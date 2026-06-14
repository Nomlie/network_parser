#!/usr/bin/env python3
# network_parser/model_bundle.py
"""
NetworkParser binary model bundle
=================================

Purpose
-------
Package a trained two-level NetworkParser registry into one portable binary
object that can be loaded for end-to-end query inference.

The bundle intentionally stores more than sklearn-style model objects.  It also
stores the selected-feature metadata needed to reconstruct the model-ready
matrix from a new query sample, including selected feature manifests, context
sequences, baseline/REF/ALT allele definitions, and feature-state evidence
needed by raw-sequence query mode.

Design rule
-----------
Training-time statistical decisions remain training-time decisions.  Querying a
bundle is inference-only: it does not rerun RF-FDR, chi-square/Fisher FDR,
permutation testing, model selection, decision-tree fitting, or bootstrap
confidence scoring.  It reloads the trained knowledge object and applies it to a
new sample in the saved feature space.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import logging
import pickle
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from network_parser.config import NetworkParserConfig
    from network_parser.query_engine import (
        NetworkParserQueryEngine,
        extract_model_importance,
        read_ranked_feature_table,
    )
except Exception:  # pragma: no cover - supports direct source-tree execution
    from config import NetworkParserConfig  # type: ignore
    from query_engine import (  # type: ignore
        NetworkParserQueryEngine,
        extract_model_importance,
        read_ranked_feature_table,
    )

logger = logging.getLogger(__name__)

BUNDLE_SCHEMA_VERSION = "1.0"
DEFAULT_BUNDLE_SUFFIX = ".npb"


# -----------------------------------------------------------------------------
# Small serialisation / path helpers
# -----------------------------------------------------------------------------


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return asdict(obj)
    return str(obj)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _resolve_path(path_value: Optional[str], base_dir: Path) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return path


def _load_pickle_or_joblib(path: Path) -> Any:
    """Load a trained model payload using joblib first, then pickle."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model payload not found: {path}")

    try:
        import joblib  # type: ignore

        return joblib.load(path)
    except Exception:
        with open(path, "rb") as handle:
            return pickle.load(handle)


def _safe_token(value: Any, max_len: int = 80) -> str:
    raw = str(value)
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)
    cleaned = cleaned.strip("_") or "item"
    if len(cleaned) <= max_len:
        return cleaned
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{cleaned[: max_len - 13]}_{digest}"


def _slot_id(tokens: Sequence[Any]) -> str:
    raw = json.dumps([str(t) for t in tokens], ensure_ascii=False)
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:12]
    label = "__".join(_safe_token(t, max_len=30) for t in tokens[:-1])
    return f"{label or 'slot'}__{digest}"


def _get_nested(mapping: Dict[str, Any], tokens: Sequence[Any]) -> Any:
    current: Any = mapping
    for token in tokens:
        if not isinstance(current, dict):
            return None
        current = current.get(token)
    return current


def _set_nested(mapping: Dict[str, Any], tokens: Sequence[Any], value: Any) -> None:
    current: Dict[str, Any] = mapping
    for token in tokens[:-1]:
        child = current.get(token)
        if not isinstance(child, dict):
            child = {}
            current[token] = child
        current = child
    current[tokens[-1]] = value


def _set_nested_payload_key(mapping: Dict[str, Any], tokens: Sequence[Any], key: str, value: Any) -> None:
    current = _get_nested(mapping, tokens)
    if isinstance(current, dict):
        current[key] = value


def _feature_hash(features: Iterable[Any]) -> str:
    canonical = [str(f) for f in features]
    payload = "\n".join(canonical).encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()


# -----------------------------------------------------------------------------
# Registry traversal helpers
# -----------------------------------------------------------------------------


def iter_model_file_slots(registry: Dict[str, Any]) -> Iterator[Tuple[List[Any], str, Dict[str, Any]]]:
    """
    Yield model-file locations in a two-level registry.

    Each yield is: (path_tokens_to_model_file, model_id, payload_dict).
    The payload dict is the registry section holding model_file/features/status.
    """
    level1 = registry.get("level1", {}) if isinstance(registry, dict) else {}
    if isinstance(level1, dict):
        yield ["level1", "model_file"], "level1", level1

    level2 = registry.get("level2", {}) if isinstance(registry, dict) else {}
    if not isinstance(level2, dict):
        return

    global_fallback = level2.get("global_fallback", {})
    if isinstance(global_fallback, dict):
        yield ["level2", "global_fallback", "model_file"], "level2.global_fallback", global_fallback

    global_binary = level2.get("global_binary_fallback", {})
    if isinstance(global_binary, dict):
        yield ["level2", "global_binary_fallback", "model_file"], "level2.global_binary_fallback", global_binary

    by_group = level2.get("by_level1_group", {})
    if isinstance(by_group, dict):
        for group, payload in by_group.items():
            if not isinstance(payload, dict):
                continue
            model_id = f"level2.by_level1_group::{str(group)}"
            yield ["level2", "by_level1_group", str(group), "model_file"], model_id, payload


def iter_manifest_slots(registry: Dict[str, Any]) -> Iterator[Tuple[List[Any], str]]:
    """Yield manifest-file locations in a two-level registry."""
    training_matrix = registry.get("training_matrix", {}) if isinstance(registry, dict) else {}
    if isinstance(training_matrix, dict):
        yield ["training_matrix", "feature_manifest_file"], "training_matrix.feature_manifest"

    level1 = registry.get("level1", {}) if isinstance(registry, dict) else {}
    if isinstance(level1, dict):
        yield ["level1", "feature_manifest", "manifest_file"], "level1.feature_manifest"

    level2 = registry.get("level2", {}) if isinstance(registry, dict) else {}
    if not isinstance(level2, dict):
        return

    for name in ("global_fallback", "global_binary_fallback"):
        payload = level2.get(name, {})
        if isinstance(payload, dict):
            yield ["level2", name, "feature_manifest", "manifest_file"], f"level2.{name}.feature_manifest"

    by_group = level2.get("by_level1_group", {})
    if isinstance(by_group, dict):
        for group, payload in by_group.items():
            if not isinstance(payload, dict):
                continue
            yield (
                ["level2", "by_level1_group", str(group), "feature_manifest", "manifest_file"],
                f"level2.by_level1_group::{str(group)}.feature_manifest",
            )


def iter_ranked_table_slots(registry: Dict[str, Any]) -> Iterator[Tuple[List[Any], str]]:
    """
    Yield ranked-feature table artifact locations where available.

    These are optional.  They improve supporting-marker ranking during bundled
    query mode, but predictions can still run without them.
    """

    def _walk(obj: Any, tokens: List[Any]) -> Iterator[Tuple[List[Any], str]]:
        if not isinstance(obj, dict):
            return
        artifacts = obj.get("artifacts")
        if isinstance(artifacts, dict):
            for key in ("rf_fdr_results_csv", "feature_results_csv"):
                if artifacts.get(key):
                    yield tokens + ["artifacts", key], f"{'.'.join(map(str, tokens))}.{key}"
        for key, value in obj.items():
            if isinstance(value, dict):
                yield from _walk(value, tokens + [key])

    yield from _walk(registry, [])


def collect_required_features(registry: Dict[str, Any]) -> List[str]:
    """Collect the ordered union of all features required by bundled models."""
    ordered: List[str] = []
    seen = set()
    for _, _, payload in iter_model_file_slots(registry):
        features = payload.get("features", []) if isinstance(payload, dict) else []
        for feature in features or []:
            f = str(feature)
            if f and f not in seen:
                seen.add(f)
                ordered.append(f)
    return ordered


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------


@dataclass
class BundleFileRecord:
    """Serializable representation of a text table stored inside the bundle."""

    slot_tokens: List[Any]
    bundle_id: str
    original_path: Optional[str]
    kind: str
    columns: List[str]
    records: List[Dict[str, Any]]
    sep: str = "\t"

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.records)
        if self.columns:
            for col in self.columns:
                if col not in df.columns:
                    df[col] = ""
            df = df.loc[:, self.columns]
        return df


@dataclass
class NetworkParserModelBundle:
    """
    Portable binary NetworkParser trained-knowledge object.

    The bundle keeps the JSON registry for transparency, but also embeds the
    model payloads and query-critical biological metadata so the object remains
    usable after being moved away from the original training directory.
    """

    schema_version: str
    created_at: str
    registry: Dict[str, Any]
    registry_source: str
    model_payloads: Dict[str, Any] = field(default_factory=dict)
    model_sources: Dict[str, str] = field(default_factory=dict)
    manifest_records: List[BundleFileRecord] = field(default_factory=list)
    ranked_table_records: List[BundleFileRecord] = field(default_factory=list)
    feature_space: Dict[str, Any] = field(default_factory=dict)
    validation_evidence: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def materialize_runtime_files(self, output_dir: Path) -> Dict[str, Any]:
        """
        Write embedded manifests / ranked feature tables to a runtime directory
        and return a registry copy whose paths point to the materialized files.
        """
        runtime_dir = _ensure_dir(Path(output_dir))
        materialized_registry = copy.deepcopy(self.registry)

        manifest_dir = _ensure_dir(runtime_dir / "manifests")
        for record in self.manifest_records:
            path = manifest_dir / f"{_safe_token(record.bundle_id)}.tsv"
            record.to_dataframe().to_csv(path, sep=record.sep or "\t", index=False)
            _set_nested(materialized_registry, record.slot_tokens, str(path))

        table_dir = _ensure_dir(runtime_dir / "ranked_feature_tables")
        for record in self.ranked_table_records:
            path = table_dir / f"{_safe_token(record.bundle_id)}.csv"
            record.to_dataframe().to_csv(path, index=False)
            _set_nested(materialized_registry, record.slot_tokens, str(path))

        # Attach bundle model IDs to the copied registry so the bundled query
        # engine can load payloads from memory instead of from model_file paths.
        for tokens, model_id, _payload in iter_model_file_slots(materialized_registry):
            _set_nested_payload_key(materialized_registry, tokens[:-1], "bundle_model_id", model_id)

        registry_path = runtime_dir / "bundled_runtime_registry.json"
        _write_json(materialized_registry, registry_path)

        return {
            "registry": materialized_registry,
            "registry_path": registry_path,
            "runtime_dir": runtime_dir,
        }

    def to_payload(self) -> Dict[str, Any]:
        """Return a plain pickle payload for cross-entry-point portability."""
        return {
            "__networkparser_model_bundle__": True,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "registry": self.registry,
            "registry_source": self.registry_source,
            "model_payloads": self.model_payloads,
            "model_sources": self.model_sources,
            "manifest_records": [asdict(record) for record in self.manifest_records],
            "ranked_table_records": [asdict(record) for record in self.ranked_table_records],
            "feature_space": self.feature_space,
            "validation_evidence": self.validation_evidence,
            "notes": self.notes,
        }

    def save(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        if path.suffix == "":
            path = path.with_suffix(DEFAULT_BUNDLE_SUFFIX)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as handle:
            pickle.dump(self.to_payload(), handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Wrote NetworkParser model bundle: %s", path)
        return path


# -----------------------------------------------------------------------------
# Bundle construction / loading
# -----------------------------------------------------------------------------


def _read_manifest_record(tokens: List[Any], bundle_id: str, original_path: Path) -> BundleFileRecord:
    df = pd.read_csv(original_path, sep="\t", dtype=str).fillna("")
    return BundleFileRecord(
        slot_tokens=list(tokens),
        bundle_id=bundle_id,
        original_path=str(original_path),
        kind="feature_manifest",
        columns=[str(c) for c in df.columns],
        records=df.to_dict(orient="records"),
        sep="\t",
    )


def _read_ranked_table_record(tokens: List[Any], bundle_id: str, original_path: Path) -> BundleFileRecord:
    df = pd.read_csv(original_path)
    return BundleFileRecord(
        slot_tokens=list(tokens),
        bundle_id=bundle_id,
        original_path=str(original_path),
        kind="ranked_feature_table",
        columns=[str(c) for c in df.columns],
        records=df.to_dict(orient="records"),
        sep=",",
    )


def build_bundle_from_registry(
    registry_path: str | Path,
    output_path: Optional[str | Path] = None,
    *,
    include_model_payloads: bool = True,
    include_feature_manifests: bool = True,
    include_ranked_feature_tables: bool = True,
) -> NetworkParserModelBundle:
    """
    Build a binary NetworkParser bundle from an existing two-level registry.

    Parameters
    ----------
    registry_path
        Path to ``two_level_model_registry.json``.
    output_path
        Optional path to write the ``.npb`` bundle immediately.
    include_model_payloads
        Embed trained model objects into the bundle.
    include_feature_manifests
        Embed selected-feature manifests / context-sequence tables.
    include_ranked_feature_tables
        Embed optional RF-FDR / feature-result tables used to rank supporting
        markers in query reports.
    """
    registry_path = Path(registry_path)
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")

    registry_base = registry_path.parent
    registry = _read_json(registry_path)
    registry_for_bundle = copy.deepcopy(registry)

    model_payloads: Dict[str, Any] = {}
    model_sources: Dict[str, str] = {}

    if include_model_payloads:
        for tokens, model_id, payload in iter_model_file_slots(registry):
            model_file = _get_nested(registry, tokens)
            resolved = _resolve_path(model_file, registry_base)
            status = payload.get("status") if isinstance(payload, dict) else None

            # Level 1 may not have a status field; Level 2 skipped payloads often do.
            if not model_file or resolved is None or not resolved.exists():
                if status == "success" or model_id == "level1":
                    logger.warning("Model file for %s is missing and will not be embedded: %s", model_id, model_file)
                continue

            model_payloads[model_id] = _load_pickle_or_joblib(resolved)
            model_sources[model_id] = str(resolved)
            _set_nested_payload_key(registry_for_bundle, tokens[:-1], "bundle_model_id", model_id)

    manifest_records: List[BundleFileRecord] = []
    if include_feature_manifests:
        seen_manifest_paths = set()
        for tokens, logical_name in iter_manifest_slots(registry):
            manifest_path = _get_nested(registry, tokens)
            resolved = _resolve_path(manifest_path, registry_base)
            if resolved is None or not resolved.exists():
                continue
            dedupe_key = (tuple(tokens), str(resolved))
            if dedupe_key in seen_manifest_paths:
                continue
            seen_manifest_paths.add(dedupe_key)
            bundle_id = f"{_slot_id(tokens)}__manifest"
            try:
                manifest_records.append(_read_manifest_record(tokens, bundle_id, resolved))
            except Exception as exc:
                logger.warning("Could not embed feature manifest %s: %s", resolved, exc)

    ranked_table_records: List[BundleFileRecord] = []
    if include_ranked_feature_tables:
        seen_tables = set()
        for tokens, logical_name in iter_ranked_table_slots(registry):
            table_path = _get_nested(registry, tokens)
            resolved = _resolve_path(table_path, registry_base)
            if resolved is None or not resolved.exists():
                continue
            dedupe_key = (tuple(tokens), str(resolved))
            if dedupe_key in seen_tables:
                continue
            seen_tables.add(dedupe_key)
            bundle_id = f"{_slot_id(tokens)}__ranked_table"
            try:
                ranked_table_records.append(_read_ranked_table_record(tokens, bundle_id, resolved))
            except Exception as exc:
                logger.warning("Could not embed ranked feature table %s: %s", resolved, exc)

    required_features = collect_required_features(registry)
    feature_space = {
        "required_feature_count": int(len(required_features)),
        "required_features": required_features,
        "required_features_sha256": _feature_hash(required_features),
        "level1_feature_count": int(len(registry.get("level1", {}).get("features", []) or [])),
        "n_embedded_models": int(len(model_payloads)),
        "n_embedded_manifests": int(len(manifest_records)),
        "n_embedded_ranked_tables": int(len(ranked_table_records)),
    }

    validation_evidence = {
        "publication_summary": registry.get("publication_summary", {}),
        "level1_filter": registry.get("level1", {}).get("filter", {}),
        "level1_feature_panel_separability": registry.get("level1", {}).get("feature_panel_separability", {}),
        "level2_global_filter": registry.get("level2", {}).get("global_fallback", {}).get("filter", {}),
        "level2_global_feature_panel_separability": registry.get("level2", {}).get("global_fallback", {}).get("feature_panel_separability", {}),
    }

    bundle = NetworkParserModelBundle(
        schema_version=BUNDLE_SCHEMA_VERSION,
        created_at=_now_utc_iso(),
        registry=registry_for_bundle,
        registry_source=str(registry_path),
        model_payloads=model_payloads,
        model_sources=model_sources,
        manifest_records=manifest_records,
        ranked_table_records=ranked_table_records,
        feature_space=feature_space,
        validation_evidence=validation_evidence,
        notes=[
            "Bundle query mode is inference-only.",
            "The bundle stores trained models plus selected-feature manifests and context-sequence metadata.",
            "Statistical filtering, model screening, tree construction, and bootstrap confidence are training-time steps and are not rerun during bundled query.",
        ],
    )

    if output_path is not None:
        bundle.save(output_path)

    logger.info(
        "Built NetworkParser bundle | models=%d | manifests=%d | ranked_tables=%d | required_features=%d",
        len(model_payloads),
        len(manifest_records),
        len(ranked_table_records),
        len(required_features),
    )
    return bundle


def save_bundle(bundle: NetworkParserModelBundle, output_path: str | Path) -> Path:
    """Save a bundle to disk."""
    return bundle.save(output_path)


def _bundle_from_payload(payload: Dict[str, Any]) -> NetworkParserModelBundle:
    manifest_records = [
        BundleFileRecord(**record)
        for record in payload.get("manifest_records", [])
        if isinstance(record, dict)
    ]
    ranked_table_records = [
        BundleFileRecord(**record)
        for record in payload.get("ranked_table_records", [])
        if isinstance(record, dict)
    ]
    return NetworkParserModelBundle(
        schema_version=str(payload.get("schema_version", "unknown")),
        created_at=str(payload.get("created_at", "")),
        registry=payload.get("registry", {}) or {},
        registry_source=str(payload.get("registry_source", "")),
        model_payloads=payload.get("model_payloads", {}) or {},
        model_sources=payload.get("model_sources", {}) or {},
        manifest_records=manifest_records,
        ranked_table_records=ranked_table_records,
        feature_space=payload.get("feature_space", {}) or {},
        validation_evidence=payload.get("validation_evidence", {}) or {},
        notes=payload.get("notes", []) or [],
    )


def load_bundle(bundle_path: str | Path) -> NetworkParserModelBundle:
    """Load a NetworkParser binary bundle from disk."""
    path = Path(bundle_path)
    if not path.exists():
        raise FileNotFoundError(f"Bundle not found: {path}")
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)

    # Backward compatibility for any early bundles saved as the dataclass object.
    if isinstance(payload, NetworkParserModelBundle):
        bundle = payload
    elif isinstance(payload, dict) and payload.get("__networkparser_model_bundle__"):
        bundle = _bundle_from_payload(payload)
    else:
        raise TypeError(f"File is not a NetworkParser model bundle: {path}")

    if bundle.schema_version != BUNDLE_SCHEMA_VERSION:
        logger.warning(
            "Bundle schema version differs from runtime | bundle=%s | runtime=%s",
            bundle.schema_version,
            BUNDLE_SCHEMA_VERSION,
        )
    return bundle


# -----------------------------------------------------------------------------
# Bundled query engine
# -----------------------------------------------------------------------------


class BundledNetworkParserQueryEngine(NetworkParserQueryEngine):
    """Query engine that loads models/manifests from a binary bundle."""

    def __init__(
        self,
        bundle: NetworkParserModelBundle,
        config: NetworkParserConfig,
        runtime_dir: str | Path,
    ):
        runtime = bundle.materialize_runtime_files(Path(runtime_dir))
        self.bundle = bundle
        self.registry_path = Path(runtime["registry_path"])
        self.registry_base = Path(runtime["runtime_dir"])
        self.registry = runtime["registry"]
        self.config = config

    def _payload_from_registry_section(self, section: Dict[str, Any], level_name: str) -> Any:
        model_id = section.get("bundle_model_id") if isinstance(section, dict) else None
        if model_id and model_id in self.bundle.model_payloads:
            return self.bundle.model_payloads[str(model_id)]

        # Fallback for partially bundled objects: use original query-engine file loading.
        model_file = section.get("model_file") if isinstance(section, dict) else None
        resolved = _resolve_path(model_file, self.registry_base)
        if resolved is not None and resolved.exists():
            return _load_pickle_or_joblib(resolved)

        raise ValueError(f"Bundle is missing a usable {level_name} model payload.")

    def _load_level1(self) -> Tuple[List[str], Any, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        level1 = self.registry.get("level1", {})
        features = [str(f) for f in level1.get("features", [])]
        if not features:
            raise ValueError("Bundle registry is missing Level 1 selected features.")

        payload = self._payload_from_registry_section(level1, "Level 1")
        ranked = read_ranked_feature_table(level1.get("filter", {}), self.registry_base)
        model_importance = extract_model_importance(payload, features)
        return features, payload, ranked, model_importance

    def _select_level2_payload(self, predicted_level1: str) -> Tuple[str, List[str], Any, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        level2 = self.registry.get("level2", {})
        by_group = level2.get("by_level1_group", {}) if isinstance(level2, dict) else {}
        group_payload = by_group.get(str(predicted_level1), {}) if isinstance(by_group, dict) else {}

        source = "level1_group_specific"
        selected = group_payload
        if not selected or selected.get("status") != "success":
            selected = level2.get("global_fallback", {}) if isinstance(level2, dict) else {}
            source = "global_fallback"
            if not selected or selected.get("status") != "success":
                selected = level2.get("global_binary_fallback", {}) if isinstance(level2, dict) else {}
                source = "global_binary_fallback"

        features = [str(f) for f in selected.get("features", [])]
        if not features:
            raise ValueError(
                "No usable Level 2 feature set found for predicted Level 1 group and no global fallback is available."
            )

        payload = self._payload_from_registry_section(selected, f"Level 2 ({source})")
        ranked = read_ranked_feature_table(selected.get("filter", {}), self.registry_base)
        model_importance = extract_model_importance(payload, features)
        return source, features, payload, ranked, model_importance



def query_bundle(
    *,
    bundle_path: str | Path,
    genomic_path: str,
    output_dir: str | Path,
    config: NetworkParserConfig,
    ref_fasta: Optional[str] = None,
    max_markers: int = 10,
    n_jobs: Optional[int] = None,
    query_input_type: str = "auto",
    raw_sequence_mapping_mode: str = "auto",
    meta_path: Optional[str] = None,
    level1_label: Optional[str] = None,
    level2_label: Optional[str] = None,
    sample_id_column: Optional[str] = None,
) -> pd.DataFrame:
    """Run query inference from a binary NetworkParser bundle."""
    out = _ensure_dir(Path(output_dir))
    bundle = load_bundle(bundle_path)
    engine = BundledNetworkParserQueryEngine(
        bundle=bundle,
        config=config,
        runtime_dir=out / "_bundle_runtime",
    )
    return engine.query(
        genomic_path=genomic_path,
        output_dir=str(out),
        ref_fasta=ref_fasta,
        max_markers=int(max_markers),
        n_jobs=n_jobs,
        query_input_type=query_input_type,
        raw_sequence_mapping_mode=raw_sequence_mapping_mode,
        meta_path=meta_path,
        level1_label=level1_label,
        level2_label=level2_label,
        sample_id_column=sample_id_column,
    )


__all__ = [
    "NetworkParserModelBundle",
    "BundledNetworkParserQueryEngine",
    "build_bundle_from_registry",
    "save_bundle",
    "load_bundle",
    "query_bundle",
]

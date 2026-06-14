#!/usr/bin/env python3
# network_parser/query_engine.py
"""
NetworkParser user-facing query engine
=====================================

Purpose
-------
Apply a trained two-level NetworkParser model registry to new strain/sample
input and produce a user-facing prediction report.

The query engine is inference-only:

    new strain/sample -> same feature representation -> saved Level 1 model
                      -> saved Level 2 model -> report

It does not rerun RF-FDR, permutation testing, FDR correction, decision-tree
training, or bootstrap confidence. Those are training/discovery-time operations.

Expected trained input
----------------------
A two-level model registry produced by ``two_level_protocol.py`` with:

    level1.model_file
    level1.features
    level2.global_fallback.model_file / features
    level2.by_level1_group.<group>.model_file / features, where available

Outputs
-------
    query_predictions.csv
    query_predictions_compact.tsv
    query_predictions_readable.html
    query_route_audit.json
    query_report.json
    query_report.txt
    query_alignment_summary.json
"""

from __future__ import annotations

import argparse
import copy
import html
import json
import logging
import pickle
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from network_parser.config import NetworkParserConfig
    from network_parser.data_loader import DataLoader
    from network_parser.sequence_query_encoder import (
        encode_raw_sequence_query,
        encode_vcf_query_from_manifest,
        load_feature_manifest,
    )
    from network_parser.utils import normalize_sample_id
    from network_parser.fastq_processor import FastqProcessor
except Exception:  # pragma: no cover - supports direct source-tree execution
    from config import NetworkParserConfig  # type: ignore
    from data_loader import DataLoader  # type: ignore
    from sequence_query_encoder import (  # type: ignore
        encode_raw_sequence_query,
        encode_vcf_query_from_manifest,
        load_feature_manifest,
    )
    from utils import normalize_sample_id  # type: ignore
    from fastq_processor import FastqProcessor  # type: ignore


logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(obj: Any) -> Any:
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
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)


def load_config(config_path: Optional[str]) -> NetworkParserConfig:
    config = NetworkParserConfig()
    if config_path is None:
        config.__post_init__()
        return config

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        overrides = json.load(handle)

    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning("Ignoring unknown config key in query config: %s", key)

    config.__post_init__()
    return config


def resolve_path(path_value: Optional[str], base_dir: Path) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return path


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pickle(path: Path) -> Any:
    """
    Load saved model payloads robustly.

    Uses joblib first for sklearn-style models, then pickle as fallback.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model payload not found: {path}")

    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as handle:
            return pickle.load(handle)


# -----------------------------------------------------------------------------
# Matrix loading and feature alignment
# -----------------------------------------------------------------------------

def load_query_matrix(
    genomic_path: str,
    output_dir: Path,
    config: NetworkParserConfig,
    ref_fasta: Optional[str] = None,
    n_jobs: Optional[int] = None,
) -> pd.DataFrame:
    """Load/construct the query sample × genomic-feature matrix.

    Query inputs are not discovery cohorts. For VCF-derived query samples we
    therefore relax cohort-level feature-retention filters so observed query
    variants are preserved and later aligned to the trained selected-feature
    space. Missing trained features are still filled as 0 by
    align_to_training_features().
    """
    query_config = copy.copy(config)
    query_config.min_sample_presence = 1
    query_config.remove_invariant = False
    query_config.min_minor_count = 0
    query_config.matrices_min_count = 0

    loader = DataLoader(config=query_config, n_jobs=n_jobs if n_jobs is not None else getattr(query_config, "n_jobs", -1))
    X = loader.load_genomic_matrix(
        file_path=genomic_path,
        output_dir=str(output_dir / "query_matrix_artifacts"),
        ref_fasta=ref_fasta,
    )
    if not isinstance(X, pd.DataFrame):
        raise TypeError("DataLoader.load_genomic_matrix did not return a pandas DataFrame.")
    X = X.copy()
    X.index = X.index.astype(str).map(normalize_sample_id)
    return X




def is_hierarchical_registry(registry: Dict[str, Any]) -> bool:
    """Return True when the registry uses the recursive hierarchy schema."""
    if not isinstance(registry, dict):
        return False
    protocol = str(registry.get("protocol", "")).strip().lower()
    hierarchy = registry.get("hierarchy", {})
    return protocol == "multi_level_hierarchy_protocol" or (
        isinstance(hierarchy, dict) and isinstance(hierarchy.get("root"), dict)
    )


def _add_unique_features(
    ordered: List[str],
    seen: set,
    features: Iterable[Any],
) -> None:
    for feature in features or []:
        f = str(feature)
        if f and f not in seen:
            seen.add(f)
            ordered.append(f)


def _collect_features_from_hierarchy_node(
    node: Dict[str, Any],
    ordered: List[str],
    seen: set,
) -> None:
    if not isinstance(node, dict):
        return
    _add_unique_features(ordered, seen, node.get("features", []))
    children = node.get("children", {})
    if isinstance(children, dict):
        for child in children.values():
            if isinstance(child, dict):
                _collect_features_from_hierarchy_node(child, ordered, seen)


def collect_required_features_from_registry(registry: Dict[str, Any]) -> List[str]:
    """Collect the union of every feature required by two-level or hierarchy models."""
    ordered: List[str] = []
    seen: set = set()

    if is_hierarchical_registry(registry):
        hierarchy = registry.get("hierarchy", {}) if isinstance(registry, dict) else {}
        root = hierarchy.get("root", {}) if isinstance(hierarchy, dict) else {}
        _collect_features_from_hierarchy_node(root, ordered, seen)
        return ordered

    level1 = registry.get("level1", {}) if isinstance(registry, dict) else {}
    _add_unique_features(ordered, seen, level1.get("features", []))

    level2 = registry.get("level2", {}) if isinstance(registry, dict) else {}
    global_payload = level2.get("global_fallback", {}) if isinstance(level2, dict) else {}
    _add_unique_features(ordered, seen, global_payload.get("features", []))
    global_binary_payload = level2.get("global_binary_fallback", {}) if isinstance(level2, dict) else {}
    _add_unique_features(ordered, seen, global_binary_payload.get("features", []))

    by_group = level2.get("by_level1_group", {}) if isinstance(level2, dict) else {}
    if isinstance(by_group, dict):
        for payload in by_group.values():
            if isinstance(payload, dict):
                _add_unique_features(ordered, seen, payload.get("features", []))

    return ordered


def resolve_registry_feature_manifest(registry: Dict[str, Any], registry_base: Path) -> Optional[Path]:
    """Resolve the all-feature manifest saved during training."""
    candidates: List[Optional[str]] = []
    training_matrix = registry.get("training_matrix", {}) if isinstance(registry, dict) else {}
    candidates.append(training_matrix.get("feature_manifest_file"))

    level1 = registry.get("level1", {}) if isinstance(registry, dict) else {}
    l1_manifest = level1.get("feature_manifest", {}) if isinstance(level1, dict) else {}
    if isinstance(l1_manifest, dict):
        candidates.append(l1_manifest.get("manifest_file"))

    level2 = registry.get("level2", {}) if isinstance(registry, dict) else {}
    global_payload = level2.get("global_fallback", {}) if isinstance(level2, dict) else {}
    g_manifest = global_payload.get("feature_manifest", {}) if isinstance(global_payload, dict) else {}
    if isinstance(g_manifest, dict):
        candidates.append(g_manifest.get("manifest_file"))

    global_binary_payload = level2.get("global_binary_fallback", {}) if isinstance(level2, dict) else {}
    gb_manifest = global_binary_payload.get("feature_manifest", {}) if isinstance(global_binary_payload, dict) else {}
    if isinstance(gb_manifest, dict):
        candidates.append(gb_manifest.get("manifest_file"))

    by_group = level2.get("by_level1_group", {}) if isinstance(level2, dict) else {}
    if isinstance(by_group, dict):
        for payload in by_group.values():
            if not isinstance(payload, dict):
                continue
            group_manifest = payload.get("feature_manifest", {})
            if isinstance(group_manifest, dict):
                candidates.append(group_manifest.get("manifest_file"))

    for candidate in candidates:
        resolved = resolve_path(candidate, registry_base)
        if resolved is not None and resolved.exists():
            return resolved
    return None


def feature_call_metadata_by_sample(calls: Optional[pd.DataFrame]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if calls is None or calls.empty:
        return {}
    if "sample_id" not in calls.columns or "feature_id" not in calls.columns:
        return {}
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for _, row in calls.iterrows():
        sample_id = str(row.get("sample_id", ""))
        feature_id = str(row.get("feature_id", ""))
        if not sample_id or not feature_id:
            continue
        out.setdefault(sample_id, {})[feature_id] = row.to_dict()
    return out

def align_to_training_features(
    X_new: pd.DataFrame,
    features: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Align new samples to a saved trained feature list.

    Missing trained features are filled with 0. Extra query features are ignored.
    This keeps query-time inference consistent with the training-time feature space,
    but the alignment summary records missing-feature burden so reports can flag
    low-coverage query inputs.
    """
    requested = [str(f) for f in features]

    if not requested:
        raise ValueError("Training feature list is empty; cannot align query matrix.")

    X = X_new.copy()
    X.columns = X.columns.astype(str)

    requested_set = set(requested)
    missing = [f for f in requested if f not in X.columns]
    extra = [f for f in X.columns if f not in requested_set]

    if missing:
        fill_block = pd.DataFrame(0, index=X.index, columns=missing)
        X = pd.concat([X, fill_block], axis=1)

    X_aligned = X.loc[:, requested].copy()
    X_aligned = X_aligned.apply(pd.to_numeric, errors="coerce").fillna(0)

    missing_fraction = float(len(missing) / max(1, len(requested)))

    if missing_fraction == 0:
        alignment_status = "complete"
        warning = None
    elif missing_fraction < 0.5:
        alignment_status = "partial"
        warning = (
            "Some trained features were missing from the query input and were filled as 0. "
            "Interpret prediction support with caution."
        )
    else:
        alignment_status = "low_feature_coverage"
        warning = (
            "Many trained features were missing from the query input and were filled as 0. "
            "This may indicate a feature-space mismatch between training and query data."
        )

    summary = {
        "requested_training_features": int(len(requested)),
        "features_present_in_query": int(len(requested) - len(missing)),
        "missing_training_features_filled_as_zero": int(len(missing)),
        "missing_training_feature_fraction": missing_fraction,
        "extra_query_features_ignored": int(len(extra)),
        "alignment_status": alignment_status,
        "warning": warning,
        "missing_feature_names": missing,
    }

    if warning:
        logger.warning(warning)

    if missing_fraction > 0.3:
        logger.warning(
            "High missing-feature fraction (%.2f). "
            "Check that query input uses the same reference/contig naming as training.",
            missing_fraction,
        )

    return X_aligned, summary

# -----------------------------------------------------------------------------
# Model prediction helpers
# -----------------------------------------------------------------------------

def unpack_model_payload(payload: Any) -> Tuple[Any, Optional[Any], Optional[List[str]]]:
    """
    Support the fallback payload written by two_level_protocol.py and plain
    sklearn-like model objects.
    """
    if isinstance(payload, dict) and "model" in payload:
        model = payload.get("model")
        label_encoder = payload.get("label_encoder")
        features = payload.get("features")
        return model, label_encoder, list(features) if features is not None else None
    return payload, None, None


def predict_labels_and_support(
    payload: Any,
    X: pd.DataFrame,
) -> Tuple[List[str], List[Optional[float]], List[Dict[str, float]]]:
    """
    Predict labels plus probability-like support.

    Supports:
      - sklearn-like models with predict()/predict_proba()
      - fallback payloads containing model + label_encoder
      - NetworkParser-style models exposing identify()
    """
    model, label_encoder, _ = unpack_model_payload(payload)

    labels: List[str] = []
    max_support: List[Optional[float]] = []
    class_support: List[Dict[str, float]] = []

    if hasattr(model, "predict"):
        raw_pred = model.predict(X)

        if label_encoder is not None:
            try:
                labels = [str(v) for v in label_encoder.inverse_transform(raw_pred)]
            except Exception:
                labels = [str(v) for v in raw_pred]
        else:
            labels = [str(v) for v in raw_pred]

        max_support = [None for _ in labels]
        class_support = [{} for _ in labels]

        if hasattr(model, "predict_proba"):
            try:
                proba = np.asarray(model.predict_proba(X), dtype=float)

                if label_encoder is not None and hasattr(label_encoder, "classes_"):
                    classes = [str(c) for c in label_encoder.classes_]
                elif hasattr(model, "classes_"):
                    classes = [str(c) for c in model.classes_]
                else:
                    classes = [str(i) for i in range(proba.shape[1])]

                max_support = [float(np.max(row)) for row in proba]
                class_support = [
                    {
                        classes[i]: float(row[i])
                        for i in range(min(len(classes), len(row)))
                    }
                    for row in proba
                ]

            except Exception as exc:
                logger.warning(
                    "Model exposes predict_proba but support extraction failed: %s",
                    exc,
                )

        return labels, max_support, class_support

    if hasattr(model, "identify"):
        for _, row in X.iterrows():
            marker_dict = {
                str(col): float(value)
                for col, value in row.items()
            }

            result = model.identify(marker_dict)
            pred_list = result.get("predictions", []) if isinstance(result, dict) else []

            if not pred_list:
                labels.append("unavailable")
                max_support.append(None)
                class_support.append({})
                continue

            first = pred_list[0]

            if isinstance(first, dict):
                label = first.get("label") or first.get("class") or first.get("prediction")
                prob = first.get("probability") or first.get("support") or first.get("score")
            elif isinstance(first, (tuple, list)):
                label = first[0] if len(first) >= 1 else "unavailable"
                prob = first[1] if len(first) >= 2 else None
            else:
                label = first
                prob = None

            labels.append(str(label))

            try:
                max_support.append(float(prob) if prob is not None else None)
            except Exception:
                max_support.append(None)

            class_support.append({str(label): max_support[-1]} if max_support[-1] is not None else {})

        return labels, max_support, class_support

    raise TypeError(
        "Model payload does not expose predict() or identify(); cannot perform inference."
    )


def read_ranked_feature_table(filter_summary: Dict[str, Any], registry_base: Path) -> Optional[pd.DataFrame]:
    artifacts = filter_summary.get("artifacts", {}) if isinstance(filter_summary, dict) else {}
    table_path = artifacts.get("rf_fdr_results_csv") or artifacts.get("feature_results_csv")
    resolved = resolve_path(table_path, registry_base)
    if resolved is None or not resolved.exists():
        return None
    try:
        df = pd.read_csv(resolved)
        if "feature" not in df.columns:
            return None
        return df
    except Exception as exc:
        logger.warning("Could not read ranked feature table %s: %s", resolved, exc)
        return None


def extract_model_importance(payload: Any, features: Sequence[str]) -> Optional[pd.DataFrame]:
    model, _, _ = unpack_model_payload(payload)
    if not hasattr(model, "feature_importances_"):
        return None
    values = np.asarray(getattr(model, "feature_importances_"), dtype=float)
    if values.shape[0] != len(features):
        return None
    return pd.DataFrame({"feature": list(features), "model_importance": values}).sort_values(
        "model_importance", ascending=False
    )


RESOLVED_ALLELE_CALLS = {"baseline_match", "alt_match", "known_nonbaseline_match"}
NONBASELINE_ALLELE_CALLS = {"alt_match", "known_nonbaseline_match"}
BASELINE_ALLELE_CALLS = {"baseline_match"}
UNRESOLVED_ALLELE_CALLS = {
    "not_called",
    "ambiguous_base",
    "not_called_multi_hit_context",
    "non_training_allele",
}
SUPPORTING_EVIDENCE_ROLES = {
    "resolved_nonbaseline_state",
    "resolved_baseline_state",
    "resolved_trained_zero_state",
    "aligned_matrix_state",
}


def _normalised_text(value: Any) -> str:
    return str(value or "").strip()


def _feature_evidence_role(
    *,
    feature_id: str,
    value: Any,
    metadata: Optional[Dict[str, Any]] = None,
    available_features: Optional[set] = None,
) -> str:
    """Classify whether a zero is a real trained state or a cautious fill value."""
    meta = metadata or {}
    allele_call = _normalised_text(meta.get("allele_call"))
    mapping_status = _normalised_text(meta.get("mapping_status"))
    numeric_value = _safe_float(value)
    numeric_value = 0.0 if numeric_value is None else float(numeric_value)

    if allele_call in NONBASELINE_ALLELE_CALLS:
        return "resolved_nonbaseline_state"
    if allele_call in BASELINE_ALLELE_CALLS:
        return "resolved_baseline_state"
    if allele_call in RESOLVED_ALLELE_CALLS:
        return "resolved_trained_zero_state" if numeric_value == 0.0 else "resolved_nonbaseline_state"
    if allele_call in UNRESOLVED_ALLELE_CALLS or mapping_status:
        return "unresolved_zero_fill"

    # Matrix-only query mode has no allele-call metadata. In that setting, a
    # feature that came from the user-supplied matrix is an aligned matrix state;
    # a feature absent from the matrix and injected by alignment is a zero-fill.
    if available_features is not None:
        return "aligned_matrix_state" if feature_id in available_features else "unresolved_zero_fill"

    return "aligned_matrix_state"


def _is_supporting_marker_role(role: str) -> bool:
    return str(role) in SUPPORTING_EVIDENCE_ROLES


def supporting_markers_for_sample(
    sample_values: pd.Series,
    ranked_features: Optional[pd.DataFrame],
    model_importance: Optional[pd.DataFrame],
    max_markers: int = 10,
    feature_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    available_features: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """
    Return top trained-marker evidence for a sample.

    This intentionally includes resolved baseline 0 states. A baseline 0 is a
    valid trained marker state when the query feature was genuinely resolved.
    Unresolved, ambiguous, repeated, or non-training allele zero-fills are kept
    out of the supporting-marker list and reported through evidence summaries.
    """
    feature_metadata = feature_metadata or {}
    available_features = {str(f) for f in available_features} if available_features is not None else None

    def _attach_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
        feature_id = str(record.get("feature", ""))
        meta = feature_metadata.get(feature_id, {})
        for key in (
            "observed_allele",
            "mapping_status",
            "mapping_quality",
            "allele_call",
            "ref_allele",
            "alt_allele",
            "baseline_allele",
            "sequence",
            "position",
            "gene_annotation",
            "nucleotide_change",
            "amino_acid_change",
            "subject_id",
            "subject_position",
            "strand",
            "mapping_method",
            "n_context_hits",
            "n_blast_hits",
            "n_equivalent_best_hits",
            "blast_pident",
            "blast_query_coverage",
            "blast_bitscore",
        ):
            if key in meta and meta.get(key) not in (None, ""):
                record[key] = meta.get(key)
        role = _feature_evidence_role(
            feature_id=feature_id,
            value=record.get("value", 0),
            metadata=meta,
            available_features=available_features,
        )
        record["evidence_role"] = role
        record["supports_trained_marker_pattern"] = bool(_is_supporting_marker_role(role))
        return record

    def _candidate_record(feature: Any, extra: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        feature_id = str(feature)
        if feature_id not in sample_values.index:
            return None
        value = _safe_float(sample_values.get(feature_id))
        value = 0.0 if value is None else float(value)
        record: Dict[str, Any] = {"feature": feature_id, "value": value}
        if extra:
            record.update(extra)
        record = _attach_metadata(record)
        if not record.get("supports_trained_marker_pattern"):
            return None
        return record

    records: List[Dict[str, Any]] = []
    seen = set()

    if ranked_features is not None and "feature" in ranked_features.columns:
        df = ranked_features.copy()
        df["feature"] = df["feature"].astype(str)
        df = df[df["feature"].isin(set(map(str, sample_values.index)))]
        for _, row in df.iterrows():
            feature = str(row["feature"])
            if feature in seen:
                continue
            rec = _candidate_record(
                feature,
                {
                    "rf_mean_importance": _safe_float(row.get("rf_mean_importance")),
                    "empirical_p_value": _safe_float(row.get("empirical_p_value")),
                    "corrected_p_value": _safe_float(row.get("corrected_p_value")),
                },
            )
            if rec is not None:
                records.append(rec)
                seen.add(feature)
            if len(records) >= max_markers:
                return records

    if model_importance is not None and "feature" in model_importance.columns:
        df = model_importance.copy()
        df["feature"] = df["feature"].astype(str)
        df = df[df["feature"].isin(set(map(str, sample_values.index)))]
        for _, row in df.iterrows():
            feature = str(row["feature"])
            if feature in seen:
                continue
            rec = _candidate_record(
                feature,
                {"model_importance": _safe_float(row.get("model_importance"))},
            )
            if rec is not None:
                records.append(rec)
                seen.add(feature)
            if len(records) >= max_markers:
                return records

    for feature in map(str, sample_values.index):
        if feature in seen:
            continue
        rec = _candidate_record(feature)
        if rec is not None:
            records.append(rec)
            seen.add(feature)
        if len(records) >= max_markers:
            break
    return records


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _status_from_unique_fraction(unique_fraction: Optional[float], has_mapping_metadata: bool) -> Tuple[str, str]:
    """Classify whether a model-specific selected feature set was mapped in the query."""
    if not has_mapping_metadata:
        return (
            "feature_space_alignment_only",
            "No per-feature allele-call metadata were available; status is based on aligned matrix states.",
        )
    if unique_fraction is None:
        return (
            "unknown_marker_recovery",
            "Per-feature mapping metadata were present, but marker recovery fraction could not be computed.",
        )
    if unique_fraction >= 0.80:
        return (
            "adequate_marker_recovery",
            "Most selected markers for this model were mapped or reported in the query input.",
        )
    if unique_fraction >= 0.50:
        return (
            "partial_marker_recovery",
            "Only part of this model's selected marker space was mapped or reported; interpret this level with caution.",
        )
    return (
        "low_marker_recovery",
        "Most selected markers for this model were unresolved or ambiguous; prediction support is likely weak.",
    )


def _status_from_active_fraction(active_fraction: float, active_count: int) -> Tuple[str, str]:
    """Classify non-baseline evidence for a model-specific selected feature set."""
    if active_count >= 10 or active_fraction >= 0.01:
        return (
            "active_marker_evidence_present",
            "This model's selected feature set contains multiple non-baseline query states.",
        )
    if active_count > 0:
        return (
            "very_low_active_marker_evidence",
            "Only a small number of this model's selected features are non-baseline in the query.",
        )
    return (
        "no_active_marker_evidence",
        "This model's selected features are recovered mainly as baseline/zero states in the query.",
    )


def _status_from_resolved_fraction(
    resolved_fraction: float,
    resolved_count: int,
    has_mapping_metadata: bool,
) -> Tuple[str, str]:
    """Classify resolved trained-marker pattern evidence independently of active 1s."""
    if not has_mapping_metadata:
        if resolved_count > 0:
            return (
                "aligned_matrix_evidence_present",
                "No per-feature allele-call metadata were available, but trained features were present in the aligned query matrix.",
            )
        return (
            "feature_space_alignment_only",
            "No per-feature allele-call metadata were available; resolved marker status cannot be separated from matrix alignment.",
        )
    if resolved_fraction >= 0.80:
        return (
            "resolved_marker_evidence_present",
            "Most model-specific trained markers were resolved, including baseline states encoded as 0.",
        )
    if resolved_fraction >= 0.50:
        return (
            "partial_resolved_marker_evidence",
            "A useful fraction of model-specific trained markers was resolved, but some calls remain caution states.",
        )
    if resolved_count > 0:
        return (
            "low_resolved_marker_evidence",
            "Only a small fraction of model-specific trained markers was resolved in the query input.",
        )
    return (
        "no_resolved_marker_evidence",
        "No model-specific trained markers were confirmed as resolved query states.",
    )


def summarize_feature_evidence_for_model(
    *,
    sample_values: pd.Series,
    features: Sequence[str],
    feature_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    available_features: Optional[set] = None,
) -> Dict[str, Any]:
    """Summarise query evidence for the exact feature list used by one model.

    Non-baseline evidence is still counted, but it is no longer the only evidence
    measure. Resolved baseline states encoded as 0 are counted as valid trained-
    marker evidence for the overall query pattern.
    """
    requested = [str(f) for f in features or []]
    feature_metadata = feature_metadata or {}
    available_features = {str(f) for f in available_features} if available_features is not None else None

    values = pd.to_numeric(sample_values.reindex(requested).fillna(0), errors="coerce").fillna(0)
    active_features = [str(f) for f, value in values.items() if float(value) != 0.0]
    n_features = int(len(requested))
    n_active = int(len(active_features))
    active_fraction = float(n_active / max(1, n_features))

    has_mapping_metadata = any(str(f) in feature_metadata for f in requested)
    metadata_rows = [feature_metadata.get(str(f), {}) for f in requested]

    mapping_status_values = [str(m.get("mapping_status", "")) for m in metadata_rows if m]
    allele_call_values = [str(m.get("allele_call", "")) for m in metadata_rows if m]

    mapping_status_counts = {
        str(k): int(v)
        for k, v in pd.Series(mapping_status_values, dtype="object").value_counts(dropna=False).to_dict().items()
    } if mapping_status_values else {}
    allele_call_counts = {
        str(k): int(v)
        for k, v in pd.Series(allele_call_values, dtype="object").value_counts(dropna=False).to_dict().items()
    } if allele_call_values else {}

    evidence_role_counts: Dict[str, int] = {}
    resolved_features: List[str] = []
    resolved_baseline_features: List[str] = []
    resolved_nonbaseline_features: List[str] = []
    unresolved_or_missing_features: List[str] = []

    for feature in requested:
        meta = feature_metadata.get(feature, {})
        role = _feature_evidence_role(
            feature_id=feature,
            value=values.get(feature, 0),
            metadata=meta,
            available_features=available_features,
        )
        evidence_role_counts[role] = evidence_role_counts.get(role, 0) + 1
        if role in {"resolved_baseline_state", "resolved_nonbaseline_state", "resolved_trained_zero_state"}:
            resolved_features.append(feature)
        elif role == "aligned_matrix_state" and not has_mapping_metadata:
            # For matrix-only queries, present matrix states are the best
            # available trained-pattern evidence even though allele resolution is absent.
            resolved_features.append(feature)
        else:
            unresolved_or_missing_features.append(feature)

        if role in {"resolved_baseline_state", "resolved_trained_zero_state"}:
            resolved_baseline_features.append(feature)
        if role == "resolved_nonbaseline_state":
            resolved_nonbaseline_features.append(feature)

    unique_mapped = int(sum(1 for status in mapping_status_values if status == "mapped_unique_context"))
    mapped_or_reported = int(len(mapping_status_values))
    unique_fraction = (
        float(unique_mapped / max(1, mapped_or_reported))
        if has_mapping_metadata else None
    )

    n_resolved = int(len(resolved_features))
    n_resolved_baseline = int(len(resolved_baseline_features))
    n_resolved_nonbaseline = int(len(resolved_nonbaseline_features))
    resolved_fraction = float(n_resolved / max(1, n_features))
    resolved_baseline_fraction = float(n_resolved_baseline / max(1, n_features))
    resolved_nonbaseline_fraction = float(n_resolved_nonbaseline / max(1, n_features))

    recovery_status, recovery_reason = _status_from_unique_fraction(unique_fraction, has_mapping_metadata)
    active_status, active_reason = _status_from_active_fraction(active_fraction, n_active)
    resolved_status, resolved_reason = _status_from_resolved_fraction(
        resolved_fraction=resolved_fraction,
        resolved_count=n_resolved,
        has_mapping_metadata=has_mapping_metadata,
    )

    n_multi_hit = int(
        allele_call_counts.get("not_called_multi_hit_context", 0)
        + sum(v for k, v in mapping_status_counts.items() if "multi_hit" in str(k))
    )
    n_ambiguous = int(
        allele_call_counts.get("ambiguous_base", 0)
        + sum(v for k, v in mapping_status_counts.items() if "ambiguous_base" in str(k))
    )
    n_non_training = int(
        allele_call_counts.get("non_training_allele", 0)
        + sum(v for k, v in mapping_status_counts.items() if "non_training_allele" in str(k))
    )
    n_unresolved_or_missing = int(
        allele_call_counts.get("not_called", 0)
        + sum(v for k, v in mapping_status_counts.items() if "missing_context" in str(k) or "unresolved_context" in str(k))
    )
    n_zero_fill_caution = int(
        evidence_role_counts.get("unresolved_zero_fill", 0)
    )

    return {
        "n_selected_features": n_features,
        "n_active_features": n_active,
        "active_feature_fraction": active_fraction,
        "active_feature_ids": active_features,
        "nonbaseline_evidence_status": active_status,
        "nonbaseline_evidence_reason": active_reason,
        "has_mapping_metadata": bool(has_mapping_metadata),
        "n_features_with_mapping_metadata": int(mapped_or_reported),
        "n_unique_mapped_features": int(unique_mapped),
        "unique_mapped_fraction": unique_fraction,
        "marker_recovery_status": recovery_status,
        "marker_recovery_reason": recovery_reason,
        "active_marker_evidence_status": active_status,
        "active_marker_evidence_reason": active_reason,
        "n_resolved_features": n_resolved,
        "resolved_feature_fraction": resolved_fraction,
        "n_resolved_baseline_features": n_resolved_baseline,
        "resolved_baseline_feature_fraction": resolved_baseline_fraction,
        "n_resolved_nonbaseline_features": n_resolved_nonbaseline,
        "resolved_nonbaseline_feature_fraction": resolved_nonbaseline_fraction,
        "resolved_marker_evidence_status": resolved_status,
        "resolved_marker_evidence_reason": resolved_reason,
        "evidence_role_counts": evidence_role_counts,
        "mapping_status_counts": mapping_status_counts,
        "allele_call_counts": allele_call_counts,
        "n_baseline_match_calls": int(allele_call_counts.get("baseline_match", 0)),
        "n_alt_match_calls": int(allele_call_counts.get("alt_match", 0)),
        "n_known_nonbaseline_match_calls": int(allele_call_counts.get("known_nonbaseline_match", 0)),
        "n_unresolved_or_missing_calls": n_unresolved_or_missing,
        "n_multi_hit_calls": n_multi_hit,
        "n_ambiguous_base_calls": n_ambiguous,
        "n_non_training_allele_calls": n_non_training,
        "n_zero_fill_caution_features": n_zero_fill_caution,
        "n_unresolved_or_missing_features": int(len(unresolved_or_missing_features)),
    }


def interpretation_confidence_for_level(
    *,
    support: Optional[float],
    evidence: Dict[str, Any],
    n_supporting_markers: int,
) -> Tuple[str, str]:
    """Combine model support and resolved trained-marker evidence into a cautious label."""
    support_value = _safe_float(support)
    active_count = int(evidence.get("n_active_features", 0) or 0)
    resolved_count = int(evidence.get("n_resolved_features", 0) or 0)
    resolved_fraction = float(evidence.get("resolved_feature_fraction", 0.0) or 0.0)
    recovery_status = str(evidence.get("marker_recovery_status", ""))
    resolved_status = str(evidence.get("resolved_marker_evidence_status", ""))

    if recovery_status == "low_marker_recovery" and resolved_count == 0:
        return (
            "low_confidence",
            "Prediction generated, but too few model-specific selected markers were resolved in the query input.",
        )

    if resolved_count == 0:
        return (
            "low_confidence",
            "Prediction generated, but this model received no confirmed resolved trained-marker states for this sample.",
        )

    if support_value is None:
        return (
            "evidence_available_support_unavailable",
            "Resolved trained-marker pattern evidence is present, but probability-like support was unavailable from the model.",
        )

    if support_value >= 0.70 and n_supporting_markers > 0 and resolved_fraction >= 0.50:
        if active_count == 0:
            return (
                "high_confidence_baseline_pattern",
                "Prediction has strong model support and many resolved trained markers, mostly as baseline states encoded as 0.",
            )
        return (
            "high_confidence",
            "Prediction has strong model support and resolved model-specific trained-marker evidence.",
        )

    if support_value >= 0.50 and (n_supporting_markers > 0 or resolved_count > 0):
        return (
            "moderate_confidence",
            "Prediction has some model support and a resolved trained-marker pattern, but should still be interpreted cautiously.",
        )

    if resolved_status in {"resolved_marker_evidence_present", "partial_resolved_marker_evidence", "aligned_matrix_evidence_present"}:
        return (
            "low_to_moderate_confidence",
            "Resolved trained-marker evidence is available, but model support is weak.",
        )

    return (
        "low_confidence",
        "Prediction has weak probability-like support despite available marker evidence.",
    )


def flatten_feature_evidence(prefix: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Small CSV-friendly subset of feature-evidence diagnostics."""
    return {
        f"{prefix}_n_selected_features": evidence.get("n_selected_features"),
        f"{prefix}_n_active_features": evidence.get("n_active_features"),
        f"{prefix}_active_feature_fraction": evidence.get("active_feature_fraction"),
        f"{prefix}_nonbaseline_evidence_status": evidence.get("nonbaseline_evidence_status"),
        f"{prefix}_nonbaseline_evidence_reason": evidence.get("nonbaseline_evidence_reason"),
        f"{prefix}_n_resolved_features": evidence.get("n_resolved_features"),
        f"{prefix}_resolved_feature_fraction": evidence.get("resolved_feature_fraction"),
        f"{prefix}_n_resolved_baseline_features": evidence.get("n_resolved_baseline_features"),
        f"{prefix}_resolved_baseline_feature_fraction": evidence.get("resolved_baseline_feature_fraction"),
        f"{prefix}_n_resolved_nonbaseline_features": evidence.get("n_resolved_nonbaseline_features"),
        f"{prefix}_resolved_nonbaseline_feature_fraction": evidence.get("resolved_nonbaseline_feature_fraction"),
        f"{prefix}_resolved_marker_evidence_status": evidence.get("resolved_marker_evidence_status"),
        f"{prefix}_resolved_marker_evidence_reason": evidence.get("resolved_marker_evidence_reason"),
        f"{prefix}_n_unique_mapped_features": evidence.get("n_unique_mapped_features"),
        f"{prefix}_unique_mapped_fraction": evidence.get("unique_mapped_fraction"),
        f"{prefix}_marker_recovery_status": evidence.get("marker_recovery_status"),
        f"{prefix}_marker_recovery_reason": evidence.get("marker_recovery_reason"),
        f"{prefix}_active_marker_evidence_status": evidence.get("active_marker_evidence_status"),
        f"{prefix}_active_marker_evidence_reason": evidence.get("active_marker_evidence_reason"),
        f"{prefix}_n_baseline_match_calls": evidence.get("n_baseline_match_calls"),
        f"{prefix}_n_alt_match_calls": evidence.get("n_alt_match_calls"),
        f"{prefix}_n_known_nonbaseline_match_calls": evidence.get("n_known_nonbaseline_match_calls"),
        f"{prefix}_n_unresolved_or_missing_calls": evidence.get("n_unresolved_or_missing_calls"),
        f"{prefix}_n_multi_hit_calls": evidence.get("n_multi_hit_calls"),
        f"{prefix}_n_ambiguous_base_calls": evidence.get("n_ambiguous_base_calls"),
        f"{prefix}_n_non_training_allele_calls": evidence.get("n_non_training_allele_calls"),
        f"{prefix}_n_zero_fill_caution_features": evidence.get("n_zero_fill_caution_features"),
    }


def decision_tree_path_explanation(payload: Any, X: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Best-effort explanation for sklearn decision-tree-like models.
    If unavailable, returns empty paths. This does not train a tree.
    """
    model, _, features_from_payload = unpack_model_payload(payload)
    if not hasattr(model, "tree_"):
        return {str(idx): [] for idx in X.index}

    feature_names = features_from_payload if features_from_payload is not None else list(X.columns)
    tree = model.tree_
    paths: Dict[str, List[str]] = {}

    for sample_id, row in X.iterrows():
        node_id = 0
        rules: List[str] = []
        values = row.values.astype(float)
        while tree.children_left[node_id] != tree.children_right[node_id]:
            feature_idx = int(tree.feature[node_id])
            threshold = float(tree.threshold[node_id])
            feature_name = str(feature_names[feature_idx]) if feature_idx < len(feature_names) else f"feature_{feature_idx}"
            value = float(values[feature_idx])
            if value <= threshold:
                rules.append(f"{feature_name} <= {threshold:.6g}")
                node_id = int(tree.children_left[node_id])
            else:
                rules.append(f"{feature_name} > {threshold:.6g}")
                node_id = int(tree.children_right[node_id])
        paths[str(sample_id)] = rules
    return paths


# -----------------------------------------------------------------------------
# Query engine
# -----------------------------------------------------------------------------

class NetworkParserQueryEngine:
    """Apply saved two-level or multi-level NetworkParser models to new samples."""

    def __init__(self, registry_path: str, config: NetworkParserConfig):
        self.registry_path = Path(registry_path)
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Model registry not found: {self.registry_path}")
        self.registry_base = self.registry_path.parent
        self.registry = load_json(self.registry_path)
        self.config = config

    def _load_level1(self) -> Tuple[List[str], Any, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        level1 = self.registry.get("level1", {})
        features = [str(f) for f in level1.get("features", [])]
        model_path = resolve_path(level1.get("model_file"), self.registry_base)
        if not features:
            raise ValueError("Registry is missing Level 1 selected features.")
        if model_path is None or not model_path.exists():
            raise ValueError("Registry is missing a readable Level 1 model file.")
        payload = load_pickle(model_path)
        ranked = read_ranked_feature_table(level1.get("filter", {}), self.registry_base)
        model_importance = extract_model_importance(payload, features)
        return features, payload, ranked, model_importance

    def _select_level2_payload(self, predicted_level1: str) -> Tuple[str, List[str], Any, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        level2 = self.registry.get("level2", {})
        by_group = level2.get("by_level1_group", {}) if isinstance(level2, dict) else {}
        group_payload = by_group.get(str(predicted_level1), {}) if isinstance(by_group, dict) else {}

        source = "level1_group_specific"
        selected = group_payload
        if not selected or selected.get("status") != "success" or not selected.get("model_file"):
            selected = level2.get("global_fallback", {}) if isinstance(level2, dict) else {}
            source = "global_fallback"
            if not selected or selected.get("status") != "success" or not selected.get("model_file"):
                selected = level2.get("global_binary_fallback", {}) if isinstance(level2, dict) else {}
                source = "global_binary_fallback"

        features = [str(f) for f in selected.get("features", [])]
        model_path = resolve_path(selected.get("model_file"), self.registry_base)
        if not features or model_path is None or not model_path.exists():
            raise ValueError(
                "No usable Level 2 model found for predicted Level 1 group and no global fallback is available."
            )

        payload = load_pickle(model_path)
        ranked = read_ranked_feature_table(selected.get("filter", {}), self.registry_base)
        model_importance = extract_model_importance(payload, features)
        return source, features, payload, ranked, model_importance

    def _load_hierarchy_node(
        self,
        node: Dict[str, Any],
    ) -> Tuple[List[str], Any, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load a trainable hierarchy node from the recursive registry."""
        features = [str(f) for f in node.get("features", [])]
        model_path = resolve_path(node.get("model_file"), self.registry_base)
        if not features:
            raise ValueError(
                f"Hierarchy node for label '{node.get('label_column')}' has no selected features."
            )
        if model_path is None or not model_path.exists():
            raise ValueError(
                f"Hierarchy node for label '{node.get('label_column')}' has no readable model file."
            )
        payload = load_pickle(model_path)
        ranked = read_ranked_feature_table(node.get("filter", {}), self.registry_base)
        model_importance = extract_model_importance(payload, features)
        return features, payload, ranked, model_importance

    @staticmethod
    def _hierarchy_node_key(node: Dict[str, Any], fallback: str = "root") -> str:
        """Stable audit key for a hierarchy node without exposing feature names."""
        level_number = str(node.get("level_number", "NA"))
        label_column = str(node.get("label_column", "label"))
        path = node.get("path", []) or []
        path_values = []
        for item in path:
            if isinstance(item, dict):
                path_values.append(str(item.get("value", "")))
        suffix = "__" + "__".join(path_values) if path_values else ""
        key = f"level_{level_number}__{label_column}{suffix}".strip("_")
        return key or fallback

    def _query_hierarchy_from_matrix(
        self,
        *,
        X_raw: pd.DataFrame,
        raw_calls: Optional[pd.DataFrame],
        raw_mapping_summary: Optional[Dict[str, Any]],
        raw_feature_metadata: Dict[str, Dict[str, Dict[str, Any]]],
        raw_sample_quality: Dict[str, Dict[str, Any]],
        raw_available_features: set,
        out: Path,
        genomic_path: str,
        query_input_type: str,
        fastq_processing_summary: Optional[Dict[str, Any]],
        max_markers: int,
    ) -> pd.DataFrame:
        """Recursive query traversal for ``hierarchical_model_registry.json``."""
        hierarchy = self.registry.get("hierarchy", {}) if isinstance(self.registry, dict) else {}
        root = hierarchy.get("root", {}) if isinstance(hierarchy, dict) else {}
        label_columns = [str(x) for x in hierarchy.get("label_columns", [])] if isinstance(hierarchy, dict) else []
        if not isinstance(root, dict) or not root:
            raise ValueError("Hierarchical registry is missing hierarchy.root.")

        rows: List[Dict[str, Any]] = []
        report_samples: List[Dict[str, Any]] = []
        alignment_by_node: Dict[str, Any] = {}

        for sample_id in X_raw.index.astype(str):
            sample_row = X_raw.loc[[sample_id]].copy()
            sample_feature_metadata = raw_feature_metadata.get(sample_id, {})
            sample_mapping_quality = raw_sample_quality.get(sample_id, {})

            row: Dict[str, Any] = {
                "sample_id": sample_id,
                "query_marker_recovery_status": sample_mapping_quality.get("marker_recovery_status"),
                "query_marker_recovery_reason": sample_mapping_quality.get("marker_recovery_reason"),
                "query_active_marker_evidence_status": sample_mapping_quality.get("active_marker_evidence_status"),
                "query_active_marker_evidence_reason": sample_mapping_quality.get("active_marker_evidence_reason"),
                "query_unique_mapped_fraction": sample_mapping_quality.get("unique_mapped_fraction"),
                "query_active_feature_fraction": sample_mapping_quality.get("active_feature_fraction"),
                "query_n_encoded_active_features": sample_mapping_quality.get("n_encoded_active_features"),
                "query_n_resolved_features": sample_mapping_quality.get("n_resolved_features"),
                "query_resolved_feature_fraction": sample_mapping_quality.get("resolved_feature_fraction"),
                "query_n_resolved_baseline_features": sample_mapping_quality.get("n_resolved_baseline_features"),
                "query_resolved_baseline_feature_fraction": sample_mapping_quality.get("resolved_baseline_feature_fraction"),
                "query_resolved_marker_evidence_status": sample_mapping_quality.get("resolved_marker_evidence_status"),
                "query_resolved_marker_evidence_reason": sample_mapping_quality.get("resolved_marker_evidence_reason"),
                "query_n_unresolved_or_missing_calls": sample_mapping_quality.get("n_unresolved_or_missing_context_calls", sample_mapping_quality.get("n_unresolved_or_missing_calls")),
                "query_n_multi_hit_calls": sample_mapping_quality.get("n_multi_hit_calls"),
                "query_n_ambiguous_base_calls": sample_mapping_quality.get("n_ambiguous_base_calls"),
                "query_n_non_training_allele_calls": sample_mapping_quality.get("n_non_training_allele_calls"),
            }

            report_entry: Dict[str, Any] = {
                **row,
                "hierarchy_steps": [],
                "fasta_mapping_quality": sample_mapping_quality if query_input_type == "fasta" else {},
                "vcf_mapping_quality": sample_mapping_quality if query_input_type in {"vcf", "fastq"} else {},
                "raw_sequence_mapping_quality": sample_mapping_quality,
            }

            current_node = root
            terminal_status = "started"
            terminal_reason = ""
            final_prediction: Optional[str] = None
            path_parts: List[str] = []
            visited = 0
            max_depth = max(1, int(hierarchy.get("n_levels", len(label_columns) or 1))) + 2

            while isinstance(current_node, dict) and current_node and visited < max_depth:
                visited += 1
                level_number = int(current_node.get("level_number", visited) or visited)
                label_column = str(current_node.get("label_column", f"level_{level_number}"))
                node_status = str(current_node.get("status", "unknown"))
                level_prefix = f"level{level_number}"
                node_key = self._hierarchy_node_key(current_node)

                row[f"{level_prefix}_label_column"] = label_column
                row[f"{level_prefix}_model_status"] = node_status

                if node_status == "constant":
                    prediction = str(current_node.get("constant_label", ""))
                    support: Optional[float] = 1.0 if prediction else None
                    confidence = "deterministic_branch"
                    confidence_note = (
                        "This branch had only one observed child state during training, so no model was fitted at this level."
                    )
                    markers: List[Dict[str, Any]] = []
                    evidence: Dict[str, Any] = {}
                    class_support: Dict[str, float] = {prediction: 1.0} if prediction else {}
                    decision_path: List[str] = []

                elif node_status == "success":
                    features, payload, ranked, importance = self._load_hierarchy_node(current_node)
                    X_node, alignment = align_to_training_features(sample_row, features)
                    alignment_by_node.setdefault(node_key, alignment)
                    pred, support_values, class_support_values = predict_labels_and_support(payload, X_node)
                    tree_paths = decision_tree_path_explanation(payload, X_node)
                    prediction = str(pred[0]) if pred else "unavailable"
                    support = support_values[0] if support_values else None
                    class_support = class_support_values[0] if class_support_values else {}
                    decision_path = tree_paths.get(sample_id, [])

                    markers = supporting_markers_for_sample(
                        X_node.loc[sample_id],
                        ranked_features=ranked,
                        model_importance=importance,
                        max_markers=max_markers,
                        feature_metadata=sample_feature_metadata,
                        available_features=raw_available_features,
                    )
                    evidence = summarize_feature_evidence_for_model(
                        sample_values=X_node.loc[sample_id],
                        features=features,
                        feature_metadata=sample_feature_metadata,
                        available_features=raw_available_features,
                    )
                    confidence, confidence_note = interpretation_confidence_for_level(
                        support=support,
                        evidence=evidence,
                        n_supporting_markers=len(markers),
                    )
                    row[f"n_{level_prefix}_supporting_markers"] = int(len(markers))
                    row.update(flatten_feature_evidence(level_prefix, evidence))

                else:
                    terminal_status = f"stopped_at_{node_status}"
                    terminal_reason = str(current_node.get("reason", "No trained or deterministic branch was available."))
                    row[f"{level_prefix}_stop_reason"] = terminal_reason
                    report_entry["hierarchy_steps"].append(
                        {
                            "level_number": level_number,
                            "label_column": label_column,
                            "node_status": node_status,
                            "stop_reason": terminal_reason,
                        }
                    )
                    break

                row[f"predicted_{level_prefix}_label"] = prediction
                row[f"{level_prefix}_support"] = support
                row[f"{level_prefix}_interpretation_confidence"] = confidence
                row[f"{level_prefix}_confidence_note"] = confidence_note
                row[f"{level_prefix}_node_key"] = node_key
                row[f"n_{level_prefix}_supporting_markers"] = int(len(markers))

                # Backward-compatible aliases for tools that still expect the
                # old two-level column names.
                if level_number == 1:
                    row["predicted_level1_identity"] = prediction
                    row["level1_support"] = support
                    row["n_level1_supporting_markers"] = int(len(markers))
                elif level_number == 2:
                    row["predicted_level2_resistance_profile"] = prediction
                    row["level2_support"] = support
                    row["level2_model_source"] = "hierarchy_node"
                    row["level2_target_label_column"] = label_column
                    row["n_level2_supporting_markers"] = int(len(markers))

                report_entry["hierarchy_steps"].append(
                    {
                        "level_number": level_number,
                        "label_column": label_column,
                        "prediction": prediction,
                        "support": support,
                        "class_support": class_support,
                        "node_status": node_status,
                        "node_key": node_key,
                        "interpretation_confidence": confidence,
                        "confidence_note": confidence_note,
                        "feature_evidence": evidence,
                        "supporting_markers": markers,
                        "decision_path": decision_path,
                    }
                )

                if prediction:
                    path_parts.append(f"{label_column}={prediction}")
                    final_prediction = prediction

                children = current_node.get("children", {})
                next_node = children.get(prediction) if isinstance(children, dict) else None
                if isinstance(next_node, dict) and next_node:
                    current_node = next_node
                    terminal_status = "continued"
                    continue

                terminal_status = "complete"
                terminal_reason = "No deeper child node exists for the predicted label."
                break

            if visited >= max_depth:
                terminal_status = "stopped_max_depth_guard"
                terminal_reason = "Traversal stopped by recursion guard."

            row["predicted_hierarchy_path"] = " > ".join(path_parts)
            row["predicted_terminal_label"] = final_prediction
            row["predicted_terminal_level"] = len(path_parts)
            row["hierarchy_terminal_status"] = terminal_status
            row["hierarchy_terminal_reason"] = terminal_reason
            report_entry.update(
                {
                    **row,
                    "predicted_hierarchy_path": row["predicted_hierarchy_path"],
                    "predicted_terminal_label": final_prediction,
                    "predicted_terminal_level": len(path_parts),
                    "hierarchy_terminal_status": terminal_status,
                    "hierarchy_terminal_reason": terminal_reason,
                }
            )
            rows.append(row)
            report_samples.append(report_entry)

        predictions = pd.DataFrame(rows)
        predictions_path = out / "query_predictions.csv"
        compact_path = out / "query_predictions_compact.tsv"
        readable_path = out / "query_predictions_readable.html"
        route_audit_path = out / "query_route_audit.json"
        alignment_path = out / "query_alignment_summary.json"

        predictions.to_csv(predictions_path, index=False)
        write_hierarchical_compact_predictions(predictions, compact_path)

        alignment_summary = {
            "mode": "multi_level_hierarchy_query_alignment",
            "hierarchy_label_columns": label_columns,
            "hierarchy_n_levels": int(hierarchy.get("n_levels", len(label_columns) or 0)),
            "alignment_by_node": alignment_by_node,
            "n_query_samples": int(X_raw.shape[0]),
            "n_query_features_raw": int(X_raw.shape[1]),
            "query_input_type": query_input_type,
            "fasta_mapping": raw_mapping_summary if query_input_type == "fasta" else None,
            "vcf_mapping": raw_mapping_summary if query_input_type in {"vcf", "fastq"} else None,
            "raw_sequence_mapping": raw_mapping_summary,
            "fastq_processing": fastq_processing_summary,
        }
        write_json(alignment_summary, alignment_path)

        route_audit = build_hierarchical_query_route_audit(
            registry_path=self.registry_path,
            query_input_type=query_input_type,
            report_samples=report_samples,
            alignment_by_node=alignment_by_node,
        )
        write_json(route_audit, route_audit_path)

        report = {
            "mode": "multi_level_hierarchy_query",
            "registry": str(self.registry_path),
            "genomic_input": str(genomic_path),
            "n_samples": int(len(report_samples)),
            "hierarchy_label_columns": label_columns,
            "samples": report_samples,
            "diagnostic_question": "Given this genomic evidence, where does the strain belong across the trained hierarchy, what phenotype is predicted at the terminal level, and which trained genomic markers support the interpretation?",
            "notes": [
                "Query mode is inference-only.",
                "Recursive hierarchy query traversal does not rerun central feature filtering, model selection, decision-tree training, or bootstrap confidence scoring.",
                "Each trainable hierarchy node aligns the query to that node's saved selected-feature space before prediction.",
                "Deterministic branches are followed when a training node had only one observed child label and therefore no model was fitted.",
                "For FASTA/VCF/FASTQ queries, NetworkParser reconstructs the saved trained feature space from the selected-feature manifest.",
                "Resolved baseline states encoded as 0 are valid trained-marker evidence for the overall query pattern.",
                "Unresolved, ambiguous, repeated, or non-training allele calls are encoded as 0 and explicitly reported in the mapping summary.",
            ],
            "artifacts": {
                "predictions_csv": str(predictions_path),
                "predictions_compact_tsv": str(compact_path),
                "predictions_readable_html": str(readable_path),
                "route_audit_json": str(route_audit_path),
                "alignment_summary_json": str(alignment_path),
                "report_json": str(out / "query_report.json"),
                "report_txt": str(out / "query_report.txt"),
                "fastq_processing_summary": (
                    str(out / "fastq_query_preprocessing" / "fastq_processing_summary.json")
                    if fastq_processing_summary is not None else None
                ),
            },
        }
        write_json(report, out / "query_report.json")
        write_hierarchical_text_report(report, out / "query_report.txt")
        write_hierarchical_readable_html_report(report, readable_path)

        logger.info(
            "Hierarchy query complete | predictions=%s | compact=%s | readable=%s",
            predictions_path,
            compact_path,
            readable_path,
        )
        return predictions

    def query(
        self,
        genomic_path: str,
        output_dir: str,
        ref_fasta: Optional[str] = None,
        max_markers: int = 10,
        n_jobs: Optional[int] = None,
        query_input_type: str = "auto",
        raw_sequence_mapping_mode: str = "auto",
    ) -> pd.DataFrame:
        out = ensure_dir(Path(output_dir))
        query_input_type = str(query_input_type or "auto").lower()

        # Query mode must reconstruct the *trained* feature space.  For FASTA,
        # VCF, and FASTQ-derived VCF input we therefore use the selected-feature
        # manifest and the union of registry features, instead of allowing a
        # single-sample query to rediscover/filter/collapse its own feature set.
        genomic_candidate = Path(genomic_path)
        fasta_suffixes = {".fa", ".fna", ".fasta", ".fas"}
        fastq_suffixes = (".fastq", ".fq", ".fastq.gz", ".fq.gz")
        vcf_suffixes = (".vcf", ".vcf.gz")

        if query_input_type == "auto":
            if genomic_candidate.is_file():
                lower_name = genomic_candidate.name.lower()
                if genomic_candidate.suffix.lower() in fasta_suffixes:
                    query_input_type = "fasta"
                elif lower_name.endswith(vcf_suffixes):
                    query_input_type = "vcf"
            elif genomic_candidate.is_dir():
                files = [p for p in genomic_candidate.iterdir() if p.is_file()]
                names = [p.name.lower() for p in files]
                if any(any(name.endswith(ext) for ext in fastq_suffixes) for name in names):
                    query_input_type = "fastq"
                elif any(any(name.endswith(ext) for ext in vcf_suffixes) for name in names):
                    query_input_type = "vcf"
                elif any(p.suffix.lower() in fasta_suffixes for p in files):
                    query_input_type = "fasta"
            if query_input_type == "auto":
                query_input_type = "matrix"

        if query_input_type in {"raw_sequence", "raw_fasta", "sequence"}:
            logger.warning("query_input_type=raw_sequence is deprecated; use query_input_type=fasta instead.")
            query_input_type = "fasta"

        raw_calls: Optional[pd.DataFrame] = None
        raw_mapping_summary: Optional[Dict[str, Any]] = None
        fastq_processing_summary: Optional[Dict[str, Any]] = None

        required_features = collect_required_features_from_registry(self.registry)
        manifest_path = resolve_registry_feature_manifest(self.registry, self.registry_base)

        def _require_feature_manifest(input_label: str) -> Path:
            if manifest_path is None:
                raise ValueError(
                    f"{input_label} query mode requires a selected-feature manifest in the model registry. "
                    "Retrain with a reference FASTA/GenBank so selected-feature context, REF/ALT, "
                    "and baseline allele metadata are saved."
                )
            manifest = load_feature_manifest(Path(manifest_path))
            context_columns = [
                col
                for col in ("Context_sequence", "Context_±40", "Context", "context_sequence")
                if col in manifest.columns
            ]
            context_present = 0
            if context_columns:
                context_present = int(
                    manifest[context_columns]
                    .astype(str)
                    .apply(lambda row: any(value.strip() for value in row), axis=1)
                    .sum()
                )
            logger.info(
                "Loaded selected-feature manifest | features=%d | context_present=%d",
                len(manifest),
                context_present,
            )
            return manifest_path

        if query_input_type == "fasta":
            resolved_manifest = _require_feature_manifest("FASTA")
            X_raw, raw_mapping_summary, raw_calls = encode_raw_sequence_query(
                raw_sequence_path=genomic_path,
                feature_manifest_path=str(resolved_manifest),
                features=required_features,
                output_dir=str(out / "fasta_query_encoding"),
                mapping_mode=raw_sequence_mapping_mode,
            )
            X_raw.index = X_raw.index.astype(str).map(normalize_sample_id)

        elif query_input_type == "vcf":
            resolved_manifest = _require_feature_manifest("VCF")
            X_raw, raw_mapping_summary, raw_calls = encode_vcf_query_from_manifest(
                vcf_path=genomic_path,
                feature_manifest_path=str(resolved_manifest),
                features=required_features,
                output_dir=str(out / "vcf_query_encoding"),
            )
            X_raw.index = X_raw.index.astype(str).map(normalize_sample_id)

        elif query_input_type == "fastq":
            if not ref_fasta:
                raise ValueError(
                    "FASTQ query mode requires --ref_fasta because reads must be aligned "
                    "and converted to VCF-derived genomic features before inference."
                )
            resolved_manifest = _require_feature_manifest("FASTQ")
            fastq_out = ensure_dir(out / "fastq_query_preprocessing")
            processor = FastqProcessor(
                config=self.config,
                fastq_dir=genomic_path,
                ref_genome=ref_fasta,
                output_dir=str(fastq_out),
                n_jobs=n_jobs,
            )
            vcf_dir, fastq_summary = processor.process_samples()
            fastq_processing_summary = asdict(fastq_summary)
            X_raw, raw_mapping_summary, raw_calls = encode_vcf_query_from_manifest(
                vcf_path=str(vcf_dir),
                feature_manifest_path=str(resolved_manifest),
                features=required_features,
                output_dir=str(out / "vcf_query_encoding"),
            )
            X_raw.index = X_raw.index.astype(str).map(normalize_sample_id)

        elif query_input_type == "matrix":
            X_raw = load_query_matrix(
                genomic_path=genomic_path,
                output_dir=out,
                config=self.config,
                ref_fasta=ref_fasta,
                n_jobs=n_jobs,
            )
        else:
            raise ValueError("query_input_type must be one of: auto, matrix, vcf, fasta, fastq")

        raw_available_features = set(map(str, X_raw.columns))

        raw_feature_metadata = feature_call_metadata_by_sample(raw_calls)
        if raw_feature_metadata:
            raw_feature_metadata = {
                normalize_sample_id(str(sample_id)): feature_map
                for sample_id, feature_map in raw_feature_metadata.items()
            }
        raw_sample_quality: Dict[str, Dict[str, Any]] = {}
        if isinstance(raw_mapping_summary, dict):
            for item in raw_mapping_summary.get("per_sample", []) or []:
                if isinstance(item, dict) and item.get("sample_id") is not None:
                    raw_sample_quality[normalize_sample_id(str(item.get("sample_id")))] = item

        if is_hierarchical_registry(self.registry):
            return self._query_hierarchy_from_matrix(
                X_raw=X_raw,
                raw_calls=raw_calls,
                raw_mapping_summary=raw_mapping_summary,
                raw_feature_metadata=raw_feature_metadata,
                raw_sample_quality=raw_sample_quality,
                raw_available_features=raw_available_features,
                out=out,
                genomic_path=genomic_path,
                query_input_type=query_input_type,
                fastq_processing_summary=fastq_processing_summary,
                max_markers=max_markers,
            )

        level1_features, level1_payload, level1_ranked, level1_importance = self._load_level1()
        X_l1, l1_alignment = align_to_training_features(X_raw, level1_features)
        l1_pred, l1_support, l1_class_support = predict_labels_and_support(level1_payload, X_l1)
        l1_tree_paths = decision_tree_path_explanation(level1_payload, X_l1)

        rows: List[Dict[str, Any]] = []
        report_samples: List[Dict[str, Any]] = []
        alignment_by_level2_source: Dict[str, Any] = {}

        for idx, sample_id in enumerate(X_l1.index.astype(str)):
            predicted_l1 = str(l1_pred[idx])
            level2_source, l2_features, l2_payload, l2_ranked, l2_importance = self._select_level2_payload(predicted_l1)
            X_l2, l2_alignment = align_to_training_features(X_raw.loc[[sample_id]], l2_features)
            alignment_by_level2_source.setdefault(level2_source, l2_alignment)

            l2_pred, l2_support, l2_class_support = predict_labels_and_support(l2_payload, X_l2)
            l2_tree_paths = decision_tree_path_explanation(l2_payload, X_l2)

            sample_feature_metadata = raw_feature_metadata.get(sample_id, {})
            sample_mapping_quality = raw_sample_quality.get(sample_id, {})
            l1_markers = supporting_markers_for_sample(
                X_l1.loc[sample_id],
                ranked_features=level1_ranked,
                model_importance=level1_importance,
                max_markers=max_markers,
                feature_metadata=sample_feature_metadata,
                available_features=raw_available_features,
            )
            l2_markers = supporting_markers_for_sample(
                X_l2.loc[sample_id],
                ranked_features=l2_ranked,
                model_importance=l2_importance,
                max_markers=max_markers,
                feature_metadata=sample_feature_metadata,
                available_features=raw_available_features,
            )

            # Model-specific query evidence.  This is intentionally separate
            # from union-level FASTA/VCF recovery so we can diagnose whether
            # Level 1 or the selected Level 2 model received active marker
            # evidence.
            l1_evidence = summarize_feature_evidence_for_model(
                sample_values=X_l1.loc[sample_id],
                features=level1_features,
                feature_metadata=sample_feature_metadata,
                available_features=raw_available_features,
            )
            l2_evidence = summarize_feature_evidence_for_model(
                sample_values=X_l2.loc[sample_id],
                features=l2_features,
                feature_metadata=sample_feature_metadata,
                available_features=raw_available_features,
            )

            l1_confidence, l1_confidence_note = interpretation_confidence_for_level(
                support=l1_support[idx],
                evidence=l1_evidence,
                n_supporting_markers=len(l1_markers),
            )
            l2_confidence, l2_confidence_note = interpretation_confidence_for_level(
                support=l2_support[0],
                evidence=l2_evidence,
                n_supporting_markers=len(l2_markers),
            )

            row = {
                "sample_id": sample_id,
                "predicted_level1_identity": predicted_l1,
                "level1_support": l1_support[idx],
                "predicted_level2_resistance_profile": str(l2_pred[0]),
                "level2_support": l2_support[0],
                "level2_model_source": level2_source,
                "level2_target_label_column": (
                    self.registry.get("level2", {}).get("global_label_column")
                    if level2_source == "global_fallback"
                    else self.registry.get("level2", {}).get("label_column")
                ),
                "n_level1_supporting_markers": int(len(l1_markers)),
                "n_level2_supporting_markers": int(len(l2_markers)),
                "level1_interpretation_confidence": l1_confidence,
                "level1_confidence_note": l1_confidence_note,
                "level2_interpretation_confidence": l2_confidence,
                "level2_confidence_note": l2_confidence_note,
                **flatten_feature_evidence("level1", l1_evidence),
                **flatten_feature_evidence("level2", l2_evidence),
                "query_marker_recovery_status": sample_mapping_quality.get("marker_recovery_status"),
                "query_marker_recovery_reason": sample_mapping_quality.get("marker_recovery_reason"),
                "query_active_marker_evidence_status": sample_mapping_quality.get("active_marker_evidence_status"),
                "query_active_marker_evidence_reason": sample_mapping_quality.get("active_marker_evidence_reason"),
                "query_unique_mapped_fraction": sample_mapping_quality.get("unique_mapped_fraction"),
                "query_active_feature_fraction": sample_mapping_quality.get("active_feature_fraction"),
                "query_n_encoded_active_features": sample_mapping_quality.get("n_encoded_active_features"),
                "query_n_resolved_features": sample_mapping_quality.get("n_resolved_features"),
                "query_resolved_feature_fraction": sample_mapping_quality.get("resolved_feature_fraction"),
                "query_n_resolved_baseline_features": sample_mapping_quality.get("n_resolved_baseline_features"),
                "query_resolved_baseline_feature_fraction": sample_mapping_quality.get("resolved_baseline_feature_fraction"),
                "query_resolved_marker_evidence_status": sample_mapping_quality.get("resolved_marker_evidence_status"),
                "query_resolved_marker_evidence_reason": sample_mapping_quality.get("resolved_marker_evidence_reason"),
                "query_n_unresolved_or_missing_calls": sample_mapping_quality.get("n_unresolved_or_missing_context_calls", sample_mapping_quality.get("n_unresolved_or_missing_calls")),
                "query_n_multi_hit_calls": sample_mapping_quality.get("n_multi_hit_calls"),
                "query_n_ambiguous_base_calls": sample_mapping_quality.get("n_ambiguous_base_calls"),
                "query_n_non_training_allele_calls": sample_mapping_quality.get("n_non_training_allele_calls"),
            }
            rows.append(row)

            report_samples.append(
                {
                    **row,
                    "level1_class_support": l1_class_support[idx],
                    "level2_class_support": l2_class_support[0],
                    "level1_feature_evidence": l1_evidence,
                    "level2_feature_evidence": l2_evidence,
                    "level1_supporting_markers": l1_markers,
                    "level2_supporting_markers": l2_markers,
                    "fasta_mapping_quality": sample_mapping_quality if query_input_type == "fasta" else {},
                    "vcf_mapping_quality": sample_mapping_quality if query_input_type in {"vcf", "fastq"} else {},
                    "raw_sequence_mapping_quality": sample_mapping_quality,
                    "level1_decision_path": l1_tree_paths.get(sample_id, []),
                    "level2_decision_path": l2_tree_paths.get(sample_id, []),
                }
            )

        predictions = pd.DataFrame(rows)
        predictions_path = out / "query_predictions.csv"
        compact_path = out / "query_predictions_compact.tsv"
        readable_path = out / "query_predictions_readable.html"
        route_audit_path = out / "query_route_audit.json"

        predictions.to_csv(predictions_path, index=False)
        write_compact_predictions(predictions, compact_path)

        alignment_summary = {
            "level1": l1_alignment,
            "level2_by_source": alignment_by_level2_source,
            "n_query_samples": int(X_raw.shape[0]),
            "n_query_features_raw": int(X_raw.shape[1]),
            "query_input_type": query_input_type,
            "fasta_mapping": raw_mapping_summary if query_input_type == "fasta" else None,
            "vcf_mapping": raw_mapping_summary if query_input_type in {"vcf", "fastq"} else None,
            "raw_sequence_mapping": raw_mapping_summary,
            "fastq_processing": fastq_processing_summary,
        }
        write_json(alignment_summary, out / "query_alignment_summary.json")

        route_audit = build_query_route_audit(
            registry_path=self.registry_path,
            query_input_type=query_input_type,
            report_samples=report_samples,
            l1_alignment=l1_alignment,
            alignment_by_level2_source=alignment_by_level2_source,
        )
        write_json(route_audit, route_audit_path)

        report = {
            "mode": "two_level_query",
            "registry": str(self.registry_path),
            "genomic_input": str(genomic_path),
            "n_samples": int(len(report_samples)),
            "samples": report_samples,
            "diagnostic_question": "Given this genomic evidence, where does the strain belong, what phenotype is predicted, and which trained genomic markers support the interpretation?",
            "notes": [
                "Query mode is inference-only.",
                "RF-FDR feature selection is not rerun on query samples.",
                "For FASTA/VCF/FASTQ queries, NetworkParser reconstructs the saved trained feature space from the selected-feature manifest.",
                "Query mode does not rerun cohort-level matrix refinement, redundancy reduction, RF-FDR, model selection, or tree construction.",
                "For FASTA queries, saved context sequences are mapped to the query genome and the centre nucleotide is encoded with the saved baseline/REF/ALT rule.",
                "For VCF queries, saved feature coordinates are looked up directly in the query VCF; absent variant records are treated as reference-state calls and encoded with the same saved rule.",
                "Resolved baseline states encoded as 0 are valid trained-marker evidence for the overall query pattern.",
                "Unresolved, ambiguous, repeated, or non-training allele calls are encoded as 0 and explicitly reported in the mapping summary.",
            ],
            "artifacts": {
                "predictions_csv": str(predictions_path),
                "predictions_compact_tsv": str(compact_path),
                "predictions_readable_html": str(readable_path),
                "route_audit_json": str(route_audit_path),
                "alignment_summary_json": str(out / "query_alignment_summary.json"),
                "report_json": str(out / "query_report.json"),
                "report_txt": str(out / "query_report.txt"),
                "fastq_processing_summary": (
                    str(out / "fastq_query_preprocessing" / "fastq_processing_summary.json")
                    if fastq_processing_summary is not None else None
                ),
            },
        }
        write_json(report, out / "query_report.json")
        write_text_report(report, out / "query_report.txt")
        write_readable_html_report(report, readable_path)

        logger.info(
            "Query complete | predictions=%s | compact=%s | readable=%s",
            predictions_path,
            compact_path,
            readable_path,
        )
        return predictions


def write_compact_predictions(predictions: pd.DataFrame, path: Path) -> None:
    """Write a terminal-friendly compact TSV with key prediction/evidence fields."""
    preferred = [
        "sample_id",
        "predicted_level1_identity",
        "level1_support",
        "level1_interpretation_confidence",
        "level1_n_supporting_markers",
        "n_level1_supporting_markers",
        "level1_n_resolved_features",
        "level1_resolved_feature_fraction",
        "level1_n_resolved_baseline_features",
        "level1_n_active_features",
        "level1_resolved_marker_evidence_status",
        "level1_nonbaseline_evidence_status",
        "predicted_level2_resistance_profile",
        "level2_support",
        "level2_model_source",
        "level2_interpretation_confidence",
        "n_level2_supporting_markers",
        "level2_n_resolved_features",
        "level2_resolved_feature_fraction",
        "level2_n_resolved_baseline_features",
        "level2_n_active_features",
        "level2_resolved_marker_evidence_status",
        "level2_nonbaseline_evidence_status",
        "query_marker_recovery_status",
        "query_resolved_marker_evidence_status",
        "query_n_unresolved_or_missing_calls",
        "query_n_multi_hit_calls",
        "query_n_ambiguous_base_calls",
        "query_n_non_training_allele_calls",
    ]
    cols = [col for col in preferred if col in predictions.columns]
    if not cols:
        cols = list(predictions.columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.loc[:, cols].to_csv(path, sep="\t", index=False)


def build_query_route_audit(
    *,
    registry_path: Path,
    query_input_type: str,
    report_samples: List[Dict[str, Any]],
    l1_alignment: Dict[str, Any],
    alignment_by_level2_source: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a compact audit of how each sample moved through query inference."""
    routes: List[Dict[str, Any]] = []
    for sample in report_samples:
        routes.append(
            {
                "sample_id": sample.get("sample_id"),
                "predicted_level1_identity": sample.get("predicted_level1_identity"),
                "level1_support": sample.get("level1_support"),
                "level1_interpretation_confidence": sample.get("level1_interpretation_confidence"),
                "level1_resolved_marker_evidence_status": sample.get("level1_resolved_marker_evidence_status"),
                "level1_nonbaseline_evidence_status": sample.get("level1_nonbaseline_evidence_status"),
                "predicted_level2_resistance_profile": sample.get("predicted_level2_resistance_profile"),
                "level2_support": sample.get("level2_support"),
                "level2_model_source": sample.get("level2_model_source"),
                "level2_target_label_column": sample.get("level2_target_label_column"),
                "level2_interpretation_confidence": sample.get("level2_interpretation_confidence"),
                "level2_resolved_marker_evidence_status": sample.get("level2_resolved_marker_evidence_status"),
                "level2_nonbaseline_evidence_status": sample.get("level2_nonbaseline_evidence_status"),
                "query_marker_recovery_status": sample.get("query_marker_recovery_status"),
                "query_resolved_marker_evidence_status": sample.get("query_resolved_marker_evidence_status"),
            }
        )

    return {
        "mode": "two_level_query_route_audit",
        "registry": str(registry_path),
        "query_input_type": query_input_type,
        "n_samples": int(len(report_samples)),
        "level1_alignment_status": l1_alignment.get("alignment_status") if isinstance(l1_alignment, dict) else None,
        "level2_alignment_by_source": alignment_by_level2_source,
        "routes": routes,
    }


def write_hierarchical_compact_predictions(predictions: pd.DataFrame, path: Path) -> None:
    """Write a compact TSV for arbitrary-depth hierarchy predictions."""
    base_cols = [
        "sample_id",
        "predicted_hierarchy_path",
        "predicted_terminal_label",
        "predicted_terminal_level",
        "hierarchy_terminal_status",
        "hierarchy_terminal_reason",
    ]
    level_cols: List[str] = []
    for col in predictions.columns:
        if col.startswith("predicted_level") or col.endswith("_support") or col.endswith("_interpretation_confidence"):
            level_cols.append(col)
    evidence_cols = [
        col for col in predictions.columns
        if col.endswith("_resolved_marker_evidence_status")
        or col.endswith("_nonbaseline_evidence_status")
        or col.startswith("n_level") and col.endswith("_supporting_markers")
    ]
    query_cols = [
        "query_marker_recovery_status",
        "query_resolved_marker_evidence_status",
        "query_n_unresolved_or_missing_calls",
        "query_n_multi_hit_calls",
        "query_n_ambiguous_base_calls",
        "query_n_non_training_allele_calls",
    ]
    ordered: List[str] = []
    for col in base_cols + sorted(level_cols) + sorted(evidence_cols) + query_cols:
        if col in predictions.columns and col not in ordered:
            ordered.append(col)
    if not ordered:
        ordered = list(predictions.columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.loc[:, ordered].to_csv(path, sep="\t", index=False)


def build_hierarchical_query_route_audit(
    *,
    registry_path: Path,
    query_input_type: str,
    report_samples: List[Dict[str, Any]],
    alignment_by_node: Dict[str, Any],
) -> Dict[str, Any]:
    """Build an audit payload for recursive hierarchy traversal."""
    routes: List[Dict[str, Any]] = []
    for sample in report_samples:
        routes.append(
            {
                "sample_id": sample.get("sample_id"),
                "predicted_hierarchy_path": sample.get("predicted_hierarchy_path"),
                "predicted_terminal_label": sample.get("predicted_terminal_label"),
                "predicted_terminal_level": sample.get("predicted_terminal_level"),
                "hierarchy_terminal_status": sample.get("hierarchy_terminal_status"),
                "hierarchy_terminal_reason": sample.get("hierarchy_terminal_reason"),
                "steps": [
                    {
                        "level_number": step.get("level_number"),
                        "label_column": step.get("label_column"),
                        "prediction": step.get("prediction"),
                        "support": step.get("support"),
                        "node_status": step.get("node_status"),
                        "node_key": step.get("node_key"),
                        "interpretation_confidence": step.get("interpretation_confidence"),
                        "resolved_marker_evidence_status": (step.get("feature_evidence") or {}).get("resolved_marker_evidence_status"),
                        "nonbaseline_evidence_status": (step.get("feature_evidence") or {}).get("nonbaseline_evidence_status"),
                    }
                    for step in sample.get("hierarchy_steps", []) or []
                ],
                "query_marker_recovery_status": sample.get("query_marker_recovery_status"),
                "query_resolved_marker_evidence_status": sample.get("query_resolved_marker_evidence_status"),
            }
        )

    return {
        "mode": "multi_level_hierarchy_query_route_audit",
        "registry": str(registry_path),
        "query_input_type": query_input_type,
        "n_samples": int(len(report_samples)),
        "alignment_by_node": alignment_by_node,
        "routes": routes,
    }


def write_hierarchical_readable_html_report(report: Dict[str, Any], path: Path) -> None:
    """Write a browser-readable report for arbitrary-depth hierarchy query."""
    def _marker_table(markers: List[Dict[str, Any]]) -> str:
        rows: List[str] = []
        for marker in markers[:10]:
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(marker.get('feature', '')))}</td>"
                f"<td>{html.escape(str(marker.get('value', '')))}</td>"
                f"<td>{html.escape(str(marker.get('evidence_role', '')))}</td>"
                f"<td>{html.escape(str(marker.get('allele_call', '')))}</td>"
                f"<td>{html.escape(str(marker.get('observed_allele', '')))}</td>"
                "</tr>"
            )
        if not rows:
            return ""
        return (
            "<table><thead><tr><th>Feature</th><th>Value</th><th>Evidence role</th>"
            "<th>Allele call</th><th>Observed</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    cards: List[str] = []
    for sample in report.get("samples", []) or []:
        sid = html.escape(str(sample.get("sample_id", "NA")))
        steps_html: List[str] = []
        for step in sample.get("hierarchy_steps", []) or []:
            level = html.escape(str(step.get("level_number", "NA")))
            label = html.escape(str(step.get("label_column", "NA")))
            evidence = step.get("feature_evidence") or {}
            markers = step.get("supporting_markers") or []
            steps_html.append(
                "<div class='step'>"
                f"<h3>Level {level}: {label}</h3>"
                + _html_kv("Prediction", step.get("prediction"))
                + _html_kv("Support", step.get("support"))
                + _html_kv("Node status", step.get("node_status"))
                + _html_kv("Interpretation confidence", step.get("interpretation_confidence"))
                + _html_kv("Resolved marker evidence", evidence.get("resolved_marker_evidence_status"))
                + _html_kv("Resolved features", evidence.get("n_resolved_features"))
                + _html_kv("Resolved baseline features", evidence.get("n_resolved_baseline_features"))
                + _html_kv("Nonbaseline features", evidence.get("n_active_features"))
                + _marker_table(markers)
                + "</div>"
            )

        cards.append(
            "<section class='card'>"
            f"<h2>Sample: {sid}</h2>"
            + _html_kv("Predicted hierarchy path", sample.get("predicted_hierarchy_path"))
            + _html_kv("Terminal label", sample.get("predicted_terminal_label"))
            + _html_kv("Terminal status", sample.get("hierarchy_terminal_status"))
            + "<div class='steps'>"
            + "".join(steps_html)
            + "</div></section>"
        )

    notes = "".join(f"<li>{html.escape(str(note))}</li>" for note in report.get("notes", []) or [])
    labels = ", ".join(html.escape(str(x)) for x in report.get("hierarchy_label_columns", []) or [])
    question = html.escape(str(report.get("diagnostic_question", "")))
    document = f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>NetworkParser hierarchy query report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; line-height: 1.45; background: #f7f7f7; color: #222; }}
.card {{ background: white; border: 1px solid #ddd; border-radius: 12px; padding: 1rem 1.25rem; margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
.step {{ border-left: 4px solid #ddd; padding: 0.75rem 1rem; margin: 0.75rem 0; background: #fbfbfb; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 0.5rem; font-size: 0.92rem; }}
th, td {{ border: 1px solid #ddd; padding: 0.4rem; text-align: left; vertical-align: top; }}
th {{ background: #f0f0f0; }}
.question {{ background: #eef5ff; border-left: 4px solid #6699cc; padding: 0.75rem 1rem; }}
</style>
</head>
<body>
<h1>NetworkParser multi-level hierarchy query report</h1>
<p class=\"question\"><strong>Diagnostic question:</strong> {question}</p>
<p><strong>Hierarchy labels:</strong> {labels}</p>
<p>Samples queried: {html.escape(str(report.get('n_samples', 0)))}</p>
{''.join(cards)}
<h2>Notes</h2>
<ul>{notes}</ul>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def write_hierarchical_text_report(report: Dict[str, Any], path: Path) -> None:
    """Write a plain-text arbitrary-depth hierarchy query report."""
    lines: List[str] = []
    lines.append("NetworkParser multi-level hierarchy query report")
    lines.append("=" * 54)
    lines.append(f"Samples queried: {report.get('n_samples', 0)}")
    if report.get("hierarchy_label_columns"):
        lines.append("Hierarchy labels: " + " > ".join(map(str, report.get("hierarchy_label_columns", []))))
    if report.get("diagnostic_question"):
        lines.append("")
        lines.append("Diagnostic question")
        lines.append("-------------------")
        lines.append(str(report.get("diagnostic_question")))
    lines.append("")

    for sample in report.get("samples", []) or []:
        lines.append(f"Sample: {sample.get('sample_id')}")
        lines.append("-" * (8 + len(str(sample.get("sample_id", "")))))
        lines.append(f"Predicted hierarchy path: {sample.get('predicted_hierarchy_path')}")
        lines.append(f"Terminal label: {sample.get('predicted_terminal_label')}")
        lines.append(f"Terminal status: {sample.get('hierarchy_terminal_status')}")
        if sample.get("hierarchy_terminal_reason"):
            lines.append(f"Terminal reason: {sample.get('hierarchy_terminal_reason')}")
        lines.append("")

        for step in sample.get("hierarchy_steps", []) or []:
            lines.append(f"Level {step.get('level_number')} — {step.get('label_column')}")
            lines.append(f"  Node status: {step.get('node_status')}")
            if step.get("prediction") is not None:
                lines.append(f"  Prediction: {step.get('prediction')}")
            if step.get("support") is not None:
                try:
                    lines.append(f"  Support: {float(step.get('support')):.4f}")
                except Exception:
                    lines.append(f"  Support: {step.get('support')}")
            if step.get("interpretation_confidence"):
                lines.append(f"  Interpretation confidence: {step.get('interpretation_confidence')}")
            evidence = step.get("feature_evidence") or {}
            if evidence:
                lines.append(f"  Selected features: {evidence.get('n_selected_features', 'NA')}")
                lines.append(f"  Resolved trained-marker states: {evidence.get('n_resolved_features', 'NA')}")
                lines.append(f"  Resolved baseline states: {evidence.get('n_resolved_baseline_features', 'NA')}")
                lines.append(f"  Nonbaseline features: {evidence.get('n_active_features', 'NA')}")
                if evidence.get("resolved_marker_evidence_status"):
                    lines.append(f"  Resolved marker evidence: {evidence.get('resolved_marker_evidence_status')}")
            markers = step.get("supporting_markers") or []
            if markers:
                lines.append("  Supporting markers:")
                for marker in markers[:10]:
                    extra = f" | role={marker.get('evidence_role', 'NA')}"
                    if marker.get("observed_allele"):
                        extra += f" | observed={marker.get('observed_allele')} | status={marker.get('mapping_status', '')}"
                    lines.append(f"    - {marker.get('feature')} = {marker.get('value')}{extra}")
            lines.append("")
        lines.append("")

    lines.append("Notes")
    lines.append("-----")
    for note in report.get("notes", []) or []:
        lines.append(f"- {note}")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _html_kv(label: str, value: Any) -> str:
    value_text = "NA" if value is None else str(value)
    return f"<div><strong>{html.escape(label)}:</strong> {html.escape(value_text)}</div>"


def write_readable_html_report(report: Dict[str, Any], path: Path) -> None:
    """Write a browser-readable query report with one card per sample."""
    cards: List[str] = []
    for sample in report.get("samples", []) or []:
        sid = html.escape(str(sample.get("sample_id", "NA")))
        marker_items: List[str] = []
        for level_key, title in (("level1_supporting_markers", "Level 1 supporting markers"), ("level2_supporting_markers", "Level 2 supporting markers")):
            markers = sample.get(level_key, []) or []
            rows = []
            for marker in markers[:10]:
                feature = html.escape(str(marker.get("feature", "")))
                value = html.escape(str(marker.get("value", "")))
                role = html.escape(str(marker.get("evidence_role", "")))
                call = html.escape(str(marker.get("allele_call", "")))
                observed = html.escape(str(marker.get("observed_allele", "")))
                rows.append(
                    f"<tr><td>{feature}</td><td>{value}</td><td>{role}</td><td>{call}</td><td>{observed}</td></tr>"
                )
            if rows:
                marker_items.append(
                    f"<h4>{html.escape(title)}</h4>"
                    "<table><thead><tr><th>Feature</th><th>Value</th><th>Evidence role</th><th>Allele call</th><th>Observed</th></tr></thead>"
                    f"<tbody>{''.join(rows)}</tbody></table>"
                )

        cards.append(
            "<section class='card'>"
            f"<h2>Sample: {sid}</h2>"
            "<div class='grid'>"
            "<div><h3>Level 1</h3>"
            + _html_kv("Prediction", sample.get("predicted_level1_identity"))
            + _html_kv("Support", sample.get("level1_support"))
            + _html_kv("Interpretation confidence", sample.get("level1_interpretation_confidence"))
            + _html_kv("Resolved marker evidence", sample.get("level1_resolved_marker_evidence_status"))
            + _html_kv("Resolved features", sample.get("level1_n_resolved_features"))
            + _html_kv("Resolved baseline features", sample.get("level1_n_resolved_baseline_features"))
            + _html_kv("Nonbaseline features", sample.get("level1_n_active_features"))
            + "</div>"
            "<div><h3>Level 2</h3>"
            + _html_kv("Prediction", sample.get("predicted_level2_resistance_profile"))
            + _html_kv("Support", sample.get("level2_support"))
            + _html_kv("Model source", sample.get("level2_model_source"))
            + _html_kv("Interpretation confidence", sample.get("level2_interpretation_confidence"))
            + _html_kv("Resolved marker evidence", sample.get("level2_resolved_marker_evidence_status"))
            + _html_kv("Resolved features", sample.get("level2_n_resolved_features"))
            + _html_kv("Resolved baseline features", sample.get("level2_n_resolved_baseline_features"))
            + _html_kv("Nonbaseline features", sample.get("level2_n_active_features"))
            + "</div></div>"
            + "".join(marker_items)
            + "</section>"
        )

    notes = "".join(f"<li>{html.escape(str(note))}</li>" for note in report.get("notes", []) or [])
    question = html.escape(str(report.get("diagnostic_question", "")))
    document = f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>NetworkParser query report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; line-height: 1.45; background: #f7f7f7; color: #222; }}
.card {{ background: white; border: 1px solid #ddd; border-radius: 12px; padding: 1rem 1.25rem; margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 0.5rem; font-size: 0.92rem; }}
th, td {{ border: 1px solid #ddd; padding: 0.4rem; text-align: left; vertical-align: top; }}
th {{ background: #f0f0f0; }}
.question {{ background: #eef5ff; border-left: 4px solid #6699cc; padding: 0.75rem 1rem; }}
</style>
</head>
<body>
<h1>NetworkParser two-level query report</h1>
<p class=\"question\"><strong>Diagnostic question:</strong> {question}</p>
<p>Samples queried: {html.escape(str(report.get('n_samples', 0)))}</p>
{''.join(cards)}
<h2>Notes</h2>
<ul>{notes}</ul>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def write_text_report(report: Dict[str, Any], path: Path) -> None:
    lines: List[str] = []
    lines.append("NetworkParser two-level query report")
    lines.append("=" * 42)
    lines.append(f"Samples queried: {report.get('n_samples', 0)}")
    if report.get("diagnostic_question"):
        lines.append("")
        lines.append("Diagnostic question")
        lines.append("-------------------")
        lines.append(str(report.get("diagnostic_question")))
    lines.append("")

    def _evidence_lines(prefix: str, evidence: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        if not evidence:
            return out
        out.append("  Level-specific marker evidence:")
        out.append(f"    Selected features: {evidence.get('n_selected_features', 'NA')}")
        out.append(f"    Nonbaseline features: {evidence.get('n_active_features', 'NA')}")
        out.append(f"    Resolved trained-marker states: {evidence.get('n_resolved_features', 'NA')}")
        out.append(f"    Resolved baseline states: {evidence.get('n_resolved_baseline_features', 'NA')}")
        if evidence.get("resolved_feature_fraction") is not None:
            out.append(f"    Resolved feature fraction: {float(evidence.get('resolved_feature_fraction')):.4f}")
        if evidence.get("unique_mapped_fraction") is not None:
            out.append(f"    Unique recovery fraction: {float(evidence.get('unique_mapped_fraction')):.4f}")
        out.append(f"    Marker recovery: {evidence.get('marker_recovery_status', 'NA')}")
        out.append(f"    Resolved marker pattern: {evidence.get('resolved_marker_evidence_status', 'NA')}")
        out.append(f"    Nonbaseline evidence: {evidence.get('nonbaseline_evidence_status', evidence.get('active_marker_evidence_status', 'NA'))}")
        if evidence.get("n_zero_fill_caution_features", 0):
            out.append(f"    Zero-fill caution states: {evidence.get('n_zero_fill_caution_features')}")
        if evidence.get("n_multi_hit_calls", 0):
            out.append(f"    Multi-hit caution states: {evidence.get('n_multi_hit_calls')}")
        if evidence.get("n_ambiguous_base_calls", 0):
            out.append(f"    Ambiguous-base caution states: {evidence.get('n_ambiguous_base_calls')}")
        if evidence.get("n_non_training_allele_calls", 0):
            out.append(f"    Non-training-allele caution states: {evidence.get('n_non_training_allele_calls')}")
        return out

    def _marker_lines(markers: List[Dict[str, Any]]) -> List[str]:
        out: List[str] = []
        for marker in markers[:10]:
            role = marker.get("evidence_role") or "NA"
            extra = f" | role={role}"
            if marker.get("observed_allele"):
                quality = marker.get("mapping_quality") or marker.get("allele_call") or ""
                quality_txt = f" | quality={quality}" if quality else ""
                extra += f" | observed={marker.get('observed_allele')} | status={marker.get('mapping_status', '')}{quality_txt}"
            out.append(f"    - {marker.get('feature')} = {marker.get('value')}{extra}")
        return out

    for sample in report.get("samples", []):
        lines.append(f"Sample: {sample.get('sample_id')}")
        lines.append("-" * (8 + len(str(sample.get("sample_id", "")))))
        if sample.get("fasta_mapping_quality") or sample.get("vcf_mapping_quality") or sample.get("raw_sequence_mapping_quality"):
            rq = sample.get("fasta_mapping_quality") or sample.get("vcf_mapping_quality") or sample.get("raw_sequence_mapping_quality") or {}
            label = "FASTA context recovery" if sample.get("fasta_mapping_quality") else "VCF trained-feature recovery"
            lines.append(label)
            lines.append(f"  Marker recovery: {rq.get('marker_recovery_status', 'NA')}")
            if rq.get("resolved_marker_evidence_status"):
                lines.append(f"  Resolved marker pattern: {rq.get('resolved_marker_evidence_status', 'NA')}")
            lines.append(f"  Nonbaseline marker evidence: {rq.get('active_marker_evidence_status', 'NA')}")
            lines.append(f"  Unique mapped/resolved calls: {rq.get('n_unique_mapped_calls', 0)} / {rq.get('n_feature_calls', 0)}")
            if rq.get("n_resolved_features") is not None:
                lines.append(f"  Resolved trained-marker calls: {rq.get('n_resolved_features', 0)}")
            if rq.get("n_resolved_baseline_features") is not None:
                lines.append(f"  Resolved baseline calls encoded as 0: {rq.get('n_resolved_baseline_features', 0)}")
            lines.append(f"  Active encoded calls: {rq.get('n_encoded_active_features', 0)}")
            if rq.get("n_multi_hit_calls", 0):
                lines.append(f"  Multi-hit contexts/coordinates filled as 0: {rq.get('n_multi_hit_calls')}")
            if rq.get("n_ambiguous_base_calls", 0):
                lines.append(f"  Ambiguous-base calls filled as 0: {rq.get('n_ambiguous_base_calls')}")
            if rq.get("n_non_training_allele_calls", 0):
                lines.append(f"  Non-training alleles filled as 0: {rq.get('n_non_training_allele_calls')}")
            if rq.get("n_unresolved_or_missing_context_calls", 0):
                lines.append(f"  Unresolved/missing contexts filled as 0: {rq.get('n_unresolved_or_missing_context_calls')}")
            lines.append("")

        lines.append("Level 1 — strain/sample placement")
        lines.append(f"  Prediction: {sample.get('predicted_level1_identity')}")
        if sample.get("level1_support") is not None:
            lines.append(f"  Support: {float(sample.get('level1_support')):.4f}")
        if sample.get("level1_interpretation_confidence"):
            lines.append(f"  Interpretation confidence: {sample.get('level1_interpretation_confidence')}")
            if sample.get("level1_confidence_note"):
                lines.append(f"  Confidence note: {sample.get('level1_confidence_note')}")
        lines.extend(_evidence_lines("level1", sample.get("level1_feature_evidence") or {}))
        if sample.get("level1_supporting_markers"):
            lines.append("  Supporting markers:")
            lines.extend(_marker_lines(sample.get("level1_supporting_markers", [])))
        if sample.get("level1_decision_path"):
            lines.append("  Decision path:")
            for rule in sample.get("level1_decision_path", []):
                lines.append(f"    - {rule}")

        lines.append("")
        lines.append("Level 2 — resistance profile")
        lines.append(f"  Prediction: {sample.get('predicted_level2_resistance_profile')}")
        if sample.get("level2_support") is not None:
            lines.append(f"  Support: {float(sample.get('level2_support')):.4f}")
        lines.append(f"  Model source: {sample.get('level2_model_source')}")
        if sample.get("level2_target_label_column"):
            lines.append(f"  Target label column: {sample.get('level2_target_label_column')}")
        if sample.get("level2_interpretation_confidence"):
            lines.append(f"  Interpretation confidence: {sample.get('level2_interpretation_confidence')}")
            if sample.get("level2_confidence_note"):
                lines.append(f"  Confidence note: {sample.get('level2_confidence_note')}")
        lines.extend(_evidence_lines("level2", sample.get("level2_feature_evidence") or {}))
        if sample.get("level2_supporting_markers"):
            lines.append("  Supporting markers:")
            lines.extend(_marker_lines(sample.get("level2_supporting_markers", [])))
        if sample.get("level2_decision_path"):
            lines.append("  Decision path:")
            for rule in sample.get("level2_decision_path", []):
                lines.append(f"    - {rule}")
        lines.append("")

    lines.append("Notes")
    lines.append("-----")
    for note in report.get("notes", []):
        lines.append(f"- {note}")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply trained two-level or multi-level NetworkParser models to new strain/sample input.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--genomic", required=True, help="New genomic matrix file or VCF directory.")
    parser.add_argument("--registry", required=True, help="Path to two_level_model_registry.json or hierarchical_model_registry.json from training.")
    parser.add_argument("--output_dir", required=True, help="Directory for query outputs.")
    parser.add_argument("--config", default=None, help="Optional JSON config override file.")
    parser.add_argument("--ref_fasta", default=None, help="Optional reference FASTA for VCF parsing context.")
    parser.add_argument("--max_markers", type=int, default=10, help="Maximum supporting markers to show per level per sample.")
    parser.add_argument("--n_jobs", type=int, default=None, help="Runtime worker override.")
    parser.add_argument(
        "--query_input_type",
        choices=["auto", "matrix", "vcf", "fasta", "raw_sequence", "fastq"],
        default="auto",
        help="Interpret --genomic as a prebuilt matrix/VCF input, FASTA DNA, or paired FASTQ reads. raw_sequence remains a deprecated alias for fasta.",
    )
    parser.add_argument(
        "--fasta_mapping_mode",
        "--raw_sequence_mapping_mode",
        dest="raw_sequence_mapping_mode",
        choices=["auto", "blast", "exact"],
        default="auto",
        help="How FASTA query sequences should be mapped to selected feature contexts. The old --raw_sequence_mapping_mode option remains as an alias.",
    )
    parser.add_argument("--fastq_max_parallel_samples", type=int, default=None)
    parser.add_argument("--fastq_threads", type=int, default=None)
    parser.add_argument("--fastq_memory_per_sample_mb", type=int, default=None)
    parser.add_argument("--fastq_clean_intermediates", action="store_true")
    parser.add_argument("--fastq_no_auto_index_reference", action="store_true")
    parser.add_argument("--fastq_min_mapping_quality", type=int, default=None)
    return parser


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main(argv: Optional[List[str]] = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.n_jobs is not None:
        config.n_jobs = int(args.n_jobs)
    for key in [
        "fastq_max_parallel_samples",
        "fastq_threads",
        "fastq_memory_per_sample_mb",
        "fastq_min_mapping_quality",
    ]:
        value = getattr(args, key, None)
        if value is not None:
            setattr(config, key, value)
    if bool(getattr(args, "fastq_clean_intermediates", False)):
        config.fastq_clean_intermediates = True
    if bool(getattr(args, "fastq_no_auto_index_reference", False)):
        config.fastq_auto_index_reference = False
    config.__post_init__()

    engine = NetworkParserQueryEngine(registry_path=args.registry, config=config)
    engine.query(
        genomic_path=args.genomic,
        output_dir=args.output_dir,
        ref_fasta=args.ref_fasta,
        max_markers=int(args.max_markers),
        n_jobs=args.n_jobs,
        query_input_type=args.query_input_type,
        raw_sequence_mapping_mode=args.raw_sequence_mapping_mode,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

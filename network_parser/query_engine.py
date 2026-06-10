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
    query_report.json
    query_report.txt
    query_alignment_summary.json
"""

from __future__ import annotations

import argparse
import copy
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




def collect_required_features_from_registry(registry: Dict[str, Any]) -> List[str]:
    """Collect the union of every feature required by Level 1 and Level 2 models."""
    ordered: List[str] = []
    seen = set()

    def _add(features: Iterable[Any]) -> None:
        for feature in features or []:
            f = str(feature)
            if f and f not in seen:
                seen.add(f)
                ordered.append(f)

    level1 = registry.get("level1", {}) if isinstance(registry, dict) else {}
    _add(level1.get("features", []))

    level2 = registry.get("level2", {}) if isinstance(registry, dict) else {}
    global_payload = level2.get("global_fallback", {}) if isinstance(level2, dict) else {}
    _add(global_payload.get("features", []))
    global_binary_payload = level2.get("global_binary_fallback", {}) if isinstance(level2, dict) else {}
    _add(global_binary_payload.get("features", []))

    by_group = level2.get("by_level1_group", {}) if isinstance(level2, dict) else {}
    if isinstance(by_group, dict):
        for payload in by_group.values():
            if isinstance(payload, dict):
                _add(payload.get("features", []))

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


def supporting_markers_for_sample(
    sample_values: pd.Series,
    ranked_features: Optional[pd.DataFrame],
    model_importance: Optional[pd.DataFrame],
    max_markers: int = 10,
    feature_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Return top active markers for a sample, prioritising RF-FDR ranking if
    available, otherwise model feature importance, otherwise column order.
    """
    active_features = set(sample_values.index[pd.to_numeric(sample_values, errors="coerce").fillna(0).values != 0])
    feature_metadata = feature_metadata or {}

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
        return record

    if ranked_features is not None and "feature" in ranked_features.columns:
        df = ranked_features.copy()
        df = df[df["feature"].astype(str).isin(active_features)]
        if not df.empty:
            records = []
            for _, row in df.head(max_markers).iterrows():
                feature = str(row["feature"])
                records.append(
                    _attach_metadata(
                        {
                            "feature": feature,
                            "value": float(sample_values.get(feature, 0)),
                            "rf_mean_importance": _safe_float(row.get("rf_mean_importance")),
                            "empirical_p_value": _safe_float(row.get("empirical_p_value")),
                            "corrected_p_value": _safe_float(row.get("corrected_p_value")),
                        }
                    )
                )
            return records

    if model_importance is not None and "feature" in model_importance.columns:
        df = model_importance.copy()
        df = df[df["feature"].astype(str).isin(active_features)]
        if not df.empty:
            records = []
            for _, row in df.head(max_markers).iterrows():
                feature = str(row["feature"])
                records.append(
                    _attach_metadata(
                        {
                            "feature": feature,
                            "value": float(sample_values.get(feature, 0)),
                            "model_importance": _safe_float(row.get("model_importance")),
                        }
                    )
                )
            return records

    fallback = []
    for feature in sample_values.index:
        value = _safe_float(sample_values.get(feature))
        if value is not None and value != 0:
            fallback.append(_attach_metadata({"feature": str(feature), "value": value}))
        if len(fallback) >= max_markers:
            break
    return fallback


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _status_from_unique_fraction(unique_fraction: Optional[float], has_mapping_metadata: bool) -> Tuple[str, str]:
    """Classify whether a model-specific selected feature set was resolved in the query."""
    if not has_mapping_metadata:
        return (
            "feature_space_alignment_only",
            "No per-feature mapping metadata were available; status is based only on matrix alignment.",
        )
    if unique_fraction is None:
        return (
            "unknown_marker_recovery",
            "Per-feature mapping metadata were present, but marker recovery fraction could not be computed.",
        )
    if unique_fraction >= 0.80:
        return (
            "adequate_marker_recovery",
            "Most selected markers for this model were resolved in the query input.",
        )
    if unique_fraction >= 0.50:
        return (
            "partial_marker_recovery",
            "Only part of this model's selected marker space was resolved; interpret this level with caution.",
        )
    return (
        "low_marker_recovery",
        "Most selected markers for this model were unresolved or ambiguous; prediction support is likely weak.",
    )


def _status_from_active_fraction(active_fraction: float, active_count: int) -> Tuple[str, str]:
    """Classify active non-baseline evidence for a model-specific selected feature set."""
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


def summarize_feature_evidence_for_model(
    *,
    sample_values: pd.Series,
    features: Sequence[str],
    feature_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Summarise query evidence for the exact feature list used by one model.

    This separates model-specific evidence from union-level query evidence.  A
    query can have many active markers in the global union while the Level 1
    model, for example, still receives mostly baseline/zero states.
    """
    requested = [str(f) for f in features or []]
    feature_metadata = feature_metadata or {}

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

    unique_mapped = int(sum(1 for status in mapping_status_values if status == "mapped_unique_context"))
    mapped_or_reported = int(len(mapping_status_values))
    unique_fraction = (
        float(unique_mapped / max(1, mapped_or_reported))
        if has_mapping_metadata else None
    )

    recovery_status, recovery_reason = _status_from_unique_fraction(unique_fraction, has_mapping_metadata)
    active_status, active_reason = _status_from_active_fraction(active_fraction, n_active)

    return {
        "n_selected_features": n_features,
        "n_active_features": n_active,
        "active_feature_fraction": active_fraction,
        "active_feature_ids": active_features,
        "has_mapping_metadata": bool(has_mapping_metadata),
        "n_features_with_mapping_metadata": int(mapped_or_reported),
        "n_unique_mapped_features": int(unique_mapped),
        "unique_mapped_fraction": unique_fraction,
        "marker_recovery_status": recovery_status,
        "marker_recovery_reason": recovery_reason,
        "active_marker_evidence_status": active_status,
        "active_marker_evidence_reason": active_reason,
        "mapping_status_counts": mapping_status_counts,
        "allele_call_counts": allele_call_counts,
        "n_baseline_match_calls": int(allele_call_counts.get("baseline_match", 0)),
        "n_alt_match_calls": int(allele_call_counts.get("alt_match", 0)),
        "n_known_nonbaseline_match_calls": int(allele_call_counts.get("known_nonbaseline_match", 0)),
        "n_unresolved_or_missing_calls": int(
            allele_call_counts.get("not_called", 0)
            + mapping_status_counts.get("missing_context_filled_as_zero", 0)
            + mapping_status_counts.get("unresolved_context_filled_as_zero", 0)
        ),
        "n_multi_hit_calls": int(
            allele_call_counts.get("not_called_multi_hit_context", 0)
            + sum(v for k, v in mapping_status_counts.items() if str(k).startswith("multi_hit"))
        ),
        "n_non_training_allele_calls": int(allele_call_counts.get("non_training_allele", 0)),
    }


def interpretation_confidence_for_level(
    *,
    support: Optional[float],
    evidence: Dict[str, Any],
    n_supporting_markers: int,
) -> Tuple[str, str]:
    """Combine model support and model-specific marker evidence into a cautious label."""
    support_value = _safe_float(support)
    active_count = int(evidence.get("n_active_features", 0) or 0)
    recovery_status = str(evidence.get("marker_recovery_status", ""))
    active_status = str(evidence.get("active_marker_evidence_status", ""))

    if recovery_status == "low_marker_recovery":
        return (
            "low_confidence",
            "Prediction generated, but too few model-specific selected markers were resolved in the query input.",
        )

    if active_count == 0:
        return (
            "low_confidence",
            "Prediction generated, but this model received no active non-baseline selected markers for this sample.",
        )

    if support_value is None:
        return (
            "evidence_available_support_unavailable",
            "Model-specific active marker evidence is present, but probability-like support was unavailable from the model.",
        )

    if support_value >= 0.70 and n_supporting_markers > 0 and active_status == "active_marker_evidence_present":
        return (
            "high_confidence",
            "Prediction has strong model support and active model-specific marker evidence.",
        )

    if support_value >= 0.50 and (n_supporting_markers > 0 or active_count > 0):
        return (
            "moderate_confidence",
            "Prediction has some model support and model-specific marker evidence, but should still be interpreted cautiously.",
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
        f"{prefix}_n_non_training_allele_calls": evidence.get("n_non_training_allele_calls"),
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
    """Apply saved two-level NetworkParser models to new samples."""

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
            )
            l2_markers = supporting_markers_for_sample(
                X_l2.loc[sample_id],
                ranked_features=l2_ranked,
                model_importance=l2_importance,
                max_markers=max_markers,
                feature_metadata=sample_feature_metadata,
            )

            # Model-specific query evidence.  This is intentionally separate
            # from union-level FASTA/VCF recovery so we can diagnose whether
            # Level 1 or the selected Level 2 model received active marker
            # evidence.
            l1_evidence = summarize_feature_evidence_for_model(
                sample_values=X_l1.loc[sample_id],
                features=level1_features,
                feature_metadata=sample_feature_metadata,
            )
            l2_evidence = summarize_feature_evidence_for_model(
                sample_values=X_l2.loc[sample_id],
                features=l2_features,
                feature_metadata=sample_feature_metadata,
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
        predictions.to_csv(predictions_path, index=False)

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

        report = {
            "mode": "two_level_query",
            "registry": str(self.registry_path),
            "genomic_input": str(genomic_path),
            "n_samples": int(len(report_samples)),
            "samples": report_samples,
            "notes": [
                "Query mode is inference-only.",
                "RF-FDR feature selection is not rerun on query samples.",
                "For FASTA/VCF/FASTQ queries, NetworkParser reconstructs the saved trained feature space from the selected-feature manifest.",
                "Query mode does not rerun cohort-level matrix refinement, redundancy reduction, RF-FDR, model selection, or tree construction.",
                "For FASTA queries, saved context sequences are mapped to the query genome and the centre nucleotide is encoded with the saved baseline/REF/ALT rule.",
                "For VCF queries, saved feature coordinates are looked up directly in the query VCF; absent variant records are treated as reference-state calls and encoded with the same saved rule.",
                "Unresolved, ambiguous, repeated, or non-training allele calls are encoded as 0 and explicitly reported in the mapping summary.",
            ],
            "artifacts": {
                "predictions_csv": str(predictions_path),
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

        logger.info("Query complete: %s", predictions_path)
        return predictions


def write_text_report(report: Dict[str, Any], path: Path) -> None:
    lines: List[str] = []
    lines.append("NetworkParser two-level query report")
    lines.append("=" * 42)
    lines.append(f"Samples queried: {report.get('n_samples', 0)}")
    lines.append("")

    for sample in report.get("samples", []):
        lines.append(f"Sample: {sample.get('sample_id')}")
        lines.append("-" * (8 + len(str(sample.get("sample_id", "")))))
        if sample.get("fasta_mapping_quality") or sample.get("vcf_mapping_quality") or sample.get("raw_sequence_mapping_quality"):
            rq = sample.get("fasta_mapping_quality") or sample.get("vcf_mapping_quality") or sample.get("raw_sequence_mapping_quality") or {}
            label = "FASTA context recovery" if sample.get("fasta_mapping_quality") else "VCF trained-feature recovery"
            lines.append(label)
            lines.append(f"  Marker recovery: {rq.get('marker_recovery_status', 'NA')}")
            lines.append(f"  Active marker evidence: {rq.get('active_marker_evidence_status', 'NA')}")
            lines.append(f"  Unique mapped/resolved calls: {rq.get('n_unique_mapped_calls', 0)} / {rq.get('n_feature_calls', 0)}")
            lines.append(f"  Active encoded calls: {rq.get('n_encoded_active_features', 0)}")
            if rq.get("n_multi_hit_calls", 0):
                lines.append(f"  Multi-hit contexts/coordinates filled as 0: {rq.get('n_multi_hit_calls')}")
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
        l1_ev = sample.get("level1_feature_evidence") or {}
        if l1_ev:
            lines.append("  Level-specific marker evidence:")
            lines.append(f"    Selected features: {l1_ev.get('n_selected_features', 'NA')}")
            lines.append(f"    Active selected features: {l1_ev.get('n_active_features', 'NA')}")
            if l1_ev.get("unique_mapped_fraction") is not None:
                lines.append(f"    Unique recovery fraction: {float(l1_ev.get('unique_mapped_fraction')):.4f}")
            lines.append(f"    Marker recovery: {l1_ev.get('marker_recovery_status', 'NA')}")
            lines.append(f"    Active evidence: {l1_ev.get('active_marker_evidence_status', 'NA')}")
        if sample.get("level1_supporting_markers"):
            lines.append("  Supporting markers:")
            for marker in sample.get("level1_supporting_markers", [])[:10]:
                extra = ""
                if marker.get("observed_allele"):
                    quality = marker.get("mapping_quality") or marker.get("allele_call") or ""
                    quality_txt = f" | quality={quality}" if quality else ""
                    extra = f" | observed={marker.get('observed_allele')} | status={marker.get('mapping_status', '')}{quality_txt}"
                lines.append(f"    - {marker.get('feature')} = {marker.get('value')}{extra}")
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
        l2_ev = sample.get("level2_feature_evidence") or {}
        if l2_ev:
            lines.append("  Level-specific marker evidence:")
            lines.append(f"    Selected features: {l2_ev.get('n_selected_features', 'NA')}")
            lines.append(f"    Active selected features: {l2_ev.get('n_active_features', 'NA')}")
            if l2_ev.get("unique_mapped_fraction") is not None:
                lines.append(f"    Unique recovery fraction: {float(l2_ev.get('unique_mapped_fraction')):.4f}")
            lines.append(f"    Marker recovery: {l2_ev.get('marker_recovery_status', 'NA')}")
            lines.append(f"    Active evidence: {l2_ev.get('active_marker_evidence_status', 'NA')}")
        if sample.get("level2_supporting_markers"):
            lines.append("  Supporting markers:")
            for marker in sample.get("level2_supporting_markers", [])[:10]:
                extra = ""
                if marker.get("observed_allele"):
                    quality = marker.get("mapping_quality") or marker.get("allele_call") or ""
                    quality_txt = f" | quality={quality}" if quality else ""
                    extra = f" | observed={marker.get('observed_allele')} | status={marker.get('mapping_status', '')}{quality_txt}"
                lines.append(f"    - {marker.get('feature')} = {marker.get('value')}{extra}")
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
        description="Apply trained two-level NetworkParser models to new strain/sample input.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--genomic", required=True, help="New genomic matrix file or VCF directory.")
    parser.add_argument("--registry", required=True, help="Path to two_level_model_registry.json from training.")
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

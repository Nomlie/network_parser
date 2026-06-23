#!/usr/bin/env python3
# network_parser/two_level_protocol.py
"""
Two-level NetworkParser protocol
================================

Purpose
-------
Train and apply a hierarchical genomic classifier:

    Level 1: strain / lineage / group placement
    Level 2: drug-resistance phenotype or resistance-profile prediction

The protocol keeps the NetworkParser architecture explicit:

    input -> DataLoader/preprocessing -> configurable central feature filtering -> level-1 model
                                              -> level-specific configurable filtering -> level-2 models
                                              -> optional global level-2 fallback model

Important
---------
The configured central feature-selection method is used here. RF-FDR remains the
default, but chi-square/Fisher with multiple-testing correction or chi-square
permutation-FDR can be selected for faster statistically defensible screening. This is not the post-tree
confidence layer. Decision-tree interpretability can still be run elsewhere on
the filtered matrices when required.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.multitest import multipletests

try:
    from network_parser.config import NetworkParserConfig
    from network_parser.data_loader import DataLoader
    from network_parser.ml_protocol import MLProtocolRunner
    from network_parser.network_parser import normalize_labels
    from network_parser.statistical_validation_branch import StatisticalValidatorBranch
    from network_parser.feature_panel_selection import run_feature_panel_separability_check
    from network_parser.utils import (
        log_pipeline_header,
        log_stage_start,
        log_stage_complete,
        log_branch_decision,
        log_artifact,
        log_flow_step,
        PipelineProgress,
        progress_iter,
    )
except Exception:  # pragma: no cover - supports running from source tree
    from config import NetworkParserConfig  # type: ignore
    from data_loader import DataLoader  # type: ignore
    from ml_protocol import MLProtocolRunner  # type: ignore
    from network_parser import normalize_labels  # type: ignore
    from statistical_validation_branch import StatisticalValidatorBranch  # type: ignore
    from feature_panel_selection import run_feature_panel_separability_check  # type: ignore
    from utils import (  # type: ignore
        log_pipeline_header,
        log_stage_start,
        log_stage_complete,
        log_branch_decision,
        log_artifact,
        log_flow_step,
        PipelineProgress,
        progress_iter,
    )

try:
    from network_parser.feature_selection import rf_fdr_feature_selection
except Exception:  # pragma: no cover - supports direct source-tree execution
    from feature_selection import rf_fdr_feature_selection  # type: ignore

logger = logging.getLogger(__name__)


def _planned_two_level_stages(config: NetworkParserConfig, train_global_level2: bool) -> List[str]:
    """Return ordered stage labels for the two-level training progress bar."""
    stages = [
        "load and preprocess genomic matrix",
        "load metadata and align two labels",
        "Level 1 placement filtering and model training",
    ]
    if train_global_level2:
        stages.append("global Level-2 fallback training")
    if bool(getattr(config, "level2_train_binary_global_fallback", False)):
        stages.append("optional global binary Level-2 fallback")
    stages.append("per-group Level-2 training")
    stages.append("finalize two-level registry")
    return stages


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)

def resolve_central_filter_method(config: NetworkParserConfig) -> str:
    """Resolve the configured central feature filter with legacy compatibility."""
    method = str(
        getattr(
            config,
            "resolved_central_feature_filter_method",
            getattr(config, "central_feature_filter_method", "auto"),
        )
    ).lower()

    if method == "auto":
        if bool(getattr(config, "run_rf_fdr_feature_selection", False)):
            return "rf_fdr"
        if str(getattr(config, "statistical_test", "chi2")).lower() == "fisher":
            return "fisher_fdr"
        return "chi2_fdr"

    if method not in {"rf_fdr", "chi2_fdr", "fisher_fdr", "chi2_perm_fdr"}:
        raise ValueError(
            "central_feature_filter_method must resolve to one of: "
            "'rf_fdr', 'chi2_fdr', 'fisher_fdr', or 'chi2_perm_fdr'"
        )

    return method


def run_configured_feature_filter(
    X: pd.DataFrame,
    y: pd.Series,
    output_base_dir: Path,
    config: NetworkParserConfig,
    stage_name: str,
) -> Dict[str, Any]:
    """Run RF-FDR or association-FDR according to the central filter config."""
    method = resolve_central_filter_method(config)
    output_base_dir = Path(output_base_dir)

    filter_reason = {
        "rf_fdr": "Uses repeated random-forest importance estimates against label permutations, then applies FDR correction before model training.",
        "chi2_fdr": "Uses per-feature chi-square association tests with multiple-testing correction before model training.",
        "fisher_fdr": "Uses Fisher exact testing where appropriate, followed by multiple-testing correction before model training.",
        "chi2_perm_fdr": "Uses empirical chi-square association scores from label permutations, then applies FDR correction before model training.",
    }.get(method, "Runs configured statistical filtering before model training.")

    log_flow_step(
        logger,
        step=f"{stage_name} — configured feature filtering",
        happened="Started central feature filtering for this training target.",
        reason=filter_reason,
        before_samples=int(X.shape[0]),
        before_features=int(X.shape[1]),
        threshold=f"method={method}",
        status="started",
    )

    if method == "rf_fdr":
        result = rf_fdr_feature_selection(
            X=X,
            y=y,
            output_dir=output_base_dir / "rf_fdr_filter",
            config=config,
            stage_name=stage_name,
        )
        X_filtered = result["filtered_matrix"]
        summary = result.get("summary", {})
        log_flow_step(
            logger,
            step=f"{stage_name} — feature filtering complete",
            happened="Retained the RF-FDR-supported filtered matrix for model training.",
            reason="The downstream model should consume statistically screened genomic features rather than the full high-dimensional matrix.",
            before_samples=int(X.shape[0]),
            before_features=int(X.shape[1]),
            after_samples=int(X_filtered.shape[0]),
            after_features=int(X_filtered.shape[1]),
            threshold="method=rf_fdr",
            status=str(summary.get("status", "success")),
            artifact=summary.get("artifacts", {}).get("filtered_matrix"),
        )
        return result

    local_config = config
    local_config.statistical_test = "fisher" if method == "fisher_fdr" else "chi2"

    filter_dir = output_base_dir / f"{method}_filter"
    filter_dir.mkdir(parents=True, exist_ok=True)

    validator = StatisticalValidatorBranch(local_config)

    if method == "chi2_perm_fdr":
        result = validator.chi2_permutation_feature_selection(
            genomic_df=X,
            labels=y,
            output_dir=str(filter_dir),
            stage_name=stage_name,
        )
        X_filtered = result["filtered_matrix"]
        summary = result.get("summary", {})
        log_flow_step(
            logger,
            step=f"{stage_name} — feature filtering complete",
            happened="Retained the permutation-FDR-supported filtered matrix for model training.",
            reason="Permutation-derived empirical p-values improve robust inference before downstream model fitting.",
            before_samples=int(X.shape[0]),
            before_features=int(X.shape[1]),
            after_samples=int(X_filtered.shape[0]),
            after_features=int(X_filtered.shape[1]),
            threshold="method=chi2_perm_fdr",
            status=str(summary.get("status", "success")),
            artifact=summary.get("artifacts", {}).get("filtered_matrix"),
        )
        return result
    assoc = validator.association_tests(
        data=X,
        labels=y,
        output_dir=str(filter_dir),
    )
    corrected = validator.multiple_testing_correction(
        test_results=assoc,
        output_dir=str(filter_dir),
    )

    retained_features = [
        feature
        for feature, result in corrected.items()
        if bool(result.get("significant", False)) and feature in X.columns
    ]

    fallback_strategy = str(
        getattr(config, "feature_filter_fallback_strategy", "stop")
    ).lower()
    used_fallback = False

    if retained_features:
        X_filtered = X.loc[:, retained_features].copy()
    elif fallback_strategy == "stop":
        raise ValueError(
            f"{stage_name}: {method} retained no significant genomic features. "
            "Stopping is statistically defensible for publication-grade runs. "
            "For exploratory smoke testing only, set "
            "feature_filter_fallback_strategy='unfiltered'."
        )
    elif fallback_strategy == "unfiltered":
        logger.warning(
            "%s %s retained no significant genomic features. Using the aligned "
            "matrix as an exploratory fallback.",
            stage_name,
            method,
        )
        X_filtered = X.copy()
        retained_features = list(X_filtered.columns)
        used_fallback = True
    else:
        raise ValueError(
            "feature_filter_fallback_strategy must be one of: 'stop' or 'unfiltered'"
        )

    filtered_matrix_path = filter_dir / "filtered_matrix.csv"
    X_filtered.to_csv(filtered_matrix_path)

    summary = {
        "method": method,
        "status": "success",
        "stage_name": stage_name,
        "input_features": int(X.shape[1]),
        "tested_features": int(len(assoc)),
        "significant_features": int(
            sum(1 for result in corrected.values() if bool(result.get("significant", False)))
        ),
        "retained_features": int(X_filtered.shape[1]),
        "fallback_strategy": fallback_strategy,
        "used_fallback_unfiltered_matrix": bool(used_fallback),
        "retention_fraction": float(X_filtered.shape[1] / max(1, X.shape[1])),
        "retained_feature_names": list(X_filtered.columns),
        "artifacts": {
            "filter_dir": str(filter_dir),
            "association_json": str(filter_dir / "chi_squared_results.json"),
            "multiple_testing_json": str(filter_dir / "multiple_testing_results.json"),
            "filtered_matrix": str(filtered_matrix_path),
            "summary_json": str(filter_dir / "feature_filtering_summary.json"),
        },
    }

    write_json(summary, filter_dir / "feature_filtering_summary.json")

    log_flow_step(
        logger,
        step=f"{stage_name} — feature filtering complete",
        happened="Retained the association-FDR-supported filtered matrix for model training.",
        reason="The downstream model should only see features that pass the configured pre-model statistical screen, unless an explicit exploratory fallback was requested.",
        before_samples=int(X.shape[0]),
        before_features=int(X.shape[1]),
        after_samples=int(X_filtered.shape[0]),
        after_features=int(X_filtered.shape[1]),
        threshold=f"method={method}; fallback={fallback_strategy}",
        status="fallback_unfiltered" if used_fallback else "success",
        artifact=str(filtered_matrix_path),
    )

    return {
        "method": method,
        "summary": summary,
        "association": assoc,
        "multiple_testing": corrected,
        "retained_features": list(X_filtered.columns),
        "filtered_matrix": X_filtered,
    }


def run_feature_panel_check_after_filter(
    filter_result: Dict[str, Any],
    y: pd.Series,
    output_base_dir: Path,
    config: NetworkParserConfig,
    stage_name: str,
) -> Dict[str, Any]:
    """Run the ranked feature-panel check after central filtering."""
    panel_result = run_feature_panel_separability_check(
        X=filter_result["filtered_matrix"],
        y=y,
        output_dir=output_base_dir,
        config=config,
        stage_name=stage_name,
        filter_result=filter_result,
    )

    updated = dict(filter_result)
    updated["central_filtered_matrix"] = filter_result["filtered_matrix"]
    updated["filtered_matrix"] = panel_result["selected_matrix"]
    updated["feature_panel_separability"] = panel_result["summary"]
    updated["retained_features"] = list(panel_result["selected_features"])

    summary = panel_result.get("summary", {})
    log_flow_step(
        logger,
        step=f"{stage_name} — feature-panel check complete",
        happened="Selected the model-ready feature panel from the centrally filtered matrix.",
        reason="This keeps model training compact and interpretable while preserving the statistically ranked signal carried forward from central filtering.",
        before_samples=int(filter_result["filtered_matrix"].shape[0]),
        before_features=int(filter_result["filtered_matrix"].shape[1]),
        after_samples=int(panel_result["selected_matrix"].shape[0]),
        after_features=int(panel_result["selected_matrix"].shape[1]),
        threshold=f"panel_sizes={getattr(config, 'feature_panel_sizes', 'configured')}; min_score={getattr(config, 'feature_panel_min_score', 'configured')}",
        status=str(summary.get("selection_reason", summary.get("status", "complete"))),
        artifact=summary.get("artifacts", {}).get("selected_panel_matrix"),
    )
    return updated

def load_artifact_filtered_binary_matrix(
    artifact_root: Path,
    fallback_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load the DataLoader artifact-filtered binary matrix for downstream modeling.

    Important:
    - *_filtered.tsv is the filtered marker / annotation table.
    - *_binary.tsv is the sample × marker binary matrix.
    - RF-FDR must consume the artifact-filtered binary matrix when available.
    """

    artifact_root = Path(artifact_root)

    if fallback_matrix is None or fallback_matrix.empty:
        raise ValueError("Fallback matrix is empty; cannot validate artifact-filtered matrix.")

    fallback_index = pd.Index(
        fallback_matrix.index.astype(str).map(normalize_sample_id)
    )

    candidate_paths = sorted(artifact_root.rglob("*_binary.tsv"))

    if not candidate_paths:
        logger.warning(
            "No artifact-filtered binary matrix was found under %s. "
            "Using DataLoader returned matrix.",
            artifact_root,
        )
        return fallback_matrix.copy()

    valid_candidates = []

    for path in candidate_paths:
        try:
            candidate = pd.read_csv(path, sep="\t", index_col=0)
        except Exception as exc:
            logger.warning(
                "Could not read candidate artifact binary matrix %s: %s",
                path,
                exc,
            )
            continue

        if candidate.empty:
            logger.warning(
                "Skipping empty candidate artifact binary matrix: %s",
                path,
            )
            continue

        candidate.index = candidate.index.astype(str).map(normalize_sample_id)

        # Remove reference/control rows if present. They are useful in FASTA/TSV
        # artifacts but should not enter supervised model training.
        drop_rows = [
            idx for idx in candidate.index
            if str(idx).strip().upper() in {"REF", "REFERENCE"}
        ]
        if drop_rows:
            candidate = candidate.drop(index=drop_rows, errors="ignore")

        overlap = candidate.index.intersection(fallback_index)

        if overlap.empty:
            logger.warning(
                "Candidate artifact binary matrix %s has no sample-ID overlap "
                "with the DataLoader returned matrix. Skipping.",
                path,
            )
            continue

        valid_candidates.append(
            {
                "path": path,
                "matrix": candidate,
                "n_overlap": len(overlap),
                "n_features": candidate.shape[1],
            }
        )

    if not valid_candidates:
        logger.warning(
            "No valid artifact-filtered binary matrix could be aligned. "
            "Using DataLoader returned matrix."
        )
        return fallback_matrix.copy()

    # Prefer the candidate with the strongest sample overlap.
    # If tied, prefer the one with fewer features, because this should represent
    # the artifact-filtered matrix after structural/redundancy filtering.
    best = sorted(
        valid_candidates,
        key=lambda item: (-item["n_overlap"], item["n_features"]),
    )[0]

    X_artifact = best["matrix"].copy()

    logger.info(
        "Using artifact-filtered binary matrix for downstream modeling: %s | "
        "samples=%d | features=%d",
        best["path"],
        int(X_artifact.shape[0]),
        int(X_artifact.shape[1]),
    )

    return X_artifact

def find_feature_manifest(artifact_root: Path) -> Optional[Path]:
    """
    Locate the final DataLoader feature manifest synchronized to the artifact-filtered
    binary matrix used by model training.
    """
    artifact_root = Path(artifact_root)
    candidates = sorted(artifact_root.rglob("*_feature_manifest.tsv"))
    if not candidates:
        candidates = sorted(artifact_root.rglob("*_filtered.tsv"))

    for candidate in candidates:
        try:
            header = pd.read_csv(candidate, sep="\t", nrows=0).columns.astype(str).tolist()
        except Exception:
            continue
        if "Feature_ID" in header or "Position" in header:
            return candidate
    return None


def load_feature_manifest(manifest_path: Optional[Path]) -> Optional[pd.DataFrame]:
    if manifest_path is None:
        return None
    path = Path(manifest_path)
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    if "Feature_ID" not in df.columns and "Position" in df.columns:
        df["Feature_ID"] = df["Position"].astype(str)
    return df


def selected_feature_manifest_integrity(
    selected: pd.DataFrame,
    features: List[str],
    missing_metadata: List[str],
) -> Dict[str, Any]:
    """Summarise whether a selected-feature manifest is query-ready.

    This does not change model training. It records whether selected genomic
    features retain enough metadata to support raw-sequence query reconstruction.
    """
    if selected is None or selected.empty:
        return {
            "status": "empty",
            "requested_features": int(len(features)),
            "raw_sequence_query_ready": False,
            "warnings": ["Selected manifest is empty."],
        }

    context_cols = [
        col for col in ["Context_±40", "Context", "Context_sequence", "context_sequence"]
        if col in selected.columns
    ]

    def _nonempty(row: pd.Series, cols: List[str]) -> bool:
        for col in cols:
            if str(row.get(col, "")).strip():
                return True
        return False

    context_missing: List[str] = []
    allele_incomplete: List[str] = []
    for _, row in selected.iterrows():
        feature_id = str(row.get("Feature_ID", ""))
        if context_cols and not _nonempty(row, context_cols):
            context_missing.append(feature_id)
        elif not context_cols:
            context_missing.append(feature_id)

        has_ref = _nonempty(row, [col for col in ["Ref_allele", "REF"] if col in selected.columns])
        has_alt = _nonempty(row, [col for col in ["Alt_allele", "ALT"] if col in selected.columns])
        has_baseline = _nonempty(row, [col for col in ["Baseline_allele", "baseline_allele"] if col in selected.columns])
        if not (has_ref and has_alt and has_baseline):
            allele_incomplete.append(feature_id)

    warnings: List[str] = []
    if missing_metadata:
        warnings.append("Some selected features were not found in the source manifest.")
    if context_missing:
        warnings.append("Some selected features lack context sequence needed for raw-sequence query mapping.")
    if allele_incomplete:
        warnings.append("Some selected features lack complete REF/ALT/baseline allele metadata.")

    raw_ready = not missing_metadata and not context_missing and not allele_incomplete
    status = "query_ready" if raw_ready else ("partial" if len(selected) > 0 else "empty")

    return {
        "status": status,
        "requested_features": int(len(features)),
        "raw_sequence_query_ready": bool(raw_ready),
        "features_with_context_sequence": int(len(selected) - len(context_missing)),
        "features_missing_context_sequence": int(len(context_missing)),
        "features_with_complete_allele_metadata": int(len(selected) - len(allele_incomplete)),
        "features_missing_allele_metadata": int(len(allele_incomplete)),
        "features_missing_source_metadata": int(len(missing_metadata)),
        "missing_context_feature_ids": context_missing,
        "missing_allele_metadata_feature_ids": allele_incomplete,
        "warnings": warnings,
    }


def write_selected_feature_manifest(
    *,
    features: List[str],
    source_manifest: Optional[pd.DataFrame],
    output_path: Path,
) -> Dict[str, Any]:
    """Write an ordered manifest for the exact features retained by a model."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "manifest_file": str(output_path),
        "requested_features": int(len(features)),
        "features_with_metadata": 0,
        "features_missing_metadata": int(len(features)),
        "status": "missing_source_manifest",
    }

    if source_manifest is None or source_manifest.empty:
        pd.DataFrame({"Feature_ID": [str(f) for f in features]}).to_csv(output_path, sep="\t", index=False)
        return summary

    manifest = source_manifest.copy()
    feature_col = "Feature_ID" if "Feature_ID" in manifest.columns else "Position"
    manifest[feature_col] = manifest[feature_col].astype(str)
    manifest = manifest.drop_duplicates(subset=[feature_col], keep="first")
    lookup = manifest.set_index(feature_col, drop=False)

    ordered_rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for feature in [str(f) for f in features]:
        if feature in lookup.index:
            row = lookup.loc[feature].to_dict()
            row["Feature_ID"] = str(row.get("Feature_ID") or feature)
            ordered_rows.append(row)
        else:
            ordered_rows.append({"Feature_ID": feature})
            missing.append(feature)

    selected = pd.DataFrame(ordered_rows)
    selected.to_csv(output_path, sep="\t", index=False)

    integrity = selected_feature_manifest_integrity(
        selected=selected,
        features=features,
        missing_metadata=missing,
    )
    summary.update(
        {
            "features_with_metadata": int(len(features) - len(missing)),
            "features_missing_metadata": int(len(missing)),
            "missing_feature_ids": missing,
            "integrity": integrity,
            "status": "success" if not missing else "partial",
        }
    )
    return summary

def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
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
            logger.warning("Ignoring unknown config key: %s", key)

    config.__post_init__()
    return config


def normalize_sample_id(value: Any) -> str:
    sample = str(value).strip()
    sample = sample.replace(".vcf.gz", "").replace(".vcf", "")
    if sample.endswith(".gz"):
        sample = sample[:-3]
    return sample


def align_matrix_and_label(
    X: pd.DataFrame,
    meta: pd.DataFrame,
    label_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    if label_column not in meta.columns:
        raise ValueError(f"Label column '{label_column}' not found in metadata.")

    X = X.copy()
    X.index = X.index.astype(str).map(normalize_sample_id)

    y = normalize_labels(meta[label_column], drop_missing=True, lowercase=False)
    y.index = y.index.astype(str).map(normalize_sample_id)

    common = X.index.intersection(y.index)
    if common.empty:
        raise ValueError(
            f"No overlapping sample IDs between genomic matrix and metadata for label '{label_column}'."
        )

    X_aligned = X.loc[common].copy()
    y_aligned = y.loc[common].copy()

    logger.info(
        "Aligned label '%s' | samples=%d | features=%d | classes=%d",
        label_column,
        int(X_aligned.shape[0]),
        int(X_aligned.shape[1]),
        int(y_aligned.nunique(dropna=True)),
    )
    return X_aligned, y_aligned


def align_two_labels(
    X: pd.DataFrame,
    meta: pd.DataFrame,
    level1_label: str,
    level2_label: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    X1, y1 = align_matrix_and_label(X, meta, level1_label)

    y2 = normalize_labels(meta[level2_label], drop_missing=True, lowercase=False)
    y2.index = y2.index.astype(str).map(normalize_sample_id)

    common = X1.index.intersection(y1.index).intersection(y2.index)
    if common.empty:
        raise ValueError("No samples have both level-1 and level-2 labels after alignment.")

    X_final = X1.loc[common].copy()
    y1_final = y1.loc[common].copy()
    y2_final = y2.loc[common].copy()

    logger.info(
        "Aligned two-level supervision | samples=%d | features=%d | level1_classes=%d | level2_classes=%d",
        int(X_final.shape[0]),
        int(X_final.shape[1]),
        int(y1_final.nunique(dropna=True)),
        int(y2_final.nunique(dropna=True)),
    )
    return X_final, y1_final, y2_final

def safe_hierarchy_token(value: Any, max_len: int = 80) -> str:
    """Return a filesystem-safe token for hierarchy node directories."""
    raw = str(value).strip() or "node"
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)
    cleaned = cleaned.strip("_") or "node"
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[:max_len].rstrip("_") or "node"


def align_hierarchy_labels(
    X: pd.DataFrame,
    meta: pd.DataFrame,
    hierarchy_labels: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align a feature matrix to samples carrying every requested hierarchy label."""
    labels = [str(label).strip() for label in hierarchy_labels if str(label).strip()]
    if len(labels) < 1:
        raise ValueError("At least one hierarchy label column is required.")

    missing = [label for label in labels if label not in meta.columns]
    if missing:
        raise ValueError(
            "Hierarchy label column(s) not found in metadata: " + ", ".join(missing)
        )

    X_local = X.copy()
    X_local.index = X_local.index.astype(str).map(normalize_sample_id)

    label_series: Dict[str, pd.Series] = {}
    common = pd.Index(X_local.index)
    for label in labels:
        y = normalize_labels(meta[label], drop_missing=True, lowercase=False)
        y.index = y.index.astype(str).map(normalize_sample_id)
        label_series[label] = y
        common = common.intersection(y.index)

    if common.empty:
        raise ValueError(
            "No samples have all requested hierarchy labels after metadata alignment."
        )

    X_aligned = X_local.loc[common].copy()
    labels_df = pd.DataFrame(index=common)
    for label, y in label_series.items():
        labels_df[label] = y.loc[common].astype(str).values

    logger.info(
        "Aligned multi-level hierarchy supervision | samples=%d | features=%d | levels=%d",
        int(X_aligned.shape[0]),
        int(X_aligned.shape[1]),
        int(len(labels)),
    )
    for idx, label in enumerate(labels, start=1):
        logger.info(
            "Hierarchy level %d aligned | label_column=%s | classes=%d",
            int(idx),
            str(label),
            int(labels_df[label].nunique(dropna=True)),
        )

    return X_aligned, labels_df


def build_global_level2_training_labels(
    *,
    X: pd.DataFrame,
    meta: pd.DataFrame,
    y_level2: pd.Series,
    level2_label: str,
    global_level2_label: Optional[str],
    config: NetworkParserConfig,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Resolve labels for the standard global Level-2 fallback.

    Group-specific Level-2 models always use the detailed level2_label. The
    global fallback may optionally use a broader metadata column, for example
    AMR_binary, so that underrepresented lineages can fall back to a robust
    resistant/susceptible endpoint rather than an under-supported detailed
    resistance profile.
    """
    requested = (
        global_level2_label
        or getattr(config, "global_level2_label_column", None)
        or level2_label
    )
    requested = str(requested).strip() if requested is not None else level2_label

    if requested == level2_label:
        y_global = y_level2.copy()
        X_global = X.loc[X.index.intersection(y_global.index)].copy()
        y_global = y_global.loc[X_global.index].copy()
        summary = {
            "global_level2_label_column": level2_label,
            "source": "level2_label",
            "description": "standard global fallback uses the detailed Level-2 label",
            "n_samples": int(y_global.shape[0]),
            "n_classes": int(y_global.nunique(dropna=True)),
        }
        return X_global, y_global, summary

    if requested not in meta.columns:
        raise ValueError(
            f"global_level2_label column '{requested}' was not found in metadata. "
            "Use --global_level2_label with an existing metadata column."
        )

    y_global = normalize_labels(meta[requested], drop_missing=True, lowercase=False)
    y_global.index = y_global.index.astype(str).map(normalize_sample_id)
    common = X.index.intersection(y_global.index)
    if common.empty:
        raise ValueError(
            f"No aligned samples have non-missing labels in global_level2_label column '{requested}'."
        )

    X_global = X.loc[common].copy()
    y_global = y_global.loc[common].copy()
    summary = {
        "global_level2_label_column": requested,
        "source": "global_level2_label",
        "description": (
            "standard global Level-2 fallback uses a separate metadata endpoint; "
            "group-specific Level-2 models still use the detailed --level2_label"
        ),
        "detailed_level2_label_column": level2_label,
        "n_samples": int(y_global.shape[0]),
        "n_classes": int(y_global.nunique(dropna=True)),
    }
    logger.info(
        "Global Level-2 fallback target resolved | label_column=%s | source=%s | samples=%d | classes=%d",
        requested,
        summary["source"],
        int(summary["n_samples"]),
        int(summary["n_classes"]),
    )
    return X_global, y_global, summary


def align_prediction_matrix(X_new: pd.DataFrame, training_features: List[str]) -> pd.DataFrame:
    X_new = X_new.copy()
    X_new.index = X_new.index.astype(str).map(normalize_sample_id)

    for feature in training_features:
        if feature not in X_new.columns:
            X_new[feature] = 0

    return X_new.loc[:, training_features].copy()


# -----------------------------------------------------------------------------
# Model fitting / prediction helpers
# -----------------------------------------------------------------------------

def run_ml_model(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    config: NetworkParserConfig,
    algorithm: Optional[str],
) -> Dict[str, Any]:
    """Run the existing ML protocol and return its summary payload."""
    ensure_dir(output_dir)
    runner = MLProtocolRunner(config=config)
    return runner.run(
        genomic_df=X,
        labels=y,
        output_dir=str(output_dir),
        algorithm=algorithm if algorithm is not None else getattr(config, "ml_algorithm", "auto"),
    )


def train_fallback_rf_model(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    config: NetworkParserConfig,
    model_name: str,
) -> Dict[str, Any]:
    """
    Small fallback used only if the external ML protocol fails.
    It keeps the two-level protocol runnable while recording the failure clearly.
    """
    ensure_dir(output_dir)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y.astype(str))

    model = RandomForestClassifier(
        n_estimators=int(getattr(config, "rf_selector_n_estimators", 300)),
        max_features=getattr(config, "rf_selector_max_features", "sqrt"),
        min_samples_leaf=int(getattr(config, "rf_selector_min_samples_leaf", 1)),
        class_weight=getattr(config, "rf_selector_class_weight", "balanced"),
        random_state=int(getattr(config, "rf_selector_random_state", 42)),
        n_jobs=int(getattr(config, "n_jobs", -1)),
    )
    model.fit(X, y_encoded)

    payload = {"model": model, "label_encoder": encoder, "features": list(X.columns)}
    model_path = output_dir / f"{model_name}.pkl"
    with open(model_path, "wb") as handle:
        pickle.dump(payload, handle)

    summary = {
        "status": "success",
        "selected_algorithm": "RF_fallback",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "artifacts": {"model_file": str(model_path)},
    }
    write_json(summary, output_dir / f"{model_name}_summary.json")
    return summary


def train_model_safely(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    config: NetworkParserConfig,
    algorithm: Optional[str],
    model_name: str,
) -> Dict[str, Any]:
    """
    Train a model for the two-level protocol.

    Publication-safe behaviour:
    - by default, fail loudly if the requested ML protocol fails;
    - only use the RF fallback if config.allow_two_level_rf_fallback=True.
    """
    try:
        return run_ml_model(
            X=X,
            y=y,
            output_dir=output_dir,
            config=config,
            algorithm=algorithm,
        )

    except Exception as exc:
        diagnostics = label_distribution_diagnostics(y, requested_cv_splits=5)
        diagnostics.update(
            {
                "model_name": str(model_name),
                "n_features": int(X.shape[1]) if hasattr(X, "shape") else None,
                "failure_reason": classify_ml_failure(exc),
                "ml_protocol_error": str(exc),
                "rf_fallback_enabled": bool(getattr(config, "allow_two_level_rf_fallback", False)),
            }
        )

        if bool(getattr(config, "allow_two_level_rf_fallback", False)):
            logger.warning(
                "%s ML protocol failed, so an explicitly enabled RF fallback will be fitted because %s | "
                "samples=%d | features=%d | classes=%d | "
                "min_class_count=%d | feasible_cv_splits=%d",
                model_name,
                diagnostics["failure_reason"],
                diagnostics["n_samples"],
                int(X.shape[1]),
                diagnostics["n_classes"],
                diagnostics["min_class_count"],
                diagnostics["feasible_selector_cv_splits"],
            )
            logger.debug("%s ML protocol failure traceback", model_name, exc_info=True)
            fallback = train_fallback_rf_model(
                X=X,
                y=y,
                output_dir=output_dir,
                config=config,
                model_name=model_name,
            )
            fallback["ml_protocol_error"] = str(exc)
            fallback["failure_reason"] = diagnostics["failure_reason"]
            fallback["training_diagnostics"] = diagnostics
            fallback["fallback_enabled_by_config"] = True
            return fallback

        logger.warning(
            "%s ML protocol failed, so the group-specific model will be skipped if a global fallback is available because %s | "
            "samples=%d | features=%d | classes=%d | min_class_count=%d | "
            "feasible_cv_splits=%d | rf_fallback_enabled=False",
            model_name,
            diagnostics["failure_reason"],
            diagnostics["n_samples"],
            int(X.shape[1]),
            diagnostics["n_classes"],
            diagnostics["min_class_count"],
            diagnostics["feasible_selector_cv_splits"],
        )
        logger.debug("%s ML protocol failure traceback", model_name, exc_info=True)

        raise RuntimeError(
            f"{model_name}: ML protocol failed ({diagnostics['failure_reason']}). "
            f"samples={diagnostics['n_samples']}; "
            f"features={diagnostics['n_features']}; "
            f"classes={diagnostics['n_classes']}; "
            f"min_class_count={diagnostics['min_class_count']}; "
            f"feasible_selector_cv_splits={diagnostics['feasible_selector_cv_splits']}. "
            "RF fallback is disabled; group-specific Level-2 prediction should use "
            "the global fallback when available. Set allow_two_level_rf_fallback=True "
            "only for exploratory runs."
        ) from exc


def load_model_payload(model_path: str) -> Any:
    path = Path(model_path)
    with open(path, "rb") as handle:
        return pickle.load(handle)


def predict_from_model_payload(model_payload: Any, X: pd.DataFrame) -> List[str]:
    """
    Predict from either an MLProtocol model object or the fallback payload.
    """
    if isinstance(model_payload, dict) and "model" in model_payload and "label_encoder" in model_payload:
        model = model_payload["model"]
        encoder = model_payload["label_encoder"]
        raw = model.predict(X)
        return [str(v) for v in encoder.inverse_transform(raw)]

    raw = model_payload.predict(X)
    return [str(v) for v in raw]


def get_model_file(model_summary: Dict[str, Any]) -> Optional[str]:
    artifacts = model_summary.get("artifacts", {}) if isinstance(model_summary, dict) else {}
    for key in ("model_file", "model_path"):
        value = artifacts.get(key)
        if value:
            return str(value)
    return None


def label_distribution_diagnostics(
    labels: pd.Series,
    requested_cv_splits: int = 5,
) -> Dict[str, Any]:
    """Summarise label support for Level-2 eligibility checks.

    The public diagnostic payload intentionally reports count structure rather
    than label names.  This keeps logs generic while making small-group and
    stratified-CV failures interpretable.
    """
    y = pd.Series(labels).astype(str).str.strip()
    y = y.replace({"": pd.NA, "-": pd.NA, "NA": pd.NA, "N/A": pd.NA, "None": pd.NA, "nan": pd.NA, "NaN": pd.NA})
    y = y.dropna()

    counts = y.value_counts(dropna=True)
    values = [int(v) for v in counts.tolist()]
    min_count = int(min(values)) if values else 0
    max_count = int(max(values)) if values else 0
    requested_cv_splits = max(2, int(requested_cv_splits))
    feasible_cv_splits = int(min(requested_cv_splits, min_count)) if min_count > 0 else 0
    min_samples_for_binary_split = int(counts.shape[0] * 2) if counts.shape[0] >= 2 else 0

    return {
        "n_samples": int(y.shape[0]),
        "n_classes": int(counts.shape[0]),
        "min_class_count": min_count,
        "max_class_count": max_count,
        "n_singleton_classes": int(sum(v == 1 for v in values)),
        "class_count_values_sorted": sorted(values),
        "requested_selector_cv_splits": requested_cv_splits,
        "feasible_selector_cv_splits": feasible_cv_splits,
        "min_samples_for_twofold_per_label": min_samples_for_binary_split,
        "stratified_cv_feasible": bool(counts.shape[0] >= 2 and feasible_cv_splits >= 2),
    }


def adaptive_level2_min_samples(
    label_diagnostics: Dict[str, Any],
    min_samples_per_label: int = 2,
) -> int:
    """Return the minimum group size implied by the Level-2 label structure.

    Group-specific Level-2 training should not be controlled by a fixed cohort
    size.  The minimum required samples scale with the number of Level-2 labels
    that must be separated.  A two-fold stratified probe needs at least two
    samples per label, so the adaptive default is:

        required samples = n_level2_labels * 2

    This remains a pre-model eligibility check; statistical filtering still runs
    only after the label structure is trainable.
    """
    try:
        n_classes = int(label_diagnostics.get("n_classes", 0))
    except Exception:
        n_classes = 0

    min_samples_per_label = max(2, int(min_samples_per_label))
    if n_classes < 2:
        return 0
    return int(n_classes * min_samples_per_label)


def apply_level2_class_support_filter(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    config: NetworkParserConfig,
    stage_name: str,
    requested_cv_splits: int = 5,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Optionally remove Level-2 classes that cannot support stratified CV.

    This is a label-support eligibility gate. It runs before Level-2
    statistical filtering and model screening, and it does not use genomic
    features to decide which samples to retain.
    """
    output_dir = ensure_dir(Path(output_dir))
    enabled = bool(getattr(config, "level2_drop_low_support_classes", False))
    min_count = int(getattr(config, "level2_min_class_count", 2))

    y_clean = pd.Series(y, index=y.index).astype(str).str.strip()
    y_clean = y_clean.replace({
        "": pd.NA,
        "-": pd.NA,
        "NA": pd.NA,
        "N/A": pd.NA,
        "None": pd.NA,
        "nan": pd.NA,
        "NaN": pd.NA,
    })
    non_missing_idx = y_clean.dropna().index
    X0 = X.loc[X.index.intersection(non_missing_idx)].copy()
    y0 = y_clean.loc[X0.index].astype(str).copy()

    before_diag = label_distribution_diagnostics(y0, requested_cv_splits=requested_cv_splits)
    counts = y0.value_counts(dropna=True)

    class_support_table = {str(label): int(count) for label, count in counts.items()}

    summary: Dict[str, Any] = {
        "stage_name": str(stage_name),
        "enabled": bool(enabled),
        "min_class_count_required": int(min_count),
        "before": before_diag,
        "after": before_diag,
        "n_samples_removed": 0,
        "n_classes_removed": 0,
        "n_classes_retained": int(counts.shape[0]),
        "class_support_table": class_support_table,
        "adaptive_min_training_samples": adaptive_level2_min_samples(before_diag),
        "status": "not_applied",
        "reason": "disabled" if not enabled else "no_low_support_classes",
        "artifacts": {
            "summary_json": str(output_dir / "level2_class_support_filter_summary.json"),
        },
    }

    if not enabled:
        write_json(summary, output_dir / "level2_class_support_filter_summary.json")
        return X0, y0, summary

    keep_classes = counts[counts >= min_count].index.astype(str)
    drop_classes = counts[counts < min_count]

    audit_path = output_dir / "level2_low_support_classes.tsv"
    if not drop_classes.empty:
        pd.DataFrame(
            {
                "level2_class": drop_classes.index.astype(str),
                "sample_count": drop_classes.astype(int).values,
                "min_class_count_required": int(min_count),
            }
        ).to_csv(audit_path, sep="\t", index=False)
        summary["artifacts"]["low_support_classes_tsv"] = str(audit_path)

    if drop_classes.empty:
        write_json(summary, output_dir / "level2_class_support_filter_summary.json")
        return X0, y0, summary

    keep_mask = y0.isin(set(keep_classes))
    X1 = X0.loc[keep_mask].copy()
    y1 = y0.loc[X1.index].copy()
    after_diag = label_distribution_diagnostics(y1, requested_cv_splits=requested_cv_splits)

    summary.update(
        {
            "status": "applied",
            "reason": "low_support_classes_removed",
            "after": after_diag,
            "n_samples_removed": int(X0.shape[0] - X1.shape[0]),
            "n_classes_removed": int(drop_classes.shape[0]),
            "n_classes_retained": int(len(keep_classes)),
            "class_support_table": {str(label): int(count) for label, count in y1.value_counts(dropna=True).items()},
            "adaptive_min_training_samples": adaptive_level2_min_samples(after_diag),
        }
    )

    logger.info(
        "%s Level-2 class-support filter applied | min_class_count=%d | "
        "removed_samples=%d | removed_classes=%d | retained_classes=%d | "
        "post_filter_min_class_count=%d | feasible_cv_splits=%d",
        stage_name,
        int(min_count),
        int(summary["n_samples_removed"]),
        int(summary["n_classes_removed"]),
        int(summary["n_classes_retained"]),
        int(after_diag["min_class_count"]),
        int(after_diag["feasible_selector_cv_splits"]),
    )

    write_json(summary, output_dir / "level2_class_support_filter_summary.json")
    return X1, y1, summary


def _comma_set(value: Any) -> set:
    """Parse comma-separated config values into a case-insensitive lookup set."""
    if value is None:
        return set()
    return {str(item).strip().lower() for item in str(value).split(",") if str(item).strip()}


def _read_level2_binary_mapping(mapping_path: str) -> Dict[str, str]:
    """Read an explicit detailed-label -> binary-label mapping table."""
    path = Path(mapping_path)
    if not path.exists():
        raise FileNotFoundError(f"Level-2 binary label mapping file not found: {path}")

    df = pd.read_csv(path, sep=None, engine="python")
    if df.empty:
        raise ValueError(f"Level-2 binary label mapping file is empty: {path}")

    lower_to_col = {str(col).strip().lower(): col for col in df.columns}
    original_col = (
        lower_to_col.get("original_level2_label")
        or lower_to_col.get("original_label")
        or lower_to_col.get("source_label")
        or lower_to_col.get("from")
    )
    binary_col = (
        lower_to_col.get("binary_level2_label")
        or lower_to_col.get("binary_label")
        or lower_to_col.get("target_label")
        or lower_to_col.get("to")
    )
    if original_col is None or binary_col is None:
        raise ValueError(
            "Level-2 binary label mapping file must contain columns "
            "original_level2_label and binary_level2_label."
        )

    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        source = str(row.get(original_col, "")).strip()
        target = str(row.get(binary_col, "")).strip()
        if source and target and source.lower() not in {"nan", "none", "na", "n/a"}:
            mapping[source] = target
    if not mapping:
        raise ValueError("Level-2 binary label mapping file produced no usable mappings.")
    return mapping


def _canonical_binary_level2_labels(labels: pd.Series, config: NetworkParserConfig) -> Tuple[pd.Series, Dict[str, Any]]:
    """Convert binary endpoint labels to canonical resistant/susceptible values."""
    resistant_values = _comma_set(getattr(config, "level2_binary_resistant_values", ""))
    susceptible_values = _comma_set(getattr(config, "level2_binary_susceptible_values", ""))

    canonical = []
    dropped_values: Dict[str, int] = {}
    for value in labels.astype(str).str.strip():
        key = value.lower()
        if key in resistant_values:
            canonical.append("resistant")
        elif key in susceptible_values:
            canonical.append("susceptible")
        else:
            canonical.append(pd.NA)
            dropped_values[value] = dropped_values.get(value, 0) + 1

    out = pd.Series(canonical, index=labels.index, name="level2_binary_label", dtype="object")
    summary = {
        "canonical_classes": sorted([str(v) for v in out.dropna().unique()]),
        "n_unmapped_or_invalid_labels": int(out.isna().sum()),
        "unmapped_or_invalid_label_counts": {str(k): int(v) for k, v in dropped_values.items()},
    }
    return out.dropna().astype(str), summary


def build_level2_binary_labels(
    *,
    X: pd.DataFrame,
    meta: pd.DataFrame,
    y_level2: pd.Series,
    config: NetworkParserConfig,
    output_dir: Path,
    level2_label: str,
) -> Tuple[Optional[pd.Series], Dict[str, Any]]:
    """Build the optional global binary Level-2 endpoint.

    Source priority:
      1. dedicated metadata column from config.level2_binary_label_column
      2. explicit mapping file from config.level2_binary_label_mapping_file

    The returned labels are aligned to X and canonicalised to resistant/susceptible.
    """
    output_dir = ensure_dir(Path(output_dir))
    enabled = bool(getattr(config, "level2_train_binary_global_fallback", False))
    label_col = getattr(config, "level2_binary_label_column", None)
    mapping_file = getattr(config, "level2_binary_label_mapping_file", None)

    summary: Dict[str, Any] = {
        "enabled": bool(enabled),
        "status": "skipped",
        "reason": "disabled" if not enabled else None,
        "source": None,
        "level2_label_column": str(level2_label),
        "binary_label_column": label_col,
        "binary_label_mapping_file": mapping_file,
        "artifacts": {
            "summary_json": str(output_dir / "level2_binary_label_summary.json"),
            "labels_csv": str(output_dir / "level2_binary_training_labels.csv"),
        },
    }

    if not enabled:
        write_json(summary, output_dir / "level2_binary_label_summary.json")
        return None, summary

    raw_binary: Optional[pd.Series] = None
    if label_col:
        if label_col not in meta.columns:
            summary.update({"status": "skipped", "reason": "binary_label_column_not_found"})
            write_json(summary, output_dir / "level2_binary_label_summary.json")
            return None, summary
        raw_binary = normalize_labels(meta[label_col], drop_missing=True, lowercase=False)
        raw_binary.index = raw_binary.index.astype(str).map(normalize_sample_id)
        summary["source"] = "metadata_column"

    elif mapping_file:
        mapping = _read_level2_binary_mapping(str(mapping_file))
        raw_binary = y_level2.astype(str).str.strip().map(mapping)
        raw_binary = raw_binary.dropna().astype(str)
        raw_binary.index = raw_binary.index.astype(str).map(normalize_sample_id)
        summary["source"] = "mapping_file"
        summary["n_mapping_entries"] = int(len(mapping))

    else:
        summary.update({"status": "skipped", "reason": "no_binary_label_source_configured"})
        write_json(summary, output_dir / "level2_binary_label_summary.json")
        return None, summary

    common = X.index.astype(str).intersection(raw_binary.index.astype(str))
    if common.empty:
        summary.update({"status": "skipped", "reason": "no_overlap_with_binary_level2_labels"})
        write_json(summary, output_dir / "level2_binary_label_summary.json")
        return None, summary

    raw_binary = raw_binary.loc[common].astype(str).str.strip()
    y_binary, canonical_summary = _canonical_binary_level2_labels(raw_binary, config)
    y_binary = y_binary.loc[y_binary.index.intersection(X.index)].copy()

    diagnostics = label_distribution_diagnostics(y_binary, requested_cv_splits=5)
    summary.update(
        {
            "status": "success" if diagnostics["n_classes"] >= 2 else "skipped",
            "reason": None if diagnostics["n_classes"] >= 2 else "binary_endpoint_has_fewer_than_two_classes",
            "n_aligned_binary_labels_before_canonicalisation": int(raw_binary.shape[0]),
            "n_binary_training_labels": int(y_binary.shape[0]),
            "canonicalisation": canonical_summary,
            "label_diagnostics": diagnostics,
        }
    )

    pd.DataFrame(
        {
            "sample_id": y_binary.index.astype(str),
            "level2_binary_label": y_binary.astype(str).values,
        }
    ).to_csv(output_dir / "level2_binary_training_labels.csv", index=False)

    write_json(summary, output_dir / "level2_binary_label_summary.json")
    if summary["status"] != "success":
        return None, summary
    return y_binary, summary


def _successful_level2_fallback_source(
    global_profile_payload: Dict[str, Any],
    global_binary_payload: Dict[str, Any],
) -> str:
    """Return the best available Level-2 fallback source name."""
    if isinstance(global_profile_payload, dict) and global_profile_payload.get("status") == "success":
        return "global_fallback"
    if isinstance(global_binary_payload, dict) and global_binary_payload.get("status") == "success":
        return "global_binary_fallback"
    return "unavailable"


def classify_ml_failure(exc: Exception) -> str:
    """Convert nested ML exceptions into concise registry/log reason codes."""
    messages: List[str] = []
    current: Optional[BaseException] = exc
    while current is not None:
        messages.append(str(current))
        current = current.__cause__ if current.__cause__ is not None else current.__context__

    msg = " | ".join(messages).lower()
    if "no finite probe scores" in msg:
        return "model_selector_no_finite_probe_scores"
    if "no feature" in msg or "empty feature" in msg or "feature columns" in msg:
        return "empty_or_invalid_filtered_matrix"
    if "one class" in msg or "at least two" in msg or "single" in msg:
        return "insufficient_level2_class_diversity"
    if "converge" in msg or "convergence" in msg:
        return "model_convergence_failure"
    return "ml_protocol_failure"


def _extract_model_publication_metrics(model_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Extract compact, JSON-safe model metrics for the registry summary."""
    if not isinstance(model_summary, dict):
        return {"status": "unavailable"}

    evaluation = model_summary.get("evaluation", {})
    if not isinstance(evaluation, dict):
        evaluation = {}

    return {
        "status": model_summary.get("status", "unknown"),
        "selected_algorithm": model_summary.get("selected_algorithm"),
        "n_samples": model_summary.get("n_samples"),
        "n_features": model_summary.get("n_features"),
        "training_metrics": model_summary.get("training_metrics", {}),
        "best_threshold_summary": evaluation.get("best_threshold_summary", {}),
        "feature_overlap_coverage": evaluation.get("feature_overlap_coverage"),
    }


def _build_publication_summary(
    *,
    X: pd.DataFrame,
    y_level1: pd.Series,
    y_level2: pd.Series,
    level1_filter: Dict[str, Any],
    level1_model: Dict[str, Any],
    global_level2_payload: Dict[str, Any],
    global_binary_level2_payload: Optional[Dict[str, Any]] = None,
    subgroup_payload: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Build a compact registry section for reporting and manuscript tables."""
    subgroup_payload = subgroup_payload or {}
    global_binary_level2_payload = global_binary_level2_payload or {"status": "skipped"}

    subgroup_summary: Dict[str, Any] = {}
    for group_name, payload in subgroup_payload.items():
        payload = payload if isinstance(payload, dict) else {}
        subgroup_summary[str(group_name)] = {
            "status": payload.get("status", "unknown"),
            "reason": payload.get("reason"),
            "n_samples": payload.get("n_samples"),
            "n_level2_classes": payload.get("n_level2_classes"),
            "retained_features": int(len(payload.get("features", []))) if isinstance(payload.get("features", []), list) else 0,
            "model_file": payload.get("model_file"),
            "level2_source_for_prediction": payload.get("level2_source_for_prediction"),
        }

    global_features = global_level2_payload.get("features", []) if isinstance(global_level2_payload, dict) else []
    global_binary_features = (
        global_binary_level2_payload.get("features", [])
        if isinstance(global_binary_level2_payload, dict)
        else []
    )

    return {
        "cohort_statistics": {
            "aligned_samples": int(X.shape[0]),
            "artifact_filtered_features": int(X.shape[1]),
            "level1_classes": int(y_level1.nunique(dropna=True)),
            "level2_classes": int(y_level2.nunique(dropna=True)),
        },
        "retained_features_per_level": {
            "level1": int(len(level1_filter.get("retained_features", []))),
            "level2_global_fallback": int(len(global_features)) if isinstance(global_features, list) else 0,
            "level2_global_binary_fallback": int(len(global_binary_features)) if isinstance(global_binary_features, list) else 0,
            "level2_by_level1_group": subgroup_summary,
        },
        "model_performance_metrics": {
            "level1": _extract_model_publication_metrics(level1_model),
            "level2_global_fallback": _extract_model_publication_metrics(global_level2_payload.get("model", {}))
            if isinstance(global_level2_payload, dict)
            else {"status": "unavailable"},
            "level2_global_binary_fallback": _extract_model_publication_metrics(global_binary_level2_payload.get("model", {}))
            if isinstance(global_binary_level2_payload, dict)
            else {"status": "unavailable"},
            "level2_by_level1_group": {
                str(group): _extract_model_publication_metrics(payload.get("model", {}))
                for group, payload in subgroup_payload.items()
                if isinstance(payload, dict) and payload.get("status") == "success"
            },
        },
        "known_marker_overlap": {
            "status": "not_evaluated",
            "reason": "known marker set was not provided to the two-level protocol",
        },
    }


# -----------------------------------------------------------------------------
# Two-level protocol
# -----------------------------------------------------------------------------

class TwoLevelProtocol:
    """Train and apply the two-level strain-placement and resistance protocol."""

    def __init__(self, config: NetworkParserConfig):
        self.config = config
        self.loader = DataLoader(config=config, n_jobs=getattr(config, "n_jobs", -1))

    def train_hierarchy(
        self,
        genomic_path: str,
        meta_path: str,
        hierarchy_labels: List[str],
        output_dir: str,
        ref_fasta: Optional[str] = None,
        algorithm: Optional[str] = None,
        min_samples_per_node: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Train a true multi-level hierarchy from an ordered list of labels.

        Each level is trained only within the parent branch defined by the
        previous observed label path.  Nodes with only one child state are stored
        as deterministic branches rather than forced through model training.
        """
        labels = [str(label).strip() for label in hierarchy_labels if str(label).strip()]
        if len(labels) < 2:
            raise ValueError("--hierarchy_labels requires at least two metadata columns.")

        out = ensure_dir(Path(output_dir))
        matrices_dir = ensure_dir(out / "matrices")
        hierarchy_dir = ensure_dir(out / "hierarchy_models")

        log_pipeline_header(
            logger,
            "NetworkParser multi-level hierarchy training started",
            central_filter=getattr(self.config, "central_feature_filter_method", "auto"),
            feature_panel_check=bool(getattr(self.config, "run_feature_panel_separability_check", True)),
            n_jobs=getattr(self.config, "n_jobs", "NA"),
        )

        log_stage_start(logger, 1, "load and preprocess genomic matrix")
        X_raw_unfiltered = self.loader.load_genomic_matrix(
            file_path=genomic_path,
            output_dir=str(matrices_dir),
            ref_fasta=ref_fasta,
        )
        X_raw = load_artifact_filtered_binary_matrix(
            artifact_root=matrices_dir,
            fallback_matrix=X_raw_unfiltered,
        )
        log_stage_complete(
            logger,
            1,
            "load and preprocess genomic matrix",
            samples=int(X_raw.shape[0]),
            features=int(X_raw.shape[1]),
        )

        feature_manifest_path = find_feature_manifest(matrices_dir)
        feature_manifest_df = load_feature_manifest(feature_manifest_path)
        if feature_manifest_path is None:
            logger.warning(
                "No feature manifest was found under %s. Raw-sequence query mode "
                "will be unavailable for this registry unless a manifest is supplied.",
                str(matrices_dir),
            )
        else:
            logger.info("Using feature manifest for query annotation: %s", str(feature_manifest_path))

        log_stage_start(logger, 2, "load metadata and align hierarchy labels")
        meta = self.loader.load_metadata(meta_path, output_dir=str(out))
        X, labels_df = align_hierarchy_labels(
            X=X_raw,
            meta=meta,
            hierarchy_labels=labels,
        )
        log_flow_step(
            logger,
            step="Multi-level hierarchy alignment checkpoint",
            happened="Aligned the feature matrix to samples with every requested hierarchy label.",
            reason="Recursive hierarchy training requires each retained sample to carry all supervised labels used along the path.",
            before_samples=int(X_raw.shape[0]),
            before_features=int(X_raw.shape[1]),
            after_samples=int(X.shape[0]),
            after_features=int(X.shape[1]),
            threshold="sample_id present in matrix and all requested hierarchy labels",
            status="complete",
        )
        log_stage_complete(
            logger,
            2,
            "load metadata and align hierarchy labels",
            samples=int(X.shape[0]),
            features=int(X.shape[1]),
            levels=int(len(labels)),
        )

        X.to_csv(out / "aligned_hierarchy_matrix.csv")
        aligned_labels_df = labels_df.copy()
        aligned_labels_df.insert(0, "sample_id", aligned_labels_df.index.astype(str))
        aligned_labels_df.to_csv(out / "aligned_hierarchy_labels.csv", index=False)

        explicit_min = (
            int(min_samples_per_node)
            if min_samples_per_node is not None
            else getattr(self.config, "min_level2_samples_per_group", None)
        )
        explicit_min = int(explicit_min) if explicit_min is not None else None

        log_stage_start(logger, 3, "recursive hierarchy model training")
        root_node = self._train_hierarchy_node(
            X=X,
            labels_df=labels_df,
            hierarchy_labels=labels,
            level_index=0,
            sample_index=list(X.index),
            node_dir=hierarchy_dir / "level_1_root",
            feature_manifest_df=feature_manifest_df,
            algorithm=algorithm,
            explicit_min_samples=explicit_min,
            path=[] ,
        )
        log_stage_complete(logger, 3, "recursive hierarchy model training")

        log_stage_start(logger, 4, "terminal hierarchy fallback training")
        terminal_fallbacks = self._train_hierarchy_terminal_fallbacks(
            X=X,
            labels_df=labels_df,
            hierarchy_labels=labels,
            output_dir=hierarchy_dir / "terminal_fallbacks",
            feature_manifest_df=feature_manifest_df,
            algorithm=algorithm,
        )
        log_stage_complete(
            logger,
            4,
            "terminal hierarchy fallback training",
            status=terminal_fallbacks.get("status"),
            target=terminal_fallbacks.get("target_label_column"),
        )

        registry = {
            "protocol": "multi_level_hierarchy_protocol",
            "compatible_with": "NetworkParser hierarchical training registry",
            "hierarchy": {
                "label_columns": labels,
                "n_levels": int(len(labels)),
                "root": root_node,
                "terminal_fallbacks": terminal_fallbacks,
            },
            "training_matrix": {
                "aligned_matrix_csv": str(out / "aligned_hierarchy_matrix.csv"),
                "aligned_labels_csv": str(out / "aligned_hierarchy_labels.csv"),
                "feature_manifest_file": str(feature_manifest_path) if feature_manifest_path else None,
            },
            "publication_summary": {
                "cohort_statistics": {
                    "aligned_samples": int(X.shape[0]),
                    "artifact_filtered_features": int(X.shape[1]),
                    "hierarchy_levels": int(len(labels)),
                    "classes_per_level": {
                        str(label): int(labels_df[label].nunique(dropna=True))
                        for label in labels
                    },
                },
                "model_tree": self._summarise_hierarchy_node(root_node),
            },
            "config": asdict(self.config) if is_dataclass(self.config) else vars(self.config),
        }

        registry_path = out / "hierarchical_model_registry.json"
        write_json(registry, registry_path)
        logger.info("Multi-level hierarchy training complete: %s", registry_path)
        return registry

    def _train_hierarchy_node(
        self,
        *,
        X: pd.DataFrame,
        labels_df: pd.DataFrame,
        hierarchy_labels: List[str],
        level_index: int,
        sample_index: List[Any],
        node_dir: Path,
        feature_manifest_df: Optional[pd.DataFrame],
        algorithm: Optional[str],
        explicit_min_samples: Optional[int],
        path: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Train one hierarchy node and recursively train child nodes."""
        node_dir = ensure_dir(Path(node_dir))
        label_column = hierarchy_labels[level_index]
        sample_index = [idx for idx in sample_index if idx in X.index]

        y_node = labels_df.loc[sample_index, label_column].astype(str).str.strip()
        y_node = y_node.replace({"": pd.NA, "-": pd.NA, "NA": pd.NA, "N/A": pd.NA, "None": pd.NA, "nan": pd.NA, "NaN": pd.NA}).dropna()
        common = X.index.intersection(y_node.index)
        X_node = X.loc[common].copy()
        y_node = y_node.loc[common].astype(str).copy()

        level_number = int(level_index + 1)
        stage_token = f"hierarchy_level_{level_number}__{safe_hierarchy_token(label_column, 40)}"
        if path:
            stage_token = stage_token + "__" + safe_hierarchy_token("__".join(item["value"] for item in path), 80)

        raw_diag = label_distribution_diagnostics(y_node, requested_cv_splits=5)
        node_payload: Dict[str, Any] = {
            "level_index": int(level_index),
            "level_number": level_number,
            "label_column": str(label_column),
            "path": list(path),
            "n_samples": int(X_node.shape[0]),
            "n_features_available": int(X_node.shape[1]),
            "label_diagnostics": raw_diag,
            "children": {},
        }

        if X_node.empty or raw_diag["n_classes"] == 0:
            node_payload.update({"status": "skipped", "reason": "no_aligned_samples_or_labels"})
            write_json(node_payload, node_dir / "node_summary.json")
            return node_payload

        # A single observed child label is a deterministic branch, not a failed model.
        if raw_diag["n_classes"] == 1:
            constant_label = str(y_node.iloc[0])
            node_payload.update(
                {
                    "status": "constant",
                    "reason": "single_child_label_in_parent_branch",
                    "constant_label": constant_label,
                    "features": [],
                    "model_file": None,
                    "feature_manifest": None,
                }
            )
            if level_index + 1 < len(hierarchy_labels):
                child_path = list(path) + [
                    {"level_number": str(level_number), "label_column": str(label_column), "value": constant_label}
                ]
                child_dir = node_dir / "children" / safe_hierarchy_token(constant_label)
                node_payload["children"][constant_label] = self._train_hierarchy_node(
                    X=X,
                    labels_df=labels_df,
                    hierarchy_labels=hierarchy_labels,
                    level_index=level_index + 1,
                    sample_index=list(y_node.index),
                    node_dir=child_dir,
                    feature_manifest_df=feature_manifest_df,
                    algorithm=algorithm,
                    explicit_min_samples=explicit_min_samples,
                    path=child_path,
                )
            write_json(node_payload, node_dir / "node_summary.json")
            return node_payload

        X_train = X_node
        y_train = y_node
        support_summary: Dict[str, Any] = {"status": "not_applied", "reason": "root_or_disabled"}
        if level_index > 0:
            X_train, y_train, support_summary = apply_level2_class_support_filter(
                X=X_node,
                y=y_node,
                output_dir=node_dir,
                config=self.config,
                stage_name=stage_token,
                requested_cv_splits=5,
            )

        label_diag = label_distribution_diagnostics(y_train, requested_cv_splits=5)
        adaptive_min = adaptive_level2_min_samples(label_diag)

        skip_reasons: List[str] = []
        if label_diag["n_classes"] < 2:
            skip_reasons.append("single_class_after_support_filter")
        elif label_diag["min_class_count"] < 2:
            skip_reasons.append("insufficient_per_label_support_for_stratified_cv")
        elif X_train.shape[0] < adaptive_min:
            skip_reasons.append("insufficient_samples_for_hierarchy_label_structure")
        if explicit_min_samples is not None and X_train.shape[0] < explicit_min_samples:
            skip_reasons.append("below_user_requested_absolute_node_minimum")

        node_payload.update(
            {
                "n_training_samples": int(X_train.shape[0]),
                "training_label_diagnostics": label_diag,
                "class_support_filter": support_summary,
                "adaptive_min_training_samples": int(adaptive_min),
                "explicit_absolute_min_samples": explicit_min_samples,
            }
        )

        if skip_reasons:
            counts = y_train.value_counts(dropna=True)
            support_table = pd.DataFrame(
                {
                    "label_state": counts.index.astype(str),
                    "sample_count": counts.astype(int).values,
                    "meets_twofold_cv_minimum": [int(v) >= 2 for v in counts.astype(int).values],
                }
            )
            support_path = node_dir / "hierarchy_label_support_diagnostics.tsv"
            support_table.to_csv(support_path, sep="\t", index=False)
            node_payload.update(
                {
                    "status": "skipped",
                    "reason": "+".join(skip_reasons),
                    "features": [],
                    "model_file": None,
                    "feature_manifest": None,
                    "label_support_diagnostics_file": str(support_path),
                }
            )
            write_json(node_payload, node_dir / "node_summary.json")
            return node_payload

        filter_result: Optional[Dict[str, Any]] = None
        X_filtered: Optional[pd.DataFrame] = None
        try:
            filter_result = run_configured_feature_filter(
                X=X_train,
                y=y_train,
                output_base_dir=node_dir,
                config=self.config,
                stage_name=stage_token,
            )
            filter_result = run_feature_panel_check_after_filter(
                filter_result=filter_result,
                y=y_train,
                output_base_dir=node_dir,
                config=self.config,
                stage_name=f"{stage_token}__model_matrix",
            )
            X_filtered = filter_result["filtered_matrix"]
            manifest_summary = write_selected_feature_manifest(
                features=list(X_filtered.columns),
                source_manifest=feature_manifest_df,
                output_path=node_dir / "selected_feature_manifest.tsv",
            )
            model_summary = train_model_safely(
                X=X_filtered,
                y=y_train.loc[X_filtered.index],
                output_dir=node_dir / "model",
                config=self.config,
                algorithm=algorithm,
                model_name=f"hierarchy_level_{level_number}_model",
            )
        except Exception as exc:
            failure_reason = classify_ml_failure(exc)
            matrix_for_diag = X_filtered if isinstance(X_filtered, pd.DataFrame) else X_train
            labels_for_diag = y_train.loc[matrix_for_diag.index] if isinstance(matrix_for_diag, pd.DataFrame) else y_train
            failure_diag = label_distribution_diagnostics(labels_for_diag, requested_cv_splits=5)
            failure_diag.update(
                {
                    "failure_reason": failure_reason,
                    "error": str(exc),
                    "n_features_at_failure": int(matrix_for_diag.shape[1]) if isinstance(matrix_for_diag, pd.DataFrame) else None,
                    "stage": "hierarchy_node_training",
                }
            )
            node_payload.update(
                {
                    "status": "skipped",
                    "reason": failure_reason,
                    "error": str(exc),
                    "training_failure_diagnostics": failure_diag,
                    "filter": filter_result.get("summary", {}) if isinstance(filter_result, dict) else {},
                    "feature_panel_separability": filter_result.get("feature_panel_separability", {}) if isinstance(filter_result, dict) else {},
                    "features": [],
                    "model_file": None,
                    "feature_manifest": None,
                }
            )
            write_json(node_payload, node_dir / "node_summary.json")
            return node_payload

        node_payload.update(
            {
                "status": "success",
                "filter": filter_result["summary"],
                "feature_panel_separability": filter_result.get("feature_panel_separability", {}),
                "model": model_summary,
                "features": list(X_filtered.columns),
                "model_file": get_model_file(model_summary),
                "feature_manifest": manifest_summary,
            }
        )

        if level_index + 1 < len(hierarchy_labels):
            for child_value in sorted([str(v) for v in y_train.dropna().unique()]):
                child_samples = list(y_train[y_train.astype(str) == child_value].index)
                child_path = list(path) + [
                    {"level_number": str(level_number), "label_column": str(label_column), "value": child_value}
                ]
                child_dir = node_dir / "children" / safe_hierarchy_token(child_value)
                node_payload["children"][child_value] = self._train_hierarchy_node(
                    X=X,
                    labels_df=labels_df,
                    hierarchy_labels=hierarchy_labels,
                    level_index=level_index + 1,
                    sample_index=child_samples,
                    node_dir=child_dir,
                    feature_manifest_df=feature_manifest_df,
                    algorithm=algorithm,
                    explicit_min_samples=explicit_min_samples,
                    path=child_path,
                )

        write_json(node_payload, node_dir / "node_summary.json")
        return node_payload

    def _terminal_fallback_skip_payload(
        self,
        *,
        status: str,
        reason: str,
        target_label_column: str,
        fallback_scope: str,
        output_dir: Path,
        conditioning_label_column: Optional[str] = None,
        conditioning_value: Optional[str] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a registry-safe skipped fallback payload."""
        payload: Dict[str, Any] = {
            "status": status,
            "reason": reason,
            "target_label_column": str(target_label_column),
            "fallback_scope": str(fallback_scope),
            "conditioning_label_column": conditioning_label_column,
            "conditioning_value": conditioning_value,
            "features": [],
            "model_file": None,
            "feature_manifest": None,
            "filter": {},
            "feature_panel_separability": {},
            "diagnostics": diagnostics or {},
        }
        if error is not None:
            payload["error"] = str(error)
        write_json(payload, Path(output_dir) / "fallback_summary.json")
        return payload

    def _train_one_terminal_fallback_model(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        output_dir: Path,
        feature_manifest_df: Optional[pd.DataFrame],
        algorithm: Optional[str],
        target_label_column: str,
        fallback_scope: str,
        conditioning_label_column: Optional[str] = None,
        conditioning_value: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Train one terminal-label fallback model using the normal filtering path.

        This is still training-time feature discovery: central statistical
        filtering and panel selection run before model fitting. Query mode will
        later use this model only when a deeper hierarchy branch is unavailable.
        """
        output_dir = ensure_dir(Path(output_dir))

        y_clean = pd.Series(y, index=y.index).astype(str).str.strip()
        y_clean = y_clean.replace({
            "": pd.NA,
            "-": pd.NA,
            "NA": pd.NA,
            "N/A": pd.NA,
            "None": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
        }).dropna()

        common = X.index.intersection(y_clean.index)
        X0 = X.loc[common].copy()
        y0 = y_clean.loc[common].astype(str).copy()

        stage_token = "hierarchy_terminal_fallback"
        if conditioning_label_column is not None:
            stage_token += f"__by_{safe_hierarchy_token(conditioning_label_column, 40)}"
        if conditioning_value is not None:
            stage_token += f"__{safe_hierarchy_token(conditioning_value, 60)}"

        X_train, y_train, support_summary = apply_level2_class_support_filter(
            X=X0,
            y=y0,
            output_dir=output_dir,
            config=self.config,
            stage_name=stage_token,
            requested_cv_splits=5,
        )
        label_diag = label_distribution_diagnostics(y_train, requested_cv_splits=5)
        adaptive_min = adaptive_level2_min_samples(label_diag)

        skip_reasons: List[str] = []
        if label_diag["n_classes"] < 2:
            skip_reasons.append("single_class_after_support_filter")
        elif label_diag["min_class_count"] < 2:
            skip_reasons.append("insufficient_per_label_support_for_stratified_cv")
        elif X_train.shape[0] < adaptive_min:
            skip_reasons.append("insufficient_samples_for_terminal_fallback")

        diagnostics = {
            "n_samples": int(X_train.shape[0]),
            "n_features_available": int(X_train.shape[1]),
            "label_diagnostics": label_diag,
            "class_support_filter": support_summary,
            "adaptive_min_training_samples": int(adaptive_min),
        }

        if skip_reasons:
            return self._terminal_fallback_skip_payload(
                status="skipped",
                reason="+".join(skip_reasons),
                target_label_column=target_label_column,
                fallback_scope=fallback_scope,
                output_dir=output_dir,
                conditioning_label_column=conditioning_label_column,
                conditioning_value=conditioning_value,
                diagnostics=diagnostics,
            )

        filter_result: Optional[Dict[str, Any]] = None
        X_filtered: Optional[pd.DataFrame] = None
        try:
            filter_result = run_configured_feature_filter(
                X=X_train,
                y=y_train,
                output_base_dir=output_dir,
                config=self.config,
                stage_name=stage_token,
            )
            filter_result = run_feature_panel_check_after_filter(
                filter_result=filter_result,
                y=y_train,
                output_base_dir=output_dir,
                config=self.config,
                stage_name=f"{stage_token}__model_matrix",
            )
            X_filtered = filter_result["filtered_matrix"]
            manifest_summary = write_selected_feature_manifest(
                features=list(X_filtered.columns),
                source_manifest=feature_manifest_df,
                output_path=output_dir / "selected_feature_manifest.tsv",
            )
            model_summary = train_model_safely(
                X=X_filtered,
                y=y_train.loc[X_filtered.index],
                output_dir=output_dir / "model",
                config=self.config,
                algorithm=algorithm,
                model_name="hierarchy_terminal_fallback_model",
            )
        except Exception as exc:
            failure_diag = diagnostics.copy()
            failure_diag.update(
                {
                    "failure_reason": classify_ml_failure(exc),
                    "error": str(exc),
                    "n_features_at_failure": int(X_filtered.shape[1]) if isinstance(X_filtered, pd.DataFrame) else int(X_train.shape[1]),
                }
            )
            return self._terminal_fallback_skip_payload(
                status="skipped",
                reason=classify_ml_failure(exc),
                target_label_column=target_label_column,
                fallback_scope=fallback_scope,
                output_dir=output_dir,
                conditioning_label_column=conditioning_label_column,
                conditioning_value=conditioning_value,
                diagnostics=failure_diag,
                error=str(exc),
            )

        payload = {
            "status": "success",
            "reason": "trained_terminal_fallback",
            "target_label_column": str(target_label_column),
            "fallback_scope": str(fallback_scope),
            "conditioning_label_column": conditioning_label_column,
            "conditioning_value": conditioning_value,
            "n_training_samples": int(X_filtered.shape[0]),
            "training_label_diagnostics": label_diag,
            "class_support_filter": support_summary,
            "filter": filter_result["summary"],
            "feature_panel_separability": filter_result.get("feature_panel_separability", {}),
            "model": model_summary,
            "features": list(X_filtered.columns),
            "model_file": get_model_file(model_summary),
            "feature_manifest": manifest_summary,
        }
        write_json(payload, output_dir / "fallback_summary.json")
        return payload

    def _train_hierarchy_terminal_fallbacks(
        self,
        *,
        X: pd.DataFrame,
        labels_df: pd.DataFrame,
        hierarchy_labels: List[str],
        output_dir: Path,
        feature_manifest_df: Optional[pd.DataFrame],
        algorithm: Optional[str],
    ) -> Dict[str, Any]:
        """Train broad terminal-label fallback models for recursive hierarchy query.

        For a hierarchy such as Country -> Lineage -> Pheno, this trains:
          1. Pheno within each Lineage across all countries.
          2. A global Pheno fallback across all aligned samples.

        These fallbacks do not replace the hierarchy. They are used only when a
        deeper branch is skipped because the country-specific slice is too small
        or otherwise untrainable.
        """
        output_dir = ensure_dir(Path(output_dir))
        labels = [str(label).strip() for label in hierarchy_labels if str(label).strip()]
        target_label = labels[-1]
        parent_label = labels[-2] if len(labels) >= 2 else None

        payload: Dict[str, Any] = {
            "status": "started",
            "target_label_column": str(target_label),
            "preferred_parent_label_column": str(parent_label) if parent_label else None,
            "fallback_order": ["by_parent_label", "global"],
            "by_parent_label": {},
            "global": {},
        }

        if target_label not in labels_df.columns:
            payload.update({"status": "skipped", "reason": "target_label_missing_from_aligned_metadata"})
            write_json(payload, output_dir / "terminal_fallbacks_summary.json")
            return payload

        if parent_label is not None and parent_label in labels_df.columns:
            parent_block: Dict[str, Any] = {
                "label_column": str(parent_label),
                "target_label_column": str(target_label),
                "models": {},
            }
            parent_values = sorted([str(v) for v in labels_df[parent_label].dropna().astype(str).str.strip().unique() if str(v).strip()])
            for parent_value in parent_values:
                idx = labels_df.index[labels_df[parent_label].astype(str).str.strip() == parent_value]
                model_dir = output_dir / "by_parent_label" / safe_hierarchy_token(str(parent_label)) / safe_hierarchy_token(parent_value)
                parent_block["models"][parent_value] = self._train_one_terminal_fallback_model(
                    X=X.loc[X.index.intersection(idx)].copy(),
                    y=labels_df.loc[idx, target_label],
                    output_dir=model_dir,
                    feature_manifest_df=feature_manifest_df,
                    algorithm=algorithm,
                    target_label_column=target_label,
                    fallback_scope="by_parent_label",
                    conditioning_label_column=parent_label,
                    conditioning_value=parent_value,
                )
            payload["by_parent_label"][str(parent_label)] = parent_block

        payload["global"] = self._train_one_terminal_fallback_model(
            X=X.copy(),
            y=labels_df[target_label],
            output_dir=output_dir / "global",
            feature_manifest_df=feature_manifest_df,
            algorithm=algorithm,
            target_label_column=target_label,
            fallback_scope="global_terminal",
            conditioning_label_column=None,
            conditioning_value=None,
        )

        n_parent_success = 0
        for block in payload.get("by_parent_label", {}).values():
            if isinstance(block, dict):
                for model_payload in block.get("models", {}).values():
                    if isinstance(model_payload, dict) and model_payload.get("status") == "success":
                        n_parent_success += 1

        global_success = isinstance(payload.get("global"), dict) and payload["global"].get("status") == "success"
        payload.update(
            {
                "status": "success" if (n_parent_success > 0 or global_success) else "skipped",
                "reason": "trained_at_least_one_terminal_fallback" if (n_parent_success > 0 or global_success) else "no_trainable_terminal_fallbacks",
                "n_parent_conditioned_fallbacks": int(n_parent_success),
                "global_fallback_available": bool(global_success),
            }
        )
        write_json(payload, output_dir / "terminal_fallbacks_summary.json")
        return payload

    def _summarise_hierarchy_node(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Return a compact node summary for reports without duplicating model payloads."""
        children = node.get("children", {}) if isinstance(node, dict) else {}
        return {
            "level_number": node.get("level_number"),
            "label_column": node.get("label_column"),
            "status": node.get("status"),
            "reason": node.get("reason"),
            "n_samples": node.get("n_samples"),
            "n_training_samples": node.get("n_training_samples"),
            "n_classes": (node.get("training_label_diagnostics") or node.get("label_diagnostics") or {}).get("n_classes"),
            "retained_features": int(len(node.get("features", []))) if isinstance(node.get("features", []), list) else 0,
            "model_file": node.get("model_file"),
            "constant_branch": node.get("constant_label") is not None,
            "children": {
                str(key): self._summarise_hierarchy_node(value)
                for key, value in children.items()
                if isinstance(value, dict)
            },
        }

    def train(
        self,
        genomic_path: str,
        meta_path: str,
        level1_label: str,
        level2_label: str,
        output_dir: str,
        ref_fasta: Optional[str] = None,
        algorithm: Optional[str] = None,
        train_global_level2: bool = True,
        min_level2_samples_per_group: Optional[int] = None,
        global_level2_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        out = ensure_dir(Path(output_dir))
        matrices_dir = ensure_dir(out / "matrices")
        level1_dir = ensure_dir(out / "level1_strain_identity")
        level2_dir = ensure_dir(out / "level2_resistance_profile")

        log_pipeline_header(
            logger,
            "NetworkParser two-level training started",
            central_filter=getattr(self.config, "central_feature_filter_method", "auto"),
            feature_panel_check=bool(getattr(self.config, "run_feature_panel_separability_check", True)),
            n_jobs=getattr(self.config, "n_jobs", "NA"),
        )

        with PipelineProgress(
            _planned_two_level_stages(self.config, train_global_level2),
            title="Two-level training",
        ) as pipeline_progress:
            return self._train_two_level_body(
                genomic_path=genomic_path,
                meta_path=meta_path,
                level1_label=level1_label,
                level2_label=level2_label,
                out=out,
                matrices_dir=matrices_dir,
                level1_dir=level1_dir,
                level2_dir=level2_dir,
                ref_fasta=ref_fasta,
                algorithm=algorithm,
                train_global_level2=train_global_level2,
                min_level2_samples_per_group=min_level2_samples_per_group,
                global_level2_label=global_level2_label,
                pipeline_progress=pipeline_progress,
            )

    def _train_two_level_body(
        self,
        *,
        genomic_path: str,
        meta_path: str,
        level1_label: str,
        level2_label: str,
        out: Path,
        matrices_dir: Path,
        level1_dir: Path,
        level2_dir: Path,
        ref_fasta: Optional[str],
        algorithm: Optional[str],
        train_global_level2: bool,
        min_level2_samples_per_group: Optional[int],
        global_level2_label: Optional[str],
        pipeline_progress: PipelineProgress,
    ) -> Dict[str, Any]:
        log_stage_start(
            logger,
            1,
            "load and preprocess genomic matrix",
            progress=pipeline_progress,
        )
        X_raw_unfiltered = self.loader.load_genomic_matrix(
            file_path=genomic_path,
            output_dir=str(matrices_dir),
            ref_fasta=ref_fasta,
        )

        X_raw = load_artifact_filtered_binary_matrix(
            artifact_root=matrices_dir,
            fallback_matrix=X_raw_unfiltered,
        )
        log_stage_complete(
            logger,
            1,
            "load and preprocess genomic matrix",
            progress=pipeline_progress,
            samples=int(X_raw.shape[0]),
            features=int(X_raw.shape[1]),
        )

        feature_manifest_path = find_feature_manifest(matrices_dir)
        feature_manifest_df = load_feature_manifest(feature_manifest_path)
        if feature_manifest_path is None:
            logger.warning(
                "No feature manifest was found under %s. Raw-sequence query mode "
                "will be unavailable for this registry unless a manifest is supplied.",
                str(matrices_dir),
            )
        else:
            logger.info("Using feature manifest for query annotation: %s", str(feature_manifest_path))

        log_stage_start(
            logger,
            2,
            "load metadata and align two labels",
            progress=pipeline_progress,
        )
        meta = self.loader.load_metadata(meta_path, output_dir=str(out))

        X, y_level1, y_level2 = align_two_labels(
            X=X_raw,
            meta=meta,
            level1_label=level1_label,
            level2_label=level2_label,
        )

        log_flow_step(
            logger,
            step="Two-level alignment checkpoint",
            happened="Aligned the feature matrix to samples with both Level-1 and Level-2 labels.",
            reason="Hierarchical training needs the same retained sample set for placement and downstream phenotype/profile interpretation.",
            before_samples=int(X_raw.shape[0]),
            before_features=int(X_raw.shape[1]),
            after_samples=int(X.shape[0]),
            after_features=int(X.shape[1]),
            threshold="sample_id present in matrix and both supervised labels",
            status="complete",
        )
        log_stage_complete(
            logger,
            2,
            "load metadata and align two labels",
            progress=pipeline_progress,
            samples=int(X.shape[0]),
            features=int(X.shape[1]),
        )

        X_global_level2_base, y_level2_global_base, global_level2_label_summary = build_global_level2_training_labels(
            X=X,
            meta=meta,
            y_level2=y_level2,
            level2_label=level2_label,
            global_level2_label=global_level2_label,
            config=self.config,
        )

        explicit_group_min_n = (
            int(min_level2_samples_per_group)
            if min_level2_samples_per_group is not None
            else getattr(self.config, "min_level2_samples_per_group", None)
        )
        explicit_group_min_n = (
            int(explicit_group_min_n)
            if explicit_group_min_n is not None
            else None
        )

        X.to_csv(out / "aligned_two_level_matrix.csv")
        aligned_labels_df = pd.DataFrame(
            {
                "sample_id": X.index.astype(str),
                "level1_label": y_level1.astype(str).values,
                "level2_label": y_level2.astype(str).values,
            }
        )
        aligned_labels_df["global_level2_label"] = (
            y_level2_global_base.reindex(X.index).astype("object").where(lambda v: ~v.isna(), "").values
        )
        aligned_labels_df.to_csv(out / "aligned_two_level_labels.csv", index=False)
        write_json(global_level2_label_summary, out / "global_level2_label_summary.json")

        # ------------------------------------------------------------------
        # Level 1: strain / lineage / group placement
        # ------------------------------------------------------------------
        log_stage_start(
            logger,
            3,
            "Level 1 placement filtering and model training",
            progress=pipeline_progress,
        )
        level1_filter = run_configured_feature_filter(
            X=X,
            y=y_level1,
            output_base_dir=level1_dir,
            config=self.config,
            stage_name="level1_strain_identity",
        )
        level1_filter = run_feature_panel_check_after_filter(
            filter_result=level1_filter,
            y=y_level1,
            output_base_dir=level1_dir,
            config=self.config,
            stage_name="level1_strain_identity_model_matrix",
        )
        X_level1 = level1_filter["filtered_matrix"]
        level1_manifest = write_selected_feature_manifest(
            features=list(X_level1.columns),
            source_manifest=feature_manifest_df,
            output_path=level1_dir / "selected_feature_manifest.tsv",
        )

        level1_model = train_model_safely(
            X=X_level1,
            y=y_level1.loc[X_level1.index],
            output_dir=level1_dir / "model",
            config=self.config,
            algorithm=algorithm,
            model_name="level1_strain_identity_model",
        )
        log_stage_complete(
            logger,
            3,
            "Level 1 placement filtering and model training",
            progress=pipeline_progress,
            features=int(X_level1.shape[1]),
            status=level1_model.get("status", "complete") if isinstance(level1_model, dict) else "complete",
        )

        # ------------------------------------------------------------------
        # Level 2 global fallback: resistance prediction across all samples
        # ------------------------------------------------------------------
        global_level2_payload: Dict[str, Any] = {"status": "skipped"}
        if train_global_level2:
            pipeline_progress.begin_stage("global Level-2 fallback training")
            global_dir = ensure_dir(level2_dir / "global_fallback")
            X_level2_global_source, y_level2_global_source, global_class_support = apply_level2_class_support_filter(
                X=X_global_level2_base,
                y=y_level2_global_base,
                output_dir=global_dir,
                config=self.config,
                stage_name="level2_global_resistance_profile",
                requested_cv_splits=5,
            )
            global_label_diag = label_distribution_diagnostics(y_level2_global_source, requested_cv_splits=5)

            if global_label_diag["n_classes"] < 2:
                global_level2_payload = {
                    "status": "skipped",
                    "reason": "insufficient_global_level2_class_diversity_after_support_filter",
                    "target_type": "global_level2_fallback",
                    "target_label_column": global_level2_label_summary.get("global_level2_label_column"),
                    "target_label_source": global_level2_label_summary.get("source"),
                    "global_level2_label_summary": global_level2_label_summary,
                    "level2_class_support_filter": global_class_support,
                    "level2_label_diagnostics": global_label_diag,
                }
                write_json(global_level2_payload, global_dir / "global_level2_summary.json")
            elif global_label_diag["min_class_count"] < 2:
                global_level2_payload = {
                    "status": "skipped",
                    "reason": "insufficient_global_level2_class_count_for_stratified_cv",
                    "target_type": "global_level2_fallback",
                    "target_label_column": global_level2_label_summary.get("global_level2_label_column"),
                    "target_label_source": global_level2_label_summary.get("source"),
                    "global_level2_label_summary": global_level2_label_summary,
                    "level2_class_support_filter": global_class_support,
                    "level2_label_diagnostics": global_label_diag,
                }
                write_json(global_level2_payload, global_dir / "global_level2_summary.json")
            else:
                global_filter: Optional[Dict[str, Any]] = None
                X_global: Optional[pd.DataFrame] = None
                try:
                    global_filter = run_configured_feature_filter(
                        X=X_level2_global_source,
                        y=y_level2_global_source,
                        output_base_dir=global_dir,
                        config=self.config,
                        stage_name="level2_global_resistance_profile",
                    )
                    global_filter = run_feature_panel_check_after_filter(
                        filter_result=global_filter,
                        y=y_level2_global_source,
                        output_base_dir=global_dir,
                        config=self.config,
                        stage_name="level2_global_resistance_profile_model_matrix",
                    )
                    X_global = global_filter["filtered_matrix"]
                    global_manifest = write_selected_feature_manifest(
                        features=list(X_global.columns),
                        source_manifest=feature_manifest_df,
                        output_path=global_dir / "selected_feature_manifest.tsv",
                    )
                    global_model = train_model_safely(
                        X=X_global,
                        y=y_level2_global_source.loc[X_global.index],
                        output_dir=global_dir / "model",
                        config=self.config,
                        algorithm=algorithm,
                        model_name="level2_global_resistance_model",
                    )
                    global_level2_payload = {
                        "status": "success",
                        "filter": global_filter["summary"],
                        "feature_panel_separability": global_filter.get("feature_panel_separability", {}),
                        "model": global_model,
                        "features": list(X_global.columns),
                        "model_file": get_model_file(global_model),
                        "feature_manifest": global_manifest,
                        "target_type": "global_level2_fallback",
                        "target_label_column": global_level2_label_summary.get("global_level2_label_column"),
                        "target_label_source": global_level2_label_summary.get("source"),
                        "global_level2_label_summary": global_level2_label_summary,
                        "level2_class_support_filter": global_class_support,
                        "level2_label_diagnostics": global_label_diag,
                    }
                    write_json(global_level2_payload, global_dir / "global_level2_summary.json")
                except Exception as exc:
                    failure_reason = classify_ml_failure(exc)
                    matrix_for_diag = X_global if isinstance(X_global, pd.DataFrame) else X_level2_global_source
                    labels_for_diag = y_level2_global_source.loc[matrix_for_diag.index]
                    failure_diag = label_distribution_diagnostics(labels_for_diag, requested_cv_splits=5)
                    failure_diag.update(
                        {
                            "failure_reason": failure_reason,
                            "error": str(exc),
                            "n_features_at_failure": int(matrix_for_diag.shape[1]),
                            "stage": "global_level2_training",
                        }
                    )
                    logger.warning(
                        "Global Level-2 model skipped because %s | samples=%d | "
                        "features=%d | classes=%d | min_class_count=%d | feasible_cv_splits=%d",
                        failure_reason,
                        int(failure_diag["n_samples"]),
                        int(failure_diag["n_features_at_failure"]),
                        int(failure_diag["n_classes"]),
                        int(failure_diag["min_class_count"]),
                        int(failure_diag["feasible_selector_cv_splits"]),
                    )
                    logger.debug("Global Level-2 training traceback", exc_info=True)
                    global_level2_payload = {
                        "status": "skipped",
                        "reason": failure_reason,
                        "error": str(exc),
                        "training_failure_diagnostics": failure_diag,
                        "filter": global_filter.get("summary", {}) if isinstance(global_filter, dict) else {},
                        "feature_panel_separability": global_filter.get("feature_panel_separability", {}) if isinstance(global_filter, dict) else {},
                        "target_type": "global_level2_fallback",
                        "target_label_column": global_level2_label_summary.get("global_level2_label_column"),
                        "target_label_source": global_level2_label_summary.get("source"),
                        "global_level2_label_summary": global_level2_label_summary,
                        "level2_class_support_filter": global_class_support,
                        "level2_label_diagnostics": global_label_diag,
                    }
                    write_json(global_level2_payload, global_dir / "global_level2_summary.json")
            pipeline_progress.complete_stage("global Level-2 fallback training")

        global_binary_level2_payload: Dict[str, Any] = {"status": "skipped", "reason": "disabled"}
        # ------------------------------------------------------------------
        # Optional global binary Level-2 fallback
        # ------------------------------------------------------------------
        if getattr(self.config, "level2_train_binary_global_fallback", False):
            binary_dir = ensure_dir(level2_dir / "global_binary_fallback")
            log_stage_start(
                logger,
                "5",
                "optional global binary Level-2 fallback",
                progress=pipeline_progress,
            )

            y_binary_source, binary_label_summary = build_level2_binary_labels(
                X=X,
                meta=meta,
                y_level2=y_level2,
                config=self.config,
                output_dir=binary_dir,
                level2_label=level2_label,
            )

            if y_binary_source is None:
                global_binary_level2_payload = {
                    "status": "skipped",
                    "reason": binary_label_summary.get("reason", "binary_label_unavailable"),
                    "target_type": "global_binary_resistant_susceptible",
                    "binary_label_summary": binary_label_summary,
                }
                write_json(global_binary_level2_payload, binary_dir / "global_binary_level2_summary.json")
                log_branch_decision(
                    logger,
                    "global binary Level-2 fallback",
                    "skipped",
                    reason=global_binary_level2_payload["reason"],
                )
                log_stage_complete(
                    logger,
                    "5",
                    "optional global binary Level-2 fallback",
                    progress=pipeline_progress,
                    status="skipped",
                )
            else:
                X_binary_source = X.loc[X.index.intersection(y_binary_source.index)].copy()
                y_binary_source = y_binary_source.loc[X_binary_source.index].copy()
                X_binary_train, y_binary_train, binary_class_support = apply_level2_class_support_filter(
                    X=X_binary_source,
                    y=y_binary_source,
                    output_dir=binary_dir,
                    config=self.config,
                    stage_name="global_binary_resistant_susceptible",
                    requested_cv_splits=5,
                )
                binary_label_diag = label_distribution_diagnostics(y_binary_train, requested_cv_splits=5)

                log_flow_step(
                    logger,
                    step="Level-2 binary fallback checkpoint — label support",
                    happened="Checked whether the optional binary Level-2 endpoint has enough class support for model training.",
                    reason="A fallback model should only be trained when the endpoint has at least two supported classes and feasible stratified probes.",
                    before_samples=int(X_binary_source.shape[0]),
                    before_features=int(X_binary_source.shape[1]),
                    after_samples=int(X_binary_train.shape[0]),
                    after_features=int(X_binary_train.shape[1]),
                    threshold=f"min_class_count={getattr(self.config, 'level2_min_class_count', 2)}",
                    status="eligible" if binary_label_diag["n_classes"] >= 2 and binary_label_diag["min_class_count"] >= 2 else "not_eligible",
                    artifact=binary_class_support.get("artifacts", {}).get("summary_json"),
                )

                if binary_label_diag["n_classes"] < 2 or binary_label_diag["min_class_count"] < 2:
                    reason = (
                        "insufficient_binary_class_diversity"
                        if binary_label_diag["n_classes"] < 2
                        else "insufficient_binary_class_count_for_stratified_cv"
                    )
                    global_binary_level2_payload = {
                        "status": "skipped",
                        "reason": reason,
                        "target_type": "global_binary_resistant_susceptible",
                        "binary_label_summary": binary_label_summary,
                        "level2_class_support_filter": binary_class_support,
                        "level2_label_diagnostics": binary_label_diag,
                    }
                    write_json(global_binary_level2_payload, binary_dir / "global_binary_level2_summary.json")
                    log_branch_decision(
                        logger,
                        "global binary Level-2 fallback",
                        "skipped",
                        reason=reason,
                        samples=int(binary_label_diag["n_samples"]),
                        classes=int(binary_label_diag["n_classes"]),
                    )
                    log_stage_complete(
                        logger,
                        "5",
                        "optional global binary Level-2 fallback",
                        progress=pipeline_progress,
                        status="skipped",
                    )
                else:
                    binary_filter: Optional[Dict[str, Any]] = None
                    X_binary_filtered: Optional[pd.DataFrame] = None
                    try:
                        binary_filter = run_configured_feature_filter(
                            X=X_binary_train,
                            y=y_binary_train,
                            output_base_dir=binary_dir,
                            config=self.config,
                            stage_name="level2_global_binary_resistant_susceptible",
                        )
                        binary_filter = run_feature_panel_check_after_filter(
                            filter_result=binary_filter,
                            y=y_binary_train,
                            output_base_dir=binary_dir,
                            config=self.config,
                            stage_name="level2_global_binary_resistant_susceptible_model_matrix",
                        )
                        X_binary_filtered = binary_filter["filtered_matrix"]
                        binary_manifest = write_selected_feature_manifest(
                            features=list(X_binary_filtered.columns),
                            source_manifest=feature_manifest_df,
                            output_path=binary_dir / "selected_feature_manifest.tsv",
                        )
                        binary_model = train_model_safely(
                            X=X_binary_filtered,
                            y=y_binary_train.loc[X_binary_filtered.index],
                            output_dir=binary_dir / "model",
                            config=self.config,
                            algorithm=algorithm,
                            model_name="level2_global_binary_resistance_model",
                        )

                        support_table = pd.DataFrame({
                            "binary_class": list(binary_class_support.get("class_support_table", {}).keys()),
                            "sample_count": list(binary_class_support.get("class_support_table", {}).values()),
                        })
                        support_table_path = binary_dir / "global_binary_class_support_diagnostics.tsv"
                        support_table.to_csv(support_table_path, sep="\t", index=False)

                        global_binary_level2_payload = {
                            "status": "success",
                            "target_type": "global_binary_resistant_susceptible",
                            "description": "binary Level-2 endpoint trained across all Level-1 groups",
                            "filter": binary_filter.get("summary", {}),
                            "feature_panel_separability": binary_filter.get("feature_panel_separability", {}),
                            "model": binary_model,
                            "features": list(X_binary_filtered.columns),
                            "model_file": get_model_file(binary_model),
                            "feature_manifest": binary_manifest,
                            "binary_label_summary": binary_label_summary,
                            "level2_class_support_filter": binary_class_support,
                            "level2_label_diagnostics": binary_label_diag,
                            "class_support_table": binary_class_support.get("class_support_table", {}),
                            "diagnostics_file": str(support_table_path),
                        }
                        write_json(global_binary_level2_payload, binary_dir / "global_binary_level2_summary.json")
                        log_stage_complete(
                            logger,
                            "5",
                            "optional global binary Level-2 fallback",
                            progress=pipeline_progress,
                            status="success",
                            features=int(X_binary_filtered.shape[1]),
                        )
                    except Exception as exc:
                        failure_reason = classify_ml_failure(exc)
                        matrix_for_diag = X_binary_filtered if isinstance(X_binary_filtered, pd.DataFrame) else X_binary_train
                        labels_for_diag = y_binary_train.loc[matrix_for_diag.index]
                        failure_diag = label_distribution_diagnostics(labels_for_diag, requested_cv_splits=5)
                        failure_diag.update({
                            "failure_reason": failure_reason,
                            "error": str(exc),
                            "n_features_at_failure": int(matrix_for_diag.shape[1]),
                            "stage": "global_binary_level2_training",
                        })
                        global_binary_level2_payload = {
                            "status": "skipped",
                            "reason": failure_reason,
                            "error": str(exc),
                            "target_type": "global_binary_resistant_susceptible",
                            "training_failure_diagnostics": failure_diag,
                            "filter": binary_filter.get("summary", {}) if isinstance(binary_filter, dict) else {},
                            "feature_panel_separability": binary_filter.get("feature_panel_separability", {}) if isinstance(binary_filter, dict) else {},
                            "binary_label_summary": binary_label_summary,
                            "level2_class_support_filter": binary_class_support,
                            "level2_label_diagnostics": binary_label_diag,
                        }
                        write_json(global_binary_level2_payload, binary_dir / "global_binary_level2_summary.json")
                        logger.warning(
                            "Global binary Level-2 fallback skipped because %s | samples=%d | features=%d | classes=%d | min_class_count=%d | feasible_cv_splits=%d",
                            failure_reason,
                            int(failure_diag["n_samples"]),
                            int(failure_diag["n_features_at_failure"]),
                            int(failure_diag["n_classes"]),
                            int(failure_diag["min_class_count"]),
                            int(failure_diag["feasible_selector_cv_splits"]),
                        )
                        logger.debug("Global binary Level-2 training traceback", exc_info=True)
                        log_stage_complete(
                            logger,
                            "5",
                            "optional global binary Level-2 fallback",
                            progress=pipeline_progress,
                            status="skipped",
                        )

        # ------------------------------------------------------------------
        # Level 2 per level-1 group: resistance prediction within placement
        # ------------------------------------------------------------------
        pipeline_progress.begin_stage("per-group Level-2 training")
        subgroup_payload: Dict[str, Any] = {}
        group_values = sorted(y_level1.astype(str).unique())
        for group_value in progress_iter(
            group_values,
            desc="Level-2 per-group training",
            unit="group",
            leave=False,
        ):
            group_mask = y_level1.astype(str) == str(group_value)
            group_samples = y_level1.index[group_mask]
            X_group = X.loc[group_samples].copy()
            y2_group = y_level2.loc[group_samples].copy()

            safe_group_name = str(group_value).replace("/", "_").replace(" ", "_")
            group_dir = ensure_dir(level2_dir / "by_level1_group" / safe_group_name)

            raw_label_diag = label_distribution_diagnostics(y2_group, requested_cv_splits=5)
            X_group_train, y2_group_train, group_class_support = apply_level2_class_support_filter(
                X=X_group,
                y=y2_group,
                output_dir=group_dir,
                config=self.config,
                stage_name=f"level2_resistance_profile__{safe_group_name}",
                requested_cv_splits=5,
            )
            label_diag = label_distribution_diagnostics(y2_group_train, requested_cv_splits=5)
            adaptive_min_group_n = adaptive_level2_min_samples(label_diag)
            group_summary: Dict[str, Any] = {
                "level1_group": str(group_value),
                "n_samples": int(X_group.shape[0]),
                "n_training_samples": int(X_group_train.shape[0]),
                "n_level2_classes": int(label_diag["n_classes"]),
                "adaptive_min_training_samples": int(adaptive_min_group_n),
                "adaptive_min_rule": "n_level2_classes * 2_samples_per_label",
                "explicit_absolute_min_samples": explicit_group_min_n,
                "level2_label_diagnostics_before_support_filter": raw_label_diag,
                "level2_label_diagnostics": label_diag,
                "level2_class_support_filter": group_class_support,
            }

            fallback_source = _successful_level2_fallback_source(global_level2_payload, global_binary_level2_payload)
            fallback_available = fallback_source != "unavailable"

            skip_reasons: List[str] = []
            if label_diag["n_classes"] < 2:
                skip_reasons.append("single_level2_class_within_group")
            elif label_diag["min_class_count"] < 2:
                skip_reasons.append("insufficient_per_label_support_for_stratified_cv")
            elif X_group_train.shape[0] < adaptive_min_group_n:
                skip_reasons.append("insufficient_samples_for_level2_label_structure")

            if explicit_group_min_n is not None and X_group_train.shape[0] < explicit_group_min_n:
                skip_reasons.append("below_user_requested_absolute_group_minimum")

            # === SKIP DIAGNOSTICS ===
            if skip_reasons:
                # Always write a clear support table for reviewers. The JSON
                # summary stays generic, while this TSV gives auditable detail.
                support_table = group_class_support.get("class_support_table", {})
                skip_table = pd.DataFrame({
                    "level2_class": list(support_table.keys()),
                    "sample_count": list(support_table.values()),
                    "meets_twofold_cv_minimum": [int(v) >= 2 for v in support_table.values()],
                })

                skip_table_path = group_dir / "level2_class_support_diagnostics.tsv"
                skip_table.to_csv(skip_table_path, sep="\t", index=False)

                logger.info(
                    "Group %s: skipping group-specific Level-2 model because %s | "
                    "training_samples=%d | adaptive_min_samples=%d | labels=%d | "
                    "min_class_count=%d | feasible_cv_splits=%d | wrote_diagnostics=%s",
                    str(group_value),
                    "+".join(skip_reasons),
                    int(X_group_train.shape[0]),
                    int(adaptive_min_group_n),
                    int(label_diag["n_classes"]),
                    int(label_diag["min_class_count"]),
                    int(label_diag["feasible_selector_cv_splits"]),
                    str(skip_table_path),
                )

                group_summary.update({
                    "status": "skipped",
                    "reason": "+".join(skip_reasons),
                    "level2_class_support_diagnostics_file": str(skip_table_path),
                    "level2_class_support_table": support_table,
                    "level2_source_for_prediction": fallback_source,
                })

                write_json(group_summary, group_dir / "group_summary.json")
                subgroup_payload[str(group_value)] = group_summary
                continue

            group_filter: Optional[Dict[str, Any]] = None
            X_group_filtered: Optional[pd.DataFrame] = None

            try:
                group_filter = run_configured_feature_filter(
                    X=X_group_train,
                    y=y2_group_train,
                    output_base_dir=group_dir,
                    config=self.config,
                    stage_name=f"level2_resistance_profile__{safe_group_name}",
                )
                group_filter = run_feature_panel_check_after_filter(
                    filter_result=group_filter,
                    y=y2_group_train,
                    output_base_dir=group_dir,
                    config=self.config,
                    stage_name=f"level2_resistance_profile__{safe_group_name}__model_matrix",
                )
                X_group_filtered = group_filter["filtered_matrix"]
                group_manifest = write_selected_feature_manifest(
                    features=list(X_group_filtered.columns),
                    source_manifest=feature_manifest_df,
                    output_path=group_dir / "selected_feature_manifest.tsv",
                )
                group_model = train_model_safely(
                    X=X_group_filtered,
                    y=y2_group_train.loc[X_group_filtered.index],
                    output_dir=group_dir / "model",
                    config=self.config,
                    algorithm=algorithm,
                    model_name="level2_resistance_model",
                )
            except Exception as exc:
                failure_reason = classify_ml_failure(exc)
                matrix_for_diag = X_group_filtered if isinstance(X_group_filtered, pd.DataFrame) else X_group_train
                labels_for_diag = y2_group_train.loc[matrix_for_diag.index] if isinstance(matrix_for_diag, pd.DataFrame) else y2_group_train
                failure_diag = label_distribution_diagnostics(labels_for_diag, requested_cv_splits=5)
                failure_diag.update(
                    {
                        "failure_reason": failure_reason,
                        "error": str(exc),
                        "n_features_at_failure": int(matrix_for_diag.shape[1]) if isinstance(matrix_for_diag, pd.DataFrame) else None,
                        "stage": "group_specific_level2_training",
                    }
                )

                logger.warning(
                    "Group %s: group-specific Level-2 model skipped because %s | "
                    "samples=%d | features=%s | classes=%d | min_class_count=%d | "
                    "feasible_cv_splits=%d | prediction_source=%s",
                    str(group_value),
                    failure_reason,
                    int(failure_diag["n_samples"]),
                    str(failure_diag.get("n_features_at_failure")),
                    int(failure_diag["n_classes"]),
                    int(failure_diag["min_class_count"]),
                    int(failure_diag["feasible_selector_cv_splits"]),
                    fallback_source,
                )
                logger.debug(
                    "Group %s: group-specific Level-2 training traceback",
                    str(group_value),
                    exc_info=True,
                )
                group_summary.update(
                    {
                        "status": "skipped",
                        "reason": failure_reason,
                        "error": str(exc),
                        "training_failure_diagnostics": failure_diag,
                        "filter": group_filter.get("summary", {}) if isinstance(group_filter, dict) else {},
                        "feature_panel_separability": group_filter.get("feature_panel_separability", {}) if isinstance(group_filter, dict) else {},
                        "level2_source_for_prediction": fallback_source,
                    }
                )
                write_json(group_summary, group_dir / "group_summary.json")
                subgroup_payload[str(group_value)] = group_summary
                continue

            group_summary.update(
                {
                    "status": "success",
                    "filter": group_filter["summary"],
                    "feature_panel_separability": group_filter.get("feature_panel_separability", {}),
                    "model": group_model,
                    "features": list(X_group_filtered.columns),
                    "model_file": get_model_file(group_model),
                    "feature_manifest": group_manifest,
                    "level2_source_for_prediction": "level1_group_specific",
                }
            )
            write_json(group_summary, group_dir / "group_summary.json")
            subgroup_payload[str(group_value)] = group_summary

        registry = {
            "protocol": "two_level_protocol",
            "level1": {
                "label_column": level1_label,
                "description": "strain / lineage / group placement",
                "filter": level1_filter["summary"],
                "feature_panel_separability": level1_filter.get("feature_panel_separability", {}),
                "model": level1_model,
                "features": list(X_level1.columns),
                "model_file": get_model_file(level1_model),
                "feature_manifest": level1_manifest,
            },
            "level2": {
                "label_column": level2_label,
                "global_label_column": global_level2_label_summary.get("global_level2_label_column"),
                "global_label_source": global_level2_label_summary.get("source"),
                "description": "drug-resistance phenotype or resistance-profile prediction",
                "global_fallback": global_level2_payload,
                "global_binary_fallback": global_binary_level2_payload,
                "by_level1_group": subgroup_payload,
            },
            "training_matrix": {
                "aligned_matrix_csv": str(out / "aligned_two_level_matrix.csv"),
                "aligned_labels_csv": str(out / "aligned_two_level_labels.csv"),
                "feature_manifest_file": str(feature_manifest_path) if feature_manifest_path else None,
            },
            "publication_summary": _build_publication_summary(
                X=X,
                y_level1=y_level1,
                y_level2=y_level2,
                level1_filter=level1_filter,
                level1_model=level1_model,
                global_level2_payload=global_level2_payload,
                global_binary_level2_payload=global_binary_level2_payload,
                subgroup_payload=subgroup_payload,
            ),
            "config": asdict(self.config) if is_dataclass(self.config) else vars(self.config),
        }

        pipeline_progress.complete_stage("per-group Level-2 training")

        registry_path = out / "two_level_model_registry.json"
        write_json(registry, registry_path)
        logger.info("Two-level training complete: %s", registry_path)
        pipeline_progress.complete_stage("finalize two-level registry")
        return registry

    def predict(
        self,
        genomic_path: str,
        registry_path: str,
        output_dir: str,
        ref_fasta: Optional[str] = None,
    ) -> pd.DataFrame:
        out = ensure_dir(Path(output_dir))
        with open(registry_path, "r", encoding="utf-8") as handle:
            registry = json.load(handle)

        prediction_artifact_dir = out / "prediction_matrix_artifacts"

        X_new_raw_unfiltered = self.loader.load_genomic_matrix(
            file_path=genomic_path,
            output_dir=str(prediction_artifact_dir),
            ref_fasta=ref_fasta,
        )

        X_new_raw = load_artifact_filtered_binary_matrix(
            artifact_root=prediction_artifact_dir,
            fallback_matrix=X_new_raw_unfiltered,
        )

        level1_features = list(registry["level1"].get("features", []))
        level1_model_file = registry["level1"].get("model_file")
        if not level1_features or not level1_model_file:
            raise ValueError("Registry is missing the level-1 feature list or model file.")

        X_level1 = align_prediction_matrix(X_new_raw, level1_features)
        level1_payload = load_model_payload(level1_model_file)
        level1_predictions = predict_from_model_payload(level1_payload, X_level1)

        rows = []
        for sample_id, predicted_group in zip(X_level1.index.astype(str), level1_predictions):
            group_payload = registry["level2"].get("by_level1_group", {}).get(str(predicted_group), {})
            model_file = group_payload.get("model_file")
            features = group_payload.get("features", [])
            level2_source = "level1_group_specific"

            if not model_file or not features:
                fallback = registry["level2"].get("global_fallback", {})
                if not fallback or fallback.get("status") != "success" or not fallback.get("model_file"):
                    fallback = registry["level2"].get("global_binary_fallback", {})
                    level2_source = "global_binary_fallback"
                else:
                    level2_source = "global_fallback"
                model_file = fallback.get("model_file")
                features = fallback.get("features", [])

            if model_file and features:
                X_level2 = align_prediction_matrix(X_new_raw.loc[[sample_id]], list(features))
                level2_payload = load_model_payload(model_file)
                level2_prediction = predict_from_model_payload(level2_payload, X_level2)[0]
            else:
                level2_prediction = "unavailable"
                level2_source = "unavailable"

            rows.append(
                {
                    "sample_id": sample_id,
                    "predicted_level1_identity": str(predicted_group),
                    "predicted_level2_resistance_profile": str(level2_prediction),
                    "level2_model_source": level2_source,
                    "level2_target_label_column": registry.get("level2", {}).get("global_label_column") if level2_source == "global_fallback" else registry.get("level2", {}).get("label_column"),
                }
            )

        predictions = pd.DataFrame(rows)
        predictions_path = out / "two_level_predictions.csv"
        predictions.to_csv(predictions_path, index=False)
        logger.info("Two-level prediction complete: %s", predictions_path)
        return predictions


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or apply a two-level NetworkParser model: strain identity first, resistance profile second.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train the two-level protocol.")
    train.add_argument("--genomic", required=True, help="Genomic matrix file or VCF directory.")
    train.add_argument("--meta", required=True, help="Metadata CSV/TSV.")
    train.add_argument("--level1_label", default=None, help="Metadata column for first-level strain/lineage/group placement. Required unless --hierarchy_labels is used.")
    train.add_argument("--level2_label", default=None, help="Metadata column for second-level phenotype/profile. Required unless --hierarchy_labels is used.")
    train.add_argument(
        "--hierarchy_labels",
        nargs="+",
        default=None,
        help=(
            "Ordered metadata columns for true recursive hierarchy training. "
            "When provided, this supersedes --level1_label/--level2_label."
        ),
    )
    train.add_argument("--global_level2_label", default=None, help="Optional metadata column for the standard global Level-2 fallback, e.g. AMR_binary.")
    train.add_argument("--output_dir", required=True, help="Output directory.")
    train.add_argument("--config", default=None, help="Optional JSON config override file.")
    train.add_argument("--ref_fasta", default=None, help="Optional reference FASTA for VCF parsing context.")
    train.add_argument("--algorithm", default=None, help="Optional ML algorithm override passed to MLProtocolRunner.")
    train.add_argument("--no_global_level2", action="store_true", help="Disable the global level-2 fallback model.")
    train.add_argument(
        "--min_level2_samples_per_group",
        type=int,
        default=None,
        help=(
            "Optional absolute minimum samples for group-specific Level-2 models. "
            "When unset, eligibility is adaptive and scales with the number of Level-2 labels."
        ),
    )
    train.add_argument("--level2_drop_low_support_classes", action="store_true", help="Exclude Level 2 classes below the configured sample-count threshold before Level 2 training.")
    train.add_argument("--level2_min_class_count", type=int, default=None, help="Minimum samples per Level 2 class when low-support class exclusion is enabled.")
    train.add_argument("--level2_train_binary_global_fallback", action="store_true", help="Train an additional resistant/susceptible global Level 2 fallback model across all lineages.")
    train.add_argument("--level2_binary_label_column", default=None, help="Metadata column containing resistant/susceptible labels for the binary fallback model.")
    train.add_argument("--level2_binary_label_mapping_file", default=None, help="CSV/TSV mapping file from detailed Level 2 labels to resistant/susceptible labels.")
    train.add_argument("--level2_binary_resistant_values", default=None, help="Comma-separated values interpreted as resistant for the binary fallback.")
    train.add_argument("--level2_binary_susceptible_values", default=None, help="Comma-separated values interpreted as susceptible for the binary fallback.")
    train.add_argument("--n_jobs", type=int, default=None, help="Runtime worker override.")

    predict = sub.add_parser("predict", help="Apply a trained two-level protocol to new strain/sample input.")
    predict.add_argument("--genomic", required=True, help="New genomic matrix file or VCF directory.")
    predict.add_argument("--registry", required=True, help="Path to two_level_model_registry.json from training.")
    predict.add_argument("--output_dir", required=True, help="Prediction output directory.")
    predict.add_argument("--config", default=None, help="Optional JSON config override file.")
    predict.add_argument("--ref_fasta", default=None, help="Optional reference FASTA for VCF parsing context.")
    predict.add_argument("--n_jobs", type=int, default=None, help="Runtime worker override.")

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
    if getattr(args, "global_level2_label", None) is not None:
        config.global_level2_label_column = args.global_level2_label
    if bool(getattr(args, "level2_drop_low_support_classes", False)):
        config.level2_drop_low_support_classes = True
    if getattr(args, "level2_min_class_count", None) is not None:
        config.level2_min_class_count = int(args.level2_min_class_count)
    if bool(getattr(args, "level2_train_binary_global_fallback", False)):
        config.level2_train_binary_global_fallback = True
    if getattr(args, "level2_binary_label_column", None) is not None:
        config.level2_binary_label_column = args.level2_binary_label_column
    if getattr(args, "level2_binary_label_mapping_file", None) is not None:
        config.level2_binary_label_mapping_file = args.level2_binary_label_mapping_file
    if getattr(args, "level2_binary_resistant_values", None) is not None:
        config.level2_binary_resistant_values = args.level2_binary_resistant_values
    if getattr(args, "level2_binary_susceptible_values", None) is not None:
        config.level2_binary_susceptible_values = args.level2_binary_susceptible_values
    config.__post_init__()

    protocol = TwoLevelProtocol(config=config)

    if args.command == "train":
        hierarchy_labels = getattr(args, "hierarchy_labels", None)
        if hierarchy_labels:
            protocol.train_hierarchy(
                genomic_path=args.genomic,
                meta_path=args.meta,
                hierarchy_labels=list(hierarchy_labels),
                output_dir=args.output_dir,
                ref_fasta=args.ref_fasta,
                algorithm=args.algorithm,
                min_samples_per_node=args.min_level2_samples_per_group,
            )
            return 0

        if not args.level1_label or not args.level2_label:
            raise ValueError(
                "train requires either --hierarchy_labels with at least two columns "
                "or both --level1_label and --level2_label."
            )

        protocol.train(
            genomic_path=args.genomic,
            meta_path=args.meta,
            level1_label=args.level1_label,
            level2_label=args.level2_label,
            output_dir=args.output_dir,
            global_level2_label=getattr(args, "global_level2_label", None),
            ref_fasta=args.ref_fasta,
            algorithm=args.algorithm,
            train_global_level2=not bool(args.no_global_level2),
            min_level2_samples_per_group=args.min_level2_samples_per_group,
        )
        return 0

    if args.command == "predict":
        protocol.predict(
            genomic_path=args.genomic,
            registry_path=args.registry,
            output_dir=args.output_dir,
            ref_fasta=args.ref_fasta,
        )
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())

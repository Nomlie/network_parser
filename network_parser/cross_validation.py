#!/usr/bin/env python3
# network_parser/cross_validation.py
"""
Leakage-aware repeated cross-validation for NetworkParser.

Purpose
-------
Run repeated stratified cross-validation for a single supervised metadata target
while keeping the most important statistical boundary intact:

    split first -> fit feature filtering/panel selection on the training fold only
                -> train model on the training fold only
                -> evaluate the held-out fold

This module is deliberately model-agnostic with respect to the active dataset.
It is intended as a validation wrapper around the existing NetworkParser
feature-filtering and ML protocol, not as a new discovery branch.

Outputs
-------
    cross_validation_summary.json
    cv_fold_metrics.tsv
    cv_predictions.tsv
    cv_feature_stability.tsv
    cv_by_class_metrics.tsv
    aggregate_performance/*

Notes
-----
VCF parsing/QC and initial matrix construction may still be performed once at
input load time for practical runtime reasons. Cohort-level structural filters
are relaxed during this load where possible; supervised statistical feature
filtering and panel selection are always fit inside each training fold.
"""

from __future__ import annotations

import copy
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

try:
    from network_parser.config import NetworkParserConfig
    from network_parser.data_loader import DataLoader
    from network_parser.ml_protocol import MLProtocolRunner
    from network_parser.model_evaluation import evaluate_predictions, score_dicts_to_frame
    from network_parser.network_parser import normalize_labels
    from network_parser.two_level_protocol import (
        run_configured_feature_filter,
        run_feature_panel_check_after_filter,
    )
    from network_parser.utils import normalize_sample_id
except Exception:  # pragma: no cover - supports direct source-tree execution
    from config import NetworkParserConfig  # type: ignore
    from data_loader import DataLoader  # type: ignore
    from ml_protocol import MLProtocolRunner  # type: ignore
    from model_evaluation import evaluate_predictions, score_dicts_to_frame  # type: ignore
    from network_parser import normalize_labels  # type: ignore
    from two_level_protocol import (  # type: ignore
        run_configured_feature_filter,
        run_feature_panel_check_after_filter,
    )
    from utils import normalize_sample_id  # type: ignore

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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
    return str(obj)


def _write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
        handle.write("\n")


# -----------------------------------------------------------------------------
# Input preparation
# -----------------------------------------------------------------------------

def _cv_loader_config(config: NetworkParserConfig) -> NetworkParserConfig:
    """Return a copy with cohort-level structural filters relaxed for CV loading."""
    local = copy.copy(config)
    # Keep per-record VCF QC intact, but avoid filtering markers using the full
    # labelled cohort before folds are created.
    local.min_sample_presence = 1
    local.remove_invariant = False
    local.min_minor_count = 0
    local.matrices_min_count = 0
    return local


def _drop_invariant_from_training(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Remove columns that are invariant in the training fold only."""
    keep: List[str] = []
    for col in X_train.columns.astype(str):
        series = X_train[col]
        clean = series.where(~series.isna(), "__MISSING__").astype(str).str.strip()
        if clean.nunique(dropna=False) > 1:
            keep.append(str(col))

    if not keep:
        raise ValueError("No non-invariant genomic features remain in this training fold.")

    return X_train.loc[:, keep].copy(), X_test.reindex(columns=keep, fill_value=0).copy(), keep


def load_cv_matrix_and_labels(
    *,
    genomic_path: str,
    meta_path: str,
    label_column: str,
    output_dir: Path,
    config: NetworkParserConfig,
    ref_fasta: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Load genomic input and one metadata target for repeated CV."""
    loader_config = _cv_loader_config(config)
    loader = DataLoader(config=loader_config, n_jobs=getattr(config, "n_jobs", -1))

    matrix_dir = _ensure_dir(output_dir / "cv_input_matrix_artifacts")
    X = loader.load_genomic_matrix(
        file_path=genomic_path,
        output_dir=str(matrix_dir),
        ref_fasta=ref_fasta,
    )
    if not isinstance(X, pd.DataFrame):
        raise TypeError("DataLoader did not return a pandas DataFrame for CV input.")

    meta = loader.load_metadata(meta_path, output_dir=str(output_dir / "metadata"))
    if label_column not in meta.columns:
        raise ValueError(f"label_column '{label_column}' not found in metadata columns")

    X = X.copy()
    X.index = X.index.astype(str).map(normalize_sample_id)
    X = X[~X.index.duplicated(keep="first")]
    X.columns = X.columns.astype(str)

    labels = normalize_labels(meta[label_column], drop_missing=True, lowercase=False)
    labels.index = labels.index.astype(str).map(normalize_sample_id)
    labels = labels[~labels.index.duplicated(keep="first")]

    common = X.index.intersection(labels.index)
    if common.empty:
        raise ValueError("No overlapping sample IDs between genomic matrix and metadata labels for CV.")

    X = X.loc[common].copy()
    y = labels.loc[common].astype(str).copy()

    summary = {
        "status": "loaded",
        "genomic_path": str(genomic_path),
        "meta_path": str(meta_path),
        "target_label_column": str(label_column),
        "samples_after_alignment": int(X.shape[0]),
        "features_after_relaxed_input_load": int(X.shape[1]),
        "n_classes": int(y.nunique(dropna=True)),
        "class_count_values_sorted": sorted([int(v) for v in y.value_counts().tolist()]),
        "input_load_note": (
            "Per-record input QC was preserved. Cohort-level structural filters were relaxed "
            "where possible so supervised feature filtering can be fit inside each CV training fold."
        ),
    }

    X.to_csv(output_dir / "cv_aligned_input_matrix.csv")
    pd.DataFrame({"sample_id": y.index.astype(str), "label": y.values}).to_csv(
        output_dir / "cv_aligned_labels.tsv",
        sep="\t",
        index=False,
    )
    _write_json(summary, output_dir / "cv_input_summary.json")
    return X, y, summary


# -----------------------------------------------------------------------------
# Prediction helpers
# -----------------------------------------------------------------------------

def _predict_one_model_row(model: Any, row: pd.Series, features: Sequence[str]) -> Tuple[str, Optional[float], Dict[str, float]]:
    marker_dict: Dict[str, str] = {}
    for feature in features:
        value = row.get(feature, 0)
        if pd.isna(value):
            marker_dict[str(feature)] = ""
        else:
            marker_dict[str(feature)] = str(value).strip()

    if not hasattr(model, "identify"):
        raise TypeError("Trained model does not expose identify(); cannot run NetworkParser-style fold prediction.")

    result = model.identify(marker_dict)
    predictions = result.get("predictions", []) if isinstance(result, dict) else []

    class_support: Dict[str, float] = {}
    for item in predictions:
        try:
            label = str(item[0])
            support = float(item[1])
        except Exception:
            continue
        class_support[label] = support

    if not class_support:
        return "unavailable", None, {}

    top_label = max(class_support.items(), key=lambda kv: kv[1])[0]
    top_support = float(class_support[top_label])
    return top_label, top_support, class_support


def _predict_fold(
    *,
    model: Any,
    X_test: pd.DataFrame,
    features: Sequence[str],
    y_test: pd.Series,
    repeat: int,
    fold: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    score_dicts: List[Dict[str, float]] = []
    sample_ids: List[str] = []

    X_local = X_test.reindex(columns=list(features), fill_value=0).copy()

    for sample_id, row in X_local.iterrows():
        pred_label, support, class_support = _predict_one_model_row(model, row, features)
        sid = normalize_sample_id(sample_id)
        sample_ids.append(sid)
        score_dicts.append(class_support)
        rows.append(
            {
                "prediction_id": f"r{repeat}_f{fold}_{sid}",
                "repeat": int(repeat),
                "fold": int(fold),
                "sample_id": sid,
                "true_label": str(y_test.loc[sample_id]),
                "predicted_label": str(pred_label),
                "top_support": float(support) if support is not None else np.nan,
                "class_support_json": json.dumps(class_support, sort_keys=True),
            }
        )

    pred_df = pd.DataFrame(rows)
    score_df = score_dicts_to_frame(sample_ids=sample_ids, score_dicts=score_dicts)
    # Use prediction_id as index for aggregate evaluation so repeated samples
    # across repeated CV are not deduplicated by sample id.
    if not pred_df.empty:
        score_df.index = pred_df["prediction_id"].astype(str).values
    return pred_df, score_df


# -----------------------------------------------------------------------------
# Fold training/evaluation
# -----------------------------------------------------------------------------

def _resolve_fold_algorithm(
    *,
    runner: MLProtocolRunner,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    requested_algorithm: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    requested = requested_algorithm
    if requested is None or str(requested).lower() == "auto":
        run_selector = bool(getattr(runner.config, "run_model_selector", True)) if runner.config is not None else True
        if run_selector:
            selector = runner.select_model(X_train, y_train)
            selector = runner._normalize_selector_output(selector)
            selected = runner.resolve_algorithm(
                selector_recommendation=selector.get("recommendation", "RF"),
                requested_algorithm="auto",
            )
            return selected, selector
        selected = runner.resolve_algorithm("RF", requested_algorithm="RF")
        return selected, {"selector_enabled": False, "recommendation": selected}

    selected = runner.resolve_algorithm("RF", requested_algorithm=str(requested))
    return selected, {"selector_enabled": False, "requested_algorithm": str(requested), "recommendation": selected}


def _run_one_fold(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    repeat: int,
    fold: int,
    output_dir: Path,
    config: NetworkParserConfig,
    algorithm: Optional[str],
) -> Dict[str, Any]:
    fold_dir = _ensure_dir(output_dir / f"repeat_{repeat:02d}" / f"fold_{fold:02d}")

    X_train = X.loc[list(train_ids)].copy()
    X_test = X.loc[list(test_ids)].copy()
    y_train = y.loc[list(train_ids)].astype(str).copy()
    y_test = y.loc[list(test_ids)].astype(str).copy()

    X_train, X_test, train_variant_features = _drop_invariant_from_training(X_train, X_test)

    filter_result = run_configured_feature_filter(
        X=X_train,
        y=y_train,
        output_base_dir=fold_dir / "feature_filtering",
        config=config,
        stage_name=f"cv_repeat_{repeat}_fold_{fold}",
    )

    panel_status = "not_requested"
    if bool(getattr(config, "run_feature_panel_separability_check", True)):
        try:
            filter_result = run_feature_panel_check_after_filter(
                filter_result=filter_result,
                y=y_train,
                output_base_dir=fold_dir / "feature_panel",
                config=config,
                stage_name=f"cv_repeat_{repeat}_fold_{fold}",
            )
            panel_status = "success"
        except Exception as exc:
            # Do not convert a diagnostic panel failure into a failed CV fold.
            # Central statistical filtering has already happened; retaining that
            # matrix preserves the main statistically defensible boundary.
            panel_status = f"skipped_after_failure: {exc}"
            logger.warning(
                "CV panel check failed; using centrally filtered matrix for this fold | repeat=%d | fold=%d | error=%s",
                repeat,
                fold,
                exc,
            )

    X_train_selected = filter_result["filtered_matrix"].copy()
    selected_features = [str(f) for f in filter_result.get("retained_features", list(X_train_selected.columns))]
    selected_features = [f for f in selected_features if f in X_train_selected.columns]
    if not selected_features:
        raise ValueError("No selected features available after fold filtering.")
    X_train_selected = X_train_selected.loc[:, selected_features].copy()
    X_test_selected = X_test.reindex(columns=selected_features, fill_value=0).copy()

    runner = MLProtocolRunner(config=config)
    selected_algorithm, selector_payload = _resolve_fold_algorithm(
        runner=runner,
        X_train=X_train_selected,
        y_train=y_train,
        requested_algorithm=algorithm,
    )
    model = runner.train_model(
        genomic_df=X_train_selected,
        labels=y_train,
        algorithm=selected_algorithm,
    )

    pred_df, score_df = _predict_fold(
        model=model,
        X_test=X_test_selected,
        features=selected_features,
        y_test=y_test,
        repeat=repeat,
        fold=fold,
    )
    pred_df.to_csv(fold_dir / "fold_predictions.tsv", sep="\t", index=False)
    score_df.to_csv(fold_dir / "fold_class_support.tsv", sep="\t")

    truth = pd.Series(
        pred_df["true_label"].astype(str).values,
        index=pred_df["prediction_id"].astype(str).values,
        name="true_label",
    )
    pred = pd.Series(
        pred_df["predicted_label"].astype(str).values,
        index=pred_df["prediction_id"].astype(str).values,
        name="predicted_label",
    )
    fold_eval = evaluate_predictions(
        y_true=truth,
        y_pred=pred,
        class_support_scores=score_df,
        output_dir=fold_dir / "evaluation",
        level_name=f"cv_repeat_{repeat}_fold_{fold}",
    )

    by_class_path = Path(fold_eval.get("artifacts", {}).get("by_class_tsv", ""))
    if by_class_path.exists():
        by_class = pd.read_csv(by_class_path, sep="\t")
        by_class.insert(0, "fold", int(fold))
        by_class.insert(0, "repeat", int(repeat))
        by_class.to_csv(fold_dir / "fold_by_class_metrics.tsv", sep="\t", index=False)

    fold_summary = {
        "status": "success",
        "repeat": int(repeat),
        "fold": int(fold),
        "n_train_samples": int(X_train.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
        "n_features_after_train_invariant_filter": int(len(train_variant_features)),
        "n_selected_features": int(len(selected_features)),
        "selected_algorithm": str(selected_algorithm),
        "panel_status": panel_status,
        "selector": selector_payload,
        "selected_features": selected_features,
        "evaluation": fold_eval,
        "artifacts": {
            "fold_dir": str(fold_dir),
            "fold_predictions": str(fold_dir / "fold_predictions.tsv"),
            "fold_class_support": str(fold_dir / "fold_class_support.tsv"),
            "fold_evaluation": str(fold_dir / "evaluation" / "model_performance_summary.json"),
        },
    }
    _write_json(fold_summary, fold_dir / "fold_summary.json")
    return fold_summary


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------

def _flatten_fold_metrics(fold_summaries: List[Dict[str, Any]]) -> pd.DataFrame:
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision_ppv",
        "macro_true_positive_rate",
        "macro_sensitivity_recall",
        "macro_true_negative_rate",
        "macro_specificity",
        "macro_npv",
        "macro_f1",
        "weighted_f1",
        "matthews_corrcoef",
    ]
    rows: List[Dict[str, Any]] = []
    for fold in fold_summaries:
        evaluation = fold.get("evaluation", {}) if isinstance(fold, dict) else {}
        row: Dict[str, Any] = {
            "repeat": fold.get("repeat"),
            "fold": fold.get("fold"),
            "status": fold.get("status"),
            "n_train_samples": fold.get("n_train_samples"),
            "n_test_samples": fold.get("n_test_samples"),
            "n_selected_features": fold.get("n_selected_features"),
            "selected_algorithm": fold.get("selected_algorithm"),
            "panel_status": fold.get("panel_status"),
        }
        for name in metric_names:
            row[name] = evaluation.get(name)
        roc_pr = evaluation.get("roc_pr", {}) if isinstance(evaluation.get("roc_pr", {}), dict) else {}
        row["macro_roc_auc_ovr"] = roc_pr.get("macro_roc_auc_ovr")
        row["weighted_roc_auc_ovr"] = roc_pr.get("weighted_roc_auc_ovr")
        row["macro_pr_auc_average_precision"] = roc_pr.get("macro_pr_auc_average_precision")
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_stability(fold_summaries: List[Dict[str, Any]]) -> pd.DataFrame:
    total_folds = max(1, len(fold_summaries))
    counts: Counter[str] = Counter()
    ranks: Dict[str, List[int]] = defaultdict(list)

    for fold in fold_summaries:
        features = fold.get("selected_features", [])
        if not isinstance(features, list):
            continue
        for rank, feature in enumerate(features, start=1):
            f = str(feature)
            counts[f] += 1
            ranks[f].append(int(rank))

    rows: List[Dict[str, Any]] = []
    for feature, count in counts.items():
        rank_values = ranks.get(feature, [])
        rows.append(
            {
                "feature": feature,
                "selection_count": int(count),
                "selection_frequency": float(count / total_folds),
                "mean_selected_rank": float(np.mean(rank_values)) if rank_values else np.nan,
                "median_selected_rank": float(np.median(rank_values)) if rank_values else np.nan,
                "n_successful_folds": int(total_folds),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            by=["selection_frequency", "selection_count", "mean_selected_rank", "feature"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
    return df


def _collect_by_class_metrics(fold_summaries: List[Dict[str, Any]]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for fold in fold_summaries:
        fold_dir = Path(fold.get("artifacts", {}).get("fold_dir", ""))
        path = fold_dir / "fold_by_class_metrics.tsv"
        if path.exists():
            frames.append(pd.read_csv(path, sep="\t"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _aggregate_prediction_performance(predictions: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    if predictions.empty:
        return {"status": "skipped", "message": "No fold predictions available."}

    truth = pd.Series(
        predictions["true_label"].astype(str).values,
        index=predictions["prediction_id"].astype(str).values,
        name="true_label",
    )
    pred = pd.Series(
        predictions["predicted_label"].astype(str).values,
        index=predictions["prediction_id"].astype(str).values,
        name="predicted_label",
    )

    score_dicts: List[Dict[str, Any]] = []
    for raw in predictions["class_support_json"].astype(str).tolist():
        try:
            parsed = json.loads(raw)
            score_dicts.append(parsed if isinstance(parsed, dict) else {})
        except Exception:
            score_dicts.append({})
    scores = score_dicts_to_frame(predictions["prediction_id"].astype(str).tolist(), score_dicts)

    return evaluate_predictions(
        y_true=truth,
        y_pred=pred,
        class_support_scores=scores,
        output_dir=output_dir / "aggregate_performance",
        level_name="repeated_cv_aggregate",
    )


# -----------------------------------------------------------------------------
# Public runner
# -----------------------------------------------------------------------------

def run_repeated_cv(
    *,
    genomic_path: str,
    meta_path: str,
    label_column: str,
    output_dir: str,
    config: NetworkParserConfig,
    ref_fasta: Optional[str] = None,
    n_repeats: int = 3,
    n_splits: int = 5,
    algorithm: Optional[str] = None,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Run leakage-aware repeated stratified CV for one supervised target."""
    out = _ensure_dir(Path(output_dir))
    random_state = int(getattr(config, "random_state", 42) if random_state is None else random_state)
    n_repeats = max(1, int(n_repeats))
    requested_splits = max(2, int(n_splits))

    X, y, input_summary = load_cv_matrix_and_labels(
        genomic_path=genomic_path,
        meta_path=meta_path,
        label_column=label_column,
        output_dir=out,
        config=config,
        ref_fasta=ref_fasta,
    )

    class_counts = y.value_counts(dropna=True)
    feasible_splits = min(requested_splits, int(class_counts.min())) if not class_counts.empty else 0
    if y.nunique(dropna=True) < 2 or feasible_splits < 2:
        raise ValueError(
            "Repeated CV requires at least two classes and at least two samples in the smallest class."
        )

    logger.info(
        "Repeated CV started | target=configured | repeats=%d | requested_folds=%d | actual_folds=%d",
        n_repeats,
        requested_splits,
        feasible_splits,
    )

    fold_summaries: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    sample_ids = np.asarray(X.index.astype(str))
    y_values = y.astype(str).to_numpy()

    for repeat in range(1, n_repeats + 1):
        cv = StratifiedKFold(
            n_splits=feasible_splits,
            shuffle=True,
            random_state=random_state + repeat - 1,
        )
        for fold, (train_idx, test_idx) in enumerate(cv.split(sample_ids, y_values), start=1):
            train_ids = sample_ids[train_idx].tolist()
            test_ids = sample_ids[test_idx].tolist()
            try:
                fold_summary = _run_one_fold(
                    X=X,
                    y=y,
                    train_ids=train_ids,
                    test_ids=test_ids,
                    repeat=repeat,
                    fold=fold,
                    output_dir=out,
                    config=config,
                    algorithm=algorithm,
                )
                fold_summaries.append(fold_summary)
                logger.info(
                    "CV fold complete | repeat=%d | fold=%d | selected_features=%d | algorithm=%s",
                    repeat,
                    fold,
                    int(fold_summary.get("n_selected_features", 0)),
                    fold_summary.get("selected_algorithm", "NA"),
                )
            except Exception as exc:
                failure = {
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "status": "failed",
                    "error": str(exc),
                }
                failures.append(failure)
                fail_dir = _ensure_dir(out / f"repeat_{repeat:02d}" / f"fold_{fold:02d}")
                _write_json(failure, fail_dir / "fold_failure.json")
                logger.warning("CV fold failed | repeat=%d | fold=%d | error=%s", repeat, fold, exc)

    fold_metrics = _flatten_fold_metrics(fold_summaries)
    fold_metrics.to_csv(out / "cv_fold_metrics.tsv", sep="\t", index=False)

    prediction_frames: List[pd.DataFrame] = []
    for fold in fold_summaries:
        path = Path(fold.get("artifacts", {}).get("fold_predictions", ""))
        if path.exists():
            prediction_frames.append(pd.read_csv(path, sep="\t"))
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    predictions.to_csv(out / "cv_predictions.tsv", sep="\t", index=False)

    stability = _feature_stability(fold_summaries)
    stability.to_csv(out / "cv_feature_stability.tsv", sep="\t", index=False)

    by_class = _collect_by_class_metrics(fold_summaries)
    by_class.to_csv(out / "cv_by_class_metrics.tsv", sep="\t", index=False)

    aggregate_eval = _aggregate_prediction_performance(predictions, out)

    metric_summary: Dict[str, Any] = {}
    numeric_cols = [
        "accuracy",
        "balanced_accuracy",
        "macro_true_positive_rate",
        "macro_sensitivity_recall",
        "macro_true_negative_rate",
        "macro_specificity",
        "macro_precision_ppv",
        "macro_npv",
        "macro_f1",
        "weighted_f1",
        "matthews_corrcoef",
        "macro_roc_auc_ovr",
        "macro_pr_auc_average_precision",
    ]
    for col in numeric_cols:
        if col in fold_metrics.columns:
            values = pd.to_numeric(fold_metrics[col], errors="coerce").dropna()
            metric_summary[col] = {
                "mean": float(values.mean()) if not values.empty else None,
                "std": float(values.std(ddof=1)) if values.shape[0] > 1 else None,
                "min": float(values.min()) if not values.empty else None,
                "max": float(values.max()) if not values.empty else None,
            }

    summary = {
        "status": "success" if fold_summaries else "failed",
        "target_label_column": str(label_column),
        "requested_repeats": int(n_repeats),
        "requested_folds": int(requested_splits),
        "actual_folds_per_repeat": int(feasible_splits),
        "successful_folds": int(len(fold_summaries)),
        "failed_folds": int(len(failures)),
        "input_summary": input_summary,
        "metric_summary_across_folds": metric_summary,
        "aggregate_prediction_performance": aggregate_eval,
        "failures": failures,
        "artifacts": {
            "cv_input_summary": str(out / "cv_input_summary.json"),
            "cv_fold_metrics": str(out / "cv_fold_metrics.tsv"),
            "cv_predictions": str(out / "cv_predictions.tsv"),
            "cv_feature_stability": str(out / "cv_feature_stability.tsv"),
            "cv_by_class_metrics": str(out / "cv_by_class_metrics.tsv"),
            "aggregate_performance_dir": str(out / "aggregate_performance"),
        },
        "method_note": (
            "Each CV fold split samples before supervised statistical feature filtering, "
            "feature-panel selection, and model training. This prevents the held-out fold "
            "from driving marker selection."
        ),
    }
    _write_json(summary, out / "cross_validation_summary.json")

    logger.info(
        "Repeated CV complete | successful_folds=%d | failed_folds=%d | out=%s",
        int(len(fold_summaries)),
        int(len(failures)),
        str(out),
    )
    return summary

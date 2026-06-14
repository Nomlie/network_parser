#!/usr/bin/env python3
# network_parser/model_evaluation.py
"""
Model-performance evaluation utilities for NetworkParser.

Purpose
-------
Evaluate saved-model predictions against labelled holdout metadata without
rerunning statistical filtering, feature selection, model training, tree
construction, bootstrapping, or confidence scoring.

This module is intentionally generic with respect to the active dataset.  It
operates on true labels, predicted labels, and optional class-support scores.
It writes diagnostic metrics that are useful for AMR/strain-classification
validation:

    - overall accuracy / balanced accuracy / macro and weighted F1
    - per-class sensitivity, specificity, PPV, NPV, F1
    - confusion matrix
    - one-vs-rest ROC-AUC and PR-AUC where class-support scores are available
    - ROC / PR curve point tables for plotting downstream

Design rule
-----------
These metrics are evaluation-only.  They do not affect marker discovery or
model fitting.  For robust inference, use this on a held-out set, or later wrap
it inside cross-validation where filtering is rerun inside each training fold.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    from network_parser.utils import normalize_sample_id
except Exception:  # pragma: no cover - supports direct source-tree execution
    try:
        from utils import normalize_sample_id  # type: ignore
    except Exception:  # pragma: no cover
        def normalize_sample_id(value: Any, strip_library_suffix: bool = True) -> str:  # type: ignore
            return str(value).strip()

logger = logging.getLogger(__name__)

MISSING_LABEL_TOKENS = {"", "-", "NA", "N/A", "None", "none", "nan", "NaN", "null", "NULL"}


# -----------------------------------------------------------------------------
# JSON / IO helpers
# -----------------------------------------------------------------------------

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


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
        handle.write("\n")


# -----------------------------------------------------------------------------
# Metadata / label helpers
# -----------------------------------------------------------------------------

def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".tsv") or suffixes.endswith(".txt"):
        return pd.read_csv(path, sep="\t")
    if suffixes.endswith(".csv"):
        return pd.read_csv(path)

    # Robust fallback for whitespace/comma/tab-delimited metadata.
    return pd.read_csv(path, sep=None, engine="python")


def _normalise_label_series(values: pd.Series) -> pd.Series:
    clean = values.astype(str).str.strip()
    clean = clean.replace({token: pd.NA for token in MISSING_LABEL_TOKENS})
    clean = clean.str.replace("-", "_", regex=False)
    return clean


def _resolve_sample_id_column(df: pd.DataFrame, sample_id_column: Optional[str]) -> Optional[str]:
    if sample_id_column:
        if sample_id_column not in df.columns:
            raise ValueError(f"sample_id_column '{sample_id_column}' not found in metadata columns")
        return sample_id_column

    for candidate in ("sample_id", "Sample_ID", "sample", "Sample", "SampleID", "sampleID", "id", "ID"):
        if candidate in df.columns:
            return candidate
    return None


def load_labels_from_metadata(
    meta_path: str | Path,
    label_column: str,
    sample_id_column: Optional[str] = None,
) -> pd.Series:
    """Load one labelled metadata column and index it by normalized sample ID."""
    df = _read_table(meta_path)
    if label_column not in df.columns:
        raise ValueError(f"label_column '{label_column}' not found in metadata columns")

    sid_col = _resolve_sample_id_column(df, sample_id_column)
    if sid_col is not None:
        index = df[sid_col].astype(str).map(normalize_sample_id)
    else:
        index = pd.Index(df.index.astype(str).map(normalize_sample_id))
        logger.warning(
            "No sample-id column was detected in metadata. Using metadata row index for evaluation alignment."
        )

    labels = _normalise_label_series(df[label_column])
    labels.index = index
    labels = labels.dropna()
    labels = labels[labels.index.astype(str).str.len() > 0]
    labels = labels[~labels.index.duplicated(keep="first")]
    return labels


def score_dicts_to_frame(
    sample_ids: Sequence[Any],
    score_dicts: Sequence[Optional[Dict[str, Any]]],
) -> pd.DataFrame:
    """Convert per-sample class-support dictionaries into a numeric score matrix."""
    rows: List[Dict[str, float]] = []
    index: List[str] = []
    for sample_id, scores in zip(sample_ids, score_dicts):
        index.append(normalize_sample_id(str(sample_id)))
        row: Dict[str, float] = {}
        if isinstance(scores, dict):
            for key, value in scores.items():
                try:
                    row[str(key)] = float(value)
                except Exception:
                    continue
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, index=index).fillna(0.0)


def _as_label_series(values: Any, name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        out = values.copy()
    elif isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError(f"{name} DataFrame must have exactly one column")
        out = values.iloc[:, 0].copy()
    else:
        out = pd.Series(values)
    out = _normalise_label_series(out)
    return out.rename(name)


def _align_truth_and_predictions(
    y_true: Any,
    y_pred: Any,
    score_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    truth = _as_label_series(y_true, "true_label")
    pred = _as_label_series(y_pred, "predicted_label")

    truth.index = truth.index.astype(str).map(normalize_sample_id)
    pred.index = pred.index.astype(str).map(normalize_sample_id)

    truth = truth.dropna()
    pred = pred.dropna()
    truth = truth[~truth.index.duplicated(keep="first")]
    pred = pred[~pred.index.duplicated(keep="first")]

    common = truth.index.intersection(pred.index)
    df = pd.DataFrame(
        {
            "sample_id": common.astype(str),
            "true_label": truth.loc[common].astype(str).values,
            "predicted_label": pred.loc[common].astype(str).values,
        },
        index=common,
    )

    if score_df is None or score_df.empty:
        return df, None

    scores = score_df.copy()
    scores.index = scores.index.astype(str).map(normalize_sample_id)
    scores = scores[~scores.index.duplicated(keep="first")]
    scores = scores.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scores = scores.reindex(df.index).fillna(0.0)
    return df, scores


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------

def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = int(cm.sum())

    rows: List[Dict[str, Any]] = []
    for i, label in enumerate(labels):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(total - tp - fn - fp)

        sensitivity = _safe_div(tp, tp + fn)
        specificity = _safe_div(tn, tn + fp)
        ppv = _safe_div(tp, tp + fp)
        npv = _safe_div(tn, tn + fn)
        f1 = _safe_div(2.0 * ppv * sensitivity, ppv + sensitivity)

        rows.append(
            {
                "class_label": str(label),
                "support": int(tp + fn),
                "predicted_count": int(tp + fp),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "sensitivity_recall": sensitivity,
                "specificity": specificity,
                "ppv_precision": ppv,
                "npv": npv,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)


def _normalise_score_rows(scores: pd.DataFrame) -> pd.DataFrame:
    scores = scores.copy().astype(float)
    row_sums = scores.sum(axis=1)
    nonzero = row_sums > 0
    scores.loc[nonzero, :] = scores.loc[nonzero, :].div(row_sums.loc[nonzero], axis=0)
    return scores.fillna(0.0)


def _roc_pr_outputs(
    eval_df: pd.DataFrame,
    scores: Optional[pd.DataFrame],
    labels: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if scores is None or scores.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "status": "skipped",
            "message": "No class-support score matrix was available for ROC/PR evaluation.",
        }

    usable_labels = [str(label) for label in labels if str(label) in scores.columns]
    if not usable_labels:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "status": "skipped",
            "message": "Class-support scores did not contain columns matching evaluated labels.",
        }

    scores_local = scores.reindex(columns=usable_labels).fillna(0.0)
    scores_local = _normalise_score_rows(scores_local)
    y_true = eval_df["true_label"].astype(str).to_numpy()

    auc_rows: List[Dict[str, Any]] = []
    roc_rows: List[Dict[str, Any]] = []
    pr_rows: List[Dict[str, Any]] = []

    for label in usable_labels:
        binary_true = (y_true == label).astype(int)
        if len(np.unique(binary_true)) < 2:
            auc_rows.append(
                {
                    "class_label": label,
                    "roc_auc_ovr": np.nan,
                    "pr_auc_average_precision": np.nan,
                    "status": "skipped_single_class_truth",
                }
            )
            continue

        score = scores_local[label].to_numpy(dtype=float)
        try:
            roc_auc = float(roc_auc_score(binary_true, score))
            fpr, tpr, roc_thresholds = roc_curve(binary_true, score)
            for fpr_i, tpr_i, thr_i in zip(fpr, tpr, roc_thresholds):
                roc_rows.append(
                    {
                        "class_label": label,
                        "false_positive_rate": float(fpr_i),
                        "true_positive_rate": float(tpr_i),
                        "threshold": float(thr_i),
                    }
                )
        except Exception as exc:
            roc_auc = np.nan
            logger.warning("ROC calculation failed for one class: %s", exc)

        try:
            pr_auc = float(average_precision_score(binary_true, score))
            precision, recall, pr_thresholds = precision_recall_curve(binary_true, score)
            # precision/recall has one extra point beyond thresholds.
            padded_thresholds = list(pr_thresholds) + [np.nan]
            for precision_i, recall_i, thr_i in zip(precision, recall, padded_thresholds):
                pr_rows.append(
                    {
                        "class_label": label,
                        "precision_ppv": float(precision_i),
                        "recall_sensitivity": float(recall_i),
                        "threshold": float(thr_i) if np.isfinite(thr_i) else np.nan,
                    }
                )
        except Exception as exc:
            pr_auc = np.nan
            logger.warning("PR calculation failed for one class: %s", exc)

        auc_rows.append(
            {
                "class_label": label,
                "roc_auc_ovr": roc_auc,
                "pr_auc_average_precision": pr_auc,
                "status": "ok" if np.isfinite(roc_auc) or np.isfinite(pr_auc) else "failed",
            }
        )

    macro_roc_auc = np.nan
    weighted_roc_auc = np.nan
    macro_pr_auc = np.nan

    try:
        present_true_labels = sorted(set(map(str, y_true)))
        multiclass_labels = [label for label in usable_labels if label in present_true_labels]
        if len(multiclass_labels) >= 2:
            score_all = scores_local.reindex(columns=multiclass_labels).fillna(0.0)
            score_all = _normalise_score_rows(score_all)
            macro_roc_auc = float(
                roc_auc_score(
                    y_true,
                    score_all.to_numpy(dtype=float),
                    labels=multiclass_labels,
                    multi_class="ovr",
                    average="macro",
                )
            )
            weighted_roc_auc = float(
                roc_auc_score(
                    y_true,
                    score_all.to_numpy(dtype=float),
                    labels=multiclass_labels,
                    multi_class="ovr",
                    average="weighted",
                )
            )
    except Exception:
        # Per-class one-vs-rest AUC rows remain the primary output.
        pass

    auc_df = pd.DataFrame(auc_rows)
    if not auc_df.empty and "pr_auc_average_precision" in auc_df.columns:
        macro_pr_auc = float(pd.to_numeric(auc_df["pr_auc_average_precision"], errors="coerce").mean())

    summary = {
        "status": "ok" if not auc_df.empty else "skipped",
        "scored_classes": int(len(usable_labels)),
        "macro_roc_auc_ovr": macro_roc_auc if np.isfinite(macro_roc_auc) else None,
        "weighted_roc_auc_ovr": weighted_roc_auc if np.isfinite(weighted_roc_auc) else None,
        "macro_pr_auc_average_precision": macro_pr_auc if np.isfinite(macro_pr_auc) else None,
    }
    return auc_df, pd.DataFrame(roc_rows), pd.DataFrame(pr_rows), summary


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def evaluate_predictions(
    *,
    y_true: Any,
    y_pred: Any,
    class_support_scores: Optional[pd.DataFrame] = None,
    output_dir: str | Path,
    level_name: str = "model",
) -> Dict[str, Any]:
    """Evaluate predictions and write diagnostic performance artifacts."""
    out = _ensure_dir(Path(output_dir))
    eval_df, scores = _align_truth_and_predictions(y_true, y_pred, class_support_scores)

    artifacts = {
        "sample_predictions": str(out / "evaluated_sample_predictions.tsv"),
        "summary_json": str(out / "model_performance_summary.json"),
        "by_class_tsv": str(out / "model_performance_by_class.tsv"),
        "confusion_matrix_tsv": str(out / "confusion_matrix.tsv"),
        "roc_auc_summary_tsv": str(out / "roc_auc_summary.tsv"),
        "roc_curve_points_tsv": str(out / "roc_curve_points.tsv"),
        "pr_curve_points_tsv": str(out / "pr_curve_points.tsv"),
    }

    eval_df.to_csv(artifacts["sample_predictions"], sep="\t", index=False)

    if eval_df.empty:
        summary = {
            "status": "skipped",
            "level_name": level_name,
            "n_evaluated_samples": 0,
            "message": "No overlapping labelled samples between predictions and metadata.",
            "artifacts": artifacts,
        }
        _write_json(summary, out / "model_performance_summary.json")
        return summary

    y_true_arr = eval_df["true_label"].astype(str).to_numpy()
    y_pred_arr = eval_df["predicted_label"].astype(str).to_numpy()
    labels = sorted(set(y_true_arr).union(set(y_pred_arr)))

    by_class = _per_class_metrics(y_true_arr, y_pred_arr, labels)
    by_class.to_csv(artifacts["by_class_tsv"], sep="\t", index=False)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true::{x}" for x in labels], columns=[f"pred::{x}" for x in labels])
    cm_df.to_csv(artifacts["confusion_matrix_tsv"], sep="\t")

    precision_macro = precision_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0)
    recall_macro = recall_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0)
    f1_macro = f1_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0)
    precision_weighted = precision_score(y_true_arr, y_pred_arr, labels=labels, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_true_arr, y_pred_arr, labels=labels, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true_arr, y_pred_arr, labels=labels, average="weighted", zero_division=0)

    macro_specificity = float(pd.to_numeric(by_class["specificity"], errors="coerce").mean()) if not by_class.empty else 0.0
    macro_npv = float(pd.to_numeric(by_class["npv"], errors="coerce").mean()) if not by_class.empty else 0.0

    auc_df, roc_points, pr_points, auc_summary = _roc_pr_outputs(eval_df, scores, labels)
    auc_df.to_csv(artifacts["roc_auc_summary_tsv"], sep="\t", index=False)
    roc_points.to_csv(artifacts["roc_curve_points_tsv"], sep="\t", index=False)
    pr_points.to_csv(artifacts["pr_curve_points_tsv"], sep="\t", index=False)

    try:
        mcc = float(matthews_corrcoef(y_true_arr, y_pred_arr))
    except Exception:
        mcc = 0.0

    summary = {
        "status": "success",
        "level_name": level_name,
        "n_evaluated_samples": int(eval_df.shape[0]),
        "n_classes_observed": int(len(set(y_true_arr))),
        "n_prediction_classes": int(len(set(y_pred_arr))),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "macro_precision_ppv": float(precision_macro),
        "macro_sensitivity_recall": float(recall_macro),
        "macro_specificity": macro_specificity,
        "macro_npv": macro_npv,
        "macro_f1": float(f1_macro),
        "weighted_precision_ppv": float(precision_weighted),
        "weighted_sensitivity_recall": float(recall_weighted),
        "weighted_f1": float(f1_weighted),
        "matthews_corrcoef": mcc,
        "roc_pr": auc_summary,
        "artifacts": artifacts,
    }

    _write_json(summary, out / "model_performance_summary.json")
    logger.info(
        "Model evaluation complete | level=%s | samples=%d | balanced_accuracy=%.4f | macro_sensitivity=%.4f | macro_specificity=%.4f",
        level_name,
        int(eval_df.shape[0]),
        float(summary["balanced_accuracy"]),
        float(summary["macro_sensitivity_recall"]),
        float(summary["macro_specificity"]),
    )
    return summary

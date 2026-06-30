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

import ast
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


def _balanced_accuracy_no_warning(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    """Compute balanced accuracy without sklearn warnings for extra predicted classes.

    sklearn warns when y_pred contains labels absent from y_true. For evaluation
    diagnostics this is not an exception condition; it means the model predicted
    an unsupported/out-of-scope class for the evaluated subset. Balanced accuracy
    is therefore computed as the mean one-vs-rest recall over classes observed in
    y_true only.
    """
    y_true_arr = np.asarray([str(x) for x in y_true], dtype=object)
    y_pred_arr = np.asarray([str(x) for x in y_pred], dtype=object)
    observed = sorted(set(y_true_arr.tolist()))
    if not observed:
        return 0.0
    recalls: List[float] = []
    for label in observed:
        mask = y_true_arr == label
        denom = int(mask.sum())
        recalls.append(_safe_div(int(np.sum(y_pred_arr[mask] == label)), denom))
    return float(np.mean(recalls)) if recalls else 0.0


def _safe_macro_metric(metric_func, y_true: Sequence[Any], y_pred: Sequence[Any], *, labels: Optional[List[str]] = None) -> float:
    """Return a macro metric while treating extra prediction-only labels as zero-support classes."""
    y_true_arr = np.asarray([str(x) for x in y_true], dtype=object)
    y_pred_arr = np.asarray([str(x) for x in y_pred], dtype=object)
    if labels is None:
        labels = sorted(set(y_true_arr.tolist()).union(set(y_pred_arr.tolist())))
    try:
        return float(metric_func(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0))
    except Exception:
        return 0.0


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
        false_positive_rate = _safe_div(fp, fp + tn)
        false_negative_rate = _safe_div(fn, fn + tp)
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
                "true_positive_rate": sensitivity,
                "sensitivity_recall": sensitivity,
                "true_negative_rate": specificity,
                "specificity": specificity,
                "false_positive_rate": false_positive_rate,
                "false_negative_rate": false_negative_rate,
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
# Prediction-table loaders
# -----------------------------------------------------------------------------

def _resolve_required_column(
    df: pd.DataFrame,
    requested: Optional[str],
    candidates: Sequence[str],
    role: str,
) -> str:
    """Resolve a required column by explicit name or common NetworkParser names."""
    if requested:
        if requested not in df.columns:
            raise ValueError(f"{role} column '{requested}' not found in table columns")
        return requested

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        f"Could not auto-detect {role} column. Provide it explicitly. "
        f"Available columns: {', '.join(map(str, df.columns[:30]))}"
    )


def _parse_score_mapping(value: Any) -> Dict[str, float]:
    """Parse one JSON/dict-like support-score value into class -> score."""
    if isinstance(value, dict):
        raw = value
    else:
        text = str(value or "").strip()
        if not text or text.lower() in MISSING_LABEL_TOKENS:
            return {}
        try:
            parsed = json.loads(text)
        except Exception:
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                return {}
        raw = parsed if isinstance(parsed, dict) else {}

    out: Dict[str, float] = {}
    for key, val in raw.items():
        try:
            out[str(key)] = float(val)
        except Exception:
            continue
    return out


def load_predictions_from_table(
    predictions_path: str | Path,
    *,
    prediction_column: Optional[str] = None,
    sample_id_column: Optional[str] = None,
    score_json_column: Optional[str] = None,
    score_prefix: Optional[str] = None,
) -> Tuple[pd.Series, Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Load predicted labels and optional class-support scores from a table.

    This is intentionally generic. It supports NetworkParser query outputs and
    simple holdout-prediction tables. It does not rerun feature filtering,
    model training, tree construction, or bootstrap confidence scoring.
    """
    df = _read_table(predictions_path)
    if df.empty:
        raise ValueError(f"Prediction table is empty: {predictions_path}")

    sample_col = _resolve_required_column(
        df,
        sample_id_column,
        candidates=("sample_id", "Sample_ID", "sample", "Sample", "id", "ID"),
        role="sample-id",
    )
    pred_col = _resolve_required_column(
        df,
        prediction_column,
        candidates=(
            "predicted_label",
            "prediction",
            "predicted_terminal_label",
            "predicted_level2_identity",
            "predicted_level2",
            "predicted_level1_identity",
            "predicted_level1",
        ),
        role="prediction",
    )

    index = df[sample_col].astype(str).map(normalize_sample_id)
    y_pred = _normalise_label_series(df[pred_col])
    y_pred.index = index
    y_pred = y_pred.dropna()
    y_pred = y_pred[y_pred.index.astype(str).str.len() > 0]
    y_pred = y_pred[~y_pred.index.duplicated(keep="first")]
    y_pred = y_pred.rename("predicted_label")

    score_df: Optional[pd.DataFrame] = None
    score_source: Optional[str] = None

    if score_json_column:
        if score_json_column not in df.columns:
            raise ValueError(f"score_json_column '{score_json_column}' not found in prediction table")
        score_dicts = [_parse_score_mapping(v) for v in df[score_json_column].tolist()]
        score_df = score_dicts_to_frame(df[sample_col].astype(str).tolist(), score_dicts)
        score_source = score_json_column
    elif score_prefix:
        score_cols = [col for col in df.columns if str(col).startswith(str(score_prefix))]
        if not score_cols:
            raise ValueError(f"No score columns found with prefix '{score_prefix}'")
        score_df = df.loc[:, score_cols].copy()
        score_df.columns = [str(c)[len(str(score_prefix)):] for c in score_cols]
        score_df.index = index
        score_df = score_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        score_source = f"prefix:{score_prefix}"
    else:
        for candidate in ("class_support_json", "class_support", "support_scores_json"):
            if candidate in df.columns:
                score_dicts = [_parse_score_mapping(v) for v in df[candidate].tolist()]
                score_df = score_dicts_to_frame(df[sample_col].astype(str).tolist(), score_dicts)
                score_source = candidate
                break

    summary = {
        "predictions_path": str(predictions_path),
        "sample_id_column": str(sample_col),
        "prediction_column": str(pred_col),
        "n_prediction_rows": int(df.shape[0]),
        "n_predicted_samples_after_normalization": int(y_pred.shape[0]),
        "score_source": score_source,
        "score_matrix_available": bool(score_df is not None and not score_df.empty),
    }
    return y_pred, score_df, summary




def _prediction_table_with_sample_index(
    predictions_path: str | Path,
    sample_id_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """Read a prediction table and normalize its sample identifier index."""
    df = _read_table(predictions_path)
    if df.empty:
        raise ValueError(f"Prediction table is empty: {predictions_path}")
    sample_col = _resolve_required_column(
        df,
        sample_id_column,
        candidates=("sample_id", "Sample_ID", "sample", "Sample", "id", "ID"),
        role="sample-id",
    )
    df = df.copy()
    df["__sample_id_normalized"] = df[sample_col].astype(str).map(normalize_sample_id)
    df = df[df["__sample_id_normalized"].astype(str).str.len() > 0]
    df = df.drop_duplicates(subset=["__sample_id_normalized"], keep="first")
    df = df.set_index("__sample_id_normalized", drop=False)
    return df, sample_col


def _metadata_truth_frame(
    meta_path: str | Path,
    hierarchy_labels: Sequence[str],
    sample_id_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Load multiple metadata truth columns with a normalized sample-id index."""
    meta = _read_table(meta_path)
    missing = [str(label) for label in hierarchy_labels if str(label) not in meta.columns]
    if missing:
        raise ValueError(f"Metadata is missing hierarchy label column(s): {', '.join(missing)}")

    sid_col = _resolve_sample_id_column(meta, sample_id_column)
    if sid_col is not None:
        index = meta[sid_col].astype(str).map(normalize_sample_id)
    else:
        index = pd.Index(meta.index.astype(str).map(normalize_sample_id))
        logger.warning(
            "No sample-id column was detected in metadata. Using metadata row index for hierarchy evaluation alignment."
        )

    truth = pd.DataFrame(index=index)
    for label in hierarchy_labels:
        truth[str(label)] = _normalise_label_series(meta[str(label)]).values
    truth = truth.dropna(how="any")
    truth = truth[truth.index.astype(str).str.len() > 0]
    truth = truth[~truth.index.duplicated(keep="first")]
    return truth, sid_col


def _hierarchy_prediction_columns(df: pd.DataFrame, n_levels: int) -> List[Optional[str]]:
    """Resolve predicted_level columns for hierarchy outputs, one per level."""
    cols: List[Optional[str]] = []
    for i in range(1, int(n_levels) + 1):
        candidates = [
            f"predicted_level{i}",
            f"predicted_level{i}_identity",
            f"level{i}_prediction",
            f"level_{i}_prediction",
        ]
        found = next((col for col in candidates if col in df.columns), None)
        cols.append(found)
    return cols


def _parse_predicted_hierarchy_path_value(value: Any, hierarchy_labels: Sequence[str]) -> Dict[str, str]:
    """Parse strings like label=value / label=value into label -> predicted value."""
    text = str(value or "").strip()
    out: Dict[str, str] = {}
    if not text or text.lower() in MISSING_LABEL_TOKENS:
        return out
    pieces = [piece.strip() for piece in text.split("/") if piece.strip()]
    for idx, piece in enumerate(pieces):
        if "=" in piece:
            key, val = piece.split("=", 1)
            out[str(key).strip()] = str(val).strip()
        elif idx < len(hierarchy_labels):
            out[str(hierarchy_labels[idx])] = piece
    return out


def evaluate_hierarchy_prediction_table(
    *,
    predictions_path: str | Path,
    meta_path: str | Path,
    hierarchy_labels: Sequence[str],
    output_dir: str | Path,
    metadata_sample_id_column: Optional[str] = None,
    prediction_sample_id_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a recursive hierarchy prediction table against ordered metadata labels.

    Outputs include per-sample full-path correctness and prefix-depth correctness.
    Per-level TP/FP/TN/FN metrics are still produced by evaluate_prediction_table;
    this helper adds the hierarchy-wide route-level summary.
    """
    labels = [str(x) for x in hierarchy_labels if str(x).strip()]
    if not labels:
        raise ValueError("hierarchy_labels must contain at least one metadata label column")

    out = _ensure_dir(Path(output_dir))
    pred_df, pred_sample_col = _prediction_table_with_sample_index(
        predictions_path=predictions_path,
        sample_id_column=prediction_sample_id_column,
    )
    truth_df, meta_sample_col = _metadata_truth_frame(
        meta_path=meta_path,
        hierarchy_labels=labels,
        sample_id_column=metadata_sample_id_column,
    )

    common = truth_df.index.intersection(pred_df.index)
    if common.empty:
        summary = {
            "status": "skipped",
            "message": "No overlapping labelled samples between hierarchy predictions and metadata.",
            "hierarchy_labels": labels,
            "n_evaluated_samples": 0,
            "artifacts": {
                "full_path_predictions_tsv": str(out / "hierarchy_full_path_predictions.tsv"),
                "summary_json": str(out / "hierarchy_full_path_summary.json"),
            },
        }
        _write_json(summary, out / "hierarchy_full_path_summary.json")
        return summary

    pred_level_cols = _hierarchy_prediction_columns(pred_df, len(labels))
    rows: List[Dict[str, Any]] = []
    for sample_id in common.astype(str):
        truth_row = truth_df.loc[sample_id]
        pred_row = pred_df.loc[sample_id]
        parsed_path = _parse_predicted_hierarchy_path_value(
            pred_row.get("predicted_hierarchy_path", ""),
            labels,
        )

        true_parts: List[str] = []
        pred_parts: List[str] = []
        level_correct: List[bool] = []
        row: Dict[str, Any] = {"sample_id": sample_id}

        for i, label in enumerate(labels, start=1):
            true_value = str(truth_row.get(label, "")).strip()
            pred_col = pred_level_cols[i - 1]
            if pred_col is not None:
                pred_value = str(pred_row.get(pred_col, "")).strip()
            else:
                pred_value = str(parsed_path.get(label, "")).strip()

            true_parts.append(f"{label}={true_value}")
            pred_parts.append(f"{label}={pred_value}")
            is_correct = bool(true_value and pred_value and true_value == pred_value)
            level_correct.append(is_correct)

            row[f"level{i}_label_column"] = label
            row[f"level{i}_true_label"] = true_value
            row[f"level{i}_predicted_label"] = pred_value
            row[f"level{i}_correct"] = is_correct

        deepest_correct_prefix = 0
        for ok in level_correct:
            if ok:
                deepest_correct_prefix += 1
            else:
                break

        row.update(
            {
                "true_hierarchy_path": " / ".join(true_parts),
                "predicted_hierarchy_path_normalized": " / ".join(pred_parts),
                "full_path_correct": bool(all(level_correct)),
                "deepest_correct_prefix_level": int(deepest_correct_prefix),
                "terminal_level_correct": bool(level_correct[-1]) if level_correct else False,
            }
        )
        rows.append(row)

    full_path_df = pd.DataFrame(rows)
    full_path_path = out / "hierarchy_full_path_predictions.tsv"
    full_path_df.to_csv(full_path_path, sep="\t", index=False)

    level_summaries: List[Dict[str, Any]] = []
    for i, label in enumerate(labels, start=1):
        col = f"level{i}_correct"
        values = full_path_df[col].astype(bool) if col in full_path_df.columns else pd.Series(dtype=bool)
        level_summaries.append(
            {
                "level_number": int(i),
                "label_column": label,
                "prediction_column": pred_level_cols[i - 1],
                "n_evaluated_samples": int(values.shape[0]),
                "level_accuracy": float(values.mean()) if not values.empty else None,
            }
        )

    prefix_counts = (
        full_path_df["deepest_correct_prefix_level"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("deepest_correct_prefix_level")
        .reset_index(name="n_samples")
    )
    prefix_path = out / "hierarchy_prefix_depth_counts.tsv"
    prefix_counts.to_csv(prefix_path, sep="\t", index=False)

    summary = {
        "status": "success",
        "hierarchy_labels": labels,
        "n_evaluated_samples": int(full_path_df.shape[0]),
        "full_path_accuracy": float(full_path_df["full_path_correct"].mean()) if not full_path_df.empty else None,
        "terminal_level_accuracy": float(full_path_df["terminal_level_correct"].mean()) if not full_path_df.empty else None,
        "mean_deepest_correct_prefix_level": float(full_path_df["deepest_correct_prefix_level"].mean()) if not full_path_df.empty else None,
        "per_level": level_summaries,
        "input": {
            "predictions_path": str(predictions_path),
            "prediction_sample_id_column": pred_sample_col,
            "metadata_path": str(meta_path),
            "metadata_sample_id_column": meta_sample_col,
        },
        "artifacts": {
            "full_path_predictions_tsv": str(full_path_path),
            "prefix_depth_counts_tsv": str(prefix_path),
            "summary_json": str(out / "hierarchy_full_path_summary.json"),
        },
        "method_note": (
            "Full-path correctness requires every ordered hierarchy level to match for a sample. "
            "This is evaluation-only and does not rerun feature filtering, tree construction, bootstrapping, or model training."
        ),
    }
    _write_json(summary, out / "hierarchy_full_path_summary.json")
    return summary


def evaluate_hierarchy_branch_diagnostics(
    *,
    predictions_path: str | Path,
    meta_path: str | Path,
    hierarchy_labels: Sequence[str],
    output_dir: str | Path,
    metadata_sample_id_column: Optional[str] = None,
    prediction_sample_id_column: Optional[str] = None,
    parent_level: int = 1,
    min_samples_for_reliable_profile: int = 20,
    min_balanced_accuracy_for_exact: float = 0.70,
    min_prediction_coverage_for_exact: float = 0.80,
) -> Dict[str, Any]:
    """Write branch-conditioned hierarchy diagnostics.

    This evaluates downstream hierarchy labels within each true parent branch,
    e.g. within each lineage, how well profile-level and binary-endpoint
    predictions are recovered. It separates two failure modes that ordinary
    per-level metrics collapse together:

      1. performance failure: prediction exists but is wrong;
      2. routing/coverage failure: downstream prediction is missing because the
         hierarchy traversal stopped or took an unsupported branch.

    The function is evaluation-only. It does not rerun filtering, model
    training, tree construction, bootstrapping, or confidence scoring.
    """
    labels = [str(x) for x in hierarchy_labels if str(x).strip()]
    if len(labels) < 2:
        raise ValueError("Branch diagnostics require at least two hierarchy labels.")

    parent_level = int(parent_level)
    if parent_level < 1 or parent_level >= len(labels):
        raise ValueError("parent_level must be between 1 and len(hierarchy_labels) - 1")

    out = _ensure_dir(Path(output_dir))
    pred_df, pred_sample_col = _prediction_table_with_sample_index(
        predictions_path=predictions_path,
        sample_id_column=prediction_sample_id_column,
    )
    truth_df, meta_sample_col = _metadata_truth_frame(
        meta_path=meta_path,
        hierarchy_labels=labels,
        sample_id_column=metadata_sample_id_column,
    )

    common = truth_df.index.intersection(pred_df.index)
    pred_level_cols = _hierarchy_prediction_columns(pred_df, len(labels))

    if common.empty:
        summary = {
            "status": "skipped",
            "message": "No overlapping labelled samples between hierarchy predictions and metadata.",
            "hierarchy_labels": labels,
            "n_evaluated_samples": 0,
            "artifacts": {
                "per_parent_child_metrics_tsv": str(out / "per_parent_child_metrics.tsv"),
                "summary_json": str(out / "branch_diagnostics_summary.json"),
            },
        }
        _write_json(summary, out / "branch_diagnostics_summary.json")
        return summary

    merged = truth_df.loc[common, labels].copy()
    for i, label in enumerate(labels, start=1):
        pred_col = pred_level_cols[i - 1]
        if pred_col is not None:
            merged[f"__pred_level{i}"] = _normalise_label_series(pred_df.loc[common, pred_col])
        else:
            parsed_values: List[str] = []
            for sample_id in common.astype(str):
                parsed = _parse_predicted_hierarchy_path_value(
                    pred_df.loc[sample_id].get("predicted_hierarchy_path", ""),
                    labels,
                )
                parsed_values.append(parsed.get(label, ""))
            merged[f"__pred_level{i}"] = _normalise_label_series(pd.Series(parsed_values, index=common))

    parent_label_col = labels[parent_level - 1]
    parent_pred_col = f"__pred_level{parent_level}"

    rows: List[Dict[str, Any]] = []
    per_sample_rows: List[Dict[str, Any]] = []

    parent_values = sorted(
        [str(x) for x in merged[parent_label_col].dropna().astype(str).unique() if str(x).strip()]
    )

    for parent_value in parent_values:
        branch = merged[merged[parent_label_col].astype(str) == str(parent_value)].copy()
        if branch.empty:
            continue

        parent_pred_available = branch[parent_pred_col].dropna().astype(str).str.len() > 0 if parent_pred_col in branch.columns else pd.Series(dtype=bool)
        parent_correct = (
            branch[parent_pred_col].astype(str) == branch[parent_label_col].astype(str)
        ) if parent_pred_col in branch.columns else pd.Series(False, index=branch.index)

        for child_level in range(parent_level + 1, len(labels) + 1):
            child_label_col = labels[child_level - 1]
            child_pred_col = f"__pred_level{child_level}"

            true_child = _normalise_label_series(branch[child_label_col]).dropna()
            pred_child = _normalise_label_series(branch[child_pred_col]) if child_pred_col in branch.columns else pd.Series(index=branch.index, dtype=object)
            pred_child = pred_child.dropna()

            truth_index = true_child.index
            eval_index = truth_index.intersection(pred_child.index)
            y_true = true_child.loc[eval_index].astype(str) if len(eval_index) else pd.Series(dtype=str)
            y_pred = pred_child.loc[eval_index].astype(str) if len(eval_index) else pd.Series(dtype=str)

            n_parent_samples = int(branch.shape[0])
            n_truth_child = int(true_child.shape[0])
            n_with_child_prediction = int(pred_child.reindex(truth_index).dropna().shape[0]) if n_truth_child else 0
            n_evaluated = int(y_true.shape[0])
            prediction_coverage = float(n_with_child_prediction / max(1, n_truth_child))
            parent_prediction_coverage = float(parent_pred_available.mean()) if len(parent_pred_available) else 0.0
            parent_accuracy_within_branch = float(parent_correct.mean()) if len(parent_correct) else None

            true_classes = sorted(set(y_true.astype(str).tolist())) if n_evaluated else sorted(set(true_child.astype(str).tolist()))
            pred_classes = sorted(set(y_pred.astype(str).tolist())) if n_evaluated else sorted(set(pred_child.astype(str).tolist()))
            predicted_absent_from_truth = sorted(set(pred_classes) - set(true_classes))
            true_never_predicted = sorted(set(true_classes) - set(pred_classes))

            if n_evaluated and len(set(y_true.astype(str))) >= 2:
                labels_union = sorted(set(y_true.astype(str).tolist()).union(set(y_pred.astype(str).tolist())))
                balanced_accuracy = _balanced_accuracy_no_warning(y_true, y_pred)
                macro_tpr = _safe_macro_metric(recall_score, y_true, y_pred, labels=labels_union)
                macro_f1 = _safe_macro_metric(f1_score, y_true, y_pred, labels=labels_union)
            else:
                balanced_accuracy = None
                macro_tpr = None
                macro_f1 = None

            if n_truth_child < int(min_samples_for_reliable_profile) or len(set(true_child.astype(str))) < 2:
                recommended_route = "insufficient_support"
                reason = "Branch has too little evaluable support or only one observed child class."
            elif prediction_coverage < float(min_prediction_coverage_for_exact):
                recommended_route = "routing_or_prediction_coverage_issue"
                reason = "Many samples in this parent branch lack a downstream child prediction."
            elif balanced_accuracy is not None and balanced_accuracy >= float(min_balanced_accuracy_for_exact):
                recommended_route = "use_exact_child_prediction"
                reason = "Child-level prediction is sufficiently recovered within this parent branch."
            elif len(set(true_child.astype(str))) > 2:
                recommended_route = "fallback_to_simpler_endpoint"
                reason = "Multi-class child endpoint is not recovered strongly enough in this parent branch."
            else:
                recommended_route = "needs_review"
                reason = "Binary or low-cardinality child endpoint is underperforming and needs inspection."

            rows.append(
                {
                    "parent_level": int(parent_level),
                    "parent_label_column": parent_label_col,
                    "parent_label": str(parent_value),
                    "child_level": int(child_level),
                    "child_label": child_label_col,
                    "prediction_column": pred_level_cols[child_level - 1],
                    "n_parent_truth_samples": n_parent_samples,
                    "n_child_truth_samples": n_truth_child,
                    "n_with_child_prediction": n_with_child_prediction,
                    "n_evaluated_samples": n_evaluated,
                    "prediction_coverage": prediction_coverage,
                    "parent_prediction_coverage": parent_prediction_coverage,
                    "parent_accuracy_within_branch": parent_accuracy_within_branch,
                    "n_true_child_classes": int(len(set(true_child.astype(str)))) if n_truth_child else 0,
                    "n_pred_child_classes": int(len(set(pred_child.astype(str)))) if not pred_child.empty else 0,
                    "balanced_accuracy": balanced_accuracy,
                    "macro_true_positive_rate": macro_tpr,
                    "macro_f1": macro_f1,
                    "predicted_classes_absent_from_truth": ";".join(predicted_absent_from_truth),
                    "true_classes_never_predicted": ";".join(true_never_predicted),
                    "recommended_route": recommended_route,
                    "reason": reason,
                }
            )

            for sample_id in branch.index.astype(str):
                true_value = branch.loc[sample_id].get(child_label_col, pd.NA)
                pred_value = branch.loc[sample_id].get(child_pred_col, pd.NA)
                per_sample_rows.append(
                    {
                        "sample_id": sample_id,
                        "parent_label_column": parent_label_col,
                        "parent_label": str(parent_value),
                        "child_level": int(child_level),
                        "child_label": child_label_col,
                        "true_child_label": "" if pd.isna(true_value) else str(true_value),
                        "predicted_child_label": "" if pd.isna(pred_value) else str(pred_value),
                        "child_prediction_present": bool(not pd.isna(pred_value) and str(pred_value).strip()),
                        "child_prediction_correct": bool(
                            not pd.isna(true_value)
                            and not pd.isna(pred_value)
                            and str(true_value).strip() == str(pred_value).strip()
                        ),
                    }
                )

    metrics_df = pd.DataFrame(rows)
    metrics_path = out / "per_parent_child_metrics.tsv"
    metrics_df.to_csv(metrics_path, sep="\t", index=False)

    per_sample_df = pd.DataFrame(per_sample_rows)
    per_sample_path = out / "per_sample_branch_predictions.tsv"
    per_sample_df.to_csv(per_sample_path, sep="\t", index=False)

    recommendations_path = out / "per_parent_fallback_recommendations.tsv"
    if not metrics_df.empty:
        metrics_df.sort_values(
            ["child_level", "recommended_route", "prediction_coverage", "balanced_accuracy"],
            ascending=[True, True, True, False],
            na_position="first",
        ).to_csv(recommendations_path, sep="\t", index=False)
    else:
        pd.DataFrame().to_csv(recommendations_path, sep="\t", index=False)

    route_counts = (
        metrics_df["recommended_route"].value_counts(dropna=False).rename_axis("recommended_route").reset_index(name="n_branches")
        if not metrics_df.empty and "recommended_route" in metrics_df.columns
        else pd.DataFrame(columns=["recommended_route", "n_branches"])
    )
    route_counts_path = out / "recommendation_counts.tsv"
    route_counts.to_csv(route_counts_path, sep="\t", index=False)

    summary = {
        "status": "success",
        "hierarchy_labels": labels,
        "parent_level": int(parent_level),
        "parent_label_column": parent_label_col,
        "n_evaluated_samples": int(len(common)),
        "n_parent_branches": int(len(parent_values)),
        "n_branch_child_rows": int(metrics_df.shape[0]),
        "prediction_columns": {f"level_{i}": pred_level_cols[i - 1] for i in range(1, len(labels) + 1)},
        "input": {
            "predictions_path": str(predictions_path),
            "prediction_sample_id_column": pred_sample_col,
            "metadata_path": str(meta_path),
            "metadata_sample_id_column": meta_sample_col,
        },
        "artifacts": {
            "per_parent_child_metrics_tsv": str(metrics_path),
            "per_parent_fallback_recommendations_tsv": str(recommendations_path),
            "per_sample_branch_predictions_tsv": str(per_sample_path),
            "recommendation_counts_tsv": str(route_counts_path),
            "summary_json": str(out / "branch_diagnostics_summary.json"),
        },
        "method_note": (
            "Branch diagnostics evaluate downstream child-label performance within each true parent branch. "
            "They explicitly report prediction coverage so missing downstream predictions are not mistaken for ordinary misclassification."
        ),
    }
    _write_json(summary, out / "branch_diagnostics_summary.json")
    return summary

def evaluate_prediction_table(
    *,
    predictions_path: str | Path,
    meta_path: str | Path,
    label_column: str,
    prediction_column: Optional[str] = None,
    output_dir: str | Path,
    level_name: str = "model",
    metadata_sample_id_column: Optional[str] = None,
    prediction_sample_id_column: Optional[str] = None,
    score_json_column: Optional[str] = None,
    score_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a saved prediction table against labelled metadata.

    Use this for held-out query outputs or any external test-set predictions.
    True-positive rate is reported per class as sensitivity_recall in the
    by-class table, alongside FP/TN/FN, specificity, PPV, NPV, F1, and the
    confusion matrix.
    """
    y_true = load_labels_from_metadata(
        meta_path=meta_path,
        label_column=label_column,
        sample_id_column=metadata_sample_id_column,
    )
    y_pred, score_df, input_summary = load_predictions_from_table(
        predictions_path=predictions_path,
        prediction_column=prediction_column,
        sample_id_column=prediction_sample_id_column,
        score_json_column=score_json_column,
        score_prefix=score_prefix,
    )

    summary = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        class_support_scores=score_df,
        output_dir=output_dir,
        level_name=level_name,
    )
    summary["input"] = {
        **input_summary,
        "metadata_path": str(meta_path),
        "metadata_label_column": str(label_column),
        "metadata_sample_id_column": metadata_sample_id_column,
    }
    _write_json(summary, Path(output_dir) / "model_performance_summary.json")
    return summary

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
        "balanced_accuracy": float(_balanced_accuracy_no_warning(y_true_arr, y_pred_arr)),
        "macro_precision_ppv": float(precision_macro),
        "macro_true_positive_rate": float(recall_macro),
        "macro_sensitivity_recall": float(recall_macro),
        "macro_true_negative_rate": macro_specificity,
        "macro_specificity": macro_specificity,
        "macro_npv": macro_npv,
        "macro_f1": float(f1_macro),
        "weighted_precision_ppv": float(precision_weighted),
        "weighted_true_positive_rate": float(recall_weighted),
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


# -----------------------------------------------------------------------------
# Run-root artifact layout helpers
# -----------------------------------------------------------------------------

_LEGACY_TESTING_ARTIFACTS = {
    "query": "query_results",
    "evaluate": "validation",
}


def _safe_label_token(value: Any, max_len: int = 40) -> str:
    raw = str(value).strip() or "label"
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)
    cleaned = cleaned.strip("_") or "label"
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[:max_len].rstrip("_") or "label"


def resolve_run_artifact_dir(
    anchor_path: str | Path,
    artifact_name: str,
) -> Path:
    """Resolve a standard run artifact directory next to the training run root.

    The canonical layout mirrors successful hierarchy runs such as
    ``training_1/`` + sibling ``query_results/`` + sibling ``validation/``::

        hierarchy_testing_AMR/
        ├── training_1/
        ├── query_results/
        └── validation/

    Legacy ``testing/query`` and ``testing/evaluate`` paths are normalized back
    to ``query_results`` and ``validation``.
    """
    if artifact_name not in {"query_results", "validation"}:
        raise ValueError("artifact_name must be 'query_results' or 'validation'")

    path = Path(anchor_path).expanduser().resolve()
    if path.is_file():
        current = path.parent
    else:
        current = path

    parts = current.parts
    if "testing" in parts:
        testing_idx = parts.index("testing")
        run_root = Path(*parts[:testing_idx])
        legacy_name = parts[testing_idx + 1] if testing_idx + 1 < len(parts) else ""
        mapped = _LEGACY_TESTING_ARTIFACTS.get(str(legacy_name))
        if mapped:
            return run_root / mapped
        return run_root / artifact_name

    if current.name in {"query_results", "validation"}:
        return current.parent / artifact_name

    if current.name.startswith("training"):
        return current.parent / artifact_name

    return current / artifact_name


def normalize_run_artifact_dir(
    output_dir: str | Path,
    artifact_name: str,
) -> Path:
    """Normalize a user-supplied output directory to the canonical artifact name."""
    path = Path(output_dir).expanduser()
    parts = path.parts
    if "testing" in parts:
        testing_idx = parts.index("testing")
        run_root = Path(*parts[:testing_idx])
        legacy_name = parts[testing_idx + 1] if testing_idx + 1 < len(parts) else ""
        mapped = _LEGACY_TESTING_ARTIFACTS.get(str(legacy_name))
        if mapped:
            return run_root / mapped
    if path.name in _LEGACY_TESTING_ARTIFACTS:
        mapped = _LEGACY_TESTING_ARTIFACTS[path.name]
        if mapped == artifact_name:
            return path.parent / artifact_name
    return path


def run_networkparser_evaluation(
    *,
    predictions_path: str | Path,
    meta_path: str | Path,
    hierarchy_labels: Sequence[str],
    output_dir: str | Path,
    metadata_sample_id_column: Optional[str] = None,
    prediction_sample_id_column: Optional[str] = None,
    parent_level: int = 1,
    min_samples_for_reliable_profile: int = 20,
    min_balanced_accuracy_for_exact: float = 0.70,
    min_prediction_coverage_for_exact: float = 0.80,
) -> Dict[str, Any]:
    """Evaluate saved hierarchy predictions and write the validation artifact tree."""
    labels = [str(x) for x in hierarchy_labels if str(x).strip()]
    if not labels:
        raise ValueError("hierarchy_labels must contain at least one metadata label column")

    out = _ensure_dir(normalize_run_artifact_dir(output_dir, "validation"))
    pred_df, _ = _prediction_table_with_sample_index(
        predictions_path=predictions_path,
        sample_id_column=prediction_sample_id_column,
    )
    pred_level_cols = _hierarchy_prediction_columns(pred_df, len(labels))

    targets: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for level_number, label in enumerate(labels, start=1):
        level_name = f"hierarchy_level_{level_number:02d}__{_safe_label_token(label)}"
        level_dir = out / level_name
        prediction_column = pred_level_cols[level_number - 1]
        try:
            summary = evaluate_prediction_table(
                predictions_path=predictions_path,
                meta_path=meta_path,
                label_column=label,
                prediction_column=prediction_column,
                output_dir=level_dir,
                level_name=level_name,
                metadata_sample_id_column=metadata_sample_id_column,
                prediction_sample_id_column=prediction_sample_id_column,
            )
            targets.append(summary)
        except Exception as exc:
            logger.exception("Hierarchy level evaluation failed | level=%s | label=%s", level_number, label)
            failure = {
                "status": "failed",
                "level_number": int(level_number),
                "label_column": label,
                "level_name": level_name,
                "prediction_column": prediction_column,
                "error": str(exc),
            }
            failures.append(failure)
            targets.append(failure)

    try:
        hierarchy_full_path = evaluate_hierarchy_prediction_table(
            predictions_path=predictions_path,
            meta_path=meta_path,
            hierarchy_labels=labels,
            output_dir=out / "hierarchy_full_path",
            metadata_sample_id_column=metadata_sample_id_column,
            prediction_sample_id_column=prediction_sample_id_column,
        )
    except Exception as exc:
        logger.exception("Hierarchy full-path evaluation failed")
        hierarchy_full_path = {
            "status": "failed",
            "error": str(exc),
        }
        failures.append({"stage": "hierarchy_full_path", "error": str(exc)})

    try:
        hierarchy_branch_diagnostics = evaluate_hierarchy_branch_diagnostics(
            predictions_path=predictions_path,
            meta_path=meta_path,
            hierarchy_labels=labels,
            output_dir=out / "hierarchy_branch_diagnostics",
            metadata_sample_id_column=metadata_sample_id_column,
            prediction_sample_id_column=prediction_sample_id_column,
            parent_level=parent_level,
            min_samples_for_reliable_profile=min_samples_for_reliable_profile,
            min_balanced_accuracy_for_exact=min_balanced_accuracy_for_exact,
            min_prediction_coverage_for_exact=min_prediction_coverage_for_exact,
        )
    except Exception as exc:
        logger.exception("Hierarchy branch diagnostics failed")
        hierarchy_branch_diagnostics = {
            "status": "failed",
            "error": str(exc),
        }
        failures.append({"stage": "hierarchy_branch_diagnostics", "error": str(exc)})

    successful_targets = [item for item in targets if item.get("status") == "success"]
    summary = {
        "status": "success" if successful_targets and not failures else ("partial" if successful_targets else "failed"),
        "n_successful_targets": int(len(successful_targets)),
        "n_failed_targets": int(len([item for item in targets if item.get("status") != "success"]) + len(failures)),
        "targets": targets,
        "hierarchy_full_path": hierarchy_full_path,
        "hierarchy_branch_diagnostics": hierarchy_branch_diagnostics,
        "failures": failures,
        "artifacts": {
            "summary_json": str(out / "networkparser_validation_summary.json"),
        },
        "method_note": (
            "This is evaluation-only. It compares saved predictions to metadata truth labels "
            "and does not rerun feature filtering, tree construction, bootstrapping, or model training."
        ),
    }
    _write_json(summary, out / "networkparser_validation_summary.json")
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
        "balanced_accuracy": float(_balanced_accuracy_no_warning(y_true_arr, y_pred_arr)),
        "macro_precision_ppv": float(precision_macro),
        "macro_true_positive_rate": float(recall_macro),
        "macro_sensitivity_recall": float(recall_macro),
        "macro_true_negative_rate": macro_specificity,
        "macro_specificity": macro_specificity,
        "macro_npv": macro_npv,
        "macro_f1": float(f1_macro),
        "weighted_precision_ppv": float(precision_weighted),
        "weighted_true_positive_rate": float(recall_weighted),
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

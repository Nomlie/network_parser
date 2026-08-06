#!/usr/bin/env python3
# network_parser/model_evaluation.py
"""
Model-performance evaluation utilities for NetworkParser.

Purpose
-------
Evaluate saved-model predictions against labelled holdout metadata without
rerunning statistical filtering, feature selection, model training, tree
construction, bootstrapping, or confidence scoring.

Evaluation roles
----------------
- **held_out / external**: generalisation performance on untouched external data.
- **out_of_fold / internal_cv**: internal estimates from leakage-aware CV.
- **training_fit_diagnostics**: same-data scores on the fitting set. These are
  **not** generalisation metrics and must never be reported as such.

Always report end-to-end denominators including abstentions/missing predictions
(total truth, total predictions, matched, called, abstained, coverage).
"""

from __future__ import annotations

import ast
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
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

MISSING_LABEL_TOKENS = frozenset(
    {
        "",
        "-",
        "NA",
        "N/A",
        "<NA>",
        "None",
        "none",
        "nan",
        "NaN",
        "null",
        "NULL",
    }
)
MISSING_LABEL_TOKENS_NORMALIZED = frozenset(
    str(token).strip().casefold() for token in MISSING_LABEL_TOKENS
)

# Prediction tokens treated as abstention / non-called (not class assignments).
DEFAULT_ABSTENTION_TOKENS = frozenset(
    {
        "",
        "unavailable",
        "review",
        "review_required",
        "low_support_review_required",
        "amr_evidence_review_required",
        "abstain",
        "abstain_review_unresolved",
        "unresolved",
        "not_called",
        "missing",
        "na",
        "<na>",
        "nan",
        "none",
    }
)

EVALUATION_ROLES = frozenset(
    {
        "held_out",
        "external",
        "out_of_fold",
        "internal_cv",
        "training_fit_diagnostics",
        "calibration",
    }
)


# -----------------------------------------------------------------------------
# Sample-ID integrity and end-to-end evaluation helpers
# -----------------------------------------------------------------------------


def detect_duplicate_normalized_ids(ids: Sequence[Any]) -> List[str]:
    """Return sorted list of normalized IDs that appear more than once."""
    counts: Dict[str, int] = {}
    for value in ids:
        key = normalize_sample_id(str(value))
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return sorted([k for k, n in counts.items() if n > 1])


def assert_unique_normalized_ids(
    ids: Sequence[Any], *, context: str = "samples"
) -> None:
    """Fail hard on colliding normalized sample IDs (no silent keep-first)."""
    dups = detect_duplicate_normalized_ids(ids)
    if dups:
        raise ValueError(
            f"Duplicate normalized sample IDs in {context}: {dups[:20]}"
            + (" ..." if len(dups) > 20 else "")
            + ". Refusing to silently keep the first occurrence."
        )


def wilson_interval(
    successes: int, n: int, *, z: float = 1.96
) -> Tuple[Optional[float], Optional[float]]:
    """Wilson score interval for a binomial proportion (or (None, None) if n==0)."""
    if n <= 0:
        return None, None
    successes = max(0, min(int(successes), int(n)))
    n = int(n)
    phat = successes / n
    denom = 1.0 + (z * z) / n
    centre = (phat + (z * z) / (2.0 * n)) / denom
    margin = (
        z * math.sqrt((phat * (1.0 - phat) / n) + (z * z) / (4.0 * n * n))
    ) / denom
    return float(max(0.0, centre - margin)), float(min(1.0, centre + margin))


def _is_missing_label(value: Any) -> bool:
    """Return True for native or textual missing-label representations.

    Literal ``0`` is intentionally not missing. It remains a valid binary
    class/baseline label, matching the DataLoader's 0/1 matrix contract.
    """
    if pd.isna(value):
        return True
    return str(value).strip().casefold() in MISSING_LABEL_TOKENS_NORMALIZED


def _is_abstention_label(value: Any, abstention_tokens: Set[str]) -> bool:
    if _is_missing_label(value):
        return True
    return str(value).strip().casefold() in abstention_tokens


def end_to_end_prediction_counts(
    *,
    y_true: pd.Series,
    y_pred: pd.Series,
    abstention_tokens: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Count truth/prediction/matched/called/abstained with full denominators.

    Called = matched samples whose prediction is a non-abstention class label.
    Missing predictions (truth without pred) count as abstained for end-to-end.
    """
    tokens = {t.lower() for t in (abstention_tokens or DEFAULT_ABSTENTION_TOKENS)}

    truth = y_true.copy()
    truth.index = truth.index.astype(str).map(normalize_sample_id)
    truth = truth.dropna()
    truth = truth[truth.index.astype(str).str.len() > 0]
    assert_unique_normalized_ids(
        truth.index.tolist(), context="y_true for end-to-end metrics"
    )

    pred = y_pred.copy()
    pred.index = pred.index.astype(str).map(normalize_sample_id)
    # Keep abstention labels; only drop empty index
    pred = pred[pred.index.astype(str).str.len() > 0]
    assert_unique_normalized_ids(
        pred.index.tolist(), context="y_pred for end-to-end metrics"
    )

    n_truth = int(len(truth))
    n_pred_rows = int(len(pred))
    matched_ids = truth.index.intersection(pred.index)
    n_matched = int(len(matched_ids))

    called_mask = []
    correct_called = 0
    correct_all_matched = 0
    abstained_matched = 0
    for sid in matched_ids:
        t = str(truth.loc[sid])
        p = pred.loc[sid]
        if _is_abstention_label(p, tokens):
            abstained_matched += 1
            called_mask.append(False)
            continue
        called_mask.append(True)
        p_str = str(p)
        if p_str == t:
            correct_called += 1
            correct_all_matched += 1
        # unmatched abstentions don't contribute to correct_all_matched

    n_called = int(sum(called_mask))
    # Truth samples with no prediction row
    missing_pred = int(n_truth - n_matched)
    n_abstained_or_missing = int(abstained_matched + missing_pred)

    # End-to-end accuracy: correct class calls / all truth samples
    # (missing/abstained count as incorrect for e2e assignment accuracy)
    e2e_correct = correct_called  # only non-abstention correct counts
    coverage = float(n_called / n_truth) if n_truth else 0.0
    called_only_acc = float(correct_called / n_called) if n_called else 0.0
    e2e_acc = float(e2e_correct / n_truth) if n_truth else 0.0

    ci_called = (
        wilson_interval(correct_called, n_called) if n_called >= 5 else (None, None)
    )
    ci_e2e = wilson_interval(e2e_correct, n_truth) if n_truth >= 5 else (None, None)
    ci_cov = wilson_interval(n_called, n_truth) if n_truth >= 5 else (None, None)

    return {
        "n_truth_samples": n_truth,
        "n_prediction_samples": n_pred_rows,
        "n_matched_samples": n_matched,
        "n_called_samples": n_called,
        "n_abstained_matched_samples": int(abstained_matched),
        "n_missing_prediction_samples": missing_pred,
        "n_abstained_or_missing_samples": n_abstained_or_missing,
        "coverage_call_rate": coverage,
        "accuracy_called_only": called_only_acc,
        "accuracy_end_to_end_all_truth": e2e_acc,
        "n_correct_called": int(correct_called),
        "n_correct_end_to_end": int(e2e_correct),
        "accuracy_called_only_ci95": {"low": ci_called[0], "high": ci_called[1]},
        "accuracy_end_to_end_ci95": {"low": ci_e2e[0], "high": ci_e2e[1]},
        "coverage_call_rate_ci95": {"low": ci_cov[0], "high": ci_cov[1]},
    }


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
    """Strip and map missing tokens only — never rewrite punctuation globally.

    Replacing every ``-`` with ``_`` can collapse distinct biological labels
    (``A-B`` vs ``A_B``). Use an explicit mapping when renaming is required.
    """
    # Pandas' nullable string dtype preserves native pd.NA/np.nan instead of
    # converting them to the literal strings "<NA>"/"nan". Textual missing
    # tokens are then canonicalised to pd.NA. Literal "0" remains a valid label.
    clean = values.astype("string").str.strip()
    missing_mask = clean.str.casefold().isin(MISSING_LABEL_TOKENS_NORMALIZED)
    return clean.mask(missing_mask, pd.NA)


def _resolve_sample_id_column(
    df: pd.DataFrame, sample_id_column: Optional[str]
) -> Optional[str]:
    if sample_id_column:
        if sample_id_column not in df.columns:
            raise ValueError(
                f"sample_id_column '{sample_id_column}' not found in metadata columns"
            )
        return sample_id_column

    for candidate in (
        "sample_id",
        "Sample_ID",
        "sample",
        "Sample",
        "SampleID",
        "sampleID",
        "id",
        "ID",
    ):
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
    assert_unique_normalized_ids(
        labels.index.tolist(), context=f"metadata labels ({label_column})"
    )
    return labels


def score_dicts_to_frame(
    sample_ids: Sequence[Any],
    score_dicts: Sequence[Optional[Dict[str, Any]]],
) -> pd.DataFrame:
    """Convert per-sample class-support dictionaries into a numeric score matrix.

    Absent class keys remain NaN. Omission is **not** treated as zero unless a
    predictor contract explicitly documents that semantics.
    """
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
    # Do not fillna(0): missing class scores stay NaN.
    return pd.DataFrame(rows, index=index)


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
    *,
    drop_abstentions_for_called_metrics: bool = False,
    abstention_tokens: Optional[Iterable[str]] = None,
    include_truth_without_prediction: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
    truth = _as_label_series(y_true, "true_label")
    pred = _as_label_series(y_pred, "predicted_label")

    truth.index = truth.index.astype(str).map(normalize_sample_id)
    pred.index = pred.index.astype(str).map(normalize_sample_id)

    truth = truth.dropna()
    # Keep abstention predictions for end-to-end accounting; drop empty index only.
    pred = pred[pred.index.astype(str).str.len() > 0]
    truth = truth[truth.index.astype(str).str.len() > 0]
    assert_unique_normalized_ids(truth.index.tolist(), context="aligned y_true")
    assert_unique_normalized_ids(pred.index.tolist(), context="aligned y_pred")

    e2e = end_to_end_prediction_counts(
        y_true=truth,
        y_pred=pred,
        abstention_tokens=abstention_tokens,
    )

    tokens = {t.lower() for t in (abstention_tokens or DEFAULT_ABSTENTION_TOKENS)}
    common = truth.index.intersection(pred.index)

    if include_truth_without_prediction:
        # Full audit: every truth sample, including those with no prediction row.
        ordered = truth.index
        pred_labels: List[Any] = []
        statuses: List[str] = []
        is_abs: List[bool] = []
        for sid in ordered:
            if sid not in pred.index:
                pred_labels.append(pd.NA)
                statuses.append("missing_prediction")
                is_abs.append(True)
                continue
            p = pred.loc[sid]
            abs_flag = _is_abstention_label(p, tokens)
            pred_labels.append(pd.NA if _is_missing_label(p) else str(p).strip())
            statuses.append("abstention" if abs_flag else "called")
            is_abs.append(abs_flag)
        df = pd.DataFrame(
            {
                "sample_id": ordered.astype(str),
                "true_label": truth.loc[ordered].astype(str).values,
                "predicted_label": pred_labels,
                "prediction_status": statuses,
                "is_abstention": is_abs,
            },
            index=ordered,
        )
    else:
        df = pd.DataFrame(
            {
                "sample_id": common.astype(str),
                "true_label": truth.loc[common].astype(str).values,
                # Preserve native NA values until abstention classification.
                # Stringifying here used to turn pd.NA into a false "<NA>"
                # prediction class and contaminate called-only metrics.
                "predicted_label": pred.loc[common].to_numpy(dtype=object),
            },
            index=common,
        )
        df["is_abstention"] = [
            _is_abstention_label(p, tokens) for p in df["predicted_label"].tolist()
        ]
        df["prediction_status"] = [
            "abstention" if a else "called" for a in df["is_abstention"].tolist()
        ]

    if drop_abstentions_for_called_metrics and not df.empty:
        called_df = df.loc[~df["is_abstention"]].copy()
    else:
        called_df = df.copy()

    if score_df is None or score_df.empty:
        return called_df, None, e2e

    scores = score_df.copy()
    scores.index = scores.index.astype(str).map(normalize_sample_id)
    assert_unique_normalized_ids(
        scores.index.tolist(), context="class-support score matrix"
    )
    scores = scores.apply(pd.to_numeric, errors="coerce")
    # Do not fabricate zeros for missing score rows — reindex with NaN.
    scores = scores.reindex(called_df.index)
    return called_df, scores, e2e


def bootstrap_metric_confidence_intervals(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    groups: Optional[Sequence[Any]] = None,
    n_bootstrap: int = 500,
    random_state: int = 42,
    metrics: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Sample- or group-aware bootstrap CIs for principal classification metrics.

    When ``groups`` is provided, resampling is over unique groups (cluster
    bootstrap) so correlated samples within a group are kept together.
    """
    y_true_arr = np.asarray([str(x) for x in y_true], dtype=object)
    y_pred_arr = np.asarray([str(x) for x in y_pred], dtype=object)
    n = int(len(y_true_arr))
    if n == 0 or n != len(y_pred_arr):
        return {
            "status": "skipped",
            "skip_reason": "empty_or_mismatched_inputs",
            "n_bootstrap": 0,
        }

    wanted = (
        list(metrics)
        if metrics is not None
        else [
            "accuracy_called_only",
            "balanced_accuracy",
            "macro_f1",
        ]
    )
    rng = np.random.default_rng(int(random_state))
    n_boot = max(1, int(n_bootstrap))

    if groups is not None:
        g_arr = np.asarray([str(g) for g in groups], dtype=object)
        if len(g_arr) != n:
            return {
                "status": "skipped",
                "skip_reason": "groups_length_mismatch",
                "n_bootstrap": 0,
            }
        unique_groups = np.unique(g_arr)
        unit_ids = unique_groups
        unit_mode = "group"
        unit_to_idx = {ug: np.where(g_arr == ug)[0] for ug in unique_groups}
    else:
        unit_ids = np.arange(n)
        unit_mode = "sample"
        unit_to_idx = {i: np.array([i], dtype=int) for i in unit_ids}

    n_units = int(len(unit_ids))
    if n_units < 5:
        return {
            "status": "skipped",
            "skip_reason": "too_few_resample_units",
            "n_bootstrap": 0,
            "n_units": n_units,
            "unit_mode": unit_mode,
        }

    def _compute(yt: np.ndarray, yp: np.ndarray) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if "accuracy_called_only" in wanted:
            out["accuracy_called_only"] = (
                float(accuracy_score(yt, yp)) if len(yt) else float("nan")
            )
        if "balanced_accuracy" in wanted:
            out["balanced_accuracy"] = (
                float(_balanced_accuracy_no_warning(yt.tolist(), yp.tolist()))
                if len(yt)
                else float("nan")
            )
        if "macro_f1" in wanted:
            labs = sorted(set(yt.tolist()).union(set(yp.tolist())))
            try:
                out["macro_f1"] = (
                    float(
                        f1_score(yt, yp, labels=labs, average="macro", zero_division=0)
                    )
                    if len(yt)
                    else float("nan")
                )
            except Exception:
                out["macro_f1"] = float("nan")
        return out

    point = _compute(y_true_arr, y_pred_arr)
    collected: Dict[str, List[float]] = {k: [] for k in point}

    for _ in range(n_boot):
        draw = rng.choice(unit_ids, size=n_units, replace=True)
        idx_parts = [unit_to_idx[u] for u in draw]
        idx = np.concatenate(idx_parts) if idx_parts else np.array([], dtype=int)
        if idx.size == 0:
            continue
        boot = _compute(y_true_arr[idx], y_pred_arr[idx])
        for k, v in boot.items():
            if np.isfinite(v):
                collected[k].append(float(v))

    intervals: Dict[str, Any] = {}
    for k, vals in collected.items():
        if len(vals) < 20:
            intervals[k] = {
                "point": point.get(k),
                "low": None,
                "high": None,
                "n_successful_resamples": int(len(vals)),
                "status": "insufficient_successful_resamples",
            }
            continue
        arr = np.asarray(vals, dtype=float)
        intervals[k] = {
            "point": point.get(k),
            "low": float(np.quantile(arr, 0.025)),
            "high": float(np.quantile(arr, 0.975)),
            "n_successful_resamples": int(len(vals)),
            "status": "ok",
        }

    return {
        "status": "ok",
        "unit_mode": unit_mode,
        "n_units": n_units,
        "n_bootstrap_requested": n_boot,
        "n_samples": n,
        "intervals": intervals,
        "method": (
            "cluster bootstrap over groups (resample groups with replacement)"
            if unit_mode == "group"
            else "sample bootstrap (resample samples with replacement)"
        ),
    }


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _balanced_accuracy_no_warning(
    y_true: Sequence[Any], y_pred: Sequence[Any]
) -> float:
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


def _safe_macro_metric(
    metric_func,
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    labels: Optional[List[str]] = None,
) -> float:
    """Return a macro metric while treating extra prediction-only labels as zero-support classes."""
    y_true_arr = np.asarray([str(x) for x in y_true], dtype=object)
    y_pred_arr = np.asarray([str(x) for x in y_pred], dtype=object)
    if labels is None:
        labels = sorted(set(y_true_arr.tolist()).union(set(y_pred_arr.tolist())))
    try:
        return float(
            metric_func(
                y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0
            )
        )
    except Exception:
        return 0.0


def _per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]
) -> pd.DataFrame:
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


def _class_has_genuine_support(
    scores: pd.DataFrame, label: str, *, min_finite: int = 5
) -> bool:
    """True when a class column has enough finite scores (not fabricated zeros)."""
    if scores is None or scores.empty or str(label) not in scores.columns:
        return False
    col = pd.to_numeric(scores[str(label)], errors="coerce")
    return int(col.notna().sum()) >= int(min_finite)


def _scores_are_top_class_only(scores: pd.DataFrame, labels: Sequence[str]) -> bool:
    """Detect matrices that only expose a single top-class support column."""
    if scores is None or scores.empty:
        return True
    present = [str(c) for c in scores.columns if str(c) in {str(x) for x in labels}]
    if len(present) <= 1:
        return True
    # Per-row: at most one finite non-NaN score among evaluated labels → top-class only.
    sub = scores.reindex(columns=present).apply(pd.to_numeric, errors="coerce")
    finite_per_row = sub.notna().sum(axis=1)
    if finite_per_row.empty:
        return True
    return bool((finite_per_row <= 1).mean() >= 0.9)


def _scores_usable_for_auc(scores: pd.DataFrame) -> bool:
    """Require finite, non-fabricated score mass for a majority of evaluated rows."""
    if scores is None or scores.empty:
        return False
    arr = scores.to_numpy(dtype=float)
    row_ok = np.isfinite(arr).any(axis=1)
    return bool(row_ok.mean() >= 0.9 and int(row_ok.sum()) >= 5)


def _roc_pr_outputs(
    eval_df: pd.DataFrame,
    scores: Optional[pd.DataFrame],
    labels: List[str],
    *,
    scores_are_calibrated_probabilities: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    ROC/PR from class-support scores.

    Rules
    -----
    - Do not fill absent class scores with zero.
    - One-vs-rest AUC uses genuine finite scores for that class as a ranking
      score (not renamed to probability unless calibrated).
    - Multiclass AUC requires genuine support for **every** evaluated class;
      skip when only top-class support is available and record why.
    - Row-normalisation to a simplex is only applied when scores are documented
      calibrated probabilities.
    """
    if scores is None or scores.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "status": "skipped",
                "skip_reason": "no_score_matrix",
                "message": "No class-support score matrix was available for ROC/PR evaluation.",
            },
        )

    present_true = sorted({str(x) for x in eval_df["true_label"].astype(str).tolist()})
    score_cols = {str(c) for c in scores.columns}
    usable_labels = [str(label) for label in labels if str(label) in score_cols]
    if not usable_labels:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "status": "skipped",
                "skip_reason": "no_matching_score_columns",
                "message": "Class-support scores did not contain columns matching evaluated labels.",
            },
        )

    scores_local = scores.reindex(columns=usable_labels).apply(
        pd.to_numeric, errors="coerce"
    )
    # Rows need at least one finite score for per-class ranking metrics.
    row_ok = scores_local.notna().any(axis=1)
    if int(row_ok.sum()) < 5 or float(row_ok.mean()) < 0.5:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "status": "skipped",
                "skip_reason": "insufficient_finite_score_rows",
                "message": (
                    "Class-support scores are sparse (too few rows with finite scores); "
                    "AUC was not computed. Absent class entries are not filled with zero."
                ),
            },
        )

    eval_local = eval_df.loc[row_ok].copy()
    scores_local = scores_local.loc[row_ok]
    y_true = eval_local["true_label"].astype(str).to_numpy()

    # Ranking scores for OVR; optional simplex only if calibrated probabilities.
    if scores_are_calibrated_probabilities:
        row_sums = scores_local.sum(axis=1, min_count=1)
        nonzero = row_sums > 0
        scores_rank = scores_local.copy()
        scores_rank.loc[nonzero, :] = scores_local.loc[nonzero, :].div(
            row_sums.loc[nonzero], axis=0
        )
        score_kind = "calibrated_probability"
    else:
        scores_rank = scores_local
        score_kind = "support_score_ranking"

    auc_rows: List[Dict[str, Any]] = []
    roc_rows: List[Dict[str, Any]] = []
    pr_rows: List[Dict[str, Any]] = []

    for label in usable_labels:
        if not _class_has_genuine_support(scores_rank, label, min_finite=5):
            auc_rows.append(
                {
                    "class_label": label,
                    "roc_auc_ovr": np.nan,
                    "pr_auc_average_precision": np.nan,
                    "status": "skipped_no_genuine_class_support",
                    "skip_reason": "class_score_column_sparse_or_absent",
                }
            )
            continue

        binary_true = (y_true == label).astype(int)
        if len(np.unique(binary_true)) < 2:
            auc_rows.append(
                {
                    "class_label": label,
                    "roc_auc_ovr": np.nan,
                    "pr_auc_average_precision": np.nan,
                    "status": "skipped_single_class_truth",
                    "skip_reason": "single_class_truth_for_ovr",
                }
            )
            continue

        score = scores_rank[label].to_numpy(dtype=float)
        # Only use rows with finite score for this class (no zero-fill).
        finite_mask = np.isfinite(score)
        if int(finite_mask.sum()) < 5 or len(np.unique(binary_true[finite_mask])) < 2:
            auc_rows.append(
                {
                    "class_label": label,
                    "roc_auc_ovr": np.nan,
                    "pr_auc_average_precision": np.nan,
                    "status": "skipped_insufficient_finite_scores",
                    "skip_reason": "insufficient_finite_scores_for_class",
                }
            )
            continue

        try:
            roc_auc = float(roc_auc_score(binary_true[finite_mask], score[finite_mask]))
            fpr, tpr, roc_thresholds = roc_curve(
                binary_true[finite_mask], score[finite_mask]
            )
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
            pr_auc = float(
                average_precision_score(binary_true[finite_mask], score[finite_mask])
            )
            precision, recall, pr_thresholds = precision_recall_curve(
                binary_true[finite_mask], score[finite_mask]
            )
            padded_thresholds = list(pr_thresholds) + [np.nan]
            for precision_i, recall_i, thr_i in zip(
                precision, recall, padded_thresholds
            ):
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
                "status": "ok"
                if np.isfinite(roc_auc) or np.isfinite(pr_auc)
                else "failed",
                "score_kind": score_kind,
            }
        )

    macro_roc_auc = np.nan
    weighted_roc_auc = np.nan
    macro_pr_auc = np.nan
    multiclass_skip_reason: Optional[str] = None

    present_true_labels = sorted(set(map(str, y_true)))
    multiclass_labels = [
        label for label in usable_labels if label in present_true_labels
    ]
    if len(multiclass_labels) < 2:
        multiclass_skip_reason = "fewer_than_two_truth_classes"
    elif _scores_are_top_class_only(scores_rank, multiclass_labels):
        multiclass_skip_reason = "only_top_class_support_available"
    else:
        missing_support = [
            lab
            for lab in multiclass_labels
            if not _class_has_genuine_support(scores_rank, lab, min_finite=5)
        ]
        if missing_support:
            multiclass_skip_reason = "missing_genuine_support_for_classes:" + ",".join(
                missing_support[:20]
            )
        else:
            score_all = scores_rank.reindex(columns=multiclass_labels)
            # Multiclass OVR requires finite scores for every class on every row.
            complete = score_all.notna().all(axis=1)
            if int(complete.sum()) < 5:
                multiclass_skip_reason = "too_few_rows_with_complete_multiclass_scores"
            else:
                try:
                    y_mc = y_true[complete.to_numpy()]
                    score_arr = score_all.loc[complete].to_numpy(dtype=float)
                    if scores_are_calibrated_probabilities:
                        row_sums = score_arr.sum(axis=1, keepdims=True)
                        row_sums = np.where(row_sums > 0, row_sums, np.nan)
                        score_arr = score_arr / row_sums
                    if np.isfinite(score_arr).all() and len(set(map(str, y_mc))) >= 2:
                        macro_roc_auc = float(
                            roc_auc_score(
                                y_mc,
                                score_arr,
                                labels=multiclass_labels,
                                multi_class="ovr",
                                average="macro",
                            )
                        )
                        weighted_roc_auc = float(
                            roc_auc_score(
                                y_mc,
                                score_arr,
                                labels=multiclass_labels,
                                multi_class="ovr",
                                average="weighted",
                            )
                        )
                    else:
                        multiclass_skip_reason = "non_finite_multiclass_score_block"
                except Exception as exc:
                    multiclass_skip_reason = f"multiclass_auc_failed:{exc}"

    auc_df = pd.DataFrame(auc_rows)
    if not auc_df.empty and "pr_auc_average_precision" in auc_df.columns:
        macro_pr_auc = float(
            pd.to_numeric(auc_df["pr_auc_average_precision"], errors="coerce").mean()
        )

    any_ok = (
        bool(auc_df["status"].eq("ok").any())
        if not auc_df.empty and "status" in auc_df.columns
        else False
    )
    summary: Dict[str, Any] = {
        "status": "ok" if any_ok else "skipped",
        "scored_classes": int(len(usable_labels)),
        "score_kind": score_kind,
        "scores_are_calibrated_probabilities": bool(
            scores_are_calibrated_probabilities
        ),
        "macro_roc_auc_ovr": macro_roc_auc if np.isfinite(macro_roc_auc) else None,
        "weighted_roc_auc_ovr": weighted_roc_auc
        if np.isfinite(weighted_roc_auc)
        else None,
        "macro_pr_auc_average_precision": macro_pr_auc
        if np.isfinite(macro_pr_auc)
        else None,
        "n_truth_classes_in_eval": int(len(present_true)),
    }
    if multiclass_skip_reason is not None:
        summary["multiclass_auc_status"] = "skipped"
        summary["multiclass_auc_skip_reason"] = multiclass_skip_reason
        summary["message"] = f"Multiclass AUC skipped: {multiclass_skip_reason}"
    elif np.isfinite(macro_roc_auc):
        summary["multiclass_auc_status"] = "ok"
    if not any_ok and summary["status"] == "skipped" and "message" not in summary:
        summary["skip_reason"] = "no_per_class_auc_computed"
        summary[
            "message"
        ] = "No per-class AUC could be computed from available support scores."
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
    y_pred = y_pred[y_pred.index.astype(str).str.len() > 0]
    assert_unique_normalized_ids(
        y_pred.index.tolist(), context="prediction table sample IDs"
    )
    y_pred = y_pred.rename("predicted_label")

    score_df: Optional[pd.DataFrame] = None
    score_source: Optional[str] = None

    if score_json_column:
        if score_json_column not in df.columns:
            raise ValueError(
                f"score_json_column '{score_json_column}' not found in prediction table"
            )
        score_dicts = [_parse_score_mapping(v) for v in df[score_json_column].tolist()]
        score_df = score_dicts_to_frame(
            df[sample_col].astype(str).tolist(), score_dicts
        )
        score_source = score_json_column
    elif score_prefix:
        score_cols = [
            col for col in df.columns if str(col).startswith(str(score_prefix))
        ]
        if not score_cols:
            raise ValueError(f"No score columns found with prefix '{score_prefix}'")
        score_df = df.loc[:, score_cols].copy()
        score_df.columns = [str(c)[len(str(score_prefix)) :] for c in score_cols]
        score_df.index = index
        # Keep NaN for missing score cells; do not fabricate zeros.
        score_df = score_df.apply(pd.to_numeric, errors="coerce")
        score_source = f"prefix:{score_prefix}"
    else:
        for candidate in ("class_support_json", "class_support", "support_scores_json"):
            if candidate in df.columns:
                score_dicts = [_parse_score_mapping(v) for v in df[candidate].tolist()]
                score_df = score_dicts_to_frame(
                    df[sample_col].astype(str).tolist(), score_dicts
                )
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
    assert_unique_normalized_ids(
        df["__sample_id_normalized"].tolist(),
        context=f"prediction table ({predictions_path})",
    )
    df = df.set_index("__sample_id_normalized", drop=False)
    return df, sample_col


def _metadata_truth_frame(
    meta_path: str | Path,
    hierarchy_labels: Sequence[str],
    sample_id_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Load multiple metadata truth columns with a normalized sample-id index."""
    meta = _read_table(meta_path)
    missing = [
        str(label) for label in hierarchy_labels if str(label) not in meta.columns
    ]
    if missing:
        raise ValueError(
            f"Metadata is missing hierarchy label column(s): {', '.join(missing)}"
        )

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
    assert_unique_normalized_ids(
        truth.index.tolist(), context="metadata hierarchy truth"
    )
    return truth, sid_col


def _hierarchy_prediction_columns(
    df: pd.DataFrame, n_levels: int
) -> List[Optional[str]]:
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


def _parse_predicted_hierarchy_path_value(
    value: Any, hierarchy_labels: Sequence[str]
) -> Dict[str, str]:
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
        raise ValueError(
            "hierarchy_labels must contain at least one metadata label column"
        )

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
        summary: Dict[str, Any] = {
            "status": "skipped",
            "message": "No overlapping labelled samples between hierarchy predictions and metadata.",
            "hierarchy_labels": labels,
            "n_evaluated_samples": 0,
            "artifacts": {
                "full_path_predictions_tsv": str(
                    out / "hierarchy_full_path_predictions.tsv"
                ),
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
                "terminal_level_correct": bool(level_correct[-1])
                if level_correct
                else False,
            }
        )
        rows.append(row)

    full_path_df = pd.DataFrame(rows)
    full_path_path = out / "hierarchy_full_path_predictions.tsv"
    full_path_df.to_csv(full_path_path, sep="\t", index=False)

    level_summaries: List[Dict[str, Any]] = []
    for i, label in enumerate(labels, start=1):
        col = f"level{i}_correct"
        values = (
            full_path_df[col].astype(bool)
            if col in full_path_df.columns
            else pd.Series(dtype=bool)
        )
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
        "full_path_accuracy": float(full_path_df["full_path_correct"].mean())
        if not full_path_df.empty
        else None,
        "terminal_level_accuracy": float(full_path_df["terminal_level_correct"].mean())
        if not full_path_df.empty
        else None,
        "mean_deepest_correct_prefix_level": float(
            full_path_df["deepest_correct_prefix_level"].mean()
        )
        if not full_path_df.empty
        else None,
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
                "per_parent_child_metrics_tsv": str(
                    out / "per_parent_child_metrics.tsv"
                ),
                "summary_json": str(out / "branch_diagnostics_summary.json"),
            },
        }
        _write_json(summary, out / "branch_diagnostics_summary.json")
        return summary

    merged = truth_df.loc[common, labels].copy()
    for i, label in enumerate(labels, start=1):
        pred_col = pred_level_cols[i - 1]
        if pred_col is not None:
            merged[f"__pred_level{i}"] = _normalise_label_series(
                pred_df.loc[common, pred_col]
            )
        else:
            parsed_values: List[str] = []
            for sample_id in common.astype(str):
                parsed = _parse_predicted_hierarchy_path_value(
                    pred_df.loc[sample_id].get("predicted_hierarchy_path", ""),
                    labels,
                )
                parsed_values.append(parsed.get(label, ""))
            merged[f"__pred_level{i}"] = _normalise_label_series(
                pd.Series(parsed_values, index=common)
            )

    parent_label_col = labels[parent_level - 1]
    parent_pred_col = f"__pred_level{parent_level}"

    rows: List[Dict[str, Any]] = []
    per_sample_rows: List[Dict[str, Any]] = []

    parent_values = sorted(
        [
            str(x)
            for x in merged[parent_label_col].dropna().astype(str).unique()
            if str(x).strip()
        ]
    )

    for parent_value in parent_values:
        branch = merged[
            merged[parent_label_col].astype(str) == str(parent_value)
        ].copy()
        if branch.empty:
            continue

        parent_pred_available = (
            branch[parent_pred_col].dropna().astype(str).str.len() > 0
            if parent_pred_col in branch.columns
            else pd.Series(dtype=bool)
        )
        parent_correct = (
            (
                branch[parent_pred_col].astype(str)
                == branch[parent_label_col].astype(str)
            )
            if parent_pred_col in branch.columns
            else pd.Series(False, index=branch.index)
        )

        for child_level in range(parent_level + 1, len(labels) + 1):
            child_label_col = labels[child_level - 1]
            child_pred_col = f"__pred_level{child_level}"

            true_child = _normalise_label_series(branch[child_label_col]).dropna()
            pred_child = (
                _normalise_label_series(branch[child_pred_col])
                if child_pred_col in branch.columns
                else pd.Series(index=branch.index, dtype=object)
            )
            pred_child = pred_child.dropna()

            truth_index = true_child.index
            eval_index = truth_index.intersection(pred_child.index)
            y_true = (
                true_child.loc[eval_index].astype(str)
                if len(eval_index)
                else pd.Series(dtype=str)
            )
            y_pred = (
                pred_child.loc[eval_index].astype(str)
                if len(eval_index)
                else pd.Series(dtype=str)
            )

            n_parent_samples = int(branch.shape[0])
            n_truth_child = int(true_child.shape[0])
            n_with_child_prediction = (
                int(pred_child.reindex(truth_index).dropna().shape[0])
                if n_truth_child
                else 0
            )
            n_evaluated = int(y_true.shape[0])
            prediction_coverage = float(n_with_child_prediction / max(1, n_truth_child))
            parent_prediction_coverage = (
                float(parent_pred_available.mean())
                if len(parent_pred_available)
                else 0.0
            )
            parent_accuracy_within_branch = (
                float(parent_correct.mean()) if len(parent_correct) else None
            )

            true_classes = (
                sorted(set(y_true.astype(str).tolist()))
                if n_evaluated
                else sorted(set(true_child.astype(str).tolist()))
            )
            pred_classes = (
                sorted(set(y_pred.astype(str).tolist()))
                if n_evaluated
                else sorted(set(pred_child.astype(str).tolist()))
            )
            predicted_absent_from_truth = sorted(set(pred_classes) - set(true_classes))
            true_never_predicted = sorted(set(true_classes) - set(pred_classes))

            if n_evaluated and len(set(y_true.astype(str))) >= 2:
                labels_union = sorted(
                    set(y_true.astype(str).tolist()).union(
                        set(y_pred.astype(str).tolist())
                    )
                )
                balanced_accuracy = _balanced_accuracy_no_warning(y_true, y_pred)
                macro_tpr = _safe_macro_metric(
                    recall_score, y_true, y_pred, labels=labels_union
                )
                macro_f1 = _safe_macro_metric(
                    f1_score, y_true, y_pred, labels=labels_union
                )
            else:
                balanced_accuracy = None
                macro_tpr = None
                macro_f1 = None

            if (
                n_truth_child < int(min_samples_for_reliable_profile)
                or len(set(true_child.astype(str))) < 2
            ):
                recommended_route = "insufficient_support"
                reason = "Branch has too little evaluable support or only one observed child class."
            elif prediction_coverage < float(min_prediction_coverage_for_exact):
                recommended_route = "routing_or_prediction_coverage_issue"
                reason = "Many samples in this parent branch lack a downstream child prediction."
            elif balanced_accuracy is not None and balanced_accuracy >= float(
                min_balanced_accuracy_for_exact
            ):
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
                    "n_true_child_classes": int(len(set(true_child.astype(str))))
                    if n_truth_child
                    else 0,
                    "n_pred_child_classes": int(len(set(pred_child.astype(str))))
                    if not pred_child.empty
                    else 0,
                    "balanced_accuracy": balanced_accuracy,
                    "macro_true_positive_rate": macro_tpr,
                    "macro_f1": macro_f1,
                    "predicted_classes_absent_from_truth": ";".join(
                        predicted_absent_from_truth
                    ),
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
                        "true_child_label": ""
                        if pd.isna(true_value)
                        else str(true_value),
                        "predicted_child_label": ""
                        if pd.isna(pred_value)
                        else str(pred_value),
                        "child_prediction_present": bool(
                            not pd.isna(pred_value) and str(pred_value).strip()
                        ),
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
            [
                "child_level",
                "recommended_route",
                "prediction_coverage",
                "balanced_accuracy",
            ],
            ascending=[True, True, True, False],
            na_position="first",
        ).to_csv(recommendations_path, sep="\t", index=False)
    else:
        pd.DataFrame().to_csv(recommendations_path, sep="\t", index=False)

    route_counts = (
        metrics_df["recommended_route"]
        .value_counts(dropna=False)
        .rename_axis("recommended_route")
        .reset_index(name="n_branches")
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
        "prediction_columns": {
            f"level_{i}": pred_level_cols[i - 1] for i in range(1, len(labels) + 1)
        },
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
    evaluation_role: str = "held_out",
    abstention_tokens: Optional[Iterable[str]] = None,
    groups: Optional[Sequence[Any]] = None,
    n_bootstrap: int = 500,
    scores_are_calibrated_probabilities: bool = False,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Evaluate predictions and write diagnostic performance artifacts.

    Parameters
    ----------
    evaluation_role
        ``held_out`` / ``external`` / ``out_of_fold`` / ``internal_cv`` for
        generalisation or internal estimates; ``training_fit_diagnostics`` for
        same-data scores that must not be reported as generalisation.
    groups
        Optional group labels aligned to called samples (or truth index order
        after alignment) for cluster bootstrap CIs.
    scores_are_calibrated_probabilities
        When False (default), class scores are treated as support/ranking
        scores, not probabilities. Row-normalisation to a simplex is skipped.
    """
    role = str(evaluation_role or "held_out").strip().lower()
    if role not in EVALUATION_ROLES:
        role = "held_out"

    out = _ensure_dir(Path(output_dir))
    eval_df, scores, e2e = _align_truth_and_predictions(
        y_true,
        y_pred,
        class_support_scores,
        drop_abstentions_for_called_metrics=True,
        abstention_tokens=abstention_tokens,
    )

    artifacts = {
        "sample_predictions": str(out / "evaluated_sample_predictions.tsv"),
        "summary_json": str(out / "model_performance_summary.json"),
        "by_class_tsv": str(out / "model_performance_by_class.tsv"),
        "confusion_matrix_tsv": str(out / "confusion_matrix.tsv"),
        "roc_auc_summary_tsv": str(out / "roc_auc_summary.tsv"),
        "roc_curve_points_tsv": str(out / "roc_curve_points.tsv"),
        "pr_curve_points_tsv": str(out / "pr_curve_points.tsv"),
        "bootstrap_ci_json": str(out / "bootstrap_metric_cis.json"),
    }

    # Full audit table: every truth sample, including those without predictions.
    full_df, _, _ = _align_truth_and_predictions(
        y_true,
        y_pred,
        class_support_scores,
        drop_abstentions_for_called_metrics=False,
        abstention_tokens=abstention_tokens,
        include_truth_without_prediction=True,
    )
    full_df.to_csv(artifacts["sample_predictions"], sep="\t", index=False)

    if eval_df.empty and e2e.get("n_truth_samples", 0) == 0:
        summary: Dict[str, Any] = {
            "status": "skipped",
            "level_name": level_name,
            "evaluation_role": role,
            "is_generalization_estimate": role not in {"training_fit_diagnostics"},
            "n_evaluated_samples": 0,
            "end_to_end": e2e,
            "message": "No overlapping labelled samples between predictions and metadata.",
            "artifacts": artifacts,
        }
        _write_json(summary, out / "model_performance_summary.json")
        return summary

    if eval_df.empty:
        # Truth exists but no called predictions
        summary = {
            "status": "success",
            "level_name": level_name,
            "evaluation_role": role,
            "is_generalization_estimate": role not in {"training_fit_diagnostics"},
            "n_evaluated_samples": 0,
            "n_called_samples": 0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "accuracy_called_only": 0.0,
            "end_to_end": e2e,
            "coverage_call_rate": e2e.get("coverage_call_rate", 0.0),
            "accuracy_end_to_end_all_truth": e2e.get(
                "accuracy_end_to_end_all_truth", 0.0
            ),
            "method_note": (
                "All matched predictions abstained or were missing; called-only metrics are empty. "
                "End-to-end metrics use the full truth denominator. "
                "Per-sample audit includes truth samples without predictions."
            ),
            "artifacts": artifacts,
        }
        if role == "training_fit_diagnostics":
            summary[
                "method_note"
            ] += " Role=training_fit_diagnostics: not a generalisation performance claim."
        _write_json(summary, out / "model_performance_summary.json")
        return summary

    y_true_arr = eval_df["true_label"].astype(str).to_numpy()
    y_pred_arr = eval_df["predicted_label"].astype(str).to_numpy()
    labels = sorted(set(y_true_arr).union(set(y_pred_arr)))

    by_class = _per_class_metrics(y_true_arr, y_pred_arr, labels)
    by_class.to_csv(artifacts["by_class_tsv"], sep="\t", index=False)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    cm_df = pd.DataFrame(
        cm, index=[f"true::{x}" for x in labels], columns=[f"pred::{x}" for x in labels]
    )
    cm_df.to_csv(artifacts["confusion_matrix_tsv"], sep="\t")

    precision_macro = precision_score(
        y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0
    )
    recall_macro = recall_score(
        y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0
    )
    f1_macro = f1_score(
        y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0
    )
    precision_weighted = precision_score(
        y_true_arr, y_pred_arr, labels=labels, average="weighted", zero_division=0
    )
    recall_weighted = recall_score(
        y_true_arr, y_pred_arr, labels=labels, average="weighted", zero_division=0
    )
    f1_weighted = f1_score(
        y_true_arr, y_pred_arr, labels=labels, average="weighted", zero_division=0
    )

    macro_specificity = (
        float(pd.to_numeric(by_class["specificity"], errors="coerce").mean())
        if not by_class.empty
        else 0.0
    )
    macro_npv = (
        float(pd.to_numeric(by_class["npv"], errors="coerce").mean())
        if not by_class.empty
        else 0.0
    )

    n_called = int(len(eval_df))
    n_correct = int((y_true_arr == y_pred_arr).sum())
    called_acc = float(accuracy_score(y_true_arr, y_pred_arr))
    called_acc_ci = (
        wilson_interval(n_correct, n_called) if n_called >= 5 else (None, None)
    )

    auc_df, roc_points, pr_points, auc_summary = _roc_pr_outputs(
        eval_df,
        scores,
        labels,
        scores_are_calibrated_probabilities=bool(scores_are_calibrated_probabilities),
    )
    auc_df.to_csv(artifacts["roc_auc_summary_tsv"], sep="\t", index=False)
    roc_points.to_csv(artifacts["roc_curve_points_tsv"], sep="\t", index=False)
    pr_points.to_csv(artifacts["pr_curve_points_tsv"], sep="\t", index=False)

    # Align optional groups to called eval_df index when possible.
    boot_groups: Optional[Sequence[Any]] = None
    if groups is not None:
        try:
            gser = pd.Series(list(groups))
            if len(gser) == n_called:
                boot_groups = gser.tolist()
            elif hasattr(groups, "index"):
                gser = pd.Series(groups)
                gser.index = gser.index.astype(str).map(normalize_sample_id)
                boot_groups = gser.reindex(eval_df.index).tolist()
            else:
                boot_groups = None
        except Exception:
            boot_groups = None

    boot_ci = bootstrap_metric_confidence_intervals(
        y_true_arr,
        y_pred_arr,
        groups=boot_groups,
        n_bootstrap=int(n_bootstrap),
        random_state=int(random_state),
    )
    _write_json(boot_ci, out / "bootstrap_metric_cis.json")

    try:
        mcc = float(matthews_corrcoef(y_true_arr, y_pred_arr))
    except Exception:
        mcc = 0.0

    summary = {
        "status": "success",
        "level_name": level_name,
        "evaluation_role": role,
        "is_generalization_estimate": role not in {"training_fit_diagnostics"},
        "n_evaluated_samples": n_called,
        "n_called_samples": n_called,
        "n_classes_observed": int(len(set(y_true_arr))),
        "n_prediction_classes": int(len(set(y_pred_arr))),
        "accuracy": called_acc,
        "accuracy_called_only": called_acc,
        "accuracy_called_only_ci95": {
            "low": called_acc_ci[0],
            "high": called_acc_ci[1],
        },
        "balanced_accuracy": float(
            _balanced_accuracy_no_warning(y_true_arr, y_pred_arr)
        ),
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
        "end_to_end": e2e,
        "n_truth_samples": e2e.get("n_truth_samples"),
        "n_prediction_samples": e2e.get("n_prediction_samples"),
        "n_matched_samples": e2e.get("n_matched_samples"),
        "n_abstained_or_missing_samples": e2e.get("n_abstained_or_missing_samples"),
        "coverage_call_rate": e2e.get("coverage_call_rate"),
        "accuracy_end_to_end_all_truth": e2e.get("accuracy_end_to_end_all_truth"),
        "roc_pr": auc_summary,
        "bootstrap_metric_cis": boot_ci,
        "artifacts": artifacts,
        "method_note": (
            "Called-only metrics exclude abstentions. End-to-end metrics use all truth samples "
            "as the denominator (NA/abstentions/missing contribute zero correct assignments, "
            "reducing coverage and end-to-end accuracy). Literal label 0 remains a valid class. "
            "Per-sample audit includes truth samples without predictions. "
            "Class-support scores are not treated as probabilities unless "
            "scores_are_calibrated_probabilities=True. "
            "Absent class-score entries are not filled with zero."
            + (
                " Role=training_fit_diagnostics: same-data scores are not generalisation performance."
                if role == "training_fit_diagnostics"
                else ""
            )
        ),
    }

    _write_json(summary, out / "model_performance_summary.json")
    logger.info(
        "Model evaluation complete | role=%s | level=%s | called=%d | e2e_acc=%.4f | called_acc=%.4f | coverage=%.4f",
        role,
        level_name,
        n_called,
        float(summary.get("accuracy_end_to_end_all_truth") or 0.0),
        float(summary["accuracy_called_only"]),
        float(summary.get("coverage_call_rate") or 0.0),
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
        raise ValueError(
            "hierarchy_labels must contain at least one metadata label column"
        )

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
            logger.exception(
                "Hierarchy level evaluation failed | level=%s | label=%s",
                level_number,
                label,
            )
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
    # Level failures are stored once in `targets`. `failures` also lists them for
    # detail plus separate stage failures (full_path / branch). Do not double-count.
    n_failed_level_targets = int(
        len([item for item in targets if item.get("status") != "success"])
    )
    n_failed_stages = int(len([item for item in failures if "stage" in item]))
    n_failed_total = int(n_failed_level_targets + n_failed_stages)
    summary = {
        "status": "success"
        if successful_targets and n_failed_total == 0
        else ("partial" if successful_targets else "failed"),
        "n_successful_targets": int(len(successful_targets)),
        "n_failed_level_targets": n_failed_level_targets,
        "n_failed_stages": n_failed_stages,
        "n_failed_targets": n_failed_total,
        "targets": targets,
        "hierarchy_full_path": hierarchy_full_path,
        "hierarchy_branch_diagnostics": hierarchy_branch_diagnostics,
        "failures": failures,
        "artifacts": {
            "summary_json": str(out / "networkparser_validation_summary.json"),
        },
        "method_note": (
            "This is evaluation-only. It compares saved predictions to metadata truth labels "
            "and does not rerun feature filtering, tree construction, bootstrapping, or model training. "
            "Failure counts: each level is counted once; stage failures (full-path/branch) are separate."
        ),
    }
    _write_json(summary, out / "networkparser_validation_summary.json")
    return summary

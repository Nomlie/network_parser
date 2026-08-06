#!/usr/bin/env python3
"""
Standard holdout evaluation pack for hierarchical predictions.

Writes per-level metrics, confusions, bootstrap CIs, full-path accuracy,
and optional resistance-label harmonization (susceptible ↔ Sensitive).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

RESISTANCE_LABEL_MAP = {
    "susceptible": "Sensitive",
    "sensitive": "Sensitive",
    "pre-xdr": "Pre_XDR",
    "pre_xdr": "Pre_XDR",
    "other*": "Other",
}


def _harmonize_series(s: pd.Series) -> pd.Series:
    def _one(v: Any) -> Any:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return v
        key = str(v).strip()
        low = key.lower()
        if low in RESISTANCE_LABEL_MAP:
            return RESISTANCE_LABEL_MAP[low]
        return key

    return s.map(_one)


def run_hierarchy_evaluation_pack(
    *,
    predictions_path: str | Path,
    meta_path: str | Path,
    hierarchy_labels: Sequence[str],
    output_dir: str | Path,
    sample_id_column: Optional[str] = None,
    harmonize_resistance_labels: bool = True,
    n_bootstrap: int = 500,
) -> Dict[str, Any]:
    """
    Evaluate each hierarchy level and full-path correctness.

    Uses model_evaluation.evaluate_predictions for per-level packs.
    """
    try:
        from network_parser.model_evaluation import evaluate_predictions
        from network_parser.utils import normalize_sample_id
    except ImportError:  # pragma: no cover
        from model_evaluation import evaluate_predictions  # type: ignore
        from utils import normalize_sample_id  # type: ignore

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = [str(x).strip() for x in hierarchy_labels if str(x).strip()]
    if len(labels) < 1:
        raise ValueError("hierarchy_labels required")

    # Load full prediction table for multi-level columns
    pred_df = pd.read_csv(predictions_path)
    # sample id
    sid_col = sample_id_column
    if sid_col is None:
        for c in ("sample_id", "Sample_ID", "sample", "ID"):
            if c in pred_df.columns:
                sid_col = c
                break
    if sid_col is None:
        raise ValueError("Could not find sample_id column in predictions")
    pred_df = pred_df.copy()
    pred_df["__sid"] = pred_df[sid_col].astype(str).map(normalize_sample_id)
    pred_df = pred_df[pred_df["__sid"].astype(str).str.len() > 0]

    meta = pd.read_csv(meta_path, sep=None, engine="python")
    meta_sid = None
    for c in ("sample_id", "Sample_ID", "sample", "Sample", "ID", "id"):
        if c in meta.columns:
            meta_sid = c
            break
    if meta_sid is None:
        raise ValueError("Could not find sample_id column in metadata")
    meta = meta.copy()
    meta["__sid"] = meta[meta_sid].astype(str).map(normalize_sample_id)

    # Restrict meta to prediction samples with complete hierarchy labels
    for lab in labels:
        if lab not in meta.columns:
            raise ValueError(f"Metadata missing hierarchy column {lab!r}")
    complete = meta.dropna(subset=labels).copy()
    for lab in labels:
        complete = complete[
            ~complete[lab].astype(str).str.strip().isin({"", "nan", "NA", "None", "-"})
        ]
    complete = complete[complete["__sid"].isin(set(pred_df["__sid"]))]

    if harmonize_resistance_labels:
        for lab in labels:
            if "resistance" in lab.lower() or lab.lower() in {
                "pheno",
                "amr_binary",
            }:
                complete[lab] = _harmonize_series(complete[lab])
                # also harmonize prediction columns later

    # Write restricted eval meta for provenance
    eval_meta_path = out / "holdout_evaluation_metadata.csv"
    complete.drop(columns=["__sid"], errors="ignore").to_csv(eval_meta_path, index=False)

    per_level: Dict[str, Any] = {}
    pred_cols_guess = [
        "predicted_level1",
        "predicted_level2",
        "predicted_level3",
        "predicted_level4",
        "predicted_level5",
    ]

    for i, lab in enumerate(labels):
        level_name = f"level{i+1}_{lab}"
        level_dir = out / level_name
        level_dir.mkdir(parents=True, exist_ok=True)
        pred_col = pred_cols_guess[i] if i < len(pred_cols_guess) else None
        if pred_col is None or pred_col not in pred_df.columns:
            # try match by label name
            candidates = [c for c in pred_df.columns if lab.lower() in c.lower() and "pred" in c.lower()]
            pred_col = candidates[0] if candidates else None
        if pred_col is None or pred_col not in pred_df.columns:
            per_level[level_name] = {
                "status": "skipped",
                "message": f"No prediction column for level {i+1} ({lab})",
            }
            continue

        y_true = complete.set_index("__sid")[lab]
        y_pred = pred_df.set_index("__sid")[pred_col]
        if harmonize_resistance_labels and (
            "resistance" in lab.lower() or lab.lower() in {"pheno", "amr_binary"}
        ):
            y_pred = _harmonize_series(y_pred)
            y_true = _harmonize_series(y_true)

        summary = evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            output_dir=level_dir,
            level_name=level_name,
            evaluation_role="held_out",
            n_bootstrap=n_bootstrap,
        )
        per_level[level_name] = summary

    # Full-path correctness among complete samples with all predicted levels called
    path_rows = []
    meta_idx = complete.set_index("__sid")
    pred_idx = pred_df.set_index("__sid")
    n_full = 0
    n_correct_path = 0
    for sid in meta_idx.index:
        truths = []
        preds = []
        ok = True
        for i, lab in enumerate(labels):
            pred_col = pred_cols_guess[i] if i < len(pred_cols_guess) else None
            if pred_col is None or pred_col not in pred_idx.columns or sid not in pred_idx.index:
                ok = False
                break
            t = str(meta_idx.loc[sid, lab])
            p = str(pred_idx.loc[sid, pred_col])
            if harmonize_resistance_labels:
                t = str(_harmonize_series(pd.Series([t])).iloc[0])
                p = str(_harmonize_series(pd.Series([p])).iloc[0])
            if p.lower() in {
                "unavailable",
                "nan",
                "amr_evidence_review_required",
                "low_support_review_required",
            }:
                ok = False
                break
            truths.append(t)
            preds.append(p)
        if not ok:
            continue
        n_full += 1
        correct = truths == preds
        if correct:
            n_correct_path += 1
        path_rows.append(
            {
                "sample_id": sid,
                "true_path": " | ".join(truths),
                "pred_path": " | ".join(preds),
                "full_path_correct": bool(correct),
            }
        )

    path_df = pd.DataFrame(path_rows)
    path_tsv = out / "full_path_predictions.tsv"
    path_df.to_csv(path_tsv, sep="\t", index=False)

    pack = {
        "status": "success",
        "predictions": str(predictions_path),
        "meta": str(meta_path),
        "evaluation_meta_restricted": str(eval_meta_path),
        "hierarchy_labels": labels,
        "harmonize_resistance_labels": bool(harmonize_resistance_labels),
        "n_complete_truth_in_predictions": int(len(complete)),
        "full_path": {
            "n_fully_called": int(n_full),
            "n_correct": int(n_correct_path),
            "accuracy": float(n_correct_path / n_full) if n_full else None,
            "tsv": str(path_tsv),
        },
        "per_level": {
            k: {
                "status": v.get("status"),
                "n_evaluated_samples": v.get("n_evaluated_samples"),
                "accuracy": v.get("accuracy"),
                "balanced_accuracy": v.get("balanced_accuracy"),
                "accuracy_end_to_end_all_truth": v.get("accuracy_end_to_end_all_truth"),
                "coverage_call_rate": v.get("coverage_call_rate"),
            }
            if isinstance(v, dict)
            else v
            for k, v in per_level.items()
        },
        "artifacts": {
            "evaluation_summary": str(out / "evaluation_summary.json"),
            "full_path_tsv": str(path_tsv),
        },
    }
    summary_path = out / "evaluation_summary.json"
    summary_path.write_text(json.dumps(pack, indent=2, default=str) + "\n", encoding="utf-8")
    logger.info(
        "Hierarchy evaluation pack complete | levels=%d | full_path_acc=%s | out=%s",
        len(labels),
        pack["full_path"]["accuracy"],
        out,
    )
    return pack

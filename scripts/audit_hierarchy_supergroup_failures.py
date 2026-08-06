#!/usr/bin/env python3
"""Audit hierarchy supergroup / lineage failures against query predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        required=True,
        help="query_predictions.csv from hierarchy query run",
    )
    parser.add_argument(
        "--metadata", required=True, help="Metadata CSV with true hierarchy labels"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory for audit TSV outputs"
    )
    parser.add_argument("--sample-id-col", default="sample_id")
    parser.add_argument("--meta-id-col", default="ID")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = pd.read_csv(args.predictions)
    meta = pd.read_csv(args.metadata).rename(
        columns={args.meta_id_col: args.sample_id_col}
    )
    df = preds.merge(meta, on=args.sample_id_col, how="inner", suffixes=("", "_meta"))

    df["l1_correct"] = df["Lineage_Supergroup"] == df["predicted_level1"]
    df["l2_correct"] = df["Lineage_clean"] == df["predicted_level2"]
    if "AMR_binary" in df.columns and "predicted_level3" in df.columns:
        df["true_l3"] = df["AMR_binary"].astype(str).str.lower()
        df["pred_l3"] = df["predicted_level3"].astype(str).str.lower()
        df["l3_correct"] = df["true_l3"] == df["pred_l3"]
    elif (
        "Resistance_Profile_Collapsed" in df.columns
        and "predicted_level3" in df.columns
    ):
        df["true_l3"] = df["Resistance_Profile_Collapsed"].astype(str)
        df["pred_l3"] = df["predicted_level3"].astype(str)
        df["l3_correct"] = df["true_l3"] == df["pred_l3"]

    l1_fail = df[~df["l1_correct"]].copy()
    l1_fail.to_csv(out_dir / "supergroup_l1_failures.tsv", sep="\t", index=False)

    summary = {
        "n_query": len(df),
        "l1_accuracy": float(df["l1_correct"].mean()),
        "l2_accuracy": float(df["l2_correct"].mean()),
        "l1_failures": int((~df["l1_correct"]).sum()),
        "l2_failures": int((~df["l2_correct"]).sum()),
    }
    if "l3_correct" in df.columns:
        summary["l3_accuracy"] = float(df["l3_correct"].mean())
        summary["l3_failures"] = int((~df["l3_correct"]).sum())

    crosswalk = (
        l1_fail.groupby(
            ["Lineage_clean", "Lineage_Supergroup", "predicted_level1"], dropna=False
        )
        .size()
        .reset_index(name="n_samples")
        .sort_values("n_samples", ascending=False)
    )
    crosswalk.to_csv(
        out_dir / "supergroup_l1_failure_crosswalk.tsv", sep="\t", index=False
    )

    pd.DataFrame([summary]).to_csv(
        out_dir / "supergroup_audit_summary.tsv", sep="\t", index=False
    )
    print(f"Wrote audits to {out_dir}")
    print(summary)


if __name__ == "__main__":
    main()

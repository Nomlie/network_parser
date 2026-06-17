#!/usr/bin/env python3
"""
add_test_hierarchy_column.py

Reusable helper to add a coarse parent label for testing multi-level
NetworkParser hierarchy training.

Purpose
-------
When the first hierarchy level has weak genomic signal, this script can create a
coarse, source-label-derived parent column so you can test whether recursive
hierarchy training/querying works mechanically.

Example
-------
python add_test_hierarchy_column.py \
    --input metadata.csv \
    --source-col Lineage_clean \
    --output-col Lineage_Supergroup_Test \
    --strategy frequency_bins

Then use:
    --hierarchy_labels Lineage_Supergroup_Test Lineage_clean AMR_binary

Notes
-----
This creates a TESTING column. It should not be interpreted as a validated
biological taxonomy unless you define and justify the grouping rules.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd


DEFAULT_MISSING_TOKENS = {
    "",
    "-",
    "NA",
    "N/A",
    "None",
    "none",
    "nan",
    "NaN",
    "NULL",
    "null",
}


def read_table(path: Path, sep: Optional[str] = None) -> pd.DataFrame:
    """Read CSV/TSV/unknown-delimited metadata."""
    if not path.exists():
        raise FileNotFoundError(f"Input metadata file not found: {path}")

    suffix = "".join(path.suffixes).lower()
    if sep is not None:
        return pd.read_csv(path, sep=sep)
    if suffix.endswith(".tsv") or suffix.endswith(".txt"):
        return pd.read_csv(path, sep="\t")
    if suffix.endswith(".csv"):
        return pd.read_csv(path)

    return pd.read_csv(path, sep=None, engine="python")


def write_table(df: pd.DataFrame, path: Path) -> None:
    """Write CSV or TSV based on output suffix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = "".join(path.suffixes).lower()
    if suffix.endswith(".tsv") or suffix.endswith(".txt"):
        df.to_csv(path, sep="\t", index=False)
    else:
        df.to_csv(path, index=False)


def clean_label_series(
    values: pd.Series,
    *,
    missing_tokens: Iterable[str],
    missing_label: str,
    lowercase: bool = False,
    replace_hyphen: bool = True,
) -> pd.Series:
    """Conservative label normalization for grouping only."""
    clean = values.astype(str).str.strip()
    clean = clean.replace({token: pd.NA for token in missing_tokens})
    if replace_hyphen:
        clean = clean.str.replace("-", "_", regex=False)
    if lowercase:
        clean = clean.str.lower()
    clean = clean.fillna(missing_label)
    clean = clean.replace({"": missing_label})
    return clean.astype(str)


def load_manual_map(path: Path) -> Dict[str, str]:
    """
    Load a manual source->parent mapping from JSON or two-column CSV/TSV.

    JSON example:
        {"source_label_a": "parent_a", "source_label_b": "parent_b"}

    Table example:
        source,parent
        source_label_a,parent_a
        source_label_b,parent_b
    """
    if not path.exists():
        raise FileNotFoundError(f"Mapping file not found: {path}")

    suffix = "".join(path.suffixes).lower()
    if suffix.endswith(".json"):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("JSON mapping file must contain an object/dictionary.")
        return {str(k).strip(): str(v).strip() for k, v in payload.items()}

    mapping_df = read_table(path)
    if mapping_df.shape[1] < 2:
        raise ValueError("Mapping table must contain at least two columns: source,parent")

    source_col = "source" if "source" in mapping_df.columns else mapping_df.columns[0]
    parent_col = "parent" if "parent" in mapping_df.columns else mapping_df.columns[1]

    return {
        str(row[source_col]).strip(): str(row[parent_col]).strip()
        for _, row in mapping_df.iterrows()
        if str(row[source_col]).strip()
    }


def make_frequency_bins(
    labels: pd.Series,
    *,
    missing_label: str,
    high_quantile: float,
    mid_quantile: float,
    prefix: str,
) -> pd.Series:
    """
    Group labels into coarse parents based on source-label frequency.

    This is useful for testing hierarchy mechanics because it creates parent
    classes with different support levels without hard-coding label names.
    """
    if not 0 <= mid_quantile <= 1 or not 0 <= high_quantile <= 1:
        raise ValueError("Quantiles must be between 0 and 1.")
    if mid_quantile > high_quantile:
        raise ValueError("--mid-quantile must be <= --high-quantile.")

    counts = labels.value_counts(dropna=False)
    non_missing_counts = counts.drop(labels=[missing_label], errors="ignore")

    if non_missing_counts.empty:
        return pd.Series([f"{prefix}_unknown_parent"] * len(labels), index=labels.index)

    high_threshold = float(non_missing_counts.quantile(high_quantile))
    mid_threshold = float(non_missing_counts.quantile(mid_quantile))

    def assign(value: str) -> str:
        if value == missing_label:
            return f"{prefix}_unknown_parent"
        count = int(counts.get(value, 0))
        if count >= high_threshold:
            return f"{prefix}_common_parent"
        if count >= mid_threshold:
            return f"{prefix}_intermediate_parent"
        return f"{prefix}_low_support_parent"

    return labels.map(assign).astype(str)


def make_prefix_groups(
    labels: pd.Series,
    *,
    missing_label: str,
    prefix_parts: int,
    prefix: str,
) -> pd.Series:
    """
    Group labels by their first token(s) after splitting on common separators.

    Example conceptually:
        source label: group_a_subtype_1
        parent with prefix_parts=1: test_group_a_parent
    """
    if prefix_parts < 1:
        raise ValueError("--prefix-parts must be >= 1.")

    def assign(value: str) -> str:
        if value == missing_label:
            return f"{prefix}_unknown_parent"
        parts = [p for p in re.split(r"[\s_./:|]+", value.strip()) if p]
        if not parts:
            return f"{prefix}_unknown_parent"
        parent_core = "_".join(parts[:prefix_parts])
        parent_core = re.sub(r"[^A-Za-z0-9_]+", "_", parent_core).strip("_")
        return f"{prefix}_{parent_core}_parent" if parent_core else f"{prefix}_unknown_parent"

    return labels.map(assign).astype(str)


def make_manual_groups(
    labels: pd.Series,
    *,
    mapping: Dict[str, str],
    missing_label: str,
    unmapped_parent: str,
) -> pd.Series:
    """Group labels using a user-provided mapping."""
    def assign(value: str) -> str:
        if value == missing_label:
            return unmapped_parent
        return mapping.get(value, unmapped_parent)

    return labels.map(assign).astype(str)


def default_output_path(input_path: Path, output_col: str) -> Path:
    """Build a safe default output filename."""
    safe_col = re.sub(r"[^A-Za-z0-9_.-]+", "_", output_col).strip("_") or "test_hierarchy"
    suffix = "".join(input_path.suffixes) or ".csv"
    stem = input_path.name
    for s in input_path.suffixes:
        stem = stem[: -len(s)]
    return input_path.with_name(f"{stem}_with_{safe_col}{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add a reusable coarse parent label column for testing NetworkParser "
            "multi-level hierarchy training."
        )
    )
    parser.add_argument("--input", required=True, help="Input metadata CSV/TSV file.")
    parser.add_argument("--output", default=None, help="Output metadata CSV/TSV file. Default: input_with_<output_col>.")
    parser.add_argument("--sep", default=None, help="Optional input separator override, e.g. ',' or '\\t'.")

    parser.add_argument("--source-col", default="Lineage_clean", help="Column used to derive the test parent label.")
    parser.add_argument("--output-col", default="Lineage_Supergroup_Test", help="Name of the new parent label column.")

    parser.add_argument(
        "--strategy",
        choices=["frequency_bins", "prefix", "manual_map"],
        default="frequency_bins",
        help=(
            "Grouping strategy: frequency_bins = support-based test parents; "
            "prefix = first token(s) of source label; manual_map = use mapping file."
        ),
    )
    parser.add_argument("--high-quantile", type=float, default=0.75, help="High-support quantile for frequency_bins.")
    parser.add_argument("--mid-quantile", type=float, default=0.40, help="Middle-support quantile for frequency_bins.")
    parser.add_argument("--prefix-parts", type=int, default=1, help="Number of leading tokens to use for prefix strategy.")
    parser.add_argument("--mapping-file", default=None, help="JSON/CSV/TSV source->parent mapping for manual_map strategy.")

    parser.add_argument("--test-prefix", default="test", help="Prefix added to generated parent labels.")
    parser.add_argument("--missing-label", default="unknown", help="Internal label for missing source values.")
    parser.add_argument("--unmapped-parent", default="test_unmapped_parent", help="Parent label for unmapped manual_map values.")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase source labels before grouping.")
    parser.add_argument("--keep-hyphen", action="store_true", help="Do not replace '-' with '_' in source labels.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output column.")
    parser.add_argument("--summary", default=None, help="Optional JSON summary path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else default_output_path(input_path, args.output_col)

    df = read_table(input_path, sep=args.sep)

    if args.source_col not in df.columns:
        raise ValueError(
            f"source column '{args.source_col}' not found. Available columns: {list(df.columns)}"
        )

    if args.output_col in df.columns and not args.overwrite:
        raise ValueError(
            f"output column '{args.output_col}' already exists. Use --overwrite to replace it."
        )

    source = clean_label_series(
        df[args.source_col],
        missing_tokens=DEFAULT_MISSING_TOKENS,
        missing_label=args.missing_label,
        lowercase=bool(args.lowercase),
        replace_hyphen=not bool(args.keep_hyphen),
    )

    if args.strategy == "frequency_bins":
        parent = make_frequency_bins(
            source,
            missing_label=args.missing_label,
            high_quantile=args.high_quantile,
            mid_quantile=args.mid_quantile,
            prefix=args.test_prefix,
        )
    elif args.strategy == "prefix":
        parent = make_prefix_groups(
            source,
            missing_label=args.missing_label,
            prefix_parts=args.prefix_parts,
            prefix=args.test_prefix,
        )
    elif args.strategy == "manual_map":
        if not args.mapping_file:
            raise ValueError("--mapping-file is required when --strategy manual_map is used.")
        mapping = load_manual_map(Path(args.mapping_file))
        parent = make_manual_groups(
            source,
            mapping=mapping,
            missing_label=args.missing_label,
            unmapped_parent=args.unmapped_parent,
        )
    else:
        raise ValueError(f"Unsupported strategy: {args.strategy}")

    df[args.output_col] = parent
    write_table(df, output_path)

    summary = {
        "status": "success",
        "input": str(input_path),
        "output": str(output_path),
        "source_col": args.source_col,
        "output_col": args.output_col,
        "strategy": args.strategy,
        "rows": int(df.shape[0]),
        "source_unique_labels": int(source.nunique(dropna=False)),
        "parent_unique_labels": int(parent.nunique(dropna=False)),
        "parent_distribution": {str(k): int(v) for k, v in parent.value_counts(dropna=False).to_dict().items()},
        "recommended_hierarchy_example": f"--hierarchy_labels {args.output_col} {args.source_col} <terminal_label>",
        "note": "This column is intended for testing hierarchy mechanics, not as a validated biological label unless manually defined and justified.",
    }

    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(f"Wrote metadata: {output_path}")
    print(f"Wrote summary:  {summary_path}")
    print(f"Added column:   {args.output_col}")
    print(f"Use example:    --hierarchy_labels {args.output_col} {args.source_col} <terminal_label>")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
extract_subset.py

Create a random or stratified subset of VCF samples based on metadata,
then split the selected subset into train/test sets.

Default:
    selected subset -> 80% train + 20% test

Important:
    Only samples with matching VCF files AND valid metadata labels are selected.

This version normalizes metadata label values before stratification/splitting.
Example:
    lineage BOV-AFRI -> lineage BOV_AFRI
    lineage BOV_AFRI -> lineage BOV_AFRI
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import random
import shutil
import re
from typing import Dict, List, Optional, Tuple


# -------------------------
# Setup logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


MISSING_LABEL_TOKENS = {
    "",
    "-",
    ".",
    "NA",
    "N/A",
    "None",
    "none",
    "nan",
    "NaN",
    "NULL",
    "null",
    "unknown",
    "Unknown",
    "UNK",
    "missing",
    "Missing",
}


# -------------------------
# Helper functions
# -------------------------
def example_text() -> str:
    return """
Examples
--------
# Stratified by lineage, then split 80/20
python3 extract_subset.py \\
    --vcf-dir AFRO_TB_ANNOTATION_VCF \\
    --meta-file AFRO_dataset_meta.csv \\
    --stratify-by lineage \\
    --n-samples 100 \\
    --output-dir subset_lineage_100 \\
    --random-seed 42

# Require valid hierarchy labels too
python3 extract_subset.py \\
    --vcf-dir AFRO_TB_ANNOTATION_VCF \\
    --meta-file AFRO_dataset_meta.csv \\
    --stratify-by lineage \\
    --required-label-columns Country Lineage Pheno \\
    --n-samples 100 \\
    --output-dir subset_valid_hierarchy_100 \\
    --random-seed 42

# Random subset, then 80/20 split
python3 extract_subset.py \\
    --vcf-dir AFRO_TB_ANNOTATION_VCF \\
    --meta-file AFRO_dataset_meta.csv \\
    --stratify-by none \\
    --n-samples 100 \\
    --output-dir subset_random_100 \\
    --random-seed 42

# Custom split, e.g. 70/30
python3 extract_subset.py \\
    --vcf-dir AFRO_TB_ANNOTATION_VCF \\
    --meta-file AFRO_dataset_meta.csv \\
    --stratify-by phenotype \\
    --n-samples 100 \\
    --train-fraction 0.7 \\
    --output-dir subset_pheno_70_30
""".strip()


def print_examples():
    print(example_text())


def is_valid_metadata_label(value) -> bool:
    """Return True if a metadata label is usable."""
    if pd.isna(value):
        return False

    text = str(value).strip()
    if text in MISSING_LABEL_TOKENS:
        return False

    return True


def normalize_metadata_label(value) -> str:
    """
    Normalize metadata labels used for stratification.

    This prevents equivalent labels with small formatting differences from
    being treated as separate groups.

    Examples:
        lineage BOV-AFRI -> lineage BOV_AFRI
        lineage BOV_AFRI -> lineage BOV_AFRI
        lineage BOV - AFRI -> lineage BOV_AFRI
    """
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text in MISSING_LABEL_TOKENS:
        return ""

    # Normalize Unicode dash variants to normal hyphen first.
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")

    # Normalize separator-only differences.
    # Handles "BOV-AFRI", "BOV_AFRI", "BOV - AFRI", "BOV _ AFRI".
    text = re.sub(r"\s*[-_]\s*", "_", text)

    # Collapse repeated whitespace.
    text = " ".join(text.split())

    # Collapse repeated underscores.
    text = re.sub(r"_+", "_", text)

    return text.strip()


def normalize_required_metadata_columns(
    meta_df: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:
    """
    Normalize label values in required metadata columns before sampling/splitting.

    This is important because pandas groupby() treats raw strings as distinct,
    so labels like BOV-AFRI and BOV_AFRI would otherwise become separate groups.
    """
    df = meta_df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        before_n = df[col].nunique(dropna=False)
        df[col] = df[col].map(normalize_metadata_label)
        after_n = df[col].nunique(dropna=False)

        logger.info(
            "Normalized metadata labels in '%s': unique_before=%d | unique_after=%d",
            col,
            before_n,
            after_n,
        )

    return df


def normalize_sample_id_from_vcf(vcf_path: Path, delimiter: str = "_") -> str:
    """Extract canonical sample ID from VCF filename."""
    name = vcf_path.name

    if name.endswith(".vcf.gz"):
        name = name[:-7]
    elif name.endswith(".vcf"):
        name = name[:-4]

    if delimiter:
        return name.split(delimiter)[0]

    return name


def load_metadata(meta_file: Path) -> pd.DataFrame:
    """Load metadata CSV with sample IDs in the first column."""
    df = pd.read_csv(meta_file)

    if df.empty:
        raise ValueError(f"Metadata file is empty: {meta_file}")

    first_col = df.columns[0]
    df[first_col] = df[first_col].astype(str).str.strip()
    df = df.set_index(first_col, drop=True)
    df.index = df.index.astype(str)

    logger.info("Loaded metadata for %d samples.", len(df))
    return df


def scan_vcf_dir(vcf_dir: Path, delimiter: str = "_") -> Dict[str, Path]:
    """Map canonical sample_id -> full VCF path."""
    vcf_files = list(vcf_dir.glob("*.vcf.gz")) + list(vcf_dir.glob("*.vcf"))
    logger.info("Found %d VCF files in %s", len(vcf_files), vcf_dir)

    vcf_map: Dict[str, Path] = {}

    for vcf in sorted(vcf_files):
        sid = normalize_sample_id_from_vcf(vcf, delimiter=delimiter)

        if sid in vcf_map:
            logger.warning("Duplicate sample ID '%s' found - keeping first", sid)
            continue

        vcf_map[sid] = vcf

    return vcf_map


def resolve_column_case_insensitive(meta_df: pd.DataFrame, column_name: str) -> str:
    """Resolve a metadata column name case-insensitively."""
    meta_columns_lower = {col.lower(): col for col in meta_df.columns}
    key = str(column_name).strip().lower()

    if key not in meta_columns_lower:
        raise ValueError(
            f"Metadata column '{column_name}' not found. "
            f"Available columns: {list(meta_df.columns)}"
        )

    return meta_columns_lower[key]


def resolve_stratify_column(meta_df: pd.DataFrame, stratify_by: Optional[str]) -> Optional[str]:
    """Resolve stratification column case-insensitively."""
    if stratify_by is None:
        return None

    text = str(stratify_by).strip()
    if text.lower() in {"", "none", "no", "false"}:
        return None

    return resolve_column_case_insensitive(meta_df, text)


def resolve_required_label_columns(
    meta_df: pd.DataFrame,
    stratify_by: Optional[str],
    required_label_columns: Optional[List[str]],
) -> Tuple[Optional[str], List[str]]:
    """
    Resolve required metadata label columns.

    The stratification column is always required if supplied.
    Additional columns can be supplied using --required-label-columns.
    """
    resolved_required: List[str] = []

    real_stratify_col = resolve_stratify_column(meta_df, stratify_by)
    if real_stratify_col:
        resolved_required.append(real_stratify_col)

    for col in required_label_columns or []:
        text = str(col).strip()
        if text.lower() in {"", "none", "no", "false"}:
            continue

        real_col = resolve_column_case_insensitive(meta_df, text)

        if real_col not in resolved_required:
            resolved_required.append(real_col)

    return real_stratify_col, resolved_required


def filter_valid_metadata_labels(
    meta_df: pd.DataFrame,
    required_label_columns: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep only samples with valid metadata labels in required columns.

    Returns:
        valid_meta_df, invalid_meta_df
    """
    if not required_label_columns:
        return meta_df.copy(), meta_df.iloc[0:0].copy()

    valid_mask = pd.Series(True, index=meta_df.index)

    for col in required_label_columns:
        valid_mask &= meta_df[col].map(is_valid_metadata_label)

    valid_df = meta_df.loc[valid_mask].copy()
    invalid_df = meta_df.loc[~valid_mask].copy()

    logger.info(
        "Metadata label filtering: kept %d/%d samples with valid labels in: %s",
        len(valid_df),
        len(meta_df),
        ", ".join(required_label_columns),
    )

    if len(invalid_df) > 0:
        logger.warning(
            "Excluded %d samples because required metadata labels were missing/invalid.",
            len(invalid_df),
        )

    return valid_df, invalid_df


def random_sample(
    sample_ids: List[str],
    n_total: int,
    random_seed: Optional[int] = None,
) -> List[str]:
    """Randomly select up to n_total sample IDs."""
    rng = random.Random(random_seed)
    ids = list(sample_ids)
    rng.shuffle(ids)
    return ids[: min(n_total, len(ids))]


def stratified_sample(
    meta_df: pd.DataFrame,
    stratify_col: str,
    n_total: int,
    random_seed: Optional[int] = None,
) -> List[str]:
    """
    Select a roughly balanced subset across groups in stratify_col.

    If some groups are too small, the script fills the remaining target size
    from the remaining eligible samples.
    """
    rng = random.Random(random_seed)

    groups = meta_df.groupby(stratify_col, dropna=False)
    group_sizes = groups.size()

    logger.info(
        "Stratifying by '%s' — found %d groups:\n%s",
        stratify_col,
        len(group_sizes),
        group_sizes,
    )

    n_groups = len(group_sizes)
    if n_groups == 0:
        raise ValueError("No groups found in stratification column.")

    base_per_group = n_total // n_groups
    remainder = n_total % n_groups

    selected: List[str] = []

    for group_name, group_df in groups:
        n_wanted = base_per_group + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1

        available_ids = list(group_df.index.astype(str))
        rng.shuffle(available_ids)

        n_take = min(n_wanted, len(available_ids))

        if n_take == 0:
            logger.warning("Group '%s' has 0 samples — skipped", group_name)
            continue

        selected_ids = available_ids[:n_take]
        selected.extend(selected_ids)

        logger.info(
            "Group '%s': requested %d, took %d/%d",
            group_name,
            n_wanted,
            n_take,
            len(available_ids),
        )

    # Fill shortfall from remaining eligible samples.
    if len(selected) < min(n_total, len(meta_df)):
        selected_set = set(selected)
        remaining = [
            sid
            for sid in meta_df.index.astype(str).tolist()
            if sid not in selected_set
        ]
        rng.shuffle(remaining)

        n_needed = min(n_total, len(meta_df)) - len(selected)
        selected.extend(remaining[:n_needed])

        logger.info(
            "Filled stratified subset shortfall with %d additional eligible samples.",
            min(n_needed, len(remaining)),
        )

    if len(selected) < n_total:
        logger.warning(
            "Could only select %d/%d samples due to limited eligible samples.",
            len(selected),
            n_total,
        )

    return selected


def select_samples(
    meta_df: pd.DataFrame,
    vcf_map: Dict[str, Path],
    n: int,
    stratify_by: Optional[str] = None,
    random_seed: Optional[int] = None,
    required_label_columns: Optional[List[str]] = None,
) -> Tuple[List[str], Optional[str], List[str], pd.DataFrame, pd.DataFrame]:
    """
    Select samples only from VCFs with valid metadata labels.

    Returns:
        selected_ids,
        real_stratify_col,
        resolved_required_columns,
        invalid_metadata_df,
        valid_normalized_metadata_df
    """
    overlap_ids = sorted(set(meta_df.index.astype(str)).intersection(vcf_map.keys()))

    logger.info(
        "Found %d overlapping samples between metadata and VCFs.",
        len(overlap_ids),
    )

    if len(overlap_ids) == 0:
        raise ValueError("No overlapping samples between metadata and VCFs.")

    meta_overlap = meta_df.loc[overlap_ids].copy()

    real_stratify_col, resolved_required_columns = resolve_required_label_columns(
        meta_df=meta_overlap,
        stratify_by=stratify_by,
        required_label_columns=required_label_columns,
    )

    # Critical fix:
    # Normalize metadata labels before validity filtering, sampling, and splitting.
    meta_overlap = normalize_required_metadata_columns(
        meta_df=meta_overlap,
        columns=resolved_required_columns,
    )

    meta_valid, meta_invalid = filter_valid_metadata_labels(
        meta_df=meta_overlap,
        required_label_columns=resolved_required_columns,
    )

    if meta_valid.empty:
        raise ValueError(
            "No samples remain after filtering for valid metadata labels. "
            f"Required columns: {resolved_required_columns}"
        )

    valid_overlap_ids = sorted(meta_valid.index.astype(str).tolist())

    logger.info(
        "Eligible VCF samples after metadata-label filtering: %d",
        len(valid_overlap_ids),
    )

    if real_stratify_col:
        logger.info("Performing balanced stratified sampling by '%s'", real_stratify_col)
        selected_ids = stratified_sample(
            meta_df=meta_valid,
            stratify_col=real_stratify_col,
            n_total=n,
            random_seed=random_seed,
        )
    else:
        logger.info("Performing simple random sampling from metadata-valid samples.")
        selected_ids = random_sample(
            sample_ids=valid_overlap_ids,
            n_total=n,
            random_seed=random_seed,
        )

    selected_ids = [sid for sid in selected_ids if sid in vcf_map]

    logger.info("Selected %d samples with valid metadata labels.", len(selected_ids))

    return (
        selected_ids,
        real_stratify_col,
        resolved_required_columns,
        meta_invalid,
        meta_valid,
    )


def split_train_test_random(
    selected_ids: List[str],
    train_fraction: float,
    random_seed: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """Random train/test split."""
    rng = random.Random(random_seed)
    ids = list(selected_ids)
    rng.shuffle(ids)

    n_train = int(round(len(ids) * train_fraction))
    n_train = max(1, min(n_train, len(ids) - 1)) if len(ids) > 1 else len(ids)

    train_ids = ids[:n_train]
    test_ids = ids[n_train:]

    return train_ids, test_ids


def split_train_test_stratified(
    selected_ids: List[str],
    meta_df: pd.DataFrame,
    stratify_col: str,
    train_fraction: float,
    random_seed: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """
    Stratified train/test split.

    Groups with at least 2 samples get at least 1 train and 1 test sample.
    Singleton groups go to train only because they cannot be split.
    """
    rng = random.Random(random_seed)

    selected_meta = meta_df.loc[selected_ids].copy()
    selected_meta["_sample_id"] = selected_meta.index.astype(str)

    train_ids: List[str] = []
    test_ids: List[str] = []

    for group_name, group_df in selected_meta.groupby(stratify_col, dropna=False):
        group_ids = list(group_df["_sample_id"].astype(str))
        rng.shuffle(group_ids)

        n_group = len(group_ids)

        if n_group == 1:
            train_ids.extend(group_ids)
            logger.warning(
                "Group '%s' has only 1 selected sample; assigning it to train only.",
                group_name,
            )
            continue

        n_train = int(round(n_group * train_fraction))
        n_train = max(1, min(n_train, n_group - 1))

        train_ids.extend(group_ids[:n_train])
        test_ids.extend(group_ids[n_train:])

        logger.info(
            "Split group '%s': train=%d, test=%d, total=%d",
            group_name,
            n_train,
            n_group - n_train,
            n_group,
        )

    rng.shuffle(train_ids)
    rng.shuffle(test_ids)

    return train_ids, test_ids


def split_train_test(
    selected_ids: List[str],
    meta_df: pd.DataFrame,
    stratify_col: Optional[str],
    train_fraction: float,
    random_seed: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """Split selected sample IDs into train and test IDs."""
    if not 0 < train_fraction < 1:
        raise ValueError("--train-fraction must be between 0 and 1.")

    if len(selected_ids) == 0:
        raise ValueError("No selected samples available for train/test split.")

    if len(selected_ids) == 1:
        logger.warning("Only one sample selected; assigning it to train set.")
        return selected_ids, []

    if stratify_col:
        logger.info("Performing stratified train/test split by '%s'", stratify_col)
        return split_train_test_stratified(
            selected_ids=selected_ids,
            meta_df=meta_df,
            stratify_col=stratify_col,
            train_fraction=train_fraction,
            random_seed=random_seed,
        )

    logger.info("Performing random train/test split.")
    return split_train_test_random(
        selected_ids=selected_ids,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )


def copy_vcfs(
    sample_ids: List[str],
    vcf_map: Dict[str, Path],
    output_dir: Path,
) -> None:
    """Copy selected VCFs into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for sid in sample_ids:
        vcf_path = vcf_map[sid]

        if vcf_path.name.endswith(".vcf.gz"):
            suffix = ".vcf.gz"
        elif vcf_path.name.endswith(".vcf"):
            suffix = ".vcf"
        else:
            suffix = vcf_path.suffix

        dest = output_dir / f"{sid}{suffix}"
        shutil.copy2(vcf_path, dest)

    logger.info("Copied %d VCFs to %s", len(sample_ids), output_dir)


def write_sample_manifest(
    sample_ids: List[str],
    meta_df: pd.DataFrame,
    split_name: str,
    output_path: Path,
) -> None:
    """Write selected sample IDs plus metadata for a split."""
    if not sample_ids:
        df = pd.DataFrame(columns=["sample_id", "split"])
    else:
        df = meta_df.loc[sample_ids].copy()
        df.insert(0, "sample_id", df.index.astype(str))
        df.insert(1, "split", split_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Wrote %s manifest: %s", split_name, output_path)


def write_split_summary(
    train_ids: List[str],
    test_ids: List[str],
    meta_df: pd.DataFrame,
    stratify_col: Optional[str],
    output_path: Path,
) -> None:
    """Write compact train/test summary."""
    rows = []

    if stratify_col:
        for split_name, ids in [("train", train_ids), ("test", test_ids)]:
            if not ids:
                continue

            counts = meta_df.loc[ids, stratify_col].value_counts(dropna=False)
            for label, count in counts.items():
                rows.append(
                    {
                        "split": split_name,
                        "stratify_column": stratify_col,
                        "label": label,
                        "n_samples": int(count),
                    }
                )
    else:
        rows = [
            {
                "split": "train",
                "stratify_column": "",
                "label": "all",
                "n_samples": len(train_ids),
            },
            {
                "split": "test",
                "stratify_column": "",
                "label": "all",
                "n_samples": len(test_ids),
            },
        ]

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Wrote split summary: %s", output_path)


def write_invalid_metadata_report(
    invalid_metadata_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write samples excluded because of invalid metadata labels."""
    if invalid_metadata_df.empty:
        return

    df = invalid_metadata_df.copy()
    df.insert(0, "sample_id", df.index.astype(str))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Wrote excluded invalid-metadata samples: %s", output_path)


def write_run_summary(
    output_path: Path,
    selected_ids: List[str],
    train_ids: List[str],
    test_ids: List[str],
    required_label_columns: List[str],
    stratify_col: Optional[str],
    train_fraction: float,
) -> None:
    """Write one-row summary of the subsetting run."""
    row = {
        "selected_total": len(selected_ids),
        "train_samples": len(train_ids),
        "test_samples": len(test_ids),
        "train_fraction": train_fraction,
        "stratify_column": stratify_col or "",
        "required_label_columns": ";".join(required_label_columns),
        "label_normalization": "hyphen/dash/underscore variants normalized to underscore",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(output_path, index=False)
    logger.info("Wrote run summary: %s", output_path)


# -------------------------
# Main
# -------------------------
def main(args):
    meta_file = Path(args.meta_file)
    vcf_dir = Path(args.vcf_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    meta_df = load_metadata(meta_file)
    vcf_map = scan_vcf_dir(vcf_dir, delimiter=args.vcf_sample_delimiter)

    (
        selected_ids,
        real_stratify_col,
        required_label_columns,
        invalid_metadata_df,
        meta_valid_df,
    ) = select_samples(
        meta_df=meta_df,
        vcf_map=vcf_map,
        n=args.n_samples,
        stratify_by=args.stratify_by,
        random_seed=args.random_seed,
        required_label_columns=args.required_label_columns,
    )

    write_invalid_metadata_report(
        invalid_metadata_df=invalid_metadata_df,
        output_path=output_dir / "excluded_invalid_metadata_labels.csv",
    )

    train_ids, test_ids = split_train_test(
        selected_ids=selected_ids,
        meta_df=meta_valid_df,
        stratify_col=real_stratify_col,
        train_fraction=args.train_fraction,
        random_seed=args.random_seed,
    )

    train_dir = output_dir / "train"
    test_dir = output_dir / "test"

    copy_vcfs(train_ids, vcf_map, train_dir)
    copy_vcfs(test_ids, vcf_map, test_dir)

    write_sample_manifest(
        sample_ids=train_ids,
        meta_df=meta_valid_df,
        split_name="train",
        output_path=output_dir / "train_samples.csv",
    )

    write_sample_manifest(
        sample_ids=test_ids,
        meta_df=meta_valid_df,
        split_name="test",
        output_path=output_dir / "test_samples.csv",
    )

    write_split_summary(
        train_ids=train_ids,
        test_ids=test_ids,
        meta_df=meta_valid_df,
        stratify_col=real_stratify_col,
        output_path=output_dir / "subset_split_summary.csv",
    )

    write_run_summary(
        output_path=output_dir / "subset_run_summary.csv",
        selected_ids=selected_ids,
        train_ids=train_ids,
        test_ids=test_ids,
        required_label_columns=required_label_columns,
        stratify_col=real_stratify_col,
        train_fraction=args.train_fraction,
    )

    logger.info(
        "Subset creation completed. total=%d | train=%d | test=%d | output=%s",
        len(selected_ids),
        len(train_ids),
        len(test_ids),
        output_dir,
    )


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Create a reproducible, optionally stratified VCF subset and "
            "split it into train/test directories."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=example_text(),
    )

    parser.add_argument(
        "--vcf-dir",
        required=True,
        help="Directory containing VCF files (*.vcf.gz or *.vcf).",
    )

    parser.add_argument(
        "--meta-file",
        required=True,
        help="Metadata CSV. First column must contain sample IDs.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory. Creates train/ and test/ inside it.",
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Target total number of selected samples before train/test split.",
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of selected samples assigned to training. Default: 0.8.",
    )

    parser.add_argument(
        "--stratify-by",
        default=None,
        help=(
            "Metadata column to stratify by, e.g. lineage or phenotype. "
            "Use 'none' for simple random sampling."
        ),
    )

    parser.add_argument(
        "--required-label-columns",
        nargs="*",
        default=None,
        help=(
            "Additional metadata columns that must contain valid labels before "
            "a VCF can be selected. Example: --required-label-columns Country Lineage Pheno"
        ),
    )

    parser.add_argument(
        "--vcf-sample-delimiter",
        default="_",
        help="Delimiter in VCF filenames separating sampleID from suffix. Default: '_'.",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Set random seed for reproducibility.",
    )

    args = parser.parse_args()

    if args.random_seed is not None:
        random.seed(args.random_seed)
        logger.info("Random seed set to %d for reproducibility.", args.random_seed)

    main(args)
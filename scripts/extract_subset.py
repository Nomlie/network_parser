#!/usr/bin/env python3
"""
extract_subset.py

Create a balanced or random subset of samples from VCFs based on metadata.
Supports stratification by 'lineage' or 'phenotype'.
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import random
import shutil

# -------------------------
# Setup logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def print_examples():
    examples = """
Examples
--------
# Stratified by lineage (balanced across lineage groups)
python3 extract_subset.py \
    --vcf-dir /mnt/lustre/users/nmfuphi/AFRO_TB/AFRO_TB_ANNOTATION_VCF \
    --meta-file /mnt/lustre/users/nmfuphi/AFRO_TB/AFRO_dataset_meta.csv \
    --stratify-by Lineage \
    --n-samples 5000 \
    --output-dir /mnt/lustre/users/nmfuphi/AFRO_TB/AFRO_TB_5000_LineageBalanced

# Stratified by phenotype
python3 extract_subset.py \\
    --vcf-dir AFRO_TB_ANNOTATION_VCF \\
    --meta-file AFRO_dataset_meta.csv \\
    --stratify-by phenotype \\
    --n-samples 100 \\
    --output-dir test_subset_pheno

    
# Simple random (no stratification)
python3 extract_subset.py \\
    --vcf-dir AFRO_TB_ANNOTATION_VCF \\
    --meta-file AFRO_dataset_meta.csv \\
    --n-samples 100 \\
    --output-dir test_random_100

# Reproducible with seed
python3 extract_subset.py \\
    --n-samples 100 \\
    --stratify-by lineage \\
    --random-seed 42
"""
    print(examples.strip())

# -------------------------
# Core functions
# -------------------------
def normalize_sample_id(vcf_path: Path, delimiter="_") -> str:
    """Extract canonical sample ID from VCF filename."""
    name = vcf_path.name
    if name.endswith(".vcf.gz"):
        name = name[:-7]
    elif name.endswith(".vcf"):
        name = name[:-4]
    return name.split(delimiter)[0]

def load_metadata(meta_file: Path) -> pd.DataFrame:
    """Load metadata CSV with sample IDs in first column."""
    df = pd.read_csv(meta_file)
    df.iloc[:, 0] = df.iloc[:, 0].astype(str)  # ensure string IDs
    df = df.set_index(df.columns[0])           # set first column as index
    logger.info(f"Loaded metadata for {len(df)} samples.")
    return df

def scan_vcf_dir(vcf_dir: Path, delimiter="_") -> dict:
    """Map canonical sample_id → full VCF path."""
    vcf_files = list(vcf_dir.glob("*.vcf.gz")) + list(vcf_dir.glob("*.vcf"))
    logger.info(f"Found {len(vcf_files)} VCF files in {vcf_dir}")
    
    vcf_map = {}
    for vcf in vcf_files:
        sid = normalize_sample_id(vcf, delimiter)
        if sid in vcf_map:
            logger.warning(f"Duplicate sample ID '{sid}' found - keeping first")
        else:
            vcf_map[sid] = vcf
    return vcf_map

def stratified_sample(meta_df: pd.DataFrame, stratify_col: str, n_total: int) -> list:
    stratify_col = stratify_col.lower() if stratify_col else None
    meta_columns_lower = {col.lower(): col for col in meta_df.columns}
    
    if stratify_col and stratify_col not in meta_columns_lower:
        raise ValueError(f"Stratify column '{stratify_col}' not found (case-insensitive). Available: {list(meta_df.columns)}")
    
    real_col = meta_columns_lower.get(stratify_col, stratify_col)
    if real_col:
        logger.info(f"Using stratification column '{real_col}'")
    groups = meta_df.groupby(real_col) if real_col else None
    group_sizes = groups.size()
    logger.info(f"Stratifying by '{stratify_col}' — found {len(group_sizes)} groups:\n{group_sizes}")

    n_groups = len(group_sizes)
    if n_groups == 0:
        raise ValueError("No groups found in stratification column")

    # Ideal samples per group (balanced)
    base_per_group = n_total // n_groups
    remainder = n_total % n_groups

    selected = []
    for group_name, group_df in groups:
        n_wanted = base_per_group + (1 if remainder > 0 else 0)
        remainder -= 1 if remainder > 0 else 0

        # Don't ask for more than available
        n_take = min(n_wanted, len(group_df))

        if n_take == 0:
            logger.warning(f"Group '{group_name}' has 0 samples — skipped")
            continue

        # Sample randomly within group
        sampled_ids = group_df.index.to_series().sample(n=n_take, random_state=None).tolist()
        selected.extend(sampled_ids)
        logger.info(f"Group '{group_name}': requested {n_wanted}, took {n_take}/{len(group_df)}")

    if len(selected) < n_total:
        logger.warning(f"Could only select {len(selected)}/{n_total} samples due to small group sizes")

    return selected

def select_samples(meta_df: pd.DataFrame, vcf_map: dict, n: int, stratify_by: str = None) -> dict:
    """Select n samples, optionally stratified."""
    overlap_ids = set(meta_df.index).intersection(vcf_map.keys())
    logger.info(f"Found {len(overlap_ids)} overlapping samples between metadata and VCFs.")

    if len(overlap_ids) == 0:
        logger.error("No overlapping samples! Check VCF naming vs metadata.")
        raise ValueError("No overlapping samples between metadata and VCFs.")

    if stratify_by:
        logger.info(f"Performing stratified sampling by '{stratify_by}'")
        selected_ids = stratified_sample(meta_df.loc[list(overlap_ids)], stratify_by, n)
    else:
        logger.info("Performing simple random sampling (no stratification)")
        selected_ids = random.sample(list(overlap_ids), min(n, len(overlap_ids)))

    selected_vcfs = {sid: vcf_map[sid] for sid in selected_ids if sid in vcf_map}
    logger.info(f"Selected {len(selected_vcfs)} samples")

    return selected_vcfs

def copy_vcfs(selected_vcfs: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for sid, vcf_path in selected_vcfs.items():
        new_name = f"{sid}.vcf.gz"
        dest = output_dir / new_name
        shutil.copy2(vcf_path, dest)
        logger.info(f"Copied {vcf_path.name} → {dest}")

# -------------------------
# Main
# -------------------------
def main(args):
    meta_file = Path(args.meta_file)
    vcf_dir = Path(args.vcf_dir)
    output_dir = Path(args.output_dir)

    meta_df = load_metadata(meta_file)
    vcf_map = scan_vcf_dir(vcf_dir, delimiter=args.vcf_sample_delimiter)
    selected_vcfs = select_samples(meta_df, vcf_map, args.n_samples, stratify_by=args.stratify_by)
    copy_vcfs(selected_vcfs, output_dir)

    logger.info(f"Subset creation completed. {len(selected_vcfs)} VCFs copied to {output_dir}")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a reproducible, optionally stratified subset of VCFs based on metadata.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=print_examples.__doc__ or ""
    )

    parser.add_argument("--vcf-dir", required=True, help="Directory containing VCF files (*.vcf.gz)")
    parser.add_argument("--meta-file", required=True, help="Metadata CSV (first column = sample IDs)")
    parser.add_argument("--output-dir", required=True, help="Directory to copy selected VCFs")
    parser.add_argument("--n-samples", type=int, default=100, help="Target number of samples")
    
    parser.add_argument(
    "--stratify-by",
    type=str.lower,   # ← force lowercase
    choices=["lineage", "phenotype", "none", None],
    default=None,
    help="Column to stratify by (balanced groups). Options: lineage, phenotype, or none (random)"
)
    
    parser.add_argument(
        "--vcf-sample-delimiter",
        default="_",
        help="Delimiter in VCF filenames separating sampleID from suffix (default '_')"
    )
    
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Set random seed for reproducibility"
    )

    args = parser.parse_args()

    if args.random_seed is not None:
        random.seed(args.random_seed)
        logger.info(f"Random seed set to {args.random_seed} for reproducibility")

    main(args)

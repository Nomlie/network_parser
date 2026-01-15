#!/usr/bin/env python3
"""
extracts_subset.py

Create a subset of samples from VCFs based on metadata.

Universal renaming: keeps only the sampleID prefix from VCF filenames,
stripping any suffix after the first delimiter.
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
# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def print_examples():
    examples = """
Examples
--------
python3 /home/nmfuphi/network_parser/scripts/extract_subset.py \
--vcf-dir /mnt/lustre/users/nmfuphi/AFRO_TB/AFRO_TB_ANNOTATION_VCF \
--meta-file /mnt/lustre/users/nmfuphi/AFRO_TB/AFRO_dataset_meta.csv \
--output-dir /mnt/lustre/users/nmfuphi/AFRO_TB/AFRO_TB_100_VCFs \
--n-samples 100 

Random 100-sample subset (no merge):
  ./extract_subset.py \\
    --vcf-dir AFRO_TB_ANNOTATION_VCF \\
    --meta-file AFRO_dataset_meta.csv \\
    --n-samples 100 \\
    --out-dir test_subset_basic

Phenotype-balanced subset:
  ./extract_subset.py \\
    --n-samples 100 \\
    --stratify-by phenotype \\
    --out-dir test_subset_pheno

Lineage-balanced subset + merged VCF (NetworkParser-ready):
  ./extract_subset.py \\
    --n-samples 100 \\
    --stratify-by lineage \\
    --merge-vcfs \\
    --out-dir test_subset_lineage

Reproducible subset with fixed seed:
  ./extract_subset.py \\
    --n-samples 100 \\
    --random-seed 1234 \\
    --merge-vcfs
"""
    print(examples.strip())

# -------------------------
# Functions
# -------------------------
def normalize_sample_id(vcf_path: Path, delimiter="_") -> str:
    """
    Extract canonical sample ID from VCF filename.
    Example:
      AFR123_library1.vcf.gz -> AFR123
      AFR456-seqA.vcf.gz -> AFR456
    """
    name = vcf_path.name
    if name.endswith(".vcf.gz"):
        name = name[:-7]  # remove .vcf.gz
    # Take everything before the first delimiter
    return name.split(delimiter)[0]

def load_metadata(meta_file: Path) -> set:
    """Load metadata CSV and return set of sample IDs (first column)."""
    df = pd.read_csv(meta_file)
    sample_ids = set(df.iloc[:, 0].astype(str))
    logging.info(f"Loaded {len(sample_ids)} samples from metadata.")
    return sample_ids

def scan_vcf_dir(vcf_dir: Path, delimiter="_") -> dict:
    """Return dictionary mapping canonical sample_id -> VCF path."""
    vcf_files = list(vcf_dir.glob("*.vcf.gz"))
    logging.info(f"Found {len(vcf_files)} VCF files in {vcf_dir}")
    vcf_map = {}
    for vcf in vcf_files:
        sid = normalize_sample_id(vcf, delimiter)
        vcf_map[sid] = vcf
    return vcf_map

def select_samples(meta_ids: set, vcf_ids: dict, n: int) -> dict:
    """Select n overlapping samples between metadata and VCFs."""
    overlap = meta_ids.intersection(vcf_ids.keys())
    logging.info(f"Found {len(overlap)} overlapping samples between metadata and VCFs.")

    if len(overlap) == 0:
        logging.error("No overlapping samples found! Check VCF naming vs metadata.")
        logging.info(f"Metadata sample IDs (first 5): {list(meta_ids)[:5]}")
        logging.info(f"VCF sample IDs (first 5): {list(vcf_ids.keys())[:5]}")
        raise ValueError("No overlapping samples between metadata and VCFs.")

    n_select = min(n, len(overlap))
    logging.info(f"Selecting {n_select} samples randomly from overlap.")
    selected_ids = random.sample(list(overlap), n_select)
    return {sid: vcf_ids[sid] for sid in selected_ids}

def copy_vcfs(selected_vcfs: dict, output_dir: Path):
    """Copy and rename selected VCFs to output directory, matching sample ID."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for sid, vcf_path in selected_vcfs.items():
        # Rename to match metadata sample ID
        new_name = f"{sid}.vcf.gz"
        dest = output_dir / new_name
        shutil.copy2(vcf_path, dest)
        logging.info(f"Copied {vcf_path.name} -> {dest}")

# -------------------------
# Main
# -------------------------
def main(args):
    meta_file = Path(args.meta_file)
    vcf_dir = Path(args.vcf_dir)
    output_dir = Path(args.output_dir)

    meta_ids = load_metadata(meta_file)
    vcf_map = scan_vcf_dir(vcf_dir, delimiter=args.vcf_sample_delimiter)
    selected_vcfs = select_samples(meta_ids, vcf_map, args.n_samples)
    copy_vcfs(selected_vcfs, output_dir)
    logging.info(f"Subset creation completed. {len(selected_vcfs)} VCFs copied to {output_dir}")
# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Extract a reproducible subset of VCFs and matching metadata for "
            "NetworkParser testing. Supports stratified sampling and optional "
            "VCF merging with bcftools."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:

  Random 100-sample subset:
    ./extract_subset.py --n-samples 100

  Phenotype-balanced subset:
    ./extract_subset.py --n-samples 100 --stratify-by phenotype

  Lineage-balanced subset + merged VCF:
    ./extract_subset.py --n-samples 100 --stratify-by lineage --merge-vcfs

  Show full examples:
    ./extract_subset.py --example
"""
    )
# -------------------------
# Argument parser
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create subset of VCFs based on metadata")
    parser.add_argument("--vcf-dir", required=True, help="Directory containing VCF files")
    parser.add_argument("--meta-file", required=True, help="Metadata CSV file")
    parser.add_argument("--output-dir", required=True, help="Directory to copy selected VCFs")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of samples to select")
    parser.add_argument(
        "--vcf-sample-delimiter",
        default="_",
        help="Delimiter in VCF filenames separating sampleID from suffix (default '_')"
    )
    args = parser.parse_args()
    main(args)

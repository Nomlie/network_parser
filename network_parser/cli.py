# network_parser/cli.py
import argparse
import sys
import logging
from pathlib import Path
from typing import Optional
import time
import json

# Configure detailed logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log"),
    ],
)

logger = logging.getLogger(__name__)

# ✅ Correct imports for package execution.
# Also includes a safe fallback for running the file directly (less recommended).
try:
    from .config import NetworkParserConfig
    from .network_parser import run_networkparser_analysis
except ImportError:  # pragma: no cover
    # Fallback for: python network_parser/cli.py
    from network_parser.config import NetworkParserConfig
    from network_parser.network_parser import run_networkparser_analysis


def load_config(config_path: Optional[str], default_config: NetworkParserConfig) -> NetworkParserConfig:
    if not config_path:
        return default_config

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as fh:
        data = json.load(fh)

    # Shallow override on NetworkParserConfig fields (legacy behavior)
    for k, v in data.items():
        if hasattr(default_config, k):
            setattr(default_config, k, v)

    return default_config


def validate_input_path(path_str: str) -> Path:
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Genomic input not found: {p}")
    return p


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NetworkParser: VCF/Matrix → statistically validated features + interpretable networks"
    )

    input_group = parser.add_argument_group("Input Files")
    input_group.add_argument(
        "--genomic",
        required=True,
        type=str,
        help=(
            "Genomic input: single file (CSV/TSV/VCF/FASTA) "
            "or DIRECTORY containing multiple VCF(.gz) files"
        ),
    )
    input_group.add_argument(
        "--regions",
        type=str,
        default=None,
        help=(
            "Optional regions/targets restriction for VCF processing. "
            "Examples: 'chrom:start-end' (e.g. 'NC_000962.3:1-1000000') "
            "or a BED file path supported by bcftools."
        ),
    )
    input_group.add_argument("--meta", type=str, default=None, help="Metadata CSV/TSV with sample IDs and labels")
    input_group.add_argument(
        "--label",
        required=True,
        type=str,
        help="Metadata label column to use for phenotype (e.g. Lineage, AMR)",
    )
    input_group.add_argument("--known_markers", type=str, default=None, help="Optional known markers file path")
    input_group.add_argument(
        "--ref_fasta",
        type=str,
        default=None,
        help="Optional reference FASTA (enables bcftools consensus + reference-aware normalization)",
    )

    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output_dir", required=True, type=str, help="Output directory for results")

    cfg_group = parser.add_argument_group("Config")
    cfg_group.add_argument("--config", type=str, default=None, help="Optional JSON config file to override defaults")

    flags_group = parser.add_argument_group("Validation Flags")
    flags_group.add_argument(
        "--validate_statistics",
        action="store_true",
        help="Run association testing + multiple testing correction (pre-tree)",
    )
    flags_group.add_argument(
        "--validate_interactions",
        action="store_true",
        help="Run post-tree interaction permutation validation",
    )

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    genomic_path = validate_input_path(args.genomic)

    meta_path = None
    if args.meta:
        meta_p = Path(args.meta)
        if not meta_p.is_file():
            logger.error(f"Metadata file not found: {meta_p}")
            sys.exit(1)
        meta_path = meta_p

    known_markers_path = None
    if args.known_markers:
        km_p = Path(args.known_markers)
        if not km_p.is_file():
            logger.error(f"Known markers file not found: {km_p}")
            sys.exit(1)
        known_markers_path = km_p

    ref_fasta_path = None
    if args.ref_fasta:
        rf_p = Path(args.ref_fasta)
        if not rf_p.is_file():
            logger.error(f"Reference FASTA file not found: {rf_p}")
            sys.exit(1)
        ref_fasta_path = rf_p

    # Load config
    config = load_config(args.config, NetworkParserConfig())

    # Logging summary
    logger.info("Starting NetworkParser pipeline")
    logger.info(f"Genomic input:    {genomic_path.resolve()}")
    logger.info(f"Metadata:         {meta_path.resolve() if meta_path else 'None'}")
    logger.info(f"Label column:     {args.label}")
    logger.info(f"Output directory: {Path(args.output_dir).resolve()}")
    if ref_fasta_path:
        logger.info(f"Reference FASTA:  {ref_fasta_path.resolve()}")
    if args.regions:
        logger.info(f"Regions/targets:  {args.regions}")

    try:
        start_time = time.time()
        run_networkparser_analysis(
            genomic_path=str(genomic_path),
            meta_path=str(meta_path) if meta_path else None,
            label_column=args.label,
            known_markers_path=str(known_markers_path) if known_markers_path else None,
            output_dir=args.output_dir,
            config=config,
            validate_statistics=args.validate_statistics,
            validate_interactions=args.validate_interactions,
            ref_fasta=str(ref_fasta_path) if ref_fasta_path else None,
            regions=args.regions,
        )
        elapsed_time = time.time() - start_time
        logger.info(f"NetworkParser pipeline completed successfully in {elapsed_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

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
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

try:
    from network_parser.config import NetworkParserConfig
    from network_parser import run_networkparser_analysis
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)


def load_config(config_path: Optional[str], default_config: NetworkParserConfig) -> NetworkParserConfig:
    """Load configuration from file or use default."""
    if config_path:
        logger.info(f"Loading config from: {config_path}")
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            for key, value in config_data.items():
                if hasattr(default_config, key):
                    setattr(default_config, key, value)
                else:
                    logger.warning(f"Config key '{key}' not found in NetworkParserConfig")
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            sys.exit(1)
    return default_config


def setup_parser() -> argparse.ArgumentParser:
    """Setup CLI parser with groups."""
    parser = argparse.ArgumentParser(
        prog="network_parser",
        description="NetworkParser: Interpretable genomic feature discovery pipeline.\n"
                    "Supports CSV/TSV/VCF/FASTA files or a directory of VCF files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Examples:\n"
               "  python -m network_parser.cli --genomic data/matrix.csv --label Group\n"
               "  python -m network_parser.cli --genomic vcfs_folder/ --meta metadata.csv --label Lineage"
    )

    input_group = parser.add_argument_group('Input Files')
    input_group.add_argument("--genomic", required=True, type=str,
                             help="Genomic input: single file (CSV/TSV/VCF/FASTA) "
                                  "or DIRECTORY containing multiple VCF(.gz) files")
    input_group.add_argument("--meta", type=str, default=None,
                             help="Metadata CSV/TSV with sample IDs and labels")
    input_group.add_argument("--label", required=True, type=str,
                             help="Column name containing the phenotype/label")
    input_group.add_argument("--known-markers", type=str, default=None,
                             help="File with known resistance markers (optional)")
    input_group.add_argument("--ref-fasta", type=str, default=None,
                             help="Reference genome FASTA file (required for consensus FASTA from VCF)")

    opt_group = parser.add_argument_group('Options')
    opt_group.add_argument("--output-dir", type=str, default="results/",
                           help="Output directory")
    opt_group.add_argument("--config", type=str, default=None,
                           help="JSON/YAML config file")
    opt_group.add_argument("--validate-statistics", action="store_true", default=True,
                           help="Run statistical validation")
    opt_group.add_argument("--validate-interactions", action="store_true", default=True,
                           help="Validate epistatic interactions")
    opt_group.add_argument("--export-format", choices=["json", "pickle"], default="json",
                           help="Export format")
    opt_group.add_argument("--version", action="version", version="NetworkParser 1.0.0")

    return parser


def validate_input_path(path_str: str) -> Path:
    """Validate that --genomic is either a file or a directory."""
    path = Path(path_str)
    if not path.exists():
        logger.error(f"Input path does not exist: {path}")
        sys.exit(1)

    if path.is_dir():
        vcf_files = list(path.glob("*.vcf")) + list(path.glob("*.vcf.gz"))
        if not vcf_files:
            logger.error(f"No VCF files (.vcf or .vcf.gz) found in directory: {path}")
            sys.exit(1)
        logger.info(f"Detected directory input with {len(vcf_files)} VCF files")
    elif path.is_file():
        supported = {'.csv', '.tsv', '.vcf', '.vcf.gz', '.fasta', '.fa'}
        if path.suffix.lower() not in supported:
            logger.error(f"Unsupported file extension: {path.suffix}. "
                         f"Supported: {', '.join(supported)}")
            sys.exit(1)
    else:
        logger.error(f"Input path is neither a file nor a directory: {path}")
        sys.exit(1)

    return path


def main():
    """CLI entrypoint."""
    parser = setup_parser()
    args = parser.parse_args()

    # ────────────────────────────────────────────────
    # Validate inputs
    # ────────────────────────────────────────────────
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
    logger.info(f"Genomic input:   {genomic_path.resolve()}")
    logger.info(f"Metadata:        {meta_path.resolve() if meta_path else 'None'}")
    logger.info(f"Label column:    {args.label}")
    logger.info(f"Output directory: {Path(args.output_dir).resolve()}")
    if ref_fasta_path:
        logger.info(f"Reference FASTA: {ref_fasta_path.resolve()}")

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
            ref_fasta=str(ref_fasta_path) if ref_fasta_path else None   # pass ref_fasta if provided
        )
        elapsed_time = time.time() - start_time
        logger.info(f"NetworkParser pipeline completed successfully in {elapsed_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
    
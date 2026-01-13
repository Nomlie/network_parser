# network_parser/cli.py
import argparse
import sys
import logging
from pathlib import Path
from typing import Optional
import time
import numpy as np

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

def validate_file_path(file_path: str, file_type: str) -> None:
    """Validate if a file exists."""
    if not Path(file_path).is_file():
        logger.error(f"{file_type} file not found: {file_path}")
        sys.exit(1)

def load_config(config_path: Optional[str], default_config: NetworkParserConfig) -> NetworkParserConfig:
    """Load configuration from file or use default."""
    if config_path:
        logger.info(f"Loading config from: {config_path}")
        # Implement JSON/YAML loading here if needed
        import json
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            for key, value in config_data.items():
                setattr(default_config, key, value)
    return default_config

def setup_parser() -> argparse.ArgumentParser:
    """Setup CLI parser with groups."""
    parser = argparse.ArgumentParser(
        prog="network_parser",
        description="NetworkParser: Interpretable genomic feature discovery pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example: python -m network_parser.cli \
            --genomic  /home/nmfuphi/network_parser/data/AFRO_TB/test_subset\
            --meta /home/nmfuphi/network_parser/data/AFRO_TB/AFRO_dataset_meta.csv \
            --label    Lineage \
            --output-dir results_tb_2026/")
    
    input_group = parser.add_argument_group('Input Files')
    input_group.add_argument("--genomic", required=True, type=str,
                             help="Genomic matrix (CSV/TSV/FASTA/VCF) containing features and optionally labels.")
    input_group.add_argument("--meta", type=str, default=None,
                             help="Metadata (CSV/TSV) with annotations including labels (optional if labels are in genomic).")
    input_group.add_argument("--label", required=True, type=str,
                             help="Label column name, must exist in meta or genomic file.")
    input_group.add_argument("--known-markers", type=str, default=None,
                             help="Known markers file (TXT/CSV/TSV)")
    input_group.add_argument("--ref-fasta", type=str, default=None,
                             help="Reference genome FASTA file (e.g., H37Rv.fa). Required for consensus FASTA output from VCF.")
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

def main():
    """CLI entrypoint."""
    parser = setup_parser()
    args = parser.parse_args()
    
    # Validate inputs
    validate_file_path(args.genomic, 'genomic')
    if args.meta:
        validate_file_path(args.meta, 'meta')
    if args.known_markers:
        validate_file_path(args.known_markers, 'known_markers')
    
    # Config
    config = load_config(args.config, NetworkParserConfig())
    
    logger.info("Attempting to import run_networkparser_analysis and NetworkParserConfig")
    logger.info(f"Running cli.py from: {Path(__file__).resolve()}")
    logger.info("Starting NetworkParser pipeline")
    logger.info(f"Genomic data: {Path(args.genomic).resolve()}")
    logger.info(f"Label column: {args.label}")
    logger.info(f"Output directory: {Path(args.output_dir).resolve()}")
    
    try:
        start_time = time.time()
        run_networkparser_analysis(
            genomic_path=args.genomic,  # Changed from genomic_data_path to genomic_path
            meta_path=args.meta,        # Changed from metadata_path to meta_path
            label_column=args.label,
            known_markers_path=args.known_markers,
            output_dir=args.output_dir,
            config=config,
            validate_statistics=args.validate_statistics,
            validate_interactions=args.validate_interactions,
            ref_fasta=args.ref_fasta
        )
        elapsed_time = time.time() - start_time
        logger.info(f"NetworkParser pipeline completed successfully in {elapsed_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
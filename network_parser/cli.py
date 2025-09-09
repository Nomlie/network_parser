import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Debug import attempt
logger.info("Attempting to import run_networkparser_analysis and NetworkParserConfig")
try:
    from network_parser.network_parser import run_networkparser_analysis
    from network_parser.config import NetworkParserConfig
    logger.info("Successfully imported run_networkparser_analysis and NetworkParserConfig")
except ImportError as e:
    logger.error(f"ImportError: {e}. Ensure 'network_parser' is in PYTHONPATH and network_parser.py exists at /Users/nmfuphicsir.co.za/Documents/pHDProject/Code/network_parser/network_parser/network_parser.py")
    sys.exit(1)

def validate_file_path(file_path: str, file_type: str) -> Path:
    """Validate if file exists and has correct extension."""
    path = Path(file_path)
    if not path.is_file():
        logger.error(f"{file_type} file does not exist: {file_path}")
        sys.exit(1)
    
    valid_extensions = {
        'genomic': ('.csv', '.tsv', '.fasta', '.vcf'),
        'meta': ('.csv', '.tsv'),
        'known_markers': ('.txt', '.csv', '.tsv'),
        'config': ('.json',)
    }
    
    if file_type in valid_extensions and not path.suffix.lower() in valid_extensions[file_type]:
        logger.error(f"Invalid {file_type} file format: {file_path}. Expected extensions: {valid_extensions[file_type]}")
        sys.exit(1)
    
    return path

def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up the argument parser with organized argument groups."""
    parser = argparse.ArgumentParser(
        prog="network_parser.cli",
        description=(
            "NetworkParser: Interpretable framework for genomic feature discovery.\n"
            "Integrates decision tree-based feature discovery, statistical validation, "
            "and interaction detection into a single reproducible pipeline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example: python -m network_parser.cli --genomic data/matrix.csv --meta data/labels.csv --label label --output-dir results/"
    )

    # Input File Arguments
    input_group = parser.add_argument_group('Required Input Files')
    input_group.add_argument(
        "--genomic", required=True, type=str,
        help="Path to genomic matrix file (CSV, TSV, FASTA, or VCF supported)."
    )
    input_group.add_argument(
        "--label", required=True, type=str,
        help="Column in genomic data or metadata to use as labels (phenotype or class)."
    )

    # Optional Arguments
    optional_group = parser.add_argument_group('Optional Parameters')
    optional_group.add_argument(
        "--meta", type=str, default=None,
        help="Optional path to metadata file (CSV/TSV with sample annotations)."
    )
    optional_group.add_argument(
        "--known-markers", type=str, default=None,
        help="Optional file containing a list of known genetic markers for benchmarking."
    )
    optional_group.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save results and intermediate files."
    )
    optional_group.add_argument(
        "--export-format", type=str, choices=["json", "pickle"], default="json",
        help="Format for saving results."
    )
    optional_group.add_argument(
        "--validate-statistics", action="store_true",
        help="Enable statistical validation of features (bootstrap, chi-squared, multiple testing)."
    )
    optional_group.add_argument(
        "--validate-interactions", action="store_true",
        help="Enable validation of epistatic interactions with permutation tests."
    )

    # Configuration Arguments
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument(
        "--config", type=str, default=None,
        help="Optional JSON config file with advanced parameters."
    )
    config_group.add_argument(
        "--version", action="version", version="NetworkParser 0.1.0",
        help="Show program's version number and exit."
    )

    return parser

def load_config(config_path: Optional[str]) -> NetworkParserConfig:
    """Load and validate configuration from JSON file."""
    config = NetworkParserConfig()
    if config_path:
        try:
            config_path = validate_file_path(config_path, 'config')
            with config_path.open('r') as f:
                user_cfg = json.load(f)
            for key, value in user_cfg.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logger.warning(f"Unknown configuration parameter: {key}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            sys.exit(1)
    return config

def main():
    """Main function to orchestrate the NetworkParser pipeline."""
    try:
        logger.info(f"Running cli.py from: {Path(__file__).resolve()}")
        parser = setup_argument_parser()
        args = parser.parse_args()

        genomic_path = validate_file_path(args.genomic, 'genomic')
        meta_path = validate_file_path(args.meta, 'meta') if args.meta else None
        known_markers = validate_file_path(args.known_markers, 'known_markers') if args.known_markers else None
        output_dir = args.output_dir

        logger.info("Starting NetworkParser pipeline")
        logger.info(f"Genomic data: {genomic_path}")
        if meta_path:
            logger.info(f"Metadata: {meta_path}")
        logger.info(f"Label column: {args.label}")
        if known_markers:
            logger.info(f"Known markers: {known_markers}")
        if output_dir:
            logger.info(f"Output directory: {output_dir}")

        config = load_config(args.config)

        run_networkparser_analysis(
            genomic_data_path=str(genomic_path),
            metadata_path=str(meta_path) if meta_path else None,
            label_column=args.label,
            known_markers_path=str(known_markers) if known_markers else None,
            output_dir=output_dir,
            config=config,
            validate_statistics=args.validate_statistics,
            validate_interactions=args.validate_interactions,
            export_format=args.export_format
        )

        logger.info("NetworkParser pipeline completed successfully")

    except KeyboardInterrupt:
        logger.error("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

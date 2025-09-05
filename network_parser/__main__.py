import argparse
from . import run_networkparser_analysis, NetworkParserConfig

def main():
    parser = argparse.ArgumentParser(
        prog="network_parser",
        description="NetworkParser: Interpretable framework for genomic feature discovery.\n"
                    "Integrates decision tree-based feature discovery, statistical validation, "
                    "and interaction detection into a single reproducible pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--genomic", required=True, type=str,
        help="Path to genomic matrix file (CSV, TSV, FASTA, or VCF supported)."
    )
    parser.add_argument(
        "--meta", required=True, type=str,
        help="Path to metadata file (CSV/TSV with sample annotations)."
    )
    parser.add_argument(
        "--label", required=True, type=str,
        help="Column in metadata to use as labels (phenotype or class)."
    )

    # Optional arguments
    parser.add_argument(
        "--known-markers", type=str, default=None,
        help="Optional file containing a list of known genetic markers for benchmarking."
    )
    parser.add_argument(
        "--export-format", type=str, choices=["json", "pickle"], default="json",
        help="Format for saving results."
    )
    parser.add_argument(
        "--validate-statistics", action="store_true",
        help="Enable statistical validation of features (bootstrap, chi-squared, multiple testing)."
    )
    parser.add_argument(
        "--validate-interactions", action="store_true",
        help="Enable validation of epistatic interactions with permutation tests."
    )

    # Convenience
    parser.add_argument(
        "--config", type=str, default=None,
        help="Optional JSON config file with advanced parameters."
    )
    parser.add_argument(
        "--version", action="version", version="NetworkParser 0.1.0"
    )

    args = parser.parse_args()

    # Load config if provided
    config = NetworkParserConfig()
    if args.config:
        import json
        with open(args.config) as f:
            user_cfg = json.load(f)
        for k, v in user_cfg.items():
            setattr(config, k, v)

    # Run pipeline
    run_networkparser_analysis(
        genomic_data_path=args.genomic,
        metadata_path=args.meta,
        label_column=args.label,
        config=config,
        validate_statistics=args.validate_statistics,
        validate_interactions=args.validate_interactions
    )

if __name__ == "__main__":
    main()

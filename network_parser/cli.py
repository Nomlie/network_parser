#!/usr/bin/env python3
# network_parser/cli.py
"""
NetworkParser command-line interface.

Supports three entry points:

1) run
   Single-label NetworkParser workflow:
   load -> align -> RF-FDR central feature filtering -> ML protocol/model selector
   -> conditional decision-tree interpretability -> optional downstream validation.

2) train-two-level
   Two-label hierarchical protocol:
   Level 1: strain / lineage / group placement
   Level 2: drug-resistance phenotype / resistance-profile prediction

3) query
   User-facing inference on a new strain/sample using a saved two-level registry.

Backward compatibility:
If no subcommand is supplied, arguments are interpreted as the single-label
``run`` workflow so existing calls such as ``python -m network_parser.cli --genomic ...``
continue to work.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from network_parser.config import NetworkParserConfig
    from network_parser.network_parser import run_networkparser_analysis
    from network_parser.two_level_protocol import TwoLevelProtocol
    from network_parser.query_engine import NetworkParserQueryEngine
except Exception:  # pragma: no cover - allows direct execution from source tree
    from config import NetworkParserConfig  # type: ignore
    from network_parser import run_networkparser_analysis  # type: ignore
    from two_level_protocol import TwoLevelProtocol  # type: ignore
    from query_engine import NetworkParserQueryEngine  # type: ignore


LOGGER = logging.getLogger(__name__)
VALID_PIPELINE_MODES = {"matrix_only", "decision_tree_only", "ml_only", "both"}
VALID_SUBCOMMANDS = {"run", "train-two-level", "query"}


# -----------------------------------------------------------------------------
# Logging / config helpers
# -----------------------------------------------------------------------------

def configure_logging(verbose: bool = False, quiet: bool = False) -> None:
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_config(config_path: Optional[str]) -> NetworkParserConfig:
    """Load NetworkParserConfig and apply optional JSON overrides."""
    config = NetworkParserConfig()

    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as handle:
            overrides: Dict[str, Any] = json.load(handle)

        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                LOGGER.warning("Ignoring unknown config key: %s", key)

    if hasattr(config, "__post_init__"):
        config.__post_init__()
    return config


def set_if_provided(config: NetworkParserConfig, key: str, value: Any) -> None:
    """Set a config value only when the CLI provided a non-None value."""
    if value is not None:
        setattr(config, key, value)


def apply_common_overrides(config: NetworkParserConfig, args: argparse.Namespace) -> NetworkParserConfig:
    """Apply CLI overrides that are shared across commands."""
    set_if_provided(config, "n_jobs", getattr(args, "n_jobs", None))
    set_if_provided(config, "central_feature_filter_method", getattr(args, "central_feature_filter_method", None))

    # RF-FDR selector controls. These are optional so config files remain the
    # canonical place for tuned runs, while CLI remains convenient for testing.
    set_if_provided(config, "rf_selector_n_estimators", getattr(args, "rf_selector_n_estimators", None))
    set_if_provided(config, "rf_selector_n_observed_repeats", getattr(args, "rf_selector_n_observed_repeats", None))
    set_if_provided(config, "rf_selector_n_permutations", getattr(args, "rf_selector_n_permutations", None))
    set_if_provided(config, "rf_selector_fdr_alpha", getattr(args, "rf_selector_fdr_alpha", None))
    set_if_provided(config, "rf_selector_random_state", getattr(args, "rf_selector_random_state", None))
    set_if_provided(config, "rf_selector_top_n", getattr(args, "rf_selector_top_n", None))
    set_if_provided(config, "rf_selector_min_importance", getattr(args, "rf_selector_min_importance", None))
    set_if_provided(config, "rf_selector_fallback_strategy", getattr(args, "rf_selector_fallback_strategy", None))
    set_if_provided(config, "rf_selector_fallback_top_n", getattr(args, "rf_selector_fallback_top_n", None))
    set_if_provided(config, "feature_filter_fallback_strategy", getattr(args, "feature_filter_fallback_strategy", None))
    set_if_provided(config, "n_permutation_tests", getattr(args, "n_permutation_tests", None))
    set_if_provided(config, "fdr_alpha", getattr(args, "fdr_alpha", None))
    set_if_provided(config, "multiple_testing_method", getattr(args, "multiple_testing_method", None))

    if hasattr(config, "__post_init__"):
        config.__post_init__()
    return config


def add_logging_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument("--quiet", action="store_true", help="Only show warnings and errors.")


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="Optional JSON file with NetworkParserConfig overrides.")
    parser.add_argument("--n_jobs", type=int, default=None, help="Number of parallel workers where supported.")


def add_rf_fdr_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Central feature-selection controls")
    group.add_argument(
        "--central_feature_filter_method",
        "--central_filter_method",
        dest="central_feature_filter_method",
        choices=["rf_fdr", "chi2_fdr", "fisher_fdr", "chi2_perm_fdr"],
        default=None,
        help=(
            "Central feature-filtering method. "
            "rf_fdr uses RF importance with permutation-derived empirical p-values and FDR correction; "
            "chi2_fdr uses chi-square association testing plus multiple-testing correction; "
            "fisher_fdr uses Fisher where appropriate plus multiple-testing correction; "
            "chi2_perm_fdr uses chi-square label permutations, empirical p-values, and FDR correction."
        ),
    )
    group.add_argument("--n_permutation_tests", type=int, default=None, help="Label permutations for chi2_perm_fdr and downstream permutation utilities.")
    group.add_argument("--fdr_alpha", type=float, default=None, help="FDR alpha used by association-FDR and chi2_perm_fdr.")
    group.add_argument("--multiple_testing_method", choices=["fdr_bh", "bonferroni"], default=None, help="Multiple-testing correction method for association-FDR and chi2_perm_fdr.")
    group.add_argument("--rf_selector_n_estimators", type=int, default=None, help="Random Forest trees used during RF-FDR scoring.")
    group.add_argument("--rf_selector_n_observed_repeats", type=int, default=None, help="Observed RF importance repeats.")
    group.add_argument("--rf_selector_n_permutations", type=int, default=None, help="Label permutations for empirical p-values.")
    group.add_argument("--rf_selector_fdr_alpha", type=float, default=None, help="FDR-BH alpha for RF-FDR retention.")
    group.add_argument("--rf_selector_random_state", type=int, default=None, help="Random seed for RF-FDR.")
    group.add_argument("--rf_selector_top_n", type=int, default=None, help="Optional cap on retained RF-FDR features.")
    group.add_argument("--rf_selector_min_importance", type=float, default=None, help="Minimum observed RF importance for retained features.")
    group.add_argument(
        "--rf_selector_fallback_strategy",
        choices=["stop", "top_n", "unfiltered"],
        default=None,
        help="What to do if RF-FDR retains no features. Use 'stop' for publication-grade runs.",
    )

    group.add_argument(
        "--rf_selector_fallback_top_n",
        type=int,
        default=None,
        help="Number of top RF-ranked features to retain only when fallback strategy is top_n.",
    )

    group.add_argument(
        "--feature_filter_fallback_strategy",
        choices=["stop", "unfiltered"],
        default=None,
        help="Fallback for chi-square/Fisher central filtering when no features survive FDR.",
    )

# -----------------------------------------------------------------------------
# Parser builders
# -----------------------------------------------------------------------------

def build_run_parser(prog: Optional[str] = None, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run the single-label NetworkParser workflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=add_help,
    )

    parser.add_argument("--genomic", required=True, help="Genomic input file or directory: VCF/VCF.gz/CSV/TSV.")
    parser.add_argument("--meta", required=True, help="Metadata CSV/TSV containing the supervised label column.")
    parser.add_argument("--label", required=True, help="Metadata column used as the supervised target.")
    parser.add_argument("--known_markers", default=None, help="Optional known-marker file for comparison or annotation.")
    parser.add_argument("--ref_fasta", default=None, help="Optional reference FASTA/GenBank context for VCF-oriented workflows.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")

    parser.add_argument(
        "--pipeline_mode",
        default="both",
        choices=sorted(VALID_PIPELINE_MODES),
        help="Workflow mode.",
    )
    parser.add_argument("--validate_statistics", action="store_true", help="Keep validation flag for compatibility; central filtering is config-controlled.")
    parser.add_argument("--validate_interactions", action="store_true", help="Run optional post-tree interaction validation when available.")

    parser.add_argument("--run_ml_protocol", action="store_true", help="Force ML protocol branch on through config.")
    parser.add_argument("--disable_central_feature_filtering", action="store_true", help="Pass aligned matrix forward without central feature filtering.")
    parser.add_argument("--disable_model_selector", action="store_true", help="Disable automatic model-selector behaviour where supported.")
    parser.add_argument("--disable_conditional_dt", action="store_true", help="Prevent selector-driven decision-tree triggering where supported.")

    parser.add_argument("--ml_algorithm", default=None, help="Optional ML algorithm override, e.g. auto, RF, MLP, LR, DT, SVC, MBCS, DNL.")
    parser.add_argument("--ml_min_sensitivity", type=float, default=None, help="Optional ML protocol sensitivity lower bound.")
    parser.add_argument("--ml_max_sensitivity", type=float, default=None, help="Optional ML protocol sensitivity upper bound.")
    parser.add_argument("--ml_step_sensitivity", type=float, default=None, help="Optional ML protocol sensitivity step.")
    parser.add_argument("--ml_empty_symbol", default=None, help="Optional empty symbol for ML-formatted data.")
    parser.add_argument("--ml_remove_empty_field_threshold", type=float, default=None, help="Optional empty-field removal threshold.")

    add_config_args(parser)
    add_rf_fdr_args(parser)
    add_logging_args(parser)
    return parser


def build_train_two_level_parser(prog: Optional[str] = None, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Train the two-level NetworkParser protocol: strain placement first, resistance profile second.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=add_help,
    )

    parser.add_argument("--genomic", required=True, help="Genomic input file or directory: VCF/VCF.gz/CSV/TSV.")
    parser.add_argument("--meta", required=True, help="Metadata CSV/TSV containing both supervised labels.")
    parser.add_argument("--level1_label", required=True, help="Metadata column for strain/lineage/group placement.")
    parser.add_argument("--level2_label", required=True, help="Metadata column for drug-resistance phenotype/profile.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--ref_fasta", default=None, help="Optional reference FASTA/GenBank context for VCF-oriented workflows.")
    parser.add_argument("--algorithm", default=None, help="Optional ML algorithm override passed to the ML protocol.")
    parser.add_argument("--no_global_level2", action="store_true", help="Disable the global Level 2 fallback model.")
    parser.add_argument("--min_level2_samples_per_group", type=int, default=None, help="Minimum samples needed for group-specific Level 2 models.")

    add_config_args(parser)
    add_rf_fdr_args(parser)
    add_logging_args(parser)
    return parser


def build_query_parser(prog: Optional[str] = None, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Apply a trained two-level NetworkParser registry to new strain/sample input.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=add_help,
    )

    parser.add_argument("--genomic", required=True, help="New genomic input file or directory: VCF/VCF.gz/CSV/TSV/FASTA.")
    parser.add_argument("--registry", required=True, help="Path to two_level_model_registry.json from training.")
    parser.add_argument("--output_dir", required=True, help="Prediction output directory.")
    parser.add_argument("--ref_fasta", default=None, help="Optional reference FASTA/GenBank context for VCF-oriented workflows.")
    parser.add_argument("--max_markers", type=int, default=10, help="Maximum supporting markers to report per prediction level.")
    parser.add_argument(
        "--query_input_type",
        choices=["auto", "matrix", "vcf", "raw_sequence"],
        default="auto",
        help="How to interpret --genomic. Use raw_sequence for raw FASTA DNA queries.",
    )
    parser.add_argument(
        "--raw_sequence_mapping_mode",
        choices=["auto", "blast", "exact"],
        default="auto",
        help="How raw FASTA query sequences are mapped to selected feature contexts.",
    )

    add_config_args(parser)
    add_logging_args(parser)
    return parser


def build_top_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NetworkParser command-line interface.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the single-label NetworkParser workflow.", parents=[build_run_parser(add_help=False)], add_help=False)
    run.set_defaults(command="run")

    train_two = subparsers.add_parser(
        "train-two-level",
        help="Train the two-label strain-identity and resistance-profile protocol.",
        parents=[build_train_two_level_parser(add_help=False)],
        add_help=False,
    )
    train_two.set_defaults(command="train-two-level")

    query = subparsers.add_parser(
        "query",
        help="Run user-facing prediction on new strain/sample input.",
        parents=[build_query_parser(add_help=False)],
        add_help=False,
    )
    query.set_defaults(command="query")
    return parser


# -----------------------------------------------------------------------------
# Command runners
# -----------------------------------------------------------------------------

def run_single_label(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_config(args.config)
    config = apply_common_overrides(config, args)

    config.pipeline_mode = args.pipeline_mode

    if args.run_ml_protocol:
        config.run_ml_protocol = True

    if args.disable_central_feature_filtering:
        config.run_central_feature_filtering = False
    else:
        config.run_central_feature_filtering = bool(getattr(config, "run_central_feature_filtering", True))

    if args.disable_model_selector:
        config.run_model_selector = False
        config.disable_model_selector = True

    if args.disable_conditional_dt:
        config.run_conditional_dt = False
        config.disable_conditional_dt = True

    set_if_provided(config, "ml_algorithm", args.ml_algorithm)
    set_if_provided(config, "ml_min_sensitivity", args.ml_min_sensitivity)
    set_if_provided(config, "ml_max_sensitivity", args.ml_max_sensitivity)
    set_if_provided(config, "ml_step_sensitivity", args.ml_step_sensitivity)
    set_if_provided(config, "ml_empty_symbol", args.ml_empty_symbol)
    set_if_provided(config, "ml_remove_empty_field_threshold", args.ml_remove_empty_field_threshold)

    if hasattr(config, "__post_init__"):
        config.__post_init__()

    LOGGER.info("Starting NetworkParser single-label workflow | mode=%s", config.pipeline_mode)
    return run_networkparser_analysis(
        genomic_path=args.genomic,
        meta_path=args.meta,
        label_column=args.label,
        known_markers_path=args.known_markers,
        output_dir=args.output_dir,
        config=config,
        validate_statistics=bool(args.validate_statistics),
        validate_interactions=bool(args.validate_interactions),
        ref_fasta=args.ref_fasta,
    )


def run_train_two_level(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_config(args.config)
    config = apply_common_overrides(config, args)

    if hasattr(config, "__post_init__"):
        config.__post_init__()

    LOGGER.info("Starting NetworkParser two-level training")
    protocol = TwoLevelProtocol(config=config)
    return protocol.train(
        genomic_path=args.genomic,
        meta_path=args.meta,
        level1_label=args.level1_label,
        level2_label=args.level2_label,
        output_dir=args.output_dir,
        ref_fasta=args.ref_fasta,
        algorithm=args.algorithm,
        train_global_level2=not bool(args.no_global_level2),
        min_level2_samples_per_group=args.min_level2_samples_per_group,
    )


def run_query(args: argparse.Namespace) -> Any:
    config = load_config(args.config)
    config = apply_common_overrides(config, args)

    LOGGER.info("Starting NetworkParser query workflow")
    engine = NetworkParserQueryEngine(registry_path=args.registry, config=config)
    return engine.query(
        genomic_path=args.genomic,
        output_dir=args.output_dir,
        ref_fasta=args.ref_fasta,
        max_markers=int(args.max_markers),
        n_jobs=args.n_jobs,
        query_input_type=args.query_input_type,
        raw_sequence_mapping_mode=args.raw_sequence_mapping_mode,
    )


# -----------------------------------------------------------------------------
# Main dispatcher
# -----------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    tokens = list(sys.argv[1:] if argv is None else argv)

    # Backward-compatible mode: no subcommand means "run".
    if not tokens or tokens[0] not in VALID_SUBCOMMANDS:
        parser = build_run_parser(prog="network_parser.cli")
        args = parser.parse_args(tokens)
        args.command = "run"
        return args

    parser = build_top_parser()
    return parser.parse_args(tokens)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(verbose=bool(getattr(args, "verbose", False)), quiet=bool(getattr(args, "quiet", False)))

    try:
        if args.command == "run":
            run_single_label(args)
        elif args.command == "train-two-level":
            run_train_two_level(args)
        elif args.command == "query":
            run_query(args)
        else:
            raise ValueError(f"Unsupported command: {args.command}")
    except Exception as exc:
        LOGGER.exception("NetworkParser CLI failed: %s", exc)
        return 1

    LOGGER.info("NetworkParser CLI completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

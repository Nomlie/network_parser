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

    feature_panel_check = getattr(args, "feature_panel_check", None)
    if feature_panel_check is not None:
        config.run_feature_panel_separability_check = feature_panel_check == "on"
    set_if_provided(config, "feature_panel_sizes", getattr(args, "feature_panel_sizes", None))
    set_if_provided(config, "feature_panel_metric", getattr(args, "feature_panel_metric", None))
    set_if_provided(config, "feature_panel_classifier", getattr(args, "feature_panel_classifier", None))
    set_if_provided(config, "feature_panel_lr_max_iter", getattr(args, "feature_panel_lr_max_iter", None))
    set_if_provided(config, "feature_panel_lr_tol", getattr(args, "feature_panel_lr_tol", None))
    set_if_provided(config, "feature_panel_rf_n_estimators", getattr(args, "feature_panel_rf_n_estimators", None))
    set_if_provided(config, "feature_panel_rf_max_features", getattr(args, "feature_panel_rf_max_features", None))
    set_if_provided(config, "feature_panel_rf_min_samples_leaf", getattr(args, "feature_panel_rf_min_samples_leaf", None))
    set_if_provided(config, "feature_panel_rf_class_weight", getattr(args, "feature_panel_rf_class_weight", None))
    set_if_provided(config, "feature_panel_rf_n_jobs", getattr(args, "feature_panel_rf_n_jobs", None))
    set_if_provided(config, "feature_panel_min_score", getattr(args, "feature_panel_min_score", None))
    set_if_provided(config, "feature_panel_selection_rule", getattr(args, "feature_panel_selection_rule", None))
    set_if_provided(config, "feature_panel_cv_splits", getattr(args, "feature_panel_cv_splits", None))

    set_if_provided(config, "global_level2_label_column", getattr(args, "global_level2_label", None))

    if bool(getattr(args, "level2_drop_low_support_classes", False)):
        config.level2_drop_low_support_classes = True
    set_if_provided(config, "level2_min_class_count", getattr(args, "level2_min_class_count", None))

    if bool(getattr(args, "level2_train_binary_global_fallback", False)):
        config.level2_train_binary_global_fallback = True
    set_if_provided(config, "level2_binary_label_column", getattr(args, "level2_binary_label_column", None))
    set_if_provided(config, "level2_binary_label_mapping_file", getattr(args, "level2_binary_label_mapping_file", None))
    set_if_provided(config, "level2_binary_resistant_values", getattr(args, "level2_binary_resistant_values", None))
    set_if_provided(config, "level2_binary_susceptible_values", getattr(args, "level2_binary_susceptible_values", None))

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

    group.add_argument(
        "--feature_panel_check",
        choices=["on", "off"],
        default=None,
        help="Enable or disable the ranked feature-panel separability check after central filtering.",
    )
    group.add_argument(
        "--feature_panel_sizes",
        default=None,
        help="Comma-separated top-N panel sizes to evaluate after FDR filtering, for example: 100,200,500.",
    )
    group.add_argument(
        "--feature_panel_metric",
        choices=["balanced_accuracy", "adjusted_rand", "normalized_mutual_info", "silhouette"],
        default=None,
        help="Metric used to choose the model-ready feature panel.",
    )
    group.add_argument(
        "--feature_panel_classifier",
        choices=["lr", "rf"],
        default=None,
        help=(
            "Supervised classifier used for the balanced-accuracy feature-panel probe. "
            "lr is faster; rf is usually slower but can capture nonlinear separability."
        ),
    )
    group.add_argument(
        "--feature_panel_lr_max_iter",
        type=int,
        default=None,
        help="Maximum iterations for the LR feature-panel probe.",
    )
    group.add_argument(
        "--feature_panel_lr_tol",
        type=float,
        default=None,
        help="Optimization tolerance for the LR feature-panel probe.",
    )
    group.add_argument(
        "--feature_panel_rf_n_estimators",
        type=int,
        default=None,
        help="Number of trees for the RF feature-panel probe.",
    )
    group.add_argument(
        "--feature_panel_rf_max_features",
        choices=["sqrt", "log2", "none"],
        default=None,
        help="Feature subsampling rule for the RF feature-panel probe.",
    )
    group.add_argument(
        "--feature_panel_rf_min_samples_leaf",
        type=int,
        default=None,
        help="Minimum leaf size for the RF feature-panel probe.",
    )
    group.add_argument(
        "--feature_panel_rf_class_weight",
        choices=["balanced", "balanced_subsample", "none"],
        default=None,
        help="Class weighting strategy for the RF feature-panel probe.",
    )
    group.add_argument(
        "--feature_panel_rf_n_jobs",
        type=int,
        default=None,
        help="Parallel workers used inside the RF feature-panel probe. Use -1 for all available cores.",
    )
    group.add_argument(
        "--feature_panel_min_score",
        type=float,
        default=None,
        help="Minimum separability score required for the selected panel threshold.",
    )
    group.add_argument(
        "--feature_panel_selection_rule",
        choices=["smallest_passing", "best_passing", "best_available"],
        default=None,
        help="Rule for choosing among scored feature panels.",
    )
    group.add_argument(
        "--feature_panel_cv_splits",
        type=int,
        default=None,
        help="Cross-validation folds used by the supervised panel diagnostic.",
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
    parser.add_argument(
        "--global_level2_label",
        default=None,
        help=(
            "Optional metadata column for the standard global Level 2 fallback. "
            "Use this when group-specific Level 2 models should learn the detailed "
            "--level2_label, but the global fallback should learn a broader endpoint "
            "such as AMR_binary."
        ),
    )
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--ref_fasta", default=None, help="Optional reference FASTA/GenBank context for VCF-oriented workflows.")
    parser.add_argument("--algorithm", default=None, help="Optional ML algorithm override passed to the ML protocol.")
    parser.add_argument("--no_global_level2", action="store_true", help="Disable the global Level 2 fallback model.")
    parser.add_argument("--min_level2_samples_per_group", type=int, default=None, help="Minimum samples needed for group-specific Level 2 models.")
    parser.add_argument(
        "--level2_drop_low_support_classes",
        action="store_true",
        help=(
            "Exclude Level 2 classes with too few samples before Level 2 statistical "
            "filtering and model screening. This is useful when singleton or very rare "
            "phenotype classes make stratified cross-validation impossible."
        ),
    )
    parser.add_argument(
        "--level2_min_class_count",
        type=int,
        default=None,
        help="Minimum samples per Level 2 class when --level2_drop_low_support_classes is enabled.",
    )
    parser.add_argument(
        "--level2_train_binary_global_fallback",
        action="store_true",
        help=(
            "Train an additional global Level 2 resistant/susceptible fallback model "
            "across all lineages. This model is used when a group-specific Level 2 "
            "model is unavailable."
        ),
    )
    parser.add_argument(
        "--level2_binary_label_column",
        default=None,
        help=(
            "Optional metadata column containing the global binary Level 2 endpoint "
            "for the additional fallback model."
        ),
    )
    parser.add_argument(
        "--level2_binary_label_mapping_file",
        default=None,
        help=(
            "Optional CSV/TSV mapping file with columns original_level2_label and "
            "binary_level2_label. Used to collapse detailed Level 2 labels into a "
            "resistant/susceptible endpoint when no dedicated binary column exists."
        ),
    )
    parser.add_argument(
        "--level2_binary_resistant_values",
        default=None,
        help="Comma-separated values interpreted as resistant for the binary Level 2 fallback.",
    )
    parser.add_argument(
        "--level2_binary_susceptible_values",
        default=None,
        help="Comma-separated values interpreted as susceptible for the binary Level 2 fallback.",
    )

    add_config_args(parser)
    add_rf_fdr_args(parser)
    add_logging_args(parser)
    return parser


def build_query_parser(prog: Optional[str] = None, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Apply a trained two-level NetworkParser registry or binary model bundle to new strain/sample input.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=add_help,
    )

    parser.add_argument("--genomic", required=True, help="New genomic input file or directory: VCF/VCF.gz/CSV/TSV/FASTA/FASTQ directory.")
    parser.add_argument(
        "--registry",
        default=None,
        help="Path to two_level_model_registry.json from training. For backward compatibility, a .npb path here is treated as --bundle.",
    )
    parser.add_argument(
        "--bundle",
        default=None,
        help="Path to networkparser_model_bundle.npb. Preferred for portable end-to-end query inference.",
    )
    parser.add_argument("--output_dir", required=True, help="Prediction output directory.")
    parser.add_argument("--ref_fasta", default=None, help="Optional reference FASTA/GenBank context for VCF-oriented workflows.")
    parser.add_argument("--max_markers", type=int, default=10, help="Maximum supporting markers to report per prediction level.")
    parser.add_argument(
        "--query_input_type",
        choices=["auto", "matrix", "vcf", "fasta", "raw_sequence", "fastq"],
        default="auto",
        help="How to interpret --genomic. Use fasta for FASTA DNA queries or fastq for paired-end FASTQ directories. raw_sequence is kept as a deprecated alias for fasta.",
    )
    parser.add_argument(
        "--fasta_mapping_mode",
        "--raw_sequence_mapping_mode",
        dest="raw_sequence_mapping_mode",
        choices=["auto", "blast", "exact"],
        default="auto",
        help="How FASTA query sequences are mapped to selected feature contexts. The old --raw_sequence_mapping_mode option remains as an alias.",
    )
    parser.add_argument(
        "--fastq_max_parallel_samples",
        type=int,
        default=None,
        help="Maximum number of paired FASTQ samples to align/call concurrently during FASTQ query mode.",
    )
    parser.add_argument(
        "--fastq_threads",
        type=int,
        default=None,
        help="Total threads available to FASTQ preprocessing. Per-sample threads are derived from this value.",
    )
    parser.add_argument(
        "--fastq_memory_per_sample_mb",
        type=int,
        default=None,
        help="Optional virtual-memory limit per FASTQ sample in MB. Leave unset unless your scheduler requires it.",
    )
    parser.add_argument(
        "--fastq_clean_intermediates",
        action="store_true",
        help="Remove FASTQ intermediate working files after successful preprocessing. BAMs, VCFs, stats, and logs are preserved.",
    )
    parser.add_argument(
        "--fastq_no_auto_index_reference",
        action="store_true",
        help="Do not automatically create missing BWA/samtools reference indexes for FASTQ query mode.",
    )
    parser.add_argument(
        "--fastq_min_mapping_quality",
        type=int,
        default=None,
        help="Minimum mapping quality passed to bcftools mpileup during FASTQ query mode.",
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

    run = subparsers.add_parser("run", help="Run the single-label NetworkParser workflow.", parents=[build_run_parser(add_help=False)], add_help=True)
    run.set_defaults(command="run")

    train_two = subparsers.add_parser(
        "train-two-level",
        help="Train the two-label strain-identity and resistance-profile protocol.",
        parents=[build_train_two_level_parser(add_help=False)],
        add_help=True,
    )
    train_two.set_defaults(command="train-two-level")

    query = subparsers.add_parser(
        "query",
        help="Run user-facing prediction on new strain/sample input.",
        parents=[build_query_parser(add_help=False)],
        add_help=True,
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
        global_level2_label=getattr(args, "global_level2_label", None),
        ref_fasta=args.ref_fasta,
        algorithm=args.algorithm,
        train_global_level2=not bool(args.no_global_level2),
        min_level2_samples_per_group=args.min_level2_samples_per_group,
    )


def run_query(args: argparse.Namespace) -> Any:
    config = load_config(args.config)
    config = apply_common_overrides(config, args)

    set_if_provided(config, "fastq_max_parallel_samples", getattr(args, "fastq_max_parallel_samples", None))
    set_if_provided(config, "fastq_threads", getattr(args, "fastq_threads", None))
    set_if_provided(config, "fastq_memory_per_sample_mb", getattr(args, "fastq_memory_per_sample_mb", None))
    set_if_provided(config, "fastq_min_mapping_quality", getattr(args, "fastq_min_mapping_quality", None))
    if bool(getattr(args, "fastq_clean_intermediates", False)):
        config.fastq_clean_intermediates = True
    if bool(getattr(args, "fastq_no_auto_index_reference", False)):
        config.fastq_auto_index_reference = False
    if hasattr(config, "__post_init__"):
        config.__post_init__()

    registry_path = getattr(args, "registry", None)
    bundle_path = getattr(args, "bundle", None)

    # Backward-compatible convenience: allow users to pass the binary bundle
    # through --registry while newer commands use the clearer --bundle flag.
    if registry_path and str(registry_path).lower().endswith(".npb") and not bundle_path:
        bundle_path = registry_path
        registry_path = None

    if bool(registry_path) == bool(bundle_path):
        raise ValueError(
            "Query mode requires exactly one trained model source: "
            "provide --bundle networkparser_model_bundle.npb or --registry two_level_model_registry.json."
        )

    if bundle_path:
        LOGGER.info("Starting NetworkParser bundled query workflow")
        try:
            from network_parser.model_bundle import query_bundle
        except Exception:  # pragma: no cover - supports direct source-tree execution
            from model_bundle import query_bundle  # type: ignore

        return query_bundle(
            bundle_path=bundle_path,
            genomic_path=args.genomic,
            output_dir=args.output_dir,
            config=config,
            ref_fasta=args.ref_fasta,
            max_markers=int(args.max_markers),
            n_jobs=args.n_jobs,
            query_input_type=args.query_input_type,
            raw_sequence_mapping_mode=args.raw_sequence_mapping_mode,
        )

    LOGGER.info("Starting NetworkParser registry query workflow")
    engine = NetworkParserQueryEngine(registry_path=registry_path, config=config)
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

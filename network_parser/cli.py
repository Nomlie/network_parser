#!/usr/bin/env python3
# network_parser/cli.py
"""
NetworkParser command-line interface.

Supports four entry points:

1) run
   Single-label NetworkParser workflow:
   load -> align -> RF-FDR central feature filtering -> ML protocol/model selector
   -> conditional decision-tree interpretability -> optional downstream validation.

2) train-hierarchy  (alias: train-two-level)
   Two-label / recursive hierarchical protocol:
   Level 1: strain / lineage / group placement
   Level 2+: phenotype / AMR-profile / terminal endpoint prediction

3) bundle
   Package a trained registry into a portable .npb binary model bundle.

4) query
   User-facing inference on a new strain/sample using a saved registry or bundle.

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
from typing import Any, Dict, List, Optional, Tuple

try:
    from network_parser.config import NetworkParserConfig
    from network_parser.network_parser import run_networkparser_analysis
    from network_parser.hierarchy_protocol import HierarchyProtocol
    from network_parser.query_engine import NetworkParserQueryEngine
except ImportError:  # pragma: no cover - allows direct execution from source tree
    from config import NetworkParserConfig  # type: ignore
    from network_parser import run_networkparser_analysis  # type: ignore
    from hierarchy_protocol import HierarchyProtocol  # type: ignore
    from query_engine import NetworkParserQueryEngine  # type: ignore


LOGGER = logging.getLogger(__name__)
VALID_PIPELINE_MODES = {"matrix_only", "decision_tree_only", "ml_only", "both"}
VALID_SUBCOMMANDS = {
    "run",
    "train-hierarchy",
    "train-two-level",
    "bundle",
    "query",
    "evaluate",
    "evaluate-hierarchy",
    "cross-validate",
    "cross_validation",
    "annotate-panels",
}


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


def apply_common_overrides(
    config: NetworkParserConfig, args: argparse.Namespace
) -> NetworkParserConfig:
    """Apply CLI overrides that are shared across commands."""
    set_if_provided(config, "n_jobs", getattr(args, "n_jobs", None))
    set_if_provided(
        config,
        "central_feature_filter_method",
        getattr(args, "central_feature_filter_method", None),
    )

    # RF-FDR selector controls. These are optional so config files remain the
    # canonical place for tuned runs, while CLI remains convenient for testing.
    set_if_provided(
        config,
        "rf_selector_n_estimators",
        getattr(args, "rf_selector_n_estimators", None),
    )
    set_if_provided(
        config,
        "rf_selector_n_observed_repeats",
        getattr(args, "rf_selector_n_observed_repeats", None),
    )
    set_if_provided(
        config,
        "rf_selector_n_permutations",
        getattr(args, "rf_selector_n_permutations", None),
    )
    set_if_provided(
        config, "rf_selector_fdr_alpha", getattr(args, "rf_selector_fdr_alpha", None)
    )
    set_if_provided(
        config,
        "rf_selector_random_state",
        getattr(args, "rf_selector_random_state", None),
    )
    set_if_provided(
        config, "rf_selector_top_n", getattr(args, "rf_selector_top_n", None)
    )
    set_if_provided(
        config,
        "rf_selector_min_importance",
        getattr(args, "rf_selector_min_importance", None),
    )
    set_if_provided(
        config,
        "rf_selector_fallback_strategy",
        getattr(args, "rf_selector_fallback_strategy", None),
    )
    set_if_provided(
        config,
        "rf_selector_fallback_top_n",
        getattr(args, "rf_selector_fallback_top_n", None),
    )
    set_if_provided(
        config,
        "feature_filter_fallback_strategy",
        getattr(args, "feature_filter_fallback_strategy", None),
    )
    set_if_provided(
        config, "n_permutation_tests", getattr(args, "n_permutation_tests", None)
    )
    set_if_provided(config, "fdr_alpha", getattr(args, "fdr_alpha", None))
    set_if_provided(
        config,
        "multiple_testing_method",
        getattr(args, "multiple_testing_method", None),
    )

    feature_panel_check = getattr(args, "feature_panel_check", None)
    if feature_panel_check is not None:
        config.run_feature_panel_separability_check = feature_panel_check == "on"
    set_if_provided(
        config, "feature_panel_sizes", getattr(args, "feature_panel_sizes", None)
    )
    set_if_provided(
        config, "feature_panel_metric", getattr(args, "feature_panel_metric", None)
    )
    set_if_provided(
        config,
        "feature_panel_classifier",
        getattr(args, "feature_panel_classifier", None),
    )
    set_if_provided(
        config,
        "feature_panel_lr_max_iter",
        getattr(args, "feature_panel_lr_max_iter", None),
    )
    set_if_provided(
        config, "feature_panel_lr_tol", getattr(args, "feature_panel_lr_tol", None)
    )
    set_if_provided(
        config,
        "feature_panel_rf_n_estimators",
        getattr(args, "feature_panel_rf_n_estimators", None),
    )
    set_if_provided(
        config,
        "feature_panel_rf_max_features",
        getattr(args, "feature_panel_rf_max_features", None),
    )
    set_if_provided(
        config,
        "feature_panel_rf_min_samples_leaf",
        getattr(args, "feature_panel_rf_min_samples_leaf", None),
    )
    set_if_provided(
        config,
        "feature_panel_rf_class_weight",
        getattr(args, "feature_panel_rf_class_weight", None),
    )
    set_if_provided(
        config,
        "feature_panel_rf_n_jobs",
        getattr(args, "feature_panel_rf_n_jobs", None),
    )
    set_if_provided(
        config,
        "feature_panel_min_score",
        getattr(args, "feature_panel_min_score", None),
    )
    set_if_provided(
        config,
        "feature_panel_selection_rule",
        getattr(args, "feature_panel_selection_rule", None),
    )
    set_if_provided(
        config,
        "feature_panel_cv_splits",
        getattr(args, "feature_panel_cv_splits", None),
    )

    set_if_provided(
        config, "global_level2_label_column", getattr(args, "global_level2_label", None)
    )

    if bool(getattr(args, "keep_low_support_classes", False)):
        config.level2_drop_low_support_classes = False
    elif bool(getattr(args, "level2_drop_low_support_classes", False)):
        config.level2_drop_low_support_classes = True
    set_if_provided(
        config, "level2_min_class_count", getattr(args, "level2_min_class_count", None)
    )

    set_if_provided(
        config,
        "hierarchy_global_fallback_labels",
        getattr(args, "global_fallback_labels", None),
    )
    if bool(getattr(args, "no_parent_conditioned_fallbacks", False)):
        config.hierarchy_train_parent_conditioned_fallbacks = False
    if bool(getattr(args, "no_hierarchy_global_lineage_fallback", False)):
        config.hierarchy_train_global_lineage_fallback = False
    set_if_provided(
        config,
        "hierarchy_global_lineage_fallback_label",
        getattr(args, "hierarchy_global_lineage_fallback_label", None),
    )
    if bool(
        getattr(args, "no_hierarchy_global_lineage_low_confidence_fallback", False)
    ):
        config.hierarchy_global_lineage_fallback_on_low_confidence = False
    if bool(getattr(args, "no_hierarchy_global_lineage_disagreement_fallback", False)):
        config.hierarchy_global_lineage_fallback_on_disagreement = False
    set_if_provided(
        config,
        "hierarchy_global_lineage_fallback_min_support_delta",
        getattr(args, "hierarchy_global_lineage_fallback_min_support_delta", None),
    )

    if bool(getattr(args, "level2_train_binary_global_fallback", False)):
        config.level2_train_binary_global_fallback = True
    apply_low_support_review_overrides(config, args)
    apply_amr_evidence_guard_overrides(config, args)
    apply_performance_overrides(config, args)
    set_if_provided(
        config,
        "level2_binary_label_column",
        getattr(args, "level2_binary_label_column", None),
    )
    set_if_provided(
        config,
        "level2_binary_label_mapping_file",
        getattr(args, "level2_binary_label_mapping_file", None),
    )
    set_if_provided(
        config,
        "level2_binary_resistant_values",
        getattr(args, "level2_binary_resistant_values", None),
    )
    set_if_provided(
        config,
        "level2_binary_susceptible_values",
        getattr(args, "level2_binary_susceptible_values", None),
    )

    if hasattr(config, "__post_init__"):
        config.__post_init__()
    return config


def add_logging_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--quiet", action="store_true", help="Only show warnings and errors."
    )


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=None,
        help="Optional JSON file with NetworkParserConfig overrides.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=None,
        help="Number of parallel workers where supported.",
    )


def add_amr_evidence_guard_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("AMR evidence guard")
    group.add_argument(
        "--no_amr_weak_evidence_review",
        action="store_true",
        help=(
            "Disable the query-time AMR weak-evidence guard entirely "
            "(no warn flags and no blocking of susceptible calls)."
        ),
    )
    group.add_argument(
        "--amr_weak_evidence_mode",
        choices=["warn", "block"],
        default=None,
        help=(
            "How to handle weak-evidence susceptible AMR calls: "
            "'warn' keeps the class label (cleaner TP/FP/TN/FN) and attaches a reason; "
            "'block' replaces the prediction with the review label (legacy)."
        ),
    )
    group.add_argument(
        "--amr_weak_evidence_min_resolved_fraction",
        type=float,
        default=None,
        help=(
            "Minimum fraction of branch AMR panel features that must resolve before "
            "a susceptible call is reported without a weak-evidence warning."
        ),
    )
    group.add_argument(
        "--amr_weak_evidence_review_label",
        default=None,
        help=(
            "Label used only when --amr_weak_evidence_mode=block replaces a weak "
            "susceptible call (default: amr_evidence_review_required)."
        ),
    )
    group.add_argument(
        "--no_hierarchy_global_amr_fallback_on_weak_evidence",
        action="store_true",
        help=(
            "Do not try lineage/global terminal AMR fallback models before blocking "
            "weak-evidence susceptible branch predictions."
        ),
    )
    group.add_argument(
        "--hierarchy_global_amr_fallback_min_resistant_probability",
        type=float,
        default=None,
        help=(
            "Minimum resistant probability required from a terminal AMR fallback "
            "model before it overrides a weak-evidence susceptible branch call."
        ),
    )
    group.add_argument(
        "--amr_evidence_guard_label_columns",
        default=None,
        help="Comma-separated metadata label columns guarded by the AMR evidence policy.",
    )


def apply_amr_evidence_guard_overrides(
    config: NetworkParserConfig, args: argparse.Namespace
) -> None:
    if bool(getattr(args, "no_amr_weak_evidence_review", False)):
        config.amr_weak_evidence_review_enabled = False
    set_if_provided(
        config,
        "amr_weak_evidence_mode",
        getattr(args, "amr_weak_evidence_mode", None),
    )
    set_if_provided(
        config,
        "amr_weak_evidence_min_resolved_fraction",
        getattr(args, "amr_weak_evidence_min_resolved_fraction", None),
    )
    set_if_provided(
        config,
        "amr_weak_evidence_review_label",
        getattr(args, "amr_weak_evidence_review_label", None),
    )
    if bool(getattr(args, "no_hierarchy_global_amr_fallback_on_weak_evidence", False)):
        config.hierarchy_global_amr_fallback_on_weak_evidence = False
    set_if_provided(
        config,
        "hierarchy_global_amr_fallback_min_resistant_probability",
        getattr(args, "hierarchy_global_amr_fallback_min_resistant_probability", None),
    )
    set_if_provided(
        config,
        "amr_evidence_guard_label_columns",
        getattr(args, "amr_evidence_guard_label_columns", None),
    )


def add_low_support_review_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Low-support review reporting")
    group.add_argument(
        "--no_low_support_review",
        action="store_true",
        help=(
            "Disable query-time low-support review labels. By default, rare-class "
            "predictions are reported as low_support_review_required instead of "
            "merging rare groups."
        ),
    )
    group.add_argument(
        "--low_support_review_min_class_count",
        type=int,
        default=None,
        help=(
            "Minimum training samples per class required for confident reporting at "
            "query time. Classes below this threshold are flagged for manual review."
        ),
    )
    group.add_argument(
        "--low_support_review_label",
        default=None,
        help="Label reported when a prediction maps to a low-support training class.",
    )


def apply_low_support_review_overrides(
    config: NetworkParserConfig, args: argparse.Namespace
) -> None:
    if bool(getattr(args, "no_low_support_review", False)):
        config.low_support_review_enabled = False
    set_if_provided(
        config,
        "low_support_review_min_class_count",
        getattr(args, "low_support_review_min_class_count", None),
    )
    set_if_provided(
        config,
        "low_support_review_label",
        getattr(args, "low_support_review_label", None),
    )


def add_performance_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Performance controls")
    group.add_argument(
        "--no_query_parallel_samples",
        action="store_true",
        help="Disable parallel per-sample hierarchy query traversal.",
    )
    group.add_argument(
        "--query_parallel_n_jobs",
        type=int,
        default=None,
        help="Worker count for parallel query traversal. Defaults to --n_jobs when unset.",
    )
    group.add_argument(
        "--no_hierarchy_parallel_fallback_training",
        action="store_true",
        help="Train hierarchy terminal fallback models sequentially instead of in parallel.",
    )
    group.add_argument(
        "--no_level2_parallel_group_training",
        action="store_true",
        help="Train Level-2 group-specific models sequentially instead of in parallel.",
    )
    group.add_argument(
        "--no_hierarchy_parallel_child_nodes",
        action="store_true",
        help=(
            "Train sibling hierarchy child nodes sequentially. "
            "By default children train in parallel when RAM/CPU allow."
        ),
    )
    group.add_argument(
        "--no_feature_panel_parallel_scoring",
        action="store_true",
        help="Score feature-panel candidate sizes sequentially instead of in parallel.",
    )
    group.add_argument(
        "--parallel_memory_per_worker_gb",
        type=float,
        default=None,
        help=(
            "Soft RAM budget per concurrent outer model fit (default 4). "
            "Caps parallel model training on small machines (e.g. 16 GB → ~1–2 workers)."
        ),
    )
    group.add_argument(
        "--parallel_max_workers",
        type=int,
        default=None,
        help="Hard cap on parallel workers regardless of --n_jobs / CPU count.",
    )
    group.add_argument(
        "--association_test_batch_size",
        type=int,
        default=None,
        help="Features dispatched per association-test parallel batch.",
    )


def apply_performance_overrides(
    config: NetworkParserConfig, args: argparse.Namespace
) -> None:
    if bool(getattr(args, "no_query_parallel_samples", False)):
        config.query_parallel_samples = False
    set_if_provided(
        config, "query_parallel_n_jobs", getattr(args, "query_parallel_n_jobs", None)
    )
    if bool(getattr(args, "no_hierarchy_parallel_fallback_training", False)):
        config.hierarchy_parallel_fallback_training = False
    if bool(getattr(args, "no_level2_parallel_group_training", False)):
        config.level2_parallel_group_training = False
    if bool(getattr(args, "no_hierarchy_parallel_child_nodes", False)):
        config.hierarchy_parallel_child_nodes = False
    if bool(getattr(args, "no_feature_panel_parallel_scoring", False)):
        config.feature_panel_parallel_scoring = False
    set_if_provided(
        config,
        "parallel_memory_per_worker_gb",
        getattr(args, "parallel_memory_per_worker_gb", None),
    )
    set_if_provided(
        config,
        "parallel_max_workers",
        getattr(args, "parallel_max_workers", None),
    )
    set_if_provided(
        config,
        "association_test_batch_size",
        getattr(args, "association_test_batch_size", None),
    )


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
    group.add_argument(
        "--n_permutation_tests",
        type=int,
        default=None,
        help="Label permutations for chi2_perm_fdr and downstream permutation utilities.",
    )
    group.add_argument(
        "--fdr_alpha",
        type=float,
        default=None,
        help="FDR alpha used by association-FDR and chi2_perm_fdr.",
    )
    group.add_argument(
        "--multiple_testing_method",
        choices=["fdr_bh", "bonferroni"],
        default=None,
        help="Multiple-testing correction method for association-FDR and chi2_perm_fdr.",
    )
    group.add_argument(
        "--rf_selector_n_estimators",
        type=int,
        default=None,
        help="Random Forest trees used during RF-FDR scoring.",
    )
    group.add_argument(
        "--rf_selector_n_observed_repeats",
        type=int,
        default=None,
        help="Observed RF importance repeats.",
    )
    group.add_argument(
        "--rf_selector_n_permutations",
        type=int,
        default=None,
        help="Label permutations for empirical p-values.",
    )
    group.add_argument(
        "--rf_selector_fdr_alpha",
        type=float,
        default=None,
        help="FDR-BH alpha for RF-FDR retention.",
    )
    group.add_argument(
        "--rf_selector_random_state",
        type=int,
        default=None,
        help="Random seed for RF-FDR.",
    )
    group.add_argument(
        "--rf_selector_top_n",
        type=int,
        default=None,
        help="Optional cap on retained RF-FDR features.",
    )
    group.add_argument(
        "--rf_selector_min_importance",
        type=float,
        default=None,
        help="Minimum observed RF importance for retained features.",
    )
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
        choices=[
            "balanced_accuracy",
            "adjusted_rand",
            "normalized_mutual_info",
            "silhouette",
        ],
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


def build_run_parser(
    prog: Optional[str] = None, add_help: bool = True
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run the single-label NetworkParser workflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=add_help,
    )

    parser.add_argument(
        "--genomic",
        required=True,
        help="Genomic input file or directory: VCF/VCF.gz/CSV/TSV.",
    )
    parser.add_argument(
        "--meta",
        required=True,
        help="Metadata CSV/TSV containing the supervised label column.",
    )
    parser.add_argument(
        "--label", required=True, help="Metadata column used as the supervised target."
    )
    parser.add_argument(
        "--known_markers",
        default=None,
        help=(
            "Optional known-marker file for comparison against the feature space only. "
            "Writes known_markers_feature_overlap.json; does NOT filter or train on these markers."
        ),
    )
    parser.add_argument(
        "--ref_fasta",
        default=None,
        help="Optional reference FASTA/GenBank context for VCF-oriented workflows.",
    )
    parser.add_argument("--output_dir", required=True, help="Output directory.")

    parser.add_argument(
        "--pipeline_mode",
        default="both",
        choices=sorted(VALID_PIPELINE_MODES),
        help="Workflow mode.",
    )
    parser.add_argument(
        "--validate_statistics",
        action="store_true",
        help=(
            "Deprecated alias: forces run_central_feature_filtering=True. "
            "Prefer --disable_central_feature_filtering / config.run_central_feature_filtering."
        ),
    )
    parser.add_argument(
        "--validate_interactions",
        action="store_true",
        help="Run optional post-tree interaction validation when available.",
    )

    parser.add_argument(
        "--run_ml_protocol",
        action="store_true",
        help="Force ML protocol branch on through config.",
    )
    parser.add_argument(
        "--disable_central_feature_filtering",
        action="store_true",
        help="Pass aligned matrix forward without central feature filtering.",
    )
    parser.add_argument(
        "--disable_model_selector",
        action="store_true",
        help="Disable automatic model-selector behaviour where supported.",
    )
    parser.add_argument(
        "--disable_conditional_dt",
        action="store_true",
        help="Prevent selector-driven decision-tree triggering where supported.",
    )

    parser.add_argument(
        "--ml_algorithm",
        default=None,
        help="Optional ML algorithm override, e.g. auto, RF, MLP, LR, DT, SVC, MBCS, DNL.",
    )
    parser.add_argument(
        "--ml_min_sensitivity",
        type=float,
        default=None,
        help="Optional ML protocol sensitivity lower bound.",
    )
    parser.add_argument(
        "--ml_max_sensitivity",
        type=float,
        default=None,
        help="Optional ML protocol sensitivity upper bound.",
    )
    parser.add_argument(
        "--ml_step_sensitivity",
        type=float,
        default=None,
        help="Optional ML protocol sensitivity step.",
    )
    parser.add_argument(
        "--ml_empty_symbol",
        default=None,
        help="Optional empty symbol for ML-formatted data.",
    )
    parser.add_argument(
        "--ml_remove_empty_field_threshold",
        type=float,
        default=None,
        help="Optional empty-field removal threshold.",
    )

    add_config_args(parser)
    add_low_support_review_args(parser)
    add_amr_evidence_guard_args(parser)
    add_performance_args(parser)
    add_rf_fdr_args(parser)
    add_logging_args(parser)
    return parser


def build_train_hierarchy_parser(
    prog: Optional[str] = None, add_help: bool = True
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Train the hierarchical NetworkParser protocol (2+ levels): placement first, phenotype endpoints under parent branches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=add_help,
    )

    parser.add_argument(
        "--genomic",
        required=True,
        help="Genomic input file or directory: VCF/VCF.gz/CSV/TSV.",
    )
    parser.add_argument(
        "--meta",
        required=True,
        help="Metadata CSV/TSV containing both supervised labels.",
    )
    parser.add_argument(
        "--level1_label",
        default=None,
        help="Metadata column for first-level strain/lineage/group placement. Required unless --hierarchy_labels is used.",
    )
    parser.add_argument(
        "--level2_label",
        default=None,
        help="Metadata column for second-level phenotype/profile. Required unless --hierarchy_labels is used.",
    )
    parser.add_argument(
        "--hierarchy_labels",
        nargs="+",
        default=None,
        help=(
            "Ordered metadata columns for true recursive hierarchy training. "
            "When provided, this supersedes --level1_label/--level2_label and trains one model per hierarchy node."
        ),
    )
    parser.add_argument(
        "--hierarchy_preset",
        default=None,
        choices=[
            "lineage_amr_profile",
            "lineage_family_amr_profile",
            "lineage_amr_binary",
        ],
        help=(
            "Biological hierarchy preset (no artificial Lineage_Supergroup). "
            "lineage_amr_profile: Lineage_clean→AMR_binary→Resistance_Profile_Collapsed; "
            "lineage_family_amr_profile: Lineage_family→Lineage_clean→AMR_binary→Resistance_Profile_Collapsed; "
            "lineage_amr_binary: Lineage_clean→AMR_binary. "
            "Can be combined with --hierarchy_labels (labels win if both set)."
        ),
    )
    parser.add_argument(
        "--hierarchy_resume",
        action="store_true",
        help="Skip nodes that already have node_summary.json + model under the output directory.",
    )
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
    parser.add_argument(
        "--ref_fasta",
        default=None,
        help="Optional reference FASTA/GenBank context for VCF-oriented workflows.",
    )
    parser.add_argument(
        "--algorithm",
        default=None,
        help="Optional ML algorithm override passed to the ML protocol.",
    )
    parser.add_argument(
        "--no_global_level2",
        action="store_true",
        help="Disable the global Level 2 fallback model.",
    )
    parser.add_argument(
        "--min_level2_samples_per_group",
        type=int,
        default=None,
        help=(
            "Optional absolute minimum samples for group-specific Level 2 models. "
            "When unset, eligibility is adaptive and scales with the number of Level 2 labels."
        ),
    )
    parser.add_argument(
        "--level2_drop_low_support_classes",
        action="store_true",
        help=(
            "Force dropping of low-support classes before hierarchy-node and Level-2 "
            "training. Enabled by default in the current architecture."
        ),
    )
    parser.add_argument(
        "--keep_low_support_classes",
        action="store_true",
        help=(
            "Retain singleton or very rare classes during hierarchy-node and Level-2 "
            "training instead of applying the default low-support class filter."
        ),
    )
    parser.add_argument(
        "--level2_min_class_count",
        type=int,
        default=None,
        help="Minimum samples per class required when low-support class dropping is enabled.",
    )
    parser.add_argument(
        "--global_fallback_labels",
        default=None,
        help=(
            "Which hierarchy levels get cohort-wide global models. "
            "Comma-separated label names, or special tokens: "
            "none (default; no globals), terminal (last hierarchy label only), "
            "lineage (resolved lineage column), legacy (old defaults: terminal+lineage). "
            "Example: --global_fallback_labels Lineage_clean,AMR_binary"
        ),
    )
    parser.add_argument(
        "--no_parent_conditioned_fallbacks",
        action="store_true",
        help=(
            "Do not train parent-conditioned fallbacks "
            "(e.g. terminal phenotype within each lineage). "
            "Path-local hierarchy nodes still train."
        ),
    )
    parser.add_argument(
        "--no_hierarchy_global_lineage_fallback",
        action="store_true",
        help=(
            "Disable the dedicated global lineage fallback model even if lineage "
            "appears in --global_fallback_labels / legacy mode."
        ),
    )
    parser.add_argument(
        "--hierarchy_global_lineage_fallback_label",
        default=None,
        help=(
            "Metadata column for the global lineage fallback model. When unset, "
            "NetworkParser resolves a lineage-like label from --hierarchy_labels."
        ),
    )
    parser.add_argument(
        "--no_hierarchy_global_lineage_low_confidence_fallback",
        action="store_true",
        help="Do not replace low-confidence branch lineage predictions with the global lineage model.",
    )
    parser.add_argument(
        "--no_hierarchy_global_lineage_disagreement_fallback",
        action="store_true",
        help=(
            "Do not replace branch lineage predictions when the global lineage model "
            "disagrees and has stronger support."
        ),
    )
    parser.add_argument(
        "--hierarchy_global_lineage_fallback_min_support_delta",
        type=float,
        default=None,
        help=(
            "Minimum support advantage required before a disagreeing global lineage "
            "prediction overrides a branch prediction."
        ),
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
    parser.add_argument(
        "--bundle_output",
        default=None,
        help=(
            "Optional output path for the automatically created .npb model bundle. "
            "Default: <output_dir>/networkparser_model_bundle.npb."
        ),
    )
    parser.add_argument(
        "--no_model_bundle",
        action="store_true",
        help="Do not automatically create networkparser_model_bundle.npb after training.",
    )

    add_config_args(parser)
    add_low_support_review_args(parser)
    add_amr_evidence_guard_args(parser)
    add_performance_args(parser)
    add_rf_fdr_args(parser)
    add_logging_args(parser)
    return parser


def build_query_parser(
    prog: Optional[str] = None, add_help: bool = True
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Apply a trained hierarchical NetworkParser registry or binary model bundle to new strain/sample input.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=add_help,
    )

    parser.add_argument(
        "--genomic",
        required=True,
        help="New genomic input file or directory: VCF/VCF.gz/CSV/TSV/FASTA/FASTQ directory.",
    )
    parser.add_argument(
        "--registry",
        default=None,
        help="Path to hierarchy_model_registry.json / two_level_model_registry.json / hierarchical_model_registry.json from training. For backward compatibility, a .npb path here is treated as --bundle.",
    )
    parser.add_argument(
        "--bundle",
        default=None,
        help="Path to networkparser_model_bundle.npb. Preferred for portable end-to-end query inference.",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Prediction output directory."
    )
    parser.add_argument(
        "--ref_fasta",
        default=None,
        help="Optional reference FASTA/GenBank context for VCF-oriented workflows.",
    )
    parser.add_argument(
        "--max_markers",
        type=int,
        default=10,
        help="Maximum supporting markers to report per prediction level.",
    )
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
    add_low_support_review_args(parser)
    add_amr_evidence_guard_args(parser)
    add_performance_args(parser)
    add_logging_args(parser)
    return parser


def build_evaluate_parser(
    prog: Optional[str] = None, add_help: bool = True
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Evaluate saved NetworkParser query predictions against labelled metadata. "
            "This is evaluation-only and does not rerun feature filtering, model training, "
            "decision-tree construction, or bootstrap confidence scoring."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=add_help,
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to query_predictions.csv or another saved prediction table.",
    )
    parser.add_argument(
        "--meta",
        required=True,
        help="Path to metadata file containing the true labels.",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for evaluation artifacts."
    )
    parser.add_argument(
        "--label", default=None, help="Single metadata label column to evaluate."
    )
    parser.add_argument(
        "--hierarchy_labels",
        nargs="+",
        default=None,
        help="Ordered metadata label columns to evaluate, for example: Lineage Resistance_Profile AMR_binary",
    )
    parser.add_argument(
        "--global_level2_label",
        default=None,
        help=(
            "Optional truth label for standard two-level/hierarchy global fallback evaluation. "
            "When supplied with --hierarchy_labels, it replaces the second requested "
            "truth label during evaluation only."
        ),
    )
    parser.add_argument(
        "--predicted_column",
        default=None,
        help="Optional explicit prediction column when evaluating a single label.",
    )
    parser.add_argument(
        "--sample_id_column",
        default=None,
        help="Optional sample-id column in metadata. If omitted, common sample-id column names are auto-detected.",
    )
    parser.add_argument(
        "--skip_missing_prediction_levels",
        action="store_true",
        help="Skip a requested hierarchy level if the expected prediction column is absent.",
    )
    add_logging_args(parser)
    return parser


def build_cross_validate_parser(
    prog: Optional[str] = None, add_help: bool = True
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Run leakage-aware repeated cross-validation for one supervised label. "
            "Each fold performs feature filtering, feature-panel selection, and model training "
            "inside the training split only."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=add_help,
    )
    parser.add_argument(
        "--genomic", required=True, help="Training genomic input file or directory."
    )
    parser.add_argument(
        "--meta", required=True, help="Metadata file containing the supervised label."
    )
    parser.add_argument(
        "--label", required=True, help="Metadata label column to cross-validate."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for repeated cross-validation artifacts.",
    )
    parser.add_argument(
        "--ref_fasta",
        default=None,
        help="Optional reference FASTA/GenBank context for VCF-oriented workflows.",
    )
    parser.add_argument(
        "--algorithm",
        default=None,
        help="Optional fixed downstream algorithm. Leave unset to use the normal selector pathway.",
    )
    parser.add_argument(
        "--n_repeats", type=int, default=3, help="Number of repeated CV rounds."
    )
    parser.add_argument(
        "--n_splits", type=int, default=5, help="Requested stratified folds per repeat."
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=None,
        help="Random seed for repeat/fold generation.",
    )
    add_rf_fdr_args(parser)
    add_config_args(parser)
    add_performance_args(parser)
    add_logging_args(parser)
    return parser


def build_bundle_parser(
    prog: Optional[str] = None, add_help: bool = True
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Package a trained NetworkParser registry into a portable .npb model bundle. "
            "This is a deployment/inference artifact and does not rerun feature filtering, "
            "model selection, tree construction, or bootstrap confidence scoring."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=add_help,
    )
    parser.add_argument(
        "--registry",
        required=True,
        help="Path to hierarchy/two-level/hierarchical model registry JSON from training.",
    )
    parser.add_argument(
        "--output",
        "--output_bundle",
        dest="output",
        default=None,
        help=(
            "Output .npb path. If omitted, networkparser_model_bundle.npb is written "
            "next to the registry."
        ),
    )
    parser.add_argument(
        "--no_model_payloads",
        action="store_true",
        help=(
            "Do not embed trained model objects. Usually leave this off; without embedded "
            "models the bundle is not fully portable for query inference."
        ),
    )
    parser.add_argument(
        "--no_feature_manifests",
        action="store_true",
        help="Do not embed selected-feature manifests/context tables.",
    )
    parser.add_argument(
        "--no_ranked_feature_tables",
        action="store_true",
        help="Do not embed ranked feature-result tables used for supporting-marker ranking.",
    )
    add_logging_args(parser)
    return parser


def build_top_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NetworkParser command-line interface.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run",
        help="Run the single-label NetworkParser workflow.",
        parents=[build_run_parser(add_help=False)],
        add_help=True,
    )
    run.set_defaults(command="run")

    train_hier = subparsers.add_parser(
        "train-hierarchy",
        help="Train hierarchical models (2+ levels): placement and nested phenotype endpoints.",
        parents=[build_train_hierarchy_parser(add_help=False)],
        add_help=True,
    )
    train_hier.set_defaults(command="train-hierarchy")

    # Backward-compatible alias (same implementation as train-hierarchy)
    train_two = subparsers.add_parser(
        "train-two-level",
        help="Deprecated alias for train-hierarchy (classic two-level or multi-level).",
        parents=[build_train_hierarchy_parser(add_help=False)],
        add_help=True,
    )
    train_two.set_defaults(command="train-hierarchy")

    annotate = subparsers.add_parser(
        "annotate-panels",
        help=(
            "Annotate selected feature panels with genes, predicted consequences "
            "and optional resistance-catalogue labels (post-training biological summary)."
        ),
        add_help=True,
    )
    annotate.add_argument(
        "--registry",
        required=True,
        help="Path to hierarchical_model_registry.json or two_level_model_registry.json",
    )
    annotate.add_argument(
        "--output_dir", required=True, help="Directory for annotation tables"
    )
    annotate.add_argument(
        "--catalogue",
        default=None,
        help="Optional TSV/CSV of known resistance mutations/genes (flexible column names)",
    )
    annotate.add_argument(
        "--stability",
        default=None,
        help="Optional TSV/CSV with Feature_ID and selection_frequency from leakage-aware CV",
    )
    annotate.add_argument(
        "--min_stability",
        type=float,
        default=0.0,
        help="If --stability is set, keep features with selection_frequency >= this value (0-1)",
    )
    annotate.add_argument(
        "--write_stable_report",
        action="store_true",
        help="Write stable_panel_features_annotated.tsv (uses min_stability or 0.5 default).",
    )
    annotate.add_argument(
        "--write_catalogue_circularity",
        action="store_true",
        help="Write catalogue circularity audit (known vs non-catalogue by node).",
    )
    annotate.set_defaults(command="annotate-panels")

    bundle = subparsers.add_parser(
        "bundle",
        help="Package a trained registry into a portable .npb model bundle.",
        parents=[build_bundle_parser(add_help=False)],
        add_help=True,
    )
    bundle.set_defaults(command="bundle")

    query = subparsers.add_parser(
        "query",
        help="Run user-facing prediction on new strain/sample input.",
        parents=[build_query_parser(add_help=False)],
        add_help=True,
    )
    query.set_defaults(command="query")

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate saved query predictions against labelled metadata.",
        parents=[build_evaluate_parser(add_help=False)],
        add_help=True,
    )
    evaluate.set_defaults(command="evaluate")

    evaluate_hier = subparsers.add_parser(
        "evaluate-hierarchy",
        help=(
            "Hierarchy evaluation pack: per-level metrics/confusions, full-path "
            "accuracy, bootstrap CIs, optional resistance-label harmonization."
        ),
        add_help=True,
    )
    evaluate_hier.add_argument(
        "--predictions",
        required=True,
        help="query_predictions.csv from hierarchy query",
    )
    evaluate_hier.add_argument(
        "--meta",
        required=True,
        help="Metadata with hierarchy label columns",
    )
    evaluate_hier.add_argument(
        "--hierarchy_labels",
        nargs="+",
        required=True,
        help="Ordered hierarchy label columns (same as training)",
    )
    evaluate_hier.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for the evaluation pack",
    )
    evaluate_hier.add_argument(
        "--sample_id_column",
        default=None,
        help="Optional sample-id column in predictions",
    )
    evaluate_hier.add_argument(
        "--harmonize_resistance_labels",
        action="store_true",
        help="Map susceptible/Sensitive (and similar) for fair resistance scoring",
    )
    evaluate_hier.add_argument(
        "--n_bootstrap",
        type=int,
        default=500,
        help="Bootstrap resamples for principal metric CIs",
    )
    evaluate_hier.set_defaults(command="evaluate-hierarchy")

    cross_validate = subparsers.add_parser(
        "cross-validate",
        aliases=["cross_validation"],
        help="Run leakage-aware repeated cross-validation for one supervised label.",
        parents=[build_cross_validate_parser(add_help=False)],
        add_help=True,
    )
    cross_validate.set_defaults(command="cross-validate")
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
        config.run_central_feature_filtering = bool(
            getattr(config, "run_central_feature_filtering", True)
        )

    if args.disable_model_selector:
        config.run_model_selector = False

    if args.disable_conditional_dt:
        config.run_conditional_dt = False
        config.disable_conditional_dt = True

    set_if_provided(config, "ml_algorithm", args.ml_algorithm)
    set_if_provided(config, "ml_min_sensitivity", args.ml_min_sensitivity)
    set_if_provided(config, "ml_max_sensitivity", args.ml_max_sensitivity)
    set_if_provided(config, "ml_step_sensitivity", args.ml_step_sensitivity)
    set_if_provided(config, "ml_empty_symbol", args.ml_empty_symbol)
    set_if_provided(
        config, "ml_remove_empty_field_threshold", args.ml_remove_empty_field_threshold
    )

    if hasattr(config, "__post_init__"):
        config.__post_init__()

    LOGGER.info(
        "Starting NetworkParser single-label workflow | mode=%s", config.pipeline_mode
    )
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


def _build_training_bundle(
    *,
    args: argparse.Namespace,
    registry_path: Path,
) -> Optional[Path]:
    """Create the default portable .npb bundle after successful training."""
    if bool(getattr(args, "no_model_bundle", False)):
        LOGGER.info(
            "Skipping automatic model bundle creation because --no_model_bundle was supplied."
        )
        return None

    registry_path = Path(registry_path)
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Training registry was expected but not found, so the model bundle cannot be created: {registry_path}"
        )

    output_path = (
        Path(args.bundle_output)
        if getattr(args, "bundle_output", None)
        else Path(args.output_dir) / "networkparser_model_bundle.npb"
    )

    try:
        from network_parser.model_bundle import build_bundle_from_registry
    except ImportError:  # pragma: no cover - supports direct source-tree execution
        from model_bundle import build_bundle_from_registry  # type: ignore

    LOGGER.info(
        "Creating NetworkParser model bundle after training | registry=%s | output=%s",
        registry_path,
        output_path,
    )
    build_bundle_from_registry(
        registry_path=registry_path,
        output_path=output_path,
        include_model_payloads=True,
        include_feature_manifests=True,
        include_ranked_feature_tables=True,
    )
    LOGGER.info(
        "Automatic NetworkParser model bundle complete | output=%s", output_path
    )
    return output_path


def run_train_hierarchy(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_config(args.config)
    config = apply_common_overrides(config, args)
    if bool(getattr(args, "hierarchy_resume", False)):
        config.hierarchy_resume_completed_nodes = True
    preset = getattr(args, "hierarchy_preset", None)
    if preset:
        config.hierarchy_preset = str(preset)

    if hasattr(config, "__post_init__"):
        config.__post_init__()

    protocol = HierarchyProtocol(config=config)

    try:
        from network_parser.hierarchy_artifacts import resolve_hierarchy_labels
    except ImportError:  # pragma: no cover
        from hierarchy_artifacts import resolve_hierarchy_labels  # type: ignore

    hierarchy_labels = getattr(args, "hierarchy_labels", None)
    try:
        resolved_labels = None
        if hierarchy_labels or preset:
            resolved_labels = resolve_hierarchy_labels(
                hierarchy_labels=hierarchy_labels,
                preset=preset,
            )
    except ValueError:
        resolved_labels = list(hierarchy_labels) if hierarchy_labels else None

    if resolved_labels:
        if getattr(args, "global_level2_label", None):
            LOGGER.warning(
                "Ignoring --global_level2_label in recursive hierarchy mode. "
                "Use a terminal hierarchy label such as AMR_binary directly in --hierarchy_labels."
            )
        LOGGER.info(
            "Starting NetworkParser multi-level hierarchy training | labels=%s",
            resolved_labels,
        )
        registry = protocol.train_hierarchy(
            genomic_path=args.genomic,
            meta_path=args.meta,
            hierarchy_labels=list(resolved_labels),
            output_dir=args.output_dir,
            ref_fasta=args.ref_fasta,
            algorithm=args.algorithm,
            min_samples_per_node=args.min_level2_samples_per_group,
        )
        _build_training_bundle(
            args=args,
            registry_path=Path(args.output_dir) / "hierarchical_model_registry.json",
        )
        return registry

    if not args.level1_label or not args.level2_label:
        raise ValueError(
            "train-hierarchy requires either --hierarchy_labels / --hierarchy_preset "
            "with at least two columns, or both --level1_label and --level2_label."
        )

    LOGGER.info("Starting NetworkParser hierarchy training")
    registry = protocol.train(
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
    _build_training_bundle(
        args=args,
        registry_path=Path(args.output_dir) / "two_level_model_registry.json",
    )
    return registry


def run_bundle(args: argparse.Namespace) -> Any:
    registry_path = Path(args.registry)
    output_path = (
        Path(args.output)
        if args.output
        else registry_path.parent / "networkparser_model_bundle.npb"
    )

    try:
        from network_parser.model_bundle import build_bundle_from_registry
    except ImportError:  # pragma: no cover - supports direct source-tree execution
        from model_bundle import build_bundle_from_registry  # type: ignore

    LOGGER.info(
        "Starting NetworkParser bundle build | registry=%s | output=%s",
        registry_path,
        output_path,
    )

    bundle = build_bundle_from_registry(
        registry_path=registry_path,
        output_path=output_path,
        include_model_payloads=not bool(getattr(args, "no_model_payloads", False)),
        include_feature_manifests=not bool(
            getattr(args, "no_feature_manifests", False)
        ),
        include_ranked_feature_tables=not bool(
            getattr(args, "no_ranked_feature_tables", False)
        ),
    )

    LOGGER.info(
        "NetworkParser bundle complete | output=%s | embedded_models=%s | embedded_manifests=%s | required_features=%s",
        output_path,
        bundle.feature_space.get("n_embedded_models"),
        bundle.feature_space.get("n_embedded_manifests"),
        bundle.feature_space.get("required_feature_count"),
    )
    return bundle


def _registry_candidates_for_bundle_dir(bundle_dir: Path) -> List[Path]:
    """Return likely registry JSON paths inside one training output directory."""
    return [
        bundle_dir / "hierarchical_model_registry.json",
        bundle_dir / "two_level_model_registry.json",
    ]


def _find_registry_for_bundle(path: Path) -> Tuple[Optional[Path], List[Path]]:
    """Locate a registry JSON that can rebuild the requested bundle."""
    checked: List[Path] = []
    bundle_dir = path.parent
    run_name = bundle_dir.name

    search_dirs: List[Path] = []
    if bundle_dir.exists():
        search_dirs.append(bundle_dir)
    else:
        base_dir = bundle_dir.parent
        if base_dir.exists():
            prefix = run_name.rstrip("0123456789").rstrip("_") or run_name
            for child in sorted(base_dir.iterdir()):
                if not child.is_dir():
                    continue
                if child.name == run_name or child.name.startswith(prefix):
                    search_dirs.append(child)

    seen_dirs: set[str] = set()
    for directory in search_dirs:
        key = str(directory.resolve())
        if key in seen_dirs:
            continue
        seen_dirs.add(key)
        for candidate in _registry_candidates_for_bundle_dir(directory):
            checked.append(candidate)
            if candidate.exists():
                if directory != bundle_dir:
                    LOGGER.warning(
                        "Requested bundle directory %s was not found. "
                        "Using registry from %s instead.",
                        bundle_dir,
                        directory,
                    )
                return candidate, checked

    return None, checked


def _ensure_query_bundle_available(bundle_path: str | Path) -> Path:
    """Return an existing bundle path, rebuilding from a nearby registry if possible."""
    path = Path(bundle_path)
    if path.exists():
        return path

    registry_path, checked = _find_registry_for_bundle(path)
    if registry_path is None:
        bundle_dir = path.parent
        hint = ""
        base_dir = bundle_dir.parent
        if base_dir.exists():
            siblings = sorted(
                child.name
                for child in base_dir.iterdir()
                if child.is_dir()
                and any(
                    candidate.exists()
                    for candidate in _registry_candidates_for_bundle_dir(child)
                )
            )
            if siblings:
                hint = (
                    " Available trained run directories with registries: "
                    + ", ".join(siblings)
                    + "."
                )
        raise FileNotFoundError(
            "Bundle not found and no nearby registry could be used to rebuild it. "
            f"Expected bundle: {path}; checked registries: "
            + ", ".join(str(candidate) for candidate in checked)
            + hint
            + " Retrain with `train-hierarchy` (alias: `train-two-level`) (bundle is created automatically), "
            "run `network_parser bundle`, or query with --registry instead of --bundle."
        )

    try:
        from network_parser.model_bundle import build_bundle_from_registry
    except ImportError:  # pragma: no cover - supports direct source-tree execution
        from model_bundle import build_bundle_from_registry  # type: ignore

    LOGGER.warning(
        "Bundle not found at %s. Rebuilding it from nearby registry %s before query.",
        path,
        registry_path,
    )
    build_bundle_from_registry(
        registry_path=registry_path,
        output_path=path,
        include_model_payloads=True,
        include_feature_manifests=True,
        include_ranked_feature_tables=True,
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Bundle rebuild completed but bundle is still missing: {path}"
        )
    return path


def run_query(args: argparse.Namespace) -> Any:
    try:
        from network_parser.hierarchy_artifacts import write_resource_profile
    except ImportError:  # pragma: no cover
        from hierarchy_artifacts import write_resource_profile  # type: ignore
    try:
        config_for_log = load_config(getattr(args, "config", None))
        config_for_log = apply_common_overrides(config_for_log, args)
    except Exception:
        config_for_log = None
    write_resource_profile(
        getattr(args, "output_dir", "."),
        config=config_for_log,
        stage="query",
    )
    config = load_config(args.config)
    config = apply_common_overrides(config, args)

    set_if_provided(
        config,
        "fastq_max_parallel_samples",
        getattr(args, "fastq_max_parallel_samples", None),
    )
    set_if_provided(config, "fastq_threads", getattr(args, "fastq_threads", None))
    set_if_provided(
        config,
        "fastq_memory_per_sample_mb",
        getattr(args, "fastq_memory_per_sample_mb", None),
    )
    set_if_provided(
        config,
        "fastq_min_mapping_quality",
        getattr(args, "fastq_min_mapping_quality", None),
    )
    if bool(getattr(args, "fastq_clean_intermediates", False)):
        config.fastq_clean_intermediates = True
    if bool(getattr(args, "fastq_no_auto_index_reference", False)):
        config.fastq_auto_index_reference = False
    apply_performance_overrides(config, args)
    if hasattr(config, "__post_init__"):
        config.__post_init__()

    registry_path = getattr(args, "registry", None)
    bundle_path = getattr(args, "bundle", None)

    # Backward-compatible convenience: allow users to pass the binary bundle
    # through --registry while newer commands use the clearer --bundle flag.
    if (
        registry_path
        and str(registry_path).lower().endswith(".npb")
        and not bundle_path
    ):
        bundle_path = registry_path
        registry_path = None

    if bool(registry_path) == bool(bundle_path):
        raise ValueError(
            "Query mode requires exactly one trained model source: "
            "provide --bundle networkparser_model_bundle.npb or --registry hierarchical_model_registry.json / two_level_model_registry.json."
        )

    if bundle_path:
        bundle_path = _ensure_query_bundle_available(bundle_path)
        LOGGER.info(
            "Starting NetworkParser bundled query workflow | bundle=%s", bundle_path
        )
        try:
            from network_parser.model_bundle import query_bundle
        except ImportError:  # pragma: no cover - supports direct source-tree execution
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


def _read_cli_table(path_value: str):
    import pandas as pd

    path = Path(path_value)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".tsv") or suffixes.endswith(".txt"):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _candidate_prediction_columns(level_index: int) -> List[str]:
    idx = int(level_index)
    candidates = [f"predicted_level{idx}"]
    if idx == 1:
        candidates.append("predicted_level1_identity")
    if idx == 2:
        candidates.append("predicted_level2_identity")
    if idx == 1:
        candidates.append("predicted_terminal_label")
    return candidates


def run_evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    import pandas as pd

    try:
        from network_parser.model_evaluation import (
            evaluate_predictions,
            load_labels_from_metadata,
        )
    except ImportError:  # pragma: no cover - supports direct source-tree execution
        from model_evaluation import evaluate_predictions, load_labels_from_metadata  # type: ignore

    labels: List[str] = []
    if getattr(args, "hierarchy_labels", None):
        labels = [str(x) for x in args.hierarchy_labels]
    elif getattr(args, "label", None):
        labels = [str(args.label)]
    else:
        raise ValueError("evaluate requires either --label or --hierarchy_labels.")

    global_level2_label = getattr(args, "global_level2_label", None)
    if global_level2_label and len(labels) >= 2:
        LOGGER.info(
            "Using --global_level2_label=%s as the Level-2 truth label for evaluation.",
            global_level2_label,
        )
        labels[1] = str(global_level2_label)

    predictions = _read_cli_table(args.predictions)
    if "sample_id" not in predictions.columns:
        raise ValueError("Predictions table must contain a sample_id column.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}
    skipped: List[Dict[str, Any]] = []

    for level_idx, label_column in enumerate(labels, start=1):
        if len(labels) == 1 and getattr(args, "predicted_column", None):
            pred_col = str(args.predicted_column)
        else:
            pred_col = next(
                (
                    c
                    for c in _candidate_prediction_columns(level_idx)
                    if c in predictions.columns
                ),
                None,
            )

        if not pred_col:
            message = {
                "status": "skipped",
                "label_column": label_column,
                "reason": f"No prediction column found for requested level {level_idx}.",
                "candidate_columns": _candidate_prediction_columns(level_idx),
            }
            if bool(getattr(args, "skip_missing_prediction_levels", False)):
                skipped.append(message)
                continue
            raise ValueError(
                f"No prediction column found for requested level {level_idx} ({label_column}). "
                f"Tried: {', '.join(_candidate_prediction_columns(level_idx))}"
            )

        y_true = load_labels_from_metadata(
            meta_path=args.meta,
            label_column=label_column,
            sample_id_column=getattr(args, "sample_id_column", None),
        )
        y_pred = pd.Series(
            predictions[pred_col].astype(str).values,
            index=predictions["sample_id"].astype(str).values,
            name=pred_col,
        )

        level_name = f"level{level_idx}_{label_column}"
        result = evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            class_support_scores=None,
            output_dir=out_dir / level_name,
            level_name=level_name,
        )
        result["truth_label_column"] = label_column
        result["prediction_column"] = pred_col
        results[level_name] = result

    summary = {
        "status": "success" if results else "skipped",
        "predictions": str(args.predictions),
        "meta": str(args.meta),
        "requested_label_columns": labels,
        "evaluated_levels": list(results.keys()),
        "skipped_levels": skipped,
        "per_level": results,
    }
    with open(out_dir / "evaluation_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    LOGGER.info(
        "Evaluation complete | levels=%d | output_dir=%s", len(results), out_dir
    )
    return summary


def run_cross_validate(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        from network_parser.cross_validation import run_repeated_cv
    except ImportError:  # pragma: no cover - supports direct source-tree execution
        from cross_validation import run_repeated_cv  # type: ignore

    config = load_config(args.config)
    config = apply_common_overrides(config, args)
    if hasattr(config, "__post_init__"):
        config.__post_init__()

    LOGGER.info(
        "Starting NetworkParser repeated cross-validation | label=%s | repeats=%d | folds=%d",
        args.label,
        int(args.n_repeats),
        int(args.n_splits),
    )
    return run_repeated_cv(
        genomic_path=args.genomic,
        meta_path=args.meta,
        label_column=args.label,
        output_dir=args.output_dir,
        config=config,
        ref_fasta=args.ref_fasta,
        n_repeats=int(args.n_repeats),
        n_splits=int(args.n_splits),
        algorithm=getattr(args, "algorithm", None),
        random_state=getattr(args, "random_state", None),
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
    configure_logging(
        verbose=bool(getattr(args, "verbose", False)),
        quiet=bool(getattr(args, "quiet", False)),
    )

    try:
        if args.command == "run":
            run_single_label(args)
        elif args.command in {"train-hierarchy", "train-two-level"}:
            run_train_hierarchy(args)
        elif args.command == "bundle":
            run_bundle(args)
        elif args.command == "query":
            run_query(args)
        elif args.command == "evaluate":
            run_evaluate(args)
        elif args.command == "evaluate-hierarchy":
            try:
                from network_parser.hierarchy_evaluation_pack import (
                    run_hierarchy_evaluation_pack,
                )
            except ImportError:  # pragma: no cover
                from hierarchy_evaluation_pack import (  # type: ignore
                    run_hierarchy_evaluation_pack,
                )
            run_hierarchy_evaluation_pack(
                predictions_path=args.predictions,
                meta_path=args.meta,
                hierarchy_labels=list(args.hierarchy_labels),
                output_dir=args.output_dir,
                sample_id_column=getattr(args, "sample_id_column", None),
                harmonize_resistance_labels=bool(
                    getattr(args, "harmonize_resistance_labels", False)
                ),
                n_bootstrap=int(getattr(args, "n_bootstrap", 500)),
            )
        elif args.command == "cross-validate":
            run_cross_validate(args)
        elif args.command == "annotate-panels":
            try:
                from network_parser.panel_annotation import annotate_registry_panels
            except ImportError:  # pragma: no cover
                from panel_annotation import annotate_registry_panels  # type: ignore
            annotate_registry_panels(
                registry_path=Path(args.registry),
                output_dir=Path(args.output_dir),
                catalogue_path=Path(args.catalogue) if args.catalogue else None,
                stability_path=Path(args.stability) if args.stability else None,
                min_stability=float(args.min_stability),
                write_stable_report=bool(getattr(args, "write_stable_report", False)),
                write_catalogue_circularity=bool(
                    getattr(args, "write_catalogue_circularity", False)
                ),
            )
        else:
            raise ValueError(f"Unsupported command: {args.command}")
    except Exception as exc:
        LOGGER.exception("NetworkParser CLI failed: %s", exc)
        return 1

    LOGGER.info("NetworkParser CLI completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

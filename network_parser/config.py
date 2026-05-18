# network_parser/config.py
"""
network_parser.config

Central configuration object for NetworkParser.

Updated architecture
--------------------
Input -> preprocessing
      -> central feature filtering
      -> ML protocol / model selector
      -> conditional decision tree branch
      -> optional downstream interaction validation
"""

from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class NetworkParserConfig:
    # -------------------------------------------------
    # 1) Input / Output behavior
    # -------------------------------------------------
    include_intermediate_files: bool = True
    generic_name: str = "matrix"

    # -------------------------------------------------
    # 2) VCF-level QC (DataLoader)
    # -------------------------------------------------
    qual_threshold: float = 30.0
    min_dp_per_sample: int = 10
    min_gq_per_sample: int = 20
    mq_threshold: float = 40.0
    mq0f_threshold: float = 0.1
    biallelic_only: bool = True

    max_missing_fraction: float = 0.1
    min_spacing_bp: int = 10

    # -------------------------------------------------
    # 3) Cohort-level SNP filtering
    # -------------------------------------------------
    min_sample_presence: int = 10

    # -------------------------------------------------
    # 4) Binary encoding strategy
    # -------------------------------------------------
    ancestral_allele: Literal["Y", "N"] = "Y"

    # -------------------------------------------------
    # 5) Lightweight preprocessing 
    # -------------------------------------------------
    remove_invariant: bool = True
    min_minor_count: int = 10

    # -------------------------------------------------
    # 6) Artifact filtering controls
    # -------------------------------------------------
    matrices_min_count: int = 10
    matrices_repeat_number: int = 5
    matrices_type: Literal["all", "coding", "sense-mutations"] = "all"
    matrices_fix: str = ""

    # -------------------------------------------------
    # 7) Central statistical feature filtering
    # -------------------------------------------------
    run_central_feature_filtering: bool = True

    # Explicit central feature-filter choice.
    # "auto" preserves legacy behavior:
    #   run_rf_fdr_feature_selection=True  -> rf_fdr
    #   run_rf_fdr_feature_selection=False -> chi2_fdr or fisher_fdr via statistical_test
    central_feature_filter_method: Literal["auto", "rf_fdr", "chi2_fdr", "fisher_fdr", "chi2_perm_fdr"] = "auto"
    resolved_central_feature_filter_method: str = "rf_fdr"

    statistical_test: Literal["chi2", "fisher"] = "chi2"
    significance_level: float = 0.05
    fdr_alpha: float = 0.05
    fdr_threshold: float = 0.05
    chi2_min_expected: int = 5
    # Used by chi2_perm_fdr and optional downstream interaction permutation tests.
    n_permutation_tests: int = 1000
    feature_filter_fallback_strategy: Literal["stop", "unfiltered"] = "stop"

    # Legacy internal prefilter compatibility
    prefilter_alpha: float = 0.05
    min_nonmissing_prefilter: float = 0.20
    min_maf_prefilter: float = 0.0
    max_prefiltered_features: Optional[int] = 10000

    multiple_testing_method: Literal["fdr_bh", "bonferroni"] = "fdr_bh"
    # -------------------------------------------------
    # RF-FDR central feature selection
    # -------------------------------------------------
    run_rf_fdr_feature_selection: bool = True

    rf_selector_n_estimators: int = 300
    rf_selector_n_observed_repeats: int = 10

    # Higher default gives better empirical p-value resolution for FDR-BH.
    # This improves robust inference when RF-FDR is used as the main filter.
    rf_selector_n_permutations: int = 1000

    rf_selector_fdr_alpha: float = 0.05
    rf_selector_max_features: str = "sqrt"
    rf_selector_min_samples_leaf: int = 1
    rf_selector_class_weight: Optional[str] = "balanced"
    rf_selector_min_importance: float = 0.0
    rf_selector_top_n: Optional[int] = None
    rf_selector_random_state: int = 42

    # stop = do not silently pass exploratory top-N features as if they were FDR-significant.
    # Use "top_n" only for exploratory/smoke testing.
    rf_selector_fallback_strategy: Literal["top_n", "unfiltered", "stop"] = "stop"
    rf_selector_fallback_top_n: int = 500

    # Two-level fallback control.
    # False = fail loudly if ML training fails.
    # True = allow RF fallback model and record it explicitly.
    allow_two_level_rf_fallback: bool = False
    # -------------------------------------------------
    # 8) Decision Tree parameters
    # -------------------------------------------------
    max_depth: Optional[int] = None
    max_branch_depth: int = 3
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_information_gain: float = 0.001

    min_group_size: int = 2

    # -------------------------------------------------
    # 9) Interaction / Epistasis Mining (POST-TREE)
    # -------------------------------------------------
    epistasis_strength_threshold: float = 0.05
    max_epistatic_interactions: int = 50

    # -------------------------------------------------
    # 10) Post-tree bootstrap / stability
    # -------------------------------------------------
    n_bootstrap: int = 100
    bootstrap_sample_fraction: float = 0.8
    n_bootstrap_samples: int = 1000
    bootstrap_samples_per_iter: int = 100
    bootstrap_outer_iters: int = 5
    min_bootstrap_support: float = 0.7

    # -------------------------------------------------
    # 11) Optional matrix compression
    # -------------------------------------------------
    use_integer_variant_ids: bool = False

    # -------------------------------------------------
    # 12) Performance & reproducibility
    # -------------------------------------------------
    n_jobs: int = -1
    random_state: int = 42
    memory_efficient: bool = False

    # -------------------------------------------------
    # 13) Pipeline mode
    # -------------------------------------------------
    pipeline_mode: Literal[
        "matrix_only",
        "decision_tree_only",
        "ml_only",
        "both",
        "two_level",
    ] = "both"

    # Backward compatibility
    run_ml_protocol: bool = False
    # -------------------------------------------------
    # Two-level protocol
    # -------------------------------------------------
    level1_label_column: Optional[str] = None
    level2_label_column: Optional[str] = None
    train_global_level2: bool = True
    min_level2_samples_per_group: Optional[int] = None

    # -------------------------------------------------
    # 14) Updated orchestration flags
    # -------------------------------------------------
    run_model_selector: bool = True
    run_conditional_dt: bool = True
    disable_conditional_dt: bool = False
    trigger_decision_tree_on_selected: bool = True
    trigger_decision_tree_if_candidate: bool = True
    decision_tree_requires_selector_match: bool = False

    # -------------------------------------------------
    # 15) ML protocol branch
    # -------------------------------------------------
    ml_algorithm: str = "auto"
    ml_min_sensitivity: float = 0.5
    ml_max_sensitivity: float = 1.0
    ml_step_sensitivity: float = 0.1

    ml_empty_symbol: str = ""
    ml_remove_empty_field_threshold: float = 1.0

    def __post_init__(self) -> None:
        self.min_group_size = self.min_samples_split

        if self.ancestral_allele not in {"Y", "N"}:
            raise ValueError("ancestral_allele must be 'Y' or 'N'")

        if self.statistical_test not in {"chi2", "fisher"}:
            raise ValueError("statistical_test must be 'chi2' or 'fisher'")

        supported_central_filters = {"auto", "rf_fdr", "chi2_fdr", "fisher_fdr", "chi2_perm_fdr"}
        if self.central_feature_filter_method not in supported_central_filters:
            raise ValueError(
                "central_feature_filter_method must be one of: "
                "'auto', 'rf_fdr', 'chi2_fdr', 'fisher_fdr', or 'chi2_perm_fdr'"
            )

        # Resolve the explicit selector into the legacy booleans/tests used by
        # older parts of the codebase. This preserves backward compatibility
        # while giving users a clean method switch.
        if self.central_feature_filter_method == "auto":
            if bool(self.run_rf_fdr_feature_selection):
                self.resolved_central_feature_filter_method = "rf_fdr"
            elif self.statistical_test == "fisher":
                self.resolved_central_feature_filter_method = "fisher_fdr"
            else:
                self.resolved_central_feature_filter_method = "chi2_fdr"
        else:
            self.resolved_central_feature_filter_method = self.central_feature_filter_method

        if self.resolved_central_feature_filter_method == "rf_fdr":
            self.run_rf_fdr_feature_selection = True
        elif self.resolved_central_feature_filter_method == "chi2_fdr":
            self.run_rf_fdr_feature_selection = False
            self.statistical_test = "chi2"
        elif self.resolved_central_feature_filter_method == "fisher_fdr":
            self.run_rf_fdr_feature_selection = False
            self.statistical_test = "fisher"
        elif self.resolved_central_feature_filter_method == "chi2_perm_fdr":
            self.run_rf_fdr_feature_selection = False
            self.statistical_test = "chi2"

        if self.feature_filter_fallback_strategy not in {"stop", "unfiltered"}:
            raise ValueError(
                "feature_filter_fallback_strategy must be one of: 'stop' or 'unfiltered'"
            )

        if self.matrices_type not in {"all", "coding", "sense-mutations"}:
            raise ValueError("matrices_type must be one of: all, coding, sense-mutations")

        if self.multiple_testing_method not in {"fdr_bh", "bonferroni"}:
            raise ValueError("multiple_testing_method must be 'fdr_bh' or 'bonferroni'")

        supported_modes = {
            "matrix_only",
            "decision_tree_only",
            "ml_only",
            "both",
            "two_level",
        }
        if self.pipeline_mode not in supported_modes:
            raise ValueError(f"pipeline_mode must be one of: {sorted(supported_modes)}")
        if self.min_level2_samples_per_group is not None and self.min_level2_samples_per_group < 2:
            raise ValueError("min_level2_samples_per_group must be >= 2 or None")

        # Correct common typo while preserving backward compatibility.
        if self.ml_algorithm == "SCV":
            self.ml_algorithm = "SVC"

        supported_ml = {"auto", "RF", "MLP", "LR", "MBCS", "DT", "SVC", "DNL"}
        if self.ml_algorithm not in supported_ml:
            raise ValueError(f"ml_algorithm must be one of: {sorted(supported_ml)}")

        if self.run_ml_protocol and self.pipeline_mode == "decision_tree_only":
            self.pipeline_mode = "both"

        if self.qual_threshold < 0:
            raise ValueError("qual_threshold must be >= 0")
        if self.min_dp_per_sample < 0:
            raise ValueError("min_dp_per_sample must be >= 0")
        if self.min_gq_per_sample < 0:
            raise ValueError("min_gq_per_sample must be >= 0")
        if self.mq_threshold < 0:
            raise ValueError("mq_threshold must be >= 0")
        if not 0 <= self.mq0f_threshold <= 1:
            raise ValueError("mq0f_threshold must be in [0, 1]")
        if self.min_sample_presence < 1:
            raise ValueError("min_sample_presence must be >= 1")
        if not 0 <= self.max_missing_fraction <= 1:
            raise ValueError("max_missing_fraction must be in [0, 1]")
        if self.min_spacing_bp < 0:
            raise ValueError("min_spacing_bp must be >= 0")
        if self.min_minor_count < 0:
            raise ValueError("min_minor_count must be >= 0")
        if self.matrices_min_count < 0:
            raise ValueError("matrices_min_count must be >= 0")
        if self.matrices_repeat_number < 1:
            raise ValueError("matrices_repeat_number must be >= 1")
        if not 0 < self.significance_level <= 1:
            raise ValueError("significance_level must be in (0, 1]")
        if not 0 < self.fdr_alpha <= 1:
            raise ValueError("fdr_alpha must be in (0, 1]")
        if not 0 < self.fdr_threshold <= 1:
            raise ValueError("fdr_threshold must be in (0, 1]")
        if self.chi2_min_expected < 1:
            raise ValueError("chi2_min_expected must be >= 1")
        if self.n_permutation_tests < 0:
            raise ValueError("n_permutation_tests must be >= 0")
        if self.resolved_central_feature_filter_method == "chi2_perm_fdr" and self.n_permutation_tests < 1:
            raise ValueError("n_permutation_tests must be >= 1 when using chi2_perm_fdr")
        if not 0 < self.prefilter_alpha <= 1:
            raise ValueError("prefilter_alpha must be in (0, 1]")
        if not 0 <= self.min_nonmissing_prefilter <= 1:
            raise ValueError("min_nonmissing_prefilter must be in [0, 1]")
        if not 0 <= self.min_maf_prefilter <= 0.5:
            raise ValueError("min_maf_prefilter must be in [0, 0.5]")
        if self.max_prefiltered_features is not None and self.max_prefiltered_features < 1:
            raise ValueError("max_prefiltered_features must be >= 1 or None")
        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError("max_depth must be positive or None")
        if self.max_branch_depth < 1:
            raise ValueError("max_branch_depth must be >= 1")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if self.epistasis_strength_threshold < 0:
            raise ValueError("epistasis_strength_threshold must be >= 0")
        if self.max_epistatic_interactions < 1:
            raise ValueError("max_epistatic_interactions must be >= 1")
        if self.n_bootstrap < 0:
            raise ValueError("n_bootstrap must be >= 0")
        if self.n_bootstrap_samples < 0:
            raise ValueError("n_bootstrap_samples must be >= 0")
        if self.bootstrap_samples_per_iter < 1:
            raise ValueError("bootstrap_samples_per_iter must be >= 1")
        if self.bootstrap_outer_iters < 1:
            raise ValueError("bootstrap_outer_iters must be >= 1")
        if not 0 <= self.min_bootstrap_support <= 1:
            raise ValueError("min_bootstrap_support must be in [0, 1]")
        if not 0 <= self.ml_min_sensitivity <= 1:
            raise ValueError("ml_min_sensitivity must be in [0, 1]")
        if not 0 <= self.ml_max_sensitivity <= 1:
            raise ValueError("ml_max_sensitivity must be in [0, 1]")
        if self.ml_min_sensitivity > self.ml_max_sensitivity:
            raise ValueError("ml_min_sensitivity cannot exceed ml_max_sensitivity")
        if self.ml_step_sensitivity <= 0:
            raise ValueError("ml_step_sensitivity must be > 0")
        if not 0 <= self.ml_remove_empty_field_threshold <= 1:
            raise ValueError("ml_remove_empty_field_threshold must be in [0, 1]")
        if self.rf_selector_n_estimators < 1:
            raise ValueError("rf_selector_n_estimators must be >= 1")

        if self.rf_selector_n_observed_repeats < 1:
            raise ValueError("rf_selector_n_observed_repeats must be >= 1")

        if self.rf_selector_n_permutations < 1:
            raise ValueError("rf_selector_n_permutations must be >= 1")

        if not 0 < self.rf_selector_fdr_alpha <= 1:
            raise ValueError("rf_selector_fdr_alpha must be in (0, 1]")

        if self.rf_selector_min_samples_leaf < 1:
            raise ValueError("rf_selector_min_samples_leaf must be >= 1")

        if self.rf_selector_min_importance < 0:
            raise ValueError("rf_selector_min_importance must be >= 0")

        if self.rf_selector_top_n is not None and self.rf_selector_top_n < 1:
            raise ValueError("rf_selector_top_n must be >= 1 or None")

        if self.rf_selector_fallback_strategy not in {"top_n", "unfiltered", "stop"}:
            raise ValueError(
                "rf_selector_fallback_strategy must be one of: "
                "'top_n', 'unfiltered', or 'stop'"
            )

        if self.rf_selector_fallback_top_n < 1:
            raise ValueError("rf_selector_fallback_top_n must be >= 1")

        if not isinstance(self.allow_two_level_rf_fallback, bool):
            raise ValueError("allow_two_level_rf_fallback must be boolean")
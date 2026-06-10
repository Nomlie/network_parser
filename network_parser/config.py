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
    matrices_redundancy_sample_threshold: int = 2000
    matrices_redundancy_sample_size: int = 256

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
    # 13) FASTQ query preprocessing
    # -------------------------------------------------
    # These controls are used only when query_input_type="fastq".
    # FASTQ reads are converted to per-sample VCF.GZ files and then routed
    # through the existing DataLoader VCF-directory pathway. No statistical
    # filtering or model training happens in this stage.
    fastq_max_parallel_samples: int = 1
    fastq_threads: Optional[int] = None
    fastq_memory_per_sample_mb: Optional[int] = None
    fastq_clean_intermediates: bool = False
    fastq_auto_index_reference: bool = True
    fastq_min_mapping_quality: int = 20
    fastq_sample_platform: str = "ILLUMINA"
    fastq_sort_memory: str = "1G"

    # -------------------------------------------------
    # 14) Pipeline mode
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
    # Optional target for the global Level-2 fallback. When unset, the global
    # fallback uses level2_label_column / --level2_label. When set, group-specific
    # Level-2 models still use the detailed --level2_label, while the global
    # fallback can use a broader endpoint such as AMR_binary.
    global_level2_label_column: Optional[str] = None
    train_global_level2: bool = True
    min_level2_samples_per_group: Optional[int] = None

    # Optional Level-2 label-support gate. Disabled by default so legacy runs
    # remain unchanged. When enabled, Level-2 classes with fewer than
    # level2_min_class_count samples are excluded before Level-2 statistical
    # filtering and model screening. This prevents impossible stratified CV
    # caused by singleton or extremely underrepresented phenotype classes.
    level2_drop_low_support_classes: bool = False
    level2_min_class_count: int = 2

    # Optional additional Level-2 global binary fallback. This trains a
    # resistant/susceptible endpoint across all lineages and is used only when
    # a group-specific Level-2 model is unavailable. The binary target can come
    # from a dedicated metadata column or from an explicit mapping file that
    # collapses detailed Level-2 labels into resistant/susceptible states.
    level2_train_binary_global_fallback: bool = False
    level2_binary_label_column: Optional[str] = None
    level2_binary_label_mapping_file: Optional[str] = None
    level2_binary_resistant_values: str = "R,resistant,RESISTANT,Resistant,1,true,TRUE,True"
    level2_binary_susceptible_values: str = "S,susceptible,SUSCEPTIBLE,Susceptible,0,false,FALSE,False"

    # -------------------------------------------------
    # 15) Updated orchestration flags
    # -------------------------------------------------
    run_model_selector: bool = True
    run_conditional_dt: bool = True
    disable_conditional_dt: bool = False
    trigger_decision_tree_on_selected: bool = True
    trigger_decision_tree_if_candidate: bool = True
    decision_tree_requires_selector_match: bool = False

    # -------------------------------------------------
    # 16) ML protocol branch
    # -------------------------------------------------
    ml_algorithm: str = "auto"
    ml_lr_max_iter: int = 2000
    ml_min_sensitivity: float = 0.5
    ml_max_sensitivity: float = 1.0
    ml_step_sensitivity: float = 0.1

    ml_empty_symbol: str = ""
    ml_remove_empty_field_threshold: float = 1.0

    # -------------------------------------------------
    # 17) Ranked feature-panel separability check
    # -------------------------------------------------
    # Runs after central statistical filtering and before ML / tree fitting.
    # It ranks retained features by corrected/empirical/raw p-value, evaluates
    # top-N panels, and forwards the smallest panel with acceptable separability.
    run_feature_panel_separability_check: bool = True
    feature_panel_sizes: tuple = (100, 200, 500, 1000)
    feature_panel_metric: Literal[
        "balanced_accuracy",
        "adjusted_rand",
        "normalized_mutual_info",
        "silhouette",
    ] = "balanced_accuracy"

    # Supervised probe used to score top-N panels by balanced accuracy.
    # "lr" is fast and stable after scaling; "rf" is slower but captures
    # nonlinear feature combinations while keeping the stage pre-model.
    feature_panel_classifier: Literal["lr", "rf"] = "lr"
    feature_panel_lr_max_iter: int = 2000
    feature_panel_lr_tol: float = 1e-4
    feature_panel_rf_n_estimators: int = 300
    feature_panel_rf_max_features: Optional[str] = "sqrt"
    feature_panel_rf_min_samples_leaf: int = 1
    feature_panel_rf_class_weight: Optional[str] = "balanced"
    feature_panel_rf_n_jobs: Optional[int] = None

    feature_panel_min_score: float = 0.75
    feature_panel_selection_rule: Literal["smallest_passing", "best_passing", "best_available"] = "smallest_passing"
    feature_panel_cv_splits: int = 5
    feature_panel_always_include_full_filtered: bool = True
    feature_panel_max_silhouette_samples: int = 5000
    feature_panel_large_feature_threshold: int = 5000
    feature_panel_large_max_scoring_features: int = 5000
    feature_panel_large_pool_multiplier: int = 4
    feature_panel_score_full_large_matrix: bool = False

    # -------------------------------------------------
    # 18) Two-level binary model bundle output
    # -------------------------------------------------
    # Build a portable .npb bundle automatically at the end of two-level
    # training. This keeps query-ready deployment artifacts next to the
    # registry without requiring a second manual CLI command.
    build_model_bundle: bool = True
    model_bundle_filename: str = "networkparser_model_bundle.npb"
    model_bundle_include_model_payloads: bool = True
    model_bundle_include_feature_manifests: bool = True
    model_bundle_include_ranked_feature_tables: bool = True
    model_bundle_fail_on_error: bool = False

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
        if self.global_level2_label_column is not None:
            self.global_level2_label_column = str(self.global_level2_label_column).strip() or None
        if self.min_level2_samples_per_group is not None and self.min_level2_samples_per_group < 2:
            raise ValueError("min_level2_samples_per_group must be >= 2 or None")
        if not isinstance(self.level2_drop_low_support_classes, bool):
            raise ValueError("level2_drop_low_support_classes must be boolean")
        if int(self.level2_min_class_count) < 2:
            raise ValueError("level2_min_class_count must be >= 2")
        self.level2_min_class_count = int(self.level2_min_class_count)
        if not isinstance(self.level2_train_binary_global_fallback, bool):
            raise ValueError("level2_train_binary_global_fallback must be boolean")
        if self.level2_binary_label_column is not None:
            self.level2_binary_label_column = str(self.level2_binary_label_column).strip() or None
        if self.level2_binary_label_mapping_file is not None:
            self.level2_binary_label_mapping_file = str(self.level2_binary_label_mapping_file).strip() or None
        if not str(self.level2_binary_resistant_values).strip():
            raise ValueError("level2_binary_resistant_values cannot be empty")
        if not str(self.level2_binary_susceptible_values).strip():
            raise ValueError("level2_binary_susceptible_values cannot be empty")

        # Correct common typo while preserving backward compatibility.
        if self.ml_algorithm == "SCV":
            self.ml_algorithm = "SVC"

        supported_ml = {"auto", "RF", "MLP", "LR", "MBCS", "DT", "SVC", "DNL"}
        if self.ml_algorithm not in supported_ml:
            raise ValueError(f"ml_algorithm must be one of: {sorted(supported_ml)}")

        if self.run_ml_protocol and self.pipeline_mode == "decision_tree_only":
            self.pipeline_mode = "both"

        if int(self.fastq_max_parallel_samples) < 1:
            raise ValueError("fastq_max_parallel_samples must be >= 1")
        self.fastq_max_parallel_samples = int(self.fastq_max_parallel_samples)
        if self.fastq_threads is not None:
            self.fastq_threads = int(self.fastq_threads)
            if self.fastq_threads < 1:
                raise ValueError("fastq_threads must be >= 1 or None")
        if self.fastq_memory_per_sample_mb is not None:
            self.fastq_memory_per_sample_mb = int(self.fastq_memory_per_sample_mb)
            if self.fastq_memory_per_sample_mb < 256:
                raise ValueError("fastq_memory_per_sample_mb must be >= 256 or None")
        if not isinstance(self.fastq_clean_intermediates, bool):
            raise ValueError("fastq_clean_intermediates must be boolean")
        if not isinstance(self.fastq_auto_index_reference, bool):
            raise ValueError("fastq_auto_index_reference must be boolean")
        if int(self.fastq_min_mapping_quality) < 0:
            raise ValueError("fastq_min_mapping_quality must be >= 0")
        self.fastq_min_mapping_quality = int(self.fastq_min_mapping_quality)
        self.fastq_sample_platform = str(self.fastq_sample_platform).strip() or "ILLUMINA"
        self.fastq_sort_memory = str(self.fastq_sort_memory).strip() or "1G"

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
        if self.matrices_redundancy_sample_threshold < 1:
            raise ValueError("matrices_redundancy_sample_threshold must be >= 1")
        if self.matrices_redundancy_sample_size < 1:
            raise ValueError("matrices_redundancy_sample_size must be >= 1")
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
        if self.ml_lr_max_iter < 100:
            raise ValueError("ml_lr_max_iter must be >= 100")

        if not isinstance(self.run_feature_panel_separability_check, bool):
            raise ValueError("run_feature_panel_separability_check must be boolean")
        if not self.feature_panel_sizes:
            raise ValueError("feature_panel_sizes must contain at least one positive integer")
        try:
            self.feature_panel_sizes = tuple(
                int(x.strip()) if isinstance(x, str) else int(x)
                for x in (self.feature_panel_sizes.split(",") if isinstance(self.feature_panel_sizes, str) else self.feature_panel_sizes)
            )
        except Exception as exc:
            raise ValueError("feature_panel_sizes must be a tuple/list or comma-separated string of positive integers") from exc
        if any(x < 1 for x in self.feature_panel_sizes):
            raise ValueError("feature_panel_sizes must contain positive integers")
        if self.feature_panel_metric not in {"balanced_accuracy", "adjusted_rand", "normalized_mutual_info", "silhouette"}:
            raise ValueError(
                "feature_panel_metric must be one of: balanced_accuracy, adjusted_rand, "
                "normalized_mutual_info, silhouette"
            )

        panel_classifier_aliases = {
            "logistic": "lr",
            "logistic_regression": "lr",
            "randomforest": "rf",
            "random_forest": "rf",
            "random_forest_classifier": "rf",
        }
        self.feature_panel_classifier = panel_classifier_aliases.get(
            str(self.feature_panel_classifier).strip().lower().replace("-", "_"),
            str(self.feature_panel_classifier).strip().lower(),
        )
        if self.feature_panel_classifier not in {"lr", "rf"}:
            raise ValueError("feature_panel_classifier must be one of: lr, rf")
        if self.feature_panel_lr_max_iter < 100:
            raise ValueError("feature_panel_lr_max_iter must be >= 100")
        if not 0 < self.feature_panel_lr_tol <= 1:
            raise ValueError("feature_panel_lr_tol must be in (0, 1]")
        if self.feature_panel_rf_n_estimators < 1:
            raise ValueError("feature_panel_rf_n_estimators must be >= 1")
        if self.feature_panel_rf_min_samples_leaf < 1:
            raise ValueError("feature_panel_rf_min_samples_leaf must be >= 1")
        if self.feature_panel_rf_n_jobs is not None and int(self.feature_panel_rf_n_jobs) == 0:
            raise ValueError("feature_panel_rf_n_jobs cannot be 0; use 1, -1, or another non-zero integer")
        if isinstance(self.feature_panel_rf_class_weight, str):
            cw = self.feature_panel_rf_class_weight.strip().lower()
            self.feature_panel_rf_class_weight = None if cw in {"", "none", "null"} else cw
        if self.feature_panel_rf_class_weight not in {None, "balanced", "balanced_subsample"}:
            raise ValueError("feature_panel_rf_class_weight must be one of: balanced, balanced_subsample, none")
        if isinstance(self.feature_panel_rf_max_features, str):
            mf = self.feature_panel_rf_max_features.strip().lower()
            self.feature_panel_rf_max_features = None if mf in {"", "none", "null"} else mf
            if self.feature_panel_rf_max_features not in {None, "sqrt", "log2"}:
                raise ValueError("feature_panel_rf_max_features must be one of: sqrt, log2, none")

        if not 0 <= self.feature_panel_min_score <= 1:
            raise ValueError("feature_panel_min_score must be in [0, 1]")
        if self.feature_panel_selection_rule not in {"smallest_passing", "best_passing", "best_available"}:
            raise ValueError("feature_panel_selection_rule must be one of: smallest_passing, best_passing, best_available")
        if self.feature_panel_cv_splits < 2:
            raise ValueError("feature_panel_cv_splits must be >= 2")
        if not isinstance(self.feature_panel_always_include_full_filtered, bool):
            raise ValueError("feature_panel_always_include_full_filtered must be boolean")
        if self.feature_panel_max_silhouette_samples < 2:
            raise ValueError("feature_panel_max_silhouette_samples must be >= 2")
        if self.feature_panel_large_feature_threshold < 1:
            raise ValueError("feature_panel_large_feature_threshold must be >= 1")
        if self.feature_panel_large_max_scoring_features < 1:
            raise ValueError("feature_panel_large_max_scoring_features must be >= 1")
        if self.feature_panel_large_pool_multiplier < 1:
            raise ValueError("feature_panel_large_pool_multiplier must be >= 1")
        if not isinstance(self.feature_panel_score_full_large_matrix, bool):
            raise ValueError("feature_panel_score_full_large_matrix must be boolean")

        if not isinstance(self.build_model_bundle, bool):
            raise ValueError("build_model_bundle must be boolean")
        self.model_bundle_filename = str(self.model_bundle_filename).strip()
        if not self.model_bundle_filename:
            raise ValueError("model_bundle_filename cannot be empty")
        if not self.model_bundle_filename.endswith(".npb"):
            self.model_bundle_filename = f"{self.model_bundle_filename}.npb"
        if not isinstance(self.model_bundle_include_model_payloads, bool):
            raise ValueError("model_bundle_include_model_payloads must be boolean")
        if not isinstance(self.model_bundle_include_feature_manifests, bool):
            raise ValueError("model_bundle_include_feature_manifests must be boolean")
        if not isinstance(self.model_bundle_include_ranked_feature_tables, bool):
            raise ValueError("model_bundle_include_ranked_feature_tables must be boolean")
        if not isinstance(self.model_bundle_fail_on_error, bool):
            raise ValueError("model_bundle_fail_on_error must be boolean")

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
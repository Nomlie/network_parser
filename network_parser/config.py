# network_parser/config.py
"""
network_parser.config

Central configuration object for NetworkParser.

Design notes
------------
- Keep everything generic with respect to the current dataset.
- Statistical validation is pre-tree.
- Bootstrap / confidence is post-tree.
- ML protocol is an optional downstream branch that consumes the
  already-created sample x feature dataframe from DataLoader.

These values can be overridden via JSON config.
"""

from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class NetworkParserConfig:
    # ─────────────────────────────────────────────
    # 1) Input / Output behavior
    # ─────────────────────────────────────────────
    include_intermediate_files: bool = True
    generic_name: str = "matrix"

    # ─────────────────────────────────────────────
    # 2) VCF-level QC (DataLoader)
    #    Applied while streaming each per-sample VCF
    # ─────────────────────────────────────────────
    qual_threshold: float = 30.0
    min_dp_per_sample: int = 10
    min_gq_per_sample: int = 20
    mq_threshold: float = 40.0
    mq0f_threshold: float = 0.1
    biallelic_only: bool = True

    # Optional cohort / site missingness controls
    max_missing_fraction: float = 0.1
    min_spacing_bp: int = 10

    # ─────────────────────────────────────────────
    # 3) Cohort-level SNP filtering
    # ─────────────────────────────────────────────
    min_sample_presence: int = 3

    # ─────────────────────────────────────────────
    # 4) Binary encoding strategy
    # ─────────────────────────────────────────────
    ancestral_allele: Literal["Y", "N"] = "Y"

    # ─────────────────────────────────────────────
    # 5) Lightweight preprocessing (NOT statistical)
    # ─────────────────────────────────────────────
    remove_invariant: bool = True
    min_minor_count: int = 0

    # ─────────────────────────────────────────────
    # 6) Artifact (matrices/*) filtering controls
    # ─────────────────────────────────────────────
    matrices_min_count: int = 3
    matrices_repeat_number: int = 5
    matrices_type: Literal["all", "coding", "sense-mutations"] = "all"
    matrices_fix: str = ""

    # ─────────────────────────────────────────────
    # 7) Statistical validation (PRE-TREE)
    # ─────────────────────────────────────────────
    statistical_test: Literal["chi2", "fisher"] = "chi2"
    significance_level: float = 0.05
    fdr_alpha: float = 0.05
    fdr_threshold: float = 0.05
    chi2_min_expected: int = 5
    n_permutation_tests: int = 500

    # Pre-tree statistical filtering
    prefilter_alpha: float = 0.05
    min_nonmissing_prefilter: float = 0.20
    min_maf_prefilter: float = 0.0
    max_prefiltered_features: Optional[int] = 10000

    # Multiple testing
    multiple_testing_method: Literal["fdr_bh", "bonferroni"] = "fdr_bh"

    # ─────────────────────────────────────────────
    # 8) Decision Tree parameters
    # ─────────────────────────────────────────────
    max_depth: Optional[int] = None
    max_branch_depth: int = 3
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_information_gain: float = 0.001

    # Compatibility alias
    min_group_size: int = 2

    # ─────────────────────────────────────────────
    # 9) Interaction / Epistasis Mining (POST-TREE)
    # ─────────────────────────────────────────────
    epistasis_strength_threshold: float = 0.05
    max_epistatic_interactions: int = 50

    # ─────────────────────────────────────────────
    # 10) Post-tree bootstrap / stability (POST-TREE)
    # ─────────────────────────────────────────────
    n_bootstrap: int = 100
    bootstrap_sample_fraction: float = 0.8

    # Advanced stability interface
    n_bootstrap_samples: int = 1000
    bootstrap_samples_per_iter: int = 100
    bootstrap_outer_iters: int = 5
    min_bootstrap_support: float = 0.7

    # ─────────────────────────────────────────────
    # 11) Optional matrix compression
    # ─────────────────────────────────────────────
    use_integer_variant_ids: bool = False

    # ─────────────────────────────────────────────
    # 12) Performance & Reproducibility
    # ─────────────────────────────────────────────
    n_jobs: int = -1
    random_state: int = 42
    memory_efficient: bool = False

    # ─────────────────────────────────────────────
    # 13) Pipeline branch mode
    # ─────────────────────────────────────────────
    pipeline_mode: Literal[
        "matrix_only",
        "decision_tree_only",
        "ml_only",
        "both",
    ] = "decision_tree_only"

    # Backward compatibility with older configs / CLI
    run_ml_protocol: bool = False

    # ─────────────────────────────────────────────
    # 14) Downstream ML protocol branch
    # ─────────────────────────────────────────────
    ml_algorithm: str = "auto"
    ml_min_sensitivity: float = 0.5
    ml_max_sensitivity: float = 1.0
    ml_step_sensitivity: float = 0.1

    # train.py-derived optional controls
    ml_empty_symbol: str = ""
    ml_remove_empty_field_threshold: float = 1.0

    def __post_init__(self) -> None:
        self.min_group_size = self.min_samples_split

        if self.ancestral_allele not in {"Y", "N"}:
            raise ValueError("ancestral_allele must be 'Y' or 'N'")

        if self.statistical_test not in {"chi2", "fisher"}:
            raise ValueError("statistical_test must be 'chi2' or 'fisher'")

        if self.matrices_type not in {"all", "coding", "sense-mutations"}:
            raise ValueError("matrices_type must be one of: all, coding, sense-mutations")

        if self.multiple_testing_method not in {"fdr_bh", "bonferroni"}:
            raise ValueError("multiple_testing_method must be 'fdr_bh' or 'bonferroni'")

        supported_modes = {"matrix_only", "decision_tree_only", "ml_only", "both"}
        if self.pipeline_mode not in supported_modes:
            raise ValueError(f"pipeline_mode must be one of: {sorted(supported_modes)}")

        supported_ml = {"auto", "RF", "MLP", "LR", "MBCS", "DT", "SVC", "SCV", "DNL"}
        if self.ml_algorithm not in supported_ml:
            raise ValueError(f"ml_algorithm must be one of: {sorted(supported_ml)}")

        if self.run_ml_protocol and self.pipeline_mode == "decision_tree_only":
            # preserve old behavior: run_ml_protocol=True used to mean "add ML branch"
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
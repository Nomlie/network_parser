# network_parser/config.py
"""
network_parser.config

Central configuration object for NetworkParser.

All filtering logic that can safely be parameterized is defined here,
including:

• VCF / DataLoader QC thresholds
• Cohort-level SNP presence filtering
• Binary baseline strategy
• Lightweight preprocessing
• Intermediate artifact generation filters (fasta / matrices outputs)
• Statistical validation thresholds (pre-tree)
• Decision tree parameters
• Post-tree bootstrap settings

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
    # ─────────────────────────────────────────────
    qual_threshold: float = 30.0
    min_dp_per_sample: int = 10
    min_gq_per_sample: int = 20  # kept from earlier config (safe default)
    mq_threshold: float = 40.0
    mq0f_threshold: float = 0.1
    biallelic_only: bool = True

    # Optional cohort/variant missingness controls (from earlier config)
    max_missing_fraction: float = 0.1
    min_spacing_bp: int = 10

    # Union-matrix mode (directory of single-sample VCFs)
    force_union_matrix: bool = False
    union_matrix_threshold: int = 500
    union_dp_min: int = 10

    # ─────────────────────────────────────────────
    # 3) Cohort-level SNP filtering
    # ─────────────────────────────────────────────
    min_sample_presence: int = 3

    # ─────────────────────────────────────────────
    # 4) Binary encoding strategy
    # ─────────────────────────────────────────────
    ancestral_allele: Literal["Y", "N"] = "Y"
    # "Y" → baseline = reference allele
    # "N" → baseline = cohort mode allele

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
    matrices_fix: str = ""  # comma/space separated list of 1-based positions to force-keep

    # ─────────────────────────────────────────────
    # 7) Statistical validation (PRE-TREE)
    # ─────────────────────────────────────────────
    statistical_test: Literal["chi2", "fisher"] = "chi2"
    significance_level: float = 0.05  # global alpha
    fdr_alpha: float = 0.05           # (alias-like; kept for your existing JSONs)
    fdr_threshold: float = 0.05       # kept for earlier code paths
    chi2_min_expected: int = 5
    n_permutation_tests: int = 500

    # Pre-filtering / feature selection (from earlier config; optional stage)
    prefilter_alpha: float = 0.05
    min_nonmissing_prefilter: float = 0.20
    min_maf_prefilter: float = 0.0
    max_prefiltered_features: Optional[int] = 10000

    # Multiple testing method (from earlier config)
    multiple_testing_method: Literal["fdr_bh", "bonferroni"] = "fdr_bh"

    # ─────────────────────────────────────────────
    # 8) Decision Tree parameters
    # ─────────────────────────────────────────────
    max_depth: Optional[int] = None
    max_branch_depth: int = 3
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_information_gain: float = 0.001

    # Legacy compatibility
    min_group_size: int = 2  # populated from min_samples_split in __post_init__
    # ──────────────────────────────────────────────────────────────
    # Interaction / Epistasis Mining
    # ──────────────────────────────────────────────────────────────
    epistasis_strength_threshold: float = 0.05
    max_epistatic_interactions: int = 50
    n_permutation_tests: int = 500
    # ─────────────────────────────────────────────
    # 9) Post-tree bootstrap / stability
    # ─────────────────────────────────────────────
    # Simple bootstrap interface (what your new config already has)
    n_bootstrap: int = 100
    bootstrap_sample_fraction: float = 0.8

    # Advanced stability interface (kept for earlier pipeline stages)
    n_bootstrap_samples: int = 1000
    bootstrap_samples_per_iter: int = 100
    bootstrap_outer_iters: int = 5
    min_bootstrap_support: float = 0.7

    # ─────────────────────────────────────────────
    # 10) Optional matrix compression
    # ─────────────────────────────────────────────
    use_integer_variant_ids: bool = False

    # ─────────────────────────────────────────────
    # 11) Performance & Reproducibility
    # ─────────────────────────────────────────────
    n_jobs: int = -1  # -1 = all cores
    random_state: int = 42
    memory_efficient: bool = False

    # ─────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────
    def __post_init__(self) -> None:
        # legacy alias
        self.min_group_size = self.min_samples_split

        # enums
        if self.ancestral_allele not in {"Y", "N"}:
            raise ValueError("ancestral_allele must be 'Y' or 'N'")

        if self.statistical_test not in {"chi2", "fisher"}:
            raise ValueError("statistical_test must be 'chi2' or 'fisher'")

        if self.matrices_type not in {"all", "coding", "sense-mutations"}:
            raise ValueError("matrices_type must be one of: all, coding, sense-mutations")

        if self.multiple_testing_method not in {"fdr_bh", "bonferroni"}:
            raise ValueError("multiple_testing_method must be 'fdr_bh' or 'bonferroni'")

        # ranges
        if self.qual_threshold < 0:
            raise ValueError("qual_threshold must be >= 0")

        if self.min_dp_per_sample < 0 or self.union_dp_min < 0:
            raise ValueError("DP thresholds must be >= 0")

        if not 0 <= self.mq0f_threshold <= 1:
            raise ValueError("mq0f_threshold must be in [0, 1]")

        if self.min_sample_presence < 1:
            raise ValueError("min_sample_presence must be >= 1")

        if not 0 <= self.max_missing_fraction <= 1:
            raise ValueError("max_missing_fraction must be in [0, 1]")

        if self.min_spacing_bp < 0:
            raise ValueError("min_spacing_bp must be >= 0")

        if not 0 < self.significance_level <= 1:
            raise ValueError("significance_level must be in (0, 1]")

        if not 0 < self.fdr_alpha <= 1:
            raise ValueError("fdr_alpha must be in (0, 1]")

        if not 0 < self.fdr_threshold <= 1:
            raise ValueError("fdr_threshold must be in (0, 1]")

        if self.chi2_min_expected < 1:
            raise ValueError("chi2_min_expected must be >= 1")

        if not 0 < self.prefilter_alpha <= 1:
            raise ValueError("prefilter_alpha must be in (0, 1]")

        if not 0 <= self.min_nonmissing_prefilter <= 1:
            raise ValueError("min_nonmissing_prefilter must be in [0, 1]")

        if not 0 <= self.min_maf_prefilter <= 0.5:
            raise ValueError("min_maf_prefilter must be in [0, 0.5]")

        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError("max_depth must be positive or None")

        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")

        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")

        if self.min_samples_leaf >= self.min_samples_split:
            raise ValueError("min_samples_leaf must be smaller than min_samples_split")

        if not 0 < self.bootstrap_sample_fraction <= 1:
            raise ValueError("bootstrap_sample_fraction must be in (0, 1]")

        if self.n_bootstrap < 0:
            raise ValueError("n_bootstrap must be >= 0")

        if self.n_bootstrap_samples < 0:
            raise ValueError("n_bootstrap_samples must be >= 0")

        if self.bootstrap_samples_per_iter <= 0:
            raise ValueError("bootstrap_samples_per_iter must be > 0")

        if self.bootstrap_outer_iters <= 0:
            raise ValueError("bootstrap_outer_iters must be > 0")

        if not 0 <= self.min_bootstrap_support <= 1:
            raise ValueError("min_bootstrap_support must be in [0, 1]")
# network_parser/config.py
"""
Configuration module for NetworkParser.

Central place for all tunable parameters of the pipeline.
Defaults are chosen with microbial SNP datasets in mind
(thousands of samples, tens to hundreds of thousands of variants).

Parameters are grouped roughly in order of pipeline stages:
    1. Input / Data Loading
    2. Pre-filtering
    3. Decision Tree & Feature Discovery
    4. Statistical Validation & Confidence
    5. Interaction / Epistasis Mining
    6. Performance & Reproducibility
    7. Output & Post-processing
"""

from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class NetworkParserConfig:
    """
    Comprehensive configuration class for the NetworkParser pipeline.
    """

    # ──────────────────────────────────────────────────────────────
    # 1. Input / Data Loading & VCF Filtering
    # ──────────────────────────────────────────────────────────────
    qual_threshold: float = 30.0
    min_dp_per_sample: int = 10
    min_gq_per_sample: int = 20
    max_missing_fraction: float = 0.1               # Max allowed missingness per variant
    min_spacing_bp: int = 10                        # Min distance between retained variants

    # Union-matrix mode (for directories of single-sample VCFs)
    force_union_matrix: bool = False
    union_matrix_threshold: int = 500               # Auto-switch to union mode above this # of VCFs
    union_dp_min: int = 10


    # ──────────────────────────────────────────────────────────────
    # 2. Pre-filtering / Feature Selection
    # ──────────────────────────────────────────────────────────────
    prefilter_alpha: float = 0.05                   # Significance level for chi²/Fisher pre-filter
    min_nonmissing_prefilter: float = 0.20          # Min fraction of non-missing genotypes (0.0–1.0)
    min_maf_prefilter: float = 0.005                # Min minor allele frequency (0.0–0.5)
    max_prefiltered_features: Optional[int] = 10000 # Max features kept after pre-filter (None = no limit)


    # ──────────────────────────────────────────────────────────────
    # 3. Decision Tree & Feature Discovery
    # ──────────────────────────────────────────────────────────────
    max_depth: Optional[int] = None                 # None = unlimited
    max_branch_depth: int = 3                       # Max depth to consider features as "branch"
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    min_information_gain: float = 0.001
    use_integer_variant_ids = True
    # Legacy compatibility
    min_group_size: int = 10                        # Alias for min_samples_split


    # ──────────────────────────────────────────────────────────────
    # 4. Statistical Validation & Confidence Scoring
    # ──────────────────────────────────────────────────────────────
    significance_level: float = 0.05
    fdr_threshold: float = 0.05
    multiple_testing_method: Literal["fdr_bh", "bonferroni"] = "fdr_bh"

    ##chi-square
    chi2_min_expected: int = 5

    # Bootstrap settings
    n_bootstrap_samples: int = 1000
    bootstrap_samples_per_iter: int = 100
    bootstrap_outer_iters: int = 5
    min_bootstrap_support: float = 0.7              # Min fraction of stable rankings
    max_depth: Optional[int] = 5

    # ──────────────────────────────────────────────────────────────
    # 5. Interaction / Epistasis Mining
    # ──────────────────────────────────────────────────────────────
    epistasis_strength_threshold: float = 0.05
    max_epistatic_interactions: int = 50
    n_permutation_tests: int = 500


    # ──────────────────────────────────────────────────────────────
    # 6. Performance & Reproducibility
    # ──────────────────────────────────────────────────────────────
    n_jobs: int = -1                                # -1 = all cores
    random_state: int = 42
    memory_efficient: bool = False


    # ──────────────────────────────────────────────────────────────
    # 7. Output & Post-processing
    # ──────────────────────────────────────────────────────────────
    generate_consensus_fasta: bool = True
    consensus_fasta_type: Literal["individual", "multi"] = "individual"
    include_intermediate_files: bool = True


    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        valid_methods = {"fdr_bh", "bonferroni"}
        if self.multiple_testing_method not in valid_methods:
            raise ValueError(
                f"multiple_testing_method must be one of {valid_methods}, "
                f"got {self.multiple_testing_method!r}"
            )

        valid_fasta_types = {"individual", "multi"}
        if self.consensus_fasta_type not in valid_fasta_types:
            raise ValueError(
                f"consensus_fasta_type must be one of {valid_fasta_types}, "
                f"got {self.consensus_fasta_type!r}"
            )

        # Maintain legacy compatibility
        self.min_group_size = self.min_samples_split

        # Basic range validation for critical numeric params
        if not 0 < self.prefilter_alpha <= 1:
            raise ValueError("prefilter_alpha must be in (0, 1]")
        if not 0 <= self.min_maf_prefilter <= 0.5:
            raise ValueError("min_maf_prefilter must be in [0, 0.5]")
        if not 0 <= self.max_missing_fraction <= 1:
            raise ValueError("max_missing_fraction must be in [0, 1]")
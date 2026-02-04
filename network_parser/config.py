# network_parser/config.py
"""
Configuration module for NetworkParser.

Defines configurable parameters controlling the behavior of the pipeline,
including statistical thresholds, decision tree settings, parallelization,
and validation options.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VCFProcessingConfig:
    """Configuration for VCF → matrix preprocessing (bcftools-based)."""

    # Variant-level filters
    qual_min: float = 30.0
    dp_min_variant: int = 10  # INFO/DP if present

    # Genotype-level filters (FORMAT)
    dp_min_genotype: int = 10
    gq_min: int = 20

    # Site-level filters (post-tags)
    max_missing: float = 0.20  # uses F_MISSING if present
    min_maf: Optional[float] = 0.01  # uses MAF if present (None disables)

    # Normalization / representation
    snp_gap_bp: int = 10
    normalize: bool = True
    remove_invariants: bool = True
    keep_only_snps: bool = True


@dataclass
class NetworkParserConfig:
    """
    Comprehensive configuration for the NetworkParser pipeline.
    """

    # Decision Tree Parameters (scikit-learn compatible)
    max_depth: Optional[int] = None
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    min_information_gain: float = 0.001

    # Legacy aliases for backward compatibility (used in older code)
    min_group_size: int = 10  # legacy alias for min_samples_split

    # Statistical Validation
    significance_level: float = 0.05
    fdr_threshold: float = 0.05
    multiple_testing_method: str = "fdr_bh"  # 'fdr_bh' or 'bonferroni'
    n_bootstrap_samples: int = 1000
    n_permutation_tests: int = 500
    min_bootstrap_support: float = 0.7

    # Performance
    n_jobs: int = -1
    random_state: int = 42
    memory_efficient: bool = False

    # VCF Processing (passed into DataLoader)
    vcf_processing: VCFProcessingConfig = field(default_factory=VCFProcessingConfig)

    # --- Legacy VCF fields (kept so older CLI/configs don't break) ---
    # These are mapped into vcf_processing in __post_init__.
    qual_threshold: Optional[float] = None
    min_dp_per_sample: Optional[int] = None
    min_gq_per_sample: Optional[int] = None
    max_missing_fraction: Optional[float] = None
    min_spacing_bp: Optional[int] = None

    # Output & FASTA Options
    generate_consensus_fasta: bool = True
    consensus_fasta_type: str = "individual"  # 'individual' or 'multi'
    include_intermediate_files: bool = True

    def __post_init__(self):
        if self.multiple_testing_method not in {"fdr_bh", "bonferroni"}:
            raise ValueError("multiple_testing_method must be 'fdr_bh' or 'bonferroni'")
        if self.consensus_fasta_type not in {"individual", "multi"}:
            raise ValueError("consensus_fasta_type must be 'individual' or 'multi'")

        # Set legacy tree field to match new standard
        self.min_group_size = self.min_samples_split

        # Map legacy VCF fields (if provided) into the new VCFProcessingConfig
        if self.qual_threshold is not None:
            self.vcf_processing.qual_min = float(self.qual_threshold)
        if self.min_dp_per_sample is not None:
            self.vcf_processing.dp_min_genotype = int(self.min_dp_per_sample)
            self.vcf_processing.dp_min_variant = int(self.min_dp_per_sample)
        if self.min_gq_per_sample is not None:
            self.vcf_processing.gq_min = int(self.min_gq_per_sample)
        if self.max_missing_fraction is not None:
            self.vcf_processing.max_missing = float(self.max_missing_fraction)
        if self.min_spacing_bp is not None:
            self.vcf_processing.snp_gap_bp = int(self.min_spacing_bp)

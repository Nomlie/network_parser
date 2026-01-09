# network_parser/config.py
"""
Configuration module for NetworkParser.

Defines configurable parameters controlling the behavior of the pipeline,
including statistical thresholds, decision tree settings, parallelization,
and validation options.

Defaults are optimized for microbial SNP datasets (e.g., thousands of samples,
tens to hundreds of thousands of variants).
"""

from dataclasses import dataclass
from typing import Optional

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
    min_group_size: int = 10  # ← This fixes your error!
    """Legacy alias for min_samples_split — kept for compatibility."""

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

    # VCF Filtering Parameters (used in DataLoader)
    qual_threshold: float = 30.0
    min_dp_per_sample: int = 10
    min_gq_per_sample: int = 20
    max_missing_fraction: float = 0.1
    min_spacing_bp: int = 10

    # Output & FASTA Options
    generate_consensus_fasta: bool = True
    consensus_fasta_type: str = "individual"  # 'individual' or 'multi'
    include_intermediate_files: bool = True

    def __post_init__(self):
        if self.multiple_testing_method not in {"fdr_bh", "bonferroni"}:
            raise ValueError("multiple_testing_method must be 'fdr_bh' or 'bonferroni'")
        if self.consensus_fasta_type not in {"individual", "multi"}:
            raise ValueError("consensus_fasta_type must be 'individual' or 'multi'")
        # Set legacy field to match new standard
        self.min_group_size = self.min_samples_split
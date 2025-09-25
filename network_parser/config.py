# network_parser/config.py
"""
Configuration module for NetworkParser.
This file defines a dataclass to hold configurable parameters for the pipeline,
allowing easy modification of settings like statistical thresholds and tree parameters.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class NetworkParserConfig:
    """
    Configuration class for NetworkParser pipeline parameters.
    This dataclass encapsulates settings that control the behavior of data processing,
    feature discovery, and validation steps in the pipeline.

    Attributes:
        max_depth (Optional[int]): Maximum depth for decision trees. If None, the depth
            is auto-tuned based on data size (default: None).
        min_group_size (int): Minimum number of samples required to split a node or
            form a leaf in the decision tree, ensuring statistical reliability
            (default: 5).
        significance_level (float): Alpha level for statistical tests (e.g., chi-squared),
            determining the threshold for significance (default: 0.05).
        n_bootstrap_samples (int): Number of bootstrap iterations for stability
            assessment, higher values improve confidence but increase computation time
            (default: 1000).
        n_permutation_tests (int): Number of permutation iterations for interaction
            significance testing, balancing accuracy and performance (default: 500).
        multiple_testing_method (str): Method for correcting multiple hypothesis tests
            ('fdr_bh' for Benjamini-Hochberg FDR or 'bonferroni' for Bonferroni correction,
            default: 'fdr_bh').
        min_information_gain (float): Minimum impurity decrease required for a split
            in the decision tree, preventing overfitting on noisy data (default: 0.001).
        n_jobs (int): Number of parallel jobs for computation. -1 uses all available
            CPU cores, 1 uses a single core (default: -1).
        random_state (int): Random seed for reproducibility in stochastic processes
            (default: 42)
    """
    max_depth: Optional[int] = None
    min_group_size: int = 5
    significance_level: float = 0.05
    n_bootstrap_samples: int = 1000
    n_permutation_tests: int = 500
    multiple_testing_method: str = 'fdr_bh'
    min_information_gain: float = 0.001
    n_jobs: int = -1
    random_state: int = 42  

# Optional: Add a test print to verify
if __name__ == "__main__":
    config = NetworkParserConfig()
    print(f"Significance level: {config.significance_level}")
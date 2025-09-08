"""
Configuration module for NetworkParser.
"""

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class NetworkParserConfig:
    """Configuration class for NetworkParser analysis."""
    # Decision tree parameters
    max_depth: Optional[int] = 5  # Maximum depth of decision trees
    min_samples_split: int = 2  # Minimum samples required to split a node
    min_samples_leaf: int = 1  # Minimum samples required at a leaf node
    max_features: Optional[str] = None  # Number of features to consider for splits ('auto', 'sqrt', 'log2', or None)
    
    # Statistical validation parameters
    bootstrap_iterations: int = 1000  # Number of bootstrap samples for statistical validation
    confidence_threshold: float = 0.95  # Confidence threshold for feature selection
    fdr_threshold: float = 0.05  # False discovery rate threshold for multiple testing
    min_group_size: int = 5  # Minimum group size for statistical tests
    correction_method: str = 'fdr_bh'  # Multiple testing correction method
    min_bootstrap_support: float = 0.8  # Minimum bootstrap support for features
    
    # Interaction analysis parameters
    max_interaction_order: int = 2  # Maximum order of epistatic interactions
    interaction_interpretation: bool = True  # Whether to interpret interactions
    
    # Processing parameters
    max_workers: int = 4  # Number of parallel workers
    memory_efficient: bool = False  # Enable memory-efficient processing
    chunk_size: int = 1000  # Chunk size for large datasets
    
    # Validation parameters
    cross_validation_folds: int = 5  # Number of cross-validation folds
    stability_threshold: float = 0.9  # Stability threshold for feature selection
    
    # Output parameters
    output_formats: List[str] = field(default_factory=lambda: ["text", "json"])  # Output formats
    include_matrices: bool = True  # Include matrices in output
    generate_plots: bool = True  # Generate visualization plots
    annotate_features: bool = True  # Annotate features with biological context
    biological_context: bool = True  # Include biological context in output
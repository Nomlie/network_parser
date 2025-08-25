# networkparser/config/config.py
"""
Configuration module for NetworkParser.
"""

from dataclasses import dataclass, field
from typing import List

@dataclass
class NetworkParserConfig:
    """Configuration class for NetworkParser analysis"""
    bootstrap_iterations: int = 1000
    confidence_threshold: float = 0.95
    max_interaction_order: int = 2
    fdr_threshold: float = 0.05
    min_group_size: int = 5
    correction_method: str = 'fdr_bh'
    max_workers: int = 4
    memory_efficient: bool = False
    chunk_size: int = 1000
    cross_validation_folds: int = 5
    stability_threshold: float = 0.9
    min_bootstrap_support: float = 0.8
    output_formats: List[str] = field(default_factory=lambda: ["text", "json"])
    include_matrices: bool = True
    generate_plots: bool = True
    annotate_features: bool = True
    biological_context: bool = True
    interaction_interpretation: bool = True
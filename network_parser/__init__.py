# networkparser/config/__init__.py
from .network_parser import run_networkparser_analysis
__version__ = "0.1.0"

"""
NetworkParser: Interpretable Framework for Epistatic Cluster Segregation Analysis

Designed for interpretable genomic feature discovery. 
It processes genomic data (e.g., SNP matrices) and metadata to identify discriminative features, detect 
epistatic interactions, and validate results statistically. The pipeline integrates decision tree-based 
analysis with advanced validation techniques, producing ML-ready outputs like graphs and adjacency matrices.
"""

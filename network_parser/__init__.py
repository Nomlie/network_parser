# networkparser/config/__init__.py
"""
NetworkParser: Interpretable Framework for Epistatic Cluster Segregation Analysis

A bioinformatics framework for identifying statistically validated features 
that drive cluster segregation using interpretable machine learning and 
epistatic interaction modeling.
"""
# network_parser/__init__.py

from .config import NetworkParserConfig
from .main import NetworkParser,  run_networkparser_analysis # <-- was network_parser.py, renamed to main.py

__all__ = ["NetworkParser", "NetworkParserConfig"]
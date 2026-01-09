# network_parser/__init__.py
"""
NetworkParser: Interpretable Genomic Feature Discovery Framework

A scalable, modular pipeline for identifying statistically validated genomic markers
and epistatic interactions driving phenotype segregation (e.g., antimicrobial resistance,
lineage diversification in microbes).

Core features:
- End-to-end processing from VCF/CSV/FASTA → clean binary SNP matrix
- High-quality biallelic SNP filtering (bcftools-based)
- Optional consensus pseudogenome FASTA generation (bcftools consensus)
- Interpretable decision tree-based feature discovery
- Rigorous statistical validation (bootstrap, permutation tests)
- Network integration and GNN-ready outputs
- Designed for microbial genomics (e.g., TB, Staphylococcus, etc.)

Author: Nomlindelo Mfuphi
GitHub: https://github.com/Nomlie/network_parser/
"""

from .config import NetworkParserConfig
from .network_parser import NetworkParser, run_networkparser_analysis
from .data_loader import DataLoader
from .statistical_validation import StatisticalValidator
from .decision_tree_builder import EnhancedDecisionTreeBuilder

# Package version (update this as you develop!)
__version__ = "0.2.0"  # Increment from 0.1.0 to reflect VCF + FASTA support

# Define what gets imported with "from network_parser import *"
__all__ = [
    "NetworkParserConfig",
    "NetworkParser",
    "run_networkparser_analysis",
    "DataLoader",
    "StatisticalValidator",
    "EnhancedDecisionTreeBuilder",
    "__version__"
]

# Optional: Friendly welcome message when imported
def _welcome():
    print("""
╔═══════════════════════════════════════════════════════════╗
║             Welcome to NetworkParser v{}             ║
║  Interpretable genomic feature discovery for microbes     ║
║                                                           ║
║  Now with full VCF → SNP matrix + consensus FASTA support ║
║  Perfect for AMR surveillance and lineage analysis        ║
╚═══════════════════════════════════════════════════════════╝
    """.format(__version__))


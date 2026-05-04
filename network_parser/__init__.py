"""
NetworkParser: Interpretable Genomic Feature Discovery Framework.

NetworkParser converts genomic variant matrices or VCF-derived feature spaces
into statistically defensible marker rankings, interpretable decision-tree
outputs, interaction summaries, and ML-ready matrices.
"""

from .config import NetworkParserConfig
from .data_loader import DataLoader
from .statistical_validation_branch import StatisticalValidatorBranch
from .ml_protocol import MLProtocolRunner
from .network_parser import (
    NetworkParser,
    run_networkparser_analysis,
    normalize_labels,
)
from .utils import normalize_sample_id

try:
    from .decision_tree_branch import DecisionTreeBranch
except ImportError:  # pragma: no cover
    DecisionTreeBranch = None  # type: ignore

try:
    from .two_level_protocol import TwoLevelProtocol
except ImportError:  # pragma: no cover
    TwoLevelProtocol = None  # type: ignore

try:
    from .query_engine import NetworkParserQueryEngine
except ImportError:  # pragma: no cover
    NetworkParserQueryEngine = None  # type: ignore

NetworkParserPipeline = NetworkParser

__version__ = "0.1.0"

__all__ = [
    "NetworkParserConfig",
    "DataLoader",
    "StatisticalValidatorBranch",
    "DecisionTreeBranch",
    "MLProtocolRunner",
    "NetworkParser",
    "NetworkParserPipeline",
    "run_networkparser_analysis",
    "normalize_sample_id",
    "normalize_labels",
    "TwoLevelProtocol",
    "NetworkParserQueryEngine",
    "__version__",
]
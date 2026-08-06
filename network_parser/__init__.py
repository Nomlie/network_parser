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
from .cross_validation import run_repeated_cv
from .model_evaluation import (
    evaluate_predictions,
    evaluate_prediction_table,
    evaluate_hierarchy_prediction_table,
    evaluate_hierarchy_branch_diagnostics,
    load_predictions_from_table,
    load_labels_from_metadata,
    resolve_run_artifact_dir,
    normalize_run_artifact_dir,
    run_networkparser_evaluation,
)

try:
    from .decision_tree_branch import DecisionTreeBranch
except ImportError:  # pragma: no cover
    DecisionTreeBranch = None  # type: ignore

try:
    from .hierarchy_protocol import HierarchyProtocol

    TwoLevelProtocol = HierarchyProtocol  # backward-compatible alias
except ImportError:  # pragma: no cover
    try:
        from .two_level_protocol import TwoLevelProtocol, HierarchyProtocol  # type: ignore
    except ImportError:
        HierarchyProtocol = None  # type: ignore
        TwoLevelProtocol = None  # type: ignore

try:
    from .query_engine import NetworkParserQueryEngine
except ImportError:  # pragma: no cover
    NetworkParserQueryEngine = None  # type: ignore

try:
    from .sequence_query_encoder import encode_raw_sequence_query
except ImportError:  # pragma: no cover
    encode_raw_sequence_query = None  # type: ignore

try:
    from .fastq_processor import FastqProcessor
except ImportError:  # pragma: no cover
    FastqProcessor = None  # type: ignore

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
    "evaluate_predictions",
    "evaluate_prediction_table",
    "evaluate_hierarchy_prediction_table",
    "evaluate_hierarchy_branch_diagnostics",
    "load_predictions_from_table",
    "load_labels_from_metadata",
    "resolve_run_artifact_dir",
    "normalize_run_artifact_dir",
    "run_networkparser_evaluation",
    "run_repeated_cv",
    "HierarchyProtocol",
    "TwoLevelProtocol",
    "NetworkParserQueryEngine",
    "encode_raw_sequence_query",
    "FastqProcessor",
    "__version__",
]

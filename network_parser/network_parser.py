# network_parser/network_parser.py
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

from network_parser.config import NetworkParserConfig
from network_parser.data_loader import DataLoader
from network_parser.decision_tree_builder import EnhancedDecisionTreeBuilder
from network_parser.statistical_validation import StatisticalValidator
from network_parser.utils import save_json, ensure_dir, timestamp

logger = logging.getLogger(__name__)


class NetworkParser:
    """
    Main orchestrator class for the NetworkParser pipeline.
    """

    def __init__(self, config: NetworkParserConfig):
        logger.info(f"Initializing NetworkParser with config: {vars(config)}")
        self.config = config

        # Data loading behavior (including folder-of-VCFs union-matrix mode) is driven via config
        self.loader = DataLoader(config=config, n_jobs=config.n_jobs)  # Uses enhanced DataLoader with bcftools + FASTA support

        self.validator = StatisticalValidator(config)
        self.tree_builder = EnhancedDecisionTreeBuilder(config)

    def run_pipeline(
        self,
        genomic_path: str,
        meta_path: Optional[str],
        label_column: str,
        known_markers_path: Optional[str],
        output_dir: str,
        validate_statistics: bool = False,
        validate_interactions: bool = False,
        ref_fasta: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute full NetworkParser pipeline.
        """
        output_dir_path = Path(output_dir)
        ensure_dir(output_dir_path)

        logger.info("Stage 1: Loading genomic matrix")
        genomic_df = self.loader.load_genomic_matrix(
            file_path=genomic_path,
            output_dir=output_dir,
            ref_fasta=ref_fasta
        )

        logger.info(f"Loaded genomic matrix with shape: {genomic_df.shape}")

        meta_df = None
        if meta_path:
            logger.info("Loading metadata")
            meta_df = self.loader.load_metadata(meta_path, output_dir=output_dir)
            logger.info(f"Loaded metadata with shape: {meta_df.shape}")

        known_markers = None
        if known_markers_path:
            logger.info("Loading known markers")
            known_markers = self.loader.load_known_markers(known_markers_path, output_dir=output_dir)
            logger.info(f"Loaded {len(known_markers)} known markers")

        # Stage 2: Decision tree feature discovery
        logger.info("Stage 2: Decision tree feature discovery")
        discovery_results = self.tree_builder.run_feature_discovery(
            genomic_df=genomic_df,
            meta_df=meta_df,
            label_column=label_column,
            known_markers=known_markers,
            output_dir=output_dir
        )

        # Stage 3: Statistical validation (optional)
        validation_results = {}
        if validate_statistics:
            logger.info("Stage 3: Statistical validation (features)")
            validation_results["features"] = self.validator.validate_features(
                genomic_df=genomic_df,
                meta_df=meta_df,
                label_column=label_column,
                discovered_features=discovery_results.get("ranked_features", []),
                output_dir=output_dir
            )

        if validate_interactions:
            logger.info("Stage 3: Statistical validation (interactions)")
            validation_results["interactions"] = self.validator.validate_interactions(
                genomic_df=genomic_df,
                meta_df=meta_df,
                label_column=label_column,
                interactions=discovery_results.get("epistatic_interactions", []),
                output_dir=output_dir
            )

        # Stage 4: Final synthesis / report writing
        logger.info("Stage 4: Writing final results")
        results = {
            "timestamp": timestamp(),
            "config": vars(self.config),
            "discovery": discovery_results,
            "validation": validation_results
        }

        results_path = output_dir_path / f"networkparser_results_{timestamp()}.json"
        save_json(results, results_path)
        logger.info(f"Saved final results: {results_path}")

        return results


def run_networkparser_analysis(
    genomic_path: str,
    meta_path: Optional[str],
    label_column: str,
    known_markers_path: Optional[str],
    output_dir: str,
    config: NetworkParserConfig,
    validate_statistics: bool = False,
    validate_interactions: bool = False,
    ref_fasta: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience wrapper to run the pipeline.
    """
    parser = NetworkParser(config)
    return parser.run_pipeline(
        genomic_path=genomic_path,
        meta_path=meta_path,
        label_column=label_column,
        known_markers_path=known_markers_path,
        output_dir=output_dir,
        validate_statistics=validate_statistics,
        validate_interactions=validate_interactions,
        ref_fasta=ref_fasta
    )

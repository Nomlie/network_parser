# network_parser/network_parser.py
import logging
import pandas as pd
import numpy as np
import os
import time
from pathlib import Path
import json
import networkx as nx  # For graph creation
from typing import Optional, Dict, List, Tuple

from .config import NetworkParserConfig
from .data_loader import DataLoader
from .statistical_validation import StatisticalValidator
from .decision_tree_builder import EnhancedDecisionTreeBuilder

logger = logging.getLogger(__name__)


class NetworkParser:
    """
    Main orchestrator for NetworkParser pipeline.
    """

    def __init__(self, config: NetworkParserConfig):
        logger.info(f"Initializing NetworkParser with config: {vars(config)}")
        self.config = config
        self.loader = DataLoader(vcf_config=config.vcf_processing)  # Uses config-driven VCF processing
        self.tree_builder = EnhancedDecisionTreeBuilder(config)
        self.validator = StatisticalValidator(config)

    def run_pipeline(
        self,
        genomic_path: str,
        meta_path: Optional[str] = None,
        label_column: str = None,
        known_markers_path: Optional[str] = None,
        regions: Optional[str] = None,
        output_dir: str = "results/",
        validate_statistics: bool = True,
        validate_interactions: bool = True,
        ref_fasta: Optional[str] = None,
    ) -> Dict:
        """
        Run full end-to-end pipeline.

        Parameters
        ----------
        genomic_path : str
            CSV matrix OR VCF file/folder.
        meta_path : Optional[str]
            Metadata file.
        label_column : str
            Label column to use as phenotype.
        known_markers_path : Optional[str]
            Optional known marker list.
        regions : Optional[str]
            Optional bcftools --regions/--targets string (e.g. "chrom:start-end" or BED file).
        output_dir : str
            Output directory.
        validate_statistics : bool
            Run pre-tree statistical validation (association testing + multiple testing correction).
        validate_interactions : bool
            Run post-tree interaction permutation validation.
        ref_fasta : Optional[str]
            Reference FASTA for consensus generation and reference-aware normalization.
        """

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # Load genomic data → clean binary matrix (handles VCF filtering + consensus FASTA if ref_fasta given)
        genomic_df = self.loader.load_genomic_matrix(
            file_path=genomic_path,          # ← FIXED: changed from genomic_path= to file_path=
            output_dir=output_dir,
            ref_fasta=ref_fasta,             # Triggers bcftools consensus for VCF inputs
            regions=regions
        )

        # Load optional metadata
        meta_df = None
        if meta_path:
            meta_df = pd.read_csv(meta_path, sep=None, engine="python")
            if label_column not in meta_df.columns:
                raise ValueError(f"Label column '{label_column}' not found in metadata columns: {list(meta_df.columns)}")

        # Align samples
        if meta_df is not None:
            meta_df[label_column] = meta_df[label_column].astype(str)
            meta_df.index = meta_df.iloc[:, 0].astype(str) if meta_df.index.name is None else meta_df.index.astype(str)

            common = genomic_df.index.intersection(meta_df.index)
            if len(common) == 0:
                raise ValueError("No overlapping sample IDs between genomic matrix and metadata.")
            genomic_df = genomic_df.loc[common]
            meta_df = meta_df.loc[common]

        # Extract labels
        y = None
        if meta_df is not None and label_column:
            y = meta_df[label_column].values

        # Stage 2: Feature discovery (pre-tree filtering happens inside tree builder)
        tree_results = self.tree_builder.build_tree(genomic_df, y)

        # Stage 3: Statistical validation
        stats_results = {}
        if validate_statistics:
            stats_results = self.validator.validate_features(genomic_df, y, tree_results)

        interaction_results = {}
        if validate_interactions:
            interaction_results = self.validator.validate_interactions(genomic_df, y, tree_results)

        # Stage 4: Integration and network creation
        results = {
            "tree_results": tree_results,
            "stats_results": stats_results,
            "interaction_results": interaction_results,
        }

        # Save results JSON
        out_json = output_dir_path / f"networkparser_results_{int(time.time())}.json"
        with open(out_json, "w") as fh:
            json.dump(results, fh, indent=2)
        logger.info(f"Saved results: {out_json}")

        return results


def run_networkparser_analysis(**kwargs):
    """
    Entry-point function for pipeline execution (used by CLI).
    """
    logger.info("Initializing NetworkParser with provided configuration.")
    config = kwargs.pop('config', NetworkParserConfig())
    parser = NetworkParser(config=config)

    try:
        parser.run_pipeline(**kwargs)
    finally:
        # Clean up temporary files from DataLoader (e.g., intermediate VCFs)
        if hasattr(parser.loader, 'cleanup'):
            parser.loader.cleanup()


if __name__ == "__main__":
    import sys
    raise SystemExit("This module is intended to be run via the CLI entrypoint.")

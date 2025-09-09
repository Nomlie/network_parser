"""
NetworkParser main module for genomic feature discovery.

This module orchestrates the pipeline:
1. Data loading and preprocessing
2. Decision tree-based feature discovery
3. Statistical validation
4. Interaction detection
5. Result export

Supports intermediate file saving to output_dir.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import pickle
from typing import Optional, Dict, List, Tuple

# Use absolute imports to avoid circular imports
try:
    from network_parser.config import NetworkParserConfig
    from network_parser.data_loader import DataLoader
    from network_parser.decision_tree_builder import EnhancedDecisionTreeBuilder
    from network_parser.statistical_validation import StatisticalValidator
except ImportError as e:
    print(f"ImportError: {e}. Ensure 'network_parser' is in PYTHONPATH and all modules are correctly placed.")
    import sys
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class NetworkParser:
    """
    Main class for the NetworkParser pipeline.
    """
    def __init__(self, config: NetworkParserConfig):
        """
        Initialize NetworkParser with configuration.

        Args:
            config: Configuration object for pipeline parameters.
        """
        self.config = config
        self.loader = DataLoader()
        self.tree_builder = EnhancedDecisionTreeBuilder(config)
        self.validator = StatisticalValidator(config)
        logger.info("Initialized NetworkParser with provided configuration.")

    def load_data(self,
                  genomic_data_path: str,
                  metadata_path: Optional[str],
                  label_column: str,
                  known_markers_path: Optional[str],
                  output_dir: Optional[str]) -> Tuple[pd.DataFrame, pd.Series, Optional[List[str]]]:
        """
        Load and preprocess genomic and metadata.

        Args:
            genomic_data_path: Path to genomic data file.
            metadata_path: Path to metadata file (optional).
            label_column: Column name for labels.
            known_markers_path: Path to known markers file (optional).
            output_dir: Directory to save intermediate files (optional).

        Returns:
            Tuple of (aligned genomic DataFrame, labels Series, known markers list).
        """
        logger.info("Loading genomic data...")
        genomic_data = self.loader.load_genomic_matrix(genomic_data_path, output_dir=output_dir)

        metadata = None
        if metadata_path:
            logger.info("Loading metadata...")
            metadata = self.loader.load_metadata(metadata_path, output_dir=output_dir)

        known_markers = None
        if known_markers_path:
            logger.info("Loading known markers...")
            known_markers = self.loader.load_known_markers(known_markers_path, output_dir=output_dir)

        logger.info("Aligning data...")
        aligned_data, aligned_labels = self.loader.align_data(
            genomic_data, metadata, label_column, output_dir=output_dir
        )

        return aligned_data, aligned_labels, known_markers

    def run_feature_discovery(self,
                             data: pd.DataFrame,
                             labels: pd.Series,
                             output_dir: Optional[str] = None) -> Dict:
        """
        Run feature discovery and validation pipeline.

        Args:
            data: Genomic data DataFrame.
            labels: Labels Series.
            output_dir: Directory to save intermediate files (optional).

        Returns:
            Dictionary with discovered features, trees, and validation results.
        """
        logger.info(f"Starting feature discovery on {len(data)} samples with {len(data.columns)} features...")
        all_features = list(data.columns)

        # Feature discovery with decision trees
        feature_results = self.tree_builder.discover_features(data, labels, all_features, output_dir=output_dir)

        # Statistical validation
        bootstrap_results = self.validator.bootstrap_validation(data, labels, all_features, output_dir=output_dir)
        chi2_results = self.validator.chi_squared_test(data[all_features], labels, output_dir=output_dir)
        corrected_results = self.validator.multiple_testing_correction(chi2_results, output_dir=output_dir)

        # Compile results
        results = {
            'discovered_features': feature_results['discovered_features'],
            'decision_trees': feature_results['decision_trees'],
            'feature_confidence': feature_results['feature_confidence'],
            'epistatic_interactions': feature_results['epistatic_interactions'],
            'bootstrap_validation': bootstrap_results,
            'chi_squared_tests': chi2_results,
            'corrected_p_values': corrected_results
        }

        # Save intermediate results
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "decision_tree_features.json", 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved decision tree features to: {output_dir / 'decision_tree_features.json'}")

        return results

    def validate_interactions(self,
                             data: pd.DataFrame,
                             labels: pd.Series,
                             interactions: List[Tuple[str, str]],
                             output_dir: Optional[str] = None) -> Dict:
        """
        Validate epistatic interactions with permutation tests.

        Args:
            data: Genomic data DataFrame.
            labels: Labels Series.
            interactions: List of feature pairs to validate.
            output_dir: Directory to save intermediate files (optional).

        Returns:
            Dictionary with interaction validation results.
        """
        logger.info("Validating epistatic interactions...")
        interaction_results = self.validator.permutation_test_interactions(
            data, labels, interactions, output_dir=output_dir
        )

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "interaction_validation.json", 'w') as f:
                json.dump(interaction_results, f, indent=2)
            logger.info(f"Saved interaction validation results to: {output_dir / 'interaction_validation.json'}")

        return interaction_results

def run_networkparser_analysis(
    genomic_data_path: str,
    metadata_path: Optional[str] = None,
    label_column: str = "label",
    known_markers_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    config: Optional[NetworkParserConfig] = None,
    validate_statistics: bool = False,
    validate_interactions: bool = False,
    export_format: str = "json"
) -> None:
    """
    Run the full NetworkParser pipeline.

    Args:
        genomic_data_path: Path to genomic data file.
        metadata_path: Path to metadata file (optional).
        label_column: Column name for labels.
        known_markers_path: Path to known markers file (optional).
        output_dir: Directory to save results and intermediate files (optional).
        config: Configuration object (optional).
        validate_statistics: Whether to run statistical validation.
        validate_interactions: Whether to validate epistatic interactions.
        export_format: Format for saving results ('json' or 'pickle').

    Returns:
        None. Saves results to output_dir.
    """
    try:
        logger.info(f"Running network_parser.py from: {Path(__file__).resolve()}")
        config = config or NetworkParserConfig()
        parser = NetworkParser(config)

        # Load data
        data, labels, known_markers = parser.load_data(
            genomic_data_path, metadata_path, label_column, known_markers_path, output_dir
        )

        # Run feature discovery
        results = parser.run_feature_discovery(data, labels, output_dir=output_dir)

        # Validate interactions if requested
        if validate_interactions:
            interactions = results.get('epistatic_interactions', [])
            if interactions:
                interaction_results = parser.validate_interactions(data, labels, interactions, output_dir=output_dir)
                results['interaction_validation'] = interaction_results

        # Compare with known markers if provided
        if known_markers:
            logger.info("Comparing with known markers...")
            known_results = parser.validator.validate_feature_set(
                data, labels, results['discovered_features'], baseline_features=known_markers, output_dir=output_dir
            )
            results['known_marker_comparison'] = known_results

        # Save final results
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"networkparser_results_{timestamp}.{export_format}"

            if export_format == "json":
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
            elif export_format == "pickle":
                with open(output_file, 'wb') as f:
                    pickle.dump(results, f)

            logger.info(f"Saved final results to: {output_file}")

    except Exception as e:
        logger.error(f"NetworkParser pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NetworkParser direct execution (use cli.py for full functionality).")
    parser.add_argument("--genomic", required=True, type=str, help="Path to genomic matrix file.")
    parser.add_argument("--label", required=True, type=str, help="Column in genomic data or metadata with labels.")
    parser.add_argument("--meta", type=str, default=None, help="Optional path to metadata file.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save results.")
    args = parser.parse_args()

    run_networkparser_analysis(
        genomic_data_path=args.genomic,
        metadata_path=args.meta,
        label_column=args.label,
        output_dir=args.output_dir
    )

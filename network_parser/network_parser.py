# network_parser.py
import logging
import pandas as pd
import numpy as np
from datetime import datetime
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
    Orchestrates the NetworkParser pipeline.
    """

    def __init__(self, config: NetworkParserConfig):
        logger.info(f"Initializing NetworkParser with config: {vars(config)}")
        self.config = config
        self.loader = DataLoader()
        self.validator = StatisticalValidator(config)
        self.tree_builder = EnhancedDecisionTreeBuilder(config)

    def run_pipeline(self, genomic_path: str, meta_path: Optional[str] = None, label_column: str = None,
                     known_markers_path: Optional[str] = None, output_dir: str = "results/",
                     validate_statistics: bool = True, validate_interactions: bool = True) -> Dict:
        """
        Execute the full pipeline: load data, discover features, validate, and integrate results.

        Args:
            genomic_path (str): Path to the genomic data file.
            meta_path (Optional[str]): Path to the metadata file. Defaults to None.
            label_column (str): Name of the column containing labels.
            known_markers_path (Optional[str]): Path to known markers file. Defaults to None.
            output_dir (str, optional): Output directory. Defaults to "results/".
            validate_statistics (bool, optional): Whether to run statistical validation. Defaults to True.
            validate_interactions (bool, optional): Whether to validate interactions. Defaults to True.

        Returns:
            dict: Pipeline results.
        """
        logger.info("📥 Stage 1: Input Processing")
        data, labels, known_markers = self._load_and_preprocess(
            genomic_path, meta_path, label_column, known_markers_path, output_dir
        )

        logger.info("🌳 Stage 2: Feature Discovery")
        chi2_results = self.validator.chi_squared_test(data, labels, output_dir)
        corrected = self.validator.multiple_testing_correction(chi2_results, output_dir=output_dir)
        significant_features = [f for f, res in corrected.items() if res['significant']]
        logger.info(f"Filtered {len(significant_features)} significant features.")
        discovery_results = self.tree_builder.discover_features(data, labels, significant_features, output_dir)

        logger.info("✅ Stage 3: Statistical Validation")
        if validate_statistics:
            bootstrap = self.validator.bootstrap_validation(data, labels, significant_features, output_dir)
            discovery_results['bootstrap'] = bootstrap

        if validate_interactions and discovery_results.get('epistatic_interactions'):
            logger.info("🔄 Stage 3b: Interaction Validation")
            interactions = [(i['parent'], i['child']) for i in discovery_results['epistatic_interactions']]
            interaction_results = self.validator.permutation_test_interactions(data, labels, interactions, output_dir)
            discovery_results['interaction_validation'] = interaction_results

        logger.info("🔗 Stage 4: Integration")
        integrated = self._integrate_features(discovery_results, data, labels, output_dir)

        logger.info("📤 Stage 5: Output Generation")
        results = {**discovery_results, **integrated}
        if known_markers:
            results['known_comparison'] = self._compare_known_markers(data, labels, significant_features, known_markers)

        if output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            with open(output_path / f"networkparser_results_{timestamp}.json", 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved final results to: {output_path / f'networkparser_results_{timestamp}.json'}")

        logger.info("✅ Pipeline completed successfully")
        return results

    def _load_and_preprocess(self, genomic_path: str, meta_path: Optional[str],
                             label_column: str, known_markers_path: Optional[str],
                             out_dir: str) -> Tuple[pd.DataFrame, pd.Series, Optional[List[str]]]:
        """Stage 1: Load/align data (as per diagram)."""
        logger.info(f"Loading genomic data from: {genomic_path}")
        genomic = self.loader.load_genomic_matrix(genomic_path, out_dir)
        meta = self.loader.load_metadata(meta_path, out_dir) if meta_path else None
        markers = self.loader.load_known_markers(known_markers_path, out_dir) if known_markers_path else None
        logger.info("Aligning data...")
        aligned_data, aligned_labels = self.loader.align_data(genomic, meta, label_column, out_dir)
        logger.info(f"Aligned data: {len(aligned_labels)} samples, {aligned_data.shape[1]} features retained.")
        return aligned_data, aligned_labels, markers

    def _integrate_features(self, results: Dict, data: pd.DataFrame, labels: pd.Series, output_dir: Optional[str]) -> Dict:
        """Stage 4: Integrate features into networks and rankings."""
        ranked = sorted(results['feature_confidence'].items(), key=lambda x: x[1]['confidence'], reverse=True)
        logger.info(f"Ranked {len(ranked)} features by confidence.")

        # Sample-Feature Network (bipartite)
        sample_feature_graph = nx.Graph()
        for sample in data.index:
            sample_feature_graph.add_node(sample, type='sample')
        for feature in results['discovered_features']:
            sample_feature_graph.add_node(feature, type='feature')
            for sample in data.index[data[feature] == 1]:
                sample_feature_graph.add_edge(sample, feature)

        # Interaction Graph
        interaction_graph = nx.Graph()
        for i in results.get('epistatic_interactions', []):
            interaction_graph.add_edge(i['parent'], i['child'], weight=i.get('strength', 1.0))

        if output_dir:
            output_path = Path(output_dir)
            nx.write_graphml(sample_feature_graph, output_path / "sample_feature_network.graphml")
            nx.write_graphml(interaction_graph, output_path / "interaction_graph.graphml")
            logger.info("Saved network graphs to GraphML files.")
            # iGNN matrices: adjacency
            np.savez(output_path / "ignn_matrices.npz",
                     sample_feature_adj=nx.to_numpy_array(sample_feature_graph),
                     interaction_adj=nx.to_numpy_array(interaction_graph))
            logger.info("Saved iGNN adjacency matrices.")

        return {
            'ranked_features': ranked,
            'sample_feature_network': {'nodes': list(sample_feature_graph.nodes()), 'edges': list(sample_feature_graph.edges())},
            'interaction_graph': {'nodes': list(interaction_graph.nodes()), 'edges': list(interaction_graph.edges())}
        }

    def _compare_known_markers(self, data: pd.DataFrame, labels: pd.Series, discovered: List[str], known: List[str]) -> Dict:
        """Compare discovered features with known markers."""
        overlap = set(discovered) & set(known)
        logger.info(f"Found {len(overlap)} overlapping markers with known set.")
        return {'overlap': list(overlap), 'overlap_ratio': len(overlap) / len(known) if known else 0}

def run_networkparser_analysis(**kwargs):
    """
    Entry-point function for pipeline execution.
    """
    logger.info("Initializing NetworkParser with provided configuration.")
    config = kwargs.pop('config', NetworkParserConfig())
    parser = NetworkParser(config)
    logger.info("Starting pipeline execution...")
    return parser.run_pipeline(**kwargs)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python network_parser.py --genomic data.csv --label phenotype")
        sys.exit(1)
    # Simple arg parse for direct run
    results = run_networkparser_analysis(genomic_path=sys.argv[1], label_column='label')
    print(json.dumps(results, indent=2))
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
        self.loader = DataLoader()  # Uses enhanced DataLoader with bcftools + FASTA support
        self.validator = StatisticalValidator(config)
        self.tree_builder = EnhancedDecisionTreeBuilder(config)

    def run_pipeline(self, genomic_path: str, meta_path: Optional[str] = None,
                     label_column: str = None, known_markers_path: Optional[str] = None,
                     output_dir: str = "results/", validate_statistics: bool = True,
                     validate_interactions: bool = True,
                     ref_fasta: Optional[str] = None) -> Dict:
        """
        Execute the full pipeline: load data, discover features, validate, and integrate results.
        """
        logger.info("\033[1m📥 Stage 1: Input Processing\033[0m")

        # Load genomic data → clean binary matrix (handles VCF filtering + consensus FASTA if ref_fasta given)
        genomic_df = self.loader.load_genomic_matrix(
            file_path=genomic_path,          # ← FIXED: changed from genomic_path= to file_path=
            output_dir=output_dir,
            ref_fasta=ref_fasta              # Triggers bcftools consensus for VCF inputs
        )

        # Load optional metadata and known markers
        meta = self.loader.load_metadata(meta_path, output_dir) if meta_path else None
        known_markers = self.loader.load_known_markers(known_markers_path, output_dir) if known_markers_path else None

        # Align data and extract labels (supports labels in matrix or metadata)
        data, labels = self.loader.align_data(
            genomic_data=genomic_df,
            metadata=meta,
            label_column=label_column,
            output_dir=output_dir
        )

        logger.info(f"Final aligned dataset: {len(data)} samples, {data.shape[1]} features")

        logger.info("\033[1m🌳 Stage 2: Feature Discovery\033[0m")
        chi2_results = self.validator.chi_squared_test(data, labels, output_dir)
        corrected = self.validator.multiple_testing_correction(chi2_results, output_dir=output_dir)
        significant_features = [f for f, res in corrected.items() if res['significant']]
        logger.info(f"Filtered {len(significant_features)} significant features.")

        discovery_results = self.tree_builder.discover_features(data, labels, significant_features, output_dir)

        logger.info("\033[1m✅ Stage 3: Statistical Validation\033[0m")
        if validate_statistics:
            bootstrap = self.validator.bootstrap_validation(data, labels, significant_features, output_dir)
            discovery_results['bootstrap'] = bootstrap

        if validate_interactions and discovery_results.get('epistatic_interactions'):
            logger.info("\033[1m🔄 Stage 3b: Interaction Validation\033[0m")
            interactions = [(i['parent'], i['child']) for i in discovery_results['epistatic_interactions']]
            interaction_results = self.validator.permutation_test_interactions(data, labels, interactions, output_dir)
            discovery_results['interaction_validation'] = interaction_results

        logger.info("\033[1m🔗 Stage 4: Integration\033[0m")
        integrated = self._integrate_features(discovery_results, data, labels, output_dir)

        logger.info("\033[1m📤 Stage 5: Output Generation\033[0m")
        results = {**discovery_results, **integrated}
        if known_markers:
            results['known_comparison'] = self._compare_known_markers(data, labels, significant_features, known_markers)

        if output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            final_json = output_path / f"networkparser_results_{timestamp}.json"
            with open(final_json, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved final results to: {final_json}")

            # Extra note if consensus FASTAs were generated
            if ref_fasta and Path(genomic_path).suffix.lower() in {'.vcf', '.vcf.gz'}:
                consensus_dir = output_path / "consensus_fastas"
                if consensus_dir.exists():
                    logger.info(f"Consensus FASTA pseudogenomes saved in: {consensus_dir}")

        logger.info("✅ Pipeline completed successfully")
        return results

    def _integrate_features(self, results: Dict, data: pd.DataFrame, labels: pd.Series,
                            output_dir: Optional[str]) -> Dict:
        """Stage 4: Integrate features into networks and rankings."""
        ranked = sorted(results['feature_confidence'].items(),
                        key=lambda x: x[1]['confidence'], reverse=True)
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

            np.savez(output_path / "ignn_matrices.npz",
                     sample_feature_adj=nx.to_numpy_array(sample_feature_graph),
                     interaction_adj=nx.to_numpy_array(interaction_graph))
            logger.info("Saved iGNN adjacency matrices.")

        return {
            'ranked_features': ranked,
            'sample_feature_network': {'nodes': list(sample_feature_graph.nodes()),
                                       'edges': list(sample_feature_graph.edges())},
            'interaction_graph': {'nodes': list(interaction_graph.nodes()),
                                  'edges': list(interaction_graph.edges())}
        }

    def _compare_known_markers(self, data: pd.DataFrame, labels: pd.Series,
                               discovered: List[str], known: List[str]) -> Dict:
        """Compare discovered features with known markers."""
        overlap = set(discovered) & set(known)
        logger.info(f"Found {len(overlap)} overlapping markers with known set.")
        return {'overlap': list(overlap),
                'overlap_ratio': len(overlap) / len(known) if known else 0}


def run_networkparser_analysis(**kwargs):
    """
    Entry-point function for pipeline execution (used by CLI).
    """
    logger.info("Initializing NetworkParser with provided configuration.")
    config = kwargs.pop('config', NetworkParserConfig())
    parser = NetworkParser(config)
    logger.info("Starting pipeline execution...")
    try:
        return parser.run_pipeline(**kwargs)
    finally:
        # Clean up temporary files from DataLoader (e.g., intermediate VCFs)
        if hasattr(parser.loader, 'cleanup'):
            parser.loader.cleanup()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python network_parser.py <genomic_file_or_dir> <label_column>")
        sys.exit(1)
    # Simple direct run (for testing)
    results = run_networkparser_analysis(
        genomic_path=sys.argv[1],
        label_column=sys.argv[2]
    )
    print(json.dumps(results, indent=2))
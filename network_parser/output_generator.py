# network_parser/output_generator.py
"""
Output generation and visualization module.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

from .config import NetworkParserConfig

class OutputGenerator:
    """Generates various output formats and visualizations"""
    
    def __init__(self, config: NetworkParserConfig):
        self.config = config
    
    def generate_text_report(self, results: Dict, output_dir: str) -> None:
        """Generate comprehensive text report"""
        output_path = Path(output_dir) / "networkparser_results.txt"
        
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("NetworkParser Analysis Results (v1.0.0)\n")
            f.write("=" * 80 + "\n\n")
            
            # Dataset Summary
            f.write("Dataset Summary:\n")
            f.write(f"  Samples: {results['dataset_info']['n_samples']}\n")
            f.write(f"  Features: {results['dataset_info']['n_features']} genomic markers\n")
            f.write(f"  Groups: {len(results['dataset_info']['unique_labels'])} classes\n\n")
            
            # Hierarchical Analysis
            if 'hierarchical_analysis' in results:
                f.write("Hierarchical Analysis:\n")
                for level, info in results['hierarchical_analysis']['cluster_levels'].items():
                    n_clusters = level.split('_')[1]
                    f.write(f"  Level {n_clusters}: {n_clusters} clusters "
                           f"(silhouette: {info['silhouette_score']:.3f})\n")
                f.write("\n")
            
            # Discriminative Features
            f.write("Discriminative Features:\n")
            significant_features = [f for f, data in results['statistical_validation'].items() 
                                  if data['significant']]
            f.write(f"  Single Features: {len(significant_features)} significant markers "
                   f"(FDR < {self.config.fdr_threshold})\n")
            
            if 'interactions' in results:
                significant_interactions = [i for i, data in results['interactions'].items() 
                                          if data.get('bootstrap_stable', False)]
                f.write(f"  Epistatic Interactions: {len(significant_interactions)} "
                       f"validated interactions\n")
            
            if 'known_markers_validated' in results:
                f.write(f"  Known Markers Validated: {results['known_markers_validated']['n_validated']}/"
                       f"{results['known_markers_validated']['n_total']} previously known features confirmed\n")
            f.write("\n")
            
            # Top Features
            f.write("Top Discriminative Features:\n")
            sorted_features = sorted(
                [(f, data) for f, data in results['statistical_validation'].items() if data['significant']], 
                key=lambda x: x[1]['corrected_p_value']
            )[:10]
            
            for feature, data in sorted_features:
                f.write(f"  {feature}: p = {data['corrected_p_value']:.6f}\n")
            f.write("\n")
            
            # Decision Tree Summary
            if 'decision_tree' in results:
                f.write("Decision Tree Summary:\n")
                f.write(f"  Accuracy: {results['decision_tree']['accuracy']:.3f}\n")
                f.write("  Tree Rules:\n")
                tree_rules = results['decision_tree']['tree_rules'].split('\n')
                for rule in tree_rules[:10]:  # First 10 lines
                    f.write(f"    {rule}\n")
                f.write("\n")
            
            # Statistical Validation
            f.write("Statistical Validation:\n")
            f.write(f"  Bootstrap iterations: {self.config.bootstrap_iterations}\n")
            f.write(f"  FDR correction: {self.config.correction_method}\n")
            f.write(f"  Confidence threshold: {self.config.confidence_threshold}\n")
        
    def generate_json_output(self, results: Dict, output_dir: str) -> None:
        """Generate JSON output for programmatic access"""
        output_path = Path(output_dir) / "networkparser_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = self._convert_for_json(results)
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
    
    def generate_gnn_matrices(self, data: pd.DataFrame, results: Dict, 
                            output_dir: str) -> None:
        """Generate matrices ready for Graph Neural Network training"""
        gnn_dir = Path(output_dir) / "gnn_ready_matrices"
        gnn_dir.mkdir(exist_ok=True)
        
        # Extract significant features
        significant_features = [f for f, data in results['statistical_validation'].items() 
                              if data['significant']]
        
        # Node features (validated genomic markers)
        node_features = data[significant_features]
        node_features.to_csv(gnn_dir / "node_features.csv")
        
        # Edge features (epistatic interactions)
        if 'interactions' in results:
            interaction_matrix = pd.DataFrame(index=data.index)
            
            for interaction_name, interaction_data in results['interactions'].items():
                if interaction_data.get('bootstrap_stable', False):
                    interaction_matrix[interaction_name] = interaction_data['interaction_values']
            
            if not interaction_matrix.empty:
                interaction_matrix.to_csv(gnn_dir / "edge_features.csv")
        
        # Combined matrix
        combined_features = [node_features]
        if 'interactions' in results and not interaction_matrix.empty:
            combined_features.append(interaction_matrix)
        
        combined_matrix = pd.concat(combined_features, axis=1)
        combined_matrix.to_csv(gnn_dir / "combined_features.csv")
        
        # Feature annotations
        feature_annotations = {}
        for feature in significant_features:
            feature_annotations[feature] = {
                'type': 'individual_marker',
                'p_value': results['statistical_validation'][feature]['corrected_p_value'],
                'significant': results['statistical_validation'][feature]['significant']
            }
        
        if 'interactions' in results:
            for interaction_name, interaction_data in results['interactions'].items():
                if interaction_data.get('bootstrap_stable', False):
                    feature_annotations[interaction_name] = {
                        'type': 'epistatic_interaction',
                        'features': interaction_data['features'],
                        'p_value': interaction_data['p_value'],
                        'bootstrap_support': interaction_data['bootstrap_support']
                    }
        
        with open(gnn_dir / "feature_annotations.json", 'w') as f:
            json.dump(feature_annotations, f, indent=2)
    
    def generate_plots(self, results: Dict, output_dir: str) -> None:
        """Generate visualization plots"""
        plots_dir = Path(output_dir) / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Feature importance plot
        if 'statistical_validation' in results:
            self._plot_feature_importance(results['statistical_validation'], plots_dir)
        
        # Hierarchical clustering dendrogram
        if 'hierarchical_analysis' in results:
            self._plot_dendrogram(results['hierarchical_analysis'], plots_dir)
        
        # Interaction network
        if 'interactions' in results:
            self._plot_interaction_network(results['interactions'], plots_dir)
    
    def _plot_feature_importance(self, validation_results: Dict, plots_dir: Path) -> None:
        """Plot feature importance"""
        significant_features = {f: data for f, data in validation_results.items() 
                              if data['significant']}
        
        if not significant_features:
            return
        
        features = list(significant_features.keys())[:20]  # Top 20
        p_values = [-np.log10(significant_features[f]['corrected_p_value']) for f in features]
        
        plt.figure(figsize=(12, 8))
        plt.barh(features, p_values)
        plt.xlabel('-log10(FDR-corrected p-value)')
        plt.title('Top Significant Features')
        plt.tight_layout()
        plt.savefig(plots_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_dendrogram(self, hierarchical_results: Dict, plots_dir: Path) -> None:
        """Plot hierarchical clustering dendrogram"""
        from scipy.cluster.hierarchy import dendrogram
        
        plt.figure(figsize=(15, 8))
        dendrogram(hierarchical_results['linkage_matrix'], truncate_mode='lastp', p=30)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.savefig(plots_dir / "dendrogram.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_interaction_network(self, interactions: Dict, plots_dir: Path) -> None:
        """Plot epistatic interaction network"""
        significant_interactions = {i: data for i, data in interactions.items() 
                                  if data.get('bootstrap_stable', False)}
        
        if not significant_interactions:
            return
        
        # Simple network visualization using matplotlib
        plt.figure(figsize=(12, 10))
        
        # Extract all features involved in interactions
        all_features = set()
        for interaction_data in significant_interactions.values():
            all_features.update(interaction_data['features'])
        
        # Position features in a circle
        n_features = len(all_features)
        feature_positions = {}
        for i, feature in enumerate(sorted(all_features)):
            angle = 2 * np.pi * i / n_features
            x = np.cos(angle)
            y = np.sin(angle)
            feature_positions[feature] = (x, y)
            plt.scatter(x, y, s=100, c='lightblue', zorder=3)
            plt.text(x*1.1, y*1.1, feature, ha='center', va='center', fontsize=8)
        
        # Draw interaction edges
        for interaction_name, interaction_data in significant_interactions.items():
            features = interaction_data['features']
            if len(features) == 2:  # Only plot pairwise interactions
                x1, y1 = feature_positions[features[0]]
                x2, y2 = feature_positions[features[1]]
                
                # Line thickness based on interaction strength
                strength = interaction_data.get('cramers_v', 0.1)
                plt.plot([x1, x2], [y1, y2], 'r-', alpha=0.6, 
                        linewidth=strength*10, zorder=1)
        
        plt.title('Epistatic Interaction Network')
        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(plots_dir / "interaction_network.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _convert_for_json(self, obj):
        """Convert numpy arrays and other non-JSON serializable objects"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._convert_for_json(obj.__dict__)
        else:
            return obj
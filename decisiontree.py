#!/usr/bin/env python3
"""
NetworkParser: Interpretable Framework for Epistatic Cluster Segregation Analysis

A bioinformatics framework for identifying statistically validated features 
that drive cluster segregation using interpretable machine learning and 
epistatic interaction modeling.
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Set
from dataclasses import dataclass, asdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from scipy.spatial.distance import pdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class NetworkParserConfig:
    """Configuration class for NetworkParser analysis"""
    bootstrap_iterations: int = 1000
    confidence_threshold: float = 0.95
    max_interaction_order: int = 2
    fdr_threshold: float = 0.05
    min_group_size: int = 5
    correction_method: str = 'fdr_bh'
    max_workers: int = 4
    memory_efficient: bool = False
    chunk_size: int = 1000
    cross_validation_folds: int = 5
    stability_threshold: float = 0.9
    min_bootstrap_support: float = 0.8
    output_formats: List[str] = None
    include_matrices: bool = True
    generate_plots: bool = True
    annotate_features: bool = True
    biological_context: bool = True
    interaction_interpretation: bool = True
    
    def __post_init__(self):
        if self.output_formats is None:
            self.output_formats = ["text", "json"]

class DataLoader:
    """Handles loading and preprocessing of various input formats"""
    
    @staticmethod
    def load_genomic_matrix(filepath: str) -> pd.DataFrame:
        """Load genomic data matrix from various formats"""
        filepath = Path(filepath)
        
        if filepath.suffix.lower() == '.csv':
            return pd.read_csv(filepath, index_col=0)
        elif filepath.suffix.lower() == '.tsv':
            return pd.read_csv(filepath, sep='\t', index_col=0)
        elif filepath.suffix.lower() in ['.fasta', '.fa']:
            return DataLoader._load_fasta_binary(filepath)
        elif filepath.suffix.lower() == '.vcf':
            return DataLoader._load_vcf_binary(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    @staticmethod
    def _load_fasta_binary(filepath: str) -> pd.DataFrame:
        """Convert FASTA sequences to binary matrix"""
        sequences = {}
        with open(filepath, 'r') as f:
            current_id = None
            current_seq = ""
            
            for line in f:
                if line.startswith('>'):
                    if current_id:
                        sequences[current_id] = current_seq
                    current_id = line[1:].strip()
                    current_seq = ""
                else:
                    current_seq += line.strip()
            
            if current_id:
                sequences[current_id] = current_seq
        
        # Convert to binary matrix
        max_length = max(len(seq) for seq in sequences.values())
        binary_matrix = []
        
        for sample_id, sequence in sequences.items():
            # Pad sequence if necessary
            padded_seq = sequence.ljust(max_length, '0')
            binary_row = [int(char) for char in padded_seq]
            binary_matrix.append(binary_row)
        
        columns = [f"pos_{i}" for i in range(max_length)]
        return pd.DataFrame(binary_matrix, index=list(sequences.keys()), columns=columns)
    
    @staticmethod
    def _load_vcf_binary(filepath: str) -> pd.DataFrame:
        """Convert VCF to binary matrix (simplified implementation)"""
        # This is a simplified VCF parser - in practice you'd use libraries like cyvcf2
        variants = []
        samples = []
        
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('##'):
                    continue
                elif line.startswith('#CHROM'):
                    # Extract sample names
                    samples = line.strip().split('\t')[9:]
                elif not line.startswith('#'):
                    # Process variant line
                    fields = line.strip().split('\t')
                    chrom, pos, ref, alt = fields[0], fields[1], fields[3], fields[4]
                    genotypes = fields[9:]
                    
                    # Convert genotypes to binary (simplified)
                    binary_genotypes = []
                    for gt in genotypes:
                        gt_field = gt.split(':')[0]
                        if gt_field in ['0/0', '0|0']:
                            binary_genotypes.append(0)
                        elif gt_field in ['1/1', '1|1', '0/1', '1/0', '0|1', '1|0']:
                            binary_genotypes.append(1)
                        else:
                            binary_genotypes.append(0)  # Missing data as reference
                    
                    variants.append({
                        'id': f"{chrom}_{pos}_{ref}_{alt}",
                        'genotypes': binary_genotypes
                    })
        
        # Create DataFrame
        matrix = pd.DataFrame(
            [variant['genotypes'] for variant in variants],
            columns=samples,
            index=[variant['id'] for variant in variants]
        ).T  # Transpose to have samples as rows
        
        return matrix
    
    @staticmethod
    def load_metadata(filepath: str) -> pd.DataFrame:
        """Load sample metadata"""
        return pd.read_csv(filepath, index_col=0)
    
    @staticmethod
    def load_known_markers(filepath: str) -> List[str]:
        """Load list of known markers"""
        with open(filepath, 'r') as f:
            return [line.strip() for line in f if line.strip()]

class StatisticalValidator:
    """Handles statistical validation and multiple testing correction"""
    
    def __init__(self, config: NetworkParserConfig):
        self.config = config
    
    def bootstrap_validation(self, data: pd.DataFrame, labels: pd.Series, 
                           features: List[str], n_iterations: int = None) -> Dict[str, float]:
        """Perform bootstrap validation for feature significance"""
        if n_iterations is None:
            n_iterations = self.config.bootstrap_iterations
        
        feature_scores = defaultdict(list)
        
        for i in range(n_iterations):
            # Bootstrap sample
            bootstrap_indices = np.random.choice(
                len(data), size=len(data), replace=True
            )
            
            bootstrap_data = data.iloc[bootstrap_indices]
            bootstrap_labels = labels.iloc[bootstrap_indices]
            
            # Calculate feature importance using Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=i)
            rf.fit(bootstrap_data[features], bootstrap_labels)
            
            for feature, importance in zip(features, rf.feature_importances_):
                feature_scores[feature].append(importance)
        
        # Calculate p-values and confidence intervals
        results = {}
        for feature, scores in feature_scores.items():
            scores = np.array(scores)
            # Calculate p-value (proportion of bootstraps where importance > 0)
            p_value = np.sum(scores <= np.mean(scores) * 0.1) / n_iterations
            results[feature] = {
                'bootstrap_score': np.mean(scores),
                'p_value': max(p_value, 1/n_iterations),  # Avoid p=0
                'confidence_interval': np.percentile(scores, [2.5, 97.5])
            }
        
        return results
    
    def chi_squared_test(self, data: pd.DataFrame, labels: pd.Series) -> Dict[str, float]:
        """Perform chi-squared tests for feature-label associations"""
        results = {}
        
        for feature in data.columns:
            contingency_table = pd.crosstab(data[feature], labels)
            chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
            results[feature] = p_value
        
        return results
    
    def multiple_testing_correction(self, p_values: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Apply multiple testing correction"""
        features = list(p_values.keys())
        p_vals = list(p_values.values())
        
        if self.config.correction_method == 'fdr_bh':
            rejected, p_corrected = fdrcorrection(p_vals, alpha=self.config.fdr_threshold)
        else:
            # Bonferroni correction
            p_corrected = np.array(p_vals) * len(p_vals)
            p_corrected = np.minimum(p_corrected, 1.0)
            rejected = p_corrected < self.config.fdr_threshold
        
        results = {}
        for feature, p_val, p_corr, is_significant in zip(features, p_vals, p_corrected, rejected):
            results[feature] = {
                'raw_p_value': p_val,
                'corrected_p_value': p_corr,
                'significant': is_significant
            }
        
        return results

class EpistaticInteractionDetector:
    """Detects and validates epistatic interactions between features"""
    
    def __init__(self, config: NetworkParserConfig):
        self.config = config
    
    def detect_interactions(self, data: pd.DataFrame, labels: pd.Series, 
                          significant_features: List[str]) -> Dict[str, Dict]:
        """Detect epistatic interactions among significant features"""
        interactions = {}
        
        # Generate all possible combinations up to max_interaction_order
        for order in range(2, self.config.max_interaction_order + 1):
            for feature_combo in combinations(significant_features, order):
                interaction_name = ' × '.join(feature_combo)
                
                # Create interaction feature (logical AND for binary data)
                interaction_values = data[list(feature_combo)].prod(axis=1)
                
                # Test interaction significance
                contingency_table = pd.crosstab(interaction_values, labels)
                if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                    chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
                    
                    # Calculate interaction strength (Cramér's V)
                    n = contingency_table.sum().sum()
                    cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                    
                    interactions[interaction_name] = {
                        'features': feature_combo,
                        'p_value': p_value,
                        'cramers_v': cramers_v,
                        'chi2_statistic': chi2,
                        'interaction_values': interaction_values
                    }
        
        return interactions
    
    def validate_interactions(self, interactions: Dict[str, Dict], 
                            data: pd.DataFrame, labels: pd.Series) -> Dict[str, Dict]:
        """Validate interactions using bootstrap sampling"""
        validated_interactions = {}
        
        for interaction_name, interaction_data in interactions.items():
            # Bootstrap validation
            bootstrap_scores = []
            
            for i in range(self.config.bootstrap_iterations):
                bootstrap_indices = np.random.choice(len(data), size=len(data), replace=True)
                bootstrap_labels = labels.iloc[bootstrap_indices]
                bootstrap_interaction = interaction_data['interaction_values'].iloc[bootstrap_indices]
                
                contingency_table = pd.crosstab(bootstrap_interaction, bootstrap_labels)
                if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                    chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
                    bootstrap_scores.append(p_value)
            
            if bootstrap_scores:
                bootstrap_p_value = np.mean(np.array(bootstrap_scores) < 0.05)
                
                interaction_data['bootstrap_support'] = bootstrap_p_value
                interaction_data['bootstrap_stable'] = bootstrap_p_value >= self.config.min_bootstrap_support
                
                validated_interactions[interaction_name] = interaction_data
        
        return validated_interactions

class DecisionTreeBuilder:
    """Constructs interpretable decision trees for sample classification"""
    
    def __init__(self, config: NetworkParserConfig):
        self.config = config
    
    def build_hierarchy(self, data: pd.DataFrame, labels: pd.Series, 
                       significant_features: List[str]) -> Tuple[DecisionTreeClassifier, Dict]:
        """Build hierarchical decision tree"""
        # Encode labels
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)
        
        # Build decision tree
        dt = DecisionTreeClassifier(
            max_depth=3,  # Reasonable depth for interpretability
            min_samples_split=self.config.min_group_size,
            min_samples_leaf=self.config.min_group_size // 2,
            random_state=42
        )
        
        dt.fit(data[significant_features], encoded_labels)
        
        # Generate tree structure information
        tree_info = {
            'accuracy': dt.score(data[significant_features], encoded_labels),
            'feature_importances': dict(zip(significant_features, dt.feature_importances_)),
            'tree_rules': export_text(dt, feature_names=significant_features),
            'label_encoder': le
        }
        
        return dt, tree_info
    
    def hierarchical_clustering(self, data: pd.DataFrame, 
                              significant_features: List[str]) -> Dict:
        """Perform hierarchical clustering of samples"""
        # Calculate distance matrix
        distances = pdist(data[significant_features], metric='hamming')
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(distances, method='ward')
        
        # Extract cluster information at different levels
        cluster_info = {}
        for n_clusters in [2, 4, 8]:
            if n_clusters <= len(data):
                clusters = cut_tree(linkage_matrix, n_clusters=n_clusters).flatten()
                cluster_info[f'level_{n_clusters}'] = {
                    'clusters': clusters,
                    'silhouette_score': self._calculate_silhouette_score(data[significant_features], clusters)
                }
        
        return {
            'linkage_matrix': linkage_matrix,
            'cluster_levels': cluster_info,
            'distances': distances
        }
    
    def _calculate_silhouette_score(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering quality"""
        from sklearn.metrics import silhouette_score
        if len(np.unique(labels)) > 1:
            return silhouette_score(data, labels)
        return 0.0

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
        
        logger.info(f"Text report saved to {output_path}")
    
    def generate_json_output(self, results: Dict, output_dir: str) -> None:
        """Generate JSON output for programmatic access"""
        output_path = Path(output_dir) / "networkparser_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = self._convert_for_json(results)
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        logger.info(f"JSON results saved to {output_path}")
    
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
        
        logger.info(f"GNN-ready matrices saved to {gnn_dir}")
    
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

class NetworkParser:
    """Main NetworkParser class that orchestrates the analysis pipeline"""
    
    def __init__(self, config: NetworkParserConfig = None):
        self.config = config or NetworkParserConfig()
        self.validator = StatisticalValidator(self.config)
        self.interaction_detector = EpistaticInteractionDetector(self.config)
        self.tree_builder = DecisionTreeBuilder(self.config)
        self.output_generator = OutputGenerator(self.config)
    
    def run_analysis(self, input_matrix: str, metadata: str = None, 
                    phenotype_file: str = None, hierarchy_column: str = None,
                    target_groups: str = None, known_markers: str = None,
                    output_dir: str = "networkparser_results") -> Dict:
        """Run complete NetworkParser analysis pipeline"""
        
        logger.info("Starting NetworkParser analysis...")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Load data
        logger.info("Loading input data...")
        data = DataLoader.load_genomic_matrix(input_matrix)
        
        # Load labels/groups
        if metadata and hierarchy_column:
            metadata_df = DataLoader.load_metadata(metadata)
            labels = metadata_df[hierarchy_column]
        elif phenotype_file:
            phenotype_df = pd.read_csv(phenotype_file, index_col=0)
            labels = phenotype_df.iloc[:, 0]  # First column as labels
        else:
            raise ValueError("Must provide either metadata with hierarchy_column or phenotype_file")
        
        # Filter for common samples
        common_samples = data.index.intersection(labels.index)
        data = data.loc[common_samples]
        labels = labels.loc[common_samples]
        
        logger.info(f"Analysis dataset: {len(data)} samples, {len(data.columns)} features")
        
        # Load known markers if provided
        known_marker_list = []
        if known_markers:
            known_marker_list = DataLoader.load_known_markers(known_markers)
            known_marker_list = [m for m in known_marker_list if m in data.columns]
        
        # Initialize results dictionary
        results = {
            'dataset_info': {
                'n_samples': len(data),
                'n_features': len(data.columns),
                'unique_labels': labels.unique().tolist(),
                'label_counts': labels.value_counts().to_dict()
            }
        }
        
        # Statistical validation
        logger.info("Performing statistical validation...")
        chi2_results = self.validator.chi_squared_test(data, labels)
        corrected_results = self.validator.multiple_testing_correction(chi2_results)
        results['statistical_validation'] = corrected_results
        
        # Get significant features
        significant_features = [f for f, data in corrected_results.items() if data['significant']]
        logger.info(f"Found {len(significant_features)} statistically significant features")
        
        if not significant_features:
            logger.warning("No significant features found. Consider adjusting thresholds.")
            return results
        
        # Bootstrap validation for significant features
        logger.info("Performing bootstrap validation...")
        bootstrap_results = self.validator.bootstrap_validation(data, labels, significant_features)
        results['bootstrap_validation'] = bootstrap_results
        
        # Epistatic interaction detection
        logger.info("Detecting epistatic interactions...")
        interactions = self.interaction_detector.detect_interactions(data, labels, significant_features)
        validated_interactions = self.interaction_detector.validate_interactions(interactions, data, labels)
        results['interactions'] = validated_interactions
        
        # Decision tree construction
        logger.info("Building decision trees...")
        decision_tree, tree_info = self.tree_builder.build_hierarchy(data, labels, significant_features)
        results['decision_tree'] = tree_info
        
        # Hierarchical clustering
        logger.info("Performing hierarchical clustering...")
        hierarchical_results = self.tree_builder.hierarchical_clustering(data, significant_features)
        results['hierarchical_analysis'] = hierarchical_results
        
        # Validate known markers if provided
        if known_marker_list:
            logger.info("Validating known markers...")
            known_validation = self._validate_known_markers(known_marker_list, corrected_results)
            results['known_markers_validated'] = known_validation
        
        # Cross-validation
        logger.info("Performing cross-validation...")
        cv_results = self._cross_validate_model(data, labels, significant_features)
        results['cross_validation'] = cv_results
        
        # Generate outputs
        logger.info("Generating outputs...")
        if 'text' in self.config.output_formats:
            self.output_generator.generate_text_report(results, output_dir)
        
        if 'json' in self.config.output_formats:
            self.output_generator.generate_json_output(results, output_dir)
        
        if self.config.include_matrices:
            self.output_generator.generate_gnn_matrices(data, results, output_dir)
        
        if self.config.generate_plots:
            self.output_generator.generate_plots(results, output_dir)
        
        # Save processed data matrices
        self._save_processed_matrices(data, labels, significant_features, validated_interactions, output_dir)
        
        logger.info("NetworkParser analysis completed successfully!")
        return results
    
    def _validate_known_markers(self, known_markers: List[str], 
                               validation_results: Dict) -> Dict:
        """Validate known markers against statistical results"""
        validated_count = 0
        known_marker_results = {}
        
        for marker in known_markers:
            if marker in validation_results:
                is_significant = validation_results[marker]['significant']
                p_value = validation_results[marker]['corrected_p_value']
                
                known_marker_results[marker] = {
                    'validated': is_significant,
                    'p_value': p_value
                }
                
                if is_significant:
                    validated_count += 1
        
        return {
            'n_total': len(known_markers),
            'n_validated': validated_count,
            'validation_rate': validated_count / len(known_markers) if known_markers else 0,
            'details': known_marker_results
        }
    
    def _cross_validate_model(self, data: pd.DataFrame, labels: pd.Series, 
                             significant_features: List[str]) -> Dict:
        """Perform cross-validation to assess model stability"""
        # Random Forest cross-validation
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        cv_scores = cross_val_score(
            rf, data[significant_features], labels, 
            cv=StratifiedKFold(n_splits=self.config.cross_validation_folds, shuffle=True, random_state=42),
            scoring='accuracy'
        )
        
        # Decision Tree cross-validation
        dt = DecisionTreeClassifier(max_depth=3, min_samples_split=self.config.min_group_size, random_state=42)
        dt_cv_scores = cross_val_score(
            dt, data[significant_features], labels,
            cv=StratifiedKFold(n_splits=self.config.cross_validation_folds, shuffle=True, random_state=42),
            scoring='accuracy'
        )
        
        return {
            'random_forest': {
                'mean_accuracy': np.mean(cv_scores),
                'std_accuracy': np.std(cv_scores),
                'scores': cv_scores.tolist()
            },
            'decision_tree': {
                'mean_accuracy': np.mean(dt_cv_scores),
                'std_accuracy': np.std(dt_cv_scores),
                'scores': dt_cv_scores.tolist()
            }
        }
    
    def _save_processed_matrices(self, data: pd.DataFrame, labels: pd.Series,
                                significant_features: List[str], interactions: Dict,
                                output_dir: str) -> None:
        """Save processed matrices for downstream analysis"""
        matrices_dir = Path(output_dir) / "processed_matrices"
        matrices_dir.mkdir(exist_ok=True)
        
        # Original data with significant features only
        filtered_data = data[significant_features]
        filtered_data.to_csv(matrices_dir / "significant_features_matrix.csv")
        
        # Labels
        labels.to_csv(matrices_dir / "sample_labels.csv", header=['label'])
        
        # Interaction features
        if interactions:
            interaction_matrix = pd.DataFrame(index=data.index)
            
            for interaction_name, interaction_data in interactions.items():
                if interaction_data.get('bootstrap_stable', False):
                    interaction_matrix[interaction_name] = interaction_data['interaction_values']
            
            if not interaction_matrix.empty:
                interaction_matrix.to_csv(matrices_dir / "interaction_features_matrix.csv")
                
                # Combined matrix (features + interactions)
                combined_matrix = pd.concat([filtered_data, interaction_matrix], axis=1)
                combined_matrix.to_csv(matrices_dir / "combined_features_matrix.csv")
        
        logger.info(f"Processed matrices saved to {matrices_dir}")

def create_config_from_args(args) -> NetworkParserConfig:
    """Create configuration from command line arguments"""
    config = NetworkParserConfig()
    
    # Update config with provided arguments
    if args.bootstrap_iterations:
        config.bootstrap_iterations = args.bootstrap_iterations
    if args.confidence_threshold:
        config.confidence_threshold = args.confidence_threshold
    if args.max_interaction_order:
        config.max_interaction_order = args.max_interaction_order
    if args.fdr_threshold:
        config.fdr_threshold = args.fdr_threshold
    if args.min_group_size:
        config.min_group_size = args.min_group_size
    if args.correction_method:
        config.correction_method = args.correction_method
    if args.max_workers:
        config.max_workers = args.max_workers
    if args.output_format:
        config.output_formats = args.output_format.split(',')
    
    # Set boolean flags
    config.memory_efficient = getattr(args, 'memory_efficient', False)
    config.include_matrices = getattr(args, 'include_matrices', True)
    config.generate_plots = getattr(args, 'generate_plots', True)
    config.include_interactions = getattr(args, 'include_interactions', True)
    
    return config

def load_config_file(config_path: str) -> NetworkParserConfig:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create config object
    config = NetworkParserConfig()
    
    # Update with file contents
    if 'analysis' in config_dict:
        analysis_config = config_dict['analysis']
        config.bootstrap_iterations = analysis_config.get('bootstrap_iterations', config.bootstrap_iterations)
        config.confidence_threshold = analysis_config.get('confidence_threshold', config.confidence_threshold)
        config.max_interaction_order = analysis_config.get('max_interaction_order', config.max_interaction_order)
        config.fdr_threshold = analysis_config.get('fdr_threshold', config.fdr_threshold)
    
    if 'processing' in config_dict:
        processing_config = config_dict['processing']
        config.max_workers = processing_config.get('max_workers', config.max_workers)
        config.memory_efficient = processing_config.get('memory_efficient', config.memory_efficient)
        config.chunk_size = processing_config.get('chunk_size', config.chunk_size)
    
    if 'output' in config_dict:
        output_config = config_dict['output']
        config.output_formats = output_config.get('formats', config.output_formats)
        config.include_matrices = output_config.get('include_matrices', config.include_matrices)
        config.generate_plots = output_config.get('generate_plots', config.generate_plots)
    
    if 'validation' in config_dict:
        validation_config = config_dict['validation']
        config.cross_validation_folds = validation_config.get('cross_validation_folds', config.cross_validation_folds)
        config.stability_threshold = validation_config.get('stability_threshold', config.stability_threshold)
        config.min_bootstrap_support = validation_config.get('min_bootstrap_support', config.min_bootstrap_support)
    
    return config

def main():
    """Main command line interface"""
    parser = argparse.ArgumentParser(
        description="NetworkParser: Interpretable Framework for Epistatic Cluster Segregation Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Hierarchical analysis
  python networkparser.py --input_matrix data/genomic_features.csv \\
                         --metadata data/sample_metadata.csv \\
                         --hierarchy_column "lineage" \\
                         --output_dir results

  # Phenotype-based analysis  
  python networkparser.py --input_matrix data/resistance_profiles.csv \\
                         --phenotype_file data/phenotypes.txt \\
                         --target_groups "resistant,sensitive" \\
                         --output_dir results

  # With known markers
  python networkparser.py --input_matrix data/snp_matrix.csv \\
                         --metadata data/metadata.csv \\
                         --hierarchy_column "cluster" \\
                         --known_markers data/resistance_snps.txt \\
                         --output_dir results
        """
    )
    
    # Required arguments
    parser.add_argument('--input_matrix', required=True,
                       help='Path to binary genomic matrix (CSV, FASTA, or VCF)')
    
    # Analysis mode arguments (mutually exclusive groups)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--metadata',
                           help='Path to sample metadata file (for hierarchical analysis)')
    mode_group.add_argument('--phenotype_file', 
                           help='Path to phenotype classifications (for phenotype-based analysis)')
    
    # Mode-specific arguments
    parser.add_argument('--hierarchy_column',
                       help='Column name for hierarchical grouping (required with --metadata)')
    parser.add_argument('--target_groups',
                       help='Comma-separated list of target groups (for phenotype analysis)')
    
    # Optional inputs
    parser.add_argument('--known_markers',
                       help='Path to known trait-associated features')
    parser.add_argument('--config_file',
                       help='Path to YAML configuration file')
    
    # Output options
    parser.add_argument('--output_dir', default='networkparser_results',
                       help='Output directory (default: networkparser_results)')
    parser.add_argument('--output_format', default='text,json',
                       help='Output formats: text,json,xml (default: text,json)')
    
    # Analysis parameters
    parser.add_argument('--bootstrap_iterations', type=int, default=1000,
                       help='Number of bootstrap iterations (default: 1000)')
    parser.add_argument('--confidence_threshold', type=float, default=0.95,
                       help='Statistical confidence level (default: 0.95)')
    parser.add_argument('--max_interaction_order', type=int, default=2,
                       help='Maximum epistatic interaction order (default: 2)')
    parser.add_argument('--fdr_threshold', type=float, default=0.05,
                       help='False discovery rate threshold (default: 0.05)')
    parser.add_argument('--min_group_size', type=int, default=5,
                       help='Minimum samples per group (default: 5)')
    parser.add_argument('--correction_method', default='fdr_bh',
                       choices=['fdr_bh', 'bonferroni'],
                       help='Multiple testing correction method (default: fdr_bh)')
    parser.add_argument('--max_workers', type=int, default=4,
                       help='Number of parallel processing threads (default: 4)')
    
    # Boolean flags
    parser.add_argument('--memory_efficient', action='store_true',
                       help='Enable memory-efficient processing for large datasets')
    parser.add_argument('--no_matrices', action='store_false', dest='include_matrices',
                       help='Skip generating processed matrices')
    parser.add_argument('--no_plots', action='store_false', dest='generate_plots',
                       help='Skip generating plots')
    parser.add_argument('--json_output', action='store_true',
                       help='Generate JSON output (sets format to include json)')
    parser.add_argument('--include_interactions', action='store_true', default=True,
                       help='Include epistatic interaction analysis')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate argument combinations
    if args.metadata and not args.hierarchy_column:
        parser.error("--hierarchy_column is required when using --metadata")
    
    # Load configuration
    if args.config_file:
        config = load_config_file(args.config_file)
    else:
        config = create_config_from_args(args)
    
    # Override output format if json_output flag is set
    if args.json_output:
        config.output_formats = list(set(config.output_formats + ['json']))
    
    try:
        # Initialize and run NetworkParser
        networkparser = NetworkParser(config)
        
        results = networkparser.run_analysis(
            input_matrix=args.input_matrix,
            metadata=args.metadata,
            phenotype_file=args.phenotype_file,
            hierarchy_column=args.hierarchy_column,
            target_groups=args.target_groups,
            known_markers=args.known_markers,
            output_dir=args.output_dir
        )
        
        # Print summary
        print("\n" + "="*80)
        print("NETWORKPARSER ANALYSIS COMPLETED")
        print("="*80)
        print(f"Samples analyzed: {results['dataset_info']['n_samples']}")
        print(f"Features analyzed: {results['dataset_info']['n_features']}")
        
        significant_features = sum(1 for data in results['statistical_validation'].values() 
                                 if data['significant'])
        print(f"Significant features: {significant_features}")
        
        if 'interactions' in results:
            significant_interactions = sum(1 for data in results['interactions'].values() 
                                         if data.get('bootstrap_stable', False))
            print(f"Validated interactions: {significant_interactions}")
        
        if 'cross_validation' in results:
            cv_acc = results['cross_validation']['random_forest']['mean_accuracy']
            print(f"Cross-validation accuracy: {cv_acc:.3f}")
        
        print(f"\nResults saved to: {args.output_dir}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# Example configuration file (config.yml) content:
"""
# NetworkParser Configuration File

analysis:
  bootstrap_iterations: 2000
  confidence_threshold: 0.99
  max_interaction_order: 3
  fdr_threshold: 0.01
  min_group_size: 5
  correction_method: fdr_bh

processing:
  max_workers: 16
  memory_efficient: true
  chunk_size: 1000

output:
  formats: ["text", "json", "xml"]
  include_matrices: true
  generate_plots: true

validation:
  cross_validation_folds: 5
  stability_threshold: 0.9
  min_bootstrap_support: 0.8

explainable_ai:
  annotate_features: true
  biological_context: true
  interaction_interpretation: true
"""

# Example usage as Python library:
"""
from networkparser import NetworkParser, NetworkParserConfig

# Create custom configuration
config = NetworkParserConfig(
    bootstrap_iterations=2000,
    max_interaction_order=3,
    fdr_threshold=0.01
)

# Initialize NetworkParser
np = NetworkParser(config)

# Run analysis
results = np.run_analysis(
    input_matrix="data/genomic_matrix.csv",
    metadata="data/sample_metadata.csv", 
    hierarchy_column="lineage",
    output_dir="my_results"
)

# Access results
print(f"Found {len([f for f, d in results['statistical_validation'].items() if d['significant']])} significant features")
"""
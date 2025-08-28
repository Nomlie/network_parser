"""
NetworkParser - Integrated Genomic Feature Discovery Pipeline

A comprehensive pipeline for discovering discriminative genomic features using:
1. Decision trees as feature discovery engines
2. Statistical validation (bootstrap, chi-squared, multiple testing)
3. Epistatic interaction detection
4. Multi-format data loading (CSV, TSV, FASTA, VCF)

Main pipeline class that orchestrates all components.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import json
import pickle
from datetime import datetime
import warnings

from .config import NetworkParserConfig
from .data_loader import DataLoader
from .enhanced_decision_tree import EnhancedDecisionTreeBuilder
from .statistical_validator import StatisticalValidator


class NetworkParser:
    """
    Main NetworkParser pipeline for genomic feature discovery.
    
    Integrates decision tree feature discovery with statistical validation
    to identify truly discriminative genomic markers while controlling
    for multiple testing and validating epistatic interactions.
    """
    
    def __init__(self, config: NetworkParserConfig):
        self.config = config
        self.data = None
        self.labels = None
        self.metadata = None
        self.results = None
        
        # Initialize components
        self.data_loader = DataLoader()
        self.tree_builder = EnhancedDecisionTreeBuilder(config)
        self.validator = StatisticalValidator(config)
        
        # Pipeline state
        self.pipeline_complete = False
        self.processing_time = None
        
    def load_data(self, 
                  genomic_data_path: str,
                  metadata_path: str,
                  label_column: str,
                  known_markers_path: Optional[str] = None) -> 'NetworkParser':
        """
        Load genomic data and metadata.
        
        Args:
            genomic_data_path: Path to genomic matrix (CSV, TSV, FASTA, VCF)
            metadata_path: Path to sample metadata with labels
            label_column: Column name containing class labels
            known_markers_path: Optional path to known marker list for comparison
            
        Returns:
            Self for method chaining
        """
        print("Loading genomic data and metadata...")
        
        # Load genomic matrix
        try:
            self.data = self.data_loader.load_genomic_matrix(genomic_data_path)
            print(f"Loaded genomic matrix: {self.data.shape[0]} samples × {self.data.shape[1]} features")
        except Exception as e:
            raise ValueError(f"Failed to load genomic data from {genomic_data_path}: {e}")
        
        # Load metadata
        try:
            self.metadata = self.data_loader.load_metadata(metadata_path)
            print(f"Loaded metadata for {len(self.metadata)} samples")
        except Exception as e:
            raise ValueError(f"Failed to load metadata from {metadata_path}: {e}")
        
        # Extract labels
        if label_column not in self.metadata.columns:
            raise ValueError(f"Label column '{label_column}' not found in metadata")
        
        # Align data and metadata
        common_samples = list(set(self.data.index) & set(self.metadata.index))
        if len(common_samples) == 0:
            raise ValueError("No common samples found between genomic data and metadata")
        
        self.data = self.data.loc[common_samples]
        self.metadata = self.metadata.loc[common_samples]
        self.labels = self.metadata[label_column]
        
        print(f"Aligned data: {len(common_samples)} samples")
        print(f"Label distribution: {self.labels.value_counts().to_dict()}")
        
        # Load known markers if provided
        self.known_markers = None
        if known_markers_path:
            try:
                self.known_markers = self.data_loader.load_known_markers(known_markers_path)
                print(f"Loaded {len(self.known_markers)} known markers for comparison")
            except Exception as e:
                warnings.warn(f"Could not load known markers: {e}")
        
        return self
    
    def run_feature_discovery(self, 
                            validate_statistics: bool = True,
                            validate_interactions: bool = False) -> 'NetworkParser':
        """
        Run the complete feature discovery pipeline.
        
        Args:
            validate_statistics: Whether to run statistical validation
            validate_interactions: Whether to validate epistatic interactions with permutation tests
            
        Returns:
            Self for method chaining
        """
        if self.data is None or self.labels is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        start_time = datetime.now()
        print("\n" + "="*70)
        print("NETWORKPARSER FEATURE DISCOVERY PIPELINE")
        print("="*70)
        
        # Phase 1: Decision tree feature discovery
        print("\nPHASE 1: Decision Tree Feature Discovery")
        print("-" * 40)
        
        discovery_results = self.tree_builder.discover_features(
            self.data, 
            self.labels,
            validate_statistics=validate_statistics
        )
        
        # Phase 2: Additional statistical validation if requested
        additional_validation = {}
        if validate_interactions and discovery_results.get('epistatic_interactions'):
            print("\nPHASE 2: Interaction Validation")
            print("-" * 40)
            
            # Extract interaction pairs for permutation testing
            interaction_pairs = [
                (interaction['parent_feature'], interaction['child_feature'])
                for interaction in discovery_results['epistatic_interactions'][:10]  # Limit to top 10
            ]
            
            if interaction_pairs:
                interaction_validation = self.validator.permutation_test_interactions(
                    self.data, self.labels, interaction_pairs
                )
                additional_validation['interaction_permutation_tests'] = interaction_validation
        
        # Phase 3: Comprehensive feature set validation
        print("\nPHASE 3: Feature Set Validation")
        print("-" * 40)
        
        all_discovered_features = (
            discovery_results['root_features'] + 
            discovery_results['branch_features']
        )
        
        if all_discovered_features:
            feature_set_validation = self.validator.validate_feature_set(
                self.data, self.labels, all_discovered_features
            )
            additional_validation['feature_set_validation'] = feature_set_validation
        
        # Phase 4: Known marker comparison if available
        known_marker_comparison = {}
        if self.known_markers:
            print("\nPHASE 4: Known Marker Comparison")
            print("-" * 40)
            known_marker_comparison = self._compare_with_known_markers(
                all_discovered_features, self.known_markers
            )
        
        # Compile comprehensive results
        self.results = {
            'discovery_results': discovery_results,
            'additional_validation': additional_validation,
            'known_marker_comparison': known_marker_comparison,
            'pipeline_metadata': {
                'processing_time': datetime.now() - start_time,
                'data_shape': self.data.shape,
                'n_classes': len(self.labels.unique()),
                'class_distribution': self.labels.value_counts().to_dict(),
                'config': self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config)
            }
        }
        
        self.processing_time = self.results['pipeline_metadata']['processing_time']
        self.pipeline_complete = True
        
        # Print final summary
        self._print_pipeline_summary()
        
        return self
    
    def get_discovered_features(self, 
                              feature_type: str = 'all',
                              significance_threshold: float = 0.05,
                              confidence_threshold: float = 0.5) -> Dict[str, List[str]]:
        """
        Extract discovered features with filtering options.
        
        Args:
            feature_type: 'root', 'branch', or 'all'
            significance_threshold: P-value threshold for statistical significance
            confidence_threshold: Confidence score threshold
            
        Returns:
            Dictionary with filtered feature lists
        """
        if not self.pipeline_complete:
            raise ValueError("Pipeline not complete. Run run_feature_discovery() first.")
        
        results = self.results['discovery_results']
        
        # Get base feature lists
        root_features = results.get('root_features', [])
        branch_features = results.get('branch_features', [])
        
        # Apply confidence filtering
        confidence_scores = results.get('feature_confidence', {})
        
        filtered_root = [
            f for f in root_features 
            if confidence_scores.get(f, {}).get('confidence', 0) >= confidence_threshold
        ]
        
        filtered_branch = [
            f for f in branch_features 
            if confidence_scores.get(f, {}).get('confidence', 0) >= confidence_threshold
        ]
        
        # Apply statistical significance filtering if available
        if 'statistical_validation' in results:
            corrected_results = results['statistical_validation'].get('multiple_testing', {})
            
            statistically_significant = [
                f for f, stats in corrected_results.items()
                if isinstance(stats, dict) and stats.get('significant', False)
            ]
            
            filtered_root = [f for f in filtered_root if f in statistically_significant]
            filtered_branch = [f for f in filtered_branch if f in statistically_significant]
        
        # Return based on requested type
        if feature_type == 'root':
            return {'root_features': filtered_root}
        elif feature_type == 'branch':
            return {'branch_features': filtered_branch}
        else:  # 'all'
            return {
                'root_features': filtered_root,
                'branch_features': filtered_branch,
                'all_features': filtered_root + filtered_branch
            }
    
    def get_epistatic_interactions(self, 
                                 significance_threshold: float = 0.05,
                                 strength_threshold: float = 0.1) -> List[Dict]:
        """
        Get validated epistatic interactions.
        
        Args:
            significance_threshold: P-value threshold for interaction validation
            strength_threshold: Minimum interaction strength
            
        Returns:
            List of significant epistatic interactions
        """
        if not self.pipeline_complete:
            raise ValueError("Pipeline not complete. Run run_feature_discovery() first.")
        
        interactions = self.results['discovery_results'].get('epistatic_interactions', [])
        
        # Filter by strength threshold
        filtered_interactions = [
            interaction for interaction in interactions
            if interaction.get('interaction_strength', 0) >= strength_threshold
        ]
        
        # Add permutation test results if available
        permutation_results = self.results['additional_validation'].get(
            'interaction_permutation_tests', {}
        )
        
        for interaction in filtered_interactions:
            parent = interaction['parent_feature']
            child = interaction['child_feature']
            key = f"{parent}_x_{child}"
            
            if key in permutation_results:
                interaction['permutation_p_value'] = permutation_results[key]['p_value']
                interaction['permutation_significant'] = permutation_results[key]['significant']
        
        return filtered_interactions
    
    def export_results(self, 
                      output_dir: str,
                      export_format: str = 'json') -> str:
        """
        Export pipeline results to file.
        
        Args:
            output_dir: Directory to save results
            export_format: 'json' or 'pickle'
            
        Returns:
            Path to exported file
        """
        if not self.pipeline_complete:
            raise ValueError("Pipeline not complete. Run run_feature_discovery() first.")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if export_format == 'json':
            # Convert non-serializable objects for JSON
            exportable_results = self._prepare_for_json_export(self.results)
            output_file = output_dir / f"networkparser_results_{timestamp}.json"
            
            with open(output_file, 'w') as f:
                json.dump(exportable_results, f, indent=2, default=str)
                
        elif export_format == 'pickle':
            output_file = output_dir / f"networkparser_results_{timestamp}.pkl"
            
            with open(output_file, 'wb') as f:
                pickle.dump(self.results, f)
        
        else:
            raise ValueError("export_format must be 'json' or 'pickle'")
        
        print(f"Results exported to: {output_file}")
        return str(output_file)
    
    def _compare_with_known_markers(self, 
                                  discovered_features: List[str],
                                  known_markers: List[str]) -> Dict:
        """Compare discovered features with known markers."""
        discovered_set = set(discovered_features)
        known_set = set(known_markers)
        
        overlap = discovered_set & known_set
        discovered_only = discovered_set - known_set
        known_only = known_set - discovered_set
        
        return {
            'n_discovered': len(discovered_set),
            'n_known': len(known_set),
            'n_overlap': len(overlap),
            'overlap_features': list(overlap),
            'discovered_only': list(discovered_only),
            'known_only': list(known_only),
            'jaccard_index': len(overlap) / len(discovered_set | known_set) if (discovered_set | known_set) else 0,
            'precision': len(overlap) / len(discovered_set) if discovered_set else 0,
            'recall': len(overlap) / len(known_set) if known_set else 0
        }
    
    def _prepare_for_json_export(self, data):
        """Recursively convert data to JSON-serializable format."""
        if isinstance(data, dict):
            return {k: self._prepare_for_json_export(v) for k, v in data.items() 
                   if not k.startswith('decision_tree')}  # Skip sklearn objects
        elif isinstance(data, (list, tuple)):
            return [self._prepare_for_json_export(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.int64, np.int32)):
            return int(data)
        elif isinstance(data, (np.float64, np.float32)):
            return float(data)
        elif hasattr(data, '__dict__'):  # Complex objects
            return str(data)
        else:
            return data
    
    def _print_pipeline_summary(self):
        """Print comprehensive pipeline summary."""
        if not self.results:
            return
            
        results = self.results['discovery_results']
        metadata = self.results['pipeline_metadata']
        
        print("\n" + "="*70)
        print("NETWORKPARSER PIPELINE SUMMARY")
        print("="*70)
        
        print(f"Processing Time: {metadata['processing_time']}")
        print(f"Data Shape: {metadata['data_shape'][0]} samples × {metadata['data_shape'][1]} features")
        print(f"Classes: {metadata['n_classes']} ({list(metadata['class_distribution'].keys())})")
        
        print(f"\nFEATURE DISCOVERY RESULTS:")
        print(f"  Root Features (Global): {len(results.get('root_features', []))}")
        print(f"  Branch Features (Conditional): {len(results.get('branch_features', []))}")
        print(f"  Total Discovered: {len(results.get('root_features', [])) + len(results.get('branch_features', []))}")
        
        if results.get('epistatic_interactions'):
            print(f"  Epistatic Interactions: {len(results['epistatic_interactions'])}")
        
        # Statistical validation summary
        if 'statistical_validation' in results:
            stats = results['statistical_validation']
            if 'multiple_testing' in stats and '_summary' in stats['multiple_testing']:
                summary = stats['multiple_testing']['_summary']
                print(f"\nSTATISTICAL VALIDATION:")
                print(f"  Features Tested: {summary['n_features_tested']}")
                print(f"  Significant (Raw): {summary['n_significant_original']}")
                print(f"  Significant (Corrected): {summary['n_significant_corrected']}")
                print(f"  Correction Method: {summary['correction_method']}")
        
        # Known marker comparison
        if self.results.get('known_marker_comparison'):
            comp = self.results['known_marker_comparison']
            print(f"\nKNOWN MARKER COMPARISON:")
            print(f"  Overlap: {comp['n_overlap']}/{comp['n_known']} known markers")
            print(f"  Precision: {comp['precision']:.3f}")
            print(f"  Recall: {comp['recall']:.3f}")
            print(f"  Jaccard Index: {comp['jaccard_index']:.3f}")
        
        print("="*70)


# Convenience function for quick analysis
def run_networkparser_analysis(genomic_data_path: str,
                              metadata_path: str,
                              label_column: str,
                              output_dir: Optional[str] = None,
                              config: Optional[NetworkParserConfig] = None,
                              **kwargs) -> NetworkParser:
    """
    Convenience function to run complete NetworkParser analysis.
    
    Args:
        genomic_data_path: Path to genomic data file
        metadata_path: Path to metadata file
        label_column: Column name with class labels
        output_dir: Optional directory to save results
        config: Optional custom configuration
        **kwargs: Additional arguments for run_feature_discovery()
        
    Returns:
        Completed NetworkParser instance
    """
    if config is None:
        config = NetworkParserConfig()
    
    # Run analysis
    parser = (NetworkParser(config)
              .load_data(genomic_data_path, metadata_path, label_column)
              .run_feature_discovery(**kwargs))
    
    # Export results if output directory provided
    if output_dir:
        parser.export_results(output_dir)
    
    return parser
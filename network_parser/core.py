# networkparser/core.py
"""
Core NetworkParser orchestration module.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from typing import Dict, List, Optional

from .config import NetworkParserConfig
from .data_loader import DataLoader
from .statistical_validator import StatisticalValidator
from .interaction_detector import EpistaticInteractionDetector
from .decision_tree_builder import DecisionTreeBuilder
from .output_generator import OutputGenerator

logger = logging.getLogger(__name__)

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
                'mean_accuracy': float(np.mean(cv_scores)),
                'std_accuracy': float(np.std(cv_scores)),
                'scores': cv_scores.tolist()
            },
            'decision_tree': {
                'mean_accuracy': float(np.mean(dt_cv_scores)),
                'std_accuracy': float(np.std(dt_cv_scores)),
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
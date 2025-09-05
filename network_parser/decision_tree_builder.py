"""
Enhanced Decision Tree and Statistical Validation module.
Implements: 
1. Decision tree as a feature discovery engine with improved feature role extraction
2. Statistical validation (bootstrap, chi-squared, multiple testing correction)
3. Epistatic interaction detection through conditional splits
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mutual_info_score
from collections import defaultdict, Counter
from typing import Dict, List, Optional

from .config import NetworkParserConfig
from .statistical_validation import StatisticalValidator


class EnhancedDecisionTreeBuilder:
    """
    Enhanced decision tree feature discovery engine that:
    1. Identifies root vs branch features with confidence scores
    2. Detects epistatic interactions through conditional analysis
    3. Validates discoveries through statistical testing
    4. Filters noise from true discriminative features
    """

    def __init__(self, config: NetworkParserConfig):
        self.config = config
        self.feature_roles = {}
        self.epistatic_pairs = []
        self.validation_scores = {}

    def discover_features(self, 
                          data: pd.DataFrame, 
                          labels: pd.Series,
                          validate_statistics: bool = True) -> Dict:
        """
        Feature discovery using decision tree as discovery engine.
        
        Returns comprehensive analysis of:
        - Root features (globally discriminative)
        - Branch features (conditionally discriminative) 
        - Epistatic interactions
        - Statistical validation scores
        """
        print(f"🔍 Starting feature discovery on {len(data)} samples with {len(data.columns)} features...")
        
        # Encode labels for tree building
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)
        n_classes = len(le.classes_)
        
        print(f"📊 Found {n_classes} distinct labels: {le.classes_.tolist()}")

        # Build enhanced decision tree with optimal parameters
        dt = self._build_optimal_tree(data, encoded_labels)
        
        # Extract detailed feature roles and interactions
        feature_analysis = self._analyze_tree_structure(dt, data.columns, data, encoded_labels)
        
        # Calculate feature confidence scores
        confidence_scores = self._calculate_feature_confidence(
            dt, data, encoded_labels, feature_analysis['root_features'], feature_analysis['branch_features']
        )
        
        # Detect epistatic interactions
        epistatic_interactions = self._detect_epistatic_interactions(dt, data.columns, data, encoded_labels)
        
        # Validate statistically if requested
        statistical_validation = {}
        if validate_statistics:
            validator = StatisticalValidator(self.config)
            all_features = feature_analysis['root_features'] + feature_analysis['branch_features']
            
            bootstrap_results = validator.bootstrap_validation(data, labels, all_features)
            chi2_results = validator.chi_squared_test(data[all_features], labels)
            corrected_results = validator.multiple_testing_correction(chi2_results)
            
            statistical_validation = {
                'bootstrap': bootstrap_results,
                'chi_squared': chi2_results,
                'multiple_testing': corrected_results
            }
        
        # Compile comprehensive results
        results = {
            'decision_tree': dt,
            'tree_accuracy': dt.score(data, encoded_labels),
            'tree_rules': export_text(dt, feature_names=list(data.columns)),
            'label_classes': le.classes_.tolist(),
            
            # Enhanced feature analysis
            'root_features': feature_analysis['root_features'],
            'branch_features': feature_analysis['branch_features'], 
            'feature_depths': feature_analysis['feature_depths'],
            'split_purities': feature_analysis['split_purities'],
            
            # Confidence and validation
            'feature_confidence': confidence_scores,
            'epistatic_interactions': epistatic_interactions,
            'statistical_validation': statistical_validation,
            
            # Raw importance for comparison
            'sklearn_importances': dict(zip(data.columns, dt.feature_importances_))
        }
        
        self._print_discovery_summary(results)
        return results

    def _build_optimal_tree(self, data: pd.DataFrame, encoded_labels: np.ndarray) -> DecisionTreeClassifier:
        """Build decision tree with parameters optimized for feature discovery."""
        dt = DecisionTreeClassifier(
            criterion='gini',  # Good balance of interpretability and performance
            max_depth=self.config.max_depth or min(10, int(np.log2(len(data)) + 3)),
            min_samples_split=max(2, self.config.min_group_size),
            min_samples_leaf=max(1, self.config.min_group_size // 3),
            min_impurity_decrease=0.001,  # Prevent overfitting to noise
            random_state=42
        )
        dt.fit(data, encoded_labels)
        return dt

    def _analyze_tree_structure(self, dt: DecisionTreeClassifier, 
                               feature_names: List[str],
                               data: pd.DataFrame,
                               encoded_labels: np.ndarray) -> Dict:
        """
        Enhanced analysis of tree structure to extract feature roles.
        
        Returns:
        - Root features: Global discriminators (depth 0-1)
        - Branch features: Conditional discriminators (depth 2+)
        - Feature depths: Depth at which each feature first appears
        - Split purities: Information gain at each split
        """
        tree = dt.tree_
        root_features = set()
        branch_features = set()
        feature_depths = defaultdict(list)
        split_purities = {}

        def analyze_node(node_id: int, depth: int, parent_feature: Optional[str] = None):
            if tree.feature[node_id] == -2:  # Leaf node
                return
                
            feature_idx = tree.feature[node_id]
            feature_name = feature_names[feature_idx]
            
            # Calculate information gain at this split
            parent_impurity = tree.impurity[node_id]
            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]
            
            if left_child != -1 and right_child != -1:
                left_samples = tree.n_node_samples[left_child]
                right_samples = tree.n_node_samples[right_child]
                total_samples = tree.n_node_samples[node_id]
                
                left_impurity = tree.impurity[left_child]
                right_impurity = tree.impurity[right_child]
                
                weighted_child_impurity = (
                    (left_samples / total_samples) * left_impurity +
                    (right_samples / total_samples) * right_impurity
                )
                
                information_gain = parent_impurity - weighted_child_impurity
                split_purities[f"{feature_name}_depth_{depth}"] = {
                    'information_gain': information_gain,
                    'samples': total_samples,
                    'parent_feature': parent_feature
                }
            
            # Categorize feature based on depth and context
            feature_depths[feature_name].append(depth)
            
            if depth <= 1:  # Root level features (global discriminators)
                root_features.add(feature_name)
            else:  # Branch features (conditional discriminators)
                branch_features.add(feature_name)
            
            # Recursively analyze children
            if left_child != -1:
                analyze_node(left_child, depth + 1, feature_name)
            if right_child != -1:
                analyze_node(right_child, depth + 1, feature_name)

        # Start analysis from root
        analyze_node(0, 0)
        
        # Clean up overlapping classifications (prioritize root if appears at both levels)
        branch_features = branch_features - root_features
        
        return {
            'root_features': list(root_features),
            'branch_features': list(branch_features),
            'feature_depths': dict(feature_depths),
            'split_purities': split_purities
        }

    def _calculate_feature_confidence(self, dt: DecisionTreeClassifier,
                                    data: pd.DataFrame,
                                    encoded_labels: np.ndarray,
                                    root_features: List[str],
                                    branch_features: List[str]) -> Dict:
        """
        Calculate confidence scores for discovered features.
        Higher scores = more reliable discriminators, less likely to be noise.
        """
        confidence_scores = {}
        
        # For root features: measure global discrimination power
        for feature in root_features:
            # Mutual information with labels
            mi_score = mutual_info_score(encoded_labels, data[feature].values)
            
            # Stability across bootstrap samples
            stability_scores = []
            for _ in range(10):  # Bootstrap validation
                boot_indices = np.random.choice(len(data), size=len(data), replace=True)
                boot_data = data.iloc[boot_indices]
                boot_labels = encoded_labels[boot_indices]
                
                boot_dt = DecisionTreeClassifier(
                    max_depth=3, min_samples_split=5, random_state=np.random.randint(1000)
                )
                boot_dt.fit(boot_data, boot_labels)
                
                # Check if feature appears in top 3 most important
                importances = dict(zip(boot_data.columns, boot_dt.feature_importances_))
                top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:3]
                stability_scores.append(1.0 if feature in [f[0] for f in top_features] else 0.0)
            
            confidence_scores[feature] = {
                'type': 'root',
                'mutual_info': mi_score,
                'stability': np.mean(stability_scores),
                'confidence': (mi_score + np.mean(stability_scores)) / 2
            }
        
        # For branch features: measure conditional discrimination power
        for feature in branch_features:
            mi_score = mutual_info_score(encoded_labels, data[feature].values)
            
            # Conditional entropy reduction (epistatic potential)
            conditional_score = self._calculate_conditional_discrimination(
                data, encoded_labels, feature, root_features
            )
            
            confidence_scores[feature] = {
                'type': 'branch',
                'mutual_info': mi_score,
                'conditional_power': conditional_score,
                'confidence': (mi_score + conditional_score) / 2
            }
        
        return confidence_scores

    def _calculate_conditional_discrimination(self, data: pd.DataFrame,
                                           encoded_labels: np.ndarray,
                                           target_feature: str,
                                           root_features: List[str]) -> float:
        """Calculate how much target_feature improves discrimination when conditioned on root features."""
        if not root_features:
            return 0.0
            
        scores = []
        for root_feature in root_features[:3]:  # Limit to top 3 to avoid overfitting
            # Split data based on root feature
            root_values = data[root_feature].unique()
            for value in root_values:
                mask = data[root_feature] == value
                if mask.sum() < 5:  # Skip small groups
                    continue
                    
                subset_data = data[mask]
                subset_labels = encoded_labels[mask]
                
                if len(np.unique(subset_labels)) > 1:  # Still heterogeneous after root split
                    # Measure additional discrimination from target feature
                    mi_gain = mutual_info_score(subset_labels, subset_data[target_feature].values)
                    scores.append(mi_gain)
        
        return np.mean(scores) if scores else 0.0

    def _detect_epistatic_interactions(self, dt: DecisionTreeClassifier,
                                     feature_names: List[str],
                                     data: pd.DataFrame,
                                     encoded_labels: np.ndarray) -> List[Dict]:
        """
        Detect epistatic (conditional) interactions from tree structure.
        Look for feature pairs where one conditions the effect of another.
        """
        tree = dt.tree_
        interactions = []
        
        def find_interactions(node_id: int, path_features: List[str]):
            if tree.feature[node_id] == -2:  # Leaf node
                return
                
            current_feature = feature_names[tree.feature[node_id]]
            new_path = path_features + [current_feature]
            
            # If path length >= 2, we have potential epistatic interaction
            if len(new_path) >= 2:
                parent_feature = path_features[-1] if path_features else None
                if parent_feature and parent_feature != current_feature:
                    # Calculate interaction strength
                    interaction_strength = self._measure_epistatic_strength(
                        data, encoded_labels, parent_feature, current_feature
                    )
                    
                    if interaction_strength > 0.1:  # Threshold for significant interaction
                        interactions.append({
                            'parent_feature': parent_feature,
                            'child_feature': current_feature,
                            'depth': len(new_path) - 1,
                            'interaction_strength': interaction_strength,
                            'samples_at_split': tree.n_node_samples[node_id]
                        })
            
            # Recurse to children
            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]
            
            if left_child != -1:
                find_interactions(left_child, new_path)
            if right_child != -1:
                find_interactions(right_child, new_path)
        
        find_interactions(0, [])
        
        # Sort by interaction strength
        interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)
        return interactions

    def _measure_epistatic_strength(self, data: pd.DataFrame,
                                  encoded_labels: np.ndarray,
                                  feature1: str, feature2: str) -> float:
        """
        Measure epistatic interaction strength between two features.
        Returns how much feature2 depends on feature1 for discrimination.
        """
        try:
            # Overall MI of feature2 with labels
            overall_mi = mutual_info_score(encoded_labels, data[feature2].values)
            
            # Conditional MI: average MI of feature2 with labels, given feature1 values
            conditional_mis = []
            for value1 in data[feature1].unique():
                mask = data[feature1] == value1
                if mask.sum() < 3:  # Skip very small groups
                    continue
                subset_labels = encoded_labels[mask]
                subset_feature2 = data[feature2].values[mask]
                
                if len(np.unique(subset_labels)) > 1:
                    cond_mi = mutual_info_score(subset_labels, subset_feature2)
                    conditional_mis.append(cond_mi)
            
            if conditional_mis:
                avg_conditional_mi = np.mean(conditional_mis)
                # Epistatic strength = how much conditional context matters
                return abs(avg_conditional_mi - overall_mi) / (overall_mi + 1e-6)
            
        except Exception as e:
            print(f"Warning: Could not calculate epistatic strength for {feature1}-{feature2}: {e}")
        
        return 0.0

    def _print_discovery_summary(self, results: Dict):
        """Print a comprehensive summary of feature discovery results."""
        print("\n" + "="*60)
        print("🎯 FEATURE DISCOVERY SUMMARY")
        print("="*60)
        
        print(f"📈 Tree Accuracy: {results['tree_accuracy']:.3f}")
        print(f"🏷️  Label Classes: {results['label_classes']}")
        
        print(f"\n🌳 ROOT FEATURES (Global Discriminators): {len(results['root_features'])}")
        for i, feature in enumerate(results['root_features'][:5]):  # Show top 5
            confidence = results['feature_confidence'].get(feature, {}).get('confidence', 0)
            print(f"  {i+1}. {feature} (confidence: {confidence:.3f})")
        
        print(f"\n🌿 BRANCH FEATURES (Conditional Discriminators): {len(results['branch_features'])}")
        for i, feature in enumerate(results['branch_features'][:5]):  # Show top 5
            confidence = results['feature_confidence'].get(feature, {}).get('confidence', 0)
            print(f"  {i+1}. {feature} (confidence: {confidence:.3f})")
        
        if results['epistatic_interactions']:
            print(f"\n🔗 EPISTATIC INTERACTIONS: {len(results['epistatic_interactions'])}")
            for interaction in results['epistatic_interactions'][:3]:  # Show top 3
                print(f"  {interaction['parent_feature']} → {interaction['child_feature']} "
                      f"(strength: {interaction['interaction_strength']:.3f})")
        
        if 'statistical_validation' in results and results['statistical_validation']:
            print(f"\n✅ STATISTICAL VALIDATION:")
            corrected = results['statistical_validation']['multiple_testing']
            sig_features = [f for f, v in corrected.items() if v['significant']]
            print(f"  Significant features after correction: {len(sig_features)}")
            for f in sig_features[:5]:
                print(f"   - {f}: corrected p={corrected[f]['corrected_p_value']:.3e}")
        
        print("="*60 + "\n")

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
from collections import defaultdict
from typing import Dict, List, Optional
from pathlib import Path
import json

try:
    from network_parser.config import NetworkParserConfig
    from network_parser.statistical_validation import StatisticalValidator
except ImportError as e:
    print(f"ImportError: {e}. Ensure 'network_parser' is in PYTHONPATH.")
    import sys
    sys.exit(1)

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
                         all_features: List[str],
                         output_dir: Optional[str] = None) -> Dict:
        """
        Feature discovery using decision tree as discovery engine.
        
        Args:
            data: Genomic data DataFrame.
            labels: Labels Series.
            all_features: List of all feature names.
            output_dir: Directory to save results (optional).
        
        Returns:
            Dictionary with discovered features, trees, and validation results.
        """
        print(f"🔍 Starting feature discovery on {len(data)} samples with {len(all_features)} features...")
        
        # Encode labels for tree building
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)
        n_classes = len(le.classes_)
        
        print(f"📊 Found {n_classes} distinct labels: {le.classes_.tolist()}")

        # Build enhanced decision tree
        dt = self._build_optimal_tree(data, encoded_labels)
        
        # Extract feature roles and interactions
        feature_analysis = self._analyze_tree_structure(dt, all_features, data, encoded_labels)
        
        # Calculate feature confidence scores
        confidence_scores = self._calculate_feature_confidence(
            dt, data, encoded_labels, feature_analysis['root_features'], feature_analysis['branch_features']
        )
        
        # Detect epistatic interactions
        epistatic_interactions = self._detect_epistatic_interactions(dt, all_features, data, encoded_labels)
        
        # Validate statistically
        statistical_validation = {}
        validator = StatisticalValidator(self.config)
        bootstrap_results = validator.bootstrap_validation(data, labels, all_features, output_dir=output_dir)
        chi2_results = validator.chi_squared_test(data[all_features], labels, output_dir=output_dir)
        corrected_results = validator.multiple_testing_correction(chi2_results, output_dir=output_dir)
        
        statistical_validation = {
            'bootstrap': bootstrap_results,
            'chi_squared': chi2_results,
            'multiple_testing': corrected_results
        }
        
        # Compile results
        results = {
            'discovered_features': feature_analysis['root_features'] + feature_analysis['branch_features'],
            'decision_trees': {
                'tree_accuracy': float(dt.score(data, encoded_labels)),
                'tree_rules': export_text(dt, feature_names=list(all_features))
            },
            'feature_confidence': confidence_scores,
            'epistatic_interactions': epistatic_interactions,
            'statistical_validation': statistical_validation,
            'sklearn_importances': dict(zip(all_features, dt.feature_importances_))
        }
        
        # Save results if output_dir is provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_dir / "decision_tree_rules.txt", 'w') as f:
                f.write(results['decision_trees']['tree_rules'])
            print(f"Saved decision tree rules to: {output_dir / 'decision_tree_rules.txt'}")
            
            with open(output_dir / "feature_confidence.json", 'w') as f:
                json.dump(results['feature_confidence'], f, indent=2)
            print(f"Saved feature confidence scores to: {output_dir / 'feature_confidence.json'}")
            
            with open(output_dir / "epistatic_interactions.json", 'w') as f:
                json.dump(results['epistatic_interactions'], f, indent=2)
            print(f"Saved epistatic interactions to: {output_dir / 'epistatic_interactions.json'}")
        
        self._print_discovery_summary(results)
        return results

    def _build_optimal_tree(self, data: pd.DataFrame, encoded_labels: np.ndarray) -> DecisionTreeClassifier:
        """Build decision tree with parameters optimized for feature discovery."""
        dt = DecisionTreeClassifier(
            criterion='gini',
            max_depth=self.config.max_depth or min(10, int(np.log2(len(data)) + 3)),
            min_samples_split=max(2, self.config.min_group_size),
            min_samples_leaf=max(1, self.config.min_group_size // 3),
            min_impurity_decrease=0.001,
            random_state=42
        )
        dt.fit(data, encoded_labels)
        return dt

    def _analyze_tree_structure(self, dt: DecisionTreeClassifier, 
                               feature_names: List[str],
                               data: pd.DataFrame,
                               encoded_labels: np.ndarray) -> Dict:
        """
        Analyze tree structure to extract feature roles.
        
        Returns:
            Dictionary with root features, branch features, feature depths, and split purities.
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
            
            # Calculate information gain
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
                    'information_gain': float(information_gain),
                    'samples': int(total_samples),
                    'parent_feature': parent_feature
                }
            
            feature_depths[feature_name].append(depth)
            
            if depth <= 1:
                root_features.add(feature_name)
            else:
                branch_features.add(feature_name)
            
            if left_child != -1:
                analyze_node(left_child, depth + 1, feature_name)
            if right_child != -1:
                analyze_node(right_child, depth + 1, feature_name)

        analyze_node(0, 0)
        branch_features = branch_features - root_features
        
        return {
            'root_features': list(root_features),
            'branch_features': list(branch_features),
            'feature_depths': {k: [int(d) for d in v] for k, v in feature_depths.items()},
            'split_purities': split_purities
        }

    def _calculate_feature_confidence(self, dt: DecisionTreeClassifier,
                                    data: pd.DataFrame,
                                    encoded_labels: np.ndarray,
                                    root_features: List[str],
                                    branch_features: List[str]) -> Dict:
        """
        Calculate confidence scores for discovered features.
        """
        confidence_scores = {}
        
        for feature in root_features:
            mi_score = mutual_info_score(encoded_labels, data[feature].values)
            stability_scores = []
            for _ in range(10):
                boot_indices = np.random.choice(len(data), size=len(data), replace=True)
                boot_data = data.iloc[boot_indices]
                boot_labels = encoded_labels[boot_indices]
                
                boot_dt = DecisionTreeClassifier(
                    max_depth=3, min_samples_split=5, random_state=np.random.randint(1000)
                )
                boot_dt.fit(boot_data, boot_labels)
                
                importances = dict(zip(boot_data.columns, boot_dt.feature_importances_))
                top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:3]
                stability_scores.append(1.0 if feature in [f[0] for f in top_features] else 0.0)
            
            confidence_scores[feature] = {
                'type': 'root',
                'mutual_info': float(mi_score),
                'stability': float(np.mean(stability_scores)),
                'confidence': float((mi_score + np.mean(stability_scores)) / 2)
            }
        
        for feature in branch_features:
            mi_score = mutual_info_score(encoded_labels, data[feature].values)
            conditional_score = self._calculate_conditional_discrimination(
                data, encoded_labels, feature, root_features
            )
            confidence_scores[feature] = {
                'type': 'branch',
                'mutual_info': float(mi_score),
                'conditional_power': float(conditional_score),
                'confidence': float((mi_score + conditional_score) / 2)
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
        for root_feature in root_features[:3]:
            root_values = data[root_feature].unique()
            for value in root_values:
                mask = data[root_feature] == value
                if mask.sum() < 5:
                    continue
                subset_data = data[mask]
                subset_labels = encoded_labels[mask]
                if len(np.unique(subset_labels)) > 1:
                    mi_gain = mutual_info_score(subset_labels, subset_data[target_feature].values)
                    scores.append(float(mi_gain))
        
        return float(np.mean(scores)) if scores else 0.0

    def _detect_epistatic_interactions(self, dt: DecisionTreeClassifier,
                                     feature_names: List[str],
                                     data: pd.DataFrame,
                                     encoded_labels: np.ndarray) -> List[Dict]:
        """
        Detect epistatic interactions from tree structure.
        """
        tree = dt.tree_
        interactions = []
        
        def find_interactions(node_id: int, path_features: List[str]):
            if tree.feature[node_id] == -2:
                return
                
            current_feature = feature_names[tree.feature[node_id]]
            new_path = path_features + [current_feature]
            
            if len(new_path) >= 2:
                parent_feature = path_features[-1] if path_features else None
                if parent_feature and parent_feature != current_feature:
                    interaction_strength = self._measure_epistatic_strength(
                        data, encoded_labels, parent_feature, current_feature
                    )
                    if interaction_strength > 0.1:
                        interactions.append({
                            'parent_feature': parent_feature,
                            'child_feature': current_feature,
                            'depth': int(len(new_path) - 1),
                            'interaction_strength': float(interaction_strength),
                            'samples_at_split': int(tree.n_node_samples[node_id])  # Convert to Python int
                        })
            
            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]
            
            if left_child != -1:
                find_interactions(left_child, new_path)
            if right_child != -1:
                find_interactions(right_child, new_path)
        
        find_interactions(0, [])
        interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)
        return interactions

    def _measure_epistatic_strength(self, data: pd.DataFrame,
                                  encoded_labels: np.ndarray,
                                  feature1: str, feature2: str) -> float:
        """
        Measure epistatic interaction strength between two features.
        """
        try:
            overall_mi = mutual_info_score(encoded_labels, data[feature2].values)
            conditional_mis = []
            for value1 in data[feature1].unique():
                mask = data[feature1] == value1
                if mask.sum() < 3:
                    continue
                subset_labels = encoded_labels[mask]
                subset_feature2 = data[feature2].values[mask]
                if len(np.unique(subset_labels)) > 1:
                    cond_mi = mutual_info_score(subset_labels, subset_feature2)
                    conditional_mis.append(float(cond_mi))
            
            if conditional_mis:
                avg_conditional_mi = float(np.mean(conditional_mis))
                return float(abs(avg_conditional_mi - overall_mi) / (overall_mi + 1e-6))
        except Exception as e:
            print(f"Warning: Could not calculate epistatic strength for {feature1}-{feature2}: {e}")
        
        return 0.0

    def _print_discovery_summary(self, results: Dict):
        """Print a comprehensive summary of feature discovery results."""
        print("\n" + "="*60)
        print("🎯 FEATURE DISCOVERY SUMMARY")
        print("="*60)
        
        print(f"📈 Tree Accuracy: {results['decision_trees']['tree_accuracy']:.3f}")
        print(f"🏷️  Label Classes: {results['decision_trees']['tree_rules'].splitlines()[0]}")
        
        print(f"\n🌳 ROOT FEATURES (Global Discriminators): {len(results['discovered_features'])}")
        for i, feature in enumerate(results['discovered_features'][:5]):
            confidence = results['feature_confidence'].get(feature, {}).get('confidence', 0)
            print(f"  {i+1}. {feature} (confidence: {confidence:.3f})")
        
        print(f"\n🔗 EPISTATIC INTERACTIONS: {len(results['epistatic_interactions'])}")
        for interaction in results['epistatic_interactions'][:3]:
            print(f"  {interaction['parent_feature']} → {interaction['child_feature']} "
                  f"(strength: {interaction['interaction_strength']:.3f})")
        
        print(f"\n✅ STATISTICAL VALIDATION:")
        corrected = results['statistical_validation']['multiple_testing']
        sig_features = [f for f, v in corrected.items() if isinstance(v, dict) and v.get('significant')]
        print(f"  Significant features after correction: {len(sig_features)}")
        for f in sig_features[:5]:
            print(f"   - {f}: corrected p={corrected[f]['corrected_p_value']:.3e}")
        
        print("="*60 + "\n")

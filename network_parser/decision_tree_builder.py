# network_parser/decision_tree_builder.py
"""
Enhanced Decision Tree Builder for NetworkParser.
Discovers root/branch features, detects epistatic interactions, and computes confidence scores.
Integrates with statistical validation for robust feature extraction.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mutual_info_score, accuracy_score
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import logging
from scipy.stats import chi2_contingency, fisher_exact  # Added missing import for Cramer's V calculation

def log_feature_summary(name: str, features: list, max_show: int = 3):
        count = len(features)
        if count == 0:
            logger.info(f"{name}: 0 features")
            return
        
        if count <= max_show:
            shown = ", ".join(map(str, features))
            logger.info(f"{name}: {count} features → {shown}")
        else:
            shown = ", ".join(map(str, features[:max_show]))
            logger.info(f"{name}: {count:,} features → {shown} ... +{count - max_show:,} more")
# Configure logging for tracking tree-building steps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from network_parser.config import NetworkParserConfig
except ImportError:
    from config import NetworkParserConfig  # Fallback for direct execution

class EnhancedDecisionTreeBuilder:
    """
    Decision tree engine for genomic feature discovery.
    - Hierarchical feature classification (root/branch).
    - Epistatic interaction mining via path analysis.
    - Confidence scoring with mutual information and stability.
    """
    
    def __init__(self, config: NetworkParserConfig):
        self.config = config
        np.random.seed(self.config.random_state)  # Ensure reproducibility
    def discover_features(self, data: pd.DataFrame, labels: pd.Series,
                          all_features: List[str], output_dir: Optional[str] = None) -> Dict:
        """
        Core feature discovery pipeline aligned with workflow: Chi-sq/Fisher, FDR, Tree Construction.
        Builds a decision tree on prefiltered features and analyzes its structure.
        
        Args:
            data: Aligned genomic DataFrame.
            labels: Labels Series.
            all_features: List of feature names.
            output_dir: Output directory.
        
        Returns:
            Dict: Comprehensive results with features, trees, interactions.
        """
        log_feature_summary("Input features", all_features)        
        # Validate input features
        valid_features = [f for f in all_features if f in data.columns]
        if not valid_features:
            raise ValueError("No valid features provided for discovery.")
        if len(valid_features) < len(all_features):
            logger.warning(f"Some features not in data.columns: {set(all_features) - set(valid_features)}")
        
        # Ensure data and labels are aligned
        if not data.index.equals(labels.index):
            common_index = data.index.intersection(labels.index)
            if not common_index.empty:
                data = data.loc[common_index]
                labels = labels.loc[common_index]
                logger.info(f"Aligned data and labels to {len(common_index)} common samples.")
            else:
                raise ValueError("No common indices between data and labels.")
        
        # Pre-filter features using statistical tests
        prefiltered_features = self._prefilter_features(data, labels, valid_features)
        log_feature_summary("Prefiltered features", prefiltered_features)        
        if not prefiltered_features:
            logger.warning("No significant features after prefiltering; using all valid features.")
            prefiltered_features = valid_features
        
        # Subset data to prefiltered features
        try:
            X = data[prefiltered_features].copy()
        except KeyError as e:
            logger.error(f"Error subsetting data with prefiltered features: {e}")
            raise
        logger.info(f"Checking for NaN in X ({X.shape}) ...")
        # Handle missing values explicitly
        nan_per_feature = X.isna().mean() * 100
        bad_features = nan_per_feature[nan_per_feature > 5].index.tolist()  # >5% missing
        
        if bad_features:
            logger.warning(f"Number of features with >5% missing values: {len(bad_features)}")

        
        total_nan = X.isna().sum().sum()
        if total_nan > 0:
            logger.info(f"Imputing {total_nan:,} NaN values → treating as REF (0)")
            X = X.fillna(0).astype(np.float64)
            
            # Optional: drop very sparse features after imputation
            post_impute_maf = X.mean().clip(0, 1)
            too_rare = (post_impute_maf < 0.001) | (post_impute_maf > 0.999)
            if too_rare.any():
                drop_cols = too_rare[too_rare].index.tolist()
                logger.info(f"Dropping {len(drop_cols)} near-monomorphic features after imputation")
                X = X.drop(columns=drop_cols)
                # Also update prefiltered_features if you want to keep lists consistent
                prefiltered_features = [f for f in prefiltered_features if f in X.columns]
        # Encode labels for classification
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)
        n_classes = len(le.classes_)
        logger.info(f"📊 {n_classes} classes: {le.classes_.tolist()}")
        
        # Build tree with optimized parameters
        dt = self._build_enhanced_tree(X, encoded_labels)
        
        # Analyze tree to identify feature roles and interactions
        analysis = self._analyze_hierarchical_structure(dt, prefiltered_features, X, encoded_labels)
        
        # Compute confidence and epistatic interactions
        confidence = self._compute_advanced_confidence(dt, X, encoded_labels, analysis)
        interactions = self._mine_epistatic_networks(dt, prefiltered_features, X, encoded_labels)
        
        results = {
            'discovered_features': analysis['features'],
            'root_features': analysis['root_features'],
            'branch_features': analysis['branch_features'],
            'decision_trees': {
                'accuracy': float(accuracy_score(encoded_labels, dt.predict(X))),
                'rules': export_text(dt, feature_names=prefiltered_features),
                'n_classes': int(n_classes)
            },
            'feature_confidence': confidence,
            'epistatic_interactions': interactions,
            'prefiltered_features': prefiltered_features
        }
        
        self._export_discovery_results(results, output_dir)
        self._print_enhanced_summary(results)
        
        return results
    
    def _prefilter_features(self, data: pd.DataFrame, labels: pd.Series,
                            features: List[str], alpha: Optional[float] = None) -> List[str]:
        """
        Prefilter features using association tests + MAF fallback.
        Returns list of kept features.
        """
        from statsmodels.stats.multitest import multipletests
        
        alpha = alpha or self.config.prefilter_alpha
        MIN_NONMISSING = self.config.min_nonmissing_prefilter
        MIN_MAF = self.config.min_maf_prefilter
        MAX_FEATURES = self.config.max_prefiltered_features
        
        pre_candidates = []
        skip_count = 0
        skip_reasons = defaultdict(int)
        
        for feature in features:
            try:
                col = data[feature]
                nonmissing_frac = col.notna().mean()
                if nonmissing_frac < MIN_NONMISSING:
                    skip_count += 1
                    skip_reasons["Too many missing"] += 1
                    continue
                
                # Compute AF/MAF on non-NA values
                col_non_na = col.dropna()
                af = col_non_na.mean()
                maf = min(af, 1 - af)
                if maf < MIN_MAF:
                    skip_count += 1
                    skip_reasons["MAF too low"] += 1
                    continue
                
                pre_candidates.append(feature)
            except Exception as e:
                skip_count += 1
                reason = str(e).split('\n')[0]
                skip_reasons[reason] += 1
        
        if skip_count > 0:
            logger.info(f"Skipped {skip_count:,} features during missing/MAF pre-filtering")
            for reason, cnt in sorted(skip_reasons.items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f" • {cnt:,} × {reason}")
        
        if not pre_candidates:
            logger.warning("No features passed missing/MAF filters; returning empty list.")
            return []
        
        # Limit the number of stat tests if too many pre_candidates
        max_for_tests = 20000 if MAX_FEATURES is None else max(10000, MAX_FEATURES * 2)
        if len(pre_candidates) > max_for_tests:
            variances = data[pre_candidates].var()
            pre_candidates = variances.nlargest(max_for_tests).index.tolist()
            logger.info(f"Reduced {len(pre_candidates):,} pre-candidates to top {max_for_tests} by variance for stat tests.")
        
        # Now perform association tests
        p_values = []
        valid_features = []
        for feature in pre_candidates:
            try:
                table = pd.crosstab(data[feature], labels)
                if table.empty or min(table.shape) < 2:
                    continue
                
                if table.shape == (2, 2):
                    _, p = fisher_exact(table)
                else:
                    _, p, _, _ = chi2_contingency(table)
                
                p_values.append(p)
                valid_features.append(feature)
            except Exception as e:
                logger.warning(f"Skipping feature {feature} due to error in statistical test: {e}")
        
        if not p_values:
            logger.warning("No valid p-values computed; falling back to top by variance.")
            variances = data[pre_candidates].var()
            n_take = min(5000, len(pre_candidates)) if MAX_FEATURES is None else min(MAX_FEATURES, len(pre_candidates))
            return variances.nlargest(n_take).index.tolist()
        
        # Apply FDR correction
        try:
            reject, corrected_p, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
            significant = [valid_features[i] for i in range(len(valid_features)) if reject[i]]
        except Exception as e:
            logger.error(f"Error in multiple testing correction: {e}")
            significant = valid_features
        
        if not significant:
            logger.warning("No FDR-significant features; falling back to top by variance from pre-candidates.")
            variances = data[pre_candidates].var()
            n_take = min(5000, len(pre_candidates)) if MAX_FEATURES is None else min(MAX_FEATURES, len(pre_candidates))
            significant = variances.nlargest(n_take).index.tolist()
        
        # Cap the final list if needed (rank significant by smallest p-value)
        if MAX_FEATURES is not None and len(significant) > MAX_FEATURES:
            p_df = pd.Series(p_values, index=valid_features)
            sig_p = p_df[significant]
            significant = sig_p.nsmallest(MAX_FEATURES).index.tolist()
            logger.info(f"Capped {len(significant)} significant features to {MAX_FEATURES} by smallest p-value.")
        
        return significant
    
    def _build_enhanced_tree(self, X: pd.DataFrame, y: np.ndarray) -> DecisionTreeClassifier:
        """Build decision tree with config constraints."""
        dt = DecisionTreeClassifier(
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state
        )
        dt.fit(X, y)
        return dt
    
    def _analyze_hierarchical_structure(self, dt: DecisionTreeClassifier,
                                        features: List[str], X: pd.DataFrame,
                                        y: np.ndarray) -> Dict:
        """Analyze tree for root/branch classification."""
        tree_ = dt.tree_
        root_features: Set[str] = set()
        branch_features: Set[str] = set()
        feature_depths: Dict[str, List[int]] = defaultdict(list)
        
        def traverse(node_id: int, depth: int = 0):
            feat_idx = tree_.feature[node_id]
            if feat_idx == -2:  # Leaf
                return
            if feat_idx >= 0:
                feat = features[feat_idx]
                feature_depths[feat].append(depth)
                if depth <= 1:
                    root_features.add(feat)
                else:
                    branch_features.add(feat)
            left, right = tree_.children_left[node_id], tree_.children_right[node_id]
            if left != -1:
                traverse(left, depth + 1)
            if right != -1:
                traverse(right, depth + 1)
        
        traverse(0)
        
        for feat in feature_depths:
            mean_depth = np.mean(feature_depths[feat])
            if mean_depth <= 1.5:
                root_features.add(feat)
                branch_features.discard(feat)
            else:
                branch_features.add(feat)
                root_features.discard(feat)
        
        return {
            'root_features': list(root_features),
            'branch_features': list(branch_features),
            'features': list(root_features | branch_features),
            'depths': dict(feature_depths)
        }
    
    def _compute_advanced_confidence(self, dt: DecisionTreeClassifier,
                                     data: pd.DataFrame, labels: np.ndarray,
                                     analysis: Dict) -> Dict:
        """Compute multi-metric confidence: MI + Bootstrap stability + Effect size."""
        root_feats, branch_feats = analysis['root_features'], analysis['branch_features']
        confidences = {}
        
        for feats, ftype in [(root_feats, 'root'), (branch_feats, 'branch')]:
            for feat in feats:
                mi = mutual_info_score(labels, data[feat])
                stability = np.mean([self._bootstrap_importance(data, labels, feat, n=self.config.bootstrap_samples_per_iter)
                    for _ in range(self.config.bootstrap_outer_iters)
])
                table = pd.crosstab(data[feat], pd.Series(labels, index=data.index))
                cv = self._cramers_v(table) if table.shape[1] > 1 else 0.0
                conf = (mi * 0.4 + stability * 0.4 + cv * 0.2)
                confidences[feat] = {
                    'type': ftype,
                    'mutual_info': float(mi),
                    'stability': float(stability),
                    'cramers_v': float(cv),
                    'confidence': float(conf)
                }
        return confidences
    
    def _bootstrap_importance(self, data: pd.DataFrame, labels: np.ndarray,
                              target_feat: str, n: int = 100) -> float:
        """Single bootstrap importance for target feature."""
        boot_idx = np.random.choice(len(data), len(data), replace=True)
        boot_data, boot_labels = data.iloc[boot_idx], labels[boot_idx]
        dt = DecisionTreeClassifier(max_depth = self.config.max_depth, random_state=self.config.random_state)
        dt.fit(boot_data, boot_labels)
        feat_idx = list(data.columns).index(target_feat)
        return dt.feature_importances_[feat_idx] if feat_idx < len(dt.feature_importances_) else 0.0
    
    def _cramers_v(self, table: pd.DataFrame) -> float:
        """Compute Cramer's V."""
        chi2 = chi2_contingency(table)[0]
        n = table.sum().sum()
        min_dim = min(table.shape) - 1
        return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0
    
    def _mine_epistatic_networks(self, dt: DecisionTreeClassifier,
                                 features: List[str], data: pd.DataFrame,
                                 labels: np.ndarray) -> List[Dict]:
        """Detect interactions via path synergies (workflow step 2: Epistatic)."""
        tree_ = dt.tree_
        interactions = []
        
        def traverse_path(node_id: int, path: List[str]):
            if tree_.feature[node_id] == -2:
                return
            feat_idx = tree_.feature[node_id]
            if feat_idx >= 0:
                feat = features[feat_idx]
                new_path = path + [feat]
                if len(new_path) >= 2:
                    strength = self._epistasis_strength(data, labels, new_path[-2], new_path[-1])
                    if strength > self.config.epistasis_strength_threshold:
                        interactions.append({
                            'features': new_path[-2:],
                            'strength': float(strength),
                            'path_depth': len(new_path) - 1,
                            'support': int(tree_.n_node_samples[node_id]),
                            'type': 'conditional' if len(new_path) > 2 else 'pairwise'
                        })
            left, right = tree_.children_left[node_id], tree_.children_right[node_id]
            if left != -1:
                traverse_path(left, new_path)
            if right != -1:
                traverse_path(right, new_path)
        
        traverse_path(0, [])
        interactions.sort(key=lambda x: x['strength'], reverse=True)
        return interactions[:self.config.max_epistatic_interactions]
    
    def _epistasis_strength(self, data: pd.DataFrame, labels: np.ndarray,
                            feat1: str, feat2: str) -> float:
        """Mutual info synergy: MI(f1,f2|label) - MI(f1|label) - MI(f2|label)."""
        mi1 = mutual_info_score(labels, data[feat1])
        mi2 = mutual_info_score(labels, data[feat2])
        combined = pd.Categorical(data[[feat1, feat2]].astype(str).agg('_'.join, axis=1))
        mi_comb = mutual_info_score(labels, combined)
        return max(0, mi_comb - mi1 - mi2)
    
    def _export_discovery_results(self, results: Dict, output_dir: Optional[str]) -> None:
        """Export tree rules, confidence, interactions."""
        if not output_dir:
            return
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        (out_dir / "decision_tree_rules.txt").write_text(results['decision_trees']['rules'])
        (out_dir / "feature_confidence.json").write_text(json.dumps(results['feature_confidence'], indent=2))
        (out_dir / "epistatic_interactions.json").write_text(json.dumps(results['epistatic_interactions'], indent=2))
    
    def _print_enhanced_summary(self, results: Dict) -> None:
        """Print workflow-aligned summary."""
        print("\n" + "="*70)
        print("\033[1m 🎯 FEATURE DISCOVERY SUMMARY (Workflow Stage 2)\033[0m")
        print("="*70)
        tree = results['decision_trees']
        print(f"📈 Tree Accuracy: {tree['accuracy']:.3f} | Classes: {tree['n_classes']}")
        print(f"🌳 Root Features: {len(results['root_features'])} | Branch: {len(results['branch_features'])}")
        for i, feat in enumerate(results['root_features'][:3], 1):
            conf = results['feature_confidence'][feat]['confidence']
            print(f"  {i}. {feat} (conf: {conf:.3f})")
        print(f"🔗 Epistatic Interactions: {len(results['epistatic_interactions'])}")
        for inter in results['epistatic_interactions'][:2]:
            print(f"  {inter['features'][0]} × {inter['features'][1]} (strength: {inter['strength']:.3f})")
        print("="*70 + "\n")
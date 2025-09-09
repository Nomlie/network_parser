"""
Statistical Validator module for NetworkParser.
Implements rigorous statistical validation for discovered features:
1. Bootstrap validation for feature stability
2. Chi-squared testing for feature-label associations
3. Multiple testing corresctions (Bonferroni, FDR)
4. Permutation tests for interaction validation
5. Effect size calculations

This module provides a comprehensive suite of statistical tests to validate
features discovered by the decision tree builder, ensuring robustness and
significance of findings.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mutual_info_score, accuracy_score
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple, Optional, Union
import warnings
from collections import defaultdict
from pathlib import Path
import json

try:
    from network_parser.config import NetworkParserConfig
except ImportError as e:
    print(f"ImportError: {e}. Ensure 'network_parser' is in PYTHONPATH.")
    import sys
    sys.exit(1)

class StatisticalValidator:
    """
    Comprehensive statistical validation suite for feature discovery.
    
    Validates discovered features through:
    - Bootstrap resampling for stability testing
    - Chi-squared tests for independence
    - Multiple testing corrections
    - Permutation tests for interaction effects
    - Effect size measurements (Cramér's V, Cohen's d)
    """

    def __init__(self, config: NetworkParserConfig):
        """
        Initialize the StatisticalValidator.

        Args:
            config: Configuration object containing parameters like significance level,
                    number of bootstrap samples, and permutation tests.
        """
        self.config = config
        self.alpha = getattr(config, 'significance_level', 0.05)
        self.n_bootstrap = getattr(config, 'n_bootstrap_samples', 1000)
        self.n_permutations = getattr(config, 'n_permutation_tests', 500)
        print("Initialized StatisticalValidator with provided configuration.")

    def bootstrap_validation(self, 
                            data: pd.DataFrame, 
                            labels: pd.Series, 
                            features: List[str],
                            output_dir: Optional[str] = None,
                            n_bootstrap: Optional[int] = None) -> Dict:
        """
        Bootstrap validation of feature importance and stability.
        
        Tests whether discovered features remain important across 
        different bootstrap samples of the data.
        
        Args:
            data: DataFrame containing features and samples.
            labels: Series of target labels.
            features: List of features to validate.
            output_dir: Directory to save results (optional).
            n_bootstrap: Number of bootstrap samples (overrides config if provided).

        Returns:
            Dictionary with stability scores, confidence intervals,
            bootstrap distributions, and significance tests.

        Raises:
            ValueError: If data or labels are empty or mismatched.
        """
        if data.empty or labels.empty:
            raise ValueError("Data or labels cannot be empty.")
        if len(data) != len(labels):
            raise ValueError("Data and labels must have the same number of samples.")
        if not features:
            return {}
        
        n_bootstrap = n_bootstrap or self.n_bootstrap
        print(f"🔄 Running bootstrap validation with {n_bootstrap} samples...")
        
        results = {
            'stability_scores': {},
            'confidence_intervals': {},
            'bootstrap_distributions': {},
            'significance_tests': {}
        }
        
        feature_importances = defaultdict(list)
        feature_rankings = defaultdict(list)
        
        for i in range(n_bootstrap):
            if i % 100 == 0:
                print(f"   Bootstrap sample {i+1}/{n_bootstrap}")
                
            boot_indices = np.random.choice(len(data), size=len(data), replace=True)
            boot_data = data.iloc[boot_indices]
            boot_labels = labels.iloc[boot_indices]
            
            if len(boot_labels.unique()) < 2:
                continue
                
            try:
                dt = DecisionTreeClassifier(
                    max_depth=self.config.max_depth or 5,
                    min_samples_split=max(2, self.config.min_group_size),
                    min_samples_leaf=max(1, self.config.min_group_size // 3),
                    random_state=i
                )
                dt.fit(boot_data[features], boot_labels)
                
                importances = dict(zip(features, dt.feature_importances_))
                for feature in features:
                    feature_importances[feature].append(float(importances.get(feature, 0.0)))
                
                sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                for rank, (feature, _) in enumerate(sorted_features, 1):
                    feature_rankings[feature].append(int(rank))
                    
            except Exception as e:
                warnings.warn(f"Bootstrap sample {i} failed: {e}")
                continue
        
        for feature in features:
            if not feature_importances[feature]:
                continue
                
            importances = np.array(feature_importances[feature])
            rankings = np.array(feature_rankings[feature])
            
            median_rank = len(features) / 2
            stability_score = float(np.mean(rankings <= median_rank))
            
            ci_lower, ci_upper = np.percentile(importances, [2.5, 97.5])
            
            t_stat, p_value = stats.ttest_1samp(importances, 0.0)
            
            results['stability_scores'][feature] = stability_score
            results['confidence_intervals'][feature] = {
                'mean_importance': float(np.mean(importances)),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'std': float(np.std(importances))
            }
            results['bootstrap_distributions'][feature] = importances.tolist()
            results['significance_tests'][feature] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': bool(p_value < self.alpha)  # Convert to Python bool
            }
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "bootstrap_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved bootstrap results to: {output_dir / 'bootstrap_results.json'}")
        
        return results

    def chi_squared_test(self, 
                         data: pd.DataFrame, 
                         labels: pd.Series,
                         output_dir: Optional[str] = None) -> Dict:
        """
        Chi-squared tests of independence between features and labels.
        
        Tests the null hypothesis that each feature is independent
        of the class labels. Uses Fisher's exact test for small samples.
        
        Args:
            data: DataFrame with features to test.
            labels: Series of target labels.
            output_dir: Directory to save results (optional).

        Returns:
            Dictionary with test results per feature, including statistics,
            p-values, effect sizes, and contingency tables.
        """
        print(f"🧮 Running chi-squared tests for {len(data.columns)} features...")
        
        results = {}
        
        for feature in data.columns:
            try:
                contingency_table = pd.crosstab(data[feature], labels)
                
                if contingency_table.min().min() < 5:
                    if contingency_table.shape == (2, 2):
                        odds_ratio, p_value = fisher_exact(contingency_table.values)
                        chi2_stat = np.nan
                        cramers_v = self._calculate_cramers_v_from_table(contingency_table)
                        test_type = 'fisher_exact'
                    else:
                        results[feature] = {
                            'chi2_statistic': None,
                            'p_value': 1.0,
                            'cramers_v': 0.0,
                            'mutual_information': 0.0,
                            'error': 'Sparse contingency table; test skipped',
                            'test_type': 'skipped'
                        }
                        continue
                else:
                    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table.values)
                    cramers_v = self._calculate_cramers_v(chi2_stat, contingency_table.sum().sum(), 
                                                          min(contingency_table.shape) - 1)
                    test_type = 'chi_squared'
                
                mi_score = mutual_info_score(labels.values, data[feature].values)
                
                results[feature] = {
                    'chi2_statistic': float(chi2_stat) if not np.isnan(chi2_stat) else None,
                    'p_value': float(p_value),
                    'cramers_v': float(cramers_v),
                    'mutual_information': float(mi_score),
                    'contingency_table': contingency_table.to_dict(),
                    'test_type': test_type
                }
                
            except Exception as e:
                warnings.warn(f"Chi-squared test failed for feature {feature}: {e}")
                results[feature] = {
                    'chi2_statistic': None,
                    'p_value': 1.0,
                    'cramers_v': 0.0,
                    'mutual_information': 0.0,
                    'error': str(e),
                    'test_type': 'failed'
                }
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "chi_squared_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved chi-squared results to: {output_dir / 'chi_squared_results.json'}")
        
        return results

    def multiple_testing_correction(self, 
                                    chi2_results: Dict,
                                    output_dir: Optional[str] = None,
                                    method: str = 'fdr_bh') -> Dict:
        """
        Apply multiple testing corrections to control family-wise error rate.
        
        Available methods: 'bonferroni', 'fdr_bh', 'fdr_by', 'holm'.
        
        Args:
            chi2_results: Results from chi_squared_test.
            output_dir: Directory to save results (optional).
            method: Correction method (default: 'fdr_bh').

        Returns:
            Dictionary with corrected p-values, significance flags,
            effect sizes, and summary statistics.
        """
        print(f"🔧 Applying multiple testing correction using {method}...")
        
        features = []
        p_values = []
        
        for feature, results in chi2_results.items():
            if 'p_value' in results and results['p_value'] is not None:
                features.append(feature)
                p_values.append(results['p_value'])
        
        if not p_values:
            return {}
        
        try:
            rejected, corrected_p_values, alpha_sidak, alpha_bonf = multipletests(
                p_values, alpha=self.alpha, method=method
            )
        except Exception as e:
            warnings.warn(f"Multiple testing correction failed: {e}")
            return {}
        
        corrected_results = {}
        
        for i, feature in enumerate(features):
            original_results = chi2_results[feature]
            
            cramers_v = original_results.get('cramers_v', 0.0)
            effect_size = self._classify_effect_size(cramers_v)
            
            corrected_results[feature] = {
                'original_p_value': float(original_results['p_value']),
                'corrected_p_value': float(corrected_p_values[i]),
                'significant': bool(rejected[i]),  # Convert to Python bool
                'cramers_v': float(cramers_v),
                'effect_size': effect_size,
                'mutual_information': float(original_results.get('mutual_information', 0.0)),
                'chi2_statistic': original_results.get('chi2_statistic'),
                'correction_method': method
            }
        
        corrected_results['_summary'] = {
            'n_features_tested': len(features),
            'n_significant_original': int(sum(p < self.alpha for p in p_values)),
            'n_significant_corrected': int(sum(rejected)),
            'alpha_level': float(self.alpha),
            'correction_method': method,
            'family_wise_error_rate': float(alpha_bonf) if 'bonferroni' in method else None
        }
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "multiple_testing_results.json", 'w') as f:
                json.dump(corrected_results, f, indent=2)
            print(f"Saved multiple testing results to: {output_dir / 'multiple_testing_results.json'}")
        
        return corrected_results

    def permutation_test_interactions(self, 
                                     data: pd.DataFrame,
                                     labels: pd.Series,
                                     interaction_pairs: List[Tuple[str, str]],
                                     output_dir: Optional[str] = None,
                                     n_permutations: Optional[int] = None) -> Dict:
        """
        Permutation tests for epistatic interactions.
        
        Compares observed interaction strengths against null distribution
        from permuted labels.
        
        Args:
            data: Feature DataFrame.
            labels: Target labels.
            interaction_pairs: List of (feature1, feature2) pairs.
            output_dir: Directory to save results (optional).
            n_permutations: Number of permutations (overrides config if provided).

        Returns:
            Dictionary with test results per interaction pair.
        """
        n_permutations = n_permutations or self.n_permutations
        print(f"🔀 Running permutation tests for {len(interaction_pairs)} interactions...")
        
        results = {}
        
        for feature1, feature2 in interaction_pairs:
            print(f"   Testing interaction: {feature1} × {feature2}")
            
            observed_strength = self._calculate_interaction_strength(
                data, labels, feature1, feature2
            )
            
            null_strengths = []
            for i in range(n_permutations):
                permuted_labels = labels.sample(frac=1, random_state=i).reset_index(drop=True)
                
                null_strength = self._calculate_interaction_strength(
                    data, permuted_labels, feature1, feature2
                )
                null_strengths.append(float(null_strength))
            
            null_strengths = np.array(null_strengths)
            p_value = float(np.mean(np.abs(null_strengths) >= abs(observed_strength)))
            
            results[f"{feature1}_x_{feature2}"] = {
                'observed_strength': float(observed_strength),
                'null_mean': float(np.mean(null_strengths)),
                'null_std': float(np.std(null_strengths)),
                'p_value': p_value,
                'significant': bool(p_value < self.alpha),  # Convert to Python bool
                'z_score': float((observed_strength - np.mean(null_strengths)) / 
                                 (np.std(null_strengths) + 1e-8)),
                'n_permutations': int(n_permutations)
            }
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "interaction_permutation_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved interaction permutation results to: {output_dir / 'interaction_permutation_results.json'}")
        
        return results

    def validate_feature_set(self,
                            data: pd.DataFrame,
                            labels: pd.Series, 
                            discovered_features: List[str],
                            output_dir: Optional[str] = None,
                            baseline_features: Optional[List[str]] = None) -> Dict:
        """
        Comprehensive validation of a discovered feature set.
        
        Compares against random sets and individual features.
        Baseline features comparison is optional.
        
        Args:
            data: Feature DataFrame.
            labels: Target labels.
            discovered_features: List of discovered features.
            output_dir: Directory to save results (optional).
            baseline_features: Optional list of baseline features for comparison.

        Returns:
            Dictionary with performance metrics and comparisons.
        """
        print(f"📊 Comprehensive validation of {len(discovered_features)} discovered features...")
        
        results = {
            'discovered_performance': {},
            'random_baseline_comparison': {},
            'individual_feature_analysis': {},
            'feature_set_statistics': {}
        }
        
        discovered_performance = self._test_feature_set_performance(
            data, labels, discovered_features
        )
        results['discovered_performance'] = discovered_performance
        
        random_comparison = self._compare_against_random_features(
            data, labels, discovered_features
        )
        results['random_baseline_comparison'] = random_comparison
        
        individual_analysis = {}
        for feature in discovered_features:
            individual_perf = self._test_feature_set_performance(
                data, labels, [feature]
            )
            individual_analysis[feature] = individual_perf
        results['individual_feature_analysis'] = individual_analysis
        
        results['feature_set_statistics'] = {
            'n_features': int(len(discovered_features)),
            'mean_individual_accuracy': float(np.mean([
                perf['accuracy'] for perf in individual_analysis.values()
            ])),
            'feature_set_accuracy': float(discovered_performance['accuracy']),
            'synergy_score': float(discovered_performance['accuracy'] - np.mean([
                perf['accuracy'] for perf in individual_analysis.values()
            ]))
        }
        
        if baseline_features:
            baseline_perf = self._test_feature_set_performance(
                data, labels, baseline_features
            )
            results['baseline_comparison'] = {
                'baseline_accuracy': float(baseline_perf['accuracy']),
                'discovered_accuracy': float(discovered_performance['accuracy']),
                'improvement': float(discovered_performance['accuracy'] - baseline_perf['accuracy'])
            }
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "feature_set_validation.json", 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved feature set validation results to: {output_dir / 'feature_set_validation.json'}")
        
        return results

    def _calculate_cramers_v(self, chi2: float, n: int, min_dim: int) -> float:
        """
        Calculate Cramér's V effect size measure.

        Args:
            chi2: Chi-squared statistic.
            n: Total number of observations.
            min_dim: Minimum dimension minus 1.

        Returns:
            Cramér's V value.
        """
        return float(np.sqrt(chi2 / (n * min_dim)))
    
    def _calculate_cramers_v_from_table(self, contingency_table: pd.DataFrame) -> float:
        """
        Calculate Cramér's V directly from contingency table.

        Args:
            contingency_table: Crosstab of feature and labels.

        Returns:
            Cramér's V value.
        """
        chi2, _, _, _ = chi2_contingency(contingency_table.values)
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        return float(self._calculate_cramers_v(chi2, n, min_dim))
    
    def _classify_effect_size(self, cramers_v: float) -> str:
        """
        Classify effect size based on Cramér's V value.

        Args:
            cramers_v: Cramér's V value.

        Returns:
            String classification: 'negligible', 'small', 'medium', 'large'.
        """
        if cramers_v < 0.1:
            return 'negligible'
        elif cramers_v < 0.3:
            return 'small'
        elif cramers_v < 0.5:
            return 'medium'
        else:
            return 'large'
    
    def _calculate_interaction_strength(self, 
                                      data: pd.DataFrame,
                                      labels: pd.Series,
                                      feature1: str, 
                                      feature2: str) -> float:
        """
        Calculate statistical interaction strength between two features.
        Uses mutual information difference method.

        Args:
            data: Feature DataFrame.
            labels: Target labels.
            feature1: First feature.
            feature2: Second feature.

        Returns:
            Interaction strength value.
        """
        try:
            mi1 = mutual_info_score(labels.values, data[feature1].values)
            mi2 = mutual_info_score(labels.values, data[feature2].values)
            combined_feature = data[feature1].astype(str) + "_" + data[feature2].astype(str)
            mi_combined = mutual_info_score(labels.values, combined_feature.values)
            interaction_strength = mi_combined - (mi1 + mi2)
            return float(interaction_strength)
        except Exception as e:
            warnings.warn(f"Could not calculate interaction strength for {feature1}-{feature2}: {e}")
            return 0.0
    
    def _test_feature_set_performance(self, 
                                     data: pd.DataFrame,
                                     labels: pd.Series,
                                     features: List[str]) -> Dict:
        """
        Test predictive performance of a feature set using a simple decision tree.

        Args:
            data: Feature DataFrame.
            labels: Target labels.
            features: List of features to use.

        Returns:
            Dictionary with accuracy and other metrics.
        """
        try:
            dt = DecisionTreeClassifier(
                max_depth=3,
                min_samples_split=5,
                random_state=42
            )
            dt.fit(data[features], labels)
            accuracy = dt.score(data[features], labels)
            return {
                'accuracy': float(accuracy),
                'n_features': int(len(features)),
                'features': features
            }
        except Exception as e:
            return {
                'accuracy': 0.0,
                'n_features': int(len(features)),
                'features': features,
                'error': str(e)
            }
    
    def _compare_against_random_features(self,
                                        data: pd.DataFrame, 
                                        labels: pd.Series,
                                        discovered_features: List[str],
                                        n_random_tests: int = 100) -> Dict:
        """
        Compare discovered features against random feature sets of same size.

        Args:
            data: Feature DataFrame.
            labels: Target labels.
            discovered_features: Discovered features.
            n_random_tests: Number of random sets to test.

        Returns:
            Dictionary with statistical comparison results.
        """
        n_features = len(discovered_features)
        all_features = list(data.columns)
        
        random_accuracies = []
        for i in range(n_random_tests):
            np.random.seed(i)
            random_features = np.random.choice(
                all_features, size=min(n_features, len(all_features)), replace=False
            ).tolist()
            random_perf = self._test_feature_set_performance(
                data, labels, random_features
            )
            random_accuracies.append(float(random_perf['accuracy']))
        
        discovered_perf = self._test_feature_set_performance(
            data, labels, discovered_features
        )
        
        random_mean = float(np.mean(random_accuracies))
        random_std = float(np.std(random_accuracies))
        
        z_score = float((discovered_perf['accuracy'] - random_mean) / (random_std + 1e-8))
        p_value = float(2 * (1 - stats.norm.cdf(abs(z_score))))
        
        return {
            'discovered_accuracy': float(discovered_perf['accuracy']),
            'random_mean_accuracy': random_mean,
            'random_std_accuracy': random_std,
            'z_score': z_score,
            'p_value': p_value,
            'significant': bool(p_value < self.alpha),  # Convert to Python bool
            'n_random_tests': int(n_random_tests),
            'percentile_rank': float(np.mean(
                np.array(random_accuracies) < discovered_perf['accuracy']
            ) * 100)
        }

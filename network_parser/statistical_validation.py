# network_parser/statistical_validator.py
"""
Statistical validation suite for feature discovery and interactions.
"""

import logging
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mutual_info_score
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from pathlib import Path
import json
from joblib import Parallel, delayed  # For parallel processing
from network_parser.config import NetworkParserConfig

logger = logging.getLogger(__name__)

class StatisticalValidator:
    """
    Statistical validation suite for feature discovery and interactions.
    """

    def __init__(self, config: NetworkParserConfig):
        self.config = config
        self.alpha = self.config.significance_level
        self.n_bootstrap = self.config.n_bootstrap_samples
        self.n_permutations = self.config.n_permutation_tests
        self.n_jobs = self.config.n_jobs
        logger.info("Initialized StatisticalValidator.")

    def chi_squared_test(self, data: pd.DataFrame, labels: pd.Series,
                         output_dir: Optional[str] = None) -> Dict:
        """
        Perform chi-squared or Fisher's exact tests for feature-label associations.
        """
        logger.info(f"Running association tests for {len(data.columns)} features.")
        results = {}

        def test_feature(feature):
            contingency = pd.crosstab(data[feature], labels)
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                return feature, None
            if contingency.values.min() < self.config.chi2_min_expected:
                _, p_value = fisher_exact(contingency.values)
                statistic = None
                dof = None
            else:
                statistic, p_value, dof, _ = chi2_contingency(contingency.values)
            cramers_v = self._calculate_cramers_v_from_table(contingency)
            mi = mutual_info_score(labels.values, data[feature].values)
            return feature, {
                'statistic': float(statistic) if statistic is not None else None,
                'p_value': float(p_value),
                'dof': int(dof) if dof is not None else None,
                'cramers_v': float(cramers_v),
                'mutual_info': float(mi),
                'effect_size_class': self._classify_effect_size(cramers_v),
                'contingency_table': contingency.to_dict()
            }

        feature_results = Parallel(n_jobs=self.n_jobs)(
            delayed(test_feature)(feature) for feature in data.columns
        )
        for feature, result in feature_results:
            if result:
                results[feature] = result

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(output_dir) / "chi_squared_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved chi-squared results to: {output_dir}")

        return results

    def multiple_testing_correction(self, test_results: Dict, method: str = None,
                                    output_dir: Optional[str] = None) -> Dict:
        """
        Apply multiple testing correction to p-values.
        """
        method = method or self.config.multiple_testing_method
        logger.info(f"Applying multiple testing correction using {method}.")
        p_values = [res['p_value'] for res in test_results.values() if 'p_value' in res]
        features = [f for f, res in test_results.items() if 'p_value' in res]
        if not p_values:
            return {}

        corrected = multipletests(p_values, alpha=self.alpha, method=method)
        results = {f: {
            'corrected_p_value': float(corrected[1][i]),
            'significant': bool(corrected[0][i]),
            **test_results[f]
        } for i, f in enumerate(features)}

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(output_dir) / "multiple_testing_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved multiple testing results to: {output_dir}")

        return results

    def bootstrap_validation(self, data: pd.DataFrame, labels: pd.Series, features: List[str],
                             output_dir: Optional[str] = None) -> Dict:
        """
        Bootstrap validation for feature stability.
        """
        logger.info(f"Running bootstrap validation with {self.n_bootstrap} samples.")

        def bootstrap_sample(i):
            boot_indices = np.random.choice(len(data), len(data), replace=True)
            boot_data = data.iloc[boot_indices]
            boot_labels = labels.iloc[boot_indices]
            if len(boot_labels.unique()) < 2:
                return None
            dt = DecisionTreeClassifier(
                max_depth=self.config.max_depth or 5,
                min_samples_split=max(2, self.config.min_group_size),
                min_samples_leaf=max(1, self.config.min_group_size // 3),
                random_state=i
            )
            dt.fit(boot_data[features], boot_labels)
            importances = dict(zip(features, dt.feature_importances_))
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            rankings = {f: rank for rank, (f, _) in enumerate(sorted_features, 1)}
            return importances, rankings

        bootstrap_results = Parallel(n_jobs=self.n_jobs)(
            delayed(bootstrap_sample)(i) for i in range(self.n_bootstrap)
        )
        bootstrap_results = [res for res in bootstrap_results if res]

        feature_importances = defaultdict(list)
        feature_rankings = defaultdict(list)
        for importances, rankings in bootstrap_results:
            for f in features:
                feature_importances[f].append(importances.get(f, 0.0))
                feature_rankings[f].append(rankings.get(f, len(features) + 1))

        results = {}
        for f in features:
            importances = np.array(feature_importances[f])
            rankings = np.array(feature_rankings[f])
            stability = float(np.mean(rankings <= len(features) / 2))
            mean_imp = float(np.mean(importances))
            ci = tuple(map(float, np.percentile(importances, [2.5, 97.5])))
            t_stat, p_value = stats.ttest_1samp(importances, 0.0)
            results[f] = {
                'stability_score': stability,
                'mean_importance': mean_imp,
                'ci': ci,
                'p_value': float(p_value),
                'significant': bool(p_value < self.alpha)  # Convert numpy.bool_ to Python bool
            }

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(output_dir) / "bootstrap_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved bootstrap results to: {output_dir}")

        return results

    def permutation_test_interactions(self, data: pd.DataFrame, labels: pd.Series,
                                      interactions: List[Tuple[str, str]],
                                      output_dir: Optional[str] = None) -> Dict:
        """
        Permutation tests for epistatic interactions.
        """
        logger.info(f"Running permutation tests for {len(interactions)} interactions.")

        def test_interaction(pair):
            f1, f2 = pair
            observed = self._calculate_interaction_strength(data, labels, f1, f2)
            perms = []
            for _ in range(self.n_permutations):
                perm_labels = np.random.permutation(labels)
                perms.append(self._calculate_interaction_strength(data, perm_labels, f1, f2))
            p_value = np.mean([p >= observed for p in perms])
            return pair, {'observed_strength': observed, 'p_value': p_value, 'significant': bool(p_value < self.alpha)}

        interaction_results = dict(Parallel(n_jobs=self.n_jobs)(
            delayed(test_interaction)(pair) for pair in interactions
        ))

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(output_dir) / "interaction_permutation_results.json", 'w') as f:
                json.dump(interaction_results, f, indent=2)
            logger.info(f"Saved interaction permutation results to: {output_dir}")

        return interaction_results

    def _calculate_cramers_v_from_table(self, contingency: pd.DataFrame) -> float:
        """Compute Cramer's V from contingency table."""
        chi2 = chi2_contingency(contingency.values)[0]
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0

    def _classify_effect_size(self, v: float) -> str:
        """Classify Cramer's V effect size."""
        if v < 0.1: return 'negligible'
        if v < 0.3: return 'small'
        if v < 0.5: return 'medium'
        return 'large'

    def _calculate_interaction_strength(self, data: pd.DataFrame, labels: pd.Series, f1: str, f2: str) -> float:
        """Calculate interaction strength via mutual information."""
        mi1 = mutual_info_score(labels, data[f1])
        mi2 = mutual_info_score(labels, data[f2])
        combined = data[f1].astype(str) + "_" + data[f2].astype(str)
        mi_combined = mutual_info_score(labels, combined)
        return mi_combined - (mi1 + mi2)
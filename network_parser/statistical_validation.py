# network_parser/statistical_validator.py
"""
Statistical validation module.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.multitest import fdrcorrection
from collections import defaultdict
from typing import Dict, List

from .config import NetworkParserConfig

class StatisticalValidator:
    """Handles statistical validation and multiple testing correction"""
    
    def __init__(self, config: NetworkParserConfig):
        self.config = config
    
    def bootstrap_validation(self, data: pd.DataFrame, labels: pd.Series, 
                           features: List[str], n_iterations: int = None) -> Dict[str, Dict]:
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
                'confidence_interval': np.percentile(scores, [2.5, 97.5]).tolist()
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
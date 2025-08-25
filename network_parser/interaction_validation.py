# network_parser/interaction_detector.py
"""
Epistatic interaction detection module.
"""

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
from typing import Dict, List, Tuple

from .config import NetworkParserConfig

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
                        'features': list(feature_combo),
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
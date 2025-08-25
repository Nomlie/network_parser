# network_parser/decision_tree_builder.py
"""
Decision tree and hierarchical clustering module.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import pdist
from typing import Dict, List, Tuple

from .config import NetworkParserConfig

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
            'label_encoder': le.classes_.tolist()
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
                    'clusters': clusters.tolist(),
                    'silhouette_score': self._calculate_silhouette_score(data[significant_features].values, clusters)
                }
        
        return {
            'linkage_matrix': linkage_matrix.tolist(),
            'cluster_levels': cluster_info,
            'distances': distances.tolist()
        }
    
    def _calculate_silhouette_score(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering quality"""
        if len(np.unique(labels)) > 1:
            return float(silhouette_score(data, labels))
        return 0.0
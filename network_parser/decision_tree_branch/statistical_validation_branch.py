# network_parser/decision_tree_branch/statistical_validation_branch.py
"""
Statistical validation branch (NetworkParser).

Refactor of your statistical_validation.py:
- feature association tests (chi2/fisher) + MI + Cramer's V
- multiple testing correction
- bootstrap validation of feature stability
- permutation testing of interaction pairs
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.metrics import mutual_info_score
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.multitest import multipletests

try:
    from network_parser.config import NetworkParserConfig
except Exception:  # pragma: no cover
    from config import NetworkParserConfig  # type: ignore

logger = logging.getLogger(__name__)


class StatisticalValidatorBranch:
    """
    Validation suite for:
      - discovered feature set
      - interaction candidate set
    """

    def __init__(self, config: NetworkParserConfig):
        self.config = config
        self.alpha = float(getattr(self.config, "significance_level", 0.05))
        self.n_bootstrap = int(getattr(self.config, "n_bootstrap_samples", 1000))
        self.n_permutations = int(getattr(self.config, "n_permutation_tests", 500))
        self.n_jobs = int(getattr(self.config, "n_jobs", -1))

        logger.info("Initialized StatisticalValidatorBranch.")

    # ------------------------- Feature association -------------------------

    def association_tests(
        self,
        data: pd.DataFrame,
        labels: pd.Series,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        For each feature:
          - chi2 or fisher (fallback if low expected)
          - Cramer's V
          - mutual information
        """
        results: Dict[str, Any] = {}

        def test_feature(feature: str):
            contingency = pd.crosstab(data[feature], labels)
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                return feature, None

            if int(contingency.values.min()) < int(getattr(self.config, "chi2_min_expected", 5)):
                _, p_value = fisher_exact(contingency.values)
                statistic = None
                dof = None
            else:
                statistic, p_value, dof, _ = chi2_contingency(contingency.values)

            cramers_v = self._cramers_v_from_table(contingency)
            mi = mutual_info_score(labels.values, data[feature].values)

            return feature, {
                "statistic": float(statistic) if statistic is not None else None,
                "p_value": float(p_value),
                "dof": int(dof) if dof is not None else None,
                "cramers_v": float(cramers_v),
                "mutual_info": float(mi),
                "effect_size_class": self._classify_effect_size(float(cramers_v)),
                "contingency_table": contingency.to_dict(),
            }

        feature_results = Parallel(n_jobs=self.n_jobs)(delayed(test_feature)(f) for f in data.columns)
        for feature, res in feature_results:
            if res:
                results[feature] = res

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "chi_squared_results.json").write_text(json.dumps(results, indent=2))

        return results

    def multiple_testing_correction(
        self,
        test_results: Dict[str, Any],
        method: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        method = method or str(getattr(self.config, "multiple_testing_method", "fdr_bh"))

        p_values = [res["p_value"] for res in test_results.values() if "p_value" in res]
        feats = [f for f, res in test_results.items() if "p_value" in res]
        if not p_values:
            return {}

        corrected = multipletests(p_values, alpha=self.alpha, method=method)
        out: Dict[str, Any] = {}
        for i, f in enumerate(feats):
            out[f] = {
                "corrected_p_value": float(corrected[1][i]),
                "significant": bool(corrected[0][i]),
                **test_results[f],
            }

        if output_dir:
            p = Path(output_dir)
            p.mkdir(parents=True, exist_ok=True)
            (p / "multiple_testing_results.json").write_text(json.dumps(out, indent=2))

        return out

    # ------------------------- Bootstrap stability -------------------------

    def bootstrap_validation(
        self,
        data: pd.DataFrame,
        labels: pd.Series,
        features: List[str],
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Bootstrap stability on a feature set:
          - fit a tree per bootstrap sample
          - collect importances and ranks
          - stability = fraction of times rank <= half of feature count
        """
        logger.info("Bootstrap validation: n_bootstrap=%d", self.n_bootstrap)

        def bootstrap_sample(i: int):
            boot_idx = np.random.choice(len(data), len(data), replace=True)
            boot_data = data.iloc[boot_idx]
            boot_labels = labels.iloc[boot_idx]
            if boot_labels.nunique() < 2:
                return None

            dt = DecisionTreeClassifier(
                max_depth=(getattr(self.config, "max_depth", None) or 5),
                min_samples_split=max(2, int(getattr(self.config, "min_group_size", 2))),
                min_samples_leaf=max(1, int(getattr(self.config, "min_group_size", 2)) // 3),
                random_state=i,
            )
            dt.fit(boot_data[features], boot_labels)

            importances = dict(zip(features, dt.feature_importances_))
            sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            rankings = {f: rank for rank, (f, _) in enumerate(sorted_feats, 1)}
            return importances, rankings

        boot = Parallel(n_jobs=self.n_jobs)(delayed(bootstrap_sample)(i) for i in range(self.n_bootstrap))
        boot = [b for b in boot if b is not None]

        imp_map = defaultdict(list)
        rank_map = defaultdict(list)
        for importances, rankings in boot:
            for f in features:
                imp_map[f].append(float(importances.get(f, 0.0)))
                rank_map[f].append(int(rankings.get(f, len(features) + 1)))

        results: Dict[str, Any] = {}
        half = max(1, len(features) / 2)

        for f in features:
            imps = np.asarray(imp_map[f], dtype=float)
            ranks = np.asarray(rank_map[f], dtype=float)
            stability = float(np.mean(ranks <= half)) if ranks.size else 0.0
            mean_imp = float(np.mean(imps)) if imps.size else 0.0
            ci = tuple(map(float, np.percentile(imps, [2.5, 97.5]))) if imps.size else (0.0, 0.0)
            _, p_value = stats.ttest_1samp(imps, 0.0) if imps.size else (0.0, 1.0)

            results[f] = {
                "stability_score": stability,
                "mean_importance": mean_imp,
                "ci": ci,
                "p_value": float(p_value),
                "significant": bool(float(p_value) < self.alpha),
            }

        if output_dir:
            p = Path(output_dir)
            p.mkdir(parents=True, exist_ok=True)
            (p / "bootstrap_results.json").write_text(json.dumps(results, indent=2))

        return results

    # ------------------------- Interaction permutation -------------------------

    def permutation_test_interactions(
        self,
        data: pd.DataFrame,
        labels: pd.Series,
        interactions: List[Tuple[str, str]],
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        logger.info("Permutation testing interactions: n_pairs=%d", len(interactions))

        def test_pair(pair: Tuple[str, str]):
            f1, f2 = pair
            observed = self._interaction_strength(data, labels, f1, f2)
            perms = []
            for _ in range(self.n_permutations):
                perm_labels = np.random.permutation(labels)
                perms.append(self._interaction_strength(data, perm_labels, f1, f2))
            p_value = float(np.mean([p >= observed for p in perms]))
            return pair, {"observed_strength": float(observed), "p_value": p_value, "significant": bool(p_value < self.alpha)}

        out = dict(Parallel(n_jobs=self.n_jobs)(delayed(test_pair)(pair) for pair in interactions))

        if output_dir:
            p = Path(output_dir)
            p.mkdir(parents=True, exist_ok=True)
            (p / "interaction_permutation_results.json").write_text(json.dumps(out, indent=2))

        return out

    # ------------------------- Orchestrator wrappers -------------------------

    def validate_features(
        self,
        genomic_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        label_column: str,
        discovered_features: Optional[List[str]],
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        if meta_df is None:
            raise ValueError("meta_df is required for validation")
        if label_column not in meta_df.columns:
            raise ValueError(f"label_column not found: {label_column}")

        labels = meta_df[label_column]
        features = discovered_features or list(genomic_df.columns)
        data = genomic_df[features]

        assoc = self.association_tests(data=data, labels=labels, output_dir=output_dir)
        mtest = self.multiple_testing_correction(assoc, output_dir=output_dir)
        boot = self.bootstrap_validation(data=genomic_df, labels=labels, features=features, output_dir=output_dir)

        return {"association": assoc, "multiple_testing": mtest, "bootstrap": boot}

    def validate_interactions(
        self,
        genomic_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        label_column: str,
        interactions: List[Tuple[str, str]],
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        if meta_df is None:
            raise ValueError("meta_df is required for interaction validation")
        if label_column not in meta_df.columns:
            raise ValueError(f"label_column not found: {label_column}")
        if not interactions:
            return {}
        labels = meta_df[label_column]
        return self.permutation_test_interactions(genomic_df, labels, interactions, output_dir=output_dir)

    # ------------------------- Helpers -------------------------

    @staticmethod
    def _cramers_v_from_table(contingency: pd.DataFrame) -> float:
        chi2 = float(chi2_contingency(contingency.values)[0])
        n = float(contingency.sum().sum())
        min_dim = min(contingency.shape) - 1
        if min_dim <= 0 or n <= 0:
            return 0.0
        return float(np.sqrt(chi2 / (n * min_dim)))

    @staticmethod
    def _classify_effect_size(v: float) -> str:
        if v < 0.1:
            return "negligible"
        if v < 0.3:
            return "small"
        if v < 0.5:
            return "medium"
        return "large"

    @staticmethod
    def _interaction_strength(data: pd.DataFrame, labels: pd.Series, f1: str, f2: str) -> float:
        mi1 = float(mutual_info_score(labels, data[f1]))
        mi2 = float(mutual_info_score(labels, data[f2]))
        combined = data[f1].astype(str) + "_" + data[f2].astype(str)
        mi_comb = float(mutual_info_score(labels, combined))
        return float(mi_comb - (mi1 + mi2))
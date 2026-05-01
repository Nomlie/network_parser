# network_parser/statistical_validation_branch.py
"""
Statistical validation branch (NetworkParser).

Updated role in pipeline
------------------------
This module now supports two distinct responsibilities:

1) Central feature filtering (PRE-ML / PRE-tree)
   - per-feature association testing
   - multiple testing correction
   - return a statistically defensible filtered matrix

2) Optional downstream validation utilities
   - bootstrap stability for an already selected feature set
   - permutation testing for interaction pairs

Important architectural rule
----------------------------
Statistical filtering happens BEFORE model screening / tree construction.
Bootstrap confidence/stability is NOT part of the central filtering stage.
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
    Statistical validation suite for NetworkParser.

    Main intended use in the updated pipeline:
      - run_feature_filtering(): central pre-ML / pre-tree filtering

    Optional downstream use:
      - bootstrap_validation(): post-selection stability scoring
      - permutation_test_interactions(): interaction validation
    """

    def __init__(self, config: NetworkParserConfig):
        self.config = config
        self.alpha = float(getattr(self.config, "significance_level", 0.05))
        self.fdr_alpha = float(getattr(self.config, "fdr_alpha", self.alpha))
        self.n_bootstrap = int(getattr(self.config, "n_bootstrap_samples", 1000))
        self.n_permutations = int(getattr(self.config, "n_permutation_tests", 500))
        self.n_jobs = int(getattr(self.config, "n_jobs", -1))

        logger.info("Initialized StatisticalValidatorBranch.")

    # ------------------------------------------------------------------
    # Central pre-ML / pre-tree feature filtering
    # ------------------------------------------------------------------
    def run_feature_filtering(
        self,
        genomic_df: pd.DataFrame,
        labels: pd.Series,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Central statistically defensible feature filtering stage.

        Steps
        -----
        1) Per-feature association testing
        2) Multiple testing correction
        3) Retain significant features
        4) If none survive, return the original matrix as a controlled fallback

        Returns
        -------
        dict with:
          - association
          - multiple_testing
          - retained_features
          - filtered_matrix
          - summary
        """
        self._validate_feature_inputs(genomic_df, labels)

        logger.info(
            "Central feature filtering started | samples=%d | features=%d",
            int(genomic_df.shape[0]),
            int(genomic_df.shape[1]),
        )

        assoc = self.association_tests(
            data=genomic_df,
            labels=labels,
            output_dir=output_dir,
        )

        corrected = self.multiple_testing_correction(
            test_results=assoc,
            output_dir=output_dir,
        )

        retained_features = [
            feature
            for feature, res in corrected.items()
            if bool(res.get("significant", False))
        ]
        retained_features = [f for f in retained_features if f in genomic_df.columns]

        used_fallback = False
        if retained_features:
            filtered_df = genomic_df.loc[:, retained_features].copy()
        else:
            logger.warning(
                "No features survived multiple testing correction. "
                "Using the unfiltered aligned matrix as fallback to avoid an empty downstream matrix."
            )
            filtered_df = genomic_df.copy()
            retained_features = list(filtered_df.columns)
            used_fallback = True

        result = {
            "association": assoc,
            "multiple_testing": corrected,
            "retained_features": retained_features,
            "filtered_matrix": filtered_df,
            "summary": {
                "input_features": int(genomic_df.shape[1]),
                "tested_features": int(len(assoc)),
                "retained_features": int(len(retained_features)),
                "retention_fraction": float(len(retained_features) / max(1, genomic_df.shape[1])),
                "used_fallback_unfiltered_matrix": bool(used_fallback),
            },
        }

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)

            filtered_df.to_csv(out / "filtered_matrix.csv")
            summary_to_write = {
                **result["summary"],
                "retained_feature_names": retained_features,
                "artifacts": {
                    "association_json": str(out / "chi_squared_results.json"),
                    "multiple_testing_json": str(out / "multiple_testing_results.json"),
                    "filtered_matrix_csv": str(out / "filtered_matrix.csv"),
                },
            }
            (out / "feature_filtering_summary.json").write_text(
                json.dumps(summary_to_write, indent=2)
            )

        logger.info(
            "Central feature filtering complete | retained=%d / %d",
            int(filtered_df.shape[1]),
            int(genomic_df.shape[1]),
        )

        return result

    # ------------------------------------------------------------------
    # Per-feature association
    # ------------------------------------------------------------------
    def association_tests(
        self,
        data: pd.DataFrame,
        labels: pd.Series,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        For each feature:
          - choose chi2 or Fisher where appropriate
          - compute effect size (Cramer's V)
          - compute mutual information

        Notes
        -----
        This method is intentionally generic for binary / categorical columns.
        """
        self._validate_feature_inputs(data, labels)

        results: Dict[str, Any] = {}

        def test_feature(feature: str):
            try:
                feature_series = data[feature]

                # Drop rows where label is missing
                valid_mask = ~labels.isna()
                feature_series = feature_series.loc[valid_mask]
                local_labels = labels.loc[valid_mask]

                contingency = pd.crosstab(feature_series, local_labels)

                # Need at least 2x2 support to test association
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    return feature, None

                test_name = self._choose_test(contingency)

                if test_name == "fisher":
                    if contingency.shape == (2, 2):
                        _, p_value = fisher_exact(contingency.values)
                        statistic = None
                        dof = None
                    else:
                        # Fisher is not generally available beyond 2x2 in scipy.
                        # Fallback to chi2 to keep the run operational.
                        statistic, p_value, dof, _ = chi2_contingency(contingency.values)
                        test_name = "chi2"
                else:
                    statistic, p_value, dof, _ = chi2_contingency(contingency.values)

                cramers_v = self._cramers_v_from_table(contingency)
                mi = mutual_info_score(local_labels.values, feature_series.values)

                return feature, {
                    "test": test_name,
                    "statistic": float(statistic) if statistic is not None else None,
                    "p_value": float(p_value),
                    "dof": int(dof) if dof is not None else None,
                    "cramers_v": float(cramers_v),
                    "mutual_info": float(mi),
                    "effect_size_class": self._classify_effect_size(float(cramers_v)),
                    "n_rows": int(contingency.values.sum()),
                    "n_feature_states": int(contingency.shape[0]),
                    "n_label_states": int(contingency.shape[1]),
                    "contingency_table": contingency.to_dict(),
                }
            except Exception as exc:
                logger.warning("Association testing failed for feature '%s': %s", feature, exc)
                return feature, None

        feature_results = Parallel(n_jobs=self.n_jobs)(
            delayed(test_feature)(f) for f in data.columns
        )

        for feature, res in feature_results:
            if res is not None:
                results[feature] = res

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "chi_squared_results.json").write_text(json.dumps(results, indent=2))

        logger.info(
            "Association testing complete | valid_tests=%d / %d",
            int(len(results)),
            int(data.shape[1]),
        )

        return results

    # ------------------------------------------------------------------
    # Multiple testing correction
    # ------------------------------------------------------------------
    def multiple_testing_correction(
        self,
        test_results: Dict[str, Any],
        method: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Apply multiple testing correction across per-feature p-values.
        """
        method = method or str(getattr(self.config, "multiple_testing_method", "fdr_bh"))

        p_values = [res["p_value"] for res in test_results.values() if "p_value" in res]
        features = [f for f, res in test_results.items() if "p_value" in res]

        if not p_values:
            logger.warning("No p-values available for multiple testing correction.")
            return {}

        corrected = multipletests(p_values, alpha=self.fdr_alpha, method=method)

        out: Dict[str, Any] = {}
        for i, feature in enumerate(features):
            out[feature] = {
                **test_results[feature],
                "multiple_testing_method": method,
                "corrected_p_value": float(corrected[1][i]),
                "significant": bool(corrected[0][i]),
            }

        if output_dir:
            p = Path(output_dir)
            p.mkdir(parents=True, exist_ok=True)
            (p / "multiple_testing_results.json").write_text(json.dumps(out, indent=2))

        logger.info(
            "Multiple testing correction complete | significant=%d / %d | method=%s",
            int(sum(1 for r in out.values() if r.get("significant", False))),
            int(len(out)),
            method,
        )

        return out

    # ------------------------------------------------------------------
    # Optional post-selection bootstrap stability
    # ------------------------------------------------------------------
    def bootstrap_validation(
        self,
        data: pd.DataFrame,
        labels: pd.Series,
        features: List[str],
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Bootstrap stability on an already selected feature set.

        This is NOT part of the central filtering stage.
        It is intended for downstream confidence scoring once a feature set
        has already been selected by an interpretable branch.
        """
        self._validate_feature_inputs(data, labels)

        features = [f for f in features if f in data.columns]
        if not features:
            logger.warning("Bootstrap validation received no valid feature names.")
            return {}

        logger.info(
            "Bootstrap validation started | n_bootstrap=%d | features=%d",
            self.n_bootstrap,
            len(features),
        )

        def bootstrap_sample(i: int):
            try:
                boot_idx = np.random.choice(len(data), len(data), replace=True)
                boot_data = data.iloc[boot_idx]
                boot_labels = labels.iloc[boot_idx]

                if boot_labels.nunique() < 2:
                    return None

                dt = DecisionTreeClassifier(
                    max_depth=(getattr(self.config, "max_depth", None) or 5),
                    min_samples_split=max(2, int(getattr(self.config, "min_group_size", 2))),
                    min_samples_leaf=max(1, int(getattr(self.config, "min_group_size", 2)) // 2),
                    random_state=i,
                )
                dt.fit(boot_data[features], boot_labels)

                importances = dict(zip(features, dt.feature_importances_))
                sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                rankings = {f: rank for rank, (f, _) in enumerate(sorted_feats, 1)}
                return importances, rankings
            except Exception:
                return None

        boot = Parallel(n_jobs=self.n_jobs)(
            delayed(bootstrap_sample)(i) for i in range(self.n_bootstrap)
        )
        boot = [b for b in boot if b is not None]

        imp_map = defaultdict(list)
        rank_map = defaultdict(list)

        for importances, rankings in boot:
            for feature in features:
                imp_map[feature].append(float(importances.get(feature, 0.0)))
                rank_map[feature].append(int(rankings.get(feature, len(features) + 1)))

        results: Dict[str, Any] = {}
        half_rank = max(1, len(features) / 2)

        for feature in features:
            imps = np.asarray(imp_map[feature], dtype=float)
            ranks = np.asarray(rank_map[feature], dtype=float)

            stability = float(np.mean(ranks <= half_rank)) if ranks.size else 0.0
            mean_imp = float(np.mean(imps)) if imps.size else 0.0
            ci = (
                tuple(map(float, np.percentile(imps, [2.5, 97.5])))
                if imps.size
                else (0.0, 0.0)
            )

            if imps.size > 1 and not np.allclose(imps, 0.0):
                _, p_value = stats.ttest_1samp(imps, 0.0)
                p_value = float(p_value)
            else:
                p_value = 1.0

            results[feature] = {
                "stability_score": stability,
                "mean_importance": mean_imp,
                "ci": ci,
                "p_value": p_value,
                "significant": bool(p_value < self.alpha),
            }

        if output_dir:
            p = Path(output_dir)
            p.mkdir(parents=True, exist_ok=True)
            (p / "bootstrap_results.json").write_text(json.dumps(results, indent=2))

        logger.info("Bootstrap validation complete | evaluated_features=%d", len(results))
        return results

    # ------------------------------------------------------------------
    # Optional downstream interaction permutation testing
    # ------------------------------------------------------------------
    def permutation_test_interactions(
        self,
        data: pd.DataFrame,
        labels: pd.Series,
        interactions: List[Tuple[str, str]],
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Permutation testing for candidate interaction pairs.
        """
        self._validate_feature_inputs(data, labels)

        valid_interactions = [
            (f1, f2)
            for f1, f2 in interactions
            if f1 in data.columns and f2 in data.columns
        ]

        if not valid_interactions:
            logger.warning("No valid interaction pairs available for permutation testing.")
            return {}

        logger.info(
            "Permutation testing interactions started | n_pairs=%d | n_permutations=%d",
            len(valid_interactions),
            self.n_permutations,
        )

        def test_pair(pair: Tuple[str, str]):
            f1, f2 = pair
            try:
                observed = self._interaction_strength(data, labels, f1, f2)
                perms = []

                for _ in range(self.n_permutations):
                    perm_labels = np.random.permutation(labels)
                    perms.append(self._interaction_strength(data, perm_labels, f1, f2))

                p_value = float(np.mean([p >= observed for p in perms]))
                return pair, {
                    "observed_strength": float(observed),
                    "p_value": p_value,
                    "significant": bool(p_value < self.alpha),
                }
            except Exception as exc:
                logger.warning("Interaction permutation testing failed for %s: %s", pair, exc)
                return pair, None

        raw_out = Parallel(n_jobs=self.n_jobs)(
            delayed(test_pair)(pair) for pair in valid_interactions
        )

        out: Dict[str, Any] = {}
        for pair, res in raw_out:
            if res is not None:
                key = f"{pair[0]}__{pair[1]}"
                out[key] = {
                    "pair": [pair[0], pair[1]],
                    **res,
                }

        if output_dir:
            p = Path(output_dir)
            p.mkdir(parents=True, exist_ok=True)
            (p / "interaction_permutation_results.json").write_text(json.dumps(out, indent=2))

        logger.info("Permutation interaction testing complete | valid_pairs=%d", len(out))
        return out

    # ------------------------------------------------------------------
    # Public wrappers for orchestrator compatibility
    # ------------------------------------------------------------------
    def validate_features(
        self,
        genomic_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        label_column: str,
        discovered_features: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compatibility wrapper.

        Updated behavior:
          - if discovered_features is None:
                run central feature filtering on the full matrix
          - if discovered_features is provided:
                run central feature filtering only on that subset

        No bootstrap is included here anymore.
        """
        if meta_df is None:
            raise ValueError("meta_df is required for validation")
        if label_column not in meta_df.columns:
            raise ValueError(f"label_column not found: {label_column}")

        labels = meta_df[label_column]

        if discovered_features is None:
            data = genomic_df.copy()
        else:
            valid_features = [f for f in discovered_features if f in genomic_df.columns]
            if not valid_features:
                raise ValueError("No discovered_features were found in genomic_df.")
            data = genomic_df.loc[:, valid_features].copy()

        result = self.run_feature_filtering(
            genomic_df=data,
            labels=labels,
            output_dir=output_dir,
        )

        # Preserve legacy-like keys expected elsewhere
        return {
            "association": result["association"],
            "multiple_testing": result["multiple_testing"],
            "retained_features": result["retained_features"],
            "summary": result["summary"],
        }

    def validate_interactions(
        self,
        genomic_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        label_column: str,
        interactions: List[Tuple[str, str]],
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Wrapper for optional downstream interaction validation.
        """
        if meta_df is None:
            raise ValueError("meta_df is required for interaction validation")
        if label_column not in meta_df.columns:
            raise ValueError(f"label_column not found: {label_column}")
        if not interactions:
            return {}

        labels = meta_df[label_column]
        return self.permutation_test_interactions(
            data=genomic_df,
            labels=labels,
            interactions=interactions,
            output_dir=output_dir,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _validate_feature_inputs(
        self,
        data: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not isinstance(labels, pd.Series):
            raise TypeError("labels must be a pandas Series")
        if data.empty:
            raise ValueError("data is empty")
        if labels.empty:
            raise ValueError("labels is empty")

        common = data.index.intersection(labels.index)
        if len(common) == 0:
            raise ValueError("No overlapping sample IDs between data and labels.")

    def _choose_test(self, contingency: pd.DataFrame) -> str:
        """
        Decide whether to use chi2 or fisher according to config and table structure.
        """
        requested = str(getattr(self.config, "statistical_test", "chi2")).lower()

        if requested == "fisher" and contingency.shape == (2, 2):
            return "fisher"

        # Use expected counts rule for 2x2 when chi2 may be unstable
        try:
            _, _, _, expected = chi2_contingency(contingency.values)
            if contingency.shape == (2, 2) and float(expected.min()) < float(
                getattr(self.config, "chi2_min_expected", 5)
            ):
                return "fisher"
        except Exception:
            pass

        return "chi2"

    @staticmethod
    def _cramers_v_from_table(contingency: pd.DataFrame) -> float:
        try:
            chi2 = float(chi2_contingency(contingency.values)[0])
            n = float(contingency.sum().sum())
            min_dim = min(contingency.shape) - 1
            if min_dim <= 0 or n <= 0:
                return 0.0
            return float(np.sqrt(chi2 / (n * min_dim)))
        except Exception:
            return 0.0

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
    def _interaction_strength(
        data: pd.DataFrame,
        labels: pd.Series,
        f1: str,
        f2: str,
    ) -> float:
        """
        Interaction strength proxy:
          MI(label ; joint feature state) - [MI(label ; f1) + MI(label ; f2)]
        """
        mi1 = float(mutual_info_score(labels, data[f1]))
        mi2 = float(mutual_info_score(labels, data[f2]))

        combined = data[f1].astype(str) + "_" + data[f2].astype(str)
        mi_comb = float(mutual_info_score(labels, combined))

        return float(mi_comb - (mi1 + mi2))
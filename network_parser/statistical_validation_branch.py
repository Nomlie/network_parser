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
    from network_parser.feature_selection import (
        rf_fdr_feature_selection as _shared_rf_fdr_feature_selection,
    )
except ImportError:  # pragma: no cover
    from config import NetworkParserConfig  # type: ignore
    from feature_selection import (  # type: ignore
        rf_fdr_feature_selection as _shared_rf_fdr_feature_selection,
    )
try:
    from network_parser.feature_selection import (
        rf_fdr_feature_selection as _rf_fdr_select,
    )
except ImportError:  # pragma: no cover
    try:
        from feature_selection import rf_fdr_feature_selection as _rf_fdr_select  # type: ignore
    except ImportError:  # pragma: no cover
        _rf_fdr_select = None  # type: ignore

logger = logging.getLogger(__name__)

try:
    from network_parser.utils import progress_iter
except ImportError:  # pragma: no cover
    try:
        from utils import progress_iter  # type: ignore
    except ImportError:  # pragma: no cover

        def progress_iter(iterable, **kwargs):  # type: ignore
            return iterable


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
    # RF-FDR central pre-ML / pre-tree feature filtering
    # ------------------------------------------------------------------
    def rf_fdr_feature_selection(
        self,
        genomic_df: pd.DataFrame,
        labels: pd.Series,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        RF-FDR central feature selection.

        This is a thin compatibility wrapper around the shared RF-FDR
        implementation in feature_selection.py.

        Purpose
        -------
        Keep exactly one RF-FDR implementation in the codebase so that
        single-label and hierarchy protocols behave consistently.

        This method is PRE-ML / PRE-tree. It does not perform decision-tree
        construction, interaction mining, or bootstrap confidence scoring.
        """
        self._validate_feature_inputs(genomic_df, labels)

        out_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path("central_feature_filtering")
        )

        result = _shared_rf_fdr_feature_selection(
            X=genomic_df,
            y=labels,
            output_dir=out_dir,
            config=self.config,
            stage_name="central_feature_filtering",
        )

        result.setdefault("method", "rf_fdr")
        return result

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
                "retention_fraction": float(
                    len(retained_features) / max(1, genomic_df.shape[1])
                ),
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
        Per-feature association testing with memory-aware chunking for large matrices.
        """
        self._validate_feature_inputs(data, labels)

        n_features = data.shape[1]
        logger.info(
            "Association testing started | features=%d | n_jobs=%s",
            n_features,
            self.n_jobs,
        )

        # Memory-aware chunking
        chunk_size = max(
            500,
            min(
                2000, n_features // (abs(self.n_jobs) * 2) if self.n_jobs != 1 else 1000
            ),
        )

        def test_feature(feature: str):
            try:
                feature_series = data[feature]
                common_index = feature_series.index.intersection(labels.index)
                feature_series = feature_series.loc[common_index]
                local_labels = labels.loc[common_index]

                # NaN represents a non-callable genotype, not evidence for the
                # baseline state.  Test each feature using only sample/label
                # pairs for which both values are observed.
                valid_mask = ~(feature_series.isna() | local_labels.isna())
                feature_series = feature_series.loc[valid_mask]
                local_labels = local_labels.loc[valid_mask]

                contingency = pd.crosstab(feature_series, local_labels)

                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    return feature, None

                test_name = self._choose_test(contingency)

                if test_name == "fisher" and contingency.shape == (2, 2):
                    _, p_value = fisher_exact(contingency.values)
                    statistic = None
                    dof = None
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
                logger.warning(
                    "Association testing failed for feature '%s': %s", feature, exc
                )
                return feature, None

        features = list(data.columns)
        batch_size = int(
            getattr(self.config, "association_test_batch_size", chunk_size)
        )
        batch_size = max(50, min(batch_size, chunk_size))

        logger.info(
            "Association testing dispatch | features=%d | batch_size=%d",
            len(features),
            int(batch_size),
        )

        results = Parallel(
            n_jobs=self.n_jobs,
            batch_size=int(batch_size),
        )(
            delayed(test_feature)(feature)
            for feature in progress_iter(
                features,
                desc="Association tests",
                unit="feature",
                leave=False,
            )
        )

        # Build final dict
        final_results: Dict[str, Any] = {}
        for feature, res in results:
            if res is not None:
                final_results[feature] = res

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "chi_squared_results.json").write_text(
                json.dumps(final_results, indent=2, default=self._json_default)
            )

        logger.info(
            "Association testing complete | valid_tests=%d / %d | chunk_size=%d",
            len(final_results),
            n_features,
            chunk_size,
        )

        return final_results

    # ------------------------------------------------------------------
    # Chi-square permutation-FDR central feature filtering
    # ------------------------------------------------------------------
    def chi2_permutation_feature_selection(
        self,
        genomic_df: pd.DataFrame,
        labels: pd.Series,
        output_dir: Optional[str] = None,
        stage_name: str = "central_feature_filtering",
    ) -> Dict[str, Any]:
        """
        Chi-square permutation-FDR central feature selection.

        Purpose
        -------
        Provide a faster classical-statistics alternative to RF-FDR while
        preserving the same core idea: compare each feature's observed
        association statistic against a label-permutation null, convert that
        into empirical p-values, and apply multiple-testing correction before
        model screening or decision-tree construction.

        Notes
        -----
        - This is PRE-ML / PRE-tree central filtering.
        - The empirical p-value resolution is 1 / (n_permutations + 1).
        - A feature with zero permutation exceedances receives the minimum
          possible empirical p-value, not p=0.
        """
        self._validate_feature_inputs(genomic_df, labels)

        X = genomic_df.copy()
        y = labels.copy()
        X.index = X.index.astype(str)
        X.columns = X.columns.astype(str)
        y.index = y.index.astype(str)

        common = X.index.intersection(y.index)
        X = X.loc[common].copy()
        y = y.loc[common].copy()

        valid_label_mask = ~y.isna()
        X = X.loc[valid_label_mask].copy()
        y = y.loc[valid_label_mask].copy()

        if X.empty or X.shape[1] == 0:
            raise ValueError(
                f"{stage_name}: empty feature matrix – cannot run chi2 permutation-FDR."
            )
        if y.nunique(dropna=True) < 2:
            raise ValueError(
                f"{stage_name}: chi2 permutation-FDR requires at least two label classes "
                f"(found {y.nunique(dropna=True)})."
            )

        n_permutations = int(getattr(self.config, "n_permutation_tests", 1000))
        if n_permutations < 1:
            raise ValueError("n_permutation_tests must be >= 1 for chi2_perm_fdr.")

        fdr_alpha = float(getattr(self.config, "fdr_alpha", self.fdr_alpha))
        multiple_method = str(getattr(self.config, "multiple_testing_method", "fdr_bh"))
        random_state = int(getattr(self.config, "random_state", 42))
        rng = np.random.default_rng(random_state)

        out_dir = Path(output_dir) if output_dir is not None else Path(stage_name)
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "%s chi2 permutation-FDR started | samples=%d | features=%d | permutations=%d",
            stage_name,
            int(X.shape[0]),
            int(X.shape[1]),
            int(n_permutations),
        )

        results_df = self._chi2_permutation_results(
            X=X,
            y=y,
            n_permutations=n_permutations,
            rng=rng,
            stage_name=stage_name,
        )

        if results_df.empty:
            raise ValueError(f"{stage_name}: no valid chi-square tests were produced.")

        reject, corrected_p, _, _ = multipletests(
            results_df["empirical_p_value"].astype(float).values,
            alpha=fdr_alpha,
            method=multiple_method,
        )

        results_df["multiple_testing_method"] = multiple_method
        results_df["corrected_p_value"] = corrected_p.astype(float)
        results_df["significant"] = reject.astype(bool)

        results_df = results_df.sort_values(
            ["significant", "corrected_p_value", "empirical_p_value", "chi2_statistic"],
            ascending=[False, True, True, False],
        ).reset_index(drop=True)

        significant_features = [
            str(feature)
            for feature in results_df.loc[results_df["significant"], "feature"].tolist()
            if str(feature) in X.columns
        ]

        fallback_strategy = str(
            getattr(self.config, "feature_filter_fallback_strategy", "stop")
        ).lower()
        used_fallback = False

        if significant_features:
            X_filtered = X.loc[:, significant_features].copy()
            retained_feature_names = list(X_filtered.columns)
        elif fallback_strategy == "stop":
            raise ValueError(
                f"{stage_name}: chi2 permutation-FDR retained no significant genomic features. "
                "Stopping is statistically defensible for publication-grade runs. "
                "For exploratory smoke testing only, set "
                "feature_filter_fallback_strategy='unfiltered'."
            )
        elif fallback_strategy == "unfiltered":
            logger.warning(
                "%s chi2 permutation-FDR retained no significant genomic features. "
                "Using the aligned matrix as an exploratory fallback. Do not report "
                "downstream markers from this fallback as FDR-supported discoveries.",
                stage_name,
            )
            X_filtered = X.copy()
            retained_feature_names = list(X_filtered.columns)
            used_fallback = True
        else:
            raise ValueError(
                "feature_filter_fallback_strategy must be one of: 'stop' or 'unfiltered'."
            )

        filtered_matrix_path = out_dir / "filtered_matrix.csv"
        results_csv_path = out_dir / "chi2_permutation_results.csv"
        results_json_path = out_dir / "chi2_permutation_results.json"
        summary_json_path = out_dir / "feature_filtering_summary.json"

        X_filtered.to_csv(filtered_matrix_path)
        results_df.to_csv(results_csv_path, index=False)

        feature_records = results_df.to_dict(orient="records")
        feature_results = {
            str(row["feature"]): {k: v for k, v in row.items() if k != "feature"}
            for row in feature_records
        }
        results_json_path.write_text(
            json.dumps(feature_results, indent=2, default=self._json_default)
        )

        summary = {
            "method": "chi2_perm_fdr",
            "status": "success",
            "stage_name": stage_name,
            "input_features": int(X.shape[1]),
            "tested_features": int(results_df.shape[0]),
            "significant_features": int(len(significant_features)),
            "retained_features": int(X_filtered.shape[1]),
            "fallback_strategy": fallback_strategy,
            "used_fallback_unfiltered_matrix": bool(used_fallback),
            "retention_fraction": float(X_filtered.shape[1] / max(1, X.shape[1])),
            "n_permutations": int(n_permutations),
            "empirical_p_resolution": float(1.0 / (n_permutations + 1.0)),
            "fdr_alpha": float(fdr_alpha),
            "multiple_testing_method": multiple_method,
            "retained_feature_names": retained_feature_names,
            "artifacts": {
                "filter_dir": str(out_dir),
                "chi2_permutation_results_csv": str(results_csv_path),
                "chi2_permutation_results_json": str(results_json_path),
                "filtered_matrix": str(filtered_matrix_path),
                "summary_json": str(summary_json_path),
            },
        }
        summary_json_path.write_text(
            json.dumps(summary, indent=2, default=self._json_default)
        )

        logger.info(
            "%s chi2 permutation-FDR complete | retained_features=%d / %d | empirical_p_resolution=%.6g",
            stage_name,
            int(X_filtered.shape[1]),
            int(X.shape[1]),
            float(str(summary["empirical_p_resolution"])),
        )

        return {
            "method": "chi2_perm_fdr",
            "summary": summary,
            "association": feature_results,
            "multiple_testing": feature_results,
            "retained_features": list(X_filtered.columns),
            "filtered_matrix": X_filtered,
            "feature_results": results_df,
        }

    def _chi2_permutation_results(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_permutations: int,
        rng: np.random.Generator,
        stage_name: str,
    ) -> pd.DataFrame:
        """Dispatch to a binary-matrix fast path when possible."""
        X_numeric = X.apply(pd.to_numeric, errors="coerce")
        numeric_values = X_numeric.to_numpy(dtype=float, copy=False)
        non_missing_values = (
            np.unique(numeric_values[~np.isnan(numeric_values)])
            if numeric_values.size
            else np.array([])
        )
        is_binary_matrix = len(non_missing_values) > 0 and set(
            map(float, non_missing_values)
        ).issubset({0.0, 1.0})

        if is_binary_matrix:
            # Do not fillna(0): non-callable NaNs are not baseline evidence.
            # Use complete cases for the permutation path when any NaNs exist.
            if X_numeric.isna().any().any():
                X_complete = X_numeric.dropna(axis=0, how="any")
                y_complete = y.loc[X_complete.index]
                if X_complete.shape[0] < 4 or y_complete.nunique() < 2:
                    logger.warning(
                        "%s chi2 permutation-FDR: insufficient complete cases after "
                        "dropping non-callable genotypes; using feature-wise complete pairs.",
                        stage_name,
                    )
                    # Fall back to generic path which can handle per-feature missingness.
                    is_binary_matrix = False
                else:
                    return self._chi2_permutation_binary_results(
                        X_complete.to_numpy(dtype=float),
                        y_complete,
                        n_permutations,
                        rng,
                        stage_name,
                    )
            if is_binary_matrix:
                return self._chi2_permutation_binary_results(
                    X_numeric.to_numpy(dtype=float),
                    y,
                    n_permutations,
                    rng,
                    stage_name,
                )

        logger.info(
            "%s chi2 permutation-FDR using generic categorical path. "
            "For binary matrices, numeric 0/1 encoding is faster.",
            stage_name,
        )
        return self._chi2_permutation_generic_results(X, y, n_permutations, rng)

    def _chi2_permutation_binary_results(
        self,
        X_binary: pd.DataFrame,
        y: pd.Series,
        n_permutations: int,
        rng: np.random.Generator,
        stage_name: str,
    ) -> pd.DataFrame:
        """Vectorized chi-square permutation test for numeric 0/1 matrices."""
        X_arr = X_binary.to_numpy(dtype=np.float64, copy=True)
        feature_names = list(map(str, X_binary.columns))

        y_codes, y_levels = pd.factorize(y.astype(str), sort=True)
        y_codes = np.asarray(y_codes, dtype=np.int64)
        n_classes = int(len(y_levels))
        n_samples = int(X_arr.shape[0])

        class_counts = np.bincount(y_codes, minlength=n_classes).astype(np.float64)

        def counts1_for_codes(codes: np.ndarray) -> np.ndarray:
            out = np.empty((X_arr.shape[1], n_classes), dtype=np.float64)
            for k in range(n_classes):
                mask = codes == k
                if np.any(mask):
                    out[:, k] = X_arr[mask, :].sum(axis=0)
                else:
                    out[:, k] = 0.0
            return out

        def chi2_stats_from_counts1(
            counts1: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
            row1 = counts1.sum(axis=1)
            row0 = float(n_samples) - row1
            counts0 = class_counts.reshape(1, -1) - counts1

            expected1 = (
                row1.reshape(-1, 1) * class_counts.reshape(1, -1) / float(n_samples)
            )
            expected0 = (
                row0.reshape(-1, 1) * class_counts.reshape(1, -1) / float(n_samples)
            )

            stat = np.zeros(counts1.shape[0], dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                part1 = np.where(
                    expected1 > 0, ((counts1 - expected1) ** 2) / expected1, 0.0
                )
                part0 = np.where(
                    expected0 > 0, ((counts0 - expected0) ** 2) / expected0, 0.0
                )
            stat = part1.sum(axis=1) + part0.sum(axis=1)

            valid = (row0 > 0) & (row1 > 0) & (n_classes >= 2)
            stat[~valid] = 0.0
            return stat, valid

        observed_counts1 = counts1_for_codes(y_codes)
        observed_stat, valid = chi2_stats_from_counts1(observed_counts1)
        dof = max(1, n_classes - 1)
        asymptotic_p = stats.chi2.sf(observed_stat, dof)
        asymptotic_p[~valid] = 1.0

        exceedances = np.zeros(X_arr.shape[1], dtype=np.int64)

        for perm_idx in progress_iter(
            range(n_permutations),
            desc=f"{stage_name} chi2 permutations",
            unit="perm",
            leave=False,
        ):
            permuted_codes = rng.permutation(y_codes)
            perm_counts1 = counts1_for_codes(permuted_codes)
            perm_stat, _ = chi2_stats_from_counts1(perm_counts1)
            exceedances += (perm_stat >= observed_stat).astype(np.int64)

        empirical_p = (1.0 + exceedances.astype(float)) / float(n_permutations + 1)
        empirical_p[~valid] = 1.0

        # For a 2 x K table, min_dim is 1 when K >= 2.
        cramers_v = np.sqrt(np.maximum(observed_stat, 0.0) / max(1.0, float(n_samples)))
        cramers_v[~valid] = 0.0

        mutual_info = [
            float(mutual_info_score(y_codes, X_arr[:, j].astype(int)))
            for j in range(X_arr.shape[1])
        ]

        return pd.DataFrame(
            {
                "feature": feature_names,
                "test": "chi2_permutation",
                "chi2_statistic": observed_stat.astype(float),
                "asymptotic_p_value": asymptotic_p.astype(float),
                "empirical_p_value": empirical_p.astype(float),
                "permutation_exceedances": exceedances.astype(int),
                "n_permutations": int(n_permutations),
                "empirical_p_resolution": float(1.0 / (n_permutations + 1.0)),
                "dof": int(dof),
                "cramers_v": cramers_v.astype(float),
                "mutual_info": mutual_info,
                "n_rows": int(n_samples),
                "n_feature_states": np.where(valid, 2, 1).astype(int),
                "n_label_states": int(n_classes),
            }
        )

    def _chi2_permutation_generic_results(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_permutations: int,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Generic categorical chi-square permutation test, feature by feature."""
        feature_names = list(map(str, X.columns))
        base_seeds = rng.integers(
            0, 2**31 - 1, size=len(feature_names), dtype=np.int64
        )

        def chi2_stat_from_table(table: np.ndarray) -> float:
            table = np.asarray(table, dtype=np.float64)
            n = float(table.sum())
            if n <= 0 or min(table.shape) < 2:
                return 0.0
            row_sum = table.sum(axis=1, keepdims=True)
            col_sum = table.sum(axis=0, keepdims=True)
            expected = row_sum @ col_sum / n
            with np.errstate(divide="ignore", invalid="ignore"):
                stat = np.where(
                    expected > 0, ((table - expected) ** 2) / expected, 0.0
                ).sum()
            return float(stat)

        def process_feature(feature: str, seed: int) -> Optional[Dict[str, Any]]:
            series = X[feature]
            valid = ~(series.isna() | y.isna())
            if int(valid.sum()) == 0:
                return None

            x_values = series.loc[valid].astype(str)
            y_values = y.loc[valid].astype(str)
            x_codes, x_levels = pd.factorize(x_values, sort=True)
            y_codes, y_levels = pd.factorize(y_values, sort=True)

            n_feature_states = int(len(x_levels))
            n_label_states = int(len(y_levels))
            if n_feature_states < 2 or n_label_states < 2:
                return None

            table = np.zeros((n_feature_states, n_label_states), dtype=np.float64)
            np.add.at(table, (x_codes, y_codes), 1.0)
            observed_stat = chi2_stat_from_table(table)
            dof = int((n_feature_states - 1) * (n_label_states - 1))
            asymptotic_p = float(stats.chi2.sf(observed_stat, max(1, dof)))

            local_rng = np.random.default_rng(int(seed))
            exceedances = 0
            for _ in range(n_permutations):
                perm_y = local_rng.permutation(y_codes)
                perm_table = np.zeros_like(table)
                np.add.at(perm_table, (x_codes, perm_y), 1.0)
                perm_stat = chi2_stat_from_table(perm_table)
                if perm_stat >= observed_stat:
                    exceedances += 1

            empirical_p = float((1.0 + exceedances) / (n_permutations + 1.0))
            cramers_v = self._cramers_v_from_table(pd.DataFrame(table))
            mi = mutual_info_score(y_codes, x_codes)

            return {
                "feature": str(feature),
                "test": "chi2_permutation",
                "chi2_statistic": float(observed_stat),
                "asymptotic_p_value": asymptotic_p,
                "empirical_p_value": empirical_p,
                "permutation_exceedances": int(exceedances),
                "n_permutations": int(n_permutations),
                "empirical_p_resolution": float(1.0 / (n_permutations + 1.0)),
                "dof": int(dof),
                "cramers_v": float(cramers_v),
                "mutual_info": float(mi),
                "n_rows": int(valid.sum()),
                "n_feature_states": n_feature_states,
                "n_label_states": n_label_states,
            }

        rows = Parallel(n_jobs=self.n_jobs)(
            delayed(process_feature)(feature, int(seed))
            for feature, seed in progress_iter(
                zip(feature_names, base_seeds),
                total=len(feature_names),
                desc="Chi2 permutation features",
                unit="feature",
                leave=False,
            )
        )
        return pd.DataFrame([row for row in rows if row is not None])

    @staticmethod
    def _json_default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Series, pd.Index)):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)

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
        method = method or str(
            getattr(self.config, "multiple_testing_method", "fdr_bh")
        )

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
                    min_samples_split=max(
                        2, int(getattr(self.config, "min_group_size", 2))
                    ),
                    min_samples_leaf=max(
                        1, int(getattr(self.config, "min_group_size", 2)) // 2
                    ),
                    random_state=i,
                )
                dt.fit(boot_data[features], boot_labels)

                importances = dict(zip(features, dt.feature_importances_))
                sorted_feats = sorted(
                    importances.items(), key=lambda x: x[1], reverse=True
                )
                rankings = {f: rank for rank, (f, _) in enumerate(sorted_feats, 1)}
                return importances, rankings
            except Exception:
                return None

        boot = Parallel(n_jobs=self.n_jobs)(
            delayed(bootstrap_sample)(i)
            for i in progress_iter(
                range(self.n_bootstrap),
                desc="Bootstrap validation",
                unit="sample",
                leave=False,
            )
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
            if imps.size:
                percentiles = np.asarray(
                    np.percentile(imps, [2.5, 97.5]), dtype=float
                ).ravel()
                ci = (float(percentiles[0]), float(percentiles[1]))
            else:
                ci = (0.0, 0.0)

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

        logger.info(
            "Bootstrap validation complete | evaluated_features=%d", len(results)
        )
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
            logger.warning(
                "No valid interaction pairs available for permutation testing."
            )
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
                logger.warning(
                    "Interaction permutation testing failed for %s: %s", pair, exc
                )
                return pair, None

        raw_out = Parallel(n_jobs=self.n_jobs)(
            delayed(test_pair)(pair)
            for pair in progress_iter(
                valid_interactions,
                desc="Interaction permutation tests",
                unit="pair",
                leave=False,
            )
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
            (p / "interaction_permutation_results.json").write_text(
                json.dumps(out, indent=2)
            )

        logger.info(
            "Permutation interaction testing complete | valid_pairs=%d", len(out)
        )
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
        first = pd.to_numeric(data[f1], errors="coerce").reset_index(drop=True)
        second = pd.to_numeric(data[f2], errors="coerce").reset_index(drop=True)
        target = pd.Series(labels).reset_index(drop=True)
        valid = first.notna() & second.notna() & target.notna()
        if int(valid.sum()) < 2:
            return 0.0

        first = first.loc[valid]
        second = second.loc[valid]
        target = target.loc[valid]
        mi1 = float(mutual_info_score(target, first))
        mi2 = float(mutual_info_score(target, second))

        combined = first.astype(str) + "_" + second.astype(str)
        mi_comb = float(mutual_info_score(target, combined))

        return float(mi_comb - (mi1 + mi2))

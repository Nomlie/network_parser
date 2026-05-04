#!/usr/bin/env python3
# network_parser/feature_selection.py
"""
Central RF‑FDR feature selection for NetworkParser.

This module provides the shared RF‑permutation/FDR feature‑filtering stage
used by both single‑label and two‑level protocols.  It is the **only**
implementation; all other copies should be removed.

Expected input
--------------
- Aligned, preprocessed genomic feature matrix (`X`)
- Supervised label series (`y`)
- Output directory for artifacts
- NetworkParserConfig object (or any object exposing the required attributes)

Configurable controls
---------------------
All settings are read from `config` with sensible fallbacks:

    rf_selector_n_estimators : int = 300
    rf_selector_n_observed_repeats : int = 10
    rf_selector_n_permutations : int = 100
    rf_selector_fdr_alpha : float = 0.05
    rf_selector_max_features : str = "sqrt"
    rf_selector_min_samples_leaf : int = 1
    rf_selector_class_weight : str or None = "balanced"
    rf_selector_min_importance : float = 0.0
    rf_selector_top_n : int or None = None
    rf_selector_random_state : int = 42
    rf_selector_fallback_strategy : "stop" | "top_n" | "unfiltered" = "stop"
    rf_selector_fallback_top_n : int = 500

Output
------
A dictionary containing:

    - filtered_matrix        : pd.DataFrame (samples × retained features)
    - retained_features      : list of feature names
    - feature_results        : pd.DataFrame with per‑feature statistics
    - summary                : dict with stage summary and artifact paths
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local utility helpers – kept simple to avoid cross‑module dependencies
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(obj: Any) -> Any:
    """Serialise numpy ints, floats, arrays, paths."""
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


def write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)


# =====================================================================
# Core RF‑FDR feature selection
# =====================================================================

def rf_fdr_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    config: Any,
    stage_name: str,
) -> Dict[str, Any]:
    """
    Random Forest permutation‑FDR feature selection.

    Workflow
    --------
    1. Align X and y on shared sample index.
    2. Estimate observed RF importance across repeated fits.
    3. Build null importance distribution via label permutations.
    4. Compute empirical p‑values and apply Benjamini‑Hochberg FDR.
    5. Retain features based on significance, minimum importance, and optional top‑N cap.
    6. If no features survive, apply the configured fallback strategy.

    Returns
    -------
    dict
        {
            "filtered_matrix": pd.DataFrame,
            "retained_features": List[str],
            "feature_results": pd.DataFrame,
            "summary": dict
        }
    """
    # ------------------------------------------------------------------
    # 0. Sanity checks & alignment
    # ------------------------------------------------------------------
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    X = X.copy()
    y = y.copy()
    X.index = X.index.astype(str)
    y.index = y.index.astype(str)

    common = X.index.intersection(y.index)
    X = X.loc[common].copy()
    y = y.loc[common].copy()

    if X.empty or X.shape[1] == 0:
        raise ValueError(f"{stage_name}: empty feature matrix – cannot run RF‑FDR.")
    if y.nunique(dropna=True) < 2:
        raise ValueError(f"{stage_name}: RF‑FDR requires at least two label classes (found {y.nunique(dropna=True)}).")

    # ------------------------------------------------------------------
    # 1. Extract runtime configuration
    # ------------------------------------------------------------------
    rng = np.random.default_rng(int(getattr(config, "rf_selector_random_state", 42)))

    n_estimators        = int(getattr(config, "rf_selector_n_estimators", 300))
    n_observed_repeats  = int(getattr(config, "rf_selector_n_observed_repeats", 10))
    n_permutations      = int(getattr(config, "rf_selector_n_permutations", 100))
    fdr_alpha           = float(getattr(config, "rf_selector_fdr_alpha", 0.05))
    max_features        = getattr(config, "rf_selector_max_features", "sqrt")
    min_samples_leaf    = int(getattr(config, "rf_selector_min_samples_leaf", 1))
    class_weight        = getattr(config, "rf_selector_class_weight", "balanced")
    min_importance      = float(getattr(config, "rf_selector_min_importance", 0.0))
    top_n               = getattr(config, "rf_selector_top_n", None)
    n_jobs              = int(getattr(config, "n_jobs", -1))

    fallback_strategy   = str(getattr(config, "rf_selector_fallback_strategy", "stop")).lower()
    fallback_top_n      = int(getattr(config, "rf_selector_fallback_top_n", 500))

    feature_names = list(X.columns)

    logger.info(
        "%s RF‑FDR started | samples=%d | features=%d | observed_repeats=%d | permutations=%d",
        stage_name,
        int(X.shape[0]),
        int(X.shape[1]),
        int(n_observed_repeats),
        int(n_permutations),
    )

    # ------------------------------------------------------------------
    # 2. Observed importance estimation
    # ------------------------------------------------------------------
    def fit_importance(local_y: pd.Series, seed: int) -> np.ndarray:
        """Fit a single RF and return feature importances."""
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=seed,
            n_jobs=n_jobs,
        )
        model.fit(X, local_y.astype(str))
        return np.asarray(model.feature_importances_, dtype=float)

    observed_list: List[np.ndarray] = []
    for _ in range(max(1, n_observed_repeats)):
        seed = int(rng.integers(0, 2**31 - 1))
        observed_list.append(fit_importance(y, seed))

    observed_matrix = np.vstack(observed_list)
    observed_mean = observed_matrix.mean(axis=0)
    observed_std  = observed_matrix.std(axis=0)

    # ------------------------------------------------------------------
    # 3. Null importance via label permutation
    # ------------------------------------------------------------------
    null_sum              = np.zeros(len(feature_names), dtype=float)
    null_sum_sq           = np.zeros(len(feature_names), dtype=float)
    null_exceedance_counts = np.zeros(len(feature_names), dtype=int)

    total_permutations = max(1, n_permutations)
    progress_step = max(1, total_permutations // 10)

    for perm_idx in range(total_permutations):
        permuted_y = pd.Series(
            rng.permutation(y.values),
            index=y.index,
            name=y.name,
        )
        seed = int(rng.integers(0, 2**31 - 1))
        null_importance = fit_importance(permuted_y, seed)

        null_sum              += null_importance
        null_sum_sq           += null_importance ** 2
        null_exceedance_counts += (null_importance >= observed_mean).astype(int)

        if (perm_idx + 1) % progress_step == 0 or (perm_idx + 1) == total_permutations:
            logger.info(
                "%s RF‑FDR permutation progress | %d / %d",
                stage_name,
                int(perm_idx + 1),
                int(total_permutations),
            )

    # ------------------------------------------------------------------
    # 4. Empirical p‑values and FDR correction
    # ------------------------------------------------------------------
    denominator = float(total_permutations + 1)
    empirical_p_resolution = 1.0 / denominator

    empirical_p = (1.0 + null_exceedance_counts.astype(float)) / denominator

    null_mean     = null_sum / float(total_permutations)
    null_variance = (null_sum_sq / float(total_permutations)) - (null_mean ** 2)
    null_variance = np.maximum(null_variance, 0.0)
    null_std      = np.sqrt(null_variance)

    reject, corrected_p, _, _ = multipletests(
        empirical_p, alpha=fdr_alpha, method="fdr_bh"
    )

    results_df = pd.DataFrame(
        {
            "feature":                   feature_names,
            "rf_mean_importance":        observed_mean,
            "rf_std_importance":         observed_std,
            "null_mean_importance":      null_mean,
            "null_std_importance":       null_std,
            "null_exceedance_count":     null_exceedance_counts.astype(int),
            "n_permutations":            int(total_permutations),
            "empirical_p_resolution":    float(empirical_p_resolution),
            "empirical_p_value":         empirical_p,
            "corrected_p_value":         corrected_p,
            "significant":               reject.astype(bool),
        }
    ).sort_values(
        ["corrected_p_value", "rf_mean_importance"],
        ascending=[True, False],
    ).reset_index(drop=True)

    # Retain significant features above the minimum importance
    retained = results_df[
        (results_df["significant"])
        & (results_df["rf_mean_importance"] > min_importance)
    ].copy()

    if top_n is not None:
        retained = retained.head(int(top_n)).copy()

    retained_features = [
        f for f in retained["feature"].tolist() if f in X.columns
    ]

    # ------------------------------------------------------------------
    # 5. Fallback handling
    # ------------------------------------------------------------------
    used_fallback = False
    used_top_n_fallback = False
    used_unfiltered_fallback = False
    selection_mode = "strict_rf_fdr"

    if retained_features:
        X_filtered = X.loc[:, retained_features].copy()
    else:
        used_fallback = True

        if fallback_strategy == "top_n":
            fallback_n = min(fallback_top_n, len(results_df))
            retained = results_df.head(fallback_n).copy()
            retained_features = [f for f in retained["feature"].tolist() if f in X.columns]

            X_filtered = X.loc[:, retained_features].copy()
            used_top_n_fallback = True
            selection_mode = "rf_top_n_fallback"

            logger.warning(
                "%s RF‑FDR retained no features at FDR alpha %.4f. "
                "Using top‑%d RF‑ranked features as exploratory fallback.",
                stage_name,
                float(fdr_alpha),
                int(len(retained_features)),
            )

        elif fallback_strategy == "unfiltered":
            retained = results_df.copy()
            retained_features = feature_names
            X_filtered = X.copy()
            used_unfiltered_fallback = True
            selection_mode = "unfiltered_fallback"

            logger.warning(
                "%s RF‑FDR retained no features at FDR alpha %.4f. "
                "Using the aligned matrix as fallback.",
                stage_name,
                float(fdr_alpha),
            )

        elif fallback_strategy == "stop":
            raise ValueError(
                f"{stage_name}: RF‑FDR retained no features at FDR alpha {fdr_alpha}. "
                "Increase rf_selector_n_permutations, relax rf_selector_fdr_alpha, "
                "or set rf_selector_fallback_strategy='top_n' only for exploratory testing."
            )

        else:
            raise ValueError(
                "rf_selector_fallback_strategy must be one of: "
                "'top_n', 'unfiltered', or 'stop'"
            )

    # ------------------------------------------------------------------
    # 6. Write artifacts
    # ------------------------------------------------------------------
    out = ensure_dir(output_dir)

    results_path   = out / "rf_fdr_feature_results.csv"
    retained_path  = out / "rf_fdr_retained_features.csv"
    matrix_path    = out / "filtered_matrix.csv"
    summary_path   = out / "feature_filtering_summary.json"

    results_df.to_csv(results_path, index=False)
    retained.to_csv(retained_path, index=False)
    X_filtered.to_csv(matrix_path)

    summary = {
        "stage":                          stage_name,
        "method":                         "rf_fdr",
        "status":                         "success",
        "input_features":                 int(X.shape[1]),
        "tested_features":                int(len(feature_names)),
        "retained_features":              int(len(retained_features)),
        "retention_fraction":             float(len(retained_features) / max(1, X.shape[1])),
        "selection_mode":                 selection_mode,
        "used_fallback":                  bool(used_fallback),
        "used_fallback_top_n":            bool(used_top_n_fallback),
        "used_fallback_unfiltered_matrix":bool(used_unfiltered_fallback),
        "fallback_strategy":              fallback_strategy,
        "fallback_top_n":                 int(fallback_top_n),
        "min_empirical_p_value":          float(results_df["empirical_p_value"].min()),
        "min_corrected_p_value":          float(results_df["corrected_p_value"].min()),
        "unique_empirical_p_values":      int(results_df["empirical_p_value"].nunique()),
        "unique_corrected_p_values":      int(results_df["corrected_p_value"].nunique()),
        "fdr_alpha":                      float(fdr_alpha),
        "n_observed_repeats":             int(n_observed_repeats),
        "n_permutations":                 int(n_permutations),
        "min_importance":                 float(min_importance),
        "top_n":                          int(top_n) if top_n is not None else None,
        "retained_feature_names":         retained_features,
        "artifacts": {
            "rf_fdr_results_csv":           str(results_path),
            "rf_fdr_retained_features_csv": str(retained_path),
            "filtered_matrix_csv":          str(matrix_path),
            "summary_json":                 str(summary_path),
        },
    }

    write_json(summary, summary_path)

    logger.info(
        "%s RF‑FDR complete | retained_features=%d / %d",
        stage_name,
        int(X_filtered.shape[1]),
        int(X.shape[1]),
    )

    return {
        "filtered_matrix":   X_filtered,
        "retained_features": retained_features,
        "feature_results":   results_df,
        "summary":           summary,
    }
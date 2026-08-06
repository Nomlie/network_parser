#!/usr/bin/env python3
# network_parser/feature_selection.py
"""
Central RF‑FDR feature selection for NetworkParser.

This module provides the shared RF‑permutation/FDR feature‑filtering stage
used by both single-label and hierarchy protocols.  It is the **only**
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
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

try:
    from network_parser.utils import progress_iter
except ImportError:  # pragma: no cover
    try:
        from utils import progress_iter  # type: ignore
    except ImportError:  # pragma: no cover

        def progress_iter(iterable, **kwargs):  # type: ignore
            return iterable


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


def _resolve_parallel_plan(
    *,
    config: Any,
    default_n_jobs: int,
    n_tasks: int,
) -> tuple[int, int, bool]:
    """Return (outer_n_jobs, inner_rf_n_jobs, use_outer_parallel).

    RF-FDR has two possible parallel layers: independent RF fits and trees
    inside each RF.  Running both with all cores causes oversubscription.  The
    default plan therefore parallelises the independent observed/permutation
    fits and keeps each RandomForest fit single-threaded.
    """
    outer = getattr(config, "rf_selector_outer_n_jobs", None)
    inner = getattr(config, "rf_selector_inner_n_jobs", None)

    outer_n_jobs = int(default_n_jobs if outer is None else outer)
    if outer_n_jobs == 0:
        outer_n_jobs = 1

    use_outer_parallel = n_tasks > 1 and outer_n_jobs != 1

    if inner is not None:
        inner_rf_n_jobs = int(inner)
    elif use_outer_parallel:
        inner_rf_n_jobs = 1
    else:
        inner_rf_n_jobs = int(default_n_jobs)

    if inner_rf_n_jobs == 0:
        inner_rf_n_jobs = 1

    return outer_n_jobs, inner_rf_n_jobs, use_outer_parallel


def _parallel_importance_map(
    tasks: Sequence[tuple[str, int, Optional[np.ndarray]]], func, n_jobs: int
) -> List[np.ndarray]:
    """Run RF-importance tasks with a shared-memory threading backend."""
    if len(tasks) <= 1 or int(n_jobs) == 1:
        return [func(kind, seed, values) for kind, seed, values in tasks]

    return Parallel(n_jobs=int(n_jobs), prefer="threads")(
        delayed(func)(kind, seed, values) for kind, seed, values in tasks
    )


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

    # Drop missing target labels *before* string conversion so "nan" is not a class.
    missing_label_mask = y.isna() | y.astype(str).str.strip().isin(
        {"", "nan", "NaN", "None", "NA", "N/A", "-", "."}
    )
    n_missing_labels = int(missing_label_mask.sum())
    if n_missing_labels:
        logger.warning(
            "%s RF‑FDR dropped %d samples with missing/invalid target labels before selection.",
            stage_name,
            n_missing_labels,
        )
        y = y.loc[~missing_label_mask].copy()

    common = X.index.intersection(y.index)
    X = X.loc[common].copy()
    y = y.loc[common].copy()

    if X.empty or X.shape[1] == 0:
        raise ValueError(f"{stage_name}: empty feature matrix – cannot run RF‑FDR.")
    if y.nunique(dropna=True) < 2:
        raise ValueError(
            f"{stage_name}: RF‑FDR requires at least two label classes (found {y.nunique(dropna=True)})."
        )

    # Matrix contract: RF cannot accept NaN. Fit missingness on this matrix
    # (fold-train only when called from CV) and impute explicitly.
    try:
        from network_parser.matrix_contract import (
            MissingnessPolicy,
            prepare_for_sklearn,
        )
    except ImportError:  # pragma: no cover
        from matrix_contract import MissingnessPolicy, prepare_for_sklearn  # type: ignore

    policy = MissingnessPolicy.from_config(config)
    X, miss_state, miss_audit = prepare_for_sklearn(X, policy=policy)
    y = y.loc[X.index].copy()
    if y.nunique(dropna=True) < 2:
        raise ValueError(
            f"{stage_name}: RF‑FDR requires at least two label classes after missingness filtering."
        )
    logger.info(
        "%s RF‑FDR missingness prep | samples=%d | features=%d | strategy=%s | audit=%s",
        stage_name,
        int(X.shape[0]),
        int(X.shape[1]),
        policy.impute_strategy,
        {
            k: miss_audit.get(k)
            for k in ("n_dropped_features", "n_dropped_samples", "impute_strategy")
        },
    )

    # ------------------------------------------------------------------
    # 1. Extract runtime configuration
    # ------------------------------------------------------------------
    rng = np.random.default_rng(int(getattr(config, "rf_selector_random_state", 42)))

    n_estimators = int(getattr(config, "rf_selector_n_estimators", 300))
    n_observed_repeats = int(getattr(config, "rf_selector_n_observed_repeats", 10))
    n_permutations = int(getattr(config, "rf_selector_n_permutations", 100))
    fdr_alpha = float(getattr(config, "rf_selector_fdr_alpha", 0.05))
    max_features = getattr(config, "rf_selector_max_features", "sqrt")
    min_samples_leaf = int(getattr(config, "rf_selector_min_samples_leaf", 1))
    class_weight = getattr(config, "rf_selector_class_weight", "balanced")
    min_importance = float(getattr(config, "rf_selector_min_importance", 0.0))
    top_n = getattr(config, "rf_selector_top_n", None)
    n_jobs = int(getattr(config, "n_jobs", -1))
    perm_resolution_policy = str(
        getattr(config, "rf_selector_permutation_resolution_policy", "warn")
    ).lower()

    fallback_strategy = str(
        getattr(config, "rf_selector_fallback_strategy", "stop")
    ).lower()
    fallback_top_n = int(getattr(config, "rf_selector_fallback_top_n", 500))

    # Optional group blocks for permutations (population structure)
    group_values: Optional[np.ndarray] = None
    group_col = getattr(config, "cv_group_column", None) or getattr(
        config, "rf_selector_permutation_group_column", None
    )
    groups_arg = getattr(config, "_rf_fdr_groups", None)  # optional in-memory Series
    if groups_arg is not None:
        gser = pd.Series(groups_arg).astype(str)
        gser.index = gser.index.astype(str)
        gser = gser.loc[gser.index.intersection(X.index)]
        if len(gser) == len(X):
            group_values = gser.loc[X.index].to_numpy()
    elif group_col and hasattr(config, "_rf_fdr_group_series"):
        pass  # reserved

    feature_names = list(X.columns)

    if int(n_permutations) < 1:
        raise ValueError(
            f"{stage_name}: rf_selector_n_permutations must be >= 1 "
            f"(got {n_permutations}). Refusing to silently convert zero permutations to one."
        )

    # Minimum empirical p = 1/(B+1). BH-FDR requires resolution fine enough that
    # the smallest attainable corrected p can pass alpha when many tests exist.
    n_tests = max(1, len(feature_names))
    min_empirical_p = 1.0 / float(int(n_permutations) + 1)
    # Conservative: need min_p * n_tests roughly <= alpha for at least one rejection possible
    # Conservative attainable BH bound when only the smallest p-value is at
    # the permutation floor: p_min * m. This prevents a run from claiming a
    # meaningful FDR screen when its permutation grid cannot possibly reject.
    minimum_attainable_bh_adjusted_p = min(1.0, min_empirical_p * n_tests)
    resolution_ok = minimum_attainable_bh_adjusted_p <= fdr_alpha + 1e-15
    resolution_msg = (
        f"permutation_resolution: B={n_permutations} => min_empirical_p={min_empirical_p:.6g}; "
        f"fdr_alpha={fdr_alpha}; n_tests={n_tests}; "
        f"min_attainable_bh_adjusted_p={minimum_attainable_bh_adjusted_p:.6g}; "
        f"{'adequate' if resolution_ok else 'INADEQUATE for intended BH-FDR'}"
    )
    if not resolution_ok:
        if perm_resolution_policy == "fail":
            raise ValueError(
                f"{stage_name}: {resolution_msg}. "
                "Increase rf_selector_n_permutations or relax rf_selector_fdr_alpha."
            )
        logger.warning("%s RF‑FDR %s", stage_name, resolution_msg)

    total_fits = max(1, n_observed_repeats) + int(n_permutations)
    outer_n_jobs, inner_rf_n_jobs, use_outer_parallel = _resolve_parallel_plan(
        config=config,
        default_n_jobs=n_jobs,
        n_tasks=total_fits,
    )

    logger.info(
        "%s RF‑FDR started | samples=%d | features=%d | observed_repeats=%d | "
        "permutations=%d | outer_n_jobs=%s | inner_rf_n_jobs=%s | group_blocked_perm=%s",
        stage_name,
        int(X.shape[0]),
        int(X.shape[1]),
        int(n_observed_repeats),
        int(n_permutations),
        str(outer_n_jobs),
        str(inner_rf_n_jobs),
        bool(group_values is not None),
    )

    # Labels already cleaned; now safe to stringify for RF
    y_values = y.astype(str).str.strip().to_numpy(copy=True)

    if group_values is not None:
        group_sizes = pd.Series(group_values).value_counts()
        permutable_size_counts = group_sizes.value_counts()
        if not any(int(count) >= 2 for count in permutable_size_counts.tolist()):
            raise ValueError(
                f"{stage_name}: group-aware permutation requires at least two groups "
                "with the same sample count so whole related-sample blocks can be exchanged."
            )

    def _permute_labels(seed: int) -> np.ndarray:
        local_rng = np.random.default_rng(int(seed))
        if group_values is None:
            return local_rng.permutation(y_values)
        # Group-aware permutation: move entire within-group label vectors only
        # between groups of equal size. This preserves related-sample blocks,
        # label counts and within-group correlation while breaking the mapping
        # between genomic rows and group-level outcomes.
        fit_y = y_values.copy()
        groups_by_size: Dict[int, List[np.ndarray]] = {}
        for group in np.unique(group_values):
            idx = np.where(group_values == group)[0]
            groups_by_size.setdefault(int(len(idx)), []).append(idx)
        for blocks in groups_by_size.values():
            if len(blocks) <= 1:
                continue
            source_order = local_rng.permutation(len(blocks))
            for target_idx, source_position in zip(blocks, source_order):
                fit_y[target_idx] = y_values[blocks[int(source_position)]]
        return fit_y

    # ------------------------------------------------------------------
    # 2. Observed importance estimation
    # ------------------------------------------------------------------
    def fit_importance(
        kind: str, seed: int, local_y_values: Optional[np.ndarray]
    ) -> np.ndarray:
        """Fit a single RF and return feature importances."""
        if kind == "permuted":
            fit_y = _permute_labels(int(seed))
        else:
            fit_y = y_values if local_y_values is None else local_y_values

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=int(seed),
            n_jobs=inner_rf_n_jobs,
        )
        model.fit(X, fit_y)
        return np.asarray(model.feature_importances_, dtype=float)

    observed_seeds = [
        int(rng.integers(0, 2**31 - 1)) for _ in range(max(1, n_observed_repeats))
    ]
    observed_tasks = [("observed", seed, y_values) for seed in observed_seeds]
    observed_list = _parallel_importance_map(
        tasks=observed_tasks,
        func=fit_importance,
        n_jobs=outer_n_jobs if use_outer_parallel else 1,
    )

    observed_matrix = np.vstack(observed_list)
    observed_mean = observed_matrix.mean(axis=0)
    observed_std = observed_matrix.std(axis=0)

    # ------------------------------------------------------------------
    # 3. Null importance via label permutation (group-blocked when available)
    # ------------------------------------------------------------------
    null_sum = np.zeros(len(feature_names), dtype=float)
    null_sum_sq = np.zeros(len(feature_names), dtype=float)
    null_exceedance_counts = np.zeros(len(feature_names), dtype=int)

    total_permutations = int(n_permutations)  # never silently promote 0 → 1
    progress_step = max(1, total_permutations // 10)
    permutation_seeds = [
        int(rng.integers(0, 2**31 - 1)) for _ in range(total_permutations)
    ]

    batch_starts = list(range(0, total_permutations, progress_step))
    for start in progress_iter(
        batch_starts,
        desc=f"{stage_name} RF-FDR permutations",
        unit="batch",
        leave=False,
    ):
        end = min(total_permutations, start + progress_step)
        batch_tasks = [
            ("permuted", seed, None) for seed in permutation_seeds[start:end]
        ]
        batch_importances = _parallel_importance_map(
            tasks=batch_tasks,
            func=fit_importance,
            n_jobs=outer_n_jobs if use_outer_parallel else 1,
        )

        for null_importance in batch_importances:
            null_sum += null_importance
            null_sum_sq += null_importance**2
            null_exceedance_counts += (null_importance >= observed_mean).astype(int)

    # ------------------------------------------------------------------
    # 4. Empirical p‑values and FDR correction
    # ------------------------------------------------------------------
    # Phipson & Smyth: p = (1 + exceedances) / (B + 1); min p = 1/(B+1)
    denominator = float(total_permutations + 1)
    empirical_p_resolution = 1.0 / denominator

    empirical_p = (1.0 + null_exceedance_counts.astype(float)) / denominator

    null_mean = null_sum / float(total_permutations)
    null_variance = (null_sum_sq / float(total_permutations)) - (null_mean**2)
    null_variance = np.maximum(null_variance, 0.0)
    null_std = np.sqrt(null_variance)

    reject, corrected_p, _, _ = multipletests(
        empirical_p, alpha=fdr_alpha, method="fdr_bh"
    )

    results_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "rf_mean_importance": observed_mean,
                "rf_std_importance": observed_std,
                "null_mean_importance": null_mean,
                "null_std_importance": null_std,
                "null_exceedance_count": null_exceedance_counts.astype(int),
                "n_permutations": int(total_permutations),
                "empirical_p_resolution": float(empirical_p_resolution),
                "empirical_p_value": empirical_p,
                "corrected_p_value": corrected_p,
                "significant": reject.astype(bool),
            }
        )
        .sort_values(
            ["corrected_p_value", "rf_mean_importance"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )

    # Retain significant features above the minimum importance
    retained = results_df[
        (results_df["significant"])
        & (results_df["rf_mean_importance"] > min_importance)
    ].copy()

    if top_n is not None:
        retained = retained.head(int(top_n)).copy()

    retained_features = [f for f in retained["feature"].tolist() if f in X.columns]

    # ------------------------------------------------------------------
    # 5. Fallback handling
    # ------------------------------------------------------------------
    used_fallback = False
    used_top_n_fallback = False
    used_unfiltered_fallback = False
    selection_mode = "strict_rf_fdr"

    results_df["selection_status"] = np.where(
        results_df["significant"] & (results_df["rf_mean_importance"] > min_importance),
        "fdr_significant",
        "not_significant",
    )
    results_df["exploratory_non_significant"] = False

    if retained_features:
        X_filtered = X.loc[:, retained_features].copy()
    else:
        used_fallback = True

        if fallback_strategy == "top_n":
            fallback_n = min(fallback_top_n, len(results_df))
            retained = results_df.head(fallback_n).copy()
            retained["selection_status"] = "exploratory_fallback_top_n"
            retained["exploratory_non_significant"] = True
            retained["significant"] = False  # never claim FDR significance for fallback
            retained_features = [
                f for f in retained["feature"].tolist() if f in X.columns
            ]
            results_df.loc[
                results_df["feature"].isin(retained_features), "selection_status"
            ] = "exploratory_fallback_top_n"
            results_df.loc[
                results_df["feature"].isin(retained_features),
                "exploratory_non_significant",
            ] = True

            X_filtered = X.loc[:, retained_features].copy()
            used_top_n_fallback = True
            selection_mode = "rf_top_n_fallback_exploratory"

            logger.warning(
                "%s RF‑FDR retained no FDR-significant features at alpha %.4f. "
                "Using top‑%d RF‑ranked features as EXPLORATORY/NON-SIGNIFICANT fallback.",
                stage_name,
                float(fdr_alpha),
                int(len(retained_features)),
            )

        elif fallback_strategy == "unfiltered":
            retained = results_df.copy()
            retained["selection_status"] = "exploratory_fallback_unfiltered"
            retained["exploratory_non_significant"] = True
            retained["significant"] = False
            retained_features = feature_names
            results_df["selection_status"] = "exploratory_fallback_unfiltered"
            results_df["exploratory_non_significant"] = True
            X_filtered = X.copy()
            used_unfiltered_fallback = True
            selection_mode = "unfiltered_fallback_exploratory"

            logger.warning(
                "%s RF‑FDR retained no FDR-significant features at alpha %.4f. "
                "Using the full aligned matrix as EXPLORATORY/NON-SIGNIFICANT fallback.",
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

    results_path = out / "rf_fdr_feature_results.csv"
    retained_path = out / "rf_fdr_retained_features.csv"
    matrix_path = out / "filtered_matrix.csv"
    summary_path = out / "feature_filtering_summary.json"

    results_df.to_csv(results_path, index=False)
    retained.to_csv(retained_path, index=False)
    X_filtered.to_csv(matrix_path)

    summary = {
        "stage": stage_name,
        "method": "rf_fdr",
        "status": "success" if not used_fallback else "exploratory_fallback",
        "input_features": int(X.shape[1]),
        "tested_features": int(len(feature_names)),
        "retained_features": int(len(retained_features)),
        "retention_fraction": float(len(retained_features) / max(1, X.shape[1])),
        "selection_mode": selection_mode,
        "used_fallback": bool(used_fallback),
        "fallback_features_are_exploratory_non_significant": bool(used_fallback),
        "used_fallback_top_n": bool(used_top_n_fallback),
        "used_fallback_unfiltered_matrix": bool(used_unfiltered_fallback),
        "fallback_strategy": fallback_strategy,
        "fallback_top_n": int(fallback_top_n),
        "min_empirical_p_value": float(results_df["empirical_p_value"].min()),
        "min_empirical_p_value_formula": "1/(B+1)",
        "empirical_p_resolution": float(empirical_p_resolution),
        "minimum_attainable_bh_adjusted_p": float(minimum_attainable_bh_adjusted_p),
        "permutation_resolution_adequate": bool(resolution_ok),
        "permutation_resolution_note": resolution_msg,
        "min_corrected_p_value": float(results_df["corrected_p_value"].min()),
        "unique_empirical_p_values": int(results_df["empirical_p_value"].nunique()),
        "unique_corrected_p_values": int(results_df["corrected_p_value"].nunique()),
        "fdr_alpha": float(fdr_alpha),
        "n_observed_repeats": int(n_observed_repeats),
        "n_permutations": int(n_permutations),
        "n_missing_labels_dropped": int(n_missing_labels),
        "group_blocked_permutations": bool(group_values is not None),
        "group_permutation_mode": (
            "equal_size_whole_group_label_vectors"
            if group_values is not None
            else "sample_level"
        ),
        "missingness_impute_strategy": str(policy.impute_strategy),
        "missingness_state": miss_state.to_dict()
        if hasattr(miss_state, "to_dict")
        else {},
        "rf_selector_outer_n_jobs": int(outer_n_jobs),
        "rf_selector_inner_n_jobs": int(inner_rf_n_jobs),
        "rf_selector_outer_parallel": bool(use_outer_parallel),
        "min_importance": float(min_importance),
        "top_n": int(top_n) if top_n is not None else None,
        "retained_feature_names": retained_features,
        "artifacts": {
            "rf_fdr_results_csv": str(results_path),
            "rf_fdr_retained_features_csv": str(retained_path),
            "filtered_matrix_csv": str(matrix_path),
            "summary_json": str(summary_path),
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
        "filtered_matrix": X_filtered,
        "retained_features": retained_features,
        "feature_results": results_df,
        "summary": summary,
    }

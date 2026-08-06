# network_parser/feature_panel_selection.py
"""
Ranked feature-panel separability check for NetworkParser.

Role in pipeline
----------------
This module runs AFTER central statistical filtering and BEFORE model training.
It does not replace RF-FDR, chi-square/Fisher-FDR, or chi-square permutation-FDR.
Instead, it ranks the already retained genomic features by statistical strength,
evaluates top-N feature panels, and selects a compact model-ready matrix when a
panel shows adequate label separability.

This is a pre-model diagnostic / panel-selection layer. It is not post-tree
bootstrap confidence scoring and should not be interpreted as final unbiased
model performance.
"""

from __future__ import annotations

import json
import logging
import copy
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FeaturePanelThresholdError(RuntimeError):
    """Raised when no configured feature panel reaches the acceptance score."""

    def __init__(self, message: str, *, summary: Dict[str, Any]):
        super().__init__(message)
        self.summary = summary


try:
    from joblib import Parallel, delayed
except ImportError:  # pragma: no cover
    Parallel = None  # type: ignore
    delayed = None  # type: ignore

try:
    from network_parser.utils import (
        progress_iter,
        resolve_effective_n_jobs,
        should_run_parallel,
    )
except ImportError:  # pragma: no cover - package vs source-tree layout
    try:
        from utils import progress_iter, resolve_effective_n_jobs, should_run_parallel  # type: ignore
    except ImportError:  # pragma: no cover

        def progress_iter(iterable, **kwargs):  # type: ignore
            return iterable

        def resolve_effective_n_jobs(config, *, override=None, minimum_tasks=1):  # type: ignore
            return 1

        def should_run_parallel(config, *, enabled_attr, n_tasks, min_tasks=2):  # type: ignore
            return False


# -----------------------------------------------------------------------------
# Small local utilities
# -----------------------------------------------------------------------------


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


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_panel_sizes(value: Any) -> List[int]:
    """Accept tuple/list/string config values and return positive unique sizes."""
    if value is None:
        return []

    if isinstance(value, str):
        raw_items: Iterable[Any] = [x.strip() for x in value.split(",") if x.strip()]
    elif isinstance(value, (list, tuple, set)):
        raw_items = value
    else:
        raw_items = [value]

    sizes: List[int] = []
    for item in raw_items:
        try:
            size = int(item)
        except Exception:
            continue
        if size > 0 and size not in sizes:
            sizes.append(size)

    return sorted(sizes)


def _align_inputs(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    X_local = X.copy()
    y_local = y.copy()
    X_local.index = X_local.index.astype(str)
    y_local.index = y_local.index.astype(str)

    y_local = y_local.astype(str).str.strip()
    y_local = y_local.replace(
        {
            "": pd.NA,
            "-": pd.NA,
            "NA": pd.NA,
            "N/A": pd.NA,
            "None": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
        }
    )
    y_local = y_local.dropna()

    common = X_local.index.intersection(y_local.index)
    if len(common) == 0:
        raise ValueError("No overlapping sample IDs between feature matrix and labels")

    X_local = X_local.loc[common].copy()
    y_local = y_local.loc[common].copy()

    if X_local.empty or X_local.shape[1] == 0:
        raise ValueError("Feature-panel separability check received an empty matrix")
    if y_local.nunique(dropna=True) < 2:
        raise ValueError(
            "Feature-panel separability check requires at least two label classes"
        )

    return X_local, y_local


def _encode_dataframe(X: pd.DataFrame) -> np.ndarray:
    """
    Validate and encode a preprocessed binary feature frame as numeric values.

    Missing or categorical marker states are rejected here. The caller must
    supply the output of an explicit train-fitted missingness transformer.
    """
    encoded_cols: List[np.ndarray] = []

    for col in X.columns:
        series = X[col]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.isna().any():
            raise ValueError(
                f"Feature-panel encoder received unresolved/non-numeric states in {col!r}"
            )
        encoded_cols.append(numeric.to_numpy(dtype=float, copy=False))

    if not encoded_cols:
        raise ValueError("No feature columns available for panel encoding")

    X_arr = np.column_stack(encoded_cols).astype(float, copy=False)
    if not np.isfinite(X_arr).all():
        raise ValueError("Feature-panel encoder received non-finite values")
    return X_arr


# -----------------------------------------------------------------------------
# Ranking logic
# -----------------------------------------------------------------------------


def _records_from_mapping(mapping: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for feature, payload in mapping.items():
        if not isinstance(payload, dict):
            continue
        row = {"feature": str(feature)}
        row.update(payload)
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_results_to_dataframe(
    filter_result: Optional[Dict[str, Any]]
) -> pd.DataFrame:
    if not isinstance(filter_result, dict):
        return pd.DataFrame()

    feature_results = filter_result.get("feature_results")
    if isinstance(feature_results, pd.DataFrame):
        df = feature_results.copy()
    elif isinstance(feature_results, dict):
        df = _records_from_mapping(feature_results)
    else:
        df = pd.DataFrame()

    if df.empty:
        multiple_testing = filter_result.get("multiple_testing")
        if isinstance(multiple_testing, dict):
            df = _records_from_mapping(multiple_testing)

    if df.empty:
        association = filter_result.get("association")
        if isinstance(association, dict):
            df = _records_from_mapping(association)

    if not df.empty and "feature" not in df.columns:
        df = df.reset_index().rename(columns={"index": "feature"})

    return df


def rank_features_for_panel_selection(
    X: pd.DataFrame,
    filter_result: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Rank retained features from strongest to weakest statistical evidence.

    Priority is corrected p-value first, then empirical/raw p-value, then effect
    / importance columns where available. If no statistical table is available,
    the incoming column order is preserved and the panel check becomes a pure
    top-column diagnostic.
    """
    feature_order = {str(feature): rank for rank, feature in enumerate(X.columns)}
    df = _feature_results_to_dataframe(filter_result)

    if df.empty:
        df = pd.DataFrame({"feature": list(map(str, X.columns))})

    df = df.copy()
    df["feature"] = df["feature"].astype(str)
    df = df[df["feature"].isin(feature_order)].copy()

    # Add any retained matrix column missing from the statistical table.
    present = set(df["feature"].astype(str))
    missing = [feature for feature in map(str, X.columns) if feature not in present]
    if missing:
        df = pd.concat([df, pd.DataFrame({"feature": missing})], ignore_index=True)

    numeric_cols = [
        "corrected_p_value",
        "empirical_p_value",
        "p_value",
        "rf_mean_importance",
        "mutual_info",
        "cramers_v",
        "chi2_statistic",
        "statistic",
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["original_order"] = df["feature"].map(feature_order).astype(int)

    # Keep all fields, but sort using evidence strength. Missing p-values go last.
    df = df.sort_values(
        by=[
            "corrected_p_value",
            "empirical_p_value",
            "p_value",
            "rf_mean_importance",
            "mutual_info",
            "cramers_v",
            "chi2_statistic",
            "statistic",
            "original_order",
        ],
        ascending=[True, True, True, False, False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)

    df.insert(0, "rank", np.arange(1, len(df) + 1, dtype=int))
    return df


def _column_variance_for_prefilter(series: pd.Series) -> float:
    """Fast variance score used only to reduce very large panel-scoring pools."""
    numeric = pd.to_numeric(series, errors="coerce")
    observed_numeric = numeric.dropna()
    observed_tokens = series.dropna().astype(str).str.strip()
    observed_tokens = observed_tokens.loc[
        ~observed_tokens.isin({"", "nan", "NaN", "nd", "NA", "N/A"})
    ]
    if observed_numeric.shape[0] == observed_tokens.shape[0]:
        arr = observed_numeric.to_numpy(dtype=float, copy=False)
    else:
        arr = pd.Categorical(observed_tokens).codes.astype(float)

    if arr.size == 0:
        return 0.0
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.var(arr)) if arr.size else 0.0


def _variance_prefilter_ranked_features(
    X: pd.DataFrame,
    ranked_features: List[str],
    max_features: int,
    pool_multiplier: int = 4,
) -> List[str]:
    """
    Reduce the scoring pool for very large filtered matrices.

    Central statistical filtering has already happened before this function.
    This helper only limits how many retained features enter the expensive
    panel-separability scoring step. It first respects the statistical ranking
    by building a candidate pool from the strongest-ranked retained features,
    then keeps the most variable features within that pool and restores the
    original statistical order among the retained candidates.
    """
    max_features = int(max_features)
    if max_features < 1 or len(ranked_features) <= max_features:
        return ranked_features

    pool_multiplier = max(1, int(pool_multiplier))
    pool_size = min(len(ranked_features), max_features * pool_multiplier)
    candidate_pool = [f for f in ranked_features[:pool_size] if f in X.columns]

    if len(candidate_pool) <= max_features:
        return candidate_pool

    variance_records: List[Tuple[str, float]] = []
    for feature in candidate_pool:
        try:
            variance_records.append(
                (feature, _column_variance_for_prefilter(X[feature]))
            )
        except Exception:
            variance_records.append((feature, 0.0))

    selected_by_variance = {
        feature
        for feature, _ in sorted(
            variance_records, key=lambda item: item[1], reverse=True
        )[:max_features]
    }

    # Restore statistical order after the variance-based pool reduction.
    return [feature for feature in ranked_features if feature in selected_by_variance]


# -----------------------------------------------------------------------------
# Panel scoring and selection
# -----------------------------------------------------------------------------


def _normalize_panel_classifier(value: Any) -> str:
    """Normalize the supervised feature-panel probe name."""
    name = str(value or "lr").strip().lower().replace("-", "_")
    aliases = {
        "logistic": "lr",
        "logistic_regression": "lr",
        "randomforest": "rf",
        "random_forest": "rf",
        "random_forest_classifier": "rf",
    }
    return aliases.get(name, name)


def _score_supervised_balanced_accuracy(
    X_arr: np.ndarray,
    y: pd.Series,
    cv_splits: int,
    random_state: int,
    classifier: str = "lr",
    lr_max_iter: int = 2000,
    lr_tol: float = 1e-4,
    rf_n_estimators: int = 300,
    rf_max_features: Any = "sqrt",
    rf_min_samples_leaf: int = 1,
    rf_class_weight: Optional[str] = "balanced",
    rf_n_jobs: int = 1,
    groups: Optional[pd.Series] = None,
    missingness_policy: Optional[Any] = None,
) -> Tuple[float, str, int]:
    """Return (mean_score, status, actual_cv_splits_used)."""
    X_frame = X_arr.copy() if isinstance(X_arr, pd.DataFrame) else pd.DataFrame(X_arr)
    X_frame.index = y.index
    X_frame.columns = X_frame.columns.astype(str)
    y_values = y.astype(str).to_numpy()
    labels, counts = np.unique(y_values, return_counts=True)
    min_class_count = int(counts.min()) if counts.size else 0
    n_splits = min(int(cv_splits), min_class_count)
    group_values: Optional[np.ndarray] = None
    if groups is not None:
        groups = pd.Series(groups).reindex(y.index)
        if groups.isna().any():
            return np.nan, "skipped_grouped_cv_missing_group_values", 0
        group_values = groups.astype(str).to_numpy()
        class_group_counts = [
            int(pd.Series(group_values[y_values == label]).nunique())
            for label in labels
        ]
        if class_group_counts:
            n_splits = min(n_splits, min(class_group_counts))
    classifier = _normalize_panel_classifier(classifier)

    if len(labels) < 2 or n_splits < 2:
        return np.nan, "skipped_cv_insufficient_class_support", int(max(0, n_splits))

    if classifier not in {"lr", "rf"}:
        return np.nan, f"skipped_unknown_supervised_classifier:{classifier}", 0

    # Cap RF inner threads at 1 when outer panel scoring is parallelized by caller.
    rf_inner_jobs = max(1, int(rf_n_jobs)) if int(rf_n_jobs) > 0 else 1

    splitter = (
        StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        if group_values is not None
        else StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
    )
    fold_scores: List[float] = []

    split_iter = splitter.split(X_frame, y_values, groups=group_values)
    for fold_idx, (train_idx, test_idx) in enumerate(split_iter):
        try:
            try:
                from network_parser.matrix_contract import (
                    MissingnessPolicy,
                    fit_missingness_state,
                    transform_with_missingness_state,
                )
            except ImportError:  # pragma: no cover
                from matrix_contract import (  # type: ignore
                    MissingnessPolicy,
                    fit_missingness_state,
                    transform_with_missingness_state,
                )

            policy = copy.deepcopy(missingness_policy or MissingnessPolicy())
            # Panel CV evaluates all labelled samples; feature filtering and
            # imputation are fitted only on this inner training split.
            policy.drop_exceeding_samples = False
            X_train_raw = X_frame.iloc[train_idx].copy()
            X_test_raw = X_frame.iloc[test_idx].copy()
            X_train_filtered, state, _ = fit_missingness_state(
                X_train_raw,
                policy=policy,
            )
            X_train_frame, _ = transform_with_missingness_state(
                X_train_filtered,
                state,
                apply_imputation=True,
                drop_high_missing_samples=False,
            )
            X_test_frame, _ = transform_with_missingness_state(
                X_test_raw,
                state,
                apply_imputation=True,
                drop_high_missing_samples=False,
            )
            if X_train_frame.isna().any().any() or X_test_frame.isna().any().any():
                raise ValueError(
                    "panel fold contains NaN after train-fitted preprocessing"
                )
            X_train_arr = _encode_dataframe(X_train_frame)
            X_test_arr = _encode_dataframe(X_test_frame)

            if classifier == "rf":
                model = RandomForestClassifier(
                    n_estimators=int(rf_n_estimators),
                    max_features=rf_max_features,
                    min_samples_leaf=int(rf_min_samples_leaf),
                    class_weight=rf_class_weight,
                    random_state=int(random_state) + int(fold_idx),
                    n_jobs=rf_inner_jobs,
                )
                X_train = X_train_arr
                X_test = X_test_arr
                model.fit(X_train, y_values[train_idx])
            else:
                model = LogisticRegression(
                    max_iter=int(lr_max_iter),
                    class_weight="balanced",
                    solver="lbfgs",
                    tol=float(lr_tol),
                    n_jobs=1,
                )
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train_arr)
                X_test = scaler.transform(X_test_arr)

                with warnings.catch_warnings():
                    # Treat persistent convergence warnings as a failed score rather
                    # than letting noisy, unstable folds influence panel selection.
                    warnings.filterwarnings("error", category=ConvergenceWarning)
                    model.fit(X_train, y_values[train_idx])

            pred = model.predict(X_test)
            fold_scores.append(float(balanced_accuracy_score(y_values[test_idx], pred)))
        except ConvergenceWarning as exc:
            logger.warning(
                "Feature-panel LR scoring did not converge for one fold: %s", exc
            )
            return (
                np.nan,
                f"supervised_scoring_failed_convergence: {exc}",
                int(n_splits),
            )
        except Exception as exc:
            logger.warning(
                "Feature-panel supervised scoring failed for one fold | classifier=%s | error=%s",
                classifier,
                exc,
            )
            return (
                np.nan,
                f"supervised_scoring_failed_{classifier}: {exc}",
                int(n_splits),
            )

    if not fold_scores:
        return np.nan, "supervised_scoring_failed_no_valid_folds", int(n_splits)

    return float(np.mean(fold_scores)), f"success_{classifier}", int(n_splits)


def _score_unsupervised_clustering(
    X_arr: np.ndarray,
    y: pd.Series,
    random_state: int,
    max_silhouette_samples: int,
) -> Dict[str, float]:
    y_values = y.astype(str).to_numpy()
    y_codes, y_levels = pd.factorize(y_values, sort=True)
    n_classes = int(len(y_levels))

    scores = {
        "adjusted_rand": np.nan,
        "normalized_mutual_info": np.nan,
        "silhouette": np.nan,
    }

    if n_classes < 2 or X_arr.shape[0] <= n_classes:
        return scores

    try:
        kmeans = KMeans(n_clusters=n_classes, random_state=random_state, n_init=5)
        cluster_labels = kmeans.fit_predict(X_arr)
        scores["adjusted_rand"] = float(adjusted_rand_score(y_codes, cluster_labels))
        scores["normalized_mutual_info"] = float(
            normalized_mutual_info_score(y_codes, cluster_labels)
        )

        unique_clusters = np.unique(cluster_labels)
        if 1 < len(unique_clusters) < X_arr.shape[0]:
            if X_arr.shape[0] > max_silhouette_samples:
                rng = np.random.default_rng(random_state)
                subset = rng.choice(
                    X_arr.shape[0], size=max_silhouette_samples, replace=False
                )
                scores["silhouette"] = float(
                    silhouette_score(X_arr[subset], cluster_labels[subset])
                )
            else:
                scores["silhouette"] = float(silhouette_score(X_arr, cluster_labels))
    except Exception as exc:
        logger.warning("Feature-panel clustering diagnostic failed: %s", exc)

    return scores


def _score_panel(
    X_panel: pd.DataFrame,
    y: pd.Series,
    config: Any,
) -> Dict[str, Any]:
    """
    Score a feature panel with enhanced robustness and diagnostics.
    """
    random_state = int(getattr(config, "random_state", 42))
    cv_splits = int(getattr(config, "feature_panel_cv_splits", 5))
    max_silhouette_samples = int(
        getattr(config, "feature_panel_max_silhouette_samples", 5000)
    )
    classifier = _normalize_panel_classifier(
        getattr(config, "feature_panel_classifier", "rf")
    )

    rf_n_jobs = getattr(config, "feature_panel_rf_n_jobs", None) or getattr(
        config, "n_jobs", 1
    )

    # Matrix contract policy is fitted independently inside every panel CV fold.
    try:
        from network_parser.matrix_contract import (
            MissingnessPolicy,
            prepare_for_sklearn,
        )
    except ImportError:  # pragma: no cover
        from matrix_contract import MissingnessPolicy, prepare_for_sklearn  # type: ignore

    policy = MissingnessPolicy.from_config(config)
    panel_groups = getattr(config, "_feature_panel_groups", None)
    if panel_groups is not None:
        panel_groups = pd.Series(panel_groups).reindex(X_panel.index)

    # --- Supervised Probe ---
    # When panel sizes are scored in parallel, force RF inner n_jobs=1 to avoid
    # nested oversubscription under a single config.n_jobs budget.
    panel_parallel = bool(getattr(config, "feature_panel_parallel_scoring", True))
    effective_rf_jobs = (
        1 if panel_parallel else int(rf_n_jobs if rf_n_jobs is not None else 1)
    )

    (
        supervised_score,
        supervised_status,
        actual_cv_splits,
    ) = _score_supervised_balanced_accuracy(
        X_arr=X_panel,
        y=y,
        cv_splits=cv_splits,
        random_state=random_state,
        classifier=classifier,
        lr_max_iter=int(getattr(config, "feature_panel_lr_max_iter", 2000)),
        lr_tol=float(getattr(config, "feature_panel_lr_tol", 1e-4)),
        rf_n_estimators=int(getattr(config, "feature_panel_rf_n_estimators", 300)),
        rf_max_features=getattr(config, "feature_panel_rf_max_features", "sqrt"),
        rf_min_samples_leaf=int(
            getattr(config, "feature_panel_rf_min_samples_leaf", 1)
        ),
        rf_class_weight=getattr(config, "feature_panel_rf_class_weight", "balanced"),
        rf_n_jobs=effective_rf_jobs,
        groups=panel_groups,
        missingness_policy=policy,
    )

    # --- Unsupervised Clustering Diagnostics ---
    run_clustering = bool(
        getattr(config, "feature_panel_run_clustering_diagnostics", False)
    )
    if run_clustering:
        X_panel_imp, _, _ = prepare_for_sklearn(X_panel, policy=policy)
        X_arr = _encode_dataframe(X_panel_imp)
        unsupervised_scores = _score_unsupervised_clustering(
            X_arr=X_arr,
            y=y.loc[X_panel_imp.index],
            random_state=random_state,
            max_silhouette_samples=max_silhouette_samples,
        )
    else:
        unsupervised_scores = {
            "adjusted_rand": np.nan,
            "normalized_mutual_info": np.nan,
            "silhouette": np.nan,
        }

    # --- Enhanced Diagnostics ---
    n_features = int(X_panel.shape[1])
    n_samples = int(X_panel.shape[0])
    n_classes = int(y.nunique())

    return {
        "balanced_accuracy": supervised_score,
        "supervised_status": supervised_status,
        "supervised_classifier": classifier,
        **unsupervised_scores,
        "n_features": n_features,
        "n_samples": n_samples,
        "n_classes": n_classes,
        "requested_cv_splits": int(cv_splits),
        "actual_cv_splits": int(actual_cv_splits),
        "panel_diagnostics": {
            "cv_splits_requested": int(cv_splits),
            "cv_splits_used": int(actual_cv_splits),
            "rf_n_jobs": effective_rf_jobs,
            "memory_usage_mb": round(
                X_panel.memory_usage(deep=True).sum() / (1024**2), 2
            ),
            "grouped_inner_cv": bool(panel_groups is not None),
            "preprocessing_scope": "fit_within_each_panel_cv_training_fold",
            "clustering_diagnostics_enabled": run_clustering,
        },
    }


def _select_panel_row(
    panel_scores: pd.DataFrame,
    metric: str,
    min_score: float,
    selection_rule: str,
) -> Tuple[pd.Series, str]:
    if panel_scores.empty:
        raise ValueError("No feature-panel scores were produced")

    metric_values = pd.to_numeric(panel_scores.get(metric, np.nan), errors="coerce")

    if metric_values.notna().sum() == 0:
        raise ValueError(
            f"Feature-panel metric '{metric}' could not be computed for any panel"
        )

    scored = panel_scores.copy()
    scored["_metric_value"] = metric_values
    scored = scored[scored["_metric_value"].notna()].copy()

    passing = scored[scored["_metric_value"] >= float(min_score)].copy()
    rule = str(selection_rule).lower()

    if not passing.empty and rule == "smallest_passing":
        selected = passing.sort_values(
            by=["n_features", "_metric_value"],
            ascending=[True, False],
        ).iloc[0]
        return selected, "smallest_panel_meeting_threshold"

    if not passing.empty and rule == "best_passing":
        selected = passing.sort_values(
            by=["_metric_value", "n_features"],
            ascending=[False, True],
        ).iloc[0]
        return selected, "best_panel_meeting_threshold"

    selected = scored.sort_values(
        by=["_metric_value", "n_features"],
        ascending=[False, True],
    ).iloc[0]
    return selected, "best_available_panel_threshold_not_met"


def log_model_input_panel_decision(
    *,
    model_name: str,
    X: pd.DataFrame,
    panel_summary: Optional[Dict[str, Any]],
    log: Optional[logging.Logger] = None,
) -> None:
    """Log and validate the feature-panel decision immediately before fitting."""
    active_logger = log or logger
    summary = panel_summary if isinstance(panel_summary, dict) else {}
    status = str(summary.get("status", "not_recorded"))
    reason = str(summary.get("selection_reason", "panel_selection_not_recorded"))
    allowed = bool(summary.get("model_training_allowed", True))
    selected_features = int(X.shape[1])
    reported_features = summary.get("selected_features")

    if not allowed:
        raise FeaturePanelThresholdError(
            f"{model_name}: model training blocked by feature-panel decision ({reason}).",
            summary=summary,
        )
    if reported_features is not None and int(reported_features) != selected_features:
        raise RuntimeError(
            f"{model_name}: feature-panel/model-input mismatch; summary selected "
            f"{reported_features} features but model input has {selected_features}."
        )

    active_logger.info(
        "Model input panel decision | model=%s | samples=%d | selected_features=%d | "
        "panel_status=%s | selection_reason=%s | candidate_panels=%s | "
        "minimum_required_score=%s | model_training=allowed",
        model_name,
        int(X.shape[0]),
        selected_features,
        status,
        reason,
        summary.get("candidate_panel_sizes", "not_recorded"),
        summary.get("minimum_required_score", "not_recorded"),
    )


def run_feature_panel_separability_check(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path | str,
    config: Any,
    stage_name: str,
    filter_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Rank filtered features, evaluate top-N panels, and return a model matrix.

    Returns a dictionary containing:
      - selected_matrix: pd.DataFrame
      - selected_features: list[str]
      - ranked_features: pd.DataFrame
      - panel_scores: pd.DataFrame
      - summary: dict
    """
    enabled = bool(getattr(config, "run_feature_panel_separability_check", True))
    out_dir = _ensure_dir(Path(output_dir) / "feature_panel_separability")

    X_aligned, y_aligned = _align_inputs(X, y)

    if not enabled:
        selected_matrix_path = out_dir / "selected_panel_matrix.csv"
        ranked_path = out_dir / "ranked_features.csv"
        panel_scores_path = out_dir / "panel_scores.csv"

        X_aligned.to_csv(selected_matrix_path)
        ranked_disabled = pd.DataFrame(
            {
                "rank": np.arange(1, X_aligned.shape[1] + 1, dtype=int),
                "feature": list(map(str, X_aligned.columns)),
            }
        )
        ranked_disabled.to_csv(ranked_path, index=False)
        panel_scores_disabled = pd.DataFrame()
        panel_scores_disabled.to_csv(panel_scores_path, index=False)

        summary = {
            "stage_name": stage_name,
            "status": "skipped",
            "input_features": int(X_aligned.shape[1]),
            "selected_features": int(X_aligned.shape[1]),
            "candidate_panel_sizes": [],
            "large_feature_scoring_pool_applied": False,
            "scoring_pool_features": int(X_aligned.shape[1]),
            "original_filtered_features": int(X_aligned.shape[1]),
            "score_full_large_matrix": False,
            "selection_metric": str(
                getattr(config, "feature_panel_metric", "balanced_accuracy")
            ),
            "supervised_classifier": _normalize_panel_classifier(
                getattr(config, "feature_panel_classifier", "lr")
            ),
            "minimum_required_score": float(
                getattr(config, "feature_panel_min_score", 0.75)
            ),
            "selection_rule": str(
                getattr(config, "feature_panel_selection_rule", "smallest_passing")
            ),
            "selection_reason": "feature_panel_check_disabled",
            "used_original_central_filtered_matrix": True,
            "selected_feature_names": list(map(str, X_aligned.columns)),
            "artifacts": {
                "feature_panel_dir": str(out_dir),
                "ranked_features_csv": str(ranked_path),
                "panel_scores_csv": str(panel_scores_path),
                "selected_panel_matrix": str(selected_matrix_path),
                "summary_json": str(
                    out_dir / "feature_panel_separability_summary.json"
                ),
            },
        }
        with open(
            out_dir / "feature_panel_separability_summary.json", "w", encoding="utf-8"
        ) as handle:
            json.dump(summary, handle, indent=2, default=_json_default)
        logger.info(
            "%s feature-panel separability check skipped | passing through filtered matrix | features=%d",
            stage_name,
            int(X_aligned.shape[1]),
        )
        return {
            "selected_matrix": X_aligned,
            "selected_features": list(X_aligned.columns),
            "ranked_features": ranked_disabled,
            "panel_scores": panel_scores_disabled,
            "summary": summary,
        }

    ranked = rank_features_for_panel_selection(X_aligned, filter_result=filter_result)
    ranked_path = out_dir / "ranked_features.csv"
    ranked.to_csv(ranked_path, index=False)

    configured_sizes = _parse_panel_sizes(
        getattr(config, "feature_panel_sizes", (100, 200, 500, 1000))
    )
    always_include_full = bool(
        getattr(config, "feature_panel_always_include_full_filtered", False)
    )

    ranked_features = ranked["feature"].astype(str).tolist()
    # Optional: seed known resistance markers for phenotype endpoints (config off by default).
    known_seed_info: Dict[str, Any] = {"enabled": False, "applied": False}
    try:
        from network_parser.known_marker_seed import apply_known_marker_seed
    except ImportError:  # pragma: no cover
        try:
            from known_marker_seed import apply_known_marker_seed  # type: ignore
        except ImportError:
            apply_known_marker_seed = None  # type: ignore
    if apply_known_marker_seed is not None:
        ranked_features, known_seed_info = apply_known_marker_seed(
            ranked_features=ranked_features,
            matrix_columns=list(map(str, X_aligned.columns)),
            config=config,
            stage_name=stage_name,
        )
        # Persist seed annotation on the ranked table for audit.
        if known_seed_info.get("applied"):
            seeded = set(known_seed_info.get("seeded_feature_ids") or [])
            ranked = ranked.copy()
            ranked["known_marker_seed"] = ranked["feature"].astype(str).isin(seeded)
            # Re-write rank order to match reordered list
            order = {f: i for i, f in enumerate(ranked_features)}
            ranked["_seed_order"] = ranked["feature"].astype(str).map(order)
            ranked = ranked.sort_values("_seed_order", kind="mergesort").reset_index(
                drop=True
            )
            ranked["rank"] = np.arange(1, len(ranked) + 1, dtype=int)
            ranked = ranked.drop(columns=["_seed_order"], errors="ignore")
            ranked.to_csv(ranked_path, index=False)
    original_feature_count = int(X_aligned.shape[1])
    large_threshold = int(
        getattr(config, "feature_panel_large_feature_threshold", 5000)
    )
    large_max_features = int(
        getattr(config, "feature_panel_large_max_scoring_features", 5000)
    )
    large_pool_multiplier = int(
        getattr(config, "feature_panel_large_pool_multiplier", 4)
    )
    score_full_large_matrix = bool(
        getattr(config, "feature_panel_score_full_large_matrix", False)
    )

    large_feature_scoring_pool_applied = False
    scoring_ranked_features = ranked_features

    allow_variance_pool = bool(
        getattr(config, "feature_panel_allow_variance_prefilter", False)
    )
    if original_feature_count > large_threshold and allow_variance_pool:
        scoring_ranked_features = _variance_prefilter_ranked_features(
            X=X_aligned,
            ranked_features=ranked_features,
            max_features=large_max_features,
            pool_multiplier=large_pool_multiplier,
        )
        large_feature_scoring_pool_applied = len(scoring_ranked_features) < len(
            ranked_features
        )
        logger.info(
            "%s large feature set detected; scoring pool reduced from %d to %d retained features "
            "using variance within the statistically ranked candidate pool "
            "(feature_panel_allow_variance_prefilter=True).",
            stage_name,
            original_feature_count,
            len(scoring_ranked_features),
        )
    elif original_feature_count > large_threshold and not allow_variance_pool:
        logger.info(
            "%s large feature set (%d features); variance prefilter disabled. "
            "Set feature_panel_allow_variance_prefilter=True to reduce the scoring pool, "
            "or lower feature_panel_large_feature_threshold.",
            stage_name,
            original_feature_count,
        )

    scoring_feature_count = max(1, len(scoring_ranked_features))

    candidate_sizes: List[int] = []
    for size in configured_sizes:
        candidate_sizes.append(min(size, scoring_feature_count))

    if always_include_full:
        if large_feature_scoring_pool_applied and not score_full_large_matrix:
            candidate_sizes.append(scoring_feature_count)
        else:
            candidate_sizes.append(original_feature_count)

    # Unique, positive, ascending panel sizes.
    candidate_sizes = sorted({size for size in candidate_sizes if size > 0})

    if not candidate_sizes:
        candidate_sizes = [scoring_feature_count]

    logger.info(
        "%s feature-panel separability check started | filtered_features=%d | scoring_pool_features=%d | candidate_panels=%s | supervised_classifier=%s",
        stage_name,
        original_feature_count,
        scoring_feature_count,
        ",".join(str(x) for x in candidate_sizes),
        _normalize_panel_classifier(getattr(config, "feature_panel_classifier", "lr")),
    )

    seeded_ids = list(known_seed_info.get("seeded_feature_ids") or [])
    force_known = bool(known_seed_info.get("applied")) and str(
        known_seed_info.get("mode") or ""
    ) in {"force_include", "rank_boost"}

    def _score_one_panel(panel_size: int) -> Optional[Dict[str, Any]]:
        feature_source = (
            ranked_features
            if panel_size > len(scoring_ranked_features)
            else scoring_ranked_features
        )
        if force_known and seeded_ids:
            try:
                from network_parser.known_marker_seed import build_panel_with_forced_known
            except ImportError:  # pragma: no cover
                from known_marker_seed import build_panel_with_forced_known  # type: ignore
            panel_features = build_panel_with_forced_known(
                ranked_features=feature_source,
                panel_size=int(panel_size),
                known_seeded=seeded_ids,
                matrix_columns=list(map(str, X_aligned.columns)),
            )
        else:
            panel_features = [
                feature
                for feature in feature_source[:panel_size]
                if feature in X_aligned.columns
            ]
        if not panel_features:
            return None

        X_panel = X_aligned.loc[:, panel_features].copy()
        score_payload = _score_panel(
            X_panel=X_panel, y=y_aligned.loc[X_panel.index], config=config
        )
        return {
            "stage_name": stage_name,
            "n_features": int(X_panel.shape[1]),
            "panel_size_requested": int(panel_size),
            "n_known_markers_in_panel": int(
                sum(1 for f in panel_features if f in set(seeded_ids))
            ),
            **score_payload,
        }

    panel_records: List[Dict[str, Any]] = []
    use_parallel = (
        Parallel is not None
        and delayed is not None
        and should_run_parallel(
            config,
            enabled_attr="feature_panel_parallel_scoring",
            n_tasks=len(candidate_sizes),
        )
    )
    if use_parallel:
        n_jobs = resolve_effective_n_jobs(config, minimum_tasks=len(candidate_sizes))
        scored = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_score_one_panel)(panel_size) for panel_size in candidate_sizes
        )
        panel_records = [record for record in scored if isinstance(record, dict)]
    else:
        for panel_size in progress_iter(
            candidate_sizes,
            desc=f"{stage_name} feature-panel scoring",
            unit="panel",
            leave=False,
        ):
            record = _score_one_panel(panel_size)
            if isinstance(record, dict):
                panel_records.append(record)

    panel_scores = pd.DataFrame(panel_records)
    panel_scores_path = out_dir / "panel_scores.csv"
    panel_scores.to_csv(panel_scores_path, index=False)

    selection_metric = str(getattr(config, "feature_panel_metric", "balanced_accuracy"))
    min_score = float(getattr(config, "feature_panel_min_score", 0.75))
    selection_rule = str(
        getattr(config, "feature_panel_selection_rule", "smallest_passing")
    )
    supervised_classifier = _normalize_panel_classifier(
        getattr(config, "feature_panel_classifier", "lr")
    )

    try:
        selected_row, selection_reason = _select_panel_row(
            panel_scores=panel_scores,
            metric=selection_metric,
            min_score=min_score,
            selection_rule=selection_rule,
        )
        selected_n = int(selected_row["n_features"])
        selected_feature_source = (
            ranked_features
            if selected_n > len(scoring_ranked_features)
            else scoring_ranked_features
        )
        if force_known and seeded_ids:
            try:
                from network_parser.known_marker_seed import build_panel_with_forced_known
            except ImportError:  # pragma: no cover
                from known_marker_seed import build_panel_with_forced_known  # type: ignore
            selected_features = build_panel_with_forced_known(
                ranked_features=selected_feature_source,
                panel_size=selected_n,
                known_seeded=seeded_ids,
                matrix_columns=list(map(str, X_aligned.columns)),
            )
        else:
            selected_features = [
                feature
                for feature in selected_feature_source[:selected_n]
                if feature in X_aligned.columns
            ]
        X_selected = X_aligned.loc[:, selected_features].copy()
        selected_score = float(selected_row[selection_metric])
        if selection_reason in {
            "smallest_panel_meeting_threshold",
            "best_panel_meeting_threshold",
        }:
            status = "success"
        else:
            status = "threshold_not_met"
        used_original_matrix = selected_n == int(X_aligned.shape[1])
    except Exception as exc:
        logger.error(
            "%s feature-panel selection failed; model training will not start | error=%s",
            stage_name,
            exc,
        )
        raise RuntimeError(
            f"{stage_name}: feature-panel selection failed; model training was blocked: {exc}"
        ) from exc

    threshold_failure_strategy = (
        str(getattr(config, "feature_panel_threshold_failure_strategy", "stop"))
        .strip()
        .lower()
    )
    threshold_not_met = selection_reason == "best_available_panel_threshold_not_met"
    block_training = threshold_not_met and threshold_failure_strategy == "stop"

    selected_matrix_path: Optional[Path] = None
    best_available_matrix_path: Optional[Path] = None
    if block_training:
        status = "unsupported_threshold_not_met"
        best_available_matrix_path = out_dir / "best_available_panel_matrix.csv"
        X_selected.to_csv(best_available_matrix_path)
    else:
        selected_matrix_path = out_dir / "selected_panel_matrix.csv"
        X_selected.to_csv(selected_matrix_path)
        if threshold_not_met:
            status = "exploratory_best_available_below_threshold"

    summary = {
        "stage_name": stage_name,
        "status": status,
        "score_role": "panel_tuning_estimate_not_unbiased_performance",
        "score_role_note": (
            "Panel separability scores are internal tuning diagnostics used to choose "
            "a compact feature subset. They are not nested held-out generalisation estimates."
        ),
        "input_features": int(X_aligned.shape[1]),
        "selected_features": 0 if block_training else int(X_selected.shape[1]),
        "best_available_features": int(X_selected.shape[1]),
        "best_available_score": float(selected_score),
        "candidate_panel_sizes": candidate_sizes,
        "large_feature_scoring_pool_applied": bool(large_feature_scoring_pool_applied),
        "variance_prefilter_enabled": bool(allow_variance_pool),
        "scoring_pool_features": int(scoring_feature_count),
        "original_filtered_features": int(original_feature_count),
        "score_full_large_matrix": bool(score_full_large_matrix),
        "selection_metric": selection_metric,
        "supervised_classifier": supervised_classifier,
        "minimum_required_score": float(min_score),
        "selection_rule": selection_rule,
        "threshold_failure_strategy": threshold_failure_strategy,
        "selection_reason": selection_reason,
        "model_training_allowed": not block_training,
        "used_original_central_filtered_matrix": bool(used_original_matrix),
        "selected_feature_names": [] if block_training else selected_features,
        "best_available_feature_names": selected_features,
        "known_marker_seed": known_seed_info,
        "n_known_markers_in_selected_panel": int(
            0
            if block_training
            else sum(1 for f in selected_features if f in set(seeded_ids))
        ),
        "artifacts": {
            "feature_panel_dir": str(out_dir),
            "ranked_features_csv": str(ranked_path),
            "panel_scores_csv": str(panel_scores_path),
            "selected_panel_matrix": str(selected_matrix_path)
            if selected_matrix_path is not None
            else None,
            "best_available_panel_matrix": str(best_available_matrix_path)
            if best_available_matrix_path is not None
            else None,
            "summary_json": str(out_dir / "feature_panel_separability_summary.json"),
        },
    }

    with open(
        out_dir / "feature_panel_separability_summary.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, default=_json_default)

    if block_training:
        message = (
            f"{stage_name}: no configured feature panel reached the required "
            f"{selection_metric} >= {min_score:.4f}; best panel had "
            f"{selected_n} features with score={selected_score:.4f}. "
            "This node is unsupported and model training was blocked."
        )
        logger.error(
            "Feature-panel decision | stage=%s | status=unsupported | candidates=%s | "
            "selected_features=0 | best_available_features=%d | best_available_score=%.6f | "
            "minimum_required_score=%.6f | reason=%s | model_training=blocked",
            stage_name,
            candidate_sizes,
            selected_n,
            selected_score,
            min_score,
            selection_reason,
        )
        raise FeaturePanelThresholdError(message, summary=summary)

    log_method = logger.warning if threshold_not_met else logger.info
    log_method(
        "Feature-panel decision | stage=%s | status=%s | candidates=%s | "
        "selected_features=%d | selected_score=%.6f | minimum_required_score=%.6f | "
        "reason=%s | model_training=allowed",
        stage_name,
        status,
        candidate_sizes,
        int(X_selected.shape[1]),
        selected_score,
        min_score,
        selection_reason,
    )

    return {
        "selected_matrix": X_selected,
        "selected_features": selected_features,
        "ranked_features": ranked,
        "panel_scores": panel_scores,
        "summary": summary,
    }

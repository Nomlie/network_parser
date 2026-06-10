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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


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
        {"": pd.NA, "-": pd.NA, "NA": pd.NA, "N/A": pd.NA, "None": pd.NA, "nan": pd.NA, "NaN": pd.NA}
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
        raise ValueError("Feature-panel separability check requires at least two label classes")

    return X_local, y_local


def _encode_dataframe(X: pd.DataFrame) -> np.ndarray:
    """
    Encode binary / categorical / mixed feature columns into numeric arrays.

    Numeric columns are kept numeric where possible. Non-numeric columns are
    encoded as categorical integer codes. Missing values are treated as baseline.
    """
    encoded_cols: List[np.ndarray] = []

    for col in X.columns:
        series = X[col]
        numeric = pd.to_numeric(series, errors="coerce")

        if numeric.notna().sum() == len(numeric):
            encoded_cols.append(numeric.to_numpy(dtype=float, copy=False))
            continue

        clean = series.where(~series.isna(), "__MISSING__")
        clean = clean.astype(str).str.strip()
        clean = clean.replace({"": "__MISSING__", "nan": "__MISSING__", "NaN": "__MISSING__", "nd": "__MISSING__"})
        cat = pd.Categorical(clean)
        encoded_cols.append(cat.codes.astype(float))

    if not encoded_cols:
        raise ValueError("No feature columns available for panel encoding")

    X_arr = np.column_stack(encoded_cols).astype(float, copy=False)
    X_arr[~np.isfinite(X_arr)] = 0.0
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


def _feature_results_to_dataframe(filter_result: Optional[Dict[str, Any]]) -> pd.DataFrame:
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
    if numeric.notna().sum() == len(numeric):
        arr = numeric.to_numpy(dtype=float, copy=False)
    else:
        clean = series.where(~series.isna(), "__MISSING__")
        clean = clean.astype(str).str.strip()
        clean = clean.replace({"": "__MISSING__", "nan": "__MISSING__", "NaN": "__MISSING__", "nd": "__MISSING__"})
        arr = pd.Categorical(clean).codes.astype(float)

    if arr.size == 0:
        return 0.0
    arr = np.asarray(arr, dtype=float)
    arr[~np.isfinite(arr)] = 0.0
    return float(np.var(arr))


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
            variance_records.append((feature, _column_variance_for_prefilter(X[feature])))
        except Exception:
            variance_records.append((feature, 0.0))

    selected_by_variance = {
        feature
        for feature, _ in sorted(variance_records, key=lambda item: item[1], reverse=True)[:max_features]
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
) -> Tuple[float, str]:
    y_values = y.astype(str).to_numpy()
    labels, counts = np.unique(y_values, return_counts=True)
    min_class_count = int(counts.min()) if counts.size else 0
    n_splits = min(int(cv_splits), min_class_count)
    classifier = _normalize_panel_classifier(classifier)

    if len(labels) < 2 or n_splits < 2:
        return np.nan, "skipped_cv_insufficient_class_support"

    if classifier not in {"lr", "rf"}:
        return np.nan, f"skipped_unknown_supervised_classifier:{classifier}"

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores: List[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X_arr, y_values)):
        try:
            if classifier == "rf":
                model = RandomForestClassifier(
                    n_estimators=int(rf_n_estimators),
                    max_features=rf_max_features,
                    min_samples_leaf=int(rf_min_samples_leaf),
                    class_weight=rf_class_weight,
                    random_state=int(random_state) + int(fold_idx),
                    n_jobs=int(rf_n_jobs),
                )
                X_train = X_arr[train_idx]
                X_test = X_arr[test_idx]
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
                X_train = scaler.fit_transform(X_arr[train_idx])
                X_test = scaler.transform(X_arr[test_idx])

                with warnings.catch_warnings():
                    # Treat persistent convergence warnings as a failed score rather
                    # than letting noisy, unstable folds influence panel selection.
                    warnings.filterwarnings("error", category=ConvergenceWarning)
                    model.fit(X_train, y_values[train_idx])

            pred = model.predict(X_test)
            fold_scores.append(float(balanced_accuracy_score(y_values[test_idx], pred)))
        except ConvergenceWarning as exc:
            logger.warning("Feature-panel LR scoring did not converge for one fold: %s", exc)
            return np.nan, f"supervised_scoring_failed_convergence: {exc}"
        except Exception as exc:
            logger.warning(
                "Feature-panel supervised scoring failed for one fold | classifier=%s | error=%s",
                classifier,
                exc,
            )
            return np.nan, f"supervised_scoring_failed_{classifier}: {exc}"

    if not fold_scores:
        return np.nan, "supervised_scoring_failed_no_valid_folds"

    return float(np.mean(fold_scores)), f"success_{classifier}"


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
        scores["normalized_mutual_info"] = float(normalized_mutual_info_score(y_codes, cluster_labels))

        unique_clusters = np.unique(cluster_labels)
        if 1 < len(unique_clusters) < X_arr.shape[0]:
            if X_arr.shape[0] > max_silhouette_samples:
                rng = np.random.default_rng(random_state)
                subset = rng.choice(X_arr.shape[0], size=max_silhouette_samples, replace=False)
                scores["silhouette"] = float(silhouette_score(X_arr[subset], cluster_labels[subset]))
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
    max_silhouette_samples = int(getattr(config, "feature_panel_max_silhouette_samples", 5000))
    classifier = _normalize_panel_classifier(getattr(config, "feature_panel_classifier", "rf"))

    rf_n_jobs = getattr(config, "feature_panel_rf_n_jobs", None) or getattr(config, "n_jobs", 1)

    X_arr = _encode_dataframe(X_panel)

    # --- Supervised Probe ---
    supervised_score, supervised_status = _score_supervised_balanced_accuracy(
        X_arr=X_arr,
        y=y,
        cv_splits=cv_splits,
        random_state=random_state,
        classifier=classifier,
        lr_max_iter=int(getattr(config, "feature_panel_lr_max_iter", 2000)),
        lr_tol=float(getattr(config, "feature_panel_lr_tol", 1e-4)),
        rf_n_estimators=int(getattr(config, "feature_panel_rf_n_estimators", 300)),
        rf_max_features=getattr(config, "feature_panel_rf_max_features", "sqrt"),
        rf_min_samples_leaf=int(getattr(config, "feature_panel_rf_min_samples_leaf", 1)),
        rf_class_weight=getattr(config, "feature_panel_rf_class_weight", "balanced"),
        rf_n_jobs=int(rf_n_jobs),
    )

    # --- Unsupervised Clustering Diagnostics ---
    unsupervised_scores = _score_unsupervised_clustering(
        X_arr=X_arr,
        y=y,
        random_state=random_state,
        max_silhouette_samples=max_silhouette_samples,
    )

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
        "panel_diagnostics": {
            "cv_splits_used": cv_splits,
            "rf_n_jobs": rf_n_jobs,
            "memory_usage_mb": round(X_panel.memory_usage(deep=True).sum() / (1024**2), 2),
        }
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
        raise ValueError(f"Feature-panel metric '{metric}' could not be computed for any panel")

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
        ranked_disabled = pd.DataFrame({
            "rank": np.arange(1, X_aligned.shape[1] + 1, dtype=int),
            "feature": list(map(str, X_aligned.columns)),
        })
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
            "selection_metric": str(getattr(config, "feature_panel_metric", "balanced_accuracy")),
            "supervised_classifier": _normalize_panel_classifier(getattr(config, "feature_panel_classifier", "lr")),
            "minimum_required_score": float(getattr(config, "feature_panel_min_score", 0.75)),
            "selection_rule": str(getattr(config, "feature_panel_selection_rule", "smallest_passing")),
            "selection_reason": "feature_panel_check_disabled",
            "used_original_central_filtered_matrix": True,
            "selected_feature_names": list(map(str, X_aligned.columns)),
            "artifacts": {
                "feature_panel_dir": str(out_dir),
                "ranked_features_csv": str(ranked_path),
                "panel_scores_csv": str(panel_scores_path),
                "selected_panel_matrix": str(selected_matrix_path),
                "summary_json": str(out_dir / "feature_panel_separability_summary.json"),
            },
        }
        with open(out_dir / "feature_panel_separability_summary.json", "w", encoding="utf-8") as handle:
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

    configured_sizes = _parse_panel_sizes(getattr(config, "feature_panel_sizes", (100, 200, 500, 1000)))
    always_include_full = bool(getattr(config, "feature_panel_always_include_full_filtered", True))

    ranked_features = ranked["feature"].astype(str).tolist()
    original_feature_count = int(X_aligned.shape[1])
    large_threshold = int(getattr(config, "feature_panel_large_feature_threshold", 5000))
    large_max_features = int(getattr(config, "feature_panel_large_max_scoring_features", 5000))
    large_pool_multiplier = int(getattr(config, "feature_panel_large_pool_multiplier", 4))
    score_full_large_matrix = bool(getattr(config, "feature_panel_score_full_large_matrix", False))

    large_feature_scoring_pool_applied = False
    scoring_ranked_features = ranked_features

    if original_feature_count > large_threshold:
        scoring_ranked_features = _variance_prefilter_ranked_features(
            X=X_aligned,
            ranked_features=ranked_features,
            max_features=large_max_features,
            pool_multiplier=large_pool_multiplier,
        )
        large_feature_scoring_pool_applied = len(scoring_ranked_features) < len(ranked_features)
        logger.info(
            "%s large feature set detected; scoring pool reduced from %d to %d retained features "
            "using variance within the statistically ranked candidate pool.",
            stage_name,
            original_feature_count,
            len(scoring_ranked_features),
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

    panel_records: List[Dict[str, Any]] = []

    for panel_size in candidate_sizes:
        feature_source = ranked_features if panel_size > len(scoring_ranked_features) else scoring_ranked_features
        panel_features = [feature for feature in feature_source[:panel_size] if feature in X_aligned.columns]
        if not panel_features:
            continue

        X_panel = X_aligned.loc[:, panel_features].copy()
        score_payload = _score_panel(X_panel=X_panel, y=y_aligned.loc[X_panel.index], config=config)
        panel_records.append(
            {
                "stage_name": stage_name,
                "n_features": int(X_panel.shape[1]),
                "panel_size_requested": int(panel_size),
                **score_payload,
            }
        )

    panel_scores = pd.DataFrame(panel_records)
    panel_scores_path = out_dir / "panel_scores.csv"
    panel_scores.to_csv(panel_scores_path, index=False)

    selection_metric = str(getattr(config, "feature_panel_metric", "balanced_accuracy"))
    min_score = float(getattr(config, "feature_panel_min_score", 0.75))
    selection_rule = str(getattr(config, "feature_panel_selection_rule", "smallest_passing"))
    supervised_classifier = _normalize_panel_classifier(getattr(config, "feature_panel_classifier", "lr"))

    try:
        selected_row, selection_reason = _select_panel_row(
            panel_scores=panel_scores,
            metric=selection_metric,
            min_score=min_score,
            selection_rule=selection_rule,
        )
        selected_n = int(selected_row["n_features"])
        selected_feature_source = ranked_features if selected_n > len(scoring_ranked_features) else scoring_ranked_features
        selected_features = [feature for feature in selected_feature_source[:selected_n] if feature in X_aligned.columns]
        X_selected = X_aligned.loc[:, selected_features].copy()
        status = "success"
        used_original_matrix = selected_n == int(X_aligned.shape[1])
    except Exception as exc:
        logger.warning(
            "%s feature-panel separability scoring failed; passing through central filtered matrix: %s",
            stage_name,
            exc,
        )
        selected_n = int(X_aligned.shape[1])
        selected_features = list(X_aligned.columns)
        X_selected = X_aligned.copy()
        selection_reason = f"scoring_failed_pass_through: {exc}"
        status = "fallback_pass_through"
        used_original_matrix = True

    selected_matrix_path = out_dir / "selected_panel_matrix.csv"
    X_selected.to_csv(selected_matrix_path)

    summary = {
        "stage_name": stage_name,
        "status": status,
        "input_features": int(X_aligned.shape[1]),
        "selected_features": int(X_selected.shape[1]),
        "candidate_panel_sizes": candidate_sizes,
        "large_feature_scoring_pool_applied": bool(large_feature_scoring_pool_applied),
        "scoring_pool_features": int(scoring_feature_count),
        "original_filtered_features": int(original_feature_count),
        "score_full_large_matrix": bool(score_full_large_matrix),
        "selection_metric": selection_metric,
        "supervised_classifier": supervised_classifier,
        "minimum_required_score": float(min_score),
        "selection_rule": selection_rule,
        "selection_reason": selection_reason,
        "used_original_central_filtered_matrix": bool(used_original_matrix),
        "selected_feature_names": selected_features,
        "artifacts": {
            "feature_panel_dir": str(out_dir),
            "ranked_features_csv": str(ranked_path),
            "panel_scores_csv": str(panel_scores_path),
            "selected_panel_matrix": str(selected_matrix_path),
            "summary_json": str(out_dir / "feature_panel_separability_summary.json"),
        },
    }

    with open(out_dir / "feature_panel_separability_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=_json_default)

    logger.info(
        "%s feature-panel separability complete | selected_features=%d / %d | reason=%s",
        stage_name,
        int(X_selected.shape[1]),
        int(X_aligned.shape[1]),
        selection_reason,
    )

    return {
        "selected_matrix": X_selected,
        "selected_features": selected_features,
        "ranked_features": ranked,
        "panel_scores": panel_scores,
        "summary": summary,
    }

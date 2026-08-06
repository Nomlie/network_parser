# network_parser/model_selector.py
from __future__ import annotations

import os
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from joblib import Parallel, delayed

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    cross_val_score,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.cluster import KMeans

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.exceptions import ConvergenceWarning

try:
    from network_parser.utils import progress_iter, resolve_effective_n_jobs
except ImportError:  # pragma: no cover - package vs source-tree layout
    try:
        from utils import progress_iter, resolve_effective_n_jobs  # type: ignore
    except ImportError:  # pragma: no cover

        def progress_iter(iterable, **kwargs):  # type: ignore
            return iterable

        def resolve_effective_n_jobs(config, *, override=None, minimum_tasks=1):  # type: ignore
            cpu = max(1, os.cpu_count() or 1)
            if override is not None:
                req = int(override)
            else:
                req = (
                    int(getattr(config, "n_jobs", 1) or 1) if config is not None else 1
                )
            if req < 0:
                return cpu
            return max(1, req)


def _probe_one(
    name: str,
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: int = 5,
    groups: Optional[np.ndarray] = None,
) -> Tuple[str, float, int, str]:
    score, n_splits, err = _cv_score(
        estimator,
        X,
        y,
        cv_splits=cv_splits,
        groups=groups,
        return_n_splits=True,
        return_error=True,
    )
    return name, score, n_splits, err


def _basic_stats(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    n_samples, n_features = X.shape
    missing_frac = np.isnan(X).mean() if np.isnan(X).any() else 0.0
    zeros = (X == 0).sum() if np.issubdtype(X.dtype, np.number) else 0
    sparsity = (
        float(zeros) / (n_samples * n_features) if n_samples * n_features > 0 else 0.0
    )

    counts = Counter(y)
    max_c = max(counts.values())
    min_c = min(counts.values())
    imbalance_ratio = max_c / max(1, min_c)

    return {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "missing_frac": float(missing_frac),
        "sparsity": float(sparsity),
        "class_counts": dict(counts),
        "imbalance_ratio": float(imbalance_ratio),
        "n_classes": int(len(counts)),
    }


def _cluster_scores(X: np.ndarray, n_clusters: int) -> Dict[str, float]:
    """Optional unsupervised diagnostics only — never used for model selection.

    Gated by ``selector_run_clustering_diagnostics`` (default False) so these
    expensive scores are not produced without an explicit consumer.
    """
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
    labels = km.fit_predict(Xs)

    scores = {}
    try:
        scores["silhouette"] = float(silhouette_score(Xs, labels))
    except Exception:
        scores["silhouette"] = float("nan")
    try:
        scores["calinski_harabasz"] = float(calinski_harabasz_score(Xs, labels))
    except Exception:
        scores["calinski_harabasz"] = float("nan")
    try:
        scores["davies_bouldin"] = float(davies_bouldin_score(Xs, labels))
    except Exception:
        scores["davies_bouldin"] = float("nan")

    return scores


def _cv_score(
    estimator,
    X,
    y,
    cv_splits: int = 5,
    scoring: str = "balanced_accuracy",
    groups: Optional[np.ndarray] = None,
    *,
    return_n_splits: bool = False,
    return_error: bool = False,
):
    """
    Cross-validated probe score for model selection.

    Uses balanced accuracy by default because AMR phenotype matrices are often
    class-imbalanced. Keeps cross_val_score single-threaded (n_jobs=1) so outer
    probe Parallel does not nest worker pools.

    Failures are recorded as NaN with an error string (not silent unexplained NaN).
    """
    class_counts = Counter(y)

    def _pack(score, n_splits, err=""):
        if return_error and return_n_splits:
            return score, int(n_splits), err
        if return_n_splits:
            return score, int(n_splits)
        if return_error:
            return score, err
        return score

    if not class_counts:
        return _pack(float("nan"), 0, "empty_labels")

    min_class_count = min(class_counts.values())
    n_splits = min(cv_splits, min_class_count)
    group_values: Optional[np.ndarray] = None
    if groups is not None:
        group_values = np.asarray(groups).astype(str)
        if group_values.shape[0] != len(y):
            return _pack(float("nan"), 0, "group_length_mismatch")
        distinct_groups_per_class = [
            len(set(group_values[np.asarray(y) == label])) for label in class_counts
        ]
        if distinct_groups_per_class:
            n_splits = min(n_splits, min(distinct_groups_per_class))

    if n_splits < 2:
        return _pack(float("nan"), int(n_splits), "insufficient_class_support_for_cv")

    cv = (
        StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        if group_values is not None
        else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    )

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            scores = cross_val_score(
                estimator,
                X,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                groups=group_values,
            )
        mean_score = float(np.mean(scores))
        return _pack(mean_score, int(n_splits), "")
    except Exception as exc:
        return _pack(float("nan"), int(n_splits), f"{type(exc).__name__}: {exc}")


def _dt_probe_params(config: Any = None) -> Dict[str, Any]:
    """Match DecisionTreeBranch constrained configuration for probes and final fit."""
    max_depth = getattr(config, "max_depth", 12) if config is not None else 12
    if max_depth is None:
        max_depth = 12
    return {
        "max_depth": int(max_depth),
        "min_samples_split": int(
            getattr(config, "min_samples_split", 4) if config is not None else 4
        ),
        "min_samples_leaf": max(
            2, int(getattr(config, "min_samples_leaf", 2) if config is not None else 2)
        ),
        "random_state": int(
            getattr(config, "random_state", 42) if config is not None else 42
        ),
    }


def probe_models(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_jobs: int = 1,
    cv_splits: int = 5,
    config: Any = None,
    groups: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, str]]:
    """
    Fast cross-validated probes for supported downstream classifiers.

    Returns (probe_scores, actual_cv_splits_per_probe, probe_errors).

    Nested parallelism is avoided: outer Parallel uses ``n_jobs`` and every
    estimator (including RF) is forced to ``n_jobs=1`` inside probes.
    DT probe uses the same constrained tree hyperparameters as final DT fitting.
    """
    probes: Dict[str, float] = {}
    actual_splits: Dict[str, int] = {}
    probe_errors: Dict[str, str] = {}

    impute_strategy = (
        str(
            getattr(config, "genotype_impute_strategy", "baseline")
            if config is not None
            else "baseline"
        )
        .strip()
        .lower()
    )
    if impute_strategy == "none":
        imputer = None
    elif impute_strategy == "feature_mode":
        imputer = SimpleImputer(strategy="most_frequent")
    else:
        imputer = SimpleImputer(
            strategy="constant",
            fill_value=float(
                getattr(config, "genotype_impute_constant", 0.0)
                if impute_strategy == "constant" and config is not None
                else 0.0
            ),
        )

    def _pipeline(estimator: Any, *, scale: bool = False) -> Pipeline:
        steps: List[Tuple[str, Any]] = []
        if imputer is not None:
            steps.append(("train_fold_imputer", imputer))
        if scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(("clf", estimator))
        return Pipeline(steps)

    lr = _pipeline(
        LogisticRegression(max_iter=2000, n_jobs=None, solver="lbfgs", tol=1e-4),
        scale=True,
    )

    linsvc = _pipeline(LinearSVC(C=1.0, tol=1e-3, dual="auto"), scale=True)

    svc_rbf = _pipeline(
        SVC(kernel="rbf", C=1.0, gamma="scale", probability=False),
        scale=True,
    )

    dt_params = _dt_probe_params(config)
    dt = _pipeline(
        DecisionTreeClassifier(
            criterion="gini",
            max_depth=dt_params["max_depth"],
            min_samples_split=dt_params["min_samples_split"],
            min_samples_leaf=dt_params["min_samples_leaf"],
            random_state=dt_params["random_state"],
        )
    )

    # Always single-thread estimators inside probes to respect outer budget.
    probe_specs: List[Tuple[str, Any]] = [
        ("LR", lr),
        ("LinearSVC", linsvc),
        ("SVC_RBF", svc_rbf),
        (
            "RF",
            _pipeline(
                RandomForestClassifier(
                    n_estimators=300,
                    max_features="sqrt",
                    n_jobs=1,
                    random_state=42,
                )
            ),
        ),
        ("DT", dt),
        (
            "MLP_small",
            _pipeline(
                MLPClassifier(
                    hidden_layer_sizes=(64,),
                    max_iter=1000,
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    random_state=42,
                ),
                scale=True,
            ),
        ),
    ]

    n_probe_jobs = max(1, min(int(n_jobs), len(probe_specs)))

    if n_probe_jobs <= 1:
        for name, estimator in progress_iter(
            probe_specs, desc="Model screening probes", unit="model", leave=False
        ):
            score, n_splits_used, err = _cv_score(
                estimator,
                X,
                y,
                cv_splits=cv_splits,
                groups=groups,
                return_n_splits=True,
                return_error=True,
            )
            probes[name] = score
            actual_splits[name] = int(n_splits_used)
            if err:
                probe_errors[name] = err
        return probes, actual_splits, probe_errors

    probe_results = Parallel(n_jobs=n_probe_jobs)(
        delayed(_probe_one)(name, estimator, X, y, cv_splits, groups)
        for name, estimator in progress_iter(
            probe_specs,
            desc="Model screening probes",
            unit="model",
            leave=False,
        )
    )
    for name, score, n_splits_used, err in probe_results:
        probes[name] = score
        actual_splits[name] = int(n_splits_used)
        if err:
            probe_errors[name] = err
    return probes, actual_splits, probe_errors


def _normalize_algo_name(name: str) -> str:
    mapping = {
        "LinearSVC": "SVC",
        "SVC_RBF": "SVC",
        "MLP_small": "MLP",
    }
    return mapping.get(str(name).strip(), str(name).strip())


def _scored_algo_list(probes: Dict[str, float]) -> List[Tuple[str, float]]:
    scored: List[Tuple[str, float]] = []
    for algo, score in probes.items():
        if algo == "delta_nonlinear_minus_linear":
            continue
        try:
            fscore = float(score)
        except Exception:
            continue
        if np.isfinite(fscore):
            scored.append((_normalize_algo_name(algo), fscore))
    # Keep best score per normalized name
    best: Dict[str, float] = {}
    for algo, score in scored:
        if algo not in best or score > best[algo]:
            best[algo] = score
    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
    return ranked


def _rank_candidates_from_probes(
    probes: Dict[str, float],
    *,
    top_k: int = 2,
    max_margin: float = 0.05,
) -> List[str]:
    """
    Shortlist algorithms within top_k of finite scores AND within max_margin of best.

    Not every finite probe enters the candidate list.
    """
    ranked = _scored_algo_list(probes)
    if not ranked:
        return []
    best_score = ranked[0][1]
    k = max(1, int(top_k))
    out: List[str] = []
    for algo, score in ranked[:k]:
        if (best_score - score) <= float(max_margin) + 1e-12:
            out.append(algo)
    # Always include the best even if margin is 0
    if ranked[0][0] not in out:
        out.insert(0, ranked[0][0])
    return out


def _dt_meets_candidate_rule(
    probes: Dict[str, float],
    *,
    top_k: int,
    max_margin: float,
) -> bool:
    ranked = _scored_algo_list(probes)
    if not ranked:
        return False
    best_score = ranked[0][1]
    k = max(1, int(top_k))
    top = ranked[:k]
    for algo, score in top:
        if algo == "DT" and (best_score - score) <= float(max_margin) + 1e-12:
            return True
    return False


def recommend_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_jobs: int = 1,
    cv_splits: int = 5,
    config: Any = None,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Recommend a classifier and return a ranked shortlist suitable for
    downstream branch decisions.

    On total probe failure this returns ``selector_status='failed_no_finite_probe_scores'``
    with ``recommendation=None`` — callers must not silently substitute RF.

    ``candidate_ranked`` contains only algorithms within configured top-k and
    score margin of the best probe (not every finite probe).
    ``dt_candidate`` is True only when DT is selected or meets the same top-k/margin rule.
    """
    top_k = 2
    max_margin = 0.05
    linear_high = 0.85
    delta_ignore = 0.03
    delta_prefer = 0.05
    run_clustering = False
    if config is not None:
        n_jobs = resolve_effective_n_jobs(
            config, override=n_jobs if n_jobs != 1 else None, minimum_tasks=2
        )
        cv_splits = int(
            getattr(config, "ml_threshold_cv_splits", cv_splits) or cv_splits
        )
        top_k = int(getattr(config, "decision_tree_candidate_top_k", top_k))
        max_margin = float(
            getattr(config, "decision_tree_candidate_max_margin", max_margin)
        )
        linear_high = float(getattr(config, "selector_linear_score_high", linear_high))
        delta_ignore = float(
            getattr(config, "selector_nonlinear_delta_ignore", delta_ignore)
        )
        delta_prefer = float(
            getattr(config, "selector_nonlinear_delta_prefer", delta_prefer)
        )
        run_clustering = bool(
            getattr(config, "selector_run_clustering_diagnostics", False)
        )

    desc = _basic_stats(X, y)
    clus: Dict[str, float] = {}
    if run_clustering:
        try:
            clus = _cluster_scores(X, n_clusters=max(2, desc["n_classes"]))
        except Exception as exc:
            clus = {"error": str(exc)}  # type: ignore[dict-item]

    probes, actual_cv_splits, probe_errors = probe_models(
        X,
        y,
        n_jobs=n_jobs,
        cv_splits=cv_splits,
        config=config,
        groups=groups,
    )

    finite_probe_scores = [
        float(v)
        for k, v in probes.items()
        if k != "delta_nonlinear_minus_linear"
        and isinstance(v, (int, float, np.integer, np.floating))
        and np.isfinite(float(v))
    ]
    if not finite_probe_scores:
        return {
            "recommendation": None,
            "candidate_ranked": [],
            "dt_candidate": False,
            "recommended_interpretable_models": [],
            "rationale": [
                "No finite cross-validated probe scores were obtained; automatic selection failed. "
                "Do not default to RF; inspect label support / feasible CV folds / probe_errors."
            ],
            "dataset_summary": desc,
            "clustering_scores": clus,
            "clustering_diagnostics_optional": True,
            "probe_scores": {
                "LR": probes.get("LR", np.nan),
                "LinearSVC": probes.get("LinearSVC", np.nan),
                "SVC_RBF": probes.get("SVC_RBF", np.nan),
                "RF": probes.get("RF", np.nan),
                "DT": probes.get("DT", np.nan),
                "MLP_small": probes.get("MLP_small", np.nan),
                "delta_nonlinear_minus_linear": np.nan,
            },
            "probe_errors": probe_errors,
            "actual_cv_splits": actual_cv_splits,
            "requested_cv_splits": int(cv_splits),
            "selector_status": "failed_no_finite_probe_scores",
            "error": "failed_no_finite_probe_scores",
            "selector_rule_thresholds": {
                "linear_score_high": linear_high,
                "nonlinear_delta_ignore": delta_ignore,
                "nonlinear_delta_prefer": delta_prefer,
                "candidate_top_k": top_k,
                "candidate_max_margin": max_margin,
            },
        }

    linear_candidates = [probes.get("LR", np.nan), probes.get("LinearSVC", np.nan)]
    linear_candidates = [float(x) for x in linear_candidates if np.isfinite(float(x))]
    linear_score = float(np.mean(linear_candidates)) if linear_candidates else np.nan
    nonlinear_score = probes.get("SVC_RBF", np.nan)
    rf_score = probes.get("RF", np.nan)
    mlp_score = probes.get("MLP_small", np.nan)
    dt_score = probes.get("DT", np.nan)

    delta_nonlinear = (
        nonlinear_score - linear_score
        if (not np.isnan(nonlinear_score) and not np.isnan(linear_score))
        else np.nan
    )

    n, p = desc["n_samples"], desc["n_features"]
    large_data = (n >= 2000) or (p >= 1000)
    very_sparse = desc["sparsity"] > 0.8
    noisy_or_sparse = (desc["missing_frac"] > 0.0) or very_sparse
    small_medium = n < 10000

    rationale: List[str] = []

    if (
        not np.isnan(linear_score)
        and (linear_score >= linear_high)
        and (np.isnan(delta_nonlinear) or delta_nonlinear < delta_ignore)
    ):
        rec = "LR"
        rationale.append(
            f"High linear probe balanced accuracy (≥{linear_high}) with nonlinear gain "
            f"<{delta_ignore}."
        )
    elif (
        not np.isnan(delta_nonlinear)
        and (delta_nonlinear >= delta_prefer)
        and small_medium
    ):
        if (
            large_data
            and (not np.isnan(mlp_score))
            and (mlp_score >= nonlinear_score - 0.01)
        ):
            rec = "MLP"
            rationale.append(
                f"Nonlinear boundary indicated (delta≥{delta_prefer}); MLP preferred on larger matrices."
            )
        else:
            rec = "SVC"
            rationale.append(
                f"Nonlinear boundary indicated (delta≥{delta_prefer}) on a small-to-medium matrix."
            )
    elif noisy_or_sparse or (
        max(
            rf_score if np.isfinite(rf_score) else -np.inf,
            dt_score if np.isfinite(dt_score) else -np.inf,
        )
        >= max(
            linear_score if np.isfinite(linear_score) else -np.inf,
            nonlinear_score if np.isfinite(nonlinear_score) else -np.inf,
            mlp_score if np.isfinite(mlp_score) else -np.inf,
        )
        - 0.01
    ):
        rec = (
            "RF"
            if (
                np.isfinite(rf_score)
                and (not np.isfinite(dt_score) or rf_score >= dt_score)
            )
            else "DT"
        )
        rationale.append(
            "Sparse/noisy pattern or competitive tree-family performance favored a tree-based method."
        )
    else:
        best = {
            "LR": linear_score,
            "SVC": nonlinear_score,
            "RF": rf_score,
            "DT": dt_score,
            "MLP": mlp_score,
        }
        rec = max(best, key=lambda k: (best[k] if not np.isnan(best[k]) else -np.inf))
        rationale.append(
            "Selected the best cross-validated probe among supported candidates."
        )

    probe_scores = {
        "LR": probes.get("LR", np.nan),
        "LinearSVC": probes.get("LinearSVC", np.nan),
        "SVC_RBF": probes.get("SVC_RBF", np.nan),
        "RF": probes.get("RF", np.nan),
        "DT": probes.get("DT", np.nan),
        "MLP_small": probes.get("MLP_small", np.nan),
        "delta_nonlinear_minus_linear": delta_nonlinear,
    }

    candidate_ranked = _rank_candidates_from_probes(
        probe_scores, top_k=top_k, max_margin=max_margin
    )
    if rec not in candidate_ranked:
        candidate_ranked.insert(0, rec)

    dt_candidate = bool(rec == "DT") or _dt_meets_candidate_rule(
        probe_scores, top_k=top_k, max_margin=max_margin
    )
    # Interpretable: DT / LR / RF only — never auto-label SVC/MLP as interpretable
    interpretable_ranked = [a for a in candidate_ranked if a in {"DT", "LR", "RF"}]

    return {
        "recommendation": rec,
        "candidate_ranked": candidate_ranked,
        "dt_candidate": dt_candidate,
        "recommended_interpretable_models": interpretable_ranked,
        "rationale": rationale,
        "dataset_summary": desc,
        "clustering_scores": clus,
        "clustering_diagnostics_optional": True,
        "clustering_diagnostics_ran": bool(run_clustering),
        "probe_scores": probe_scores,
        "probe_errors": probe_errors,
        "actual_cv_splits": actual_cv_splits,
        "requested_cv_splits": int(cv_splits),
        "selector_status": "ok",
        "dt_probe_params": _dt_probe_params(config),
        "selector_rule_thresholds": {
            "linear_score_high": linear_high,
            "nonlinear_delta_ignore": delta_ignore,
            "nonlinear_delta_prefer": delta_prefer,
            "candidate_top_k": top_k,
            "candidate_max_margin": max_margin,
            "meanings": {
                "linear_score_high": "Recommend LR when mean linear probe ≥ this and nonlinear gain small",
                "nonlinear_delta_ignore": "Nonlinear gain below this is treated as negligible for LR",
                "nonlinear_delta_prefer": "Nonlinear gain at/above this prefers SVC/MLP over linear",
                "candidate_top_k": "Shortlist size for branch candidates / DT candidacy",
                "candidate_max_margin": "Max score gap from best probe for shortlist membership",
            },
        },
    }


if __name__ == "__main__":
    import argparse

    def _infer_sep(filename: str) -> str:
        low = filename.lower()
        if low.endswith(".csv"):
            return ","
        if low.endswith(".tsv") or low.endswith(".txt"):
            return "\t"
        return ","

    def _read_matrix_with_labels(path: str) -> tuple[pd.DataFrame, np.ndarray]:
        sep = _infer_sep(path)
        df = pd.read_csv(path, sep=sep, header=0, dtype=str)
        if df.shape[1] < 3:
            raise ValueError(
                "Input matrix must contain sample column, label column, and feature columns."
            )

        label_candidates = {"label", "class", "group", "y"}
        cols_lower = {c.lower(): c for c in df.columns}
        label_col = None
        for k in label_candidates:
            if k in cols_lower:
                label_col = cols_lower[k]
                break
        if label_col is None:
            raise ValueError(
                "No label column found. Use one of: label, class, group, y."
            )

        row_titles_col = df.columns[0]
        y = df[label_col].astype(str).to_numpy()

        feature_cols = [c for c in df.columns if c not in (row_titles_col, label_col)]
        X_raw = df[feature_cols].copy()

        for c in X_raw.columns:
            X_raw[c] = X_raw[c].apply(
                lambda v: (str(v).strip() if pd.notna(v) else np.nan)
            )

        X_codes = []
        for c in X_raw.columns:
            codes, _ = pd.factorize(X_raw[c].astype(str), sort=True)
            codes = codes.astype(np.int64) + 1
            X_codes.append(codes)

        X_mat = np.column_stack(X_codes).astype(np.float64)
        X_df = pd.DataFrame(X_mat, columns=X_raw.columns)
        return X_df, y

    parser = argparse.ArgumentParser(
        description="Classifier recommendation for genomic feature matrices."
    )
    parser.add_argument("-i", "--input_folder", default="input")
    parser.add_argument("-o", "--output_folder", default="output")
    parser.add_argument("-f", "--in_file", required=True)
    args = parser.parse_args()

    in_path = os.path.join(args.input_folder, args.in_file)
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Input file not found: {in_path}")

    os.makedirs(args.output_folder, exist_ok=True)
    base, _ = os.path.splitext(os.path.basename(args.in_file))
    out_path = os.path.join(args.output_folder, f"{base}_recommendations.txt")

    X_df, y = _read_matrix_with_labels(in_path)
    X = X_df.to_numpy()

    result = recommend_classifier(X, y)

    lines = []
    lines.append("# Matrix-Based Classifier Recommendation")
    lines.append(f"Recommendation: {result['recommendation']}")
    lines.append(f"Candidate ranked: {', '.join(result['candidate_ranked'])}")
    lines.append(f"DT candidate: {result['dt_candidate']}")
    lines.append("")
    lines.append("Rationale:")
    for r in result["rationale"]:
        lines.append(f"- {r}")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"Saved recommendation report to: {out_path}")

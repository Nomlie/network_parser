# network_parser/model_selector.py
from __future__ import annotations

import os
import warnings
from collections import Counter
from typing import Any, Dict, List, Tuple

from joblib import Parallel, delayed

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.exceptions import ConvergenceWarning

try:
    from network_parser.utils import progress_iter
except Exception:  # pragma: no cover
    try:
        from utils import progress_iter  # type: ignore
    except Exception:  # pragma: no cover
        progress_iter = lambda iterable, **kwargs: iterable  # type: ignore


def _probe_one(name: str, estimator, X: np.ndarray, y: np.ndarray) -> Tuple[str, float]:
    return name, _cv_score(estimator, X, y)


def _basic_stats(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    n_samples, n_features = X.shape
    missing_frac = np.isnan(X).mean() if np.isnan(X).any() else 0.0
    zeros = (X == 0).sum() if np.issubdtype(X.dtype, np.number) else 0
    sparsity = float(zeros) / (n_samples * n_features) if n_samples * n_features > 0 else 0.0

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
) -> float:
    """
    Cross-validated probe score for model selection.

    Uses balanced accuracy by default because AMR phenotype matrices are often
    class-imbalanced. Also avoids nested parallel oversubscription by keeping
    cross_val_score single-threaded here; model-level estimators can still use
    their own configured parallelism.
    """
    class_counts = Counter(y)

    if not class_counts:
        return float("nan")

    min_class_count = min(class_counts.values())
    n_splits = min(cv_splits, min_class_count)

    if n_splits < 2:
        return float("nan")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

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
            )
        return float(np.mean(scores))
    except Exception:
        return float("nan")

def probe_models(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Fast cross-validated probes for supported downstream classifiers.
    """
    probes: Dict[str, float] = {}

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, n_jobs=None, solver="lbfgs", tol=1e-4)),
    ])

    linsvc = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(C=1.0, tol=1e-3, dual="auto")),
    ])

    svc_rbf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=False)),
    ])

    dt = DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
    )

    probe_specs: List[Tuple[str, Any]] = [
        ("LR", lr),
        ("LinearSVC", linsvc),
        ("SVC_RBF", svc_rbf),
        ("RF", RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        )),
        ("DT", dt),
        ("MLP_small", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64,),
                max_iter=1000,
                alpha=1e-4,
                learning_rate_init=1e-3,
                random_state=42,
            )),
        ])),
    ]

    n_probe_jobs = min(len(probe_specs), max(1, os.cpu_count() or 1))
    if n_probe_jobs > 1:
        parallel_specs: List[Tuple[str, Any]] = []
        for name, estimator in probe_specs:
            if name == "RF" and isinstance(estimator, RandomForestClassifier):
                parallel_specs.append((
                    name,
                    RandomForestClassifier(
                        n_estimators=300,
                        max_features="sqrt",
                        n_jobs=1,
                        random_state=42,
                    ),
                ))
            else:
                parallel_specs.append((name, estimator))
        probe_specs = parallel_specs

    if n_probe_jobs <= 1:
        for name, estimator in progress_iter(probe_specs, desc="Model screening probes", unit="model", leave=False):
            probes[name] = _cv_score(estimator, X, y)
        return probes

    probe_results = Parallel(n_jobs=n_probe_jobs)(
        delayed(_probe_one)(name, estimator, X, y)
        for name, estimator in progress_iter(
            probe_specs,
            desc="Model screening probes",
            unit="model",
            leave=False,
        )
    )
    return dict(probe_results)


def _normalize_algo_name(name: str) -> str:
    mapping = {
        "LinearSVC": "SVC",
        "SVC_RBF": "SVC",
        "MLP_small": "MLP",
    }
    return mapping.get(str(name).strip(), str(name).strip())


def _rank_candidates_from_probes(probes: Dict[str, float]) -> List[str]:
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

    scored.sort(key=lambda x: x[1], reverse=True)

    ranked: List[str] = []
    seen = set()
    for algo, _ in scored:
        if algo not in seen:
            ranked.append(algo)
            seen.add(algo)
    return ranked


def recommend_classifier(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Recommend a classifier and return a ranked shortlist suitable for
    downstream branch decisions.
    """
    desc = _basic_stats(X, y)
    clus = _cluster_scores(X, n_clusters=max(2, desc["n_classes"]))
    probes = probe_models(X, y)

    finite_probe_scores = [
        float(v)
        for k, v in probes.items()
        if k != "delta_nonlinear_minus_linear"
        and isinstance(v, (int, float, np.integer, np.floating))
        and np.isfinite(float(v))
    ]
    if not finite_probe_scores:
        return {
            "recommendation": "RF",
            "candidate_ranked": ["RF"],
            "dt_candidate": False,
            "recommended_interpretable_models": ["RF"],
            "rationale": [
                "No finite cross-validated probe scores were obtained; automatic selection should be treated as failed."
            ],
            "dataset_summary": desc,
            "clustering_scores": clus,
            "probe_scores": {
                "LR": probes.get("LR", np.nan),
                "LinearSVC": probes.get("LinearSVC", np.nan),
                "SVC_RBF": probes.get("SVC_RBF", np.nan),
                "RF": probes.get("RF", np.nan),
                "DT": probes.get("DT", np.nan),
                "MLP_small": probes.get("MLP_small", np.nan),
                "delta_nonlinear_minus_linear": np.nan,
            },
            "selector_status": "failed_no_finite_probe_scores",
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

    if not np.isnan(linear_score) and (linear_score >= 0.85) and (
        np.isnan(delta_nonlinear) or delta_nonlinear < 0.03
    ):
        rec = "LR"
        rationale.append("High linear probe balanced accuracy with negligible nonlinear gain.")
    elif not np.isnan(delta_nonlinear) and (delta_nonlinear >= 0.05) and small_medium:
        if large_data and (not np.isnan(mlp_score)) and (mlp_score >= nonlinear_score - 0.01):
            rec = "MLP"
            rationale.append("Nonlinear boundary indicated and MLP scales better on larger high-dimensional matrices.")
        else:
            rec = "SVC"
            rationale.append("Nonlinear boundary indicated on a small-to-medium matrix.")
    elif noisy_or_sparse or (
        max(
            rf_score if np.isfinite(rf_score) else -np.inf,
            dt_score if np.isfinite(dt_score) else -np.inf,
        ) >= max(
            linear_score if np.isfinite(linear_score) else -np.inf,
            nonlinear_score if np.isfinite(nonlinear_score) else -np.inf,
            mlp_score if np.isfinite(mlp_score) else -np.inf,
        ) - 0.01
    ):
        rec = "RF" if (rf_score >= dt_score) else "DT"
        rationale.append("Sparse/noisy pattern or competitive tree-family performance favored a tree-based method.")
    else:
        best = {
            "LR": linear_score,
            "SVC": nonlinear_score,
            "RF": rf_score,
            "DT": dt_score,
            "MLP": mlp_score,
        }
        rec = max(best, key=lambda k: (best[k] if not np.isnan(best[k]) else -np.inf))
        rationale.append("Selected the best cross-validated probe among supported candidates.")

    probe_scores = {
        "LR": probes.get("LR", np.nan),
        "LinearSVC": probes.get("LinearSVC", np.nan),
        "SVC_RBF": probes.get("SVC_RBF", np.nan),
        "RF": probes.get("RF", np.nan),
        "DT": probes.get("DT", np.nan),
        "MLP_small": probes.get("MLP_small", np.nan),
        "delta_nonlinear_minus_linear": delta_nonlinear,
    }

    candidate_ranked = _rank_candidates_from_probes(probe_scores)
    if rec not in candidate_ranked:
        candidate_ranked.insert(0, rec)

    interpretable_ranked = [a for a in candidate_ranked if a in {"DT", "LR", "RF", "SVC", "MLP"}]

    return {
        "recommendation": rec,
        "candidate_ranked": candidate_ranked,
        "dt_candidate": ("DT" in candidate_ranked),
        "recommended_interpretable_models": interpretable_ranked,
        "rationale": rationale,
        "dataset_summary": desc,
        "clustering_scores": clus,
        "probe_scores": probe_scores,
    }


if __name__ == "__main__":
    import os
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
            raise ValueError("Input matrix must contain sample column, label column, and feature columns.")

        label_candidates = {"label", "class", "group", "y"}
        cols_lower = {c.lower(): c for c in df.columns}
        label_col = None
        for k in label_candidates:
            if k in cols_lower:
                label_col = cols_lower[k]
                break
        if label_col is None:
            raise ValueError("No label column found. Use one of: label, class, group, y.")

        row_titles_col = df.columns[0]
        y = df[label_col].astype(str).to_numpy()

        feature_cols = [c for c in df.columns if c not in (row_titles_col, label_col)]
        X_raw = df[feature_cols].copy()

        for c in X_raw.columns:
            X_raw[c] = X_raw[c].apply(lambda v: (str(v).strip() if pd.notna(v) else np.nan))

        X_codes = []
        for c in X_raw.columns:
            codes, _ = pd.factorize(X_raw[c].astype(str), sort=True)
            codes = codes.astype(np.int64) + 1
            X_codes.append(codes)

        X_mat = np.column_stack(X_codes).astype(np.float64)
        X_df = pd.DataFrame(X_mat, columns=X_raw.columns)
        return X_df, y

    parser = argparse.ArgumentParser(description="Classifier recommendation for genomic feature matrices.")
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
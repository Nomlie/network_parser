import numpy as np
import pandas as pd
from typing import Dict, Any
from collections import Counter

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB

import warnings
from sklearn.exceptions import ConvergenceWarning


# ------------------------- optional xgboost -------------------------
def _try_make_xgb(random_state: int = 42, n_jobs: int = -1):
    """
    Returns an XGBClassifier instance if xgboost is installed, else None.
    """
    try:
        from xgboost import XGBClassifier
    except Exception:
        return None

    # modest defaults for "probe"
    return XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=1.0,
        gamma=0.0,
        objective="multi:softprob",   # will be overwritten if binary
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=0,
    )


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
        "n_samples": n_samples,
        "n_features": n_features,
        "missing_frac": missing_frac,
        "sparsity": sparsity,
        "class_counts": dict(counts),
        "imbalance_ratio": imbalance_ratio,
        "n_classes": len(counts),
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


def _cv_score(estimator, X, y, cv_splits=5) -> float:
    class_counts = Counter(y)
    min_class_count = min(class_counts.values())
    n_splits = min(cv_splits, max(2, min_class_count))
    if n_splits < 2:
        return float("nan")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            scores = cross_val_score(
                estimator, X, y,
                cv=cv,
                scoring="accuracy",
                n_jobs=-1
            )
        return float(np.mean(scores))
    except Exception:
        return float("nan")


def probe_models(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Fast cross-validated probes for ML-protocol-supported supervised classifiers.
    Restricted to models that can actually be selected downstream.
    """
    probes: Dict[str, float] = {}

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=None, solver="lbfgs"))
    ])

    linsvc = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(C=1.0, tol=1e-3, dual="auto"))
    ])

    svc_rbf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=False))
    ])

    dt = DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )

    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64,),
            max_iter=1000,
            alpha=1e-4,
            learning_rate_init=1e-3,
            random_state=42
        ))
    ])

    probes["LR"] = _cv_score(lr, X, y)
    probes["LinearSVC"] = _cv_score(linsvc, X, y)
    probes["SVC_RBF"] = _cv_score(svc_rbf, X, y)
    probes["RF"] = _cv_score(rf, X, y)
    probes["DT"] = _cv_score(dt, X, y)
    probes["MLP_small"] = _cv_score(mlp, X, y)

    return probes


def remove_empty_columns(dm: pd.DataFrame, thr: float, empty_symbol: str) -> pd.DataFrame:
    """
    Remove feature columns where fraction of empty cells > thr.

    IMPORTANT NOTE (empty_symbol visibility):
      This function MUST be applied BEFORE any factorization/encoding step.
      After factorization, tokens such as "", "nd", "-", "0" become numeric codes,
      so the original empty_symbol is no longer visible.

    Empty is defined STRICTLY by string comparison:
      - NaN
      - stripped value == ""
      - stripped value == empty_symbol (as STRING, even if '0')

    Here dm is expected to contain ONLY feature columns (no ID/label columns).
    """

    if dm.shape[1] == 0:
        return dm

    empty_sym = str(empty_symbol).strip()

    keep_cols = []
    removed = []

    for c in dm.columns:
        s = dm[c]
        is_empty = s.isna()
        s_str = s.astype(str).str.strip()

        is_empty |= (s_str == "")
        if empty_sym != "":
            is_empty |= (s_str == empty_sym)

        frac_empty = float(is_empty.mean())
        if frac_empty <= thr:
            keep_cols.append(c)
        else:
            removed.append((c, frac_empty))

    if removed:
        msg = ", ".join(f"{c}({p:.2f})" for c, p in removed[:10])
        extra = "" if len(removed) <= 10 else f" ... +{len(removed)-10} more"
        print(f"⚠️  Removed {len(removed)} column(s) with empty fraction > {thr}: {msg}{extra}")

    return dm.loc[:, keep_cols].copy()


def recommend_classifier(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    desc = _basic_stats(X, y)
    clus = _cluster_scores(X, n_clusters=max(2, desc["n_classes"]))
    probes = probe_models(X, y)

    linear_score = np.nanmean([probes.get("LR", np.nan), probes.get("LinearSVC", np.nan)])
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
    noisy_or_missing = (desc["missing_frac"] > 0.0) or very_sparse
    small_medium = n < 10000

    rationale = []

    if not np.isnan(linear_score) and (linear_score >= 0.85) and (np.isnan(delta_nonlinear) or delta_nonlinear < 0.03):
        rec = "LR"
        rationale.append("High linear probe accuracy and negligible benefit from nonlinear kernel.")
    elif not np.isnan(delta_nonlinear) and (delta_nonlinear >= 0.05) and small_medium:
        if large_data and (mlp_score >= nonlinear_score - 0.01):
            rec = "MLP"
            rationale.append("Nonlinear boundary indicated; dataset is large/high-dim → MLP scales better.")
        else:
            rec = "SVC"
            rationale.append("Nonlinear boundary indicated on small/medium dataset → SVC (RBF) fits well.")
    elif noisy_or_missing or (max(rf_score, dt_score) >= max(linear_score, nonlinear_score, mlp_score) - 0.01):
        rec = "RF" if (rf_score >= dt_score) else "DT"
        rationale.append("Data appear noisy/sparse or tree models perform competitively → choose robust tree (RF/DT).")
    else:
        best = {
            "LR": linear_score,
            "SVC_RBF": nonlinear_score,
            "RF": rf_score,
            "DT": dt_score,
            "MLP_small": mlp_score,
        }
        rec = max(best, key=lambda k: (best[k] if not np.isnan(best[k]) else -np.inf))
        rationale.append("Selected the best cross-validated probe among supported candidates.")

    return {
        "recommendation": rec,
        "rationale": rationale,
        "dataset_summary": desc,
        "clustering_scores": clus,
        "probe_scores": {
            "LR": probes.get("LR", np.nan),
            "LinearSVC": probes.get("LinearSVC", np.nan),
            "SVC_RBF": probes.get("SVC_RBF", np.nan),
            "RF": probes.get("RF", np.nan),
            "DT": probes.get("DT", np.nan),
            "MLP_small": probes.get("MLP_small", np.nan),
            "delta_nonlinear_minus_linear": delta_nonlinear,
        },
    }


# -------------------------
if __name__ == "__main__":
    import os
    import sys
    import argparse

    def _infer_sep(filename: str) -> str:
        low = filename.lower()
        if low.endswith(".csv"):
            return ","
        if low.endswith(".tsv") or low.endswith(".txt"):
            return "\t"
        return ","

    def _read_matrix_with_labels(path: str, thr: float, empty_symbol: str) -> tuple[pd.DataFrame, np.ndarray]:
        sep = _infer_sep(path)
        df = pd.read_csv(path, sep=sep, header=0, dtype=str)
        if df.shape[1] < 3:
            raise ValueError("Input matrix must have at least: 1st col (row titles), ≥1 feature column, and a label column.")

        label_candidates = {"label", "class", "group", "y"}
        cols_lower = {c.lower(): c for c in df.columns}
        label_col = None
        for k in label_candidates:
            if k in cols_lower:
                label_col = cols_lower[k]
                break
        if label_col is None:
            raise ValueError("No label column found. Please include a column named one of: label, class, group, y.")

        row_titles_col = df.columns[0]
        y = df[label_col].astype(str).to_numpy()

        feature_cols = [c for c in df.columns if c not in (row_titles_col, label_col)]
        X_raw = df[feature_cols].copy()

        # Normalize tokens: strip spaces; keep missing as NaN
        # NOTE: This normalization MUST happen before remove_empty_columns(),
        # otherwise whitespace-only cells may not be detected as empty.
        for c in X_raw.columns:
            X_raw[c] = X_raw[c].apply(
                lambda v: (str(v).strip() if pd.notna(v) else np.nan)
            )

        # Remove empty columns BEFORE factorization (keeps empty_symbol visible)
        if thr < 1.0:
            X_raw = remove_empty_columns(X_raw, thr=thr, empty_symbol=str(empty_symbol))

        # factorize each column independently (symbols -> codes), shift to >=0
        X_codes = []
        for c in X_raw.columns:
            codes, _uniques = pd.factorize(X_raw[c].astype(str), sort=True)
            codes = codes.astype(np.int64) + 1
            X_codes.append(codes)
        X_mat = np.column_stack(X_codes).astype(np.float64)

        X_df = pd.DataFrame(X_mat, columns=X_raw.columns)
        return X_df, y

    parser = argparse.ArgumentParser(description="Matrix-based classifier recommendation (extended).")
    parser.add_argument("-i", "--input_folder", default="input", help="Folder with input matrix file (default: input)")
    parser.add_argument("-o", "--output_folder", default="output", help="Folder to save report (default: output)")
    parser.add_argument("-f", "--in_file", required=True, help="CSV/TSV matrix file name (within input folder)")
    # Empty-field filtering
    parser.add_argument("--empty_symbol", default="", help="Symbol/string representing empty values in feature columns")
    parser.add_argument("--remove_empty_filed", type=float, default=1.0,
                        help="If < 1.0, drop feature columns where empty fraction >= this threshold (range: 0.0..1.0)")

    args = parser.parse_args()

    # Validate remove_empty_filed
    try:
        thr = float(args.remove_empty_filed)
    except Exception:
        raise SystemExit("[ERROR] --remove_empty_filed must be a floating number.")
    if thr < 0.0 or thr > 1.0:
        raise SystemExit("[ERROR] --remove_empty_filed must be within [0.0, 1.0].")

    in_path = os.path.join(args.input_folder, args.in_file)
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Input file not found: {in_path}")

    os.makedirs(args.output_folder, exist_ok=True)
    base, _ = os.path.splitext(os.path.basename(args.in_file))
    out_path = os.path.join(args.output_folder, f"{base}_recommendations.txt")

    X_df, y = _read_matrix_with_labels(in_path, thr=thr, empty_symbol=str(args.empty_symbol))
    X = X_df.to_numpy()

    result = recommend_classifier(X, y)

    lines = []
    lines.append("# Matrix-Based Classifier Recommendation")
    lines.append(f"Input file         : {args.in_file}")
    lines.append(f"Objects (rows)     : {result['dataset_summary']['n_samples']}")
    lines.append(f"Features (columns) : {result['dataset_summary']['n_features']}")
    lines.append(f"Classes            : {result['dataset_summary']['n_classes']}")
    cc = result['dataset_summary']['class_counts']
    lines.append(f"Class counts       : {', '.join(f'{k}={v}' for k, v in cc.items())}")
    lines.append(f"Missing fraction   : {result['dataset_summary']['missing_frac']:.4f}")
    lines.append(f"Sparsity (zeros)   : {result['dataset_summary']['sparsity']:.4f}")
    lines.append("")

    cs = result["clustering_scores"]
    lines.append("Unsupervised clustering scores (KMeans, k≈#classes):")
    lines.append(f"  Silhouette        : {cs['silhouette']:.4f}" if np.isfinite(cs['silhouette']) else "  Silhouette        : nan")
    lines.append(f"  Calinski-Harabasz : {cs['calinski_harabasz']:.4f}" if np.isfinite(cs['calinski_harabasz']) else "  Calinski-Harabasz : nan")
    lines.append(f"  Davies-Bouldin    : {cs['davies_bouldin']:.4f}" if np.isfinite(cs['davies_bouldin']) else "  Davies-Bouldin    : nan")
    lines.append("")

    ps = result["probe_scores"]

    def _fmt(v):
        return f"{v:.4f}" if (v == v and np.isfinite(v)) else "nan"

    lines.append("Supervised probe accuracies (CV):")
    lines.append(f"  LR         : {_fmt(ps['LR'])}")
    lines.append(f"  LinearSVC  : {_fmt(ps['LinearSVC'])}")
    lines.append(f"  SVC_RBF    : {_fmt(ps['SVC_RBF'])}")
    lines.append(f"  RF         : {_fmt(ps['RF'])}")
    lines.append(f"  DT         : {_fmt(ps['DT'])}")
    lines.append(f"  MLP_small  : {_fmt(ps['MLP_small'])}")
    lines.append(f"  KNN        : {_fmt(ps['KNN'])}")
    lines.append(f"  NBayes     : {_fmt(ps['NBayes'])}")
    lines.append(f"  XGBoost    : {_fmt(ps['XGBoost'])}  (requires xgboost)")
    lines.append(f"  Δ(nonlin−lin): {_fmt(ps['delta_nonlinear_minus_linear'])}")
    lines.append("")

    lines.append(f"RECOMMENDATION: {result['recommendation']}")
    lines.append("Rationale:")
    for r in result["rationale"]:
        lines.append(f"  - {r}")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"✓ Recommendation written to: {out_path}")
    print("\n".join(lines))

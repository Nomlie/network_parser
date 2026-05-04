# network_parser/decision_tree_branch.py
"""
Decision Tree discovery branch (NetworkParser).

Updated role in pipeline
------------------------
This branch now assumes that statistical filtering has already happened
upstream in the central feature-filtering stage.

Default flow:
    aligned + centrally filtered matrix
        -> decision tree
        -> tree hierarchy extraction
        -> post-tree confidence scoring
        -> path-based epistatic interaction mining

Backward compatibility:
    A legacy internal prefilter is still available, but it is OFF by default.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.metrics import accuracy_score, mutual_info_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from statsmodels.stats.multitest import multipletests

try:
    from network_parser.config import NetworkParserConfig
except Exception:  # pragma: no cover
    from config import NetworkParserConfig  # type: ignore


logger = logging.getLogger(__name__)


def normalize_labels(
    labels: pd.Series,
    drop_missing: bool = True,
    lowercase: bool = False,
) -> pd.Series:
    """
    Normalize labels using the same general strategy used elsewhere in the pipeline.
    """
    if not isinstance(labels, pd.Series):
        raise TypeError("labels must be a pandas Series")

    clean = labels.astype(str).str.strip()
    missing_tokens = {"", "-", "NA", "N/A", "None", "nan", "NaN"}
    clean = clean.replace(missing_tokens, pd.NA)
    clean = clean.str.replace("-", "_", regex=False)

    if lowercase:
        clean = clean.str.lower()

    if drop_missing:
        clean = clean[~clean.isna()]

    return clean


def log_feature_summary(name: str, features: List[str], max_show: int = 3) -> None:
    n = len(features)
    if n == 0:
        logger.info("%s: 0 features", name)
        return

    if n <= max_show:
        logger.info("%s: %d features -> %s", name, n, ", ".join(map(str, features)))
    else:
        logger.info(
            "%s: %d features -> %s ... +%d more",
            name,
            n,
            ", ".join(map(str, features[:max_show])),
            n - max_show,
        )


@dataclass
class DecisionTreeBranchArtifacts:
    rules_txt: str = "decision_tree_rules.txt"
    confidence_json: str = "feature_confidence.json"
    interactions_json: str = "epistatic_interactions.json"


class DecisionTreeBranch:
    """
    Decision tree interpretability branch.

    Output dict structure:
      {
        "discovered_features": [...],
        "root_features": [...],
        "branch_features": [...],
        "decision_trees": {"accuracy": float, "rules": str, "n_classes": int},
        "feature_confidence": {feature: {...}},
        "epistatic_interactions": [ {...}, ... ],
        "prefiltered_features": [...]
      }
    """

    def __init__(
        self,
        config: NetworkParserConfig,
        artifacts: Optional[DecisionTreeBranchArtifacts] = None,
    ):
        self.config = config
        self.artifacts = artifacts or DecisionTreeBranchArtifacts()
        np.random.seed(int(getattr(self.config, "random_state", 42)))

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------
    def run(
        self,
        data: pd.DataFrame,
        labels: pd.Series,
        all_features: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        prefiltered_input: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the decision-tree branch on a sample x feature matrix.

        Parameters
        ----------
        data
            Input feature matrix. In the updated architecture this should already
            be centrally filtered.
        labels
            Sample labels aligned to data.index.
        all_features
            Optional subset of features to consider.
        output_dir
            Output directory for tree rules / confidence / interactions.
        prefiltered_input
            If True, trust incoming matrix as already filtered and skip the
            legacy internal prefilter.
            If False, apply the legacy internal prefilter for backward compatibility.
        """
        self._validate_inputs(data, labels)

        if all_features is None:
            all_features = list(data.columns)

        valid_features = [f for f in all_features if f in data.columns]
        if not valid_features:
            raise ValueError("No valid features provided for decision-tree discovery.")

        if len(valid_features) < len(all_features):
            logger.warning("Some requested features were not found in data and were ignored.")

        # Defensive sample alignment
        if not data.index.equals(labels.index):
            common = data.index.intersection(labels.index)
            if common.empty:
                raise ValueError("No common indices between data and labels.")
            data = data.loc[common].copy()
            labels = labels.loc[common].copy()
            logger.info("Aligned data and labels to %d common samples.", len(common))

        log_feature_summary("Decision-tree input features", valid_features)

        if prefiltered_input:
            prefiltered = valid_features.copy()
            logger.info(
                "DecisionTreeBranch received centrally filtered input; skipping internal prefilter."
            )
        else:
            logger.info(
                "DecisionTreeBranch running legacy internal prefilter for backward compatibility."
            )
            prefiltered = self._prefilter_features(
                data=data,
                labels=labels,
                features=valid_features,
            )

            if not prefiltered:
                logger.warning(
                    "Internal prefilter retained no features; falling back to supplied valid features."
                )
                prefiltered = valid_features.copy()

        log_feature_summary("Decision-tree feature set used", prefiltered)

        X = data.loc[:, prefiltered].copy()

        # NaN handling
        total_nan = int(X.isna().sum().sum())
        if total_nan > 0:
            logger.info("Imputing NaNs (%d) -> treating as baseline (0).", total_nan)
            X = X.fillna(0)

        # Ensure numeric tree input
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        if int(X.isna().sum().sum()) > 0:
            logger.info(
                "Coercion produced NaNs; filling remaining missing values with 0 before tree fitting."
            )
            X = X.fillna(0)

        X = X.astype(np.float64)

        # Drop invariant / near-monomorphic columns after final coercion
        kept_features = self._drop_post_impute_monomorphic(X)
        X = X.loc[:, kept_features].copy()
        prefiltered = kept_features

        if X.shape[1] == 0:
            raise ValueError(
                "No usable features remain for decision-tree fitting after post-imputation cleanup."
            )

        # Encode labels
        labels = normalize_labels(labels, drop_missing=True, lowercase=False)
        if not X.index.equals(labels.index):
            common = X.index.intersection(labels.index)
            X = X.loc[common].copy()
            labels = labels.loc[common].copy()

        if labels.nunique() < 2:
            raise ValueError("Decision tree requires at least 2 label classes.")

        le = LabelEncoder()
        y = le.fit_transform(labels.astype(str))
        n_classes = int(len(le.classes_))
        logger.info("Labels encoded for DT branch: n_classes=%d", n_classes)

        # Build tree
        dt = self._build_tree(X, y)

        # Analyze hierarchy
        analysis = self._analyze_tree_structure(dt, feature_names=list(X.columns))

        # Confidence scoring
        confidence = self._compute_confidence(dt, X, y, analysis)

        # Interaction mining
        interactions = self._mine_epistatic_interactions(
            dt=dt,
            feature_names=list(X.columns),
            X=X,
            y=y,
        )

        results: Dict[str, Any] = {
            "discovered_features": analysis["features"],
            "root_features": analysis["root_features"],
            "branch_features": analysis["branch_features"],
            "decision_trees": {
                "training_accuracy": float(accuracy_score(y, dt.predict(X))),
                "rules": export_text(dt, feature_names=list(X.columns)),
                "n_classes": n_classes,
            },
            "feature_confidence": confidence,
            "epistatic_interactions": interactions,
            "prefiltered_features": prefiltered,
        }

        self._export_results(results, output_dir)
        self._print_summary(results)

        return results

    # ------------------------------------------------------------------
    # Legacy internal prefilter
    # ------------------------------------------------------------------
    def _prefilter_features(
        self,
        data: pd.DataFrame,
        labels: pd.Series,
        features: List[str],
        alpha: Optional[float] = None,
    ) -> List[str]:
        """
        Legacy internal feature prefilter.

        Kept only for backward compatibility. In the updated architecture,
        central filtering should happen before this branch is called.
        """
        alpha = float(alpha if alpha is not None else getattr(self.config, "prefilter_alpha", 0.05))
        min_nonmissing = float(getattr(self.config, "min_nonmissing_prefilter", 0.20))
        min_maf = float(getattr(self.config, "min_maf_prefilter", 0.0))
        max_features = getattr(self.config, "max_prefiltered_features", 10000)

        pre_candidates: List[str] = []
        skip_reasons: Dict[str, int] = defaultdict(int)

        for feat in features:
            try:
                col = pd.to_numeric(data[feat], errors="coerce")
                nonmissing_frac = float(col.notna().mean())
                if nonmissing_frac < min_nonmissing:
                    skip_reasons["too_many_missing"] += 1
                    continue

                col_non_na = col.dropna()
                if col_non_na.empty:
                    skip_reasons["all_missing"] += 1
                    continue

                af = float(col_non_na.mean())
                maf = min(af, 1.0 - af)
                if maf < min_maf:
                    skip_reasons["maf_too_low"] += 1
                    continue

                pre_candidates.append(feat)
            except Exception:
                skip_reasons["prefilter_error"] += 1

        if skip_reasons:
            logger.info(
                "Legacy DT prefilter skipped features: %s",
                dict(skip_reasons),
            )

        if not pre_candidates:
            logger.warning("Legacy DT prefilter: no features passed missing/MAF gate.")
            return []

        if max_features is None:
            max_for_tests = 20000
        else:
            max_for_tests = max(10000, int(max_features) * 2)

        if len(pre_candidates) > max_for_tests:
            variances = data[pre_candidates].apply(pd.to_numeric, errors="coerce").var()
            pre_candidates = variances.nlargest(max_for_tests).index.tolist()
            logger.info(
                "Legacy DT prefilter reduced candidates to top %d by variance for testing.",
                max_for_tests,
            )

        p_values: List[float] = []
        valid: List[str] = []

        for feat in pre_candidates:
            try:
                col = pd.to_numeric(data[feat], errors="coerce")
                keep = ~(col.isna() | labels.isna())
                if int(keep.sum()) == 0:
                    continue

                table = pd.crosstab(col.loc[keep], labels.loc[keep])
                if table.empty or min(table.shape) < 2:
                    continue

                if table.shape == (2, 2):
                    _, p = fisher_exact(table.values)
                else:
                    _, p, _, _ = chi2_contingency(table.values)

                p_values.append(float(p))
                valid.append(feat)
            except Exception:
                logger.warning("Legacy DT prefilter skipped feature '%s' due to test error.", feat)

        if not p_values:
            logger.warning("Legacy DT prefilter found no valid p-values; falling back to variance ranking.")
            variances = data[pre_candidates].apply(pd.to_numeric, errors="coerce").var()
            n_take = min(5000, len(pre_candidates)) if max_features is None else min(int(max_features), len(pre_candidates))
            return variances.nlargest(n_take).index.tolist()

        try:
            reject, _, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
            significant = [valid[i] for i in range(len(valid)) if bool(reject[i])]
        except Exception:
            significant = valid.copy()

        if not significant:
            logger.warning("Legacy DT prefilter retained no FDR-significant features; falling back to variance ranking.")
            variances = data[pre_candidates].apply(pd.to_numeric, errors="coerce").var()
            n_take = min(5000, len(pre_candidates)) if max_features is None else min(int(max_features), len(pre_candidates))
            significant = variances.nlargest(n_take).index.tolist()

        if max_features is not None and len(significant) > int(max_features):
            p_ser = pd.Series(p_values, index=valid)
            sig_p = p_ser.loc[significant]
            significant = sig_p.nsmallest(int(max_features)).index.tolist()
            logger.info(
                "Legacy DT prefilter capped retained features to max_prefiltered_features=%d.",
                int(max_features),
            )

        return significant

    # ------------------------------------------------------------------
    # Tree building
    # ------------------------------------------------------------------
    def _build_tree(self, X: pd.DataFrame, y: np.ndarray) -> DecisionTreeClassifier:
        dt = DecisionTreeClassifier(
            max_depth=getattr(self.config, "max_depth", None),
            min_samples_split=int(getattr(self.config, "min_samples_split", 2)),
            min_samples_leaf=int(getattr(self.config, "min_samples_leaf", 1)),
            random_state=int(getattr(self.config, "random_state", 42)),
        )
        dt.fit(X, y)

        logger.info(
            "Built decision tree | depth=%d | leaves=%d | features_used=%d",
            int(dt.get_depth()),
            int(dt.get_n_leaves()),
            int(X.shape[1]),
        )

        return dt

    # ------------------------------------------------------------------
    # Tree structure analysis
    # ------------------------------------------------------------------
    def _analyze_tree_structure(
        self,
        dt: DecisionTreeClassifier,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        tree_ = dt.tree_
        root_features: Set[str] = set()
        branch_features: Set[str] = set()
        feature_depths: Dict[str, List[int]] = defaultdict(list)

        def traverse(node_id: int, depth: int = 0) -> None:
            feat_idx = int(tree_.feature[node_id])
            if feat_idx == -2:
                return

            if feat_idx >= 0:
                feat = feature_names[feat_idx]
                feature_depths[feat].append(depth)

                if depth <= 1:
                    root_features.add(feat)
                else:
                    branch_features.add(feat)

            left = int(tree_.children_left[node_id])
            right = int(tree_.children_right[node_id])

            if left != -1:
                traverse(left, depth + 1)
            if right != -1:
                traverse(right, depth + 1)

        traverse(0)

        # Mean-depth reclassification
        for feat, depths in feature_depths.items():
            mean_depth = float(np.mean(depths)) if depths else 999.0
            if mean_depth <= 1.5:
                root_features.add(feat)
                branch_features.discard(feat)
            else:
                branch_features.add(feat)
                root_features.discard(feat)

        return {
            "root_features": list(root_features),
            "branch_features": list(branch_features),
            "features": list(root_features | branch_features),
            "depths": dict(feature_depths),
        }

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------
    def _compute_confidence(
        self,
        dt: DecisionTreeClassifier,
        X: pd.DataFrame,
        y: np.ndarray,
        analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Confidence function:
            confidence = 0.4 * mutual_information
                       + 0.4 * bootstrap_stability
                       + 0.2 * Cramer's V
        """
        del dt  # kept in signature for future expansion

        root_feats = list(analysis.get("root_features", []))
        branch_feats = list(analysis.get("branch_features", []))

        confidences: Dict[str, Any] = {}

        for feats, ftype in [(root_feats, "root"), (branch_feats, "branch")]:
            for feat in feats:
                try:
                    mi = float(mutual_info_score(y, X[feat]))

                    stability_values = []
                    outer = int(getattr(self.config, "bootstrap_outer_iters", 5))
                    per_iter = int(getattr(self.config, "bootstrap_samples_per_iter", 100))

                    for _ in range(max(1, outer)):
                        stability_values.append(
                            self._bootstrap_importance(X, y, feat, n=per_iter)
                        )

                    stability = float(np.mean(stability_values)) if stability_values else 0.0

                    table = pd.crosstab(X[feat], pd.Series(y, index=X.index))
                    cv = float(self._cramers_v(table)) if table.shape[1] > 1 else 0.0

                    conf = float(mi * 0.4 + stability * 0.4 + cv * 0.2)

                    confidences[feat] = {
                        "type": ftype,
                        "mutual_info": mi,
                        "stability": stability,
                        "cramers_v": cv,
                        "confidence": conf,
                    }
                except Exception as exc:
                    logger.warning("Confidence computation failed for feature '%s': %s", feat, exc)

        return confidences

    def _bootstrap_importance(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        target_feat: str,
        n: int = 100,
    ) -> float:
        """
        Bootstrap tree importance for a target feature.
        """
        if X.shape[0] == 0:
            return 0.0

        sample_size = max(1, min(int(n), int(X.shape[0])))
        boot_idx = np.random.choice(len(X), sample_size, replace=True)
        Xb = X.iloc[boot_idx]
        yb = y[boot_idx]

        if len(np.unique(yb)) < 2:
            return 0.0

        dt = DecisionTreeClassifier(
            max_depth=getattr(self.config, "max_depth", None),
            min_samples_split=int(getattr(self.config, "min_samples_split", 2)),
            min_samples_leaf=int(getattr(self.config, "min_samples_leaf", 1)),
            random_state=int(getattr(self.config, "random_state", 42)),
        )
        dt.fit(Xb, yb)

        try:
            feat_idx = list(X.columns).index(target_feat)
        except ValueError:
            return 0.0

        if feat_idx >= len(dt.feature_importances_):
            return 0.0

        return float(dt.feature_importances_[feat_idx])

    @staticmethod
    def _cramers_v(table: pd.DataFrame) -> float:
        try:
            chi2 = float(chi2_contingency(table)[0])
            n = float(table.sum().sum())
            min_dim = min(table.shape) - 1
            if min_dim <= 0 or n <= 0:
                return 0.0
            return float(np.sqrt(chi2 / (n * min_dim)))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Interaction mining
    # ------------------------------------------------------------------
    def _mine_epistatic_interactions(
        self,
        dt: DecisionTreeClassifier,
        feature_names: List[str],
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Mine interactions by walking tree paths and scoring the last two features
        by mutual-information-derived synergy.
        """
        tree_ = dt.tree_
        interactions: List[Dict[str, Any]] = []

        def traverse(node_id: int, path: List[str]) -> None:
            if int(tree_.feature[node_id]) == -2:
                return

            feat_idx = int(tree_.feature[node_id])
            new_path = path

            if feat_idx >= 0:
                feat = feature_names[feat_idx]
                new_path = path + [feat]

                if len(new_path) >= 2:
                    f1, f2 = new_path[-2], new_path[-1]
                    strength = float(self._epistasis_strength(X, y, f1, f2))

                    if strength > float(getattr(self.config, "epistasis_strength_threshold", 0.05)):
                        interactions.append(
                            {
                                "features": [f1, f2],
                                "strength": strength,
                                "path_depth": int(len(new_path) - 1),
                                "support": int(tree_.n_node_samples[node_id]),
                                "type": "conditional" if len(new_path) > 2 else "pairwise",
                            }
                        )

            left = int(tree_.children_left[node_id])
            right = int(tree_.children_right[node_id])

            if left != -1:
                traverse(left, new_path)
            if right != -1:
                traverse(right, new_path)

        traverse(0, [])

        # Deduplicate on ordered feature pair, keeping strongest score
        best_by_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for item in interactions:
            pair = tuple(item["features"])
            prev = best_by_pair.get(pair)
            if prev is None or float(item["strength"]) > float(prev["strength"]):
                best_by_pair[pair] = item

        deduped = list(best_by_pair.values())
        deduped.sort(key=lambda d: d["strength"], reverse=True)

        return deduped[: int(getattr(self.config, "max_epistatic_interactions", 50))]

    @staticmethod
    def _epistasis_strength(
        X: pd.DataFrame,
        y: np.ndarray,
        f1: str,
        f2: str,
    ) -> float:
        """
        Synergy proxy:
            MI(y ; joint_state(f1,f2)) - [MI(y ; f1) + MI(y ; f2)]
        """
        mi1 = float(mutual_info_score(y, X[f1]))
        mi2 = float(mutual_info_score(y, X[f2]))
        combined = X[f1].astype(str) + "_" + X[f2].astype(str)
        mi_comb = float(mutual_info_score(y, combined))
        return float(mi_comb - (mi1 + mi2))

    # ------------------------------------------------------------------
    # Export + Summary
    # ------------------------------------------------------------------
    def _export_results(self, results: Dict[str, Any], output_dir: Optional[str]) -> None:
        if not output_dir:
            return

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        (out / self.artifacts.rules_txt).write_text(
            results["decision_trees"]["rules"],
            encoding="utf-8",
        )
        (out / self.artifacts.confidence_json).write_text(
            json.dumps(results["feature_confidence"], indent=2),
            encoding="utf-8",
        )
        (out / self.artifacts.interactions_json).write_text(
            json.dumps(results["epistatic_interactions"], indent=2),
            encoding="utf-8",
        )

    def _print_summary(self, results: Dict[str, Any]) -> None:
        def _shorten_feature_name(name: str, max_len: int = 60) -> str:
            s = str(name)
            if len(s) <= max_len:
                return s

            keep = max_len - 3
            left = keep // 2
            right = keep - left
            return f"{s[:left]}...{s[-right:]}"

        tree = results["decision_trees"]
        root_features = results.get("root_features", [])
        branch_features = results.get("branch_features", [])
        interactions = results.get("epistatic_interactions", [])
        confidence_map = results.get("feature_confidence", {})

        print("\n" + "=" * 70)
        print("FEATURE DISCOVERY SUMMARY (DecisionTree Branch)")
        print("=" * 70)
        training_accuracy = tree.get("training_accuracy", tree.get("accuracy", float("nan")))
        print(
            f"Tree Training Accuracy: {training_accuracy:.3f} | "
            f"Classes: {tree['n_classes']}"
        )
        print(f"Root Features: {len(root_features)} | Branch Features: {len(branch_features)}")

        shown = root_features[:3]
        if shown:
            for i, feat in enumerate(shown, 1):
                conf = confidence_map.get(feat, {}).get("confidence", float("nan"))
                feat_disp = _shorten_feature_name(feat, max_len=60)
                print(f"  {i}. {feat_disp} (conf: {conf:.3f})")
        else:
            print("  No root features identified.")

        print(f"Epistatic Interactions: {len(interactions)}")
        print("=" * 70 + "\n")

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    def _validate_inputs(self, data: pd.DataFrame, labels: pd.Series) -> None:
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

    def _drop_post_impute_monomorphic(self, X: pd.DataFrame) -> List[str]:
        """
        Drop invariant / near-monomorphic columns after final coercion and imputation.
        """
        if X.empty:
            return []

        # Mean-based rule assumes binary-ish encoding and matches the current pipeline design
        col_means = X.mean().clip(0, 1)
        too_rare = (col_means < 0.001) | (col_means > 0.999)

        if bool(too_rare.any()):
            drop_cols = too_rare[too_rare].index.tolist()
            logger.info(
                "Dropping %d invariant / near-monomorphic features before tree fitting.",
                len(drop_cols),
            )

        kept = [c for c in X.columns if c not in set(too_rare[too_rare].index.tolist())]
        return kept
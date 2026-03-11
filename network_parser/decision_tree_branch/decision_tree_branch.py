# network_parser/decision_tree_branch/decision_tree_branch.py
"""
Decision Tree discovery branch (NetworkParser).

This module is a refactor of:
- decision_tree_builder.py (discovery + confidence + interaction mining)
- statistical_validation.py (validation lives in validator module)

Design goals:
- Keep behavior consistent with your current scripts:
    * prefilter: missing/MAF -> chi2/fisher -> FDR-BH, with variance fallbacks
    * tree: sklearn DecisionTreeClassifier with config params
    * post-tree: root/branch classification, MI+bootstrap+Cramer's V confidence,
      and path-based epistasis mining
- Keep everything generic to dataset.
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


def normalize_labels(labels: pd.Series, drop_missing: bool = True, lowercase: bool = False) -> pd.Series:
    """
    Same normalization pattern as in your existing scripts:
      - strip
      - treat '-', '', NA-like as missing
      - unify '-' -> '_'
      - optional lowercase
      - optionally drop missing
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
        logger.info("%s: %d features → %s", name, n, ", ".join(map(str, features)))
    else:
        logger.info(
            "%s: %d features → %s ... +%d more",
            name, n, ", ".join(map(str, features[:max_show])), n - max_show
        )


@dataclass
class DecisionTreeBranchArtifacts:
    rules_txt: str = "decision_tree_rules.txt"
    confidence_json: str = "feature_confidence.json"
    interactions_json: str = "epistatic_interactions.json"


class DecisionTreeBranch:
    """
    Decision tree discovery branch.

    Output dict structure intentionally matches your current builder outputs:
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

    def __init__(self, config: NetworkParserConfig, artifacts: Optional[DecisionTreeBranchArtifacts] = None):
        self.config = config
        self.artifacts = artifacts or DecisionTreeBranchArtifacts()
        np.random.seed(int(getattr(self.config, "random_state", 42)))

    # ----------------------------- Public entry -----------------------------

    def run(
        self,
        data: pd.DataFrame,
        labels: pd.Series,
        all_features: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the full decision-tree discovery branch on aligned X (samples×features) and labels.
        """
        if all_features is None:
            all_features = list(data.columns)

        valid_features = [f for f in all_features if f in data.columns]
        if not valid_features:
            raise ValueError("No valid features provided for discovery.")
        if len(valid_features) < len(all_features):
            logger.warning("Some provided features are missing from data columns (ignored).")

        # Align indices (defensive)
        if not data.index.equals(labels.index):
            common = data.index.intersection(labels.index)
            if common.empty:
                raise ValueError("No common indices between data and labels.")
            data = data.loc[common]
            labels = labels.loc[common]
            logger.info("Aligned data and labels to %d common samples.", len(common))

        log_feature_summary("Input features", valid_features)

        # Prefilter before tree (statistically defensible for discovery)
        prefiltered = self._prefilter_features(data=data, labels=labels, features=valid_features)
        if not prefiltered:
            logger.warning("No significant features after prefiltering; using all valid features.")
            prefiltered = valid_features
        log_feature_summary("Prefiltered features", prefiltered)

        # Subset and handle NaNs like your builder
        X = data[prefiltered].copy()
        total_nan = int(X.isna().sum().sum())
        if total_nan > 0:
            logger.info("Imputing NaNs (%d) → treating as baseline (0).", total_nan)
            X = X.fillna(0).astype(np.float64)

            # Optional: drop near-monomorphic columns after imputation
            post_impute_maf = X.mean().clip(0, 1)
            too_rare = (post_impute_maf < 0.001) | (post_impute_maf > 0.999)
            if bool(too_rare.any()):
                drop_cols = too_rare[too_rare].index.tolist()
                logger.info("Dropping %d near-monomorphic features after imputation.", len(drop_cols))
                X = X.drop(columns=drop_cols)
                prefiltered = [f for f in prefiltered if f in X.columns]

        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(labels.astype(str))
        n_classes = int(len(le.classes_))
        logger.info("Labels encoded: n_classes=%d", n_classes)

        # Build tree
        dt = self._build_tree(X, y)

        # Analyze hierarchy
        analysis = self._analyze_tree_structure(dt, feature_names=prefiltered)

        # Confidence scoring
        confidence = self._compute_confidence(dt, X, y, analysis)

        # Interactions
        interactions = self._mine_epistatic_interactions(dt, feature_names=prefiltered, X=X, y=y)

        results: Dict[str, Any] = {
            "discovered_features": analysis["features"],
            "root_features": analysis["root_features"],
            "branch_features": analysis["branch_features"],
            "decision_trees": {
                "accuracy": float(accuracy_score(y, dt.predict(X))),
                "rules": export_text(dt, feature_names=prefiltered),
                "n_classes": n_classes,
            },
            "feature_confidence": confidence,
            "epistatic_interactions": interactions,
            "prefiltered_features": prefiltered,
        }

        self._export_results(results, output_dir)
        self._print_summary(results)

        return results

    # ----------------------------- Prefilter -----------------------------

    def _prefilter_features(
        self,
        data: pd.DataFrame,
        labels: pd.Series,
        features: List[str],
        alpha: Optional[float] = None,
    ) -> List[str]:
        """
        Prefilter (same logic as your current decision_tree_builder):
          1) drop features with too much missing
          2) drop features with too low MAF
          3) association tests (chi2 or fisher for 2x2)
          4) FDR-BH
          5) fallbacks: top by variance if nothing survives or p-values empty
          6) optional cap max_prefiltered_features by smallest p-value
        """
        alpha = float(alpha if alpha is not None else self.config.prefilter_alpha)
        min_nonmissing = float(getattr(self.config, "min_nonmissing_prefilter", 0.20))
        min_maf = float(getattr(self.config, "min_maf_prefilter", 0.0))
        max_features = getattr(self.config, "max_prefiltered_features", 10000)

        pre_candidates: List[str] = []
        skip_reasons: Dict[str, int] = defaultdict(int)

        for f in features:
            try:
                col = data[f]
                nonmissing_frac = float(col.notna().mean())
                if nonmissing_frac < min_nonmissing:
                    skip_reasons["Too many missing"] += 1
                    continue

                col_non_na = col.dropna()
                # NOTE: This assumes binary-ish (0/1) encoding; matches your current builder.
                af = float(col_non_na.mean())
                maf = min(af, 1.0 - af)
                if maf < min_maf:
                    skip_reasons["MAF too low"] += 1
                    continue

                pre_candidates.append(f)
            except Exception as e:
                skip_reasons[str(e).split("\n")[0]] += 1

        skipped = sum(skip_reasons.values())
        if skipped > 0:
            logger.info("Prefilter: skipped %d feature(s) in missing/MAF gating.", skipped)

        if not pre_candidates:
            logger.warning("Prefilter: no features passed missing/MAF gating.")
            return []

        # Limit tests if enormous; mirror your variance cut
        if max_features is None:
            max_for_tests = 20000
        else:
            max_for_tests = max(10000, int(max_features) * 2)

        if len(pre_candidates) > max_for_tests:
            variances = data[pre_candidates].var()
            pre_candidates = variances.nlargest(max_for_tests).index.tolist()
            logger.info("Prefilter: reduced candidates to top %d by variance for tests.", max_for_tests)

        p_values: List[float] = []
        valid: List[str] = []

        for f in pre_candidates:
            try:
                table = pd.crosstab(data[f], labels)
                if table.empty or min(table.shape) < 2:
                    continue

                if table.shape == (2, 2):
                    _, p = fisher_exact(table)
                else:
                    _, p, _, _ = chi2_contingency(table)
                p_values.append(float(p))
                valid.append(f)
            except Exception as e:
                logger.warning("Prefilter: skipping feature due to test error.")

        if not p_values:
            logger.warning("Prefilter: no valid p-values; fallback to variance ranking.")
            variances = data[pre_candidates].var()
            n_take = min(5000, len(pre_candidates)) if max_features is None else min(int(max_features), len(pre_candidates))
            return variances.nlargest(n_take).index.tolist()

        # FDR-BH
        try:
            reject, _, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
            significant = [valid[i] for i in range(len(valid)) if bool(reject[i])]
        except Exception:
            significant = valid

        if not significant:
            logger.warning("Prefilter: no FDR-significant; fallback to variance ranking.")
            variances = data[pre_candidates].var()
            n_take = min(5000, len(pre_candidates)) if max_features is None else min(int(max_features), len(pre_candidates))
            significant = variances.nlargest(n_take).index.tolist()

        # Cap by smallest p-value if requested
        if max_features is not None and len(significant) > int(max_features):
            p_ser = pd.Series(p_values, index=valid)
            sig_p = p_ser.loc[significant]
            significant = sig_p.nsmallest(int(max_features)).index.tolist()
            logger.info("Prefilter: capped significant set to max_prefiltered_features=%d.", int(max_features))

        return significant

    # ----------------------------- Tree build -----------------------------

    def _build_tree(self, X: pd.DataFrame, y: np.ndarray) -> DecisionTreeClassifier:
        dt = DecisionTreeClassifier(
            max_depth=getattr(self.config, "max_depth", None),
            min_samples_split=int(getattr(self.config, "min_samples_split", 2)),
            min_samples_leaf=int(getattr(self.config, "min_samples_leaf", 1)),
            random_state=int(getattr(self.config, "random_state", 42)),
        )
        dt.fit(X, y)
        return dt

    # ----------------------------- Tree analysis -----------------------------

    def _analyze_tree_structure(self, dt: DecisionTreeClassifier, feature_names: List[str]) -> Dict[str, Any]:
        tree_ = dt.tree_
        root_features: Set[str] = set()
        branch_features: Set[str] = set()
        feature_depths: Dict[str, List[int]] = defaultdict(list)

        def traverse(node_id: int, depth: int = 0) -> None:
            feat_idx = int(tree_.feature[node_id])
            if feat_idx == -2:  # leaf
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

        # mean-depth reclassification (matches your current logic)
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

    # ----------------------------- Confidence -----------------------------

    def _compute_confidence(
        self,
        dt: DecisionTreeClassifier,
        X: pd.DataFrame,
        y: np.ndarray,
        analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Same confidence function:
          confidence = 0.4*MI + 0.4*stability + 0.2*Cramer'sV
        """
        root_feats = list(analysis.get("root_features", []))
        branch_feats = list(analysis.get("branch_features", []))

        confidences: Dict[str, Any] = {}

        for feats, ftype in [(root_feats, "root"), (branch_feats, "branch")]:
            for feat in feats:
                mi = float(mutual_info_score(y, X[feat]))

                stability_values = []
                outer = int(getattr(self.config, "bootstrap_outer_iters", 5))
                per_iter = int(getattr(self.config, "bootstrap_samples_per_iter", 100))
                for _ in range(max(1, outer)):
                    stability_values.append(self._bootstrap_importance(X, y, feat, n=per_iter))
                stability = float(np.mean(stability_values))

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

        return confidences

    def _bootstrap_importance(self, X: pd.DataFrame, y: np.ndarray, target_feat: str, n: int = 100) -> float:
        """
        Single bootstrap run: fit a tree on a bootstrap sample and return importance for target_feat.
        (This mirrors your current implementation, including using a tree as the probe.)
        """
        if X.shape[0] == 0:
            return 0.0
        boot_idx = np.random.choice(len(X), len(X), replace=True)
        Xb = X.iloc[boot_idx]
        yb = y[boot_idx]

        dt = DecisionTreeClassifier(
            max_depth=getattr(self.config, "max_depth", None),
            random_state=int(getattr(self.config, "random_state", 42)),
        )
        dt.fit(Xb, yb)

        # importance index follows X columns order
        try:
            feat_idx = list(X.columns).index(target_feat)
        except ValueError:
            return 0.0

        if feat_idx >= len(dt.feature_importances_):
            return 0.0
        return float(dt.feature_importances_[feat_idx])

    @staticmethod
    def _cramers_v(table: pd.DataFrame) -> float:
        chi2 = float(chi2_contingency(table)[0])
        n = float(table.sum().sum())
        min_dim = min(table.shape) - 1
        if min_dim <= 0 or n <= 0:
            return 0.0
        return float(np.sqrt(chi2 / (n * min_dim)))

    # ----------------------------- Interactions -----------------------------

    def _mine_epistatic_interactions(
        self,
        dt: DecisionTreeClassifier,
        feature_names: List[str],
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Mine interactions by walking tree paths and scoring the last two features by MI synergy.
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
        interactions.sort(key=lambda d: d["strength"], reverse=True)
        return interactions[: int(getattr(self.config, "max_epistatic_interactions", 50))]

    @staticmethod
    def _epistasis_strength(X: pd.DataFrame, y: np.ndarray, feat1: str, feat2: str) -> float:
        """
        MI synergy:
          max(0, MI(label ; (f1,f2)) - MI(label ; f1) - MI(label ; f2))
        """
        mi1 = float(mutual_info_score(y, X[feat1]))
        mi2 = float(mutual_info_score(y, X[feat2]))
        combined = pd.Categorical(X[[feat1, feat2]].astype(str).agg("_".join, axis=1))
        mi_comb = float(mutual_info_score(y, combined))
        return float(max(0.0, mi_comb - mi1 - mi2))

    # ----------------------------- Export + Summary -----------------------------

    def _export_results(self, results: Dict[str, Any], output_dir: Optional[str]) -> None:
        if not output_dir:
            return
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        (out / self.artifacts.rules_txt).write_text(results["decision_trees"]["rules"])
        (out / self.artifacts.confidence_json).write_text(json.dumps(results["feature_confidence"], indent=2))
        (out / self.artifacts.interactions_json).write_text(json.dumps(results["epistatic_interactions"], indent=2))

    def _print_summary(self, results: Dict[str, Any]) -> None:
        def _shorten_feature_name(name: str, max_len: int = 60) -> str:
            """
            Keep terminal logs readable without losing the true feature identity.
            The full feature name remains unchanged in output files/results;
            this only shortens the console display.
            """
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
        print("🎯 FEATURE DISCOVERY SUMMARY (DecisionTree Branch)")
        print("=" * 70)
        print(f"📈 Tree Accuracy: {tree['accuracy']:.3f} | Classes: {tree['n_classes']}")
        print(f"🌳 Root Features: {len(root_features)} | Branch: {len(branch_features)}")

        shown = root_features[:3]
        if shown:
            for i, feat in enumerate(shown, 1):
                conf = confidence_map.get(feat, {}).get("confidence", float("nan"))
                feat_disp = _shorten_feature_name(feat, max_len=60)
                print(f"  {i}. {feat_disp} (conf: {conf:.3f})")
        else:
            print("  No root features identified.")

        print(f"🔗 Epistatic Interactions: {len(interactions)}")
        print("=" * 70 + "\n")
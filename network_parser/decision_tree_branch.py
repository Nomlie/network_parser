# network_parser/decision_tree_branch.py
"""
Decision Tree discovery branch (NetworkParser).

Default flow:
    aligned + centrally filtered matrix
        -> capacity-constrained decision tree
        -> tree hierarchy extraction (min-depth root features)
        -> post-tree evidence fields (MI, Cramér's V, bootstrap stability)
        -> tree-path interaction candidates (+ optional validation)

Evidence fields are reported separately. Any weighted mixture is an explicit
``evidence_score``, not calibrated probability confidence.
"""

from __future__ import annotations

import json
import logging
import math
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
except ImportError:  # pragma: no cover - package vs source-tree layout
    from config import NetworkParserConfig  # type: ignore


logger = logging.getLogger(__name__)

# Explicit missing tokens only — no global punctuation rewriting (Phase 6).
_MISSING_LABEL_TOKENS = frozenset(
    {
        "",
        "-",
        "NA",
        "N/A",
        "None",
        "nan",
        "NaN",
        "NULL",
        "null",
        ".",
    }
)


def normalize_labels(
    labels: pd.Series,
    drop_missing: bool = True,
    lowercase: bool = False,
    **kwargs,
) -> pd.Series:
    """
    Normalize labels without silent punctuation rewriting.

    Matches the NetworkParser policy: strip, map missing tokens, optional
    lowercase. Does not rewrite ``-`` to ``_`` (that can merge distinct classes).
    """
    if not isinstance(labels, pd.Series):
        raise TypeError("labels must be a pandas Series")

    clean = labels.astype(str).str.strip()
    clean = clean.replace(set(_MISSING_LABEL_TOKENS), pd.NA)

    if lowercase:
        clean = clean.map(lambda v: v.lower() if isinstance(v, str) else v)

    if drop_missing:
        clean = clean[~clean.isna()]

    return clean


def log_feature_summary(name: str, features: List[str], max_show: int = 3) -> None:
    """Log feature-set size at INFO and exact identifiers only at DEBUG."""
    n = len(features)
    logger.info("%s: %d features", name, n)
    if n == 0:
        return

    if n <= max_show:
        logger.debug("%s feature identifiers: %s", name, ", ".join(map(str, features)))
    else:
        logger.debug(
            "%s feature identifiers: %s ... +%d more",
            name,
            ", ".join(map(str, features[:max_show])),
            n - max_show,
        )


@dataclass
class DecisionTreeBranchArtifacts:
    rules_txt: str = "decision_tree_rules.txt"
    confidence_json: str = (
        "feature_evidence.json"  # legacy filename still written below
    )
    evidence_json: str = "feature_evidence.json"
    interactions_json: str = "tree_path_interaction_candidates.json"
    # Legacy alias path still produced for older readers
    legacy_confidence_json: str = "feature_confidence.json"
    legacy_interactions_json: str = "epistatic_interactions.json"


class DecisionTreeBranch:
    """
    Decision tree interpretability branch.

    Output includes separate evidence fields (MI, Cramér's V, bootstrap
    stability) and an optional documented ``evidence_score`` mixture. Tree-path
    interaction candidates are never labelled as biological epistasis unless
    validation gates pass (and even then status is ``validated_candidate``).
    """

    def __init__(
        self,
        config: NetworkParserConfig,
        artifacts: Optional[DecisionTreeBranchArtifacts] = None,
    ):
        self.config = config
        self.artifacts = artifacts or DecisionTreeBranchArtifacts()
        # Local RNG only — never mutate the process-global NumPy state.
        self._rng = np.random.default_rng(int(getattr(self.config, "random_state", 42)))

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
            logger.warning(
                "Some requested features were not found in data and were ignored."
            )

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

        # Matrix contract: coerce to float; impute only via fitted train policy.
        try:
            from network_parser.matrix_contract import (
                MissingnessPolicy,
                prepare_for_sklearn,
            )
        except ImportError:  # pragma: no cover
            from matrix_contract import MissingnessPolicy, prepare_for_sklearn  # type: ignore

        policy = MissingnessPolicy.from_config(self.config)
        # Trees need dense input: impute with train-fit baseline/mode (explicit).
        X, miss_state, miss_audit = prepare_for_sklearn(X, policy=policy)
        logger.info(
            "Decision-tree missingness prep | strategy=%s | n_nan_before=%s | dropped_features=%d",
            policy.impute_strategy,
            miss_audit.get("transform", miss_audit).get(
                "n_nan_before_impute", miss_audit.get("n_nan_before_impute")
            ),
            len(miss_state.dropped_features),
        )
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

        # Evidence fields (not probability confidence)
        evidence = self._compute_feature_evidence(dt, X, y, analysis)

        # Tree-path interaction candidates (+ validation status)
        interactions = self._mine_tree_path_interaction_candidates(
            dt=dt,
            feature_names=list(X.columns),
            X=X,
            y=y,
        )

        results: Dict[str, Any] = {
            "discovered_features": analysis["features"],
            "root_features": analysis["root_features"],
            "branch_features": analysis["branch_features"],
            "feature_min_depths": analysis.get("min_depths", {}),
            "decision_trees": {
                "training_fit_accuracy": float(accuracy_score(y, dt.predict(X))),
                "training_accuracy": float(
                    accuracy_score(y, dt.predict(X))
                ),  # legacy alias
                "rules": export_text(dt, feature_names=list(X.columns)),
                "n_classes": n_classes,
                "max_depth_config": getattr(self.config, "max_depth", None),
                "min_samples_leaf": int(getattr(self.config, "min_samples_leaf", 2)),
                "tree_depth": int(dt.get_depth()),
                "n_leaves": int(dt.get_n_leaves()),
            },
            "feature_evidence": evidence,
            "tree_path_interaction_candidates": interactions,
            "prefiltered_features": prefiltered,
            # Versioned deprecated aliases (planned removal: schema v2.0).
            "deprecated": {
                "removal_target": "2.0",
                "feature_confidence": evidence,
                "epistatic_interactions": interactions,
                "note": (
                    "feature_confidence and epistatic_interactions are deprecated aliases. "
                    "Use feature_evidence and tree_path_interaction_candidates. "
                    "evidence_score is not confidence; interactions are not epistasis claims."
                ),
            },
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
        alpha = float(
            alpha
            if alpha is not None
            else getattr(self.config, "prefilter_alpha", 0.05)
        )
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
                logger.warning(
                    "Legacy DT prefilter skipped feature '%s' due to test error.", feat
                )

        if not p_values:
            logger.warning(
                "Legacy DT prefilter found no valid p-values; falling back to variance ranking."
            )
            variances = data[pre_candidates].apply(pd.to_numeric, errors="coerce").var()
            n_take = (
                min(5000, len(pre_candidates))
                if max_features is None
                else min(int(max_features), len(pre_candidates))
            )
            return variances.nlargest(n_take).index.tolist()

        try:
            reject, _, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
            significant = [valid[i] for i in range(len(valid)) if bool(reject[i])]
        except Exception:
            significant = valid.copy()

        if not significant:
            logger.warning(
                "Legacy DT prefilter retained no FDR-significant features; falling back to variance ranking."
            )
            variances = data[pre_candidates].apply(pd.to_numeric, errors="coerce").var()
            n_take = (
                min(5000, len(pre_candidates))
                if max_features is None
                else min(int(max_features), len(pre_candidates))
            )
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
    def _tree_params(self) -> Dict[str, Any]:
        """Capacity-constrained tree parameters (no unrestricted leaf=1 trees)."""
        max_depth = getattr(self.config, "max_depth", 12)
        if max_depth is None:
            max_depth = 12
        return {
            "max_depth": int(max_depth),
            "min_samples_split": int(getattr(self.config, "min_samples_split", 4)),
            "min_samples_leaf": max(
                2, int(getattr(self.config, "min_samples_leaf", 2))
            ),
            "random_state": int(getattr(self.config, "random_state", 42)),
        }

    def _build_tree(self, X: pd.DataFrame, y: np.ndarray) -> DecisionTreeClassifier:
        params = self._tree_params()
        dt = DecisionTreeClassifier(**params)
        dt.fit(X, y)
        logger.info(
            "Built decision tree | depth=%d | leaves=%d | features_used=%d | max_depth=%s | min_leaf=%d",
            int(dt.get_depth()),
            int(dt.get_n_leaves()),
            int(X.shape[1]),
            params["max_depth"],
            params["min_samples_leaf"],
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
        """Root features by minimum depth (not average depth); deterministic order."""
        tree_ = dt.tree_
        feature_depths: Dict[str, List[int]] = defaultdict(list)

        def traverse(node_id: int, depth: int = 0) -> None:
            feat_idx = int(tree_.feature[node_id])
            if feat_idx == -2:
                return
            if feat_idx >= 0:
                feat = feature_names[feat_idx]
                feature_depths[feat].append(depth)
            left = int(tree_.children_left[node_id])
            right = int(tree_.children_right[node_id])
            if left != -1:
                traverse(left, depth + 1)
            if right != -1:
                traverse(right, depth + 1)

        traverse(0)

        min_depths = {
            feat: int(min(depths)) if depths else 999
            for feat, depths in feature_depths.items()
        }
        root_features = sorted(
            [f for f, d in min_depths.items() if d <= 1],
            key=lambda f: (min_depths[f], f),
        )
        branch_features = sorted(
            [f for f, d in min_depths.items() if d > 1],
            key=lambda f: (min_depths[f], f),
        )
        all_feats = sorted(set(root_features) | set(branch_features))

        return {
            "root_features": root_features,
            "branch_features": branch_features,
            "features": all_feats,
            "depths": {k: sorted(v) for k, v in sorted(feature_depths.items())},
            "min_depths": dict(sorted(min_depths.items())),
        }

    # ------------------------------------------------------------------
    # Genuine bootstrap stability + separate evidence fields
    # ------------------------------------------------------------------
    def _compute_feature_evidence(
        self,
        dt: DecisionTreeClassifier,
        X: pd.DataFrame,
        y: np.ndarray,
        analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Per-feature evidence with separate fields:

        - mutual_info
        - cramers_v
        - bootstrap_stability = selection_count / B
        - evidence_score = optional documented weighted mixture (NOT probability)

        Bootstrap: B independent cohort resamples; fit a tree each time; record
        whether each feature was used (feature_importances_ > 0) and min depth.
        """
        del dt
        B = int(
            getattr(
                self.config,
                "n_bootstrap_resamples",
                getattr(self.config, "n_bootstrap", 100),
            )
        )
        frac = float(
            getattr(
                self.config,
                "bootstrap_cohort_sample_fraction",
                getattr(self.config, "bootstrap_sample_fraction", 1.0),
            )
        )
        features = list(analysis.get("features", []))
        min_depths = analysis.get("min_depths", {})
        root_set = set(analysis.get("root_features", []))

        stability = self._bootstrap_feature_stability(
            X, y, features, n_resamples=B, sample_fraction=frac
        )
        successful_B = int(stability.get("__meta__", {}).get("n_successful_fits", 0))
        skipped_single = int(
            stability.get("__meta__", {}).get("n_skipped_single_class", 0)
        )
        requested_B = int(stability.get("__meta__", {}).get("n_requested_resamples", B))
        min_success = int(
            getattr(self.config, "bootstrap_min_successful_resamples", 20)
        )
        if successful_B < min_success:
            logger.warning(
                "Bootstrap stability: only %d successful fits of %d requested "
                "(skipped_single_class=%d). Minimum recommended=%d. "
                "Stability denominators use successful fits only.",
                successful_B,
                requested_B,
                skipped_single,
                min_success,
            )

        evidence: Dict[str, Any] = {}
        for feat in sorted(features):
            # Pairwise-complete MI / Cramér's V (drop NaN feature rows)
            try:
                xf = pd.to_numeric(X[feat], errors="coerce")
                mask = xf.notna()
                if mask.sum() >= 2 and len(np.unique(y[mask.to_numpy()])) >= 2:
                    mi = float(
                        mutual_info_score(y[mask.to_numpy()], xf.loc[mask].to_numpy())
                    )
                else:
                    mi = 0.0
            except Exception:
                mi = 0.0
            try:
                xf = pd.to_numeric(X[feat], errors="coerce")
                mask = xf.notna()
                table = pd.crosstab(xf.loc[mask], pd.Series(y, index=X.index).loc[mask])
                cv = float(self._cramers_v(table)) if min(table.shape) > 1 else 0.0
            except Exception:
                cv = 0.0

            stab = stability.get(feat, {})
            sel_count = int(stab.get("selection_count", 0))
            # Denominator = successful fitted bootstrap resamples (not requested B)
            denom = int(stab.get("n_successful_fits", successful_B))
            stab_frac = float(sel_count / denom) if denom > 0 else 0.0
            ci_low, ci_high = self._wilson_interval(sel_count, denom)

            # Documented mixture — NOT calibrated confidence / probability
            evidence_score = float(
                0.4 * min(1.0, mi) + 0.4 * stab_frac + 0.2 * min(1.0, cv)
            )

            evidence[feat] = {
                "type": "root" if feat in root_set else "branch",
                "min_tree_depth": int(min_depths.get(feat, 999)),
                "mutual_info": mi,
                "cramers_v": cv,
                "bootstrap_stability": stab_frac,
                "bootstrap_selection_count": sel_count,
                "bootstrap_denominator": denom,
                "bootstrap_n_requested_resamples": requested_B,
                "bootstrap_n_successful_fits": successful_B,
                "bootstrap_n_skipped_single_class": skipped_single,
                "bootstrap_stability_ci95": {"low": ci_low, "high": ci_high},
                "bootstrap_min_depth_mean": stab.get("min_depth_mean"),
                "bootstrap_min_depth_median": stab.get("min_depth_median"),
                "evidence_score": evidence_score,
                "evidence_score_definition": (
                    "0.4*clip(mutual_info,0,1)+0.4*bootstrap_stability+0.2*clip(cramers_v,0,1); "
                    "arbitrary documented weighting; not a calibrated probability or confidence."
                ),
                # Deprecated aliases (removal planned v2.0) — not preferred
                "stability": stab_frac,
            }
        evidence["__bootstrap_meta__"] = {
            "n_requested_resamples": requested_B,
            "n_successful_fits": successful_B,
            "n_skipped_single_class": skipped_single,
            "denominator_policy": "successful_fits_only",
            "min_successful_resamples": min_success,
            "warning_insufficient_successful_fits": bool(successful_B < min_success),
        }
        return evidence

    def _bootstrap_feature_stability(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        features: List[str],
        *,
        n_resamples: int,
        sample_fraction: float,
    ) -> Dict[str, Dict[str, Any]]:
        """
        B independent cohort bootstrap trees.

        Stability denominator is the number of *successful* multi-class fits,
        not the requested resample count.
        """
        n = int(X.shape[0])
        if n == 0 or n_resamples <= 0:
            meta: Dict[str, Any] = {
                "n_requested_resamples": max(0, n_resamples),
                "n_successful_fits": 0,
                "n_skipped_single_class": 0,
            }
            out: Dict[str, Dict[str, Any]] = {
                f: {
                    "selection_count": 0,
                    "n_successful_fits": 0,
                    "n_bootstrap_resamples": 0,
                    "min_depths": [],
                }
                for f in features
            }
            out["__meta__"] = meta
            return out

        sample_size = max(1, int(round(n * float(sample_fraction))))
        params = self._tree_params()
        feature_list = list(X.columns.astype(str))
        selection_count = {f: 0 for f in features}
        min_depth_lists: Dict[str, List[int]] = {f: [] for f in features}
        n_success = 0
        n_skipped = 0

        for b in range(int(n_resamples)):
            boot_idx = self._rng.choice(n, size=sample_size, replace=True)
            Xb = X.iloc[boot_idx]
            yb = y[boot_idx]
            if len(np.unique(yb)) < 2:
                n_skipped += 1
                continue
            rs = int(self._rng.integers(0, 2**31 - 1))
            dt_b = DecisionTreeClassifier(
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                random_state=rs,
            )
            try:
                dt_b.fit(Xb, yb)
            except Exception:
                n_skipped += 1
                continue
            n_success += 1
            used = set()
            tree_ = dt_b.tree_
            depths_this: Dict[str, List[int]] = defaultdict(list)

            def walk(node_id: int, depth: int) -> None:
                feat_idx = int(tree_.feature[node_id])
                if feat_idx == -2:
                    return
                if feat_idx >= 0:
                    fname = feature_list[feat_idx]
                    used.add(fname)
                    depths_this[fname].append(depth)
                left = int(tree_.children_left[node_id])
                right = int(tree_.children_right[node_id])
                if left != -1:
                    walk(left, depth + 1)
                if right != -1:
                    walk(right, depth + 1)

            walk(0, 0)
            for f in features:
                if f in used:
                    selection_count[f] += 1
                    if depths_this.get(f):
                        min_depth_lists[f].append(int(min(depths_this[f])))

        out = {}
        for f in features:
            depths = min_depth_lists[f]
            out[f] = {
                "selection_count": int(selection_count[f]),
                "n_successful_fits": int(n_success),
                "n_bootstrap_resamples": int(n_success),  # successful-fit denominator
                "min_depth_mean": float(np.mean(depths)) if depths else None,
                "min_depth_median": float(np.median(depths)) if depths else None,
            }
        out["__meta__"] = {
            "n_requested_resamples": int(n_resamples),
            "n_successful_fits": int(n_success),
            "n_skipped_single_class": int(n_skipped),
        }
        return out

    @staticmethod
    def _wilson_interval(
        successes: int, n: int, z: float = 1.96
    ) -> Tuple[Optional[float], Optional[float]]:
        if n <= 0:
            return None, None
        successes = max(0, min(int(successes), int(n)))
        phat = successes / n
        denom = 1.0 + (z * z) / n
        centre = (phat + (z * z) / (2.0 * n)) / denom
        margin = (
            z * math.sqrt((phat * (1.0 - phat) / n) + (z * z) / (4.0 * n * n))
        ) / denom
        return float(max(0.0, centre - margin)), float(min(1.0, centre + margin))

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
    # Tree-path interaction candidates (not biological epistasis claims)
    # ------------------------------------------------------------------
    def _mine_tree_path_interaction_candidates(
        self,
        dt: DecisionTreeClassifier,
        feature_names: List[str],
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Mine co-occurring features along tree paths as interaction *candidates*.

        Validation requires joint support, bootstrap co-selection stability,
        association test with FDR correction. Status is never "biological_epistasis".
        """
        tree_ = dt.tree_
        raw: List[Dict[str, Any]] = []

        def traverse(node_id: int, path: List[str]) -> None:
            feat_idx = int(tree_.feature[node_id])
            if feat_idx == -2:
                return
            new_path = path
            if feat_idx >= 0:
                feat = feature_names[feat_idx]
                new_path = path + [feat]
                if len(new_path) >= 2:
                    f1, f2 = new_path[-2], new_path[-1]
                    strength = float(self._path_synergy_score(X, y, f1, f2))
                    raw.append(
                        {
                            "features": sorted([f1, f2]),
                            "synergy_score": strength,
                            "path_depth": int(len(new_path) - 1),
                            "node_support": int(tree_.n_node_samples[node_id]),
                            "path_type": "conditional"
                            if len(new_path) > 2
                            else "pairwise",
                            "candidate_kind": "tree_path_interaction_candidate",
                        }
                    )
            left = int(tree_.children_left[node_id])
            right = int(tree_.children_right[node_id])
            if left != -1:
                traverse(left, new_path)
            if right != -1:
                traverse(right, new_path)

        traverse(0, [])

        # Aggregate candidates by unordered pair
        by_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for item in raw:
            pair = tuple(item["features"])
            prev = by_pair.get(pair)  # type: ignore[arg-type]
            if prev is None or float(item["synergy_score"]) > float(
                prev["synergy_score"]
            ):
                by_pair[pair] = item  # type: ignore[index]

        candidates = list(by_pair.values())
        # Bootstrap co-selection stability for pairs
        pair_stability = self._bootstrap_pair_stability(
            X, y, [tuple(c["features"]) for c in candidates]
        )

        min_joint = int(getattr(self.config, "interaction_min_joint_support", 10))
        min_stab = float(
            getattr(self.config, "interaction_min_bootstrap_stability", 0.5)
        )
        fdr_alpha = float(getattr(self.config, "interaction_fdr_alpha", 0.05))
        strength_gate = float(
            getattr(self.config, "epistasis_strength_threshold", 0.05)
        )

        pvals: List[float] = []
        for c in candidates:
            f1, f2 = c["features"]
            joint = self._joint_support(X, f1, f2)
            c["joint_support"] = int(joint)
            key = tuple(sorted([f1, f2]))
            stab = pair_stability.get(key, {})
            c["pair_bootstrap_stability"] = stab.get("stability", 0.0)
            c["pair_bootstrap_selection_count"] = stab.get("selection_count", 0)
            c["pair_bootstrap_denominator"] = stab.get(
                "n_successful_fits", stab.get("n_bootstrap_resamples", 0)
            )
            c["pair_stability_definition"] = stab.get(
                "definition",
                "fraction of successful bootstrap trees where the pair co-occurs on a root-to-leaf path",
            )
            # Association of joint state vs label (pairwise-complete)
            p = self._joint_label_association_pvalue(X, y, f1, f2)
            c["association_pvalue"] = p
            pvals.append(p if p is not None and np.isfinite(p) else 1.0)

        if pvals:
            reject, p_adj, _, _ = multipletests(pvals, alpha=fdr_alpha, method="fdr_bh")
        else:
            reject, p_adj = [], []

        for i, c in enumerate(candidates):
            c["association_pvalue_fdr"] = float(p_adj[i]) if len(p_adj) else None
            passes = (
                float(c.get("synergy_score", 0.0)) > strength_gate
                and int(c.get("joint_support", 0)) >= min_joint
                and float(c.get("pair_bootstrap_stability", 0.0)) >= min_stab
                and bool(reject[i])
                if len(reject)
                else False
            )
            c["validation_status"] = (
                "validated_candidate" if passes else "unvalidated_candidate"
            )
            c["biological_epistasis_claim"] = False
            c["note"] = (
                "Tree-path interaction candidate only. "
                "Not a claim of biological epistasis."
            )

        candidates.sort(
            key=lambda d: (
                0 if d.get("validation_status") == "validated_candidate" else 1,
                -float(d.get("synergy_score", 0.0)),
                d["features"][0],
                d["features"][1],
            )
        )
        limit = int(
            getattr(
                self.config,
                "max_tree_path_interaction_candidates",
                getattr(self.config, "max_epistatic_interactions", 50),
            )
        )
        return candidates[:limit]

    def _bootstrap_pair_stability(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        pairs: List[Tuple[str, str]],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        Path co-occurrence stability: count successful bootstrap trees where the
        pair appears on the *same* root-to-leaf path (not merely both used anywhere
        in the tree).
        """
        B = int(getattr(self.config, "n_bootstrap_resamples", 50))
        B = min(B, 50)  # pair stability can be expensive; cap for practicality
        frac = float(getattr(self.config, "bootstrap_cohort_sample_fraction", 1.0))
        n = int(X.shape[0])

        def pair_key(pair: Tuple[str, str]) -> Tuple[str, str]:
            first, second = pair
            return (first, second) if first <= second else (second, first)

        empty = {
            pair_key(p): {
                "selection_count": 0,
                "stability": 0.0,
                "n_successful_fits": 0,
                "n_bootstrap_resamples": 0,
                "n_requested_resamples": B,
                "n_skipped_single_class": 0,
                "definition": "path_cooccurrence_on_successful_fits",
            }
            for p in pairs
        }
        if n == 0 or B <= 0 or not pairs:
            return empty

        sample_size = max(1, int(round(n * frac)))
        params = self._tree_params()
        feature_list = list(X.columns.astype(str))
        counts = {pair_key(p): 0 for p in pairs}
        n_success = 0
        n_skipped = 0

        for _ in range(B):
            boot_idx = self._rng.choice(n, size=sample_size, replace=True)
            Xb = X.iloc[boot_idx]
            yb = y[boot_idx]
            if len(np.unique(yb)) < 2:
                n_skipped += 1
                continue
            rs = int(self._rng.integers(0, 2**31 - 1))
            dt_b = DecisionTreeClassifier(
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                random_state=rs,
            )
            try:
                dt_b.fit(Xb, yb)
            except Exception:
                n_skipped += 1
                continue
            n_success += 1
            tree_ = dt_b.tree_
            path_pairs: Set[Tuple[str, str]] = set()

            def walk(node_id: int, path: List[str]) -> None:
                feat_idx = int(tree_.feature[node_id])
                left = int(tree_.children_left[node_id])
                right = int(tree_.children_right[node_id])
                # Leaf: record all unordered pairs on this root-to-leaf path
                if feat_idx == -2 or (left == right == -1):
                    for i in range(len(path)):
                        for j in range(i + 1, len(path)):
                            path_pairs.add(pair_key((path[i], path[j])))
                    return
                new_path = path
                if feat_idx >= 0:
                    new_path = path + [feature_list[feat_idx]]
                if left != -1:
                    walk(left, new_path)
                if right != -1:
                    walk(right, new_path)

            walk(0, [])
            for p in counts:
                if p in path_pairs:
                    counts[p] += 1

        denom = max(0, n_success)
        return {
            p: {
                "selection_count": int(c),
                "n_successful_fits": int(denom),
                "n_bootstrap_resamples": int(denom),
                "n_requested_resamples": int(B),
                "n_skipped_single_class": int(n_skipped),
                "stability": float(c / denom) if denom else 0.0,
                "definition": "path_cooccurrence_on_successful_fits",
            }
            for p, c in counts.items()
        }

    @staticmethod
    def _joint_support(X: pd.DataFrame, f1: str, f2: str) -> int:
        if f1 not in X.columns or f2 not in X.columns:
            return 0
        # Count samples where both markers are non-baseline (typically 1)
        # Pairwise-complete association: drop rows where either feature is non-callable.
        a = pd.to_numeric(X[f1], errors="coerce")
        b = pd.to_numeric(X[f2], errors="coerce")
        both = a.notna() & b.notna()
        a = a.loc[both]
        b = b.loc[both]
        if a.empty:
            return 0
        return int(((a != 0) & (b != 0)).sum())

    @staticmethod
    def _joint_label_association_pvalue(
        X: pd.DataFrame, y: np.ndarray, f1: str, f2: str
    ) -> Optional[float]:
        try:
            a = pd.to_numeric(X[f1], errors="coerce")
            b = pd.to_numeric(X[f2], errors="coerce")
            both = a.notna() & b.notna()
            if both.sum() < 4:
                return 1.0
            joint = a.loc[both].astype(str) + "_" + b.loc[both].astype(str)
            y_s = pd.Series(y, index=X.index).loc[both]
            table = pd.crosstab(joint, y_s)
            if table.shape[0] < 2 or table.shape[1] < 2:
                return 1.0
            if table.shape == (2, 2):
                _, p = fisher_exact(table.values)
                return float(p)
            _, p, _, _ = chi2_contingency(table.values)
            return float(p)
        except Exception:
            return 1.0

    @staticmethod
    def _path_synergy_score(X: pd.DataFrame, y: np.ndarray, f1: str, f2: str) -> float:
        """MI synergy proxy for path co-occurrence (not epistasis claim)."""
        first = pd.to_numeric(X[f1], errors="coerce").reset_index(drop=True)
        second = pd.to_numeric(X[f2], errors="coerce").reset_index(drop=True)
        target = pd.Series(y).reset_index(drop=True)
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

    # Compatibility wrappers (planned removal v2.0). Prefer the non-epistasis names.
    def _mine_epistatic_interactions(self, *args, **kwargs):
        """Deprecated alias of ``_mine_tree_path_interaction_candidates`` (removal_target=2.0)."""
        logger.warning(
            "Deprecated: _mine_epistatic_interactions; use "
            "_mine_tree_path_interaction_candidates (not biological epistasis)."
        )
        return self._mine_tree_path_interaction_candidates(*args, **kwargs)

    def _epistasis_strength(self, X, y, f1, f2):
        """Deprecated alias of ``_path_synergy_score`` (removal_target=2.0)."""
        return self._path_synergy_score(X, y, f1, f2)

    # ------------------------------------------------------------------
    # Export + Summary
    # ------------------------------------------------------------------
    def _export_results(
        self, results: Dict[str, Any], output_dir: Optional[str]
    ) -> None:
        if not output_dir:
            return

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        (out / self.artifacts.rules_txt).write_text(
            results["decision_trees"]["rules"],
            encoding="utf-8",
        )
        evidence = results.get("feature_evidence", {})
        if not evidence and isinstance(results.get("deprecated"), dict):
            evidence = results["deprecated"].get("feature_confidence", {})
        interactions = results.get("tree_path_interaction_candidates", [])
        if not interactions and isinstance(results.get("deprecated"), dict):
            interactions = results["deprecated"].get("epistatic_interactions", [])

        (out / self.artifacts.evidence_json).write_text(
            json.dumps(evidence, indent=2), encoding="utf-8"
        )
        (out / self.artifacts.interactions_json).write_text(
            json.dumps(interactions, indent=2), encoding="utf-8"
        )
        # Deprecated filenames for compatibility (planned removal v2.0)
        deprecated_blob = results.get("deprecated") or {
            "removal_target": "2.0",
            "feature_confidence": evidence,
            "epistatic_interactions": interactions,
        }
        (out / self.artifacts.legacy_confidence_json).write_text(
            json.dumps(
                {
                    "deprecated": True,
                    "removal_target": "2.0",
                    "use_instead": self.artifacts.evidence_json,
                    "payload": deprecated_blob.get("feature_confidence", evidence),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (out / self.artifacts.legacy_interactions_json).write_text(
            json.dumps(
                {
                    "deprecated": True,
                    "removal_target": "2.0",
                    "use_instead": self.artifacts.interactions_json,
                    "note": "Not biological epistasis; tree-path interaction candidates only.",
                    "payload": deprecated_blob.get(
                        "epistatic_interactions", interactions
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _print_summary(self, results: Dict[str, Any]) -> None:
        tree = results["decision_trees"]
        root_features = results.get("root_features", [])
        branch_features = results.get("branch_features", [])
        interactions = results.get(
            "tree_path_interaction_candidates",
            results.get("epistatic_interactions", []),
        )
        evidence_map = results.get(
            "feature_evidence", results.get("feature_confidence", {})
        )

        print("\n" + "=" * 70)
        print("FEATURE DISCOVERY SUMMARY (DecisionTree Branch)")
        print("=" * 70)
        training_accuracy = tree.get(
            "training_fit_accuracy", tree.get("training_accuracy", float("nan"))
        )
        print(
            f"Tree Training-Fit Accuracy: {training_accuracy:.3f} | "
            f"Classes: {tree['n_classes']} | Depth: {tree.get('tree_depth', '?')}"
        )
        print(
            f"Root Features (min depth≤1): {len(root_features)} | Branch Features: {len(branch_features)}"
        )

        if root_features:
            root_scores = [
                evidence_map.get(feat, {}).get("evidence_score", float("nan"))
                for feat in root_features
            ]
            finite = [float(x) for x in root_scores if np.isfinite(float(x))]
            if finite:
                print(f"Mean Root Feature Evidence Score: {float(np.mean(finite)):.3f}")
            print(
                "  Evidence fields (MI, Cramér V, bootstrap) written separately; not probability confidence."
            )
        else:
            print("  No root features identified.")

        n_val = sum(
            1
            for i in interactions
            if i.get("validation_status") == "validated_candidate"
        )
        print(
            f"Tree-path interaction candidates: {len(interactions)} "
            f"(validated_candidate={n_val}; not biological epistasis claims)"
        )
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
        Drop invariant columns using nunique / value counts (not clipped mean).
        """
        if X.empty:
            return []

        kept: List[str] = []
        dropped = 0
        for col in X.columns:
            series = pd.to_numeric(X[col], errors="coerce")
            nunique = int(series.nunique(dropna=True))
            if nunique <= 1:
                dropped += 1
                continue
            # Near-monomorphic: dominant state ≥ 99.9% of non-null samples
            vc = series.value_counts(dropna=True)
            if len(vc) and float(vc.iloc[0]) / max(1, int(vc.sum())) >= 0.999:
                dropped += 1
                continue
            kept.append(col)

        if dropped:
            logger.info(
                "Dropping %d invariant / near-monomorphic features before tree fitting (nunique/value-count rule).",
                dropped,
            )
        return kept

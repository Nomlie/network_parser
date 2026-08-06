# network_parser/network_parser.py
from __future__ import annotations

import json
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd

try:
    from network_parser.config import NetworkParserConfig
    from network_parser.data_loader import DataLoader
    from network_parser.statistical_validation_branch import StatisticalValidatorBranch
    from network_parser.ml_protocol import MLProtocolRunner
    from network_parser.feature_panel_selection import (
        log_model_input_panel_decision,
        run_feature_panel_separability_check,
    )
    from network_parser.utils import (
        save_json,
        ensure_dir,
        timestamp,
        normalize_sample_id,
        log_pipeline_header,
        log_stage_start,
        log_stage_complete,
        log_branch_decision,
        log_artifact,
        log_flow_step,
        log_final_run_summary,
        collect_common_warnings,
        write_run_audit,
        write_stage_checkpoint,
        load_stage_checkpoint,
        build_checkpoint_hashes,
        CHECKPOINT_SCHEMA_VERSION,
        PipelineProgress,
    )
except ImportError:  # pragma: no cover - package vs source-tree layout
    from config import NetworkParserConfig  # type: ignore
    from data_loader import DataLoader  # type: ignore
    from statistical_validation_branch import StatisticalValidatorBranch  # type: ignore
    from ml_protocol import MLProtocolRunner  # type: ignore
    from feature_panel_selection import (  # type: ignore
        log_model_input_panel_decision,
        run_feature_panel_separability_check,
    )
    from utils import (  # type: ignore
        save_json,
        ensure_dir,
        timestamp,
        normalize_sample_id,
        log_pipeline_header,
        log_stage_start,
        log_stage_complete,
        log_branch_decision,
        log_artifact,
        log_flow_step,
        log_final_run_summary,
        collect_common_warnings,
        write_run_audit,
        write_stage_checkpoint,
        load_stage_checkpoint,
        build_checkpoint_hashes,
        CHECKPOINT_SCHEMA_VERSION,
        PipelineProgress,
    )


def _planned_single_label_stages(
    config: NetworkParserConfig,
    mode: str,
    meta_path: Optional[str],
    validate_interactions: bool,
) -> List[str]:
    """Return the ordered stage labels used for the pipeline progress bar."""
    stages: List[str] = ["load and preprocess genomic matrix"]
    if meta_path:
        stages.append("load metadata")
    if mode == "matrix_only":
        stages.append("finalize results")
        return stages

    if meta_path:
        stages.append("supervised sample alignment")
    stages.append("central statistical filtering")
    stages.append("ranked feature-panel separability check")

    run_ml = mode in {"ml_only", "both"} or bool(
        getattr(config, "run_ml_protocol", False)
    )
    if run_ml:
        stages.append("ML protocol and model screening")
    if mode in {"both", "decision_tree_only"}:
        stages.append("conditional decision-tree interpretability")
        if validate_interactions:
            stages.append("optional post-tree interaction validation")
    stages.append("finalize results")
    return stages


try:
    from network_parser.decision_tree_branch import DecisionTreeBranch
except ImportError:  # pragma: no cover - package vs source-tree layout
    try:
        from decision_tree_branch import DecisionTreeBranch  # type: ignore
    except ImportError:
        DecisionTreeBranch = None  # type: ignore
logger = logging.getLogger(__name__)

# Explicit missing-label tokens only (no global punctuation rewriting).
DEFAULT_MISSING_LABEL_TOKENS = frozenset(
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
    *,
    explicit_mapping: Optional[Dict[str, str]] = None,
    missing_tokens: Optional[set] = None,
    allow_mapping_collisions: bool = False,
) -> pd.Series:
    """
    Normalize phenotype / class labels without silent punctuation rewriting.

    Rules
    -----
    1. Strip whitespace.
    2. Map explicit missing tokens → NA (configurable set).
    3. Apply optional ``explicit_mapping`` (old_label → new_label) with collision checks.
    4. Optionally lowercase.

    Global punctuation rewrites (e.g. ``-`` → ``_``) are intentionally absent:
    they can merge distinct biological classes (``A-B`` vs ``A_B``). Use an
    explicit mapping when renaming is required.
    """
    if not isinstance(labels, pd.Series):
        raise TypeError("labels must be a pandas Series")

    original_n = labels.shape[0]
    original_unique = labels.nunique(dropna=False)

    clean = labels.astype(str).str.strip()
    tokens = (
        set(missing_tokens)
        if missing_tokens is not None
        else set(DEFAULT_MISSING_LABEL_TOKENS)
    )
    clean = clean.replace(tokens, pd.NA)

    if explicit_mapping:
        mapping = {str(k).strip(): str(v).strip() for k, v in explicit_mapping.items()}
        # Detect many-to-one collisions introduced by the mapping (distinct
        # present labels that map to the same target).
        present = {str(v).strip() for v in clean.dropna().unique()}
        image_to_sources: Dict[str, List[str]] = {}
        for src in present:
            dst = mapping.get(src, src)
            image_to_sources.setdefault(dst, []).append(src)
        collisions = {
            dst: srcs for dst, srcs in image_to_sources.items() if len(srcs) > 1
        }
        if collisions and not allow_mapping_collisions:
            detail = "; ".join(
                f"{dst!r} <- {sorted(srcs)}" for dst, srcs in sorted(collisions.items())
            )
            raise ValueError(
                "Label explicit_mapping would merge distinct labels into the same "
                f"class (collision). Refusing silent merge: {detail}. "
                "Pass allow_mapping_collisions=True only if the merge is intentional."
            )
        if collisions and allow_mapping_collisions:
            logger.warning(
                "Label explicit_mapping merges distinct labels (allow_mapping_collisions=True): %s",
                collisions,
            )
        clean = clean.map(
            lambda v: mapping.get(str(v).strip(), v) if pd.notna(v) else v
        )

    if lowercase:
        clean = clean.map(lambda v: v.lower() if isinstance(v, str) else v)

    n_missing = int(clean.isna().sum())
    if drop_missing:
        clean = clean[~clean.isna()]

    final_unique = int(clean.nunique(dropna=False))
    final_n = int(clean.shape[0])

    logger.info(
        "Label normalization: original_n=%d | final_n=%d | missing_removed=%d | "
        "unique_before=%d | unique_after=%d",
        int(original_n),
        final_n,
        n_missing,
        int(original_unique),
        final_unique,
    )

    if n_missing > 0:
        logger.warning(
            "Label normalization removed %d sample label(s) due to missing/invalid phenotype values.",
            n_missing,
        )

    return clean


class NetworkParser:
    """
    Main orchestrator class for the NetworkParser pipeline.

    Updated orchestration:
      1) Load genomic matrix / metadata
      2) Align samples between X and y
      3) Run central feature filtering once
      4) Run ranked feature-panel separability check
      5) Run ML protocol / model selector on selected model matrix
      6) Conditionally trigger decision-tree interpretability branch
      7) Optionally validate post-tree interactions
    """

    def __init__(self, config: NetworkParserConfig):
        self.config = config
        logger.info(
            "NetworkParser initialized | mode=%s | central_filter=%s | panel_check=%s | n_jobs=%s",
            getattr(config, "pipeline_mode", "NA"),
            getattr(config, "central_feature_filter_method", "auto"),
            bool(getattr(config, "run_feature_panel_separability_check", True)),
            getattr(config, "n_jobs", "NA"),
        )
        logger.debug("NetworkParser full config: %s", vars(config))

        self.loader = DataLoader(config=config, n_jobs=config.n_jobs)
        self.validator = StatisticalValidatorBranch(config)
        self.tree_builder = DecisionTreeBranch(config)

    # ------------------------------------------------------------------
    # Alignment helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_sample_id(x: str) -> str:
        return normalize_sample_id(x)

    def _align_X_y(
        self,
        genomic_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        label_column: str,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if label_column not in meta_df.columns:
            raise ValueError(
                f"label_column '{label_column}' not found in metadata columns"
            )

        labels = normalize_labels(
            meta_df[label_column], drop_missing=True, lowercase=False
        )

        genomic_df = genomic_df.copy()
        genomic_df.index = genomic_df.index.astype(str).map(self._normalize_sample_id)

        labels.index = labels.index.astype(str).map(self._normalize_sample_id)

        common = genomic_df.index.intersection(labels.index)

        logger.info(
            "Sample alignment: genomic=%d | meta=%d | labels_after_norm=%d | overlap=%d",
            int(genomic_df.shape[0]),
            int(meta_df.shape[0]),
            int(labels.shape[0]),
            int(common.shape[0]),
        )

        if len(common) == 0:
            raise ValueError(
                "No overlapping sample IDs between genomic matrix and metadata after label normalization."
            )

        X = genomic_df.loc[common].copy()
        y = labels.loc[common].copy()

        log_flow_step(
            logger,
            step="Preprocessing checkpoint — supervised sample alignment",
            happened="Intersected genomic-matrix sample identifiers with non-missing metadata labels and kept only aligned samples.",
            reason="Supervised statistical filtering and model screening require each retained sample to have both genomic features and a valid target label.",
            before_samples=int(genomic_df.shape[0]),
            before_features=int(genomic_df.shape[1]),
            after_samples=int(X.shape[0]),
            after_features=int(X.shape[1]),
            threshold="sample_id present in both matrix and metadata; label non-missing",
            status="complete",
        )

        return X, y

    # ------------------------------------------------------------------
    # Central feature filtering
    # ------------------------------------------------------------------
    def _run_central_feature_filter(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        output_dir: str,
        enabled: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
        """
        Shared central feature filtering stage used BEFORE model selection and
        BEFORE any downstream algorithm-specific branch.

        Preferred current behaviour
        ---------------------------
        config.central_feature_filter_method controls the central filter:
            - rf_fdr
            - chi2_fdr
            - fisher_fdr
            - chi2_perm_fdr

        The legacy run_rf_fdr_feature_selection flag is still honoured when
        central_feature_filter_method="auto".

        This stage is PRE-ML / PRE-tree.

        It does not perform:
            - decision-tree construction
            - post-tree bootstrap stability
            - path-based epistasis mining
            - interaction permutation testing
        """
        stats_dir = Path(output_dir) / "central_feature_filtering"
        stats_dir.mkdir(parents=True, exist_ok=True)

        if not enabled:
            log_flow_step(
                logger,
                step="Central filtering checkpoint — disabled",
                happened="Passed the aligned matrix forward without central statistical feature filtering.",
                reason="The user configuration disabled the central filtering stage; this is useful for controlled testing, but publication-grade discovery should keep pre-model statistical filtering enabled.",
                before_samples=int(X.shape[0]),
                before_features=int(X.shape[1]),
                after_samples=int(X.shape[0]),
                after_features=int(X.shape[1]),
                threshold="run_central_feature_filtering=False",
                status="skipped",
            )
            summary = {
                "method": "none",
                "status": "skipped",
                "input_features": int(X.shape[1]),
                "retained_features": int(X.shape[1]),
                "retention_fraction": 1.0,
                "used_fallback_unfiltered_matrix": False,
                "filtered_matrix": str(stats_dir / "filtered_matrix.csv"),
            }
            X.to_csv(stats_dir / "filtered_matrix.csv")
            with open(
                stats_dir / "feature_filtering_summary.json", "w", encoding="utf-8"
            ) as fh:
                json.dump(summary, fh, indent=2)
            return X.copy(), summary, {"summary": summary}

        logger.info("Stage 3: Central feature filtering")

        filter_method = str(
            getattr(
                self.config,
                "resolved_central_feature_filter_method",
                getattr(self.config, "central_feature_filter_method", "auto"),
            )
        ).lower()

        if filter_method == "auto":
            if bool(getattr(self.config, "run_rf_fdr_feature_selection", False)):
                filter_method = "rf_fdr"
            elif (
                str(getattr(self.config, "statistical_test", "chi2")).lower()
                == "fisher"
            ):
                filter_method = "fisher_fdr"
            else:
                filter_method = "chi2_fdr"

        filter_reason = {
            "rf_fdr": "Uses repeated random-forest importance estimates against label permutations, then applies FDR correction before any model is trained.",
            "chi2_fdr": "Uses per-feature chi-square association tests with multiple-testing correction before model screening.",
            "fisher_fdr": "Uses Fisher exact testing where appropriate, followed by multiple-testing correction before model screening.",
            "chi2_perm_fdr": "Uses empirical chi-square association scores from label permutations, then applies FDR correction before model screening.",
        }.get(
            filter_method, "Runs central statistical filtering before model screening."
        )

        log_flow_step(
            logger,
            step=f"Central filtering checkpoint — {filter_method}",
            happened="Started pre-model statistical filtering on the aligned feature matrix.",
            reason=filter_reason,
            before_samples=int(X.shape[0]),
            before_features=int(X.shape[1]),
            threshold=f"method={filter_method}",
            status="started",
        )

        if filter_method == "rf_fdr":
            logger.info("Central feature filtering method: RF-FDR feature selection.")

            rf_result = self.validator.rf_fdr_feature_selection(
                genomic_df=X,
                labels=y,
                output_dir=str(stats_dir),
            )

            X_filtered = rf_result["filtered_matrix"].copy()
            summary = dict(rf_result.get("summary", {}))
            summary.setdefault("method", "rf_fdr")
            summary.setdefault("status", "success")
            summary_artifacts = summary.setdefault("artifacts", {})
            if not isinstance(summary_artifacts, dict):
                summary_artifacts = {}
                summary["artifacts"] = summary_artifacts
            summary_artifacts.setdefault(
                "filter_dir",
                str(stats_dir),
            )
            summary_artifacts.setdefault(
                "filtered_matrix",
                str(stats_dir / "filtered_matrix.csv"),
            )

            log_flow_step(
                logger,
                step="Central filtering checkpoint — complete",
                happened="Retained the FDR-supported feature subset for model screening and optional tree construction.",
                reason="Only features with statistical evidence after permutation/FDR control should move into downstream model interpretation with confidence.",
                before_samples=int(X.shape[0]),
                before_features=int(X.shape[1]),
                after_samples=int(X_filtered.shape[0]),
                after_features=int(X_filtered.shape[1]),
                threshold=f"method={filter_method}",
                status=str(summary.get("status", "success")),
                artifact=summary_artifacts.get("filtered_matrix"),
            )

            return X_filtered, summary, rf_result

        if filter_method == "chi2_perm_fdr":
            logger.info(
                "Central feature filtering method: chi-square permutation testing + FDR correction."
            )

            perm_result = self.validator.chi2_permutation_feature_selection(
                genomic_df=X,
                labels=y,
                output_dir=str(stats_dir),
                stage_name="central_feature_filtering",
            )

            X_filtered = perm_result["filtered_matrix"].copy()
            summary = dict(perm_result.get("summary", {}))
            summary.setdefault("method", "chi2_perm_fdr")
            summary.setdefault("status", "success")
            summary_artifacts = summary.setdefault("artifacts", {})
            if not isinstance(summary_artifacts, dict):
                summary_artifacts = {}
                summary["artifacts"] = summary_artifacts
            summary_artifacts.setdefault("filter_dir", str(stats_dir))
            summary_artifacts.setdefault(
                "filtered_matrix", str(stats_dir / "filtered_matrix.csv")
            )

            log_flow_step(
                logger,
                step="Central filtering checkpoint — complete",
                happened="Retained the permutation-FDR-supported feature subset for model screening and optional tree construction.",
                reason="Permutation-derived empirical p-values reduce reliance on asymptotic assumptions before FDR correction.",
                before_samples=int(X.shape[0]),
                before_features=int(X.shape[1]),
                after_samples=int(X_filtered.shape[0]),
                after_features=int(X_filtered.shape[1]),
                threshold=f"method={filter_method}",
                status=str(summary.get("status", "success")),
                artifact=summary_artifacts.get("filtered_matrix"),
            )

            return X_filtered, summary, perm_result

        if filter_method not in {"chi2_fdr", "fisher_fdr"}:
            raise ValueError(
                "central_feature_filter_method must resolve to one of: "
                "'rf_fdr', 'chi2_fdr', 'fisher_fdr', or 'chi2_perm_fdr'"
            )

        self.config.statistical_test = (
            "fisher" if filter_method == "fisher_fdr" else "chi2"
        )

        logger.info(
            "Central feature filtering method: %s association testing + multiple testing correction.",
            self.config.statistical_test,
        )

        assoc = self.validator.association_tests(
            data=X,
            labels=y,
            output_dir=str(stats_dir),
        )

        corrected = self.validator.multiple_testing_correction(
            test_results=assoc,
            output_dir=str(stats_dir),
        )

        significant_features = [
            feat
            for feat, res in corrected.items()
            if bool(res.get("significant", False))
        ]
        significant_features = [f for f in significant_features if f in X.columns]
        fallback_strategy = str(
            getattr(self.config, "feature_filter_fallback_strategy", "stop")
        ).lower()

        used_fallback = False

        if significant_features:
            X_filtered = X.loc[:, significant_features].copy()
            retained_feature_names = list(X_filtered.columns)

        elif fallback_strategy == "stop":
            raise ValueError(
                "Central association-FDR filtering retained no significant genomic features. "
                "Stopping is statistically defensible for publication-grade runs. "
                "For exploratory smoke testing only, set "
                "feature_filter_fallback_strategy='unfiltered'."
            )

        elif fallback_strategy == "unfiltered":
            logger.warning(
                "Central association-FDR filtering retained no significant genomic features. "
                "Using the aligned matrix as an exploratory fallback. "
                "Do not report downstream markers from this fallback as FDR-supported discoveries."
            )
            X_filtered = X.copy()
            retained_feature_names = list(X_filtered.columns)
            used_fallback = True

        else:
            raise ValueError(
                "feature_filter_fallback_strategy must be one of: 'stop' or 'unfiltered'."
            )

        filtered_matrix_path = stats_dir / "filtered_matrix.csv"
        X_filtered.to_csv(filtered_matrix_path)

        summary = {
            "method": filter_method,
            "status": "success",
            "input_features": int(X.shape[1]),
            "tested_features": int(len(assoc)),
            "significant_features": int(len(significant_features)),
            "retained_features": int(X_filtered.shape[1]),
            "fallback_strategy": fallback_strategy,
            "retention_fraction": float(X_filtered.shape[1] / max(1, X.shape[1])),
            "used_fallback_unfiltered_matrix": bool(used_fallback),
            "retained_feature_names": retained_feature_names,
            "artifacts": {
                "filter_dir": str(stats_dir),
                "association_json": str(stats_dir / "chi_squared_results.json"),
                "multiple_testing_json": str(
                    stats_dir / "multiple_testing_results.json"
                ),
                "filtered_matrix": str(filtered_matrix_path),
                "summary_json": str(stats_dir / "feature_filtering_summary.json"),
            },
        }

        with open(
            stats_dir / "feature_filtering_summary.json", "w", encoding="utf-8"
        ) as fh:
            json.dump(summary, fh, indent=2)

        log_flow_step(
            logger,
            step="Central filtering checkpoint — complete",
            happened="Retained the association-FDR-supported feature subset for model screening and optional tree construction.",
            reason="Association testing and multiple-testing correction occur before model training so downstream interpretation is based on a statistically defensible filtered matrix.",
            before_samples=int(X.shape[0]),
            before_features=int(X.shape[1]),
            after_samples=int(X_filtered.shape[0]),
            after_features=int(X_filtered.shape[1]),
            threshold=f"method={filter_method}; fallback={fallback_strategy}",
            status="fallback_unfiltered" if used_fallback else "success",
            artifact=str(filtered_matrix_path),
        )

        filter_result = {
            "method": filter_method,
            "summary": summary,
            "association": assoc,
            "multiple_testing": corrected,
            "retained_features": retained_feature_names,
            "filtered_matrix": X_filtered,
        }

        return X_filtered, summary, filter_result

    # ------------------------------------------------------------------
    # Decision-tree trigger logic
    # ------------------------------------------------------------------
    def _dt_meets_top_k_margin_rule(
        self,
        ml_results: Dict[str, Any],
    ) -> bool:
        """
        True when DT probe score is among top-k finite scores and within
        max_margin of the best finite probe score.
        """
        top_k = max(1, int(getattr(self.config, "decision_tree_candidate_top_k", 2)))
        max_margin = float(
            getattr(self.config, "decision_tree_candidate_max_margin", 0.05)
        )

        selector = (
            ml_results.get("selector", {}) if isinstance(ml_results, dict) else {}
        )
        if not isinstance(selector, dict):
            selector = {}
        probe_scores = selector.get("probe_scores", {})
        if not isinstance(probe_scores, dict):
            probe_scores = {}

        # Prefer normalized algorithm names (DT, RF, LR, SVC, MLP).
        name_map = {
            "DT": "DT",
            "RF": "RF",
            "LR": "LR",
            "LinearSVC": "SVC",
            "SVC_RBF": "SVC",
            "SVC": "SVC",
            "MLP_small": "MLP",
            "MLP": "MLP",
        }
        scored: List[Tuple[str, float]] = []
        for raw_name, raw_score in probe_scores.items():
            if str(raw_name) == "delta_nonlinear_minus_linear":
                continue
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(score):
                continue
            algo = name_map.get(str(raw_name).strip(), str(raw_name).strip())
            scored.append((algo, score))

        if not scored:
            return False

        # Keep best score per algorithm, then rank.
        best_by_algo: Dict[str, float] = {}
        for algo, score in scored:
            prev = best_by_algo.get(algo)
            if prev is None or score > prev:
                best_by_algo[algo] = score

        ranked = sorted(best_by_algo.items(), key=lambda kv: kv[1], reverse=True)
        best_score = ranked[0][1]
        top_algos = [name for name, _ in ranked[:top_k]]
        dt_score = best_by_algo.get("DT")
        if dt_score is None:
            return False
        in_top_k = "DT" in top_algos
        within_margin = (best_score - dt_score) <= max_margin + 1e-12
        logger.info(
            "DT top-k/margin rule | top_k=%d | max_margin=%.4f | best=%.4f | "
            "dt=%.4f | in_top_k=%s | within_margin=%s | ranked=%s",
            top_k,
            max_margin,
            best_score,
            dt_score,
            in_top_k,
            within_margin,
            [(n, round(s, 4)) for n, s in ranked],
        )
        return bool(in_top_k and within_margin)

    def _should_run_decision_tree(
        self,
        mode: str,
        ml_results: Dict[str, Any],
    ) -> bool:
        """
        Resolve whether the decision-tree interpretability branch should run.

        Policy
        ------
        - matrix_only: never
        - decision_tree_only: always
        - ml_only: never
        - both / mixed:
            * always if decision_tree_always_run_interpretability=True
              (documented always-run interpretability stage)
            * always if ml_algorithm=DT
            * if decision_tree_requires_selector_match: only when selected/recommended is DT
            * if trigger_decision_tree_on_selected and selected/recommended is DT
            * if trigger_decision_tree_if_candidate: only when DT meets the
              explicit top-k + margin rule (not merely presence in a longlist)
        """
        if mode == "matrix_only":
            return False

        if mode == "decision_tree_only":
            return True

        if mode == "ml_only":
            return False

        if not bool(getattr(self.config, "run_conditional_dt", True)):
            logger.info(
                "Decision tree branch trigger: disabled by config.run_conditional_dt=False"
            )
            return False

        if bool(
            getattr(self.config, "decision_tree_always_run_interpretability", False)
        ):
            logger.info(
                "Decision tree branch trigger: always-run interpretability stage "
                "(decision_tree_always_run_interpretability=True)"
            )
            return True

        branch_decision = (
            ml_results.get("branch_decision", {})
            if isinstance(ml_results, dict)
            else {}
        )

        selected_algorithm = str(
            branch_decision.get(
                "selected_algorithm", ml_results.get("selected_algorithm", "")
            )
        ).strip()

        selector_recommendation = str(
            branch_decision.get(
                "selector_recommendation",
                ml_results.get("selector", {}).get("recommendation", ""),
            )
        ).strip()

        configured_algorithm = str(getattr(self.config, "ml_algorithm", "auto")).strip()
        trigger_on_selected = bool(
            getattr(self.config, "trigger_decision_tree_on_selected", True)
        )
        trigger_if_candidate = bool(
            getattr(self.config, "trigger_decision_tree_if_candidate", True)
        )
        requires_match = bool(
            getattr(self.config, "decision_tree_requires_selector_match", False)
        )

        if configured_algorithm == "DT":
            logger.info(
                "Decision tree branch trigger: running because config.ml_algorithm=DT"
            )
            return True

        if requires_match:
            run_dt = selected_algorithm == "DT" or selector_recommendation == "DT"
            logger.info(
                "Decision tree branch trigger with strict selector match | selected=%s | recommended=%s | run=%s",
                selected_algorithm or "n/a",
                selector_recommendation or "n/a",
                run_dt,
            )
            return run_dt

        if trigger_on_selected and selected_algorithm == "DT":
            logger.info(
                "Decision tree branch trigger: running because selected_algorithm=DT"
            )
            return True

        if trigger_on_selected and selector_recommendation == "DT":
            logger.info(
                "Decision tree branch trigger: running because selector recommendation=DT"
            )
            return True

        if trigger_if_candidate and self._dt_meets_top_k_margin_rule(ml_results):
            logger.info(
                "Decision tree branch trigger: running because DT meets top-k/margin candidate rule"
            )
            return True

        logger.info(
            "Decision tree branch trigger: not activated | selected=%s | recommended=%s",
            selected_algorithm or "n/a",
            selector_recommendation or "n/a",
        )
        return False

    # ------------------------------------------------------------------
    # Interaction extraction helper
    # ------------------------------------------------------------------
    def _extract_interaction_pairs(
        self,
        interaction_payload: Any,
    ) -> List[Tuple[str, str]]:
        """
        Normalize interaction payload from DT branch into a list of feature pairs.
        """
        pairs: List[Tuple[str, str]] = []

        if not interaction_payload:
            return pairs

        if isinstance(interaction_payload, list):
            for item in interaction_payload:
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    f1, f2 = str(item[0]), str(item[1])
                    if f1 and f2:
                        pairs.append((f1, f2))
                    continue

                if isinstance(item, dict):
                    if (
                        "pair" in item
                        and isinstance(item["pair"], (tuple, list))
                        and len(item["pair"]) >= 2
                    ):
                        f1, f2 = str(item["pair"][0]), str(item["pair"][1])
                        if f1 and f2:
                            pairs.append((f1, f2))
                        continue

                    if (
                        "features" in item
                        and isinstance(item["features"], (tuple, list))
                        and len(item["features"]) >= 2
                    ):
                        f1, f2 = str(item["features"][0]), str(item["features"][1])
                        if f1 and f2:
                            pairs.append((f1, f2))
                        continue

                    keys = list(item.keys())
                    if len(keys) >= 2 and all(isinstance(k, str) for k in keys[:2]):
                        f1, f2 = str(keys[0]), str(keys[1])
                        if f1 and f2:
                            pairs.append((f1, f2))

        seen = set()
        uniq_pairs: List[Tuple[str, str]] = []
        for pair in pairs:
            if pair not in seen:
                uniq_pairs.append(pair)
                seen.add(pair)

        return uniq_pairs

    # ------------------------------------------------------------------
    # Audit / checkpoint helpers
    # ------------------------------------------------------------------
    def _pipeline_config_subset(self) -> Dict[str, Any]:
        """Stable config fields that invalidate checkpoints when changed."""
        keys = (
            "pipeline_mode",
            "central_feature_filter_method",
            "run_central_feature_filtering",
            "run_feature_panel_separability_check",
            "feature_panel_sizes",
            "feature_panel_min_score",
            "feature_panel_selection_rule",
            "feature_panel_always_include_full_filtered",
            "feature_panel_threshold_failure_strategy",
            "feature_panel_metric",
            "feature_panel_classifier",
            "ml_algorithm",
            "run_model_selector",
            "run_conditional_dt",
            "decision_tree_candidate_top_k",
            "decision_tree_candidate_max_margin",
            "decision_tree_always_run_interpretability",
            "n_jobs",
            "random_state",
            "qual_threshold",
            "min_dp_per_sample",
            "biallelic_only",
            "assume_absent_variant_is_reference",
        )
        cfg = self.config
        return {k: getattr(cfg, k, None) for k in keys}

    def _build_stage_hashes(
        self,
        *,
        input_paths: Optional[Dict[str, Any]] = None,
        content_objects: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return build_checkpoint_hashes(
            input_paths=input_paths,
            config_subset=self._pipeline_config_subset(),
            content_objects=content_objects,
            schema_version=CHECKPOINT_SCHEMA_VERSION,
        )

    def _write_checkpoint(
        self,
        output_dir: Path,
        stage_name: str,
        payload: Dict[str, Any],
        *,
        status: str = "complete",
        hashes: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        if not bool(getattr(self.config, "write_stage_checkpoints", True)):
            return None
        try:
            path = write_stage_checkpoint(
                output_dir,
                stage_name,
                payload,
                status=status,
                hashes=hashes,
                schema_version=CHECKPOINT_SCHEMA_VERSION,
            )
            logger.debug("Stage checkpoint written: %s", path)
            return path
        except (
            Exception
        ) as exc:  # pragma: no cover - checkpointing must never break analysis
            logger.warning("Could not write checkpoint for %s: %s", stage_name, exc)
            return None

    def _checkpoint_matrix_path(self, output_dir: Path, name: str) -> Path:
        checkpoint_root = output_dir / "_checkpoints"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        return checkpoint_root / f"{name}.csv"

    def _write_matrix_checkpoint(
        self, output_dir: Path, name: str, matrix: pd.DataFrame
    ) -> Optional[Path]:
        if not bool(getattr(self.config, "write_stage_checkpoints", True)):
            return None
        try:
            path = self._checkpoint_matrix_path(output_dir, name)
            matrix.to_csv(path)
            return path
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not write matrix checkpoint %s: %s", name, exc)
            return None

    def _load_matrix_checkpoint(
        self,
        output_dir: Path,
        name: str,
        *,
        expected_hashes: Optional[Dict[str, Any]] = None,
        stage_name: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        if not bool(getattr(self.config, "resume_from_checkpoints", False)):
            return None
        # When stage metadata exists, enforce hash compatibility before reusing matrix.
        if stage_name is not None and expected_hashes is not None:
            meta = load_stage_checkpoint(
                output_dir,
                stage_name,
                expected_hashes=expected_hashes,
                refuse_incompatible=True,
            )
            if meta is None:
                return None
        path = self._checkpoint_matrix_path(output_dir, name)
        if not path.exists():
            return None
        try:
            return pd.read_csv(path, index_col=0)
        except Exception as exc:
            logger.warning("Could not load matrix checkpoint %s: %s", path, exc)
            return None

    def _summarize_known_markers(
        self,
        known_markers: Optional[List[str]],
        feature_names: List[str],
        output_dir: Path,
    ) -> Dict[str, Any]:
        """
        Documented downstream behaviour for --known_markers:

        Compare the provided marker list to the current feature space and write
        an overlap report. Markers are NOT used for filtering or model training;
        they are provenance / annotation comparison only.
        """
        if not known_markers:
            return {
                "status": "not_provided",
                "n_known_markers": 0,
                "n_overlap_with_features": 0,
                "n_missing_from_features": 0,
                "role": "comparison_only_not_used_for_filtering_or_training",
            }
        feature_set = {str(f) for f in feature_names}
        known = [str(m) for m in known_markers]
        overlap = [m for m in known if m in feature_set]
        missing = [m for m in known if m not in feature_set]
        report = {
            "status": "compared",
            "n_known_markers": int(len(known)),
            "n_overlap_with_features": int(len(overlap)),
            "n_missing_from_features": int(len(missing)),
            "overlap_fraction": float(len(overlap) / len(known)) if known else 0.0,
            "role": "comparison_only_not_used_for_filtering_or_training",
            "overlap_markers_preview": overlap[:50],
            "missing_markers_preview": missing[:50],
        }
        try:
            report_path = Path(output_dir) / "known_markers_feature_overlap.json"
            save_json(report, report_path)
            report["artifact"] = str(report_path)
            log_artifact(logger, "known-markers feature overlap", report_path)
        except Exception as exc:
            logger.warning("Could not write known-markers overlap report: %s", exc)
        logger.info(
            "Known markers comparison | provided=%d | overlap=%d | missing_from_matrix=%d "
            "(markers are comparison-only; not used for filtering or training)",
            report["n_known_markers"],
            report["n_overlap_with_features"],
            report["n_missing_from_features"],
        )
        return report

    def _write_audit_and_summary(
        self,
        *,
        output_dir: Path,
        results: Dict[str, Any],
        results_path: Path,
        title: str = "NetworkParser final run summary",
    ) -> Optional[Path]:
        warnings = collect_common_warnings(results)

        audit_path: Optional[Path] = None
        if bool(getattr(self.config, "write_run_audit", True)):
            audit_payload = {
                "timestamp": timestamp(),
                "pipeline_mode": results.get("pipeline_mode"),
                "config": results.get("config", {}),
                "input_summary": {
                    "aligned_matrix_shape": results.get("aligned_matrix_shape"),
                    "matrix_shape": results.get("matrix_shape"),
                    "central_filtered_matrix_shape": results.get(
                        "central_filtered_matrix_shape"
                    ),
                    "model_matrix_shape": results.get("model_matrix_shape"),
                },
                "stage_summaries": {
                    "feature_filtering": results.get("feature_filtering", {}),
                    "feature_panel_separability": results.get(
                        "feature_panel_separability", {}
                    ),
                    "ml_protocol": {
                        "status": results.get("ml_protocol", {}).get("status")
                        if isinstance(results.get("ml_protocol"), dict)
                        else None,
                        "selected_algorithm": results.get("ml_protocol", {}).get(
                            "selected_algorithm"
                        )
                        if isinstance(results.get("ml_protocol"), dict)
                        else None,
                    },
                    "decision_tree": {
                        "status": "generated"
                        if results.get("discovery")
                        else "not_generated",
                        "discovered_features": len(
                            results.get("discovery", {}).get("discovered_features", [])
                        )
                        if isinstance(results.get("discovery"), dict)
                        else 0,
                    },
                    "validation": results.get("validation", {}),
                },
                "warnings": warnings,
                "artifacts": {
                    "results_json": str(results_path),
                    "checkpoint_dir": str(output_dir / "_checkpoints"),
                },
            }
            audit_path = write_run_audit(output_dir, audit_payload)
            log_artifact(logger, "run audit", audit_path)

        if bool(getattr(self.config, "write_final_run_summary", True)):
            if results.get("pipeline_mode") in {"hierarchy", "two_level"}:
                registry = (
                    results.get("hierarchy_protocol")
                    if isinstance(results.get("hierarchy_protocol"), dict)
                    else None
                )
                if registry is None:
                    registry = (
                        results.get("two_level_protocol", {})
                        if isinstance(results.get("two_level_protocol"), dict)
                        else {}
                    )
                level1 = (
                    registry.get("level1", {})
                    if isinstance(registry.get("level1"), dict)
                    else {}
                )
                level2 = (
                    registry.get("level2", {})
                    if isinstance(registry.get("level2"), dict)
                    else {}
                )
                by_group = (
                    level2.get("by_level1_group", {})
                    if isinstance(level2.get("by_level1_group"), dict)
                    else {}
                )
                global_fallback = (
                    level2.get("global_fallback", {})
                    if isinstance(level2.get("global_fallback"), dict)
                    else {}
                )
                global_binary = (
                    level2.get("global_binary_fallback", {})
                    if isinstance(level2.get("global_binary_fallback"), dict)
                    else {}
                )
                sections = [
                    {
                        "name": "Hierarchy training",
                        "message": "The run trained a hierarchical registry that first places samples into genomic context and then evaluates nested phenotype/profile endpoints under parent branches.",
                        "fields": {
                            "registry": registry.get(
                                "registry_file", "hierarchical_model_registry.json"
                            ),
                        },
                    },
                    {
                        "name": "Level 1",
                        "message": "The Level-1 model uses the configured filtered feature space for strain, lineage, or group placement before phenotype interpretation.",
                        "fields": {
                            "features": len(level1.get("features", []))
                            if isinstance(level1.get("features", []), list)
                            else None,
                            "status": level1.get("status", "trained"),
                        },
                    },
                    {
                        "name": "Global Level 2",
                        "message": "The global Level-2 fallback provides a documented endpoint when group-specific training is unavailable or under-supported.",
                        "fields": {
                            "status": global_fallback.get("status"),
                            "features": len(global_fallback.get("features", []))
                            if isinstance(global_fallback.get("features", []), list)
                            else None,
                        },
                    },
                    {
                        "name": "Group-specific Level 2",
                        "message": "Group-specific Level-2 models are retained where the data support filtered, context-aware phenotype/profile training.",
                        "fields": {
                            "groups_recorded": len(by_group),
                        },
                    },
                    {
                        "name": "Binary Level-2 fallback",
                        "message": "The optional resistant/susceptible fallback is recorded separately so broad fallback behaviour is explicit rather than hidden.",
                        "fields": {
                            "status": global_binary.get("status", "not_requested"),
                        },
                    },
                ]
            else:
                aligned = (
                    results.get("aligned_matrix_shape")
                    or results.get("matrix_shape")
                    or {}
                )
                filtered = results.get("central_filtered_matrix_shape") or {}
                model_shape = results.get("model_matrix_shape") or aligned or {}
                feature_filtering = (
                    results.get("feature_filtering", {})
                    if isinstance(results.get("feature_filtering"), dict)
                    else {}
                )
                panel = (
                    results.get("feature_panel_separability", {})
                    if isinstance(results.get("feature_panel_separability"), dict)
                    else {}
                )
                ml = (
                    results.get("ml_protocol", {})
                    if isinstance(results.get("ml_protocol"), dict)
                    else {}
                )
                discovery = (
                    results.get("discovery", {})
                    if isinstance(results.get("discovery"), dict)
                    else {}
                )

                sections = [
                    {
                        "name": "Input/preprocessing",
                        "message": "The genomic matrix was loaded and structurally cleaned so downstream steps receive a model-ready sample × feature matrix.",
                        "fields": {
                            "samples": aligned.get("samples"),
                            "features": aligned.get("features"),
                        },
                    },
                    {
                        "name": "Central filtering",
                        "message": "Only features retained by the configured pre-model statistical screen were forwarded unless the run explicitly used a documented fallback.",
                        "fields": {
                            "method": feature_filtering.get("method"),
                            "retained_features": filtered.get("features")
                            or feature_filtering.get("retained_features"),
                        },
                    },
                    {
                        "name": "Feature panel",
                        "message": "The model matrix was selected from the ranked filtered feature space to keep the downstream analysis compact and interpretable.",
                        "fields": {
                            "selected_features": model_shape.get("features"),
                            "status": panel.get("reason", panel.get("status")),
                        },
                    },
                    {
                        "name": "ML protocol",
                        "message": "Model screening used the filtered matrix as input, preserving the separation between statistical feature selection and classifier choice.",
                        "fields": {
                            "status": ml.get("status") if ml else "skipped",
                            "selected_algorithm": ml.get("selected_algorithm")
                            if ml
                            else None,
                        },
                    },
                    {
                        "name": "Decision-tree branch",
                        "message": "Decision-tree output is used for interpretable rules and path-based interaction evidence after filtering, not as the primary statistical screen.",
                        "fields": {
                            "status": "generated" if discovery else "skipped",
                            "discovered_features": len(
                                discovery.get("discovered_features", [])
                            )
                            if discovery
                            else 0,
                        },
                    },
                ]
            artifacts: Dict[str, Any] = {"results_json": results_path}
            if audit_path is not None:
                artifacts["run_audit_json"] = audit_path
            log_final_run_summary(
                logger,
                title=title,
                sections=sections,
                artifacts=artifacts,
                warnings=warnings,
            )

        return audit_path

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def run_pipeline(
        self,
        genomic_path: str,
        meta_path: Optional[str],
        label_column: Optional[str],
        known_markers_path: Optional[str],
        output_dir: str,
        validate_statistics: bool = False,
        validate_interactions: bool = False,
        ref_fasta: Optional[str] = None,
        level1_label_column: Optional[str] = None,
        level2_label_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute NetworkParser pipeline using the updated logic:

          load -> align -> central feature filtering -> feature-panel separability
               -> ML protocol/model selector -> conditional DT -> optional interaction validation
        """
        output_dir_path = Path(output_dir)
        ensure_dir(output_dir_path)
        mode = getattr(self.config, "pipeline_mode", "decision_tree_only")
        log_pipeline_header(
            logger,
            "NetworkParser training/discovery run started",
            mode=mode,
            central_filter=getattr(
                self.config, "central_feature_filter_method", "auto"
            ),
            feature_panel_check=bool(
                getattr(self.config, "run_feature_panel_separability_check", True)
            ),
            n_jobs=getattr(self.config, "n_jobs", "NA"),
        )

        progress_stages = (
            []
            if mode in {"hierarchy", "two_level"}
            else _planned_single_label_stages(
                self.config, mode, meta_path, validate_interactions
            )
        )
        progress_ctx = (
            PipelineProgress(progress_stages, title="NetworkParser pipeline")
            if progress_stages
            else nullcontext()
        )

        with progress_ctx as pipeline_progress:
            return self._run_single_label_pipeline_body(
                genomic_path=genomic_path,
                meta_path=meta_path,
                label_column=label_column,
                known_markers_path=known_markers_path,
                output_dir=output_dir,
                validate_statistics=validate_statistics,
                validate_interactions=validate_interactions,
                ref_fasta=ref_fasta,
                level1_label_column=level1_label_column,
                level2_label_column=level2_label_column,
                mode=mode,
                pipeline_progress=pipeline_progress,
            )

    def _run_single_label_pipeline_body(
        self,
        *,
        genomic_path: str,
        meta_path: Optional[str],
        label_column: Optional[str],
        known_markers_path: Optional[str],
        output_dir: str,
        validate_statistics: bool,
        validate_interactions: bool,
        ref_fasta: Optional[str],
        level1_label_column: Optional[str],
        level2_label_column: Optional[str],
        mode: str,
        pipeline_progress: PipelineProgress | None,
    ) -> Dict[str, Any]:
        output_dir_path = Path(output_dir)

        if mode in {"hierarchy", "two_level"}:
            if meta_path is None:
                raise ValueError("meta_path is required for hierarchy mode.")

            resolved_level1_label = (
                level1_label_column
                or getattr(self.config, "level1_label_column", None)
                or label_column
            )

            resolved_level2_label = level2_label_column or getattr(
                self.config, "level2_label_column", None
            )

            if not resolved_level1_label:
                raise ValueError(
                    "hierarchy mode requires a Level 1 label column. "
                    "Provide label_column or config.level1_label_column."
                )

            if not resolved_level2_label:
                raise ValueError(
                    "hierarchy mode requires a Level 2 label column. "
                    "Provide level2_label_column or config.level2_label_column."
                )

            log_stage_start(
                logger,
                1,
                "hierarchy protocol handoff",
                level1_label="configured",
                level2_label="configured",
            )
            log_flow_step(
                logger,
                step="Branch checkpoint — hierarchy training",
                happened="Routed the run into the hierarchical protocol.",
                reason="Two-level interpretation first places samples in a genomic context, then evaluates the Level-2 phenotype/profile within that context or via a documented fallback.",
                status="selected",
            )

            try:
                from network_parser.hierarchy_protocol import HierarchyProtocol
            except Exception:  # pragma: no cover
                from hierarchy_protocol import HierarchyProtocol  # type: ignore

            hierarchy_runner = HierarchyProtocol(config=self.config)

            two_level_results = hierarchy_runner.train(
                genomic_path=genomic_path,
                meta_path=meta_path,
                level1_label=resolved_level1_label,
                level2_label=resolved_level2_label,
                output_dir=output_dir,
                ref_fasta=ref_fasta,
                algorithm=getattr(self.config, "ml_algorithm", "auto"),
                train_global_level2=bool(
                    getattr(self.config, "train_global_level2", True)
                ),
                min_level2_samples_per_group=getattr(
                    self.config,
                    "min_level2_samples_per_group",
                    None,
                ),
            )

            results: Dict[str, Any] = {
                "timestamp": timestamp(),
                "config": vars(self.config),
                "pipeline_mode": mode,
                "hierarchy_protocol": two_level_results,
                "two_level_protocol": two_level_results,  # backward-compatible key
            }

            results_path = (
                output_dir_path / f"networkparser_hierarchy_results_{timestamp()}.json"
            )
            save_json(results, results_path)

            log_artifact(logger, "hierarchy NetworkParser results", results_path)
            self._write_checkpoint(
                output_dir_path,
                "final_hierarchy_results",
                {"results_json": str(results_path), "pipeline_mode": mode},
            )
            self._write_audit_and_summary(
                output_dir=output_dir_path,
                results=results,
                results_path=results_path,
                title="NetworkParser hierarchy final run summary",
            )
            return results
        log_stage_start(
            logger,
            1,
            "load and preprocess genomic matrix",
            progress=pipeline_progress,
        )

        stage1_hashes = self._build_stage_hashes(
            input_paths={"genomic_path": genomic_path, "ref_fasta": ref_fasta},
        )
        genomic_df = self._load_matrix_checkpoint(
            output_dir_path,
            "stage1_preprocessed_matrix",
            expected_hashes=stage1_hashes,
            stage_name="stage1_load_preprocess",
        )
        if genomic_df is not None:
            log_branch_decision(
                logger,
                "checkpoint",
                "reused",
                stage="load/preprocess",
                samples=int(genomic_df.shape[0]),
                features=int(genomic_df.shape[1]),
            )
        else:
            # DataLoader returns the canonical sample×marker matrix in memory.
            # Do not rglob arbitrary *_binary.tsv files from the output tree.
            genomic_df = self.loader.load_genomic_matrix(
                file_path=genomic_path,
                output_dir=output_dir,
                ref_fasta=ref_fasta,
            )
            matrix_checkpoint = self._write_matrix_checkpoint(
                output_dir_path, "stage1_preprocessed_matrix", genomic_df
            )
            stage1_hashes = self._build_stage_hashes(
                input_paths={"genomic_path": genomic_path, "ref_fasta": ref_fasta},
                content_objects={"matrix": genomic_df},
            )
            self._write_checkpoint(
                output_dir_path,
                "stage1_load_preprocess",
                {
                    "genomic_path": str(genomic_path),
                    "samples": int(genomic_df.shape[0]),
                    "features": int(genomic_df.shape[1]),
                    "matrix_checkpoint": str(matrix_checkpoint)
                    if matrix_checkpoint
                    else None,
                    "matrix_source": "dataloader_return_value",
                },
                hashes=stage1_hashes,
            )

        log_stage_complete(
            logger,
            1,
            "load and preprocess genomic matrix",
            progress=pipeline_progress,
            samples=int(genomic_df.shape[0]),
            features=int(genomic_df.shape[1]),
        )
        meta_df = None
        if meta_path:
            log_stage_start(
                logger,
                "2a",
                "load metadata",
                progress=pipeline_progress,
            )
            meta_df = self.loader.load_metadata(meta_path, output_dir=output_dir)
            log_stage_complete(
                logger,
                "2a",
                "load metadata",
                progress=pipeline_progress,
                rows=int(meta_df.shape[0]),
                columns=int(meta_df.shape[1]),
            )

        known_markers = None
        if known_markers_path:
            logger.info(
                "Loading known markers (comparison/annotation only; not used for filtering)"
            )
            known_markers = self.loader.load_known_markers(
                known_markers_path,
                output_dir=output_dir,
            )
            logger.info("Loaded %d known markers", len(known_markers))

        known_markers_summary = self._summarize_known_markers(
            known_markers,
            list(map(str, genomic_df.columns)),
            output_dir_path,
        )

        # Single matrix_only exit: stop after load/preprocess (+ optional known-marker comparison).
        if mode == "matrix_only":
            log_branch_decision(
                logger,
                "pipeline stop",
                "matrix_only",
                reason="matrix creation requested",
                samples=int(genomic_df.shape[0]),
                features=int(genomic_df.shape[1]),
            )

            results = {
                "timestamp": timestamp(),
                "config": vars(self.config),
                "pipeline_mode": mode,
                "matrix_shape": {
                    "samples": int(genomic_df.shape[0]),
                    "features": int(genomic_df.shape[1]),
                },
                "known_markers_loaded": int(len(known_markers))
                if known_markers is not None
                else 0,
                "known_markers_summary": known_markers_summary,
                "feature_filtering": {},
                "discovery": {},
                "validation": {},
                "ml_protocol": {},
            }

            results_path = output_dir_path / f"networkparser_results_{timestamp()}.json"
            save_json(results, results_path)
            log_artifact(logger, "final NetworkParser results", results_path)
            self._write_checkpoint(
                output_dir_path,
                "final_results",
                {"results_json": str(results_path), "pipeline_mode": mode},
            )
            self._write_audit_and_summary(
                output_dir=output_dir_path,
                results=results,
                results_path=results_path,
            )
            if pipeline_progress is not None:
                pipeline_progress.complete_stage("finalize results")
            return results

        if meta_df is None:
            raise ValueError(
                "meta_path is required for supervised modes because labels are needed "
                "for RF-FDR filtering, model selection, and decision-tree analysis."
            )

        # --validate_statistics is a deprecated CLI alias that forces central filtering ON.
        # It does not run a separate statistics stage.
        if validate_statistics:
            if not bool(getattr(self.config, "run_central_feature_filtering", True)):
                logger.warning(
                    "Deprecated flag --validate_statistics forces "
                    "run_central_feature_filtering=True (was disabled)."
                )
                self.config.run_central_feature_filtering = True
            else:
                logger.info(
                    "Deprecated flag --validate_statistics noted; central filtering is already "
                    "enabled via config.run_central_feature_filtering."
                )

        log_stage_start(
            logger,
            "2b",
            "supervised sample alignment",
            progress=pipeline_progress,
        )
        X, y = self._align_X_y(genomic_df, meta_df, label_column=label_column)
        log_stage_complete(
            logger,
            "2b",
            "supervised sample alignment",
            progress=pipeline_progress,
            samples=int(X.shape[0]),
            features=int(X.shape[1]),
        )
        stage2_hashes = self._build_stage_hashes(
            input_paths={
                "genomic_path": genomic_path,
                "meta_path": meta_path,
                "ref_fasta": ref_fasta,
            },
            content_objects={"aligned_matrix": X, "labels": y},
        )
        aligned_matrix_checkpoint = self._write_matrix_checkpoint(
            output_dir_path, "stage2_aligned_matrix", X
        )
        aligned_labels_checkpoint = None
        if bool(getattr(self.config, "write_stage_checkpoints", True)):
            try:
                aligned_labels_checkpoint = (
                    output_dir_path / "_checkpoints" / "stage2_aligned_labels.csv"
                )
                aligned_labels_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                y.to_frame(name="label").to_csv(aligned_labels_checkpoint)
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not write aligned-label checkpoint: %s", exc)
        self._write_checkpoint(
            output_dir_path,
            "stage2_supervised_alignment",
            {
                "samples": int(X.shape[0]),
                "features": int(X.shape[1]),
                "matrix_checkpoint": str(aligned_matrix_checkpoint)
                if aligned_matrix_checkpoint
                else None,
                "labels_checkpoint": str(aligned_labels_checkpoint)
                if aligned_labels_checkpoint
                else None,
            },
            hashes=stage2_hashes,
        )

        validation_results: Dict[str, Any] = {}
        discovery_results: Dict[str, Any] = {}
        ml_results: Dict[str, Any] = {}

        log_stage_start(
            logger,
            3,
            "central statistical filtering",
            progress=pipeline_progress,
        )
        feature_filter_result: Dict[str, Any]
        feature_filter_summary: Dict[str, Any]
        stage3_expected = self._build_stage_hashes(
            input_paths={
                "genomic_path": genomic_path,
                "meta_path": meta_path,
                "ref_fasta": ref_fasta,
            },
            content_objects={"aligned_matrix": X, "labels": y},
        )
        X_filtered = self._load_matrix_checkpoint(
            output_dir_path,
            "stage3_central_filtered_matrix",
            expected_hashes=stage3_expected,
            stage_name="stage3_central_filtering",
        )
        central_checkpoint = (
            load_stage_checkpoint(
                output_dir_path,
                "stage3_central_filtering",
                expected_hashes=stage3_expected,
                refuse_incompatible=True,
            )
            if bool(getattr(self.config, "resume_from_checkpoints", False))
            else None
        )
        if X_filtered is not None and isinstance(central_checkpoint, dict):
            payload = (
                central_checkpoint.get("payload", {})
                if isinstance(central_checkpoint.get("payload"), dict)
                else {}
            )
            feature_filter_summary = (
                payload.get("summary", {})
                if isinstance(payload.get("summary"), dict)
                else {}
            )
            feature_filter_result = {
                "summary": feature_filter_summary,
                "filtered_matrix": X_filtered,
                "retained_features": list(X_filtered.columns),
                "method": feature_filter_summary.get("method", "checkpoint"),
            }
            log_branch_decision(
                logger,
                "checkpoint",
                "reused",
                stage="central_filtering",
                retained_features=int(X_filtered.shape[1]),
            )
        else:
            (
                X_filtered,
                feature_filter_summary,
                feature_filter_result,
            ) = self._run_central_feature_filter(
                X=X,
                y=y,
                output_dir=output_dir,
                enabled=bool(
                    getattr(self.config, "run_central_feature_filtering", True)
                ),
            )
            filtered_checkpoint = self._write_matrix_checkpoint(
                output_dir_path, "stage3_central_filtered_matrix", X_filtered
            )
            stage3_hashes = self._build_stage_hashes(
                input_paths={
                    "genomic_path": genomic_path,
                    "meta_path": meta_path,
                    "ref_fasta": ref_fasta,
                },
                content_objects={
                    "aligned_matrix": X,
                    "labels": y,
                    "filtered_matrix": X_filtered,
                },
            )
            self._write_checkpoint(
                output_dir_path,
                "stage3_central_filtering",
                {
                    "summary": feature_filter_summary,
                    "samples": int(X_filtered.shape[0]),
                    "features": int(X_filtered.shape[1]),
                    "matrix_checkpoint": str(filtered_checkpoint)
                    if filtered_checkpoint
                    else None,
                },
                hashes=stage3_hashes,
            )
        log_stage_complete(
            logger,
            3,
            "central statistical filtering",
            progress=pipeline_progress,
            retained_features=int(X_filtered.shape[1]),
            input_features=int(X.shape[1]),
        )

        log_stage_start(
            logger,
            4,
            "ranked feature-panel separability check",
            progress=pipeline_progress,
        )
        stage4_expected = self._build_stage_hashes(
            input_paths={
                "genomic_path": genomic_path,
                "meta_path": meta_path,
                "ref_fasta": ref_fasta,
            },
            content_objects={"filtered_matrix": X_filtered, "labels": y},
        )
        X_model = self._load_matrix_checkpoint(
            output_dir_path,
            "stage4_model_matrix",
            expected_hashes=stage4_expected,
            stage_name="stage4_feature_panel",
        )
        panel_checkpoint = (
            load_stage_checkpoint(
                output_dir_path,
                "stage4_feature_panel",
                expected_hashes=stage4_expected,
                refuse_incompatible=True,
            )
            if bool(getattr(self.config, "resume_from_checkpoints", False))
            else None
        )
        if X_model is not None and isinstance(panel_checkpoint, dict):
            payload = (
                panel_checkpoint.get("payload", {})
                if isinstance(panel_checkpoint.get("payload"), dict)
                else {}
            )
            feature_panel_summary = (
                payload.get("summary", {})
                if isinstance(payload.get("summary"), dict)
                else {}
            )
            log_branch_decision(
                logger,
                "checkpoint",
                "reused",
                stage="feature_panel",
                selected_features=int(X_model.shape[1]),
            )
        else:
            panel_selection = run_feature_panel_separability_check(
                X=X_filtered,
                y=y,
                output_dir=output_dir,
                config=self.config,
                stage_name="single_label_model_matrix",
                filter_result=feature_filter_result,
            )
            X_model = panel_selection["selected_matrix"]
            feature_panel_summary = panel_selection["summary"]
            model_checkpoint = self._write_matrix_checkpoint(
                output_dir_path, "stage4_model_matrix", X_model
            )
            stage4_hashes = self._build_stage_hashes(
                input_paths={
                    "genomic_path": genomic_path,
                    "meta_path": meta_path,
                    "ref_fasta": ref_fasta,
                },
                content_objects={
                    "filtered_matrix": X_filtered,
                    "labels": y,
                    "model_matrix": X_model,
                },
            )
            self._write_checkpoint(
                output_dir_path,
                "stage4_feature_panel",
                {
                    "summary": feature_panel_summary,
                    "samples": int(X_model.shape[0]),
                    "features": int(X_model.shape[1]),
                    "matrix_checkpoint": str(model_checkpoint)
                    if model_checkpoint
                    else None,
                },
                hashes=stage4_hashes,
            )
        log_flow_step(
            logger,
            step="Feature-panel checkpoint — selected model matrix",
            happened="Selected the model-ready feature panel from the centrally filtered matrix.",
            reason="The panel check keeps the smallest ranked feature subset with acceptable separability, reducing runtime while preserving interpretable signal.",
            before_samples=int(X_filtered.shape[0]),
            before_features=int(X_filtered.shape[1]),
            after_samples=int(X_model.shape[0]),
            after_features=int(X_model.shape[1]),
            threshold=f"panel_sizes={getattr(self.config, 'feature_panel_sizes', 'configured')}; min_score={getattr(self.config, 'feature_panel_min_score', 'configured')}",
            status=str(
                feature_panel_summary.get(
                    "reason", feature_panel_summary.get("status", "complete")
                )
            ),
            artifact=feature_panel_summary.get("artifacts", {}).get("selected_matrix"),
        )
        log_stage_complete(
            logger,
            4,
            "ranked feature-panel separability check",
            progress=pipeline_progress,
            selected_features=int(X_model.shape[1]),
            input_features=int(X_filtered.shape[1]),
        )

        run_ml = mode in {"ml_only", "both"} or bool(
            getattr(self.config, "run_ml_protocol", False)
        )

        if run_ml:
            log_stage_start(
                logger,
                5,
                "ML protocol and model screening",
                progress=pipeline_progress,
            )
            log_flow_step(
                logger,
                step="Model-screening checkpoint — ML protocol",
                happened="Started model screening and training on the selected model matrix.",
                reason="Model screening happens after central statistical filtering so algorithm choice does not become the primary feature-selection mechanism.",
                before_samples=int(X_model.shape[0]),
                before_features=int(X_model.shape[1]),
                threshold=f"requested_algorithm={getattr(self.config, 'ml_algorithm', 'auto')}",
                status="started",
            )
            log_model_input_panel_decision(
                model_name="single_label_ml_protocol",
                X=X_model,
                panel_summary=feature_panel_summary,
                log=logger,
            )
            ml_runner = MLProtocolRunner(config=self.config)
            ml_results = ml_runner.run(
                genomic_df=X_model,
                labels=y,
                output_dir=output_dir,
                algorithm=getattr(self.config, "ml_algorithm", "auto"),
                panel_summary=feature_panel_summary,
                model_name="single_label_ml_protocol",
            )
            log_stage_complete(
                logger,
                5,
                "ML protocol and model screening",
                progress=pipeline_progress,
                selected_algorithm=ml_results.get("selected_algorithm"),
                features=int(X_model.shape[1]),
            )
            self._write_checkpoint(
                output_dir_path,
                "stage5_ml_protocol",
                {
                    "status": ml_results.get("status"),
                    "selected_algorithm": ml_results.get("selected_algorithm"),
                    "artifacts": ml_results.get("artifacts", {}),
                },
            )
        else:
            log_branch_decision(
                logger,
                "ML protocol",
                "skipped",
                reason=f"pipeline_mode={mode}",
                features=int(X_model.shape[1]),
            )
            self._write_checkpoint(
                output_dir_path,
                "stage5_ml_protocol",
                {
                    "status": "skipped",
                    "pipeline_mode": mode,
                    "features": int(X_model.shape[1]),
                },
                status="skipped",
            )

        run_tree = self._should_run_decision_tree(mode=mode, ml_results=ml_results)

        if run_tree:
            log_stage_start(
                logger,
                6,
                "conditional decision-tree interpretability",
                progress=pipeline_progress,
            )
            log_flow_step(
                logger,
                step="Interpretability checkpoint — decision tree",
                happened="Started decision-tree fitting on the selected, pre-filtered feature matrix.",
                reason="The tree is used for interpretable rules and path-based interaction mining after statistical filtering, not as the initial feature-selection layer.",
                before_samples=int(X_model.shape[0]),
                before_features=int(X_model.shape[1]),
                threshold="prefiltered_input=True",
                status="started",
            )

            discovery_results = self.tree_builder.run(
                data=X_model,
                labels=y,
                all_features=list(X_model.columns),
                output_dir=output_dir,
                prefiltered_input=True,
            )

            _tree_interactions = discovery_results.get(
                "tree_path_interaction_candidates",
                discovery_results.get("epistatic_interactions", []),
            )
            log_stage_complete(
                logger,
                6,
                "conditional decision-tree interpretability",
                progress=pipeline_progress,
                discovered_features=len(
                    discovery_results.get("discovered_features", [])
                ),
                interactions=len(_tree_interactions),
            )
            self._write_checkpoint(
                output_dir_path,
                "stage6_decision_tree",
                {
                    "status": "generated",
                    "discovered_features": len(
                        discovery_results.get("discovered_features", [])
                    ),
                    "interactions": len(_tree_interactions),
                },
            )

            if validate_interactions:
                log_stage_start(
                    logger,
                    7,
                    "optional post-tree interaction validation",
                    progress=pipeline_progress,
                )
                interaction_pairs = self._extract_interaction_pairs(_tree_interactions)

                if interaction_pairs:
                    validation_results[
                        "interactions"
                    ] = self.validator.validate_interactions(
                        genomic_df=X_model,
                        meta_df=meta_df.loc[X_model.index],
                        label_column=label_column,
                        interactions=interaction_pairs,
                        output_dir=output_dir,
                    )
                else:
                    logger.info(
                        "No interaction pairs available for validation. Skipping interaction validation."
                    )
                    validation_results["interactions"] = {
                        "status": "skipped",
                        "reason": "no_interactions_detected",
                    }
                log_stage_complete(
                    logger,
                    7,
                    "optional post-tree interaction validation",
                    progress=pipeline_progress,
                    status=validation_results.get("interactions", {}).get(
                        "status", "complete"
                    ),
                )
        else:
            log_branch_decision(
                logger,
                "decision-tree interpretability",
                "skipped",
                reason="conditional trigger not met",
                mode=mode,
            )
            if pipeline_progress is not None and mode in {"both", "decision_tree_only"}:
                pipeline_progress.complete_stage("decision-tree skipped")
                if validate_interactions:
                    pipeline_progress.complete_stage("interaction validation skipped")
            self._write_checkpoint(
                output_dir_path,
                "stage6_decision_tree",
                {"status": "skipped", "pipeline_mode": mode},
                status="skipped",
            )

        nested_cv_results: Dict[str, Any] = {}
        if bool(getattr(self.config, "run_nested_cv_evaluation", False)) and meta_path:
            # Authoritative nested CV evaluation route (strict fold-local preprocessing).
            try:
                from network_parser.cross_validation import run_repeated_cv
            except ImportError:  # pragma: no cover
                from cross_validation import run_repeated_cv  # type: ignore

            log_stage_start(
                logger,
                "cv",
                "nested repeated cross-validation evaluation",
                progress=pipeline_progress,
            )
            nested_cv_results = run_repeated_cv(
                genomic_path=genomic_path,
                meta_path=meta_path,
                label_column=label_column,
                output_dir=str(output_dir_path / "nested_cv_evaluation"),
                config=self.config,
                ref_fasta=ref_fasta,
                n_repeats=int(getattr(self.config, "cv_n_repeats", 3)),
                n_splits=int(getattr(self.config, "cv_n_splits", 5)),
                algorithm=getattr(self.config, "ml_algorithm", "auto"),
            )
            log_stage_complete(
                logger,
                "cv",
                "nested repeated cross-validation evaluation",
                progress=pipeline_progress,
                status=nested_cv_results.get("status"),
                successful_folds=nested_cv_results.get("successful_folds"),
                publication_ready=nested_cv_results.get("publication_ready"),
            )

        results = {
            "timestamp": timestamp(),
            "config": vars(self.config),
            "pipeline_mode": mode,
            "aligned_matrix_shape": {
                "samples": int(X.shape[0]),
                "features": int(X.shape[1]),
            },
            "central_filtered_matrix_shape": {
                "samples": int(X_filtered.shape[0]),
                "features": int(X_filtered.shape[1]),
            },
            "model_matrix_shape": {
                "samples": int(X_model.shape[0]),
                "features": int(X_model.shape[1]),
            },
            "known_markers_loaded": int(len(known_markers))
            if known_markers is not None
            else 0,
            "known_markers_summary": known_markers_summary,
            "feature_filtering": feature_filter_summary,
            "feature_panel_separability": feature_panel_summary,
            "discovery": discovery_results,
            "validation": validation_results,
            "ml_protocol": ml_results,
            "nested_cv_evaluation": nested_cv_results,
            "matrix_contract": {
                "encoding": {
                    "0": "callable_baseline",
                    "1": "callable_non_baseline",
                    "NaN": "non_callable",
                },
                "impute_strategy": getattr(
                    self.config, "genotype_impute_strategy", "baseline"
                ),
                "max_missing_fraction": getattr(
                    self.config, "max_missing_fraction", None
                ),
            },
        }

        results_path = output_dir_path / f"networkparser_results_{timestamp()}.json"
        save_json(results, results_path)
        log_artifact(logger, "final NetworkParser results", results_path)
        self._write_checkpoint(
            output_dir_path,
            "final_results",
            {"results_json": str(results_path), "pipeline_mode": mode},
        )
        self._write_audit_and_summary(
            output_dir=output_dir_path,
            results=results,
            results_path=results_path,
        )

        if pipeline_progress is not None:
            pipeline_progress.complete_stage("finalize results")

        return results


def run_networkparser_analysis(
    genomic_path: str,
    meta_path: Optional[str],
    label_column: Optional[str],
    known_markers_path: Optional[str],
    output_dir: str,
    config: NetworkParserConfig,
    validate_statistics: bool = False,
    validate_interactions: bool = False,
    ref_fasta: Optional[str] = None,
    level1_label_column: Optional[str] = None,
    level2_label_column: Optional[str] = None,
) -> Dict[str, Any]:
    parser = NetworkParser(config)
    return parser.run_pipeline(
        genomic_path=genomic_path,
        meta_path=meta_path,
        label_column=label_column,
        known_markers_path=known_markers_path,
        output_dir=output_dir,
        validate_statistics=validate_statistics,
        validate_interactions=validate_interactions,
        ref_fasta=ref_fasta,
        level1_label_column=level1_label_column,
        level2_label_column=level2_label_column,
    )

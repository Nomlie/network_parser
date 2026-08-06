# network_parser/ml_protocol.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _label_distribution_diagnostics(
    labels: pd.Series,
    requested_cv_splits: int = 5,
) -> Dict[str, Any]:
    """Return label-balance diagnostics without exposing class names.

    The model selector uses stratified CV probes. If a class has too few
    samples, every probe can become non-finite. These diagnostics make that
    failure explicit in logs and JSON artifacts.
    """
    y = pd.Series(labels).astype(str).str.strip()
    y = y.replace(
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
    y = y.dropna()

    counts = y.value_counts(dropna=True)
    count_values = [int(v) for v in counts.tolist()]
    min_count = int(min(count_values)) if count_values else 0
    max_count = int(max(count_values)) if count_values else 0
    requested_cv_splits = max(2, int(requested_cv_splits))
    feasible_cv_splits = (
        int(min(requested_cv_splits, min_count)) if min_count > 0 else 0
    )

    return {
        "n_samples": int(y.shape[0]),
        "n_classes": int(counts.shape[0]),
        "min_class_count": min_count,
        "max_class_count": max_count,
        "n_singleton_classes": int(sum(v == 1 for v in count_values)),
        "class_count_values_sorted": sorted(count_values),
        "requested_selector_cv_splits": requested_cv_splits,
        "feasible_selector_cv_splits": feasible_cv_splits,
        "stratified_cv_feasible": bool(
            counts.shape[0] >= 2 and feasible_cv_splits >= 2
        ),
    }


class MLProtocolRunner:
    """
    Downstream ML protocol branch.

    Updated role in pipeline
    ------------------------
    Consume the already aligned and centrally filtered sample x feature dataframe,
    then run:

        selector -> shortlist -> resolve final algorithm -> train -> evaluate

    The selector output is also used as a branch-decision payload so the
    orchestrator can decide whether the decision-tree interpretability branch
    should be triggered.
    """

    SUPPORTED_ALGOS = {"RF", "MLP", "LR", "MBCS", "DT", "SVC", "SCV", "DNL"}

    def __init__(self, config: Optional[Any] = None):
        self.config = config

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def run(
        self,
        genomic_df: pd.DataFrame,
        labels: pd.Series,
        output_dir: str,
        algorithm: Optional[str] = None,
        panel_summary: Optional[Dict[str, Any]] = None,
        model_name: str = "ml_protocol_model",
    ) -> Dict[str, Any]:
        """
        Run the full ML protocol on an already aligned, centrally filtered dataframe.
        """
        self._validate_inputs(genomic_df, labels)

        out_dir = _ensure_dir(Path(output_dir) / "ml_protocol")

        genomic_df_aligned, labels_aligned = self._align_inputs(genomic_df, labels)
        protocol_df = self.build_protocol_df(genomic_df_aligned, labels_aligned)

        empty_thr = 1.0
        empty_symbol = ""
        if self.config is not None:
            empty_thr = float(
                getattr(self.config, "ml_remove_empty_field_threshold", 1.0)
            )
            empty_symbol = str(getattr(self.config, "ml_empty_symbol", ""))

        protocol_df = self.remove_empty_columns(
            protocol_df=protocol_df,
            thr=empty_thr,
            empty_symbol=empty_symbol,
        )

        genomic_df_aligned = protocol_df.iloc[:, 2:].copy()
        labels_aligned = protocol_df.iloc[:, 1].copy()
        genomic_df_aligned.index = protocol_df.iloc[:, 0].astype(str)

        protocol_matrix_path = out_dir / "ml_protocol_matrix.csv"
        protocol_df.to_csv(protocol_matrix_path, index=False)

        requested_algo = algorithm
        if requested_algo is None and self.config is not None:
            requested_algo = getattr(self.config, "ml_algorithm", "auto")

        run_selector = (
            bool(getattr(self.config, "run_model_selector", True))
            if self.config is not None
            else True
        )

        # --------------------------------------------------------------
        # Selector / ranking stage
        # --------------------------------------------------------------
        if run_selector:
            selector_result = self.select_model(genomic_df_aligned, labels_aligned)
            selector_result = self._normalize_selector_output(selector_result)
        else:
            logger.info(
                "Model selector disabled by config.run_model_selector=False; using requested algorithm pathway."
            )
            selector_result = self._selector_disabled_payload(
                requested_algorithm=requested_algo
            )

        if run_selector and (
            requested_algo is None or str(requested_algo).lower() == "auto"
        ):
            probe_scores = selector_result.get("probe_scores", {})
            finite_probe_scores = []
            if isinstance(probe_scores, dict):
                for key, value in probe_scores.items():
                    if key == "delta_nonlinear_minus_linear":
                        continue
                    try:
                        numeric_value = float(value)
                    except Exception:
                        continue
                    if np.isfinite(numeric_value):
                        finite_probe_scores.append(numeric_value)
            if not finite_probe_scores:
                diagnostics = _label_distribution_diagnostics(
                    labels_aligned, requested_cv_splits=5
                )
                diagnostics.update(
                    {
                        "n_features_after_ml_empty_column_filter": int(
                            genomic_df_aligned.shape[1]
                        ),
                        "probe_scores": probe_scores,
                        "selector_status": selector_result.get(
                            "selector_status", "no_finite_probe_scores"
                        ),
                        "requested_algorithm": "auto"
                        if requested_algo is None
                        else str(requested_algo),
                    }
                )

                failure_path = out_dir / "ml_protocol_selector_failure.json"
                with open(failure_path, "w", encoding="utf-8") as fh:
                    json.dump(diagnostics, fh, indent=2, default=_json_default)

                logger.error(
                    "Model selector produced no finite probe scores | samples=%d | features=%d | "
                    "classes=%d | min_class_count=%d | feasible_cv_splits=%d | "
                    "singleton_classes=%d | wrote=%s",
                    diagnostics["n_samples"],
                    diagnostics["n_features_after_ml_empty_column_filter"],
                    diagnostics["n_classes"],
                    diagnostics["min_class_count"],
                    diagnostics["feasible_selector_cv_splits"],
                    diagnostics["n_singleton_classes"],
                    str(failure_path),
                )

                raise RuntimeError(
                    "Model selector produced no finite probe scores; refusing automatic model selection. "
                    f"samples={diagnostics['n_samples']}; "
                    f"features={diagnostics['n_features_after_ml_empty_column_filter']}; "
                    f"classes={diagnostics['n_classes']}; "
                    f"min_class_count={diagnostics['min_class_count']}; "
                    f"feasible_selector_cv_splits={diagnostics['feasible_selector_cv_splits']}. "
                    f"Diagnostics written to {failure_path}."
                )

        selected_algo = self.resolve_algorithm(
            selector_recommendation=selector_result.get("recommendation", "RF"),
            requested_algorithm=requested_algo,
        )

        branch_decision = self.build_branch_decision_payload(
            selector_result=selector_result,
            selected_algorithm=selected_algo,
            requested_algorithm=requested_algo,
        )

        # --------------------------------------------------------------
        # Freeze decision threshold on nested OOF *before* final fit
        # --------------------------------------------------------------
        groups = None
        group_col = (
            getattr(self.config, "cv_group_column", None)
            if self.config is not None
            else None
        )
        if group_col and group_col in genomic_df_aligned.columns:
            groups = genomic_df_aligned[group_col]
        # groups may also come from labels index alignment via protocol — optional

        threshold_selection = self.select_decision_threshold_out_of_fold(
            genomic_df=genomic_df_aligned,
            labels=labels_aligned,
            algorithm=selected_algo,
            out_dir=out_dir / "threshold_selection_oof",
            groups=groups,
            panel_summary=panel_summary,
            model_name=model_name,
        )
        selected_threshold = float(
            threshold_selection.get(
                "selected_decision_threshold",
                getattr(self.config, "ml_min_decision_threshold", 0.5)
                if self.config
                else 0.5,
            )
        )

        # --------------------------------------------------------------
        # Final deployment model (fit only after threshold is frozen)
        # --------------------------------------------------------------
        model = self.train_model(
            genomic_df=genomic_df_aligned,
            labels=labels_aligned,
            algorithm=selected_algo,
            panel_summary=panel_summary,
            model_name=f"{model_name}__final",
        )
        missingness_state = getattr(model, "networkparser_missingness_state", {})
        missingness_audit = getattr(model, "networkparser_missingness_audit", {})

        model_path = out_dir / f"{selected_algo}_ml_protocol_model.pkl"
        self.save_model(model, model_path)

        # Same-data scores are training-fit diagnostics only — not generalisation.
        training_fit_diagnostics = self.evaluate_model(
            model=model,
            protocol_df=protocol_df,
            out_dir=out_dir / "training_fit_diagnostics",
            decision_threshold=selected_threshold,
            evaluation_role="training_fit_diagnostics",
        )

        interpretability = (
            model.get_interpretability()
            if hasattr(model, "get_interpretability")
            else {}
        )

        summary = {
            "status": "success",
            "n_samples": int(genomic_df_aligned.shape[0]),
            "n_features": int(genomic_df_aligned.shape[1]),
            "selected_algorithm": selected_algo,
            "requested_algorithm": "auto"
            if requested_algo is None
            else str(requested_algo),
            "selector_enabled": bool(run_selector),
            "selector": selector_result,
            "branch_decision": branch_decision,
            "training_metrics": getattr(model, "training_metrics", {}),
            "interpretability": interpretability,
            "selected_decision_threshold": selected_threshold,
            "threshold_selection": threshold_selection,
            "missingness_state": missingness_state,
            "preprocessing_state": missingness_state,
            "missingness_audit": missingness_audit,
            "training_fit_diagnostics": training_fit_diagnostics,
            # Backward-compatible key: explicitly not a generalisation metric.
            "evaluation": {
                "role": "training_fit_diagnostics",
                "is_generalization_estimate": False,
                "note": (
                    "Same-data training-fit diagnostics only. "
                    "Use threshold_selection (OOF) and held-out/CV evaluation for performance claims."
                ),
                **training_fit_diagnostics,
            },
            "artifacts": {
                "protocol_matrix": str(protocol_matrix_path),
                "model_file": str(model_path),
                "threshold_selection_json": str(
                    out_dir
                    / "threshold_selection_oof"
                    / "threshold_selection_summary.json"
                ),
                "training_fit_diagnostics_json": str(
                    out_dir / "training_fit_diagnostics" / "ml_protocol_evaluation.json"
                ),
                "evaluation_json": str(
                    out_dir / "training_fit_diagnostics" / "ml_protocol_evaluation.json"
                ),
                "evaluation_tsv": str(
                    out_dir / "training_fit_diagnostics" / "ml_protocol_thresholds.tsv"
                ),
                "sample_predictions_tsv": str(
                    out_dir
                    / "training_fit_diagnostics"
                    / "ml_protocol_sample_predictions.tsv"
                ),
                "results_json": str(out_dir / "ml_protocol_results.json"),
            },
        }

        with open(out_dir / "ml_protocol_results.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=_json_default)

        logger.info(
            "ML protocol complete | samples=%d | features=%d | selected_algorithm=%s | dt_candidate=%s | selector_enabled=%s | out=%s",
            genomic_df_aligned.shape[0],
            genomic_df_aligned.shape[1],
            selected_algo,
            branch_decision.get("run_decision_tree_branch", False),
            run_selector,
            out_dir,
        )

        return summary

    # ------------------------------------------------------------------
    # validation / alignment
    # ------------------------------------------------------------------
    def _validate_inputs(self, genomic_df: pd.DataFrame, labels: pd.Series) -> None:
        if not isinstance(genomic_df, pd.DataFrame):
            raise TypeError("genomic_df must be a pandas DataFrame")
        if not isinstance(labels, pd.Series):
            raise TypeError("labels must be a pandas Series")
        if genomic_df.empty:
            raise ValueError("genomic_df is empty")
        if genomic_df.shape[1] == 0:
            raise ValueError("genomic_df has no feature columns")
        if labels.empty:
            raise ValueError("labels is empty")

    def _align_inputs(
        self,
        genomic_df: pd.DataFrame,
        labels: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        genomic_df = genomic_df.copy()
        labels = labels.copy()

        genomic_df.index = genomic_df.index.astype(str)
        labels.index = labels.index.astype(str)

        labels = labels.astype(str).str.strip()
        labels = labels.replace(
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
        labels = labels.dropna()

        common = genomic_df.index.intersection(labels.index)
        if len(common) == 0:
            raise ValueError("No overlapping sample IDs between genomic_df and labels")

        genomic_df_aligned = genomic_df.loc[common].copy()
        labels_aligned = labels.loc[common].copy()

        logger.info(
            "ML protocol alignment | genomic=%d | labels=%d | overlap=%d",
            len(genomic_df.index),
            len(labels.index),
            len(common),
        )

        return genomic_df_aligned, labels_aligned

    # ------------------------------------------------------------------
    # protocol matrix construction
    # ------------------------------------------------------------------
    def build_protocol_df(
        self,
        genomic_df: pd.DataFrame,
        labels: pd.Series,
    ) -> pd.DataFrame:
        """
        Build the structure expected by train/test style logic:

            col0 = sample_id
            col1 = label
            col2+ = features
        """
        labels = labels.loc[genomic_df.index]
        df = genomic_df.copy()
        df.insert(0, "label", labels.astype(str).values)
        df.insert(0, "sample_id", genomic_df.index.astype(str))
        return df

    def remove_empty_columns(
        self,
        protocol_df: pd.DataFrame,
        thr: float,
        empty_symbol: str = "",
    ) -> pd.DataFrame:
        """
        Remove feature columns where empty fraction > thr.

        The first two columns (sample_id, label) are preserved.
        """
        if protocol_df.shape[1] < 3:
            return protocol_df

        if thr >= 1.0:
            return protocol_df

        empty_sym = str(empty_symbol).strip()
        feature_cols = list(protocol_df.columns[2:])
        keep_cols: List[str] = []
        removed: List[Tuple[str, float]] = []

        for col in feature_cols:
            s = protocol_df[col]
            is_empty = s.isna()
            s_str = s.astype(str).str.strip()
            is_empty = is_empty | (s_str == "")
            if empty_sym != "":
                is_empty = is_empty | (s_str == empty_sym)

            frac_empty = float(is_empty.mean())
            if frac_empty > thr:
                removed.append((str(col), frac_empty))
            else:
                keep_cols.append(str(col))

        if removed:
            logger.warning(
                "ML protocol removed %d feature column(s) with empty fraction > %.3f",
                len(removed),
                thr,
            )
            return protocol_df.loc[:, list(protocol_df.columns[:2]) + keep_cols].copy()

        return protocol_df

    # ------------------------------------------------------------------
    # selector integration
    # ------------------------------------------------------------------
    def import_selector(self):
        try:
            from network_parser.model_selector import recommend_classifier

            return recommend_classifier
        except ImportError:  # package vs source-tree layout only
            from model_selector import recommend_classifier

            return recommend_classifier

    def _encode_for_selector(self, genomic_df: pd.DataFrame) -> np.ndarray:
        """
        Encode feature columns for selector probing under the matrix contract.

        By default, missing/non-callable values are imputed with the configured
        train-fit strategy (baseline/mode/constant) and are NOT represented as
        an ordinary genotype category. Set config.allow_missing_as_category=True
        only when an explicit missing-as-category encoding is intentional.
        """
        allow_cat = bool(
            getattr(self.config, "allow_missing_as_category", False)
            if self.config is not None
            else False
        )
        df = genomic_df.copy()
        if not allow_cat:
            # Preserve NaN into model-selector CV. Every probe pipeline fits its
            # own imputer inside each training split; pre-imputing here would
            # leak validation-fold distribution into model selection.
            numeric = df.apply(pd.to_numeric, errors="coerce")
            invalid_observed = df.notna() & numeric.isna()
            if invalid_observed.any().any():
                raise ValueError(
                    "Selector received non-numeric observed genotype states under the binary matrix contract"
                )
            return numeric.to_numpy(dtype=float)

        encoded_cols: List[np.ndarray] = []
        for col in df.columns:
            s = df[col].copy()
            if allow_cat:
                s = s.where(~s.isna(), "__MISSING__")
            s = s.astype(str).str.strip()
            s = s.replace(
                {
                    "": "__MISSING__",
                    "nan": "__MISSING__",
                    "NaN": "__MISSING__",
                    "nd": "__MISSING__",
                }
            )
            cat = pd.Categorical(s)
            encoded_cols.append(cat.codes.astype(float))

        if not encoded_cols:
            raise ValueError("No feature columns available for selector encoding")
        return np.column_stack(encoded_cols)

    def select_model(
        self,
        genomic_df: pd.DataFrame,
        labels: pd.Series,
        groups: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Run the selector stage using in-memory dataframe + labels.
        """
        recommend_classifier = self.import_selector()

        X = self._encode_for_selector(genomic_df)
        y = labels.astype(str).to_numpy()

        try:
            group_values = None
            if groups is not None:
                aligned_groups = pd.Series(groups).reindex(genomic_df.index)
                if aligned_groups.isna().any():
                    raise ValueError(
                        "Model selector groups are missing for one or more samples"
                    )
                group_values = aligned_groups.astype(str).to_numpy()
            result = recommend_classifier(
                X,
                y,
                config=self.config,
                groups=group_values,
            )
        except Exception as exc:
            # Never silently substitute RF; preserve the underlying exception.
            logger.error("Model selector failed: %s", exc, exc_info=True)
            raise RuntimeError(
                f"Model selector failed; refusing to default to RF. Underlying error: {exc}"
            ) from exc

        status = str(result.get("selector_status", "ok"))
        if status.startswith("failed") or result.get("recommendation") is None:
            err = result.get("error") or status or "selector_failed"
            rationale = result.get("rationale", [])
            raise RuntimeError(
                "Model selector failed; refusing to default to RF. "
                f"status={status}; error={err}; rationale={rationale}"
            )

        if "recommendation" not in result or not result["recommendation"]:
            raise RuntimeError(
                "Model selector returned no recommendation; refusing to default to RF."
            )

        logger.info("ML selector recommendation: %s", result["recommendation"])
        logger.info("ML selector probe scores | %s", result.get("probe_scores", {}))
        logger.info(
            "ML selector CV folds | requested=%s | actual_per_probe=%s",
            result.get("requested_cv_splits"),
            result.get("actual_cv_splits"),
        )
        return result

    def _selector_disabled_payload(
        self,
        requested_algorithm: Optional[str],
    ) -> Dict[str, Any]:
        """
        Build a consistent selector-like payload when model screening is disabled.
        """
        req = (
            "auto" if requested_algorithm is None else str(requested_algorithm).strip()
        )

        if req.lower() == "auto":
            rec = "RF"
            rationale = [
                "Model selector disabled; falling back to default RF recommendation because requested_algorithm='auto'."
            ]
        else:
            rec = "SVC" if req == "SCV" else req
            rationale = [
                f"Model selector disabled; using requested algorithm '{rec}' as the recommendation."
            ]

        candidate_ranked = [rec]
        interpretable = [a for a in candidate_ranked if a in {"DT", "LR", "RF"}]

        payload = {
            "recommendation": rec,
            "candidate_ranked": candidate_ranked,
            "dt_candidate": bool(rec == "DT"),
            "recommended_interpretable_models": interpretable,
            "rationale": rationale,
            "probe_scores": {},
            "selector_enabled": False,
        }
        return payload

    def _normalize_selector_output(
        self,
        selector_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Normalize selector output so the rest of the pipeline can rely on a
        consistent structure, even if model_selector.py still returns the older format.
        """
        result = dict(selector_result or {})

        raw_rec = result.get("recommendation", None)
        if raw_rec is None or str(raw_rec).strip() == "":
            # Preserve failure state; do not invent RF.
            result["recommendation"] = None
            result["selector_status"] = result.get(
                "selector_status", "failed_missing_recommendation"
            )
            return result
        recommendation = str(raw_rec).strip()
        result["recommendation"] = recommendation

        probe_scores = result.get("probe_scores", {})
        if not isinstance(probe_scores, dict):
            probe_scores = {}

        normalized_recommendation = self._normalize_selector_name(recommendation)

        # Prefer selector shortlist (top-k / margin). Do not rebuild from all finite probes.
        candidate_ranked = result.get("candidate_ranked")
        if not isinstance(candidate_ranked, list) or not candidate_ranked:
            scored_items: List[Tuple[str, float]] = []
            for k, v in probe_scores.items():
                if k == "delta_nonlinear_minus_linear":
                    continue
                fv = _safe_float(v, default=float("-inf"))
                if np.isfinite(fv):
                    scored_items.append((self._normalize_selector_name(str(k)), fv))
            scored_items.sort(key=lambda x: x[1], reverse=True)
            candidate_ranked = [name for name, _ in scored_items]

        candidate_ranked = [self._normalize_selector_name(x) for x in candidate_ranked]
        seen: set = set()
        deduped: List[str] = []
        for a in candidate_ranked:
            if a not in seen:
                deduped.append(a)
                seen.add(a)
        candidate_ranked = deduped
        if (
            normalized_recommendation
            and normalized_recommendation not in candidate_ranked
        ):
            candidate_ranked.insert(0, normalized_recommendation)

        # DT candidacy comes from selector top-k/margin rule, not "any finite DT score".
        dt_candidate = bool(
            result.get("dt_candidate", False) or normalized_recommendation == "DT"
        )

        # Interpretable models: DT / LR / RF only (SVC/MLP are not labelled interpretable).
        interpretable_models = [a for a in candidate_ranked if a in {"DT", "LR", "RF"}]

        result["recommendation"] = normalized_recommendation
        result["candidate_ranked"] = candidate_ranked
        result["dt_candidate"] = dt_candidate
        result["recommended_interpretable_models"] = interpretable_models

        if "rationale" not in result:
            result["rationale"] = []

        if "selector_enabled" not in result:
            result["selector_enabled"] = True

        return result

    def _normalize_selector_name(self, name: Any) -> str:
        """
        Normalize selector names to the downstream vocabulary used by this branch.
        """
        raw = str(name).strip()

        mapping = {
            "MLP_small": "MLP",
            "LinearSVC": "SVC",
            "SVC_RBF": "SVC",
            "SCV": "SVC",
        }
        return mapping.get(raw, raw)

    def resolve_algorithm(
        self,
        selector_recommendation: Optional[str],
        requested_algorithm: Optional[str] = None,
    ) -> str:
        """
        Resolve final algorithm to one supported by neural_network.py.

        Does not silently default to RF when the selector recommendation is
        missing or unrecognised under auto mode.
        """
        req = (
            "auto" if requested_algorithm is None else str(requested_algorithm).strip()
        )

        if req and req.lower() != "auto":
            if req not in self.SUPPORTED_ALGOS:
                raise ValueError(
                    f"Unsupported requested ML algorithm '{req}'. "
                    f"Supported: {sorted(self.SUPPORTED_ALGOS)} plus 'auto'."
                )
            return "SVC" if req == "SCV" else req

        if selector_recommendation is None or str(selector_recommendation).strip() in {
            "",
            "None",
            "nan",
        }:
            raise RuntimeError(
                "No selector recommendation available under ml_algorithm='auto'; "
                "refusing to default to RF."
            )

        rec = str(selector_recommendation).strip()
        mapping = {
            "RF": "RF",
            "DT": "DT",
            "LR": "LR",
            "MLP": "MLP",
            "MBCS": "MBCS",
            "DNL": "DNL",
            "SVC": "SVC",
            "LinearSVC": "SVC",
            "SVC_RBF": "SVC",
            "SCV": "SVC",
        }
        if rec not in mapping:
            raise RuntimeError(
                f"Unrecognised selector recommendation '{rec}' under ml_algorithm='auto'; "
                "refusing to default to RF. Supported recommendations: "
                f"{sorted(set(mapping.values()))}."
            )
        return mapping[rec]

    def build_branch_decision_payload(
        self,
        selector_result: Dict[str, Any],
        selected_algorithm: str,
        requested_algorithm: Optional[str],
    ) -> Dict[str, Any]:
        """
        Create a normalized branch-decision payload for the orchestrator.

        This is what enables conditional triggering of the DT branch after model selection.
        """
        requested = (
            "auto" if requested_algorithm is None else str(requested_algorithm).strip()
        )
        recommendation = str(selector_result.get("recommendation") or "").strip()
        candidate_ranked = selector_result.get("candidate_ranked", [])
        if not isinstance(candidate_ranked, list):
            candidate_ranked = []

        # DT candidate only from explicit selector rule / selected algorithm.
        dt_candidate = bool(
            selector_result.get("dt_candidate", False)
            or selected_algorithm == "DT"
            or recommendation == "DT"
        )
        run_dt = bool(
            selected_algorithm == "DT" or recommendation == "DT" or dt_candidate
        )

        payload = {
            "requested_algorithm": requested,
            "selector_recommendation": recommendation,
            "selected_algorithm": selected_algorithm,
            "candidate_algorithms": candidate_ranked,
            "recommended_interpretable_models": selector_result.get(
                "recommended_interpretable_models", []
            ),
            "dt_candidate": dt_candidate,
            "run_decision_tree_branch": run_dt,
        }

        return payload

    # ------------------------------------------------------------------
    # neural_network.py integration
    # ------------------------------------------------------------------
    def import_nn(self):
        try:
            import network_parser.neural_network as NN

            return NN
        except Exception:
            import neural_network as NN

            return NN

    def select_nn_model(self, NN: Any, algorithm: str, marker_style: str = "plain"):
        algo = "SVC" if algorithm == "SCV" else algorithm

        mapping = {
            "MLP": NN.MLP,
            "LR": NN.LR,
            "MBCS": NN.MBCS,
            "RF": NN.RF,
            "SVC": NN.SVC,
            "DT": NN.DT,
            "DNL": NN.DeltaNonlinLin,
        }
        if algo not in mapping:
            raise ValueError(f"Unknown algorithm '{algo}'")

        model = mapping[algo](marker_style=marker_style)

        if algo == "LR" and hasattr(model, "max_iter"):
            model.max_iter = int(getattr(self.config, "ml_lr_max_iter", 2000))

        if algo == "DNL" and hasattr(model, "max_iter"):
            model.max_iter = int(getattr(self.config, "ml_lr_max_iter", 2000))

        if algo == "RF" and getattr(model, "max_features", None) == "auto":
            model.max_features = "sqrt"

        return model

    def train_model(
        self,
        genomic_df: pd.DataFrame,
        labels: pd.Series,
        algorithm: str,
        panel_summary: Optional[Dict[str, Any]] = None,
        model_name: str = "ml_protocol_model",
    ):
        """
        Train a model directly from the centrally filtered dataframe.

        This follows current train.py-style logic:
          - plain marker style
          - one column per original feature
        """
        if panel_summary is None:
            panel_summary = getattr(self, "_networkparser_panel_summary", None)
            model_name = str(getattr(self, "_networkparser_model_name", model_name))

        NN = self.import_nn()

        try:
            from network_parser.matrix_contract import (
                MissingnessPolicy,
                prepare_for_sklearn,
            )
        except ImportError:  # pragma: no cover
            from matrix_contract import MissingnessPolicy, prepare_for_sklearn  # type: ignore

        # Fit an explicit imputer on this training set. Structural
        # sample/feature filtering belongs to the enclosing training fold; this
        # layer retains its ordered model feature list and only fits values.
        policy = MissingnessPolicy.from_config(self.config)
        policy.drop_exceeding_samples = False
        policy.drop_exceeding_features = False
        X, missingness_state, missingness_audit = prepare_for_sklearn(
            genomic_df,
            policy=policy,
        )
        if X.isna().any().any():
            raise ValueError(
                "Model training still contains non-callable values after the configured "
                "train-fitted imputer. Set genotype_impute_strategy to baseline, "
                "feature_mode, or constant."
            )
        labels = labels.loc[X.index]
        feature_titles = X.columns.astype(str).tolist()
        for col in X.columns:
            numeric = pd.to_numeric(X[col], errors="raise")
            X[col] = numeric.map(
                lambda value: str(int(value))
                if float(value).is_integer()
                else str(float(value))
            )

        X_values = X.values
        y_values = labels.astype(str).to_numpy()

        model = self.select_nn_model(NN, algorithm=algorithm, marker_style="plain")

        try:
            from network_parser.feature_panel_selection import (
                log_model_input_panel_decision,
            )
        except ImportError:  # pragma: no cover
            from feature_panel_selection import (  # type: ignore
                log_model_input_panel_decision,
            )

        log_model_input_panel_decision(
            model_name=model_name,
            X=X,
            panel_summary=panel_summary,
            log=logger,
        )
        logger.info(
            "Training ML protocol model | algorithm=%s | samples=%d | features=%d",
            algorithm,
            X_values.shape[0],
            X_values.shape[1],
        )

        model.train(
            X=X_values,
            y=y_values,
            feature_titles=feature_titles,
        )
        model.networkparser_missingness_state = missingness_state.to_dict()
        model.networkparser_missingness_audit = missingness_audit

        return model

    def save_model(self, model: Any, out_path: Path) -> None:
        try:
            import joblib

            joblib.dump(model, out_path)
        except Exception:
            import pickle

            with open(out_path, "wb") as fh:
                pickle.dump(model, fh)

    # ------------------------------------------------------------------
    # evaluation helpers
    # ------------------------------------------------------------------
    def _feature_overlap(
        self,
        model: Any,
        protocol_df: pd.DataFrame,
    ) -> Tuple[List[str], float]:
        if not hasattr(model, "feature_titles"):
            raise AttributeError("Model has no attribute 'feature_titles'")

        model_features = list(model.feature_titles or [])
        if not model_features:
            raise ValueError("model.feature_titles is empty")

        matrix_features = [str(c) for c in protocol_df.columns[2:]]
        matrix_feature_set = set(matrix_features)

        used_markers = [m for m in model_features if m in matrix_feature_set]
        coverage = len(used_markers) / max(1, len(model_features))

        if coverage < 0.75:
            raise ValueError(
                f"Only {coverage * 100:.1f}% of model features are present in evaluation matrix "
                f"({len(used_markers)}/{len(model_features)}). At least 75% is required."
            )

        return used_markers, coverage

    def _predict_one(
        self,
        model: Any,
        marker_dict: Dict[str, str],
    ) -> Dict[str, Any]:
        result = model.identify(marker_dict)

        if not isinstance(result, dict):
            raise ValueError("model.identify() returned a non-dict result")

        predictions = result.get("predictions", [])
        if not isinstance(predictions, list):
            predictions = []

        norm_predictions: List[Tuple[str, float]] = []
        for item in predictions:
            try:
                label, prob = item[0], item[1]
                norm_predictions.append((str(label), float(prob)))
            except Exception:
                continue

        return {
            "predictions": norm_predictions,
            "raw": result,
        }

    @staticmethod
    def _marker_symbol(value: Any) -> str:
        if pd.isna(value):
            raise ValueError(
                "Non-callable marker reached model inference without train-fitted imputation."
            )
        numeric = float(value)
        return str(int(numeric)) if numeric.is_integer() else str(numeric)

    @staticmethod
    def _prepare_protocol_features(
        model: Any,
        protocol_df: pd.DataFrame,
        used_markers: Sequence[str],
    ) -> pd.DataFrame:
        """Apply the imputer fitted with ``model`` to evaluation/query markers."""
        try:
            from network_parser.matrix_contract import (
                FittedMissingnessState,
                transform_with_missingness_state,
            )
        except ImportError:  # pragma: no cover
            from matrix_contract import (  # type: ignore
                FittedMissingnessState,
                transform_with_missingness_state,
            )

        X = protocol_df.loc[:, list(used_markers)].copy()
        raw_state = getattr(model, "networkparser_missingness_state", None)
        if raw_state:
            state = (
                raw_state
                if isinstance(raw_state, FittedMissingnessState)
                else FittedMissingnessState.from_dict(raw_state)
            )
            X, _ = transform_with_missingness_state(
                X,
                state,
                apply_imputation=True,
                drop_high_missing_samples=False,
            )
        else:
            X = X.apply(pd.to_numeric, errors="coerce")
        if X.isna().any().any():
            raise ValueError(
                "Evaluation contains non-callable required markers but the model has no "
                "usable train-fitted preprocessing state."
            )
        return X

    def identify_records(
        self,
        model: Any,
        protocol_df: pd.DataFrame,
        used_markers: List[str],
        decision_threshold: float,
        *,
        sensitivity: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Per-sample identification at a fixed decision (minimum support) threshold.

        ``sensitivity`` is accepted as a deprecated alias for ``decision_threshold``.
        """
        if sensitivity is not None:
            decision_threshold = float(sensitivity)
        thr = float(decision_threshold)
        records: List[Dict[str, Any]] = []
        prepared_features = self._prepare_protocol_features(
            model,
            protocol_df,
            used_markers,
        )

        for row_position, (_, row) in enumerate(protocol_df.iterrows()):
            sample_id = str(row.iloc[0])
            true_label = str(row.iloc[1])

            marker_dict = {}
            for marker in used_markers:
                value = prepared_features.iloc[row_position][marker]
                marker_dict[marker] = self._marker_symbol(value)

            pred = self._predict_one(model, marker_dict)
            predictions = pred["predictions"]

            if not predictions:
                top_label = ""
                top_support = 0.0
            else:
                top_label, top_support = predictions[0]

            is_called = bool(top_support >= thr)
            called_label = top_label if is_called else ""

            records.append(
                {
                    "sample_id": sample_id,
                    "true_label": true_label,
                    "predicted_label": top_label if is_called else "not_called",
                    "raw_top_label": top_label,
                    # Primary field: support score (not calibrated probability unless noted).
                    "support_score": float(top_support),
                    "top_support_score": float(top_support),
                    "score_kind": "support_score",
                    "called_label": called_label,
                    "called": is_called,
                    "decision_threshold": thr,
                    "minimum_support_threshold": thr,
                    "sensitivity": thr,  # deprecated alias
                }
            )

        return records

    def score_records(
        self,
        records: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Score a fixed decision threshold.

        Components are always reported separately. The configured objective
        value is added by ``score_threshold_objective``.
        """
        if not records:
            return {
                "n_truth_samples": 0,
                "n_records": 0,
                "n_called": 0,
                "n_abstained": 0,
                "n_correct_called": 0,
                "n_wrong_called": 0,
                "call_rate": 0.0,
                "coverage_call_rate": 0.0,
                "accuracy_called_only": 0.0,
                "called_error_rate": 0.0,
                "accuracy_end_to_end_all_truth": 0.0,
                "accuracy_all_samples": 0.0,
                "utility": 0.0,
            }

        n_records = len(records)
        called = [r for r in records if bool(r.get("called", False))]
        n_called = len(called)
        n_abstained = n_records - n_called

        correct_called = sum(
            1
            for r in called
            if str(r.get("called_label", "")) == str(r.get("true_label", ""))
        )
        wrong_called = n_called - correct_called
        e2e_correct = correct_called
        call_rate = float(n_called / max(1, n_records))
        acc_called = float(correct_called / max(1, n_called)) if n_called else 0.0
        called_err = float(wrong_called / max(1, n_called)) if n_called else 0.0
        acc_e2e = float(e2e_correct / max(1, n_records))

        reward = 1.0
        cost_wrong = 2.0
        cost_abs = 0.5
        if self.config is not None:
            reward = float(
                getattr(self.config, "ml_threshold_utility_reward_correct", 1.0)
            )
            cost_wrong = float(
                getattr(self.config, "ml_threshold_utility_cost_wrong", 2.0)
            )
            cost_abs = float(
                getattr(self.config, "ml_threshold_utility_cost_abstain", 0.5)
            )
        utility = (
            reward * correct_called - cost_wrong * wrong_called - cost_abs * n_abstained
        )

        return {
            "n_truth_samples": int(n_records),
            "n_records": int(n_records),
            "n_called": int(n_called),
            "n_abstained": int(n_abstained),
            "n_correct_called": int(correct_called),
            "n_wrong_called": int(wrong_called),
            "call_rate": call_rate,
            "coverage_call_rate": call_rate,
            "accuracy_called_only": acc_called,
            "called_error_rate": called_err,
            "accuracy_end_to_end_all_truth": acc_e2e,
            "accuracy_all_samples": acc_e2e,
            "utility": float(utility),
        }

    def score_threshold_objective(self, scored: Dict[str, float]) -> Tuple[float, str]:
        """
        Selective-classification objective for threshold selection.

        Default: minimise called-sample error subject to minimum coverage.
        Higher objective_value is better for all objectives (including the
        min-error form, which is scored as ``1 - called_error_rate`` when the
        coverage constraint is met, else ``-inf``).
        """
        objective = "min_called_error_subject_to_coverage"
        w_called = 0.5
        w_cov = 0.5
        min_cov = 0.5
        if self.config is not None:
            objective = str(getattr(self.config, "ml_threshold_objective", objective))
            w_called = float(
                getattr(self.config, "ml_threshold_objective_called_weight", 0.5)
            )
            w_cov = float(
                getattr(self.config, "ml_threshold_objective_coverage_weight", 0.5)
            )
            min_cov = float(getattr(self.config, "ml_threshold_min_coverage", 0.5))

        call_rate = float(scored.get("call_rate", 0.0))
        called_err = float(scored.get("called_error_rate", 1.0))
        acc_called = float(scored.get("accuracy_called_only", 0.0))

        if objective == "min_called_error_subject_to_coverage":
            if call_rate + 1e-12 < min_cov:
                return float("-inf"), objective
            # Higher is better: prefer lower called error; break ties by higher coverage
            return float(1.0 - called_err + 1e-6 * call_rate), objective
        if objective == "utility":
            return float(scored.get("utility", 0.0)), objective
        if objective == "accuracy_called_only":
            return acc_called, objective
        if objective == "call_rate":
            return call_rate, objective
        if objective == "balanced_called_and_coverage":
            total_w = max(1e-12, w_called + w_cov)
            val = (w_called * acc_called + w_cov * call_rate) / total_w
            return float(val), objective
        # Legacy end-to-end (not recommended for selective classification)
        return (
            float(scored.get("accuracy_end_to_end_all_truth", 0.0)),
            "accuracy_end_to_end",
        )

    def evaluate_model(
        self,
        model: Any,
        protocol_df: pd.DataFrame,
        out_dir: Path,
        *,
        decision_threshold: Optional[float] = None,
        evaluation_role: str = "training_fit_diagnostics",
    ) -> Dict[str, Any]:
        """
        Score a fitted model on the provided protocol matrix.

        When ``evaluation_role`` is ``training_fit_diagnostics``, results are
        same-data diagnostics and must not be reported as generalisation.
        Threshold grids here are for diagnostics only; production threshold
        selection uses ``select_decision_threshold_out_of_fold``.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        used_markers, coverage = self._feature_overlap(model, protocol_df)

        min_thr = 0.5
        max_thr = 1.0
        step_thr = 0.1
        if self.config is not None:
            min_thr = float(
                getattr(
                    self.config,
                    "ml_min_decision_threshold",
                    getattr(self.config, "ml_min_sensitivity", 0.5),
                )
            )
            max_thr = float(
                getattr(
                    self.config,
                    "ml_max_decision_threshold",
                    getattr(self.config, "ml_max_sensitivity", 1.0),
                )
            )
            step_thr = float(
                getattr(
                    self.config,
                    "ml_step_decision_threshold",
                    getattr(self.config, "ml_step_sensitivity", 0.1),
                )
            )

        thresholds = np.arange(min_thr, max_thr + (step_thr / 10.0), step_thr)

        threshold_rows: List[Dict[str, Any]] = []
        for thr in thresholds:
            records = self.identify_records(
                model=model,
                protocol_df=protocol_df,
                used_markers=used_markers,
                decision_threshold=float(thr),
            )
            scored = self.score_records(records)
            obj_val, obj_name = self.score_threshold_objective(scored)
            row = {
                "decision_threshold": float(thr),
                "minimum_support_threshold": float(thr),
                "sensitivity": float(thr),  # deprecated alias
                "objective_name": obj_name,
                "objective_value": float(obj_val),
                **scored,
            }
            threshold_rows.append(row)

        if decision_threshold is None:
            # Diagnostic pick only — not used for final threshold selection
            best_row = (
                max(
                    threshold_rows,
                    key=lambda r: r.get("objective_value", float("-inf")),
                )
                if threshold_rows
                else {}
            )
            decision_threshold = (
                float(best_row.get("decision_threshold", min_thr))
                if best_row
                else min_thr
            )
        else:
            decision_threshold = float(decision_threshold)
            best_row = (
                min(
                    threshold_rows,
                    key=lambda r: abs(
                        float(r.get("decision_threshold", 0.0)) - decision_threshold
                    ),
                )
                if threshold_rows
                else {}
            )

        records = self.identify_records(
            model=model,
            protocol_df=protocol_df,
            used_markers=used_markers,
            decision_threshold=decision_threshold,
        )
        scored_best = self.score_records(records)
        obj_val, obj_name = self.score_threshold_objective(scored_best)

        thresholds_df = pd.DataFrame(threshold_rows)
        predictions_df = pd.DataFrame(records)

        thresholds_path = out_dir / "ml_protocol_thresholds.tsv"
        preds_path = out_dir / "ml_protocol_sample_predictions.tsv"
        eval_json_path = out_dir / "ml_protocol_evaluation.json"

        thresholds_df.to_csv(thresholds_path, sep="\t", index=False)
        predictions_df.to_csv(preds_path, sep="\t", index=False)

        evaluation = {
            "evaluation_role": evaluation_role,
            "is_generalization_estimate": evaluation_role
            not in {"training_fit_diagnostics"},
            "feature_overlap_coverage": float(coverage),
            "used_marker_count": int(len(used_markers)),
            "used_markers": used_markers,
            "decision_threshold": float(decision_threshold),
            "minimum_support_threshold": float(decision_threshold),
            "objective_name": obj_name,
            "objective_value": float(obj_val),
            "objective_components": {
                "accuracy_called_only": scored_best.get("accuracy_called_only"),
                "call_rate": scored_best.get("call_rate"),
                "accuracy_end_to_end_all_truth": scored_best.get(
                    "accuracy_end_to_end_all_truth"
                ),
                "n_truth_samples": scored_best.get("n_truth_samples"),
                "n_called": scored_best.get("n_called"),
                "n_abstained": scored_best.get("n_abstained"),
            },
            "threshold_grid_diagnostics": threshold_rows,
            "selected_threshold_row": best_row,
            "method_note": (
                "Decision threshold is a minimum class-support cutoff, not a clinical sensitivity. "
                + (
                    "These are same-data training-fit diagnostics, not generalisation performance."
                    if evaluation_role == "training_fit_diagnostics"
                    else "Evaluate generalisation on held-out or out-of-fold predictions."
                )
            ),
        }

        with open(eval_json_path, "w", encoding="utf-8") as fh:
            json.dump(evaluation, fh, indent=2, default=_json_default)

        return evaluation

    def _raw_support_scores_for_frame(
        self,
        model: Any,
        protocol_df: pd.DataFrame,
        used_markers: List[str],
    ) -> List[Dict[str, Any]]:
        """Run inference once; return raw top-label support scores (no threshold)."""
        rows: List[Dict[str, Any]] = []
        prepared_features = self._prepare_protocol_features(
            model,
            protocol_df,
            used_markers,
        )
        for row_position, (_, row) in enumerate(protocol_df.iterrows()):
            sample_id = str(row.iloc[0])
            true_label = str(row.iloc[1])
            marker_dict = {}
            for marker in used_markers:
                value = prepared_features.iloc[row_position][marker]
                marker_dict[marker] = self._marker_symbol(value)
            pred = self._predict_one(model, marker_dict)
            predictions = pred["predictions"]
            if not predictions:
                top_label, top_support = "", 0.0
            else:
                top_label, top_support = predictions[0]
            rows.append(
                {
                    "sample_id": sample_id,
                    "true_label": true_label,
                    "raw_top_label": top_label,
                    "support_score": float(top_support),
                    "score_kind": "support_score",
                }
            )
        return rows

    @staticmethod
    def _apply_threshold_to_raw_scores(
        raw_rows: List[Dict[str, Any]],
        thr: float,
    ) -> List[Dict[str, Any]]:
        """Apply a decision threshold to precomputed support scores (no re-inference)."""
        thr = float(thr)
        out: List[Dict[str, Any]] = []
        for r in raw_rows:
            score = float(r.get("support_score", 0.0))
            top_label = str(r.get("raw_top_label", ""))
            is_called = bool(score >= thr)
            called_label = top_label if is_called else ""
            out.append(
                {
                    **r,
                    "predicted_label": top_label if is_called else "not_called",
                    "called_label": called_label,
                    "called": is_called,
                    "decision_threshold": thr,
                    "minimum_support_threshold": thr,
                    "top_support_score": score,
                }
            )
        return out

    def select_decision_threshold_out_of_fold(
        self,
        *,
        genomic_df: pd.DataFrame,
        labels: pd.Series,
        algorithm: str,
        out_dir: Path,
        groups: Optional[pd.Series] = None,
        panel_summary: Optional[Dict[str, Any]] = None,
        model_name: str = "ml_protocol_model",
    ) -> Dict[str, Any]:
        """
        Select decision threshold using nested out-of-fold support scores.

        For each fold: fit on train only; score validation once; then evaluate
        the full threshold grid from stored support scores (no re-inference).
        """
        from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        use_oof = True
        n_splits = 5
        if self.config is not None:
            use_oof = bool(
                getattr(self.config, "ml_select_threshold_out_of_fold", True)
            )
            n_splits = int(getattr(self.config, "ml_threshold_cv_splits", 5))

        min_thr = float(
            getattr(self.config, "ml_min_decision_threshold", 0.5)
            if self.config
            else 0.5
        )
        max_thr = float(
            getattr(self.config, "ml_max_decision_threshold", 1.0)
            if self.config
            else 1.0
        )
        step_thr = float(
            getattr(self.config, "ml_step_decision_threshold", 0.1)
            if self.config
            else 0.1
        )
        thresholds = [
            float(t) for t in np.arange(min_thr, max_thr + (step_thr / 10.0), step_thr)
        ]

        objective_name = str(
            getattr(
                self.config,
                "ml_threshold_objective",
                "min_called_error_subject_to_coverage",
            )
            if self.config
            else "min_called_error_subject_to_coverage"
        )
        min_coverage = float(
            getattr(self.config, "ml_threshold_min_coverage", 0.5)
            if self.config
            else 0.5
        )
        calibrate = bool(
            getattr(self.config, "ml_calibrate_support_scores", False)
            if self.config
            else False
        )
        calib_method = str(
            getattr(self.config, "ml_calibration_method", "none")
            if self.config
            else "none"
        )
        if not calibrate:
            calib_method = "none"

        y = labels.astype(str)
        X = genomic_df.copy()
        common = X.index.intersection(y.index)
        X = X.loc[common]
        y = y.loc[common]
        g = None
        if groups is not None:
            g = groups.astype(str)
            g = g.loc[g.index.intersection(common)]
            common2 = X.index.intersection(g.index)
            X = X.loc[common2]
            y = y.loc[common2]
            g = g.loc[common2]

        sample_ids = np.asarray(X.index.astype(str))
        y_values = y.astype(str).to_numpy()
        g_values = g.astype(str).to_numpy() if g is not None else None

        def _fallback(method: str, note: str) -> Dict[str, Any]:
            mid = float(thresholds[len(thresholds) // 2]) if thresholds else 0.5
            payload = {
                "status": "fallback_fixed_threshold",
                "selected_decision_threshold": mid,
                "minimum_support_threshold": mid,
                "method": method,
                "objective_name": objective_name,
                "min_coverage_constraint": min_coverage,
                "calibration_method": calib_method,
                "score_kind": "support_score",
                "oof_provenance": "none",
                "note": note,
            }
            with open(
                out_dir / "threshold_selection_summary.json", "w", encoding="utf-8"
            ) as fh:
                json.dump(payload, fh, indent=2, default=_json_default)
            return payload

        if not use_oof or y.nunique() < 2 or len(y) < max(4, n_splits):
            return _fallback(
                "fallback_no_oof",
                "Insufficient samples/classes or OOF disabled; mid-grid threshold without maximisation.",
            )

        class_counts = pd.Series(y_values).value_counts()
        feasible = min(n_splits, int(class_counts.min()))
        if g_values is not None:
            class_group_counts = [
                int(pd.Series(g_values[y_values == label]).nunique())
                for label in class_counts.index
            ]
            feasible = min(
                feasible,
                int(pd.Series(g_values).nunique()),
                min(class_group_counts) if class_group_counts else 0,
            )
        if feasible < 2:
            return _fallback("fallback_rare_class", "Feasible stratified splits < 2.")

        rs = int(getattr(self.config, "random_state", 42) if self.config else 42)
        if g_values is not None:
            cv = StratifiedGroupKFold(n_splits=feasible, shuffle=True, random_state=rs)
            split_iter = cv.split(sample_ids, y_values, groups=g_values)
            split_method = "out_of_fold_stratified_group_cv"
        else:
            cv = StratifiedKFold(n_splits=feasible, shuffle=True, random_state=rs)
            split_iter = cv.split(sample_ids, y_values)
            split_method = "out_of_fold_stratified_cv"

        # One inference pass per fold → raw support scores
        raw_oof: List[Dict[str, Any]] = []
        for fold_i, (train_idx, val_idx) in enumerate(split_iter, start=1):
            train_ids = sample_ids[train_idx]
            val_ids = sample_ids[val_idx]
            if g_values is not None:
                if set(g_values[train_idx]).intersection(set(g_values[val_idx])):
                    raise RuntimeError(
                        "Grouped OOF threshold selection leaked groups across folds"
                    )
            X_train = X.loc[train_ids]
            y_train = y.loc[train_ids]
            X_val = X.loc[val_ids]
            y_val = y.loc[val_ids]
            train_kwargs: Dict[str, Any] = {
                "genomic_df": X_train,
                "labels": y_train,
                "algorithm": algorithm,
            }
            if panel_summary is not None:
                train_kwargs.update(
                    {
                        "panel_summary": panel_summary,
                        "model_name": f"{model_name}__threshold_oof_fold_{fold_i}",
                    }
                )
            model = self.train_model(**train_kwargs)
            used_markers = [
                c
                for c in X_train.columns.astype(str).tolist()
                if c in set(map(str, X_val.columns))
            ]
            proto = X_val.copy()
            proto.insert(0, "__label__", y_val.astype(str).values)
            proto.insert(0, "__sample_id__", val_ids)
            fold_raw = self._raw_support_scores_for_frame(model, proto, used_markers)
            for r in fold_raw:
                r["fold"] = int(fold_i)
                if g is not None:
                    r["group"] = str(g.loc[str(r["sample_id"])])
                raw_oof.append(r)

        raw_df = pd.DataFrame(raw_oof)
        raw_df.to_csv(out_dir / "oof_raw_support_scores.tsv", sep="\t", index=False)

        # Optional isotonic calibration of top support scores (multiclass: score only)
        calibration_note = "none"
        if calib_method == "isotonic" and not raw_df.empty:
            try:
                from sklearn.isotonic import IsotonicRegression

                # Binary-style: correctness of top label as target for score calibration
                y_bin = (
                    (
                        raw_df["raw_top_label"].astype(str)
                        == raw_df["true_label"].astype(str)
                    )
                    .astype(int)
                    .to_numpy()
                )
                x_sc = raw_df["support_score"].astype(float).to_numpy()
                if len(np.unique(y_bin)) >= 2 and len(x_sc) >= 5:
                    iso = IsotonicRegression(out_of_bounds="clip")
                    raw_df["support_score"] = iso.fit_transform(x_sc, y_bin)
                    calibration_note = "isotonic_on_oof_top_label_correctness"
                else:
                    calibration_note = "isotonic_skipped_insufficient_labels"
            except Exception as exc:
                calibration_note = f"isotonic_failed:{exc}"

        # Threshold grid from frozen raw scores (no re-inference)
        grid_rows: List[Dict[str, Any]] = []
        best_score = float("-inf")
        best_thr = float(thresholds[0]) if thresholds else 0.5
        best_obj_name = objective_name
        all_threshold_records: List[Dict[str, Any]] = []

        raw_records = raw_df.to_dict(orient="records")
        for thr in thresholds:
            subset = self._apply_threshold_to_raw_scores(raw_records, thr)
            for r in subset:
                all_threshold_records.append(r)
            scored = self.score_records(subset)
            obj_val, obj_name = self.score_threshold_objective(scored)
            row = {
                "decision_threshold": float(thr),
                "objective_name": obj_name,
                "objective_value": float(obj_val) if np.isfinite(obj_val) else None,
                "meets_min_coverage": bool(
                    scored.get("call_rate", 0.0) + 1e-12 >= min_coverage
                ),
                **scored,
            }
            grid_rows.append(row)
            if np.isfinite(obj_val) and obj_val > best_score:
                best_score = obj_val
                best_thr = float(thr)
                best_obj_name = obj_name

        # If no threshold met coverage constraint, pick highest coverage then lowest called error
        if not np.isfinite(best_score) or best_score == float("-inf"):
            feasible_rows = [r for r in grid_rows if r.get("n_called", 0) > 0]
            if feasible_rows:
                best_row = sorted(
                    feasible_rows,
                    key=lambda r: (
                        -float(r.get("call_rate", 0.0)),
                        float(r.get("called_error_rate", 1.0)),
                    ),
                )[0]
                best_thr = float(best_row["decision_threshold"])
                best_score = float(best_row.get("objective_value") or float("-inf"))
                best_obj_name = str(best_row.get("objective_name", objective_name))
                constraint_status = "relaxed_no_threshold_met_min_coverage"
            else:
                constraint_status = "no_valid_threshold"
        else:
            constraint_status = "ok"

        pd.DataFrame(grid_rows).to_csv(
            out_dir / "oof_threshold_grid.tsv", sep="\t", index=False
        )
        pd.DataFrame(all_threshold_records).to_csv(
            out_dir / "oof_threshold_records.tsv", sep="\t", index=False
        )

        payload = {
            "status": "success",
            "method": split_method,
            "n_splits": int(feasible),
            "selected_decision_threshold": float(best_thr),
            "minimum_support_threshold": float(best_thr),
            "objective_name": best_obj_name,
            "objective_value": float(best_score) if np.isfinite(best_score) else None,
            "min_coverage_constraint": float(min_coverage),
            "constraint_status": constraint_status,
            "calibration_method": calib_method,
            "calibration_note": calibration_note,
            "score_kind": (
                "calibrated_support_score"
                if calibration_note.startswith("isotonic_on")
                else "support_score"
            ),
            "oof_provenance": {
                "split_method": split_method,
                "n_oof_samples": int(len(raw_df)),
                "n_folds": int(feasible),
                "grouped": bool(g_values is not None),
                "inference_policy": "score_once_per_fold_then_threshold_grid",
                "final_model_fit": "after_threshold_frozen",
            },
            "grid": grid_rows,
            "note": (
                "Selective-classification threshold selected on nested OOF support scores only. "
                "Raw scores generated once per fold; threshold grid applied without re-inference. "
                "Final deployment model is fit after this threshold is frozen. "
                "Scores are support scores unless calibration_note indicates isotonic calibration."
            ),
        }
        with open(
            out_dir / "threshold_selection_summary.json", "w", encoding="utf-8"
        ) as fh:
            json.dump(payload, fh, indent=2, default=_json_default)
        return payload

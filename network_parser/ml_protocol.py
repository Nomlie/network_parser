# network_parser/ml_protocol.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
            empty_thr = float(getattr(self.config, "ml_remove_empty_field_threshold", 1.0))
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

        run_selector = bool(getattr(self.config, "run_model_selector", True)) if self.config is not None else True

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
            selector_result = self._selector_disabled_payload(requested_algorithm=requested_algo)

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
        # Training stage
        # --------------------------------------------------------------
        model = self.train_model(
            genomic_df=genomic_df_aligned,
            labels=labels_aligned,
            algorithm=selected_algo,
        )

        model_path = out_dir / f"{selected_algo}_ml_protocol_model.pkl"
        self.save_model(model, model_path)

        # --------------------------------------------------------------
        # Evaluation stage
        # --------------------------------------------------------------
        evaluation = self.evaluate_model(
            model=model,
            protocol_df=protocol_df,
            out_dir=out_dir,
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
            "requested_algorithm": "auto" if requested_algo is None else str(requested_algo),
            "selector_enabled": bool(run_selector),
            "selector": selector_result,
            "branch_decision": branch_decision,
            "training_metrics": getattr(model, "training_metrics", {}),
            "interpretability": interpretability,
            "evaluation": evaluation,
            "artifacts": {
                "protocol_matrix": str(protocol_matrix_path),
                "model_file": str(model_path),
                "evaluation_json": str(out_dir / "ml_protocol_evaluation.json"),
                "evaluation_tsv": str(out_dir / "ml_protocol_thresholds.tsv"),
                "sample_predictions_tsv": str(out_dir / "ml_protocol_sample_predictions.tsv"),
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
            {"": pd.NA, "-": pd.NA, "NA": pd.NA, "N/A": pd.NA, "None": pd.NA, "nan": pd.NA, "NaN": pd.NA}
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
        except Exception:
            from model_selector import recommend_classifier
            return recommend_classifier

    def _encode_for_selector(self, genomic_df: pd.DataFrame) -> np.ndarray:
        """
        Encode mixed/categorical feature columns into integer codes for selector probing.

        This keeps selector probing separate from the raw dataframe used for training.
        """
        encoded_cols: List[np.ndarray] = []

        for col in genomic_df.columns:
            s = genomic_df[col].copy()
            s = s.where(~s.isna(), "__MISSING__")
            s = s.astype(str).str.strip()
            s = s.replace({"": "__MISSING__", "nan": "__MISSING__", "NaN": "__MISSING__", "nd": "__MISSING__"})

            cat = pd.Categorical(s)
            codes = cat.codes.astype(float)
            encoded_cols.append(codes)

        if not encoded_cols:
            raise ValueError("No feature columns available for selector encoding")

        X = np.column_stack(encoded_cols)
        return X

    def select_model(
        self,
        genomic_df: pd.DataFrame,
        labels: pd.Series,
    ) -> Dict[str, Any]:
        """
        Run the selector stage using in-memory dataframe + labels.
        """
        recommend_classifier = self.import_selector()

        X = self._encode_for_selector(genomic_df)
        y = labels.astype(str).to_numpy()

        try:
            result = recommend_classifier(X, y)
        except Exception as exc:
            logger.warning("Model selector failed, defaulting to RF: %s", exc)
            result = {
                "recommendation": "RF",
                "rationale": [f"selector_failed: {exc}"],
                "probe_scores": {},
                "candidate_ranked": ["RF"],
                "dt_candidate": False,
            }

        if "recommendation" not in result:
            result["recommendation"] = "RF"

        logger.info("ML selector recommendation: %s", result["recommendation"])
        logger.info("ML selector probe scores | %s", result.get("probe_scores", {}))
        return result

    def _selector_disabled_payload(
        self,
        requested_algorithm: Optional[str],
    ) -> Dict[str, Any]:
        """
        Build a consistent selector-like payload when model screening is disabled.
        """
        req = "auto" if requested_algorithm is None else str(requested_algorithm).strip()

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
        interpretable = [a for a in candidate_ranked if a in {"DT", "LR", "RF", "SVC", "MLP"}]

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

        recommendation = str(result.get("recommendation", "RF")).strip() or "RF"
        result["recommendation"] = recommendation

        probe_scores = result.get("probe_scores", {})
        if not isinstance(probe_scores, dict):
            probe_scores = {}

        ranked: List[str] = []
        scored_items: List[Tuple[str, float]] = []
        for k, v in probe_scores.items():
            if k == "delta_nonlinear_minus_linear":
                continue
            fv = _safe_float(v, default=float("-inf"))
            if np.isfinite(fv):
                scored_items.append((str(k), fv))

        scored_items.sort(key=lambda x: x[1], reverse=True)
        ranked = [name for name, _ in scored_items]

        normalized_ranked: List[str] = []
        for item in ranked:
            normalized_ranked.append(self._normalize_selector_name(item))

        normalized_recommendation = self._normalize_selector_name(recommendation)
        if normalized_recommendation not in normalized_ranked:
            normalized_ranked.insert(0, normalized_recommendation)

        candidate_ranked = result.get("candidate_ranked", normalized_ranked)
        if not isinstance(candidate_ranked, list):
            candidate_ranked = normalized_ranked

        candidate_ranked = [self._normalize_selector_name(x) for x in candidate_ranked]

        dt_candidate = bool(
            result.get("dt_candidate", False)
            or normalized_recommendation == "DT"
            or "DT" in candidate_ranked
        )

        interpretable_models = []
        for algo in candidate_ranked:
            if algo in {"DT", "LR", "RF", "SVC", "MLP"} and algo not in interpretable_models:
                interpretable_models.append(algo)

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
        selector_recommendation: str,
        requested_algorithm: Optional[str] = None,
    ) -> str:
        """
        Resolve final algorithm to one supported by model_wrapper.py.
        """
        req = "auto" if requested_algorithm is None else str(requested_algorithm).strip()
        rec = str(selector_recommendation).strip()

        if req and req.lower() != "auto":
            if req not in self.SUPPORTED_ALGOS:
                raise ValueError(
                    f"Unsupported requested ML algorithm '{req}'. "
                    f"Supported: {sorted(self.SUPPORTED_ALGOS)} plus 'auto'."
                )
            return "SVC" if req == "SCV" else req

        mapping = {
            "RF": "RF",
            "DT": "DT",
            "LR": "LR",
            "MLP": "MLP",
            "MBCS": "MBCS",
            "DNL": "DNL",
            "SVC": "SVC",
            "KNN": "RF",
            "NBayes": "RF",
            "XGBoost": "RF",
        }

        return mapping.get(rec, "RF")

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
        requested = "auto" if requested_algorithm is None else str(requested_algorithm).strip()
        recommendation = str(selector_result.get("recommendation", "RF")).strip()
        candidate_ranked = selector_result.get("candidate_ranked", [])
        if not isinstance(candidate_ranked, list):
            candidate_ranked = []

        dt_candidate = bool(selector_result.get("dt_candidate", False) or "DT" in candidate_ranked)
        run_dt = bool(selected_algorithm == "DT" or recommendation == "DT" or dt_candidate)

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
    # model_wrapper.py integration
    # ------------------------------------------------------------------
    def import_nn(self):
        try:
            import network_parser.model_wrapper as NN
            return NN
        except Exception:
            import network_parser.model_wrapper as NN
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

        if algo == "RF" and getattr(model, "max_features", None) == "auto":
            model.max_features = "sqrt"

        return model

    def train_model(
        self,
        genomic_df: pd.DataFrame,
        labels: pd.Series,
        algorithm: str,
    ):
        """
        Train a model directly from the centrally filtered dataframe.

        This follows current train.py-style logic:
          - plain marker style
          - one column per original feature
        """
        NN = self.import_nn()

        feature_titles = genomic_df.columns.astype(str).tolist()

        X = genomic_df.copy()
        for col in X.columns:
            X[col] = X[col].where(~X[col].isna(), "")
            X[col] = X[col].astype(str).str.strip()

        X_values = X.values
        y_values = labels.astype(str).to_numpy()

        model = self.select_nn_model(NN, algorithm=algorithm, marker_style="plain")

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

    def identify_records(
        self,
        model: Any,
        protocol_df: pd.DataFrame,
        used_markers: List[str],
        sensitivity: float,
    ) -> List[Dict[str, Any]]:
        """
        Per-sample identification at a fixed sensitivity threshold.
        """
        records: List[Dict[str, Any]] = []

        for _, row in protocol_df.iterrows():
            sample_id = str(row.iloc[0])
            true_label = str(row.iloc[1])

            marker_dict = {}
            for marker in used_markers:
                value = row[marker]
                if pd.isna(value):
                    marker_dict[marker] = ""
                else:
                    marker_dict[marker] = str(value).strip()

            pred = self._predict_one(model, marker_dict)
            predictions = pred["predictions"]

            if not predictions:
                top_label = ""
                top_prob = 0.0
            else:
                top_label, top_prob = predictions[0]

            is_called = bool(top_prob >= sensitivity)
            called_label = top_label if is_called else ""

            records.append(
                {
                    "sample_id": sample_id,
                    "true_label": true_label,
                    "predicted_label": top_label,
                    "predicted_probability": float(top_prob),
                    "called_label": called_label,
                    "called": is_called,
                    "sensitivity": float(sensitivity),
                }
            )

        return records

    def score_records(
        self,
        records: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Compute simple accuracy / call-rate summaries for a fixed threshold.
        """
        if not records:
            return {
                "n_records": 0,
                "n_called": 0,
                "call_rate": 0.0,
                "accuracy_called_only": 0.0,
                "accuracy_all_samples": 0.0,
            }

        n_records = len(records)
        called = [r for r in records if bool(r.get("called", False))]
        n_called = len(called)

        correct_called = sum(
            1 for r in called if str(r.get("called_label", "")) == str(r.get("true_label", ""))
        )
        correct_all = sum(
            1 for r in records if str(r.get("predicted_label", "")) == str(r.get("true_label", ""))
        )

        return {
            "n_records": int(n_records),
            "n_called": int(n_called),
            "call_rate": float(n_called / max(1, n_records)),
            "accuracy_called_only": float(correct_called / max(1, n_called)),
            "accuracy_all_samples": float(correct_all / max(1, n_records)),
        }

    def evaluate_model(
        self,
        model: Any,
        protocol_df: pd.DataFrame,
        out_dir: Path,
    ) -> Dict[str, Any]:
        """
        Evaluate model across a sensitivity range and save artifacts.
        """
        used_markers, coverage = self._feature_overlap(model, protocol_df)

        min_sens = 0.5
        max_sens = 1.0
        step_sens = 0.1
        if self.config is not None:
            min_sens = float(getattr(self.config, "ml_min_sensitivity", 0.5))
            max_sens = float(getattr(self.config, "ml_max_sensitivity", 1.0))
            step_sens = float(getattr(self.config, "ml_step_sensitivity", 0.1))

        thresholds = np.arange(min_sens, max_sens + (step_sens / 10.0), step_sens)

        threshold_rows: List[Dict[str, Any]] = []
        best_records: List[Dict[str, Any]] = []
        best_summary: Optional[Dict[str, Any]] = None
        best_score = float("-inf")

        for thr in thresholds:
            records = self.identify_records(
                model=model,
                protocol_df=protocol_df,
                used_markers=used_markers,
                sensitivity=float(thr),
            )
            scored = self.score_records(records)

            row = {
                "sensitivity": float(thr),
                **scored,
            }
            threshold_rows.append(row)

            composite_score = scored["accuracy_called_only"] + scored["call_rate"]
            if composite_score > best_score:
                best_score = composite_score
                best_records = records
                best_summary = row

        thresholds_df = pd.DataFrame(threshold_rows)
        predictions_df = pd.DataFrame(best_records)

        thresholds_path = out_dir / "ml_protocol_thresholds.tsv"
        preds_path = out_dir / "ml_protocol_sample_predictions.tsv"
        eval_json_path = out_dir / "ml_protocol_evaluation.json"

        thresholds_df.to_csv(thresholds_path, sep="\t", index=False)
        predictions_df.to_csv(preds_path, sep="\t", index=False)

        evaluation = {
            "feature_overlap_coverage": float(coverage),
            "used_marker_count": int(len(used_markers)),
            "used_markers": used_markers,
            "best_threshold_summary": best_summary if best_summary is not None else {},
            "all_thresholds": threshold_rows,
        }

        with open(eval_json_path, "w", encoding="utf-8") as fh:
            json.dump(evaluation, fh, indent=2, default=_json_default)

        return evaluation
# network_parser/ml_protocol.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

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

    Purpose
    -------
    Consume the sample x feature dataframe already created by DataLoader,
    then run:

        selector -> training -> testing/evaluation

    using the logic derived from:
        - model_selector.py
        - train.py
        - tester.py

    Notes
    -----
    - This branch is downstream and does NOT replace the main
      decision-tree / interaction discovery branch.
    - Input dataframe is expected to be sample-centric:
          index   = sample identifiers
          columns = genomic features / polymorphic sites
          values  = categorical or binary feature states
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
        Run the full ML protocol on an already-aligned dataframe.

        Parameters
        ----------
        genomic_df
            Sample x feature dataframe from DataLoader.
        labels
            Label series aligned or alignable to genomic_df.index.
        output_dir
            Root output directory.
        algorithm
            Optional forced algorithm. If None or "auto", selector decides.

        Returns
        -------
        dict
            Summary of selector, training, and evaluation outputs.
        """
        self._validate_inputs(genomic_df, labels)

        out_dir = _ensure_dir(Path(output_dir) / "ml_protocol")

        genomic_df_aligned, labels_aligned = self._align_inputs(genomic_df, labels)
        protocol_df = self.build_protocol_df(genomic_df_aligned, labels_aligned)

        # NEW: apply train.py-style empty-column filtering before selector/train/eval
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

        # Rebuild aligned genomic/labels from the filtered protocol matrix
        genomic_df_aligned = protocol_df.iloc[:, 2:].copy()
        labels_aligned = protocol_df.iloc[:, 1].copy()
        genomic_df_aligned.index = protocol_df.iloc[:, 0].astype(str)

        protocol_matrix_path = out_dir / "ml_protocol_matrix.csv"
        protocol_df.to_csv(protocol_matrix_path, index=False)

        selector_result = self.select_model(genomic_df_aligned, labels_aligned)

        requested_algo = algorithm
        if requested_algo is None and self.config is not None:
            requested_algo = getattr(self.config, "ml_algorithm", "auto")

        selected_algo = self.resolve_algorithm(
            selector_recommendation=selector_result.get("recommendation", "RF"),
            requested_algorithm=requested_algo,
        )

        model = self.train_model(
            genomic_df=genomic_df_aligned,
            labels=labels_aligned,
            algorithm=selected_algo,
        )

        model_path = out_dir / f"{selected_algo}_ml_protocol_model.pkl"
        self.save_model(model, model_path)

        evaluation = self.evaluate_model(
            model=model,
            protocol_df=protocol_df,
            out_dir=out_dir,
        )

        summary = {
            "status": "success",
            "n_samples": int(genomic_df_aligned.shape[0]),
            "n_features": int(genomic_df_aligned.shape[1]),
            "selected_algorithm": selected_algo,
            "selector": selector_result,
            "training_metrics": getattr(model, "training_metrics", {}),
            "evaluation": evaluation,
            "artifacts": {
                "protocol_matrix": str(protocol_matrix_path),
                "model_file": str(model_path),
                "evaluation_json": str(out_dir / "ml_protocol_evaluation.json"),
                "evaluation_tsv": str(out_dir / "ml_protocol_thresholds.tsv"),
                "sample_predictions_tsv": str(out_dir / "ml_protocol_sample_predictions.tsv"),
            },
        }

        with open(out_dir / "ml_protocol_results.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=_json_default)

        logger.info(
            "ML protocol complete | samples=%d | features=%d | algorithm=%s | out=%s",
            genomic_df_aligned.shape[0],
            genomic_df_aligned.shape[1],
            selected_algo,
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
    ) -> tuple[pd.DataFrame, pd.Series]:
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
    # train.py-style matrix construction
    # ------------------------------------------------------------------
    def build_protocol_df(
        self,
        genomic_df: pd.DataFrame,
        labels: pd.Series,
    ) -> pd.DataFrame:
        """
        Build the same logical structure expected by train.py / tester.py:

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
        Derived from train.py:
        remove feature columns where empty fraction > thr.

        The first two columns (sample_id, label) are preserved.
        """
        if protocol_df.shape[1] < 3:
            return protocol_df

        if thr >= 1.0:
            return protocol_df

        empty_sym = str(empty_symbol).strip()
        feature_cols = list(protocol_df.columns[2:])
        keep_cols: List[str] = []
        removed: List[tuple[str, float]] = []

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
    # model_selector.py integration
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

        This mirrors the intent of model_selector without changing the original
        dataframe used for model training.
        """
        encoded_cols: List[np.ndarray] = []

        for col in genomic_df.columns:
            s = genomic_df[col].copy()

            # preserve missing-like as a shared token so selector sees them consistently
            s = s.where(~s.isna(), "__MISSING__")
            s = s.astype(str).str.strip()
            s = s.replace({"": "__MISSING__", "nan": "__MISSING__", "NaN": "__MISSING__", "nd": "__MISSING__"})

            cat = pd.Categorical(s)
            codes = cat.codes.astype(float)

            # categorical codes are non-negative here
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
                "reason": f"selector_failed: {exc}",
                "probe_scores": {},
            }

        if "recommendation" not in result:
            result["recommendation"] = "RF"

        logger.info("ML selector recommendation: %s", result["recommendation"])
        logger.info("ML selector probe scores | %s",
            result.get("probe_scores", {})
        )
        return result

    def resolve_algorithm(
        self,
        selector_recommendation: str,
        requested_algorithm: Optional[str] = None,
    ) -> str:
        """
        Resolve final algorithm to one supported by NeuralNetwork.py.
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
            "MLP_small": "MLP",
            "MLP": "MLP",
            "LinearSVC": "SVC",
            "SVC_RBF": "SVC",
            "SVC": "SVC",
            "SCV": "SVC",
            "MBCS": "MBCS",
            "DNL": "DNL",
            # unsupported / optional recommendations fallback safely
            "KNN": "RF",
            "NBayes": "RF",
            "XGBoost": "RF",
        }

        return mapping.get(rec, "RF")

    # ------------------------------------------------------------------
    # NeuralNetwork.py integration
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

        # preserve the train.py RF behavior
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
        Train a model directly from the DataLoader dataframe.

        This follows the train.py logic:
        - plain marker style
        - no one-hot here for plain mode
        - feature titles are original dataframe columns
        """
        NN = self.import_nn()

        feature_titles = genomic_df.columns.astype(str).tolist()

        # train.py passes categorical/raw values into the NN model in plain mode
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
    # tester.py-style evaluation
    # ------------------------------------------------------------------
    def _feature_overlap(
        self,
        model: Any,
        protocol_df: pd.DataFrame,
    ) -> tuple[List[str], float]:
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
        """
        Call model.identify(markers) and normalize the output.
        """
        result = model.identify(marker_dict)

        if not isinstance(result, dict):
            raise ValueError("model.identify() returned a non-dict result")

        predictions = result.get("predictions", [])
        if not isinstance(predictions, list):
            predictions = []

        norm_predictions: List[tuple[str, float]] = []
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
        tester.py-style per-sample identification at a fixed sensitivity threshold.
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
            all_predictions = pred["predictions"]
            identified = [lab for lab, prob in all_predictions if prob >= sensitivity]

            top_label = ""
            top_prob = 0.0
            if all_predictions:
                top_label, top_prob = max(all_predictions, key=lambda x: x[1])

            records.append(
                {
                    "sample_id": sample_id,
                    "true_label": true_label,
                    "identified_labels": identified,
                    "top_label": str(top_label),
                    "top_probability": float(top_prob),
                    "n_predictions": len(all_predictions),
                    "matched": true_label in identified,
                }
            )

        return records

    def summarize_thresholds(
        self,
        thresholds: List[float],
        per_threshold_records: Dict[float, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Build tester-style threshold summary.

        Metrics here are intentionally simple and transparent:
        - tp: true label present among identified labels
        - fp: identified labels exist but true label absent
        - fn: no identified labels above threshold
        """
        rows: List[Dict[str, Any]] = []

        for thr in thresholds:
            records = per_threshold_records[thr]

            tp = 0
            fp = 0
            fn = 0

            for rec in records:
                identified = rec["identified_labels"]
                if not identified:
                    fn += 1
                elif rec["matched"]:
                    tp += 1
                else:
                    fp += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            rows.append(
                {
                    "sensitivity": float(thr),
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                }
            )

        best = max(rows, key=lambda x: x["f1"]) if rows else {}
        return {
            "thresholds": rows,
            "best_threshold": best,
        }

    def evaluate_model(
        self,
        model: Any,
        protocol_df: pd.DataFrame,
        out_dir: Path,
    ) -> Dict[str, Any]:
        """
        Run the tester stage in memory and write summary artifacts.
        """
        min_sensitivity = _safe_float(
            getattr(self.config, "ml_min_sensitivity", 0.5) if self.config is not None else 0.5,
            0.5,
        )
        max_sensitivity = _safe_float(
            getattr(self.config, "ml_max_sensitivity", 1.0) if self.config is not None else 1.0,
            1.0,
        )
        step_sensitivity = _safe_float(
            getattr(self.config, "ml_step_sensitivity", 0.1) if self.config is not None else 0.1,
            0.1,
        )

        if step_sensitivity <= 0:
            raise ValueError("ml_step_sensitivity must be > 0")
        if min_sensitivity > max_sensitivity:
            raise ValueError("ml_min_sensitivity must be <= ml_max_sensitivity")

        used_markers, coverage = self._feature_overlap(model, protocol_df)

        thresholds: List[float] = []
        t = min_sensitivity
        eps = step_sensitivity / 10.0
        while t <= max_sensitivity + eps:
            thresholds.append(round(float(t), 6))
            t += step_sensitivity

        per_threshold_records: Dict[float, List[Dict[str, Any]]] = {}
        for thr in thresholds:
            per_threshold_records[thr] = self.identify_records(
                model=model,
                protocol_df=protocol_df,
                used_markers=used_markers,
                sensitivity=thr,
            )

        summary = self.summarize_thresholds(
            thresholds=thresholds,
            per_threshold_records=per_threshold_records,
        )

        # save threshold table
        threshold_df = pd.DataFrame(summary["thresholds"])
        threshold_df.to_csv(out_dir / "ml_protocol_thresholds.tsv", sep="\t", index=False)

        # save best-threshold sample predictions
        best_thr = summary.get("best_threshold", {}).get("sensitivity", thresholds[0] if thresholds else 0.5)
        best_records = per_threshold_records[float(best_thr)]
        pd.DataFrame(best_records).to_csv(
            out_dir / "ml_protocol_sample_predictions.tsv",
            sep="\t",
            index=False,
        )

        evaluation_payload = {
            "feature_overlap_coverage": float(coverage),
            "used_markers_count": int(len(used_markers)),
            "threshold_summary": summary,
        }

        with open(out_dir / "ml_protocol_evaluation.json", "w", encoding="utf-8") as fh:
            json.dump(evaluation_payload, fh, indent=2, default=_json_default)

        logger.info(
            "ML protocol evaluation complete | used_markers=%d | coverage=%.3f | best_threshold=%s",
            len(used_markers),
            coverage,
            summary.get("best_threshold", {}).get("sensitivity", "NA"),
        )

        return evaluation_payload
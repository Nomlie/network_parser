#!/usr/bin/env python3
# network_parser/two_level_protocol.py
"""
Two-level NetworkParser protocol
================================

Purpose
-------
Train and apply a hierarchical genomic classifier:

    Level 1: strain / lineage / group placement
    Level 2: drug-resistance phenotype or resistance-profile prediction

The protocol keeps the NetworkParser architecture explicit:

    input -> DataLoader/preprocessing -> RF-FDR feature filtering -> level-1 model
                                              -> level-specific RF-FDR filtering -> level-2 models
                                              -> optional global level-2 fallback model

Important
---------
RF-FDR is used here as a central feature-selection stage. It is not used as the
post-tree confidence layer. Decision-tree interpretability can still be run
elsewhere on the filtered matrices when required.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.multitest import multipletests

try:
    from network_parser.config import NetworkParserConfig
    from network_parser.data_loader import DataLoader
    from network_parser.ml_protocol import MLProtocolRunner
    from network_parser.network_parser import normalize_labels
except Exception:  # pragma: no cover - supports running from source tree
    from config import NetworkParserConfig  # type: ignore
    from data_loader import DataLoader  # type: ignore
    from ml_protocol import MLProtocolRunner  # type: ignore
    from network_parser import normalize_labels  # type: ignore

from network_parser.feature_selection import rf_fdr_feature_selection

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)

def load_artifact_filtered_binary_matrix(
    artifact_root: Path,
    fallback_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load the DataLoader artifact-filtered binary matrix for downstream modeling.

    Important:
    - *_filtered.tsv is the filtered marker / annotation table.
    - *_binary.tsv is the sample × marker binary matrix.
    - RF-FDR must consume the artifact-filtered binary matrix when available.
    """

    artifact_root = Path(artifact_root)

    if fallback_matrix is None or fallback_matrix.empty:
        raise ValueError("Fallback matrix is empty; cannot validate artifact-filtered matrix.")

    fallback_index = pd.Index(
        fallback_matrix.index.astype(str).map(normalize_sample_id)
    )

    candidate_paths = sorted(artifact_root.rglob("*_binary.tsv"))

    if not candidate_paths:
        logger.warning(
            "No artifact-filtered binary matrix was found under %s. "
            "Using DataLoader returned matrix.",
            artifact_root,
        )
        return fallback_matrix.copy()

    valid_candidates = []

    for path in candidate_paths:
        try:
            candidate = pd.read_csv(path, sep="\t", index_col=0)
        except Exception as exc:
            logger.warning(
                "Could not read candidate artifact binary matrix %s: %s",
                path,
                exc,
            )
            continue

        if candidate.empty:
            logger.warning(
                "Skipping empty candidate artifact binary matrix: %s",
                path,
            )
            continue

        candidate.index = candidate.index.astype(str).map(normalize_sample_id)

        # Remove reference/control rows if present. They are useful in FASTA/TSV
        # artifacts but should not enter supervised model training.
        drop_rows = [
            idx for idx in candidate.index
            if str(idx).strip().upper() in {"REF", "REFERENCE"}
        ]
        if drop_rows:
            candidate = candidate.drop(index=drop_rows, errors="ignore")

        overlap = candidate.index.intersection(fallback_index)

        if overlap.empty:
            logger.warning(
                "Candidate artifact binary matrix %s has no sample-ID overlap "
                "with the DataLoader returned matrix. Skipping.",
                path,
            )
            continue

        valid_candidates.append(
            {
                "path": path,
                "matrix": candidate,
                "n_overlap": len(overlap),
                "n_features": candidate.shape[1],
            }
        )

    if not valid_candidates:
        logger.warning(
            "No valid artifact-filtered binary matrix could be aligned. "
            "Using DataLoader returned matrix."
        )
        return fallback_matrix.copy()

    # Prefer the candidate with the strongest sample overlap.
    # If tied, prefer the one with fewer features, because this should represent
    # the artifact-filtered matrix after structural/redundancy filtering.
    best = sorted(
        valid_candidates,
        key=lambda item: (-item["n_overlap"], item["n_features"]),
    )[0]

    X_artifact = best["matrix"].copy()

    logger.info(
        "Using artifact-filtered binary matrix for downstream modeling: %s | "
        "samples=%d | features=%d",
        best["path"],
        int(X_artifact.shape[0]),
        int(X_artifact.shape[1]),
    )

    return X_artifact
def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def load_config(config_path: Optional[str]) -> NetworkParserConfig:
    config = NetworkParserConfig()
    if config_path is None:
        config.__post_init__()
        return config

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        overrides = json.load(handle)

    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning("Ignoring unknown config key: %s", key)

    config.__post_init__()
    return config


def normalize_sample_id(value: Any) -> str:
    sample = str(value).strip()
    sample = sample.replace(".vcf.gz", "").replace(".vcf", "")
    if sample.endswith(".gz"):
        sample = sample[:-3]
    return sample


def align_matrix_and_label(
    X: pd.DataFrame,
    meta: pd.DataFrame,
    label_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    if label_column not in meta.columns:
        raise ValueError(f"Label column '{label_column}' not found in metadata.")

    X = X.copy()
    X.index = X.index.astype(str).map(normalize_sample_id)

    y = normalize_labels(meta[label_column], drop_missing=True, lowercase=False)
    y.index = y.index.astype(str).map(normalize_sample_id)

    common = X.index.intersection(y.index)
    if common.empty:
        raise ValueError(
            f"No overlapping sample IDs between genomic matrix and metadata for label '{label_column}'."
        )

    X_aligned = X.loc[common].copy()
    y_aligned = y.loc[common].copy()

    logger.info(
        "Aligned label '%s' | samples=%d | features=%d | classes=%d",
        label_column,
        int(X_aligned.shape[0]),
        int(X_aligned.shape[1]),
        int(y_aligned.nunique(dropna=True)),
    )
    return X_aligned, y_aligned


def align_two_labels(
    X: pd.DataFrame,
    meta: pd.DataFrame,
    level1_label: str,
    level2_label: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    X1, y1 = align_matrix_and_label(X, meta, level1_label)

    y2 = normalize_labels(meta[level2_label], drop_missing=True, lowercase=False)
    y2.index = y2.index.astype(str).map(normalize_sample_id)

    common = X1.index.intersection(y1.index).intersection(y2.index)
    if common.empty:
        raise ValueError("No samples have both level-1 and level-2 labels after alignment.")

    X_final = X1.loc[common].copy()
    y1_final = y1.loc[common].copy()
    y2_final = y2.loc[common].copy()

    logger.info(
        "Aligned two-level supervision | samples=%d | features=%d | level1_classes=%d | level2_classes=%d",
        int(X_final.shape[0]),
        int(X_final.shape[1]),
        int(y1_final.nunique(dropna=True)),
        int(y2_final.nunique(dropna=True)),
    )
    return X_final, y1_final, y2_final


def align_prediction_matrix(X_new: pd.DataFrame, training_features: List[str]) -> pd.DataFrame:
    X_new = X_new.copy()
    X_new.index = X_new.index.astype(str).map(normalize_sample_id)

    for feature in training_features:
        if feature not in X_new.columns:
            X_new[feature] = 0

    return X_new.loc[:, training_features].copy()


# -----------------------------------------------------------------------------
# Model fitting / prediction helpers
# -----------------------------------------------------------------------------

def run_ml_model(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    config: NetworkParserConfig,
    algorithm: Optional[str],
) -> Dict[str, Any]:
    """Run the existing ML protocol and return its summary payload."""
    ensure_dir(output_dir)
    runner = MLProtocolRunner(config=config)
    return runner.run(
        genomic_df=X,
        labels=y,
        output_dir=str(output_dir),
        algorithm=algorithm if algorithm is not None else getattr(config, "ml_algorithm", "auto"),
    )


def train_fallback_rf_model(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    config: NetworkParserConfig,
    model_name: str,
) -> Dict[str, Any]:
    """
    Small fallback used only if the external ML protocol fails.
    It keeps the two-level protocol runnable while recording the failure clearly.
    """
    ensure_dir(output_dir)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y.astype(str))

    model = RandomForestClassifier(
        n_estimators=int(getattr(config, "rf_selector_n_estimators", 300)),
        max_features=getattr(config, "rf_selector_max_features", "sqrt"),
        min_samples_leaf=int(getattr(config, "rf_selector_min_samples_leaf", 1)),
        class_weight=getattr(config, "rf_selector_class_weight", "balanced"),
        random_state=int(getattr(config, "rf_selector_random_state", 42)),
        n_jobs=int(getattr(config, "n_jobs", -1)),
    )
    model.fit(X, y_encoded)

    payload = {"model": model, "label_encoder": encoder, "features": list(X.columns)}
    model_path = output_dir / f"{model_name}.pkl"
    with open(model_path, "wb") as handle:
        pickle.dump(payload, handle)

    summary = {
        "status": "success",
        "selected_algorithm": "RF_fallback",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "artifacts": {"model_file": str(model_path)},
    }
    write_json(summary, output_dir / f"{model_name}_summary.json")
    return summary


def train_model_safely(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    config: NetworkParserConfig,
    algorithm: Optional[str],
    model_name: str,
) -> Dict[str, Any]:
    """
    Train a model for the two-level protocol.

    Publication-safe behaviour:
    - by default, fail loudly if the requested ML protocol fails;
    - only use the RF fallback if config.allow_two_level_rf_fallback=True.
    """
    try:
        return run_ml_model(
            X=X,
            y=y,
            output_dir=output_dir,
            config=config,
            algorithm=algorithm,
        )

    except Exception as exc:
        if bool(getattr(config, "allow_two_level_rf_fallback", False)):
            logger.exception(
                "%s ML protocol failed; fitting explicitly enabled RF fallback model.",
                model_name,
            )
            fallback = train_fallback_rf_model(
                X=X,
                y=y,
                output_dir=output_dir,
                config=config,
                model_name=model_name,
            )
            fallback["ml_protocol_error"] = str(exc)
            fallback["fallback_enabled_by_config"] = True
            return fallback

        raise RuntimeError(
            f"{model_name}: ML protocol failed and RF fallback is disabled. "
            "Set allow_two_level_rf_fallback=True only for exploratory runs."
        ) from exc


def load_model_payload(model_path: str) -> Any:
    path = Path(model_path)
    with open(path, "rb") as handle:
        return pickle.load(handle)


def predict_from_model_payload(model_payload: Any, X: pd.DataFrame) -> List[str]:
    """
    Predict from either an MLProtocol model object or the fallback payload.
    """
    if isinstance(model_payload, dict) and "model" in model_payload and "label_encoder" in model_payload:
        model = model_payload["model"]
        encoder = model_payload["label_encoder"]
        raw = model.predict(X)
        return [str(v) for v in encoder.inverse_transform(raw)]

    raw = model_payload.predict(X)
    return [str(v) for v in raw]


def get_model_file(model_summary: Dict[str, Any]) -> Optional[str]:
    artifacts = model_summary.get("artifacts", {}) if isinstance(model_summary, dict) else {}
    for key in ("model_file", "model_path"):
        value = artifacts.get(key)
        if value:
            return str(value)
    return None


# -----------------------------------------------------------------------------
# Two-level protocol
# -----------------------------------------------------------------------------

class TwoLevelProtocol:
    """Train and apply the two-level strain-placement and resistance protocol."""

    def __init__(self, config: NetworkParserConfig):
        self.config = config
        self.loader = DataLoader(config=config, n_jobs=getattr(config, "n_jobs", -1))

    def train(
        self,
        genomic_path: str,
        meta_path: str,
        level1_label: str,
        level2_label: str,
        output_dir: str,
        ref_fasta: Optional[str] = None,
        algorithm: Optional[str] = None,
        train_global_level2: bool = True,
        min_level2_samples_per_group: Optional[int] = None,
    ) -> Dict[str, Any]:
        out = ensure_dir(Path(output_dir))
        matrices_dir = ensure_dir(out / "matrices")
        level1_dir = ensure_dir(out / "level1_strain_identity")
        level2_dir = ensure_dir(out / "level2_resistance_profile")

        logger.info("Two-level training: loading genomic matrix")
        X_raw_unfiltered = self.loader.load_genomic_matrix(
            file_path=genomic_path,
            output_dir=str(matrices_dir),
            ref_fasta=ref_fasta,
        )

        X_raw = load_artifact_filtered_binary_matrix(
            artifact_root=matrices_dir,
            fallback_matrix=X_raw_unfiltered,
        )

        logger.info("Two-level training: loading metadata")
        meta = self.loader.load_metadata(meta_path, output_dir=str(out))

        X, y_level1, y_level2 = align_two_labels(
            X=X_raw,
            meta=meta,
            level1_label=level1_label,
            level2_label=level2_label,
        )

        min_group_n = (
            int(min_level2_samples_per_group)
            if min_level2_samples_per_group is not None
            else int(getattr(self.config, "min_group_size", 2))
        )

        X.to_csv(out / "aligned_two_level_matrix.csv")
        pd.DataFrame(
            {
                "sample_id": X.index.astype(str),
                "level1_label": y_level1.astype(str).values,
                "level2_label": y_level2.astype(str).values,
            }
        ).to_csv(out / "aligned_two_level_labels.csv", index=False)

        # ------------------------------------------------------------------
        # Level 1: strain / lineage / group placement
        # ------------------------------------------------------------------
        level1_filter = rf_fdr_feature_selection(
            X=X,
            y=y_level1,
            output_dir=level1_dir / "rf_fdr_filter",
            config=self.config,
            stage_name="level1_strain_identity",
        )
        X_level1 = level1_filter["filtered_matrix"]

        level1_model = train_model_safely(
            X=X_level1,
            y=y_level1.loc[X_level1.index],
            output_dir=level1_dir / "model",
            config=self.config,
            algorithm=algorithm,
            model_name="level1_strain_identity_model",
        )

        # ------------------------------------------------------------------
        # Level 2 global fallback: resistance prediction across all samples
        # ------------------------------------------------------------------
        global_level2_payload: Dict[str, Any] = {"status": "skipped"}
        if train_global_level2 and y_level2.nunique(dropna=True) >= 2:
            global_dir = ensure_dir(level2_dir / "global_fallback")
            global_filter = rf_fdr_feature_selection(
                X=X,
                y=y_level2,
                output_dir=global_dir / "rf_fdr_filter",
                config=self.config,
                stage_name="level2_global_resistance_profile",
            )
            X_global = global_filter["filtered_matrix"]
            global_model = train_model_safely(
                X=X_global,
                y=y_level2.loc[X_global.index],
                output_dir=global_dir / "model",
                config=self.config,
                algorithm=algorithm,
                model_name="level2_global_resistance_model",
            )
            global_level2_payload = {
                "status": "success",
                "filter": global_filter["summary"],
                "model": global_model,
                "features": list(X_global.columns),
                "model_file": get_model_file(global_model),
            }

        # ------------------------------------------------------------------
        # Level 2 per level-1 group: resistance prediction within placement
        # ------------------------------------------------------------------
        subgroup_payload: Dict[str, Any] = {}
        for group_value in sorted(y_level1.astype(str).unique()):
            group_mask = y_level1.astype(str) == str(group_value)
            group_samples = y_level1.index[group_mask]
            X_group = X.loc[group_samples].copy()
            y2_group = y_level2.loc[group_samples].copy()

            safe_group_name = str(group_value).replace("/", "_").replace(" ", "_")
            group_dir = ensure_dir(level2_dir / "by_level1_group" / safe_group_name)

            group_summary: Dict[str, Any] = {
                "level1_group": str(group_value),
                "n_samples": int(X_group.shape[0]),
                "n_level2_classes": int(y2_group.nunique(dropna=True)),
            }

            if X_group.shape[0] < min_group_n or y2_group.nunique(dropna=True) < 2:
                group_summary.update(
                    {
                        "status": "skipped",
                        "reason": "insufficient_samples_or_single_resistance_class",
                    }
                )
                write_json(group_summary, group_dir / "group_summary.json")
                subgroup_payload[str(group_value)] = group_summary
                continue

            group_filter = rf_fdr_feature_selection(
                X=X_group,
                y=y2_group,
                output_dir=group_dir / "rf_fdr_filter",
                config=self.config,
                stage_name=f"level2_resistance_profile__{safe_group_name}",
            )
            X_group_filtered = group_filter["filtered_matrix"]
            group_model = train_model_safely(
                X=X_group_filtered,
                y=y2_group.loc[X_group_filtered.index],
                output_dir=group_dir / "model",
                config=self.config,
                algorithm=algorithm,
                model_name="level2_resistance_model",
            )

            group_summary.update(
                {
                    "status": "success",
                    "filter": group_filter["summary"],
                    "model": group_model,
                    "features": list(X_group_filtered.columns),
                    "model_file": get_model_file(group_model),
                }
            )
            write_json(group_summary, group_dir / "group_summary.json")
            subgroup_payload[str(group_value)] = group_summary

        registry = {
            "protocol": "two_level_protocol",
            "level1": {
                "label_column": level1_label,
                "description": "strain / lineage / group placement",
                "filter": level1_filter["summary"],
                "model": level1_model,
                "features": list(X_level1.columns),
                "model_file": get_model_file(level1_model),
            },
            "level2": {
                "label_column": level2_label,
                "description": "drug-resistance phenotype or resistance-profile prediction",
                "global_fallback": global_level2_payload,
                "by_level1_group": subgroup_payload,
            },
            "training_matrix": {
                "aligned_matrix_csv": str(out / "aligned_two_level_matrix.csv"),
                "aligned_labels_csv": str(out / "aligned_two_level_labels.csv"),
            },
            "config": asdict(self.config) if is_dataclass(self.config) else vars(self.config),
        }

        registry_path = out / "two_level_model_registry.json"
        write_json(registry, registry_path)
        logger.info("Two-level training complete: %s", registry_path)
        return registry

    def predict(
        self,
        genomic_path: str,
        registry_path: str,
        output_dir: str,
        ref_fasta: Optional[str] = None,
    ) -> pd.DataFrame:
        out = ensure_dir(Path(output_dir))
        with open(registry_path, "r", encoding="utf-8") as handle:
            registry = json.load(handle)

        prediction_artifact_dir = out / "prediction_matrix_artifacts"

        X_new_raw_unfiltered = self.loader.load_genomic_matrix(
            file_path=genomic_path,
            output_dir=str(prediction_artifact_dir),
            ref_fasta=ref_fasta,
        )

        X_new_raw = load_artifact_filtered_binary_matrix(
            artifact_root=prediction_artifact_dir,
            fallback_matrix=X_new_raw_unfiltered,
        )

        level1_features = list(registry["level1"].get("features", []))
        level1_model_file = registry["level1"].get("model_file")
        if not level1_features or not level1_model_file:
            raise ValueError("Registry is missing the level-1 feature list or model file.")

        X_level1 = align_prediction_matrix(X_new_raw, level1_features)
        level1_payload = load_model_payload(level1_model_file)
        level1_predictions = predict_from_model_payload(level1_payload, X_level1)

        rows = []
        for sample_id, predicted_group in zip(X_level1.index.astype(str), level1_predictions):
            group_payload = registry["level2"].get("by_level1_group", {}).get(str(predicted_group), {})
            model_file = group_payload.get("model_file")
            features = group_payload.get("features", [])
            level2_source = "level1_group_specific"

            if not model_file or not features:
                fallback = registry["level2"].get("global_fallback", {})
                model_file = fallback.get("model_file")
                features = fallback.get("features", [])
                level2_source = "global_fallback"

            if model_file and features:
                X_level2 = align_prediction_matrix(X_new_raw.loc[[sample_id]], list(features))
                level2_payload = load_model_payload(model_file)
                level2_prediction = predict_from_model_payload(level2_payload, X_level2)[0]
            else:
                level2_prediction = "unavailable"
                level2_source = "unavailable"

            rows.append(
                {
                    "sample_id": sample_id,
                    "predicted_level1_identity": str(predicted_group),
                    "predicted_level2_resistance_profile": str(level2_prediction),
                    "level2_model_source": level2_source,
                }
            )

        predictions = pd.DataFrame(rows)
        predictions_path = out / "two_level_predictions.csv"
        predictions.to_csv(predictions_path, index=False)
        logger.info("Two-level prediction complete: %s", predictions_path)
        return predictions


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or apply a two-level NetworkParser model: strain identity first, resistance profile second.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train the two-level protocol.")
    train.add_argument("--genomic", required=True, help="Genomic matrix file or VCF directory.")
    train.add_argument("--meta", required=True, help="Metadata CSV/TSV.")
    train.add_argument("--level1_label", required=True, help="Metadata column for strain/lineage/group placement.")
    train.add_argument("--level2_label", required=True, help="Metadata column for drug-resistance phenotype/profile.")
    train.add_argument("--output_dir", required=True, help="Output directory.")
    train.add_argument("--config", default=None, help="Optional JSON config override file.")
    train.add_argument("--ref_fasta", default=None, help="Optional reference FASTA for VCF parsing context.")
    train.add_argument("--algorithm", default=None, help="Optional ML algorithm override passed to MLProtocolRunner.")
    train.add_argument("--no_global_level2", action="store_true", help="Disable the global level-2 fallback model.")
    train.add_argument("--min_level2_samples_per_group", type=int, default=None, help="Minimum samples needed to train group-specific level-2 models.")
    train.add_argument("--n_jobs", type=int, default=None, help="Runtime worker override.")

    predict = sub.add_parser("predict", help="Apply a trained two-level protocol to new strain/sample input.")
    predict.add_argument("--genomic", required=True, help="New genomic matrix file or VCF directory.")
    predict.add_argument("--registry", required=True, help="Path to two_level_model_registry.json from training.")
    predict.add_argument("--output_dir", required=True, help="Prediction output directory.")
    predict.add_argument("--config", default=None, help="Optional JSON config override file.")
    predict.add_argument("--ref_fasta", default=None, help="Optional reference FASTA for VCF parsing context.")
    predict.add_argument("--n_jobs", type=int, default=None, help="Runtime worker override.")

    return parser


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main(argv: Optional[List[str]] = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.n_jobs is not None:
        config.n_jobs = int(args.n_jobs)
    config.__post_init__()

    protocol = TwoLevelProtocol(config=config)

    if args.command == "train":
        protocol.train(
            genomic_path=args.genomic,
            meta_path=args.meta,
            level1_label=args.level1_label,
            level2_label=args.level2_label,
            output_dir=args.output_dir,
            ref_fasta=args.ref_fasta,
            algorithm=args.algorithm,
            train_global_level2=not bool(args.no_global_level2),
            min_level2_samples_per_group=args.min_level2_samples_per_group,
        )
        return 0

    if args.command == "predict":
        protocol.predict(
            genomic_path=args.genomic,
            registry_path=args.registry,
            output_dir=args.output_dir,
            ref_fasta=args.ref_fasta,
        )
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())

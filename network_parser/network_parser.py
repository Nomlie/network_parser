# network_parser/network_parser.py
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import pandas as pd
import numpy as np

from network_parser.config import NetworkParserConfig
from network_parser.data_loader import DataLoader
from network_parser.decision_tree_branch import DecisionTreeBranch
from network_parser.decision_tree_branch import StatisticalValidatorBranch
from network_parser.utils import save_json, ensure_dir, timestamp
from network_parser.ml_protocol import MLProtocolRunner


logger = logging.getLogger(__name__)


def normalize_labels(
    labels: pd.Series,
    drop_missing: bool = True,
    lowercase: bool = False,
) -> pd.Series:
    """
    Normalize phenotype / class labels to avoid artificial class inflation.
    """
    if not isinstance(labels, pd.Series):
        raise TypeError("labels must be a pandas Series")

    original_n = labels.shape[0]
    original_unique = labels.nunique(dropna=False)

    clean = labels.astype(str).str.strip()
    missing_tokens = {"", "-", "NA", "N/A", "None", "nan", "NaN"}
    clean = clean.replace(missing_tokens, pd.NA)
    clean = clean.str.replace("-", "_", regex=False)

    if lowercase:
        clean = clean.str.lower()

    n_missing = int(clean.isna().sum())
    if drop_missing:
        clean = clean[~clean.isna()]

    final_unique = int(clean.nunique(dropna=False))
    final_n = int(clean.shape[0])

    logger.info(
        "Label normalization: original_n=%d | final_n=%d | missing_removed=%d | unique_before=%d | unique_after=%d",
        int(original_n), final_n, n_missing, int(original_unique), final_unique
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

    Core (existing):
      DataLoader → (optional stats) → decision tree → (optional interactions) → outputs

    Added (optional downstream):
      ML prototype → model_selector → train (NeuralNetwork) → test (NeuralNetwork)
    """

    def __init__(self, config: NetworkParserConfig):
        logger.info("Initializing NetworkParser with config: %s", vars(config))
        self.config = config

        self.loader = DataLoader(config=config, n_jobs=config.n_jobs)
        self.validator = StatisticalValidatorBranch(config)
        self.tree_builder =DecisionTreeBranch(config)

    def _align_X_y(
        self,
        genomic_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        label_column: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if label_column not in meta_df.columns:
            raise ValueError(f"label_column '{label_column}' not found in metadata columns")

        labels = normalize_labels(meta_df[label_column], drop_missing=True, lowercase=False)

        def _normalize_sample_id(x: str) -> str:
            s = str(x).strip()
            s = s.replace(".vcf.gz", "").replace(".vcf", "")
            s = __import__("re").sub(r"_library[0-9]+$", "", s)
            return s

        genomic_df = genomic_df.copy()
        genomic_df.index = genomic_df.index.astype(str).map(_normalize_sample_id)

        labels.index = labels.index.astype(str).map(_normalize_sample_id)

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

        X = genomic_df.loc[common]
        y = labels.loc[common]

        return X, y

    def _write_supervised_matrix(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        out_dir: Path,
        filename: str = "ml_supervised_matrix.csv",
    ) -> Path:
        """
        Write: Sample | label | feature1 | feature2 | ...
        All feature values are written as strings to work with NeuralNetwork categorical encoders.
        """
        out_path = out_dir / filename
        df = X.copy()

        # Ensure string tokens (NeuralNetwork expects symbols/categorical tokens)
        df = df.astype(str)

        df.insert(0, "label", y.loc[df.index].astype(str))
        df.insert(0, "Sample", df.index.astype(str))

        df.to_csv(out_path, index=False)
        logger.info("ML prototype: wrote supervised matrix %s", str(out_path))
        return out_path

    def _map_selector_to_nn_algo(self, rec: str) -> str:
        """
        Map model_selector recommendation keys to NeuralNetwork-supported keys.
        NeuralNetwork.py supports: RF, MLP, LR, DT, SVC, KNN, XGBoost (if installed), DNL, MBCS.
        model_selector recommends: LR, LinearSVC, SVC, RF, DT, MLP, KNN, NBayes, XGBoost.
        """
        r = (rec or "").strip()

        if r == "SVC_RBF":
            return "SVC"
        if r == "MLP_small":
            return "MLP"
        if r == "LinearSVC":
            # NeuralNetwork doesn't have LinearSVC; SVC is the nearest within that API
            return "SVC"
        if r == "NBayes":
            # NeuralNetwork doesn't expose NB directly in your CLI script; safest fallback is LR
            return "LR"
        if r in {"LR", "RF", "DT", "MLP", "SVC", "KNN", "XGBoost"}:
            return r

        # Default conservative fallback
        return "RF"

    def run_pipeline(
        self,
        genomic_path: str,
        meta_path: Optional[str],
        label_column: str,
        known_markers_path: Optional[str],
        output_dir: str,
        validate_statistics: bool = False,
        validate_interactions: bool = False,
        ref_fasta: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute NetworkParser pipeline.
        """
        output_dir_path = Path(output_dir)
        ensure_dir(output_dir_path)

        logger.info("Stage 1: Loading genomic matrix")
        genomic_df = self.loader.load_genomic_matrix(
            file_path=genomic_path,
            output_dir=output_dir,
            ref_fasta=ref_fasta,
        )
        logger.info("Loaded genomic matrix with shape: %s", str(genomic_df.shape))

        meta_df = None
        if meta_path:
            logger.info("Loading metadata")
            meta_df = self.loader.load_metadata(meta_path, output_dir=output_dir)
            logger.info("Loaded metadata with shape: %s", str(meta_df.shape))

        known_markers = None
        if known_markers_path:
            logger.info("Loading known markers")
            known_markers = self.loader.load_known_markers(known_markers_path, output_dir=output_dir)
            logger.info("Loaded %d known markers", len(known_markers))

        if meta_df is None:
            raise ValueError("meta_path is required (labels are needed for discovery and ML protocol).")

        # Final supervised boundary shared by both branches
        X, y = self._align_X_y(genomic_df, meta_df, label_column=label_column)

        mode = getattr(self.config, "pipeline_mode", "decision_tree_only")
        logger.info("Pipeline mode resolved: %s", mode)

        validation_results: Dict[str, Any] = {}
        discovery_results: Dict[str, Any] = {}
        ml_results: Dict[str, Any] = {}

        run_tree = mode in {"decision_tree_only", "both"}
        run_ml = mode in {"ml_only", "both"}

        # -------------------------------------------------
        # Matrix-only stop point
        # -------------------------------------------------
        if mode == "matrix_only":
            logger.info("Pipeline stop point reached: matrix creation / alignment only")

            results = {
                "timestamp": timestamp(),
                "config": vars(self.config),
                "pipeline_mode": mode,
                "aligned_matrix_shape": {
                    "samples": int(X.shape[0]),
                    "features": int(X.shape[1]),
                },
                "discovery": discovery_results,
                "validation": validation_results,
                "ml_protocol": ml_results,
            }

            results_path = output_dir_path / f"networkparser_results_{timestamp()}.json"
            save_json(results, results_path)
            logger.info("Saved final results: %s", results_path)
            return results

        # -------------------------------------------------
        # Decision-tree branch
        # -------------------------------------------------
        if run_tree:
            if validate_statistics:
                logger.info("Stage 2: Statistical Validation (pre-tree, optional)")
                validation_results["features"] = self.validator.validate_features(
                    genomic_df=X,
                    meta_df=meta_df.loc[X.index],
                    label_column=label_column,
                    discovered_features=None,
                    output_dir=output_dir,
                )

            logger.info("Stage 3: Decision Tree Feature Discovery")
            discovery_results = self.tree_builder.run(
                data=X,
                labels=y,
                all_features=list(X.columns),
                output_dir=output_dir,
            )

            if validate_interactions:
                logger.info("Stage 4: Statistical Validation (interactions, optional)")
                validation_results["interactions"] = self.validator.validate_interactions(
                    genomic_df=X,
                    meta_df=meta_df.loc[X.index],
                    label_column=label_column,
                    interactions=discovery_results.get("epistatic_interactions", []),
                    output_dir=output_dir,
                )
        else:
            logger.info("Decision tree branch skipped by pipeline_mode=%s", mode)

        # -------------------------------------------------
        # ML protocol branch
        # -------------------------------------------------
        
        if run_ml:
            logger.info("Stage 5: ML protocol branch")

            ml_runner = MLProtocolRunner(config=self.config)
            ml_results = ml_runner.run(
                genomic_df=X,
                labels=y,
                output_dir=output_dir,
                algorithm=getattr(self.config, "ml_algorithm", "auto"),
            )
        else:
            logger.info("ML protocol branch skipped by pipeline_mode=%s", mode)
        
        results = {
            "timestamp": timestamp(),
            "config": vars(self.config),
            "pipeline_mode": mode,
            "discovery": discovery_results,
            "validation": validation_results,
            "ml_protocol": ml_results,
        }

        results_path = output_dir_path / f"networkparser_results_{timestamp()}.json"
        save_json(results, results_path)
        logger.info("Saved final results: %s", results_path)

        return results
    
def run_networkparser_analysis(
    genomic_path: str,
    meta_path: Optional[str],
    label_column: str,
    known_markers_path: Optional[str],
    output_dir: str,
    config: NetworkParserConfig,
    validate_statistics: bool = False,
    validate_interactions: bool = False,
    ref_fasta: Optional[str] = None,
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
    )
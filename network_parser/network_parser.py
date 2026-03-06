# network_parser/network_parser.py
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

from network_parser.config import NetworkParserConfig
from network_parser.data_loader import DataLoader
from network_parser.decision_tree_builder import EnhancedDecisionTreeBuilder
from network_parser.statistical_validation import StatisticalValidator
from network_parser.utils import save_json, ensure_dir, timestamp

logger = logging.getLogger(__name__)
def normalize_labels(
        labels: pd.Series,
        drop_missing: bool = True,
        lowercase: bool = False,
    ) -> pd.Series:
        """
        Normalize phenotype / class labels to avoid artificial class inflation.

        Steps:
        - strip whitespace
        - treat '-', '', 'NA', 'None', etc. as missing
        - standardize '_' and '-' to a single form
        - optional lowercase normalization
        - optionally drop missing labels

        Returns:
            Cleaned pd.Series aligned to original index (missing rows optionally removed).
        """

        if not isinstance(labels, pd.Series):
            raise TypeError("labels must be a pandas Series")

        original_n = labels.shape[0]
        original_unique = labels.nunique(dropna=False)

        # Convert to string safely
        clean = labels.astype(str).str.strip()

        # Normalize obvious missing tokens
        missing_tokens = {"", "-", "NA", "N/A", "None", "nan", "NaN"}
        clean = clean.replace(missing_tokens, pd.NA)

        # Standardize separators: unify '-' and '_' → single underscore
        clean = clean.str.replace("-", "_", regex=False)

        # Optional lowercase normalization
        if lowercase:
            clean = clean.str.lower()

        # Count missing before drop
        n_missing = clean.isna().sum()

        if drop_missing:
            clean = clean[~clean.isna()]

        final_unique = clean.nunique(dropna=False)
        final_n = clean.shape[0]

        logger.info(
            "Label normalization: original_n=%d | final_n=%d | missing_removed=%d | "
            "unique_before=%d | unique_after=%d",
            original_n,
            final_n,
            n_missing,
            original_unique,
            final_unique,
        )

        # Optional warning if large drop
        if n_missing > 0:
            logger.warning(
                "Label normalization removed %d features due to missing/invalid labels.",
                n_missing,
            )

        return clean

class NetworkParser:
    """
    Main orchestrator class for the NetworkParser pipeline.
    """

    def __init__(self, config: NetworkParserConfig):
        logger.info(f"Initializing NetworkParser with config: {vars(config)}")
        self.config = config

        # Data loading behavior (including folder-of-VCFs union-matrix mode) is driven via config
        self.loader = DataLoader(config=config, n_jobs=config.n_jobs)  # Uses enhanced DataLoader with bcftools + FASTA support

        self.validator = StatisticalValidator(config)
        self.tree_builder = EnhancedDecisionTreeBuilder(config)

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
        Execute full NetworkParser pipeline.
        """
        output_dir_path = Path(output_dir)
        ensure_dir(output_dir_path)

        logger.info("Stage 1: Loading genomic matrix")
        genomic_df = self.loader.load_genomic_matrix(
            file_path=genomic_path,
            output_dir=output_dir,
            ref_fasta=ref_fasta,
        )

        logger.info(f"Loaded genomic matrix with shape: {genomic_df.shape}")

        meta_df = None
        if meta_path:
            logger.info("Loading metadata")
            meta_df = self.loader.load_metadata(meta_path, output_dir=output_dir)
            logger.info(f"Loaded metadata with shape: {meta_df.shape}")

        known_markers = None
        if known_markers_path:
            logger.info("Loading known markers")
            known_markers = self.loader.load_known_markers(known_markers_path, output_dir=output_dir)
            logger.info(f"Loaded {len(known_markers)} known markers")

       # Stage 2: Decision tree feature discovery
        logger.info("Stage 2: Decision tree feature discovery")

        # --- Normalize labels (remove '-', unify formatting) ---
        raw_labels = meta_df[label_column]
        labels = normalize_labels(
            raw_labels,
            drop_missing=True,     # drops '-', '', NA-like tokens
            lowercase=False        # keep case unless you want strict normalization
        )

        # --- Align genomic matrix and labels to common samples AFTER label cleaning ---
        genomic_samples = genomic_df.index.astype(str)
        labels.index = labels.index.astype(str)

        common = genomic_samples.intersection(labels.index)

        # Helpful logs (counts only; no spam)
        logger.info(
            "Sample alignment: genomic=%d | meta=%d | labels_after_norm=%d | overlap=%d | genomic_only=%d | meta_only=%d",
            len(genomic_samples),
            meta_df.shape[0],
            len(labels),
            len(common),
            len(genomic_samples) - len(common),
            len(labels) - len(common),
        )

        if len(common) == 0:
            raise ValueError(
                "No overlapping sample IDs between genomic matrix and metadata after label normalization. "
                "Check sample naming / metadata index column."
            )

        # Subset to overlap
        genomic_df_aligned = genomic_df.loc[common]
        labels_aligned = labels.loc[common]

        # Optional: warn about tiny classes (useful for trees + stats)
        try:
            vc = labels_aligned.value_counts(dropna=False)
            n_small = int((vc < getattr(self.config, "min_group_size", 2)).sum())
            if n_small > 0:
                logger.warning(
                    "Labels: %d class(es) have fewer than min_group_size=%d samples (may destabilize inference).",
                    n_small,
                    getattr(self.config, "min_group_size", 2),
                )
        except Exception:
            pass

        # --- Now call discovery with cleaned, aligned labels ---
        discovery_results = self.tree_builder.discover_features(
            data=genomic_df_aligned,
            labels=labels_aligned,
            all_features=genomic_df_aligned.columns.tolist(),
            output_dir=output_dir,
        )

        # Stage 3: Statistical validation (optional)
        validation_results = {}
        if validate_statistics:
            logger.info("Stage 3: Statistical validation (features)")
            validation_results["features"] = self.validator.validate_features(
                genomic_df=genomic_df,
                meta_df=meta_df,
                label_column=label_column,
                discovered_features=discovery_results.get("ranked_features", []),
                output_dir=output_dir
            )

        if validate_interactions:
            logger.info("Stage 3: Statistical validation (interactions)")
            validation_results["interactions"] = self.validator.validate_interactions(
                genomic_df=genomic_df,
                meta_df=meta_df,
                label_column=label_column,
                interactions=discovery_results.get("epistatic_interactions", []),
                output_dir=output_dir
            )

        # Stage 4: Final synthesis / report writing
        logger.info("Stage 4: Writing final results")
        results = {
            "timestamp": timestamp(),
            "config": vars(self.config),
            "discovery": discovery_results,
            "validation": validation_results
        }

        results_path = output_dir_path / f"networkparser_results_{timestamp()}.json"
        save_json(results, results_path)
        logger.info(f"Saved final results: {results_path}")

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
    """
    Convenience wrapper to run the pipeline.
    """
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

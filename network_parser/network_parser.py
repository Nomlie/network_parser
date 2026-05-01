# network_parser/network_parser.py
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import pandas as pd

try:
    from network_parser.config import NetworkParserConfig
    from network_parser.data_loader import DataLoader
    from network_parser.decision_tree_branch import DecisionTreeBranch
    from network_parser.statistical_validation_branch import StatisticalValidatorBranch
    from network_parser.ml_protocol import MLProtocolRunner
    from network_parser.utils import save_json, ensure_dir, timestamp
except Exception:  # pragma: no cover
    from config import NetworkParserConfig  # type: ignore
    from data_loader import DataLoader  # type: ignore
    from decision_tree_branch import DecisionTreeBranch  # type: ignore
    from statistical_validation_branch import StatisticalValidatorBranch  # type: ignore
    from ml_protocol import MLProtocolRunner  # type: ignore
    from utils import save_json, ensure_dir, timestamp  # type: ignore


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
      4) Run ML protocol / model selector on filtered matrix
      5) Conditionally trigger decision-tree interpretability branch
      6) Optionally validate post-tree interactions
    """

    def __init__(self, config: NetworkParserConfig):
        logger.info("Initializing NetworkParser with config: %s", vars(config))
        self.config = config

        self.loader = DataLoader(config=config, n_jobs=config.n_jobs)
        self.validator = StatisticalValidatorBranch(config)
        self.tree_builder = DecisionTreeBranch(config)

    # ------------------------------------------------------------------
    # Alignment helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_sample_id(x: str) -> str:
        s = str(x).strip()
        s = s.replace(".vcf.gz", "").replace(".vcf", "")
        s = re.sub(r"_library[0-9]+$", "", s)
        return s

    def _align_X_y(
        self,
        genomic_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        label_column: str,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if label_column not in meta_df.columns:
            raise ValueError(f"label_column '{label_column}' not found in metadata columns")

        labels = normalize_labels(meta_df[label_column], drop_missing=True, lowercase=False)

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

        logger.info(
            "Aligned supervised matrix: samples=%d | features=%d",
            int(X.shape[0]),
            int(X.shape[1]),
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
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Shared statistical filtering stage used BEFORE model selection and BEFORE
        any downstream algorithm-specific branch.

        This stage intentionally performs:
          - per-feature association testing
          - multiple testing correction

        It does NOT perform tree bootstrap / stability analysis.
        """
        stats_dir = Path(output_dir) / "central_feature_filtering"
        stats_dir.mkdir(parents=True, exist_ok=True)

        if not enabled:
            logger.info(
                "Central feature filtering disabled. Passing through aligned matrix unchanged."
            )
            summary = {
                "status": "skipped",
                "input_features": int(X.shape[1]),
                "retained_features": int(X.shape[1]),
                "retention_fraction": 1.0,
                "filtered_matrix": str(stats_dir / "filtered_matrix.csv"),
            }
            X.astype(str).to_csv(stats_dir / "filtered_matrix.csv")
            return X.copy(), summary

        logger.info("Stage 3: Central feature filtering")
        logger.info(
            "Running shared association testing before model screening | samples=%d | features=%d",
            int(X.shape[0]),
            int(X.shape[1]),
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

        if not significant_features:
            logger.warning(
                "Central feature filtering retained no significant genomic features after correction. "
                "Proceeding with the aligned matrix to avoid an empty downstream design matrix."
            )
            X_filtered = X.copy()
            used_fallback = True
        else:
            significant_features = [f for f in significant_features if f in X.columns]
            X_filtered = X.loc[:, significant_features].copy()
            used_fallback = False

        filtered_matrix_path = stats_dir / "filtered_matrix.csv"
        X_filtered.astype(str).to_csv(filtered_matrix_path)

        summary = {
            "status": "success",
            "input_features": int(X.shape[1]),
            "tested_features": int(len(assoc)),
            "significant_features": int(len(significant_features)),
            "retained_features": int(X_filtered.shape[1]),
            "retention_fraction": float(X_filtered.shape[1] / max(1, X.shape[1])),
            "used_fallback_unfiltered_matrix": bool(used_fallback),
            "retained_feature_names": list(X_filtered.columns),
            "artifacts": {
                "filter_dir": str(stats_dir),
                "association_json": str(stats_dir / "chi_squared_results.json"),
                "multiple_testing_json": str(stats_dir / "multiple_testing_results.json"),
                "filtered_matrix": str(filtered_matrix_path),
            },
        }

        with open(stats_dir / "feature_filtering_summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        logger.info(
            "Central feature filtering complete | retained_features=%d / %d",
            int(X_filtered.shape[1]),
            int(X.shape[1]),
        )

        return X_filtered, summary

    # ------------------------------------------------------------------
    # Decision-tree trigger logic
    # ------------------------------------------------------------------
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
        - both:
            * run if explicitly selected and trigger_decision_tree_on_selected=True
            * run if DT is a candidate and trigger_decision_tree_if_candidate=True
            * user forcing ml_algorithm=DT always enables DT
        """
        if mode == "matrix_only":
            return False

        if mode == "decision_tree_only":
            return True

        if mode == "ml_only":
            return False

        branch_decision = ml_results.get("branch_decision", {}) if isinstance(ml_results, dict) else {}

        selected_algorithm = str(
            branch_decision.get("selected_algorithm", ml_results.get("selected_algorithm", ""))
        ).strip()

        selector_recommendation = str(
            branch_decision.get(
                "selector_recommendation",
                ml_results.get("selector", {}).get("recommendation", ""),
            )
        ).strip()

        candidate_algorithms = branch_decision.get("candidate_algorithms", [])
        if not isinstance(candidate_algorithms, list):
            candidate_algorithms = []

        configured_algorithm = str(getattr(self.config, "ml_algorithm", "auto")).strip()
        trigger_on_selected = bool(getattr(self.config, "trigger_decision_tree_on_selected", True))
        trigger_if_candidate = bool(getattr(self.config, "trigger_decision_tree_if_candidate", True))
        requires_match = bool(getattr(self.config, "decision_tree_requires_selector_match", False))

        if configured_algorithm == "DT":
            logger.info("Decision tree branch trigger: running because config.ml_algorithm=DT")
            return True

        if requires_match:
            run_dt = (selected_algorithm == "DT" or selector_recommendation == "DT")
            logger.info(
                "Decision tree branch trigger with strict selector match | selected=%s | recommended=%s | run=%s",
                selected_algorithm or "n/a",
                selector_recommendation or "n/a",
                run_dt,
            )
            return run_dt

        if trigger_on_selected and selected_algorithm == "DT":
            logger.info("Decision tree branch trigger: running because selected_algorithm=DT")
            return True

        if trigger_on_selected and selector_recommendation == "DT":
            logger.info("Decision tree branch trigger: running because selector recommendation=DT")
            return True

        if trigger_if_candidate and "DT" in [str(x).strip() for x in candidate_algorithms]:
            logger.info("Decision tree branch trigger: running because DT is in candidate_algorithms")
            return True

        logger.info(
            "Decision tree branch trigger: not activated | selected=%s | recommended=%s | candidates=%s",
            selected_algorithm or "n/a",
            selector_recommendation or "n/a",
            candidate_algorithms,
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
                    if "pair" in item and isinstance(item["pair"], (tuple, list)) and len(item["pair"]) >= 2:
                        f1, f2 = str(item["pair"][0]), str(item["pair"][1])
                        if f1 and f2:
                            pairs.append((f1, f2))
                        continue

                    if "features" in item and isinstance(item["features"], (tuple, list)) and len(item["features"]) >= 2:
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
    # Main pipeline
    # ------------------------------------------------------------------
    def run_pipeline(
        self,
        genomic_path: str,
        meta_path: Optional[str],
        label_column: str,
        known_markers_path: Optional[str],
        output_dir: str,
        validate_statistics: bool = False,
        validate_interactions: bool = False,
        ref_fasta: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute NetworkParser pipeline using the updated logic:

          load -> align -> central feature filtering -> ML protocol/model selector
               -> conditional DT -> optional interaction validation
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
            logger.info("Stage 2: Loading metadata")
            meta_df = self.loader.load_metadata(meta_path, output_dir=output_dir)
            logger.info("Loaded metadata with shape: %s", str(meta_df.shape))

        known_markers = None
        if known_markers_path:
            logger.info("Loading known markers")
            known_markers = self.loader.load_known_markers(
                known_markers_path,
                output_dir=output_dir,
            )
            logger.info("Loaded %d known markers", len(known_markers))

        if meta_df is None:
            raise ValueError(
                "meta_path is required (labels are needed for central filtering and model selection)."
            )

        if validate_statistics:
            logger.info(
                "Legacy flag --validate_statistics detected. Central filtering is now controlled "
                "primarily by config.run_central_feature_filtering."
            )

        X, y = self._align_X_y(genomic_df, meta_df, label_column=label_column)

        mode = getattr(self.config, "pipeline_mode", "decision_tree_only")
        logger.info("Pipeline mode resolved: %s", mode)

        validation_results: Dict[str, Any] = {}
        discovery_results: Dict[str, Any] = {}
        ml_results: Dict[str, Any] = {}

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
                "known_markers_loaded": int(len(known_markers)) if known_markers is not None else 0,
                "feature_filtering": {},
                "discovery": discovery_results,
                "validation": validation_results,
                "ml_protocol": ml_results,
            }

            results_path = output_dir_path / f"networkparser_results_{timestamp()}.json"
            save_json(results, results_path)
            logger.info("Saved final results: %s", results_path)
            return results

        X_filtered, feature_filter_summary = self._run_central_feature_filter(
            X=X,
            y=y,
            output_dir=output_dir,
            enabled=bool(getattr(self.config, "run_central_feature_filtering", True)),
        )

        run_ml = mode in {"ml_only", "both"}

        if run_ml:
            logger.info("Stage 4: ML protocol + model screening")
            ml_runner = MLProtocolRunner(config=self.config)
            ml_results = ml_runner.run(
                genomic_df=X_filtered,
                labels=y,
                output_dir=output_dir,
                algorithm=getattr(self.config, "ml_algorithm", "auto"),
            )
        else:
            logger.info("ML protocol branch skipped by pipeline_mode=%s", mode)

        run_tree = self._should_run_decision_tree(mode=mode, ml_results=ml_results)

        if run_tree:
            logger.info("Stage 5: Conditional decision-tree interpretability branch")

            discovery_results = self.tree_builder.run(
                data=X_filtered,
                labels=y,
                all_features=list(X_filtered.columns),
                output_dir=output_dir,
                prefiltered_input=True,
            )

            if validate_interactions:
                logger.info("Stage 6: Optional interaction validation")
                interaction_pairs = self._extract_interaction_pairs(
                    discovery_results.get("epistatic_interactions", [])
                )

                if interaction_pairs:
                    validation_results["interactions"] = self.validator.validate_interactions(
                        genomic_df=X_filtered,
                        meta_df=meta_df.loc[X_filtered.index],
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
        else:
            logger.info("Decision tree branch skipped under updated conditional logic.")

        results = {
            "timestamp": timestamp(),
            "config": vars(self.config),
            "pipeline_mode": mode,
            "aligned_matrix_shape": {
                "samples": int(X.shape[0]),
                "features": int(X.shape[1]),
            },
            "filtered_matrix_shape": {
                "samples": int(X_filtered.shape[0]),
                "features": int(X_filtered.shape[1]),
            },
            "known_markers_loaded": int(len(known_markers)) if known_markers is not None else 0,
            "feature_filtering": feature_filter_summary,
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
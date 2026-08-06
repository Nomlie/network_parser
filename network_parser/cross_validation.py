#!/usr/bin/env python3
# network_parser/cross_validation.py
"""
Leakage-aware repeated cross-validation for NetworkParser.

Purpose
-------
Run repeated stratified cross-validation for a single supervised metadata target
while keeping the most important statistical boundary intact:

    split first -> fit feature filtering/panel selection on the training fold only
                -> train model on the training fold only
                -> evaluate the held-out fold

This module is deliberately model-agnostic with respect to the active dataset.
It is intended as a validation wrapper around the existing NetworkParser
feature-filtering and ML protocol, not as a new discovery branch.

Outputs
-------
    cross_validation_summary.json
    cv_fold_metrics.tsv
    cv_predictions.tsv
    cv_feature_stability.tsv
    cv_by_class_metrics.tsv
    aggregate_performance/*

Notes
-----
VCF parsing/QC and initial matrix construction may still be performed once at
input load time for practical runtime reasons. Cohort-level structural filters
are relaxed during this load where possible; supervised statistical feature
filtering and panel selection are always fit inside each training fold.
"""

from __future__ import annotations

import copy
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

try:
    from network_parser.config import NetworkParserConfig
    from network_parser.data_loader import DataLoader
    from network_parser.ml_protocol import MLProtocolRunner
    from network_parser.feature_panel_selection import log_model_input_panel_decision
    from network_parser.model_evaluation import (
        assert_unique_normalized_ids,
        evaluate_predictions,
        score_dicts_to_frame,
    )
    from network_parser.network_parser import normalize_labels
    from network_parser.hierarchy_protocol import (
        run_configured_feature_filter,
        run_feature_panel_check_after_filter,
    )
    from network_parser.utils import normalize_sample_id
except ImportError:  # pragma: no cover - supports direct source-tree execution
    from config import NetworkParserConfig  # type: ignore
    from data_loader import DataLoader  # type: ignore
    from ml_protocol import MLProtocolRunner  # type: ignore
    from feature_panel_selection import log_model_input_panel_decision  # type: ignore
    from model_evaluation import (  # type: ignore
        assert_unique_normalized_ids,
        evaluate_predictions,
        score_dicts_to_frame,
    )
    from network_parser import normalize_labels  # type: ignore
    from hierarchy_protocol import (  # type: ignore
        run_configured_feature_filter,
        run_feature_panel_check_after_filter,
    )
    from utils import normalize_sample_id  # type: ignore

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
        handle.write("\n")


# -----------------------------------------------------------------------------
# Input preparation
# -----------------------------------------------------------------------------


def _cv_loader_config(config: NetworkParserConfig) -> NetworkParserConfig:
    """
    Return a CV input-loader config.

    Default behaviour preserves the same unsupervised structural matrix filters
    used during training. This avoids loading every singleton polymorphic site
    into repeated CV, which can cause unnecessary memory pressure.

    Supervised statistical filtering is still refit inside each training fold,
    preserving the main leakage-aware validation boundary.
    """
    local = copy.copy(config)

    relax = bool(getattr(config, "cv_relax_input_filters", False))

    if relax:
        # Strictest leakage-aware mode: build a very broad input matrix first.
        # This can be memory-heavy for large VCF cohorts.
        local.min_sample_presence = 1
        local.remove_invariant = False
        local.min_minor_count = 0
        local.matrices_min_count = 0

    return local


def _drop_invariant_from_training(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Remove columns that are invariant in the training fold only."""
    keep: List[str] = []
    for col in X_train.columns.astype(str):
        series = X_train[col]
        # Missingness is not a biological allele state. A feature is variable
        # only when at least two callable values occur in the training fold.
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if clean.nunique(dropna=True) > 1:
            keep.append(str(col))

    if not keep:
        raise ValueError(
            "No non-invariant genomic features remain in this training fold."
        )

    return X_train.loc[:, keep].copy(), X_test.reindex(columns=keep).copy(), keep


def _fit_fold_local_cohort_baselines(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    ancestral_allele: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Refit cohort-mode binary baselines using outer-training calls only."""
    if str(ancestral_allele).upper() == "Y":
        return X_train, X_test, {"features_flipped_to_train_mode": 0}

    train = X_train.copy()
    test = X_test.copy()
    flipped = 0
    for column in train.columns:
        observed = pd.to_numeric(train[column], errors="coerce").dropna()
        if observed.empty:
            continue
        counts = observed.value_counts()
        # Deterministic tie: retain the existing orientation.
        if float(counts.get(1.0, 0)) <= float(counts.get(0.0, 0)):
            continue
        train_values = pd.to_numeric(train[column], errors="coerce")
        test_values = pd.to_numeric(test[column], errors="coerce")
        train[column] = train_values.where(train_values.isna(), 1.0 - train_values)
        test[column] = test_values.where(test_values.isna(), 1.0 - test_values)
        flipped += 1
    return train, test, {"features_flipped_to_train_mode": int(flipped)}


def load_cv_matrix_and_labels(
    *,
    genomic_path: str,
    meta_path: str,
    label_column: str,
    output_dir: Path,
    config: NetworkParserConfig,
    ref_fasta: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any], Optional[pd.Series]]:
    """Load genomic input and one metadata target for repeated CV."""
    loader_config = _cv_loader_config(config)
    loader = DataLoader(config=loader_config, n_jobs=getattr(config, "n_jobs", -1))

    matrix_dir = _ensure_dir(output_dir / "cv_input_matrix_artifacts")
    X = loader.load_genomic_matrix(
        file_path=genomic_path,
        output_dir=str(matrix_dir),
        ref_fasta=ref_fasta,
    )
    if not isinstance(X, pd.DataFrame):
        raise TypeError("DataLoader did not return a pandas DataFrame for CV input.")

    meta = loader.load_metadata(meta_path, output_dir=str(output_dir / "metadata"))
    if label_column not in meta.columns:
        raise ValueError(f"label_column '{label_column}' not found in metadata columns")

    X = X.copy()
    X.index = X.index.astype(str).map(normalize_sample_id)
    assert_unique_normalized_ids(X.index.tolist(), context="CV genomic matrix")
    X.columns = X.columns.astype(str)

    labels = normalize_labels(meta[label_column], drop_missing=True, lowercase=False)
    labels.index = labels.index.astype(str).map(normalize_sample_id)
    assert_unique_normalized_ids(labels.index.tolist(), context="CV metadata labels")

    common = X.index.intersection(labels.index)
    if common.empty:
        raise ValueError(
            "No overlapping sample IDs between genomic matrix and metadata labels for CV."
        )

    X = X.loc[common].copy()
    y = labels.loc[common].astype(str).copy()

    groups: Optional[pd.Series] = None
    group_column = getattr(config, "cv_group_column", None)
    if group_column:
        group_column = str(group_column)
        if group_column not in meta.columns:
            raise ValueError(f"cv_group_column '{group_column}' not found in metadata")
        g = meta[group_column].copy()
        g.index = g.index.astype(str).map(normalize_sample_id)
        assert_unique_normalized_ids(g.index.tolist(), context="CV group labels")
        groups = g.loc[common].astype(str)

    summary = {
        "status": "loaded",
        "genomic_path": str(genomic_path),
        "cv_group_column": str(group_column) if group_column else None,
        "grouped_cv": bool(groups is not None),
        "meta_path": str(meta_path),
        "target_label_column": str(label_column),
        "samples_after_alignment": int(X.shape[0]),
        "features_after_relaxed_input_load": int(X.shape[1]),
        "n_classes": int(y.nunique(dropna=True)),
        "class_count_values_sorted": sorted(
            [int(v) for v in y.value_counts().tolist()]
        ),
        "input_load_note": (
            "Per-record input QC was preserved. Cohort-level structural filters were relaxed "
            "where possible so supervised feature filtering can be fit inside each CV training fold."
        ),
    }

    X.to_csv(output_dir / "cv_aligned_input_matrix.csv")
    pd.DataFrame({"sample_id": y.index.astype(str), "label": y.values}).to_csv(
        output_dir / "cv_aligned_labels.tsv",
        sep="\t",
        index=False,
    )
    if groups is not None:
        pd.DataFrame(
            {"sample_id": groups.index.astype(str), "group": groups.values}
        ).to_csv(
            output_dir / "cv_aligned_groups.tsv",
            sep="\t",
            index=False,
        )
    _write_json(summary, output_dir / "cv_input_summary.json")
    return X, y, summary, groups


# -----------------------------------------------------------------------------
# Prediction helpers
# -----------------------------------------------------------------------------


def _predict_one_model_row(
    model: Any, row: pd.Series, features: Sequence[str]
) -> Tuple[str, Optional[float], Dict[str, float]]:
    marker_dict: Dict[str, str] = {}
    for feature in features:
        value = row.get(feature, float("nan"))
        if pd.isna(value):
            raise ValueError(
                f"Fold prediction contains unresolved marker after train-fitted preprocessing: {feature}"
            )
        marker_dict[str(feature)] = f"{float(value):g}"

    if not hasattr(model, "identify"):
        raise TypeError(
            "Trained model does not expose identify(); cannot run NetworkParser-style fold prediction."
        )

    result = model.identify(marker_dict)
    predictions = result.get("predictions", []) if isinstance(result, dict) else []

    class_support: Dict[str, float] = {}
    for item in predictions:
        try:
            label = str(item[0])
            support = float(item[1])
        except Exception:
            continue
        class_support[label] = support

    if not class_support:
        return "unavailable", None, {}

    top_label = max(class_support.items(), key=lambda kv: kv[1])[0]
    top_support = float(class_support[top_label])
    return top_label, top_support, class_support


def _predict_fold(
    *,
    model: Any,
    X_test: pd.DataFrame,
    features: Sequence[str],
    y_test: pd.Series,
    repeat: int,
    fold: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    score_dicts: List[Dict[str, float]] = []
    sample_ids: List[str] = []

    X_local = X_test.reindex(columns=list(features)).copy()
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
    state_payload = getattr(model, "networkparser_missingness_state", None)
    if not isinstance(state_payload, dict):
        raise ValueError("Fold model is missing its train-fitted preprocessing state")
    state = FittedMissingnessState.from_dict(state_payload)
    X_local, _ = transform_with_missingness_state(
        X_local,
        state,
        apply_imputation=True,
        drop_high_missing_samples=False,
    )
    if X_local.isna().any().any():
        raise ValueError("Fold model preprocessing left unresolved NaN values")

    for sample_id, row in X_local.iterrows():
        pred_label, support, class_support = _predict_one_model_row(
            model, row, features
        )
        sid = normalize_sample_id(sample_id)
        sample_ids.append(sid)
        score_dicts.append(class_support)
        rows.append(
            {
                "prediction_id": f"r{repeat}_f{fold}_{sid}",
                "repeat": int(repeat),
                "fold": int(fold),
                "sample_id": sid,
                "true_label": str(y_test.loc[sample_id]),
                "predicted_label": str(pred_label),
                "top_support": float(support) if support is not None else np.nan,
                "class_support_json": json.dumps(class_support, sort_keys=True),
            }
        )

    pred_df = pd.DataFrame(rows)
    score_df = score_dicts_to_frame(sample_ids=sample_ids, score_dicts=score_dicts)
    # Use prediction_id as index for aggregate evaluation so repeated samples
    # across repeated CV are not deduplicated by sample id.
    if not pred_df.empty:
        score_df.index = pred_df["prediction_id"].astype(str).values
    return pred_df, score_df


# -----------------------------------------------------------------------------
# Fold training/evaluation
# -----------------------------------------------------------------------------


def _resolve_fold_algorithm(
    *,
    runner: MLProtocolRunner,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    requested_algorithm: Optional[str],
    groups: Optional[pd.Series] = None,
) -> Tuple[str, Dict[str, Any]]:
    requested = requested_algorithm
    if requested is None or str(requested).lower() == "auto":
        run_selector = (
            bool(getattr(runner.config, "run_model_selector", True))
            if runner.config is not None
            else True
        )
        if run_selector:
            selector = runner.select_model(X_train, y_train, groups=groups)
            selector = runner._normalize_selector_output(selector)
            selected = runner.resolve_algorithm(
                selector_recommendation=selector.get("recommendation", "RF"),
                requested_algorithm="auto",
            )
            return selected, selector
        selected = runner.resolve_algorithm("RF", requested_algorithm="RF")
        return selected, {"selector_enabled": False, "recommendation": selected}

    selected = runner.resolve_algorithm("RF", requested_algorithm=str(requested))
    return selected, {
        "selector_enabled": False,
        "requested_algorithm": str(requested),
        "recommendation": selected,
    }


def _run_one_fold(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    repeat: int,
    fold: int,
    output_dir: Path,
    config: NetworkParserConfig,
    algorithm: Optional[str],
    groups: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Strict nested fold: all supervised preprocessing is fit on train only.

    Order: invariant filter → missingness limits/imputer → statistical filter →
    panel selection → model selection → train → evaluate held-out test.
    """
    fold_dir = _ensure_dir(output_dir / f"repeat_{repeat:02d}" / f"fold_{fold:02d}")

    try:
        from network_parser.matrix_contract import (
            MissingnessPolicy,
            fit_missingness_state,
            transform_with_missingness_state,
        )
    except ImportError:  # pragma: no cover
        from matrix_contract import (  # type: ignore
            MissingnessPolicy,
            fit_missingness_state,
            transform_with_missingness_state,
        )

    X_train = X.loc[list(train_ids)].copy()
    X_test = X.loc[list(test_ids)].copy()
    y_train = y.loc[list(train_ids)].astype(str).copy()
    y_test = y.loc[list(test_ids)].astype(str).copy()
    groups_train: Optional[pd.Series] = None
    groups_test: Optional[pd.Series] = None
    fold_config = copy.copy(config)
    if groups is not None:
        groups_train = groups.loc[list(train_ids)].astype(str).copy()
        groups_test = groups.loc[list(test_ids)].astype(str).copy()
        setattr(fold_config, "_rf_fdr_groups", groups_train)
        setattr(fold_config, "_feature_panel_groups", groups_train)

    X_train, X_test, fold_baseline_audit = _fit_fold_local_cohort_baselines(
        X_train,
        X_test,
        ancestral_allele=str(getattr(fold_config, "ancestral_allele", "Y")),
    )

    preprocessing_mode = str(getattr(config, "cv_preprocessing_mode", "strict"))
    if preprocessing_mode == "exploratory_transductive":
        logger.warning(
            "CV fold using exploratory_transductive preprocessing mode — "
            "not valid as held-out publication validation | repeat=%d | fold=%d",
            repeat,
            fold,
        )

    X_train, X_test, train_variant_features = _drop_invariant_from_training(
        X_train, X_test
    )

    # Missingness limits + imputer parameters from TRAIN only.
    policy = MissingnessPolicy.from_config(fold_config)
    X_train_m, miss_state, miss_audit = fit_missingness_state(X_train, policy=policy)
    y_train = y_train.loc[X_train_m.index].copy()
    X_train_imp, train_imp_audit = transform_with_missingness_state(
        X_train_m, miss_state, apply_imputation=True, drop_high_missing_samples=False
    )
    # Test: align + impute with train fill values; do NOT drop test samples for
    # evaluation completeness — high-missing test samples remain and may yield
    # weak predictions that are scored honestly.
    X_test_aligned, test_imp_audit = transform_with_missingness_state(
        X_test,
        miss_state,
        apply_imputation=False,
        drop_high_missing_samples=False,
    )
    y_test = y_test.loc[X_test_aligned.index.intersection(y_test.index)].copy()
    X_test_aligned = X_test_aligned.loc[y_test.index].copy()

    miss_state.save_json(fold_dir / "missingness_state.json")

    filter_result = run_configured_feature_filter(
        X=X_train_imp,
        y=y_train,
        output_base_dir=fold_dir / "feature_filtering",
        config=fold_config,
        stage_name=f"cv_repeat_{repeat}_fold_{fold}",
    )

    panel_status = "not_requested"
    if bool(getattr(config, "run_feature_panel_separability_check", True)):
        try:
            # Statistical filtering used the outer-train imputed matrix, but
            # panel-size CV must receive raw selected features so each inner
            # training split fits its own missingness filter and imputer.
            panel_filter_result = dict(filter_result)
            central_features = [
                str(value)
                for value in filter_result.get(
                    "retained_features", filter_result["filtered_matrix"].columns
                )
            ]
            panel_filter_result["filtered_matrix"] = X_train_m.reindex(
                columns=central_features
            ).copy()
            filter_result = run_feature_panel_check_after_filter(
                filter_result=panel_filter_result,
                y=y_train,
                output_base_dir=fold_dir / "feature_panel",
                config=fold_config,
                stage_name=f"cv_repeat_{repeat}_fold_{fold}",
            )
            panel_status = str(
                (filter_result.get("feature_panel_separability") or {}).get(
                    "status", "success"
                )
            )
        except Exception as exc:
            # Strict nested CV: panel failure fails the fold (does not silently continue).
            if preprocessing_mode == "strict" or bool(
                getattr(config, "feature_panel_strict_failure", True)
            ):
                raise RuntimeError(
                    f"CV panel selection failed in strict mode | repeat={repeat} | fold={fold}: {exc}"
                ) from exc
            panel_status = f"skipped_after_failure: {exc}"
            logger.warning(
                "CV panel check failed; using centrally filtered matrix for this fold | repeat=%d | fold=%d | error=%s",
                repeat,
                fold,
                exc,
            )

    selected_features = [
        str(f)
        for f in filter_result.get(
            "retained_features", list(filter_result["filtered_matrix"].columns)
        )
    ]
    selected_features = [f for f in selected_features if f in X_train_m.columns]
    if not selected_features:
        raise ValueError("No selected features available after fold filtering.")
    # Preserve raw NaNs for inner model/threshold selection. The deployment
    # model fits its preprocessor on this outer-training subset only.
    X_train_selected = X_train_m.reindex(columns=selected_features).copy()
    # Held-out features: NaN for missing columns (never structural baseline).
    X_test_selected = X_test_aligned.reindex(columns=selected_features).copy()

    panel_summary = filter_result.get("feature_panel_separability")
    if not isinstance(panel_summary, dict):
        panel_summary = {
            "status": "skipped",
            "selection_reason": "feature_panel_check_disabled_by_config",
            "selected_features": int(X_train_selected.shape[1]),
            "candidate_panel_sizes": [],
            "minimum_required_score": None,
            "model_training_allowed": True,
        }
    log_model_input_panel_decision(
        model_name=f"cv_repeat_{repeat}_fold_{fold}",
        X=X_train_selected,
        panel_summary=panel_summary,
        log=logger,
    )
    runner = MLProtocolRunner(config=fold_config)
    setattr(runner, "_networkparser_panel_summary", panel_summary)
    setattr(
        runner,
        "_networkparser_model_name",
        f"cv_repeat_{repeat}_fold_{fold}",
    )
    selected_algorithm, selector_payload = _resolve_fold_algorithm(
        runner=runner,
        X_train=X_train_selected,
        y_train=y_train,
        requested_algorithm=algorithm,
        groups=groups_train.reindex(X_train_selected.index)
        if groups_train is not None
        else None,
    )
    model = runner.train_model(
        genomic_df=X_train_selected,
        labels=y_train,
        algorithm=selected_algorithm,
    )

    # Threshold selection on train OOF when supported (nested, train-only).
    threshold_payload: Dict[str, Any] = {}
    try:
        threshold_payload = runner.select_decision_threshold_out_of_fold(
            genomic_df=X_train_selected,
            labels=y_train,
            algorithm=selected_algorithm,
            out_dir=fold_dir / "threshold_selection",
            groups=(
                groups_train.reindex(X_train_selected.index)
                if groups_train is not None
                else None
            ),
        )
    except Exception as exc:
        threshold_payload = {"status": "skipped", "reason": str(exc)}

    pred_df, score_df = _predict_fold(
        model=model,
        X_test=X_test_selected,
        features=selected_features,
        y_test=y_test,
        repeat=repeat,
        fold=fold,
    )
    pred_df.to_csv(fold_dir / "fold_predictions.tsv", sep="\t", index=False)
    score_df.to_csv(fold_dir / "fold_class_support.tsv", sep="\t")

    truth = pd.Series(
        pred_df["true_label"].astype(str).values,
        index=pred_df["prediction_id"].astype(str).values,
        name="true_label",
    )
    pred = pd.Series(
        pred_df["predicted_label"].astype(str).values,
        index=pred_df["prediction_id"].astype(str).values,
        name="predicted_label",
    )
    # Sanity: validation IDs never appear in the fold training set
    train_set = set(map(str, train_ids))
    test_set = set(map(str, test_ids))
    leakage = train_set.intersection(test_set)
    if leakage:
        raise RuntimeError(
            f"CV fold leakage: {len(leakage)} sample IDs in both train and test"
        )

    fold_eval = evaluate_predictions(
        y_true=truth,
        y_pred=pred,
        class_support_scores=score_df,
        output_dir=fold_dir / "evaluation",
        level_name=f"cv_repeat_{repeat}_fold_{fold}",
        evaluation_role="out_of_fold",
        groups=(
            pd.Series(
                groups_test.reindex(pred_df["sample_id"].astype(str)).to_numpy(),
                index=pred_df["prediction_id"].astype(str).to_numpy(),
            )
            if groups_test is not None
            else None
        ),
    )

    by_class_path = Path(fold_eval.get("artifacts", {}).get("by_class_tsv", ""))
    if by_class_path.exists():
        by_class = pd.read_csv(by_class_path, sep="\t")
        by_class.insert(0, "fold", int(fold))
        by_class.insert(0, "repeat", int(repeat))
        by_class.to_csv(fold_dir / "fold_by_class_metrics.tsv", sep="\t", index=False)

    fold_summary = {
        "status": "success",
        "repeat": int(repeat),
        "fold": int(fold),
        "n_train_samples": int(X_train_selected.shape[0]),
        "n_test_samples": int(X_test_selected.shape[0]),
        "n_features_after_train_invariant_filter": int(len(train_variant_features)),
        "n_selected_features": int(len(selected_features)),
        "selected_algorithm": str(selected_algorithm),
        "panel_status": panel_status,
        "selector": selector_payload,
        "selected_features": selected_features,
        "missingness_audit": {
            "train_fit": miss_audit,
            "train_transform": train_imp_audit,
            "test_transform": test_imp_audit,
            "policy": miss_state.policy.impute_strategy,
        },
        "fold_local_baseline_audit": fold_baseline_audit,
        "threshold_selection": threshold_payload,
        "preprocessing_mode": preprocessing_mode,
        "grouped_outer_fold": bool(groups_train is not None),
        "train_groups": sorted(groups_train.unique().tolist())
        if groups_train is not None
        else [],
        "test_groups": sorted(groups_test.unique().tolist())
        if groups_test is not None
        else [],
        "evaluation": fold_eval,
        "artifacts": {
            "fold_dir": str(fold_dir),
            "fold_predictions": str(fold_dir / "fold_predictions.tsv"),
            "fold_class_support": str(fold_dir / "fold_class_support.tsv"),
            "fold_evaluation": str(
                fold_dir / "evaluation" / "model_performance_summary.json"
            ),
            "missingness_state": str(fold_dir / "missingness_state.json"),
        },
    }
    _write_json(fold_summary, fold_dir / "fold_summary.json")
    return fold_summary


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------


def _flatten_fold_metrics(fold_summaries: List[Dict[str, Any]]) -> pd.DataFrame:
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision_ppv",
        "macro_true_positive_rate",
        "macro_sensitivity_recall",
        "macro_true_negative_rate",
        "macro_specificity",
        "macro_npv",
        "macro_f1",
        "weighted_f1",
        "matthews_corrcoef",
    ]
    rows: List[Dict[str, Any]] = []
    for fold in fold_summaries:
        evaluation = fold.get("evaluation", {}) if isinstance(fold, dict) else {}
        row: Dict[str, Any] = {
            "repeat": fold.get("repeat"),
            "fold": fold.get("fold"),
            "status": fold.get("status"),
            "n_train_samples": fold.get("n_train_samples"),
            "n_test_samples": fold.get("n_test_samples"),
            "n_selected_features": fold.get("n_selected_features"),
            "selected_algorithm": fold.get("selected_algorithm"),
            "panel_status": fold.get("panel_status"),
        }
        for name in metric_names:
            row[name] = evaluation.get(name)
        roc_pr = (
            evaluation.get("roc_pr", {})
            if isinstance(evaluation.get("roc_pr", {}), dict)
            else {}
        )
        row["macro_roc_auc_ovr"] = roc_pr.get("macro_roc_auc_ovr")
        row["weighted_roc_auc_ovr"] = roc_pr.get("weighted_roc_auc_ovr")
        row["macro_pr_auc_average_precision"] = roc_pr.get(
            "macro_pr_auc_average_precision"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_stability(fold_summaries: List[Dict[str, Any]]) -> pd.DataFrame:
    total_folds = max(1, len(fold_summaries))
    counts: Counter[str] = Counter()
    ranks: Dict[str, List[int]] = defaultdict(list)

    for fold in fold_summaries:
        features = fold.get("selected_features", [])
        if not isinstance(features, list):
            continue
        for rank, feature in enumerate(features, start=1):
            f = str(feature)
            counts[f] += 1
            ranks[f].append(int(rank))

    rows: List[Dict[str, Any]] = []
    for feature, count in counts.items():
        rank_values = ranks.get(feature, [])
        rows.append(
            {
                "feature": feature,
                "selection_count": int(count),
                "selection_frequency": float(count / total_folds),
                "mean_selected_rank": float(np.mean(rank_values))
                if rank_values
                else np.nan,
                "median_selected_rank": float(np.median(rank_values))
                if rank_values
                else np.nan,
                "n_successful_folds": int(total_folds),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            by=[
                "selection_frequency",
                "selection_count",
                "mean_selected_rank",
                "feature",
            ],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
    return df


def _collect_by_class_metrics(fold_summaries: List[Dict[str, Any]]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for fold in fold_summaries:
        fold_dir = Path(fold.get("artifacts", {}).get("fold_dir", ""))
        path = fold_dir / "fold_by_class_metrics.tsv"
        if path.exists():
            frames.append(pd.read_csv(path, sep="\t"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _aggregate_prediction_performance(
    predictions: pd.DataFrame,
    output_dir: Path,
    groups: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    if predictions.empty:
        return {"status": "skipped", "message": "No fold predictions available."}

    # Repeated OOF predictions of the same biological sample are correlated.
    # Aggregate once per sample before metrics/CIs rather than treating repeats
    # as independent observations.
    aggregate_rows: List[Dict[str, Any]] = []
    score_dicts: List[Dict[str, float]] = []
    for sample_id, sample_frame in predictions.groupby("sample_id", sort=True):
        truths = sorted(sample_frame["true_label"].astype(str).unique().tolist())
        if len(truths) != 1:
            raise ValueError(
                f"Conflicting truth labels across repeated CV predictions for {sample_id}: {truths}"
            )
        called = sample_frame["predicted_label"].astype(str).tolist()
        counts = Counter(called)
        majority = sorted(counts, key=lambda value: (-counts[value], value))[0]

        parsed_scores: List[Dict[str, float]] = []
        for raw in sample_frame["class_support_json"].astype(str).tolist():
            try:
                parsed = json.loads(raw)
                parsed_scores.append(
                    {str(key): float(value) for key, value in parsed.items()}
                    if isinstance(parsed, dict)
                    else {}
                )
            except Exception:
                parsed_scores.append({})
        key_sets = [set(item) for item in parsed_scores]
        complete_keys = (
            key_sets[0]
            if key_sets and all(keys == key_sets[0] for keys in key_sets)
            else set()
        )
        mean_scores = {
            key: float(np.mean([item[key] for item in parsed_scores]))
            for key in sorted(complete_keys)
        }
        score_dicts.append(mean_scores)
        aggregate_rows.append(
            {
                "sample_id": str(sample_id),
                "true_label": truths[0],
                "predicted_label": majority,
                "n_repeat_predictions": int(sample_frame.shape[0]),
                "prediction_counts_json": json.dumps(dict(counts), sort_keys=True),
                "class_support_json": json.dumps(mean_scores, sort_keys=True),
            }
        )

    aggregate_frame = pd.DataFrame(aggregate_rows)
    aggregate_dir = _ensure_dir(output_dir / "aggregate_performance")
    aggregate_frame.to_csv(
        aggregate_dir / "per_sample_aggregated_predictions.tsv",
        sep="\t",
        index=False,
    )
    sample_index = aggregate_frame["sample_id"].astype(str)
    truth = pd.Series(
        aggregate_frame["true_label"].astype(str).to_numpy(),
        index=sample_index,
        name="true_label",
    )
    pred = pd.Series(
        aggregate_frame["predicted_label"].astype(str).to_numpy(),
        index=sample_index,
        name="predicted_label",
    )
    scores = score_dicts_to_frame(sample_index.tolist(), score_dicts)

    summary = evaluate_predictions(
        y_true=truth,
        y_pred=pred,
        class_support_scores=scores,
        output_dir=aggregate_dir,
        level_name="repeated_cv_per_sample_aggregate",
        evaluation_role="out_of_fold",
        groups=groups.reindex(sample_index) if groups is not None else None,
    )
    summary["repeated_prediction_handling"] = {
        "unit_of_analysis": "unique_sample",
        "n_unique_samples": int(aggregate_frame.shape[0]),
        "n_fold_prediction_rows": int(predictions.shape[0]),
        "label_aggregation": "deterministic_majority_vote",
        "support_aggregation": "mean_only_when_every_repeat_has_same_complete_class_vector",
    }
    _write_json(summary, aggregate_dir / "model_performance_summary.json")
    return summary


# -----------------------------------------------------------------------------
# Public runner
# -----------------------------------------------------------------------------


def run_repeated_cv(
    *,
    genomic_path: str,
    meta_path: str,
    label_column: str,
    output_dir: str,
    config: NetworkParserConfig,
    ref_fasta: Optional[str] = None,
    n_repeats: int = 3,
    n_splits: int = 5,
    algorithm: Optional[str] = None,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Run leakage-aware repeated stratified CV for one supervised target."""
    out = _ensure_dir(Path(output_dir))
    random_state = int(
        getattr(config, "random_state", 42) if random_state is None else random_state
    )
    n_repeats = max(1, int(n_repeats))
    requested_splits = max(2, int(n_splits))

    X, y, input_summary, groups = load_cv_matrix_and_labels(
        genomic_path=genomic_path,
        meta_path=meta_path,
        label_column=label_column,
        output_dir=out,
        config=config,
        ref_fasta=ref_fasta,
    )

    class_counts = y.value_counts(dropna=True)
    feasible_splits = (
        min(requested_splits, int(class_counts.min())) if not class_counts.empty else 0
    )
    if y.nunique(dropna=True) < 2 or feasible_splits < 2:
        raise ValueError(
            "Repeated CV requires at least two classes and at least two samples in the smallest class."
        )

    use_groups = groups is not None
    if use_groups:
        n_groups = int(groups.nunique())
        feasible_splits = min(feasible_splits, n_groups)
        if feasible_splits < 2:
            raise ValueError(
                "Grouped CV requires at least two groups and feasible_splits>=2 "
                f"(n_groups={n_groups})."
            )
        # Every class must appear in enough distinct groups.
        min_groups_per_class = int(getattr(config, "cv_min_groups_per_class", 2))
        class_group_counts: Dict[str, int] = {}
        tmp = pd.DataFrame({"y": y.astype(str), "g": groups.astype(str)})
        for cls, sub in tmp.groupby("y"):
            class_group_counts[str(cls)] = int(sub["g"].nunique())
        weak = {c: n for c, n in class_group_counts.items() if n < min_groups_per_class}
        if weak:
            raise ValueError(
                "Grouped CV requires every class in at least "
                f"cv_min_groups_per_class={min_groups_per_class} groups. "
                f"Under-represented classes: {weak}"
            )
        feasible_splits = min(feasible_splits, min(class_group_counts.values()))

    expected_folds_total = int(n_repeats * feasible_splits)
    allow_partial = bool(getattr(config, "cv_allow_partial_results", False))
    preprocessing_mode = str(getattr(config, "cv_preprocessing_mode", "strict"))

    logger.info(
        "Repeated CV started | target=configured | repeats=%d | requested_folds=%d | "
        "actual_folds=%d | expected_total=%d | grouped=%s | preprocessing=%s",
        n_repeats,
        requested_splits,
        feasible_splits,
        expected_folds_total,
        use_groups,
        preprocessing_mode,
    )

    fold_summaries: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    sample_ids = np.asarray(X.index.astype(str))
    y_values = y.astype(str).to_numpy()
    group_values = groups.astype(str).to_numpy() if use_groups else None

    for repeat in range(1, n_repeats + 1):
        if use_groups:
            cv = StratifiedGroupKFold(
                n_splits=feasible_splits,
                shuffle=True,
                random_state=random_state + repeat - 1,
            )
            split_iter = cv.split(sample_ids, y_values, groups=group_values)
        else:
            cv = StratifiedKFold(
                n_splits=feasible_splits,
                shuffle=True,
                random_state=random_state + repeat - 1,
            )
            split_iter = cv.split(sample_ids, y_values)
        for fold, (train_idx, test_idx) in enumerate(split_iter, start=1):
            train_ids = sample_ids[train_idx].tolist()
            test_ids = sample_ids[test_idx].tolist()
            if use_groups:
                train_groups = set(group_values[train_idx])
                test_groups = set(group_values[test_idx])
                if train_groups.intersection(test_groups):
                    raise RuntimeError(
                        "Grouped CV leakage: groups appear in both train and test"
                    )
            try:
                fold_summary = _run_one_fold(
                    X=X,
                    y=y,
                    train_ids=train_ids,
                    test_ids=test_ids,
                    repeat=repeat,
                    fold=fold,
                    output_dir=out,
                    config=config,
                    algorithm=algorithm,
                    groups=groups,
                )
                fold_summaries.append(fold_summary)
                logger.info(
                    "CV fold complete | repeat=%d | fold=%d | selected_features=%d | algorithm=%s",
                    repeat,
                    fold,
                    int(fold_summary.get("n_selected_features", 0)),
                    fold_summary.get("selected_algorithm", "NA"),
                )
            except Exception as exc:
                # Failed fold remains failed — never silently substituted.
                failure = {
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "status": "failed",
                    "error": str(exc),
                }
                failures.append(failure)
                fail_dir = _ensure_dir(
                    out / f"repeat_{repeat:02d}" / f"fold_{fold:02d}"
                )
                _write_json(failure, fail_dir / "fold_failure.json")
                logger.warning(
                    "CV fold failed | repeat=%d | fold=%d | error=%s", repeat, fold, exc
                )

    fold_metrics = _flatten_fold_metrics(fold_summaries)
    fold_metrics.to_csv(out / "cv_fold_metrics.tsv", sep="\t", index=False)

    prediction_frames: List[pd.DataFrame] = []
    for fold_summary in fold_summaries:
        path = Path(fold_summary.get("artifacts", {}).get("fold_predictions", ""))
        if path.exists():
            prediction_frames.append(pd.read_csv(path, sep="\t"))
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    predictions.to_csv(out / "cv_predictions.tsv", sep="\t", index=False)

    # Per-sample aggregation across repeats (not treating repeats as independent N).
    per_sample_summary: Dict[str, Any] = {}
    if not predictions.empty and "sample_id" in predictions.columns:
        for sid, grp in predictions.groupby("sample_id"):
            labels = grp["predicted_label"].astype(str).tolist()
            counts = Counter(labels)
            per_sample_summary[str(sid)] = {
                "n_repeat_predictions": int(len(grp)),
                "majority_prediction": counts.most_common(1)[0][0] if counts else None,
                "prediction_counts": dict(counts),
            }
        _write_json(per_sample_summary, out / "cv_per_sample_repeat_summary.json")

    stability = _feature_stability(fold_summaries)
    stability.to_csv(out / "cv_feature_stability.tsv", sep="\t", index=False)

    by_class = _collect_by_class_metrics(fold_summaries)
    by_class.to_csv(out / "cv_by_class_metrics.tsv", sep="\t", index=False)

    aggregate_eval = _aggregate_prediction_performance(predictions, out, groups=groups)

    metric_summary: Dict[str, Any] = {}
    numeric_cols = [
        "accuracy",
        "balanced_accuracy",
        "macro_true_positive_rate",
        "macro_sensitivity_recall",
        "macro_true_negative_rate",
        "macro_specificity",
        "macro_precision_ppv",
        "macro_npv",
        "macro_f1",
        "weighted_f1",
        "matthews_corrcoef",
        "macro_roc_auc_ovr",
        "macro_pr_auc_average_precision",
    ]
    for col in numeric_cols:
        if col in fold_metrics.columns:
            values = pd.to_numeric(fold_metrics[col], errors="coerce").dropna()
            metric_summary[col] = {
                "mean": float(values.mean()) if not values.empty else None,
                "std": float(values.std(ddof=1)) if values.shape[0] > 1 else None,
                "min": float(values.min()) if not values.empty else None,
                "max": float(values.max()) if not values.empty else None,
                "n_folds_contributing": int(values.shape[0]),
            }

    n_success = int(len(fold_summaries))
    n_failed = int(len(failures))
    if n_success == 0:
        run_status = "failed"
    elif n_failed > 0 or n_success < expected_folds_total:
        run_status = "partial_failure"
    else:
        run_status = "success"

    publication_ready = bool(
        run_status == "success"
        and n_success == expected_folds_total
        and preprocessing_mode == "strict"
    )
    if not publication_ready and not allow_partial and run_status != "failed":
        logger.warning(
            "CV publication summary withheld: status=%s | successful=%d | expected=%d | "
            "set cv_allow_partial_results=True to export partial publication metrics.",
            run_status,
            n_success,
            expected_folds_total,
        )

    summary: Dict[str, Any] = {
        "status": run_status,
        "publication_ready": publication_ready,
        "cv_allow_partial_results": allow_partial,
        "preprocessing_mode": preprocessing_mode,
        "target_label_column": str(label_column),
        "requested_repeats": int(n_repeats),
        "requested_folds": int(requested_splits),
        "actual_folds_per_repeat": int(feasible_splits),
        "expected_folds_total": int(expected_folds_total),
        "successful_folds": n_success,
        "failed_folds": n_failed,
        "input_summary": input_summary,
        "metric_summary_across_folds": metric_summary
        if (publication_ready or allow_partial)
        else {
            "withheld": True,
            "reason": "partial_or_non_strict_cv; set cv_allow_partial_results=True to include",
            "raw_available_in_fold_metrics": True,
        },
        "aggregate_prediction_performance": aggregate_eval
        if (publication_ready or allow_partial)
        else {
            "withheld": True,
            "reason": "partial_or_non_strict_cv",
        },
        "failures": failures,
        "artifacts": {
            "cv_input_summary": str(out / "cv_input_summary.json"),
            "cv_fold_metrics": str(out / "cv_fold_metrics.tsv"),
            "cv_predictions": str(out / "cv_predictions.tsv"),
            "cv_feature_stability": str(out / "cv_feature_stability.tsv"),
            "cv_by_class_metrics": str(out / "cv_by_class_metrics.tsv"),
            "aggregate_performance_dir": str(out / "aggregate_performance"),
            "cv_per_sample_repeat_summary": str(
                out / "cv_per_sample_repeat_summary.json"
            ),
        },
        "method_note": (
            "Strict nested CV: each fold fits missingness limits/imputation, statistical "
            "feature filtering, panel selection, model selection, and optional OOF threshold "
            "selection on outer-training samples only. Failed folds remain failed. "
            "Repeated predictions of the same sample are summarised per sample, not treated "
            "as independent observations. "
            f"preprocessing_mode={preprocessing_mode}."
        ),
    }
    if preprocessing_mode == "exploratory_transductive":
        summary[
            "method_note"
        ] += " EXPLORATORY TRANSDUCTIVE MODE — do not present as held-out validation."
    _write_json(summary, out / "cross_validation_summary.json")

    logger.info(
        "Repeated CV complete | status=%s | successful_folds=%d | failed_folds=%d | "
        "publication_ready=%s | out=%s",
        run_status,
        n_success,
        n_failed,
        publication_ready,
        str(out),
    )
    return summary

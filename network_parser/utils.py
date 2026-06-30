# network_parser/utils.py
"""
Utility functions for NetworkParser.

Includes:
- Safe YAML config loading (optional dependency: PyYAML)
- CLI args → NetworkParserConfig builder
- Filesystem helpers (ensure_dir)
- Timestamp helper
- JSON save helper (save_json)

Keep this module lightweight because it is imported early by the CLI/package.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

try:
    from .config import NetworkParserConfig
except Exception:  # pragma: no cover - supports direct source-tree execution
    from config import NetworkParserConfig  # type: ignore

logger = logging.getLogger(__name__)

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

import logging
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# General helpers expected by the pipeline
# ──────────────────────────────────────────────────────────────

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists and return it as a Path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """
    Return a filesystem-friendly timestamp string.
    """
    return datetime.now().strftime(fmt)


def json_default(obj: Any) -> Any:
    """JSON serializer for common NetworkParser runtime objects."""
    try:
        import numpy as _np  # local import keeps utils lightweight at import time
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_np.ndarray,)):
            return obj.tolist()
    except Exception:
        pass

    try:
        import pandas as _pd
        if isinstance(obj, (_pd.Series, _pd.Index)):
            return obj.tolist()
        if isinstance(obj, _pd.DataFrame):
            return obj.to_dict(orient="records")
    except Exception:
        pass

    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        try:
            return vars(obj)
        except Exception:
            pass
    return str(obj)


def save_json(data: Any, out_path: Union[str, Path], indent: int = 2) -> Path:
    """
    Save JSON to disk with sane defaults.
    Creates parent directories automatically.

    Parameters
    ----------
    data : Any
        JSON-serializable object.
    out_path : str | Path
        Output file path.
    indent : int
        JSON indent level.

    Returns
    -------
    Path
        Path to written JSON file.
    """
    out_path = Path(out_path)
    if out_path.parent:
        ensure_dir(out_path.parent)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=json_default)
        f.write("\n")

    logger.info("Wrote JSON: %s", out_path)
    return out_path

def normalize_sample_id(
    value: Any,
    strip_library_suffix: bool = True,
) -> str:
    """
    Normalize sample identifiers consistently across training, query, and
    metadata-alignment stages.

    This deliberately performs only conservative filename cleanup plus the
    existing NetworkParser library-suffix cleanup. It should not rewrite
    biologically meaningful sample names.
    """
    sample = str(value).strip()

    if sample.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return ""

    sample = re.sub(r"(?i)(\.vcf\.gz|\.vcf|\.bcf\.gz|\.bcf|\.gz)$", "", sample)

    if strip_library_suffix:
        sample = re.sub(r"(?i)_library[0-9]+$", "", sample)

    return sample


# ──────────────────────────────────────────────────────────────
# User-facing pipeline logging helpers
# ──────────────────────────────────────────────────────────────

def _compact_log_value(value: Any) -> str:
    """Return a compact, stable representation for user-facing log fields."""
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        if len(items) > 6:
            shown = ",".join(str(x) for x in items[:6])
            return f"[{shown},...+{len(items) - 6}]"
        return "[" + ",".join(str(x) for x in items) + "]"
    return str(value)


def format_log_kv(**fields: Any) -> str:
    """Format key=value fields for concise, readable pipeline logs."""
    clean = []
    for key, value in fields.items():
        if value is None:
            continue
        clean.append(f"{key}={_compact_log_value(value)}")
    return " | ".join(clean)


def _normalize_sentence(text: str) -> str:
    """Return a single sentence with stable terminal punctuation."""
    body = str(text or "").strip()
    if not body:
        return ""
    if body[-1] not in ".!?":
        body += "."
    return body


def _join_step_narrative(happened: str, reason: str) -> str:
    """Combine action and rationale into plain prose without section labels."""
    parts = [_normalize_sentence(part) for part in (happened, reason)]
    return " ".join(part for part in parts if part)


def progress_enabled() -> bool:
    """Return whether tqdm progress bars should be shown."""
    flag = os.environ.get("NETWORKPARSER_DISABLE_PROGRESS", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return False
    isatty = getattr(sys.stderr, "isatty", None)
    return bool(isatty and isatty())


def progress_iter(
    iterable: Iterable[Any],
    *,
    desc: str = "",
    total: int | None = None,
    unit: str = "",
    leave: bool = False,
    position: int | None = None,
):
    """Wrap an iterable with tqdm when progress display is enabled."""
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover - tqdm is an environment dependency
        return iterable

    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        leave=leave,
        position=position,
        disable=not progress_enabled(),
        dynamic_ncols=True,
    )


class PipelineProgress:
    """Track high-level pipeline stage completion with a progress bar."""

    def __init__(self, stages: Sequence[str], *, title: str = "Pipeline") -> None:
        self._stages = [str(stage) for stage in stages]
        self._title = str(title)
        self._bar = None
        if not self._stages:
            return
        try:
            from tqdm.auto import tqdm
        except Exception:  # pragma: no cover
            return
        self._bar = tqdm(
            total=len(self._stages),
            desc=self._title,
            unit="stage",
            leave=True,
            disable=not progress_enabled(),
            dynamic_ncols=True,
        )

    def begin_stage(self, name: str) -> None:
        if self._bar is not None:
            self._bar.set_description(f"{self._title} — {name}")

    def complete_stage(self, name: str = "") -> None:
        if self._bar is None:
            return
        if name:
            self._bar.set_postfix_str(str(name)[:48], refresh=False)
        self._bar.update(1)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def __enter__(self) -> "PipelineProgress":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def log_pipeline_header(log: logging.Logger, title: str, **fields: Any) -> None:
    """Emit a clear run-level banner without exposing feature names or labels."""
    line = "=" * 78
    log.info(line)
    log.info("%s", title)
    if fields:
        log.info("Run settings | %s", format_log_kv(**fields))
    log.info(line)


def log_stage_start(
    log: logging.Logger,
    stage: Union[int, str],
    name: str,
    *,
    progress: PipelineProgress | None = None,
    **fields: Any,
) -> None:
    """Emit a standard stage-start message."""
    if progress is not None:
        progress.begin_stage(name)
    suffix = f" | {format_log_kv(**fields)}" if fields else ""
    log.info("▶ Stage %s — %s%s", stage, name, suffix)


def log_stage_complete(
    log: logging.Logger,
    stage: Union[int, str],
    name: str,
    *,
    progress: PipelineProgress | None = None,
    **fields: Any,
) -> None:
    """Emit a standard stage-complete message."""
    suffix = f" | {format_log_kv(**fields)}" if fields else ""
    log.info("✓ Stage %s complete — %s%s", stage, name, suffix)
    if progress is not None:
        progress.complete_stage(name)


def log_branch_decision(log: logging.Logger, branch: str, status: str, **fields: Any) -> None:
    """Emit a concise branch decision message for optional workflow branches."""
    fields = dict(fields)
    reason = fields.pop("reason", None)
    suffix = f" | {format_log_kv(**fields)}" if fields else ""
    if reason:
        reason_text = _normalize_sentence(str(reason)).rstrip(".")
        log.info("Branch decision — %s: %s; %s%s", branch, status, reason_text, suffix)
    else:
        log.info("Branch decision — %s: %s%s", branch, status, suffix)


def log_artifact(log: logging.Logger, label: str, path: Union[str, Path]) -> None:
    """Emit a standard artifact message."""
    log.info("Artifact written — %s | path=%s", label, str(path))


def log_final_run_summary(
    log: logging.Logger,
    *,
    title: str,
    sections: Iterable[Dict[str, Any]],
    artifacts: Optional[Dict[str, Union[str, Path]]] = None,
    warnings: Optional[Iterable[Dict[str, Any]]] = None,
) -> None:
    """Log a compact final summary without using a separate 'reason' label.

    Each section should include:
      - name: short stage label
      - message: one sentence explaining what the stage means
      - optional fields: compact movement/status values
    """
    line = "=" * 78
    log.info(line)
    log.info("%s", title)
    log.info(line)

    for section in sections:
        if not isinstance(section, dict):
            continue
        name = str(section.get("name", "Stage"))
        message = str(section.get("message", "")).strip()
        fields = section.get("fields", {})
        field_text = format_log_kv(**fields) if isinstance(fields, dict) and fields else ""
        suffix = f" | {field_text}" if field_text else ""
        if message:
            log.info("%s: %s%s", name, message, suffix)
        elif suffix:
            log.info("%s:%s", name, suffix)

    if artifacts:
        log.info("Outputs:")
        for label, path in artifacts.items():
            if path:
                log.info("  %s: %s", label, str(path))

    warning_list = [w for w in (warnings or []) if isinstance(w, dict)]
    if warning_list:
        log.info("Warnings captured in run_audit.json: %d", len(warning_list))

    log.info(line)


def audit_warning(
    *,
    stage: str,
    message: str,
    code: str = "warning",
    severity: str = "warning",
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a standardized warning record for run_audit.json."""
    return {
        "timestamp": timestamp(),
        "severity": str(severity),
        "stage": str(stage),
        "code": str(code),
        "message": str(message),
        "details": details or {},
    }


def collect_common_warnings(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect high-level warnings from a NetworkParser result payload.

    This intentionally avoids pre-flight validation. It only records issues that
    were observed during the actual run, so the audit trail remains faithful to
    what happened.
    """
    warnings: List[Dict[str, Any]] = []

    feature_filter = results.get("feature_filtering", {}) if isinstance(results, dict) else {}
    if isinstance(feature_filter, dict):
        if feature_filter.get("used_fallback_unfiltered_matrix"):
            warnings.append(audit_warning(
                stage="central_feature_filtering",
                code="exploratory_unfiltered_fallback",
                message=(
                    "Central filtering retained no supported features and an unfiltered fallback was used. "
                    "Treat this as exploratory rather than publication-grade FDR-supported output."
                ),
                details={
                    "method": feature_filter.get("method"),
                    "fallback_strategy": feature_filter.get("fallback_strategy"),
                },
            ))
        if feature_filter.get("status") in {"skipped", "disabled"}:
            warnings.append(audit_warning(
                stage="central_feature_filtering",
                code="central_filtering_skipped",
                message="Central statistical filtering was skipped, so the downstream matrix is not FDR-filtered.",
                details={"status": feature_filter.get("status")},
            ))

    panel = results.get("feature_panel_separability", {}) if isinstance(results, dict) else {}
    if isinstance(panel, dict):
        status = str(panel.get("status", "")).lower()
        reason = str(panel.get("reason", "")).lower()
        if status in {"skipped", "failed"} or "fallback" in reason:
            warnings.append(audit_warning(
                stage="feature_panel_selection",
                code="feature_panel_not_cleanly_selected",
                message="The ranked feature-panel step did not complete as a clean smallest-passing panel selection.",
                details={"status": panel.get("status"), "reason": panel.get("reason")},
            ))

    ml = results.get("ml_protocol", {}) if isinstance(results, dict) else {}
    if isinstance(ml, dict) and ml:
        selector = ml.get("selector", {}) if isinstance(ml.get("selector", {}), dict) else {}
        selector_status = str(selector.get("selector_status", "")).lower()
        if selector_status and selector_status not in {"success", "ok"}:
            warnings.append(audit_warning(
                stage="ml_protocol",
                code="model_selector_status",
                message="The model selector reported a non-standard status during model screening.",
                details={"selector_status": selector.get("selector_status")},
            ))

    discovery = results.get("discovery", {}) if isinstance(results, dict) else {}
    if not discovery and results.get("pipeline_mode") in {"both", "decision_tree_only"}:
        warnings.append(audit_warning(
            stage="decision_tree_branch",
            code="decision_tree_not_run",
            message="Decision-tree interpretability output was not generated for this run.",
            details={"pipeline_mode": results.get("pipeline_mode")},
        ))

    return warnings


def write_run_audit(
    output_dir: Union[str, Path],
    audit_payload: Dict[str, Any],
    filename: str = "run_audit.json",
) -> Path:
    """Write the run audit file into the requested output directory."""
    out = ensure_dir(output_dir) / filename
    save_json(audit_payload, out)
    return out


def _safe_stage_name(stage_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(stage_name).strip())
    return safe.strip("_") or "stage"


def checkpoint_dir(output_dir: Union[str, Path]) -> Path:
    return ensure_dir(Path(output_dir) / "_checkpoints")


def write_stage_checkpoint(
    output_dir: Union[str, Path],
    stage_name: str,
    payload: Dict[str, Any],
    *,
    status: str = "complete",
) -> Path:
    """Write a lightweight stage checkpoint metadata file."""
    path = checkpoint_dir(output_dir) / f"{_safe_stage_name(stage_name)}.json"
    checkpoint_payload = {
        "stage_name": str(stage_name),
        "status": str(status),
        "timestamp": timestamp(),
        "payload": payload,
    }
    save_json(checkpoint_payload, path)
    return path


def load_stage_checkpoint(output_dir: Union[str, Path], stage_name: str) -> Optional[Dict[str, Any]]:
    """Load a stage checkpoint metadata file if present."""
    path = checkpoint_dir(output_dir) / f"{_safe_stage_name(stage_name)}.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def log_flow_step(
    log: logging.Logger,
    *,
    step: str,
    happened: str,
    reason: str,
    before_samples: int | None = None,
    before_features: int | None = None,
    after_samples: int | None = None,
    after_features: int | None = None,
    threshold: str | None = None,
    status: str | None = None,
    artifact: Union[str, Path, None] = None,
) -> None:
    """Log a user-facing explanation block with action, rationale, and movement.

    This is intentionally INFO-level because it explains the statistical and
    preprocessing rationale. Exact feature/sample identifiers should remain in
    DEBUG logs or audit artifacts.
    """
    log.info("%s", step)
    narrative = _join_step_narrative(happened, reason)
    if narrative:
        log.info("  %s", narrative)

    movement = format_log_kv(
        input_samples=before_samples,
        input_features=before_features,
        output_samples=after_samples,
        output_features=after_features,
        threshold=threshold,
        status=status,
    )
    if movement:
        log.info("  %s", movement)
    if artifact is not None:
        log.info("  Wrote audit record to %s", str(artifact))


def log_filter_step(
    log: logging.Logger,
    *,
    filter_name: str,
    happened: str,
    reason: str,
    before_samples: int | None = None,
    before_features: int | None = None,
    after_samples: int | None = None,
    after_features: int | None = None,
    threshold: str | None = None,
    status: str | None = None,
    artifact: Union[str, Path, None] = None,
) -> None:
    """Convenience wrapper for filters/preprocessing gates."""
    log_flow_step(
        log,
        step=f"Filter — {filter_name}",
        happened=happened,
        reason=reason,
        before_samples=before_samples,
        before_features=before_features,
        after_samples=after_samples,
        after_features=after_features,
        threshold=threshold,
        status=status,
        artifact=artifact,
    )

# ──────────────────────────────────────────────────────────────
# Config handling
# ──────────────────────────────────────────────────────────────

def create_config_from_args(args: argparse.Namespace) -> NetworkParserConfig:
    """
    Create a NetworkParserConfig object from CLI args.

    Notes:
    - Only updates fields that are present and not None.
    - Keeps defaults from NetworkParserConfig for everything else.
    """
    config = NetworkParserConfig()

    # Defensive: use getattr because some args may not exist depending on CLI evolution
    for attr in [
        "bootstrap_iterations",
        "confidence_threshold",
        "max_interaction_order",
        "fdr_threshold",
        "min_group_size",
        "correction_method",
        "max_workers",
        "chunk_size",
        "cross_validation_folds",
        "stability_threshold",
        "min_bootstrap_support",
    ]:
        val = getattr(args, attr, None)
        if val is not None:
            setattr(config, attr, val)

    # Output formats
    out_fmt = getattr(args, "output_format", None)
    if out_fmt:
        config.output_formats = [x.strip() for x in out_fmt.split(",") if x.strip()]

    # Boolean flags
    # (Only set if present; otherwise leave config defaults as-is)
    for battr in ["memory_efficient", "include_matrices", "generate_plots"]:
        if hasattr(args, battr):
            setattr(config, battr, bool(getattr(args, battr)))

    return config


def load_config_file(config_path: Union[str, Path]) -> NetworkParserConfig:
    """
    Load configuration from a YAML file into NetworkParserConfig.

    Requires PyYAML to be installed (conda-forge: pyyaml).

    Parameters
    ----------
    config_path : str | Path
        Path to YAML config file.

    Returns
    -------
    NetworkParserConfig
    """
    if yaml is None:
        raise ImportError(
            "PyYAML is required to load YAML config files. "
            "Install it with: conda install -c conda-forge pyyaml "
            "or: pip install pyyaml"
        )

    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f) or {}

    config = NetworkParserConfig()

    # Sections are optional; use .get with defaults
    analysis_cfg: Dict[str, Any] = config_dict.get("analysis", {}) or {}
    processing_cfg: Dict[str, Any] = config_dict.get("processing", {}) or {}
    output_cfg: Dict[str, Any] = config_dict.get("output", {}) or {}
    validation_cfg: Dict[str, Any] = config_dict.get("validation", {}) or {}

    # Analysis
    config.bootstrap_iterations = analysis_cfg.get("bootstrap_iterations", config.bootstrap_iterations)
    config.confidence_threshold = analysis_cfg.get("confidence_threshold", config.confidence_threshold)
    config.max_interaction_order = analysis_cfg.get("max_interaction_order", config.max_interaction_order)
    config.fdr_threshold = analysis_cfg.get("fdr_threshold", config.fdr_threshold)

    # Processing
    config.max_workers = processing_cfg.get("max_workers", config.max_workers)
    config.memory_efficient = processing_cfg.get("memory_efficient", config.memory_efficient)
    config.chunk_size = processing_cfg.get("chunk_size", config.chunk_size)

    # Output
    config.output_formats = output_cfg.get("formats", config.output_formats)
    config.include_matrices = output_cfg.get("include_matrices", config.include_matrices)
    config.generate_plots = output_cfg.get("generate_plots", config.generate_plots)

    # Validation
    config.cross_validation_folds = validation_cfg.get("cross_validation_folds", config.cross_validation_folds)
    config.stability_threshold = validation_cfg.get("stability_threshold", config.stability_threshold)
    config.min_bootstrap_support = validation_cfg.get("min_bootstrap_support", config.min_bootstrap_support)

    logger.info("Loaded config from %s", config_path)
    return config


def resolve_effective_n_jobs(
    config: Any,
    *,
    override: Optional[int] = None,
    minimum_tasks: int = 1,
) -> int:
    """Resolve worker count for a parallel stage."""
    if override is not None:
        requested = int(override)
    else:
        requested = int(getattr(config, "n_jobs", -1))

    if requested == 0:
        return 1
    if requested < 0:
        cpu_count = max(1, int(os.cpu_count() or 1))
        return cpu_count if minimum_tasks > 1 else 1
    return max(1, requested)


def should_run_parallel(
    config: Any,
    *,
    enabled_attr: str,
    n_tasks: int,
    min_tasks: int = 2,
) -> bool:
    """Return True when a parallel code path should be used."""
    if int(n_tasks) < int(min_tasks):
        return False
    return bool(getattr(config, enabled_attr, True))

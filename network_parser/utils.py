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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .config import NetworkParserConfig

logger = logging.getLogger(__name__)

# Optional dependency: PyYAML
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


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
        json.dump(data, f, indent=indent, ensure_ascii=False)
        f.write("\n")

    logger.info("Wrote JSON: %s", out_path)
    return out_path


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

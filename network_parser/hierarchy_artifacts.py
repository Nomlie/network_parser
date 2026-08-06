#!/usr/bin/env python3
"""
Hierarchy training/query support artifacts:

- resource profile log
- train/query sample-ID collision checks
- per-node training dashboard
- hierarchy dendrogram export
- resume/skip completed nodes
- catalogue circularity summary (from annotated panels)
- biological hierarchy presets
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Biological hierarchy presets (no artificial Lineage_Supergroup)
# ---------------------------------------------------------------------------

HIERARCHY_PRESETS: Dict[str, Dict[str, Any]] = {
    "lineage_amr_profile": {
        "labels": ["Lineage_clean", "AMR_binary", "Resistance_Profile_Collapsed"],
        "description": (
            "Biological 3-level hierarchy: major lineage → any resistance → "
            "collapsed resistance profile."
        ),
    },
    "lineage_family_amr_profile": {
        "labels": [
            "Lineage_family",
            "Lineage_clean",
            "AMR_binary",
            "Resistance_Profile_Collapsed",
        ],
        "description": (
            "Biological 4-level hierarchy: lineage family → major lineage → "
            "any resistance → collapsed resistance profile."
        ),
    },
    "lineage_amr_binary": {
        "labels": ["Lineage_clean", "AMR_binary"],
        "description": "Two-level: major lineage → binary AMR.",
    },
}


def resolve_global_fallback_label_columns(
    hierarchy_labels: Sequence[str],
    config: Any,
) -> List[str]:
    """
    Decide which hierarchy levels get a cohort-wide global model.

    Returns an ordered unique list of label column names subset of hierarchy_labels.
    """
    labels = [str(x).strip() for x in hierarchy_labels if str(x).strip()]
    raw = str(getattr(config, "hierarchy_global_fallback_labels", "none") or "none").strip()
    if not raw or raw.lower() in {"none", "off", "false", "0", "no"}:
        return []

    tokens = [t.strip() for t in raw.replace(";", ",").split(",") if t.strip()]
    if not tokens:
        return []

    lowered = {t.lower() for t in tokens}
    if lowered & {"legacy", "*", "all", "default"}:
        # Previous behaviour: global terminal + global lineage (if flag allows)
        out: List[str] = []
        if labels:
            out.append(labels[-1])  # terminal
        if bool(getattr(config, "hierarchy_train_global_lineage_fallback", True)):
            # Prefer explicit lineage fallback label, else Lineage_clean if present
            lin = getattr(config, "hierarchy_global_lineage_fallback_label", None)
            if lin and str(lin).strip() in labels:
                out.append(str(lin).strip())
            elif "Lineage_clean" in labels:
                out.append("Lineage_clean")
            else:
                for cand in labels:
                    if "lineage" in cand.lower() and cand not in out:
                        out.append(cand)
                        break
        # unique preserve order
        seen: Set[str] = set()
        uniq = []
        for x in out:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq

    resolved: List[str] = []
    label_lower = {lab.lower(): lab for lab in labels}
    for tok in tokens:
        tl = tok.lower()
        if tl == "terminal":
            if labels:
                resolved.append(labels[-1])
            continue
        if tl in {"lineage", "global_lineage"}:
            lin = getattr(config, "hierarchy_global_lineage_fallback_label", None)
            if lin and str(lin).strip() in labels:
                resolved.append(str(lin).strip())
            elif "Lineage_clean" in labels:
                resolved.append("Lineage_clean")
            else:
                for cand in labels:
                    if "lineage" in cand.lower():
                        resolved.append(cand)
                        break
            continue
        # Exact column name (case-insensitive)
        if tok in labels:
            resolved.append(tok)
        elif tl in label_lower:
            resolved.append(label_lower[tl])
        else:
            logger.warning(
                "hierarchy_global_fallback_labels token %r not in hierarchy %s; ignored",
                tok,
                labels,
            )

    # If lineage global is disabled, drop lineage-like entries that were only
    # for the dedicated lineage fallback path — still allow explicit user list.
    seen2: Set[str] = set()
    uniq2: List[str] = []
    for x in resolved:
        if x not in seen2 and x in labels:
            seen2.add(x)
            uniq2.append(x)
    return uniq2


def resolve_hierarchy_labels(
    *,
    hierarchy_labels: Optional[Sequence[str]] = None,
    preset: Optional[str] = None,
) -> List[str]:
    """Resolve label list from explicit columns and/or a named preset."""
    labels: List[str] = []
    if preset:
        key = str(preset).strip().lower()
        if key not in HIERARCHY_PRESETS:
            raise ValueError(
                f"Unknown hierarchy_preset={preset!r}. "
                f"Supported: {sorted(HIERARCHY_PRESETS)}"
            )
        labels = list(HIERARCHY_PRESETS[key]["labels"])
    if hierarchy_labels:
        explicit = [str(x).strip() for x in hierarchy_labels if str(x).strip()]
        if labels and explicit and explicit != labels:
            logger.warning(
                "hierarchy_labels %s override preset %s (%s)",
                explicit,
                preset,
                labels,
            )
        if explicit:
            labels = explicit
    if len(labels) < 2:
        raise ValueError(
            "Need at least two hierarchy labels (use --hierarchy_labels or "
            "--hierarchy_preset)."
        )
    return labels


# ---------------------------------------------------------------------------
# Resource profile
# ---------------------------------------------------------------------------

def write_resource_profile(
    output_dir: str | Path,
    *,
    config: Any = None,
    stage: str = "train",
) -> Path:
    """Log CPU/RAM and parallel budget at stage start."""
    try:
        from network_parser.utils import (
            available_cpu_count,
            available_memory_gb,
            resolve_parallel_worker_budget,
        )
    except ImportError:  # pragma: no cover
        from utils import (  # type: ignore
            available_cpu_count,
            available_memory_gb,
            resolve_parallel_worker_budget,
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    budget = resolve_parallel_worker_budget(config, n_tasks=8) if config is not None else {}
    payload = {
        "stage": str(stage),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": os.uname().nodename if hasattr(os, "uname") else None,
        "cpu_count": available_cpu_count(),
        "available_memory_gb": available_memory_gb(),
        "n_jobs_config": getattr(config, "n_jobs", None) if config is not None else None,
        "parallel_flags": {
            "hierarchy_parallel_child_nodes": getattr(
                config, "hierarchy_parallel_child_nodes", None
            )
            if config is not None
            else None,
            "level2_parallel_group_training": getattr(
                config, "level2_parallel_group_training", None
            )
            if config is not None
            else None,
            "hierarchy_parallel_fallback_training": getattr(
                config, "hierarchy_parallel_fallback_training", None
            )
            if config is not None
            else None,
            "memory_efficient": getattr(config, "memory_efficient", None)
            if config is not None
            else None,
            "parallel_memory_per_worker_gb": getattr(
                config, "parallel_memory_per_worker_gb", None
            )
            if config is not None
            else None,
            "parallel_max_workers": getattr(config, "parallel_max_workers", None)
            if config is not None
            else None,
        },
        "example_worker_budget_for_8_tasks": budget,
    }
    path = out / f"resource_profile_{stage}.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    logger.info(
        "Resource profile (%s) | cpus=%s | mem_gb=%s | outer≈%s | path=%s",
        stage,
        payload["cpu_count"],
        payload["available_memory_gb"],
        (budget or {}).get("outer_jobs"),
        path,
    )
    return path


# ---------------------------------------------------------------------------
# Sample-ID integrity
# ---------------------------------------------------------------------------

def assert_disjoint_sample_ids(
    train_ids: Sequence[Any],
    other_ids: Sequence[Any],
    *,
    train_context: str = "training",
    other_context: str = "query/holdout",
) -> None:
    """Hard-fail on overlapping sample IDs (leakage guard)."""
    try:
        from network_parser.utils import normalize_sample_id
    except ImportError:  # pragma: no cover
        from utils import normalize_sample_id  # type: ignore

    a = {normalize_sample_id(str(x)) for x in train_ids if str(x).strip()}
    b = {normalize_sample_id(str(x)) for x in other_ids if str(x).strip()}
    a.discard("")
    b.discard("")
    overlap = sorted(a & b)
    if overlap:
        raise ValueError(
            f"Sample-ID collision between {train_context} and {other_context}: "
            f"{overlap[:20]}{' ...' if len(overlap) > 20 else ''}. "
            "Refusing to continue (leakage risk)."
        )


# ---------------------------------------------------------------------------
# Node dashboard + resume
# ---------------------------------------------------------------------------

def _node_rows_from_registry_node(
    node: Dict[str, Any],
    rows: List[Dict[str, Any]],
    parent_route: str = "",
) -> None:
    if not isinstance(node, dict):
        return
    path = node.get("path") or []
    path_str = " / ".join(
        f"{p.get('label_column')}={p.get('value')}"
        for p in path
        if isinstance(p, dict)
    )
    model = node.get("model") if isinstance(node.get("model"), dict) else {}
    algo = None
    if isinstance(model, dict):
        algo = model.get("selected_algorithm") or model.get("algorithm")
    rows.append(
        {
            "level_number": node.get("level_number"),
            "label_column": node.get("label_column"),
            "path": path_str,
            "route_from_parent": parent_route,
            "status": node.get("status"),
            "reason": node.get("reason"),
            "algorithm": algo,
            "n_samples": node.get("n_samples"),
            "n_training_samples": node.get("n_training_samples"),
            "n_features": len(node.get("features") or [])
            if isinstance(node.get("features"), list)
            else node.get("n_features_available"),
            "model_file": node.get("model_file"),
            "constant_label": node.get("constant_label"),
        }
    )
    children = node.get("children") or {}
    if isinstance(children, dict):
        for route, child in children.items():
            if isinstance(child, dict):
                _node_rows_from_registry_node(child, rows, parent_route=str(route))


def write_hierarchy_node_dashboard(
    registry: Dict[str, Any],
    output_dir: str | Path,
) -> Path:
    """Write one-row-per-node dashboard TSV from a hierarchy registry."""
    rows: List[Dict[str, Any]] = []
    hierarchy = registry.get("hierarchy") or {}
    root = hierarchy.get("root")
    if isinstance(root, dict):
        _node_rows_from_registry_node(root, rows)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "hierarchy_node_dashboard.tsv"
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    logger.info("Wrote hierarchy node dashboard | nodes=%d | path=%s", len(rows), path)
    return path


def try_load_completed_node(
    node_dir: Path,
    *,
    resume: bool,
) -> Optional[Dict[str, Any]]:
    """
    If resume is enabled and a completed node_summary.json exists with a usable
    model (or constant/skipped terminal state), return it and skip re-training.
    """
    if not resume:
        return None
    summary_path = Path(node_dir) / "node_summary.json"
    if not summary_path.is_file():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    status = str(payload.get("status") or "")
    if status in {"skipped", "constant"}:
        logger.info("Resume: reusing %s node at %s", status, node_dir)
        return payload
    if status == "success":
        model_file = payload.get("model_file")
        if model_file and Path(str(model_file)).is_file():
            logger.info("Resume: reusing trained node at %s", node_dir)
            return payload
        # model path may be relative
        if model_file and (Path(node_dir) / Path(str(model_file)).name).is_file():
            logger.info("Resume: reusing trained node (local model) at %s", node_dir)
            return payload
    return None


def export_hierarchy_dendrogram(
    registry_path: str | Path,
    output_dir: str | Path,
    *,
    figure_number: int = 6,
    stem: str = "hierarchy_dendrogram",
) -> Optional[Path]:
    """Best-effort dendrogram PNG/PDF next to training outputs."""
    try:
        # Prefer in-repo manuscript helper if available
        from network_parser.hierarchy_tree_plot import draw_registry_tree
    except ImportError:
        try:
            from hierarchy_tree_plot import draw_registry_tree  # type: ignore
        except ImportError:
            logger.warning("hierarchy_tree_plot not available; skipping dendrogram")
            return None
    out = Path(output_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{stem}.png"
    pdf = out / f"{stem}.pdf"
    try:
        draw_registry_tree(
            Path(registry_path),
            png,
            pdf,
            figure_number=figure_number,
        )
        return png
    except Exception as exc:
        logger.warning("Could not write hierarchy dendrogram: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Catalogue circularity
# ---------------------------------------------------------------------------

def write_catalogue_circularity_report(
    annotated_features_tsv: str | Path,
    output_dir: str | Path,
) -> Path:
    """
    Summarise how much of each node panel is exact catalogue mutations vs
    candidate genes vs non-catalogue (circularity / mechanism recovery audit).
    """
    path = Path(annotated_features_tsv)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t")
    status_col = "catalogue_status" if "catalogue_status" in df.columns else None
    if status_col is None:
        raise ValueError("annotated features lack catalogue_status")

    def _norm(s: Any) -> str:
        t = str(s or "").strip().lower()
        if "known" in t:
            return "known_mutation"
        if "candidate" in t:
            return "candidate_gene"
        return "not_in_catalogue"

    df = df.copy()
    df["_cat"] = df[status_col].map(_norm)
    group_cols = [c for c in ("node_label", "hierarchy_path") if c in df.columns]
    if not group_cols:
        group_cols = ["_all"]
        df["_all"] = "all_nodes"

    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = len(sub)
        n_known = int((sub["_cat"] == "known_mutation").sum())
        n_cand = int((sub["_cat"] == "candidate_gene").sum())
        n_other = int((sub["_cat"] == "not_in_catalogue").sum())
        rec = {col: val for col, val in zip(group_cols, keys)}
        rec.update(
            {
                "n_features": n,
                "n_known_mutation": n_known,
                "n_candidate_gene": n_cand,
                "n_not_in_catalogue": n_other,
                "fraction_known_mutation": float(n_known / n) if n else 0.0,
                "fraction_catalogue_related": float((n_known + n_cand) / n) if n else 0.0,
                "fraction_not_in_catalogue": float(n_other / n) if n else 0.0,
            }
        )
        rows.append(rec)

    summary = pd.DataFrame(rows)
    tsv = out / "catalogue_circularity_by_node.tsv"
    summary.to_csv(tsv, sep="\t", index=False)
    overall = {
        "n_features": int(len(df)),
        "n_known_mutation": int((df["_cat"] == "known_mutation").sum()),
        "n_candidate_gene": int((df["_cat"] == "candidate_gene").sum()),
        "n_not_in_catalogue": int((df["_cat"] == "not_in_catalogue").sum()),
        "fraction_known_mutation": float((df["_cat"] == "known_mutation").mean())
        if len(df)
        else 0.0,
        "note": (
            "High known-mutation fraction may indicate reconstruction of "
            "catalogue-derived labels rather than independent biology."
        ),
        "by_node_tsv": str(tsv),
    }
    jpath = out / "catalogue_circularity_summary.json"
    jpath.write_text(json.dumps(overall, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Catalogue circularity | known=%.1f%% | path=%s",
        100.0 * overall["fraction_known_mutation"],
        jpath,
    )
    return jpath

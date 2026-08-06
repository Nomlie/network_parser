#!/usr/bin/env python3
"""
Optional seeding of known resistance markers into feature panels.

When enabled, catalogue / known mutations that exist in the filtered matrix
are force-included (or rank-boosted) for phenotype-related hierarchy nodes
(AMR binary, resistance profile, etc.). Lineage-only stages are left alone
unless configured otherwise.

Default: disabled — statistical ranking is unchanged until the user opts in.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

_FEATURE_RE = re.compile(
    r"^(?P<contig>[^:]+):(?P<pos>\d+):(?P<ref>[^:]+):(?P<alt>.+)$"
)


def _norm_allele(a: str) -> str:
    return str(a or "").strip().upper()


def parse_matrix_feature_id(feature_id: str) -> Optional[Tuple[str, int, str, str]]:
    m = _FEATURE_RE.match(str(feature_id).strip())
    if not m:
        return None
    return (
        m.group("contig"),
        int(m.group("pos")),
        _norm_allele(m.group("ref")),
        _norm_allele(m.group("alt")),
    )


def stage_should_seed_known_markers(stage_name: str, config: Any) -> bool:
    """True if this hierarchy stage is a phenotype endpoint eligible for seeding."""
    if not bool(getattr(config, "seed_known_markers", False)):
        return False
    stage = str(stage_name or "").lower()
    raw = getattr(
        config,
        "seed_known_markers_stage_substrings",
        ("amr", "resistance", "pheno", "profile", "resistant", "susceptible"),
    )
    if isinstance(raw, str):
        subs = [s.strip().lower() for s in raw.split(",") if s.strip()]
    else:
        subs = [str(s).strip().lower() for s in raw if str(s).strip()]
    if not subs:
        return True  # empty list = all stages
    return any(s in stage for s in subs)


def load_known_marker_keys(
    path: Path,
) -> Tuple[Set[Tuple[str, int, str, str]], Set[Tuple[int, str, str]], Dict[str, Any]]:
    """
    Load known markers from:
      - resistance catalogue TSV (Position, Ref, Alt, Contig/...)
      - plain text / CSV of Feature_ID lines Contig:Pos:Ref:Alt

    Returns:
      exact_keys: (contig, pos, ref, alt)
      pos_ref_alt: (pos, ref, alt) for contig-agnostic match
      meta: load stats
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"known markers file not found: {path}")

    exact: Set[Tuple[str, int, str, str]] = set()
    loose: Set[Tuple[int, str, str]] = set()
    n_rows = 0

    # Detect TSV with header
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        sample = f.read(4096)
        f.seek(0)
        has_header = "Position" in sample or "position" in sample or "Feature_ID" in sample
        if has_header and ("\t" in sample or "," in sample):
            delim = "\t" if sample.count("\t") >= sample.count(",") else ","
            reader = csv.DictReader(f, delimiter=delim)
            fields = {c.lower(): c for c in (reader.fieldnames or [])}

            def col(*names: str) -> Optional[str]:
                for n in names:
                    if n.lower() in fields:
                        return fields[n.lower()]
                return None

            c_pos = col("Position", "pos", "Pos")
            c_ref = col("Ref", "REF", "ref_allele", "Reference")
            c_alt = col("Alt", "ALT", "alt_allele", "Alternate")
            c_ctg = col("Contig", "CHROM", "Chrom", "chrom", "Source_chromosome")
            c_fid = col("Feature_ID", "feature_id", "feature", "id")

            for row in reader:
                n_rows += 1
                if c_fid and row.get(c_fid):
                    parsed = parse_matrix_feature_id(str(row[c_fid]))
                    if parsed:
                        exact.add(parsed)
                        loose.add((parsed[1], parsed[2], parsed[3]))
                        continue
                if not (c_pos and c_ref and c_alt):
                    continue
                try:
                    pos = int(float(str(row[c_pos]).strip()))
                except (TypeError, ValueError):
                    continue
                ref = _norm_allele(row.get(c_ref, ""))
                alt = _norm_allele(row.get(c_alt, ""))
                if not ref or not alt:
                    continue
                contig = str(row.get(c_ctg, "") or "").strip() if c_ctg else ""
                if contig:
                    exact.add((contig, pos, ref, alt))
                loose.add((pos, ref, alt))
        else:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                n_rows += 1
                # allow whitespace / comma separated first token
                token = re.split(r"[\s,;]+", line)[0]
                parsed = parse_matrix_feature_id(token)
                if parsed:
                    exact.add(parsed)
                    loose.add((parsed[1], parsed[2], parsed[3]))

    meta = {
        "path": str(path),
        "n_input_rows_or_lines": n_rows,
        "n_exact_keys": len(exact),
        "n_pos_ref_alt_keys": len(loose),
    }
    logger.info(
        "Loaded known markers | path=%s | exact=%d | pos_ref_alt=%d",
        path,
        len(exact),
        len(loose),
    )
    return exact, loose, meta


def match_known_markers_in_matrix(
    matrix_features: Sequence[str],
    exact_keys: Set[Tuple[str, int, str, str]],
    loose_keys: Set[Tuple[int, str, str]],
    contig_aliases: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Return matrix feature IDs that match the known-marker set (preserve matrix order)."""
    aliases = contig_aliases or {}
    matched: List[str] = []
    for fid in matrix_features:
        parsed = parse_matrix_feature_id(str(fid))
        if not parsed:
            continue
        contig, pos, ref, alt = parsed
        contig_norm = aliases.get(contig, contig)
        if (contig, pos, ref, alt) in exact_keys or (
            contig_norm,
            pos,
            ref,
            alt,
        ) in exact_keys:
            matched.append(str(fid))
            continue
        if (pos, ref, alt) in loose_keys:
            matched.append(str(fid))
    # unique preserve order
    seen = set()
    out: List[str] = []
    for f in matched:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def apply_known_marker_seed(
    ranked_features: List[str],
    matrix_columns: Sequence[str],
    config: Any,
    stage_name: str,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Reorder / annotate ranked features with optional known-marker seed.

    Modes:
      force_include — known markers first, then remaining statistical rank
      rank_boost    — same as force_include for ranking order (known first)

    Panel construction uses this reordered list as top-N source, so known
    markers occupy the first slots of every candidate panel size.
    """
    info: Dict[str, Any] = {
        "enabled": False,
        "applied": False,
        "stage_name": stage_name,
        "mode": None,
        "n_known_in_matrix": 0,
        "n_seeded": 0,
        "seeded_feature_ids": [],
        "reason": "seed_known_markers_disabled",
    }

    if not bool(getattr(config, "seed_known_markers", False)):
        return ranked_features, info

    if not stage_should_seed_known_markers(stage_name, config):
        info["reason"] = "stage_not_in_seed_known_markers_stage_substrings"
        info["enabled"] = True
        return ranked_features, info

    path = getattr(config, "known_markers_path", None) or getattr(
        config, "seed_known_markers_path", None
    )
    if not path:
        info["enabled"] = True
        info["reason"] = "known_markers_path_not_set"
        logger.warning(
            "seed_known_markers=True but known_markers_path is empty | stage=%s",
            stage_name,
        )
        return ranked_features, info

    info["enabled"] = True
    mode = str(getattr(config, "seed_known_markers_mode", "force_include") or "force_include")
    mode = mode.strip().lower()
    if mode not in {"force_include", "rank_boost"}:
        mode = "force_include"
    info["mode"] = mode

    try:
        exact, loose, load_meta = load_known_marker_keys(Path(str(path)))
        info["load"] = load_meta
    except Exception as exc:
        info["reason"] = f"load_failed:{exc}"
        logger.error("Failed to load known markers from %s: %s", path, exc)
        return ranked_features, info

    # Contig alias map from config if present
    aliases: Dict[str, str] = {}
    raw_alias = getattr(config, "vcf_contig_aliases", None) or getattr(
        config, "contig_aliases", None
    )
    if isinstance(raw_alias, dict):
        aliases = {str(k): str(v) for k, v in raw_alias.items()}

    cols = [str(c) for c in matrix_columns]
    known_in_matrix = match_known_markers_in_matrix(cols, exact, loose, aliases)

    max_n = getattr(config, "seed_known_markers_max", None)
    if max_n is not None:
        try:
            max_n_i = int(max_n)
            if max_n_i > 0:
                known_in_matrix = known_in_matrix[:max_n_i]
        except (TypeError, ValueError):
            pass

    info["n_known_in_matrix"] = len(known_in_matrix)
    info["seeded_feature_ids"] = list(known_in_matrix)
    info["n_seeded"] = len(known_in_matrix)

    if not known_in_matrix:
        info["reason"] = "no_known_markers_present_in_filtered_matrix"
        info["applied"] = False
        logger.info(
            "Known-marker seed: no catalogue hits in filtered matrix | stage=%s | path=%s",
            stage_name,
            path,
        )
        return ranked_features, info

    known_set = set(known_in_matrix)
    rest = [f for f in ranked_features if f not in known_set]
    # Prefer statistical order among known markers when possible
    known_ordered = [f for f in ranked_features if f in known_set]
    for f in known_in_matrix:
        if f not in known_ordered:
            known_ordered.append(f)

    reordered = known_ordered + rest
    info["applied"] = True
    info["reason"] = f"{mode}_seeded_{len(known_ordered)}_markers"
    logger.info(
        "Known-marker seed applied | stage=%s | mode=%s | seeded=%d | ranked_total=%d",
        stage_name,
        mode,
        len(known_ordered),
        len(reordered),
    )
    return reordered, info


def build_panel_with_forced_known(
    ranked_features: List[str],
    panel_size: int,
    known_seeded: Sequence[str],
    matrix_columns: Sequence[str],
) -> List[str]:
    """
    Build a top-N panel that always includes known markers (when present),
    then fills with statistical rank until panel_size.
    """
    colset = set(map(str, matrix_columns))
    known = [f for f in known_seeded if f in colset]
    if len(known) >= panel_size:
        return known[:panel_size]
    out = list(known)
    seen = set(out)
    for f in ranked_features:
        if f in seen or f not in colset:
            continue
        out.append(f)
        seen.add(f)
        if len(out) >= panel_size:
            break
    return out

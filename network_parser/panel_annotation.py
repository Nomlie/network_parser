#!/usr/bin/env python3
# network_parser/panel_annotation.py
"""
Annotate selected feature panels with genes, predicted consequences and
optional resistance-catalogue membership.

This module answers the biological follow-up question:

    Which genes, predicted variant consequences and known or candidate
    resistance-associated mechanisms are represented by (stable)
    branch-specific feature panels?

It does not re-run feature selection or change trained models. It joins
selected-feature manifests already written during training, optionally
filters to CV-stable markers, and optionally labels known catalogue hits.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

GENE_FIELD_CANDIDATES = (
    "Gene_annotation",
    "gene_annotation",
    "Gene",
    "gene",
    "locus_tag",
)
POS_CANDIDATES = ("Position", "position", "POS", "pos")
REF_CANDIDATES = ("Ref_allele", "REF", "ref", "Reference")
ALT_CANDIDATES = ("Alt_allele", "ALT", "alt", "Alternate")
AA_CANDIDATES = ("Amino_acid_change", "aa_change", "AA_change", "Protein_change")
NT_CANDIDATES = ("Nucleotide_change", "nt_change", "Nuc_change")
REGION_CANDIDATES = ("Region_type", "region_type", "Region")
FEATURE_ID_CANDIDATES = ("Feature_ID", "feature_id", "FeatureId", "feature")


def _first_present(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def parse_gene_annotation(text: Any) -> Dict[str, str]:
    """
    Parse GenBank-style gene annotation strings used in NetworkParser manifests.

    Example:
        -Rv3257c | pmmA | Probable phosphomannomutase ... | [3637312..3638709]
    """
    raw = (
        ""
        if text is None or (isinstance(text, float) and pd.isna(text))
        else str(text).strip()
    )
    if not raw or raw in {".", "-", "NA", "nan", "None"}:
        return {
            "strand": "",
            "locus_tag": "",
            "gene_name": "",
            "product": "",
            "gene_span": "",
            "gene_annotation_raw": raw,
        }

    parts = [p.strip() for p in raw.split("|")]
    strand = ""
    locus = ""
    gene_name = ""
    product = ""
    span = ""

    if parts:
        head = parts[0]
        m = re.match(r"^([+-])?\s*(Rv\w+|\S+)$", head)
        if m:
            strand = m.group(1) or ""
            locus = m.group(2) or head
        else:
            locus = head
    if len(parts) >= 2:
        gene_name = parts[1] if parts[1] not in {".", "-"} else ""
    if len(parts) >= 3:
        product = parts[2]
    if len(parts) >= 4:
        span = parts[3]

    return {
        "strand": strand,
        "locus_tag": locus,
        "gene_name": gene_name,
        "product": product,
        "gene_span": span,
        "gene_annotation_raw": raw,
    }


def classify_amino_acid_change(aa_change: Any) -> str:
    """Map Amino_acid_change strings like S|* or T|T into consequence classes."""
    raw = (
        ""
        if aa_change is None or (isinstance(aa_change, float) and pd.isna(aa_change))
        else str(aa_change).strip()
    )
    if not raw or raw in {".", "-", "NA", "nan", "None"}:
        return "unknown_or_noncoding"

    # Common manifest form: REF|ALT amino acids, possibly with *
    if "|" in raw:
        left, right = [x.strip() for x in raw.split("|", 1)]
    elif "/" in raw:
        left, right = [x.strip() for x in raw.split("/", 1)]
    else:
        # e.g. p.Ser450Leu-style: keep as unknown unless simple
        return "unknown_or_noncoding"

    if not left or not right:
        return "unknown_or_noncoding"
    if right in {"*", "X", "Ter", "ter"} or left in {"*", "X"}:
        return "nonsense"
    if left == right:
        return "synonymous"
    if len(left) == 1 and len(right) == 1:
        return "nonsynonymous"
    return "complex_or_indel"


def collect_manifest_paths_from_registry(
    registry: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Walk hierarchy/two-level registry dicts and collect selected-feature manifests."""
    found: List[Dict[str, Any]] = []

    def _add(path_value: Any, node_label: str, path_labels: Sequence[str]) -> None:
        if not path_value:
            return
        found.append(
            {
                "node_label": node_label,
                "hierarchy_path": " / ".join(
                    str(x) for x in path_labels if str(x).strip()
                ),
                "manifest_file": str(path_value),
            }
        )

    def walk_node(node: Dict[str, Any], path_labels: List[str]) -> None:
        if not isinstance(node, dict):
            return
        label = (
            node.get("label_value")
            or node.get("level1_group")
            or node.get("label_column")
            or node.get("name")
            or "node"
        )
        here = path_labels + [str(label)]
        fm = node.get("feature_manifest", {})
        if isinstance(fm, dict):
            _add(fm.get("manifest_file"), str(label), here)
        elif isinstance(fm, str):
            _add(fm, str(label), here)
        # direct keys
        for key in ("selected_feature_manifest", "manifest_file"):
            if key in node:
                _add(node.get(key), str(label), here)

        children = node.get("children", {})
        if isinstance(children, dict):
            for child in children.values():
                if isinstance(child, dict):
                    walk_node(child, here)
        elif isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    walk_node(child, here)

    def walk_fallback_tree(obj: Any, path_labels: List[str], node_label: str) -> None:
        """Recursively collect manifests from nested terminal-fallback payloads."""
        if not isinstance(obj, dict):
            return
        fm = obj.get("feature_manifest", {})
        if isinstance(fm, dict):
            _add(fm.get("manifest_file"), node_label, path_labels)
        elif isinstance(fm, str):
            _add(fm, node_label, path_labels)
        for key in ("selected_feature_manifest", "manifest_file"):
            if key in obj:
                _add(obj.get(key), node_label, path_labels)
        # Recurse into nested dict/list payloads (global, by_parent_label, children, ...)
        skip_keys = {
            "feature_manifest",
            "selected_feature_manifest",
            "manifest_file",
            "model_file",
            "model_path",
            "config",
        }
        for child_name, child in obj.items():
            if child_name in skip_keys:
                continue
            if isinstance(child, dict):
                walk_fallback_tree(
                    child,
                    path_labels + [str(child_name)],
                    f"{node_label}::{child_name}",
                )
            elif isinstance(child, list):
                for i, item in enumerate(child):
                    if isinstance(item, dict):
                        walk_fallback_tree(
                            item,
                            path_labels + [str(child_name), str(i)],
                            f"{node_label}::{child_name}::{i}",
                        )

    # Multi-level hierarchy
    hierarchy = registry.get("hierarchy", {})
    if isinstance(hierarchy, dict):
        root = hierarchy.get("root", hierarchy)
        if isinstance(root, dict):
            walk_node(root, [])
        term = hierarchy.get("terminal_fallbacks", {})
        if isinstance(term, dict):
            # Recurse entire terminal_fallbacks tree (global, by_parent, nested)
            walk_fallback_tree(term, ["fallback"], "terminal_fallback")

    # Classic two-level / level1-level2 registry layout
    level1 = registry.get("level1", {})
    if isinstance(level1, dict):
        fm = level1.get("feature_manifest", {})
        if isinstance(fm, dict):
            _add(fm.get("manifest_file"), "level1", ["level1"])
    level2 = registry.get("level2", {})
    if isinstance(level2, dict):
        g = level2.get("global_fallback", {})
        if isinstance(g, dict):
            fm = g.get("feature_manifest", {})
            if isinstance(fm, dict):
                _add(
                    fm.get("manifest_file"),
                    "level2_global_fallback",
                    ["level2", "global_fallback"],
                )
        by_group = level2.get("by_level1_group", {})
        if isinstance(by_group, dict):
            for group_name, payload in by_group.items():
                if not isinstance(payload, dict):
                    continue
                fm = payload.get("feature_manifest", {})
                if isinstance(fm, dict):
                    _add(
                        fm.get("manifest_file"),
                        f"level2::{group_name}",
                        ["level2", str(group_name)],
                    )

    # Deduplicate by resolved path string
    uniq: Dict[str, Dict[str, Any]] = {}
    for row in found:
        key = f"{row['manifest_file']}::{row['node_label']}"
        uniq[key] = row
    return list(uniq.values())


def annotate_manifest_table(
    manifest: pd.DataFrame,
    *,
    node_label: str = "",
    hierarchy_path: str = "",
    catalogue: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Add gene / consequence / catalogue columns to one selected-feature manifest."""
    if manifest is None or manifest.empty:
        return pd.DataFrame()

    df = manifest.copy()
    gene_col = _first_present(df.columns, GENE_FIELD_CANDIDATES)
    aa_col = _first_present(df.columns, AA_CANDIDATES)
    nt_col = _first_present(df.columns, NT_CANDIDATES)
    region_col = _first_present(df.columns, REGION_CANDIDATES)
    pos_col = _first_present(df.columns, POS_CANDIDATES)
    ref_col = _first_present(df.columns, REF_CANDIDATES)
    alt_col = _first_present(df.columns, ALT_CANDIDATES)
    fid_col = _first_present(df.columns, FEATURE_ID_CANDIDATES)

    gene_parts = (
        df[gene_col].map(parse_gene_annotation)
        if gene_col
        else pd.Series([{}] * len(df))
    )
    gene_df = pd.DataFrame(list(gene_parts))
    out = pd.concat([df.reset_index(drop=True), gene_df.reset_index(drop=True)], axis=1)
    out["node_label"] = node_label
    out["hierarchy_path"] = hierarchy_path
    out["consequence_class"] = (
        out[aa_col].map(classify_amino_acid_change)
        if aa_col
        else "unknown_or_noncoding"
    )
    if region_col:
        region = out[region_col].astype(str).str.lower()
        noncoding_mask = region.str.contains(
            "intergenic|non.?coding|upstream|downstream|promoter", na=False
        )
        out.loc[
            noncoding_mask & (out["consequence_class"] == "unknown_or_noncoding"),
            "consequence_class",
        ] = "noncoding"

    def _norm_gene_id(g: Any) -> str:
        return str(g or "").strip().upper()

    # Optional catalogue join (build + contig + position + REF + ALT when available)
    out["catalogue_status"] = "not_in_catalogue"
    out["catalogue_match_type"] = ""
    out["catalogue_note"] = ""
    if catalogue is not None and not catalogue.empty:
        cat = catalogue.copy()
        cat_cols = {c.lower(): c for c in cat.columns}
        pos_c = (
            cat_cols.get("position")
            or cat_cols.get("pos")
            or cat_cols.get("genome_position")
        )
        ref_c = (
            cat_cols.get("ref")
            or cat_cols.get("ref_allele")
            or cat_cols.get("reference")
        )
        alt_c = (
            cat_cols.get("alt")
            or cat_cols.get("alt_allele")
            or cat_cols.get("alternate")
        )
        gene_c = (
            cat_cols.get("gene")
            or cat_cols.get("gene_name")
            or cat_cols.get("locus_tag")
        )
        note_c = (
            cat_cols.get("drug")
            or cat_cols.get("mechanism")
            or cat_cols.get("note")
            or cat_cols.get("annotation")
        )
        contig_c = (
            cat_cols.get("contig")
            or cat_cols.get("sequence")
            or cat_cols.get("chrom")
            or cat_cols.get("chromosome")
            or cat_cols.get("seqid")
        )
        build_c = (
            cat_cols.get("reference_build")
            or cat_cols.get("build")
            or cat_cols.get("ref_build")
            or cat_cols.get("genome_build")
        )
        seq_col = _first_present(
            out.columns, ("Sequence", "sequence", "Contig", "contig", "Chromosome")
        )
        build_col = _first_present(
            out.columns, ("Reference_build", "reference_build", "Build")
        )

        if pos_c and pos_col:
            left = out.copy()
            left["_pos_key"] = pd.to_numeric(left[pos_col], errors="coerce").astype(
                "Int64"
            )
            cat = cat.copy()
            cat["_pos_key"] = pd.to_numeric(cat[pos_c], errors="coerce").astype("Int64")
            if contig_c:
                cat["_contig_key"] = cat[contig_c].astype(str).str.strip()
            if build_c:
                cat["_build_key"] = cat[build_c].astype(str).str.strip()

            def _filter_hits_by_contig_build(
                hits: pd.DataFrame, row: pd.Series
            ) -> Tuple[pd.DataFrame, str]:
                """Return (filtered_hits, status_if_empty). Empty hits => mismatch/unverified."""
                local = hits
                if contig_c:
                    if not seq_col:
                        return local.iloc[0:0], "candidate_unverified"
                    contig_val = str(row.get(seq_col, "")).strip()
                    if not contig_val:
                        return local.iloc[0:0], "candidate_unverified"
                    if "_contig_key" in local.columns:
                        local = local.loc[
                            local["_contig_key"].astype(str).str.strip() == contig_val
                        ]
                    if local.empty:
                        return local, "build_or_contig_mismatch"
                if build_c:
                    if not build_col:
                        # Catalogue declares build; panel lacks declared build → not exact
                        return local.iloc[0:0], "candidate_unverified"
                    build_val = str(row.get(build_col, "")).strip()
                    if not build_val:
                        return local.iloc[0:0], "candidate_unverified"
                    if "_build_key" in local.columns:
                        local = local.loc[
                            local["_build_key"].astype(str).str.strip() == build_val
                        ]
                    if local.empty:
                        return local, "build_or_contig_mismatch"
                return local, ""

            if ref_c and alt_c and ref_col and alt_col:
                left["_allele_key"] = (
                    left["_pos_key"].astype(str)
                    + ":"
                    + left[ref_col].astype(str).str.upper()
                    + ">"
                    + left[alt_col].astype(str).str.upper()
                )
                cat["_allele_key"] = (
                    cat["_pos_key"].astype(str)
                    + ":"
                    + cat[ref_c].astype(str).str.upper()
                    + ">"
                    + cat[alt_c].astype(str).str.upper()
                )
                for i, row in left.iterrows():
                    allele_key = str(row.get("_allele_key", ""))
                    hits = cat.loc[cat["_allele_key"] == allele_key]
                    if hits.empty:
                        # Gene-level candidate only with contig/build compatibility when present
                        if gene_c and (
                            (
                                row.get("locus_tag") is not None
                                and str(row.get("locus_tag")).strip()
                            )
                            or (
                                row.get("gene_name") is not None
                                and str(row.get("gene_name")).strip()
                            )
                        ):
                            genes = {
                                _norm_gene_id(row.get("locus_tag")),
                                _norm_gene_id(row.get("gene_name")),
                            }
                            genes = {
                                g for g in genes if g and g not in {".", "-", "NAN"}
                            }
                            cat_genes = {
                                _norm_gene_id(x) for x in cat[gene_c].dropna().tolist()
                            }
                            if genes & cat_genes:
                                gene_hits = cat.loc[
                                    cat[gene_c].map(_norm_gene_id).isin(genes)
                                ]
                                gene_hits, st = _filter_hits_by_contig_build(
                                    gene_hits, row
                                )
                                if (
                                    not gene_hits.empty
                                    and out.at[i, "catalogue_status"]
                                    == "not_in_catalogue"
                                ):
                                    out.at[i, "catalogue_status"] = "candidate_gene"
                                    out.at[i, "catalogue_match_type"] = "gene_name"
                                elif (
                                    st == "candidate_unverified"
                                    and out.at[i, "catalogue_status"]
                                    == "not_in_catalogue"
                                ):
                                    out.at[
                                        i, "catalogue_status"
                                    ] = "candidate_unverified"
                                    out.at[
                                        i, "catalogue_match_type"
                                    ] = "gene_name_unverified_build_contig"
                        continue

                    # Exact known_mutation requires compatible build, contig, position, REF, ALT
                    hits, st = _filter_hits_by_contig_build(hits, row)
                    if hits.empty:
                        out.at[i, "catalogue_status"] = st or "candidate_unverified"
                        out.at[i, "catalogue_match_type"] = (
                            st or "incomplete_build_contig"
                        )
                        continue

                    out.at[i, "catalogue_status"] = "known_mutation"
                    out.at[i, "catalogue_match_type"] = "position_ref_alt_contig_build"
                    if note_c is not None and len(hits):
                        out.at[i, "catalogue_note"] = str(hits.iloc[0][note_c])
            else:
                # Position-only must NEVER be labelled known_mutation
                exact_pos = set(cat["_pos_key"].dropna().tolist())
                for i, row in left.iterrows():
                    if row.get("_pos_key") in exact_pos:
                        out.at[i, "catalogue_status"] = "candidate_unverified"
                        out.at[i, "catalogue_match_type"] = "position_only_not_exact"
        elif gene_c:
            cat_genes = {_norm_gene_id(x) for x in cat[gene_c].dropna().tolist()}
            for i, row in out.iterrows():
                genes = {
                    _norm_gene_id(row.get("locus_tag")),
                    _norm_gene_id(row.get("gene_name")),
                }
                genes = {g for g in genes if g and g not in {".", "-", "NAN"}}
                if genes & cat_genes:
                    gene_hits = cat.loc[cat[gene_c].map(_norm_gene_id).isin(genes)]
                    compatible = gene_hits
                    incomplete_identity = False
                    if contig_c:
                        contig_value = (
                            str(row.get(seq_col, "")).strip() if seq_col else ""
                        )
                        if not contig_value:
                            incomplete_identity = True
                        else:
                            compatible = compatible.loc[
                                compatible[contig_c].astype(str).str.strip()
                                == contig_value
                            ]
                    if build_c:
                        build_value = (
                            str(row.get(build_col, "")).strip() if build_col else ""
                        )
                        if not build_value:
                            incomplete_identity = True
                        else:
                            compatible = compatible.loc[
                                compatible[build_c].astype(str).str.strip()
                                == build_value
                            ]
                    if incomplete_identity:
                        out.at[i, "catalogue_status"] = "candidate_unverified"
                        out.at[
                            i, "catalogue_match_type"
                        ] = "gene_name_unverified_build_contig"
                    elif not compatible.empty:
                        out.at[i, "catalogue_status"] = "candidate_gene"
                        out.at[
                            i, "catalogue_match_type"
                        ] = "gene_name_build_contig_compatible"
                    else:
                        out.at[i, "catalogue_status"] = "not_in_catalogue"
                        out.at[
                            i, "catalogue_match_type"
                        ] = "gene_name_build_contig_mismatch"

    keep_front = [
        "node_label",
        "hierarchy_path",
        fid_col,
        pos_col,
        ref_col,
        alt_col,
        "locus_tag",
        "gene_name",
        "product",
        region_col,
        nt_col,
        aa_col,
        "consequence_class",
        "catalogue_status",
        "catalogue_match_type",
        "catalogue_note",
    ]
    keep_front = [c for c in keep_front if c and c in out.columns]
    other = [c for c in out.columns if c not in keep_front]
    return out[keep_front + other]


def summarise_annotations(annotated: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Build compact summary tables from annotated panel features."""
    if annotated is None or annotated.empty:
        empty = pd.DataFrame()
        return {
            "by_gene": empty,
            "by_consequence": empty,
            "by_catalogue": empty,
            "by_node_gene": empty,
        }

    df = annotated.copy()
    by_gene = (
        df.groupby(["locus_tag", "gene_name", "product"], dropna=False)
        .size()
        .reset_index(name="n_features")
        .sort_values("n_features", ascending=False)
    )
    by_consequence = (
        df.groupby(["consequence_class"], dropna=False)
        .size()
        .reset_index(name="n_features")
        .sort_values("n_features", ascending=False)
    )
    by_catalogue = (
        df.groupby(["catalogue_status"], dropna=False)
        .size()
        .reset_index(name="n_features")
        .sort_values("n_features", ascending=False)
    )
    by_node_gene = (
        df.groupby(
            [
                "node_label",
                "hierarchy_path",
                "locus_tag",
                "gene_name",
                "consequence_class",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="n_features")
        .sort_values(["node_label", "n_features"], ascending=[True, False])
    )
    return {
        "by_gene": by_gene,
        "by_consequence": by_consequence,
        "by_catalogue": by_catalogue,
        "by_node_gene": by_node_gene,
    }


def load_registry(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_manifest_path(path_value: Path, registry_base: Path) -> Optional[Path]:
    """
    Resolve a registry manifest path that may be absolute, relative, or stale.

    Tries, in order:
    1. path as given (if absolute and exists)
    2. registry_base / path
    3. registry_base / path.name under common subtrees
    4. suffix match for .../hierarchy_models/.../selected_feature_manifest.tsv
    """
    candidates: List[Path] = []
    raw = Path(path_value)
    if raw.is_absolute():
        candidates.append(raw)
        # Also try re-rooting absolute paths that moved (keep from hierarchy_models onward)
        parts = list(raw.parts)
        for marker in (
            "hierarchy_models",
            "level1_strain_identity",
            "level2_resistance_profile",
            "matrices",
        ):
            if marker in parts:
                idx = parts.index(marker)
                rel = Path(*parts[idx:])
                candidates.append(registry_base / rel)
                candidates.append(registry_base / "training" / rel)
                break
    else:
        candidates.append(registry_base / raw)
        candidates.append(registry_base / "training" / raw)

    candidates.append(registry_base / raw.name)

    for cand in candidates:
        try:
            if cand.exists():
                return cand.resolve()
        except OSError:
            continue

    # Do not rglob for manifests — recursive search can select stale files.
    # Manifests must resolve through the registry path / re-rooted candidates above.
    return None


def annotate_registry_panels(
    registry_path: Path,
    output_dir: Path,
    *,
    catalogue_path: Optional[Path] = None,
    stability_path: Optional[Path] = None,
    min_stability: float = 0.0,
    write_stable_report: bool = False,
    write_catalogue_circularity: bool = False,
) -> Dict[str, Any]:
    """
    Annotate all selected-feature manifests referenced by a trained registry.

    Parameters
    ----------
    registry_path:
        hierarchical_model_registry.json or two_level_model_registry.json
    output_dir:
        Directory for annotated feature tables and summaries
    catalogue_path:
        Optional TSV/CSV of known resistance mutations/genes
    stability_path:
        Optional TSV/CSV with columns Feature_ID (or feature) and selection_frequency (0-1)
    min_stability:
        Keep only features with selection_frequency >= threshold when stability_path is set
    """
    registry_path = Path(registry_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = load_registry(registry_path)
    registry_base = registry_path.parent

    catalogue = None
    if catalogue_path is not None:
        catalogue_path = Path(catalogue_path)
        sep = "\t" if catalogue_path.suffix.lower() in {".tsv", ".txt"} else ","
        catalogue = pd.read_csv(catalogue_path, sep=sep)
        logger.info(
            "Loaded catalogue | rows=%d | file=%s", len(catalogue), catalogue_path
        )

    stability = None
    if stability_path is not None:
        stability_path = Path(stability_path)
        sep = "\t" if stability_path.suffix.lower() in {".tsv", ".txt"} else ","
        stability = pd.read_csv(stability_path, sep=sep)
        logger.info(
            "Loaded stability table | rows=%d | file=%s", len(stability), stability_path
        )

    manifest_rows = collect_manifest_paths_from_registry(registry)
    annotated_parts: List[pd.DataFrame] = []
    missing: List[str] = []

    for row in manifest_rows:
        mpath = _resolve_manifest_path(Path(row["manifest_file"]), registry_base)
        if mpath is None or not mpath.exists():
            missing.append(str(row["manifest_file"]))
            logger.warning("Missing manifest: %s", row["manifest_file"])
            continue
        sep = "\t" if mpath.suffix.lower() in {".tsv", ".txt"} else ","
        manifest = pd.read_csv(mpath, sep=sep)
        ann = annotate_manifest_table(
            manifest,
            node_label=row["node_label"],
            hierarchy_path=row["hierarchy_path"],
            catalogue=catalogue,
        )
        if stability is not None and not ann.empty:
            fid = _first_present(ann.columns, FEATURE_ID_CANDIDATES)
            s_fid = _first_present(
                stability.columns, FEATURE_ID_CANDIDATES + ("feature",)
            )
            s_freq = _first_present(
                stability.columns,
                ("selection_frequency", "frequency", "freq", "stability"),
            )
            if fid and s_fid and s_freq:
                stab = stability[[s_fid, s_freq]].copy()
                stab = stab.rename(columns={s_fid: fid, s_freq: "selection_frequency"})
                ann = ann.merge(stab, on=fid, how="left")
                ann["selection_frequency"] = pd.to_numeric(
                    ann["selection_frequency"], errors="coerce"
                ).fillna(0.0)
                if min_stability > 0:
                    ann = ann.loc[
                        ann["selection_frequency"] >= float(min_stability)
                    ].copy()
                    ann["stability_filter"] = f">={min_stability}"
        annotated_parts.append(ann)

    if not annotated_parts:
        raise FileNotFoundError(
            "No selected-feature manifests could be loaded from the registry. "
            f"Missing examples: {missing[:5]}"
        )

    annotated = pd.concat(annotated_parts, ignore_index=True, sort=False)
    annotated_path = output_dir / "panel_features_annotated.tsv"
    annotated.to_csv(annotated_path, sep="\t", index=False)

    summaries = summarise_annotations(annotated)
    for name, table in summaries.items():
        table.to_csv(output_dir / f"panel_summary_{name}.tsv", sep="\t", index=False)

    report = {
        "registry": str(registry_path),
        "n_manifests_listed": len(manifest_rows),
        "n_manifests_loaded": len(annotated_parts),
        "n_features_annotated": int(len(annotated)),
        "n_missing_manifests": len(missing),
        "missing_manifests": missing[:50],
        "outputs": {
            "annotated_features": str(annotated_path),
            **{
                f"summary_{k}": str(output_dir / f"panel_summary_{k}.tsv")
                for k in summaries
            },
        },
    }

    # Full selected-panel table is always written. Optionally also write a
    # CV-stable subset when frequencies are present (even if min_stability was 0
    # during the main filter step).
    if write_stable_report and "selection_frequency" in annotated.columns:
        stable_thr = float(min_stability) if float(min_stability) > 0 else 0.5
        stable = annotated.loc[
            pd.to_numeric(annotated["selection_frequency"], errors="coerce").fillna(0)
            >= stable_thr
        ].copy()
        stable_path = output_dir / "stable_panel_features_annotated.tsv"
        stable.to_csv(stable_path, sep="\t", index=False)
        report["outputs"]["stable_annotated_features"] = str(stable_path)
        report["stable_report"] = {
            "min_selection_frequency": stable_thr,
            "n_stable_features": int(len(stable)),
        }
        logger.info(
            "Stable panel report | thr=%.2f | n=%d | %s",
            stable_thr,
            len(stable),
            stable_path,
        )

    if write_catalogue_circularity:
        try:
            from network_parser.hierarchy_artifacts import (
                write_catalogue_circularity_report,
            )
        except ImportError:  # pragma: no cover
            from hierarchy_artifacts import write_catalogue_circularity_report  # type: ignore
        circ = write_catalogue_circularity_report(annotated_path, output_dir)
        report["outputs"]["catalogue_circularity"] = str(circ)

    report_path = output_dir / "panel_annotation_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    logger.info(
        "Panel annotation complete | features=%d | manifests=%d | out=%s",
        len(annotated),
        len(annotated_parts),
        output_dir,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Annotate NetworkParser selected feature panels with genes, "
            "predicted consequences and optional resistance-catalogue labels."
        )
    )
    p.add_argument(
        "--registry",
        required=True,
        help="Path to hierarchical_model_registry.json or two_level_model_registry.json",
    )
    p.add_argument(
        "--output_dir", required=True, help="Directory for annotation outputs"
    )
    p.add_argument(
        "--catalogue",
        default=None,
        help="Optional TSV/CSV of known resistance mutations/genes (flexible column names).",
    )
    p.add_argument(
        "--stability",
        default=None,
        help="Optional TSV/CSV with Feature_ID and selection_frequency from leakage-aware CV.",
    )
    p.add_argument(
        "--min_stability",
        type=float,
        default=0.0,
        help="If --stability is set, keep features with selection_frequency >= this value (0-1).",
    )
    p.add_argument(
        "--write_stable_report",
        action="store_true",
        help="Also write stable_panel_features_annotated.tsv (default thr 0.5 if min_stability=0).",
    )
    p.add_argument(
        "--write_catalogue_circularity",
        action="store_true",
        help="Write catalogue circularity audit (known vs non-catalogue fraction by node).",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    annotate_registry_panels(
        registry_path=Path(args.registry),
        output_dir=Path(args.output_dir),
        catalogue_path=Path(args.catalogue) if args.catalogue else None,
        stability_path=Path(args.stability) if args.stability else None,
        min_stability=float(args.min_stability),
        write_stable_report=bool(getattr(args, "write_stable_report", False)),
        write_catalogue_circularity=bool(
            getattr(args, "write_catalogue_circularity", False)
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

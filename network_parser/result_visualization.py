"""Metadata-hierarchy tables used by the sample-separation report.

The functions in this module deliberately produce data-first TSV/JSON/HTML
artifacts.  They do not infer a phylogeny and do not alter model inputs.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd


MISSING_LABEL = "[missing]"


def parse_visualization_label_columns(columns: Any) -> List[str]:
    """Return unique, non-empty hierarchy column names in declared order."""
    if columns is None:
        return []
    if isinstance(columns, str):
        raw = columns.replace("->", ",").replace("→", ",").split(",")
    else:
        raw = list(columns)

    parsed: List[str] = []
    for value in raw:
        name = str(value).strip()
        if name and name not in parsed:
            parsed.append(name)
    return parsed


def _metadata_with_sample_ids(meta: pd.DataFrame) -> pd.DataFrame:
    frame = meta.copy()
    if "sample_id" in frame.columns:
        ids = frame["sample_id"]
    elif "Isolate" in frame.columns:
        ids = frame["Isolate"]
    elif "isolate" in frame.columns:
        ids = frame["isolate"]
    else:
        ids = pd.Series(frame.index, index=frame.index)
    frame["sample_id"] = ids.astype(str).str.strip()
    return frame.drop_duplicates(subset=["sample_id"], keep="first")


def build_aligned_metadata_hierarchy_frame(
    *,
    meta: pd.DataFrame,
    sample_ids: Sequence[Any],
    hierarchy_columns: Sequence[str],
    include_missing: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Align declared metadata labels to the requested sample order."""
    columns = parse_visualization_label_columns(hierarchy_columns)
    metadata = _metadata_with_sample_ids(meta)
    available = [column for column in columns if column in metadata.columns]
    missing_columns = [column for column in columns if column not in metadata.columns]
    requested_ids = [str(value).strip() for value in sample_ids]

    aligned = (
        metadata.set_index("sample_id", drop=False)
        .reindex(requested_ids)
        .loc[:, ["sample_id", *available]]
    )
    aligned["sample_id"] = requested_ids
    for column in available:
        values = aligned[column].astype("string").str.strip()
        values = values.mask(values.eq(""))
        aligned[column] = values.fillna(MISSING_LABEL)

    missing_metadata = int((~pd.Index(requested_ids).isin(metadata["sample_id"])).sum())
    if not include_missing and available:
        aligned = aligned.loc[~aligned[available].eq(MISSING_LABEL).any(axis=1)].copy()

    summary: Dict[str, Any] = {
        "requested_samples": len(requested_ids),
        "aligned_samples": int(aligned.shape[0]),
        "samples_without_metadata": missing_metadata,
        "requested_columns": columns,
        "visualized_columns": available,
        "missing_columns": missing_columns,
        "include_missing": bool(include_missing),
    }
    return aligned.reset_index(drop=True), summary


def _hierarchy_tables(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    count_rows: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []
    for level_index, column in enumerate(columns, start=1):
        counts = frame.groupby(column, dropna=False).size()
        for label, count in counts.items():
            count_rows.append(
                {
                    "level_index": level_index,
                    "level_name": column,
                    "label": str(label),
                    "n_samples": int(count),
                }
            )
        if level_index > 1:
            parent = columns[level_index - 2]
            edges = frame.groupby([parent, column], dropna=False).size()
            for (parent_label, child_label), count in edges.items():
                edge_rows.append(
                    {
                        "parent_level": parent,
                        "parent_label": str(parent_label),
                        "child_level": column,
                        "child_label": str(child_label),
                        "n_samples": int(count),
                    }
                )

    path_counts = (
        frame.groupby(list(columns), dropna=False)
        .size()
        .rename("n_samples")
        .reset_index()
        if columns
        else pd.DataFrame(columns=["n_samples"])
    )
    return pd.DataFrame(count_rows), pd.DataFrame(edge_rows), path_counts


def write_sample_hierarchy_visualizations(
    *,
    meta: pd.DataFrame,
    sample_ids: Sequence[Any],
    hierarchy_columns: Sequence[str],
    output_dir: Path,
    prefix: str = "label_hierarchy",
    include_missing: bool = True,
    max_categories_per_level: int = 30,
) -> Dict[str, Any]:
    """Write auditable label hierarchy tables and a compact HTML view."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    aligned, alignment = build_aligned_metadata_hierarchy_frame(
        meta=meta,
        sample_ids=sample_ids,
        hierarchy_columns=hierarchy_columns,
        include_missing=include_missing,
    )
    columns = list(alignment["visualized_columns"])
    counts, edges, paths = _hierarchy_tables(aligned, columns)

    aligned_path = output / f"{prefix}_aligned_labels.tsv"
    counts_path = output / f"{prefix}_level_counts.tsv"
    edges_path = output / f"{prefix}_edges.tsv"
    paths_path = output / f"{prefix}_paths.tsv"
    html_path = output / f"{prefix}.html"
    summary_path = output / f"{prefix}_summary.json"
    aligned.to_csv(aligned_path, sep="\t", index=False)
    counts.to_csv(counts_path, sep="\t", index=False)
    edges.to_csv(edges_path, sep="\t", index=False)
    paths.to_csv(paths_path, sep="\t", index=False)

    displayed = counts.groupby("level_name", sort=False).head(
        max(1, int(max_categories_per_level))
    )
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>NetworkParser label hierarchy</title></head><body>"
        "<h1>Metadata label hierarchy</h1>"
        f"<p>{html.escape(' → '.join(columns))}</p>"
        f"{displayed.to_html(index=False, escape=True)}"
        "<p>This is a metadata hierarchy, not a phylogenetic tree.</p>"
        "</body></html>",
        encoding="utf-8",
    )

    summary: Dict[str, Any] = {
        "status": "success",
        **alignment,
        "max_categories_per_level": int(max_categories_per_level),
        "artifacts": {
            "aligned_labels_tsv": str(aligned_path),
            "level_counts_tsv": str(counts_path),
            "edges_tsv": str(edges_path),
            "paths_tsv": str(paths_path),
            "html": str(html_path),
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

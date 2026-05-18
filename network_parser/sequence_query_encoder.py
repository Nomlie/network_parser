#!/usr/bin/env python3
# network_parser/sequence_query_encoder.py
"""
Raw-sequence query encoder for NetworkParser
===========================================

Purpose
-------
Convert a user-provided raw DNA FASTA file into the exact selected-feature
matrix expected by a trained NetworkParser model registry.

The encoder uses the feature manifest generated during training. For each
selected genomic feature it uses the saved reference-context sequence to locate
the homologous site in the query DNA sequence, extracts the query nucleotide at
the feature centre, and encodes it using the same baseline definition as the
training matrix.

No statistical filtering, model training, decision-tree fitting, or bootstrap
confidence computation happens here. This is inference-time matrix construction.

Design rule
-----------
A query-time feature is only activated when the context mapping is sufficiently
traceable and the observed allele belongs to the allele states seen during
training. Ambiguous bases, unresolved contexts, and repeated/multi-hit contexts
are filled as 0 and explicitly reported in the per-feature call table.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

FASTA_SUFFIXES = {".fa", ".fna", ".fasta", ".fas"}
VALID_UNAMBIGUOUS_BASES = {"A", "C", "G", "T"}
_DNA_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


# -----------------------------------------------------------------------------
# Small IO helpers
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def reverse_complement(seq: str) -> str:
    return seq.translate(_DNA_COMP)[::-1].upper()


def complement_base(base: str) -> str:
    if not base:
        return ""
    return base.translate(_DNA_COMP).upper()


def is_unambiguous_base(base: Any) -> bool:
    return str(base or "").upper() in VALID_UNAMBIGUOUS_BASES


def read_fasta_records(path: Path) -> Dict[str, str]:
    """Read FASTA records without requiring Biopython."""
    records: Dict[str, List[str]] = {}
    name: Optional[str] = None

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:].split()[0] or path.stem
                records.setdefault(name, [])
                continue
            if name is None:
                name = path.stem
                records.setdefault(name, [])
            records[name].append(line.upper())

    return {key: "".join(chunks).upper() for key, chunks in records.items() if chunks}


def discover_fasta_inputs(path: str) -> List[Path]:
    p = Path(path)
    if p.is_dir():
        return sorted([x for x in p.iterdir() if x.suffix.lower() in FASTA_SUFFIXES])
    if p.exists():
        return [p]
    raise FileNotFoundError(f"Raw sequence input not found: {p}")


# -----------------------------------------------------------------------------
# Manifest handling
# -----------------------------------------------------------------------------

def load_feature_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    if "Feature_ID" not in manifest.columns:
        if "feature" in manifest.columns:
            manifest["Feature_ID"] = manifest["feature"].astype(str)
        elif "Position" in manifest.columns:
            manifest["Feature_ID"] = manifest["Position"].astype(str)
        else:
            raise ValueError("Feature manifest must contain Feature_ID, feature, or Position.")
    manifest["Feature_ID"] = manifest["Feature_ID"].astype(str)
    return manifest.drop_duplicates(subset=["Feature_ID"], keep="first")


def manifest_rows_for_features(manifest: pd.DataFrame, features: Sequence[str]) -> List[Dict[str, str]]:
    lookup = manifest.set_index("Feature_ID", drop=False)
    rows: List[Dict[str, str]] = []
    for feature in [str(f) for f in features]:
        if feature in lookup.index:
            row = {str(k): str(v) for k, v in lookup.loc[feature].to_dict().items()}
            row["Feature_ID"] = feature
            rows.append(row)
        else:
            rows.append({"Feature_ID": feature})
    return rows


def parse_feature_id(feature_id: str) -> Tuple[str, str, str, str]:
    parts = str(feature_id).split(":")
    if len(parts) >= 4:
        return parts[0], parts[1], parts[2].upper(), parts[3].upper()
    return "", "", "", ""


def normalise_manifest_row(row: Dict[str, str]) -> Dict[str, str]:
    """
    Normalize a manifest record into the fields required for query encoding.

    The selected-feature manifest is the bridge from training to inference. It
    should carry Feature_ID, reference/alternate allele, baseline allele,
    optional annotation, and the context sequence used for raw FASTA mapping.
    """
    feature_id = str(row.get("Feature_ID", ""))
    chrom, pos, ref_from_id, alt_from_id = parse_feature_id(feature_id)

    context = ""
    for key in ("Context_±40", "Context", "Context_sequence", "context_sequence"):
        if str(row.get(key, "")).strip():
            context = str(row.get(key, "")).strip().upper()
            break

    center = str(row.get("Context_center_offset", "")).strip()
    if center:
        try:
            center_offset = int(float(center))
        except Exception:
            center_offset = len(context) // 2 if context else -1
    else:
        center_offset = len(context) // 2 if context else -1

    ref = str(row.get("Ref_allele", row.get("REF", ref_from_id))).strip().upper() or ref_from_id
    alt = str(row.get("Alt_allele", row.get("ALT", alt_from_id))).strip().upper() or alt_from_id
    baseline = str(row.get("Baseline_allele", row.get("baseline_allele", ref))).strip().upper() or ref

    return {
        **row,
        "Feature_ID": feature_id,
        "Sequence": str(row.get("Sequence", chrom)).strip() or chrom,
        "Position": str(row.get("Position", pos)).strip() or pos,
        "Ref_allele": ref,
        "Alt_allele": alt,
        "Baseline_allele": baseline,
        "Context_sequence": context,
        "Context_center_offset": str(center_offset),
    }


# -----------------------------------------------------------------------------
# Exact context mapping
# -----------------------------------------------------------------------------

def _context_hit_record(
    *,
    record_id: str,
    idx: int,
    pattern_len: int,
    center_offset: int,
    strand: str,
    observed_raw: str,
) -> Dict[str, Any]:
    # ``center_offset`` is the centre coordinate in the pattern that was
    # actually searched. For reverse-complement searches this is adjusted
    # before calling this helper, so the subject coordinate is always the
    # matched centre position in the query sequence. The allele itself is
    # complemented on minus-strand hits to return it in the reference-feature
    # orientation.
    subject_position = idx + center_offset + 1
    observed = complement_base(observed_raw) if strand == "minus" else observed_raw.upper()

    return {
        "status": "mapped_unique_context",
        "method": "exact_flanking_context",
        "subject_id": record_id,
        "subject_position": int(subject_position),
        "strand": strand,
        "observed_allele": observed,
    }


def collect_flanking_context_hits(
    records: Dict[str, str],
    context: str,
    center_offset: int,
) -> List[Dict[str, Any]]:
    """
    Collect all exact flanking-context hits while allowing the centre base to differ.

    The centre base is intentionally ignored during the search because that is
    the nucleotide we want to call in the query sequence. Both plus and minus
    orientations are searched, and all hits are returned so the caller can flag
    multi-mapping instead of silently trusting the first occurrence.
    """
    context = str(context or "").upper()
    if not context or center_offset < 0 or center_offset >= len(context):
        return []

    hits: List[Dict[str, Any]] = []

    def _scan(pattern: str, strand: str, pattern_center_offset: int) -> None:
        left = pattern[:pattern_center_offset]
        right = pattern[pattern_center_offset + 1 :]
        pattern_len = len(pattern)

        for record_id, seq in records.items():
            start = 0
            while True:
                idx = seq.find(left, start) if left else start
                if idx < 0 or idx + pattern_len > len(seq):
                    break
                centre_idx = idx + pattern_center_offset
                right_start = centre_idx + 1
                if seq[right_start : right_start + len(right)] == right:
                    hits.append(
                        _context_hit_record(
                            record_id=record_id,
                            idx=idx,
                            pattern_len=pattern_len,
                            center_offset=pattern_center_offset,
                            strand=strand,
                            observed_raw=seq[centre_idx].upper(),
                        )
                    )
                start = idx + 1

    _scan(context, "plus", center_offset)
    reverse_center_offset = len(context) - 1 - center_offset
    _scan(reverse_complement(context), "minus", reverse_center_offset)
    return hits


def find_by_flanking_context(
    records: Dict[str, str],
    context: str,
    center_offset: int,
) -> Optional[Dict[str, Any]]:
    """
    Exact flanking-context search that allows the centre nucleotide to differ.

    Returns a unique mapping when possible. If the context maps repeatedly, the
    first hit is returned only as a trace record and is marked as multi-hit so
    encode_mapping() will fill the feature as baseline/0 rather than converting
    a repeated context into a false positive marker.
    """
    hits = collect_flanking_context_hits(records, context, center_offset)
    if not hits:
        return None

    first = dict(hits[0])
    first["n_context_hits"] = int(len(hits))
    if len(hits) == 1:
        first["mapping_quality"] = "unique_context"
        return first

    first["status"] = "multi_hit_context"
    first["mapping_quality"] = "multi_hit_context"
    first["alternative_hits"] = [
        {
            "subject_id": h.get("subject_id"),
            "subject_position": h.get("subject_position"),
            "strand": h.get("strand"),
            "observed_allele": h.get("observed_allele"),
        }
        for h in hits[1:10]
    ]
    return first


# -----------------------------------------------------------------------------
# BLAST context mapping
# -----------------------------------------------------------------------------

def blast_available() -> bool:
    return shutil.which("makeblastdb") is not None and shutil.which("blastn") is not None


def write_context_query_fasta(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            context = row.get("Context_sequence", "")
            if context:
                handle.write(f">{row['Feature_ID']}\n{context}\n")


def _parse_blast_hits_for_feature(
    *,
    qid: str,
    hit_rows: List[List[str]],
    row: Dict[str, str],
    records: Dict[str, str],
    min_query_coverage: float,
) -> Optional[Dict[str, Any]]:
    context = row.get("Context_sequence", "")
    if not context:
        return None

    try:
        center_offset = int(float(row.get("Context_center_offset", len(context) // 2)))
    except Exception:
        center_offset = len(context) // 2

    qcenter = center_offset + 1
    candidates: List[Dict[str, Any]] = []

    for parts in hit_rows:
        if len(parts) < 12:
            continue
        try:
            pident = float(parts[2])
            aligned_len = int(float(parts[3]))
            mismatch = int(float(parts[4]))
            gapopen = int(float(parts[5]))
            qstart = int(float(parts[6]))
            qend = int(float(parts[7]))
            sstart = int(float(parts[8]))
            send = int(float(parts[9]))
            evalue = float(parts[10])
            bitscore = float(parts[11])
        except Exception:
            continue

        q_min, q_max = sorted([qstart, qend])
        if not (q_min <= qcenter <= q_max):
            continue

        qcov = float(aligned_len / max(1, len(context)))
        if qcov < min_query_coverage:
            continue

        subject_id = parts[1]
        subject_seq = records.get(subject_id)
        if not subject_seq:
            continue

        if sstart <= send:
            subject_position = sstart + (qcenter - qstart)
            strand = "plus"
            observed = subject_seq[subject_position - 1].upper() if 1 <= subject_position <= len(subject_seq) else ""
        else:
            subject_position = sstart - (qcenter - qstart)
            strand = "minus"
            observed_raw = subject_seq[subject_position - 1].upper() if 1 <= subject_position <= len(subject_seq) else ""
            observed = complement_base(observed_raw)

        if not observed:
            continue

        candidates.append(
            {
                "status": "mapped_unique_context",
                "method": "blast_context",
                "subject_id": subject_id,
                "subject_position": int(subject_position),
                "strand": strand,
                "observed_allele": observed,
                "blast_pident": float(pident),
                "blast_aligned_length": int(aligned_len),
                "blast_query_coverage": float(qcov),
                "blast_mismatch": int(mismatch),
                "blast_gapopen": int(gapopen),
                "blast_evalue": float(evalue),
                "blast_bitscore": float(bitscore),
                "n_blast_hits": int(len(hit_rows)),
                "_sort_key": (bitscore, pident, qcov, aligned_len, -mismatch, -gapopen),
            }
        )

    if not candidates:
        return None

    candidates.sort(key=lambda h: h["_sort_key"], reverse=True)
    best = dict(candidates[0])

    # Equivalent top hits indicate that the selected context is not unique enough
    # for a robust query call. We still report the best coordinate, but encode as 0.
    best_key = best["_sort_key"]
    equivalent = [h for h in candidates if h["_sort_key"] == best_key]
    best["n_equivalent_best_hits"] = int(len(equivalent))
    best["n_context_hits"] = int(len(candidates))
    best.pop("_sort_key", None)

    if len(equivalent) == 1:
        best["mapping_quality"] = "unique_blast_context"
        return best

    best["status"] = "multi_hit_context"
    best["mapping_quality"] = "multi_hit_blast_context"
    best["alternative_hits"] = [
        {
            "subject_id": h.get("subject_id"),
            "subject_position": h.get("subject_position"),
            "strand": h.get("strand"),
            "observed_allele": h.get("observed_allele"),
            "blast_pident": h.get("blast_pident"),
            "blast_bitscore": h.get("blast_bitscore"),
        }
        for h in equivalent[1:10]
    ]
    return best


def run_blast_context_mapping(
    *,
    sample_fasta: Path,
    records: Dict[str, str],
    rows: List[Dict[str, str]],
    output_dir: Path,
    min_query_coverage: float = 0.80,
) -> Dict[str, Dict[str, Any]]:
    """Best-effort BLAST mapping of context sequences to a raw sample FASTA."""
    if not blast_available():
        raise RuntimeError("makeblastdb and/or blastn are not available on PATH.")

    blast_dir = ensure_dir(output_dir / "blast_context")
    query_fasta = blast_dir / "selected_feature_contexts.fasta"
    db_prefix = blast_dir / f"{sample_fasta.stem}.blastdb"
    out_tsv = blast_dir / f"{sample_fasta.stem}.blast.tsv"

    write_context_query_fasta(rows, query_fasta)

    subprocess.run(
        ["makeblastdb", "-in", str(sample_fasta), "-dbtype", "nucl", "-out", str(db_prefix)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        [
            "blastn",
            "-task",
            "blastn-short",
            "-query",
            str(query_fasta),
            "-db",
            str(db_prefix),
            "-outfmt",
            "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore",
            "-out",
            str(out_tsv),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    row_by_feature = {row["Feature_ID"]: row for row in rows}
    hits_by_feature: Dict[str, List[List[str]]] = defaultdict(list)

    with open(out_tsv, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 12:
                hits_by_feature[parts[0]].append(parts)

    mapped: Dict[str, Dict[str, Any]] = {}
    for qid, hit_rows in hits_by_feature.items():
        row = row_by_feature.get(qid)
        if row is None:
            continue
        hit = _parse_blast_hits_for_feature(
            qid=qid,
            hit_rows=hit_rows,
            row=row,
            records=records,
            min_query_coverage=float(min_query_coverage),
        )
        if hit is not None:
            mapped[qid] = hit

    return mapped


# -----------------------------------------------------------------------------
# Allele encoding
# -----------------------------------------------------------------------------

def encode_mapping(row: Dict[str, str], mapping: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert a mapped query nucleotide to the saved NetworkParser binary encoding.

    Conservative inference rules:
      - unresolved or missing context -> 0
      - multi-hit/repeated context -> 0
      - ambiguous base -> 0
      - allele not observed during training -> 0
      - baseline allele -> 0
      - known non-baseline allele -> 1
    """
    feature_id = row["Feature_ID"]
    baseline = row.get("Baseline_allele", row.get("Ref_allele", "")).upper()
    ref = row.get("Ref_allele", "").upper()
    alt = row.get("Alt_allele", "").upper()
    trained_alleles = {a for a in {ref, alt, baseline} if a}

    base_record: Dict[str, Any] = {
        "feature_id": feature_id,
        "ref_allele": ref,
        "alt_allele": alt,
        "baseline_allele": baseline,
        "sequence": row.get("Sequence", ""),
        "position": row.get("Position", ""),
        "gene_annotation": row.get("Gene_annotation", ""),
        "nucleotide_change": row.get("Nucleotide_change", ""),
        "amino_acid_change": row.get("Amino_acid_change", ""),
    }

    if mapping is None:
        reason = "missing_context" if not row.get("Context_sequence") else "unresolved_context"
        return {
            **base_record,
            "encoded_value": 0,
            "observed_allele": "",
            "mapping_status": f"{reason}_filled_as_zero",
            "mapping_quality": reason,
            "allele_call": "not_called",
        }

    observed = str(mapping.get("observed_allele", "")).upper()
    raw_status = str(mapping.get("status", "mapped_unique_context"))
    mapping_quality = str(mapping.get("mapping_quality", raw_status))

    record: Dict[str, Any] = {
        **base_record,
        "observed_allele": observed,
        "subject_id": mapping.get("subject_id", ""),
        "subject_position": mapping.get("subject_position", ""),
        "strand": mapping.get("strand", ""),
        "mapping_method": mapping.get("method", ""),
        "mapping_quality": mapping_quality,
        "n_context_hits": mapping.get("n_context_hits", ""),
        "n_blast_hits": mapping.get("n_blast_hits", ""),
        "n_equivalent_best_hits": mapping.get("n_equivalent_best_hits", ""),
        "blast_pident": mapping.get("blast_pident", ""),
        "blast_aligned_length": mapping.get("blast_aligned_length", ""),
        "blast_query_coverage": mapping.get("blast_query_coverage", ""),
        "blast_mismatch": mapping.get("blast_mismatch", ""),
        "blast_gapopen": mapping.get("blast_gapopen", ""),
        "blast_evalue": mapping.get("blast_evalue", ""),
        "blast_bitscore": mapping.get("blast_bitscore", ""),
    }

    if raw_status.startswith("multi_hit"):
        return {
            **record,
            "encoded_value": 0,
            "mapping_status": f"{raw_status}_filled_as_zero",
            "allele_call": "not_called_multi_hit_context",
        }

    if not is_unambiguous_base(observed):
        return {
            **record,
            "encoded_value": 0,
            "mapping_status": "mapped_ambiguous_base_filled_as_zero",
            "allele_call": "ambiguous_base",
        }

    if trained_alleles and observed not in trained_alleles:
        return {
            **record,
            "encoded_value": 0,
            "mapping_status": "mapped_non_training_allele_filled_as_zero",
            "allele_call": "non_training_allele",
        }

    if observed == baseline:
        return {
            **record,
            "encoded_value": 0,
            "mapping_status": raw_status,
            "allele_call": "baseline_match",
        }

    allele_call = "alt_match" if observed == alt else "known_nonbaseline_match"
    return {
        **record,
        "encoded_value": 1,
        "mapping_status": raw_status,
        "allele_call": allele_call,
    }


# -----------------------------------------------------------------------------
# Summary helpers
# -----------------------------------------------------------------------------

def _status_counts(calls: pd.DataFrame, column: str) -> Dict[str, int]:
    if calls.empty or column not in calls.columns:
        return {}
    return {
        str(k): int(v)
        for k, v in calls[column].astype(str).value_counts(dropna=False).to_dict().items()
    }


def _per_sample_summary(calls: pd.DataFrame) -> List[Dict[str, Any]]:
    if calls.empty or "sample_id" not in calls.columns:
        return []

    rows: List[Dict[str, Any]] = []
    for sample_id, grp in calls.groupby("sample_id", dropna=False):
        encoded = pd.to_numeric(grp.get("encoded_value", pd.Series(dtype=int)), errors="coerce").fillna(0)
        mapping_status = grp.get("mapping_status", pd.Series(dtype=str)).astype(str)
        rows.append(
            {
                "sample_id": str(sample_id),
                "n_feature_calls": int(len(grp)),
                "n_encoded_active_features": int((encoded != 0).sum()),
                "n_unique_mapped_calls": int(mapping_status.eq("mapped_unique_context").sum()),
                "n_unresolved_or_missing_context_calls": int(mapping_status.str.contains("unresolved|missing_context", regex=True).sum()),
                "n_multi_hit_calls": int(mapping_status.str.contains("multi_hit", regex=True).sum()),
                "n_ambiguous_base_calls": int(mapping_status.str.contains("ambiguous_base", regex=True).sum()),
                "n_non_training_allele_calls": int(mapping_status.str.contains("non_training_allele", regex=True).sum()),
                "status_counts": _status_counts(grp, "mapping_status"),
                "allele_call_counts": _status_counts(grp, "allele_call"),
            }
        )
    return rows


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def encode_raw_sequence_query(
    *,
    raw_sequence_path: str,
    feature_manifest_path: str,
    features: Sequence[str],
    output_dir: str,
    mapping_mode: str = "auto",
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """
    Build a sample × selected-feature matrix from raw FASTA DNA.

    Parameters
    ----------
    raw_sequence_path
        FASTA file or directory of FASTA files representing new query samples.
    feature_manifest_path
        Selected-feature or all-feature manifest produced during training.
    features
        Ordered feature IDs required by the saved Level 1 / Level 2 models.
    output_dir
        Directory where matrix, call table, and mapping summary are written.
    mapping_mode
        - ``auto``: use BLAST if available, otherwise exact flanking-context search
        - ``blast``: require BLAST context mapping
        - ``exact``: use exact flanking-context search only
    """
    out = ensure_dir(Path(output_dir))
    mapping_mode = str(mapping_mode or "auto").lower()
    if mapping_mode not in {"auto", "blast", "exact"}:
        raise ValueError("mapping_mode must be one of: auto, blast, exact")

    feature_order = [str(f) for f in features]
    if not feature_order:
        raise ValueError("At least one selected feature is required for raw-sequence query encoding.")

    manifest = load_feature_manifest(Path(feature_manifest_path))
    rows = [normalise_manifest_row(r) for r in manifest_rows_for_features(manifest, feature_order)]
    missing_context = [r["Feature_ID"] for r in rows if not r.get("Context_sequence")]

    fasta_paths = discover_fasta_inputs(raw_sequence_path)
    if not fasta_paths:
        raise ValueError("No FASTA files were found for raw sequence query encoding.")

    matrix_rows: Dict[str, List[int]] = {}
    call_rows: List[Dict[str, Any]] = []
    selected_context_rows = [r for r in rows if r.get("Context_sequence")]
    blast_is_available = blast_available()

    for fasta_path in fasta_paths:
        sample_id = fasta_path.stem
        records = read_fasta_records(fasta_path)
        if not records:
            raise ValueError(f"No FASTA records found in raw sequence input: {fasta_path}")

        blast_mappings: Dict[str, Dict[str, Any]] = {}
        use_blast = mapping_mode == "blast" or (mapping_mode == "auto" and blast_is_available)
        if use_blast and selected_context_rows:
            try:
                blast_mappings = run_blast_context_mapping(
                    sample_fasta=fasta_path,
                    records=records,
                    rows=selected_context_rows,
                    output_dir=out / sample_id,
                )
            except Exception as exc:
                if mapping_mode == "blast":
                    raise
                logger.warning(
                    "BLAST context mapping failed for %s; falling back to exact flanking-context search. Reason: %s",
                    fasta_path,
                    exc,
                )
                blast_mappings = {}
                use_blast = False

        values: List[int] = []
        for row in rows:
            feature_id = row["Feature_ID"]
            mapping: Optional[Dict[str, Any]] = None
            if feature_id in blast_mappings:
                mapping = blast_mappings[feature_id]
            elif row.get("Context_sequence") and mapping_mode != "blast":
                mapping = find_by_flanking_context(
                    records=records,
                    context=row["Context_sequence"],
                    center_offset=int(float(row.get("Context_center_offset", -1))),
                )

            encoded = encode_mapping(row, mapping)
            encoded["sample_id"] = sample_id
            values.append(int(encoded["encoded_value"]))
            call_rows.append(encoded)

        matrix_rows[sample_id] = values

    matrix = pd.DataFrame.from_dict(matrix_rows, orient="index", columns=feature_order, dtype=int)
    matrix.index.name = "Sample"
    calls = pd.DataFrame(call_rows)

    matrix_path = out / "raw_sequence_selected_feature_matrix.csv"
    calls_path = out / "raw_sequence_feature_calls.tsv"
    per_sample_path = out / "raw_sequence_sample_mapping_summary.tsv"
    summary_path = out / "raw_sequence_mapping_summary.json"

    matrix.to_csv(matrix_path)
    calls.to_csv(calls_path, sep="\t", index=False)

    per_sample = _per_sample_summary(calls)
    pd.DataFrame(per_sample).to_csv(per_sample_path, sep="\t", index=False)

    n_calls = int(len(calls))
    mapping_status = calls.get("mapping_status", pd.Series(dtype=str)).astype(str) if n_calls else pd.Series(dtype=str)
    encoded_values = pd.to_numeric(calls.get("encoded_value", pd.Series(dtype=int)), errors="coerce").fillna(0) if n_calls else pd.Series(dtype=int)

    unique_mapped = int(mapping_status.eq("mapped_unique_context").sum()) if n_calls else 0
    active_features = int((encoded_values != 0).sum()) if n_calls else 0

    summary = {
        "mode": "raw_sequence_context_encoding",
        "mapping_mode_requested": mapping_mode,
        "blast_available": bool(blast_is_available),
        "n_samples": int(matrix.shape[0]),
        "n_features_requested": int(len(feature_order)),
        "n_features_missing_context": int(len(missing_context)),
        "missing_context_feature_ids": missing_context,
        "n_feature_calls": n_calls,
        "n_unique_mapped_calls": unique_mapped,
        "unique_mapped_fraction": float(unique_mapped / max(1, n_calls)),
        "n_encoded_active_feature_calls": active_features,
        "active_feature_fraction": float(active_features / max(1, n_calls)),
        "mapping_status_counts": _status_counts(calls, "mapping_status"),
        "mapping_quality_counts": _status_counts(calls, "mapping_quality"),
        "allele_call_counts": _status_counts(calls, "allele_call"),
        "per_sample": per_sample,
        "zero_fill_policy": {
            "unresolved_context": "encoded as 0",
            "missing_context": "encoded as 0",
            "multi_hit_context": "encoded as 0",
            "ambiguous_base": "encoded as 0",
            "non_training_allele": "encoded as 0",
        },
        "artifacts": {
            "raw_sequence_selected_feature_matrix_csv": str(matrix_path),
            "raw_sequence_feature_calls_tsv": str(calls_path),
            "raw_sequence_sample_mapping_summary_tsv": str(per_sample_path),
            "raw_sequence_mapping_summary_json": str(summary_path),
        },
    }
    write_json(summary, summary_path)

    logger.info(
        "Raw-sequence query encoding complete | samples=%d | features=%d | unique_mapped_fraction=%.3f | active_calls=%d",
        int(matrix.shape[0]),
        int(matrix.shape[1]),
        float(summary["unique_mapped_fraction"]),
        int(active_features),
    )

    return matrix, summary, calls

#!/usr/bin/env python3
# network_parser/sequence_query_encoder.py
"""
FASTA query encoder for NetworkParser
===========================================

Purpose
-------
Convert a user-provided FASTA file into the exact selected-feature
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

import gzip
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
    optional annotation, and the context sequence used for FASTA mapping.
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

    hits: List[Dict[str, Any]] = []  # keep all hits for multi-hit detection

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
        allele_call = grp.get("allele_call", pd.Series(dtype=str)).astype(str)
        resolved_mask = allele_call.isin(["baseline_match", "alt_match", "known_nonbaseline_match"])
        baseline_mask = allele_call.eq("baseline_match")
        nonbaseline_mask = allele_call.isin(["alt_match", "known_nonbaseline_match"])
        rows.append(
            {
                "sample_id": str(sample_id),
                "n_feature_calls": int(len(grp)),
                "n_encoded_active_features": int((encoded != 0).sum()),
                "n_resolved_features": int(resolved_mask.sum()),
                "n_resolved_baseline_features": int(baseline_mask.sum()),
                "n_resolved_nonbaseline_features": int(nonbaseline_mask.sum()),
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
# VCF query encoding against the trained selected-feature manifest
# -----------------------------------------------------------------------------

VCF_SUFFIXES = (".vcf", ".vcf.gz")


def _open_text_maybe_gzip(path: Path):
    return gzip.open(path, "rt") if str(path).lower().endswith(".gz") else open(path, "r", encoding="utf-8", errors="replace")


def discover_vcf_inputs(path: str) -> List[Path]:
    """Return one or more VCF/VCF.GZ files for query-time encoding."""
    p = Path(path)
    if p.is_dir():
        vcfs = [x for x in p.iterdir() if x.is_file() and x.name.lower().endswith(VCF_SUFFIXES)]
        return sorted(vcfs)
    if p.is_file() and p.name.lower().endswith(VCF_SUFFIXES):
        return [p]
    raise FileNotFoundError(f"VCF query input not found or not a VCF/VCF.GZ: {p}")


def _sample_id_from_vcf_path(path: Path) -> str:
    name = path.name
    for suffix in (".vcf.gz", ".vcf", ".bcf.gz", ".bcf"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _called_allele_from_vcf_record(ref: str, alts: List[str], fmt: str, sample_field: str) -> str:
    """Best-effort single-sample allele call from a VCF row."""
    ref = str(ref).upper()
    alts = [str(a).upper() for a in alts if str(a).strip()]

    if not fmt or not sample_field:
        return alts[0] if alts else ref

    fmt_keys = fmt.split(":")
    sample_values = sample_field.split(":")
    fmt_map = {k: v for k, v in zip(fmt_keys, sample_values)}
    gt = fmt_map.get("GT", "")

    if not gt or gt in {".", "./.", ".|."}:
        return alts[0] if alts else ref

    sep = "/" if "/" in gt else ("|" if "|" in gt else None)
    tokens = gt.split(sep) if sep else [gt]

    called_indices: List[int] = []
    for token in tokens:
        token = token.strip()
        if token in {"", "."}:
            continue
        try:
            called_indices.append(int(token))
        except Exception:
            continue

    non_ref = [idx for idx in called_indices if idx > 0]
    if not non_ref:
        return ref

    # For haploid/bacterial calls this should usually be a single ALT index.
    # If heterozygous/mixed calls occur, use the first non-reference allele so
    # query encoding remains deterministic and conservative.
    idx = non_ref[0]
    if 1 <= idx <= len(alts):
        return alts[idx - 1]
    return alts[0] if alts else ref


def parse_vcf_calls(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Parse one VCF/VCF.GZ into coordinate-indexed allele calls."""
    calls: Dict[Tuple[str, int], Dict[str, Any]] = {}
    sample_name = _sample_id_from_vcf_path(path)

    with _open_text_maybe_gzip(path) as handle:
        for raw in handle:
            if not raw:
                continue
            line = raw.rstrip("\n")
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                parts = line.split("\t")
                if len(parts) >= 10 and parts[9].strip():
                    sample_name = parts[9].strip()
                continue
            if line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 8:
                continue

            chrom = parts[0]
            try:
                pos = int(parts[1])
            except Exception:
                continue

            ref = parts[3].upper()
            alts = [a.strip().upper() for a in parts[4].split(",") if a.strip() and a.strip() != "."]
            fmt = parts[8] if len(parts) >= 9 else ""
            sample_field = parts[9] if len(parts) >= 10 else ""
            called = _called_allele_from_vcf_record(ref, alts, fmt, sample_field)

            calls[(chrom, pos)] = {
                "chrom": chrom,
                "pos": int(pos),
                "ref": ref,
                "alts": alts,
                "called_allele": called,
                "sample_name": sample_name,
                "source_vcf": str(path),
            }

    return calls


def _build_vcf_position_index(calls: Dict[Tuple[str, int], Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """Build a per-position lookup for fast contig-name fallback in VCF query mode."""
    index: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for (_, pos), payload in calls.items():
        try:
            index[int(pos)].append(payload)
        except Exception:
            continue
    return index


def _manifest_coordinate_keys(row: Dict[str, str]) -> Tuple[str, int]:
    sequence = str(row.get("Sequence", "")).strip()
    try:
        pos = int(float(str(row.get("Position", "")).strip()))
    except Exception:
        _, parsed_pos, _, _ = parse_feature_id(str(row.get("Feature_ID", "")))
        try:
            pos = int(float(parsed_pos))
        except Exception:
            pos = -1
    return sequence, pos


def _vcf_mapping_for_row(
    row: Dict[str, str],
    calls: Dict[Tuple[str, int], Dict[str, Any]],
    position_index: Optional[Dict[int, List[Dict[str, Any]]]] = None,
) -> Optional[Dict[str, Any]]:
    """Create a mapping record for one trained feature from a VCF call set.

    If the VCF has no record at the trained coordinate, we treat the observed
    allele as the manifest reference allele. This is the standard inference for
    single-sample variant-only VCFs generated against the same reference: absence
    of a variant record means the sample carries the reference state at that
    coordinate, subject to upstream caller coverage/QC.
    """
    sequence, pos = _manifest_coordinate_keys(row)
    if pos < 1:
        return None

    feature_id = str(row.get("Feature_ID", ""))
    ref = str(row.get("Ref_allele", "")).upper()

    # Prefer exact contig/chromosome match, then use position-only fallback for
    # single-reference bacterial VCFs whose CHROM naming differs from the manifest.
    call = calls.get((sequence, pos))
    coordinate_match = "exact_sequence_position"
    if call is None:
        if position_index is None:
            position_hits = [payload for (chrom, p), payload in calls.items() if p == pos]
        else:
            position_hits = position_index.get(pos, [])
        if len(position_hits) == 1:
            call = position_hits[0]
            coordinate_match = "position_only_fallback"
        elif len(position_hits) > 1:
            first = position_hits[0]
            return {
                "status": "multi_hit_context",
                "method": "vcf_coordinate",
                "subject_id": first.get("chrom", ""),
                "subject_position": int(pos),
                "strand": "reference",
                "observed_allele": str(first.get("called_allele", "")).upper(),
                "mapping_quality": "multi_vcf_records_same_position",
                "n_context_hits": int(len(position_hits)),
                "vcf_coordinate_match": "ambiguous_position_only",
                "vcf_feature_id": feature_id,
            }

    if call is None:
        return {
            "status": "mapped_unique_context",
            "method": "vcf_coordinate_absent_assumed_reference",
            "subject_id": sequence,
            "subject_position": int(pos),
            "strand": "reference",
            "observed_allele": ref,
            "mapping_quality": "absent_from_vcf_assumed_reference",
            "n_context_hits": 1,
            "vcf_coordinate_match": "absent_from_variant_vcf",
            "vcf_feature_id": feature_id,
        }

    return {
        "status": "mapped_unique_context",
        "method": "vcf_coordinate",
        "subject_id": call.get("chrom", sequence),
        "subject_position": int(pos),
        "strand": "reference",
        "observed_allele": str(call.get("called_allele", "")).upper(),
        "mapping_quality": "vcf_record_found",
        "n_context_hits": 1,
        "vcf_ref_allele": call.get("ref", ""),
        "vcf_alt_alleles": ",".join(call.get("alts", []) or []),
        "vcf_coordinate_match": coordinate_match,
        "vcf_feature_id": feature_id,
    }


def _marker_recovery_status(unique_fraction: float) -> Tuple[str, str]:
    if unique_fraction >= 0.80:
        return "adequate_marker_recovery", "Most selected marker coordinates/contexts were resolved in the query input."
    if unique_fraction >= 0.50:
        return "partial_marker_recovery", "Only part of the selected marker space was resolved; interpret predictions with caution."
    return "low_marker_recovery", "Most selected markers were unresolved or missing; prediction support is likely weak."


def _active_evidence_status(active_fraction: float, active_count: int) -> Tuple[str, str]:
    if active_count >= 10 or active_fraction >= 0.01:
        return "active_marker_evidence_present", "The query carries multiple non-baseline selected marker states."
    if active_count > 0:
        return "very_low_active_marker_evidence", "Only a few selected markers are non-baseline in the query."
    return "no_active_marker_evidence", "Selected markers were recovered mainly as baseline states."


def _resolved_marker_evidence_status(resolved_fraction: float, resolved_count: int) -> Tuple[str, str]:
    if resolved_fraction >= 0.80:
        return "resolved_marker_evidence_present", "Most selected trained markers were resolved, including baseline states encoded as 0."
    if resolved_fraction >= 0.50:
        return "partial_resolved_marker_evidence", "A useful fraction of selected trained markers was resolved, but some calls remain caution states."
    if resolved_count > 0:
        return "low_resolved_marker_evidence", "Only a small fraction of selected trained markers was resolved in the query input."
    return "no_resolved_marker_evidence", "No selected trained markers were confirmed as resolved query states."


def _write_selected_feature_outputs(
    *,
    matrix: pd.DataFrame,
    calls: pd.DataFrame,
    out: Path,
    prefix: str,
    mode: str,
    mapping_mode_requested: str,
    blast_is_available: bool = False,
    missing_context: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Write matrix/call/summary artifacts for manifest-based query encoding."""
    missing_context = missing_context or []
    matrix_path = out / f"{prefix}_selected_feature_matrix.csv"
    calls_path = out / f"{prefix}_feature_calls.tsv"
    per_sample_path = out / f"{prefix}_sample_mapping_summary.tsv"
    summary_path = out / f"{prefix}_mapping_summary.json"

    matrix.to_csv(matrix_path)
    calls.to_csv(calls_path, sep="\t", index=False)

    per_sample = _per_sample_summary(calls)
    # Enrich per-sample rows with fractions/statuses.
    for item in per_sample:
        n = max(1, int(item.get("n_feature_calls", 0)))
        unique_fraction = float(item.get("n_unique_mapped_calls", 0) / n)
        active_fraction = float(item.get("n_encoded_active_features", 0) / n)
        resolved_fraction = float(item.get("n_resolved_features", 0) / n)
        resolved_baseline_fraction = float(item.get("n_resolved_baseline_features", 0) / n)
        recovery_status, recovery_reason = _marker_recovery_status(unique_fraction)
        active_status, active_reason = _active_evidence_status(
            active_fraction, int(item.get("n_encoded_active_features", 0))
        )
        resolved_status, resolved_reason = _resolved_marker_evidence_status(
            resolved_fraction, int(item.get("n_resolved_features", 0))
        )
        item["unique_mapped_fraction"] = unique_fraction
        item["active_feature_fraction"] = active_fraction
        item["resolved_feature_fraction"] = resolved_fraction
        item["resolved_baseline_feature_fraction"] = resolved_baseline_fraction
        item["marker_recovery_status"] = recovery_status
        item["marker_recovery_reason"] = recovery_reason
        item["active_marker_evidence_status"] = active_status
        item["active_marker_evidence_reason"] = active_reason
        item["resolved_marker_evidence_status"] = resolved_status
        item["resolved_marker_evidence_reason"] = resolved_reason

    pd.DataFrame(per_sample).to_csv(per_sample_path, sep="\t", index=False)

    n_calls = int(len(calls))
    mapping_status = calls.get("mapping_status", pd.Series(dtype=str)).astype(str) if n_calls else pd.Series(dtype=str)
    allele_call = calls.get("allele_call", pd.Series(dtype=str)).astype(str) if n_calls else pd.Series(dtype=str)
    encoded_values = pd.to_numeric(calls.get("encoded_value", pd.Series(dtype=int)), errors="coerce").fillna(0) if n_calls else pd.Series(dtype=int)

    unique_mapped = int(mapping_status.eq("mapped_unique_context").sum()) if n_calls else 0
    active_features = int((encoded_values != 0).sum()) if n_calls else 0
    resolved_features = int(allele_call.isin(["baseline_match", "alt_match", "known_nonbaseline_match"]).sum()) if n_calls else 0
    resolved_baseline_features = int(allele_call.eq("baseline_match").sum()) if n_calls else 0
    resolved_nonbaseline_features = int(allele_call.isin(["alt_match", "known_nonbaseline_match"]).sum()) if n_calls else 0
    unique_fraction = float(unique_mapped / max(1, n_calls))
    active_fraction = float(active_features / max(1, n_calls))
    resolved_fraction = float(resolved_features / max(1, n_calls))
    resolved_baseline_fraction = float(resolved_baseline_features / max(1, n_calls))
    recovery_status, recovery_reason = _marker_recovery_status(unique_fraction)
    active_status, active_reason = _active_evidence_status(active_fraction, active_features)
    resolved_status, resolved_reason = _resolved_marker_evidence_status(resolved_fraction, resolved_features)

    summary = {
        "mode": mode,
        "mapping_mode_requested": mapping_mode_requested,
        "blast_available": bool(blast_is_available),
        "n_samples": int(matrix.shape[0]),
        "n_features_requested": int(matrix.shape[1]),
        "n_features_missing_context": int(len(missing_context)),
        "missing_context_feature_ids": missing_context,
        "n_feature_calls": n_calls,
        "n_unique_mapped_calls": unique_mapped,
        "unique_mapped_fraction": unique_fraction,
        "n_encoded_active_feature_calls": active_features,
        "active_feature_fraction": active_fraction,
        "n_resolved_features": resolved_features,
        "resolved_feature_fraction": resolved_fraction,
        "n_resolved_baseline_features": resolved_baseline_features,
        "resolved_baseline_feature_fraction": resolved_baseline_fraction,
        "n_resolved_nonbaseline_features": resolved_nonbaseline_features,
        "resolved_nonbaseline_feature_fraction": float(resolved_nonbaseline_features / max(1, n_calls)),
        "marker_recovery_status": recovery_status,
        "marker_recovery_reason": recovery_reason,
        "active_marker_evidence_status": active_status,
        "active_marker_evidence_reason": active_reason,
        "resolved_marker_evidence_status": resolved_status,
        "resolved_marker_evidence_reason": resolved_reason,
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
            "baseline_or_reference_state": "encoded as 0",
        },
        "artifacts": {
            f"{prefix}_selected_feature_matrix_csv": str(matrix_path),
            f"{prefix}_feature_calls_tsv": str(calls_path),
            f"{prefix}_sample_mapping_summary_tsv": str(per_sample_path),
            f"{prefix}_mapping_summary_json": str(summary_path),
        },
    }

    # Backward-compatible artifact aliases for older README/report code.
    if prefix == "fasta":
        summary["artifacts"].update(
            {
                "raw_sequence_selected_feature_matrix_csv": str(matrix_path),
                "raw_sequence_feature_calls_tsv": str(calls_path),
                "raw_sequence_sample_mapping_summary_tsv": str(per_sample_path),
                "raw_sequence_mapping_summary_json": str(summary_path),
            }
        )

    write_json(summary, summary_path)
    return summary, pd.DataFrame(per_sample)


def _enrich_vcf_per_sample_summary(rows: List[Dict[str, Any]]) -> None:
    """Add fraction/status fields to precomputed VCF per-sample summary rows."""
    for item in rows:
        n = max(1, int(item.get("n_feature_calls", 0)))
        unique_fraction = float(item.get("n_unique_mapped_calls", 0) / n)
        active_fraction = float(item.get("n_encoded_active_features", 0) / n)
        resolved_fraction = float(item.get("n_resolved_features", 0) / n)
        resolved_baseline_fraction = float(item.get("n_resolved_baseline_features", 0) / n)
        recovery_status, recovery_reason = _marker_recovery_status(unique_fraction)
        active_status, active_reason = _active_evidence_status(
            active_fraction, int(item.get("n_encoded_active_features", 0))
        )
        resolved_status, resolved_reason = _resolved_marker_evidence_status(
            resolved_fraction, int(item.get("n_resolved_features", 0))
        )
        item["unique_mapped_fraction"] = unique_fraction
        item["active_feature_fraction"] = active_fraction
        item["resolved_feature_fraction"] = resolved_fraction
        item["resolved_baseline_feature_fraction"] = resolved_baseline_fraction
        item["marker_recovery_status"] = recovery_status
        item["marker_recovery_reason"] = recovery_reason
        item["active_marker_evidence_status"] = active_status
        item["active_marker_evidence_reason"] = active_reason
        item["resolved_marker_evidence_status"] = resolved_status
        item["resolved_marker_evidence_reason"] = resolved_reason


def _write_vcf_selected_feature_outputs_from_counts(
    *,
    matrix: pd.DataFrame,
    calls: pd.DataFrame,
    out: Path,
    per_sample: List[Dict[str, Any]],
    global_counts: Dict[str, Any],
    compact_call_table: bool,
) -> Dict[str, Any]:
    """Write VCF query artifacts without requiring a full sample × feature call table."""
    matrix_path = out / "vcf_selected_feature_matrix.csv"
    calls_path = out / "vcf_feature_calls.tsv"
    per_sample_path = out / "vcf_sample_mapping_summary.tsv"
    summary_path = out / "vcf_mapping_summary.json"

    matrix.to_csv(matrix_path)
    calls.to_csv(calls_path, sep="\t", index=False)
    _enrich_vcf_per_sample_summary(per_sample)
    pd.DataFrame(per_sample).to_csv(per_sample_path, sep="\t", index=False)

    n_calls = int(global_counts.get("n_feature_calls", 0))
    unique_mapped = int(global_counts.get("n_unique_mapped_calls", 0))
    active_features = int(global_counts.get("n_encoded_active_feature_calls", 0))
    resolved_features = int(global_counts.get("n_resolved_features", 0))
    resolved_baseline_features = int(global_counts.get("n_resolved_baseline_features", 0))
    resolved_nonbaseline_features = int(global_counts.get("n_resolved_nonbaseline_features", 0))

    unique_fraction = float(unique_mapped / max(1, n_calls))
    active_fraction = float(active_features / max(1, n_calls))
    resolved_fraction = float(resolved_features / max(1, n_calls))
    resolved_baseline_fraction = float(resolved_baseline_features / max(1, n_calls))
    recovery_status, recovery_reason = _marker_recovery_status(unique_fraction)
    active_status, active_reason = _active_evidence_status(active_fraction, active_features)
    resolved_status, resolved_reason = _resolved_marker_evidence_status(resolved_fraction, resolved_features)

    summary = {
        "mode": "vcf_manifest_coordinate_encoding",
        "mapping_mode_requested": "vcf_manifest_coordinates",
        "blast_available": False,
        "n_samples": int(matrix.shape[0]),
        "n_features_requested": int(matrix.shape[1]),
        "n_features_missing_context": 0,
        "missing_context_feature_ids": [],
        "n_feature_calls": n_calls,
        "n_unique_mapped_calls": unique_mapped,
        "unique_mapped_fraction": unique_fraction,
        "n_encoded_active_feature_calls": active_features,
        "active_feature_fraction": active_fraction,
        "n_resolved_features": resolved_features,
        "resolved_feature_fraction": resolved_fraction,
        "n_resolved_baseline_features": resolved_baseline_features,
        "resolved_baseline_feature_fraction": resolved_baseline_fraction,
        "n_resolved_nonbaseline_features": resolved_nonbaseline_features,
        "resolved_nonbaseline_feature_fraction": float(resolved_nonbaseline_features / max(1, n_calls)),
        "marker_recovery_status": recovery_status,
        "marker_recovery_reason": recovery_reason,
        "active_marker_evidence_status": active_status,
        "active_marker_evidence_reason": active_reason,
        "resolved_marker_evidence_status": resolved_status,
        "resolved_marker_evidence_reason": resolved_reason,
        "mapping_status_counts": dict(global_counts.get("mapping_status_counts", {})),
        "mapping_quality_counts": dict(global_counts.get("mapping_quality_counts", {})),
        "allele_call_counts": dict(global_counts.get("allele_call_counts", {})),
        "per_sample": per_sample,
        "compact_call_table": bool(compact_call_table),
        "compact_call_table_note": (
            "For large VCF query batches, vcf_feature_calls.tsv stores non-baseline and caution calls only. "
            "Resolved baseline/reference states are retained in the selected-feature matrix and per-sample/global summaries."
            if compact_call_table else "vcf_feature_calls.tsv contains one row per sample-feature call."
        ),
        "zero_fill_policy": {
            "unresolved_context": "encoded as 0",
            "missing_context": "encoded as 0",
            "multi_hit_context": "encoded as 0",
            "ambiguous_base": "encoded as 0",
            "non_training_allele": "encoded as 0",
            "baseline_or_reference_state": "encoded as 0",
        },
        "artifacts": {
            "vcf_selected_feature_matrix_csv": str(matrix_path),
            "vcf_feature_calls_tsv": str(calls_path),
            "vcf_sample_mapping_summary_tsv": str(per_sample_path),
            "vcf_mapping_summary_json": str(summary_path),
        },
    }
    write_json(summary, summary_path)
    return summary


def _should_keep_vcf_call_row(encoded: Dict[str, Any], compact: bool) -> bool:
    """Keep full call rows for small runs; for large runs retain only evidence/caution rows."""
    if not compact:
        return True
    try:
        if int(encoded.get("encoded_value", 0)) != 0:
            return True
    except Exception:
        pass
    allele_call = str(encoded.get("allele_call", ""))
    mapping_status = str(encoded.get("mapping_status", ""))
    if allele_call not in {"baseline_match"}:
        return True
    if any(token in mapping_status for token in ("multi_hit", "ambiguous_base", "non_training_allele", "missing_context", "unresolved_context")):
        return True
    return False


def encode_vcf_query_from_manifest(
    *,
    vcf_path: str,
    feature_manifest_path: str,
    features: Sequence[str],
    output_dir: str,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """Build a query matrix directly in the trained selected-feature space from VCF.

    This deliberately bypasses DataLoader cohort-level artifact refinement. Query
    mode must not rediscover, filter, or collapse features; it must reconstruct
    the saved training feature columns and encode each selected feature using the
    same baseline/REF/ALT rule saved in the selected-feature manifest.

    Performance note
    ----------------
    Large VCF batches can contain millions of sample-feature calls. For those
    batches, the matrix and summaries remain complete, while the per-call TSV is
    compacted to non-baseline/caution rows to avoid spending most runtime writing
    baseline-reference rows that are already represented as 0 in the matrix.
    """
    out = ensure_dir(Path(output_dir))
    feature_order = [str(f) for f in features]
    if not feature_order:
        raise ValueError("At least one selected feature is required for VCF query encoding.")

    manifest = load_feature_manifest(Path(feature_manifest_path))
    rows = [normalise_manifest_row(r) for r in manifest_rows_for_features(manifest, feature_order)]
    vcf_paths = discover_vcf_inputs(vcf_path)
    if not vcf_paths:
        raise ValueError("No VCF files were found for VCF query encoding.")

    total_call_slots = int(len(vcf_paths) * len(rows))
    compact_call_table = total_call_slots > 2_000_000
    if compact_call_table:
        logger.info(
            "Large VCF query batch detected | samples=%d | trained_features=%d | call_slots=%d | writing compact call table",
            int(len(vcf_paths)),
            int(len(rows)),
            total_call_slots,
        )

    matrix_rows: Dict[str, List[int]] = {}
    call_rows: List[Dict[str, Any]] = []
    per_sample: List[Dict[str, Any]] = []
    global_counts: Dict[str, Any] = {
        "n_feature_calls": 0,
        "n_unique_mapped_calls": 0,
        "n_encoded_active_feature_calls": 0,
        "n_resolved_features": 0,
        "n_resolved_baseline_features": 0,
        "n_resolved_nonbaseline_features": 0,
        "mapping_status_counts": Counter(),
        "mapping_quality_counts": Counter(),
        "allele_call_counts": Counter(),
    }

    resolved_calls = {"baseline_match", "alt_match", "known_nonbaseline_match"}
    nonbaseline_calls = {"alt_match", "known_nonbaseline_match"}

    for sample_i, path in enumerate(vcf_paths, start=1):
        calls = parse_vcf_calls(path)
        position_index = _build_vcf_position_index(calls)
        sample_id = _sample_id_from_vcf_path(path)
        # Prefer sample name from VCF header if available.
        for payload in calls.values():
            if payload.get("sample_name"):
                sample_id = str(payload["sample_name"])
                break

        values: List[int] = []
        sample_counts: Dict[str, Any] = {
            "sample_id": sample_id,
            "n_feature_calls": 0,
            "n_encoded_active_features": 0,
            "n_resolved_features": 0,
            "n_resolved_baseline_features": 0,
            "n_resolved_nonbaseline_features": 0,
            "n_unique_mapped_calls": 0,
            "n_unresolved_or_missing_context_calls": 0,
            "n_multi_hit_calls": 0,
            "n_ambiguous_base_calls": 0,
            "n_non_training_allele_calls": 0,
            "status_counts": Counter(),
            "allele_call_counts": Counter(),
        }

        for row in rows:
            mapping = _vcf_mapping_for_row(row, calls, position_index=position_index)
            encoded = encode_mapping(row, mapping)
            encoded["sample_id"] = sample_id
            encoded["source_query_file"] = str(path)
            encoded_value = int(encoded["encoded_value"])
            values.append(encoded_value)

            mapping_status = str(encoded.get("mapping_status", ""))
            mapping_quality = str(encoded.get("mapping_quality", ""))
            allele_call = str(encoded.get("allele_call", ""))

            global_counts["n_feature_calls"] += 1
            sample_counts["n_feature_calls"] += 1
            global_counts["mapping_status_counts"][mapping_status] += 1
            global_counts["mapping_quality_counts"][mapping_quality] += 1
            global_counts["allele_call_counts"][allele_call] += 1
            sample_counts["status_counts"][mapping_status] += 1
            sample_counts["allele_call_counts"][allele_call] += 1

            if mapping_status == "mapped_unique_context":
                global_counts["n_unique_mapped_calls"] += 1
                sample_counts["n_unique_mapped_calls"] += 1
            if encoded_value != 0:
                global_counts["n_encoded_active_feature_calls"] += 1
                sample_counts["n_encoded_active_features"] += 1
            if allele_call in resolved_calls:
                global_counts["n_resolved_features"] += 1
                sample_counts["n_resolved_features"] += 1
            if allele_call == "baseline_match":
                global_counts["n_resolved_baseline_features"] += 1
                sample_counts["n_resolved_baseline_features"] += 1
            if allele_call in nonbaseline_calls:
                global_counts["n_resolved_nonbaseline_features"] += 1
                sample_counts["n_resolved_nonbaseline_features"] += 1
            if "missing_context" in mapping_status or "unresolved_context" in mapping_status:
                sample_counts["n_unresolved_or_missing_context_calls"] += 1
            if "multi_hit" in mapping_status:
                sample_counts["n_multi_hit_calls"] += 1
            if "ambiguous_base" in mapping_status:
                sample_counts["n_ambiguous_base_calls"] += 1
            if "non_training_allele" in mapping_status:
                sample_counts["n_non_training_allele_calls"] += 1

            if _should_keep_vcf_call_row(encoded, compact=compact_call_table):
                call_rows.append(encoded)

        matrix_rows[sample_id] = values
        sample_counts["status_counts"] = dict(sample_counts["status_counts"])
        sample_counts["allele_call_counts"] = dict(sample_counts["allele_call_counts"])
        per_sample.append(sample_counts)

        if sample_i == 1 or sample_i % 250 == 0 or sample_i == len(vcf_paths):
            logger.info(
                "VCF query encoding progress | samples_done=%d/%d | trained_features=%d | retained_call_rows=%d",
                int(sample_i),
                int(len(vcf_paths)),
                int(len(rows)),
                int(len(call_rows)),
            )

    matrix = pd.DataFrame.from_dict(matrix_rows, orient="index", columns=feature_order, dtype=int)
    matrix.index.name = "Sample"
    calls_df = pd.DataFrame(call_rows)

    summary = _write_vcf_selected_feature_outputs_from_counts(
        matrix=matrix,
        calls=calls_df,
        out=out,
        per_sample=per_sample,
        global_counts=global_counts,
        compact_call_table=compact_call_table,
    )

    logger.info(
        "VCF query encoding complete | samples=%d | trained_features=%d | active_calls=%d | unique_fraction=%.3f | compact_call_table=%s",
        int(matrix.shape[0]),
        int(matrix.shape[1]),
        int(summary.get("n_encoded_active_feature_calls", 0)),
        float(summary.get("unique_mapped_fraction", 0.0)),
        "yes" if compact_call_table else "no",
    )
    return matrix, summary, calls_df


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
    Build a sample × selected-feature matrix from FASTA DNA.

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
        raise ValueError("At least one selected feature is required for FASTA query encoding.")

    manifest = load_feature_manifest(Path(feature_manifest_path))
    rows = [normalise_manifest_row(r) for r in manifest_rows_for_features(manifest, feature_order)]
    missing_context = [r["Feature_ID"] for r in rows if not r.get("Context_sequence")]

    fasta_paths = discover_fasta_inputs(raw_sequence_path)
    if not fasta_paths:
        raise ValueError("No FASTA files were found for FASTA query encoding.")

    matrix_rows: Dict[str, List[int]] = {}
    call_rows: List[Dict[str, Any]] = []
    selected_context_rows = [r for r in rows if r.get("Context_sequence")]
    blast_is_available = blast_available()

    for fasta_path in fasta_paths:
        sample_id = fasta_path.stem
        records = read_fasta_records(fasta_path)
        if not records:
            raise ValueError(f"No FASTA records found in FASTA query input: {fasta_path}")

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
                    "BLAST failed for %s; falling back to exact. Reason: %s",
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

    summary, _ = _write_selected_feature_outputs(
        matrix=matrix,
        calls=calls,
        out=out,
        prefix="fasta",
        mode="fasta_context_encoding",
        mapping_mode_requested=mapping_mode,
        blast_is_available=bool(blast_is_available),
        missing_context=missing_context,
    )

    logger.info(
        "FASTA query encoding complete | samples=%d | features=%d | unique_mapped_fraction=%.3f | active_calls=%d",
        int(matrix.shape[0]),
        int(matrix.shape[1]),
        float(summary["unique_mapped_fraction"]),
        int(summary.get("n_encoded_active_feature_calls", 0)),
    )

    return matrix, summary, calls

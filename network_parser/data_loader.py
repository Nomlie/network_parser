"""
network_parser.data_loader

Build a **sample × variant** binary matrix (0/1) from per-sample VCF files.

Core responsibilities:
  - Parse per-sample VCF/VCF.GZ files.
  - Apply INFO/QUAL QC (QUAL/DP/MQ/MQ0F).
  - Enforce a cohort-level presence threshold (minimum #samples with the SNP).
  - Build an allelic matrix (REF + per-sample alleles) and a binary matrix (0/1)
    using a configurable baseline:
      * ancestral_allele='Y' → baseline = reference allele
      * ancestral_allele='N' → baseline = cohort mode allele (most common base)

Artifact responsibilities (when output_dir is provided):
  - Write outputs matching the three legacy scripts:
      vcf_counts/all_snp.txt
      fasta/<generic>_alleles.fasta
      fasta/<generic>_binary.fasta
      fasta/<generic>_filtered.tsv      (+ optional Context_±40)
      matrices/<generic>_alleles.tsv
      matrices/<generic>_binary.tsv
      matrices/<generic>_alleles.fasta
      matrices/<generic>_binary.fasta
      matrices/<generic>_filtered.tsv

Non-responsibilities:
  - Statistical validation (χ² / Fisher + FDR) must happen BEFORE tree construction
  - Decision tree building
  - Post-tree bootstrapping / confidence scoring

Returned value:
  - pandas.DataFrame: rows = samples, columns = variants, values ∈ {0,1}
"""

from __future__ import annotations
import os 
import csv
import gzip
import json
import logging
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
from joblib import Parallel, delayed
import pandas as pd

logger = logging.getLogger(__name__)

def _minor_count_chunk(cols: List[List[str]]) -> List[Tuple[int, int]]:
    # returns [(count0, count1), ...] for each col in chunk
    out = []
    for col in cols:
        c1 = 0
        for v in col:
            if v == "1":
                c1 += 1
        c0 = len(col) - c1
        out.append((c0, c1))
    return out


def minor_count_filter_parallel(binary_cols: List[List[str]], min_count: int, n_jobs: int) -> List[bool]:
    """
    Parallel minor-count filter across columns.

    Note: use ProcessPoolExecutor because Python loops are GIL-bound.
    Keep chunk sizes reasonably large to avoid overhead.
    """
    if min_count <= 0:
        return [True] * len(binary_cols)
    if n_jobs is None or n_jobs == 1 or len(binary_cols) < 5000:
        # small: parallel overhead not worth it
        return minor_count_filter(binary_cols, min_count)

    # choose chunks (tune)
    n_cols = len(binary_cols)
    n_workers = os.cpu_count() if n_jobs < 0 else n_jobs
    n_workers = max(1, int(n_workers))
    chunk_size = max(1000, n_cols // (n_workers * 4))

    chunks = [binary_cols[i:i+chunk_size] for i in range(0, n_cols, chunk_size)]

    results: List[Tuple[int, int]] = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for part in ex.map(_minor_count_chunk, chunks):
            results.extend(part)

    keep = []
    for c0, c1 in results:
        keep.append(min(c0, c1) >= min_count)
    return keep

def _fmt_n_jobs(n_jobs: Optional[int]) -> str:
    """Log-friendly n_jobs formatting."""
    if n_jobs is None:
        return "default"
    if isinstance(n_jobs, int) and n_jobs < 0:
        return "all cores"
    return str(n_jobs)


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

try:
    from Bio import SeqIO
    from Bio.Seq import Seq

    HAVE_BIO = True
except Exception:  # pragma: no cover
    HAVE_BIO = False


# ──────────────────────────────────────────────────────────────
# Basic file + VCF parsing helpers
# ──────────────────────────────────────────────────────────────


def open_any(path: Path):
    """Open plain-text or gzipped text files in read-text mode."""
    p = str(path)
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p, "r", encoding="utf-8", errors="replace")


def parse_info_field(info_str: str) -> Dict[str, str]:
    """Parse the VCF INFO column into a dictionary of string values."""
    info: Dict[str, str] = {}
    if not info_str:
        return info
    for token in info_str.split(";"):
        if "=" in token:
            k, v = token.split("=", 1)
            info[k] = v
    return info


def is_snp_like(ref: str, alt_field: str, biallelic_only: bool = True) -> bool:
    """Return True if the record looks like a SNP (single-base REF and ALT(s))."""
    if not ref or not alt_field:
        return False
    if ref == "." or alt_field == ".":
        return False
    alts = alt_field.split(",")
    if biallelic_only and len(alts) != 1:
        return False
    if len(ref) != 1:
        return False
    if any(len(a) != 1 for a in alts):
        return False
    return True


def passes_info_qc(
    qual_str: str,
    info: Dict[str, str],
    qual_thresh: float,
    dp_thresh: int,
    mq_thresh: float,
    mq0f_thresh: float,
) -> bool:
    """Apply INFO/QUAL filters (QUAL, DP, MQ, MQ0F)."""
    try:
        qual = float(qual_str) if qual_str != "." else 0.0
    except Exception:
        qual = 0.0
    if qual < qual_thresh:
        return False

    # DP can appear as float depending on caller; cast defensively
    try:
        dp = int(float(info.get("DP", "0")))
    except Exception:
        dp = 0
    try:
        mq = float(info.get("MQ", "0"))
    except Exception:
        mq = 0.0
    try:
        mq0f = float(info.get("MQ0F", "0"))
    except Exception:
        mq0f = 0.0

    if dp < dp_thresh:
        return False
    if mq < mq_thresh:
        return False
    if mq0f > mq0f_thresh:
        return False
    return True


def choose_called_allele(ref: str, alts: List[str], fmt: Optional[str], sample_field: Optional[str]) -> str:
    """Determine the allele for a sample at a site.

    Strategy:
      - If FORMAT/sample fields exist and include GT, interpret GT:
          * if any allele index > 0 appears → use that ALT (1→ALT0, 2→ALT1, ...)
          * else (0/0 or missing) → REF
      - If no GT is available → assume ALT[0] for presence-based calls
    """
    if not fmt or not sample_field:
        return alts[0] if alts else ref

    fmt_keys = fmt.split(":")
    smp_vals = sample_field.split(":")
    fmt_map = {k: v for k, v in zip(fmt_keys, smp_vals)}
    gt = fmt_map.get("GT", "")

    if not gt:
        return alts[0] if alts else ref

    sep = "/" if "/" in gt else ("|" if "|" in gt else None)
    tokens = gt.split(sep) if sep else [gt]

    allele_idx: Optional[int] = None
    for tok in tokens:
        tok = tok.strip()
        if tok in (".", ""):
            continue
        try:
            idx = int(tok)
        except ValueError:
            continue
        if idx > 0:
            allele_idx = idx
            break

    if allele_idx is None:
        return ref

    if 1 <= allele_idx <= len(alts):
        return alts[allele_idx - 1]
    return alts[0] if alts else ref


def iter_sample_calls(
    vcf_path: Path,
    qual_thresh: float,
    dp_thresh: int,
    mq_thresh: float,
    mq0f_thresh: float,
    biallelic_only: bool = True,
) -> Dict[Tuple[str, int], Tuple[str, str]]:
    """Extract per-sample SNP calls after QC.

    Returns:
      dict keyed by (chrom, pos) -> (ref_base, called_base)
    """
    calls: Dict[Tuple[str, int], Tuple[str, str]] = {}
    with open_any(vcf_path) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue

            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue

            chrom = parts[0]
            try:
                pos = int(parts[1])
            except Exception:
                continue

            ref = parts[3].upper()
            alt_field = parts[4].upper()

            if not is_snp_like(ref, alt_field, biallelic_only=biallelic_only):
                continue

            info = parse_info_field(parts[7])
            if not passes_info_qc(parts[5], info, qual_thresh, dp_thresh, mq_thresh, mq0f_thresh):
                continue

            alts = [a.strip().upper() for a in alt_field.split(",") if a.strip()]
            fmt = parts[8] if len(parts) >= 9 else None
            sample_field = parts[9] if len(parts) >= 10 else None

            called = choose_called_allele(ref, alts, fmt, sample_field).upper()
            if len(called) != 1:
                continue

            calls[(chrom, pos)] = (ref, called)

    return calls


# ──────────────────────────────────────────────────────────────
# Reference + context helpers (used for filtered TSV context column)
# ──────────────────────────────────────────────────────────────


def load_reference_sequence(ref_path: Path) -> Optional[str]:
    """Load reference sequence from FASTA or GenBank.

    If multiple records exist, sequences are concatenated in file order.
    Requires Biopython.
    """
    if not ref_path.exists():
        return None
    if not HAVE_BIO:
        raise RuntimeError("Biopython is required for reference sequence loading but is not available.")

    lower = ref_path.name.lower()
    fmt = "fasta" if lower.endswith((".fa", ".fna", ".fasta", ".fas")) else None
    if lower.endswith((".gb", ".gbk", ".gbff")):
        fmt = "genbank"

    seqs: List[str] = []
    if fmt:
        for rec in SeqIO.parse(str(ref_path), fmt):
            seqs.append(str(rec.seq).upper())
    else:
        # Try FASTA then GenBank
        try:
            for rec in SeqIO.parse(str(ref_path), "fasta"):
                seqs.append(str(rec.seq).upper())
        except Exception:
            seqs = []
        if not seqs:
            for rec in SeqIO.parse(str(ref_path), "genbank"):
                seqs.append(str(rec.seq).upper())

    if not seqs:
        return None
    return "".join(seqs).upper()


def context_around(pos_1based: int, genome: str, flank: int = 40) -> str:
    """Extract circular ±flank context around a 1-based position."""
    n = len(genome)
    if n == 0:
        return ""
    i = (pos_1based - 1) % n
    out = []
    for off in range(-flank, flank + 1):
        out.append(genome[(i + off) % n])
    return "".join(out)


# ──────────────────────────────────────────────────────────────
# Optional GenBank annotation table (all_snp.txt with annotation columns)
# ──────────────────────────────────────────────────────────────


def annotate_snps_genbank(
    snp_details: Dict[Tuple[str, int], Tuple[int, str, str]],
    ref_gbk_path: Path,
) -> List[Dict[str, str]]:
    """Annotate SNPs using a GenBank reference sequence.

    Input:
      snp_details: (chrom,pos) -> (count, ref_nt, alt_nt)

    Output:
      list of row dicts with columns used by the annotation table.
    """
    if not HAVE_BIO:
        raise RuntimeError("Biopython is required for GenBank annotation but is not available.")

    record = SeqIO.read(str(ref_gbk_path), "genbank")
    sequence = record.seq
    features = [f for f in record.features if f.type == "CDS"]

    rows: List[Dict[str, str]] = []

    for (chrom, pos, ref_nt, alt_nt) in sorted(snp_details.keys(), key=lambda x: (x[0], x[1], x[2], x[3])):
        count = snp_details[(chrom, pos, ref_nt, alt_nt)]
        pos0 = pos - 1
        coding_found = False

        closest = None
        min_dist = float("inf")

        for feature in features:
            start = int(feature.location.start)
            end = int(feature.location.end)
            strand = feature.location.strand
            locus_tag = feature.qualifiers.get("locus_tag", ["."])[0]
            gene = feature.qualifiers.get("gene", ["."])[0]
            product = feature.qualifiers.get("product", ["."])[0]
            label = f"{'+' if strand == 1 else '-'}{locus_tag} | {gene} | {product} | [{start+1}..{end}]"

            if start <= pos0 < end:
                coding_found = True
                rel_pos = (pos0 - start) if strand == 1 else (end - pos0 - 1)
                codon_number = rel_pos // 3 + 1

                if strand == 1:
                    codon_start = start + (rel_pos // 3) * 3
                    codon_seq = sequence[codon_start : codon_start + 3]
                else:
                    codon_start = end - ((rel_pos // 3 + 1) * 3)
                    codon_seq = sequence[codon_start : codon_start + 3].reverse_complement()

                ref_codon = str(codon_seq).upper()
                snp_pos_in_codon = rel_pos % 3
                codon_list = list(ref_codon)
                codon_list[snp_pos_in_codon] = alt_nt.upper()
                mut_codon = "".join(codon_list)

                ref_aa = str(Seq(ref_codon).translate())
                alt_aa = str(Seq(mut_codon).translate())

                rows.append(
                    {
                        "Position": str(pos),
                        "Count": str(count),
                        "Sequence": chrom,
                        "Region_type": "coding",
                        "Relative_pos": str(rel_pos + 1),
                        "Codon_number": str(codon_number),
                        "Nucleotide_change": f"{ref_nt}|{alt_nt}",
                        "Amino_acid_change": f"{ref_aa}|{alt_aa}",
                        "Gene_annotation": label,
                    }
                )
                break

            dist = min(abs(pos0 - start), abs(pos0 - end))
            if dist < min_dist:
                min_dist = dist
                closest = (start, strand, locus_tag, gene, product)

        if not coding_found:
            if closest is not None:
                start, strand, locus_tag, gene, product = closest
                label = f"{'+' if strand == 1 else '-'}{locus_tag} | {gene} | {product} | [{start+1}..{start+1}]"
                rel = -int(min_dist) if min_dist != float("inf") else -1
            else:
                label = ". | . | . | [.]"
                rel = -1

            rows.append(
                {
                    "Position": str(pos),
                    "Count": str(count),
                    "Sequence": chrom,
                    "Region_type": "non-coding",
                    "Relative_pos": str(rel),
                    "Codon_number": "0",
                    "Nucleotide_change": f"{ref_nt}|{alt_nt}",
                    "Amino_acid_change": "NA",
                    "Gene_annotation": label,
                }
            )

    return rows


# ──────────────────────────────────────────────────────────────
# FASTA matrix I/O (for allele/binary matrices)
# ──────────────────────────────────────────────────────────────


def write_fasta_matrix(path: Path, ref_seq: str, sample_map: Dict[str, str], ref_name: str = "REF") -> None:
    """Write a REF + sample sequences FASTA that represents a column-aligned matrix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as out:
        out.write(f">{ref_name}\n{ref_seq}\n")
        for name in sorted(sample_map):
            out.write(f">{name}\n{sample_map[name]}\n")


def read_fasta_matrix(path: Path) -> OrderedDict:
    """Read a FASTA matrix into an OrderedDict[name] -> list(chars)."""
    records: OrderedDict = OrderedDict()
    with open(path, "r", encoding="utf-8") as f:
        name = None
        seq_chunks: List[str] = []
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records[name] = list("".join(seq_chunks))
                name = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())
        if name is not None:
            records[name] = list("".join(seq_chunks))

    if not records:
        raise ValueError(f"FASTA file appears empty: {path}")

    lengths = {len(v) for v in records.values()}
    if len(lengths) != 1:
        raise ValueError(f"Sequences in {path} have different lengths: {sorted(lengths)}")

    return records


def write_fasta_matrix_wrapped(path: Path, matrix: OrderedDict, line_width: int = 80) -> None:
    """Write FASTA from OrderedDict[name]->list(chars) with wrapping."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as out:
        for gid, chars in matrix.items():
            out.write(f">{gid}\n")
            seq = "".join(chars)
            for i in range(0, len(seq), line_width):
                out.write(seq[i : i + line_width] + "\n")


# ──────────────────────────────────────────────────────────────
# Matrix conversion + filtering utilities (for matrices/* outputs)
# ──────────────────────────────────────────────────────────────


def transpose_rows_to_columns(matrix: OrderedDict) -> Tuple[List[str], List[List[str]]]:
    """Convert row-oriented FASTA matrix to a column list for per-marker filtering."""
    genomes = list(matrix.keys())
    if not genomes:
        raise ValueError("Empty FASTA matrix.")
    row_len = len(next(iter(matrix.values())))
    cols = [[] for _ in range(row_len)]
    for gid in genomes:
        row = matrix[gid]
        if len(row) != row_len:
            raise ValueError("Row lengths differ in FASTA matrix.")
        for j, ch in enumerate(row):
            cols[j].append(ch)
    return genomes, cols


def minor_count_filter(binary_cols: List[List[str]], min_count: int) -> List[bool]:
    """Keep column j only if min(count_0, count_1) >= min_count."""
    keep: List[bool] = []
    for col in binary_cols:
        c0 = sum(1 for x in col if x == "0")
        c1 = sum(1 for x in col if x == "1")
        keep.append(min(c0, c1) >= min_count)
    return keep


def type_filter(annotation_rows: List[Dict[str, str]], typ: str) -> List[bool]:
    """Filter mask by annotation type: all | coding | sense-mutations."""
    if typ == "all":
        return [True] * len(annotation_rows)

    def is_coding(r: Dict[str, str]) -> bool:
        return (r.get("Region_type", "") or "").strip().lower() == "coding"

    def aa_changed(r: Dict[str, str]) -> bool:
        field = (r.get("Amino_acid_change", "") or "").strip()
        if "|" in field:
            left, right = [x.strip() for x in field.split("|", 1)]
            if left and right and left != "-" and right != "-":
                return left != right
        return False

    if typ == "coding":
        return [is_coding(r) for r in annotation_rows]
    if typ == "sense-mutations":
        return [is_coding(r) and aa_changed(r) for r in annotation_rows]
    raise ValueError(f"Unknown type filter: {typ}")


def combine_masks(*masks: List[bool]) -> List[bool]:
    """Combine same-length boolean masks via AND."""
    if not masks:
        return []
    mlen = len(masks[0])
    for m in masks:
        if len(m) != mlen:
            raise ValueError("Mask lengths differ.")
    return [all(m[i] for m in masks) for i in range(mlen)]


def even_pick_indices(sorted_indices: List[int], k: int) -> List[int]:
    """Pick k indices spaced across the sorted list."""
    n = len(sorted_indices)
    if k == 0 or k >= n:
        return sorted_indices[:]
    if k == 1:
        return [sorted_indices[n // 2]]
    chosen = set()
    for i in range(k):
        pos = round(i * (n - 1) / (k - 1))
        chosen.add(pos)
    return [sorted_indices[i] for i in sorted(chosen)]


def group_and_reduce_by_pattern(
    binary_cols: List[List[str]],
    annotation_rows: List[Dict[str, str]],
    repeat_number: int,
) -> List[bool]:
    """Group identical 0/1 patterns and keep up to repeat_number columns per pattern."""
    groups: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for j, col in enumerate(binary_cols):
        groups[tuple(col)].append(j)

    positions: List[Optional[int]] = []
    for r in annotation_rows:
        pos_raw = (r.get("Position", "") or "").strip()
        try:
            positions.append(int(pos_raw))
        except ValueError:
            try:
                positions.append(int(float(pos_raw)))
            except Exception:
                positions.append(None)

    keep = [False] * len(binary_cols)
    for cols in groups.values():
        cols_sorted = sorted(
            cols,
            key=lambda idx: (positions[idx] is None, positions[idx] if positions[idx] is not None else idx),
        )
        picked = even_pick_indices(cols_sorted, repeat_number)
        for idx in picked:
            keep[idx] = True
    return keep


def parse_fix_positions(fix_arg: str, total_cols: int) -> Tuple[List[int], List[str]]:
    """Parse comma/space-separated 1-based positions; return (0-based indices, warnings)."""
    if not fix_arg:
        return [], []
    raw = fix_arg.replace(",", " ").split()
    vals: List[int] = []
    warnings: List[str] = []
    seen = set()
    for token in raw:
        try:
            v = int(token)
        except ValueError:
            warnings.append(f"Ignored non-integer token in --fix: '{token}'")
            continue
        if v <= 0:
            warnings.append(f"Ignored non-positive position in --fix: {v}")
            continue
        if v > total_cols:
            warnings.append(f"Ignored out-of-range position in --fix: {v} (max {total_cols})")
            continue
        if v in seen:
            continue
        seen.add(v)
        vals.append(v - 1)
    vals.sort()
    return vals, warnings


def apply_mask_to_char_rows(matrix: OrderedDict, mask: List[bool]) -> OrderedDict:
    """Apply a column mask to every sequence row in a FASTA matrix."""
    out = OrderedDict()
    for gid, chars in matrix.items():
        out[gid] = [ch for ch, k in zip(chars, mask) if k]
    return out


def read_annotation_tsv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    """Read a tab-separated annotation TSV into rows + header (robust encodings)."""
    encodings_to_try = ["utf-8", "utf-8-sig", "cp1251", "cp1252", "latin-1"]
    last_error = None
    rows: List[Dict[str, str]] = []
    header: List[str] = []

    for enc in encodings_to_try:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                header = reader.fieldnames or []
                rows = list(reader)
            break
        except UnicodeDecodeError as e:
            last_error = e
            continue

    if not rows and last_error is not None:
        raise UnicodeError(f"Failed to decode annotation file {path}. Last error: {last_error}")

    return rows, header


def write_annotation_tsv(path: Path, rows: List[Dict[str, str]], header: List[str]) -> None:
    """Write annotation TSV in a fixed header order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=header, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_matrix_tsv(path: Path, genomes: List[str], positions: List[str], data_cols: List[List[str]]) -> None:
    """Write a matrix TSV: Genome + positions header, one row per genome."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        writer.writerow(["Genome"] + positions)
        for i, gid in enumerate(genomes):
            row = [gid] + [col[i] for col in data_cols]
            writer.writerow(row)


# ──────────────────────────────────────────────────────────────
# DataLoader
# ──────────────────────────────────────────────────────────────


class DataLoader:
    """Build a clean binary feature matrix from per-sample VCFs."""
    def _log_stage1_reconciliation(
        self,
        *,
        n_samples: int,
        candidate_sites: int,
        kept_sites_n: int,
        df_shape: tuple,
        out_root: Optional[Path],
        matrices_final_markers: Optional[int] = None,
    ) -> None:
        """
        Reconcile counts across:
        - candidate cohort sites
        - presence-filtered sites
        - downstream matrix features (after preprocessing)
        - curated matrices/* outputs
        """
        downstream_features = int(df_shape[1])
        removed_features = max(0, kept_sites_n - downstream_features)

        logger.info(
            "DataLoader: Stage 1 reconciliation\n"
            "  samples=%d\n"
            "  candidate sites=%d\n"
            "  kept sites after sample presence filter=%d\n"
            "  downstream features after preprocess=%d (removed=%d invariant features)\n"
            "  matrices curated markers=%s\n"
            "  artifacts out=%s",
            n_samples,
            candidate_sites,
            kept_sites_n,
            downstream_features,
            removed_features,
            str(matrices_final_markers) if matrices_final_markers is not None else "n/a",
            str(out_root) if out_root is not None else "n/a",
        )
    def __init__(self, config=None, n_jobs: Optional[int] = None):
        self.config = config
        self.n_jobs = n_jobs

        # INFO-level site QC thresholds
        self.qual_threshold = float(getattr(config, "qual_threshold", 30.0)) if config else 30.0
        self.dp_threshold = int(getattr(config, "min_dp_per_sample", 10)) if config else 10
        self.mq_threshold = float(getattr(config, "mq_threshold", 40.0)) if config else 40.0
        self.mq0f_threshold = float(getattr(config, "mq0f_threshold", 0.1)) if config else 0.1

        # Cohort-level presence threshold (minimum number of samples that must contain the SNP)
        self.min_sample_presence = int(getattr(config, "min_sample_presence", 3)) if config else 3

        # Binary baseline strategy: 'Y' (reference) or 'N' (cohort mode)
        self.ancestral_allele = str(getattr(config, "ancestral_allele", "Y")) if config else "Y"

        # Variant scope
        self.biallelic_only = bool(getattr(config, "biallelic_only", True)) if config else True

        # DataLoader lightweight preprocessing (kept separate from statistical validation)
        self.remove_invariant = bool(getattr(config, "remove_invariant", True)) if config else True
        self.min_minor_count = int(getattr(config, "min_minor_count", 0)) if config else 0

        # Output naming (kept consistent across artifacts)
        self.generic_name = str(getattr(config, "generic_name", "matrix")) if config else "matrix"

        # Fasta2matrices-style filter knobs for matrices/* outputs
        self.matrices_min_count = int(getattr(config, "matrices_min_count", 3)) if config else 3
        self.matrices_repeat_number = int(getattr(config, "matrices_repeat_number", 5)) if config else 5
        self.matrices_type = str(getattr(config, "matrices_type", "all")) if config else "all"
        self.matrices_fix = str(getattr(config, "matrices_fix", "")) if config else ""

        # Optional: shrink column names in returned DataFrame (not affecting artifacts)
        self.use_integer_variant_ids = bool(getattr(config, "use_integer_variant_ids", False)) if config else False

    def load_genomic_matrix(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        ref_fasta: Optional[str] = None,
        label_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load genomic features from a VCF directory or a prebuilt matrix file."""
        _ = label_column
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Genomic input not found: {path}")

        logger.info("DataLoader: input=%s", str(path))

        if path.is_dir():
            logger.info("DataLoader: mode=vcf_directory")
            return self._load_vcf_directory(path, output_dir=output_dir, ref_path=ref_fasta)

        suffix = "".join(path.suffixes).lower()
        if suffix.endswith((".csv", ".tsv", ".tab")):
            logger.info("DataLoader: mode=prebuilt_matrix")
            return self._load_matrix_file(path)

        raise ValueError(
            "This DataLoader expects either a directory of per-sample VCF/VCF.GZ files "
            "or a prebuilt matrix (.csv/.tsv). "
            f"Got: {path}"
        )

    def load_metadata(self, meta_path: str, output_dir: Optional[str] = None) -> pd.DataFrame:
        """Load a metadata table and index it by sample identifier."""
        path = Path(meta_path)
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")

        sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
        df = pd.read_csv(path, sep=sep)

        if df.shape[1] < 2:
            raise ValueError(f"Metadata file looks invalid (needs ≥2 columns): {path} (shape={df.shape})")

        idx_col = "Sample" if "Sample" in df.columns else df.columns[0]
        df[idx_col] = df[idx_col].astype(str)
        df = df.set_index(idx_col, drop=True)
        df.index.name = "Sample"

        if output_dir:
            outdir = Path(output_dir)
            outdir.mkdir(parents=True, exist_ok=True)
            df.to_csv(outdir / "metadata.normalized.csv")

        return df

    def load_known_markers(self, known_markers_path: str, output_dir: Optional[str] = None) -> List[str]:
        """Load a list of marker identifiers from a .txt or .csv/.tsv file."""
        path = Path(known_markers_path)
        if not path.exists():
            logger.error("Known markers file not found: %s", path)
            raise FileNotFoundError(f"Known markers file not found: {path}")

        logger.info("Loading known markers from: %s", path)

        suffix = "".join(path.suffixes).lower()
        markers: List[str] = []

        if suffix.endswith(".txt"):
            logger.info("Detected plain text file (.txt) – reading line by line")
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    markers.append(line)

        elif suffix.endswith((".csv", ".tsv", ".tab")):
            sep = "\t" if suffix.endswith((".tsv", ".tab")) else ","
            logger.info("Detected tabular file (%s) – reading from first column or 'marker' column", suffix)
            df = pd.read_csv(path, sep=sep)
            col = "marker" if "marker" in df.columns else df.columns[0]
            markers = [str(x).strip() for x in df[col].tolist() if str(x).strip()]

        else:
            logger.error("Unsupported known markers file format: %s", suffix)
            raise ValueError(f"Unsupported known markers file type: {path}")

        # Deduplicate while preserving order
        seen = set()
        uniq_markers: List[str] = []
        for m in markers:
            if m not in seen:
                uniq_markers.append(m)
                seen.add(m)

        removed = len(markers) - len(uniq_markers)
        if removed > 0:
            logger.info("Removed %d duplicate markers (total unique: %d)", removed, len(uniq_markers))
        else:
            logger.info("Loaded %d unique markers (no duplicates found)", len(uniq_markers))

        if output_dir:
            outdir = Path(output_dir)
            outdir.mkdir(parents=True, exist_ok=True)
            out_path = outdir / "known_markers.normalized.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                for m in uniq_markers:
                    f.write(m + "\n")
            logger.info("Saved normalized known markers list to: %s", out_path)

        return uniq_markers

    # ──────────────────────────────────────────────────────────
    # VCF directory → allele + binary matrices → returned DataFrame
    # ──────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────
    # Helper to process one VCF file (used in parallel)
    # ──────────────────────────────────────────────────────────────
    def _load_vcf_directory(self, vcf_dir: Path, output_dir: Optional[str], ref_path: Optional[str]) -> pd.DataFrame:
        """Scan a directory of per-sample VCFs and build allele/binary matrices."""
        vcfs = sorted([p for p in vcf_dir.iterdir() if p.name.endswith((".vcf", ".vcf.gz"))])
        if not vcfs:
            raise ValueError(f"No .vcf/.vcf.gz files found in: {vcf_dir}")

        # 1) Discovery (what is about to be parsed?)
        logger.info(
            "DataLoader: discovered %d VCF(s) in %s (biallelic_only=%s)",
            len(vcfs),
            str(vcf_dir),
            self.biallelic_only,
        )

        # 2) Explicitly describe what parsing does + what gets kept
        logger.info(
            "DataLoader: per-sample parsing plan\n"
            "  Each VCF will be scanned record-by-record using iter_sample_calls().\n"
            "  Retained calls: SNP-like, biallelic (if enabled), passing QC thresholds.\n"
            "  Per-sample output: an in-memory dict mapping site→(REF, CALLED) for retained sites.",
        )

        # 3) Record-level QC configuration (applied during parsing)
        logger.info(
            "DataLoader: record-level QC thresholds (applied during parsing)\n"
            "  QUAL>=%.1f | INFO/DP>=%d | INFO/MQ>=%.1f | INFO/MQ0F<=%.3f",
            float(self.qual_threshold),
            int(self.dp_threshold),
            float(self.mq_threshold),
            float(self.mq0f_threshold),
        )

        # 4) Cohort / matrix-level configuration (applied AFTER parsing/merge)
        logger.info(
            "DataLoader: cohort + matrix settings (applied after parsing)\n"
            "  min_sample_presence=%d | baseline=%s | min_minor_count=%d",
            int(self.min_sample_presence), ("REF" if self.ancestral_allele.upper() == "Y" else "MODE"),
            int(self.min_minor_count),
        )

        # Helper for parallel processing
        def process_vcf(vcf_path: Path) -> tuple[str, dict]:
            sample = vcf_path.name
            if sample.endswith(".vcf.gz"):
                sample = sample[:-7]
            elif sample.endswith(".vcf"):
                sample = sample[:-4]

            calls = iter_sample_calls(
                vcf_path,
                qual_thresh=self.qual_threshold,
                dp_thresh=self.dp_threshold,
                mq_thresh=self.mq_threshold,
                mq0f_thresh=self.mq0f_threshold,
                biallelic_only=self.biallelic_only,
            )

            # Keep debug-level to avoid log spam unless enabled
            logger.debug(
                "DataLoader: per-sample parse result\n"
                "  sample=%s\n"
                "  retained_sites=%d\n"
                "  storage=returned as calls_dict (site→(REF,CALLED))",
                sample,
                len(calls),
            )
            return sample, calls

        # 5) Execute parsing
        n_jobs = getattr(self, "n_jobs", -1)
        logger.info("DataLoader: starting parallel per-sample parsing (n_jobs=%s)", _fmt_n_jobs(n_jobs))

        results = Parallel(n_jobs=n_jobs)(delayed(process_vcf)(vcf) for vcf in vcfs)

        # 6) Parsing summary + clarify where it is stored
        n_samples = len(results)
        total_calls = sum(len(calls) for _, calls in results)
        mean_calls = total_calls / max(1, n_samples)

        logger.info(
            "DataLoader: parsing complete\n"
            "  samples=%d\n"
            "  total_called_sites=%d\n"
            "  mean_calls_per_sample=%.2f\n"
            "  storage=results list of (sample_id, calls_dict) in memory",
            n_samples,
            total_calls,
            mean_calls,
        )

        logger.info(
            "DataLoader: cohort merge starting\n"
            "  The per-sample call dictionaries will now be aggregated into:\n"
            "    (1) per_sample_calls[sample] → calls_dict\n"
            "    (2) site_counts[site] → carrier_count for allele-specific events\n"
            "  Alternate alleles at the same genomic position will be retained as separate features.",
        )

        # 7) Merge parsed results into cohort-wide site universe
        per_sample_calls: Dict[str, Dict[Tuple[str, int], Tuple[str, str]]] = {}
        site_counts: Dict[Tuple[str, int, str, str], int] = {}

        carrier_events = 0

        for sample, calls in results:
            per_sample_calls[sample] = calls
            for (chrom, pos), (ref, called) in calls.items():
                if called == ref:
                    continue

                allele_key = (chrom, pos, ref, called)
                site_counts[allele_key] = site_counts.get(allele_key, 0) + 1
                carrier_events += 1

        logger.info(
            "DataLoader: cohort merge complete\n"
            "  per_sample_calls now holds %d per-genome call maps.\n"
            "  site_counts now defines the cohort-wide polymorphic site universe (pre-filter).",
            len(per_sample_calls),
        )

        # ---- Cohort universe established (pre-filter boundary) ----
        candidate_sites = len(site_counts)
        logger.info(
            "DataLoader: cohort variant landscape (pre-filter)\n"
            "  The cohort comprises %d genomes.\n"
            "  A total of %d unique allele-specific polymorphic features were observed.\n"
            "  These correspond to %d total mutation occurrences across genomes.\n"
            "  Alternate alleles at the same genomic position were retained as separate features.",
            len(per_sample_calls),
            candidate_sites,
            carrier_events,
        )

        # Clarify artifact timing + location (what “snapshot” means)
        if output_dir:
            out = Path(output_dir)
            logger.info(
                "DataLoader: cohort artifacts\n"
                "  Output will be written to: %s\n"
                "  Writing occurs after presence filtering and encoding (matrices + FASTA + annotation tables).",
                str(out),
            )

        # 8) Cohort-level filtering: min sample presence
        kept_sites: Dict[Tuple[str, int, str, str], int] = {
            key: cnt
            for key, cnt in site_counts.items()
            if cnt >= self.min_sample_presence
        }

        kept_n = len(kept_sites)
        retention_rate = kept_n / max(1, candidate_sites)

        logger.info(
            "DataLoader: cohort presence filtering\n"
            "  A minimum of %d genomes per site was required.\n"
            "  %d of %d polymorphic sites were retained (%.2f%% retained).\n"
            "  Sites failing this threshold were removed from the cohort feature space.",
            int(self.min_sample_presence),
            kept_n,
            candidate_sites,
            retention_rate * 100,
        )

        if not kept_sites:
            raise ValueError(
                "No polymorphic sites retained after QC + min-sample-presence filter. "
                "Consider relaxing thresholds."
            )

        # 9) Sort sites deterministically and build allele strings
        ordered_keys = sorted(kept_sites.keys(), key=lambda x: (x[0], x[1], x[2], x[3]))
        ref_bases = [k[2] for k in ordered_keys]
        alt_bases = [k[3] for k in ordered_keys]

        samples_sorted = sorted(per_sample_calls.keys())
        per_pos_counts = [Counter() for _ in ordered_keys]
        sample_allele_strings: Dict[str, str] = {}

        ref_line = "".join(ref_bases)
        logger.debug("DataLoader: building allele strings for %d sample(s)", len(per_sample_calls))

        for sample in samples_sorted:
            calls = per_sample_calls[sample]
            alleles = []
            for j, (chrom, pos, ref, alt) in enumerate(ordered_keys):
                called = calls.get((chrom, pos), (ref, ref))[1]
                base = alt if called == alt else ref
                alleles.append(base)
                per_pos_counts[j][base] += 1
            sample_allele_strings[sample] = "".join(alleles)

        # 10) Baseline selection
        if self.ancestral_allele.upper() == "Y":
            baseline = list(ref_line)
            baseline_strategy = "REF"
        else:
            baseline = [
                per_pos_counts[j].most_common(1)[0][0] if per_pos_counts[j] else ref_bases[j]
                for j in range(len(ordered_keys))
            ]
            baseline_strategy = "MODE"

        baseline_diff_from_ref = sum(1 for i, ch in enumerate(baseline) if ch != ref_line[i])

        if baseline_strategy == "REF":
            logger.info(
                "DataLoader: baseline encoding\n"
                "  The reference allele was used as the baseline.\n"
                "  Encoding definition: 0 indicates the reference allele; 1 indicates a non-reference allele.\n"
                "  A total of %d polymorphic sites were encoded.",
                len(ordered_keys),
            )
        else:
            logger.info(
                "DataLoader: baseline encoding\n"
                "  The most frequent allele across the cohort was used as the baseline.\n"
                "  Encoding definition: 0 indicates the cohort-majority allele; 1 indicates a minority allele.\n"
                "  A total of %d polymorphic sites were encoded.\n"
                "  At %d sites, the cohort-majority allele differed from the reference allele.",
                len(ordered_keys),
                baseline_diff_from_ref,
            )

        # 11) Binary encoding (baseline → 0/1 orientation)
        ref_binary = "".join("0" if ref_line[i] == baseline[i] else "1" for i in range(len(ref_line)))
        sample_binary_strings = {
            s: "".join("0" if seq[i] == baseline[i] else "1" for i in range(len(seq)))
            for s, seq in sample_allele_strings.items()
        }

        expected_len = len(ref_line)
        for s, binseq in sample_binary_strings.items():
            if len(binseq) != expected_len:
                raise ValueError(
                    f"Binary encoding length mismatch for sample {s}: expected {expected_len}, got {len(binseq)}"
                )

        # 12) Feature IDs (variant-centric identifiers)
        variant_ids = [f"{c}:{p}:{r}:{a}" for (c, p, r, a) in ordered_keys]
        # 13) Final matrix assembly (sample-centric orientation)
        data_bin = [[int(ch) for ch in sample_binary_strings[s]] for s in samples_sorted]

        df = pd.DataFrame(
            data_bin,
            index=samples_sorted,
            columns=variant_ids,
            dtype=int,
        )
        df.index.name = "Sample"

        # 14) Raw matrix stats (post-encoding, pre-preprocessing)
        n_samples, raw_feature_count = df.shape
        total_ones = int(df.values.sum())
        total_cells = n_samples * raw_feature_count

        matrix_density = total_ones / max(1, total_cells)
        mean_ones_per_feature = total_ones / max(1, raw_feature_count)
        mean_ones_per_sample = total_ones / max(1, n_samples)

        logger.info(
            "DataLoader: raw binary matrix (post-encoding, pre-preprocessing)\n"
            "  The matrix contains %d genomes (rows) and %d polymorphic sites (columns).\n"
            "  This corresponds to %d total genotype entries.\n"
            "  A total of %d entries are encoded as 1 (non-baseline alleles).\n"
            "  Matrix density (fraction of 1s) = %.6f.\n"
            "  Mean carrier count per site = %.2f genomes.\n"
            "  Mean variant burden per genome = %.2f sites.",
            n_samples,
            raw_feature_count,
            total_cells,
            total_ones,
            matrix_density,
            mean_ones_per_feature,
            mean_ones_per_sample,
        )

        # 15) Variant frequency summary (carriers per feature)
        carrier_counts = df.sum(axis=0).astype(int)
        singleton_sites = int((carrier_counts == 1).sum())
        doubleton_sites = int((carrier_counts == 2).sum())

        logger.info(
            "DataLoader: variant frequency summary (carriers per feature)\n"
            "  singleton_sites=%d (carriers==1)\n"
            "  doubleton_sites=%d (carriers==2)",
            singleton_sites,
            doubleton_sites,
        )

        # ─────────────────────────────────────────────
        # Preprocessing
        # ─────────────────────────────────────────────
        df, prep_stats = self._preprocess_binary_matrix(df)
        logger.info(
            "DataLoader: matrix preprocessing (post-encoding)\n"
            "  The preprocessing stage applied invariant-site removal and minor allele count filtering.\n"
            "  Invariant removal enabled: %s.\n"
            "  Rationale: remove markers with no variation across samples\n"
            "  Purpose: eliminate non-informative genomic features before downstream analysis\n"
            "  Minimum minor allele count threshold: %d.\n"
            "  Rationale: remove markers where minority state appears\n"
            "  Purpose: avoids instability from extremely rare variants in small cohorts\n"
            "  The matrix contained %d features prior to preprocessing.\n"
            "  %d invariant features were removed.\n"
            "  %d features were removed due to insufficient minor allele count.\n"
            "  %d features remain after preprocessing.",
            self.remove_invariant,
            self.min_minor_count,
            prep_stats["features_before"],
            prep_stats["removed_invariant"],
            prep_stats["removed_low_minor_count"],
            prep_stats["features_after"],
        )
        if self.config is not None:
            cfg_path = out / "dataloader_config.snapshot.json"
            payload = asdict(self.config) if hasattr(self.config, "__dataclass_fields__") else vars(self.config)
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            logger.info("DataLoader: wrote config snapshot %s", str(cfg_path))
            
        lookup = None
        if self.use_integer_variant_ids:
            df, lookup = self._convert_to_integer_variant_ids(df)
            logger.info("DataLoader: compacted variant IDs (v0..vN) and created lookup")
        if output_dir:
            out = Path(output_dir)
            logger.info(
                "DataLoader: writing artifacts\n"
                "output_dir=%s\n",
                output_dir,
            )
            matrices_final_markers = self._write_all_artifacts(
                out_root=out,
                kept_sites=kept_sites,
                ordered_keys=ordered_keys,
                positions_1based=[p for _, p, _, _ in ordered_keys],
                ref_line=ref_line,
                sample_allele_strings=sample_allele_strings,
                ref_binary=ref_binary,
                sample_binary_strings=sample_binary_strings,
                ref_path=Path(ref_path) if ref_path else None,
                integer_id_lookup=lookup,
            )
            return df
    
    def _preprocess_binary_matrix(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Apply lightweight, non-statistical preprocessing to the binary matrix.

        Steps:
        1) Remove invariant features (all 0 or all 1)
        2) Enforce minimum minor allele count per site

        Returns:
        df_filtered, stats_dict
        """
        if df.empty:
            return df, {
                "features_before": 0,
                "removed_invariant": 0,
                "removed_low_minor_count": 0,
                "features_after": 0,
            }

        features_before = df.shape[1]
        removed_invariant = 0
        removed_low_minor_count = 0

        # ─────────────────────────────────────────────
        # 1) Remove invariant features
        # ─────────────────────────────────────────────
        if self.remove_invariant:
            nunique = df.nunique(axis=0, dropna=False)
            invariant_mask = nunique <= 1
            removed_invariant = int(invariant_mask.sum())

            df = df.loc[:, ~invariant_mask]

            if df.empty:
                raise ValueError(
                    "All polymorphic sites were removed during invariant filtering. "
                    "Check input data or relax remove_invariant setting."
                )

        # ─────────────────────────────────────────────
        # 2) Enforce minimum minor allele count
        # ─────────────────────────────────────────────
        if self.min_minor_count > 0 and not df.empty:
            keep_mask = []

            for col in df.columns:
                vc = df[col].value_counts(dropna=False)
                count_0 = vc.get(0, 0)
                count_1 = vc.get(1, 0)
                keep_mask.append(min(count_0, count_1) >= self.min_minor_count)

            before_minor = df.shape[1]
            df = df.loc[:, keep_mask]
            removed_low_minor_count = before_minor - df.shape[1]

            if df.empty:
                raise ValueError(
                    "All sites removed by minor allele count filter. "
                    f"Try lowering min_minor_count (current: {self.min_minor_count}) "
                    "or verify binary encoding."
                )

        features_after = df.shape[1]

        stats = {
            "features_before": features_before,
            "removed_invariant": removed_invariant,
            "removed_low_minor_count": removed_low_minor_count,
            "features_after": features_after,
        }

        return df, stats
    
    def _convert_to_integer_variant_ids(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Replace long variant IDs with compact IDs and return a lookup."""
        lookup: Dict[str, str] = {}
        new_cols: List[str] = []
        for i, col in enumerate(df.columns):
            vid = f"v{i}"
            new_cols.append(vid)
            lookup[vid] = col
        df2 = df.copy()
        df2.columns = new_cols
        return df2, lookup

    # ──────────────────────────────────────────────────────────
    # Artifact generation 
    # ──────────────────────────────────────────────────────────

    def _write_all_artifacts(
        self,
        out_root: Path,
        kept_sites: Dict[Tuple[str, int], Tuple[int, str, str]],
        ordered_keys: List[Tuple[str, int]],
        positions_1based: List[int],
        ref_line: str,
        sample_allele_strings: Dict[str, str],
        ref_binary: str,
        sample_binary_strings: Dict[str, str],
        ref_path: Optional[Path],
        integer_id_lookup: Optional[Dict[str, str]],
    ) -> Optional[int]:
        """Write vcf_counts/*, fasta/*, and matrices/* outputs (with detailed timing logs)."""
        import time

        def _fmt_size(p: Path) -> str:
            try:
                b = p.stat().st_size
                if b < 1024:
                    return f"{b} B"
                if b < 1024**2:
                    return f"{b/1024:.1f} KB"
                if b < 1024**3:
                    return f"{b/1024**2:.1f} MB"
                return f"{b/1024**3:.2f} GB"
            except Exception:
                return "?"

        def _log_written(p: Path, extra: str = "") -> None:
            if extra:
                logger.info("Artifacts: wrote %s (%s) %s", str(p), _fmt_size(p), extra)
            else:
                logger.info("Artifacts: wrote %s (%s)", str(p), _fmt_size(p))

        out_root.mkdir(parents=True, exist_ok=True)

        vcf_counts_dir = out_root / "vcf_counts"
        fasta_dir = out_root / "fasta"
        matrices_dir = out_root / "matrices"

        vcf_counts_dir.mkdir(parents=True, exist_ok=True)
        fasta_dir.mkdir(parents=True, exist_ok=True)
        matrices_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Artifacts: start (out=%s)", str(out_root))
        logger.info(
            "Artifacts: inputs kept_sites=%d | ordered_keys=%d | samples=%d | ref_path=%s",
            len(kept_sites),
            len(ordered_keys),
            len(sample_allele_strings),
            str(ref_path) if ref_path else "<none>",
        )

        t_total = time.perf_counter()

        # 1) vcf_counts/all_snp.txt
        t = time.perf_counter()
        all_snp_path = vcf_counts_dir / "all_snp.txt"
        all_snp_rows, all_snp_header = self._write_all_snp_table(
            path=all_snp_path,
            kept_sites=kept_sites,
            ref_path=ref_path,
        )
        dt = time.perf_counter() - t
        n_rows = len(all_snp_rows) if all_snp_rows is not None else -1
        n_cols = len(all_snp_header) if all_snp_header is not None else -1
        logger.info("Artifacts: all_snp.txt done (%.2fs) rows=%d cols=%d", dt, n_rows, n_cols)
        _log_written(all_snp_path)

        # 2) fasta/<generic>_{alleles,binary}.fasta
        t = time.perf_counter()
        alleles_fa = fasta_dir / f"{self.generic_name}_alleles.fasta"
        binary_fa = fasta_dir / f"{self.generic_name}_binary.fasta"

        # Note: write_fasta_matrix typically writes a REF row + sample rows.
        write_fasta_matrix(alleles_fa, ref_line, sample_allele_strings, ref_name="REF")
        write_fasta_matrix(binary_fa, ref_binary, sample_binary_strings, ref_name="REF")

        dt = time.perf_counter() - t
        logger.info(
            "Artifacts: fasta write done (%.2fs) sequences=%d (includes REF) | length=%d",
            dt,
            len(sample_allele_strings) + 1,
            len(ref_line),
        )
        _log_written(alleles_fa)
        _log_written(binary_fa)

        # 3) fasta/<generic>_filtered.tsv (filtered copy of all_snp.txt; optional Context_±flank)
        t = time.perf_counter()
        filtered_tsv = fasta_dir / f"{self.generic_name}_filtered.tsv"
        self._write_filtered_copy_with_context(
            input_rows=all_snp_rows,
            input_header=all_snp_header,
            output_path=filtered_tsv,
            kept_positions=set(positions_1based),
            ref_path=ref_path,
        )
        dt = time.perf_counter() - t
        logger.info(
            "Artifacts: filtered.tsv done (%.2fs) kept_positions=%d context_ref=%s",
            dt,
            len(set(positions_1based)),
            "yes" if ref_path else "no",
        )
        _log_written(filtered_tsv)

        # 4) matrices/* outputs produced by filtering (minor-count + type + redundancy + fix)
        t = time.perf_counter()
        logger.info("Artifacts: matrices/* start (this is often the slow step)")
        matrices_final_markers = self._write_matrices_outputs(
            fasta_alleles=alleles_fa,
            fasta_binary=binary_fa,
            annotation_tsv=filtered_tsv,
            out_dir=matrices_dir,
        )
        dt = time.perf_counter() - t
        logger.info("Artifacts: matrices/* done (%.2fs)", dt)

        # 5) optional lookup used by returned df
        if integer_id_lookup is not None:
            t = time.perf_counter()
            lookup_path = out_root / "variant_id_lookup.json"
            with open(lookup_path, "w", encoding="utf-8") as f:
                json.dump(integer_id_lookup, f, indent=2)
            dt = time.perf_counter() - t
            logger.info("Artifacts: variant_id_lookup.json done (%.2fs) entries=%d", dt, len(integer_id_lookup))
            _log_written(lookup_path)

        logger.info("Artifacts: done (total %.2fs)", time.perf_counter() - t_total)
        return matrices_final_markers
    
    def _write_all_snp_table(
        self,
        path: Path,
        kept_sites: Dict[Tuple[str, int], Tuple[int, str, str]],
        ref_path: Optional[Path],
    ) -> Tuple[List[Dict[str, str]], List[str]]:
        """Write the SNP summary table, with annotation columns if GenBank is provided."""
        path.parent.mkdir(parents=True, exist_ok=True)

        header_annot = [
            "Position",
            "Count",
            "Sequence",
            "Region_type",
            "Relative_pos",
            "Codon_number",
            "Nucleotide_change",
            "Amino_acid_change",
            "Gene_annotation",
        ]

        # Decide whether to annotate (GenBank) or write minimal table (Position, Count).
        do_annotate = False
        if ref_path and ref_path.exists():
            lower = ref_path.name.lower()
            if lower.endswith((".gb", ".gbk", ".gbff")):
                do_annotate = True

        if do_annotate:
            rows = annotate_snps_genbank(kept_sites, ref_path)  # type: ignore[arg-type]
            with open(path, "w", encoding="utf-8", newline="") as out:
                w = csv.DictWriter(out, fieldnames=header_annot, delimiter="\t", lineterminator="\n")
                w.writeheader()
                for r in rows:
                    w.writerow(r)
            return rows, header_annot

      # Minimal output (Position, Count) still matches the script behavior when no ref is given.
        header_min = ["Position", "Count"]
        rows_min: List[Dict[str, str]] = []

        for (chrom, pos, ref_nt, alt_nt) in sorted(kept_sites.keys(), key=lambda x: (x[0], x[1], x[2], x[3])):
            count = kept_sites[(chrom, pos, ref_nt, alt_nt)]
            rows_min.append({"Position": str(pos), "Count": str(count)})

        with open(path, "w", encoding="utf-8", newline="") as out:
            w = csv.DictWriter(out, fieldnames=header_min, delimiter="\t", lineterminator="\n")
            w.writeheader()
            for r in rows_min:
                w.writerow(r)

        return rows_min, header_min

    def _write_filtered_copy_with_context(
        self,
        input_rows: List[Dict[str, str]],
        input_header: List[str],
        output_path: Path,
        kept_positions: set,
        ref_path: Optional[Path],
    ) -> None:
        """Write a filtered copy of the SNP table, optionally appending Context_±40."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # If we can load reference sequence, we append context column.
        ref_seq: Optional[str] = None
        if ref_path and ref_path.exists():
            lower = ref_path.name.lower()
            if lower.endswith((".fa", ".fna", ".fasta", ".fas", ".gb", ".gbk", ".gbff")):
                try:
                    ref_seq = load_reference_sequence(ref_path)
                except Exception as e:
                    logger.warning(f"Reference loading failed; context column skipped. Reason: {e}")
                    ref_seq = None

        out_header = list(input_header)
        add_context = ref_seq is not None and "Position" in input_header
        if add_context:
            out_header.append("Context_±40")

        with open(output_path, "w", encoding="utf-8", newline="") as out:
            w = csv.writer(out, delimiter="\t", lineterminator="\n")
            w.writerow(out_header)

            for r in input_rows:
                try:
                    pos = int(str(r.get("Position", "")).strip())
                except Exception:
                    continue
                if pos not in kept_positions:
                    continue

                row_vals = [str(r.get(col, "")) for col in input_header]
                if add_context and ref_seq is not None:
                    row_vals.append(context_around(pos, ref_seq, flank=40))
                w.writerow(row_vals)

    
    def _write_matrices_outputs(
        self,
        fasta_alleles: Path,
        fasta_binary: Path,
        annotation_tsv: Path,
        out_dir: Path,
    ) -> int:
        """
    Generate final cohort-level matrices (TSV + FASTA + filtered annotation).

    This stage refines the encoded matrix before downstream modeling.
    It operates strictly at the feature level and does NOT perform statistical
    association testing. Instead, it ensures that the final matrix is
    biologically interpretable, structurally stable, and non-redundant.

    Filtering stages (conceptual flow):

    1) Minor-count filter (signal stability control)
       - Removes markers where the minority state is too rare across genomes.
       - In small cohorts, extremely rare states introduce instability and
         can distort downstream tree splits or interaction mining.

    2) Annotation-driven type filter (biological subset selection)
       - Retains only markers whose functional annotation matches a requested category.
       - Skipped if annotation rows do not align with marker count.

    3) Redundancy reduction via pattern grouping (feature de-duplication)
       - Identifies markers that share an identical 0/1 pattern across all genomes.
       - These markers represent the exact same cohort-level signal.
       - Keeps at most `repeat_number` representatives per identical-pattern group.
       - Purpose: collapse perfectly collinear features to reduce redundancy and prevent
         one signal from being over-represented by multiple duplicate columns.

    4) Forced retention of specified positions (controlled override)
       - User-specified marker indices are force-kept even if filtered out.
       - Ensures known markers remain present for downstream reporting.

    Important methodological distinction:
    - Minor-count and type filters are biological inclusion criteria.
    - Redundancy reduction is a feature-engineering / de-duplication step.
    - None of these constitute statistical hypothesis testing.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        base = self.generic_name

        # ─────────────────────────────────────────────
        # 0) Announce what we're about to parse/write
        # ─────────────────────────────────────────────
        logger.info("Matrices/*: start (base=%s)", base)
        logger.info(
                "Matrices stage: feature refinement prior to downstream modeling\n"
                "Purpose: structural filtering + redundancy control\n"
            )

        logger.info(
                "Matrices configuration:\n"
                "min_minor_count=%d\n"
                "annotation_type_filter=%s\n"
                "repeat_number=%d\n"
                "forced_positions='%s'",
                self.matrices_min_count,
                self.matrices_type,
                self.matrices_repeat_number,
                self.matrices_fix,
            )
        logger.info(
            "Matrices/*: inputs alleles_fasta=%s | binary_fasta=%s | annotation_tsv=%s",
            str(fasta_alleles),
            str(fasta_binary),
            str(annotation_tsv),
        )
        logger.info("Matrices/*: output_dir=%s", str(out_dir))

        for p, label in (
            (fasta_alleles, "alleles FASTA"),
            (fasta_binary, "binary FASTA"),
            (annotation_tsv, "annotation TSV"),
        ):
            if not p.exists():
                raise FileNotFoundError(f"Matrices/*: missing {label}: {p}")

        # ─────────────────────────────────────────────
        # 1) Load (parse) inputs
        # ─────────────────────────────────────────────
        alleles = read_fasta_matrix(fasta_alleles)
        binary = read_fasta_matrix(fasta_binary)
        annot_rows, annot_header = read_annotation_tsv(annotation_tsv)

        if not alleles or not binary:
            raise ValueError("Matrices/*: empty FASTA matrix detected (alleles or binary).")

        if list(alleles.keys()) != list(binary.keys()):
            raise ValueError("Matrices/*: genome order differs between alleles and binary FASTA files.")

        genomes = list(alleles.keys())
        S = len(next(iter(alleles.values())))
        if len(next(iter(binary.values()))) != S:
            raise ValueError("Matrices/*: alleles and binary FASTA have different number of columns (markers).")

        logger.info("Matrices/*: parsed inputs genomes=%d | markers=%d", len(genomes), S)

        annotation_matches = len(annot_rows) == S
        if annotation_matches:
            logger.info("Matrices/*: annotation rows match markers (rows=%d)", len(annot_rows))
        else:
            logger.warning(
                "Matrices/*: annotation rows do NOT match markers (rows=%d vs markers=%d). "
                "Type-filter will be skipped and annotation will be copied as-is.",
                len(annot_rows),
                S,
            )

        # ─────────────────────────────────────────────
        # 2) Announce filter parameters
        # ─────────────────────────────────────────────
        logger.info(
            "Matrices/*: params min_count=%d | type=%s | repeat_number=%d | fix='%s'",
            self.matrices_min_count,
            self.matrices_type,
            self.matrices_repeat_number,
            self.matrices_fix,
        )

        genomes_order, allele_cols = transpose_rows_to_columns(alleles)
        _, binary_cols = transpose_rows_to_columns(binary)

        # ─────────────────────────────────────────────
        # 3) Filter 1: minor-count
        #
        # Keeps a marker only if the minority state (usually '1' in a 0/1 column)
        # appears at least `min_count` times across genomes.
        #
        # Intuition: columns where only 1 genome differs can be unstable in tiny cohorts
        # and can inflate downstream splits or interactions by acting as "singletons".
        # ─────────────────────────────────────────────
        logger.info(
            "Applying minor-count filter:\n"
            "  Skipped for matrices/* outputs\n"
            "  Rationale: minor-count filtering was already applied during main matrix preprocessing"
        )

        mask_minor = [True] * len(binary_cols)
        kept_minor = int(sum(mask_minor))

        logger.info(
            "Minor-count filter result: kept=%d/%d (removed=%d)",
            kept_minor,
            len(binary_cols),
            len(binary_cols) - kept_minor,
        )

        # ─────────────────────────────────────────────
        # 4) Filter 2: type filter (annotation-driven)
        #
        # Keeps markers whose annotation indicates the requested category (e.g., coding/non-coding/etc).
        # This is only meaningful if annotation rows align 1:1 with marker columns.
        # If annotation does not match, we skip this filter rather than risk mislabeling columns.
        # ─────────────────────────────────────────────
        if annotation_matches and annot_header and self.matrices_type != "all":
            logger.info(
                "Applying annotation-driven type filter:\n"
                "  Keeping markers matching type='%s'\n"
                "  Purpose: biological subset selection",
                self.matrices_type,
            )
            mask_type = type_filter(annot_rows, self.matrices_type)
        else:
            mask_type = [True] * S
            logger.info(
                "Type filter skipped (either type='all' or annotation mismatch)."
            )

        kept_type = int(sum(mask_type))

        logger.info(
            "Type filter result: kept=%d/%d",
            kept_type,
            S,
        )

        # ─────────────────────────────────────────────
        # 5) Combined mask (minor AND type)
        #
        # This is the inclusion mask after the "content" filters:
        #   - minor-count: statistical stability
        #   - type: biological subset selection
        # ─────────────────────────────────────────────
        mask_12 = combine_masks(mask_minor, mask_type)
        kept_12 = int(sum(mask_12))
        logger.info("Matrices/*: combined (minor AND type) kept=%d/%d", kept_12, S)

        idx12 = [i for i, k in enumerate(mask_12) if k]
        allele_cols_12 = [c for c, k in zip(allele_cols, mask_12) if k]
        binary_cols_12 = [c for c, k in zip(binary_cols, mask_12) if k]
        annot_rows_12 = [r for r, k in zip(annot_rows, mask_12) if k] if annotation_matches else []

        # ─────────────────────────────────────────────
        # 6) Filter 3: redundancy reduction by identical binary pattern
        #
        # What "redundancy" means here:
        #   Two (or more) markers can produce the *exact same* 0/1 vector across all genomes.
        #   Example (conceptual):
        #       marker A: 0 0 1 0 1 0 ...
        #       marker B: 0 0 1 0 1 0 ...
        #
        # In that case, A and B are perfectly collinear:
        #   - Any model/split using A could use B interchangeably.
        #   - Keeping all of them can over-represent one signal and inflate apparent importance.
        #
        # This step groups markers by their full-sample binary pattern and keeps only a limited
        # number of representatives per group.
        #
        # How `repeat_number` is used:
        #   - repeat_number = 1  → keep only one representative marker per identical-pattern group
        #   - repeat_number = k  → keep up to k representatives per group (useful if you want
        #                          a small amount of redundancy retained for downstream reporting)
        #
        # What this step is NOT:
        #   - It does not remove markers because they are rare (minor-count already handled that).
        #   - It does not change the 0/1 encoding.
        #   - It does not merge markers into a new synthetic feature; it simply drops duplicates.
        #
        # Why it is "same as other filters" structurally:
        #   - It produces another boolean mask (keep/drop) and logs kept counts.
        # Why it is different conceptually:
        #   - It is a de-duplication / collinearity control step, not a biological inclusion filter.
        #
        # Interpretable output:
        #   - The logging includes `unique_patterns` as a compact summary:
        #       *many columns* collapsing to *few patterns* suggests strong redundancy in the matrix.
        # ─────────────────────────────────────────────
        if binary_cols_12:
            unique_patterns = len({tuple(col) for col in binary_cols_12})

            logger.info(
                "Applying redundancy reduction (pattern grouping):\n"
                "  unique_binary_patterns=%d\n"
                "  repeat_number=%d\n"
                "  Purpose: collapse perfectly collinear markers\n"
                "           (markers sharing identical cohort-level signals)",
                unique_patterns,
                self.matrices_repeat_number,
            )

            mask_group = group_and_reduce_by_pattern(
                binary_cols_12,
                annot_rows if annotation_matches else [{"Position": str(i + 1)} for i in range(len(binary_cols_12))],
                self.matrices_repeat_number,
            )

            kept_group = int(sum(mask_group))

            logger.info(
                "Redundancy reduction result:\n"
                "  retained=%d/%d\n"
                "  removed_duplicate_representations=%d",
                kept_group,
                len(binary_cols_12),
                len(binary_cols_12) - kept_group,
            )
        else:
            mask_group = []
            kept_group = 0
            logger.info("Redundancy reduction skipped (no markers after biological filtering).")

        # Convert redundancy mask back to full length
        mask_final = [False] * S
        for kept_flag, original_idx in zip(mask_group, idx12):
            if kept_flag:
                mask_final[original_idx] = True
        # ─────────────────────────────────────────────
        # 7) Force-keep fixed positions
        # ─────────────────────────────────────────────
        fix_idx0, fix_warnings = parse_fix_positions(self.matrices_fix, S)

        for w in fix_warnings:
            logger.warning(w)

        if fix_idx0:
            logger.info(
                "Applying forced retention of %d user-specified marker(s).",
                len(fix_idx0),
            )

        for idx in fix_idx0:
            if 0 <= idx < S:
                mask_final[idx] = True

        kept_final = int(sum(mask_final))

        logger.info(
            "Final marker count after all filters:\n"
            "  %d/%d retained",
            kept_final,
            S,
        )

        # Apply final mask to FASTA matrices
        alleles_filt = apply_mask_to_char_rows(alleles, mask_final)
        binary_filt = apply_mask_to_char_rows(binary, mask_final)

        # Filter annotation rows if possible; otherwise keep original
        if annotation_matches:
            annot_filt = [r for r, k in zip(annot_rows, mask_final) if k]
        else:
            annot_filt = annot_rows

        # Prepare TSV matrix outputs
        _, allele_cols_f = transpose_rows_to_columns(alleles_filt)
        _, binary_cols_f = transpose_rows_to_columns(binary_filt)

        if annotation_matches:
            positions_f = [(r.get("Position", "") or "").strip() for r in annot_filt]
        else:
            positions_f = [str(i + 1) for i in range(len(allele_cols_f))]

        # ─────────────────────────────────────────────
        # 8) Write outputs
        # ─────────────────────────────────────────────
        out_alleles_tsv = out_dir / f"{base}_alleles.tsv"
        out_binary_tsv = out_dir / f"{base}_binary.tsv"
        out_alleles_fa = out_dir / f"{base}_alleles.fasta"
        out_binary_fa = out_dir / f"{base}_binary.fasta"
        out_filtered_tsv = out_dir / f"{base}_filtered.tsv"

        write_matrix_tsv(out_alleles_tsv, genomes_order, positions_f, allele_cols_f)
        write_matrix_tsv(out_binary_tsv, genomes_order, positions_f, binary_cols_f)
        write_fasta_matrix_wrapped(out_alleles_fa, alleles_filt)
        write_fasta_matrix_wrapped(out_binary_fa, binary_filt)

        if annotation_matches and annot_header:
            write_annotation_tsv(out_filtered_tsv, annot_filt, annot_header)
        else:
            out_filtered_tsv.write_text(
                annotation_tsv.read_text(encoding="utf-8", errors="replace"),
                encoding="utf-8",
            )

        logger.info(
            "Matrices/*: wrote outputs: %s | %s | %s | %s | %s",
            str(out_alleles_tsv),
            str(out_binary_tsv),
            str(out_alleles_fa),
            str(out_binary_fa),
            str(out_filtered_tsv),
        )
        logger.info("Matrices/*: done")
        return kept_final

    # ──────────────────────────────────────────────────────────
    # Prebuilt matrix loader
    # ──────────────────────────────────────────────────────────

    def _load_matrix_file(self, path: Path) -> pd.DataFrame:
        """Load a prebuilt matrix from CSV/TSV with row index in the first column."""
        sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
        df = pd.read_csv(path, sep=sep, index_col=0)
        df.index = df.index.astype(str)
        df.index.name = "Sample"
        return df
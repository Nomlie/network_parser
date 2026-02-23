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

import csv
import gzip
import json
import logging
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from joblib import Parallel, delayed
import pandas as pd

logger = logging.getLogger(__name__)

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

    for (chrom, pos) in sorted(snp_details.keys(), key=lambda x: (x[0], x[1])):
        count, ref_nt, alt_nt = snp_details[(chrom, pos)]
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

        if path.is_dir():
            return self._load_vcf_directory(path, output_dir=output_dir, ref_path=ref_fasta)

        suffix = "".join(path.suffixes).lower()
        if suffix.endswith((".csv", ".tsv", ".tab")):
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
            return sample, calls

        # Parallel parsing
        n_jobs = getattr(self, 'n_jobs', -1)
        logger.info("Starting parallel VCF parsing with n_jobs=%s", n_jobs if n_jobs > 0 else "all cores")

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_vcf)(vcf) for vcf in vcfs
        )

        logger.info("Parallel parsing finished — %d files processed successfully", len(results))

        # Merge parsed results
        per_sample_calls: Dict[str, Dict[Tuple[str, int], Tuple[str, str]]] = {}
        site_counts: Dict[Tuple[str, int], List] = {}

        for sample, calls in results:
            per_sample_calls[sample] = calls
            for key, (ref, called) in calls.items():
                if called == ref:
                    continue
                if key not in site_counts:
                    site_counts[key] = [0, ref, called]
                else:
                    if self.biallelic_only and called != site_counts[key][2]:
                        continue
                site_counts[key][0] += 1

        # Cohort-level filtering
        kept_sites: Dict[Tuple[str, int], Tuple[int, str, str]] = {
            key: (cnt, ref, alt) for key, (cnt, ref, alt) in site_counts.items()
            if cnt >= self.min_sample_presence
        }

        if not kept_sites:
            raise ValueError(
                "No polymorphic sites retained after QC + min-sample-presence filter. "
                "Consider relaxing thresholds."
            )

        # Sort sites deterministically
        ordered_keys = sorted(kept_sites.keys(), key=lambda x: (x[0], x[1]))
        ref_bases = [kept_sites[k][1] for k in ordered_keys]
        alt_bases = [kept_sites[k][2] for k in ordered_keys]

        samples_sorted = sorted(per_sample_calls.keys())
        per_pos_counts = [Counter() for _ in ordered_keys]
        sample_allele_strings: Dict[str, str] = {}

        ref_line = "".join(ref_bases)

        for sample in samples_sorted:
            calls = per_sample_calls[sample]
            alleles = []
            for j, key in enumerate(ordered_keys):
                ref = ref_bases[j]
                alt = alt_bases[j]
                _, called = calls.get(key, (ref, ref))
                base = called if called in {ref, alt} else ref
                alleles.append(base)
                per_pos_counts[j][base] += 1
            sample_allele_strings[sample] = "".join(alleles)

        # Baseline selection
        if self.ancestral_allele.upper() == "Y":
            baseline = list(ref_line)
        else:
            baseline = [
                per_pos_counts[j].most_common(1)[0][0] if per_pos_counts[j] else ref_bases[j]
                for j in range(len(ordered_keys))
            ]

        # Binary encoding
        ref_binary = "".join("0" if ref_line[i] == baseline[i] else "1" for i in range(len(ref_line)))
        sample_binary_strings = {
            s: "".join("0" if seq[i] == baseline[i] else "1" for i in range(len(seq)))
            for s, seq in sample_allele_strings.items()
        }

        # Final matrix
        variant_ids = [f"{c}:{p}:{r}:{a}" for (c, p), r, a in zip(ordered_keys, ref_bases, alt_bases)]
        data_bin = [[int(ch) for ch in sample_binary_strings[s]] for s in samples_sorted]
        df = pd.DataFrame(data_bin, index=samples_sorted, columns=variant_ids, dtype=int)
        df.index.name = "Sample"

        # Preprocessing (remove invariant + min minor allele filter)
        df = self._preprocess_binary_matrix(df)

        # Optional compact IDs
        lookup = None
        if self.use_integer_variant_ids:
            df, lookup = self._convert_to_integer_variant_ids(df)

        # Artifact output
        if output_dir:
            out = Path(output_dir)
            self._write_all_artifacts(
                out_root=out,
                kept_sites=kept_sites,
                ordered_keys=ordered_keys,
                positions_1based=[p for _, p in ordered_keys],
                ref_line=ref_line,
                sample_allele_strings=sample_allele_strings,
                ref_binary=ref_binary,
                sample_binary_strings=sample_binary_strings,
                ref_path=Path(ref_path) if ref_path else None,
                integer_id_lookup=lookup,
            )

            if self.config is not None:
                cfg_path = out / "dataloader_config.snapshot.json"
                payload = asdict(self.config) if hasattr(self.config, "__dataclass_fields__") else vars(self.config)
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)

        return df  
    
    def _preprocess_binary_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply lightweight, non-statistical preprocessing to the binary matrix.
        
        Removes invariant columns and (optionally) filters low minor-allele-count variants.
        This step is NOT a substitute for proper statistical filtering (χ²/Fisher + FDR).
        """
        if df.empty:
            return df

        # Remove invariant features (all 0 or all 1)
        if self.remove_invariant:
            nunique = df.nunique(axis=0, dropna=False)
            df = df.loc[:, nunique > 1]
            if df.empty:
                raise ValueError(
                    "All polymorphic sites were removed during invariant filtering. "
                    "Check input data or relax remove_invariant setting."
                )

        # Optional: enforce minimum minor allele count per site
        if self.min_minor_count > 0:
            keep_mask = []
            for col in df.columns:
                vc = df[col].value_counts(dropna=False)
                count_0 = vc.get(0, 0)
                count_1 = vc.get(1, 0)
                keep_mask.append(min(count_0, count_1) >= self.min_minor_count)

            df = df.loc[:, keep_mask]
            if df.empty:
                raise ValueError(
                    "All sites removed by minor allele count filter. "
                    f"Try lowering min_minor_count (current: {self.min_minor_count}) "
                    "or verify binary encoding."
                )

        return df
    
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
    # Artifact generation (all 3 scripts’ outputs)
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
    ) -> None:
        """Write vcf_counts/*, fasta/*, and matrices/* outputs."""
        out_root.mkdir(parents=True, exist_ok=True)

        vcf_counts_dir = out_root / "vcf_counts"
        fasta_dir = out_root / "fasta"
        matrices_dir = out_root / "matrices"

        vcf_counts_dir.mkdir(parents=True, exist_ok=True)
        fasta_dir.mkdir(parents=True, exist_ok=True)
        matrices_dir.mkdir(parents=True, exist_ok=True)

        # 1) vcf_counts/all_snp.txt
        all_snp_path = vcf_counts_dir / "all_snp.txt"
        all_snp_rows, all_snp_header = self._write_all_snp_table(
            path=all_snp_path,
            kept_sites=kept_sites,
            ref_path=ref_path,
        )

        # 2) fasta/<generic>_{alleles,binary}.fasta
        alleles_fa = fasta_dir / f"{self.generic_name}_alleles.fasta"
        binary_fa = fasta_dir / f"{self.generic_name}_binary.fasta"
        write_fasta_matrix(alleles_fa, ref_line, sample_allele_strings, ref_name="REF")
        write_fasta_matrix(binary_fa, ref_binary, sample_binary_strings, ref_name="REF")

        # 3) fasta/<generic>_filtered.tsv (filtered copy of all_snp.txt; optional Context_±40)
        filtered_tsv = fasta_dir / f"{self.generic_name}_filtered.tsv"
        self._write_filtered_copy_with_context(
            input_rows=all_snp_rows,
            input_header=all_snp_header,
            output_path=filtered_tsv,
            kept_positions=set(positions_1based),
            ref_path=ref_path,
        )

        # 4) matrices/* outputs produced by filtering (minor-count + type + redundancy + fix)
        self._write_matrices_outputs(
            fasta_alleles=alleles_fa,
            fasta_binary=binary_fa,
            annotation_tsv=filtered_tsv,
            out_dir=matrices_dir,
        )

        # 5) optional lookup used by returned df
        if integer_id_lookup is not None:
            with open(out_root / "variant_id_lookup.json", "w", encoding="utf-8") as f:
                json.dump(integer_id_lookup, f, indent=2)

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
        for (chrom, pos) in sorted(kept_sites.keys(), key=lambda x: (x[0], x[1])):
            count, _, _ = kept_sites[(chrom, pos)]
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
    ) -> None:
        """Generate matrices outputs: TSV matrices + filtered FASTA + filtered annotation TSV."""
        out_dir.mkdir(parents=True, exist_ok=True)

        alleles = read_fasta_matrix(fasta_alleles)
        binary = read_fasta_matrix(fasta_binary)
        annot_rows, annot_header = read_annotation_tsv(annotation_tsv)

        if list(alleles.keys()) != list(binary.keys()):
            raise ValueError("Genome order differs between alleles and binary FASTA files.")

        genomes = list(alleles.keys())
        S = len(next(iter(alleles.values())))
        if len(next(iter(binary.values()))) != S:
            raise ValueError("Alleles and binary FASTA have different number of columns (markers).")

        # If annotation rows length doesn’t match, we can’t apply type-filter reliably.
        # We still proceed with minor-count/redundancy only by synthesizing positions.
        annotation_matches = len(annot_rows) == S

        # Prepare column views
        genomes_order, allele_cols = transpose_rows_to_columns(alleles)
        _, binary_cols = transpose_rows_to_columns(binary)

        # 1) minor-count filter
        mask_minor = minor_count_filter(binary_cols, self.matrices_min_count)

        # 2) type filter (only if annotation matches and has required fields)
        if annotation_matches and annot_header:
            mask_type = type_filter(annot_rows, self.matrices_type)
        else:
            mask_type = [True] * S

        mask_12 = combine_masks(mask_minor, mask_type)

        # Apply mask_12 for grouping stage
        idx12 = [i for i, k in enumerate(mask_12) if k]
        allele_cols_12 = [c for c, k in zip(allele_cols, mask_12) if k]
        binary_cols_12 = [c for c, k in zip(binary_cols, mask_12) if k]
        annot_rows_12 = [r for r, k in zip(annot_rows, mask_12) if k] if annotation_matches else []

        # 3) redundancy filter by identical pattern
        if binary_cols_12:
            if annotation_matches:
                mask_group = group_and_reduce_by_pattern(binary_cols_12, annot_rows_12, self.matrices_repeat_number)
            else:
                # If no annotation, group without using positions (stable order)
                dummy_rows = [{"Position": str(i + 1)} for i in range(len(binary_cols_12))]
                mask_group = group_and_reduce_by_pattern(binary_cols_12, dummy_rows, self.matrices_repeat_number)
        else:
            mask_group = []

        # Final mask relative to original columns
        mask_final = [False] * S
        for kept_flag, original_idx in zip(mask_group, idx12):
            if kept_flag:
                mask_final[original_idx] = True

        # 4) force-keep positions
        fix_idx0, _warnings = parse_fix_positions(self.matrices_fix, S)
        for idx in fix_idx0:
            mask_final[idx] = True

        # Apply final mask
        alleles_filt = apply_mask_to_char_rows(alleles, mask_final)
        binary_filt = apply_mask_to_char_rows(binary, mask_final)

        # Filter annotation rows if possible; otherwise write the original filtered.tsv unchanged
        if annotation_matches:
            annot_filt = [r for r, k in zip(annot_rows, mask_final) if k]
        else:
            annot_filt = annot_rows

        # Prepare columns + positions for TSV
        _, allele_cols_f = transpose_rows_to_columns(alleles_filt)
        _, binary_cols_f = transpose_rows_to_columns(binary_filt)

        if annotation_matches:
            positions_f = [(r.get("Position", "") or "").strip() for r in annot_filt]
        else:
            positions_f = [str(i + 1) for i in range(len(allele_cols_f))]

        base = self.generic_name
        write_matrix_tsv(out_dir / f"{base}_alleles.tsv", genomes_order, positions_f, allele_cols_f)
        write_matrix_tsv(out_dir / f"{base}_binary.tsv", genomes_order, positions_f, binary_cols_f)
        write_fasta_matrix_wrapped(out_dir / f"{base}_alleles.fasta", alleles_filt)
        write_fasta_matrix_wrapped(out_dir / f"{base}_binary.fasta", binary_filt)

        if annotation_matches and annot_header:
            write_annotation_tsv(out_dir / f"{base}_filtered.tsv", annot_filt, annot_header)
        else:
            # Fallback: copy the TSV as-is
            out_path = out_dir / f"{base}_filtered.tsv"
            out_path.write_text(annotation_tsv.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")

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
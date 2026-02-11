# data_loader.py
"""
NetworkParser — DataLoader

This module is responsible for one job only:
    Input genomic data  →  sample × variant matrix (0/1)

It does NOT do:
    - statistical filtering (χ² / Fisher + FDR happens later)
    - decision tree building (happens later)
    - bootstrapping / confidence scoring (post-tree)

Why this separation matters:
    You want a clean, auditable conversion from genomic evidence → ML-ready matrix,
    before any inference steps.
"""

import logging
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

# Optional: sparse matrices are a major win when the matrix is huge + sparse
# (typical for variant presence/absence across many samples).
try:
    import numpy as np
    from scipy import sparse
except Exception:  # pragma: no cover
    np = None
    sparse = None

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Produces a sample × variant matrix from:
        - a folder of single-sample VCFs  (.vcf.gz)
        - a single VCF                  (.vcf / .vcf.gz)
        - a prebuilt matrix             (.csv / .tsv)

Core outputs:
    - genomic_matrix.csv  (rows=samples, columns=variants, values ∈ {0,1} or sparse)

Folder-of-VCFs behavior is controlled by NetworkParserConfig:
    - force_union_matrix (bool): always use union-of-sites mode for directories
    - union_matrix_threshold (int): automatically use union-of-sites if >= this many VCFs
    - union_dp_min (int): conservative FORMAT/DP gating in union mode
    - use_integer_variant_ids (bool): use integer column IDs + save lookup table
    """

    def __init__(
        self,
        use_bcftools: bool = True,
        temp_dir: Optional[str] = None,
        n_jobs: Optional[int] = None,
        config=None,
    ):
        # Whether we try to use bcftools/tabix workflow
        self.use_bcftools = use_bcftools

        # Temp workspace (used for intermediate outputs / folder workflows)
        self.temp_dir = Path(temp_dir or tempfile.mkdtemp(prefix="networkparser_"))
        self.temp_dir.mkdir(exist_ok=True)

        # Parallelism for I/O-heavy subprocess steps (tabix indexing mainly)
        cpu = os.cpu_count() or 1
        self.n_jobs = int(n_jobs) if (n_jobs is not None and int(n_jobs) > 0) else max(1, cpu - 1)

        # Folder union-matrix mode settings (wired via NetworkParserConfig)
        self.force_union_matrix = getattr(config, "force_union_matrix", False) if config else False
        self.union_matrix_threshold = int(getattr(config, "union_matrix_threshold", 200)) if config else 200
        self.union_dp_min = int(getattr(config, "union_dp_min", 10)) if config else 10

        # Optional: avoid huge string column headers by using integer IDs + saving a lookup table
        self.use_integer_variant_ids = getattr(config, "use_integer_variant_ids", False) if config else False

        # Optional: if config provides genotype/variant QC thresholds for single-VCF pipeline
        self.qual_threshold = float(getattr(config, "qual_threshold", 30.0)) if config else 30.0
        self.min_dp_per_sample = int(getattr(config, "min_dp_per_sample", 10)) if config else 10
        self.min_gq_per_sample = int(getattr(config, "min_gq_per_sample", 20)) if config else 20
        self.min_spacing_bp = int(getattr(config, "min_spacing_bp", 10)) if config else 10

        self._check_bcftools()

    # -------------------------------------------------------------------------
    # 1) External tool availability checks
    # -------------------------------------------------------------------------
    def _check_bcftools(self) -> None:
        """
        Purpose:
            Confirm bcftools/tabix are available.
        Why:
            - Folder-of-VCFs workflows depend on bcftools query + tabix indexing.
            - Single-VCF pipeline uses bcftools for filtering steps.
        """
        try:
            subprocess.run(["bcftools", "--version"], check=True, capture_output=True)
            subprocess.run(["tabix", "--version"], check=True, capture_output=True)
            logger.info("✅ bcftools + tabix detected - VCF workflows enabled")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("⚠️ bcftools/tabix not found - VCF parsing will be limited")
            self.use_bcftools = False

    # -------------------------------------------------------------------------
    # 2) Public entrypoint: load genomic matrix
    # -------------------------------------------------------------------------
    def load_genomic_matrix(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        regions: Optional[str] = None,
        ref_fasta: Optional[str] = None,
        label_column: Optional[str] = None,  # kept for compatibility; not used here
    ) -> pd.DataFrame:
        """
        Purpose:
            Route input to the correct loader based on path type / suffix.

        Inputs:
            - Directory: treated as folder-of-VCFs
            - .vcf / .vcf.gz: single VCF workflow
            - .csv / .tsv: already-built matrix
            - .fa/.fasta: (not implemented in this version)

        Output:
            - DataFrame indexed by sample ID
            - Columns are variant IDs (string CHROM:POS:REF:ALT OR integer IDs)
        """
        path = Path(file_path)
        logger.info(f"Loading genomic data from: {path}")

        if path.is_dir():
            return self._load_vcf_folder(path, output_dir=output_dir, ref_fasta=ref_fasta)

        suffix = "".join(path.suffixes).lower()
        if suffix.endswith(".vcf") or suffix.endswith(".vcf.gz"):
            if not self.use_bcftools:
                raise RuntimeError("bcftools/tabix required for VCF input in this build.")
            return self._vcf_bcftools_pipeline(
                vcf_path=path,
                output_dir=output_dir or str(self.temp_dir),
                ref_fasta=ref_fasta,
            )

        if suffix.endswith(".csv") or suffix.endswith(".tsv"):
            return self._load_csv_matrix(path)

        if suffix.endswith(".fa") or suffix.endswith(".fasta"):
            return self._fasta_to_matrix(path)

        raise ValueError(f"Unsupported genomic input: {path}")
     # -------------------------------------------------------------------------
    # 2b) Public helpers: metadata + known markers
    # -------------------------------------------------------------------------
    def load_metadata(self, meta_path: str, output_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Load sample metadata table.

        Expected:
            - CSV/TSV with at least one column that identifies samples.
            - If a column named 'Sample' exists, we use it as the index.
            - Otherwise we assume the FIRST column is the sample identifier.

        Output:
            - DataFrame indexed by sample ID (string)
        """
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

        # Save a normalized copy for reproducibility/debugging
        if output_dir:
            outdir = Path(output_dir)
            outdir.mkdir(parents=True, exist_ok=True)
            out_path = outdir / "metadata.normalized.csv"
            df.to_csv(out_path)
            logger.info(f"Saved normalized metadata copy: {out_path}")

        return df

    def load_known_markers(self, known_markers_path: str, output_dir: Optional[str] = None) -> List[str]:
        """
        Load a list of known markers/features to prioritize or annotate.

        Supported inputs:
            - .txt : one marker per line (comments allowed with '#')
            - .csv/.tsv : expects a column named 'marker' or uses the first column

        Returns:
            - list of unique marker strings (order preserved)
        """
        path = Path(known_markers_path)
        if not path.exists():
            raise FileNotFoundError(f"Known markers file not found: {path}")

        suffix = "".join(path.suffixes).lower()

        markers: List[str] = []
        if suffix.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    m = line.strip()
                    if m and not m.startswith("#"):
                        markers.append(m)

        elif suffix.endswith(".csv") or suffix.endswith(".tsv") or suffix.endswith(".tab"):
            sep = "\t" if (suffix.endswith(".tsv") or suffix.endswith(".tab")) else ","
            df = pd.read_csv(path, sep=sep)
            if not df.empty:
                col = "marker" if "marker" in df.columns else df.columns[0]
                markers = [str(x).strip() for x in df[col].tolist() if str(x).strip()]

        else:
            raise ValueError(f"Unsupported known markers format: {path}")

        # de-duplicate while preserving order
        seen: Set[str] = set()
        uniq: List[str] = []
        for m in markers:
            if m not in seen:
                seen.add(m)
                uniq.append(m)

        if output_dir:
            outdir = Path(output_dir)
            outdir.mkdir(parents=True, exist_ok=True)
            out_path = outdir / "known_markers.normalized.txt"
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(uniq) + ("\n" if uniq else ""))
            logger.info(f"Saved normalized known markers copy: {out_path}")

        return uniq

    # -------------------------------------------------------------------------
    # 3) CSV/TSV matrix load
    # -------------------------------------------------------------------------
    def _load_csv_matrix(self, path: Path) -> pd.DataFrame:
        """
        Purpose:
            Load a pre-built sample × variant matrix.

        Assumptions:
            - First column is sample ID index
            - Remaining columns are variant features (already encoded)
        """
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep, index_col=0)
        df.index = df.index.astype(str)
        return df

    def _fasta_to_matrix(self, path: Path) -> pd.DataFrame:
        """
        Placeholder.
        FASTA→matrix is intentionally not provided in this stripped build because it can
        encourage mixing alignment/variant-calling with ML encoding in one step.
        """
        raise NotImplementedError("FASTA → matrix not implemented in this version.")

    # -------------------------------------------------------------------------
    # 4) Tabix indexing helpers (parallel-safe)
    # -------------------------------------------------------------------------
    @staticmethod
    def _tbi_path(vcf_gz: Path) -> Path:
        """
        Purpose:
            Standard tabix index name for bgzip VCF.
        """
        return vcf_gz.with_name(vcf_gz.name + ".tbi")

    @staticmethod
    def _needs_reindex(vcf_gz: Path) -> bool:
        """
        Purpose:
            Only index if missing or outdated (VCF newer than .tbi).
        """
        tbi = DataLoader._tbi_path(vcf_gz)
        if not tbi.exists():
            return True
        return tbi.stat().st_mtime < vcf_gz.stat().st_mtime

    @staticmethod
    def _run_tabix_index(vcf_gz: Path) -> None:
        """
        Purpose:
            Create/overwrite the tabix index.
        Why force (-f):
            Deterministic end state even if partial index exists.
        """
        subprocess.run(["tabix", "-f", "-p", "vcf", str(vcf_gz)], check=True)

    def _index_vcfs_parallel(self, vcf_paths: List[Path]) -> None:
        """
        Purpose:
            Index many VCF.gz files efficiently.

        Safety:
            - Uses threads because the heavy work is external (tabix subprocess),
              and thread scheduling won’t change output (index depends only on file).
        """
        to_index = [p for p in vcf_paths if self._needs_reindex(p)]
        if not to_index:
            return

        logger.info(f"Indexing {len(to_index)} VCFs with tabix (n_jobs={self.n_jobs}) ...")
        max_workers = min(self.n_jobs, len(to_index))

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self._run_tabix_index, p): p for p in to_index}
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    raise RuntimeError(f"Failed to tabix-index: {p}") from e

    # -------------------------------------------------------------------------
    # 5) Folder-of-VCFs handling
    # -------------------------------------------------------------------------
    def _load_vcf_folder(
        self,
        folder_path: Path,
        output_dir: Optional[str] = None,
        ref_fasta: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Purpose:
            Turn a folder of single-sample VCFs into one cohort matrix.

        Two modes:
        (A) UNION-OF-SITES MATRIX  (recommended for large collections, ALT-only VCF lists)
            - Build the global feature space as the union of all biallelic SNP sites across samples.
            - Encode per sample:
                  present in sample VCF => 1
                  absent from sample VCF => 0 (implicit reference)
            - Avoids bcftools merge (which becomes extremely heavy at large N).

        (B) MERGE-THEN-PARSE  (reasonable for small collections / true multi-sample consolidation)
            - bcftools merge → multi-sample VCF
            - then run the standard single-VCF pipeline on the merged file

        """
        if not self.use_bcftools:
            raise RuntimeError("bcftools/tabix required for folder-of-VCFs input.")

        output_dir_path = Path(output_dir) if output_dir else self.temp_dir
        output_dir_path.mkdir(parents=True, exist_ok=True)

        vcfs = sorted(folder_path.glob("*.vcf.gz"))
        if not vcfs:
            raise FileNotFoundError(f"No .vcf.gz files found in: {folder_path}")

        # Ensure tabix indexes exist before any bcftools query calls
        self._index_vcfs_parallel(vcfs)

        # Decide union vs merge using config-driven logic
        use_union = self.force_union_matrix or (len(vcfs) >= self.union_matrix_threshold)
        if use_union:
            logger.info(
                f"Folder mode: UNION-OF-SITES (n_files={len(vcfs)}, "
                f"threshold={self.union_matrix_threshold}, force={self.force_union_matrix})"
            )
            return self._vcf_folder_union_matrix(vcfs, str(output_dir_path))

        logger.info(f"Folder mode: MERGE-THEN-PARSE (n_files={len(vcfs)})")

        # Merge path (small collections only)
        merged_vcf = output_dir_path / "merged.vcf.gz"

        merge_cmd = ["bcftools", "merge", "-Oz", "-o", str(merged_vcf)]
        # A sane thread clamp (bcftools supports --threads for some commands; merge can benefit too)
        merge_cmd += ["--threads", str(max(1, min(self.n_jobs, 32)))]
        merge_cmd += [str(v) for v in vcfs]

        try:
            subprocess.run(merge_cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "bcftools merge failed. This usually indicates inconsistent contigs/reference "
                "or incompatible VCF headers across files."
            ) from e

        # Index merged output for downstream steps
        subprocess.run(["tabix", "-f", "-p", "vcf", str(merged_vcf)], check=True)

        # Now treat the merged multi-sample VCF like any single VCF input
        return self._vcf_bcftools_pipeline(merged_vcf, str(output_dir_path), ref_fasta)

    # -------------------------------------------------------------------------
    # 6) Union-of-sites matrix (optimized for many single-sample ALT-only VCFs)
    # -------------------------------------------------------------------------
    def _vcf_folder_union_matrix(self, vcf_files: List[Path], output_dir: str) -> pd.DataFrame:
        """
        Purpose:
            Build a cohort matrix directly from many single-sample VCFs, without merging.

        What each step does:
            1) Read sample IDs from VCF headers (bcftools query -l)
            2) PASS 1: Build the global feature universe (union of biallelic SNP sites)
               - Each feature is CHROM:POS:REF:ALT
               - Assign a stable integer column index as features are discovered
               - Record, per sample, which column indices are present
            3) PASS 2: Build a sparse matrix from the cached per-sample presence sets
               - No need to re-read VCFs a second time
            4) Decide column naming:
               - if use_integer_variant_ids: columns are 0..N-1 and a lookup CSV is saved
               - else: columns are the CHROM:POS:REF:ALT strings

        Why this is the correct representation for ALT-only variant-list VCFs:
            - These VCFs list only ALT-present sites for each sample
            - Absence of a record implies reference (0), not missing
            - This produces a statistically defensible, interpretable binary matrix
        """
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        vcf_files = sorted([Path(v) for v in vcf_files])

        # 1) Extract single-sample IDs (from headers, not filenames)
        sample_ids: List[str] = []
        for v in vcf_files:
            sample_ids.append(self._extract_single_sample_id(v))

        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("Duplicate sample IDs detected in VCF headers (must be unique).")

        logger.info(f"Union-matrix mode: {len(sample_ids)} samples detected")

        # 2) PASS 1: Build universe + per-sample presence cache
        # universe_map maps variant_string -> column_index
        universe_map: Dict[str, int] = {}
        # variant_by_index preserves the exact discovery order of features so indices stay consistent
        variant_by_index: List[str] = []
        # per-sample cache: set of integer column indices present in each sample
        per_sample_present: List[Set[int]] = [set() for _ in sample_ids]

        logger.info("Union-matrix mode: PASS 1/2 (build universe + cache presence)...")
        for i, vcf_path in enumerate(vcf_files):
            for vid in self._iter_biallelic_snp_variant_ids(vcf_path):
                if vid not in universe_map:
                    universe_map[vid] = len(variant_by_index)
                    variant_by_index.append(vid)
                per_sample_present[i].add(universe_map[vid])

            if (i + 1) % 250 == 0:
                logger.info(
                    f"Scanned {i+1}/{len(vcf_files)} VCFs | feature universe size: {len(variant_by_index)}"
                )

        n_variants = len(variant_by_index)
        if n_variants == 0:
            logger.warning("Union-matrix mode: no variants found after filters — returning empty matrix.")
            df_empty = pd.DataFrame(index=sample_ids)
            df_empty.index.name = "Sample"
            df_empty.to_csv(outdir / "genomic_matrix.csv")
            return df_empty

        # Optional warnings on extremely large universes (helps users anticipate memory use)
        if n_variants > 2_000_000:
            logger.warning(f"VERY LARGE feature universe: {n_variants:,} variants — expect high memory usage.")
        elif n_variants > 1_000_000:
            logger.warning(f"Large feature universe: {n_variants:,} variants.")

        # 3) If integer IDs requested, write lookup mapping
        # (keeps the matrix columns small and consistent, and saves interpretability separately)
        if self.use_integer_variant_ids:
            lookup_path = outdir / "variant_lookup.csv"
            # Split “CHROM:POS:REF:ALT” into columns. POS kept as string to avoid accidental coercion.
            lookup_rows = [v.split(":", 3) for v in variant_by_index]
            lookup_df = pd.DataFrame(lookup_rows, columns=["CHROM", "POS", "REF", "ALT"])
            lookup_df.index.name = "variant_id"  # integer row index corresponds to column ID
            lookup_df.to_csv(lookup_path)
            logger.info(f"Saved variant lookup table: {lookup_path} ({n_variants:,} variants)")

        # 4) PASS 2: Build sparse coordinate lists from cache (no VCF re-reading)
        logger.info("Union-matrix mode: PASS 2/2 (build matrix from cached presence)...")
        rows: List[int] = []
        cols: List[int] = []
        vals: List[int] = []

        for r, present_cols in enumerate(per_sample_present):
            for c in present_cols:
                rows.append(r)
                cols.append(c)
                vals.append(1)

            if (r + 1) % 250 == 0:
                logger.info(f"Populated {r+1}/{len(sample_ids)} samples into matrix")

        # 5) Construct matrix
        if np is not None and sparse is not None:
            mat = sparse.csr_matrix(
                (vals, (rows, cols)),
                shape=(len(sample_ids), n_variants),
                dtype=np.int8,
            )

            if self.use_integer_variant_ids:
                col_names = [str(i) for i in range(n_variants)]
            else:
                col_names = variant_by_index

            df = pd.DataFrame.sparse.from_spmatrix(mat, index=sample_ids, columns=col_names)
        else:
            # Dense fallback is only safe for small feature spaces.
            logger.warning("scipy.sparse unavailable → building dense matrix (can be memory-intensive).")

            if self.use_integer_variant_ids:
                col_names = [str(i) for i in range(n_variants)]
            else:
                col_names = variant_by_index

            df = pd.DataFrame(0, index=sample_ids, columns=col_names, dtype="int8")
            for rr, cc in zip(rows, cols):
                df.iat[rr, cc] = 1

        df.index.name = "Sample"
        df.columns.name = "Variant"

        matrix_csv = outdir / "genomic_matrix.csv"
        df.to_csv(matrix_csv)
        logger.info(f"Union-of-sites matrix saved: {matrix_csv} (shape {df.shape})")

        # Explicit cleanup (helps large runs)
        del rows, cols, vals, per_sample_present, universe_map

        return df

    # -------------------------------------------------------------------------
    # 7) Header helpers + per-record iterators
    # -------------------------------------------------------------------------
    def _extract_single_sample_id(self, vcf_path: Path) -> str:
        """
        Purpose:
            Determine the sample ID embedded in a single-sample VCF.

        Why:
            - Avoid using filenames as sample IDs (they can be inconsistent)
            - Use the authoritative header identity for alignment with metadata
        """
        cmd = ["bcftools", "query", "-l", str(vcf_path)]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.splitlines()
        out = [x.strip() for x in out if x.strip()]
        if len(out) != 1:
            raise ValueError(
                f"Expected exactly 1 sample in {vcf_path.name}, found {len(out)}. "
                "Union mode requires single-sample VCFs."
            )
        return out[0]

    def _iter_biallelic_snp_variant_ids(self, vcf_path: Path):
        """
        Purpose:
            Stream biallelic SNP feature IDs from a VCF as "CHROM:POS:REF:ALT".

        What filters are applied (and why):
            - Skip missing GT (explicit ./.)              → avoid uncertain calls
            - Skip multiallelic ALT (contains ',')        → stay in biallelic SNP space (current scope)
            - Skip non-SNPs (len(REF)!=1 or len(ALT)!=1)  → keep encoding simple and interpretable
            - Skip non-ACGT bases                         → defensive against ambiguous symbols
            - Optional DP gate (FORMAT/DP)                → drop low-support calls conservatively

        Note:
            For ALT-only variant-list VCFs, this iterator effectively enumerates
            the ALT-present sites for that sample.
        """
        cmd = [
            "bcftools",
            "query",
            "-f",
            "%CHROM\t%POS\t%REF\t%ALT\t[%GT]\t[%DP]\n",
            str(vcf_path),
        ]

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )
        assert proc.stdout is not None

        for line in proc.stdout:
            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 4:
                continue

            chrom, pos, ref, alt = parts[0], parts[1], parts[2], parts[3]
            gt = parts[4].strip() if len(parts) > 4 else ""
            dp = parts[5].strip() if len(parts) > 5 else ""

            # 1) Must have a genotype call (skip explicit missing)
            if gt in {"", ".", "./.", ".|."}:
                continue

            # 2) Enforce biallelic ALT only
            if "," in alt:
                continue

            # 3) SNP-only
            if len(ref) != 1 or len(alt) != 1:
                continue

            # 4) Canonical bases only
            if ref not in {"A", "C", "G", "T"} or alt not in {"A", "C", "G", "T"}:
                continue

            # 5) Conservative DP gating if DP is present and parseable
            if dp and dp != ".":
                try:
                    if int(float(dp)) < self.union_dp_min:
                        continue
                except Exception:
                    # If DP is malformed, we do not over-filter by default.
                    pass

            yield f"{chrom}:{pos}:{ref}:{alt}"

        _, stderr = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"bcftools query failed for {vcf_path.name}: {stderr.strip()[:500]}"
            )

    # -------------------------------------------------------------------------
    # 8) Single VCF → preprocessing → matrix
    # -------------------------------------------------------------------------
    def _vcf_bcftools_pipeline(
        self,
        vcf_path: Path,
        output_dir: str,
        ref_fasta: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Purpose:
            Standard single-VCF pipeline:
                VCF → (optional normalize) → biallelic SNPs → QC → matrix extraction

        Important:
            - This pipeline is best when you already have a multi-sample VCF
              or a small merged VCF.
            - For huge folders of ALT-only single-sample VCFs, prefer union mode.

        Steps (high-level):
            1) (Optional) Normalize against reference FASTA (if provided)
            2) Filter biallelic SNPs
            3) Variant-level QC (QUAL and/or INFO/DP if present)
            4) Genotype-level QC (mask low-quality genotypes to missing if GQ/DP tags exist)
            5) Remove invariant sites (multi-sample contexts)
            6) Remove SNPs near indels/gaps (SnpGap)
            7) Extract GT into sample × variant 0/1 matrix
        """
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        prefix = outdir / vcf_path.stem

        # Intermediate filenames are numbered for readability and reproducibility
        files = {
            "norm": prefix.with_name(prefix.name + ".01.norm.vcf.gz"),
            "snps": prefix.with_name(prefix.name + ".02.biallelic_snps.vcf.gz"),
            "var_qc": prefix.with_name(prefix.name + ".03.variant_qc.vcf.gz"),
            "gt_qc": prefix.with_name(prefix.name + ".04.genotype_qc.vcf.gz"),
            "variable": prefix.with_name(prefix.name + ".05.variable.vcf.gz"),
            "nogap": prefix.with_name(prefix.name + ".06.no_indel_gap.vcf.gz"),
            "final": prefix.with_name(prefix.name + ".07.final_cleaned.vcf.gz"),
        }
        matrix_csv = prefix.with_name(prefix.name + ".genomic_matrix.csv")

        def run_step(cmd: List[str], output_file: Path, desc: str) -> None:
            """
            Helper:
                - Runs bcftools command producing output_file
                - Tabix-indexes the output
                - Skips if output already exists and looks valid
            """
            if output_file.exists() and output_file.stat().st_size > 100:
                logger.info(f"[SKIP] {desc} (exists): {output_file.name}")
                return

            logger.info(f"[RUN] {desc}")
            subprocess.run(cmd, check=True)

            # Index step outputs (VCF.gz)
            subprocess.run(["tabix", "-f", "-p", "vcf", str(output_file)], check=True)

            if (not output_file.exists()) or output_file.stat().st_size < 100:
                raise RuntimeError(f"{desc} failed: output missing or empty: {output_file}")

        # 1) Normalize if reference FASTA exists
        if ref_fasta and Path(ref_fasta).exists():
            run_step(
                [
                    "bcftools",
                    "norm",
                    "-f",
                    ref_fasta,
                    "-m-",
                    "-any",
                    "-Oz",
                    "-o",
                    str(files["norm"]),
                    str(vcf_path),
                ],
                files["norm"],
                "Normalize (left-align + split multiallelic if present)",
            )
            source_vcf = files["norm"]
        else:
            # If no FASTA, keep source as-is (still valid, just less normalized)
            logger.info("[SKIP] Normalization (no reference FASTA provided)")
            source_vcf = vcf_path

        # 2) Filter to biallelic SNPs
        run_step(
            [
                "bcftools",
                "view",
                "-m2",
                "-M2",
                "-v",
                "snps",
                "-Oz",
                "-o",
                str(files["snps"]),
                str(source_vcf),
            ],
            files["snps"],
            "Filter to biallelic SNPs",
        )

        # 3) Variant-level QC (QUAL + INFO/DP if present)
        header = subprocess.run(
            ["bcftools", "view", "-h", str(files["snps"])],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        has_dp_info = "INFO=<ID=DP" in header

        expr = f"QUAL < {self.qual_threshold}"
        if has_dp_info:
            expr += f" || INFO/DP < {self.min_dp_per_sample}"

        run_step(
            [
                "bcftools",
                "filter",
                "-e",
                expr,
                "-Oz",
                "-o",
                str(files["var_qc"]),
                str(files["snps"]),
            ],
            files["var_qc"],
            "Variant-level QC (QUAL/DP)",
        )

        # 4) Genotype-level QC: mask low-quality genotypes to missing (.)
        header = subprocess.run(
            ["bcftools", "view", "-h", str(files["var_qc"])],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        has_gq = "FORMAT=<ID=GQ" in header
        has_dp_fmt = "FORMAT=<ID=DP" in header

        gt_filter = ""
        if has_gq and has_dp_fmt:
            gt_filter = f"FMT/GQ < {self.min_gq_per_sample} || FMT/DP < {self.min_dp_per_sample}"
        elif has_gq:
            gt_filter = f"FMT/GQ < {self.min_gq_per_sample}"
        elif has_dp_fmt:
            gt_filter = f"FMT/DP < {self.min_dp_per_sample}"

        if gt_filter:
            run_step(
                [
                    "bcftools",
                    "filter",
                    "-S",
                    ".",
                    "-e",
                    gt_filter,
                    "-Oz",
                    "-o",
                    str(files["gt_qc"]),
                    str(files["var_qc"]),
                ],
                files["gt_qc"],
                "Genotype-level QC (mask low-quality GT to missing)",
            )
            qc_vcf = files["gt_qc"]
        else:
            logger.info("[SKIP] Genotype QC (no GQ/DP FORMAT tags found)")
            qc_vcf = files["var_qc"]

        # 5) Remove invariant sites (only meaningful in multi-sample VCFs)
        # MAX(AC)>0 keeps sites where an ALT allele appears in at least one sample.
        run_step(
            [
                "bcftools",
                "view",
                "-i",
                "MAX(AC)>0",
                "-Oz",
                "-o",
                str(files["variable"]),
                str(qc_vcf),
            ],
            files["variable"],
            "Remove invariant sites (MAX(AC)>0)",
        )

        # 6) Remove SNPs near indels/gaps (reduces alignment/context artefacts)
        run_step(
            [
                "bcftools",
                "filter",
                "--SnpGap",
                str(self.min_spacing_bp),
                "-Oz",
                "-o",
                str(files["nogap"]),
                str(files["variable"]),
            ],
            files["nogap"],
            f"Remove SNPs within {self.min_spacing_bp}bp of indels/gaps",
        )

        # Final file
        files["final"] = files["nogap"]

        # 7) Extract genotypes into a 0/1 matrix
        logger.info("Extracting GT → 0/1 matrix (streaming bcftools query)")

        # a) sample IDs
        samples = subprocess.run(
            ["bcftools", "query", "-l", str(files["final"])],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        samples = [s.strip() for s in samples if s.strip()]
        if len(set(samples)) != len(samples):
            raise ValueError("Duplicate sample IDs in VCF header — sample IDs must be unique.")

        # b) genotype encoding helper
        def gt_to_binary(gt: str):
            """
            Convert GT to binary ALT presence.

            - Haploid calls: "0" or "1"
            - Diploid calls: "0/0", "0/1", "1/1" (or phased with '|')
            - Missing: "./." or "."
            """
            gt = (gt or "").strip()
            if not gt or gt in {".", "./.", ".|."}:
                return float("nan")

            # Haploid
            if gt == "0":
                return 0
            if gt == "1":
                return 1

            # Diploid
            sep = "/" if "/" in gt else ("|" if "|" in gt else None)
            if sep is None:
                return float("nan")
            alleles = gt.split(sep)
            if any(a == "." for a in alleles):
                return float("nan")
            return 1 if any(a == "1" for a in alleles) else 0

        # c) streaming query: variant_id + per-sample GT list
        # Use CHROM:POS:REF:ALT to keep IDs collision-resistant.
        cmd = ["bcftools", "query", "-f", "%CHROM:%POS:%REF:%ALT\t[%GT\t]\n", str(files["final"])]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        assert proc.stdout is not None

        data: Dict[str, List[float]] = {s: [] for s in samples}
        variant_ids: List[str] = []

        skipped = 0
        for line in proc.stdout:
            line = line.rstrip("\n")
            if not line:
                continue

            fields = line.split("\t")
            if len(fields) < 2:
                skipped += 1
                continue

            vid = fields[0]
            gts = fields[1:]

            # bcftools pattern "[%GT\t]" often leaves a trailing empty field; strip it
            if gts and gts[-1] == "":
                gts = gts[:-1]

            if len(gts) != len(samples):
                skipped += 1
                continue

            variant_ids.append(vid)
            for i, gt in enumerate(gts):
                data[samples[i]].append(gt_to_binary(gt))

        _, stderr = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"bcftools query failed: {stderr.strip()[:500]}")

        if not variant_ids:
            logger.warning("No variants extracted — returning empty matrix.")
            df = pd.DataFrame(index=samples)
            df.index.name = "Sample"
            df.to_csv(matrix_csv)
            return df

        df = pd.DataFrame.from_dict(data, orient="index", columns=variant_ids)
        df.index.name = "Sample"
        df.columns.name = "Variant"

        df.to_csv(matrix_csv)
        logger.info(f"Saved matrix: {matrix_csv} (shape {df.shape}, skipped_lines={skipped})")

        return df

    # -------------------------------------------------------------------------
    # 9) Python fallback placeholder
    # -------------------------------------------------------------------------
    def _vcf_python_fallback(self, path: Path, output_dir: str, ref_fasta: Optional[str] = None) -> pd.DataFrame:
        """
        Purpose:
            In restricted environments without bcftools, you could implement a pure-python VCF reader.
        Note:
            Not implemented here because it’s slower and often less robust than bcftools for large VCFs.
        """
        raise NotImplementedError("Pure Python VCF fallback not implemented.")

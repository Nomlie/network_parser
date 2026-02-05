#data_loader.py
import logging
import pandas as pd
import subprocess
import os
import tempfile
from pathlib import Path
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import VCFProcessingConfig

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


class DataLoader:
    """
    NetworkParser DataLoader

    Produces a sample × variant 0/1 matrix from VCFs (or accepts pre-built matrices).

    Key design choices (current):
    - BIALLELIC SNP-ONLY VCF path: multi-allelic sites are excluded (filtered out early).
    - Filtering is configuration-driven (VCFProcessingConfig).
    - Optional region restriction via bcftools --regions/--targets.
    - Variant annotations (ID, INFO) preserved via variant_annotations.csv.
    - Streaming parse with optional tqdm progress (large VCFs).
    """

    def __init__(
        self,
        use_bcftools: bool = True,
        temp_dir: Optional[str] = None,
        n_jobs: Optional[int] = None,
        vcf_config: Optional[VCFProcessingConfig] = None,
    ):
        self.use_bcftools = use_bcftools
        self.temp_dir = Path(temp_dir or tempfile.mkdtemp(prefix="networkparser_"))
        self.temp_dir.mkdir(exist_ok=True)

        cpu = os.cpu_count() or 1
        self.n_jobs = int(n_jobs) if (n_jobs is not None and int(n_jobs) > 0) else max(1, cpu - 1)

        self.vcf_config = vcf_config or VCFProcessingConfig()

        self._check_bcftools()

    def _check_bcftools(self):
        try:
            subprocess.run(["bcftools", "--version"], check=True, capture_output=True)
            subprocess.run(["tabix", "--version"], check=True, capture_output=True)
        except Exception as e:
            msg = (
                "bcftools/tabix not available on PATH. "
                "Install via conda (bioconda) or system package manager."
            )
            raise RuntimeError(msg) from e

    def load_genomic_matrix(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        ref_fasta: Optional[str] = None,
        label_column: Optional[str] = None,  # kept for API compatibility
        regions: Optional[str] = None,        # e.g. "chrom:start-end" or BED file
    ) -> pd.DataFrame:
        path = Path(file_path)
        logger.info(f"Loading genomic data from: {path}")

        if path.is_dir():
            return self._load_vcf_folder(path, output_dir, ref_fasta, regions=regions)

        suffix = "".join(path.suffixes).lower()
        if suffix.endswith(".vcf") or suffix.endswith(".vcf.gz"):
            if self.use_bcftools:
                return self._vcf_bcftools_pipeline(
                    path,
                    output_dir or str(self.temp_dir),
                    ref_fasta,
                    regions=regions,
                )
            return self._vcf_python_fallback(path, output_dir or str(self.temp_dir), ref_fasta, regions=regions)

        if suffix.endswith(".csv") or suffix.endswith(".tsv"):
            return self._load_csv_matrix(path)

        if suffix.endswith(".fa") or suffix.endswith(".fasta"):
            return self._fasta_to_matrix(path)

        raise ValueError(f"Unsupported genomic input: {path}")

    def _load_csv_matrix(self, path: Path) -> pd.DataFrame:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep, index_col=0)
        df.index = df.index.astype(str)
        return df

    def _fasta_to_matrix(self, path: Path) -> pd.DataFrame:
        raise NotImplementedError("FASTA → matrix not implemented in this version.")

    # -------------------------------------------------------------------------
    # Multi-VCF handling
    # -------------------------------------------------------------------------
    def _tbi_path(self, vcf_path: Path) -> Path:
        return vcf_path.with_suffix(vcf_path.suffix + ".tbi")

    def _needs_reindex(self, vcf_path: Path) -> bool:
        tbi = self._tbi_path(vcf_path)
        if not tbi.exists():
            return True
        return tbi.stat().st_mtime < vcf_path.stat().st_mtime

    def _run_tabix_index(self, vcf_path: Path) -> None:
        subprocess.run(["tabix", "-f", "-p", "vcf", str(vcf_path)], check=True)

    def _index_vcfs_parallel(self, vcf_paths: List[Path]) -> None:
        to_index = [p for p in vcf_paths if self._needs_reindex(p)]
        if not to_index:
            return

        logger.info(f"Indexing {len(to_index)} VCFs with tabix (n_jobs={self.n_jobs}) ...")
        with ThreadPoolExecutor(max_workers=self.n_jobs) as ex:
            futures = {ex.submit(self._run_tabix_index, p): p for p in to_index}
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    raise RuntimeError(f"Failed to tabix-index: {p}") from e

    def _load_vcf_folder(
        self,
        folder_path: Path,
        output_dir: Optional[str] = None,
        ref_fasta: Optional[str] = None,
        regions: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load folder of VCFs → ensure all are indexed → merge → apply preprocessing.

        Notes:
        - Only .vcf.gz inputs are supported for merging (tabix indexing required).
        - Merging is done via bcftools merge.
        """
        output_dir_path = Path(output_dir) if output_dir else self.temp_dir
        output_dir_path.mkdir(parents=True, exist_ok=True)

        vcfs = sorted(folder_path.glob("*.vcf.gz"))
        if not vcfs:
            raise FileNotFoundError(f"No .vcf.gz files found in: {folder_path}")

        self._index_vcfs_parallel(vcfs)

        merged_vcf = output_dir_path / "merged.vcf.gz"
        if not merged_vcf.exists() or merged_vcf.stat().st_size < 100:
            logger.info(f"Merging {len(vcfs)} VCFs → {merged_vcf}")
            cmd = ["bcftools", "merge", "-Oz", "-o", str(merged_vcf)] + [str(v) for v in vcfs]
            subprocess.run(cmd, check=True)
            subprocess.run(["tabix", "-f", "-p", "vcf", str(merged_vcf)], check=True)

        if self.use_bcftools:
            return self._vcf_bcftools_pipeline(merged_vcf, str(output_dir_path), ref_fasta, regions=regions)
        return self._vcf_python_fallback(merged_vcf, str(output_dir_path), ref_fasta, regions=regions)

    # -------------------------------------------------------------------------
    # BCFtools pipeline (biallelic SNPs only)
    # -------------------------------------------------------------------------
    def _vcf_bcftools_pipeline(
        self,
        vcf_path: Path,
        output_dir: str,
        ref_fasta: Optional[str] = None,
        regions: Optional[str] = None,
    ) -> pd.DataFrame:
        output_dir_path = Path(output_dir) if output_dir else self.temp_dir
        output_dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing VCF: {vcf_path}")
        prefix = output_dir_path / vcf_path.stem

        files = {
            "norm": prefix.with_name(prefix.name + ".01.norm.vcf.gz"),
            "snps": prefix.with_name(prefix.name + ".02.snps_biallelic.vcf.gz"),
            "var_qc": prefix.with_name(prefix.name + ".03.variant_qc.vcf.gz"),
            "gt_qc": prefix.with_name(prefix.name + ".04.genotype_qc.vcf.gz"),
            "variable": prefix.with_name(prefix.name + ".05.variable.vcf.gz"),
            "nogap": prefix.with_name(prefix.name + ".06.no_indel_gap.vcf.gz"),
            "final": prefix.with_name(prefix.name + ".07.final_cleaned.vcf.gz"),
        }

        matrix_csv = prefix.with_name(prefix.name + ".genomic_matrix.csv")

        def run_step(cmd, output_file: Path, desc: str):
            """
            Notes:
            - Skip if output exists and non-trivial size.
            - If output exists but tiny/invalid, re-run.
            - Always tabix-index bgzipped VCF outputs.
            """
            try:
                if output_file.exists():
                    if output_file.stat().st_size > 100:
                        logger.info(f"[SKIP] {desc} (valid file exists: {output_file})")
                        return
                    logger.warning(f"[RE-RUN] {desc} (file invalid/empty)")
                    output_file.unlink(missing_ok=True)

                logger.info(f"[RUN] {desc}")
                subprocess.run(cmd, check=True)
                subprocess.run(["tabix", "-f", "-p", "vcf", str(output_file)], check=True)
            except Exception as e:
                raise RuntimeError(f"Step failed: {desc}\nCmd: {' '.join(cmd)}") from e

        cfg = self.vcf_config

        def _append_regions(cmd: List[str]) -> List[str]:
            if regions:
                cmd.extend(["--regions", regions])
            return cmd

        # 1) Normalize VCF (reference-aware, if ref fasta provided)
        if ref_fasta and Path(ref_fasta).exists():
            if cfg.normalize:
                run_step(
                    ["bcftools", "norm", "-f", ref_fasta, "-Oz", "-o", str(files["norm"]), str(vcf_path)],
                    files["norm"],
                    "Normalize VCF (reference-aware)",
                )
            else:
                logger.info("[SKIP] Normalization (disabled in config)")
                files["norm"] = vcf_path
        else:
            logger.info("[SKIP] Normalization (no valid reference FASTA)")
            files["norm"] = vcf_path

        # 2) SNP selection + enforce biallelic representation (EARLY)
        #    Multi-allelic sites are excluded for now.
        if cfg.keep_only_snps:
            cmd_view = [
                "bcftools", "view",
                "-v", "snps",
                "-m2", "-M2",
                "-Oz",
                "-o", str(files["snps"]),
                str(files["norm"]),
            ]
            _append_regions(cmd_view)
            run_step(
                cmd_view,
                files["snps"],
                "Select biallelic SNPs only (exclude multi-allelic sites)",
            )
        else:
            # If SNP-only filtering is disabled, we still recommend biallelic SNP restriction upstream.
            # Keeping behavior unchanged: downstream steps will proceed on the normalized input.
            files["snps"] = files["norm"]

        # 3) Variant-level QC (QUAL and INFO/DP if present)
        header = subprocess.run(["bcftools", "view", "-h", str(files["snps"])], capture_output=True, text=True).stdout
        has_dp_info = "##INFO=<ID=DP" in header

        var_filter = f"QUAL < {cfg.qual_min}"
        if has_dp_info:
            var_filter += f" || INFO/DP < {cfg.dp_min_variant}"

        run_step(
            ["bcftools", "filter", "-e", var_filter, "-Oz", "-o", str(files["var_qc"]), str(files["snps"])],
            files["var_qc"],
            "Variant-level QC",
        )

        # 4) Genotype-level QC: mask low-quality genotype calls to missing (.)
        header = subprocess.run(["bcftools", "view", "-h", str(files["var_qc"])], capture_output=True, text=True).stdout
        has_gq = "FORMAT=<ID=GQ" in header
        has_dp_fmt = "FORMAT=<ID=DP" in header

        gt_filter = ""
        if has_gq and has_dp_fmt:
            gt_filter = f"FMT/GQ < {cfg.gq_min} || FMT/DP < {cfg.dp_min_genotype}"
        elif has_gq:
            gt_filter = f"FMT/GQ < {cfg.gq_min}"
        elif has_dp_fmt:
            gt_filter = f"FMT/DP < {cfg.dp_min_genotype}"
        else:
            logger.warning("No GQ or DP FORMAT tags - skipping genotype QC")
            files["gt_qc"] = files["var_qc"]

        if gt_filter:
            run_step(
                ["bcftools", "filter", "-S", ".", "-e", gt_filter, "-Oz", "-o", str(files["gt_qc"]), str(files["var_qc"])],
                files["gt_qc"],
                "Genotype-level QC",
            )

        # 5) Remove invariant sites (preliminary)
        if cfg.remove_invariants:
            run_step(
                ["bcftools", "view", "-i", "MAX(AC)>0", "-Oz", "-o", str(files["variable"]), str(files["gt_qc"])],
                files["variable"],
                "Remove invariant sites",
            )
        else:
            logger.info("[SKIP] Remove invariant sites (disabled in config)")
            files["variable"] = files["gt_qc"]

        # 6) Remove SNPs near indels/gaps
        run_step(
            ["bcftools", "filter", "--SnpGap", str(cfg.snp_gap_bp), "-Oz", "-o", str(files["nogap"]), str(files["variable"])],
            files["nogap"],
            "Remove SNPs near indels/gaps",
        )

        # 7) Final missingness/MAF filters if tags exist (may be absent depending on upstream annotations)
        header = subprocess.run(["bcftools", "view", "-h", str(files["nogap"])], capture_output=True, text=True).stdout
        has_missing = "INFO=<ID=F_MISSING" in header
        has_maf = "INFO=<ID=MAF" in header

        final_filter = ""
        if has_missing or has_maf:
            parts = []
            if has_missing:
                parts.append(f"F_MISSING < {cfg.max_missing}")
            if has_maf and (cfg.min_maf is not None):
                parts.append(f"MAF > {cfg.min_maf}")
            final_filter = " && ".join(parts)

            if not final_filter:
                logger.info("[SKIP] Final MAF filter disabled (min_maf=None) and/or tags absent")
                files["final"] = files["nogap"]
        else:
            logger.warning("No F_MISSING or MAF tags - skipping final filter")
            files["final"] = files["nogap"]

        if final_filter:
            run_step(
                ["bcftools", "view", "-i", final_filter, "-Oz", "-o", str(files["final"]), str(files["nogap"])],
                files["final"],
                "Apply missingness & MAF filters",
            )

        # Save stats for auditability
        stats_file = output_dir_path / "vcf_stats.txt"
        with open(stats_file, "w") as fh:
            subprocess.run(["bcftools", "stats", str(files["final"])], stdout=fh, check=True)
            logger.info(f"VCF stats saved: {stats_file}")

        # ---------------------------------------------------------------------
        # Variant annotations (ID + INFO preservation)
        # ---------------------------------------------------------------------
        try:
            cmd_anno = [
                "bcftools", "query",
                "-f", "%CHROM\t%POS\t%REF\t%ALT\t%ID\t%INFO\n",
                str(files["final"]),
            ]
            proc_anno = subprocess.Popen(cmd_anno, stdout=subprocess.PIPE, text=True)
            anno_df = pd.read_csv(
                proc_anno.stdout,  # type: ignore[arg-type]
                sep="\t", header=None,
                names=["CHROM", "POS", "REF", "ALT", "ID", "INFO"],
            )
            proc_anno.wait()

            if not anno_df.empty:
                anno_df["Variant"] = anno_df.apply(lambda r: f"{r.CHROM}:{r.POS}:{r.REF}:{r.ALT}", axis=1)
                anno_df.set_index("Variant", inplace=True)
                anno_out = output_dir_path / "variant_annotations.csv"
                anno_df.to_csv(anno_out)
                logger.info(f"Variant annotations saved: {anno_out}")
        except Exception as e:
            logger.warning(f"Could not write variant annotations: {e}")

        # ---------------------------------------------------------------------
        # VCF → 0/1 matrix (STREAMING + GT PARSING)
        # ---------------------------------------------------------------------
        logger.info("Extracting genotypes from final cleaned VCF (streaming)...")

        # Samples: keep bcftools-provided IDs verbatim (stable, unique)
        cmd_samples = ["bcftools", "query", "-l", str(files["final"])]
        samples = subprocess.run(cmd_samples, capture_output=True, text=True, check=True).stdout.splitlines()
        samples = [s.strip() for s in samples if s.strip()]

        if len(set(samples)) != len(samples):
            raise ValueError("Duplicate sample IDs detected in VCF; cannot build stable matrix")

        # Stream variants + genotypes:
        # Format: CHROM POS REF ALT then all sample GTs
        cmd_query = ["bcftools", "query", "-f", "%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n", str(files["final"])]
        proc = subprocess.Popen(cmd_query, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

        skipped = 0
        processed = 0

        assert proc.stdout is not None  # for type checkers

        line_iter = proc.stdout
        if tqdm is not None:
            total = None
            try:
                cmd_count = f'bcftools view -H "{files["final"]}" | wc -l'
                total = int(subprocess.check_output(cmd_count, shell=True, text=True).strip())
            except Exception:
                total = None
            line_iter = tqdm(line_iter, total=total, desc="Parsing variants", unit="var")

        variant_ids: List[str] = []
        matrix_rows: List[List[float]] = []

        def gt_to_01(gt: str) -> float:
            """
            Convert GT string to 0/1/NaN:
              - 0 if all called alleles are '0'
              - 1 if any called allele is non-zero
              - NaN if missing/unknown (./., .|., any '.')
            """
            gt = (gt or "").strip()
            if not gt or gt in {"./.", ".|.", "."}:
                return float("nan")

            sep = "/" if "/" in gt else ("|" if "|" in gt else None)
            if sep is None:
                return float("nan")

            alleles = gt.split(sep)
            if any(a == "." for a in alleles):
                return float("nan")

            return 1.0 if any(a != "0" for a in alleles) else 0.0

        for line in line_iter:
            if not line:
                continue
            line = line.rstrip("\n")
            if not line:
                skipped += 1
                continue

            fields = line.split("\t")
            if len(fields) < 4:
                skipped += 1
                continue

            chrom, pos, ref, alt = fields[0], fields[1], fields[2], fields[3]
            vid = f"{chrom}:{pos}:{ref}:{alt}"
            gts = fields[4:]

            if len(gts) != len(samples):
                skipped += 1
                continue

            row = [gt_to_01(x) for x in gts]
            variant_ids.append(vid)
            matrix_rows.append(row)
            processed += 1

        if proc.stderr is not None:
            err = proc.stderr.read().strip()
            if err:
                logger.warning(f"bcftools query stderr: {err}")

        # Build DataFrame (samples × variants)
        df = pd.DataFrame(matrix_rows, columns=samples, index=variant_ids).T
        df.index.name = "Sample"

        logger.info(f"Parsed variants: {processed} (skipped: {skipped})")
        logger.info(f"Matrix shape: {df.shape[0]} samples × {df.shape[1]} variants")

        df.to_csv(matrix_csv)
        logger.info(f"Genomic matrix saved: {matrix_csv}")

        return df

    # -------------------------------------------------------------------------
    # Pure-python fallback (kept for robustness; minimal in this implementation)
    # -------------------------------------------------------------------------
    def _vcf_python_fallback(
        self,
        vcf_path: Path,
        output_dir: str,
        ref_fasta: Optional[str] = None,
        regions: Optional[str] = None,
    ) -> pd.DataFrame:
        raise NotImplementedError("Python VCF fallback not implemented in this version.")

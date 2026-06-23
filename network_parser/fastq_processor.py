#!/usr/bin/env python3
# network_parser/fastq_processor.py
"""
FASTQ-to-VCF preprocessing for NetworkParser query mode.

Purpose
-------
Convert paired-end FASTQ reads into per-sample VCF.GZ files that can be
consumed by the existing DataLoader VCF-directory pathway.

This module does NOT perform statistical filtering, model training,
decision-tree construction, bootstrapping, or confidence scoring. It is a
query-time preprocessing bridge:

    FASTQ reads -> alignment -> sorted BAM -> VCF.GZ -> DataLoader -> query matrix

External command-line tools are required on PATH:
    - bwa
    - samtools
    - bcftools
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import re
import shlex
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from network_parser.utils import progress_iter
except Exception:  # pragma: no cover
    try:
        from utils import progress_iter  # type: ignore
    except Exception:  # pragma: no cover
        progress_iter = lambda iterable, **kwargs: iterable  # type: ignore


FASTQ_EXTENSIONS = (
    ".fastq.gz",
    ".fq.gz",
    ".fastq",
    ".fq",
)


@dataclass
class FastqProcessingSummary:
    """Compact provenance summary for one FASTQ preprocessing run."""

    status: str
    input_fastq_dir: str
    reference_genome: str
    output_dir: str
    final_vcf_dir: str
    discovered_pairs: int
    successful_samples: int
    failed_samples: int
    successful_vcfs: List[str]
    failed_sample_errors: Dict[str, str]
    tool_versions: Dict[str, str]
    max_parallel_samples: int
    threads_total: int
    threads_per_sample: int


class FastqProcessor:
    """
    Convert paired FASTQ files into per-sample VCF.GZ files.

    Parameters
    ----------
    config
        NetworkParserConfig-like object. Only FASTQ-related attributes are read;
        sensible defaults are used when fields are absent.
    fastq_dir
        Directory containing paired FASTQ files.
    ref_genome
        Reference FASTA used for alignment and variant calling.
    output_dir
        Output directory for FASTQ preprocessing artifacts.
    n_jobs
        Optional total thread override. If absent, config.fastq_threads or
        config.n_jobs is used.
    """

    def __init__(
        self,
        config: Any,
        fastq_dir: str,
        ref_genome: str,
        output_dir: str,
        n_jobs: Optional[int] = None,
    ) -> None:
        self.config = config
        self.fastq_dir = Path(fastq_dir)
        self.ref_genome = Path(ref_genome)
        self.output_dir = Path(output_dir)

        self.max_parallel_samples = int(
            getattr(config, "fastq_max_parallel_samples", 1) or 1
        )
        self.max_parallel_samples = max(1, self.max_parallel_samples)

        self.total_threads = self._resolve_total_threads(n_jobs=n_jobs)
        self.threads_per_sample = max(1, self.total_threads // self.max_parallel_samples)

        self.memory_per_sample_mb = getattr(config, "fastq_memory_per_sample_mb", None)
        if self.memory_per_sample_mb is not None:
            self.memory_per_sample_mb = int(self.memory_per_sample_mb)
            if self.memory_per_sample_mb < 256:
                raise ValueError("fastq_memory_per_sample_mb must be >= 256 or None")

        self.clean_intermediates = bool(getattr(config, "fastq_clean_intermediates", False))
        self.auto_index_reference = bool(getattr(config, "fastq_auto_index_reference", True))
        self.min_mapping_quality = int(getattr(config, "fastq_min_mapping_quality", 20))
        self.sample_platform = str(getattr(config, "fastq_sample_platform", "ILLUMINA") or "ILLUMINA")
        self.sort_memory = str(getattr(config, "fastq_sort_memory", "1G") or "1G")

        self.intermediate_dir = self.output_dir / "intermediate"
        self.bam_dir = self.output_dir / "bams"
        self.vcf_dir = self.output_dir / "final" / "vcf"
        self.stats_dir = self.output_dir / "stats"
        self.logs_dir = self.output_dir / "logs"

        self.failed_sample_errors: Dict[str, str] = {}
        self.tool_versions: Dict[str, str] = {}

        self._validate_inputs()
        self._setup_directory_structure()
        self._check_required_tools()
        self._prepare_reference_indexes()

        logger.info(
            "FASTQ processor initialized | fastq_dir=%s | ref=%s | parallel_samples=%d | total_threads=%d | threads_per_sample=%d",
            self.fastq_dir,
            self.ref_genome,
            self.max_parallel_samples,
            self.total_threads,
            self.threads_per_sample,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_samples(self) -> Tuple[Path, FastqProcessingSummary]:
        """
        Process all paired FASTQ samples.

        Returns
        -------
        tuple
            (final_vcf_dir, summary)
        """
        fastq_pairs = self.find_fastq_pairs()
        if not fastq_pairs:
            raise RuntimeError(f"No valid paired FASTQ files found in {self.fastq_dir}")

        logger.info(
            "FASTQ preprocessing started | pairs=%d | output_vcf_dir=%s",
            len(fastq_pairs),
            self.vcf_dir,
        )

        successful_vcfs: List[Path] = []
        with ThreadPoolExecutor(max_workers=self.max_parallel_samples) as executor:
            futures = {
                executor.submit(self._process_one_sample, r1, r2, sample): sample
                for r1, r2, sample in fastq_pairs
            }

            for future in progress_iter(
                as_completed(futures),
                total=len(futures),
                desc="FASTQ preprocessing",
                unit="sample",
                leave=False,
            ):
                sample = futures[future]
                try:
                    vcf_path = future.result()
                    successful_vcfs.append(vcf_path)
                    logger.info("FASTQ sample complete | sample=%s | vcf=%s", sample, vcf_path)
                except Exception as exc:
                    self.failed_sample_errors[sample] = str(exc)
                    logger.error("FASTQ sample failed | sample=%s | error=%s", sample, exc)

        if not successful_vcfs:
            raise RuntimeError("FASTQ preprocessing failed for every sample; no VCFs were produced.")

        summary = FastqProcessingSummary(
            status="success" if not self.failed_sample_errors else "partial_success",
            input_fastq_dir=str(self.fastq_dir),
            reference_genome=str(self.ref_genome),
            output_dir=str(self.output_dir),
            final_vcf_dir=str(self.vcf_dir),
            discovered_pairs=int(len(fastq_pairs)),
            successful_samples=int(len(successful_vcfs)),
            failed_samples=int(len(self.failed_sample_errors)),
            successful_vcfs=[str(p) for p in sorted(successful_vcfs)],
            failed_sample_errors=dict(self.failed_sample_errors),
            tool_versions=dict(self.tool_versions),
            max_parallel_samples=int(self.max_parallel_samples),
            threads_total=int(self.total_threads),
            threads_per_sample=int(self.threads_per_sample),
        )
        self._write_json(asdict(summary), self.output_dir / "fastq_processing_summary.json")

        logger.info(
            "FASTQ preprocessing complete | successful=%d | failed=%d | vcf_dir=%s",
            summary.successful_samples,
            summary.failed_samples,
            summary.final_vcf_dir,
        )
        return self.vcf_dir, summary

    def find_fastq_pairs(self) -> List[Tuple[Path, Path, str]]:
        """Find paired-end FASTQ files using common R1/R2 naming patterns."""
        files = [p for p in self.fastq_dir.iterdir() if p.is_file() and self._is_fastq(p)]
        r1_map: Dict[str, Path] = {}
        r2_map: Dict[str, Path] = {}

        for path in files:
            parsed = self._parse_fastq_name(path.name)
            if parsed is None:
                logger.debug("Ignoring FASTQ file with unsupported pair naming: %s", path.name)
                continue
            sample, read = parsed
            if read == "R1":
                r1_map.setdefault(sample, path)
            elif read == "R2":
                r2_map.setdefault(sample, path)

        pairs: List[Tuple[Path, Path, str]] = []
        for sample in sorted(r1_map):
            r2 = r2_map.get(sample)
            if r2 is None:
                logger.warning("Missing R2 FASTQ for sample=%s", sample)
                continue
            pairs.append((r1_map[sample], r2, sample))

        for sample in sorted(set(r2_map) - set(r1_map)):
            logger.warning("Missing R1 FASTQ for sample=%s", sample)

        return pairs

    # ------------------------------------------------------------------
    # Validation and setup
    # ------------------------------------------------------------------
    def _resolve_total_threads(self, n_jobs: Optional[int]) -> int:
        cpu_count = max(1, multiprocessing.cpu_count())

        if n_jobs is not None:
            requested = int(n_jobs)
        else:
            cfg_threads = getattr(self.config, "fastq_threads", None)
            if cfg_threads is not None:
                requested = int(cfg_threads)
            else:
                cfg_n_jobs = int(getattr(self.config, "n_jobs", -1) or -1)
                requested = cpu_count if cfg_n_jobs < 0 else cfg_n_jobs

        if requested < 1:
            requested = cpu_count
        return max(1, min(requested, cpu_count))

    def _validate_inputs(self) -> None:
        if not self.fastq_dir.exists() or not self.fastq_dir.is_dir():
            raise FileNotFoundError(f"FASTQ directory not found: {self.fastq_dir}")
        if not self.ref_genome.exists() or not self.ref_genome.is_file():
            raise FileNotFoundError(f"Reference genome not found: {self.ref_genome}")
        if self.max_parallel_samples < 1:
            raise ValueError("fastq_max_parallel_samples must be >= 1")
        if self.min_mapping_quality < 0:
            raise ValueError("fastq_min_mapping_quality must be >= 0")

    def _setup_directory_structure(self) -> None:
        for path in [
            self.output_dir,
            self.intermediate_dir,
            self.bam_dir,
            self.vcf_dir,
            self.stats_dir,
            self.logs_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def _check_required_tools(self) -> None:
        missing: List[str] = []
        for tool in ["bwa", "samtools", "bcftools"]:
            if shutil.which(tool) is None:
                missing.append(tool)
            else:
                self.tool_versions[tool] = self._tool_version(tool)
        if missing:
            raise RuntimeError(
                "FASTQ query mode requires external tools on PATH: " + ", ".join(missing)
            )

    def _prepare_reference_indexes(self) -> None:
        bwa_ok = all(
            self.ref_genome.with_suffix(self.ref_genome.suffix + ext).exists()
            for ext in [".amb", ".ann", ".bwt", ".pac", ".sa"]
        )
        fai_ok = Path(str(self.ref_genome) + ".fai").exists()

        if bwa_ok and fai_ok:
            return

        if not self.auto_index_reference:
            missing = []
            if not bwa_ok:
                missing.append("BWA index")
            if not fai_ok:
                missing.append("samtools FASTA index")
            raise RuntimeError(
                "Reference indexes missing: " + ", ".join(missing) + 
                ". Enable fastq_auto_index_reference or index the reference manually."
            )

        if not bwa_ok:
            logger.info("BWA reference index missing; creating index for %s", self.ref_genome)
            self._execute(["bwa", "index", str(self.ref_genome)], "bwa_index")

        if not fai_ok:
            logger.info("samtools FASTA index missing; creating index for %s", self.ref_genome)
            self._execute(["samtools", "faidx", str(self.ref_genome)], "samtools_faidx")

    # ------------------------------------------------------------------
    # FASTQ naming
    # ------------------------------------------------------------------
    @staticmethod
    def _is_fastq(path: Path) -> bool:
        name = path.name.lower()
        return any(name.endswith(ext) for ext in FASTQ_EXTENSIONS)

    @staticmethod
    def _strip_fastq_extension(name: str) -> str:
        lower = name.lower()
        for ext in FASTQ_EXTENSIONS:
            if lower.endswith(ext):
                return name[: -len(ext)]
        return Path(name).stem

    @classmethod
    def _parse_fastq_name(cls, name: str) -> Optional[Tuple[str, str]]:
        stem = cls._strip_fastq_extension(name)
        patterns = [
            r"^(?P<sample>.+?)[._-]R(?P<read>[12])(?:[._-]001)?$",
            r"^(?P<sample>.+?)[._-](?P<read>[12])(?:[._-]001)?$",
        ]
        for pattern in patterns:
            match = re.match(pattern, stem, flags=re.IGNORECASE)
            if match:
                sample = match.group("sample").strip("._-")
                read = "R1" if match.group("read") == "1" else "R2"
                if sample:
                    return sample, read
        return None

    # ------------------------------------------------------------------
    # Per-sample processing
    # ------------------------------------------------------------------
    def _process_one_sample(self, r1: Path, r2: Path, sample: str) -> Path:
        safe_sample = self._safe_sample_name(sample)
        sample_dir = self.intermediate_dir / safe_sample
        sample_dir.mkdir(parents=True, exist_ok=True)

        sorted_bam = self.bam_dir / f"{safe_sample}.sorted.bam"
        final_vcf = self.vcf_dir / f"{safe_sample}.vcf.gz"

        try:
            self._align_sort_index_sample(
                r1=r1,
                r2=r2,
                sample=safe_sample,
                sorted_bam=sorted_bam,
            )
            self._call_variants(
                sample=safe_sample,
                sorted_bam=sorted_bam,
                final_vcf=final_vcf,
            )
            self._write_sample_metrics(sample=safe_sample, sorted_bam=sorted_bam, final_vcf=final_vcf)
        finally:
            if self.clean_intermediates:
                self._clean_sample_intermediates(sample_dir)

        if not final_vcf.exists():
            raise RuntimeError(f"Expected final VCF was not created: {final_vcf}")
        return final_vcf

    def _align_sort_index_sample(self, r1: Path, r2: Path, sample: str, sorted_bam: Path) -> None:
        threads = self.threads_per_sample
        view_threads = max(1, threads // 2)
        sort_threads = max(1, threads)
        rg = f"@RG\\tID:{sample}\\tSM:{sample}\\tPL:{self.sample_platform}"

        cmd = (
            "set -euo pipefail; "
            f"bwa mem -t {threads} -R {shlex.quote(rg)} "
            f"{shlex.quote(str(self.ref_genome))} {shlex.quote(str(r1))} {shlex.quote(str(r2))} "
            f"| samtools view -@ {view_threads} -b - "
            f"| samtools sort -@ {sort_threads} -m {shlex.quote(self.sort_memory)} "
            f"-o {shlex.quote(str(sorted_bam))} -"
        )
        self._execute_shell(cmd, f"{sample}_alignment_sort")
        self._execute(["samtools", "index", str(sorted_bam)], f"{sample}_bam_index")

    def _call_variants(self, sample: str, sorted_bam: Path, final_vcf: Path) -> None:
        threads = max(1, self.threads_per_sample)
        cmd = (
            "set -euo pipefail; "
            f"bcftools mpileup -f {shlex.quote(str(self.ref_genome))} "
            f"-Ou --min-MQ {int(self.min_mapping_quality)} {shlex.quote(str(sorted_bam))} "
            f"| bcftools call -mv --threads {threads} -Oz -o {shlex.quote(str(final_vcf))}"
        )
        self._execute_shell(cmd, f"{sample}_variant_calling")
        self._execute(["bcftools", "index", "-t", str(final_vcf)], f"{sample}_vcf_index")

    def _write_sample_metrics(self, sample: str, sorted_bam: Path, final_vcf: Path) -> None:
        self._execute_shell(
            f"samtools flagstat {shlex.quote(str(sorted_bam))} > {shlex.quote(str(self.stats_dir / (sample + '.flagstat.txt')))}",
            f"{sample}_flagstat",
        )
        self._execute_shell(
            f"samtools stats {shlex.quote(str(sorted_bam))} > {shlex.quote(str(self.stats_dir / (sample + '.alignment.stats.txt')))}",
            f"{sample}_samtools_stats",
        )
        self._execute_shell(
            f"bcftools stats {shlex.quote(str(final_vcf))} > {shlex.quote(str(self.stats_dir / (sample + '.vcf.stats.txt')))}",
            f"{sample}_bcftools_stats",
        )

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------
    def _execute(self, cmd: List[str], step_name: str) -> None:
        quoted = " ".join(shlex.quote(str(x)) for x in cmd)
        self._execute_shell(quoted, step_name)

    def _execute_shell(self, cmd: str, step_name: str) -> None:
        log_path = self.logs_dir / f"{self._safe_sample_name(step_name)}.log"
        if self.memory_per_sample_mb is not None:
            cmd = f"ulimit -v {int(self.memory_per_sample_mb) * 1024}; {cmd}"

        logger.debug("Executing %s: %s", step_name, cmd)
        result = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        log_path.write_text(
            "COMMAND\n=======\n"
            f"{cmd}\n\n"
            "STDOUT\n======\n"
            f"{result.stdout}\n\n"
            "STDERR\n======\n"
            f"{result.stderr}\n",
            encoding="utf-8",
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"{step_name} failed with exit code {result.returncode}. See log: {log_path}"
            )

    def _tool_version(self, tool: str) -> str:
        version_commands = {
            "bwa": ["bwa"],
            "samtools": ["samtools", "--version"],
            "bcftools": ["bcftools", "--version"],
        }
        cmd = version_commands.get(tool, [tool, "--version"])
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )
            text = (result.stdout or result.stderr or "").strip().splitlines()
            return text[0] if text else "available"
        except Exception:
            return "available"

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_sample_name(sample: str) -> str:
        sample = str(sample).strip()
        sample = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample)
        return sample.strip("._-") or "sample"

    @staticmethod
    def _write_json(payload: Dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)

    def _clean_sample_intermediates(self, sample_dir: Path) -> None:
        if not sample_dir.exists():
            return
        try:
            for path in sample_dir.rglob("*"):
                if path.is_file():
                    path.unlink(missing_ok=True)
            sample_dir.rmdir()
        except Exception as exc:
            logger.warning("Could not clean FASTQ intermediate directory %s: %s", sample_dir, exc)

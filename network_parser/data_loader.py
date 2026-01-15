# data_loader.py
import logging
import pandas as pd
import numpy as np
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, Tuple
import sys
import glob

logger = logging.getLogger(__name__)


class DataLoader:
    """Enhanced DataLoader with VCF→clean SNP matrix and consensus FASTA pipeline using bcftools/vcftools."""

    def __init__(self, use_bcftools: bool = True, temp_dir: Optional[str] = None):
        self.use_bcftools = use_bcftools
        self.temp_dir = Path(temp_dir or tempfile.mkdtemp(prefix="networkparser_"))
        self.temp_dir.mkdir(exist_ok=True)
        self._check_bcftools()

    def _check_bcftools(self):
        try:
            subprocess.run(['bcftools', '--version'], check=True, capture_output=True)
            subprocess.run(['vcftools', '--version'], check=True, capture_output=True)
            logger.info("✅ bcftools & vcftools detected - fast VCF processing enabled")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("⚠️ bcftools/vcftools not found - VCF parsing will be slower (pure Python fallback)")
            self.use_bcftools = False

    def load_genomic_matrix(self, file_path: str, output_dir: Optional[str] = None,
                            ref_fasta: Optional[str] = None, label_column: Optional[str] = None) -> pd.DataFrame:
        path = Path(file_path)
        logger.info(f"Loading genomic data from: {path}")

        if path.is_dir():
            return self._load_vcf_folder(path, output_dir, ref_fasta)
        elif path.suffix.lower() in {'.csv', '.tsv'}:
            return self._load_csv_matrix(path, output_dir, label_column)
        elif path.suffix.lower() in {'.vcf', '.vcf.gz'}:
            if self.use_bcftools:
                return self._vcf_bcftools_pipeline(path, output_dir, ref_fasta)
            else:
                return self._vcf_python_fallback(path, output_dir)
        elif path.suffix.lower() == '.fasta':
            return self._fasta_to_matrix(path, output_dir)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}. Use CSV/TSV/VCF/FASTA or a folder of VCF files")

    def _load_vcf_folder(self, folder_path: Path, output_dir: Optional[str] = None,
                         ref_fasta: Optional[str] = None) -> pd.DataFrame:
        if not self.use_bcftools:
            raise ValueError("bcftools is required to merge and process a folder of VCF files.")

        logger.info(f"Processing folder of VCF files: {folder_path}")

        vcf_files = list(folder_path.glob('*.vcf')) + list(folder_path.glob('*.vcf.gz'))
        if not vcf_files:
            raise ValueError(f"No VCF files found in folder: {folder_path}")

        logger.info(f"Found {len(vcf_files)} VCF files to merge.")

        output_dir_path = Path(output_dir) if output_dir else self.temp_dir
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # Index all input files before merge
        logger.info("Checking and indexing all input VCF files...")
        for vcf_file in vcf_files:
            if vcf_file.suffix.lower() == '.gz':
                tbi_file = vcf_file.with_name(vcf_file.name + '.tbi')
                if not tbi_file.exists():
                    logger.info(f"Creating index for: {vcf_file.name}")
                    subprocess.run(["tabix", "-p", "vcf", str(vcf_file)], check=True)

        logger.info("All input VCF files are indexed.")

        # Merge
        merged_vcf = output_dir_path / "merged.vcf.gz"
        cmd_merge = [
            "bcftools", "merge",
            "--threads", "8",
            "-Oz", "-o", str(merged_vcf)
        ] + [str(v) for v in vcf_files]

        logger.info("Merging VCF files...")
        subprocess.run(cmd_merge, check=True)

        # Index merged
        subprocess.run(["tabix", "-p", "vcf", str(merged_vcf)], check=True)
        logger.info("Merged VCF created and indexed successfully.")

        return self._vcf_bcftools_pipeline(merged_vcf, str(output_dir_path), ref_fasta)

    def _vcf_bcftools_pipeline(self, vcf_path: Path, output_dir: str, ref_fasta: Optional[str] = None) -> pd.DataFrame:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing VCF: {vcf_path}")

        prefix = output_dir_path / vcf_path.stem
        v_bial = prefix.with_name(prefix.name + ".biallelic.vcf.gz")
        v_final = prefix.with_name(prefix.name + ".final.vcf.gz")
        matrix_csv = prefix.with_name(prefix.name + ".genomic_matrix.csv")

        # Index input (redundant safety)
        tbi = vcf_path.with_suffix(vcf_path.suffix + '.tbi')
        if not tbi.exists():
            subprocess.run(["tabix", "-p", "vcf", str(vcf_path)], check=True)

        # Filter: biallelic SNPs
        cmd_view = [
            "bcftools", "view",
            "-m2", "-M2", "-v", "snps",
            "--threads", "8",
            "-Oz", "-o", str(v_bial),
            str(vcf_path)
        ]
        subprocess.run(cmd_view, check=True)
        subprocess.run(["tabix", "-p", "vcf", str(v_bial)], check=True)

        # Final file (you can add more filters here: QUAL, DP, GQ, missingness, thinning...)
        shutil.copy(v_bial, v_final)  # placeholder — add real filtering if needed
        subprocess.run(["tabix", "-p", "vcf", str(v_final)], check=True)

        # ────────────────────────────────────────────────
        # Extract real sample names + genotypes
        # ────────────────────────────────────────────────
        logger.info("Extracting sample names and genotypes...")

        # Get samples
        cmd_samples = ["bcftools", "query", "-l", str(v_final)]
        samples_raw = subprocess.run(cmd_samples, capture_output=True, text=True, check=True).stdout.strip().splitlines()

        # Normalize sample IDs (matching extract_subset.py logic)
        samples = []
        for s in samples_raw:
            name = s
            if name.endswith(".vcf.gz"):
                name = name[:-7]
            elif name.endswith(".vcf"):
                name = name[:-4]
            clean_id = name.split("_")[0].split(".")[0].split("/")[0].strip()
            if clean_id and clean_id not in samples:
                samples.append(clean_id)

        logger.info(f"Extracted {len(samples)} unique normalized sample IDs")
        logger.info(f"First few samples: {samples[:6]}")

        if not samples:
            raise ValueError("No samples found in VCF header")

        # Prepare data dict
        data = {sample: [] for sample in samples}

        # Query genotypes (CHROM + all GT columns)
        cmd_gt = [
            "bcftools", "query",
            "-f", "%CHROM\t[%GT\t]\n",
            str(v_final)
        ]
        gt_output = subprocess.run(cmd_gt, capture_output=True, text=True, check=True).stdout.strip()

        lines = gt_output.splitlines()
        logger.info(f"Processing {len(lines)} variant lines...")

        for line in lines:
            if not line.strip():
                continue
            fields = line.split('\t')
            chrom = fields[0]
            gts = fields[1:]  # one GT per sample

            if len(gts) != len(samples):
                logger.warning(f"GT count mismatch at {chrom}: {len(gts)} vs {len(samples)} samples")
                continue

            for i, gt in enumerate(gts):
                # Handle missing / no-call
                if gt == '.' or not gt:
                    binary = 0
                else:
                    binary = 1 if '1' in gt else 0  # alt present → 1
                data[samples[i]].append(binary)

        # Create DataFrame
        if not data or not data[samples[0]]:
            logger.error("No genotypes parsed – returning empty matrix")
            df = pd.DataFrame(index=samples)
        else:
            df = pd.DataFrame(data, index=samples)
            df.index.name = 'Sample'
            df.columns.name = 'SNP'

        # Save for debugging
        df.to_csv(matrix_csv)
        logger.info(f"💾 Saved genomic matrix: {matrix_csv}")
        logger.info(f"Matrix shape: {df.shape}")

        # Optional: consensus FASTA
        if ref_fasta:
            consensus_fa = output_dir_path / f"{vcf_path.stem}.consensus.fa"
            cmd_cons = [
                "bcftools", "consensus",
                "-f", ref_fasta,
                str(v_final),
                "-o", str(consensus_fa)
            ]
            try:
                subprocess.run(cmd_cons, check=True)
                logger.info(f"Consensus FASTA created: {consensus_fa}")
            except Exception as e:
                logger.warning(f"Consensus FASTA failed: {e}")

        return df

    # ────────────────────────────────────────────────
    # The rest of your methods remain unchanged
    # ────────────────────────────────────────────────

    def _load_csv_matrix(self, path: Path, output_dir: Optional[str], label_column: Optional[str]) -> pd.DataFrame:
        logger.info(f"Loading CSV/TSV from: {path}")
        df = pd.read_csv(path, index_col=0)
        return df

    def _vcf_python_fallback(self, path: Path, output_dir: Optional[str]) -> pd.DataFrame:
        raise NotImplementedError("Pure Python VCF fallback not implemented")

    def _fasta_to_matrix(self, fasta_path: Path, output_dir: Optional[str]) -> pd.DataFrame:
        try:
            from Bio import SeqIO
        except ImportError:
            raise ImportError("pip install biopython")
        
        sequences = {rec.id: str(rec.seq) for rec in SeqIO.parse(fasta_path, 'fasta')}
        if len(sequences) < 2:
            raise ValueError("FASTA needs ≥2 sequences")
        
        ref_id = next(iter(sequences))
        ref_seq = sequences.pop(ref_id)
        matrix = []
        
        for pos in range(len(ref_seq)):
            snp_col = [1 if seq[pos] != ref_seq[pos] else 0 for seq in sequences.values()]
            snp_col.insert(0, 0)
            matrix.append(snp_col)
        
        df = pd.DataFrame(matrix, columns=[f"pos_{i}" for i in range(len(ref_seq))])
        df.index = [ref_id] + list(sequences.keys())
        return df

    def load_metadata(self, file_path: str, output_dir: Optional[str] = None) -> pd.DataFrame:
        path = Path(file_path)
        logger.info(f"Loading metadata from: {path}")
        
        if path.suffix.lower() in {'.csv', '.tsv'}:
            sep = ',' if path.suffix.lower() == '.csv' else '\t'
            df = pd.read_csv(path, sep=sep, index_col=0)
        else:
            raise ValueError(f"Unsupported metadata format: {path.suffix}")

        duplicates = df.index.duplicated(keep=False)
        if duplicates.any():
            logger.warning(f"Found {duplicates.sum()} duplicate IDs. Keeping first.")
            df = df[~df.index.duplicated(keep='first')]

        if output_dir:
            out_path = Path(output_dir) / "deduplicated_metadata.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path)
            logger.info(f"Saved deduplicated metadata: {out_path}")

        return df

    def load_known_markers(self, file_path: str, output_dir: Optional[str] = None) -> List[str]:
        path = Path(file_path)
        logger.info(f"Loading known markers from: {path}")

        markers = [line.strip() for line in path.read_text().splitlines() if line.strip()]

        if output_dir:
            out_path = Path(output_dir) / "processed_known_markers.txt"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text('\n'.join(markers))
            logger.info(f"Saved known markers: {out_path}")

        return markers

    def align_data(self, genomic_data: pd.DataFrame, metadata: Optional[pd.DataFrame],
                   label_column: str, output_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info("Aligning genomic data and metadata...")

        if genomic_data is None:
            raise ValueError("genomic_data is None – check VCF processing in _vcf_bcftools_pipeline")

        # Debug: show sample names early
        logger.info(f"Genomic samples (first 6): {list(genomic_data.index[:6])}")
        if metadata is not None:
            logger.info(f"Metadata samples (first 6): {list(metadata.index[:6])}")

        if metadata is None and label_column in genomic_data.columns:
            logger.info(f"Extracting labels '{label_column}' from genomic matrix")
            labels = genomic_data[label_column]
            aligned_genomic = genomic_data.drop(columns=[label_column])
        else:
            if metadata is None:
                raise ValueError(f"Label column '{label_column}' not found. Provide --meta.")
            
            common_samples = genomic_data.index.intersection(metadata.index)
            if common_samples.empty:
                logger.error(f"No overlap! Genomic: {len(genomic_data)} samples | Metadata: {len(metadata)}")
                logger.error(f"Genomic sample preview: {list(genomic_data.index[:10])}")
                logger.error(f"Metadata sample preview: {list(metadata.index[:10])}")
                raise ValueError("No common samples between genomic data and metadata.")
            
            aligned_genomic = genomic_data.loc[common_samples]
            labels = metadata.loc[common_samples, label_column]

        non_na_mask = ~labels.isna()
        aligned_genomic = aligned_genomic.loc[non_na_mask]
        labels = labels[non_na_mask]

        invariants = aligned_genomic.nunique() <= 1
        if invariants.any():
            logger.info(f"Removed {invariants.sum()} invariant features")
            aligned_genomic = aligned_genomic.loc[:, ~invariants]

        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            aligned_genomic.to_csv(out_path / "aligned_genomic_matrix.csv")
            labels.to_csv(out_path / "aligned_labels.csv")

        logger.info(f"Aligned: {len(labels)} samples, {aligned_genomic.shape[1]} features")
        return aligned_genomic, labels

    def cleanup(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        logger.info(f"Cleaned temp dir: {self.temp_dir}")
# data_loader.py
import logging
import pandas as pd
import subprocess
import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, Tuple
import sys
import glob  # Added for folder scanning

logger = logging.getLogger(__name__)


class DataLoader:
    """Enhanced DataLoader with VCF→clean SNP matrix and consensus FASTA pipeline using bcftools/vcftools."""

    def __init__(self, use_bcftools: bool = True, temp_dir: Optional[str] = None):
        """
        Initialize with bcftools support.
        
        Args:
            use_bcftools: Enable fast bcftools filtering (requires bcftools/vcftools in PATH)
            temp_dir: Temporary directory for processing (auto-cleaned)
        """
        self.use_bcftools = use_bcftools
        self.temp_dir = Path(temp_dir or tempfile.mkdtemp(prefix="networkparser_"))
        self.temp_dir.mkdir(exist_ok=True)
        self._check_bcftools()

    def _check_bcftools(self):
        """Check if bcftools/vcftools available."""
        try:
            subprocess.run(['bcftools', '--version'], check=True, capture_output=True)
            subprocess.run(['vcftools', '--version'], check=True, capture_output=True)
            logger.info("✅ bcftools & vcftools detected - fast VCF processing enabled")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("⚠️  bcftools/vcftools not found - VCF parsing will be slower (pure Python fallback)")
            self.use_bcftools = False

    def load_genomic_matrix(self, file_path: str, output_dir: Optional[str] = None,
                            ref_fasta: Optional[str] = None, label_column: Optional[str] = None) -> pd.DataFrame:
        """
        Load ANY genomic format → clean binary SNP matrix for NetworkParser.
        
        Supported inputs:
        - CSV/TSV: Direct binary matrices (existing behavior)
        - VCF(.gz): Auto-filtered biallelic SNPs → binary matrix (bcftools pipeline)
        - Folder containing multiple VCF(.gz) files: Merge them using bcftools and process as a multi-sample VCF
        - FASTA: SNP alignment → binary matrix
        
        Filters applied to VCF (microbial best practices):
        - Biallelic SNPs only (QUAL≥30, DP≥10x samples, GQ≥20, <10% missing)
        - SNP thinning (≥10bp apart to avoid clustering artifacts)
        - Binary: 0=ref, 1=alt (missing→0)
        
        Args:
            file_path: Input file (CSV/TSV/VCF/FASTA) or directory containing multiple VCF files
            output_dir: Save intermediate files
            ref_fasta: Reference FASTA (optional for VCF consensus FASTA generation)
            label_column: If labels in genomic file, extract them
            
        Returns:
            pd.DataFrame: Clean binary matrix (rows=samples, cols=SNP positions)
        """
        path = Path(file_path)
        logger.info(f"Loading genomic data from: {path}")

        if path.is_dir():
            # Handle folder of VCF files
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

    def _load_vcf_folder(self, folder_path: Path, output_dir: Optional[Path] = None,
                         ref_fasta: Optional[Path] = None) -> pd.DataFrame:
        """
        Load a folder containing multiple VCF files, merge them using bcftools merge, and process via the pipeline.
        
        Assumes each VCF is for a single sample. Merges into a multi-sample VCF before filtering.
        """
        if not self.use_bcftools:
            raise ValueError("bcftools is required to merge and process a folder of VCF files.")
        
        logger.info(f"Processing folder of VCF files: {folder_path}")
        
        # Collect all VCF files in the folder (including .vcf.gz)
        vcf_files = list(folder_path.glob('*.vcf')) + list(folder_path.glob('*.vcf.gz'))
        if not vcf_files:
            raise ValueError(f"No VCF files found in folder: {folder_path}")
        
        logger.info(f"Found {len(vcf_files)} VCF files to merge.")
        
        if output_dir is None:
            output_dir = self.temp_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare merged VCF path
        merged_vcf = output_dir / "merged.vcf.gz"
        
        # Merge VCFs using bcftools merge
        cmd_merge = [
            "bcftools", "merge",
            "--threads", "8",
            "-Oz", "-o", str(merged_vcf)
        ] + [str(v) for v in vcf_files]
        
        logger.info("Merging VCF files...")
        subprocess.run(cmd_merge, check=True)
        
        # Index the merged VCF
        subprocess.run(["tabix", "-p", "vcf", str(merged_vcf)], check=True)
        
        logger.info("Merged VCF created successfully.")
        
        # Now process the merged VCF using the existing pipeline
        return self._vcf_bcftools_pipeline(merged_vcf, output_dir, ref_fasta)

    def _vcf_bcftools_pipeline(self, vcf_path: Path, output_dir: Optional[Path] = None,
                               ref_fasta: Optional[Path] = None) -> pd.DataFrame:
        """
        Fast bcftools-based pipeline: single-sample VCF → filtered, binary SNP matrix
        (samples × variants). Binary encoding: 1 = alt allele present (het or hom),
        0 = ref/ref or missing.

        Suitable for bacterial GWAS / AMR association (e.g. TB lineage / resistance).
        """
        if output_dir is None:
            output_dir = self.temp_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing VCF: {vcf_path}")

        # ────────────────────────────────────────────────
        # 1. Prepare output paths
        # ────────────────────────────────────────────────
        prefix = output_dir / vcf_path.stem
        v_bial = prefix.with_name(prefix.name + ".biallelic.vcf.gz")
        v_tagged = prefix.with_name(prefix.name + ".tagged.vcf.gz")
        v_final = prefix.with_name(prefix.name + ".final.vcf.gz")
        matrix_csv = prefix.with_name(prefix.name + ".genomic_matrix.csv")

        # ────────────────────────────────────────────────
        # 2. Index input if needed
        # ────────────────────────────────────────────────
        tbi = vcf_path.with_name(vcf_path.name + '.tbi')
        if not tbi.exists():
            subprocess.run(["tabix", "-p", "vcf", str(vcf_path)], check=True)

        # ────────────────────────────────────────────────
        # 3. Extract biallelic SNPs only
        # ────────────────────────────────────────────────
        cmd_view = [
            "bcftools", "view",
            "-m2", "-M2", "-v", "snps",
            "--threads", "8",
            "-Oz", "-o", str(v_bial),
            str(vcf_path)
        ]
        subprocess.run(cmd_view, check=True)
        subprocess.run(["tabix", "-p", "vcf", str(v_bial)], check=True)
        # ...(truncated 9216 characters)...')[0]
        binary = 1 if '1' in gt else 0
        data[samples[i]].append(binary)

        df = pd.DataFrame(data).T
        df.index.name = 'Sample'
        df.columns.name = 'SNP'
        return df

    def _fasta_to_matrix(self, fasta_path: Path, output_dir: Optional[str]) -> pd.DataFrame:
        """Convert FASTA alignment to binary matrix (ref=0, alt=1)."""
        try:
            from Bio import SeqIO
        except ImportError:
            raise ImportError("Install biopython for FASTA support: pip install biopython")

        sequences = {rec.id: str(rec.seq) for rec in SeqIO.parse(fasta_path, 'fasta')}

        if len(sequences) < 2:
            raise ValueError("FASTA needs ≥2 sequences for SNP calling")

        ref_id = next(iter(sequences))  # First sequence as reference
        ref_seq = sequences.pop(ref_id)
        matrix = []

        for pos in range(len(ref_seq)):
            snp_col = [1 if seq[pos] != ref_seq[pos] else 0 for seq in sequences.values()]
            snp_col.insert(0, 0)  # Ref is always 0
            matrix.append(snp_col)

        df = pd.DataFrame(matrix, index=[f"pos_{i}" for i in range(len(ref_seq))]).T
        df.index = [ref_id] + list(sequences.keys())
        return df

    def load_metadata(self, file_path: str, output_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Load metadata from CSV/TSV and deduplicate samples.
        """
        path = Path(file_path)
        logger.info(f"Loading metadata from: {path}")

        if path.suffix.lower() in {'.csv', '.tsv'}:
            sep = ',' if path.suffix.lower() == '.csv' else '\t'
            df = pd.read_csv(path, sep=sep, index_col=0)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

        duplicates = df.index.duplicated(keep=False)
        if duplicates.any():
            logger.warning(f"Found {duplicates.sum()} duplicate sample IDs. Keeping first occurrence.")
            df = df[~df.index.duplicated(keep='first')]

        if output_dir:
            output_path = Path(output_dir) / "deduplicated_metadata.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path)
            logger.info(f"Saved deduplicated metadata to: {output_path}")

        return df

    def load_known_markers(self, file_path: str, output_dir: Optional[str] = None) -> List[str]:
        """
        Load known markers from text-based files.
        """
        path = Path(file_path)
        logger.info(f"Loading known markers from: {path}")

        markers = [line.strip() for line in path.read_text().splitlines() if line.strip()]

        if output_dir:
            output_path = Path(output_dir) / "processed_known_markers.txt"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text('\n'.join(markers))
            logger.info(f"Saved processed known markers to: {output_path}")

        return markers

    def align_data(self, genomic_data: pd.DataFrame, metadata: Optional[pd.DataFrame],
                   label_column: str, output_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Enhanced alignment with label extraction from genomic matrix if present."""
        logger.info("Aligning genomic data and metadata...")

        # Extract labels from genomic_data if column exists and no metadata
        if metadata is None and label_column in genomic_data.columns:
            logger.info(f"📊 Extracting labels '{label_column}' directly from genomic matrix")
            labels = genomic_data[label_column]
            aligned_genomic = genomic_data.drop(columns=[label_column])
        else:
            # Original alignment logic
            if metadata is not None:
                common_samples = genomic_data.index.intersection(metadata.index)
                if common_samples.empty:
                    raise ValueError("No common samples between genomic data and metadata.")
                aligned_genomic = genomic_data.loc[common_samples]
                aligned_metadata = metadata.loc[common_samples]
                labels = aligned_metadata[label_column]
            else:
                raise ValueError(f"Label column '{label_column}' not found. Provide --meta or labels in genomic file.")

        # Filter missing labels + invariants
        non_na_mask = ~labels.isna()
        aligned_genomic = aligned_genomic.loc[non_na_mask]
        labels = labels[non_na_mask]

        invariants = aligned_genomic.nunique() <= 1
        if invariants.any():
            logger.info(f"Removed {invariants.sum()} invariant features")
            aligned_genomic = aligned_genomic.loc[:, ~invariants]

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            aligned_genomic.to_csv(Path(output_dir) / "aligned_genomic_matrix.csv")
            labels.to_csv(Path(output_dir) / "aligned_labels.csv")

        logger.info(f"✅ Aligned: {len(labels)} samples, {aligned_genomic.shape[1]} features")
        return aligned_genomic, labels

    def cleanup(self):
        """Clean temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        logger.info(f"🧹 Cleaned temp dir: {self.temp_dir}")
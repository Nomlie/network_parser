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
        - VCF(.gz): Auto-filtered biallelic SNPs → binary matrix (NEW: bcftools pipeline)
        - FASTA: SNP alignment → binary matrix
        
        Filters applied to VCF (microbial best practices):
        - Biallelic SNPs only (QUAL≥30, DP≥10x samples, GQ≥20, <10% missing)
        - SNP thinning (≥10bp apart to avoid clustering artifacts)
        - Binary: 0=ref, 1=alt (missing→0)
        
        Args:
            file_path: Input file (CSV/TSV/VCF/FASTA)
            output_dir: Save intermediate files
            ref_fasta: Reference FASTA (optional for VCF consensus FASTA generation)
            label_column: If labels in genomic file, extract them
            
        Returns:
            pd.DataFrame: Clean binary matrix (rows=samples, cols=SNP positions)
        """
        path = Path(file_path)
        logger.info(f"Loading genomic data from: {path} (format: {path.suffix})")
        
        if path.suffix.lower() in {'.csv', '.tsv'}:
            return self._load_csv_matrix(path, output_dir, label_column)
        elif path.suffix.lower() in {'.vcf', '.vcf.gz'}:
            if self.use_bcftools:
                return self._vcf_bcftools_pipeline(path, output_dir, ref_fasta)
            else:
                return self._vcf_python_fallback(path, output_dir)
        elif path.suffix.lower() == '.fasta':
            return self._fasta_to_matrix(path, output_dir)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}. Use CSV/TSV/VCF/FASTA")
    
    def _vcf_bcftools_pipeline(self, vcf_path: Path, output_dir: Optional[str], 
                             ref_fasta: Optional[str]) -> pd.DataFrame:
        """Fast bcftools VCF → clean SNP matrix pipeline (from the flow diagram)."""
        logger.info("🚀 Running fast bcftools VCF filtering pipeline...")
        
        out_prefix = self.temp_dir / "filtered_snps"
        final_vcf = out_prefix.with_suffix('.final.vcf.gz')
        
        # Step 1: Index VCF
        if not (vcf_path.with_suffix('.tbi').exists() or vcf_path.with_suffix('.csi').exists()):
            subprocess.run(['tabix', '-p', 'vcf', str(vcf_path)], check=True)
        
        # Step 2: Biallelic SNPs + QUAL filter (microbial standards)
        bcftools_view = [
            'bcftools', 'view', '-m2', '-M2', '-v', 'snps', '--threads', '8',
            str(vcf_path), '-Oz', '-o', f"{out_prefix}.biallelic.vcf.gz"
        ]
        subprocess.run(bcftools_view, check=True)
        
        # Step 3: Hard filters (QUAL≥30, DP, missingness) + per-sample GQ/DP
        subprocess.run([
            'bcftools', 'filter', '-i', 'QUAL>=30 && INFO/DP>=100',
            f"{out_prefix}.biallelic.vcf.gz", '|',
            'bcftools', '+fill-tags', '-Oz', '-o', f"{out_prefix}.tagged.vcf.gz", '--', '-t', 'F_MISSING'
        ], shell=True, check=True, executable='/bin/bash')
        
        subprocess.run([
            'bcftools', 'filter', '-i', 'F_MISSING<=0.1',
            f"{out_prefix}.tagged.vcf.gz", '|',
            'bcftools', 'view', '-i', 'FMT/GQ>=20 & FMT/DP>=10', 
            '-Oz', '-o', str(final_vcf)
        ], shell=True, check=True, executable='/bin/bash')
        
        # Step 4: Thin SNPs (avoid clustering ≥10bp)
        subprocess.run([
            'vcftools', '--gzvcf', str(final_vcf), '--thin', '10',
            '--recode', '--stdout'
        ], stdout=open(f"{final_vcf}.tmp", 'wb'), check=True)
        shutil.move(f"{final_vcf}.tmp", final_vcf)
        subprocess.run(['tabix', '-p', 'vcf', str(final_vcf)], check=True)
        
        # Step 5: Generate binary matrix (samples x SNP positions)
        matrix_csv = self.temp_dir / "genomic_matrix.csv"
        samples = subprocess.run(['bcftools', 'query', '-l', str(final_vcf)], 
                               capture_output=True, text=True).stdout.strip().split('\n')
        
        # Fast GT→binary extraction
        gt_query = subprocess.run([
            'bcftools', 'query', '-f', '%CHROM:%POS[\t%GT]\n', str(final_vcf)
        ], capture_output=True, text=True).stdout
        
        # Parse to binary matrix (0=ref, 1=alt/missing→0)
        lines = gt_query.strip().split('\n')
        positions = [line.split('\t')[0] for line in lines]
        matrix_data = {}
        for sample in samples:
            matrix_data[sample] = []
        for line in lines:
            gts = line.split('\t')[1:]
            for i, gt in enumerate(gts):
                binary = 1 if '1' in gt else 0
                matrix_data[samples[i]].append(binary)
        
        df = pd.DataFrame(matrix_data, index=positions).T
        
        # NEW: Generate consensus FASTAs if ref_fasta provided
        if ref_fasta:
            consensus_dir = self.generate_consensus_fastas(final_vcf, ref_fasta, output_dir)
            logger.info(f"Consensus FASTAs generated in: {consensus_dir}")
        
        logger.info(f"✅ VCF pipeline complete: {df.shape[0]} samples, {df.shape[1]} clean SNPs")
        
        # Save intermediates if requested
        if output_dir:
            df.to_csv(Path(output_dir) / "genomic_matrix.csv")
            shutil.copy(final_vcf, Path(output_dir) / "filtered_snps.final.vcf.gz")
            logger.info(f"💾 Saved matrix & VCF to {output_dir}")
        
        return df
    
    def generate_consensus_fastas(self, filtered_vcf: Path, ref_fasta: str, 
                                output_dir: Optional[str] = None, 
                                fasta_type: str = "individual") -> Path:
        """
        Generate consensus FASTA(s) using bcftools consensus.
        
        Args:
            filtered_vcf: Path to filtered VCF (from pipeline)
            ref_fasta: Reference genome FASTA (e.g., H37Rv for TB)
            output_dir: Save FASTAs here
            fasta_type: "individual" (one FASTA per sample) or "multi" (one file with all)
        
        Returns:
            Path to output directory with FASTAs
        """
        if not self.use_bcftools:
            raise RuntimeError("bcftools not available")
        
        ref_path = Path(ref_fasta)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference FASTA not found: {ref_fasta}")
        
        out_dir = Path(output_dir or self.temp_dir / "consensus_fastas")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        samples = subprocess.run(['bcftools', 'query', '-l', str(filtered_vcf)], 
                               capture_output=True, text=True).stdout.strip().split('\n')
        
        if fasta_type == "multi":
            multi_fasta = out_dir / "all_samples_consensus.fasta"
            with open(multi_fasta, 'w') as multi_out:
                for sample in samples:
                    logger.info(f"Generating consensus FASTA for {sample}...")
                    cmd = ['bcftools', 'consensus', '-s', sample, '-f', str(ref_path), str(filtered_vcf)]
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    multi_out.write(result.stdout)
            logger.info(f"Multi-FASTA saved: {multi_fasta}")
            return out_dir
        
        else:  # Individual files
            for sample in samples:
                sample_fasta = out_dir / f"{sample}_consensus.fasta"
                logger.info(f"Generating consensus for {sample} → {sample_fasta}")
                cmd = ['bcftools', 'consensus', '-s', sample, '-f', str(ref_path), str(filtered_vcf)]
                with open(sample_fasta, 'w') as f:
                    subprocess.run(cmd, stdout=f, check=True)
            
            logger.info(f"Individual consensus FASTAs saved in: {out_dir}")
            return out_dir
    
    def _load_csv_matrix(self, path: Path, output_dir: Optional[str], 
                        label_column: Optional[str]) -> pd.DataFrame:
        """Load existing CSV/TSV matrix (original behavior + enhancements)."""
        sep = ',' if path.suffix.lower() == '.csv' else '\t'
        df = pd.read_csv(path, sep=sep, index_col=0)
        
        # Handle labels in matrix
        if label_column and label_column in df.columns:
            logger.info(f"Extracting labels from matrix column: {label_column}")
            # Will be handled in align_data()
        
        # Deduplicate + filter invariants
        duplicates = df.index.duplicated(keep=False)
        if duplicates.any():
            logger.warning(f"Removed {duplicates.sum()} duplicate samples")
            df = df[~df.index.duplicated(keep='first')]
        
        invariants = (df.nunique() <= 1).sum()
        if invariants > 0:
            logger.info(f"Removed {invariants} invariant features")
            df = df.loc[:, df.nunique() > 1]
        
        if output_dir:
            (Path(output_dir) / "deduplicated_genomic_matrix.csv").parent.mkdir(exist_ok=True, parents=True)
            df.to_csv(Path(output_dir) / "deduplicated_genomic_matrix.csv")
        
        return df
    
    def _vcf_python_fallback(self, vcf_path: Path, output_dir: Optional[str]) -> pd.DataFrame:
        """Pure Python VCF parser (slower, for when bcftools unavailable)."""
        logger.warning("🐌 Using slow Python VCF parser (install bcftools for 100x speedup)")
        data = {}
        samples = None
        
        with open(vcf_path) as f:
            for line in f:
                if line.startswith('#CHROM'):
                    samples = line.strip().split('\t')[9:]
                    for s in samples: data[s] = []
                    continue
                if line.startswith('#'): continue
                
                fields = line.strip().split('\t')
                pos_id = f"{fields[0]}:{fields[1]}"
                for i, gt_str in enumerate(fields[9:]):
                    gt = gt_str.split(':')[0]
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
"""
Data loading and preprocessing module.

This module provides utilities for loading and preprocessing genomic data
from various file formats, including CSV, TSV, FASTA (binary sequences),
and VCF files. It also handles loading optional metadata and known markers.
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and preprocessing of various input formats.
    
    This class uses static methods to provide utility functions without
    needing instantiation.
    """
    
    @staticmethod
    def load_genomic_matrix(filepath: str, output_dir: Optional[str] = None) -> pd.DataFrame:
        """Load genomic data matrix from various formats.
        
        Supported formats: CSV, TSV, FASTA (binary sequences), and VCF.
        Removes duplicate sample IDs, keeping the first occurrence, and saves
        the deduplicated matrix to output_dir if provided.
        
        Args:
            filepath: Path to the input file.
            output_dir: Directory to save deduplicated matrix (optional).
        
        Returns:
            A pandas DataFrame representing the genomic matrix.
        
        Raises:
            ValueError: If the file format is unsupported.
            FileNotFoundError: If the file does not exist.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if filepath.suffix.lower() == '.csv':
            df = pd.read_csv(filepath, index_col=0)
        elif filepath.suffix.lower() == '.tsv':
            df = pd.read_csv(filepath, sep='\t', index_col=0)
        elif filepath.suffix.lower() in ['.fasta', '.fa']:
            df = DataLoader._load_fasta_binary(filepath)
        elif filepath.suffix.lower() == '.vcf':
            df = DataLoader._load_vcf_binary(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        if df.index.duplicated().any():
            duplicates = df.index[df.index.duplicated()].tolist()
            logger.warning(f"Duplicate sample IDs found in genomic matrix: {duplicates}. Keeping first occurrence.")
            df = df[~df.index.duplicated(keep='first')]
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / "deduplicated_genomic_matrix.csv"
                df.to_csv(output_file)
                logger.info(f"Saved deduplicated genomic matrix to: {output_file}")
        
        return df
    
    @staticmethod
    def _load_fasta_binary(filepath: Path) -> pd.DataFrame:
        """Convert FASTA sequences to binary matrix.
        
        Assumes sequences are composed of '0' and '1' characters only.
        Removes duplicates and pads shorter sequences with '0'.
        
        Args:
            filepath: Path to the FASTA file.
        
        Returns:
            A pandas DataFrame with samples as rows and positions as columns.
        """
        sequences = {}
        with open(filepath, 'r') as f:
            current_id = None
            current_seq = ""
            for line in f:
                if line.startswith('>'):
                    if current_id:
                        sequences[current_id] = current_seq
                    current_id = line[1:].strip()
                    current_seq = ""
                else:
                    current_seq += line.strip()
            if current_id:
                sequences[current_id] = current_seq
        
        if not sequences:
            return pd.DataFrame()
        
        if len(sequences) != len(set(sequences.keys())):
            duplicates = [k for k in sequences.keys() if list(sequences.keys()).count(k) > 1]
            logger.warning(f"Duplicate sample IDs found in FASTA: {duplicates}. Keeping first occurrence.")
            unique_sequences = {}
            seen = set()
            for k, v in sequences.items():
                if k not in seen:
                    unique_sequences[k] = v
                    seen.add(k)
            sequences = unique_sequences
        
        max_length = max(len(seq) for seq in sequences.values())
        binary_matrix = []
        for sample_id, sequence in sequences.items():
            padded_seq = sequence.ljust(max_length, '0')
            try:
                binary_row = [int(char) for char in padded_seq]
            except ValueError:
                raise ValueError(f"Invalid character in sequence for {sample_id}. Sequences must consist of '0' and '1' only.")
            binary_matrix.append(binary_row)
        
        columns = [f"pos_{i}" for i in range(max_length)]
        return pd.DataFrame(binary_matrix, index=list(sequences.keys()), columns=columns)
    
    @staticmethod
    def _load_vcf_binary(filepath: Path) -> pd.DataFrame:
        """Convert VCF to binary matrix (simplified implementation).
        
        Converts genotypes to binary and removes duplicate sample IDs.
        
        Args:
            filepath: Path to the VCF file.
        
        Returns:
            A pandas DataFrame with samples as rows and variants as columns.
        """
        variants = []
        samples = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('##'):
                    continue
                elif line.startswith('#CHROM'):
                    samples = line.strip().split('\t')[9:]
                elif not line.startswith('#'):
                    fields = line.strip().split('\t')
                    chrom, pos, ref, alt = fields[0], fields[1], fields[3], fields[4]
                    genotypes = fields[9:]
                    binary_genotypes = []
                    for gt in genotypes:
                        gt_field = gt.split(':')[0]
                        if gt_field in ['0/0', '0|0']:
                            binary_genotypes.append(0)
                        elif gt_field in ['1/1', '1|1', '0/1', '1/0', '0|1', '1|0']:
                            binary_genotypes.append(1)
                        else:
                            binary_genotypes.append(0)
                    variants.append({
                        'id': f"{chrom}_{pos}_{ref}_{alt}",
                        'genotypes': binary_genotypes
                    })
        
        if not variants or not samples:
            return pd.DataFrame()
        
        if len(samples) != len(set(samples)):
            duplicates = [s for s in samples if samples.count(s) > 1]
            logger.warning(f"Duplicate sample IDs found in VCF: {duplicates}. Keeping first occurrence.")
            unique_samples = []
            seen = set()
            for s in samples:
                if s not in seen:
                    unique_samples.append(s)
                    seen.add(s)
            samples = unique_samples
            variants = [
                {
                    'id': v['id'],
                    'genotypes': [v['genotypes'][i] for i in range(len(v['genotypes'])) if samples[i] in unique_samples]
                } for v in variants
            ]
        
        matrix = pd.DataFrame(
            [variant['genotypes'] for variant in variants],
            columns=samples,
            index=[variant['id'] for variant in variants]
        )
        return matrix.T
    
    @staticmethod
    def load_metadata(filepath: Optional[str], output_dir: Optional[str] = None) -> pd.DataFrame:
        """Load sample metadata if provided.
        
        Assumes CSV format with sample IDs as index. Removes duplicate sample IDs
        and saves the deduplicated metadata to output_dir if provided.
        
        Args:
            filepath: Path to the metadata CSV file, or None if no metadata.
            output_dir: Directory to save deduplicated metadata (optional).
        
        Returns:
            A pandas DataFrame with metadata, or an empty DataFrame if filepath is None.
        """
        if filepath is None:
            return pd.DataFrame()
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath, index_col=0)
        
        if df.index.duplicated().any():
            duplicates = df.index[df.index.duplicated()].tolist()
            logger.warning(f"Duplicate sample IDs found in metadata: {duplicates}. Keeping first occurrence.")
            df = df[~df.index.duplicated(keep='first')]
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / "deduplicated_metadata.csv"
                df.to_csv(output_file)
                logger.info(f"Saved deduplicated metadata to: {output_file}")
        
        return df
    
    @staticmethod
    def load_known_markers(filepath: Optional[str], output_dir: Optional[str] = None) -> List[str]:
        """Load list of known markers if provided.
        
        Reads a text file with one marker per line and saves a copy to output_dir if provided.
        
        Args:
            filepath: Path to the markers file, or None if no markers.
            output_dir: Directory to save a copy of the markers file (optional).
        
        Returns:
            List of marker strings, or empty list if filepath is None.
        """
        if filepath is None:
            return []
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r') as f:
            markers = [line.strip() for line in f if line.strip()]
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "known_markers.txt"
            with open(output_file, 'w') as f:
                f.write('\n'.join(markers))
            logger.info(f"Saved known markers to: {output_file}")
        
        return markers
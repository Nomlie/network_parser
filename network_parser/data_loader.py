# network_parser/data_loader.py
"""
Data loading and preprocessing module.

This module provides utilities for loading and preprocessing genomic data
from various file formats, including CSV, TSV, FASTA (binary sequences),
and VCF files. It also handles loading metadata and known markers.
"""

import pandas as pd
from pathlib import Path
from typing import List

class DataLoader:
    """Handles loading and preprocessing of various input formats.
    
    This class uses static methods to provide utility functions without
    needing instantiation.
    """
    
    @staticmethod
    def load_genomic_matrix(filepath: str) -> pd.DataFrame:
        """Load genomic data matrix from various formats.
        
        Supported formats: CSV, TSV, FASTA (assuming binary 0/1 sequences),
        and VCF (simplified binary conversion).
        
        Args:
            filepath: Path to the input file.
        
        Returns:
            A pandas DataFrame representing the genomic matrix.
        
        Raises:
            ValueError: If the file format is unsupported.
            FileNotFoundError: If the file does not exist.
        """
        # Convert to Path object for consistent handling
        filepath = Path(filepath)
        
        # Check if file exists before proceeding
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Handle CSV format
        if filepath.suffix.lower() == '.csv':
            return pd.read_csv(filepath, index_col=0)
        
        # Handle TSV format
        elif filepath.suffix.lower() == '.tsv':
            return pd.read_csv(filepath, sep='\t', index_col=0)
        
        # Handle FASTA/FA formats (binary sequences)
        elif filepath.suffix.lower() in ['.fasta', '.fa']:
            return DataLoader._load_fasta_binary(filepath)
        
        # Handle VCF format
        elif filepath.suffix.lower() == '.vcf':
            return DataLoader._load_vcf_binary(filepath)
        
        # Raise error for unsupported formats
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    @staticmethod
    def _load_fasta_binary(filepath: Path) -> pd.DataFrame:
        """Convert FASTA sequences to binary matrix.
        
        Assumes sequences are composed of '0' and '1' characters only.
        Pads shorter sequences with '0' to match the longest sequence.
        
        Args:
            filepath: Path to the FASTA file.
        
        Returns:
            A pandas DataFrame with samples as rows and positions as columns.
        
        Raises:
            ValueError: If a character in the sequence cannot be converted to int (not '0' or '1').
        """
        # Dictionary to store sequence IDs and their sequences
        sequences = {}
        
        # Read the FASTA file line by line
        with open(filepath, 'r') as f:
            current_id = None
            current_seq = ""
            
            for line in f:
                # Header line starts a new sequence
                if line.startswith('>'):
                    if current_id:
                        sequences[current_id] = current_seq
                    current_id = line[1:].strip()
                    current_seq = ""
                else:
                    # Append sequence data, stripping whitespace
                    current_seq += line.strip()
            
            # Add the last sequence if present
            if current_id:
                sequences[current_id] = current_seq
        
        # If no sequences found, return empty DataFrame
        if not sequences:
            return pd.DataFrame()
        
        # Determine the maximum sequence length for padding
        max_length = max(len(seq) for seq in sequences.values())
        
        # List to hold binary rows
        binary_matrix = []
        
        for sample_id, sequence in sequences.items():
            # Pad sequence with '0' if shorter than max_length
            padded_seq = sequence.ljust(max_length, '0')
            
            try:
                # Convert each character to integer (0 or 1)
                binary_row = [int(char) for char in padded_seq]
            except ValueError:
                raise ValueError(f"Invalid character in sequence for {sample_id}. Sequences must consist of '0' and '1' only.")
            
            # Append the binary row
            binary_matrix.append(binary_row)
        
        # Create column names as position indices
        columns = [f"pos_{i}" for i in range(max_length)]
        
        # Construct and return the DataFrame
        return pd.DataFrame(binary_matrix, index=list(sequences.keys()), columns=columns)
    
    @staticmethod
    def _load_vcf_binary(filepath: Path) -> pd.DataFrame:
        """Convert VCF to binary matrix (simplified implementation).
        
        This is a basic parser that converts genotypes to binary:
        - 0 for reference homozygous (0/0, 0|0)
        - 1 for any variant (heterozygous or homozygous alt)
        - 0 for missing data
        
        Note: For production use, consider libraries like cyvcf2 or hail for
        more robust VCF handling. This ignores multi-allelic variants and
        complex genotypes.
        
        Args:
            filepath: Path to the VCF file.
        
        Returns:
            A pandas DataFrame with samples as rows and variants as columns.
        """
        # Lists to store variant data and sample names
        variants = []
        samples = []
        
        # Read the VCF file line by line
        with open(filepath, 'r') as f:
            for line in f:
                # Skip metadata lines
                if line.startswith('##'):
                    continue
                
                # Extract sample names from header
                elif line.startswith('#CHROM'):
                    samples = line.strip().split('\t')[9:]
                
                # Process variant lines
                elif not line.startswith('#'):
                    fields = line.strip().split('\t')
                    
                    # Extract key fields: chrom, pos, ref, alt
                    chrom, pos, ref, alt = fields[0], fields[1], fields[3], fields[4]
                    
                    # Genotypes start from column 9
                    genotypes = fields[9:]
                    
                    # Convert to binary
                    binary_genotypes = []
                    for gt in genotypes:
                        gt_field = gt.split(':')[0]  # Take the genotype field
                        if gt_field in ['0/0', '0|0']:
                            binary_genotypes.append(0)
                        elif gt_field in ['1/1', '1|1', '0/1', '1/0', '0|1', '1|0']:
                            binary_genotypes.append(1)
                        else:
                            binary_genotypes.append(0)  # Treat missing or other as 0
                    
                    # Store variant info
                    variants.append({
                        'id': f"{chrom}_{pos}_{ref}_{alt}",
                        'genotypes': binary_genotypes
                    })
        
        # If no variants or samples, return empty DataFrame
        if not variants or not samples:
            return pd.DataFrame()
        
        # Create DataFrame with variants as rows initially
        matrix = pd.DataFrame(
            [variant['genotypes'] for variant in variants],
            columns=samples,
            index=[variant['id'] for variant in variants]
        )
        
        # Transpose to have samples as rows, variants as columns
        return matrix.T
    
    @staticmethod
    def load_metadata(filepath: str) -> pd.DataFrame:
        """Load sample metadata.
        
        Assumes CSV format with sample IDs as index.
        
        Args:
            filepath: Path to the metadata CSV file.
        
        Returns:
            A pandas DataFrame with metadata.
        
        Raises:
            FileNotFoundError: If the file does not exist.
        """
        # Convert to Path
        filepath = Path(filepath)
        
        # Check existence
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Load CSV with index_col=0
        return pd.read_csv(filepath, index_col=0)
    
    @staticmethod
    def load_known_markers(filepath: str) -> List[str]:
        """Load list of known markers.
        
        Reads a text file with one marker per line, stripping whitespace
        and ignoring empty lines.
        
        Args:
            filepath: Path to the markers file.
        
        Returns:
            List of marker strings.
        
        Raises:
            FileNotFoundError: If the file does not exist.
        """
        # Convert to Path
        filepath = Path(filepath)
        
        # Check existence
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Read lines, strip, and filter empty
        with open(filepath, 'r') as f:
            return [line.strip() for line in f if line.strip()]
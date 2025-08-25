# network_parser/data_loader.py
"""
Data loading and preprocessing module.
"""

import pandas as pd
from pathlib import Path
from typing import List

class DataLoader:
    """Handles loading and preprocessing of various input formats"""
    
    @staticmethod
    def load_genomic_matrix(filepath: str) -> pd.DataFrame:
        """Load genomic data matrix from various formats"""
        filepath = Path(filepath)
        
        if filepath.suffix.lower() == '.csv':
            return pd.read_csv(filepath, index_col=0)
        elif filepath.suffix.lower() == '.tsv':
            return pd.read_csv(filepath, sep='\t', index_col=0)
        elif filepath.suffix.lower() in ['.fasta', '.fa']:
            return DataLoader._load_fasta_binary(filepath)
        elif filepath.suffix.lower() == '.vcf':
            return DataLoader._load_vcf_binary(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    @staticmethod
    def _load_fasta_binary(filepath: str) -> pd.DataFrame:
        """Convert FASTA sequences to binary matrix"""
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
        
        # Convert to binary matrix
        max_length = max(len(seq) for seq in sequences.values())
        binary_matrix = []
        
        for sample_id, sequence in sequences.items():
            # Pad sequence if necessary
            padded_seq = sequence.ljust(max_length, '0')
            binary_row = [int(char) for char in padded_seq]
            binary_matrix.append(binary_row)
        
        columns = [f"pos_{i}" for i in range(max_length)]
        return pd.DataFrame(binary_matrix, index=list(sequences.keys()), columns=columns)
    
    @staticmethod
    def _load_vcf_binary(filepath: str) -> pd.DataFrame:
        """Convert VCF to binary matrix (simplified implementation)"""
        # This is a simplified VCF parser - in practice you'd use libraries like cyvcf2
        variants = []
        samples = []
        
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('##'):
                    continue
                elif line.startswith('#CHROM'):
                    # Extract sample names
                    samples = line.strip().split('\t')[9:]
                elif not line.startswith('#'):
                    # Process variant line
                    fields = line.strip().split('\t')
                    chrom, pos, ref, alt = fields[0], fields[1], fields[3], fields[4]
                    genotypes = fields[9:]
                    
                    # Convert genotypes to binary (simplified)
                    binary_genotypes = []
                    for gt in genotypes:
                        gt_field = gt.split(':')[0]
                        if gt_field in ['0/0', '0|0']:
                            binary_genotypes.append(0)
                        elif gt_field in ['1/1', '1|1', '0/1', '1/0', '0|1', '1|0']:
                            binary_genotypes.append(1)
                        else:
                            binary_genotypes.append(0)  # Missing data as reference
                    
                    variants.append({
                        'id': f"{chrom}_{pos}_{ref}_{alt}",
                        'genotypes': binary_genotypes
                    })
        
        # Create DataFrame
        matrix = pd.DataFrame(
            [variant['genotypes'] for variant in variants],
            columns=samples,
            index=[variant['id'] for variant in variants]
        ).T  # Transpose to have samples as rows
        
        return matrix
    
    @staticmethod
    def load_metadata(filepath: str) -> pd.DataFrame:
        """Load sample metadata"""
        return pd.read_csv(filepath, index_col=0)
    
    @staticmethod
    def load_known_markers(filepath: str) -> List[str]:
        """Load list of known markers"""
        with open(filepath, 'r') as f:
            return [line.strip() for line in f if line.strip()]
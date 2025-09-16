import logging
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class DataLoader:
    """Class to handle loading and preprocessing of genomic data and metadata."""
    
    def load_genomic_matrix(self, file_path: str, output_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Load genomic data from various file formats.

        Args:
            file_path: Path to genomic data file (CSV, TSV, FASTA, VCF).
            output_dir: Optional directory to save processed data.

        Returns:
            Genomic data as a pandas DataFrame.
        """
        logger.info(f"Loading genomic matrix from: {file_path}")
        file_path = Path(file_path)
        
        if file_path.suffix.lower() in ['.csv', '.tsv']:
            sep = ',' if file_path.suffix.lower() == '.csv' else '\t'
            df = pd.read_csv(file_path, sep=sep, index_col=0)
        elif file_path.suffix.lower() == '.fasta':
            # Placeholder for FASTA parsing (simplified)
            logger.warning("FASTA parsing not fully implemented. Using dummy data.")
            df = pd.DataFrame()  # Replace with actual FASTA parsing logic
        elif file_path.suffix.lower() == '.vcf':
            # Placeholder for VCF parsing (simplified)
            logger.warning("VCF parsing not fully implemented. Using dummy data.")
            df = pd.DataFrame()  # Replace with actual VCF parsing logic
        else:
            logger.error(f"Unsupported file format: {file_path.suffix}")
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Check for duplicate sample IDs
        duplicates = df.index[df.index.duplicated()].tolist()
        if duplicates:
            logger.warning(f"Duplicate sample IDs found in genomic matrix: {duplicates}. Keeping first occurrence.")
            df = df.loc[~df.index.duplicated(keep='first')]
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "deduplicated_genomic_matrix.csv"
            df.to_csv(output_path)
            logger.info(f"Saved deduplicated genomic matrix to: {output_path}")
        
        return df
    
    def load_metadata(self, file_path: str, output_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Load metadata from CSV or TSV files.

        Args:
            file_path: Path to metadata file (CSV or TSV).
            output_dir: Optional directory to save processed metadata.

        Returns:
            Metadata as a pandas DataFrame.
        """
        logger.info(f"Loading metadata from: {file_path}")
        file_path = Path(file_path)
        
        if file_path.suffix.lower() in ['.csv', '.tsv']:
            sep = ',' if file_path.suffix.lower() == '.csv' else '\t'
            df = pd.read_csv(file_path, sep=sep, index_col=0)
        else:
            logger.error(f"Unsupported metadata file format: {file_path.suffix}")
            raise ValueError(f"Unsupported metadata file format: {file_path.suffix}")
        
        # Check for duplicate sample IDs
        duplicates = df.index[df.index.duplicated()].tolist()
        if duplicates:
            logger.warning(f"Duplicate sample IDs found in metadata: {duplicates}. Keeping first occurrence.")
            df = df.loc[~df.index.duplicated(keep='first')]
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "deduplicated_metadata.csv"
            df.to_csv(output_path)
            logger.info(f"Saved deduplicated metadata to: {output_path}")
        
        return df
    
    def load_known_markers(self, file_path: str, output_dir: Optional[str] = None) -> List[str]:
        """
        Load known genetic markers from a file.

        Args:
            file_path: Path to file containing known markers (TXT, CSV, TSV).
            output_dir: Optional directory to save processed markers.

        Returns:
            List of known marker names.
        """
        logger.info(f"Loading known markers from: {file_path}")
        file_path = Path(file_path)
        
        if file_path.suffix.lower() in ['.txt', '.csv', '.tsv']:
            with file_path.open('r') as f:
                markers = [line.strip() for line in f if line.strip()]
            
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / "processed_known_markers.txt"
                with output_path.open('w') as f:
                    f.write('\n'.join(markers))
                logger.info(f"Saved processed known markers to: {output_path}")
            
            return markers
        else:
            logger.error(f"Unsupported known markers file format: {file_path.suffix}")
            raise ValueError(f"Unsupported known markers file format: {file_path.suffix}")
    
    def align_data(self,
                   genomic_data: pd.DataFrame,
                   metadata: Optional[pd.DataFrame],
                   label_column: str,
                   output_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Align genomic data and metadata by sample IDs, extracting labels and removing NaN labels.

        Args:
            genomic_data: Genomic data DataFrame with samples as index.
            metadata: Metadata DataFrame with samples as index (optional).
            label_column: Column name for labels (in genomic_data or metadata).
            output_dir: Directory to save aligned data (optional).

        Returns:
            Tuple of (aligned genomic DataFrame, labels Series).
        """
        logger.info("Aligning genomic data and metadata...")
        
        if metadata is not None:
            # Find common samples
            common_samples = genomic_data.index.intersection(metadata.index)
            if len(common_samples) == 0:
                logger.error("No common samples found between genomic data and metadata.")
                raise ValueError("No common samples found between genomic data and metadata.")
            
            # Align data
            aligned_genomic = genomic_data.loc[common_samples].copy()
            aligned_metadata = metadata.loc[common_samples].copy()
            
            # Extract labels
            if label_column in aligned_metadata.columns:
                labels = aligned_metadata[label_column]
            elif label_column in aligned_genomic.columns:
                aligned_genomic = aligned_genomic.drop(columns=[label_column])
                labels = genomic_data.loc[common_samples, label_column]
            else:
                logger.error(f"Label column '{label_column}' not found in metadata or genomic data.")
                raise ValueError(f"Label column '{label_column}' not found.")
            
            # Remove rows with NaN in labels
            initial_samples = len(common_samples)
            non_na_mask = ~labels.isna()
            aligned_genomic = aligned_genomic.loc[non_na_mask].copy()
            labels = labels[non_na_mask].copy()
            samples_removed = initial_samples - len(labels)
            
            if samples_removed > 0:
                logger.info(f"Removed {samples_removed} samples with missing labels")
            
            if len(labels) == 0:
                logger.error("No valid data remaining after removing samples with missing labels")
                raise ValueError("No valid data after removing missing labels")
        else:
            # No metadata provided; labels must be in genomic_data
            if label_column not in genomic_data.columns:
                logger.error(f"Label column '{label_column}' not found in genomic data.")
                raise ValueError(f"Label column '{label_column}' not found.")
            aligned_genomic = genomic_data.drop(columns=[label_column]).copy()
            labels = genomic_data[label_column]
            
            # Remove rows with NaN in labels
            initial_samples = len(genomic_data)
            non_na_mask = ~labels.isna()
            aligned_genomic = aligned_genomic.loc[non_na_mask].copy()
            labels = labels[non_na_mask].copy()
            samples_removed = initial_samples - len(labels)
            
            if samples_removed > 0:
                logger.info(f"Removed {samples_removed} samples with missing labels")
            
            if len(labels) == 0:
                logger.error("No valid data remaining after removing samples with missing labels")
                raise ValueError("No valid data after removing missing labels")
        
        # Save aligned data if output_dir is provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            aligned_genomic.to_csv(output_dir / "aligned_genomic_matrix.csv")
            logger.info(f"Saved aligned genomic matrix to: {output_dir / 'aligned_genomic_matrix.csv'}")
            
            labels.to_csv(output_dir / "aligned_metadata.csv")
            logger.info(f"Saved aligned labels to: {output_dir / 'aligned_metadata.csv'}")
        
        logger.info(f"Aligned data: {len(labels)} samples retained.")
        return aligned_genomic, labels
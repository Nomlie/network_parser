# data_loader.py
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple


logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and preprocessing of genomic data and metadata."""

    def load_genomic_matrix(self, file_path: str, output_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Load genomic data from supported formats and deduplicate samples.
        """
        path = Path(file_path)
        logger.info(f"Loading genomic matrix from: {path}")

        if path.suffix.lower() in {'.csv', '.tsv'}:
            sep = ',' if path.suffix.lower() == '.csv' else '\t'
            df = pd.read_csv(path, sep=sep, index_col=0)
        elif path.suffix.lower() == '.fasta':
            raise NotImplementedError("FASTA parsing not implemented.")
        elif path.suffix.lower() == '.vcf':
            raise NotImplementedError("VCF parsing not implemented.")
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

        duplicates = df.index.duplicated(keep=False)
        if duplicates.any():
            logger.warning(f"Found {duplicates.sum()} duplicate sample IDs. Keeping first occurrence.")
            df = df[~df.index.duplicated(keep='first')]

        if output_dir:
            output_path = Path(output_dir) / "deduplicated_genomic_matrix.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path)
            logger.info(f"Saved deduplicated genomic matrix to: {output_path}")

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

    def align_data(self, genomic_data: pd.DataFrame, metadata: Optional[pd.DataFrame], label_column: str,
                   output_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Align genomic data and metadata by common samples, extract labels, and filter NaNs/invariants.
        If metadata is None, extracts labels directly from the genomic_data using label_column.

        Args:
            genomic_data (pd.DataFrame): Genomic matrix to align.
            metadata (Optional[pd.DataFrame]): Metadata DataFrame (default: None). If None, labels are
                expected in genomic_data.
            label_column (str): Name of the column containing labels.
            output_dir (Optional[str]): Directory to save aligned data (default: None).

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Aligned genomic data and corresponding labels.

        Raises:
            ValueError: If no common samples (with metadata) or label column is missing.
        """
        logger.info("Aligning genomic data and metadata...")

        if metadata is not None:
            common_samples = genomic_data.index.intersection(metadata.index)
            if common_samples.empty:
                raise ValueError("No common samples between genomic data and metadata.")
            aligned_genomic = genomic_data.loc[common_samples]
            aligned_metadata = metadata.loc[common_samples]
            if label_column in aligned_metadata.columns:
                labels = aligned_metadata[label_column]
            else:
                raise ValueError(f"Label column '{label_column}' not found in metadata.")
        else:
            # If no metadata, extract labels from genomic_data
            if label_column not in genomic_data.columns:
                raise ValueError(f"Label column '{label_column}' not found in genomic data.")
            aligned_genomic = genomic_data.drop(columns=[label_column])  # Remove label column from features
            labels = genomic_data[label_column]

        # Filter out samples with missing labels
        non_na_mask = ~labels.isna()
        aligned_genomic = aligned_genomic.loc[non_na_mask]
        labels = labels[non_na_mask]

        # Remove features that are invariant (same value across all samples)
        invariants = aligned_genomic.nunique() <= 1
        if invariants.any():
            logger.info(f"Removing {invariants.sum()} invariant features.")
            aligned_genomic = aligned_genomic.loc[:, ~invariants]

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            aligned_genomic.to_csv(output_path / "aligned_genomic_matrix.csv")
            labels.to_csv(output_path / "aligned_metadata.csv")
            logger.info(f"Saved aligned data to: {output_path}")

        logger.info(f"Aligned data: {len(labels)} samples, {aligned_genomic.shape[1]} features retained.")
        return aligned_genomic, labels
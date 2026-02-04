# NetworkParser: Interpretable Genomic Feature Discovery Framework

NetworkParser is a scalable, modular Python pipeline designed for identifying statistically validated genomic markers and epistatic interactions that drive phenotype segregation (e.g., antimicrobial resistance or lineage diversification) in microbial genomes. It bridges machine learning, statistics, and hierarchical genomics to provide transparent, reproducible analysis of complex datasets.

---

## Core Innovation

**Goal**  
Transform raw genomic data (e.g., SNP matrices or VCF files) into interpretable insights, including ranked features, interaction networks, and ML-ready outputs (e.g., Graph Neural Network matrices).

**Impact**  
Enables rapid analysis from raw data to biological hypotheses in minutes, with full interpretability. Ideal for AMR surveillance, virulence studies, or evolutionary biology.

**Key Features**
- Handles large, noisy datasets with rigorous validation to minimize false positives.
- Detects hierarchical patterns (root vs. branch features) and epistatic interactions.
- End-to-end workflow:  
  Data loading → Feature discovery → Statistical validation → Network integration → Outputs.

---

## Challenges Addressed

**Problem 1**  
Overwhelm from noisy data (e.g., thousands of SNPs).

**Problem 2**  
Lack of interpretability in black-box ML tools.

**Problem 3**  
No integrated framework for discovery, validation, and integration.

---

## Table of Contents

- Installation  
- Quick Start  
- Usage  
  - CLI Usage  
  - Programmatic Usage  
- Pipeline Stages  
- Configuration  
- Input Formats  
- Output Formats  
- Examples  
- Scripts Overview  
- Troubleshooting  
- Contributing  
- License  
- Contact  

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Required libraries (install via `pip`):
pandas
numpy
scikit-learn
scipy
statsmodels
networkx
joblib
tqdm
biopython
rdkit
pyscf
matplotlib

Additional tools for VCF processing:
- `bcftools` and `tabix`

Install via conda:
```bash
conda install -c bioconda bcftools tabix
or via your system package manager.

Clone and Install
Clone the repository:

git clone https://github.com/Nomlie/network_parser.git
cd network_parser
Install as a Python package (recommended for CLI access):

pip install -e .
This installs network_parser as a module and makes the CLI available.

Verify installation:

python -m network_parser.cli --help
Quick Start
Run a basic analysis on example data:

python -m network_parser.cli \
  --genomic input/example.csv \
  --label Group \
  --output-dir results/
This processes a genomic matrix, uses Group as the phenotype label, and saves results to results/.

For a full log:

python -m network_parser.cli [args] 2>&1 | tee pipeline_run.log
Usage
CLI Usage
The primary entry point is cli.py, which orchestrates the full pipeline.

python -m network_parser.cli [options]
Required Arguments
--genomic: Path to genomic input (CSV/TSV matrix, VCF file, or directory of VCFs).

--label: Metadata column to use as phenotype (e.g., Lineage or AMR).

--output_dir: Directory for results (e.g., results/).

Optional Arguments
--meta: Metadata CSV/TSV with sample IDs and labels.

--known_markers: Path to known markers file.

--regions: bcftools regions/targets (e.g., NC_000962.3:1-1000000 or BED file).

--ref_fasta: Reference FASTA for consensus generation.

--config: JSON/YAML config file to override defaults.

--validate_statistics: Enable pre-tree association testing + multiple testing correction.

--validate_interactions: Enable post-tree interaction permutation validation.

Full help:

python -m network_parser.cli --help
Programmatic Usage
Import and run the pipeline in your Python scripts:

from network_parser.network_parser import run_networkparser_analysis
from network_parser.config import NetworkParserConfig

config = NetworkParserConfig()  # Customize as needed

results = run_networkparser_analysis(
    genomic_path="input/example.csv",
    meta_path="input/metadata.csv",
    label_column="Group",
    output_dir="results/",
    config=config,
    validate_statistics=True,
    validate_interactions=True
)

print(results)  # Dict with tree_results, stats_results, etc.
Pipeline Stages
NetworkParser follows a sequential workflow:

Input Processing (data_loader.py)
Load and preprocess genomic data (CSV/VCF) and metadata. Align samples, filter invariants/duplicates, and optionally restrict regions.

Pattern Discovery (decision_tree_builder.py)
Apply Chi-squared/Fisher’s tests with FDR correction. Build decision trees to identify root/branch features and epistatic interactions.

Statistical Validation (statistical_validation.py)
Bootstrap resampling for feature stability; permutation tests for interactions.

Feature Integration (network_parser.py)
Rank features and build interaction graphs using NetworkX.

Output Generation
Export JSON reports, GraphML graphs, and GNN-ready matrices.

Runtime scales with data size (e.g., ~12s for 15 samples / 89 features).

Configuration
Customize behavior via config.py or --config in the CLI.

Key parameters include:

significance_level: Alpha for tests (default: 0.05)

n_bootstrap_samples: Bootstrap iterations (default: 1000)

n_permutation_tests: Permutation tests (default: 1000)

multiple_testing_method: FDR method (e.g., fdr_bh)

max_depth: Decision tree max depth (default: 5)

min_group_size: Minimum samples per group (default: 5)

Load from YAML/JSON:

analysis:
  bootstrap_iterations: 500
  fdr_threshold: 0.01

processing:
  max_workers: 4
  memory_efficient: true
Input Formats
Genomic Data
CSV/TSV: Binary matrix (samples × variants, 0/1 for REF/ALT).

VCF: Single file or directory (supports multi-allelic splitting and filtering via bcftools).

FASTA: Limited support; converts to matrix.

Metadata
CSV/TSV with sample IDs (first column) and phenotype labels.

Known Markers
Optional CSV for feature prioritization.

Regions
bcftools-compatible region string or BED file.

Output Formats
Saved to --output_dir:

Matrices

deduplicated_genomic_matrix.csv

aligned_genomic_matrix.csv

Discovery / Validation

decision_tree_rules.txt

feature_confidence.json

bootstrap_results.json

interaction_permutation_results.json

Networks

network_graph.graphml (for Cytoscape)

Integrated Results

networkparser_results_*.json (features, interactions, GNN matrices)

Examples
Basic CSV Analysis
python -m network_parser.cli \
  --genomic input/example.csv \
  --meta input/metadata.csv \
  --label Group \
  --output_dir results/ \
  --validate_statistics \
  --validate_interactions
VCF Directory with Regions
python -m network_parser.cli \
  --genomic path/to/vcf_dir \
  --regions "NC_000962.3:1-1000000" \
  --ref_fasta reference.fasta \
  --meta metadata.csv \
  --label AMR \
  --output_dir results_vcf/
Subset Extraction (Using extract_subset.py)
Extract 100 random samples:

python scripts/extract_subset.py \
  --vcf-dir path/to/vcfs \
  --meta-file metadata.csv \
  --output-dir subset_vcfs/ \
  --n-samples 100
Stratified by phenotype:

--stratify-by phenotype
For more examples, see the presentation slides or run --example in extract_subset.py.

Scripts Overview
network_parser.py: Main pipeline orchestrator

cli.py: Command-line interface

data_loader.py: Data ingestion and preprocessing

decision_tree_builder.py: Feature discovery via decision trees

statistical_validation.py: Statistical tests and validation

extract_subset.py: Utility for sample subsetting

utils.py: Configuration loading utilities

Troubleshooting
VCF errors: Ensure bcftools and tabix are installed and on PATH.

Memory issues: Enable memory_efficient: true in config for large datasets.

No significant features: Check logs for filtering; relax fdr_threshold.

Alignment failures: Verify sample IDs match between genomic data and metadata.

Logs: Check pipeline.log in the working directory.

If issues persist, open a GitHub issue with your command, logs, and data summary.

Contributing
Contributions welcome. Fork the repository, create a branch, and submit a PR.
Follow PEP8 style, add tests in tests/, and update documentation as needed.

License
MIT License. See LICENSE for details.

Contact
Nomlindelo Mfuphi
Email: nomlindelow@gmail.com
GitHub: https://github.com/Nomlie/network_parser
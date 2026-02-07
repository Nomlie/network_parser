# NetworkParser  
**Interpretable Genomic Feature Discovery Framework**

NetworkParser is a scalable, modular Python pipeline designed for identifying statistically validated genomic markers and epistatic interactions that drive phenotype segregation (e.g., antimicrobial resistance, lineage diversification) in microbial genomes.

It combines machine learning, rigorous statistical validation, and hierarchical analysis to deliver transparent and reproducible insights from complex genomic datasets.

## Core Innovation

**Goal**  
Transform raw genomic data (SNP matrices, VCF files, etc.) into interpretable biological insights: ranked features, interaction networks, and ML-ready outputs (e.g. Graph Neural Network matrices).

**Impact**  
Enables rapid analysis — from raw data to testable biological hypotheses — in minutes, with full interpretability.  
Particularly valuable for:

- Antimicrobial resistance (AMR) surveillance  
- Virulence factor discovery  
- Microbial evolutionary studies  

**Key Features**

- Handles large, noisy datasets with strong false-positive control  
- Detects hierarchical feature importance (root vs. branch features)  
- Identifies epistatic (non-additive) interactions  
- End-to-end workflow:  
  Data loading → Feature discovery → Statistical validation → Network integration → Output generation  

**Challenges Addressed**

- Overwhelm from noisy, high-dimensional genomic data (thousands of SNPs)  
- Lack of interpretability in black-box machine learning models  
- Absence of integrated pipelines that combine discovery, rigorous validation, and network interpretation  

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Pipeline Stages](#pipeline-stages)
- [Configuration](#configuration)
- [Input Formats](#input-formats)
- [Output Files](#output-files)
- [Examples](#examples)
- [Scripts Overview](#scripts-overview)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

NetworkParser uses a Conda environment for full dependency reproducibility.

### Requirements

- Conda (Miniconda or Anaconda)
- Git
- Python 3.8+

---

### Create Environment

From the root of the repository:

```bash
git clone https://github.com/Nomlie/network_parser.git
cd network_parser
conda env create -f networkparser.yaml
conda activate networkparser
```

**Install NetworkParser**
```bash
pip install .
```

**Verification**
```bash 
python -m network_parser.cli --help
```
If successful, this will display the CLI usage.

## Quick Start
Command-line Example
```bash
conda activate networkparser
```

```bash
python -m network_parser.cli \
  --genomic input/example.csv \
  --label Group \
  --output-dir results/
```

## Usage
**Required Arguments**

* --genomic: Genomic input (CSV/TSV matrix, VCF file, or directory of VCFs).
* --label: Phenotype or group label column in metadata.
* --output-dir: Directory for output files.

**Optional Arguments**

* --meta: Metadata CSV/TSV file (with sample IDs and labels).
* --regions: Regions/targets for VCF processing (e.g., "NC_000962.3:1-1000000" or BED file).
* --ref-fasta: Reference FASTA for consensus generation and normalization.
* --config: JSON config file to override defaults.
* --validate-statistics: Enable pre-tree statistical validation.
* --validate-interactions: Enable post-tree interaction validation.

For full options, run --help.
**Output**
The output-dir will contain:

* Ranked genomic features
* Statistical validation tables
* Epistatic interaction networks
* ML-ready matrices (e.g., for GNNs)

## Pipeline Stages

1. Input Processing (data_loader.py): Loads, normalizes, filters, and aligns genomic features and metadata.
2. Feature Discovery (decision_tree_builder.py): Identifies discriminative features and hierarchical patterns using constrained decision trees.
3. Statistical Validation (statistical_validation.py): Applies bootstrap resampling, permutation tests, and multiple-testing correction for robust inference.
4. Network Integration (network_parser.py): Builds feature–feature and sample–feature interaction networks.
5. Output Generation: Exports interpretable reports, JSON, GraphML, and matrices.

## Configuration
Pipeline behavior is controlled via a centralized configuration in config.py, overrideable through CLI arguments or a JSON/YAML file (via --config).
**Example CLI Configuration**

```bash
python -m network_parser.cli \
  --genomic data/genomic_matrix.csv \
  --label Phenotype \
  --output-dir results/ \
  --max-depth 6 \
  --min-group-size 5 \
  --significance-level 0.05 \
  --multiple-testing-method fdr_bh \
  --n-bootstrap 1000 \
  --n-permutations 500 \
  --min-information-gain 0.001 \
  --n-jobs -1 \
  --random-state 42
```
For YAML/JSON config, see utils.py for loading details.
## Input Formats

* Genomic Data: CSV/TSV (binary matrix), VCF(.gz), or directory of VCFs.
* Metadata: CSV/TSV with sample IDs (first column) and label column.
* Known Markers (optional): CSV list for guided analysis.
* Reference FASTA (optional): For VCF consensus and normalization.

## Output Files

* deduplicated_genomic_matrix.csv: Cleaned matrix.
* decision_tree_rules.txt: Tree structure.
* feature_confidence.json: Confidence scores.
* epistatic_interactions.json: Detected interactions.
* network_graph.graphml: Interaction graph (for Cytoscape).
* networkparser_results_*.json: Comprehensive results.

## Examples

**CSV Matrix Analysis**
```bash
 python -m network_parser.cli \
  --genomic input/example.csv \
  --meta input/metadata.csv \
  --label Group \
  --output-dir results/ \
  --validate-statistics \
  --validate-interactions
```

**VCF Directory with Regions**

```bash
python -m network_parser.cli \
  --genomic data/vcfs/ \
  --regions "NC_000962.3:1-1000000" \
  --ref-fasta ref/MTB_H37Rv.fasta \
  --meta metadata.csv \
  --label AMR \
  --output-dir results_amr/
```

```bash
Subset Extraction Script
```bash ./scripts/extract_subset.py \
  --vcf-dir data/vcfs/ \
  --meta-file metadata.csv \
  --output-dir subsets/ \
  --n-samples 100
```

## Scripts Overview

network_parser.py: Main orchestrator.
cli.py: Command-line interface.
config.py: Configuration classes.
data_loader.py: Data ingestion and preprocessing.
decision_tree_builder.py: Feature discovery via trees.
statistical_validation.py: Validation suite.
extract_subset.py: Utility for VCF subsetting.
utils.py: Helper functions (e.g., config loading).

## Troubleshooting

bcftools/tabix missing: Install via conda install -c bioconda bcftools tabix.
High memory usage: Set --memory-efficient in config.
No features discovered: Relax thresholds (e.g., --significance-level 0.1).
Sample ID mismatch: Verify metadata first column matches VCF sample IDs.
VCF indexing errors: Ensure .tbi files exist; run tabix -p vcf file.vcf.gz.
Slow performance: Increase --n-jobs or reduce --n-bootstrap.

For logs, check pipeline.log in the working directory.

## Troubleshooting

bcftools missing → install via conda  
Memory high → memory_efficient true  
No features → relax thresholds  
ID mismatch → verify metadata  
VCF index → ensure .tbi exists  

## Contributing
Fork the repository.
Create a feature branch (git checkout -b feature/AmazingFeature).
Commit changes (git commit -m 'Add some AmazingFeature').
Push to the branch (git push origin feature/AmazingFeature).
Open a Pull Request.

Follow PEP8 style guidelines. Include tests for new features.

## License
MIT License. See LICENSE for details.

## Contact
Nomlindelo Mfuphi
Email: nomlindelow@gmail.com
GitHub: https://github.com/Nomlie/network_parser

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

## Challenges Addressed

- Overwhelm from noisy, high-dimensional genomic data (thousands of SNPs)
- Lack of interpretability in black-box machine learning models
- Absence of integrated pipelines that combine discovery, rigorous validation, and network interpretation

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [CLI Usage](#cli-usage)
  - [Programmatic Usage](#programmatic-usage)
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

### Prerequisites

- Python 3.8+
- Core dependencies:
  ```bash
  pandas numpy scikit-learn scipy statsmodels networkx joblib tqdm

Optional / visualization:Bashmatplotlib biopython
VCF processing (strongly recommended):Bashconda install -c bioconda bcftools tabix

Installation Steps
Bash# Clone repository
git clone https://github.com/Nomlie/network_parser.git
cd network_parser

# Install as editable package (recommended)
pip install -e .

# Verify CLI works
python -m network_parser.cli --help
Quick Start
Basic run on example CSV matrix:
Bashpython -m network_parser.cli \
  --genomic input/example.csv \
  --label Group \
  --output-dir results/
With full logging:
Bashpython -m network_parser.cli [your arguments] 2>&1 | tee pipeline_run.log
Usage
CLI Usage
Bashpython -m network_parser.cli [options]
Required arguments
text--genomic       Path to input (CSV/TSV matrix, single VCF, or folder of VCFs)
--label         Phenotype column name in metadata (e.g. Lineage, AMR, Group)
--output-dir    Output directory for results
Important optional arguments
text--meta                  Metadata CSV/TSV file
--regions               bcftools region string or BED file
--ref-fasta             Reference FASTA (enables consensus calling)
--config                JSON/YAML configuration file
--validate-statistics   Run association testing + FDR correction
--validate-interactions Run permutation testing for epistasis
See full options:
Bashpython -m network_parser.cli --help
Programmatic Usage
Pythonfrom network_parser.network_parser import run_networkparser_analysis
from network_parser.config import NetworkParserConfig

config = NetworkParserConfig()
# Customize e.g.:
# config.n_bootstrap_samples = 2000
# config.significance_level = 0.01

results = run_networkparser_analysis(
    genomic_path="input/example.csv",
    meta_path="input/metadata.csv",
    label_column="Group",
    output_dir="results/",
    config=config,
    validate_statistics=True,
    validate_interactions=True
)

print(results.keys())
# dict_keys(['tree_results', 'stats_results', 'interaction_results'])
Pipeline Stages

Input Processing (data_loader.py)
Load → align → clean → filter invariants/duplicates
Pattern Discovery (decision_tree_builder.py)
Chi-squared / Fisher → FDR → Decision Tree → root/branch features + epistasis mining
Statistical Validation (statistical_validation.py)
Bootstrap stability + permutation testing for interactions
Feature Integration & Graph Building (network_parser.py)
Rank features → build sample-feature & interaction graphs (NetworkX)
Output Generation
JSON, GraphML, GNN-ready matrices, rules text

Configuration
Default parameters are defined in config.py. Override via:

Command line flags (partial support)
JSON/YAML config file passed with --config

Example YAML snippet:
YAMLanalysis:
  bootstrap_iterations: 500
  fdr_threshold: 0.01
  significance_level: 0.05

processing:
  max_workers: 4
  memory_efficient: true

validation:
  min_bootstrap_support: 0.7
Input Formats

Genomic data
CSV/TSV binary matrix (samples × variants, 0/1)
VCF(.gz) single file or directory
(limited) FASTA multiple sequence alignment

Metadata
CSV/TSV with sample IDs in first column + phenotype column


Output Files
Created in --output-dir:

deduplicated_genomic_matrix.csv
aligned_genomic_matrix.csv
decision_tree_rules.txt
feature_confidence.json
bootstrap_results.json
interaction_permutation_results.json (when enabled)
network_graph.graphml (Cytoscape compatible)
networkparser_results_*.json (full structured results)

Examples
Basic CSV run
Bashpython -m network_parser.cli \
  --genomic input/example.csv \
  --meta input/metadata.csv \
  --label Group \
  --output-dir results/ \
  --validate-statistics \
  --validate-interactions
VCF folder + region filter
Bashpython -m network_parser.cli \
  --genomic data/vcfs/ \
  --regions "NC_000962.3:1-1000000" \
  --ref-fasta ref/MTB_H37Rv.fasta \
  --meta metadata.csv \
  --label AMR \
  --output-dir results_amr/
Create 100-sample test subset
Bashpython scripts/extract_subset.py \
  --vcf-dir path/to/vcfs \
  --meta-file metadata.csv \
  --output-dir subset_100_vcfs \
  --n-samples 100
Scripts Overview

network_parser.py     – main pipeline orchestrator
cli.py                – command line interface
data_loader.py        – VCF/CSV loading & preprocessing
decision_tree_builder.py – core feature & interaction discovery
statistical_validation.py – bootstrap & permutation tests
extract_subset.py     – helper: create balanced test subsets
utils.py              – configuration helpers

Troubleshooting





























IssuePossible Fix / Checkbcftools/tabix not foundInstall via conda or system package managerMemory usage too highSet memory_efficient: true in configNo significant featuresRelax --fdr-threshold or check phenotype balanceSamples not aligningVerify sample IDs match exactly between matrix/VCF and metadataVCF indexing errorsMake sure .tbi files exist or can be created
Always check pipeline.log in the working directory.
Contributing
Contributions are welcome!

Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit changes (git commit -m 'Add amazing feature')
Push to branch (git push origin feature/amazing-feature)
Open a Pull Request

Please follow PEP 8 style and add/update tests when possible.
License
MIT License
See LICENSE file for full text.
Contact
Nomlindelo Mfuphi
📧 nomlindelow@gmail.com
🐙 https://github.com/Nomlie/network_parser
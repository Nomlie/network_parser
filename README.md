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

---

### Create Environment

From the root of the repository:

```bash
git clone https://github.com/Nomlie/network_parser.git
cd network_parser
```
```bash
conda env create -f networkparser.yaml
```

```bash
conda activate networkparser
```
**Install NetworkParser**
```bash
pip install .
```

**Verification**
```bash python -m network_parser.cli --help
```
If successful, this will display the CLI usage.

Quick Start
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

Output
The results/ directory will contain:

Ranked genomic features

Statistical validation tables

Epistatic interaction networks

ML-ready matrices
**Execution with logging**

```bash python -m network_parser.cli [args] 2>&1 | tee pipeline_run.log ```

## Usage
Required arguments

--genomic – Genomic input matrix or VCF

--label – Phenotype or group label column

--output-dir – Output directory

Optional arguments

--meta

--regions

--ref-fasta

--config

--validate-statistics

--validate-interactions

**Programmatic Interface**
```bash 
from network_parser.network_parser import run_networkparser_analysis
from network_parser.config import NetworkParserConfig

config = NetworkParserConfig()

results = run_networkparser_analysis(
    genomic_path="input/example.csv",
    meta_path="input/metadata.csv",
    label_column="Group",
    output_dir="results/",
    config=config,
    validate_statistics=True,
    validate_interactions=True
)
```

## Pipeline Stages

Input Processing (data_loader.py)
Loading, normalization, filtering, and alignment of genomic features and metadata.

Feature Discovery (decision_tree_builder.py)
Identification of discriminative genomic features and hierarchical patterns using constrained decision trees.

Statistical Validation (statistical_validation.py)
Robust inference via bootstrap resampling and permutation-based testing with multiple-testing correction.

Network Integration (network_parser.py)
Construction of feature–feature and sample–feature interaction networks.

Output Generation
Export of interpretable artefacts and network representations.

Configuration

Pipeline behavior is controlled via a centralized configuration object, configurable through CLI arguments or JSON/YAML files.

**Example configuration**
analysis:
  bootstrap_iterations: 500
  fdr_threshold: 0.01

processing:
  max_workers: 4

validation:
  min_bootstrap_support: 0.7

Input Formats
Genomic inputs

CSV / TSV binary feature matrices

VCF (.vcf, .vcf.gz)

FASTA (limited support)

Metadata

CSV / TSV files containing sample identifiers and phenotype or group labels

## Output Files

Representative outputs include:

deduplicated_genomic_matrix.csv

aligned_genomic_matrix.csv

decision_tree_rules.txt

feature_confidence.json

bootstrap_results.json

interaction_permutation_results.json

network_graph.graphml

## Examples

CSV:

  ```bash python -m network_parser.cli \
  --genomic input/example.csv \
  --meta input/metadata.csv \
  --label Group \
  --output-dir results/ \
  --validate-statistics \
  --validate-interactions 
```

VCF:

  ```bash python -m network_parser.cli \
  --genomic data/vcfs/ \
  --regions "NC_000962.3:1-1000000" \
  --ref-fasta ref/MTB_H37Rv.fasta \
  --meta metadata.csv \
  --label AMR \
  --output-dir results_amr/ 
```

## Scripts

  ```bash network_parser.py  
cli.py  
data_loader.py  
decision_tree_builder.py  
statistical_validation.py  
extract_subset.py  
utils.py
```

## Troubleshooting

bcftools missing → install via conda  
Memory high → memory_efficient true  
No features → relax thresholds  
ID mismatch → verify metadata  
VCF index → ensure .tbi exists  

## Contributing

Fork → branch → commit → PR

Follow PEP8.

## License

MIT

## Contact

Nomlindelo Mfuphi  
https://github.com/Nomlie/network_parser


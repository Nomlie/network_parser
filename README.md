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

- Installation  
- Quick Start  
- Usage  
- Pipeline Stages  
- Configuration  
- Input Formats  
- Output Files  
- Examples  
- Scripts Overview  
- Troubleshooting  
- Contributing  
- License  
- Contact  

## Installation

### Prerequisites

- Python 3.8+

Core dependencies:

pandas numpy scikit-learn scipy statsmodels networkx joblib tqdm

VCF processing (recommended):

conda install -c bioconda bcftools tabix

Installation:

git clone https://github.com/Nomlie/network_parser.git  
cd network_parser  
pip install -e .  

Verify:

python -m network_parser.cli --help

## Quick Start

python -m network_parser.cli \
  --genomic input/example.csv \
  --label Group \
  --output-dir results/

With logging:

python -m network_parser.cli [args] 2>&1 | tee pipeline_run.log

## Usage

Required:

--genomic  
--label  
--output-dir  

Optional:

--meta  
--regions  
--ref-fasta  
--config  
--validate-statistics  
--validate-interactions  

## Programmatic

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

## Pipeline Stages

Input Processing – data_loader.py  
Pattern Discovery – decision_tree_builder.py  
Statistical Validation – statistical_validation.py  
Graph Building – network_parser.py  
Output Generation  

## Configuration

Override defaults via JSON/YAML or CLI.

Example:

analysis:
  bootstrap_iterations: 500
  fdr_threshold: 0.01

processing:
  max_workers: 4

validation:
  min_bootstrap_support: 0.7

## Input Formats

Genomic:
- CSV/TSV binary matrix  
- VCF (.gz)  
- FASTA (limited)

Metadata:
CSV/TSV with sample IDs + phenotype

## Output Files

deduplicated_genomic_matrix.csv  
aligned_genomic_matrix.csv  
decision_tree_rules.txt  
feature_confidence.json  
bootstrap_results.json  
interaction_permutation_results.json  
network_graph.graphml  

## Examples

CSV:

python -m network_parser.cli \
  --genomic input/example.csv \
  --meta input/metadata.csv \
  --label Group \
  --output-dir results/ \
  --validate-statistics \
  --validate-interactions

VCF:

  ```bashpython -m network_parser.cli \
  --genomic data/vcfs/ \
  --regions "NC_000962.3:1-1000000" \
  --ref-fasta ref/MTB_H37Rv.fasta \
  --meta metadata.csv \
  --label AMR \
  --output-dir results_amr/```

## Scripts

network_parser.py  
cli.py  
data_loader.py  
decision_tree_builder.py  
statistical_validation.py  
extract_subset.py  
utils.py  

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
nomlindelow@gmail.com  
https://github.com/Nomlie/network_parser

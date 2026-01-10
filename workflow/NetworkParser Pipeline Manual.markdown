# NetworkParser Pipeline Manual

## Overview

**NetworkParser** is a scalable, modular, and interpretable bioinformatics pipeline designed for microbial genomic analysis. It identifies statistically validated genomic markers and epistatic interactions driving phenotypic segregation (e.g., antimicrobial resistance, lineage diversification) while simultaneously producing phylogenetic-ready outputs.

The pipeline processes variant call format (VCF) files and phenotypic metadata to generate clean binary matrices, decision tree-based feature rankings, statistically robust validation, interaction networks, and consensus pseudogenomes suitable for phylogenetic reconstruction.

### Key Features 

- Native support for VCF(.gz) input with high-quality biallelic SNP/indel filtering (using bcftools)
- Generation of consensus pseudogenome FASTA files (`bcftools consensus`)
- Clean binary SNP matrix optimized for machine learning and epistasis analysis
- Interpretable decision tree-based feature discovery distinguishing root vs. branch markers
- Rigorous statistical validation including bootstrap resampling and permutation testing
- Rich network outputs: sample-feature bipartite graphs + epistatic interaction graphs (GraphML format + GNN-ready matrices)
- End-to-end reproducibility through conda-based environment

### Pipeline Stages

1. **Input Processing**  
   Loading, quality filtering, variant normalization, binary matrix creation, and consensus sequence generation

2. **Feature Discovery**  
   Statistical association testing + decision tree construction for marker discovery and epistasis detection

3. **Statistical Validation**  
   Bootstrap-based stability assessment + permutation testing for interaction significance

4. **Integration & Network Construction**  
   Feature ranking, bipartite sample-feature networks, and epistatic interaction graphs

5. **Output Generation**  
   Structured reports, GraphML networks, GNN-compatible matrices, and phylogenetic-ready FASTA files
## Quick Start Example — *Mycobacterium tuberculosis* Lineage Analysis

```bash
# Recommended: activate dedicated conda environment first
conda activate networkparser-env
`````

```bash
python -m network_parser.cli \
  --genomic data/tb_isolates.vcf.gz \
  --ref-fasta reference/H37Rv.fasta \
  --label Lineage \
  --output-dir results_tb_2026/ \
  --n-jobs -1 \
  --n-bootstrap 1000 \
  --n-permutations 500
`````

## Main outputs produced in results_tb_2026/
File / Directory	Description
genomic_matrix.csv	Clean binary SNP matrix (ML / epistasis ready)
filtered_snps.final.vcf.gz	High-quality filtered variant calls
consensus_fastas/*.fasta or all_samples_consensus.fasta	Individual or concatenated consensus pseudogenomes
sample_feature_network.graphml	Bipartite sample–feature network (visualise in Cytoscape)
interaction_graph.graphml	Graph of significant epistatic interactions
gnn_matrices.npz	GNN-ready adjacency / feature / label matrices
networkparser_results_*.json	Complete feature discovery, validation, and ranking report
pipeline.log	Detailed timestamped execution log

## Follow-up phylogenetic analysis example
```bash
iqtree2 \
  -s results_tb_2026/consensus_fastas/all_samples_consensus.fasta \
  -m GTR \
  -bb 1000 \
  -nt AUTO \
  --prefix tb_lineage_iqtree
  `````

## Scripts and Their Functions
**1. cli.py** 
### Purpose
Serves as the command-line interface to initiate the network_parser pipeline.

### Functionality
- Parses command-line arguments (e.g. --genomic, --label, --output-dir)
- Initializes NetworkParserConfig with parameters such as max_depth,min_group_size, significance_level, n_bootstrap_samples, n_permutation_tests, multiple_testing_method, min_information_gain, n_jobs, and random_state
- Calls network_parser.py to execute all pipeline stages
- Logs pipeline progress and errors

### Inputs
- --genomic: Path to genomic input (VCF or derived CSV)
- --label: Name of the label column (e.g. Lineage)
- --output-dir: Output directory (e.g. results/)

### Outputs
- Coordinates all output files written to output-dir
- Log file: pipeline_run.log

### Role in Pipeline
- Entry point that validates inputs and dispatches execution

**2. config.py**
### Purpose
- Defines the configuration object used across the pipeline.

### Functionality

- Implements the NetworkParserConfig class

Sets default parameters:

```bash
max_depth=None
min_group_size=5
significance_level=0.05
n_bootstrap_samples=1000
n_permutation_tests=500
multiple_testing_method='fdr_bh'
min_information_gain=0.001
n_jobs=-1
random_state=42
````

Enables consistent parameter sharing across modules

### Inputs

- Parsed CLI arguments (or configuration file if supported)

### Outputs

- A NetworkParserConfig instance

### Role in Pipeline
- Centralizes and standardizes configuration

### Dependencies

- Standard Python libraries (dataclasses, typing)

**3. data_loader.py**
### Purpose
Handles modern microbial genomics input processing (Stage 1: Input Processing)

### Functionality

Native loading of compressed VCF (.vcf.gz)
High-quality variant filtering:
biallelic SNPs/indels
quality thresholds
missingness filtering
Generation of clean binary SNP matrix (0/1/NA)
Consensus pseudogenome construction using bcftools consensus
Optional concatenation of samples into a multi-FASTA for phylogenetics
Sample deduplication and alignment of features and labels
Removal of invariant sites
Preservation of intermediate artefacts for traceability

### Inputs

VCF file (.vcf.gz)
Reference FASTA
Label column name

### Output directory

- genomic_matrix.csv
- filtered_snps.final.vcf.gz
- consensus_fastas/*.fasta or all_samples_consensus.fasta
- deduplicated_genomic_matrix.csv
- aligned_genomic_matrix.csv
- aligned_metadata.csv

### Role in Pipeline
- Converts raw variant calls into analysis-ready matrices and phylogenetic inputs with full reproducibility

**4. decision_tree_builder.py**
### Purpose
- Feature discovery and rule induction (Stage 2: Feature Discovery)

### Functionality

- Performs association testing (chi-squared / Fisher’s exact)
- Applies multiple testing correction (e.g. FDR-BH)
- Filters significant features
- Builds a constrained decision tree:
- max_depth
- min_group_size
- min_information_gain
- Identifies dominant features and candidate epistatic interactions
- Writes interpretable artefacts

### Inputs

- Aligned genomic matrix
- Labels
- Configuration parameters
- Outputs
- decision_tree_rules.txt
- feature_confidence.json
- epistatic_interactions.json

### Role in Pipeline

- Identifies discriminative variants and models their hierarchical structure


**5. statistical_validator.py**
### Purpose
- Rigorous statistical validation (Stage 3: Statistical Validation)

### Functionality

- Association testing with effect sizes and information metrics
- Multiple testing correction
- Bootstrap stability analysis (default: 1000 resamples)
- Permutation testing for epistatic interactions (default: 500 permutations)
- Parallel execution via joblib

### Inputs

- Aligned data and labels
- Discovered features and interactions
- Configuration parameters

### Outputs
- chi_squared_results.json
- multiple_testing_results.json
- bootstrap_results.json
- interaction_permutation_results.json

### Role in Pipeline

- Quantifies robustness, stability, and statistical support

**6. network_parser.py**
### Purpose
- Pipeline orchestration and integration (Stage 4: Integration)

### Functionality

- Coordinates Stages 1–3
- Integrates discovery and validation outputs
- Ranks features by confidence and stability
- Constructs feature–interaction networks using networkx
- Computes network statistics (degree, clustering, centrality)
- Writes final integrated reports

### Inputs

- Configuration
- Preprocessed data
- Discovery and validation results

### Outputs
- network_graph.graphml
- networkparser_results_*.json

### Role in Pipeline

- Final synthesis layer producing interpretable, network-aware results


## Pipeline Workflow
- cli.py — parse arguments and initialize configuration
- network_parser.py — orchestrate execution
- data_loader.py — preprocess genomic inputs
- decision_tree_builder.py — discover discriminative features
- statistical_validator.py — validate robustness and significance
- network_parser.py — integrate results and build networks

Output Files
- deduplicated_genomic_matrix.csv
- aligned_genomic_matrix.csv
- aligned_metadata.csv
- chi_squared_results.json
- multiple_testing_results.json
- decision_tree_rules.txt
- feature_confidence.json
- epistatic_interactions.json
- bootstrap_results.json
- interaction_permutation_results.json
- network_graph.graphml
- networkparser_results_*.json

pipeline_run.log

**Troubleshooting**
- Ensure dependencies are installed:
- Verify input formats and label consistency
- Inspect pipeline_run.log for detailed diagnostics
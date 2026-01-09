# NetworkParser

A next-generation, scalable, and interpretable bioinformatics framework for microbial genomic analysis. It identifies statistically validated genomic markers and epistatic interactions driving phenotypic or phylogenetic cluster segregation (e.g., antimicrobial resistance, virulence adaptation, lineage diversification) while producing phylogenetic-ready outputs and machine-learning compatible formats.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/networkparser.svg)](https://pypi.org/project/networkparser/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/[your-username]/networkparser/actions)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)  <!-- Update with real DOI when available -->

---

## Key Features (January 2026)

- Native support for **VCF(.gz)** input with high-quality biallelic SNP/indel filtering (powered by bcftools)
- Generation of **consensus pseudogenome FASTA** files (`bcftools consensus`) for phylogenetic reconstruction
- Clean **binary SNP matrix** optimized for epistasis analysis and machine learning
- Interpretable **decision tree-based** feature discovery (distinguishing root/global vs. branch/context-specific markers)
- Rigorous **statistical validation** (bootstrap resampling, permutation tests, chi-squared/Fisher’s exact tests, FDR correction)
- Rich network outputs: sample-feature **bipartite graphs** + **epistatic interaction graphs** (GraphML + GNN-ready matrices in .npz format)
- End-to-end reproducibility via **conda** environment and detailed logging

---

## Table of Contents

- [Overview](#overview)
- [Purpose](#purpose)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Input Data Formats](#input-data-formats)
- [Command Line Options](#command-line-options)
- [Output Files](#output-files)
- [Project Structure](#project-structure)
- [Example Analysis](#example-analysis)
- [Advanced Configuration](#advanced-configuration)
- [Methods](#methods)
- [Comparison to Existing Tools](#comparison-to-existing-tools)
- [Benchmarks and Performance](#benchmarks-and-performance)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)

---

## Overview

NetworkParser is designed to decode the evolutionary drivers of complex microbial phenotypes by combining interpretable machine learning, rigorous statistical validation, and phylogenetic integration. It transforms variant data into ranked markers, interaction networks, and ready-to-use phylogenetic inputs — making it particularly valuable for antimicrobial resistance (AMR), virulence, and lineage studies.

---

## Purpose

NetworkParser enables researchers to:

- Discover statistically robust **diagnostic and lineage-specific markers**
- Model **epistatic interactions** underlying non-additive phenotypic effects
- Validate findings with **bootstrap stability** and **permutation-based significance**
- Perform **hierarchical analysis** across global (root) and context-specific (branch) features
- Generate **ML-ready outputs** (binary matrices, bipartite networks, GNN adjacency matrices)
- Produce **phylogenetic-ready consensus sequences** for downstream tree inference (e.g., IQ-TREE)

---

## How It Works

### 1. Input Processing
Loads VCF(.gz) variant calls, applies quality filtering (biallelic, missingness, quality thresholds), generates a clean binary SNP matrix, and produces consensus pseudogenomes against a reference FASTA.

### 2. Pattern Discovery
Applies chi-squared/Fisher’s exact tests (with FDR correction) → builds a decision tree → identifies root/branch features → detects epistatic interactions via tree-path and mutual information analysis.

### 3. Statistical Validation
Performs bootstrap resampling (default: 1000 iterations) for feature stability/confidence intervals, permutation testing (default: 500 iterations) for interaction significance.

### 4. Integration & Output Generation
Ranks features by combined effect size + stability, constructs sample-feature and interaction networks (GraphML), exports GNN-ready matrices, and produces phylogenetic FASTA files.

---

## Installation

```bash
conda env create -f environment.yml
conda activate networkparser
````
See environment.yml for exact dependency versions (pandas, numpy, scikit-learn, networkx, joblib, bcftools, etc.).

## Quick Start – Mycobacterium tuberculosis Lineage Analysis
```bash 
python -m network_parser.cli \
  --genomic  data/tb_isolates.vcf.gz \
  --ref-fasta reference/H37Rv.fasta \
  --label    Lineage \
  --output-dir results_tb_2026/ \
  --n-jobs   -4 \
  --n-bootstrap 1000 \
  --n-permutations 500
````

## Main outputs (in results_tb_2026/):

- genomic_matrix.csv — Clean binary SNP matrix (ML/epistasis ready)
- filtered_snps.final.vcf.gz — High-quality filtered variants
- consensus_fastas/*.fasta or all_samples_consensus.fasta — Pseudogenomes for phylogeny
- sample_feature_network.graphml — Bipartite sample–feature network (visualize in Cytoscape)
- interaction_graph.graphml — Epistatic interaction graph
- ignn_matrices.npz — GNN-ready adjacency/feature/label matrices
- networkparser_results_*.json — Complete discovery + validation report
- pipeline.log — Detailed execution log


### Follow-up phylogeny:
```bash
iqtree2 -s results_tb_2026/consensus_fastas/all_samples_consensus.fasta \
        -m GTR \
        -bb 1000 \
        -nt AUTO \
        --prefix tb_lineage_iqtree
````

### Input Data Formats

- Primary input — Multi-sample VCF(.gz) (biallelic SNPs/indels preferred)
- Reference FASTA — Required for consensus sequence generation
- Label file — CSV/TSV with sample IDs and phenotypic/lineage labels (or specified column in metadata)


### Command Line Options
```bash
Run python -m network_parser.cli --help
```` 
to see the full list.

### Key flags:
```bash
--genomic — Path to VCF(.gz)
--ref-fasta — Reference genome FASTA
--label — Phenotype/lineage column name
--output-dir — Output directory
--n-bootstrap / --n-permutations — Validation iterations
--min-quality / --max-missing — VCF filtering parameters
````

---

## Example Console Summary

```bash
cd /Users/nmfuphicsir.co.za/Documents/pHDProject/Code/network_parser
python -m network_parser.cli --genomic /Users/nmfuphicsir.co.za/Documents/pHDProject/Code/MatrixSelector/input/example.csv --label Group --output-dir results/ 2>&1 | tee pipeline_run.log
2025-09-25 11:58:54,742 - INFO - Attempting to import run_networkparser_analysis and NetworkParserConfig
2025-09-25 11:58:54,743 - INFO - Running cli.py from: /Users/nmfuphicsir.co.za/Documents/pHDProject/Code/network_parser/network_parser/cli.py
2025-09-25 11:58:54,743 - INFO - Starting NetworkParser pipeline
2025-09-25 11:58:54,743 - INFO - Genomic data: /Users/nmfuphicsir.co.za/Documents/pHDProject/Code/MatrixSelector/input/example.csv
2025-09-25 11:58:54,743 - INFO - Label column: Group
2025-09-25 11:58:54,743 - INFO - Output directory: /Users/nmfuphicsir.co.za/Documents/pHDProject/Code/network_parser/results
2025-09-25 11:58:54,743 - INFO - Initializing NetworkParser with provided configuration.
2025-09-25 11:58:54,743 - INFO - Initializing NetworkParser with config: {'max_depth': None, 'min_group_size': 5, 'significance_level': 0.05, 'n_bootstrap_samples': 1000, 'n_permutation_tests': 500, 'multiple_testing_method': 'fdr_bh', 'min_information_gain': 0.001, 'n_jobs': -1, 'random_state': 42}
2025-09-25 11:58:54,743 - INFO - Initialized StatisticalValidator.
2025-09-25 11:58:54,743 - INFO - Starting pipeline execution...
2025-09-25 11:58:54,743 - INFO - 📥 Stage 1: Input Processing
2025-09-25 11:58:54,743 - INFO - Loading genomic data from: /Users/nmfuphicsir.co.za/Documents/pHDProject/Code/MatrixSelector/input/example.csv
2025-09-25 11:58:54,743 - INFO - Loading genomic matrix from: /Users/nmfuphicsir.co.za/Documents/pHDProject/Code/MatrixSelector/input/example.csv
2025-09-25 11:58:54,778 - INFO - Saved deduplicated genomic matrix to: results/deduplicated_genomic_matrix.csv
2025-09-25 11:58:54,779 - INFO - Aligning data...
2025-09-25 11:58:54,779 - INFO - Aligning genomic data and metadata...
2025-09-25 11:58:54,790 - INFO - Removing 67 invariant features.
2025-09-25 11:58:54,795 - INFO - Saved aligned data to: results
2025-09-25 11:58:54,795 - INFO - Aligned data: 15 samples, 22 features retained.
2025-09-25 11:58:54,795 - INFO - Aligned data: 15 samples, 22 features retained.
2025-09-25 11:58:54,795 - INFO - 🌳 Stage 2: Feature Discovery
2025-09-25 11:58:54,795 - INFO - Running association tests for 22 features.
2025-09-25 11:59:04,558 - INFO - Saved chi-squared results to: results/
2025-09-25 11:59:04,558 - INFO - Applying multiple testing correction using fdr_bh.
2025-09-25 11:59:04,562 - INFO - Saved multiple testing results to: results/
2025-09-25 11:59:04,562 - INFO - Filtered 7 significant features.
2025-09-25 11:59:04,562 - INFO - 🔍 Feature discovery: 15 samples, 7 features
2025-09-25 11:59:04,563 - INFO - Data columns: ['497957', '679712', '771128', '912181', '1290472', '1392623', '1470751', '1551618', '1572071', '1678339', '1901101', '1973797', '2640767', '2860050', '2909481', '3089432', '3136337', '3219764', '3349839', '3371639', '3375761', '3397479']
2025-09-25 11:59:04,563 - INFO - Input features: ['679712', '771128', '1392623', '1470751', '1572071', '1678339', '3219764']
2025-09-25 11:59:04,637 - INFO - Prefiltered 7/7 features via FDR
2025-09-25 11:59:04,637 - INFO - Prefiltered features: ['679712', '771128', '1392623', '1470751', '1572071', '1678339', '3219764']
2025-09-25 11:59:04,638 - INFO - 📊 2 classes: ['N', 'P']
2025-09-25 11:59:04,654 - INFO - Built decision tree with depth 1 and 2 leaves.
2025-09-25 11:59:04,736 - INFO - ✅ Stage 3: Statistical Validation
2025-09-25 11:59:04,736 - INFO - Running bootstrap validation with 1000 samples.
2025-09-25 11:59:05,952 - INFO - Saved bootstrap results to: results/
2025-09-25 11:59:05,953 - INFO - 🔗 Stage 4: Integration
2025-09-25 11:59:05,953 - INFO - Ranked 1 features by confidence.
2025-09-25 11:59:06,622 - INFO - Saved network graphs to GraphML files.
2025-09-25 11:59:06,623 - INFO - Saved iGNN adjacency matrices.
2025-09-25 11:59:06,623 - INFO - 📤 Stage 5: Output Generation
2025-09-25 11:59:06,624 - INFO - Saved final results to: results/networkparser_results_20250925_115906.json
2025-09-25 11:59:06,624 - INFO - ✅ Pipeline completed successfully
2025-09-25 11:59:06,624 - INFO - NetworkParser pipeline completed successfully in 11.88 seconds

======================================================================
🎯 FEATURE DISCOVERY SUMMARY (Workflow Stage 2)
======================================================================
📈 Tree Accuracy: 1.000 | Classes: 2
🌳 Root Features: 1 | Branch: 0
  1. 3219764 (conf: 0.841)
🔗 Epistatic Interactions: 0
======================================================================

```
---

## Directory Structure
```bash
network_parser/
├── network_parser/
│   ├── __init__.py          # Package version (0.1.0)
│   ├── cli.py               # Command-line interface
│   ├── config.py            # Configuration settings
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── decision_tree_builder.py  # Feature discovery and interactions
│   ├── network_parser.py    # Pipeline orchestration
│   ├── statistical_validation.py  # Statistical tests
├── data/
│   ├── matrix.csv           # Genomic data (samples × features)
│   ├── labels.csv           # Metadata (sample IDs + labels)
├── results/
│   ├── deduplicated_genomic_matrix.csv
│   ├── deduplicated_metadata.csv
│   ├── aligned_genomic_matrix.csv
│   ├── aligned_metadata.csv
│   ├── decision_tree_rules.txt
│   ├── feature_confidence.json
│   ├── epistatic_interactions.json
│   ├── bootstrap_results.json
│   ├── chi_squared_results.json
│   ├── multiple_testing_results.json
│   ├── interaction_permutation_results.json
│   ├── feature_set_validation.json
│   ├── networkparser_results_YYYYMMDD_HHMMSS.json

```
---

### Advanced Configuration
Customize significance thresholds, tree depth, minimum group size, and multiple-testing method via command-line flags or a future config file.


### Benchmarks and Performance

### Troubleshooting

Ensure bcftools is in PATH
VCF must be indexed (bgzip + tabix)
Check pipeline.log for detailed errors


### Citation

### Contributing
Contributions are welcome! Please submit pull requests with clear descriptions.

### License
MIT License – see LICENSE

### Support
Open an issue on GitHub or contact the maintainers.

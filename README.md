# NetworkParser  
A next-generation bioinformatics framework for identifying statistically validated genomic features that drive **cluster** segregation using interpretable machine learning, epistatic interaction modeling, and hierarchical analysis.

---

## Key Features
- **Data Preprocessing**: Loads, deduplicates, and aligns genomic and metadata files.  
- **Feature Discovery**: Uses decision trees to identify discriminative features, classifying them as root (global) or branch (context-specific) features.  
- **Epistatic Interaction Detection**: Identifies interactions between genetic features that jointly influence classification.  
- **Statistical Validation**: Applies bootstrap validation, chi-squared tests, multiple testing correction, permutation tests, and feature set validation to ensure robust and significant results.  
- **Comprehensive Outputs**: Generates detailed reports, including decision tree rules, feature confidence scores, and statistical validation results, saved as CSV and JSON files.  

## Badges
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)  
[![PyPI](https://img.shields.io/pypi/v/networkparser.svg)](https://pypi.org/project/networkparser/)  
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/username/networkparser/actions)  
[![Downloads](https://img.shields.io/pypi/dm/networkparser.svg)](https://pypi.org/project/networkparser/)  

---

## Table of Contents
- [Overview](#overview)  
- [Key Features](#key-features)  
- [Dependencies](#dependencies)  
- [Quick Start](#quick-start)  
- [Installation](#installation)  
- [Basic Usage](#basic-usage)  
- [Input Data Formats](#input-data-formats)  
- [Command Line Options](#command-line-options)  
- [Output](#output)  
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
NetworkParser is a next-generation genomic intelligence framework that decodes evolutionary processes underlying traits such as antimicrobial resistance emergence, virulence adaptation, and lineage diversification. By identifying statistically validated genetic drivers of cluster segregation—ranging from single polymorphisms to higher-order epistatic interactions—it transforms complex genomic variation into interpretable, actionable insights.

---

## Purpose
NetworkParser enables researchers to:

- **Discover Diagnostic Markers**: Identify genetic features specific to phenotypes or lineages.  
- **Model Epistatic Interactions**: Capture non-linear interactions between features.  
- **Ensure Statistical Rigor**: Validate findings using bootstrap resampling, chi-squared tests, and FDR correction.  
- **Support Hierarchical Analysis**: Analyze features across multiple biological levels (e.g., phylogenetic or phenotypic).  
- **Generate ML-Ready Outputs**: Produce feature matrices and interaction graphs for modern ML frameworks like GNNs.  
- **Handle Diverse Data**: Process binary-encoded datasets (SNPs, pan-genomes, motifs, or metadata), handles hierarchical or phenotypic labels.  

---

## How It Works
**1. Data Loading & Preprocessing**

Purpose: Loads and prepares genomic and metadata files for analysis, ensuring data consistency.

Inputs:

data/matrix.csv: Genomic matrix (rows: samples, columns: features like SNPs or variants).
data/labels.csv: Metadata with sample IDs and a label column (e.g., phenotypes, lineages).

Steps:
1. Loads data using pandas.
2. Removes duplicate sample IDs (e.g., 31_YP37_SZ, keeping the first occurrence).
3.  Aligns genomic and metadata files to ensure matching samples (e.g., 23 samples with 89 features).
Saves preprocessed files to results/:

- deduplicated_genomic_matrix.csv
- deduplicated_metadata.csv
- aligned_genomic_matrix.csv
- aligned_metadata.csv

Details:

- Handles hierarchical or phenotypic labels.
- Ensures robust data alignment for downstream analysis.

**2. Feature Discovery**

Purpose: Identifies discriminative features using decision trees and detects epistatic interactions.

Process:
- Decision Trees: Employs sklearn.tree.DecisionTreeClassifier to recursively partition data, identifying features that best separate classes (e.g., 11 labels: IP2666pIB1, MANG, MKUM, etc.).
- Root Features: Major discriminative features at low tree depths.
- Branch Features: Context-specific features revealing conditional dependencies.
- Epistatic Interactions: Detects feature pairs with synergistic effects by analyzing tree paths.
- Confidence Scores: Computes feature importance using mutual information and bootstrap stability.

Outputs (saved to results/):

- decision_tree_rules.txt: Text representation of the decision tree.
- feature_confidence.json: Confidence scores for root and branch features.
- epistatic_interactions.json: Feature pairs with interaction strengths and sample counts.

**3. Statistical Validation**

Purpose: Validates discovered features and interactions using rigorous statistical methods.

Methods:

Bootstrap Resampling (1000 iterations):
- Tests feature stability and computes confidence intervals.
- Saves: bootstrap_results.json.


Chi-Squared/Fisher’s Exact Tests:
- Assesses feature-label associations, using Fisher’s exact test for sparse data.
- Calculates effect sizes (Cramér’s V) and mutual information.
- Saves: chi_squared_results.json.


Multiple Testing Correction:
- Applies FDR correction (Benjamini-Hochberg, default α=0.05).
- Saves: multiple_testing_results.json.

Permutation Tests (500 iterations):

- Validates epistatic interactions against a null distribution.
- Saves: interaction_permutation_results.json.

Feature Set Validation:

- Compares discovered features against random baselines and individual features.
- Saves: feature_set_validation.json.

**4. Feature Integration & Outputs**

Purpose: Compiles results into interpretable and ML-ready formats.

Outputs:

- Feature Rankings: Lists features with effect sizes and confidence intervals.
- Interaction Graphs: Represents sample–feature networks for visualization or GNNs.
- Binary-Encoded Matrices: Provides data for ML models (e.g., GNNs, transformers).

Summary Reports:

- Human-readable console summary (tree accuracy, significant features, interactions).
- Structured JSON: networkparser_results_YYYYMMDD_HHMMSS.json.

---

## Example Console Summary

```bash
python -m network_parser.cli --genomic data/matrix.csv --meta data/labels.csv --label label --output-dir results/
2025-09-09 15:13:37,354 - INFO - Attempting to import run_networkparser_analysis and NetworkParserConfig
2025-09-09 15:13:38,148 - INFO - NumExpr defaulting to 8 threads.
2025-09-09 15:13:40,279 - INFO - Successfully imported run_networkparser_analysis and NetworkParserConfig
2025-09-09 15:13:40,279 - INFO - Running cli.py from: /Users/nmfuphicsir.co.za/Documents/pHDProject/Code/network_parser/network_parser/cli.py
2025-09-09 15:13:40,280 - INFO - Starting NetworkParser pipeline
2025-09-09 15:13:40,280 - INFO - Genomic data: data/matrix.csv
2025-09-09 15:13:40,280 - INFO - Metadata: data/labels.csv
2025-09-09 15:13:40,280 - INFO - Label column: label
2025-09-09 15:13:40,280 - INFO - Output directory: results/
2025-09-09 15:13:40,281 - INFO - Running network_parser.py from: /Users/nmfuphicsir.co.za/Documents/pHDProject/Code/network_parser/network_parser/network_parser.py
Initialized StatisticalValidator with provided configuration.
2025-09-09 15:13:40,281 - INFO - Initialized NetworkParser with provided configuration.
2025-09-09 15:13:40,281 - INFO - Loading genomic data...
2025-09-09 15:13:40,281 - INFO - Loading genomic matrix from: data/matrix.csv
2025-09-09 15:13:40,346 - INFO - Saved deduplicated genomic matrix to: results/deduplicated_genomic_matrix.csv
2025-09-09 15:13:40,346 - INFO - Loading metadata...
2025-09-09 15:13:40,346 - INFO - Loading metadata from: data/labels.csv
2025-09-09 15:13:40,350 - WARNING - Duplicate sample IDs found in metadata: ['31_YP37_SZ']. Keeping first occurrence.
2025-09-09 15:13:40,352 - INFO - Saved deduplicated metadata to: results/deduplicated_metadata.csv
2025-09-09 15:13:40,352 - INFO - Aligning data...
2025-09-09 15:13:40,352 - INFO - Aligning genomic data and metadata...
2025-09-09 15:13:40,356 - INFO - Saved aligned genomic matrix to: results/aligned_genomic_matrix.csv
2025-09-09 15:13:40,359 - INFO - Saved aligned metadata to: results/aligned_metadata.csv
2025-09-09 15:13:40,359 - INFO - Aligned data: 23 samples retained.
2025-09-09 15:13:40,359 - INFO - Starting feature discovery on 23 samples with 89 features...
🔍 Starting feature discovery on 23 samples with 89 features...
📊 Found 11 distinct labels: ['IP2666pIB1', 'MANG', 'MKUM', 'NP', 'PAK', 'PB', 'PrU', 'SZ', 'TLH', 'UST', 'VU']
Initialized StatisticalValidator with provided configuration.
🔄 Running bootstrap validation with 1000 samples...
   Bootstrap sample 1/1000
   Bootstrap sample 101/1000
   Bootstrap sample 201/1000
   Bootstrap sample 301/1000
   Bootstrap sample 401/1000
   Bootstrap sample 501/1000
   Bootstrap sample 601/1000
   Bootstrap sample 701/1000
   Bootstrap sample 801/1000
   Bootstrap sample 901/1000
Saved bootstrap results to: results/bootstrap_results.json
🧮 Running chi-squared tests for 89 features...
Saved chi-squared results to: results/chi_squared_results.json
🔧 Applying multiple testing correction using fdr_bh...
Saved multiple testing results to: results/multiple_testing_results.json
Saved decision tree rules to: results/decision_tree_rules.txt
Saved feature confidence scores to: results/feature_confidence.json
Saved epistatic interactions to: results/epistatic_interactions.json

============================================================
🎯 FEATURE DISCOVERY SUMMARY
============================================================
📈 Tree Accuracy: 0.783
🏷️  Label Classes: |--- 0.10 <= 0.50

🌳 ROOT FEATURES (Global Discriminators): 7
  1. 0.30 (confidence: 0.337)
  2. 0.10 (confidence: 0.363)
  3. 0.15 (confidence: 0.262)
  4. 0.32 (confidence: 0.390)
  5. 0.17 (confidence: 0.469)

🔗 EPISTATIC INTERACTIONS: 2
  0.30 → 0.6 (strength: 0.811)
  0.10 → 0.30 (strength: 0.414)

✅ STATISTICAL VALIDATION:
  Significant features after correction: 0
============================================================

🔄 Running bootstrap validation with 1000 samples...
   Bootstrap sample 1/1000
   Bootstrap sample 101/1000
   Bootstrap sample 201/1000
   Bootstrap sample 301/1000
   Bootstrap sample 401/1000
   Bootstrap sample 501/1000
   Bootstrap sample 601/1000
   Bootstrap sample 701/1000
   Bootstrap sample 801/1000
   Bootstrap sample 901/1000
Saved bootstrap results to: results/bootstrap_results.json
🧮 Running chi-squared tests for 89 features...
Saved chi-squared results to: results/chi_squared_results.json
🔧 Applying multiple testing correction using fdr_bh...
Saved multiple testing results to: results/multiple_testing_results.json
2025-09-09 15:13:45,582 - INFO - Saved decision tree features to: results/decision_tree_features.json
2025-09-09 15:13:45,666 - INFO - Saved final results to: results/networkparser_results_20250909_151345.json
2025-09-09 15:13:45,667 - INFO - NetworkParser pipeline completed successfully

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

## Installation
### Option 1: Conda Environment (recommended)
```bash
git clone https://github.com/Nomlie/network_parser.git
cd network_parser
conda env create -f environment.yml
conda activate network_parser
```

### Option 2: Pip Installation

```bash
git clone https://github.com/Nomlie/network_parser.git
cd network_parser
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
pip install -r requirements.txt
```
Set PYTHONPATH
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```
Add this line to your shell configuration (~/.bashrc or ~/.zshrc) if you want it permanent.

```bash
python -m network_parser -h
```

Example run:

```bash
python -m network_parser.cli  \
  --genomic data/matrix.csv \
  --meta data/labels.csv \
  --label phenotype \
  --output results/
```

## Command-Line Options:
```bash
--genomic: Path to genomic matrix (e.g., data/matrix.csv).
--meta: Path to metadata (e.g., data/labels.csv).
--label: Label column name (e.g., label).
--output-dir: Output directory (e.g., results/).
```

- Config File: Supports YAML/JSON for reproducibility (defined in config.py).
- Scalability: Multi-threaded execution for large datasets.

Outputs will be saved in the results/ directory.

---
**Analysis Modes**
- Hierarchical Mode: Analyzes phylogenetic or lineage-based contexts.
- Phenotype Mode: Focuses on metadata-driven comparisons (e.g., disease vs. healthy).

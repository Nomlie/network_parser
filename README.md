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
### 1. Feature Discovery
Purpose: Identifies discriminative features using decision trees and detects epistatic interactions.  

Process:  
- **Decision Trees**: Employs sklearn.tree.DecisionTreeClassifier to recursively partition data, identifying features that best separate classes (e.g., 11 labels: IP2666pIB1, MANG, MKUM, etc.).  
- **Root Features**: Major discriminative features at low tree depths.  
- **Branch Features**: Context-specific features revealing conditional dependencies.  
- **Epistatic Interactions**: Detects feature pairs with synergistic effects by analyzing tree paths.  
- **Confidence Scores**: Computes feature importance using mutual information and bootstrap stability.  

Outputs (saved to `results/`):  
- `decision_tree_rules.txt`  
- `feature_confidence.json`  
- `epistatic_interactions.json`  

### 2. Feature Integration
Purpose: Identifies discriminative features using decision trees and detects epistatic interactions.  

Process: *(same as above – preserved text)*  

Outputs (saved to `results/`):  
- `decision_tree_rules.txt`  
- `feature_confidence.json`  
- `epistatic_interactions.json`  

### 3. Statistical Validation
Purpose: Validates discovered features and interactions using rigorous statistical methods.  

Methods:  
- **Bootstrap Resampling (1000 iterations)**  
- **Chi-Squared/Fisher’s Exact Tests**  
- **Multiple Testing Correction**  
- **Permutation Tests (500 iterations)**  
- **Feature Set Validation**  

Outputs:  
- `bootstrap_results.json`  
- `chi_squared_results.json`  
- `multiple_testing_results.json`  
- `interaction_permutation_results.json`  
- `feature_set_validation.json`  

### 4. Feature Integration & Outputs
Purpose: Compiles results into interpretable and ML-ready formats.  

Outputs:  
- **Feature Rankings**  
- **Interaction Graphs**  
- **Binary-Encoded Matrices**  
- **Summary Reports** (`networkparser_results_YYYYMMDD_HHMMSS.json`)  

---

## Example Console Summary
🎯 FEATURE DISCOVERY SUMMARY
📈 Tree Accuracy: 0.XXX
🏷️ Label Classes: decision_tree_rules
🌳 ROOT FEATURES (Global Discriminators): X
🔗 EPISTATIC INTERACTIONS: X
✅ STATISTICAL VALIDATION:
Significant features after correction: X
yaml
Copy code

---

## Directory Structure
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


---

## Installation
### Option 1: Conda Environment (recommended)
```bash
git clone https://github.com/Nomlie/network_parser.git
cd network_parser
conda env create -f environment.yml
conda activate network_parser
Option 2: Pip Installation
```

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
Outputs will be saved in the results/ directory.

Command-Line Options:

--genomic: Path to genomic matrix (e.g., data/matrix.csv).
--meta: Path to metadata (e.g., data/labels.csv).
--label: Label column name (e.g., label).
--output-dir: Output directory (e.g., results/).
Config File: Supports YAML/JSON for reproducibility (defined in config.py).
Scalability: Multi-threaded execution for large datasets.

---
**Analysis Modes**
Hierarchical Mode: Analyzes phylogenetic or lineage-based contexts.
Phenotype Mode: Focuses on metadata-driven comparisons (e.g., disease vs. healthy).

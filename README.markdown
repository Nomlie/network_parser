# Phylogenetic Polymorphism Analysis Tool


A bioinformatics tool for identifying statistically validated polymorphisms that differentiate phylogenetic groups using machine learning and statistical analysis.

---

## 🧬 Overview

This tool analyzes genomic polymorphism data to discover diagnostic markers that distinguish between evolutionary lineages or predefined groups. It combines phylogenetic analysis with machine learning to identify minimal sets of polymorphisms for accurate group classification.

### 🔑 Key Features

- **Dual Analysis Modes**: Tree-based phylogenetic analysis or label-based group comparison  
- **Multiple Input Formats**: CSV, VCF, and FASTA support  
- **Machine Learning**: Random Forest classification with bootstrap validation  
- **Statistical Rigor**: Chi-squared tests with FDR correction  
- **Minimal Marker Discovery**: Decision tree analysis for diagnostic polymorphism sets  
- **Flexible Parsimony**: Multiple algorithms (Camin-Sokal, Wagner, Dollo, Fitch)

---

## 🚀 Quick Start

### 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/phylogenetic-analysis.git
cd phylogenetic-analysis

# Create conda environment
conda env create -f environment.yml
conda activate phylogenetic_analysis

# Or use pip
pip install -r requirements.txt
▶️ Basic Usage
Tree-based analysis
bash
Copy
Edit
python phylogenetic_analysis.py \
  --newick data/tree.tre \
  --polymorphisms data/snps.fasta \
  --target_clade "23_YP48_MANG,23_YP47_MANG,23_YP45_MANG" \
  --output_dir results \
  --json_output
Label-based analysis
bash
Copy
Edit
python decisiontree.py \
  --newick data/tree.tre \
  --polymorphisms data/snps.csv \
  --target_clade "23_YP48_MANG,23_YP47_MANG,23_YP45_MANG,23_YP49_MANG,23_YP44_MANG" \
  --output_dir results \
  --json_output
📊 Input Data Formats
Polymorphism Data
Format	Description	Example
CSV	Sample names + binary SNP columns (0/1)	sample,SNP1,SNP2,SNP3...
VCF	Standard VCF format with genotype data	Standard VCF with GT fields
FASTA	Binary (0/1) or nucleotide sequences	>sample1\n010110101...

Tree Data
Format: Newick (.nwk, .tre)

Compatibility: BEAST, MrBayes, RAxML outputs

Note: Sample names must match polymorphism data

Label Data
text
Copy
Edit
# Format: sample_name|group_label
sample1|N
sample2|P
sample3|N
unlabeled_sample
🔧 Command Line Options
Required Arguments
Tree-based mode:

--newick: Path to Newick tree file

--polymorphisms: Path to polymorphism data

--target_clade: Target clade specification

Label-based mode:

--input_file: Path to polymorphism data

--label_file: Path to group labels

Optional Parameters
Parameter	Default	Description
--algorithm	Camin-Sokal	Parsimony algorithm
--tree_levels	3	Max decision tree depth
--min_accuracy	0.95	Minimum classification accuracy
--bootstrap_iterations	1000	Bootstrap validation iterations
--fdr_threshold	0.05	False discovery rate threshold
--max_workers	4	Parallel processing threads

📈 Output
Text Report
yaml
Copy
Edit
================================================================================
Phylogenetic Polymorphism Analysis Results (v1.0.1)
================================================================================

Performance Metrics:
Random Forest Accuracy: 0.9333
Bootstrap p-value: 0.0015

Significant Polymorphisms (FDR < 0.05):
SNP1: q=0.0028
SNP7: q=0.0028
SNP17: q=0.0028

Minimal Diagnostic Set:
SNP1

Decision Tree Rules:
|--- SNP1 <= 0.50
|   |--- class: Group_N
|--- SNP1 >  0.50
|   |--- class: Group_P
JSON Output (Optional)
Structured output for programmatic parsing and tool integration.

📁 Project Structure
bash
Copy
Edit
phylogenetic_analysis/
├── phylogenetic_analysis.py    # Main script
├── environment.yml             # Conda environment
├── requirements.txt            # Python dependencies
├── config.yml                  # Configuration file
├── setup.py                    # Package setup
├── data/                       # Example data
│   ├── tree.tre
│   ├── snps.fasta
│   └── labels.txt
└── README.md
🧪 Example Analysis
Dataset:

14 samples from a phylogenetic tree

89 SNPs in binary format

Two groups: N (6 samples) vs P (9 samples)

Results:

Identified 7 significant polymorphisms (FDR < 0.05)

SNP1 selected as the minimal diagnostic marker

Classification accuracy: 93.3%

⚙️ Advanced Configuration
Use a config.yml for reproducible/batch analysis:

yaml
Copy
Edit
algorithm: "Camin-Sokal"
min_accuracy: 0.95
bootstrap_iterations: 1000
fdr_threshold: 0.05
max_workers: 8
🔬 Methods
Statistical Framework
Random Forest Classification: Initial feature importance ranking

Bootstrap Analysis: Statistical significance testing (n=1000)

Chi-squared Testing: Individual polymorphism significance

FDR Correction: Multiple testing correction

Decision Trees: Minimal diagnostic set identification

Parsimony Algorithms
Camin-Sokal: Irreversible evolution model

Wagner: Reversible evolution model

Dollo: Single-gain, multiple-loss model

Fitch: Unordered character states

🐛 Troubleshooting
Sample name mismatches:

bash
Copy
Edit
# Ensure sample names are identical across input files
grep ">" data/snps.fasta | head -5
head -5 data/labels.txt
Duplicate samples:

bash
Copy
Edit
# Check for duplicates
cut -d',' -f1 data/snps.csv | sort | uniq -d
Memory issues with large datasets:

bash
Copy
Edit
# Reduce bootstrap iterations
python phylogenetic_analysis.py --bootstrap_iterations 100 ...
📚 Citation
If you use this tool in your research, please cite:

bibtex
Copy
Edit
@software{phylogenetic_polymorphism_tool,
  title={Phylogenetic Polymorphism Analysis Tool},
  author={Your Name},
  version={2.0.1},
  year={2025},
  url={https://github.com/your-repo/phylogenetic-analysis}
}
🤝 Contributing
Contributions are welcome!

Fork the repository

Create a feature branch

Make your changes

Add tests

Submit a pull request

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

🆘 Support
Issues: https://github.com/your-repo/phylogenetic-analysis/issues

Discussions: https://github.com/your-repo/phylogenetic-analysis/discussions

Email: your.email@institution.edu
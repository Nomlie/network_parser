Phylogenetic Polymorphism Analysis Tool

A bioinformatics tool for identifying statistically validated polymorphisms that differentiate phylogenetic groups using machine learning and statistical analysis.
🧬 Overview
This tool analyzes genomic polymorphism data to discover diagnostic markers that distinguish between evolutionary lineages or predefined groups. It combines phylogenetic analysis with machine learning to identify minimal sets of polymorphisms for accurate group classification.
Key Features

Dual Analysis Modes: Tree-based phylogenetic analysis or label-based group comparison
Multiple Input Formats: CSV, VCF, and FASTA support
Machine Learning: Random Forest classification with bootstrap validation
Statistical Rigor: Chi-squared tests with FDR correction
Minimal Marker Discovery: Decision tree analysis for diagnostic polymorphism sets
Flexible Parsimony: Multiple algorithms (Camin-Sokal, Wagner, Dollo, Fitch)

🚀 Quick Start
Installation
bash# Clone the repository
git clone https://github.com/your-repo/phylogenetic-analysis.git
cd phylogenetic-analysis

# Create conda environment
conda env create -f environment.yml
conda activate phylogenetic_analysis

# Or use pip
pip install -r requirements.txt
Basic Usage
Tree-based analysis:
bashpython phylogenetic_analysis.py \
  --newick data/tree.tre \
  --polymorphisms data/snps.fasta \
  --target_clade "23_YP48_MANG,23_YP47_MANG,23_YP45_MANG" \
  --output_dir results \
  --json_output
Label-based analysis:
bashpython decisiontree.py \
  --newick data/tree.tre \
  --polymorphisms data/snps.csv \
  --target_clade "23_YP48_MANG,23_YP47_MANG,23_YP45_MANG,23_YP49_MANG,23_YP44_MANG" \
  --output_dir results \
  --json_output
📊 Input Data Formats
Polymorphism Data
FormatDescriptionExampleCSVSample names + binary SNP columns (0/1)sample,SNP1,SNP2,SNP3...VCFStandard VCF format with genotype dataStandard VCF with GT fieldsFASTABinary (0/1) or nucleotide sequences>sample1\n010110101...
Tree Data

Newick format (.nwk, .tre)
Compatible with BEAST, MrBayes, RAxML outputs
Sample names must match polymorphism data

Label Data
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
ParameterDefaultDescription--algorithmCamin-SokalParsimony algorithm--tree_levels3Max decision tree depth--min_accuracy0.95Minimum classification accuracy--bootstrap_iterations1000Bootstrap validation iterations--fdr_threshold0.05False discovery rate threshold--max_workers4Parallel processing threads
📈 Output
Text Report
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
JSON Output (optional)
Structured data for programmatic analysis and integration with other tools.
📁 Project Structure
phylogenetic_analysis/
├── phylogenetic_analysis.py    # Main script
├── environment.yml             # Conda environment
├── requirements.txt           # Python dependencies
├── config.yml                # Configuration file
├── setup.py                  # Package setup
├── data/                     # Example data
│   ├── tree.tre             # Phylogenetic tree
│   ├── snps.fasta          # SNP data
│   └── labels.txt          # Group labels
└── README.md
🧪 Example Analysis
Dataset

14 samples from phylogenetic tree
89 SNPs in binary format
Two groups: N (6 samples) vs P (9 samples)

Results
The analysis identified 7 significant polymorphisms (FDR < 0.05) with SNP1 serving as the minimal diagnostic marker, achieving 93.3% classification accuracy.
⚙️ Advanced Configuration
Create a config.yml file for batch processing:
yamlalgorithm: "Camin-Sokal"
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
Common Issues
Sample name mismatches:
bash# Ensure sample names are identical across all input files
grep ">" data/snps.fasta | head -5
head -5 data/labels.txt
Duplicate samples:
bash# Check for duplicates in your data
cut -d',' -f1 data/snps.csv | sort | uniq -d
Memory issues with large datasets:
bash# Reduce bootstrap iterations for large datasets
python phylogenetic_analysis.py --bootstrap_iterations 100 ...
📚 Citation
If you use this tool in your research, please cite:
bibtex@software{phylogenetic_polymorphism_tool,
  title={Phylogenetic Polymorphism Analysis Tool},
  author={Your Name},
  version={2.0.1},
  year={2025},
  url={https://github.com/your-repo/phylogenetic-analysis}
}
🤝 Contributing
Contributions are welcome! Please see our Contributing Guidelines for details.

Fork the repository
Create a feature branch
Make your changes
Add tests
Submit a pull request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🆘 Support

Issues: GitHub Issues
Discussions: GitHub Discussions
Email: your.email@institution.edu

🔄 Version History


Developed for genomic epidemiology and phylogenetic analysis in bioinformatics research.
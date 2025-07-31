# 🧬 Phylogenetic Polymorphism Analysis Tool

## ▶️ Basic Usage
After installation, you can run the tool in two primary modes:

### Tree-Based Analysis
```bash
python phylogenetic_analysis.py \\
  --newick data/tree.tre \\
  --polymorphisms data/snps.fasta \\
  --target_clade "23_YP48_MANG,23_YP47_MANG,23_YP45_MANG" \\
  --output_dir results \\
  --json_output

### Label-Based Analysis
bash
Copy
Edit
python phylogenetic_analysis.py \\
  --input_file data/snps.csv \\
  --label_file data/labels.txt \\
  --output_dir results \\
  --json_output
📊 Input Data Formats
Polymorphism Data
Format	Description	Example
CSV	Sample names with binary SNP columns (0/1)	sample,SNP1,SNP2,SNP3...
VCF	Standard VCF format with genotype data	Standard VCF with GT fields
FASTA	Binary (0/1) or nucleotide sequences	>sample1\n010110101...

Tree Data
Format: Newick (.nwk, .tre)

Compatibility: BEAST, MrBayes, RAxML outputs

Note: Sample names must match those in polymorphism data

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
Tree-Based Mode:

--newick: Path to Newick tree file

--polymorphisms: Path to polymorphism data

--target_clade: Comma-separated list of target clade samples

Label-Based Mode:

--input_file: Path to polymorphism data

--label_file: Path to group labels

Optional Parameters
Parameter	Default	Description
--algorithm	Camin-Sokal	Parsimony algorithm (Camin-Sokal, Wagner, Dollo, Fitch)
--tree_levels	3	Maximum decision tree depth
--min_accuracy	0.95	Minimum classification accuracy
--bootstrap_iterations	1000	Number of bootstrap validation iterations
--fdr_threshold	0.05	False discovery rate threshold
--max_workers	4	Number of parallel processing threads

📈 Output
Text Report
text
Copy
Edit
================================================================================
Phylogenetic Polymorphism Analysis Results (v2.0.1)
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
  |--- SNP1 > 0.50
      |--- class: Group_P
JSON Output (Optional)
Structured output for programmatic parsing and integration with other tools.

📁 Project Structure
text
Copy
Edit
dtree/
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
Dataset
14 samples from a phylogenetic tree

89 SNPs in binary format

Two groups: N (6 samples) vs P (9 samples)

Results
Identified 7 significant polymorphisms (FDR < 0.05)

SNP1 selected as the minimal diagnostic marker

Classification accuracy: 93.3%

⚙️ Advanced Configuration
yaml
Copy
Edit
algorithm: Camin-Sokal
min_accuracy: 0.95
bootstrap_iterations: 1000
fdr_threshold: 0.05
max_workers: 8
🔬 Methods
Statistical Framework
Random Forest Classification: Ranks feature importance

Bootstrap Analysis: Tests statistical significance (n=1000)

Chi-squared Testing: Evaluates individual polymorphism significance

FDR Correction: Adjusts for multiple testing

Decision Trees: Identifies minimal diagnostic sets

Parsimony Algorithms
Camin-Sokal: Irreversible evolution model

Wagner: Reversible evolution model

Dollo: Single-gain, multiple-loss model

Fitch: Unordered character states

🐛 Troubleshooting
Sample Name Mismatches
bash
Copy
Edit
# Check sample names in input files
grep ">" data/snps.fasta | head -5
head -5 data/labels.txt
Duplicate Samples
bash
Copy
Edit
# Check for duplicates
cut -d',' -f1 data/snps.csv | sort | uniq -d
Memory Issues with Large Datasets
bash
Copy
Edit
# Reduce bootstrap iterations
python phylogenetic_analysis.py --bootstrap_iterations 100 ...
📚 Citation
bibtex
Copy
Edit
@software{phylogenetic_polymorphism_tool,
  title = {Phylogenetic Polymorphism Analysis Tool},
  author = {Nomlie Fuphi},
  version = {2.0.1},
  year = {2025},
  url = {https://github.com/Nomlie/dtree}
}
🤝 Contributing
Fork the repository

Create a feature branch

Make your changes and add tests

Submit a pull request

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

🆘 Support
Issues: https://github.com/Nomlie/dtree/issues

Discussions: https://github.com/Nomlie/dtree/discussions

Email: nmfuphi@csir.co.za
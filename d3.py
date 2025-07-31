#!/usr/bin/env python3
"""
Phylogenetic Polymorphism Analysis Tool with Decision Tree Interface

This script analyzes phylogenetic trees and polymorphism data to identify
statistically validated polymorphisms that differentiate a target clade from its
sister clade. It uses random forest classification, bootstrap analysis, and
decision trees with optional parsimony scoring.

Version: 2.0.0
Created: 2025-07-29
Updated: 2025-07-31
"""

import os
import re
import logging
import argparse
import time
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from statsmodels.stats.multitest import fdrcorrection
from concurrent.futures import ThreadPoolExecutor
import json
from ete3 import Tree
import yaml
from Bio import SeqIO

def setup_logging(project_dir, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, project_dir, f"analysis_{timestamp}.log")
    os.makedirs(os.path.join(output_dir, project_dir), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return log_file

def load_config(config_path="config.yml"):
    try:
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        return config
    except FileNotFoundError:
        logging.warning(f"Config file {config_path} not found. Using default settings.")
        return {
            'min_accuracy': 0.95,
            'bootstrap_iterations': 1000,
            'fdr_threshold': 0.05,
            'tree_levels': 3,
            'max_workers': 4,
            'algorithm': 'Camin-Sokal'
        }

def load_polymorphism_data(file_path):
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()
    
    if file_extension == '.csv':
        df = pd.read_csv(file_path, index_col='sample')
    elif file_extension == '.vcf':
        df = pd.read_csv(file_path, sep='\t', comment='#', header=None)
        df = df.iloc[:, 9:].T
        df.columns = [f'SNP{i+1}' for i in range(df.shape[1])]
        df.index = df.index.map(str)
    elif file_extension in ['.fasta', '.fa']:
        records = list(SeqIO.parse(file_path, "fasta"))
        if not records:
            raise ValueError("FASTA file is empty")
        
        # Check sequence lengths
        seq_length = len(records[0].seq)
        if not all(len(record.seq) == seq_length for record in records):
            raise ValueError("All sequences in FASTA must have equal length")
        
        # Initialize data
        samples = [record.id for record in records]
        if len(samples) != len(set(samples)):
            raise ValueError("Duplicate sample names found in FASTA file")
        
        # Check for valid characters (0/1 or A/C/G/T)
        valid_binary = {'0', '1'}
        valid_nucleotides = {'A', 'C', 'G', 'T'}
        first_seq = str(records[0].seq).upper()
        is_binary = all(c in valid_binary for c in first_seq)
        is_nucleotide = all(c in valid_nucleotides for c in first_seq)
        
        if not (is_binary or is_nucleotide):
            raise ValueError("FASTA sequences must contain only 0/1 or A/C/G/T")
        
        # Create DataFrame
        data = []
        for record in records:
            seq = str(record.seq).upper()
            if is_binary and not all(c in valid_binary for c in seq):
                raise ValueError(f"Sequence {record.id} contains invalid characters for binary format")
            if is_nucleotide and not all(c in valid_nucleotides for c in seq):
                raise ValueError(f"Sequence {record.id} contains invalid characters for nucleotide format")
            data.append(list(seq))
        
        df = pd.DataFrame(data, index=samples, columns=[f'SNP{i+1}' for i in range(seq_length)])
        
        # Convert nucleotides to binary (if applicable)
        if is_nucleotide:
            # Assume first sample is reference (ancestral), encode others relative to it
            ref_seq = df.iloc[0]
            for col in df.columns:
                ref_allele = ref_seq[col]
                df[col] = df[col].apply(lambda x: 0 if x == ref_allele else 1)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Use .csv, .vcf, .fasta, or .fa")
    
    # Ensure data is binary
    if not df.applymap(lambda x: x in [0, 1]).all().all():
        raise ValueError("Polymorphism data must contain only 0 or 1 values")
    
    return df

def load_label_data(label_file):
    labels = {}
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '|' in line:
                sample, group = line.split('|')
                labels[sample] = group
            else:
                labels[line] = None  # Unlabeled samples
    if len(labels) != len(set(labels.keys())):
        raise ValueError("Duplicate sample names found in label file")
    return labels

def load_tree(newick_file):
    with open(newick_file, 'r') as f:
        content = f.read()
    content = re.sub(r'\[.*?\]', '', content)
    content = re.sub(r'^#NEXUS.*?\nBEGIN TREES;\n.*?Tree\s+\w+\s*=\s*', '', content, flags=re.DOTALL)
    content = content.strip().rstrip(';')
    tree = Tree(content + ';', format=1)
    return tree

def get_clade_samples(tree, target_clade):
    if ',' in target_clade:
        target_samples = set(target_clade.split(','))
        clade_node = tree.get_common_ancestor(target_samples)
        clade_samples = set(clade_node.get_leaf_names())
        sister_node = clade_node.get_sisters()[0] if clade_node.get_sisters() else None
        sister_samples = set(sister_node.get_leaf_names()) if sister_node else set()
    else:
        clade_node = tree.search_nodes(name=target_clade)[0]
        clade_samples = set(clade_node.get_leaf_names())
        sister_node = clade_node.get_sisters()[0] if clade_node.get_sisters() else None
        sister_samples = set(sister_node.get_leaf_names()) if sister_node else set()
    return clade_samples, sister_samples

def assign_parsimony_scores(data, algorithm='Camin-Sokal'):
    scores = data.copy()
    if algorithm == 'Camin-Sokal':
        pass  # 0 ancestral, 1 derived
    elif algorithm == 'Wagner':
        scores = scores.abs()
    elif algorithm == 'Dollo':
        scores = scores.where(scores == 1, 0)
    elif algorithm == 'Fitch':
        scores = scores.apply(lambda x: x % 2)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return scores

def bootstrap_rf(data, labels, n_iterations=1000, max_workers=4):
    def single_bootstrap(_):
        sample_indices = np.random.choice(len(labels), len(labels), replace=True)
        sampled_data = data.iloc[sample_indices]
        sampled_labels = labels[sample_indices]
        rf = RandomForestClassifier(n_estimators=100, random_state=np.random.randint(10000))
        rf.fit(sampled_data, sampled_labels)
        return rf.score(data, labels)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        accuracies = list(executor.map(single_bootstrap, range(n_iterations)))
    return np.mean(accuracies), stats.ttest_1samp(accuracies, 0.5)[1]

def chi_squared_tests(data, labels):
    p_values = []
    for column in data.columns:
        contingency_table = pd.crosstab(labels, data[column])
        chi2, p, _, _ = stats.chi2_contingency(contingency_table)
        p_values.append(p)
    _, q_values = fdrcorrection(p_values)
    return pd.Series(q_values, index=data.columns)

def build_decision_tree(data, labels, max_depth=3):
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt.fit(data, labels)
    return dt

def main(args):
    config = load_config(args.config)
    log_file = setup_logging(args.project_dir, args.output_dir)
    logging.info("Starting phylogenetic polymorphism analysis...")

    if args.input_file and args.label_file:
        # Label-based analysis
        logging.info(f"Loading polymorphism data from {args.input_file}")
        data = load_polymorphism_data(args.input_file)
        logging.info(f"Loading label data from {args.label_file}")
        labels_dict = load_label_data(args.label_file)
        
        # Filter samples with N or P labels
        valid_labels = {k: v for k, v in labels_dict.items() if v in ['N', 'P']}
        if not valid_labels:
            raise ValueError("No samples with 'N' or 'P' labels found in label file")
        if len(valid_labels) < 2:
            raise ValueError("At least two samples with 'N' or 'P' labels are required")
        
        # Match samples to polymorphism data
        samples = list(valid_labels.keys())
        if not all(s in data.index for s in samples):
            missing = [s for s in samples if s not in data.index]
            raise ValueError(f"Samples not found in polymorphism data: {missing}")
        
        data = data.loc[samples]
        labels = np.array([1 if valid_labels[s] == 'P' else 0 for s in samples])
        
        # Apply parsimony scoring
        logging.info(f"Applying {args.algorithm} parsimony scoring")
        data = assign_parsimony_scores(data, args.algorithm)
        
        # Random Forest and Bootstrap
        logging.info("Running Random Forest and bootstrap analysis")
        rf_accuracy, bootstrap_p = bootstrap_rf(data, labels, args.bootstrap_iterations, args.max_workers)
        
        if rf_accuracy < args.min_accuracy:
            logging.warning(f"Random Forest accuracy {rf_accuracy:.4f} below threshold {args.min_accuracy}")
        
        # Chi-squared tests with FDR correction
        logging.info("Performing chi-squared tests with FDR correction")
        q_values = chi_squared_tests(data, labels)
        significant_snps = q_values[q_values < args.fdr_threshold].index.tolist()
        
        # Decision Tree
        logging.info("Building decision tree")
        dt = build_decision_tree(data[significant_snps], labels, args.tree_levels)
        dt_rules = tree.export_text(dt, feature_names=significant_snps)
        minimal_snps = significant_snps[:dt.max_depth]
        
        # Prepare output
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        output_dir = os.path.join(args.output_dir, args.project_dir)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{base_name}_results.txt")
        
        results = {
            "Parameters": {
                "Algorithm": args.algorithm,
                "Tree Levels": args.tree_levels,
                "Minimum Accuracy": args.min_accuracy,
                "Bootstrap Iterations": args.bootstrap_iterations,
                "FDR Threshold": args.fdr_threshold
            },
            "Performance Metrics": {
                "Random Forest Accuracy": round(float(rf_accuracy), 4),
                "Bootstrap p-value": f"{bootstrap_p:.4e}"
            },
            "Significant Polymorphisms": {snp: f"{q_values[snp]:.4e}" for snp in significant_snps},
            "Minimal Diagnostic Set": minimal_snps,
            "Decision Tree Rules": dt_rules
        }
        
        # Write text output
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Phylogenetic Polymorphism Analysis Results (v2.0.1)\n")
            f.write("=" * 80 + "\n\n")
            for section, content in results.items():
                f.write(f"{section}:\n")
                if isinstance(content, dict):
                    for k, v in content.items():
                        f.write(f"{k}: {v}\n")
                elif isinstance(content, list):
                    f.write("\n".join(content) + "\n")
                else:
                    f.write(f"{content}\n")
                f.write("\n")
            f.write("=" * 80 + "\n")
            f.write(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
        
        # Write JSON output if requested
        if args.json_output:
            json_file = os.path.join(output_dir, f"{base_name}_results.json")
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=4)
        
        logging.info(f"Results saved to {output_file}")
        if args.json_output:
            logging.info(f"JSON results saved to {json_file}")
        
    elif args.newick and args.polymorphisms and args.target_clade:
        # Tree-based analysis
        logging.info(f"Loading tree from {args.newick}")
        tree = load_tree(args.newick)
        logging.info(f"Loading polymorphism data from {args.polymorphisms}")
        data = load_polymorphism_data(args.polymorphisms)
        logging.info(f"Processing target clade: {args.target_clade}")
        clade_samples, sister_samples = get_clade_samples(tree, args.target_clade)
        
        samples = list(clade_samples | sister_samples)
        if not all(s in data.index for s in samples):
            missing = [s for s in samples if s not in data.index]
            raise ValueError(f"Samples not found in polymorphism data: {missing}")
        
        data = data.loc[samples]
        labels = np.array([1 if s in clade_samples else 0 for s in samples])
        
        logging.info(f"Applying {args.algorithm} parsimony scoring")
        data = assign_parsimony_scores(data, args.algorithm)
        
        logging.info("Running Random Forest and bootstrap analysis")
        rf_accuracy, bootstrap_p = bootstrap_rf(data, labels, args.bootstrap_iterations, args.max_workers)
        
        if rf_accuracy < args.min_accuracy:
            logging.warning(f"Random Forest accuracy {rf_accuracy:.4f} below threshold {args.min_accuracy}")
        
        logging.info("Performing chi-squared tests with FDR correction")
        q_values = chi_squared_tests(data, labels)
        significant_snps = q_values[q_values < args.fdr_threshold].index.tolist()
        
        logging.info("Building decision tree")
        dt = build_decision_tree(data[significant_snps], labels, args.tree_levels)
        dt_rules = tree.export_text(dt, feature_names=significant_snps)
        minimal_snps = significant_snps[:dt.max_depth]
        
        base_name = os.path.splitext(os.path.basename(args.polymorphisms))[0]
        output_dir = os.path.join(args.output_dir, args.project_dir)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{base_name}_results.txt")
        
        results = {
            "Parameters": {
                "Algorithm": args.algorithm,
                "Tree Levels": args.tree_levels,
                "Minimum Accuracy": args.min_accuracy,
                "Bootstrap Iterations": args.bootstrap_iterations,
                "FDR Threshold": args.fdr_threshold
            },
            "Performance Metrics": {
                "Random Forest Accuracy": round(float(rf_accuracy), 4),
                "Bootstrap p-value": f"{bootstrap_p:.4e}"
            },
            "Significant Polymorphisms": {snp: f"{q_values[snp]:.4e}" for snp in significant_snps},
            "Minimal Diagnostic Set": minimal_snps,
            "Decision Tree Rules": dt_rules
        }
        
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Phylogenetic Polymorphism Analysis Results (v2.0.1)\n")
            f.write("=" * 80 + "\n\n")
            for section, content in results.items():
                f.write(f"{section}:\n")
                if isinstance(content, dict):
                    for k, v in content.items():
                        f.write(f"{k}: {v}\n")
                elif isinstance(content, list):
                    f.write("\n".join(content) + "\n")
                else:
                    f.write(f"{content}\n")
                f.write("\n")
            f.write("=" * 80 + "\n")
            f.write(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
        
        if args.json_output:
            json_file = os.path.join(output_dir, f"{base_name}_results.json")
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=4)
        
        logging.info(f"Results saved to {output_file}")
        if args.json_output:
            logging.info(f"JSON results saved to {json_file}")
    
    else:
        raise ValueError("Must provide either --newick, --polymorphisms, and --target_clade for tree-based analysis, or --input_file and --label_file for label-based analysis")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phylogenetic Polymorphism Analysis Tool (v2.0.1)")
    parser.add_argument("--newick", help="Path to Newick tree file (.nwk or .tre)")
    parser.add_argument("--polymorphisms", help="Path to polymorphism data file (CSV, VCF, FASTA)")
    parser.add_argument("--target_clade", help="Target clade name or comma-separated leaf names")
    parser.add_argument("-f", "--input_file", help="Input file for label-based analysis (CSV, FASTA)")
    parser.add_argument("-g", "--label_file", help="Label file for group assignments (e.g., data/label.txt)")
    parser.add_argument("-i", "--input_dir", default="input", help="Input folder")
    parser.add_argument("-o", "--output_dir", default="output", help="Output folder")
    parser.add_argument("-p", "--project_dir", default="", help="Project folder name")
    parser.add_argument("-a", "--algorithm", default="Camin-Sokal", choices=["Camin-Sokal", "Wagner", "Dollo", "Fitch"], help="Parsimony algorithm")
    parser.add_argument("-c", "--config", default="config.yml", help="Path to config file")
    parser.add_argument("-l", "--tree_levels", type=int, default=3, help="Max decision tree levels")
    parser.add_argument("-m", "--min_accuracy", type=float, default=0.95, help="Minimum classification accuracy")
    parser.add_argument("--bootstrap_iterations", type=int, default=1000, help="Number of bootstrap iterations")
    parser.add_argument("--fdr_threshold", type=float, default=0.05, help="FDR threshold")
    parser.add_argument("--json_output", action="store_true", help="Save results as JSON")
    parser.add_argument("--max_workers", type=int, default=4, help="Max parallel workers for bootstrap")
    parser.add_argument("-v", "--version", action="version", version="2.0.1")
    
    args = parser.parse_args()
    main(args)
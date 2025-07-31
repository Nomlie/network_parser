#!/usr/bin/env python3
"""
Phylogenetic Polymorphism Analysis Tool with Decision Tree Interface

This script analyzes phylogenetic trees and polymorphism data to identify
statistically validated polymorphisms that differentiate a target clade from its
sister clade. It uses random forest classification, bootstrap analysis, and
decision trees with optional parsimony scoring.

Version: 2.0.0
Created: 2023-11-15
Updated: 2025-07-29
"""

import argparse
import sys
import os
import logging
from typing import Tuple, List, Union, Dict, Optional
from datetime import datetime
import json
from concurrent.futures import ProcessPoolExecutor
import warnings
import yaml

import ete3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

# Version information
__version__ = "2.0.0"
date_of_creation = "2023-11-15"
date_of_update = "2025-07-29"

# Configure logging
def setup_logging(output_dir: str, project_dir: str) -> None:
    """Set up logging to console and file."""
    log_file = os.path.join(output_dir, project_dir, f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")

# Suppress warnings from external libraries
warnings.filterwarnings("ignore", category=UserWarning)

def load_config(config_file: str = "config.yml") -> Dict:
    """Load configuration from YAML file."""
    logger = logging.getLogger(__name__)
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_file}")
        return config or {}
    except FileNotFoundError:
        logger.warning(f"Config file {config_file} not found; using defaults")
        return {}
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        sys.exit(1)

def show_help() -> None:
    """Display formatted help information."""
    help_text = f"""
{'='*80}
Phylogenetic Polymorphism Analysis Tool (v{__version__})

Usage:
    python {os.path.basename(__file__)} [options]

Description:
    Identifies diagnostic polymorphisms distinguishing a target clade from its sister
    clade using phylogenetic trees and polymorphism data. Supports statistical
    validation and decision tree classification.

Required Arguments (for phylogenetic analysis):
    --newick              Path to Newick tree file
    --polymorphisms       Path to polymorphism data (CSV or VCF)
    --target_clade        Target clade node name or comma-separated leaf names

Required Arguments (for parsimony analysis):
    -f, --input_file      Input file in FASTA or CSV format
    -g, --label_file      File with group labels (CSV with sample,group columns)

Optional Arguments:
    -i, --input_dir       Input folder (default: config.yml or 'input')
    -o, --output_dir      Output folder (default: config.yml or 'output')
    -p, --project_dir     Project folder name (default: config.yml or '')
    -a, --algorithm       Parsimony algorithm (Camin-Sokal, Wagner, Dollo, Fitch)
                          (default: config.yml or Camin-Sokal)
    -l, --tree_levels     Max levels in decision tree (default: config.yml or 3)
    -m, --min_accuracy    Minimum classification accuracy (default: config.yml or 0.95)
    --bootstrap_iterations Number of bootstrap iterations (default: config.yml or 1000)
    --fdr_threshold       FDR threshold for multiple testing (default: config.yml or 0.05)
    --json_output         Save results as JSON (default: config.yml or False)
    --max_workers         Max parallel workers for bootstrap (default: config.yml or 4)

Help Options:
    -h, --help            Show help message
    -v, --version         Show version info
{'='*80}
"""
    print(help_text)

def parse_arguments(config: Dict) -> argparse.Namespace:
    """Parse command line arguments with config defaults."""
    parser = argparse.ArgumentParser(
        description="Phylogenetic Polymorphism Analysis with Decision Tree Interface",
        add_help=False
    )

    parser.add_argument("--newick", help="Path to Newick tree file")
    parser.add_argument("--polymorphisms", help="Path to polymorphism data (CSV or VCF)")
    parser.add_argument("--target_clade", help="Target clade node name or comma-separated leaf names")
    parser.add_argument("-f", "--input_file", default="", help="Input file in FASTA or CSV format")
    parser.add_argument("-g", "--label_file", default="", help="File with group labels")
    parser.add_argument("-i", "--input_dir", default=config.get("input_dir", "input"), help="Input folder")
    parser.add_argument("-o", "--output_dir", default=config.get("output_dir", "output"), help="Output folder")
    parser.add_argument("-p", "--project_dir", default=config.get("project_dir", ""), help="Project folder name")
    parser.add_argument("-a", "--algorithm", choices=["Camin-Sokal", "Wagner", "Dollo", "Fitch"],
                        default=config.get("algorithm", "Camin-Sokal"), help="Parsimony algorithm to use")
    parser.add_argument("-l", "--tree_levels", type=int, default=config.get("tree_levels", 3), help="Max levels in decision tree")
    parser.add_argument("-m", "--min_accuracy", type=float, default=config.get("min_accuracy", 0.95), help="Minimum classification accuracy")
    parser.add_argument("--bootstrap_iterations", type=int, default=config.get("bootstrap_iterations", 1000), help="Number of bootstrap iterations")
    parser.add_argument("--fdr_threshold", type=float, default=config.get("fdr_threshold", 0.05), help="FDR threshold for multiple testing")
    parser.add_argument("--json_output", action="store_true", default=config.get("json_output", False), help="Save results as JSON")
    parser.add_argument("--max_workers", type=int, default=config.get("max_workers", 4), help="Max parallel workers for bootstrap")
    parser.add_argument("-h", "--help", action="store_true", help="Show help message")
    parser.add_argument("-v", "--version", action="store_true", help="Show version info")

    args = parser.parse_args()

    if args.tree_levels < 1:
        logger.error("Tree levels must be at least 1")
        sys.exit(1)
    if not 0 < args.min_accuracy <= 1:
        logger.error("Minimum accuracy must be between 0 and 1")
        sys.exit(1)
    if args.bootstrap_iterations < 1:
        logger.error("Bootstrap iterations must be positive")
        sys.exit(1)
    if not 0 < args.fdr_threshold < 1:
        logger.error("FDR threshold must be between 0 and 1")
        sys.exit(1)
    if args.max_workers < 1:
        logger.error("Max workers must be positive")
        sys.exit(1)

    return args

def validate_file_paths(args: argparse.Namespace) -> None:
    """Validate input file paths."""
    logger = logging.getLogger(__name__)
    paths_to_check = []
    if args.input_file:
        paths_to_check.append(os.path.join(args.input_dir, args.project_dir, args.input_file))
    if args.label_file:
        paths_to_check.append(os.path.join(args.input_dir, args.project_dir, args.label_file))
    if args.newick:
        paths_to_check.append(args.newick)
    if args.polymorphisms:
        paths_to_check.append(args.polymorphisms)

    for path in paths_to_check:
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            sys.exit(1)

def load_polymorphism_data(polymorphism_file: str) -> pd.DataFrame:
    """Load polymorphism data from CSV or VCF file."""
    logger = logging.getLogger(__name__)
    try:
        if polymorphism_file.endswith('.vcf'):
            df = pd.read_csv(polymorphism_file, sep='\t', comment='#', usecols=['#CHROM', 'POS', 'REF', 'ALT'] + [col for col in pd.read_csv(polymorphism_file, sep='\t', comment='#', nrows=1).columns if col.startswith('sample')])
            df['SNP'] = df['#CHROM'] + '_' + df['POS'].astype(str)
            df = df.set_index('SNP').filter(like='sample').replace({'0/0': 0, '1/1': 1, './.': np.nan})
            df = df.T
        else:
            df = pd.read_csv(polymorphism_file, index_col=0)
        
        if df.empty:
            raise ValueError("Polymorphism data is empty")
        if df.isna().any().any():
            logger.warning("Polymorphism data contains missing values; these will be ignored")
        logger.info(f"Loaded polymorphism data with {df.shape[0]} samples and {df.shape[1]} polymorphisms")
        return df
    except Exception as e:
        logger.error(f"Error loading polymorphism data: {e}")
        sys.exit(1)

def load_tree(newick_file: str) -> ete3.Tree:
    """Load phylogenetic tree from Newick file."""
    logger = logging.getLogger(__name__)
    try:
        tree = ete3.Tree(newick_file, format=1)
        if not tree.get_leaf_names():
            raise ValueError("Tree has no leaves")
        logger.info(f"Loaded tree with {len(tree.get_leaf_names())} leaves")
        return tree
    except Exception as e:
        logger.error(f"Error loading Newick file: {e}")
        sys.exit(1)

def get_clade_leaves(tree: ete3.Tree, target_clade: Union[str, List[str]]) -> Tuple[List[str], List[str]]:
    """Get leaves of the target clade and its sister clade."""
    logger = logging.getLogger(__name__)
    try:
        if isinstance(target_clade, str):
            node = tree.search_nodes(name=target_clade)
            if not node:
                raise ValueError(f"Target clade '{target_clade}' not found")
            node = node[0]
        else:
            node = tree.get_common_ancestor(target_clade)
            if not node:
                raise ValueError("Invalid leaf names provided for target clade")
        
        target_leaves = node.get_leaf_names()
        sister_node = node.get_sisters()
        if not sister_node:
            raise ValueError("No sister clade found")
        sister_leaves = sister_node[0].get_leaf_names()
        
        logger.info(f"Target clade: {len(target_leaves)} leaves; Sister clade: {len(sister_leaves)} leaves")
        return target_leaves, sister_leaves
    except Exception as e:
        logger.error(f"Error identifying clades: {e}")
        sys.exit(1)

def apply_parsimony_score(polymorphism: pd.Series, algorithm: str) -> float:
    """Apply parsimony scoring to a polymorphism."""
    logger = logging.getLogger(__name__)
    states = polymorphism.dropna().unique()
    if len(states) < 2:
        return 0.0
    
    if algorithm == "Camin-Sokal":
        score = len(states) - 1
    elif algorithm == "Wagner":
        score = max(states) - min(states)
    elif algorithm == "Dollo":
        score = sum(1 for s in states if s != min(states))
    else:  # Fitch
        score = len(states) - 1
    logger.debug(f"Parsimony score for {polymorphism.name} ({algorithm}): {score}")
    return score

def filter_diagnostic_polymorphisms(df: pd.DataFrame, target_leaves: List[str], sister_leaves: List[str], algorithm: str) -> List[str]:
    """Identify diagnostic polymorphisms with parsimony scoring."""
    logger = logging.getLogger(__name__)
    try:
        target_data = df.loc[target_leaves]
        sister_data = df.loc[sister_leaves]
        
        diagnostic_snps = []
        for col in df.columns:
            target_states = target_data[col].dropna().unique()
            sister_states = sister_data[col].dropna().unique()
            if len(target_states) == 1 and len(sister_states) == 1 and target_states[0] != sister_states[0]:
                score = apply_parsimony_score(df[col], algorithm)
                if score > 0:
                    diagnostic_snps.append(col)
        
        logger.info(f"Found {len(diagnostic_snps)} diagnostic polymorphisms using {algorithm}")
        return diagnostic_snps
    except Exception as e:
        logger.error(f"Error filtering diagnostic polymorphisms: {e}")
        sys.exit(1)

def bootstrap_iteration(X: pd.DataFrame, y: np.ndarray, seed: int) -> float:
    """Perform a single bootstrap iteration."""
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    y_random = np.random.permutation(y)
    scores = cross_val_score(rf_clf, X, y_random, cv=5, scoring='accuracy')
    return np.mean(scores)
def run_bootstrap(args):
    """Helper function for multiprocessing; calls bootstrap_iteration with unpacked args."""
    seed, X, y = args
    return bootstrap_iteration(X, y, seed)

def statistical_validation(
    df: pd.DataFrame,
    target_leaves: List[str],
    sister_leaves: List[str],
    diagnostic_snps: List[str],
    n_iterations: int,
    fdr_threshold: float,
    max_workers: int
) -> Tuple[float, float, List[str], List[float]]:
    """Perform statistical validation of diagnostic polymorphisms."""
    logger = logging.getLogger(__name__)
    try:
        X = df.loc[target_leaves + sister_leaves, diagnostic_snps]
        y = np.array([1 if sample in target_leaves else 0 for sample in X.index])

        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        cv_scores = cross_val_score(rf_clf, X, y, cv=5, scoring='accuracy')
        rf_accuracy = np.mean(cv_scores)
        logger.info(f"Random Forest accuracy: {rf_accuracy:.3f}")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            bootstrap_scores = list(executor.map(
                run_bootstrap,
                [(seed, X, y) for seed in range(n_iterations)]
            ))

        bootstrap_p_value = np.mean(np.array(bootstrap_scores) >= rf_accuracy)
        logger.info(f"Bootstrap p-value: {bootstrap_p_value:.4f}")

        p_values = []
        for snp in diagnostic_snps:
            target_counts = df.loc[target_leaves, snp].value_counts().reindex([0, 1], fill_value=0)
            sister_counts = df.loc[sister_leaves, snp].value_counts().reindex([0, 1], fill_value=0)
            contingency_table = np.array([
                [target_counts[0], target_counts[1]],
                [sister_counts[0], sister_counts[1]]
            ])
            if np.any(contingency_table.sum(axis=0) == 0):
                p_values.append(1.0)
                continue
            _, p, _, _ = chi2_contingency(contingency_table)
            p_values.append(p)

        reject, p_values_corrected, _, _ = multipletests(p_values, alpha=fdr_threshold, method='fdr_bh')
        significant_snps = [snp for snp, reject_flag in zip(diagnostic_snps, reject) if reject_flag]
        logger.info(f"Found {len(significant_snps)} significant polymorphisms (FDR < {fdr_threshold})")

        return rf_accuracy, bootstrap_p_value, significant_snps, p_values_corrected
    except Exception as e:
        logger.error(f"Error in statistical validation: {e}")
        sys.exit(1)

def build_decision_tree(
    df: pd.DataFrame,
    target_leaves: List[str],
    sister_leaves: List[str],
    significant_snps: List[str],
    max_levels: int,
    min_accuracy: float
) -> Tuple[List[str], str]:
    """Build decision tree to select minimal polymorphisms."""
    logger = logging.getLogger(__name__)
    try:
        X = df.loc[target_leaves + sister_leaves, significant_snps]
        y = [1 if sample in target_leaves else 0 for sample in X.index]
        
        clf = DecisionTreeClassifier(
            max_depth=max_levels,
            min_impurity_decrease=1-min_accuracy,
            random_state=42
        )
        clf.fit(X, y)
        
        feature_importance = pd.Series(clf.feature_importances_, index=significant_snps)
        selected_snps = feature_importance[feature_importance > 0].index.tolist()
        tree_rules = export_text(clf, feature_names=significant_snps)
        
        logger.info(f"Decision tree selected {len(selected_snps)} polymorphisms")
        return selected_snps, tree_rules
    except Exception as e:
        logger.error(f"Error building decision tree: {e}")
        sys.exit(1)

def save_results(
    output_path: str,
    input_file: str,
    rf_accuracy: float,
    bootstrap_p_value: float,
    diagnostic_snps: List[str],
    p_values: List[float],
    selected_snps: List[str],
    tree_rules: str,
    args: argparse.Namespace
) -> None:
    """Save analysis results to text and optional JSON file."""
    logger = logging.getLogger(__name__)
    try:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        text_output = os.path.join(output_path, f"{base_name}_results.txt")
        
        with open(text_output, 'w') as f:
            f.write(f"{'='*80}\n")
            f.write(f"Phylogenetic Polymorphism Analysis Results (v{__version__})\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Analysis Parameters:\n")
            f.write(f"Algorithm: {args.algorithm}\n")
            f.write(f"Tree Levels: {args.tree_levels}\n")
            f.write(f"Minimum Accuracy: {args.min_accuracy}\n")
            f.write(f"Bootstrap Iterations: {args.bootstrap_iterations}\n")
            f.write(f"FDR Threshold: {args.fdr_threshold}\n\n")
            f.write(f"Performance Metrics:\n")
            f.write(f"Random Forest Accuracy: {rf_accuracy:.4f}\n")
            f.write(f"Bootstrap p-value: {bootstrap_p_value:.4e}\n\n")
            f.write(f"Significant Polymorphisms (FDR < {args.fdr_threshold}):\n")
            for snp, p in zip(diagnostic_snps, p_values):
                f.write(f"{snp}: q={p:.4e}\n")
            f.write(f"\nMinimal Diagnostic Set (Decision Tree):\n")
            f.write("\n".join(selected_snps))
            f.write(f"\n\nDecision Tree Rules:\n")
            f.write(tree_rules)
            f.write(f"\n{'='*80}\n")
            f.write(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if args.json_output:
            json_output = os.path.join(output_path, f"{base_name}_results.json")
            results = {
                "version": __version__,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "parameters": {
                    "algorithm": args.algorithm,
                    "tree_levels": args.tree_levels,
                    "min_accuracy": args.min_accuracy,
                    "bootstrap_iterations": args.bootstrap_iterations,
                    "fdr_threshold": args.fdr_threshold
                },
                "metrics": {
                    "random_forest_accuracy": rf_accuracy,
                    "bootstrap_p_value": bootstrap_p_value
                },
                "significant_polymorphisms": [
                    {"snp": snp, "q_value": float(p)} for snp, p in zip(diagnostic_snps, p_values)
                ],
                "selected_polymorphisms": selected_snps,
                "decision_tree_rules": tree_rules
            }
            with open(json_output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"JSON results saved to {json_output}")
        
        logger.info(f"Text results saved to {text_output}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        sys.exit(1)

def run_analysis(args: argparse.Namespace) -> None:
    """Main analysis workflow for phylogenetic polymorphism analysis."""
    logger = logging.getLogger(__name__)
    
    output_path = setup_output_directory(args.output_dir, args.project_dir)
    
    tree = load_tree(args.newick)
    df = load_polymorphism_data(args.polymorphisms)
    
    tree_leaves = set(tree.get_leaf_names())
    df_samples = set(df.index)
    if not tree_leaves.issubset(df_samples):
        missing = tree_leaves - df_samples
        logger.error(f"Missing samples in polymorphism data: {missing}")
        sys.exit(1)
    
    target_clade = args.target_clade.split(",") if "," in args.target_clade else args.target_clade
    target_leaves, sister_leaves = get_clade_leaves(tree, target_clade)
    
    diagnostic_snps = filter_diagnostic_polymorphisms(df, target_leaves, sister_leaves, args.algorithm)
    if not diagnostic_snps:
        logger.warning("No diagnostic polymorphisms found")
        return
    
    rf_accuracy, bootstrap_p_value, significant_snps, p_values = statistical_validation(
        df, target_leaves, sister_leaves, diagnostic_snps,
        args.bootstrap_iterations, args.fdr_threshold, args.max_workers
    )
    
    if not significant_snps:
        logger.warning("No significant polymorphisms found after FDR correction")
        return
    
    selected_snps, tree_rules = build_decision_tree(
        df, target_leaves, sister_leaves, significant_snps,
        args.tree_levels, args.min_accuracy
    )
    
    save_results(
        output_path, args.polymorphisms, rf_accuracy, bootstrap_p_value,
        diagnostic_snps, p_values, selected_snps, tree_rules, args
    )

def setup_output_directory(output_dir: str, project_dir: str) -> str:
    """Create and return output directory path."""
    logger = logging.getLogger(__name__)
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, project_dir)
        os.makedirs(output_path, exist_ok=True)
        return output_path
    except Exception as e:
        logger.error(f"Error creating output directory: {e}")
        sys.exit(1)

def main() -> None:
    """Entry point for the script."""
    config = load_config()
    setup_logging(config.get("output_dir", "output"), config.get("project_dir", ""))
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Phylogenetic Polymorphism Analysis Tool v{__version__}")

    args = parse_arguments(config)

    if args.help:
        show_help()
        sys.exit(0)

    if args.version:
        print(f"\nPhylogenetic Polymorphism Analysis Tool v{__version__}\nCreated: {date_of_creation}\nUpdated: {date_of_update}\n")
        sys.exit(0)

    validate_file_paths(args)

    if args.newick and args.polymorphisms and args.target_clade:
        logger.info("Running phylogenetic polymorphism analysis")
        run_analysis(args)
    elif args.input_file and args.label_file:
        logger.info("FASTA/label file analysis not implemented in this version")
        sys.exit(1)
    else:
        logger.error("Insufficient arguments provided")
        show_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
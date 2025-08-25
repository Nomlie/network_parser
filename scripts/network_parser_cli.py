# scripts/networkparser_cli.py
#!/usr/bin/env python3
"""
Command-line interface for NetworkParser.
"""

import argparse
import logging
import sys

from networkparser import NetworkParser, NetworkParserConfig
from networkparser.utils import create_config_from_args, load_config_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main command line interface"""
    parser = argparse.ArgumentParser(
        description="NetworkParser: Interpretable Framework for Epistatic Cluster Segregation Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Hierarchical analysis
  networkparser_cli.py --input_matrix data/genomic_features.csv \\
                       --metadata data/sample_metadata.csv \\
                       --hierarchy_column "lineage" \\
                       --output_dir results

  # Phenotype-based analysis  
  networkparser_cli.py --input_matrix data/resistance_profiles.csv \\
                       --phenotype_file data/phenotypes.txt \\
                       --target_groups "resistant,sensitive" \\
                       --output_dir results

  # With known markers
  networkparser_cli.py --input_matrix data/snp_matrix.csv \\
                       --metadata data/metadata.csv \\
                       --hierarchy_column "cluster" \\
                       --known_markers data/resistance_snps.txt \\
                       --output_dir results
        """
    )
    
    # Required arguments
    parser.add_argument('--input_matrix', required=True,
                       help='Path to binary genomic matrix (CSV, FASTA, or VCF)')
    
    # Analysis mode arguments (mutually exclusive groups)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--metadata',
                           help='Path to sample metadata file (for hierarchical analysis)')
    mode_group.add_argument('--phenotype_file', 
                           help='Path to phenotype classifications (for phenotype-based analysis)')
    
    # Mode-specific arguments
    parser.add_argument('--hierarchy_column',
                       help='Column name for hierarchical grouping (required with --metadata)')
    parser.add_argument('--target_groups',
                       help='Comma-separated list of target groups (for phenotype analysis)')
    
    # Optional inputs
    parser.add_argument('--known_markers',
                       help='Path to known trait-associated features')
    parser.add_argument('--config_file',
                       help='Path to YAML configuration file')
    
    # Output options
    parser.add_argument('--output_dir', default='networkparser_results',
                       help='Output directory (default: networkparser_results)')
    parser.add_argument('--output_format', default='text,json',
                       help='Output formats: text,json,xml (default: text,json)')
    
    # Analysis parameters
    parser.add_argument('--bootstrap_iterations', type=int, default=1000,
                       help='Number of bootstrap iterations (default: 1000)')
    parser.add_argument('--confidence_threshold', type=float, default=0.95,
                       help='Statistical confidence level (default: 0.95)')
    parser.add_argument('--max_interaction_order', type=int, default=2,
                       help='Maximum epistatic interaction order (default: 2)')
    parser.add_argument('--fdr_threshold', type=float, default=0.05,
                       help='False discovery rate threshold (default: 0.05)')
    parser.add_argument('--min_group_size', type=int, default=5,
                       help='Minimum samples per group (default: 5)')
    parser.add_argument('--correction_method', default='fdr_bh',
                       choices=['fdr_bh', 'bonferroni'],
                       help='Multiple testing correction method (default: fdr_bh)')
    parser.add_argument('--max_workers', type=int, default=4,
                       help='Number of parallel processing threads (default: 4)')
    
    # Boolean flags
    parser.add_argument('--memory_efficient', action='store_true',
                       help='Enable memory-efficient processing for large datasets')
    parser.add_argument('--no_matrices', action='store_false', dest='include_matrices',
                       help='Skip generating processed matrices')
    parser.add_argument('--no_plots', action='store_false', dest='generate_plots',
                       help='Skip generating plots')
    parser.add_argument('--json_output', action='store_true',
                       help='Generate JSON output (sets format to include json)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate argument combinations
    if args.metadata and not args.hierarchy_column:
        parser.error("--hierarchy_column is required when using --metadata")
    
    # Load configuration
    if args.config_file:
        config = load_config_file(args.config_file)
    else:
        config = create_config_from_args(args)
    
    # Override output format if json_output flag is set
    if args.json_output:
        config.output_formats = list(set(config.output_formats + ['json']))
    
    try:
        # Initialize and run NetworkParser
        networkparser = NetworkParser(config)
        
        results = networkparser.run_analysis(
            input_matrix=args.input_matrix,
            metadata=args.metadata,
            phenotype_file=args.phenotype_file,
            hierarchy_column=args.hierarchy_column,
            target_groups=args.target_groups,
            known_markers=args.known_markers,
            output_dir=args.output_dir
        )
        
        # Print summary
        print("\n" + "="*80)
        print("NETWORKPARSER ANALYSIS COMPLETED")
        print("="*80)
        print(f"Samples analyzed: {results['dataset_info']['n_samples']}")
        print(f"Features analyzed: {results['dataset_info']['n_features']}")
        
        significant_features = sum(1 for data in results['statistical_validation'].values() 
                                 if data['significant'])
        print(f"Significant features: {significant_features}")
        
        if 'interactions' in results:
            significant_interactions = sum(1 for data in results['interactions'].values() 
                                         if data.get('bootstrap_stable', False))
            print(f"Validated interactions: {significant_interactions}")
        
        if 'cross_validation' in results:
            cv_acc = results['cross_validation']['random_forest']['mean_accuracy']
            print(f"Cross-validation accuracy: {cv_acc:.3f}")
        
        print(f"\nResults saved to: {args.output_dir}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
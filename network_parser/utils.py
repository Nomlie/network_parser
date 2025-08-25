# networkparser/utils.py
"""
Utility functions for configuration loading and creation.
"""

import yaml
import argparse
from typing import Optional

from .config import NetworkParserConfig

def create_config_from_args(args: argparse.Namespace) -> NetworkParserConfig:
    """Create configuration from command line arguments"""
    config = NetworkParserConfig()
    
    # Update config with provided arguments
    if args.bootstrap_iterations:
        config.bootstrap_iterations = args.bootstrap_iterations
    if args.confidence_threshold:
        config.confidence_threshold = args.confidence_threshold
    if args.max_interaction_order:
        config.max_interaction_order = args.max_interaction_order
    if args.fdr_threshold:
        config.fdr_threshold = args.fdr_threshold
    if args.min_group_size:
        config.min_group_size = args.min_group_size
    if args.correction_method:
        config.correction_method = args.correction_method
    if args.max_workers:
        config.max_workers = args.max_workers
    if args.output_format:
        config.output_formats = args.output_format.split(',')
    
    # Set boolean flags
    config.memory_efficient = getattr(args, 'memory_efficient', False)
    config.include_matrices = getattr(args, 'include_matrices', True)
    config.generate_plots = getattr(args, 'generate_plots', True)
    
    return config

def load_config_file(config_path: str) -> NetworkParserConfig:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create config object
    config = NetworkParserConfig()
    
    # Update with file contents
    if 'analysis' in config_dict:
        analysis_config = config_dict['analysis']
        config.bootstrap_iterations = analysis_config.get('bootstrap_iterations', config.bootstrap_iterations)
        config.confidence_threshold = analysis_config.get('confidence_threshold', config.confidence_threshold)
        config.max_interaction_order = analysis_config.get('max_interaction_order', config.max_interaction_order)
        config.fdr_threshold = analysis_config.get('fdr_threshold', config.fdr_threshold)
    
    if 'processing' in config_dict:
        processing_config = config_dict['processing']
        config.max_workers = processing_config.get('max_workers', config.max_workers)
        config.memory_efficient = processing_config.get('memory_efficient', config.memory_efficient)
        config.chunk_size = processing_config.get('chunk_size', config.chunk_size)
    
    if 'output' in config_dict:
        output_config = config_dict['output']
        config.output_formats = output_config.get('formats', config.output_formats)
        config.include_matrices = output_config.get('include_matrices', config.include_matrices)
        config.generate_plots = output_config.get('generate_plots', config.generate_plots)
    
    if 'validation' in config_dict:
        validation_config = config_dict['validation']
        config.cross_validation_folds = validation_config.get('cross_validation_folds', config.cross_validation_folds)
        config.stability_threshold = validation_config.get('stability_threshold', config.stability_threshold)
        config.min_bootstrap_support = validation_config.get('min_bootstrap_support', config.min_bootstrap_support)
    
    return config
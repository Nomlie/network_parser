# network_parser/cli.py
import argparse
import sys
import logging
from pathlib import Path
from typing import Optional
import time
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log"),
    ],
)

logger = logging.getLogger(__name__)

try:
    from .config import NetworkParserConfig
    from .network_parser import run_networkparser_analysis
except ImportError:  # pragma: no cover
    from network_parser.config import NetworkParserConfig
    from network_parser.network_parser import run_networkparser_analysis


def load_config(config_path: Optional[str], default_config: NetworkParserConfig) -> NetworkParserConfig:
    if not config_path:
        return default_config

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    for k, v in data.items():
        if hasattr(default_config, k):
            setattr(default_config, k, v)

    default_config.__post_init__()
    return default_config


def validate_input_path(path_str: str) -> Path:
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Genomic input not found: {p}")
    return p


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NetworkParser: VCF/Matrix -> statistically validated features + interpretable networks"
    )

    input_group = parser.add_argument_group("Input Files")
    input_group.add_argument("--genomic", required=True, type=str)
    input_group.add_argument("--meta", type=str, default=None)
    input_group.add_argument("--label", required=True, type=str)
    input_group.add_argument("--known_markers", type=str, default=None)
    input_group.add_argument("--ref_fasta", type=str, default=None)

    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output_dir", required=True, type=str)

    cfg_group = parser.add_argument_group("Config")
    cfg_group.add_argument("--config", type=str, default=None)

    flags_group = parser.add_argument_group("Validation Flags")
    flags_group.add_argument(
        "--validate_statistics",
        action="store_true",
        help="Run association testing + multiple testing correction (pre-tree)",
    )
    flags_group.add_argument(
        "--validate_interactions",
        action="store_true",
        help="Run post-tree interaction permutation validation",
    )

    branch_group = parser.add_argument_group("Pipeline Branch Control")
    branch_group.add_argument(
        "--pipeline_mode",
        type=str,
        default=None,
        choices=["matrix_only", "decision_tree_only", "ml_only", "both"],
        help="Select pipeline stop/run mode",
    )
    branch_group.add_argument(
        "--run_ml_protocol",
        action="store_true",
        help="Backward-compatible shortcut: equivalent to pipeline_mode='both' unless pipeline_mode is explicitly set",
    )

    ml_group = parser.add_argument_group("ML Protocol Branch")
    ml_group.add_argument(
        "--ml_algorithm",
        type=str,
        default=None,
        choices=["auto", "RF", "MLP", "LR", "MBCS", "DT", "SVC", "SCV", "DNL"],
        help="ML protocol algorithm override",
    )
    ml_group.add_argument("--ml_min_sensitivity", type=float, default=None)
    ml_group.add_argument("--ml_max_sensitivity", type=float, default=None)
    ml_group.add_argument("--ml_step_sensitivity", type=float, default=None)
    ml_group.add_argument("--ml_empty_symbol", type=str, default=None)
    ml_group.add_argument("--ml_remove_empty_field_threshold", type=float, default=None)

    runtime_group = parser.add_argument_group("Runtime")
    runtime_group.add_argument("--n_jobs", type=int, default=None)

    return parser


def apply_cli_overrides(args: argparse.Namespace, config: NetworkParserConfig) -> NetworkParserConfig:
    if args.n_jobs is not None:
        config.n_jobs = args.n_jobs

    if args.pipeline_mode is not None:
        config.pipeline_mode = args.pipeline_mode

    # backward-compatible shortcut
    if args.run_ml_protocol and args.pipeline_mode is None:
        config.run_ml_protocol = True

    if args.ml_algorithm is not None:
        config.ml_algorithm = args.ml_algorithm
    if args.ml_min_sensitivity is not None:
        config.ml_min_sensitivity = args.ml_min_sensitivity
    if args.ml_max_sensitivity is not None:
        config.ml_max_sensitivity = args.ml_max_sensitivity
    if args.ml_step_sensitivity is not None:
        config.ml_step_sensitivity = args.ml_step_sensitivity
    if args.ml_empty_symbol is not None:
        config.ml_empty_symbol = args.ml_empty_symbol
    if args.ml_remove_empty_field_threshold is not None:
        config.ml_remove_empty_field_threshold = args.ml_remove_empty_field_threshold

    config.__post_init__()
    return config


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    genomic_path = validate_input_path(args.genomic)

    meta_path = None
    if args.meta:
        meta_p = Path(args.meta)
        if not meta_p.is_file():
            logger.error(f"Metadata file not found: {meta_p}")
            sys.exit(1)
        meta_path = meta_p

    known_markers_path = None
    if args.known_markers:
        km_p = Path(args.known_markers)
        if not km_p.is_file():
            logger.error(f"Known markers file not found: {km_p}")
            sys.exit(1)
        known_markers_path = km_p

    ref_fasta_path = None
    if args.ref_fasta:
        rf_p = Path(args.ref_fasta)
        if not rf_p.is_file():
            logger.error(f"Reference FASTA file not found: {rf_p}")
            sys.exit(1)
        ref_fasta_path = rf_p

    config = load_config(args.config, NetworkParserConfig())
    config = apply_cli_overrides(args, config)

    logger.info("Starting NetworkParser pipeline")
    logger.info(f"Genomic input:    {genomic_path.resolve()}")
    logger.info(f"Metadata:         {meta_path.resolve() if meta_path else 'None'}")
    logger.info(f"Label column:     {args.label}")
    logger.info(f"Output directory: {Path(args.output_dir).resolve()}")
    logger.info(f"Pipeline mode:    {config.pipeline_mode}")

    if config.pipeline_mode in {"ml_only", "both"}:
        logger.info(f"ML algorithm:     {config.ml_algorithm}")
        logger.info(
            "ML sensitivity:   min=%.3f max=%.3f step=%.3f",
            config.ml_min_sensitivity,
            config.ml_max_sensitivity,
            config.ml_step_sensitivity,
        )

    if ref_fasta_path:
        logger.info(f"Reference:        {ref_fasta_path.resolve()}")

    try:
        start_time = time.time()
        run_networkparser_analysis(
            genomic_path=str(genomic_path),
            meta_path=str(meta_path) if meta_path else None,
            label_column=args.label,
            known_markers_path=str(known_markers_path) if known_markers_path else None,
            output_dir=args.output_dir,
            config=config,
            validate_statistics=args.validate_statistics,
            validate_interactions=args.validate_interactions,
            ref_fasta=str(ref_fasta_path) if ref_fasta_path else None,
        )
        elapsed_time = time.time() - start_time

        total_seconds = float(elapsed_time)
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60

        if hours > 0:
            duration_str = f"{hours}h {minutes}m {seconds:.2f}s"
        elif minutes > 0:
            duration_str = f"{minutes}m {seconds:.2f}s"
        else:
            duration_str = f"{seconds:.2f}s"

        logger.info(f"NetworkParser pipeline completed successfully in {duration_str}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""User-facing entry point for NetworkParser.

Place this file in the repository root and run it with:

    python run_network_parser.py --help
    python run_network_parser.py train-hierarchy --help
    python run_network_parser.py query --help
    python run_network_parser.py evaluate --help
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LAUNCHER_PROG = "python run_network_parser.py"

GENERAL_HELP = """\
usage: python run_network_parser.py [-h] <command> ...

NetworkParser trains hierarchical genomic models and applies them to new samples.

Main commands:
  train-hierarchy     Train models from labelled genomic data
  query               Predict labels for new samples using a trained model

Other commands:
  run                 Single-label training
  bundle              Package a trained registry into a .npb model bundle
  evaluate            Evaluate saved predictions against metadata
  evaluate-hierarchy  Hierarchy evaluation pack
  cross-validate      Leakage-aware repeated cross-validation
  annotate-panels     Annotate selected feature panels
  train-two-level     Alias for train-hierarchy

Show help:
  python run_network_parser.py --help
  python run_network_parser.py train-hierarchy --help
  python run_network_parser.py query --help
  python run_network_parser.py evaluate --help

Examples:
  python run_network_parser.py train-hierarchy \\
      --genomic data/train \\
      --meta data/train_metadata.csv \\
      --hierarchy_labels Lineage_clean AMR_binary \\
      --ref_fasta data/reference/H37Rv.fasta \\
      --output_dir results/train

  python run_network_parser.py query \\
      --genomic data/test \\
      --bundle results/train/networkparser_model_bundle.npb \\
      --ref_fasta data/reference/H37Rv.fasta \\
      --output_dir results/query

  python run_network_parser.py evaluate \\
      --predictions results/query/query_predictions.csv \\
      --meta data/test_metadata.csv \\
      --label AMR_binary \\
      --output_dir results/evaluation

Optional JSON settings can be passed with --config path/to/config.json.
The launcher checks arguments, the config file, input paths, and VCF counts
before starting. Training from a VCF directory needs at least 10 files by
default (config min_sample_presence). Problems are printed as a numbered
list and the program exits.
"""


class _LauncherArgumentParser(argparse.ArgumentParser):
    """Print a short error and point the user at --help."""

    def error(self, message: str) -> None:
        sys.stderr.write(f"Error: {message}\n")
        sys.stderr.write("Use --help to see available options.\n")
        raise SystemExit(2)


def build_launcher_parser() -> argparse.ArgumentParser:
    from network_parser.cli import build_top_parser

    return build_top_parser(
        prog=LAUNCHER_PROG,
        parser_class=_LauncherArgumentParser,
    )


def _unknown_command_message(name: str) -> str:
    return f"Unknown command: {name}\n\n" "See: python run_network_parser.py --help"


def main(argv: Optional[List[str]] = None) -> int:
    from network_parser.cli import (
        VALID_SUBCOMMANDS,
        configure_logging,
        run_parsed_command,
    )

    tokens = list(sys.argv[1:] if argv is None else argv)

    if not tokens or tokens[0] in {"-h", "--help"}:
        sys.stdout.write(GENERAL_HELP)
        if not GENERAL_HELP.endswith("\n"):
            sys.stdout.write("\n")
        return 0

    command = tokens[0]
    if command not in VALID_SUBCOMMANDS:
        sys.stderr.write(_unknown_command_message(command) + "\n")
        return 2

    parser = build_launcher_parser()
    args = parser.parse_args(tokens)

    from network_parser.preflight import format_preflight_report, validate_launcher_args

    errors = validate_launcher_args(args)
    if errors:
        sys.stderr.write(
            format_preflight_report(errors, command=getattr(args, "command", None))
            + "\n"
        )
        return 2

    configure_logging(
        verbose=bool(getattr(args, "verbose", False)),
        quiet=bool(getattr(args, "quiet", False)),
    )

    try:
        run_parsed_command(args)
    except Exception as exc:
        sys.stderr.write(f"NetworkParser failed: {exc}\n")
        if bool(getattr(args, "verbose", False)):
            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

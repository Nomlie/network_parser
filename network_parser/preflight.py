#!/usr/bin/env python3
"""Preflight checks for the user-facing NetworkParser launcher.

These checks run before training or query so missing files, a bad config, or
too few VCF samples are reported in one list and the process exits cleanly.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import fields
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from network_parser.config import NetworkParserConfig
from network_parser.fastq_processor import FASTQ_EXTENSIONS
from network_parser.hierarchy_artifacts import resolve_hierarchy_labels

# Match DataLoader._load_vcf_directory: only files sitting directly in the folder.
VCF_NAME_SUFFIXES = (".vcf", ".vcf.gz")
MATRIX_SUFFIXES = (".csv", ".tsv", ".tab")
FASTA_SUFFIXES = (".fa", ".fasta", ".fna", ".fa.gz", ".fasta.gz")
GENBANK_SUFFIXES = (".gbk", ".gb", ".genbank")
REFERENCE_SUFFIXES = FASTA_SUFFIXES + GENBANK_SUFFIXES

# Query needs at least one genomic sample. Training uses min_sample_presence.
MIN_QUERY_VCF_FILES = 1
CONFIG_PATH_FIELDS = (
    "known_markers_path",
    "level2_binary_label_mapping_file",
    "fastq_panel_sites_bed",
    "fastq_panel_manifest",
)


def config_field_names() -> Set[str]:
    return {item.name for item in fields(NetworkParserConfig)}


def min_train_vcf_threshold(config: Optional[NetworkParserConfig] = None) -> int:
    """Minimum VCF count for hierarchical training.

    Defaults to ``min_sample_presence`` (10) so the cohort presence filter can
    keep variants. Always at least 2 samples.
    """
    presence = 10
    if config is not None:
        presence = int(getattr(config, "min_sample_presence", 10) or 10)
    return max(2, presence)


def is_vcf_filename(name: str) -> bool:
    lower = name.lower()
    return lower.endswith(".vcf") or lower.endswith(".vcf.gz")


def is_fastq_filename(name: str) -> bool:
    lower = name.lower()
    return any(lower.endswith(ext) for ext in FASTQ_EXTENSIONS)


def is_matrix_filename(name: str) -> bool:
    lower = name.lower()
    return any(lower.endswith(ext) for ext in MATRIX_SUFFIXES)


def is_fasta_filename(name: str) -> bool:
    lower = name.lower()
    return any(lower.endswith(ext) for ext in FASTA_SUFFIXES)


def vcf_sample_id(path: Path) -> str:
    """Sample ID implied by a VCF file name (same rules as DataLoader)."""
    name = path.name
    if name.endswith(".vcf.gz"):
        return name[: -len(".vcf.gz")]
    if name.endswith(".vcf"):
        return name[: -len(".vcf")]
    return path.stem


def discover_named_files(
    directory: Path, predicate: Callable[[str], bool]
) -> Tuple[List[Path], List[Path]]:
    """Return (readable files, unreadable matching paths) in one directory."""
    readable: List[Path] = []
    unreadable: List[Path] = []
    try:
        entries = list(directory.iterdir())
    except OSError as exc:
        raise FileNotFoundError(f"Cannot read directory {directory}: {exc}") from exc

    for path in sorted(entries, key=lambda item: item.name):
        if not predicate(path.name):
            continue
        if path.is_file() and path.stat().st_size > 0:
            readable.append(path)
        else:
            unreadable.append(path)
    return readable, unreadable


def discover_vcf_files(directory: Path) -> Tuple[List[Path], List[Path]]:
    return discover_named_files(directory, is_vcf_filename)


def discover_fastq_files(directory: Path) -> Tuple[List[Path], List[Path]]:
    return discover_named_files(directory, is_fastq_filename)


def load_and_validate_config(
    config_path: Optional[str],
) -> Tuple[NetworkParserConfig, List[str]]:
    """Load JSON config overrides. Unknown keys and invalid values are errors."""
    errors: List[str] = []
    config = NetworkParserConfig()
    if not config_path:
        if hasattr(config, "__post_init__"):
            config.__post_init__()
        return config, errors

    path = Path(config_path)
    if not path.exists():
        errors.append(f"Config file not found: {path}")
        return config, errors
    if not path.is_file():
        errors.append(f"Config path is not a file: {path}")
        return config, errors
    if path.stat().st_size == 0:
        errors.append(f"Config file is empty: {path}")
        return config, errors

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        errors.append(
            f"Config file is not valid JSON: {path} "
            f"({exc.msg} at line {exc.lineno} column {exc.colno})"
        )
        return config, errors
    except OSError as exc:
        errors.append(f"Config file could not be read: {path} ({exc})")
        return config, errors

    if not isinstance(payload, dict):
        errors.append(
            "Config file must contain a JSON object of setting names and values, "
            f"got {type(payload).__name__}: {path}"
        )
        return config, errors

    valid = config_field_names()
    unknown = [str(key) for key in payload if str(key) not in valid]
    if unknown:
        preview = ", ".join(unknown[:12])
        extra = f" (+{len(unknown) - 12} more)" if len(unknown) > 12 else ""
        errors.append(
            f"Config file {path} has unknown setting(s): {preview}{extra}. "
            "Use names from network_parser/config.py (NetworkParserConfig fields)."
        )

    for key, value in payload.items():
        if str(key) not in valid:
            continue
        setattr(config, str(key), value)

    try:
        if hasattr(config, "__post_init__"):
            config.__post_init__()
    except (TypeError, ValueError) as exc:
        errors.append(f"Config file {path} has an invalid setting: {exc}")

    if not errors:
        errors.extend(_validate_config_input_paths(config, path))
    return config, errors


def _validate_config_input_paths(
    config: NetworkParserConfig, config_path: Path
) -> List[str]:
    errors: List[str] = []
    for field_name in CONFIG_PATH_FIELDS:
        raw = getattr(config, field_name, None)
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        candidate = Path(text)
        if not candidate.is_absolute():
            relative = config_path.parent / candidate
            if relative.exists():
                candidate = relative
        if not candidate.is_file():
            errors.append(
                f"Config setting {field_name}={text!r} "
                "does not point to an existing file."
            )
    return errors


def _require_existing_file(path_value: Optional[str], label: str) -> List[str]:
    if not path_value:
        return []
    path = Path(path_value)
    if not path.exists():
        return [f"{label} not found: {path}"]
    if not path.is_file():
        return [f"{label} is not a file: {path}"]
    if path.stat().st_size == 0:
        return [f"{label} is empty: {path}"]
    return []


def _read_metadata_rows(
    path: Path,
) -> Tuple[List[str], List[Dict[str, str]], List[str]]:
    errors: List[str] = []
    suffix = "".join(path.suffixes).lower()
    delimiter = "\t" if suffix.endswith((".tsv", ".tab", ".txt")) else ","
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            headers = list(reader.fieldnames or [])
            if not headers:
                return [], [], [f"Metadata file has no header row: {path}"]
            rows = [
                {str(k): ("" if v is None else str(v)) for k, v in row.items()}
                for row in reader
            ]
    except OSError as exc:
        return [], [], [f"Metadata file could not be read: {path} ({exc})"]
    except csv.Error as exc:
        return [], [], [f"Metadata file is not a readable table: {path} ({exc})"]
    if len(headers) < 2:
        errors.append(
            f"Metadata file needs at least two columns (sample ID and labels): {path}"
        )
    if not rows:
        errors.append(f"Metadata file has a header but no sample rows: {path}")
    return headers, rows, errors


def _metadata_sample_column(headers: Sequence[str]) -> str:
    if "Sample" in headers:
        return "Sample"
    return str(headers[0])


def _validate_hierarchy_label_args(
    args: argparse.Namespace,
) -> Tuple[Optional[List[str]], List[str]]:
    errors: List[str] = []
    hierarchy_labels = getattr(args, "hierarchy_labels", None)
    preset = getattr(args, "hierarchy_preset", None)
    level1 = getattr(args, "level1_label", None)
    level2 = getattr(args, "level2_label", None)

    if hierarchy_labels or preset:
        try:
            resolved = resolve_hierarchy_labels(
                hierarchy_labels=hierarchy_labels,
                preset=preset,
            )
            return list(resolved), errors
        except ValueError as exc:
            errors.append(str(exc))
            return None, errors

    if level1 and level2:
        return [str(level1), str(level2)], errors

    errors.append(
        "train-hierarchy requires --hierarchy_labels (two or more metadata columns), "
        "or --hierarchy_preset, or both --level1_label and --level2_label."
    )
    return None, errors


def _summarize_paths(paths: Sequence[Path], limit: int = 8) -> str:
    names = [path.name for path in paths[:limit]]
    text = ", ".join(names)
    if len(paths) > limit:
        text += f" (+{len(paths) - limit} more)"
    return text


def _validate_vcf_directory(
    directory: Path,
    *,
    minimum: int,
    role: str,
    metadata_sample_ids: Optional[Set[str]] = None,
    metadata_vcf_names: Optional[Set[str]] = None,
) -> List[str]:
    errors: List[str] = []
    try:
        readable, unreadable = discover_vcf_files(directory)
    except FileNotFoundError as exc:
        return [str(exc)]

    if unreadable:
        errors.append(
            f"{role} directory {directory} has {len(unreadable)} VCF path(s) that are "
            f"unreadable or empty: {_summarize_paths(unreadable)}"
        )

    n_vcf = len(readable)
    if n_vcf < minimum:
        errors.append(
            f"{role} directory {directory} contains {n_vcf} VCF file(s); "
            f"at least {minimum} are required. Place *.vcf or *.vcf.gz files "
            "directly in this directory (subfolders are not scanned)."
        )
        return errors

    if metadata_vcf_names:
        present_names = {path.name for path in readable}
        missing = sorted(metadata_vcf_names - present_names)
        if missing:
            preview = ", ".join(missing[:8])
            extra = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
            errors.append(
                f"{role} metadata lists VCF files that are missing from {directory}: "
                f"{preview}{extra}"
            )

    if metadata_sample_ids:
        vcf_ids = {vcf_sample_id(path) for path in readable}
        overlap = vcf_ids & metadata_sample_ids
        if not overlap:
            errors.append(
                f"No sample IDs in {role} metadata match VCF file names "
                f"in {directory}. "
                "Sample IDs must match VCF names without the .vcf / .vcf.gz suffix."
            )
        elif len(overlap) < minimum:
            errors.append(
                f"{role} has {len(overlap)} sample(s) present in both "
                "metadata and VCF files; "
                f"at least {minimum} overlapping samples are required."
            )
    return errors


def _validate_genomic_training_input(
    genomic: Path,
    *,
    minimum_vcfs: int,
    metadata_sample_ids: Optional[Set[str]],
    metadata_vcf_names: Optional[Set[str]],
) -> List[str]:
    if not genomic.exists():
        return [f"Genomic input path not found: {genomic}"]

    if genomic.is_dir():
        return _validate_vcf_directory(
            genomic,
            minimum=minimum_vcfs,
            role="Training",
            metadata_sample_ids=metadata_sample_ids,
            metadata_vcf_names=metadata_vcf_names,
        )

    if genomic.is_file():
        if is_matrix_filename(genomic.name):
            if genomic.stat().st_size == 0:
                return [f"Genomic matrix file is empty: {genomic}"]
            return []
        return [
            "Training genomic input must be a directory of VCF files "
            f"or a CSV/TSV matrix. Got: {genomic}"
        ]

    return [f"Genomic input path is not a file or directory: {genomic}"]


def _detect_query_kind(
    path: Path, requested: Optional[str]
) -> Tuple[Optional[str], List[str]]:
    requested = (requested or "auto").strip().lower()
    if requested == "raw_sequence":
        requested = "fasta"

    if requested != "auto":
        return requested, []

    if path.is_dir():
        vcfs, _ = discover_vcf_files(path)
        fastqs, _ = discover_fastq_files(path)
        if vcfs:
            return "vcf", []
        if fastqs:
            return "fastq", []
        return None, [
            f"Query directory {path} has no VCF (*.vcf / *.vcf.gz) or FASTQ files. "
            "Put input files directly in this directory."
        ]

    if path.is_file():
        name = path.name
        if is_matrix_filename(name):
            return "matrix", []
        if is_fasta_filename(name):
            return "fasta", []
        if is_vcf_filename(name):
            return "vcf", []
        return None, [
            f"Could not detect query input type for {path}. "
            "Use --query_input_type with one of: matrix, vcf, fasta, fastq."
        ]

    return None, [f"Query genomic input is not a file or directory: {path}"]


def _validate_query_genomic(path: Path, query_input_type: Optional[str]) -> List[str]:
    errors: List[str] = []
    if not path.exists():
        return [f"Genomic input path not found: {path}"]

    kind, detect_errors = _detect_query_kind(path, query_input_type)
    errors.extend(detect_errors)
    if kind is None:
        return errors

    if kind == "vcf":
        if path.is_dir():
            errors.extend(
                _validate_vcf_directory(
                    path,
                    minimum=MIN_QUERY_VCF_FILES,
                    role="Query",
                )
            )
        elif is_vcf_filename(path.name):
            if path.stat().st_size == 0:
                errors.append(f"Query VCF file is empty: {path}")
        else:
            errors.append(
                "--query_input_type vcf expects a VCF file or a directory "
                f"of VCF files: {path}"
            )
        return errors

    if kind == "fastq":
        if not path.is_dir():
            errors.append(
                f"--query_input_type fastq expects a directory of FASTQ files: {path}"
            )
            return errors
        readable, unreadable = discover_fastq_files(path)
        if unreadable:
            errors.append(
                f"Query directory {path} has unreadable or empty FASTQ path(s): "
                f"{_summarize_paths(unreadable)}"
            )
        if len(readable) < 1:
            errors.append(
                f"Query directory {path} contains no FASTQ files "
                f"({', '.join(FASTQ_EXTENSIONS)})."
            )
        return errors

    if kind == "matrix":
        if not path.is_file() or not is_matrix_filename(path.name):
            errors.append(f"--query_input_type matrix expects a CSV/TSV file: {path}")
        elif path.stat().st_size == 0:
            errors.append(f"Query matrix file is empty: {path}")
        return errors

    if kind == "fasta":
        if not path.is_file():
            errors.append(f"--query_input_type fasta expects a FASTA file: {path}")
        elif path.stat().st_size == 0:
            errors.append(f"Query FASTA file is empty: {path}")
        return errors

    errors.append(f"Unsupported query input type: {kind}")
    return errors


def validate_train_hierarchy(args: argparse.Namespace) -> List[str]:
    errors: List[str] = []
    config, config_errors = load_and_validate_config(getattr(args, "config", None))
    errors.extend(config_errors)

    labels, label_errors = _validate_hierarchy_label_args(args)
    errors.extend(label_errors)

    errors.extend(_require_existing_file(getattr(args, "meta", None), "Metadata file"))
    errors.extend(
        _require_existing_file(
            getattr(args, "ref_fasta", None), "Reference FASTA/GenBank file"
        )
    )
    errors.extend(
        _require_existing_file(
            getattr(args, "level2_binary_label_mapping_file", None),
            "Level-2 binary label mapping file",
        )
    )

    errors.extend(_validate_output_dir_arg(args))

    meta_path = Path(args.meta) if getattr(args, "meta", None) else None
    metadata_sample_ids: Optional[Set[str]] = None
    metadata_vcf_names: Optional[Set[str]] = None
    if meta_path is not None and meta_path.is_file() and meta_path.stat().st_size > 0:
        headers, rows, meta_errors = _read_metadata_rows(meta_path)
        errors.extend(meta_errors)
        if headers and rows:
            sample_col = _metadata_sample_column(headers)
            metadata_sample_ids = {
                str(row.get(sample_col, "")).strip()
                for row in rows
                if str(row.get(sample_col, "")).strip()
            }
            if labels:
                missing_cols = [name for name in labels if name not in headers]
                if missing_cols:
                    errors.append(
                        f"Metadata file {meta_path} is missing hierarchy "
                        f"label column(s): {', '.join(missing_cols)}. "
                        f"Available columns: {', '.join(headers)}"
                    )
            if "vcf_file" in headers:
                metadata_vcf_names = {
                    str(row.get("vcf_file", "")).strip()
                    for row in rows
                    if str(row.get("vcf_file", "")).strip()
                }

    genomic_value = getattr(args, "genomic", None)
    if not genomic_value:
        errors.append("Missing required argument: --genomic")
    else:
        errors.extend(
            _validate_genomic_training_input(
                Path(genomic_value),
                minimum_vcfs=min_train_vcf_threshold(config),
                metadata_sample_ids=metadata_sample_ids,
                metadata_vcf_names=metadata_vcf_names,
            )
        )
    return errors


def validate_query(args: argparse.Namespace) -> List[str]:
    errors: List[str] = []
    _, config_errors = load_and_validate_config(getattr(args, "config", None))
    errors.extend(config_errors)

    registry_path = getattr(args, "registry", None)
    bundle_path = getattr(args, "bundle", None)
    if (
        registry_path
        and str(registry_path).lower().endswith(".npb")
        and not bundle_path
    ):
        bundle_path = registry_path
        registry_path = None

    if bool(registry_path) == bool(bundle_path):
        errors.append(
            "Query mode needs exactly one trained model source: "
            "--bundle networkparser_model_bundle.npb or "
            "--registry hierarchical_model_registry.json."
        )

    errors.extend(_require_existing_file(bundle_path, "Model bundle"))
    errors.extend(_require_existing_file(registry_path, "Model registry"))
    errors.extend(
        _require_existing_file(
            getattr(args, "ref_fasta", None), "Reference FASTA/GenBank file"
        )
    )
    errors.extend(_validate_output_dir_arg(args))

    genomic_value = getattr(args, "genomic", None)
    if not genomic_value:
        errors.append("Missing required argument: --genomic")
    else:
        errors.extend(
            _validate_query_genomic(
                Path(genomic_value),
                getattr(args, "query_input_type", "auto"),
            )
        )
    return errors


def _validate_output_dir_arg(args: argparse.Namespace) -> List[str]:
    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        return []
    out = Path(output_dir)
    if out.exists() and not out.is_dir():
        return [f"--output_dir exists and is not a directory: {out}"]
    return []


def _metadata_required_columns(
    meta_path: Path,
    required_columns: Sequence[str],
    *,
    kind: str = "label",
) -> List[str]:
    if not required_columns:
        return []
    if not (meta_path.is_file() and meta_path.stat().st_size > 0):
        return []
    headers, _rows, meta_errors = _read_metadata_rows(meta_path)
    if meta_errors and not headers:
        return []
    missing = [name for name in required_columns if name not in headers]
    if not missing:
        return []
    return [
        f"Metadata file {meta_path} is missing {kind} column(s): "
        f"{', '.join(missing)}. Available columns: {', '.join(headers)}"
    ]


def _validate_labelled_training(
    args: argparse.Namespace,
    required_labels: Sequence[str],
) -> List[str]:
    errors: List[str] = []
    config, config_errors = load_and_validate_config(getattr(args, "config", None))
    errors.extend(config_errors)
    errors.extend(_require_existing_file(getattr(args, "meta", None), "Metadata file"))
    errors.extend(
        _require_existing_file(
            getattr(args, "ref_fasta", None), "Reference FASTA/GenBank file"
        )
    )
    errors.extend(
        _require_existing_file(
            getattr(args, "known_markers", None), "Known-markers file"
        )
    )
    errors.extend(_validate_output_dir_arg(args))

    metadata_sample_ids: Optional[Set[str]] = None
    metadata_vcf_names: Optional[Set[str]] = None
    meta_value = getattr(args, "meta", None)
    if meta_value:
        meta_path = Path(meta_value)
        if meta_path.is_file() and meta_path.stat().st_size > 0:
            headers, rows, meta_errors = _read_metadata_rows(meta_path)
            errors.extend(meta_errors)
            if headers and rows:
                sample_col = _metadata_sample_column(headers)
                metadata_sample_ids = {
                    str(row.get(sample_col, "")).strip()
                    for row in rows
                    if str(row.get(sample_col, "")).strip()
                }
                missing_cols = [name for name in required_labels if name not in headers]
                if missing_cols:
                    errors.append(
                        f"Metadata file {meta_path} is missing label "
                        f"column(s): {', '.join(missing_cols)}. "
                        f"Available columns: {', '.join(headers)}"
                    )
                if "vcf_file" in headers:
                    metadata_vcf_names = {
                        str(row.get("vcf_file", "")).strip()
                        for row in rows
                        if str(row.get("vcf_file", "")).strip()
                    }

    genomic_value = getattr(args, "genomic", None)
    if not genomic_value:
        errors.append("Missing required argument: --genomic")
    else:
        errors.extend(
            _validate_genomic_training_input(
                Path(genomic_value),
                minimum_vcfs=min_train_vcf_threshold(config),
                metadata_sample_ids=metadata_sample_ids,
                metadata_vcf_names=metadata_vcf_names,
            )
        )
    return errors


def validate_run(args: argparse.Namespace) -> List[str]:
    label = getattr(args, "label", None)
    errors: List[str] = []
    if not label:
        errors.append("run requires --label.")
    errors.extend(_validate_labelled_training(args, [str(label)] if label else []))
    return errors


def validate_cross_validate(args: argparse.Namespace) -> List[str]:
    label = getattr(args, "label", None)
    errors: List[str] = []
    if not label:
        errors.append("cross-validate requires --label.")
    errors.extend(_validate_labelled_training(args, [str(label)] if label else []))
    return errors


def validate_bundle(args: argparse.Namespace) -> List[str]:
    errors = _require_existing_file(getattr(args, "registry", None), "Model registry")
    output = getattr(args, "output", None)
    if output:
        out = Path(output)
        if out.exists() and out.is_dir():
            errors.append(f"Bundle output path is a directory: {out}")
    return errors


def validate_evaluate(args: argparse.Namespace) -> List[str]:
    errors: List[str] = []
    errors.extend(
        _require_existing_file(getattr(args, "predictions", None), "Predictions file")
    )
    errors.extend(_require_existing_file(getattr(args, "meta", None), "Metadata file"))
    errors.extend(_validate_output_dir_arg(args))

    labels: List[str] = []
    if getattr(args, "hierarchy_labels", None):
        labels = [str(x) for x in args.hierarchy_labels]
    elif getattr(args, "label", None):
        labels = [str(args.label)]
    else:
        errors.append("evaluate requires either --label or --hierarchy_labels.")

    meta_value = getattr(args, "meta", None)
    if meta_value and labels:
        errors.extend(
            _metadata_required_columns(Path(meta_value), labels, kind="label")
        )
    return errors


def validate_evaluate_hierarchy(args: argparse.Namespace) -> List[str]:
    errors: List[str] = []
    errors.extend(
        _require_existing_file(getattr(args, "predictions", None), "Predictions file")
    )
    errors.extend(_require_existing_file(getattr(args, "meta", None), "Metadata file"))
    errors.extend(_validate_output_dir_arg(args))
    labels = [str(x) for x in (getattr(args, "hierarchy_labels", None) or []) if str(x)]
    if len(labels) < 2:
        errors.append(
            "evaluate-hierarchy requires --hierarchy_labels with at least two columns."
        )
    meta_value = getattr(args, "meta", None)
    if meta_value and labels:
        errors.extend(
            _metadata_required_columns(Path(meta_value), labels, kind="hierarchy label")
        )
    return errors


def validate_annotate_panels(args: argparse.Namespace) -> List[str]:
    errors = _require_existing_file(getattr(args, "registry", None), "Model registry")
    errors.extend(
        _require_existing_file(getattr(args, "catalogue", None), "Catalogue file")
    )
    errors.extend(
        _require_existing_file(getattr(args, "stability", None), "Stability file")
    )
    errors.extend(_validate_output_dir_arg(args))
    return errors


def validate_launcher_args(args: argparse.Namespace) -> List[str]:
    command = getattr(args, "command", None)
    validators = {
        "run": validate_run,
        "train-hierarchy": validate_train_hierarchy,
        "train-two-level": validate_train_hierarchy,
        "query": validate_query,
        "bundle": validate_bundle,
        "evaluate": validate_evaluate,
        "evaluate-hierarchy": validate_evaluate_hierarchy,
        "cross-validate": validate_cross_validate,
        "cross_validation": validate_cross_validate,
        "annotate-panels": validate_annotate_panels,
    }
    validator = validators.get(str(command) if command is not None else "")
    if validator is None:
        return [f"Unsupported command: {command}"]
    return validator(args)


def format_preflight_report(
    errors: Sequence[str],
    *,
    command: Optional[str] = None,
) -> str:
    lines = [
        "NetworkParser could not start. Please fix the following:",
        "",
    ]
    for index, message in enumerate(errors, start=1):
        lines.append(f"  {index}. {message}")
    lines.append("")
    if command:
        lines.append(f"See: python run_network_parser.py {command} --help")
    else:
        lines.append("See: python run_network_parser.py --help")
    return "\n".join(lines)

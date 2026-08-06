#!/bin/bash
# Build disjoint train/test VCF views from the authoritative sample manifests.
# Source VCFs are never moved, copied, or deleted; only symlinks are created.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_config.sh"

RAW_TRAIN_GENOMIC="${RAW_TRAIN_GENOMIC}" \
RAW_TEST_GENOMIC="${RAW_TEST_GENOMIC}" \
TRAIN_SAMPLE_MANIFEST="${TRAIN_SAMPLE_MANIFEST}" \
TEST_SAMPLE_MANIFEST="${TEST_SAMPLE_MANIFEST}" \
CLEAN_SPLIT_ROOT="${CLEAN_SPLIT_ROOT}" \
META_SOURCE="${META_SOURCE:-${META}}" \
META="${META}" \
EVALUATION_META="${EVALUATION_META}" \
"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd


def fail(message: str) -> None:
    raise SystemExit(f"INPUT PREPARATION ERROR: {message}")


def is_vcf(path: Path) -> bool:
    name = path.name.lower()
    return path.is_file() and (
        name.endswith(".vcf")
        or name.endswith(".vcf.gz")
        or name.endswith(".g.vcf")
        or name.endswith(".g.vcf.gz")
    )


def sample_id(path: Path) -> str:
    return re.sub(r"(?i)(?:\.g)?\.vcf(?:\.gz)?$", "", path.name)


def read_manifest(path: Path, expected_split: str) -> list[str]:
    if not path.is_file():
        fail(f"sample manifest not found: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    sample_column = next(
        (column for column in ("sample_id", "sample", "ID", "id") if column in frame),
        None,
    )
    if sample_column is None:
        fail(f"sample-ID column not found in {path}")
    if "split" in frame.columns:
        unexpected = sorted(
            set(frame["split"].astype(str).str.strip()) - {expected_split}
        )
        if unexpected:
            fail(f"{path} contains unexpected split values: {unexpected}")
    values = [value.strip() for value in frame[sample_column].astype(str) if value.strip()]
    if len(values) != len(set(values)):
        fail(f"duplicate sample IDs in {path}")
    return values


raw_train = Path(os.environ["RAW_TRAIN_GENOMIC"])
raw_test = Path(os.environ["RAW_TEST_GENOMIC"])
clean_root = Path(os.environ["CLEAN_SPLIT_ROOT"])
train_manifest = Path(os.environ["TRAIN_SAMPLE_MANIFEST"])
test_manifest = Path(os.environ["TEST_SAMPLE_MANIFEST"])
metadata_source_path = Path(os.environ.get("META_SOURCE") or os.environ["META"])
metadata_path = Path(os.environ["META"])
evaluation_metadata_path = Path(os.environ["EVALUATION_META"])


def add_lineage_family(frame: pd.DataFrame) -> pd.DataFrame:
    """Biologically motivated coarse lineage groups (not test Lineage_Supergroup)."""
    out = frame.copy()
    if "Lineage_clean" not in out.columns:
        return out
    mapping = {
        "lineage 1": "Indo_Oceanic",
        "lineage 2": "East_Asian",
        "lineage 3": "East_African_Indian",
        "lineage 4": "Euro_American",
        "lineage 5": "M_africanum_animal",
        "lineage 6": "M_africanum_animal",
        "lineage 7": "M_africanum_animal",
        "lineage bov_afri": "M_africanum_animal",
        "lineage bov-afri": "M_africanum_animal",
    }

    def _map(v: object) -> str:
        key = str(v or "").strip().lower()
        if not key or key in {"nan", "none", "-", "na"}:
            return ""
        return mapping.get(key, "other_or_unmapped")

    out["Lineage_family"] = out["Lineage_clean"].map(_map)
    return out

for raw_root in (raw_train, raw_test):
    if not raw_root.is_dir():
        fail(f"source VCF directory not found: {raw_root}")

train_ids = read_manifest(train_manifest, "train")
test_ids = read_manifest(test_manifest, "test")
overlap = sorted(set(train_ids) & set(test_ids))
if overlap:
    fail(f"train/test manifests overlap: {overlap[:10]}")

source_by_root: dict[Path, dict[str, Path]] = {}
for raw_root in (raw_train, raw_test):
    mapping: dict[str, Path] = {}
    for path in sorted(raw_root.rglob("*")):
        if not is_vcf(path):
            continue
        sid = sample_id(path)
        if sid in mapping:
            fail(f"duplicate VCF sample ID {sid!r} within {raw_root}")
        mapping[sid] = path.resolve()
    source_by_root[raw_root] = mapping


def build_view(
    split_name: str,
    sample_ids: list[str],
    preferred_root: Path,
    fallback_root: Path,
) -> None:
    destination = clean_root / split_name
    destination.mkdir(parents=True, exist_ok=True)
    expected_names: set[str] = set()
    missing: list[str] = []
    created = 0

    for sid in sample_ids:
        source = source_by_root[preferred_root].get(sid) or source_by_root[fallback_root].get(sid)
        if source is None:
            missing.append(sid)
            continue
        target = destination / source.name
        expected_names.add(target.name)
        if target.exists() or target.is_symlink():
            if not target.is_symlink() or target.resolve() != source:
                fail(f"unexpected existing path in clean split: {target}")
            continue
        target.symlink_to(source)
        created += 1

    if missing:
        fail(f"{len(missing)} {split_name} samples have no source VCF: {missing[:10]}")

    existing_names = {path.name for path in destination.iterdir()}
    extras = sorted(existing_names - expected_names)
    if extras:
        fail(
            f"clean {split_name} view contains stale paths: {extras[:10]}; "
            f"move them out of {destination} and rerun"
        )
    if existing_names != expected_names:
        fail(f"clean {split_name} view is incomplete")
    print(
        f"Prepared {split_name}: samples={len(sample_ids)} "
        f"new_symlinks={created} directory={destination}"
    )


build_view("train", train_ids, raw_train, raw_test)
build_view("test", test_ids, raw_test, raw_train)

# Build training + evaluation metadata from the authoritative table. Add
# Lineage_family (biological coarse groups). Source manifests stay untouched.
if not metadata_source_path.is_file():
    fail(f"authoritative metadata not found: {metadata_source_path}")
metadata = pd.read_csv(
    metadata_source_path,
    sep=None,
    engine="python",
    dtype=str,
    keep_default_na=False,
)
metadata = add_lineage_family(metadata)
metadata_sample_column = next(
    (
        column
        for column in ("sample_id", "Sample_ID", "sample", "Sample", "ID", "id")
        if column in metadata
    ),
    None,
)
if metadata_sample_column is None:
    fail(f"sample-ID column not found in authoritative metadata: {metadata_source_path}")
metadata[metadata_sample_column] = metadata[metadata_sample_column].astype(str).str.strip()
if metadata[metadata_sample_column].duplicated().any():
    duplicates = metadata.loc[
        metadata[metadata_sample_column].duplicated(keep=False), metadata_sample_column
    ].unique()
    fail(f"duplicate sample IDs in authoritative metadata: {duplicates[:10].tolist()}")

# Full enriched metadata for training (includes train + test IDs).
metadata_path.parent.mkdir(parents=True, exist_ok=True)
tmp_meta = metadata_path.with_name(f".{metadata_path.name}.tmp")
metadata.to_csv(tmp_meta, index=False)
tmp_meta.replace(metadata_path)
print(
    f"Prepared training metadata with Lineage_family: samples={len(metadata)} "
    f"path={metadata_path}"
)

metadata_by_sample = metadata.set_index(metadata_sample_column, drop=False)
missing_evaluation_ids = [sid for sid in test_ids if sid not in metadata_by_sample.index]
if missing_evaluation_ids:
    fail(
        f"{len(missing_evaluation_ids)} test samples are absent from authoritative "
        f"metadata: {missing_evaluation_ids[:10]}"
    )
evaluation_metadata = metadata_by_sample.loc[test_ids].reset_index(drop=True)
evaluation_metadata_path.parent.mkdir(parents=True, exist_ok=True)
temporary_path = evaluation_metadata_path.with_name(
    f".{evaluation_metadata_path.name}.tmp"
)
evaluation_metadata.to_csv(temporary_path, index=False)
temporary_path.replace(evaluation_metadata_path)
print(
    f"Prepared evaluation metadata: samples={len(evaluation_metadata)} "
    f"columns={len(evaluation_metadata.columns)} path={evaluation_metadata_path}"
)
PY

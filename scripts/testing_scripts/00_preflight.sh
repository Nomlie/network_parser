#!/bin/bash
# Validate shared inputs before starting a potentially long NetworkParser run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_config.sh"

requested_labels=("$@")

bash "${SCRIPT_DIR}/00_prepare_inputs.sh"

fail() {
  echo "PRE-FLIGHT ERROR: $*" >&2
  exit 2
}

[[ -d "${TRAIN_GENOMIC}" ]] || fail "Training input directory not found: ${TRAIN_GENOMIC}"
[[ -d "${TEST_GENOMIC}" ]] || fail "Test input directory not found: ${TEST_GENOMIC}"
[[ -f "${META}" ]] || fail "Metadata file not found: ${META}"
[[ -f "${EVALUATION_META}" ]] || fail "Evaluation metadata not found: ${EVALUATION_META}"
[[ -f "${REF}" ]] || fail "H37Rv GenBank file not found: ${REF}"
[[ -f "${NETWORKPARSER_CONFIG}" ]] || fail "NetworkParser config not found: ${NETWORKPARSER_CONFIG}"

if [[ "${RUN_PANEL_ANNOTATION}" == "1" ]]; then
  [[ -n "${CATALOGUE}" ]] || fail "RUN_PANEL_ANNOTATION=1 requires CATALOGUE"
  [[ -f "${CATALOGUE}" ]] || fail "Resistance catalogue not found: ${CATALOGUE}"
fi

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m network_parser.cli --help >/dev/null

TRAIN_GENOMIC="${TRAIN_GENOMIC}" \
TEST_GENOMIC="${TEST_GENOMIC}" \
META="${META}" \
EVALUATION_META="${EVALUATION_META}" \
REF="${REF}" \
CATALOGUE="${CATALOGUE}" \
RUN_PANEL_ANNOTATION="${RUN_PANEL_ANNOTATION}" \
EXPECTED_VCF_CONTIG="${EXPECTED_VCF_CONTIG}" \
EXPECTED_REFERENCE_BUILD="${EXPECTED_REFERENCE_BUILD}" \
NETWORKPARSER_CONFIG="${NETWORKPARSER_CONFIG}" \
"${PYTHON_BIN}" - "${requested_labels[@]}" <<'PY'
from __future__ import annotations

import csv
import gzip
import os
import re
import sys
from pathlib import Path

import pandas as pd

from network_parser.cli import load_config
from network_parser.vcf_call_semantics import CallState, VcfQCConfig, iter_sample_calls


def fail(message: str) -> None:
    raise SystemExit(f"PRE-FLIGHT ERROR: {message}")


def vcf_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.name.lower().endswith(".vcf")
            or path.name.lower().endswith(".vcf.gz")
            or path.name.lower().endswith(".g.vcf")
            or path.name.lower().endswith(".g.vcf.gz")
        )
    )


def sample_id(path: Path) -> str:
    return re.sub(r"(?i)(?:\.g)?\.vcf(?:\.gz)?$", "", path.name)


def first_contig(path: Path) -> str:
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("##contig=<ID="):
                return line.split("##contig=<ID=", 1)[1].split(",", 1)[0].split(">", 1)[0]
            if line and not line.startswith("#"):
                return line.split("\t", 1)[0]
    return ""


train_root = Path(os.environ["TRAIN_GENOMIC"])
test_root = Path(os.environ["TEST_GENOMIC"])
meta_path = Path(os.environ["META"])
evaluation_meta_path = Path(os.environ["EVALUATION_META"])
ref_path = Path(os.environ["REF"])
catalogue_path = Path(os.environ["CATALOGUE"]) if os.environ.get("CATALOGUE") else None
expected_contig = os.environ["EXPECTED_VCF_CONTIG"]
expected_build = os.environ["EXPECTED_REFERENCE_BUILD"]
networkparser_config_path = Path(os.environ["NETWORKPARSER_CONFIG"])

train_vcfs = vcf_paths(train_root)
test_vcfs = vcf_paths(test_root)
if not train_vcfs:
    fail(f"no VCF/VCF.GZ files found under {train_root}")
if not test_vcfs:
    fail(f"no VCF/VCF.GZ files found under {test_root}")

train_ids = {sample_id(path) for path in train_vcfs}
test_ids = {sample_id(path) for path in test_vcfs}
overlap = sorted(train_ids & test_ids)
if overlap:
    fail(f"train/test sample leakage detected: {overlap[:10]}")

meta = pd.read_csv(meta_path, sep=None, engine="python", dtype=str)
sample_column = next(
    (name for name in ("sample_id", "Sample_ID", "sample", "Sample", "ID", "id") if name in meta.columns),
    None,
)
if sample_column is None:
    fail(f"could not find a sample-ID column in metadata: {list(meta.columns)}")

missing_labels = [label for label in sys.argv[1:] if label not in meta.columns]
if missing_labels:
    fail(f"metadata is missing requested labels: {missing_labels}")

metadata_ids = set(meta[sample_column].dropna().astype(str).str.strip())
missing_train_ids = sorted(train_ids - metadata_ids)
missing_test_ids = sorted(test_ids - metadata_ids)
if missing_train_ids or missing_test_ids:
    fail(
        "VCF sample IDs missing from metadata; "
        f"train={missing_train_ids[:10]} test={missing_test_ids[:10]}"
    )

evaluation_meta = pd.read_csv(
    evaluation_meta_path,
    sep=None,
    engine="python",
    dtype=str,
    keep_default_na=False,
)
evaluation_sample_column = next(
    (
        name
        for name in ("sample_id", "Sample_ID", "sample", "Sample", "ID", "id")
        if name in evaluation_meta.columns
    ),
    None,
)
if evaluation_sample_column is None:
    fail(
        "could not find a sample-ID column in evaluation metadata: "
        f"{list(evaluation_meta.columns)}"
    )
missing_evaluation_labels = [
    label for label in sys.argv[1:] if label not in evaluation_meta.columns
]
if missing_evaluation_labels:
    fail(f"evaluation metadata is missing requested labels: {missing_evaluation_labels}")
evaluation_ids_series = evaluation_meta[evaluation_sample_column].astype(str).str.strip()
if evaluation_ids_series.duplicated().any():
    fail("evaluation metadata contains duplicate sample IDs")
evaluation_ids = set(evaluation_ids_series)
if evaluation_ids != test_ids:
    fail(
        "evaluation metadata sample set does not match held-out VCFs; "
        f"missing={sorted(test_ids - evaluation_ids)[:10]} "
        f"extra={sorted(evaluation_ids - test_ids)[:10]}"
    )

with ref_path.open(encoding="utf-8", errors="replace") as handle:
    reference_build = ""
    for line in handle:
        if line.startswith("VERSION"):
            fields = line.split()
            reference_build = fields[1] if len(fields) > 1 else ""
            break
if not reference_build:
    fail(f"could not read VERSION from GenBank file: {ref_path}")
if reference_build != expected_build:
    fail(
        f"GenBank VERSION is {reference_build!r}, expected {expected_build!r}; "
        "update EXPECTED_REFERENCE_BUILD and use a matching catalogue"
    )

observed_contigs = {first_contig(path) for path in (train_vcfs[0], test_vcfs[0])}
observed_contigs.discard("")
if observed_contigs != {expected_contig}:
    fail(
        f"representative VCF contigs are {sorted(observed_contigs)}, "
        f"expected only {expected_contig!r}"
    )

# Parse representative train/test VCFs with exactly the policy that the run
# commands will use. This catches incompatible FORMAT/QC settings before a
# full-cohort parse (the AFRO VCFs intentionally have PL but no GQ).
config = load_config(str(networkparser_config_path))
vcf_qc = VcfQCConfig.from_config(config)
qc_probe_summaries = []
for split_name, vcf_path in (("train", train_vcfs[0]), ("test", test_vcfs[0])):
    calls = iter_sample_calls(vcf_path, qc=vcf_qc)
    state_counts = {}
    reason_counts = {}
    for call in calls.by_pos.values():
        state = call.state.value
        state_counts[state] = state_counts.get(state, 0) + 1
        for reason in call.qc_reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    alternate_count = state_counts.get(CallState.CALLED_ALTERNATE.value, 0)
    if alternate_count == 0:
        top_reasons = sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
        fail(
            f"{split_name} VCF QC probe retained zero alternate calls for {vcf_path.name}; "
            f"states={state_counts}, top_qc_reasons={top_reasons}, "
            f"config={networkparser_config_path}"
        )
    qc_probe_summaries.append(
        f"{split_name}:{vcf_path.name}:alt={alternate_count}:states={state_counts}"
    )

if config.min_gq_per_sample > 0:
    fail(
        "AFRO VCFs do not provide GQ; set min_gq_per_sample to 0 in "
        f"{networkparser_config_path} or regenerate the VCFs with GQ"
    )
if not config.assume_absent_variant_is_reference:
    fail(
        "AFRO inputs are variant-only VCFs; explicitly set "
        f"assume_absent_variant_is_reference=true in {networkparser_config_path}"
    )

catalogue_rows = 0
if os.environ.get("RUN_PANEL_ANNOTATION") == "1" and catalogue_path is not None:
    catalogue = pd.read_csv(catalogue_path, sep="\t", dtype=str, keep_default_na=False)
    required = {"Position", "Ref", "Alt", "Gene", "Drug", "Contig", "Reference_build"}
    missing_columns = sorted(required - set(catalogue.columns))
    if missing_columns:
        fail(f"catalogue is missing columns: {missing_columns}")
    if not (catalogue["Contig"] == expected_contig).any():
        fail(f"catalogue has no rows for VCF contig {expected_contig!r}")
    if not (catalogue["Reference_build"] == reference_build).any():
        fail(f"catalogue has no rows for GenBank build {reference_build!r}")
    duplicate_count = int(
        catalogue.duplicated(
            ["Position", "Ref", "Alt", "Contig", "Reference_build"]
        ).sum()
    )
    if duplicate_count:
        fail(f"catalogue contains {duplicate_count} duplicate exact variant keys")
    catalogue_rows = int(len(catalogue))

print("NetworkParser validation preflight passed")
print(f"  project_root={Path.cwd()}")
print(f"  train_vcfs={len(train_vcfs)} test_vcfs={len(test_vcfs)}")
print(f"  metadata_rows={len(meta)} sample_id_column={sample_column}")
print(
    f"  evaluation_metadata_rows={len(evaluation_meta)} "
    f"sample_id_column={evaluation_sample_column}"
)
print(f"  requested_labels={','.join(sys.argv[1:])}")
print(f"  vcf_contig={expected_contig}")
print(f"  reference_build={reference_build}")
print(f"  networkparser_config={networkparser_config_path}")
print(f"  min_gq_per_sample={config.min_gq_per_sample} (GQ unavailable; GQ filter disabled)")
print(f"  assume_absent_variant_is_reference={config.assume_absent_variant_is_reference}")
for summary in qc_probe_summaries:
    print(f"  vcf_qc_probe={summary}")
if catalogue_rows:
    print(f"  resistance_catalogue_rows={catalogue_rows}")
PY

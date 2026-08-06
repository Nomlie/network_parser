#!/bin/bash
# =============================================================================
# NetworkParser CHPC / lustre configuration (cluster twin of testing_scripts).
# Paths target:
#   Data:    /mnt/lustre/users/nmfuphi/phDproject/Data/{meta,training,test}
#   Results: /mnt/lustre/users/nmfuphi/phDproject/Results
# Code:      /home/nmfuphi/network_parser  (override PROJECT_ROOT if needed)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Repo root on the login/compute node. Default matches your PBS example.
# When scripts live inside a git checkout, this is scripts/../.. from SCRIPT_DIR.
if [[ -n "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="${PROJECT_ROOT}"
elif [[ -d "/home/nmfuphi/network_parser" ]]; then
  PROJECT_ROOT="/home/nmfuphi/network_parser"
else
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

# Prefer conda env python after `conda activate networkparser` in PBS wrappers.
PYTHON_BIN="${PYTHON_BIN:-python}"

# --- Lustre data layout ---
DATA_ROOT="${DATA_ROOT:-/mnt/lustre/users/nmfuphi/phDproject/Data}"
RESULTS_ROOT="${RESULTS_ROOT:-/mnt/lustre/users/nmfuphi/phDproject/Results}"

# Unsplit AFRO VCF pool already on lustre (no re-transfer of VCFs).
# 00_prepare_inputs.sh builds train/test symlink views from the manifests.
AFRO_VCF_POOL="${AFRO_VCF_POOL:-/mnt/lustre/users/nmfuphi/AFRO_TB/AFRO_TB_ANNOTATION_VCF}"
# Both roots may point at the same pool; manifests decide membership.
RAW_TRAIN_GENOMIC="${RAW_TRAIN_GENOMIC:-${AFRO_VCF_POOL}}"
RAW_TEST_GENOMIC="${RAW_TEST_GENOMIC:-${AFRO_VCF_POOL}}"

# Sample manifests + phenotype table under Data/meta
META_DIR="${META_DIR:-${DATA_ROOT}/meta}"
TRAIN_SAMPLE_MANIFEST="${TRAIN_SAMPLE_MANIFEST:-${META_DIR}/train_samples.csv}"
TEST_SAMPLE_MANIFEST="${TEST_SAMPLE_MANIFEST:-${META_DIR}/test_samples.csv}"
META_SOURCE="${META_SOURCE:-${META_DIR}/AFRO_dataset_meta_with_test_hierarchy.csv}"
CATALOGUE="${CATALOGUE:-${META_DIR}/resistance_catalogue.tsv}"

# Clean disjoint symlink views + derived metadata (written on lustre)
CLEAN_SPLIT_ROOT="${CLEAN_SPLIT_ROOT:-${DATA_ROOT}/networkparser_manifest_split}"
TRAIN_GENOMIC="${CLEAN_SPLIT_ROOT}/train"
TEST_GENOMIC="${CLEAN_SPLIT_ROOT}/test"
META="${CLEAN_SPLIT_ROOT}/metadata_with_lineage_family.csv"
EVALUATION_META="${CLEAN_SPLIT_ROOT}/test_evaluation_metadata.csv"

# H37Rv GenBank: place under Data/ref or override REF=
REF="${REF:-${DATA_ROOT}/ref/H37Rv.gbk}"

# Resume hierarchy training when node_summary.json + model already exist.
HIERARCHY_RESUME="${HIERARCHY_RESUME:-0}"
# Global (cohort-wide) fallbacks: none | terminal | lineage | legacy | comma list
GLOBAL_FALLBACK_LABELS="${GLOBAL_FALLBACK_LABELS:-none}"

# Identity contract for AFRO VCFs + H37Rv GenBank
EXPECTED_VCF_CONTIG="${EXPECTED_VCF_CONTIG:-M.tuberculosis_H37Rv}"
EXPECTED_REFERENCE_BUILD="${EXPECTED_REFERENCE_BUILD:-AL123456.3}"

# --- Output root (all experiment RUN_DIR under here) ---
BASE_OUT="${BASE_OUT:-${RESULTS_ROOT}/All_VCFs/chi2_fdr}"
# PBS stdout/stderr for job wrappers
PBS_LOG_DIR="${PBS_LOG_DIR:-${RESULTS_ROOT}/pbs_logs}"

# --- Training / query (match typical CHPC select=1:ncpus=24:mem=120GB) ---
N_JOBS="${N_JOBS:-24}"
FILTER="${FILTER:-chi2_fdr}"
QUERY_INPUT_TYPE="${QUERY_INPUT_TYPE:-vcf}"
RANDOM_STATE="${RANDOM_STATE:-42}"

# AFRO VCF callability policy (GQ absent; absence-as-reference)
# Experiment 01 defaults to a runtime seed config when SEED_KNOWN_MARKERS=1.
NETWORKPARSER_CONFIG="${NETWORKPARSER_CONFIG:-${SCRIPT_DIR}/afro_vcf_config.json}"
# WHO catalogue seed for phenotype stages (01 default on; set 0 for control)
SEED_KNOWN_MARKERS="${SEED_KNOWN_MARKERS:-1}"

# --- Leakage-aware cross-validation ---
# Default OFF in main train/query jobs. Run pbs_11_leakage_aware_cv.pbs instead.
RUN_LEAKAGE_AWARE_CV="${RUN_LEAKAGE_AWARE_CV:-0}"
CV_REPEATS="${CV_REPEATS:-5}"
CV_SPLITS="${CV_SPLITS:-5}"
CV_PANEL_SIZES="${CV_PANEL_SIZES:-100,200,500,1000}"
CV_ALGORITHM="${CV_ALGORITHM:-}"

# --- Panel annotation ---
RUN_PANEL_ANNOTATION="${RUN_PANEL_ANNOTATION:-1}"
STABILITY_TSV="${STABILITY_TSV:-}"
MIN_STABILITY="${MIN_STABILITY:-0.0}"

# --- PRJCA040523 external (lustre after rsync) ---
PRJCA_ROOT="${PRJCA_ROOT:-${DATA_ROOT}/PRJCA040523}"
PRJCA_QUERY_CONFIG="${PRJCA_QUERY_CONFIG:-${SCRIPT_DIR}/prjca_panel_bcftools_config.json}"

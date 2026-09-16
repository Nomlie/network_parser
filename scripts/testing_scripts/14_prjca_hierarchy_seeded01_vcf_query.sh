#!/bin/bash
# =============================================================================
# Local Mac: full hierarchy (seeded_01) on existing Chinese VCFs + evaluate.
# Hierarchy: Lineage_clean → AMR_binary → Resistance_Profile_Collapsed
# No FASTQ re-calling.
#
#   bash scripts/testing_scripts/14_prjca_hierarchy_seeded01_vcf_query.sh
#   LIMIT=10 N_JOBS=8 bash scripts/testing_scripts/14_prjca_hierarchy_seeded01_vcf_query.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_config.sh"

cd "${PROJECT_ROOT}"

export PROJECT_ROOT
export PYTHON_BIN="${PYTHON_BIN:-python}"
export N_JOBS="${N_JOBS:-8}"
export LIMIT="${LIMIT:-}"
export SKIP_QUERY="${SKIP_QUERY:-0}"
export SEED_RUN_NAME="${SEED_RUN_NAME:-Hierarchy_Lineage_AMR_Resistance_Profile_seeded_01}"
export MODEL_RUN="${MODEL_RUN:-${BASE_OUT}/${SEED_RUN_NAME}}"
export BUNDLE="${BUNDLE:-${MODEL_RUN}/networkparser_model_bundle.npb}"
export RESULTS_ROOT="${BASE_OUT}"
export REF="${REF}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PHD_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
PRJCA_SCRIPT="${PRJCA_SCRIPT:-${PHD_ROOT}/Data/PRJCA040523/scripts/query_hierarchy_seeded01_chinese_vcfs.sh}"
# Always use PRJCA VCF query config unless the caller set PRJCA_NETWORKPARSER_CONFIG.
# (Do not inherit AFRO train config from 00_config.sh.)
export NETWORKPARSER_CONFIG="${PRJCA_NETWORKPARSER_CONFIG:-${PHD_ROOT}/Data/PRJCA040523/scripts/prjca_vcf_query_config.json}"

[[ -f "${PRJCA_SCRIPT}" ]] || { echo "Missing: ${PRJCA_SCRIPT}" >&2; exit 2; }
[[ -f "${BUNDLE}" ]] || { echo "Missing bundle: ${BUNDLE}" >&2; exit 2; }
[[ -f "${NETWORKPARSER_CONFIG}" ]] || { echo "Missing config: ${NETWORKPARSER_CONFIG}" >&2; exit 2; }

echo "=== 14: hierarchy seeded_01 → Chinese VCFs ==="
echo "BUNDLE=${BUNDLE}"
echo "CONFIG=${NETWORKPARSER_CONFIG}"
echo "PRJCA_SCRIPT=${PRJCA_SCRIPT}"
bash "${PRJCA_SCRIPT}"

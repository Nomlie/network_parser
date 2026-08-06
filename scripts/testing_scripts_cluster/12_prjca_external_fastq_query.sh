#!/bin/bash
# =============================================================================
# PRJCA040523 external FASTQ query + evaluate (CHPC).
# Uses panel_bcftools (trained-site only calling) — not whole-genome bcftools.
#
# Prerequisites:
#   - Seeded (or other) AFRO hierarchy bundle already trained
#   - PRJCA data on lustre: Data/PRJCA040523/{CRA025985,meta,ref}
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_config.sh"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_env_cluster.sh" 2>/dev/null || true

cd "${PROJECT_ROOT}"

PRJCA_ROOT="${PRJCA_ROOT:-${DATA_ROOT}/PRJCA040523}"
FASTQ_ROOT="${FASTQ_ROOT:-${PRJCA_ROOT}/CRA025985}"
META_DIR_PRJCA="${META_DIR_PRJCA:-${PRJCA_ROOT}/meta}"
PHENOTYPE_META="${PHENOTYPE_META:-${META_DIR_PRJCA}/networkparser_metadata.csv}"
EVAL_META_PRJCA="${EVAL_META_PRJCA:-${META_DIR_PRJCA}/networkparser_evaluation_metadata.csv}"
CRR_MAP="${CRR_MAP:-${META_DIR_PRJCA}/crr_to_strain_map.csv}"
STAGE_DIR="${STAGE_DIR:-${PRJCA_ROOT}/query_stage_fastq}"
REF_FASTA="${REF_FASTA_PRJCA:-${PRJCA_ROOT}/ref/H37Rv_M.tuberculosis_H37Rv.fasta}"
REF_GBK="${REF_GBK:-${REF}}"

# Bundle from seeded train run by default
SEED_RUN_NAME="${SEED_RUN_NAME:-Hierarchy_Lineage_AMR_Resistance_Profile_seeded_01}"
BUNDLE="${BUNDLE:-${BASE_OUT}/${SEED_RUN_NAME}/networkparser_model_bundle.npb}"

WORK_ROOT="${WORK_ROOT:-${RESULTS_ROOT}/PRJCA040523_external}"
RUN_NAME="${RUN_NAME:-Hierarchy_Lineage_AMR_Resistance_Profile_external_panel_bcftools}"
RUN_DIR="${RUN_DIR:-${WORK_ROOT}/${RUN_NAME}}"

NETWORKPARSER_CONFIG="${PRJCA_QUERY_CONFIG:-${SCRIPT_DIR}/prjca_panel_bcftools_config.json}"
HIER_LABELS=(Lineage_clean AMR_binary Resistance_Profile_Collapsed)
LIMIT="${LIMIT:-}"
RENAME_TO_STRAIN="${RENAME_TO_STRAIN:-1}"
SKIP_QUERY="${SKIP_QUERY:-0}"
SKIP_STAGE="${SKIP_STAGE:-0}"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${RUN_DIR}/query" "${RUN_DIR}/evaluate" "${STAGE_DIR}" "${PBS_LOG_DIR}"

echo "=== PRJCA040523 external FASTQ (panel_bcftools) ==="
echo "FASTQ_ROOT=${FASTQ_ROOT}"
echo "BUNDLE=${BUNDLE}"
echo "CONFIG=${NETWORKPARSER_CONFIG} (fastq_call_mode=panel_bcftools)"
echo "RUN_DIR=${RUN_DIR}"
echo "N_JOBS=${N_JOBS} LIMIT=${LIMIT:-all}"

[[ -d "${FASTQ_ROOT}" ]] || { echo "Missing FASTQ_ROOT: ${FASTQ_ROOT}" >&2; exit 2; }
[[ -f "${PHENOTYPE_META}" ]] || { echo "Missing phenotype meta: ${PHENOTYPE_META}" >&2; exit 2; }
[[ -f "${BUNDLE}" ]] || { echo "Missing model bundle: ${BUNDLE}" >&2; exit 2; }
[[ -f "${NETWORKPARSER_CONFIG}" ]] || { echo "Missing config: ${NETWORKPARSER_CONFIG}" >&2; exit 2; }

if [[ ! -f "${REF_FASTA}" ]]; then
  echo "Missing REF_FASTA: ${REF_FASTA}" >&2
  echo "Place H37Rv FASTA with header M.tuberculosis_H37Rv under PRJCA ref/." >&2
  exit 2
fi

for t in bwa samtools bcftools; do
  command -v "${t}" >/dev/null 2>&1 || {
    echo "Missing tool on PATH: ${t}" >&2
    exit 3
  }
done

REF_FOR_NP="${REF_FASTA}"
if [[ -f "${REF_GBK}" ]]; then
  REF_FOR_NP="${REF_GBK}"
fi

# Stage paired FASTQs (optional rename CRR → Strain_ID)
if [[ "${SKIP_STAGE}" != "1" ]]; then
  STAGE_PY=""
  for cand in \
    "${PRJCA_ROOT}/scripts/stage_fastq_query_dir.py" \
    "${DATA_ROOT}/PRJCA040523/scripts/stage_fastq_query_dir.py"
  do
    if [[ -f "${cand}" ]]; then
      STAGE_PY="${cand}"
      break
    fi
  done

  if [[ -n "${STAGE_PY}" ]]; then
    stage_cmd=(
      "${PYTHON_BIN}" "${STAGE_PY}"
      --fastq-root "${FASTQ_ROOT}"
      --stage-dir "${STAGE_DIR}"
    )
    if [[ "${RENAME_TO_STRAIN}" == "1" && -f "${CRR_MAP}" ]]; then
      stage_cmd+=(--crr-map "${CRR_MAP}" --rename-to-strain)
    fi
    if [[ -n "${LIMIT}" ]]; then
      stage_cmd+=(--limit "${LIMIT}")
    fi
    echo "Staging FASTQs → ${STAGE_DIR}"
    "${stage_cmd[@]}" || {
      echo "WARNING: staging failed; falling back to FASTQ_ROOT=${FASTQ_ROOT}" >&2
      STAGE_DIR="${FASTQ_ROOT}"
    }
  else
    echo "WARNING: stage_fastq_query_dir.py not found; using FASTQ_ROOT directly."
    STAGE_DIR="${FASTQ_ROOT}"
  fi
fi

QUERY_GENOMIC="${STAGE_DIR}"

if [[ "${SKIP_QUERY}" != "1" ]]; then
  echo "Querying with panel_bcftools ..."
  "${PYTHON_BIN}" -m network_parser.cli query \
    --config "${NETWORKPARSER_CONFIG}" \
    --genomic "${QUERY_GENOMIC}" \
    --bundle "${BUNDLE}" \
    --ref_fasta "${REF_FOR_NP}" \
    --query_input_type fastq \
    --output_dir "${RUN_DIR}/query" \
    --n_jobs "${N_JOBS}"
fi

EVAL_META_USE="${EVAL_META_PRJCA}"
if [[ ! -f "${EVAL_META_USE}" ]]; then
  EVAL_META_USE="${PHENOTYPE_META}"
fi

echo "Evaluating hierarchy vs ${EVAL_META_USE}"
"${PYTHON_BIN}" -m network_parser.cli evaluate-hierarchy \
  --predictions "${RUN_DIR}/query/query_predictions.csv" \
  --meta "${EVAL_META_USE}" \
  --hierarchy_labels "${HIER_LABELS[@]}" \
  --output_dir "${RUN_DIR}/evaluate" \
  --harmonize_resistance_labels

echo "Done: ${RUN_DIR}"

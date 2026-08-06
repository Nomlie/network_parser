#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_config.sh"

bash "${SCRIPT_DIR}/00_preflight.sh" Lineage_clean AMR_binary
cd "${PROJECT_ROOT}"

if [[ -n "${CATALOGUE}" && "${CATALOGUE}" != /* ]]; then
  CATALOGUE="${SCRIPT_DIR}/${CATALOGUE}"
fi
if [[ -n "${STABILITY_TSV}" && "${STABILITY_TSV}" != /* ]]; then
  STABILITY_TSV="${SCRIPT_DIR}/${STABILITY_TSV}"
fi

RUN_NAME="Two_levels_with_Lineage_AMR_02"
RUN_DIR="${BASE_OUT}/${RUN_NAME}"

mkdir -p \
  "${RUN_DIR}" \
  "${RUN_DIR}/query" \
  "${RUN_DIR}/evaluate" \
  "${RUN_DIR}/validate_cv" \
  "${RUN_DIR}/panel_annotation"

# ---------------------------------------------------------------------------
# 1. Train
# ---------------------------------------------------------------------------
"${PYTHON_BIN}" -m network_parser.cli train-hierarchy \
  --config "${NETWORKPARSER_CONFIG}" \
  --genomic "${TRAIN_GENOMIC}" \
  --meta "${META}" \
  --level1_label Lineage_clean \
  --level2_label AMR_binary \
  --central_feature_filter_method "${FILTER}" \
  --ref_fasta "${REF}" \
  --output_dir "${RUN_DIR}" \
  --n_jobs "${N_JOBS}"

# ---------------------------------------------------------------------------
# 2. Query held-out samples
# ---------------------------------------------------------------------------
"${PYTHON_BIN}" -m network_parser.cli query \
  --config "${NETWORKPARSER_CONFIG}" \
  --genomic "${TEST_GENOMIC}" \
  --bundle "${RUN_DIR}/networkparser_model_bundle.npb" \
  --ref_fasta "${REF}" \
  --query_input_type "${QUERY_INPUT_TYPE}" \
  --output_dir "${RUN_DIR}/query" \
  --n_jobs "${N_JOBS}"

# ---------------------------------------------------------------------------
# 3. Evaluate
# ---------------------------------------------------------------------------
"${PYTHON_BIN}" -m network_parser.cli evaluate \
  --predictions "${RUN_DIR}/query/query_predictions.csv" \
  --meta "${EVALUATION_META}" \
  --label Lineage_clean \
  --predicted_column predicted_level1_identity \
  --output_dir "${RUN_DIR}/evaluate/level1_Lineage_clean"

"${PYTHON_BIN}" -m network_parser.cli evaluate \
  --predictions "${RUN_DIR}/query/query_predictions.csv" \
  --meta "${EVALUATION_META}" \
  --label AMR_binary \
  --predicted_column predicted_level2_identity \
  --output_dir "${RUN_DIR}/evaluate/level2_AMR_binary"

# ---------------------------------------------------------------------------
# 4. Leakage-aware cross-validation (training partition only)
# ---------------------------------------------------------------------------
if [[ "${RUN_LEAKAGE_AWARE_CV}" == "1" ]]; then
  cv_algorithm_args=()
  if [[ -n "${CV_ALGORITHM}" ]]; then
    cv_algorithm_args+=(--algorithm "${CV_ALGORITHM}")
  fi
  for label in Lineage_clean AMR_binary; do
    out="${RUN_DIR}/validate_cv/${label}"
    mkdir -p "${out}"
    echo "Running cross-validate for label: ${label}"
    "${PYTHON_BIN}" -m network_parser.cli cross-validate \
      --config "${NETWORKPARSER_CONFIG}" \
      --genomic "${TRAIN_GENOMIC}" \
      --meta "${META}" \
      --label "${label}" \
      --central_feature_filter_method "${FILTER}" \
      --feature_panel_check on \
      --feature_panel_sizes "${CV_PANEL_SIZES}" \
      --n_repeats "${CV_REPEATS}" \
      --n_splits "${CV_SPLITS}" \
      --random_state "${RANDOM_STATE}" \
      --n_jobs "${N_JOBS}" \
      --ref_fasta "${REF}" \
      --output_dir "${out}" \
      "${cv_algorithm_args[@]}"
  done
fi

# ---------------------------------------------------------------------------
# 5. Annotate selected feature panels (genes / consequences / catalogue)
# ---------------------------------------------------------------------------
if [[ "${RUN_PANEL_ANNOTATION}" == "1" ]]; then
  registry="${RUN_DIR}/hierarchical_model_registry.json"
  if [[ ! -f "${registry}" ]]; then
    registry="${RUN_DIR}/two_level_model_registry.json"
  fi
  [[ -f "${registry}" ]] || {
    echo "Model registry not found under ${RUN_DIR}" >&2
    exit 2
  }
  out_dir="${RUN_DIR}/panel_annotation"
  mkdir -p "${out_dir}"

  cmd=(
    "${PYTHON_BIN}" -m network_parser.cli annotate-panels
    --registry "${registry}"
    --output_dir "${out_dir}"
    --min_stability "${MIN_STABILITY}"
  )

  if [[ -n "${CATALOGUE}" && -f "${CATALOGUE}" ]]; then
    cmd+=(--catalogue "${CATALOGUE}")
    echo "Using resistance catalogue: ${CATALOGUE}"
  else
    echo "No catalogue file; annotating genes/consequences only"
  fi

  if [[ -n "${STABILITY_TSV}" && -f "${STABILITY_TSV}" ]]; then
    cmd+=(--stability "${STABILITY_TSV}")
    echo "Using CV stability table: ${STABILITY_TSV} (min_stability=${MIN_STABILITY})"
  fi

  echo "Running panel annotation | registry=${registry} | out=${out_dir}"
  "${cmd[@]}"
fi

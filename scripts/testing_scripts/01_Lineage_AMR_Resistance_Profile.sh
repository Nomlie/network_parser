#!/bin/bash
# =============================================================================
# Biological 3-level hierarchy:
#   Lineage_clean → AMR_binary → Resistance_Profile_Collapsed
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_config.sh"

bash "${SCRIPT_DIR}/00_prepare_inputs.sh"
bash "${SCRIPT_DIR}/00_preflight.sh" \
  Lineage_clean AMR_binary Resistance_Profile_Collapsed
cd "${PROJECT_ROOT}"

if [[ -n "${CATALOGUE}" && "${CATALOGUE}" != /* ]]; then
  CATALOGUE="${SCRIPT_DIR}/${CATALOGUE}"
fi
if [[ -n "${STABILITY_TSV}" && "${STABILITY_TSV}" != /* ]]; then
  STABILITY_TSV="${SCRIPT_DIR}/${STABILITY_TSV}"
fi

RUN_NAME="Hierarchy_Lineage_AMR_Resistance_Profile_01"
RUN_DIR="${BASE_OUT}/${RUN_NAME}"
HIER_LABELS=(Lineage_clean AMR_binary Resistance_Profile_Collapsed)

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
  --hierarchy_labels "${HIER_LABELS[@]}" \
  --hierarchy_preset lineage_amr_profile \
  --global_fallback_labels terminal \
  --central_feature_filter_method "${FILTER}" \
  --ref_fasta "${REF}" \
  --output_dir "${RUN_DIR}" \
  --n_jobs "${N_JOBS}" \
  $( [[ "${HIERARCHY_RESUME:-0}" == "1" ]] && echo --hierarchy_resume )

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
# 3. Evaluate (hierarchy pack: confusions, e2e, bootstrap CIs)
# ---------------------------------------------------------------------------
"${PYTHON_BIN}" -m network_parser.cli evaluate-hierarchy \
  --predictions "${RUN_DIR}/query/query_predictions.csv" \
  --meta "${EVALUATION_META}" \
  --hierarchy_labels "${HIER_LABELS[@]}" \
  --output_dir "${RUN_DIR}/evaluate" \
  --harmonize_resistance_labels

# ---------------------------------------------------------------------------
# 4. Leakage-aware cross-validation (training partition only)
# ---------------------------------------------------------------------------
if [[ "${RUN_LEAKAGE_AWARE_CV}" == "1" ]]; then
  cv_algorithm_args=()
  if [[ -n "${CV_ALGORITHM}" ]]; then
    cv_algorithm_args+=(--algorithm "${CV_ALGORITHM}")
  fi
  for label in "${HIER_LABELS[@]}"; do
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
  # Prefer resistance-profile stability for AMR-related panel filtering when present
  if [[ -f "${RUN_DIR}/validate_cv/Resistance_Profile_Collapsed/cv_feature_stability.tsv" ]]; then
    STABILITY_TSV="${RUN_DIR}/validate_cv/Resistance_Profile_Collapsed/cv_feature_stability.tsv"
  fi
fi

# ---------------------------------------------------------------------------
# 5. Panel annotation + stable subset + catalogue circularity
# ---------------------------------------------------------------------------
if [[ "${RUN_PANEL_ANNOTATION}" == "1" ]]; then
  registry="${RUN_DIR}/hierarchical_model_registry.json"
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
    --write_stable_report
    --write_catalogue_circularity
  )

  if [[ -n "${CATALOGUE}" && -f "${CATALOGUE}" ]]; then
    cmd+=(--catalogue "${CATALOGUE}")
    echo "Using resistance catalogue: ${CATALOGUE}"
  fi

  if [[ -n "${STABILITY_TSV}" && -f "${STABILITY_TSV}" ]]; then
    cmd+=(--stability "${STABILITY_TSV}")
    echo "Using CV stability table: ${STABILITY_TSV} (min_stability=${MIN_STABILITY})"
  fi

  echo "Running panel annotation | registry=${registry} | out=${out_dir}"
  "${cmd[@]}"
fi

echo "Done: ${RUN_DIR}"

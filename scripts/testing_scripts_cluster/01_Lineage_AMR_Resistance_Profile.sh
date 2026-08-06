#!/bin/bash
# =============================================================================
# Biological 3-level hierarchy (CHPC):
#   Lineage_clean → AMR_binary → Resistance_Profile_Collapsed
#
# Default: WHO catalogue seeded into phenotype panels (AMR / profile stages).
# Leakage-aware CV is OFF here — submit pbs_11_leakage_aware_cv.pbs separately.
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

HIER_LABELS=(Lineage_clean AMR_binary Resistance_Profile_Collapsed)
SEED_KNOWN_MARKERS="${SEED_KNOWN_MARKERS:-1}"

# Resolve train config: seeded phenotype panels (default) or pure statistical.
if [[ "${SEED_KNOWN_MARKERS}" == "1" || "${SEED_KNOWN_MARKERS}" == "true" ]]; then
  RUN_NAME="${RUN_NAME_OVERRIDE:-Hierarchy_Lineage_AMR_Resistance_Profile_seeded_01}"
  # Build a node-local config so known_markers_path always matches CATALOGUE.
  SEED_CFG_DIR="${BASE_OUT}/.runtime_configs"
  mkdir -p "${SEED_CFG_DIR}"
  NETWORKPARSER_CONFIG="${SEED_CFG_DIR}/afro_seed_known_markers.runtime.json"
  cat > "${NETWORKPARSER_CONFIG}" <<EOF
{
  "min_gq_per_sample": 0,
  "assume_absent_variant_is_reference": true,
  "seed_known_markers": true,
  "known_markers_path": "${CATALOGUE}",
  "seed_known_markers_mode": "force_include",
  "seed_known_markers_stage_substrings": "amr,resistance,pheno,profile,resistant,susceptible",
  "seed_known_markers_max": null
}
EOF
  echo "Known-marker seed ENABLED | catalogue=${CATALOGUE}"
  echo "Runtime config: ${NETWORKPARSER_CONFIG}"
else
  RUN_NAME="${RUN_NAME_OVERRIDE:-Hierarchy_Lineage_AMR_Resistance_Profile_01}"
  NETWORKPARSER_CONFIG="${NETWORKPARSER_CONFIG:-${SCRIPT_DIR}/afro_vcf_config.json}"
  echo "Known-marker seed DISABLED | config=${NETWORKPARSER_CONFIG}"
fi

RUN_DIR="${BASE_OUT}/${RUN_NAME}"

mkdir -p \
  "${RUN_DIR}" \
  "${RUN_DIR}/query" \
  "${RUN_DIR}/evaluate" \
  "${RUN_DIR}/validate_cv" \
  "${RUN_DIR}/panel_annotation"

cat > "${RUN_DIR}/experiment_manifest.json" <<EOF
{
  "experiment": "01_Lineage_AMR_Resistance_Profile",
  "seed_known_markers": $( [[ "${SEED_KNOWN_MARKERS}" == "1" || "${SEED_KNOWN_MARKERS}" == "true" ]] && echo true || echo false ),
  "catalogue": "${CATALOGUE}",
  "hierarchy_labels": ["Lineage_clean", "AMR_binary", "Resistance_Profile_Collapsed"],
  "config": "${NETWORKPARSER_CONFIG}",
  "run_dir": "${RUN_DIR}",
  "cv_in_this_job": false,
  "note": "Leakage-aware CV: qsub pbs_11_leakage_aware_cv.pbs with RUN_DIR or RUN_NAME set."
}
EOF

# ---------------------------------------------------------------------------
# 1. Train
# ---------------------------------------------------------------------------
"${PYTHON_BIN}" -m network_parser.cli train-hierarchy \
  --config "${NETWORKPARSER_CONFIG}" \
  --genomic "${TRAIN_GENOMIC}" \
  --meta "${META}" \
  --hierarchy_labels "${HIER_LABELS[@]}" \
  --hierarchy_preset lineage_amr_profile \
  --global_fallback_labels "${GLOBAL_FALLBACK_LABELS:-none}" \
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
# 4. Panel annotation (optional; CV is a separate PBS job)
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
echo "Next (CV): qsub -v RUN_DIR=${RUN_DIR} pbs_11_leakage_aware_cv.pbs"

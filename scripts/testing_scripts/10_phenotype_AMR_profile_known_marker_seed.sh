#!/bin/bash
# =============================================================================
# LIGHT experiment: phenotype-only hierarchy + optional known-marker seed
#
#   AMR_binary → Resistance_Profile_Collapsed
#
# Skips lineage entirely (~3 nodes vs ~23 on the full 3-level tree).
# Default: seed_known_markers=true.  Set SEED_KNOWN_MARKERS=0 for control arm.
#
# Defaults (lighter than 01_):
#   RUN_LEAKAGE_AWARE_CV=0
#   RUN_PANEL_ANNOTATION=0
#
# Usage:
#   bash 10_phenotype_AMR_profile_known_marker_seed.sh              # seeded
#   SEED_KNOWN_MARKERS=0 bash 10_phenotype_AMR_profile_known_marker_seed.sh  # control
#   SEED_KNOWN_MARKERS=1 RUN_PANEL_ANNOTATION=1 bash 10_...         # + annotate
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_config.sh"

# Light defaults (override via env if needed)
RUN_LEAKAGE_AWARE_CV="${RUN_LEAKAGE_AWARE_CV:-0}"
RUN_PANEL_ANNOTATION="${RUN_PANEL_ANNOTATION:-0}"
SEED_KNOWN_MARKERS="${SEED_KNOWN_MARKERS:-1}"

bash "${SCRIPT_DIR}/00_prepare_inputs.sh"
bash "${SCRIPT_DIR}/00_preflight.sh" \
  AMR_binary Resistance_Profile_Collapsed
cd "${PROJECT_ROOT}"

if [[ -n "${CATALOGUE}" && "${CATALOGUE}" != /* ]]; then
  CATALOGUE="${SCRIPT_DIR}/${CATALOGUE}"
fi

HIER_LABELS=(AMR_binary Resistance_Profile_Collapsed)

if [[ "${SEED_KNOWN_MARKERS}" == "1" || "${SEED_KNOWN_MARKERS}" == "true" ]]; then
  NETWORKPARSER_CONFIG="${SCRIPT_DIR}/afro_seed_known_markers_config.json"
  RUN_NAME="Phenotype_AMR_Profile_seeded_10"
  SEED_TAG="seeded"
else
  # Same VCF policy, no seed (fair control arm for this light recipe)
  NETWORKPARSER_CONFIG="${SCRIPT_DIR}/afro_vcf_config.json"
  RUN_NAME="Phenotype_AMR_Profile_control_10"
  SEED_TAG="control"
fi

# Allow override of run name / out dir
RUN_NAME="${RUN_NAME_OVERRIDE:-${RUN_NAME}}"
RUN_DIR="${BASE_OUT}/${RUN_NAME}"

mkdir -p \
  "${RUN_DIR}" \
  "${RUN_DIR}/query" \
  "${RUN_DIR}/evaluate" \
  "${RUN_DIR}/validate_cv" \
  "${RUN_DIR}/panel_annotation"

# Persist experiment metadata
if [[ "${SEED_TAG}" == "seeded" ]]; then
  SEED_JSON_BOOL=true
else
  SEED_JSON_BOOL=false
fi
cat > "${RUN_DIR}/light_experiment.json" <<EOF
{
  "experiment": "phenotype_amr_profile_known_marker_seed",
  "arm": "${SEED_TAG}",
  "seed_known_markers": ${SEED_JSON_BOOL},
  "hierarchy_labels": ["AMR_binary", "Resistance_Profile_Collapsed"],
  "config": "${NETWORKPARSER_CONFIG}",
  "catalogue": "${CATALOGUE}",
  "filter": "${FILTER}",
  "note": "Lineage skipped. Compare seeded vs control holdout evaluate metrics; not a full L1→L2→L3 path claim."
}
EOF

echo "============================================================"
echo " Light phenotype experiment | arm=${SEED_TAG}"
echo " Labels: ${HIER_LABELS[*]}"
echo " Config: ${NETWORKPARSER_CONFIG}"
echo " Out:    ${RUN_DIR}"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. Train (2-level phenotype only)
# ---------------------------------------------------------------------------
"${PYTHON_BIN}" -m network_parser.cli train-hierarchy \
  --config "${NETWORKPARSER_CONFIG}" \
  --genomic "${TRAIN_GENOMIC}" \
  --meta "${META}" \
  --hierarchy_labels "${HIER_LABELS[@]}" \
  --global_fallback_labels none \
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
# 3. Evaluate hierarchy pack (AMR + profile only)
# ---------------------------------------------------------------------------
"${PYTHON_BIN}" -m network_parser.cli evaluate-hierarchy \
  --predictions "${RUN_DIR}/query/query_predictions.csv" \
  --meta "${EVALUATION_META}" \
  --hierarchy_labels "${HIER_LABELS[@]}" \
  --output_dir "${RUN_DIR}/evaluate" \
  --harmonize_resistance_labels

# ---------------------------------------------------------------------------
# 4. Optional CV (off by default for light run)
# ---------------------------------------------------------------------------
if [[ "${RUN_LEAKAGE_AWARE_CV}" == "1" ]]; then
  cv_algorithm_args=()
  if [[ -n "${CV_ALGORITHM:-}" ]]; then
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
fi

# ---------------------------------------------------------------------------
# 5. Optional panel annotation (off by default)
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
  fi
  "${cmd[@]}"
fi

# ---------------------------------------------------------------------------
# 6. Quick pointer to metrics
# ---------------------------------------------------------------------------
echo ""
echo "Done: ${RUN_DIR} (${SEED_TAG})"
echo "Compare arms:"
echo "  ${BASE_OUT}/Phenotype_AMR_Profile_seeded_10/evaluate/"
echo "  ${BASE_OUT}/Phenotype_AMR_Profile_control_10/evaluate/"
echo "Check seed applied:"
echo "  grep -R known_marker_seed ${RUN_DIR}/hierarchy_models --include='*.json' | head"
echo "Node dashboard: ${RUN_DIR}/hierarchy_node_dashboard.tsv"

#!/bin/bash
# =============================================================================
# Leakage-aware repeated cross-validation (training partition only).
# Standalone CHPC job — not part of 01 train/query/evaluate.
#
# Uses the same hierarchy labels / filter / seed config as experiment 01 by default.
# Point at an existing train run via RUN_DIR (writes validate_cv under it).
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

HIER_LABELS=(Lineage_clean AMR_binary Resistance_Profile_Collapsed)
SEED_KNOWN_MARKERS="${SEED_KNOWN_MARKERS:-1}"

# Match 01 seeded default config.
if [[ "${SEED_KNOWN_MARKERS}" == "1" || "${SEED_KNOWN_MARKERS}" == "true" ]]; then
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
  DEFAULT_RUN_NAME="Hierarchy_Lineage_AMR_Resistance_Profile_seeded_01"
else
  NETWORKPARSER_CONFIG="${NETWORKPARSER_CONFIG:-${SCRIPT_DIR}/afro_vcf_config.json}"
  DEFAULT_RUN_NAME="Hierarchy_Lineage_AMR_Resistance_Profile_01"
fi

RUN_NAME="${RUN_NAME_OVERRIDE:-${DEFAULT_RUN_NAME}}"
RUN_DIR="${RUN_DIR:-${BASE_OUT}/${RUN_NAME}}"

mkdir -p "${RUN_DIR}/validate_cv"

echo "============================================================"
echo " Leakage-aware CV (standalone)"
echo " RUN_DIR=${RUN_DIR}"
echo " Config=${NETWORKPARSER_CONFIG}"
echo " Labels=${HIER_LABELS[*]}"
echo " Repeats=${CV_REPEATS} splits=${CV_SPLITS} panels=${CV_PANEL_SIZES}"
echo "============================================================"

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

# Optional: re-run panel annotation with resistance-profile stability if requested
if [[ "${RUN_PANEL_ANNOTATION_AFTER_CV:-0}" == "1" ]]; then
  if [[ -f "${RUN_DIR}/validate_cv/Resistance_Profile_Collapsed/cv_feature_stability.tsv" ]]; then
    STABILITY_TSV="${RUN_DIR}/validate_cv/Resistance_Profile_Collapsed/cv_feature_stability.tsv"
  fi
  registry="${RUN_DIR}/hierarchical_model_registry.json"
  if [[ -f "${registry}" ]]; then
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
    [[ -n "${CATALOGUE}" && -f "${CATALOGUE}" ]] && cmd+=(--catalogue "${CATALOGUE}")
    [[ -n "${STABILITY_TSV:-}" && -f "${STABILITY_TSV}" ]] && cmd+=(--stability "${STABILITY_TSV}")
    "${cmd[@]}"
  fi
fi

echo "Done CV under: ${RUN_DIR}/validate_cv"

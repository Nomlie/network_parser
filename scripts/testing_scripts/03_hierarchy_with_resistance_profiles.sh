#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_config.sh"

RUN_NAME="Hierarchy_with_resistance_profiles_01"
RUN_DIR="${BASE_OUT}/${RUN_NAME}"

mkdir -p "${RUN_DIR}" "${RUN_DIR}/query" "${RUN_DIR}/evaluate"

python -m network_parser.cli train-two-level \
  --genomic "${TRAIN_GENOMIC}" \
  --meta "${META}" \
  --hierarchy_labels Lineage_Supergroup Lineage_clean Resistance_Profile_Collapsed \
  --central_feature_filter_method "${FILTER}" \
  --ref_fasta "${REF}" \
  --output_dir "${RUN_DIR}" \
  --n_jobs "${N_JOBS}"

python -m network_parser.cli query \
  --genomic "${TEST_GENOMIC}" \
  --bundle "${RUN_DIR}/networkparser_model_bundle.npb" \
  --ref_fasta "${REF}" \
  --query_input_type "${QUERY_INPUT_TYPE}" \
  --output_dir "${RUN_DIR}/query" \
  --n_jobs "${N_JOBS}"

python -m network_parser.cli evaluate \
  --predictions "${RUN_DIR}/query/query_predictions.csv" \
  --meta "${META}" \
  --hierarchy_labels Lineage_Supergroup Lineage_clean Resistance_Profile_Collapsed \
  --output_dir "${RUN_DIR}/evaluate"

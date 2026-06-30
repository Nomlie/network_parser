#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_config.sh"

RUN_NAME="Two_levels_01"
RUN_DIR="${BASE_OUT}/${RUN_NAME}"


mkdir -p "${RUN_DIR}" "${RUN_DIR}/query" "${RUN_DIR}/evaluate"
python -m network_parser.cli train-two-level \
  --genomic "${TRAIN_GENOMIC}" \
  --meta "${META}" \
  --level1_label Lineage_clean \
  --level2_label AMR_binary \
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
  --label Lineage_clean \
  --predicted_column predicted_level1_identity \
  --output_dir "${RUN_DIR}/evaluate/level1_Lineage_clean"

python -m network_parser.cli evaluate \
  --predictions "${RUN_DIR}/query/query_predictions.csv" \
  --meta "${META}" \
  --label AMR_binary \
  --predicted_column predicted_level2_identity \
  --output_dir "${RUN_DIR}/evaluate/level2_AMR_binary"



python -m network_parser.cli evaluate \
  --predictions /Users/nmfuphicsir.co.za/Documents/pHDProject/Results/All_VCFs/chi2_fdr/Two_levels_01/query/query_predictions.csv \
  --meta /Users/nmfuphicsir.co.za/Documents/pHDProject/Data/AFRO_TB/metadata/AFRO_dataset_meta_with_test_hierarchy.csv \
  --label AMR_binary \
  --output_dir /Users/nmfuphicsir.co.za/Documents/pHDProject/Results/All_VCFs/chi2_fdr/Two_levels_01/evaluate/level2_AMR_binary
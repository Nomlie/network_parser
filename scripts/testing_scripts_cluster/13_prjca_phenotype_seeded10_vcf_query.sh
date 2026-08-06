#!/bin/bash
# =============================================================================
# CHPC: Query Phenotype_AMR_Profile_seeded_10 on existing Chinese VCFs + evaluate.
# Hierarchy: AMR_binary → Resistance_Profile_Collapsed (no lineage).
# No FASTQ re-calling.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_config.sh"

cd "${PROJECT_ROOT}"

ARM="${ARM:-seeded}"
if [[ "${ARM}" == "control" ]]; then
  MODEL_RUN="${MODEL_RUN:-${BASE_OUT}/Phenotype_AMR_Profile_control_10}"
  OUT_TAG="control_10"
else
  MODEL_RUN="${MODEL_RUN:-${BASE_OUT}/Phenotype_AMR_Profile_seeded_10}"
  OUT_TAG="seeded_10"
fi

BUNDLE="${BUNDLE:-${MODEL_RUN}/networkparser_model_bundle.npb}"
REGISTRY="${REGISTRY:-${MODEL_RUN}/hierarchical_model_registry.json}"

PRJCA_ROOT="${PRJCA_ROOT:-${DATA_ROOT}/PRJCA040523}"
PHENOTYPE_META="${PHENOTYPE_META:-${PRJCA_ROOT}/meta/networkparser_metadata.csv}"
EVAL_META_FULL="${EVAL_META_FULL:-${PRJCA_ROOT}/meta/networkparser_evaluation_metadata.csv}"

VCF_DIR="${VCF_DIR:-${RESULTS_ROOT}/PRJCA040523_external/Hierarchy_Lineage_AMR_Resistance_Profile_external_fastq/query/fastq_query_preprocessing/final/vcf}"
NETWORKPARSER_CONFIG="${NETWORKPARSER_CONFIG:-${SCRIPT_DIR}/prjca_vcf_query_config.json}"

WORK_ROOT="${WORK_ROOT:-${RESULTS_ROOT}/PRJCA040523_external}"
RUN_NAME="${RUN_NAME:-Phenotype_AMR_Profile_${OUT_TAG}_chinese_vcfs}"
RUN_DIR="${RUN_DIR:-${WORK_ROOT}/${RUN_NAME}}"

HIER_LABELS=(AMR_binary Resistance_Profile_Collapsed)
LIMIT="${LIMIT:-}"
SKIP_QUERY="${SKIP_QUERY:-0}"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${RUN_DIR}/query" "${RUN_DIR}/evaluate" "${PBS_LOG_DIR}"

echo "=== Phenotype light model → Chinese VCFs (CHPC) ==="
echo "ARM=${ARM} MODEL_RUN=${MODEL_RUN}"
echo "BUNDLE=${BUNDLE}"
echo "VCF_DIR=${VCF_DIR}"
echo "RUN_DIR=${RUN_DIR}"
echo "Labels: ${HIER_LABELS[*]}"

[[ -d "${VCF_DIR}" ]] || { echo "Missing VCF_DIR: ${VCF_DIR}" >&2; exit 2; }
[[ -f "${BUNDLE}" || -f "${REGISTRY}" ]] || { echo "Missing bundle/registry under ${MODEL_RUN}" >&2; exit 2; }
[[ -f "${PHENOTYPE_META}" ]] || { echo "Missing phenotype meta: ${PHENOTYPE_META}" >&2; exit 2; }
[[ -f "${NETWORKPARSER_CONFIG}" ]] || { echo "Missing config: ${NETWORKPARSER_CONFIG}" >&2; exit 2; }

EVAL_META="${RUN_DIR}/evaluation_metadata_with_vcf.csv"
VCF_SAMPLE_LIST="${RUN_DIR}/vcf_sample_ids.txt"

"${PYTHON_BIN}" - <<PY
from pathlib import Path
import csv

vcf_dir = Path("${VCF_DIR}")
limit = "${LIMIT}".strip()
limit_n = int(limit) if limit else None
ids = sorted(p.name[:-7] for p in vcf_dir.glob("*.vcf.gz") if p.is_file())
# strip accidental .vcf if any
ids = [i.replace(".vcf", "") for i in ids]
if limit_n is not None:
    ids = ids[:limit_n]
Path("${VCF_SAMPLE_LIST}").write_text("\\n".join(ids) + ("\\n" if ids else ""))
print(f"VCF samples: {len(ids)}")

src = Path("${EVAL_META_FULL}") if Path("${EVAL_META_FULL}").is_file() else Path("${PHENOTYPE_META}")
id_set = set(ids)
with src.open(newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = list(reader.fieldnames or [])
    rows = [r for r in reader if (r.get("ID") or "").strip() in id_set]
have = {r["ID"].strip() for r in rows}
missing = sorted(id_set - have)
if missing:
    print(f"WARNING: {len(missing)} VCF IDs missing phenotype (e.g. {missing[:5]})")
out = Path("${EVAL_META}")
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
print(f"Eval meta n={len(rows)} → {out}")
PY

QUERY_GENOMIC="${VCF_DIR}"
if [[ -n "${LIMIT}" ]]; then
  SUBSET_DIR="${RUN_DIR}/vcf_subset"
  mkdir -p "${SUBSET_DIR}"
  find "${SUBSET_DIR}" -type l -delete 2>/dev/null || true
  while IFS= read -r sid; do
    [[ -z "${sid}" ]] && continue
    [[ -f "${VCF_DIR}/${sid}.vcf.gz" ]] && ln -sfn "${VCF_DIR}/${sid}.vcf.gz" "${SUBSET_DIR}/${sid}.vcf.gz"
    [[ -f "${VCF_DIR}/${sid}.vcf.gz.tbi" ]] && ln -sfn "${VCF_DIR}/${sid}.vcf.gz.tbi" "${SUBSET_DIR}/${sid}.vcf.gz.tbi"
  done < "${VCF_SAMPLE_LIST}"
  QUERY_GENOMIC="${SUBSET_DIR}"
fi

if [[ "${SKIP_QUERY}" != "1" ]]; then
  query_cmd=(
    "${PYTHON_BIN}" -m network_parser.cli query
    --config "${NETWORKPARSER_CONFIG}"
    --genomic "${QUERY_GENOMIC}"
    --ref_fasta "${REF}"
    --query_input_type vcf
    --output_dir "${RUN_DIR}/query"
    --n_jobs "${N_JOBS}"
  )
  if [[ -f "${BUNDLE}" ]]; then
    query_cmd+=(--bundle "${BUNDLE}")
  else
    query_cmd+=(--registry "${REGISTRY}")
  fi
  echo "Querying VCFs ..."
  "${query_cmd[@]}"
fi

PRED="${RUN_DIR}/query/query_predictions.csv"
[[ -f "${PRED}" ]] || { echo "Missing ${PRED}" >&2; exit 4; }

"${PYTHON_BIN}" -m network_parser.cli evaluate-hierarchy \
  --predictions "${PRED}" \
  --meta "${EVAL_META}" \
  --hierarchy_labels "${HIER_LABELS[@]}" \
  --output_dir "${RUN_DIR}/evaluate" \
  --harmonize_resistance_labels

echo "Done: ${RUN_DIR}"
echo "  evaluate → ${RUN_DIR}/evaluate"

#!/bin/bash
# Run WHO resistance-catalogue annotation against an existing trained registry.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_config.sh"

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "Usage: $0 RUN_DIR [CV_FEATURE_STABILITY_TSV]" >&2
  exit 2
fi

RUN_DIR="$(cd "$1" && pwd)"
stability_path="${2:-}"

bash "${SCRIPT_DIR}/00_preflight.sh"
cd "${PROJECT_ROOT}"

registry="${RUN_DIR}/hierarchical_model_registry.json"
if [[ ! -f "${registry}" ]]; then
  registry="${RUN_DIR}/two_level_model_registry.json"
fi
[[ -f "${registry}" ]] || {
  echo "No hierarchy registry found in ${RUN_DIR}" >&2
  exit 2
}

out_dir="${RUN_DIR}/panel_annotation"
mkdir -p "${out_dir}"
cmd=(
  "${PYTHON_BIN}" -m network_parser.cli annotate-panels
  --registry "${registry}"
  --output_dir "${out_dir}"
  --catalogue "${CATALOGUE}"
  --min_stability "${MIN_STABILITY}"
)

if [[ -n "${stability_path}" ]]; then
  [[ -f "${stability_path}" ]] || {
    echo "Stability table not found: ${stability_path}" >&2
    exit 2
  }
  cmd+=(--stability "${stability_path}")
fi

echo "Annotating existing run with WHO resistance catalogue"
echo "  registry=${registry}"
echo "  catalogue=${CATALOGUE}"
echo "  output_dir=${out_dir}"
"${cmd[@]}"

"${PYTHON_BIN}" - "${out_dir}/panel_features_annotated.tsv" <<'PY'
import sys
from pathlib import Path

import pandas as pd

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Expected annotation output was not created: {path}")
frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
counts = frame.get("catalogue_status", pd.Series(dtype=str)).value_counts()
print(f"Annotated panel features: {len(frame)}")
print("Catalogue status counts:")
print(counts.to_string() if not counts.empty else "  no catalogue-status rows")
if "Reference_build" not in frame.columns:
    print(
        "WARNING: this run uses legacy manifests without Reference_build. "
        "Exact alleles will remain candidate_unverified; retrain with the current "
        "NetworkParser version for strict known_mutation matches."
    )
PY

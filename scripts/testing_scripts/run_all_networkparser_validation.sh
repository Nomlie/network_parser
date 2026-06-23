#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/01_two_levels_with_global_AMR_binary_fallback.sh"
bash "${SCRIPT_DIR}/02_hierarchy_with_AMR_binary.sh"
bash "${SCRIPT_DIR}/03_hierarchy_with_resistance_profiles.sh"

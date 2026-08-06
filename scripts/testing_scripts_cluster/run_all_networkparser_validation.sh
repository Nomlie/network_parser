#!/bin/bash
# Run NetworkParser validation experiments sequentially.
# Each experiment script is self-contained (sources 00_config.sh; inline CV + annotate-panels).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$#" -eq 0 ]]; then
  # Default: biological primary recipes first
  set -- 01 02
fi

for experiment in "$@"; do
  case "${experiment}" in
    01) script="01_Lineage_AMR_Resistance_Profile.sh" ;;
    02) script="02_Lineage_family_Lineage_AMR_profile.sh" ;;
    03) script="03_two_levels_with_global_AMR_binary_fallback.sh" ;;
    04) script="04_hierarchy_with_supergroup_AMR_binary.sh" ;;
    05) script="05_hierarchy_with_supergroup_resistance_profiles.sh" ;;
    06) script="06_two_levels.sh" ;;
    07) script="07_Lineage_AMR_Resistance_Profile.sh" ;;
    08) script="08_Supergroup_Lineage_AMR_Resistance.sh" ;;
    09) script="09_annotate_existing_run.sh" ;;
    *)
      echo "Unknown experiment '${experiment}'. Choose from: 01 02 03 04 05 06 07 08 09" >&2
      exit 2
      ;;
  esac
  echo "Running NetworkParser validation experiment ${experiment}: ${script}"
  bash "${SCRIPT_DIR}/${script}"
done

echo "All NetworkParser validation runs completed."

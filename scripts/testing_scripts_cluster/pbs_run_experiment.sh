#!/bin/bash
#PBS -l select=1:ncpus=24:mem=120GB
#PBS -P RCHPC
#PBS -l walltime=96:00:00
#PBS -m abe
#PBS -M nmfuphi@csir.co.za
#PBS -N np_experiment
#PBS -o /mnt/lustre/users/nmfuphi/phDproject/Results/pbs_logs/np_experiment.out
#PBS -e /mnt/lustre/users/nmfuphi/phDproject/Results/pbs_logs/np_experiment.err
#
# Generic PBS entrypoint. Prefer the named pbs_01_*.pbs / pbs_02_*.pbs jobs,
# or submit with:
#   qsub -v EXPERIMENT=01 pbs_run_experiment.sh
#   qsub -v EXPERIMENT=02,HIERARCHY_RESUME=1 pbs_run_experiment.sh
#   qsub -v EXPERIMENT=01,RUN_LEAKAGE_AWARE_CV=0 pbs_run_experiment.sh
#
set -euo pipefail

# Resolve script directory whether qsub copies the job to a spool dir or not.
if [[ -n "${PBS_O_WORKDIR:-}" ]]; then
  cd "${PBS_O_WORKDIR}"
fi

# Prefer the checked-out scripts on the shared filesystem / home.
CLUSTER_SCRIPTS="${CLUSTER_SCRIPTS:-/home/nmfuphi/network_parser/scripts/testing_scripts_cluster}"
if [[ ! -d "${CLUSTER_SCRIPTS}" ]]; then
  # Fallback: directory of this file when run as bash pbs_run_experiment.sh
  CLUSTER_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# shellcheck source=/dev/null
source "${CLUSTER_SCRIPTS}/00_env_cluster.sh"
# shellcheck source=/dev/null
source "${CLUSTER_SCRIPTS}/00_config.sh"

mkdir -p "${PBS_LOG_DIR}" "${BASE_OUT}"

EXPERIMENT="${EXPERIMENT:-01}"
case "${EXPERIMENT}" in
  01) script="01_Lineage_AMR_Resistance_Profile.sh" ;;
  02) script="02_Lineage_family_Lineage_AMR_profile.sh" ;;
  03) script="03_two_levels_with_global_AMR_binary_fallback.sh" ;;
  04) script="04_hierarchy_with_supergroup_AMR_binary.sh" ;;
  05) script="05_hierarchy_with_supergroup_resistance_profiles.sh" ;;
  06) script="06_two_levels.sh" ;;
  07) script="07_Lineage_AMR_Resistance_Profile.sh" ;;
  08) script="08_Supergroup_Lineage_AMR_Resistance.sh" ;;
  09)
    echo "Experiment 09 needs RUN_DIR; use: bash 09_annotate_existing_run.sh RUN_DIR" >&2
    exit 2
    ;;
  11) script="11_leakage_aware_cv.sh" ;;
  12) script="12_prjca_external_fastq_query.sh" ;;
  13) script="13_prjca_phenotype_seeded10_vcf_query.sh" ;;
  *)
    echo "Unknown EXPERIMENT=${EXPERIMENT}. Use 01–08, 11 (CV), 12–13 (PRJCA)." >&2
    exit 2
    ;;
esac

export N_JOBS="${N_JOBS:-24}"
export HIERARCHY_RESUME="${HIERARCHY_RESUME:-0}"
export RUN_LEAKAGE_AWARE_CV="${RUN_LEAKAGE_AWARE_CV:-0}"
export RUN_PANEL_ANNOTATION="${RUN_PANEL_ANNOTATION:-1}"
export GLOBAL_FALLBACK_LABELS="${GLOBAL_FALLBACK_LABELS:-none}"
export SEED_KNOWN_MARKERS="${SEED_KNOWN_MARKERS:-1}"
export PROJECT_ROOT="${PROJECT_ROOT}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${PROJECT_ROOT}"
echo "Running experiment ${EXPERIMENT}: ${script}"
echo "  PROJECT_ROOT=${PROJECT_ROOT}"
echo "  DATA_ROOT=${DATA_ROOT}"
echo "  BASE_OUT=${BASE_OUT}"
echo "  N_JOBS=${N_JOBS}"
echo "  HIERARCHY_RESUME=${HIERARCHY_RESUME}"
echo "  GLOBAL_FALLBACK_LABELS=${GLOBAL_FALLBACK_LABELS}"

bash "${CLUSTER_SCRIPTS}/${script}"

echo "Finished experiment ${EXPERIMENT} at $(date)"

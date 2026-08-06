#!/usr/bin/env bash
# =============================================================================
# Copy PRJCA040523 data from the Mac workstation to CHPC lustre.
#
# Default remote (data transfer node):
#   nmfuphi@scp.chpc.ac.za:/mnt/lustre/users/nmfuphi/phDproject/Data/PRJCA040523
#
# Usage (from Mac):
#   bash scripts/testing_scripts_cluster/sync_prjca040523_to_chpc.sh
#   bash scripts/testing_scripts_cluster/sync_prjca040523_to_chpc.sh --dry-run
#   CHPC_HOST=lengau.chpc.ac.za bash .../sync_prjca040523_to_chpc.sh   # login node if needed
#
# Large transfer (~CRA025985 FASTQs is 100GB+). Prefer a stable network /
# screen/tmux. Excludes staging dirs and .DS_Store by default.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PRJCA="${LOCAL_PRJCA:-/Users/nmfuphicsir.co.za/Documents/pHDProject/Data/PRJCA040523}"
CHPC_HOST="${CHPC_HOST:-scp.chpc.ac.za}"
CHPC_USER="${CHPC_USER:-nmfuphi}"
REMOTE_BASE="${REMOTE_BASE:-/mnt/lustre/users/nmfuphi/phDproject/Data/PRJCA040523}"
SSH_ID="${SSH_ID:-${HOME}/.ssh/id_ed25519}"
if [[ ! -f "${SSH_ID}" && -f "${HOME}/ssh_key_backup_2026-06-10_1235/id_ed25519" ]]; then
  SSH_ID="${HOME}/ssh_key_backup_2026-06-10_1235/id_ed25519"
fi

DRY_RUN=0
INCLUDE_FASTQ=1
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --meta-only) INCLUDE_FASTQ=0 ;;
    --help|-h)
      sed -n '2,25p' "$0"
      exit 0
      ;;
  esac
done

# macOS ships openrsync (2.6.9-compatible): no --append-verify.
# Use portable flags only. Prefer GNU rsync if available (homebrew).
RSYNC_BIN="${RSYNC_BIN:-rsync}"
if command -v grsync >/dev/null 2>&1; then
  RSYNC_BIN="grsync"
elif [[ -x /opt/homebrew/bin/rsync ]]; then
  RSYNC_BIN="/opt/homebrew/bin/rsync"
elif [[ -x /usr/local/bin/rsync ]]; then
  # Prefer Homebrew rsync when it is not the openrsync shim
  if /usr/local/bin/rsync --version 2>&1 | grep -qi 'openrsync'; then
    :
  else
    RSYNC_BIN="/usr/local/bin/rsync"
  fi
fi

RSYNC_FLAGS=(-avh --progress --partial)
# --append is supported on openrsync and helps resume large FASTQ uploads
RSYNC_FLAGS+=(--append)
RSYNC_FLAGS+=(--exclude '.DS_Store')
RSYNC_FLAGS+=(--exclude 'query_stage_fastq/')
RSYNC_FLAGS+=(--exclude 'query_stage_fastq_phenotype_seed_compare/')

if [[ "${DRY_RUN}" == "1" ]]; then
  RSYNC_FLAGS+=(--dry-run)
  echo "[dry-run] no files will be written"
fi

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=30)
if [[ -f "${SSH_ID}" ]]; then
  SSH_OPTS+=(-i "${SSH_ID}")
fi
# Single string for rsync -e (array expansion breaks -i on some rsync builds)
RSYNC_RSH="ssh"
for opt in "${SSH_OPTS[@]}"; do
  RSYNC_RSH+=" ${opt}"
done

REMOTE="${CHPC_USER}@${CHPC_HOST}:${REMOTE_BASE}/"

echo "=== Sync PRJCA040523 → CHPC ==="
echo "Local:  ${LOCAL_PRJCA}"
echo "Remote: ${REMOTE}"
echo "rsync:  ${RSYNC_BIN} (${RSYNC_FLAGS[*]})"
echo "SSH:    ${RSYNC_RSH}"

[[ -d "${LOCAL_PRJCA}" ]] || { echo "Missing local PRJCA dir: ${LOCAL_PRJCA}" >&2; exit 2; }

# Ensure remote parent exists
ssh "${SSH_OPTS[@]}" "${CHPC_USER}@${CHPC_HOST}" \
  "mkdir -p '${REMOTE_BASE}' '/mnt/lustre/users/nmfuphi/phDproject/Results/pbs_logs'"

# 1) meta + ref + scripts (small)
echo "--- meta / ref / scripts ---"
"${RSYNC_BIN}" "${RSYNC_FLAGS[@]}" -e "${RSYNC_RSH}" \
  "${LOCAL_PRJCA}/meta/" "${REMOTE}meta/"
"${RSYNC_BIN}" "${RSYNC_FLAGS[@]}" -e "${RSYNC_RSH}" \
  "${LOCAL_PRJCA}/ref/" "${REMOTE}ref/"
"${RSYNC_BIN}" "${RSYNC_FLAGS[@]}" -e "${RSYNC_RSH}" \
  "${LOCAL_PRJCA}/scripts/" "${REMOTE}scripts/"
if [[ -f "${LOCAL_PRJCA}/README_DOWNLOAD.md" ]]; then
  "${RSYNC_BIN}" "${RSYNC_FLAGS[@]}" -e "${RSYNC_RSH}" \
    "${LOCAL_PRJCA}/README_DOWNLOAD.md" "${REMOTE}"
fi

# 2) FASTQ CRA (large)
if [[ "${INCLUDE_FASTQ}" == "1" ]]; then
  echo "--- CRA025985 FASTQs (large) ---"
  "${RSYNC_BIN}" "${RSYNC_FLAGS[@]}" -e "${RSYNC_RSH}" \
    "${LOCAL_PRJCA}/CRA025985/" "${REMOTE}CRA025985/"
else
  echo "Skipping FASTQs (--meta-only)"
fi

echo "=== Sync finished (or dry-run complete) ==="
echo "On CHPC verify:"
echo "  ls -la ${REMOTE_BASE}"
echo "  du -sh ${REMOTE_BASE}/CRA025985"

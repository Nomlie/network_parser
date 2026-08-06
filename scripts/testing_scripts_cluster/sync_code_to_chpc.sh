#!/usr/bin/env bash
# Sync network_parser code to CHPC home (login node).
#   bash scripts/testing_scripts_cluster/sync_code_to_chpc.sh
#   bash scripts/testing_scripts_cluster/sync_code_to_chpc.sh --dry-run
set -euo pipefail

LOCAL_CODE="${LOCAL_CODE:-/Users/nmfuphicsir.co.za/Documents/pHDProject/Code/network_parser}"
CHPC_HOST="${CHPC_HOST:-scp.chpc.ac.za}"
CHPC_USER="${CHPC_USER:-nmfuphi}"
REMOTE="${REMOTE:-/home/nmfuphi/network_parser}"
SSH_ID="${SSH_ID:-${HOME}/.ssh/id_ed25519}"
[[ -f "${SSH_ID}" ]] || SSH_ID="${HOME}/ssh_key_backup_2026-06-10_1235/id_ed25519"

RSYNC_FLAGS=(-avh --progress)
RSYNC_FLAGS+=(--exclude '.git' --exclude '__pycache__' --exclude '.mypy_cache' --exclude '.pytest_cache' --exclude '.DS_Store')
for arg in "$@"; do
  [[ "${arg}" == "--dry-run" ]] && RSYNC_FLAGS+=(--dry-run)
done

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=30)
[[ -f "${SSH_ID}" ]] && SSH_OPTS+=(-i "${SSH_ID}")
RSYNC_RSH="ssh"
for opt in "${SSH_OPTS[@]}"; do
  RSYNC_RSH+=" ${opt}"
done

echo "Sync code: ${LOCAL_CODE} → ${CHPC_USER}@${CHPC_HOST}:${REMOTE}"
rsync "${RSYNC_FLAGS[@]}" -e "${RSYNC_RSH}" \
  "${LOCAL_CODE}/" "${CHPC_USER}@${CHPC_HOST}:${REMOTE}/"
echo "Done."

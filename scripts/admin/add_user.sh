#!/bin/bash
#
# Add the default organisation / user group / user from a JSON config file.
#
#   bash scripts/admin/add_user.sh [path/to/user-group-organisation.json]
#
# Defaults to configs/user-group-organisation.json (create it from
# configs/user-group-organisation.example.json).
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")/../.." >/dev/null 2>&1 && pwd -P)"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/sse_lib/src:${PYTHONPATH:-}"

if [[ -f "${REPO_DIR}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_DIR}/.venv/bin/activate"
elif [[ -f "${REPO_DIR}/venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_DIR}/venv/bin/activate"
fi
USER_CONFIG="${1:-$REPO_DIR/configs/user-group-organisation.json}"

cd "$REPO_DIR"
exec python3 src/sse_tools/admin/add_org_group_user.py --user-config "$USER_CONFIG"

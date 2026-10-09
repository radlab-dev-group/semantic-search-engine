#!/bin/bash
#
# Dump every user's chats to a timestamped directory.
#
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")/../.." >/dev/null 2>&1 && pwd -P)"
OUT_DIR="${1:-$REPO_DIR/chat_dumps/$(date +%Y%m%d_%H%M%S)}"
cd "$REPO_DIR"
exec python3 src/sse_tools/admin/dump_user_chats.py --out-dir "$OUT_DIR"

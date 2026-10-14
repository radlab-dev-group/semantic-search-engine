#!/bin/bash
#
# Index a directory of files into PostgreSQL (via the data API).
#
#   bash src/sse_tools/indexing/index_to_postgresql.sh <data-dir> <collection>
#
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")/../../.." >/dev/null 2>&1 && pwd -P)"
DATA_DIR="${1:?usage: index_to_postgresql.sh <data-dir> <collection>}"
COLLECTION="${2:?usage: index_to_postgresql.sh <data-dir> <collection>}"

cd "$REPO_DIR"
exec python3 src/sse_tools/indexing/add_files_from_dir.py \
    -d "$DATA_DIR" \
    -c "$COLLECTION" \
    --proper-pages \
    --merge-document-pages

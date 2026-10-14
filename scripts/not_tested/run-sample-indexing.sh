#!/bin/bash
#
# Index a directory of files into a collection.
#
#   bash scripts/not_tested/run-sample-indexing.sh <data-dir> <collection>
#
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")/../.." >/dev/null 2>&1 && pwd -P)"
DATA_DIR="${1:?usage: run-sample-indexing.sh <data-dir> <collection>}"
COLLECTION="${2:?usage: run-sample-indexing.sh <data-dir> <collection>}"
cd "$REPO_DIR"
exec python3 src/sse_tools/indexing/add_files_from_dir.py \
    -d "$DATA_DIR" \
    -c "$COLLECTION" \
    --proper-pages \
    --merge-document-pages

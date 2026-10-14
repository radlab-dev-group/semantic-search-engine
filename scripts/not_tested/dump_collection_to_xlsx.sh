#!/bin/bash
#
# Dump a collection's documents to an xlsx file.
#
#   bash scripts/not_tested/dump_collection_to_xlsx.sh <collection-id>
#
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")/../.." >/dev/null 2>&1 && pwd -P)"
COLLECTION_ID="${1:?usage: dump_collection_to_xlsx.sh <collection-id>}"
OUT_XLSX_FILE="$REPO_DIR/collection_${COLLECTION_ID}_$(date +%Y%m%d_%H%M%S).xlsx"
cd "$REPO_DIR"
exec python3 src/sse_tools/evaluation/dataset/dump_collection_to_xlsx.py \
    --collection-id "$COLLECTION_ID" \
    --out-xlsx-file "$OUT_XLSX_FILE"

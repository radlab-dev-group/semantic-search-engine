#!/bin/bash
#
# Re-index an existing collection into a different Milvus index type.
#
#   bash src/sse_tools/indexing/index_to_milvus.sh
#
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")/../../.." >/dev/null 2>&1 && pwd -P)"

INDEX_TYPE="${1:-HNSW}"

cd "$REPO_DIR"
exec python3 src/sse_tools/indexing/index_collection_to_milvus.py \
  --index-name "$INDEX_TYPE" \
  --from-collection AiA2023 \
  --to-collection "AiA2023_${INDEX_TYPE}" \
  --chunk-type clear_texts_proper_page_chunk_max_tokens_200_overlap_20

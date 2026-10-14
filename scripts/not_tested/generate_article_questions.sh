#!/bin/bash
#
# Generate questions from a collection's articles.
#
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")/../.." >/dev/null 2>&1 && pwd -P)"
cd "$REPO_DIR"
exec python3 src/sse_tools/evaluation/dataset/generator/article_question_generator.py \
  --collection-name "${1:?collection name}"

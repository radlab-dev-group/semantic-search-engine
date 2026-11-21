#!/bin/bash
#
# Evaluate embedder search against a test configuration.
#
#   bash scripts/not_tested/evaluate_embedder_search.sh <test-configuration.json>
#
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")/../.." >/dev/null 2>&1 && pwd -P)"
TEST_CONFIGURATION="${1:?usage: evaluate_embedder_search.sh <test-configuration.json>}"
cd "$REPO_DIR"
exec python3 src/sse_tools/evaluation/evaluator/evaluate_embedder_search.py \
    --username "default_user" \
    --test-configuration "$TEST_CONFIGURATION" \
    --out-xlsx-file "$REPO_DIR/eval_$(date +%Y%m%d_%H%M%S).xlsx"

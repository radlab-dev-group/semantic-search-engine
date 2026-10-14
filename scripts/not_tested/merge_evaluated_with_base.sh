#!/bin/bash
#
# Merge evaluated results into a base dataset.
#
#   bash scripts/not_tested/merge_evaluated_with_base.sh <base.xlsx> <evaluated.xlsx> <merged.xlsx>
#
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")/../.." >/dev/null 2>&1 && pwd -P)"
cd "$REPO_DIR"
exec python3 src/sse_tools/evaluation/evaluator/merge_evaluated_results.py \
  --base-file "${1:?base.xlsx}" \
  --evaluated-file "${2:?evaluated.xlsx}" \
  --merged-out-file "${3:?merged.xlsx}"

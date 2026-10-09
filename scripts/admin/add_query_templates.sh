#!/bin/bash
#
# Load query templates into the default organisation.
#
#   bash scripts/admin/add_query_templates.sh
#
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")/../.." >/dev/null 2>&1 && pwd -P)"
AUTH_CONFIG="${AUTH_CONFIG:-$REPO_DIR/configs/auth-config.json}"
QUERY_TEMPL_CONFIG="${QUERY_TEMPL_CONFIG:-$REPO_DIR/configs/query-templates.json}"

cd "$REPO_DIR"
exec python3 src/sse_tools/admin/add_query_template_to_org.py \
  --auth-config "$AUTH_CONFIG" \
  --query-config "$QUERY_TEMPL_CONFIG"

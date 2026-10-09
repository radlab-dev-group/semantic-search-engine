#!/bin/bash
#
# Local quality gate: formatting + static analysis + type check + security.
#
#   ./check-dev.sh            # everything
#   ./check-dev.sh format     # black (in-place)
#   ./check-dev.sh lint       # flake8 + bandit
#   ./check-dev.sh types      # mypy
#   ./check-dev.sh test       # pytest (server + client)
#
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]:-$0}")"

MODE="${1:-all}"

run_format() {
  black .
}

run_lint() {
  flake8 .
  bandit -r src/ scripts/ -c .bandit -q
}

run_types() {
  mypy src/sse_api src/sse_lib
}

run_test() {
  python3 -m pytest
}

case "$MODE" in
  format) run_format ;;
  lint)   run_lint ;;
  types)  run_types ;;
  test)   run_test ;;
  all)    run_format; run_lint; run_types; run_test ;;
  *) echo "usage: $0 [all|format|lint|types|test]"; exit 1 ;;
esac

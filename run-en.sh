#!/usr/bin/env bash
#
# run-en.sh — Semantic Search Engine (SSE): one-command startup script (English edition).
# English counterpart of run.sh: same behaviour, English comments and messages.
# Combines the steps from SETUP.md: venv + dependencies (step 2), PG + Milvus (3-4),
# (optionally LLM router, 5), initialization (6), start API (7).
# Additionally: "cosine" — migrate existing Milvus collections to the COSINE metric
# (without re-embedding) for search.
#
# Usage:
#   ./run-en.sh                 # == all (full pipeline: setup+infra+init+start)
#   ./run-en.sh all             # setup + infra + init + start
#   ./run-en.sh setup           # venv + pip install + git dependencies (step 2)
#   ./run-en.sh infra           # start PostgreSQL + Milvus in Docker (steps 3-4)
#   ./run-en.sh init            # migrate + semantic + add_user + add_query_templates (step 6)
#   ./run-en.sh cosine          # migrate existing collections to COSINE (defaults to --all)
#   ./run-en.sh start|up        # run the API server in the foreground → http://localhost:8271/api/ (step 7)
#   ./run-en.sh down            # stop and remove the PG + Milvus containers
#   ./run-en.sh status          # show the state of containers and ports (5471/19530/19121/8271)
#
# Flags:
#   --no-venv        do not create/use a venv (use the system python/pip)
#   --no-infra       skip starting PG/Milvus (assume they are already running)
#   --no-init        skip migrations/initialization (step 6)
#   --skip-server    in "all": do not start the server after setup (exit after initialization)
#
#   # flags of the "cosine" command:
#   --collection X collection to migrate (repeatable; without the flag: all/--all)
#   --index TYPE   type of the new index: HNSW or IVF_FLAT (default: IVF_FLAT)
#   --dry-run      only show the planned operations (no changes in the DB)
#
# Examples:
#   ./run-en.sh all --no-infra                # PG+Milvus already running, the rest is automated
#   ./run-en.sh all --skip-server             # setup only, no foreground start
#   ./run-en.sh cosine --dry-run              # preview of the COSINE migration (all collections)
#   ./run-en.sh cosine --collection my_coll   # only one collection
#   ./run-en.sh status
#
set -euo pipefail

# ---------- locations ----------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd -P)"
ROOT_DIR="$SCRIPT_DIR"
API_DIR="$ROOT_DIR/sse_rest_api"
VENV_DIR="$ROOT_DIR/.venv"
PY_PREFERRED="python3.11"

# ---------- logging ----------
if [[ -t 1 ]]; then
  C_INFO=$'\e[32m'; C_WARN=$'\e[33m'; C_ERR=$'\e[31m'; C_DIM=$'\e[2m'; C_RST=$'\e[0m'
else
  C_INFO=""; C_WARN=""; C_ERR=""; C_DIM=""; C_RST=""
fi
log()  { printf '%s[INFO]%s %s\n' "$C_INFO" "$C_RST" "$*"; }
warn() { printf '%s[WARN]%s %s\n' "$C_WARN" "$C_RST" "$*" >&2; }
err()  { printf '%s[ERR ]%s %s\n' "$C_ERR" "$C_RST" "$*" >&2; }
step() { printf '\n%s==> %s%s\n' "$C_DIM" "$*" "$C_RST"; }

# ---------- parsing ----------
CMD="all"
NO_VENV=0; NO_INFRA=0; NO_INIT=0; SKIP_SERVER=0
DRY_RUN=0
COSINE_COLLECTIONS=()
COSINE_INDEX="IVF_FLAT"

while (( $# > 0 )); do
  case "$1" in
    --no-venv)     NO_VENV=1 ;;
    --no-infra)    NO_INFRA=1 ;;
    --no-init)     NO_INIT=1 ;;
    --skip-server) SKIP_SERVER=1 ;;
    --dry-run)     DRY_RUN=1 ;;
    --collection)  shift; [[ $# -gt 0 ]] || { err "--collection requires a collection name"; exit 1; }; COSINE_COLLECTIONS+=("$1") ;;
    --index)       shift; [[ $# -gt 0 ]] || { err "--index requires a value (HNSW|IVF_FLAT)"; exit 1; }; COSINE_INDEX="$1" ;;
    -h|--help|help) CMD="help" ;;
    all|setup|infra|init|cosine|start|up|down|status) CMD="$1" ;;
    *) warn "Unknown option: $1 (showing help)"; CMD="help" ;;
  esac
  shift
done

# ---------- helpers ----------
have() { command -v "$1" >/dev/null 2>&1; }
need() { err "Missing: $1. Install/start it first."; exit 1; }

activate_venv() {
  if [[ -f "$VENV_DIR/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
  fi
}
pick_pip() {
  if have pip; then echo "pip"; elif have pip3; then echo "pip3"; else err "Missing pip/pip3"; exit 1; fi
}
pick_python() {
  if have python; then echo "python"
  elif have python3; then echo "python3"
  else err "Missing python/python3"; exit 1
  fi
}
wait_port() {
  local port="$1" name="$2" tries="${3:-30}"
  for _ in $(seq 1 "$tries"); do
    if (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; then
      exec 3>&- 3<&- 2>/dev/null || true
      log "$name is listening on port $port"
      return 0
    fi
    sleep 1
  done
  warn "$name: port $port still unavailable after ${tries}s — check it manually"
}

# ---------- commands ----------
setup() {
  step "Step 2: virtual environment + dependencies"
  [[ -d "$API_DIR" ]] || { err "Missing sse_rest_api/ directory — are you inside the repo?"; exit 1; }

  local py="$PY_PREFERRED"
  have "$py" || { py="python3"; warn "$PY_PREFERRED not found, using python3"; }
  have "$py" || need python3

  if [[ "$NO_VENV" == "1" ]]; then
    warn "Skipping venv (--no-venv); using the system python/pip."
  else
    if [[ ! -d "$VENV_DIR" ]]; then
      log "Creating venv: $VENV_DIR"
      "$py" -m venv "$VENV_DIR"
    else
      log "venv already exists: $VENV_DIR"
    fi
    activate_venv
    log "Active python: $(command -v python3) ($(python3 --version 2>&1))"
  fi

  local pip; pip="$(pick_pip)"
  log "Installing dependencies: sse_rest_api/requirements.txt"
  "$pip" install --upgrade pip >/dev/null
  "$pip" install -r "$API_DIR/requirements.txt"

  log "Installing dependencies from git (radlab-data, llm-router)"
  ( cd "$API_DIR" && ./initialize.sh dep )
}

infra() {
  step "Steps 3-4: infrastructure (PostgreSQL + Milvus)"
  if [[ "$NO_INFRA" == "1" ]]; then
    warn "Skipping PG/Milvus startup (--no-infra). Assuming they are already running."
    return 0
  fi
  have docker || need docker

  log "PostgreSQL: bash scripts/admin/postgres.sh"
  ( cd "$ROOT_DIR" && bash scripts/admin/postgres.sh )

  log "Milvus standalone (ports 19530, 19121)"
  docker rm -f milvus-standalone >/dev/null 2>&1 || true
  docker run -d --name milvus-standalone -p 19530:19530 -p 19121:19121 milvusdb/milvus:2.3.0

  log "Waiting for ports..."
  wait_port 5471  "PostgreSQL"
  wait_port 19530 "Milvus"
}

init() {
  step "Step 6: database + account + templates"
  if [[ "$NO_INIT" == "1" ]]; then
    warn "Skipping initialization (--no-init)."
    return 0
  fi
  [[ -d "$API_DIR" ]] || { err "Missing sse_rest_api/"; exit 1; }
  [[ "$NO_VENV" == "1" ]] || activate_venv
  ( cd "$API_DIR" && ./initialize.sh migrate )
  ( cd "$API_DIR" && ./initialize.sh semantic )
  ( cd "$API_DIR" && ./initialize.sh add_user )
  ( cd "$API_DIR" && ./initialize.sh add_query_templates )
}

cosine() {
  step "COSINE: migrating indexes of existing collections (no re-embedding)"
  local script="$SCRIPT_DIR/scripts/admin/milvus_index_to_cosine.py"
  local cfg="$API_DIR/configs/milvus_config.json"
  [[ -f "$script" ]] || { err "Missing migration script: $script"; exit 1; }
  [[ -f "$cfg" ]]    || { err "Missing Milvus config: $cfg"; exit 1; }
  [[ "$NO_VENV" == "1" ]] || activate_venv

  local args=(--config "$cfg" --index "$COSINE_INDEX")
  local c
  if (( ${#COSINE_COLLECTIONS[@]} )); then
    for c in "${COSINE_COLLECTIONS[@]}"; do args+=(--collection "$c"); done
  else
    args+=(--all)
  fi
  if (( DRY_RUN )); then args+=(--dry-run); fi

  local scope="ALL"
  if (( ${#COSINE_COLLECTIONS[@]} )); then scope="${COSINE_COLLECTIONS[*]}"; fi
  log "COSINE migration: collections=$scope, index=$COSINE_INDEX, dry_run=$DRY_RUN"

  # --all requires a connection (listing collections); --collection + --dry-run works offline
  if ! { (( DRY_RUN )) && (( ${#COSINE_COLLECTIONS[@]} )); }; then
    wait_port 19530 "Milvus" 5
  fi

  local py; py="$(pick_python)"
  ( cd "$API_DIR" && "$py" "$script" "${args[@]}" )
}

start() {
  step "Step 7: starting the API server (foreground) → http://localhost:8271/api/"
  [[ -d "$API_DIR" ]] || { err "Missing sse_rest_api/"; exit 1; }
  [[ "$NO_VENV" == "1" ]] || activate_venv
  log "Test login: default_admin / password  (POST /api/login)"
  ( cd "$API_DIR" && ./run-api.sh )
}

down() {
  step "Stop: stopping and removing the PG + Milvus containers"
  have docker || need docker
  docker rm -f pg_sse_backend_engine >/dev/null 2>&1 || true
  docker rm -f milvus-standalone    >/dev/null 2>&1 || true
  log "Stopped."
}

status() {
  step "Status"
  if have docker; then
    log "Containers:"
    docker ps -a --filter name=pg_sse_backend_engine --filter name=milvus-standalone \
      --format '  table {{.Names}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null || true
  else
    warn "Docker unavailable — skipped the container list."
  fi
  log "Ports (127.0.0.1):"
  local p
  for p in 5471 19530 19121 8271; do
    if (exec 3<>"/dev/tcp/127.0.0.1/${p}") 2>/dev/null; then
      exec 3>&- 2>/dev/null || true
      log "  :$p OPEN"
    else
      log "  :$p closed"
    fi
  done
}

help() {
  awk 'NR==1{next} /^set -euo pipefail/{exit} {sub(/^# ?/,""); print}' "$0"
}

# ---------- dispatch ----------
case "$CMD" in
  all)
    setup
    infra
    init
    if [[ "$SKIP_SERVER" == "1" ]]; then
      warn "--skip-server: skipping the server start (you can run it later: ./run-en.sh start)."
      exit 0
    fi
    start
    ;;
  setup)  setup ;;
  infra)  infra ;;
  init)   init ;;
  cosine) cosine ;;
  start|up) start ;;
  down)   down ;;
  status) status ;;
  help)   help ;;
  *)      help; exit 1 ;;
esac

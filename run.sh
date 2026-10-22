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
#   ./run-en.sh start|up        # run the API server in the foreground → http://0.0.0.0:8271/api/ (step 7)
#   ./run-en.sh down            # stop and remove the PG + Milvus containers; DATA VOLUMES ARE KEPT
#   ./run-en.sh clean           # like down, but also removes the volumes (postgres_data, milvus_data)
#   ./run-en.sh status          # show the state of containers and ports (5471/19530/19121/8271)
#
# Flags:
#   --no-venv        do not create/use a venv (use the system python/pip)
#   --no-infra       skip starting PG/Milvus (assume they are already running)
#   --no-init        skip migrations/initialization (step 6)
#   --skip-server    in "all": do not start the server after setup (exit after initialization)
#   -y|--yes         for "clean": do not ask for confirmation (for CI/scripts)
#
#   # flags of the "cosine" command:
#   --collection X collection to migrate (repeatable; without the flag: all/--all)
#   --index TYPE   type of the new index: HNSW or IVF_FLAT (default: IVF_FLAT)
#   --dry-run      only show the planned operations (no changes in the DB)
#
# Volumes and data:
#   Data lives in named Docker volumes (postgres_data, milvus_data) created by
#   every "./run-en.sh infra".  "./run-en.sh down" stops and removes the
#   containers but leaves the volumes alone, so the next start finds the very
#   same database and vectors.  Only "./run-en.sh clean" deletes them
#   (containers + volumes).  A PostgreSQL/Milvus mismatch (for instance after a
#   crash in the middle of indexing) is fixed the same way: clean and re-index,
#   because PostgreSQL is the source of truth.
#
# Examples:
#   ./run-en.sh all --no-infra                # PG+Milvus already running, the rest is automated
#   ./run-en.sh all --skip-server             # setup only, no foreground start
#   ./run-en.sh cosine --dry-run              # preview of the COSINE migration (all collections)
#   ./run-en.sh cosine --collection my_coll   # only one collection
#   ./run-en.sh down                              # stop the containers; data in the volumes is intact
#   ./run-en.sh clean                             # remove containers + all data (asks first)
#   ./run-en.sh clean --yes                       # the same without asking
#   ./run-en.sh status
#
set -euo pipefail

# ---------- locations ----------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd -P)"
ROOT_DIR="$SCRIPT_DIR"
API_DIR="$ROOT_DIR"
CONFIG_DIR="$ROOT_DIR/configs"
VENV_DIR="$ROOT_DIR/.venv"
PY_PREFERRED="python3.11"

# SSE containers and persistent volumes.  The volumes outlive "./run-en.sh down",
# are created by "./run-en.sh infra" and are removed only by "./run-en.sh clean".
# Note: "docker compose" prefixes them with the project name (sse_postgres_data,
# ...), which is a different set managed by "docker compose down [-v]".
SSE_CONTAINERS=(pg_sse_backend_engine milvus-standalone)
SSE_VOLUMES=(postgres_data milvus_data)

export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/sse_lib/src:${PYTHONPATH:-}"
export DJANGO_SETTINGS_MODULE="${DJANGO_SETTINGS_MODULE:-sse_api.config.settings}"

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
ASSUME_YES=0
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
    -y|--yes)        ASSUME_YES=1 ;;
    --collection)  shift; [[ $# -gt 0 ]] || { err "--collection requires a collection name"; exit 1; }; COSINE_COLLECTIONS+=("$1") ;;
    --index)       shift; [[ $# -gt 0 ]] || { err "--index requires a value (HNSW|IVF_FLAT)"; exit 1; }; COSINE_INDEX="$1" ;;
    -h|--help|help) CMD="help" ;;
    all|setup|infra|init|cosine|start|up|down|clean|status) CMD="$1" ;;
    *) warn "Unknown option: $1 (showing help)"; CMD="help" ;;
  esac
  shift
done

# ---------- helpers ----------
have() { command -v "$1" >/dev/null 2>&1; }
need() { err "Missing: $1. Install/start it first."; exit 1; }

activate_venv() {
  local candidates=(
    "${VENV_DIR:-}"
    "$ROOT_DIR/.venv"
    "$ROOT_DIR/venv"
    "${VIRTUAL_ENV:-}"
  )
  for v in "${candidates[@]}"; do
    if [[ -n "$v" && -f "$v/bin/activate" ]]; then
      # shellcheck disable=SC1091
      source "$v/bin/activate"
      return 0
    fi
  done
}
pick_pip() {
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/pip3" ]]; then
    echo "${VIRTUAL_ENV}/bin/pip3"
  elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/pip" ]]; then
    echo "${VIRTUAL_ENV}/bin/pip"
  elif [[ -x "$VENV_DIR/bin/pip3" ]]; then
    echo "$VENV_DIR/bin/pip3"
  elif [[ -x "$VENV_DIR/bin/pip" ]]; then
    echo "$VENV_DIR/bin/pip"
  elif have pip3; then
    echo "pip3"
  elif have pip; then
    echo "pip"
  else
    err "Missing pip/pip3"; exit 1
  fi
}
pick_python() {
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python3" ]]; then
    echo "${VIRTUAL_ENV}/bin/python3"
  elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    echo "${VIRTUAL_ENV}/bin/python"
  elif [[ -x "$VENV_DIR/bin/python3" ]]; then
    echo "$VENV_DIR/bin/python3"
  elif [[ -x "$VENV_DIR/bin/python" ]]; then
    echo "$VENV_DIR/bin/python"
  elif have python3; then
    echo "python3"
  elif have python; then
    echo "python"
  else
    err "Missing python/python3"; exit 1
  fi
}
docker_cmd() {
  if docker info >/dev/null 2>&1; then
    docker "$@"
  elif have sudo; then
    sudo docker "$@"
  else
    docker "$@"
  fi
}
# Is a container of this name running right now?  Used not to recreate a
# container whose data already sits in a volume.
container_running() {
  [[ -n "$(docker_cmd ps -q --filter "name=^${1}\$" 2>/dev/null)" ]]
}

# Named volumes are created up front instead of as a side effect of 'docker run',
# so their names stay predictable for 'clean' and 'status'.
ensure_volumes() {
  local volume
  for volume in "${SSE_VOLUMES[@]}"; do
    docker_cmd volume create "$volume" >/dev/null 2>&1 || true
  done
}

remove_sse_containers() {
  local container
  for container in "${SSE_CONTAINERS[@]}"; do
    docker_cmd rm -f "$container" >/dev/null 2>&1 || true
  done
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
config() {
  step "Configuration: .env + config files from templates"
  local created=0 skipped=0 example target name rel

  if [[ -f "$ROOT_DIR/.env" ]]; then
    log "Skipped (exists): .env"
    skipped=$((skipped + 1))
  elif [[ -f "$ROOT_DIR/.env.example" ]]; then
    cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
    log "Created: .env"
    created=$((created + 1))
  fi

  for example in "$CONFIG_DIR"/*.example.*; do
    [[ -e "$example" ]] || continue
    name="$(basename "$example")"
    target="$CONFIG_DIR/${name/.example./.}"
    rel="configs/${name/.example./.}"
    if [[ -f "$target" ]]; then
      log "Skipped (exists): $rel"
      skipped=$((skipped + 1))
    else
      cp "$example" "$target"
      log "Created: $rel"
      created=$((created + 1))
    fi
  done

  log "Created files: $created, skipped (already existed): $skipped"
}

ensure_config() {
  if [[ ! -f "$CONFIG_DIR/django-config.json" ]]; then
    config
  fi
}

setup() {
  step "Step 2: virtual environment + dependencies"
  [[ -d "$API_DIR/src/sse_api" ]] || { err "Missing src/sse_api/ directory — are you inside the repo?"; exit 1; }

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
  log "Installing dependencies: requirements.txt"
  "$pip" install --upgrade pip >/dev/null
  "$pip" install -r "$API_DIR/requirements.txt"
  "$pip" install --no-deps -e "$API_DIR"

  log "Installing dependencies from git (radlab-data, llm-router)"
  ( cd "$API_DIR/src/sse_api" && ./initialize.sh dep )
}

infra() {
  step "Steps 3-4: infrastructure (PostgreSQL + Milvus)"
  if [[ "$NO_INFRA" == "1" ]]; then
    warn "Skipping PG/Milvus startup (--no-infra). Assuming they are already running."
    return 0
  fi
  have docker || need docker
  ensure_volumes

  log "PostgreSQL: bash scripts/admin/postgres.sh"
  ( cd "$ROOT_DIR" && bash scripts/admin/postgres.sh )

  log "Milvus standalone (ports 19530, 19121)"
  if container_running milvus-standalone; then
    log "Milvus is already running (data in the 'milvus_data' volume) — keeping the container."
  else
    docker_cmd rm -f milvus-standalone >/dev/null 2>&1 || true
    # Milvus >= 3.0 refuses to start with an embedded etcd unless the
    # deployment mode is marked as standalone from the very beginning.
    docker_cmd run -d --name milvus-standalone \
      --security-opt seccomp:unconfined \
      -e DEPLOY_MODE=STANDALONE \
      -e ETCD_USE_EMBED=true \
      -e ETCD_DATA_DIR=/var/lib/milvus/etcd \
      -e COMMON_STORAGETYPE=local \
      -v milvus_data:/var/lib/milvus \
      -p 0.0.0.0:19530:19530 -p 0.0.0.0:19121:19121 \
      milvusdb/milvus:v3.0.3 \
      milvus run standalone
  fi

  log "Waiting for ports..."
  wait_port 5471  "PostgreSQL"
  # Milvus 3.0 needs noticeably more time to open its port than 2.x.
  wait_port 19530 "Milvus" 60
}

ensure_infra() {
  if [[ "$NO_INFRA" == "1" ]]; then
    return 0
  fi
  local p_pg=5471
  local p_milvus=19530
  local pg_ready=0 milvus_ready=0

  if (exec 3<>"/dev/tcp/127.0.0.1/${p_pg}") 2>/dev/null; then
    exec 3>&- 2>/dev/null || true
    pg_ready=1
  fi
  if (exec 3<>"/dev/tcp/127.0.0.1/${p_milvus}") 2>/dev/null; then
    exec 3>&- 2>/dev/null || true
    milvus_ready=1
  fi

  if [[ "$pg_ready" == "0" || "$milvus_ready" == "0" ]]; then
    infra
  fi
}

init() {
  step "Step 6: database + account + templates"
  if [[ "$NO_INIT" == "1" ]]; then
    warn "Skipping initialization (--no-init)."
    return 0
  fi
  [[ -d "$API_DIR/src/sse_api" ]] || { err "Missing src/sse_api/"; exit 1; }
  ensure_config
  ensure_infra
  wait_port 5471 "PostgreSQL" 30
  wait_port 19530 "Milvus" 60
  [[ "$NO_VENV" == "1" ]] || activate_venv
  local py; py="$(pick_python)"
  ( cd "$API_DIR" && SSE_CONFIG_DIR="$CONFIG_DIR" "$py" -m sse_api.manage migrate )
  ( cd "$API_DIR" && SSE_CONFIG_DIR="$CONFIG_DIR" "$py" src/sse_tools/admin/prepare_semantic_db.py )
  ( cd "$API_DIR" && SSE_CONFIG_DIR="$CONFIG_DIR" "$py" src/sse_tools/admin/add_org_group_user.py -u configs/user-group-organisation.json )
  ( cd "$API_DIR" && SSE_CONFIG_DIR="$CONFIG_DIR" "$py" src/sse_tools/admin/add_query_template_to_org.py )
}

cosine() {
  step "COSINE: migrating indexes of existing collections (no re-embedding)"
  local script="$SCRIPT_DIR/scripts/admin/milvus_index_to_cosine.py"
  local cfg="$CONFIG_DIR/milvus_config.json"
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
  ( cd "$API_DIR" && SSE_CONFIG_DIR="$CONFIG_DIR" "$py" "$script" "${args[@]}" )
}

start() {
  step "Step 7: starting the API server (foreground) → http://0.0.0.0:8271/api/"
  [[ -d "$API_DIR/src/sse_api" ]] || { err "Missing src/sse_api/"; exit 1; }
  [[ "$NO_VENV" == "1" ]] || activate_venv
  log "Test login: default_admin / password  (POST /api/login)"
  ( cd "$API_DIR" && SSE_CONFIG_DIR="$CONFIG_DIR" ./src/sse_api/run-api.sh )
}

down() {
  step "Stop: stopping and removing the PG + Milvus containers (volumes are kept)"
  have docker || need docker
  remove_sse_containers
  log "Stopped. The data stayed in the volumes: ${SSE_VOLUMES[*]}."
  log "Start it again with ./run-en.sh infra (or ./run-en.sh all)."
  log "To begin from an empty database and empty vectors: ./run-en.sh clean"
}

# Full reset: containers + volumes.  The only command of this script that deletes data.
clean() {
  step "Clean: removing the containers AND the volumes — all SSE data is gone"
  have docker || need docker
  log "Containers: ${SSE_CONTAINERS[*]}"
  log "Volumes:    ${SSE_VOLUMES[*]}"
  if [[ "$ASSUME_YES" != "1" ]]; then
    local answer=""
    printf 'Remove the containers and the volumes (database, vectors)? [y/N]: '
    read -r answer || answer=""
    case "$answer" in
      y|Y|yes|YES) ;;
      *) log "Aborted — nothing was removed."; exit 0 ;;
    esac
  fi
  remove_sse_containers
  local volume
  for volume in "${SSE_VOLUMES[@]}"; do
    if docker_cmd volume rm "$volume" >/dev/null 2>&1; then
      log "Removed the '$volume' volume."
    fi
  done
  log "Cleaned. From scratch: ./run-en.sh all (infra + init), then re-index the collections."
}

status() {
  step "Status"
  if have docker; then
    log "Containers:"
    docker_cmd ps -a --filter name=pg_sse_backend_engine --filter name=milvus-standalone \
      --format '  table {{.Names}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null || true
    local wanted="${SSE_VOLUMES[0]}"
    local volume_name
    for volume_name in "${SSE_VOLUMES[@]:1}"; do wanted+="|${volume_name}"; done
    local existing
    existing="$(docker_cmd volume ls --format '{{.Name}}' 2>/dev/null | grep -E "^(${wanted})$" || true)"
    if [[ -n "$existing" ]]; then
      log "Volumes with data (they outlive ./run-en.sh down):"
      while IFS= read -r volume_name; do
        [[ -n "$volume_name" ]] && log "  $volume_name"
      done <<< "$existing"
    else
      log "Volumes: none (./run-en.sh infra creates them)"
    fi
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
[[ "$NO_VENV" == "1" ]] || activate_venv
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
  config) config ;;
  infra)  infra ;;
  init)   init ;;
  cosine) cosine ;;
  start|up) start ;;
  down)   down ;;
  clean)  clean ;;
  status) status ;;
  help)   help ;;
  *)      help; exit 1 ;;
esac

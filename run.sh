#!/usr/bin/env bash
#
# run.sh — Semantic Search Engine (SSE): a single startup script.
# It joins the steps of SETUP.md: venv + dependencies (step 2), PG + Milvus (3-4),
# (optionally the LLM router, 5), initialization (6), API start (7).
# It also offers "cosine" — migrating the indexes of the existing Milvus
# collections to the COSINE metric (without re-embedding) for search.
#
# Usage:
#   ./run.sh                 # == all (full pipeline: setup + infra + init + start)
#   ./run.sh all             # setup + infra + init + start
#   ./run.sh setup           # venv + pip install + git dependencies (step 2)
#   ./run.sh infra           # start PostgreSQL + Milvus + RabbitMQ in Docker (steps 3-4)
#   ./run.sh init            # migrate + semantic + add_user + add_query_templates (step 6)
#   ./run.sh cosine          # migrate the indexes of existing collections to COSINE (defaults to --all)
#   ./run.sh start|up        # run the API server in the foreground → http://0.0.0.0:8271/api/ (step 7)
#   ./run.sh worker          # run the Celery worker (indexing queue) in the foreground
#   ./run.sh down            # stop and remove the containers (+ the worker); THE DATA VOLUMES ARE KEPT
#   ./run.sh clean           # like down, but it also removes the volumes (postgres_data, milvus_data, rabbitmq_data)
#   ./run.sh status          # show the state of the containers and ports (5471/19530/19121/5672/8271)
#   ./run.sh config          # create .env and the missing configs/ files from the templates
#
# Flags:
#   --no-venv        do not create/use a venv (use the system python/pip)
#   --no-infra       skip starting PG/Milvus/RabbitMQ (assume they are already running)
#   --no-init        skip the migrations/initialization (step 6)
#   --skip-server    in "all": do not start the server after the setup (exit after the initialization)
#   -y|--yes         in "clean": do not ask for confirmation (for CI/scripts)
#
#   # flags of the "cosine" command:
#   --collection X collection to migrate (repeatable; without the flag: all/--all)
#   --index TYPE   type of the new index: HNSW or IVF_FLAT (default: IVF_FLAT)
#   --dry-run      only show the planned operations (no changes in the DB)
#
# Examples:
#   ./run.sh all --no-infra                # PG+Milvus are already running, the rest is automated
#   ./run.sh all --skip-server             # setup only, no foreground start
#   ./run.sh cosine --dry-run              # preview of the COSINE migration (all collections)
#   ./run.sh cosine --collection my_coll   # only one collection
#   ./run.sh config                        # create .env + configs/*.json from the templates
#   ./run.sh down                          # stop the containers; the data in the volumes is intact
#   ./run.sh clean                         # remove the containers + all data (asks first)
#   ./run.sh clean --yes                   # the same without asking
#   ./run.sh status
#
# The indexing queue (Celery + RabbitMQ):
#   ./run.sh infra      # also starts RabbitMQ (127.0.0.1:5672, user/password from .env)
#   ./run.sh worker     # the "indexing" queue worker (foreground)
#   ./run.sh start      # also starts a worker in the background when ENV_USE_CELERY=1 in .env
#   .env: ENV_USE_CELERY=1 — indexing through the queue; status: GET /api/indexing_status
#  The worker runs on the "solo" pool by default (CELERY_WORKER_POOL): reading the
#  files uses a multiprocessing.Pool of its own, while the tasks of a "prefork"
#  pool are daemonic children, which are not allowed to have children.
#
# Volumes and data:
#   The data lives in named Docker volumes (postgres_data, milvus_data,
#   rabbitmq_data), created by every "./run.sh infra".  "./run.sh down" stops and
#   removes the containers but leaves the volumes alone — after the next start the
#   database and the vectors are exactly the same as before.  Only "./run.sh clean"
#   deletes them (containers + volumes).  A PostgreSQL/Milvus mismatch (for
#   instance after a crash in the middle of indexing) is fixed the same way: clean
#   and re-index, because PostgreSQL is the source of truth.
#
# The .env file:
#   When it exists in the root of the repository, it is loaded before every
#   command (set -a, so the variables reach the environment of the child
#   processes: docker compose, initialize.sh, run-api.sh).  Never commit .env —
#   the template is .env.example, and "./run.sh config" creates its local copy.
#
set -euo pipefail

# ---------- locations ----------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd -P)"
ROOT_DIR="$SCRIPT_DIR"
API_DIR="$ROOT_DIR"
CONFIG_DIR="$ROOT_DIR/configs"
VENV_DIR="$ROOT_DIR/.venv"
PY_PREFERRED="python3.11"
MILVUS_IMAGE="milvusdb/milvus:v3.0.3"

export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/sse_lib/src:${PYTHONPATH:-}"
export DJANGO_SETTINGS_MODULE="${DJANGO_SETTINGS_MODULE:-sse_api.config.settings}"

# Default local (loopback) values of the indexing queue; .env overrides them.
export RABBITMQ_USER="${RABBITMQ_USER:-sse}"
export RABBITMQ_PASSWORD="${RABBITMQ_PASSWORD:-sse}"
export RABBITMQ_BIND_HOST="${RABBITMQ_BIND_HOST:-127.0.0.1}"
export RABBITMQ_HOST_PORT="${RABBITMQ_HOST_PORT:-5672}"
export RABBITMQ_MGMT_HOST_PORT="${RABBITMQ_MGMT_HOST_PORT:-15672}"
export ENV_CELERY_BROKER_URL="${ENV_CELERY_BROKER_URL:-amqp://${RABBITMQ_USER}:${RABBITMQ_PASSWORD}@localhost:5672/}"
# The pool of the worker is "solo", because the indexing pipeline itself reads the
# files with a process pool of its own (multiprocessing.Pool), while the processes
# of a "prefork" pool are daemons and may not have children ("daemonic processes
# are not allowed to have children").
export CELERY_WORKER_POOL="${CELERY_WORKER_POOL:-solo}"

# The Celery worker started by ./run.sh start (ENV_USE_CELERY=1) is a background
# process; its PID and its log live here.
WORKER_PID_FILE="$ROOT_DIR/.celery-worker.pid"

# SSE containers and persistent volumes.  The volumes outlive "./run.sh down", are
# created by "./run.sh infra" and are removed only by "./run.sh clean".
# Note: "docker compose" prefixes them with the project name (sse_postgres_data,
# etc.) — that is a different set of volumes, managed by "docker compose down [-v]".
SSE_CONTAINERS=(pg_sse_backend_engine milvus-standalone milvus-minio-init milvus-minio milvus-etcd rabbitmq-sse)
SSE_VOLUMES=(postgres_data milvus_data rabbitmq_data)

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
    -y|--yes)      ASSUME_YES=1 ;;
    --collection)  shift; [[ $# -gt 0 ]] || { err "--collection requires a collection name"; exit 1; }; COSINE_COLLECTIONS+=("$1") ;;
    --index)       shift; [[ $# -gt 0 ]] || { err "--index requires a value (HNSW|IVF_FLAT)"; exit 1; }; COSINE_INDEX="$1" ;;
    -h|--help|help) CMD="help" ;;
    all|setup|infra|init|cosine|config|start|up|down|clean|status) CMD="$1" ;;
    worker) CMD="$1" ;;
    *) warn "Unknown option: $1 (showing help)"; CMD="help" ;;
  esac
  shift
done

# ---------- helpers ----------
have() { command -v "$1" >/dev/null 2>&1; }
need() { err "Missing: $1. Install/start it first."; exit 1; }

load_env() {
  # .env in the root of the repository (created by "./run.sh config") — load it when present.
  if [[ -f "$ROOT_DIR/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.env"
    set +a
    log "Loaded the variables from $ROOT_DIR/.env"
  fi
}

# After load_env: the broker points at the RabbitMQ started by ./run.sh infra, so
# it works on localhost without any extra configuration.
apply_celery_defaults() {
  # An empty broker or a template CHANGE_ME still has to point at the RabbitMQ of
  # "./run.sh infra" — fill in the credentials and the port from .env.
  if [[ -z "${ENV_CELERY_BROKER_URL:-}" || "${ENV_CELERY_BROKER_URL}" == *CHANGE_ME* ]]; then
    export ENV_CELERY_BROKER_URL="amqp://${RABBITMQ_USER:-sse}:${RABBITMQ_PASSWORD:-sse}@localhost:${RABBITMQ_HOST_PORT:-5672}/"
    log "ENV_CELERY_BROKER_URL: using the local broker (the RabbitMQ of ./run.sh infra)."
  fi
}

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

stop_worker() {
  if [[ -f "$WORKER_PID_FILE" ]] && kill -0 "$(cat "$WORKER_PID_FILE")" 2>/dev/null; then
    kill "$(cat "$WORKER_PID_FILE")" 2>/dev/null || true
    log "Stopped the Celery worker (PID $(cat "$WORKER_PID_FILE"))."
  fi
  rm -f "$WORKER_PID_FILE"
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
# Is a container of this name running right now?  Used not to recreate a container
# whose data already sits in a volume.
container_running() {
  [[ -n "$(docker_cmd ps -q --filter "name=^${1}\$" 2>/dev/null)" ]]
}

# The named volumes are created up front instead of as a side effect of 'docker run',
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
  step "Configuration: .env + the configuration files from the templates"
  local created=0 skipped=0 example target name rel

  if [[ -f "$ROOT_DIR/.env" ]]; then
    log "Skipped (exists): .env"
    skipped=$((skipped + 1))
  elif [[ -f "$ROOT_DIR/.env.example" ]]; then
    cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
    log "Created: .env"
    created=$((created + 1))
  else
    err "Missing .env.example template — cannot create .env"
    exit 1
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

  log "Files created: $created, skipped (already existed): $skipped"
  warn "PRODUCTION: replace EVERY CHANGE_ME value with your own secret:"
  warn "  .env → ENV_SECRET_KEY, ENV_DB_PASSWORD/POSTGRES_PASSWORD,"
  warn "  MINIO_ACCESS_KEY/MINIO_SECRET_KEY; also ENV_DEBUG=0 and ENV_ALLOWED_HOSTS."
  warn "RABBITMQ_USER/RABBITMQ_PASSWORD are deliberately simple (sse/sse) — the broker"
  warn "listens on loopback only; change them when you publish the port elsewhere."
}

setup() {
  step "Step 2: virtual environment + dependencies"
  [[ -d "$API_DIR/src/sse_api" ]] || { err "Missing src/sse_api/ directory — are you inside the repository?"; exit 1; }

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
  log "Installing the dependencies: requirements.txt"
  "$pip" install --upgrade pip --break-system-packages >/dev/null
  "$pip" install -r "$API_DIR/requirements.txt" --break-system-packages

  "$pip" install --no-deps -e "$API_DIR" --break-system-packages

  log "Installing the dependencies from git (radlab-data, llm-router)"
  ( cd "$API_DIR/src/sse_api" && ./initialize.sh dep )
}

infra() {
  step "Steps 3-4: infrastructure (PostgreSQL + Milvus + RabbitMQ)"
  if [[ "$NO_INFRA" == "1" ]]; then
    warn "Skipping the PG/Milvus/RabbitMQ startup (--no-infra). Assuming they are already running."
    return 0
  fi
  have docker || need docker
  ensure_volumes

  log "PostgreSQL: bash scripts/admin/postgres.sh"
  ( cd "$ROOT_DIR" && bash scripts/admin/postgres.sh )

  local milvus_host="${MILVUS_HOST_PORT:-19530}"
  local milvus_metrics_host="${MILVUS_METRICS_HOST_PORT:-19121}"
  local milvus_bind="${MILVUS_BIND_HOST:-0.0.0.0}"
  local port_args=()
  if [[ -n "$milvus_bind" ]]; then
    port_args+=(-p "${milvus_bind}:${milvus_host}:19530" -p "${milvus_bind}:${milvus_metrics_host}:19121")
  else
    port_args+=(-p "0.0.0.0:${milvus_host}:19530" -p "0.0.0.0:${milvus_metrics_host}:19121")
  fi

  log "Milvus standalone (ports ${milvus_host}, ${milvus_metrics_host})"
  if container_running milvus-standalone; then
    log "Milvus is already running (the data is in the 'milvus_data' volume) — keeping the container."
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
      "${port_args[@]}" \
      "$MILVUS_IMAGE" \
      milvus run standalone
  fi

  log "Waiting for the ports..."
  wait_port 5471  "PostgreSQL"
  # Milvus 3.0 needs noticeably more time to open its port than 2.x.
  wait_port "$milvus_host" "Milvus" 60

  local rabbit_bind="${RABBITMQ_BIND_HOST:-127.0.0.1}"
  local rabbit_port="${RABBITMQ_HOST_PORT:-5672}"
  local rabbit_mgmt="${RABBITMQ_MGMT_HOST_PORT:-15672}"
  log "RabbitMQ (the Celery broker; ${rabbit_bind}:${rabbit_port} → 5672, management http://${rabbit_bind}:${rabbit_mgmt}, user=${RABBITMQ_USER:-sse})"
  if container_running rabbitmq-sse; then
    log "RabbitMQ is already running (the data is in the 'rabbitmq_data' volume) — keeping the container."
  else
    docker_cmd rm -f rabbitmq-sse >/dev/null 2>&1 || true
    docker_cmd run -d --name rabbitmq-sse \
      -e RABBITMQ_DEFAULT_USER="${RABBITMQ_USER:-sse}" \
      -e RABBITMQ_DEFAULT_PASS="${RABBITMQ_PASSWORD:-sse}" \
      -v rabbitmq_data:/var/lib/rabbitmq \
      -p "${rabbit_bind}:${rabbit_port}:5672" \
      -p "${rabbit_bind}:${rabbit_mgmt}:15672" \
      rabbitmq:3.13-management
  fi

  log "Waiting for RabbitMQ..."
  wait_port "$rabbit_port" "RabbitMQ (5672)"
  wait_port "$rabbit_mgmt" "RabbitMQ management (15672)" 20
}

ensure_config() {
  if [[ ! -f "$CONFIG_DIR/django-config.json" ]]; then
    config
  fi
}

ensure_infra() {
  if [[ "$NO_INFRA" == "1" ]]; then
    return 0
  fi
  local p_pg=5471
  local p_milvus="${MILVUS_HOST_PORT:-19530}"
  local p_rabbit="${RABBITMQ_HOST_PORT:-5672}"
  local pg_ready=0 milvus_ready=0 rabbit_ready=0

  if (exec 3<>"/dev/tcp/127.0.0.1/${p_pg}") 2>/dev/null; then
    exec 3>&- 2>/dev/null || true
    pg_ready=1
  fi
  if (exec 3<>"/dev/tcp/127.0.0.1/${p_milvus}") 2>/dev/null; then
    exec 3>&- 2>/dev/null || true
    milvus_ready=1
  fi
  if (exec 3<>"/dev/tcp/127.0.0.1/${p_rabbit}") 2>/dev/null; then
    exec 3>&- 2>/dev/null || true
    rabbit_ready=1
  fi

  if [[ "$pg_ready" == "0" || "$milvus_ready" == "0" || "$rabbit_ready" == "0" ]]; then
    infra
  fi
}

check_celery() {
  if ! have celery; then
    err "Missing the 'celery' command in the python environment. Install the dependencies: ./run.sh setup"
  fi
}

# The arguments of the Celery worker; "solo" does not accept --concurrency (it is 1 anyway).
celery_worker_args() {
  local pool="${CELERY_WORKER_POOL:-solo}"
  CELERY_WORKER_ARGS=(-A sse_api.config worker -Q indexing -l INFO --pool "$pool")
  if [[ "$pool" != "solo" ]]; then
    CELERY_WORKER_ARGS+=(--concurrency "${CELERY_CONCURRENCY:-1}")
  fi
}

init() {
  step "Step 6: database initialization + account + templates"
  if [[ "$NO_INIT" == "1" ]]; then
    warn "Skipping the initialization (--no-init)."
    return 0
  fi
  [[ -d "$API_DIR/src/sse_api" ]] || { err "Missing src/sse_api/"; exit 1; }
  ensure_config
  ensure_infra
  wait_port 5471 "PostgreSQL" 30
  wait_port "${MILVUS_HOST_PORT:-19530}" "Milvus" 60
  [[ "$NO_VENV" == "1" ]] || activate_venv
  local py; py="$(pick_python)"
  ( cd "$API_DIR" && SSE_CONFIG_DIR="$CONFIG_DIR" "$py" -m sse_api.manage makemigrations )
  ( cd "$API_DIR" && SSE_CONFIG_DIR="$CONFIG_DIR" "$py" -m sse_api.manage migrate )
  ( cd "$API_DIR" && SSE_CONFIG_DIR="$CONFIG_DIR" "$py" src/sse_tools/admin/prepare_semantic_db.py )
  ( cd "$API_DIR" && SSE_CONFIG_DIR="$CONFIG_DIR" "$py" src/sse_tools/admin/add_org_group_user.py -u configs/user-group-organisation.json )
  ( cd "$API_DIR" && SSE_CONFIG_DIR="$CONFIG_DIR" "$py" src/sse_tools/admin/add_query_template_to_org.py )
}

cosine() {
  step "COSINE: migrating the indexes of the existing collections (without re-embedding)"
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

  # --all needs a connection (listing the collections); --collection + --dry-run works offline
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

  if [[ "${ENV_USE_CELERY:-0}" == "1" || "${ENV_USE_CELERY:-0}" == "true" ]]; then
    check_celery
    log "ENV_USE_CELERY=1 — starting the Celery worker in the background (the 'indexing' queue)."
    mkdir -p "$ROOT_DIR/logs"
    local -a CELERY_WORKER_ARGS
    celery_worker_args
    ( cd "$API_DIR" && SSE_CONFIG_DIR="$CONFIG_DIR" celery "${CELERY_WORKER_ARGS[@]}" \
        >>"$ROOT_DIR/logs/celery-worker.log" 2>&1 &
      echo $! >"$WORKER_PID_FILE" )
    log "Worker log: logs/celery-worker.log (PID $(cat "$WORKER_PID_FILE" 2>/dev/null || echo '?'))"
    wait_port "${RABBITMQ_HOST_PORT:-5672}" "RabbitMQ" 5
    warn "The worker runs in the background — stop it with ./run.sh down (or: kill $(cat "$WORKER_PID_FILE" 2>/dev/null || echo '?'))."
  else
    warn "ENV_USE_CELERY is off — every upload is indexed inside the request that sent it."
    warn "Large uploads can outlive GUNICORN_TIMEOUT (300s) and block a worker for minutes."
    warn "For a production deployment set ENV_USE_CELERY=1 and keep the worker running."
  fi

  log "Test login: default_admin / password  (POST /api/login)"
  warn "In a production environment change the password of default_admin immediately."
  ( cd "$API_DIR" && SSE_CONFIG_DIR="$CONFIG_DIR" ./src/sse_api/run-api.sh )
}

worker() {
  step "Celery worker (the indexing queue) → foreground"
  [[ -d "$API_DIR/src/sse_api" ]] || { err "Missing src/sse_api/"; exit 1; }
  [[ "$NO_VENV" == "1" ]] || activate_venv
  check_celery
  wait_port "${RABBITMQ_HOST_PORT:-5672}" "RabbitMQ" 30
  local -a CELERY_WORKER_ARGS
  celery_worker_args
  log "Worker: pool=${CELERY_WORKER_POOL:-solo}, queue=indexing"
  log "Test login: default_admin / password  (POST /api/login)"
  ( cd "$API_DIR" && SSE_CONFIG_DIR="$CONFIG_DIR" celery "${CELERY_WORKER_ARGS[@]}" )
}

down() {
  step "Stop: stopping and removing the PG + Milvus + RabbitMQ containers (the volumes are kept)"
  stop_worker
  have docker || need docker
  remove_sse_containers
  docker_cmd network rm sse-net >/dev/null 2>&1 || true
  log "Stopped. The data stayed in the volumes: ${SSE_VOLUMES[*]}."
  log "You get it back with ./run.sh infra (or ./run.sh all)."
  log "To start from an empty database and empty vectors: ./run.sh clean"
}

# Full reset: containers + volumes.  It is the only command of this script that deletes data.
clean() {
  step "Clean: removing the containers AND the volumes — all the SSE data is gone"
  have docker || need docker
  log "Containers: ${SSE_CONTAINERS[*]}"
  log "Volumes:    ${SSE_VOLUMES[*]}"
  if [[ "$ASSUME_YES" != "1" ]]; then
    local answer=""
    printf 'Remove the containers and the volumes (database, vectors, queue)? [y/N]: '
    read -r answer || answer=""
    case "$answer" in
      y|Y|yes|YES) ;;
      *) log "Aborted — nothing was removed."; exit 0 ;;
    esac
  fi
  stop_worker
  remove_sse_containers
  local volume
  for volume in "${SSE_VOLUMES[@]}"; do
    if docker_cmd volume rm "$volume" >/dev/null 2>&1; then
      log "Removed the '$volume' volume."
    fi
  done
  docker_cmd network rm sse-net >/dev/null 2>&1 || true
  log "Cleaned. From scratch: ./run.sh all (infra + init), then re-index the collections."
}

status() {
  step "Status"
  if have docker; then
    log "Containers:"
    docker_cmd ps -a --filter name=pg_sse_backend_engine --filter name=milvus-standalone \
      --filter name=rabbitmq-sse \
      --format '  table {{.Names}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null || true
    local wanted="${SSE_VOLUMES[0]}"
    local volume_name
    for volume_name in "${SSE_VOLUMES[@]:1}"; do wanted+="|${volume_name}"; done
    local existing
    existing="$(docker_cmd volume ls --format '{{.Name}}' 2>/dev/null | grep -E "^(${wanted})$" || true)"
    if [[ -n "$existing" ]]; then
      log "Volumes with data (they outlive ./run.sh down):"
      while IFS= read -r volume_name; do
        [[ -n "$volume_name" ]] && log "  $volume_name"
      done <<< "$existing"
    else
      log "Volumes: none (./run.sh infra creates them)"
    fi
  else
    warn "Docker unavailable — skipped the container list."
  fi
  if [[ -f "$WORKER_PID_FILE" ]] && kill -0 "$(cat "$WORKER_PID_FILE")" 2>/dev/null; then
    log "Celery worker: PID $(cat "$WORKER_PID_FILE") (log: logs/celery-worker.log)"
  else
    log "Celery worker: not running"
  fi
  log "Ports (127.0.0.1):"
  local p
  for p in 5471 19530 19121 5672 8271; do
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
load_env
apply_celery_defaults
[[ "$NO_VENV" == "1" ]] || activate_venv
case "$CMD" in
  all)
    setup
    infra
    init
    if [[ "$SKIP_SERVER" == "1" ]]; then
      warn "--skip-server: skipping the server start (you can run it later: ./run.sh start)."
      exit 0
    fi
    start
    ;;
  setup)  setup ;;
  infra)  infra ;;
  init)   init ;;
  cosine) cosine ;;
  config) config ;;
  start|up) start ;;
  worker) worker ;;
  down)   down ;;
  clean)  clean ;;
  status) status ;;
  help)   help ;;
  *)      help; exit 1 ;;
esac

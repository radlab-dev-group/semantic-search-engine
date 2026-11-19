#!/usr/bin/env bash
#
# run-models.sh — start the SSE model service (embedder + reranker).
#
# A launcher for src/sse_models/: it loads .env, picks a python that really has
# the dependencies of the service, and runs "python -m sse_models" in the
# foreground (Ctrl+C stops it).  The service is independent of the engine: no
# PostgreSQL, no Milvus, no Django — only the model catalog in configs/.
#
# Difference to "./run.sh models": that command uses the venv unless you add
# --no-venv, and .env (sourced with set -a) wins over the variables you pass in
# front of it.  This script tries the interpreters in turn until one can import
# fastapi/uvicorn, and applies its flags AFTER .env, so a flag always wins.
#
# Usage:
#   ./run-models.sh                    # start on the .env/defaults
#   ./run-models.sh --port 8281        # ... on another port
#   ./run-models.sh --role rerank      # only the /v1/rerank endpoint
#   ./run-models.sh --warm             # preload the models before binding the port
#   ./run-models.sh --dry-run          # show the resolved command, start nothing
#   ./run-models.sh --help
#
# Flags:
#   --host ADDR      bind address              (SSE_MODELS_HOST,     default 0.0.0.0)
#   --port PORT      bind port                 (SSE_MODELS_PORT,     default 8280)
#   --role ROLE      all | embed | rerank      (SSE_MODELS_ROLE,     default all)
#   --warm           preload the models        (SSE_MODELS_WARM_MODELS=1)
#   --log-level L    logging level             (SSE_MODELS_LOG_LEVEL, default info)
#   --config-dir DIR directory with embedders.json / rerankers.json (SSE_CONFIG_DIR)
#   --token TOKEN    shared secret for /v1/*   (SSE_MODELS_API_TOKEN)
#   --python PATH    use exactly this interpreter (no guessing)
#   --venv DIR       use exactly this venv (no guessing)
#   --no-venv        skip the venvs, use the system python
#   -n|--dry-run     print what would run and exit
#   -h|--help        this text
#
# Which models are served, on which device, is NOT decided here: it comes from
# configs/embedders.json and configs/rerankers.json (their "active_models" and
# per-model "device"), the same files the engine reads.  See src/sse_models/README.md.
#
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd -P)"
VENV_DIR="$ROOT_DIR/.venv"

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

# ---------- helpers ----------
# "usage", not "help": help is a bash builtin, and a function of that name does
# not exist yet while the options are parsed (the parse loop is below it).
usage() {
  awk 'NR==1{next} /^set -euo pipefail/{exit} {sub(/^# ?/,""); print}' "$0"
}

# ---------- parsing ----------
HOST_OPT=""; PORT_OPT=""; ROLE_OPT=""; LOG_LEVEL_OPT=""; CONFIG_DIR_OPT=""; TOKEN_OPT=""
PY_OPT=""
VENV_MODE="auto"   # auto | force | off
DRY_RUN=0
WARM=0

while (( $# > 0 )); do
  case "$1" in
    --host)        shift; [[ $# -gt 0 ]] || { err "--host requires an address"; exit 1; }; HOST_OPT="$1" ;;
    --port)        shift; [[ $# -gt 0 ]] || { err "--port requires a number"; exit 1; }; PORT_OPT="$1" ;;
    --role)        shift; [[ $# -gt 0 ]] || { err "--role requires all|embed|rerank"; exit 1; }; ROLE_OPT="$1" ;;
    --log-level)   shift; [[ $# -gt 0 ]] || { err "--log-level requires a value"; exit 1; }; LOG_LEVEL_OPT="$1" ;;
    --config-dir)  shift; [[ $# -gt 0 ]] || { err "--config-dir requires a directory"; exit 1; }; CONFIG_DIR_OPT="$1" ;;
    --token)       shift; [[ $# -gt 0 ]] || { err "--token requires a value"; exit 1; }; TOKEN_OPT="$1" ;;
    --python)      shift; [[ $# -gt 0 ]] || { err "--python requires a path"; exit 1; }; PY_OPT="$1" ;;
    --warm)        WARM=1 ;;
    --venv)        shift; [[ $# -gt 0 ]] || { err "--venv requires a directory"; exit 1; }; VENV_DIR="$1"; VENV_MODE="force" ;;
    --no-venv)     VENV_MODE="off" ;;
    -n|--dry-run)  DRY_RUN=1 ;;
    -h|--help|help) usage; exit 0 ;;
    *) err "Unknown option: $1"; usage; exit 1 ;;
  esac
  shift
done

load_env() {
  if [[ -f "$ROOT_DIR/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.env"
    set +a
    log "Loaded the variables from $ROOT_DIR/.env"
  fi
}

has_deps() { "$1" -c 'import fastapi, uvicorn' >/dev/null 2>&1; }

# An activated venv shadows the system python: "command -v python3" then names
# the venv — the very interpreter we are looking away from.  PATH is therefore
# walked by hand and every venv directory is skipped.
system_python() {
  local name="$1" part
  local -a parts=()
  local IFS=':'
  set -f
  parts=(${PATH:-})
  set +f
  for part in ${parts[@]+"${parts[@]}"}; do
    [[ -n "$part" ]] || continue
    case "$part" in
      "$VENV_DIR"/bin|"$ROOT_DIR"/venv/bin|"${VIRTUAL_ENV:-/nonexistent}"/bin) continue ;;
    esac
    if [[ -x "$part/$name" ]]; then printf '%s\n' "$part/$name"; return 0; fi
  done
  return 1
}

# The interpreters to try in order: the project's own venv first, then a python
# from outside every venv.  VENV_MODE: auto (both) | force (only --venv) |
# off (only the system python).  On success PICKED_PY is the winner; on failure
# PROBED says what was tried and why each one was rejected.
PICKED_PY=""
PROBED=()
pick_python() {
  local -a candidates=()
  local name found

  if [[ "$VENV_MODE" != "off" ]]; then
    if [[ "$VENV_MODE" == "auto" && -n "${VIRTUAL_ENV:-}" ]]; then
      candidates+=("$VIRTUAL_ENV/bin/python3" "$VIRTUAL_ENV/bin/python")
    fi
    candidates+=("$VENV_DIR/bin/python3" "$VENV_DIR/bin/python")
    if [[ "$VENV_MODE" == "auto" ]]; then
      candidates+=("$ROOT_DIR/venv/bin/python3" "$ROOT_DIR/venv/bin/python")
    fi
  fi
  if [[ "$VENV_MODE" != "force" ]]; then
    for name in python3 python; do
      if found="$(system_python "$name")"; then candidates+=("$found"); fi
    done
  fi

  PROBED=()
  local seen=" "
  local py real
  for py in ${candidates[@]+"${candidates[@]}"}; do
    [[ -n "$py" ]] || continue
    # venv/bin/python and venv/bin/python3 are the same file: probe it once
    real="$(readlink -f -- "$py" 2>/dev/null || printf '%s' "$py")"
    case "$seen" in *" $real "*) continue ;; esac
    seen+="$real "
    if [[ ! -x "$py" ]]; then
      PROBED+=("$py — not present")
      continue
    fi
    if has_deps "$py"; then
      PICKED_PY="$py"
      return 0
    fi
    PROBED+=("$py — no fastapi/uvicorn")
  done
  return 1
}

# ---------- the service ----------
main() {
  [[ -d "$ROOT_DIR/src/sse_models" ]] || { err "Missing src/sse_models/ — is this script inside the repository?"; exit 1; }

  load_env

  # The flags are applied after .env, which makes them win over the file.
  if [[ -n "$HOST_OPT" ]];      then export SSE_MODELS_HOST="$HOST_OPT"; fi
  if [[ -n "$PORT_OPT" ]];      then export SSE_MODELS_PORT="$PORT_OPT"; fi
  if [[ -n "$ROLE_OPT" ]];      then export SSE_MODELS_ROLE="$ROLE_OPT"; fi
  if [[ -n "$LOG_LEVEL_OPT" ]]; then export SSE_MODELS_LOG_LEVEL="$LOG_LEVEL_OPT"; fi
  if [[ -n "$CONFIG_DIR_OPT" ]];then export SSE_CONFIG_DIR="$CONFIG_DIR_OPT"; fi
  if [[ -n "$TOKEN_OPT" ]];     then export SSE_MODELS_API_TOKEN="$TOKEN_OPT"; fi
  if [[ "$WARM" == "1" ]];      then export SSE_MODELS_WARM_MODELS=1; fi

  export SSE_MODELS_HOST="${SSE_MODELS_HOST:-0.0.0.0}"
  export SSE_MODELS_PORT="${SSE_MODELS_PORT:-8280}"
  export SSE_MODELS_ROLE="${SSE_MODELS_ROLE:-all}"
  export SSE_MODELS_LOG_LEVEL="${SSE_MODELS_LOG_LEVEL:-info}"
  export SSE_CONFIG_DIR="${SSE_CONFIG_DIR:-$ROOT_DIR/configs}"
  export SSE_MODELS_WARM_MODELS="${SSE_MODELS_WARM_MODELS:-0}"

  local py="" py_label=""
  if [[ -n "$PY_OPT" ]]; then
    [[ -x "$PY_OPT" ]] || { err "Not executable: $PY_OPT"; exit 1; }
    py="$PY_OPT"
    py_label="--python"
  else
    if ! pick_python; then
      err "No interpreter with the dependencies of the model service (fastapi, uvicorn).  Tried:"
      local probe
      for probe in ${PROBED[@]+"${PROBED[@]}"}; do err "  $probe"; done
      err "Install them:"
      err "  into the venv:  $VENV_DIR/bin/pip install -r $ROOT_DIR/requirements-models.txt"
      err "  or system-wide: python3 -m pip install -r $ROOT_DIR/requirements-models.txt"
      err "Or point at another interpreter: ./run-models.sh --python /path/to/python"
      exit 1
    fi
    py="$PICKED_PY"
    if [[ "$py" == "$VENV_DIR/bin/python"* || "$py" == "${VIRTUAL_ENV:-/nonexistent}/bin/python"* ]]; then
      py_label="venv"
    else
      py_label="system"
    fi
  fi

  step "Model service role=${SSE_MODELS_ROLE} → http://${SSE_MODELS_HOST}:${SSE_MODELS_PORT}/docs"
  log "interpreter: $py ($py_label)"
  log "config:      $SSE_CONFIG_DIR (warm_models=${SSE_MODELS_WARM_MODELS})"
  if [[ -n "${SSE_MODELS_API_TOKEN:-}" ]]; then
    log "auth:        /v1/* requires 'Authorization: Bearer …' (/healthz and /readyz are open)"
  else
    log "auth:        off (SSE_MODELS_API_TOKEN is empty)"
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    log "Dry run — nothing was started:"
    printf '    %s\n' "PYTHONPATH=$ROOT_DIR/src SSE_CONFIG_DIR=$SSE_CONFIG_DIR $py -m sse_models"
    exit 0
  fi

  cd "$ROOT_DIR"
  export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
  exec "$py" -m sse_models
}

main "$@"

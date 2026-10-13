#!/usr/bin/env bash
#
# run.sh — Semantic Search Engine (SSE): jednonaczyniowy skrypt uruchomieniowy.
# Łączy kroki z URUCHOMIENIE.md: venv+ zależności (krok 2), PG+Milvus (3-4),
# (opcjonalnie LLM router, 5), inicjalizacja (6), start API (7).
# Dodatkowo: "cosine" — migracja istniejących kolekcji Milvus na metrykę COSINE
# (bez re-embedingu) dla wyszukiwania.
#
# Użycie:
#   ./run.sh                 # == all (pełny pipeline: setup+infra+init+start)
#   ./run.sh all             # setup + infra + init + start
#   ./run.sh setup           # venv + pip install + zależności z git (krok 2)
#   ./run.sh infra           # uruchom PostgreSQL + Milvus w Dockerze (kroki 3-4)
#   ./run.sh init            # migrate + semantic + add_user + add_query_templates (krok 6)
#   ./run.sh cosine          # migracja istniejących kolekcji na COSINE (domyślnie --all)
#   ./run.sh start|up        # odpal serwer API w foreground → http://localhost:8271/api/ (krok 7)
#   ./run.sh down            # zatrzymaj i usuń kontenery PG + Milvus
#   ./run.sh status          # pokaż stan kontenerów i portów (5471/19530/19121/8271)
#   ./run.sh config          # utwórz .env i brakujące pliki configs/ z szablonów
#
# Flagi:
#   --no-venv        nie twórz/wykorzystuj venv (użyj systemowego pythona/pipa)
#   --no-infra       pomijaj start PG/Milvus (zakładaj, że już działają)
#   --no-init        pomijaj migracje/inicjalizację (krok 6)
#   --skip-server    w "all": nie startuj serwera po setupie (wyjdź po inicjalizacji)
#
#   # flagi komendy "cosine":
#   --collection X kolekcja do migracji (powtarzalne; bez flagi: wszystkie/--all)
#   --index TYPE   typ nowego indexu: HNSW lub IVF_FLAT (domyślnie: IVF_FLAT)
#   --dry-run      tylko pokaż planowane operacje (bez zmian w DB)
#
# Przykłady:
#   ./run.sh all --no-infra                # PG+Milvus już lecą, reszta sama
#   ./run.sh all --skip-server             # tylko setup, bez startu w foreground
#   ./run.sh cosine --dry-run              # podgląd migracji COSINE (wszystkie kolekcje)
#   ./run.sh cosine --collection my_coll   # tylko jedna kolekcja
#   ./run.sh config                        # utwórz .env + configs/*.json z szablonów
#   ./run.sh status
#
# Plik .env:
#   Jeśli istnieje w root repo, jest wczytywany przed każdą komendą (set -a,
#   czyli zmienne trafiają do środowiska procesów potomnych: docker compose,
#   initialize.sh, run-api.sh). Nigdy nie commituj .env — szablon to
#   .env.example, a jego lokalną kopię tworzy komenda "./run.sh config".
#
set -euo pipefail

# ---------- lokalizacja ----------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd -P)"
ROOT_DIR="$SCRIPT_DIR"
API_DIR="$ROOT_DIR"
CONFIG_DIR="$ROOT_DIR/configs"
VENV_DIR="$ROOT_DIR/.venv"
PY_PREFERRED="python3.11"
MILVUS_IMAGE="milvusdb/milvus:v2.3.0"

export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/sse_lib/src:${PYTHONPATH:-}"

# ---------- logi ----------
if [[ -t 1 ]]; then
  C_INFO=$'\e[32m'; C_WARN=$'\e[33m'; C_ERR=$'\e[31m'; C_DIM=$'\e[2m'; C_RST=$'\e[0m'
else
  C_INFO=""; C_WARN=""; C_ERR=""; C_DIM=""; C_RST=""
fi
log()  { printf '%s[INFO]%s %s\n' "$C_INFO" "$C_RST" "$*"; }
warn() { printf '%s[WARN]%s %s\n' "$C_WARN" "$C_RST" "$*" >&2; }
err()  { printf '%s[ERR ]%s %s\n' "$C_ERR" "$C_RST" "$*" >&2; }
step() { printf '\n%s==> %s%s\n' "$C_DIM" "$*" "$C_RST"; }

# ---------- parsowanie ----------
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
    --collection)  shift; [[ $# -gt 0 ]] || { err "--collection wymaga nazwy kolekcji"; exit 1; }; COSINE_COLLECTIONS+=("$1") ;;
    --index)       shift; [[ $# -gt 0 ]] || { err "--index wymaga wartości (HNSW|IVF_FLAT)"; exit 1; }; COSINE_INDEX="$1" ;;
    -h|--help|help) CMD="help" ;;
    all|setup|infra|init|cosine|config|start|up|down|status) CMD="$1" ;;
    *) warn "Nieznana opcja: $1 (pokażę pomoc)"; CMD="help" ;;
  esac
  shift
done

# ---------- pomocnicze ----------
have() { command -v "$1" >/dev/null 2>&1; }
need() { err "Brak: $1. Zainstaluj/uruchom wcześniej."; exit 1; }

load_env() {
  # .env w root repo (tworzony przez "./run.sh config") — wczytaj, jeżeli istnieje.
  if [[ -f "$ROOT_DIR/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.env"
    set +a
    log "Wczytano zmienne z $ROOT_DIR/.env"
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
    err "Brak pip/pip3"; exit 1
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
    err "Brak python/python3"; exit 1
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
wait_port() {
  local port="$1" name="$2" tries="${3:-30}"
  for _ in $(seq 1 "$tries"); do
    if (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; then
      exec 3>&- 3<&- 2>/dev/null || true
      log "$name nasłuchuje na porcie $port"
      return 0
    fi
    sleep 1
  done
  warn "$name: port $port wciąż niedostępny po ${tries}s — sprawdź ręcznie"
}

# ---------- komendy ----------
config() {
  step "Konfiguracja: .env + pliki konfiguracyjne z szablonów"
  local created=0 skipped=0 example target name rel

  if [[ -f "$ROOT_DIR/.env" ]]; then
    log "Pominięto (istnieje): .env"
    skipped=$((skipped + 1))
  elif [[ -f "$ROOT_DIR/.env.example" ]]; then
    cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
    log "Utworzono: .env"
    created=$((created + 1))
  else
    err "Brak szablonu .env.example — nie mogę utworzyć .env"
    exit 1
  fi

  for example in "$CONFIG_DIR"/*.example.*; do
    [[ -e "$example" ]] || continue
    name="$(basename "$example")"
    target="$CONFIG_DIR/${name/.example./.}"
    rel="configs/${name/.example./.}"
    if [[ -f "$target" ]]; then
      log "Pominięto (istnieje): $rel"
      skipped=$((skipped + 1))
    else
      cp "$example" "$target"
      log "Utworzono: $rel"
      created=$((created + 1))
    fi
  done

  log "Utworzono plików: $created, pominięto (już istniały): $skipped"
  warn "PRODUKCJA: zamień KAŻDĄ wartość CHANGE_ME na własny sekret:"
  warn "  .env → ENV_SECRET_KEY, ENV_DB_PASSWORD/POSTGRES_PASSWORD,"
  warn "  MINIO_ACCESS_KEY/MINIO_SECRET_KEY; dodatkowo ENV_DEBUG=0 i ENV_ALLOWED_HOSTS."
}

setup() {
  step "Krok 2: wirtualne środowisko + zależności"
  [[ -d "$API_DIR/src/sse_api" ]] || { err "Brak katalogu src/sse_api/ — czy jesteś w repo?"; exit 1; }

  local py="$PY_PREFERRED"
  have "$py" || { py="python3"; warn "$PY_PREFERRED nie znaleziony, używam python3"; }
  have "$py" || need python3

  if [[ "$NO_VENV" == "1" ]]; then
    warn "Pomijam venv (--no-venv); używam systemowego pythona/pipa."
  else
    if [[ ! -d "$VENV_DIR" ]]; then
      log "Tworzę venv: $VENV_DIR"
      "$py" -m venv "$VENV_DIR"
    else
      log "venv już istnieje: $VENV_DIR"
    fi
    activate_venv
    log "Aktywny python: $(command -v python3) ($(python3 --version 2>&1))"
  fi

  local pip; pip="$(pick_pip)"
  log "Instaluję zależności: requirements.txt"
  "$pip" install --upgrade pip --break-system-packages >/dev/null
  "$pip" install -r "$API_DIR/requirements.txt" --break-system-packages

  "$pip" install --no-deps -e "$API_DIR" --break-system-packages

  log "Instaluję zależności z git (radlab-data, llm-router)"
  ( cd "$API_DIR/src/sse_api" && ./initialize.sh dep )
}

infra() {
  step "Kroki 3-4: infrastruktura (PostgreSQL + Milvus)"
  if [[ "$NO_INFRA" == "1" ]]; then
    warn "Pomijam start PG/Milvus (--no-infra). Zakładam, że już działają."
    return 0
  fi
  have docker || need docker

  log "PostgreSQL: bash scripts/admin/postgres.sh"
  ( cd "$ROOT_DIR" && bash scripts/admin/postgres.sh )

  local milvus_host="${MILVUS_HOST_PORT:-19530}"
  local milvus_metrics_host="${MILVUS_METRICS_HOST_PORT:-19121}"
  local milvus_bind="${MILVUS_BIND_HOST:-}"
  local port_args=()
  if [[ -n "$milvus_bind" ]]; then
    port_args+=(-p "${milvus_bind}:${milvus_host}:19530" -p "${milvus_bind}:${milvus_metrics_host}:19121")
  else
    port_args+=(-p "${milvus_host}:19530" -p "${milvus_metrics_host}:19121")
  fi

  log "Milvus standalone (porty ${milvus_host}, ${milvus_metrics_host})"
  docker_cmd rm -f milvus-standalone >/dev/null 2>&1 || true
  docker_cmd run -d --name milvus-standalone \
    "${port_args[@]}" \
    "$MILVUS_IMAGE"

  log "Odczekuję na porty..."
  wait_port 5471  "PostgreSQL"
  wait_port "$milvus_host" "Milvus"
}

init() {
  step "Krok 6: inicjalizacja bazy + konto + szablony"
  if [[ "$NO_INIT" == "1" ]]; then
    warn "Pomijam inicjalizację (--no-init)."
    return 0
  fi
  [[ -d "$API_DIR/src/sse_api" ]] || { err "Brak src/sse_api/"; exit 1; }
  [[ "$NO_VENV" == "1" ]] || activate_venv
  local py; py="$(pick_python)"
  ( cd "$API_DIR" && SSE_CONFIG_DIR="$CONFIG_DIR" "$py" -m sse_api.manage migrate )
  ( cd "$API_DIR" && SSE_CONFIG_DIR="$CONFIG_DIR" "$py" src/sse_tools/admin/prepare_semantic_db.py )
  ( cd "$API_DIR" && SSE_CONFIG_DIR="$CONFIG_DIR" "$py" src/sse_tools/admin/add_org_group_user.py -u configs/user-group-organisation.json )
  ( cd "$API_DIR" && SSE_CONFIG_DIR="$CONFIG_DIR" "$py" src/sse_tools/admin/add_query_template_to_org.py )
}

cosine() {
  step "COSINE: migracja indexów istniejących kolekcji (bez re-embedingu)"
  local script="$SCRIPT_DIR/scripts/admin/milvus_index_to_cosine.py"
  local cfg="$CONFIG_DIR/milvus_config.json"
  [[ -f "$script" ]] || { err "Brak skryptu migracji: $script"; exit 1; }
  [[ -f "$cfg" ]]    || { err "Brak configu Milvus: $cfg"; exit 1; }
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
  log "Migracja COSINE: collections=$scope, index=$COSINE_INDEX, dry_run=$DRY_RUN"

  # --all wymaga połączenia (listowanie kolekcji); --collection + --dry-run działa offline
  if ! { (( DRY_RUN )) && (( ${#COSINE_COLLECTIONS[@]} )); }; then
    wait_port 19530 "Milvus" 5
  fi

  local py; py="$(pick_python)"
  ( cd "$API_DIR" && SSE_CONFIG_DIR="$CONFIG_DIR" "$py" "$script" "${args[@]}" )
}

start() {
  step "Krok 7: start serwera API (foreground) → http://localhost:8271/api/"
  [[ -d "$API_DIR/src/sse_api" ]] || { err "Brak src/sse_api/"; exit 1; }
  [[ "$NO_VENV" == "1" ]] || activate_venv
  log "Login testowy: default_admin / password  (POST /api/login)"
  warn "W środowisku produkcyjnym natychmiast zmień hasło konta default_admin."
  ( cd "$API_DIR" && SSE_CONFIG_DIR="$CONFIG_DIR" ./src/sse_api/run-api.sh )
}

down() {
  step "Stop: zatrzymuję i usuwam kontenery PG + Milvus"
  have docker || need docker
  docker_cmd rm -f pg_sse_backend_engine >/dev/null 2>&1 || true
  docker_cmd rm -f milvus-standalone    >/dev/null 2>&1 || true
  docker_cmd rm -f milvus-minio-init    >/dev/null 2>&1 || true
  docker_cmd rm -f milvus-minio         >/dev/null 2>&1 || true
  docker_cmd rm -f milvus-etcd          >/dev/null 2>&1 || true
  docker_cmd network rm sse-net         >/dev/null 2>&1 || true
  log "Zatrzymano."
}

status() {
  step "Status"
  if have docker; then
    log "Kontenery:"
    docker_cmd ps -a --filter name=pg_sse_backend_engine --filter name=milvus-standalone \
      --format '  table {{.Names}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null || true
  else
    warn "Docker niedostępny — pominąłem listę kontenerów."
  fi
  log "Porty (127.0.0.1):"
  local p
  for p in 5471 19530 19121 8271; do
    if (exec 3<>"/dev/tcp/127.0.0.1/${p}") 2>/dev/null; then
      exec 3>&- 2>/dev/null || true
      log "  :$p OTWARTY"
    else
      log "  :$p zamknięty"
    fi
  done
}

help() {
  awk 'NR==1{next} /^set -euo pipefail/{exit} {sub(/^# ?/,""); print}' "$0"
}

# ---------- dispatch ----------
load_env
[[ "$NO_VENV" == "1" ]] || activate_venv
case "$CMD" in
  all)
    setup
    infra
    init
    if [[ "$SKIP_SERVER" == "1" ]]; then
      warn "--skip-server: pomijam start serwera (możesz odpalić: ./run.sh start)."
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
  down)   down ;;
  status) status ;;
  help)   help ;;
  *)      help; exit 1 ;;
esac

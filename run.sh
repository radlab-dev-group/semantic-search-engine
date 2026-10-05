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
#   ./run.sh infra           # uruchom PostgreSQL + Milvus (etcd + MinIO) w Dockerze (kroki 3-4)
#   ./run.sh init            # migrate + semantic + add_user + add_query_templates (krok 6)
#   ./run.sh cosine          # migracja istniejących kolekcji na COSINE (domyślnie --all)
#   ./run.sh start|up        # odpal serwer API w foreground → http://localhost:8271/api/ (krok 7)
#   ./run.sh down            # zatrzymaj i usuń kontenery PG + Milvus (etcd/MinIO)
#   ./run.sh status          # pokaż stan kontenerów i portów (5471/19530/19121/8271)
#   ./run.sh config          # utwórz .env i brakujące pliki sse_rest_api/configs/ z szablonów
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
API_DIR="$ROOT_DIR/sse_rest_api"
VENV_DIR="$ROOT_DIR/.venv"
PY_PREFERRED="python3.11"
# Obrazy i sieć Dockera stosu Milvus standalone (te same wersje jak w
# docker-compose.yml — utrzymuj oba miejsca synchronicznie).
SSE_NETWORK="sse-net"
ETCD_IMAGE="quay.io/coreos/etcd:v3.5.5"
MINIO_IMAGE="minio/minio:RELEASE.2023-03-20T20-16-18Z"
MINIO_MC_IMAGE="minio/mc:RELEASE.2023-03-22T18-34-53Z"
MILVUS_IMAGE="milvusdb/milvus:2.3.0"

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
  if [[ -f "$VENV_DIR/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
  fi
}
pick_pip() {
  if have pip; then echo "pip"; elif have pip3; then echo "pip3"; else err "Brak pip/pip3"; exit 1; fi
}
pick_python() {
  if have python; then echo "python"
  elif have python3; then echo "python3"
  else err "Brak python/python3"; exit 1
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

  for example in "$API_DIR"/configs/*.example.*; do
    [[ -e "$example" ]] || continue
    name="$(basename "$example")"
    target="$API_DIR/configs/${name/.example./.}"
    rel="sse_rest_api/configs/${name/.example./.}"
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
  [[ -d "$API_DIR" ]] || { err "Brak katalogu sse_rest_api/ — czy jesteś w repo?"; exit 1; }

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
  log "Instaluję zależności: sse_rest_api/requirements.txt"
  "$pip" install --upgrade pip >/dev/null
  "$pip" install -r "$API_DIR/requirements.txt"

  log "Instaluję zależności z git (radlab-data, llm-router)"
  ( cd "$API_DIR" && ./initialize.sh dep )
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

  # Milvus standalone nie jest samowystarczalny: potrzebuje etcd (metadane) i
  # MinIO (storage obiektowy). Sam obraz Milvus nigdy nie osiągnie healthy i
  # nie ma trwałych danych, więc tworzę cały tercet — w jednej sieci i na
  # nazwanych wolumenach (lustrzane odbicie serwisów milvus/etcd/minio z compose).
  local minio_user="${MINIO_ACCESS_KEY:-}"
  local minio_pass="${MINIO_SECRET_KEY:-}"
  if [[ -z "$minio_user" || -z "$minio_pass" ]]; then
    minio_user="minioadmin"; minio_pass="minioadmin"
    warn "MINIO_ACCESS_KEY/MINIO_SECRET_KEY nie ustawione — używam wbudowanych danych minioadmin."
    warn "Lokalnie OK; w produkcji ustaw własne sekrety w .env (docker-compose.yml ich wymaga)."
  elif [[ "$minio_user" == "CHANGE_ME" || "$minio_pass" == "CHANGE_ME" ]]; then
    warn "MINIO_ACCESS_KEY/MINIO_SECRET_KEY nadal zawierają placeholder CHANGE_ME — w produkcji je wymień."
  fi
  local milvus_host="${MILVUS_HOST_PORT:-19530}"
  local milvus_metrics_host="${MILVUS_METRICS_HOST_PORT:-19121}"
  # Domyślnie loopback (jak w docker-compose.yml); MILVUS_BIND_HOST=0.0.0.0 tylko w zaufanej sieci.
  local milvus_bind="${MILVUS_BIND_HOST:-127.0.0.1}"

  docker network create "$SSE_NETWORK" >/dev/null 2>&1 || true

  log "etcd — metadane Milvus (wolumen sse_etcd_data)"
  docker rm -f milvus-etcd >/dev/null 2>&1 || true
  docker run -d --name milvus-etcd \
    --network "$SSE_NETWORK" --network-alias etcd --restart unless-stopped \
    -e ETCD_AUTO_COMPACTION_MODE=revision \
    -e ETCD_AUTO_COMPACTION_RETENTION=1000 \
    -e ETCD_QUOTA_BACKEND_BYTES=4294967296 \
    -e ETCD_SNAPSHOT_COUNT=50000 \
    -v sse_etcd_data:/etcd \
    "$ETCD_IMAGE" \
    etcd -advertise-client-urls=http://etcd:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  log "MinIO — storage obiektowy Milvus (wolumen sse_minio_data, bez portów na hoście)"
  docker rm -f milvus-minio >/dev/null 2>&1 || true
  docker run -d --name milvus-minio \
    --network "$SSE_NETWORK" --network-alias minio --restart unless-stopped \
    -e MINIO_ACCESS_KEY="$minio_user" \
    -e MINIO_SECRET_KEY="$minio_pass" \
    -v sse_minio_data:/minio_data \
    "$MINIO_IMAGE" minio server /minio_data

  log "Bucket MinIO milvus-bucket (czekam na serwis, maks. ~2 min)"
  docker rm -f milvus-minio-init >/dev/null 2>&1 || true
  docker run --rm --name milvus-minio-init \
    --network "$SSE_NETWORK" \
    -e MINIO_ACCESS_KEY="$minio_user" \
    -e MINIO_SECRET_KEY="$minio_pass" \
    --entrypoint /bin/sh "$MINIO_MC_IMAGE" -c \
    'i=0; until mc alias set local http://minio:9000 "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY"; do i=$((i+1)); [ "$i" -ge 60 ] && break; sleep 2; done; mc mb -p local/milvus-bucket || true'

  log "Milvus standalone (${milvus_bind}:${milvus_host} SDK, ${milvus_bind}:${milvus_metrics_host} metryki)"
  docker rm -f milvus-standalone >/dev/null 2>&1 || true
  docker run -d --name milvus-standalone \
    --network "$SSE_NETWORK" --restart unless-stopped \
    --security-opt seccomp=unconfined \
    -e ETCD_ENDPOINTS=etcd:2379 \
    -e MINIO_ADDRESS=minio:9000 \
    -e COMMON_SECURITY_AUTHORIZATIONENABLED="${MILVUS_AUTH_ENABLED:-false}" \
    -v sse_milvus_data:/var/lib/milvus \
    -p "${milvus_bind}:${milvus_host}:19530" \
    -p "${milvus_bind}:${milvus_metrics_host}:19121" \
    "$MILVUS_IMAGE" milvus run standalone

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
  [[ -d "$API_DIR" ]] || { err "Brak sse_rest_api/"; exit 1; }
  [[ "$NO_VENV" == "1" ]] || activate_venv
  ( cd "$API_DIR" && ./initialize.sh migrate )
  ( cd "$API_DIR" && ./initialize.sh semantic )
  ( cd "$API_DIR" && ./initialize.sh add_user )
  ( cd "$API_DIR" && ./initialize.sh add_query_templates )
}

cosine() {
  step "COSINE: migracja indexów istniejących kolekcji (bez re-embedingu)"
  local script="$SCRIPT_DIR/scripts/admin/milvus_index_to_cosine.py"
  local cfg="$API_DIR/configs/milvus_config.json"
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
  ( cd "$API_DIR" && "$py" "$script" "${args[@]}" )
}

start() {
  step "Krok 7: start serwera API (foreground) → http://localhost:8271/api/"
  [[ -d "$API_DIR" ]] || { err "Brak sse_rest_api/"; exit 1; }
  [[ "$NO_VENV" == "1" ]] || activate_venv
  log "Login testowy: default_admin / password  (POST /api/login)"
  warn "W środowisku produkcyjnym natychmiast zmień hasło konta default_admin."
  ( cd "$API_DIR" && ./run-api.sh )
}

down() {
  step "Stop: zatrzymuję i usuwam kontenery PG + Milvus (etcd/MinIO)"
  have docker || need docker
  docker rm -f pg_sse_backend_engine >/dev/null 2>&1 || true
  docker rm -f milvus-standalone    >/dev/null 2>&1 || true
  docker rm -f milvus-minio-init    >/dev/null 2>&1 || true
  docker rm -f milvus-minio         >/dev/null 2>&1 || true
  docker rm -f milvus-etcd          >/dev/null 2>&1 || true
  docker network rm "$SSE_NETWORK"  >/dev/null 2>&1 || true
  log "Zatrzymano."
}

status() {
  step "Status"
  if have docker; then
    log "Kontenery:"
    docker ps -a --filter name=pg_sse_backend_engine --filter name=milvus-standalone \
      --filter name=milvus-etcd --filter name=milvus-minio \
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

#!/usr/bin/env bash
#
# run-rest-api-gunicorn.sh — REST API na serwerze gunicorn (WSGI).
#
# Robi to samo co run-api.sh (te same zmienne środowiskowe i ten sam port
# 8271), ale zamiast deweloperskiego "manage.py runserver" używa gunicorn
# z konfiguracją gunicorn.conf.py.  Serwer startuje na pierwszym planie:
# logi idą na stdout/stderr, zatrzymanie to Ctrl+C albo ./run.sh down.
#
# Użycie:
#   ./run-rest-api-gunicorn.sh                        # 0.0.0.0:8271
#   ./run-rest-api-gunicorn.sh --port 8080
#   ./run-rest-api-gunicorn.sh --bind 127.0.0.1:9000 --workers 2
#   ./run.sh start                                    # to samo przez run.sh
#
# Port: 8271, czyli dokładnie tak, jak ./run.sh start do tej pory.  Kolejność
# wyboru: --bind > --port > REST_API_BIND > REST_API_PORT > 0.0.0.0:8271.
# GUNICORN_BIND z .env (8000) jest tu świadomie nadpisywany, żeby "start" nie
# zmienił adresu względem run-api.sh; produkcyjne wiązanie z .env wraz z
# walidacją sekretów daje ./run-api-prod.sh.
#
# Flagi:
#   --port N       port do nasłuchu (nadpisuje REST_API_PORT / GUNICORN_BIND)
#   --bind ADRES   pełny adres "host:port" (ma pierwszeństwo przed --port)
#   --workers N    liczba workerów gunicorn (nadpisuje GUNICORN_WORKERS)
#   -h|--help      ten komunikat
#
# Pozostałe parametry gunicorn (wątki, timeouty, klasa workera, logi) pochodzą
# z gunicorn.conf.py i zmiennych GUNICORN_* z .env.
#
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd -P)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd -P)"
GUNICORN_CONF="${SCRIPT_DIR}/gunicorn.conf.py"

# Taki port jak ./run.sh start (run-api.sh -> manage.py runserver 0.0.0.0:8271).
DEFAULT_PORT="8271"

fail() { echo -e "\e[31m[ERROR]\e[0m $*" >&2; exit 1; }
info() { echo -e "\e[32m[INFO]\e[0m $*"; }

usage() {
    sed -n '2,26p' "${BASH_SOURCE[0]:-$0}" | sed 's/^#\{1,2\} \{0,1\}//'
}

# --- argumenty ---------------------------------------------------------------
BIND_ADDRESS=""
PORT=""
WORKERS=""

if (( $# > 0 )); then
    while (( $# > 0 )); do
        case "$1" in
            --port)
                [[ -n "${2:-}" ]] || fail "--port wymaga wartości (np. --port 8080)."
                PORT="$2"; shift 2 ;;
            --port=*)
                PORT="${1#*=}"; shift ;;
            --bind)
                [[ -n "${2:-}" ]] || fail "--bind wymaga wartości (np. --bind 127.0.0.1:8080)."
                BIND_ADDRESS="$2"; shift 2 ;;
            --bind=*)
                BIND_ADDRESS="${1#*=}"; shift ;;
            --workers)
                [[ -n "${2:-}" ]] || fail "--workers wymaga wartości (np. --workers 2)."
                WORKERS="$2"; shift 2 ;;
            --workers=*)
                WORKERS="${1#*=}"; shift ;;
            -h|--help)
                usage; exit 0 ;;
            *)
                fail "Nieznany argument: $1 (dostępne: --port, --bind, --workers, --help)" ;;
        esac
    done
fi

[[ -f "$GUNICORN_CONF" ]] || fail "Brak konfiguracji gunicorn: ${GUNICORN_CONF}"

# --- .env (bez znaczenia dla --bind/--port, resztę zostawiamy jak jest) -----
if [[ -f "${ROOT_DIR}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/.env"
    set +a
fi

# --- środowisko tak jak w run-api.sh ----------------------------------------
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/sse_lib/src:${PYTHONPATH:-}"
export DJANGO_SETTINGS_MODULE="${DJANGO_SETTINGS_MODULE:-sse_api.config.settings}"
export SSE_CONFIG_DIR="${SSE_CONFIG_DIR:-${ROOT_DIR}/configs}"

if [[ -f "${ROOT_DIR}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/.venv/bin/activate"
elif [[ -f "${ROOT_DIR}/venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/venv/bin/activate"
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# DATA_UPLOAD_MAX_MEMORY_SIZE: 2621440000 = 2500 * 1024 * 1024
export DATA_UPLOAD_MAX_MEMORY_SIZE="${DATA_UPLOAD_MAX_MEMORY_SIZE:-2621440000}"
export FILE_UPLOAD_MAX_MEMORY_SIZE="${FILE_UPLOAD_MAX_MEMORY_SIZE:-2621440000}"

# Wartości placeholderowe tylko wtedy, gdy .env ich nie ustawia.
export OPENAI_API_KEY="${OPENAI_API_KEY:-foo}"
export DEEPL_AUTH_KEY="${DEEPL_AUTH_KEY:-bar}"

export ENV_USE_KC_AUTH="${ENV_USE_KC_AUTH:-0}"
export ENV_USE_OAUTH_V1_AUTH="${ENV_USE_OAUTH_V1_AUTH:-0}"
export ENV_USE_OAUTH_V2_AUTH="${ENV_USE_OAUTH_V2_AUTH:-0}"
export ENV_USE_INTROSPECT="${ENV_USE_INTROSPECT:-0}"
export ENV_INTROSPECT_EMAIL_VERIFICATION="${ENV_INTROSPECT_EMAIL_VERIFICATION:-0}"

# --- adres nasłuchu ----------------------------------------------------------
if [[ -z "$BIND_ADDRESS" ]]; then
    PORT="${PORT:-${REST_API_PORT:-$DEFAULT_PORT}}"
    [[ "$PORT" =~ ^[0-9]+$ ]] || fail "Port musi być liczbą, jest: ${PORT}"
    BIND_ADDRESS="${REST_API_HOST:-0.0.0.0}:${PORT}"
fi

if [[ -n "${GUNICORN_BIND:-}" && "$GUNICORN_BIND" != "$BIND_ADDRESS" ]]; then
    warn_bind="GUNICORN_BIND=${GUNICORN_BIND} z .env jest tu ignorowany — nasłuch na ${BIND_ADDRESS}"
    echo -e "\e[33m[WARN]\e[0m ${warn_bind} (zmień: --bind / --port)."
fi

export GUNICORN_BIND="$BIND_ADDRESS"
if [[ -n "$WORKERS" ]]; then
    [[ "$WORKERS" =~ ^[0-9]+$ ]] || fail "--workers musi być liczbą, jest: ${WORKERS}"
    export GUNICORN_WORKERS="$WORKERS"
fi

# --- start -------------------------------------------------------------------
if command -v gunicorn >/dev/null 2>&1; then
    GUNICORN_CMD=(gunicorn)
elif python3 -c "import gunicorn" >/dev/null 2>&1; then
    GUNICORN_CMD=(python3 -m gunicorn)
else
    fail "Brak gunicorn w środowisku. Zainstaluj: pip install gunicorn (albo ./run.sh setup)."
fi

cd "$ROOT_DIR"

info "REST API (gunicorn): http://${BIND_ADDRESS}/api/  (workers=${GUNICORN_WORKERS:-auto}, threads=${GUNICORN_THREADS:-4})"
info "Zatrzymanie: Ctrl+C  albo  ./run.sh down"

exec "${GUNICORN_CMD[@]}" sse_api.config.wsgi:application --config "$GUNICORN_CONF"

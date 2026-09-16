#!/usr/bin/env bash
#
# run-api-prod.sh — uruchomienie produkcyjne (gunicorn + WSGI).
# Wymaga jawnego, bezpiecznego .env; odmawia startu gdy DEBUG=1 albo gdy
# sekrety wciąż mają wartości szablonowe (CHANGE_ME).
#
# Użycie:  ./run-api-prod.sh           (domyślnie 0.0.0.0:8000, 3 workers)
#          ./run-api-prod.sh --check    (walidacja konfiguracji BEZ startu serwera)
#
# Flagi: --check / -n / --dry-run — sprawdź pliki konfiguracyjne, sekrety i
#        konfigurację Django (manage.py check --deploy), po czym wyjdz 0.
#
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd -P)"

if [[ -f "${SCRIPT_DIR}/../.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/../.env"
    set +a
fi

fail() { echo -e "\e[31m[ERROR]\e[0m $*" >&2; exit 1; }

# --- argumenty ---------------------------------------------------------------
CHECK_ONLY=0
if (( $# > 0 )); then
    for arg in "$@"; do
        case "$arg" in
            --check|-n|--dry-run) CHECK_ONLY=1 ;;
            *) fail "Nieznany argument: $arg (dostępne: --check)" ;;
        esac
    done
fi

# --- walidacja konfiguracji produkcyjnej -----------------------------------
if [[ "${ENV_DEBUG:-0}" != "0" ]]; then
    fail "ENV_DEBUG=1 w środowisku produkcyjnym — ustaw 0 w .env."
fi

# Hasła są wymagane tylko wtedy, gdy są używane (zewnętrzna baza danych nie
# wymaga POSTGRES_PASSWORD, tylko ENV_DB_PASSWORD).
for var in ENV_DB_PASSWORD POSTGRES_PASSWORD; do
    value="${!var:-}"
    if [[ -z "$value" ]]; then
        continue
    fi
    if [[ "$value" == *"CHANGE_ME"* ]]; then
        fail "Zmienna ${var} nadal zawiera wartość szablonową (CHANGE_ME)."
    fi
done

for var in ENV_MILVUS_PASSWORD ENV_CELERY_BROKER_URL OPENAI_API_KEY DEEPL_AUTH_KEY; do
    value="${!var:-}"
    if [[ -n "$value" && "$value" == *"CHANGE_ME"* ]]; then
        fail "Zmienna ${var} nadal zawiera wartość szablonową (CHANGE_ME)."
    fi
done

# SECRET_KEY może przyjść z .env (ENV_SECRET_KEY) albo z configs/secret-key.txt.
if [[ "${ENV_SECRET_KEY:-}" == *"CHANGE_ME"* ]]; then
    fail "ENV_SECRET_KEY nadal zawiera wartość szablonową (CHANGE_ME)."
fi
if [[ ! -f "${SCRIPT_DIR}/configs/secret-key.txt" && -z "${ENV_SECRET_KEY:-}" ]]; then
    fail "Brak SECRET_KEY: ustaw ENV_SECRET_KEY w .env albo utwórz configs/secret-key.txt."
fi
if [[ -f "${SCRIPT_DIR}/configs/secret-key.txt" ]]; then
    if grep -q "CHANGE_ME" "${SCRIPT_DIR}/configs/secret-key.txt"; then
        fail "configs/secret-key.txt nadal zawiera wartość szablonową (CHANGE_ME)."
    fi
fi

# --- tryb --check: walidacja bez startu serwera ------------------------------
if [[ "$CHECK_ONLY" == "1" ]]; then
    for cfg in configs/django-config.json configs/milvus_config.json; do
        [[ -f "${SCRIPT_DIR}/${cfg}" ]] || fail "Brak pliku konfiguracji: ${cfg} (uruchom: ./run.sh config)."
    done
    command -v python3 >/dev/null 2>&1 || fail "Brak python3 w PATH."

    export ENV_DEBUG=0
    export DJANGO_SETTINGS_MODULE="${DJANGO_SETTINGS_MODULE:-main.settings}"

    echo "[INFO] manage.py check --deploy"
    if ! ( cd "$SCRIPT_DIR" && python3 manage.py check --deploy ); then
        fail "manage.py check --deploy zgłosił błędy — popraw konfigurację przed wdrożeniem."
    fi

    echo "[OK] Walidacja konfiguracji produkcyjnej OK — serwer NIE został uruchomiony (--check)."
    exit 0
fi

export ENV_DEBUG=0
export DJANGO_SETTINGS_MODULE="${DJANGO_SETTINGS_MODULE:-main.settings}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export DATA_UPLOAD_MAX_MEMORY_SIZE="${DATA_UPLOAD_MAX_MEMORY_SIZE:-2621440000}"
export FILE_UPLOAD_MAX_MEMORY_SIZE="${FILE_UPLOAD_MAX_MEMORY_SIZE:-2621440000}"

cd "$SCRIPT_DIR"

echo "[INFO] collectstatic (znaki promptu są nadpisywane)"
if ! python3 manage.py collectstatic --noinput; then
    echo "[WARN] collectstatic nie powiódł się — sprawdź STATIC_ROOT." >&2
fi

echo "[INFO] start gunicorn: ${GUNICORN_BIND:-0.0.0.0:8000} (workers=${GUNICORN_WORKERS:-3})"
exec gunicorn main.wsgi:application --config "${SCRIPT_DIR}/gunicorn.conf.py"

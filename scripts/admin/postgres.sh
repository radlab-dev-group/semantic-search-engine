#!/usr/bin/env bash
# ------------------------------------------------------------
# postgres.sh
# ------------------------------------------------------------
# Uruchamia najnowszy obraz PostgreSQL w Dockerze z trwałym wolumenem
# (postgres_data -> /var/lib/postgresql/data).  Dzięki temu dane
# przeżywają zatrzymanie i usunięcie kontenera ("./run.sh down"), a ponowny
# start wznawia istniejący kontener zamiast tworzyć pustą bazę.
#
# Dane kasuje wyłącznie "./run.sh clean" (albo ręcznie: docker volume rm postgres_data).
# ------------------------------------------------------------

set -euo pipefail   # Bezpieczne ustawienia Bash

# ------------------- Konfiguracja -------------------
CONTAINER_NAME="pg_sse_backend_engine"
# Nazwa wolumenu Docker (też: docker volume inspect ${POSTGRES_DATA_VOLUME:-postgres_data})
DATA_VOLUME="${POSTGRES_DATA_VOLUME:-postgres_data}"
# Miejsce montowania wolumenu.  Od obrazu postgres:18+ dane leżą w katalogu
# /var/lib/postgresql/<wersja>/docker, a sam obraz deklaruje VOLUME
# /var/lib/postgresql — montowanie czegokolwiek głębiej (stare
# /var/lib/postgresql/data) kończy się komunikatem
# "there appears to be PostgreSQL data in: /var/lib/postgresql/data (unused mount/volume)"
# i pętlą restartów, więc wolumen musi trafić do /var/lib/postgresql.
DATA_MOUNTPOINT="/var/lib/postgresql"
# latest, bo silnik nie jest jeszcze nigdzie wystawiony produkcyjnie.  Skok dużej
# wersji może odmówić startu na istniejącym wolumenie (układ katalogów się zmienia)
# — wtedy: pg_upgrade, albo od zera (./run.sh clean) i re-indeksacja, bo PostgreSQL
# jest źródłem prawdy.  Inna wersja: POSTGRES_IMAGE=postgres:16 ./run.sh infra.
DOCKER_IMAGE="${POSTGRES_IMAGE:-postgres:latest}"
POSTGRES_PASSWORD="SuperSecretPassword123"
POSTGRES_USER="admin"
POSTGRES_DB="sse_backend"
# port dostępny w sieci
HOST_PORT=5471

# ------------------- Funkcje pomocnicze -------------------
log()   { echo -e "\e[32m[INFO]\e[0m $*"; }
warn()  { echo -e "\e[33m[WARN]\e[0m $*"; }
err()   { echo -e "\e[31m[ERROR]\e[0m $*" >&2; exit 1; }

# Docker przez socketa użytkownika, a gdyby brakowało uprawnień — przez sudo.
dc() {
    if docker info >/dev/null 2>&1; then
        docker "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo docker "$@"
    else
        docker "$@"
    fi
}

# ------------------- 1. Sprawdź Docker -------------------
if ! command -v docker >/dev/null 2>&1; then
    err "Docker nie jest zainstalowany lub nie znajduje się w \$PATH."
fi

# ------------------- 2. Trwały wolumen na dane -------------------
# Tworzony przed startem (a nie przy okazji 'docker run'), żeby jego nazwa była
# znana i przewidywalna: './run.sh clean' usuwa dokładnie ten wolumen.
dc volume create "$DATA_VOLUME" >/dev/null \
    || err "Nie udało się utworzyć wolumenu '${DATA_VOLUME}'. Sprawdź, czy daemon Dockera działa."
log "Wolumen danych PostgreSQL: ${DATA_VOLUME} -> ${DATA_MOUNTPOINT}"

# ------------------- 3. Użyj istniejącego kontenera (nie kasuj danych) -------------------
if [[ -n "$(dc ps -q --filter "name=^${CONTAINER_NAME}\$")" ]]; then
    log "Kontener '${CONTAINER_NAME}' już działa — nie tworzę go od nowa."
    exit 0
fi

if [[ -n "$(dc ps -aq --filter "name=^${CONTAINER_NAME}\$")" ]]; then
    # Kontener sprzed wprowadzenia wolumenu trzyma dane w swojej warstwie, więc
    # sam 'docker rm' bez 'pg_dump' oznaczałby ich utratę — wznawiamy go.
    if ! dc inspect --format '{{range .Mounts}}{{.Name}} {{end}}' "${CONTAINER_NAME}" | grep -qw "$DATA_VOLUME"; then
        warn "Stary kontener '${CONTAINER_NAME}' nie ma wolumenu '${DATA_VOLUME}' (dane w warstwie kontenera)."
        warn "Wznawiam go, żeby ich nie stracić. Po 'pg_dump' możesz zacząć od czystej bazy: ./run.sh clean"
    else
        log "Kontener '${CONTAINER_NAME}' istnieje (zatrzymany) — uruchamiam ponownie."
    fi
    dc start "${CONTAINER_NAME}" >/dev/null
else
    # ------------------- 4. Uruchom kontener -------------------
    log "Uruchamianie kontenera '${CONTAINER_NAME}' (host:${HOST_PORT} → 5432 w kontenerze)"
    dc run -d \
        --name "${CONTAINER_NAME}" \
        -e POSTGRES_PASSWORD="${POSTGRES_PASSWORD}" \
        -e POSTGRES_USER="${POSTGRES_USER}" \
        -e POSTGRES_DB="${POSTGRES_DB}" \
        -v "${DATA_VOLUME}:${DATA_MOUNTPOINT}" \
        -p "0.0.0.0:${HOST_PORT}:5432" \
        --restart unless-stopped \
        "${DOCKER_IMAGE}" >/dev/null
fi

log "✅ Kontener uruchomiony!"
log "Połączenie dostępne pod:"
log "  host: <IP_UBUNTU>"
log "  port: ${HOST_PORT}"
log "  user: ${POSTGRES_USER}"
log "  db:   ${POSTGRES_DB}"
log "  hasło: ${POSTGRES_PASSWORD}"
log "  dane: wolumen '${DATA_VOLUME}' (przeżywa './run.sh down'; usuwa go './run.sh clean')"

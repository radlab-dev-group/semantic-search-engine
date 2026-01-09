#!/usr/bin/env bash
# ------------------------------------------------------------
# uruchom_postgres.sh
# ------------------------------------------------------------
# Uruchamia najnowszy obraz PostgreSQL w Dockerze,
# używając trwałego katalogu /var/lib/postgresql_data
# ------------------------------------------------------------

set -euo pipefail   # Bezpieczne ustawienia Bash

# ------------------- Konfiguracja -------------------
CONTAINER_NAME="pg_sse_backend"
HOST_DATA_DIR="/var/lib/postgresql_data"
POSTGRES_PASSWORD="SuperSecretPassword123"
POSTGRES_USER="admin"
POSTGRES_DB="sse_backend"
# port dostępny w sieci
HOST_PORT=5457
DOCKER_IMAGE="postgres:latest"

# ------------------- Funkcje pomocnicze -------------------
log()   { echo -e "\e[32m[INFO]\e[0m $*"; }
warn()  { echo -e "\e[33m[WARN]\e[0m $*"; }
err()   { echo -e "\e[31m[ERROR]\e[0m $*" >&2; exit 1; }

# ------------------- 1. Sprawdź Docker -------------------
if ! command -v docker >/dev/null 2>&1; then
    err "Docker nie jest zainstalowany lub nie znajduje się w \$PATH."
fi

# ------------------- 2. Przygotuj katalog danych -------------------
# 2a. Utwórz całą ścieżkę (wraz z katalogami nadrzędnymi)
if [[ -d "$HOST_DATA_DIR" ]]; then
    log "Katalog danych już istnieje: $HOST_DATA_DIR"
else
    log "Tworzenie całej ścieżki: $HOST_DATA_DIR"
    sudo mkdir -p "$HOST_DATA_DIR"
fi

# 2b. Ustaw właściciela na UID/GID postgres (999:999), wymagane przez obraz postgres
log "Ustawianie właściciela na UID/GID 999:999"
sudo chown 999:999 "$HOST_DATA_DIR"

# 2c. Sprawdź, czy katalog jest zapisywalny
if ! sudo test -w "$HOST_DATA_DIR"; then
    err "Katalog $HOST_DATA_DIR nie jest zapisywalny. \
Sprawdź, czy system plików nie jest zamontowany jako read‑only \
lub wybierz inną lokalizację (np. /var/lib/postgresql_data)."
fi

# 2d. Ustaw rozsądne uprawnienia katalogu (dla postgres wystarczy 700/750/755)
log "Ustawianie chmod 755 na host data dir"
sudo chmod 755 "$HOST_DATA_DIR"

# ------------------- 3. Usuń istniejący kontener (jeśli jest) -------------------
if sudo docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}\$"; then
    log "Kontener '${CONTAINER_NAME}' już istnieje – zatrzymuję i usuwam."
    sudo docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    sudo docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

# ------------------- 4. Uruchom kontener -------------------
log "Uruchamianie kontenera '${CONTAINER_NAME}' (host:${HOST_PORT} → 5432 w kontenerze)"
sudo docker run -d \
    --name "${CONTAINER_NAME}" \
    -e POSTGRES_PASSWORD="${POSTGRES_PASSWORD}" \
    -e POSTGRES_USER="${POSTGRES_USER}" \
    -e POSTGRES_DB="${POSTGRES_DB}" \
    -p "0.0.0.0:${HOST_PORT}:5432" \
    --restart unless-stopped \
    "${DOCKER_IMAGE}"


# -v "${HOST_DATA_DIR}:/var/lib/postgresql/data" \

log "✅ Kontener uruchomiony!"
log "Połączenie dostępne pod:"
log "  host: <IP_UBUNTU>"
log "  port: ${HOST_PORT}"
log "  user: ${POSTGRES_USER}"
log "  db:   ${POSTGRES_DB}"
log "  hasło: ${POSTGRES_PASSWORD}"
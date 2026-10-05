# syntax=docker/dockerfile:1
#
# Semantic Search Engine — production image.
#
#   docker build -t sse-api:0.0.2 .
#
# The two git-installed dependencies are optional (they are not on PyPI), so
# they are switched on with build args, exactly like `initialize.sh dep` does:
#
#   docker build --build-arg INSTALL_RADLAB_DATA=1 \
#                --build-arg INSTALL_LLM_ROUTER=1 -t sse-api:0.0.2 .
#
# Runtime configuration (`configs/*.json`, `secret-key.txt`) is NOT baked in —
# mount `sse_rest_api/configs` at run time. See docker-compose.yml and SETUP.md.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ARG INSTALL_RADLAB_DATA=0
ARG INSTALL_LLM_ROUTER=0
ARG RADLAB_DATA_REF=master
ARG LLM_ROUTER_REF=master

# `configs/` and `static/` are resolved relative to the working directory.
WORKDIR /app/sse_rest_api

# Everything that needs a compiler is installed and purged inside a single
# layer, so the build toolchain never reaches the final image. `curl` stays —
# the container healthcheck uses it.
COPY sse_rest_api/requirements.txt ./requirements.txt
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        curl \
        build-essential \
        python3-dev \
        git \
    && pip install -r requirements.txt \
    && if [ "${INSTALL_RADLAB_DATA}" = "1" ]; then \
         pip install "git+https://github.com/radlab-dev-group/radlab-data.git@${RADLAB_DATA_REF}"; \
       fi \
    && if [ "${INSTALL_LLM_ROUTER}" = "1" ]; then \
         pip install "git+https://github.com/radlab-dev-group/llm-router.git@${LLM_ROUTER_REF}"; \
       fi \
    && apt-get purge -y --auto-remove build-essential python3-dev git \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip

# Application code. Secrets and local config are excluded by .dockerignore.
COPY sse_rest_api/ ./

# Run as an unprivileged user; the directories the API writes to are created
# here so a bind/volume mount inherits the right ownership.
RUN groupadd --gid 10001 sse \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin sse \
    && mkdir -p /app/sse_rest_api/staticfiles /app/sse_rest_api/upload_sse \
    && chown -R sse:sse /app
USER sse

EXPOSE 8000

# SECURE_SSL_REDIRECT is on whenever DEBUG=0, so the probe has to identify the
# request as https the same way nginx does — otherwise it would follow a 301.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS -H "X-Forwarded-Proto: https" http://127.0.0.1:8000/api/healthz || exit 1

CMD ["gunicorn", "main.wsgi:application", "--config", "gunicorn.conf.py"]

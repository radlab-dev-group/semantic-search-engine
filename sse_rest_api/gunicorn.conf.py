"""
gunicorn.conf.py
----------------

Production WSGI server configuration.  Every value can be tuned from the
environment (see ``.env.example``) so the same image works for a small VM and
for a container orchestrated by Kubernetes.

Usage:
    gunicorn main.wsgi:application --config gunicorn.conf.py
"""

import multiprocessing
import os


def _int_env(name: str, default: int) -> int:
    """
    Read an integer from the environment, falling back to the default value
    when the variable is missing or malformed.
    :param name: Environment variable name
    :param default: Value used when the variable is not usable
    :return: Parsed integer
    """
    raw = os.environ.get(name, "")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


# Bind address. Behind nginx keep the loopback default and publish the port
# only through the reverse proxy.
bind = os.environ.get("GUNICORN_BIND", "0.0.0.0:8000")

# Embedding models are CPU/GPU heavy, so the worker count stays deliberately
# low and has to be sized against the RAM/GPU of the host.
default_workers = min(3, max(1, multiprocessing.cpu_count() // 2))
workers = _int_env("GUNICORN_WORKERS", default_workers)
threads = _int_env("GUNICORN_THREADS", 4)
worker_class = os.environ.get("GUNICORN_WORKER_CLASS", "gthread")

# Indexing requests load large models and documents; a plain 30 s timeout
# kills them in the middle of the work.
timeout = _int_env("GUNICORN_TIMEOUT", 300)
graceful_timeout = _int_env("GUNICORN_GRACEFUL_TIMEOUT", 30)
keepalive = _int_env("GUNICORN_KEEPALIVE", 5)

# A worker that leaks memory beyond this limit is recycled (0 disables it).
max_requests = _int_env("GUNICORN_MAX_REQUESTS", 1000)
max_requests_jitter = _int_env("GUNICORN_MAX_REQUESTS_JITTER", 100)

# Model weights are loaded lazily per worker; preloading is opt-in because
# forking an initialised CUDA context breaks the embedding engine.
preload_app = os.environ.get("GUNICORN_PRELOAD", "0") == "1"

# Logs on stdout/stderr, so docker/nginx/systemd own the log rotation.
accesslog = os.environ.get("GUNICORN_ACCESSLOG", "-")
errorlog = os.environ.get("GUNICORN_ERRORLOG", "-")
loglevel = os.environ.get("GUNICORN_LOGLEVEL", "info")
capture_output = True

# `configs/` and `static/` paths are relative to the working directory.
chdir = os.path.dirname(os.path.abspath(__file__))

proc_name = "sse-api"
wsgi_app = "main.wsgi:application"

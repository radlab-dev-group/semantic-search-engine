"""
Project package.

The Celery application is created here so that ``celery -A sse_api.config``
finds it, and so that the tasks are known to Django.  Celery is optional:
an engine with ``ENV_USE_CELERY=0`` does not have it installed at all.
"""

try:
    from .celery import app as celery_app  # noqa: F401
except ImportError:  # Celery is not installed
    celery_app = None

__all__ = ("celery_app",)

"""
The Celery application of the engine.

Only the indexing runs in the queue (``sse_api.data.tasks``); the application is
created lazily and Django settings are read through the ``CELERY`` namespace.
The engine runs perfectly well without it, see ``ENV_USE_CELERY``.
"""

import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "sse_api.config.settings")

app = Celery("main")

app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()

import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "sse_api.config.settings")

app = Celery("main")

app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()

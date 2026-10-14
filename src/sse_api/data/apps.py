from django.apps import AppConfig


class DataConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "sse_api.data"
    label = "data"

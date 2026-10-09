from sse_api.authorization import app_name
from django.apps import AppConfig


class RdlAuthConfig(AppConfig):
    name = app_name
    label = "authorization"

"""
URLconf used only by the ``system`` tests.

It re-exports the real ``sse_api.system.urls`` patterns, so the probe is
resolved exactly the way ``system/urls.py`` registers it.

The full ``sse_api.config.urls`` is deliberately *not* used here: with
Keycloak enabled it imports ``sse_api.authorization.urls``, which builds an
``RdlAuthConfig`` at import time and opens ``configs/auth-config.json``
(see ``authorization/utils/config.py:105``).  Until that import-time file
access is removed, the aggregate URLconf cannot be imported without a
deployment config file, so the aggregation is covered by the container
healthcheck instead (``docker compose up`` + ``curl /api/healthz``).
"""

from sse_api.system.urls import urlpatterns  # noqa: F401  (re-export)

"""
test_healthz.py
---------------

The container healthcheck (``Dockerfile``), nginx (``nginx/sse.conf``) and
``depends_on: service_healthy`` in ``docker-compose.yml`` all poll
``/api/healthz``.  A missing route there makes every container report
unhealthy, so these tests pin the route down, its public access and its
status codes.
"""

from unittest.mock import patch

from django.test import SimpleTestCase, override_settings

# Resolved through the real ``sse_api.system.urls`` patterns (see the module
# docstring of ``sse_api/system/tests/urls.py`` for why not the aggregate one).
SYSTEM_URLCONF = "sse_api.system.tests.urls"

# The production settings default to IsAuthenticated; the test settings do
# not, so the defaults are forced here or "public probe" would pass by
# accident.
AUTHENTICATED_DEFAULTS = {
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.IsAuthenticated"],
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.TokenAuthentication"
    ],
}


@override_settings(ROOT_URLCONF=SYSTEM_URLCONF)
class HealthCheckRouteTests(SimpleTestCase):
    """The probe has to answer where the deployment actually calls it."""

    @patch("sse_api.system.api.dependencies_status", return_value={})
    def test_route_is_registered_under_the_api_root(self, _status):
        response = self.client.get("/api/healthz")
        self.assertNotEqual(
            response.status_code,
            404,
            "/api/healthz is polled by the Docker HEALTHCHECK and nginx; "
            "it must not be a 404.",
        )

    @patch("sse_api.system.api.dependencies_status", return_value={})
    def test_unversioned_alias_is_registered_too(self, _status):
        """Renaming or versioning the API root must not break the probe."""
        self.assertNotEqual(
            self.client.get("/healthz").status_code,
            404,
        )

    @override_settings(REST_FRAMEWORK=AUTHENTICATED_DEFAULTS)
    @patch("sse_api.system.api.dependencies_status", return_value={})
    def test_probe_is_public(self, _status):
        """A probe carries no token, so it must answer without one."""
        response = self.client.get("/api/healthz")
        self.assertNotIn(response.status_code, (401, 403))

    @patch("sse_api.system.api.dependencies_status", return_value={})
    def test_only_get_is_accepted(self, _status):
        self.assertEqual(self.client.post("/api/healthz").status_code, 405)


@override_settings(ROOT_URLCONF=SYSTEM_URLCONF)
class HealthCheckStatusTests(SimpleTestCase):
    """200 only when every dependency answers; anything else is 503."""

    @patch(
        "sse_api.system.api.dependencies_status",
        return_value={"database": True, "milvus": True},
    )
    def test_healthy_returns_200(self, _status):
        response = self.client.get("/api/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertIs(response.json()["healthy"], True)

    @patch(
        "sse_api.system.api.dependencies_status",
        return_value={"database": True, "milvus": False},
    )
    def test_degraded_dependency_returns_503(self, _status):
        response = self.client.get("/api/healthz")
        self.assertEqual(response.status_code, 503)
        self.assertIs(response.json()["healthy"], False)

    @patch(
        "sse_api.system.api.dependencies_status",
        return_value={"database": True, "milvus": True},
    )
    def test_body_does_not_use_the_envelope_status_key(self, _status):
        """``status`` is the envelope key; reusing it breaks sse_lib parsing."""
        self.assertNotIn("status", self.client.get("/api/healthz").json())

    @patch("sse_api.system.api.dependencies_status", side_effect=OSError("boom"))
    def test_failing_check_does_not_leak_internals(self, _status):
        """A broken dependency must not turn into a traceback in the response."""
        response = self.client.get("/api/healthz")
        self.assertIn(response.status_code, (200, 503))
        body = response.content.decode()
        self.assertNotIn("Traceback", body)
        self.assertNotIn("boom", body)

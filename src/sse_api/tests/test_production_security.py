"""
Production hardening tests: DRF throttling and Django security headers.

The throttling tests prove the API answers 429 once a client exceeds the
per-user / per-IP budget (brute force on ``login`` is the case that matters
most).  The header tests assert that the deployment settings carry the
security headers nginx relies on, so a misconfigured environment fails here
instead of in production.
"""

import os

from django.conf import settings
from django.test import override_settings
from rest_framework.test import APIClient

from sse_api.config.settings import (
    REST_FRAMEWORK as PROD_REST_FRAMEWORK,
    _production_security_settings,
)
from sse_api.system.api import OrganisationLogin

AUTH_ENDPOINT = "/api/login"


class ThrottlingTests(APIClient):
    def setUp(self):
        self.client = APIClient()

    def test_anonymous_requests_are_throttled(self):
        with override_settings(
            REST_FRAMEWORK={
                **settings.REST_FRAMEWORK,
                "DEFAULT_THROTTLE_CLASSES": [
                    "sse_api.core.throttles.BurstAnonRateThrottle"
                ],
                "DEFAULT_THROTTLE_RATES": {"burst_anon": "3/minute"},
            }
        ):
            responses = [
                self.client.get(AUTH_ENDPOINT, format="json") for _ in range(4)
            ]
        self.assertTrue(
            any(response.status_code == 429 for response in responses),
            "expected a 429 within four rapid anonymous requests",
        )

    def test_healthz_is_not_throttled(self):
        # The readiness probe must stay reachable under load; it opts out of
        # the default throttle classes.
        with override_settings(
            REST_FRAMEWORK={
                **settings.REST_FRAMEWORK,
                "DEFAULT_THROTTLE_CLASSES": [
                    "sse_api.core.throttles.BurstAnonRateThrottle"
                ],
                "DEFAULT_THROTTLE_RATES": {"burst_anon": "2/minute"},
            }
        ):
            responses = [
                self.client.get("/healthz") for _ in range(5)
            ]
        self.assertTrue(
            all(response.status_code in (200, 503) for response in responses),
            "healthz must never answer 429 to a probe",
        )

    def test_login_endpoint_uses_the_strict_auth_throttle(self):
        login_view = OrganisationLogin()
        throttle_classes = [cls() for cls in login_view.throttle_classes]
        self.assertTrue(
            any(
                "AuthThrottle" in type(throttle).__name__
                for throttle in throttle_classes
            ),
            "the login view must carry the strict authentication throttle",
        )

    def test_production_settings_define_throttling(self):
        self.assertTrue(
            PROD_REST_FRAMEWORK.get("DEFAULT_THROTTLE_CLASSES"),
            "production settings must enable a default throttle class",
        )
        rates = PROD_REST_FRAMEWORK.get("DEFAULT_THROTTLE_RATES", {})
        self.assertIn("anon", rates)
        self.assertIn("user", rates)
        self.assertIn("auth", rates)


class SecurityHeaderTests:
    def test_production_headers_are_defined(self):
        security = _production_security_settings()
        self.assertEqual(
            security["SECURE_PROXY_SSL_HEADER"],
            ("HTTP_X_FORWARDED_PROTO", "https"),
        )
        self.assertTrue(security["SECURE_SSL_REDIRECT"])
        self.assertTrue(security["SECURE_CONTENT_TYPE_NOSNIFF"])
        self.assertEqual(security["X_FRAME_OPTIONS"], "DENY")
        self.assertEqual(security["SECURE_HSTS_SECONDS"], 31536000)
        self.assertTrue(security["SESSION_COOKIE_SECURE"])
        self.assertTrue(security["CSRF_COOKIE_SECURE"])

    def test_debug_mode_disables_ssl_redirect(self):
        # With DEBUG on (local dev) the SSL redirect would break plain-http
        # access to the dev server.
        with override_settings(DEBUG=True):
            self.assertFalse(_production_security_settings()["SECURE_SSL_REDIRECT"])

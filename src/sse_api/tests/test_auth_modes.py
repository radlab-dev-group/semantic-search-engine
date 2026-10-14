"""
test_auth_modes.py
------------------

The deployment picks one of four authentication modes
(``rest_framework.authtoken``, Keycloak, OAuth v1, OAuth v2) through
``SYSTEM_HANDLER``, and ``sse_api.config.urls`` assembles a different route
table per mode.  Nothing covered that switch: the branches were only reachable
by hand-editing a config file.

These are characterization tests — they pin the route table each mode produces
today, including the ``api/login`` collision, so a future change to the
routing is a deliberate edit rather than a surprise.
"""

import importlib

from django.test import SimpleTestCase, override_settings
from django.urls import clear_url_caches

EXTERNAL_LOGIN_ROUTES = {
    "api/login_url",
    "api/logout",
    "api/refresh_token",
}


def handler_stub(
    keycloak: bool = False,
    oauth_v1: bool = False,
    oauth_v2: bool = False,
    admin_window: bool = False,
):
    """Stand-in for ``SYSTEM_HANDLER``; ``config.urls`` calls these as methods."""
    return type(
        "HandlerStub",
        (),
        {
            "use_keycloak_authentication": staticmethod(lambda: keycloak),
            "use_oauth_v1_authentication": staticmethod(lambda: oauth_v1),
            "use_oauth_v2_authentication": staticmethod(lambda: oauth_v2),
            "use_introspect": False,
            "show_admin_window": staticmethod(lambda: admin_window),
        },
    )


def routes_for(**flags):
    """Route table ``config.urls`` builds for the given authentication flags."""
    with override_settings(SYSTEM_HANDLER=handler_stub(**flags)):
        clear_url_caches()
        module = importlib.reload(importlib.import_module("sse_api.config.urls"))
        paths = [str(pattern.pattern) for pattern in module.urlpatterns]
        clear_url_caches()
    return paths


class AuthenticationModeTests(SimpleTestCase):
    def tearDown(self):
        # config.urls was re-executed with stub handlers; put the module back
        # so no later test inherits a route table built from a stub.
        with override_settings(SYSTEM_HANDLER=self.real_handler):
            clear_url_caches()
            importlib.reload(importlib.import_module("sse_api.config.urls"))
            clear_url_caches()

    def setUp(self):
        from django.conf import settings

        self.real_handler = settings.SYSTEM_HANDLER

    def test_token_mode_exposes_the_drf_token_endpoint(self):
        routes = routes_for()
        self.assertIn("api-token-auth/", routes)

    def test_token_mode_has_no_external_auth_routes(self):
        routes = set(routes_for())
        self.assertEqual(routes & EXTERNAL_LOGIN_ROUTES, set())

    def test_each_external_mode_exposes_its_own_auth_routes(self):
        for flags in (
            {"keycloak": True},
            {"oauth_v1": True},
            {"oauth_v2": True},
        ):
            with self.subTest(**flags):
                routes = set(routes_for(**flags))
                self.assertTrue(EXTERNAL_LOGIN_ROUTES <= routes)

    def test_external_mode_drops_the_drf_token_endpoint(self):
        self.assertNotIn("api-token-auth/", routes_for(keycloak=True))

    def test_two_providers_at_once_is_rejected(self):
        for flags in (
            {"keycloak": True, "oauth_v1": True},
            {"keycloak": True, "oauth_v2": True},
            {"oauth_v1": True, "oauth_v2": True},
        ):
            with self.subTest(**flags):
                with self.assertRaises(Exception) as caught:
                    routes_for(**flags)
                self.assertIn("Choose between", str(caught.exception))

    def test_admin_window_flag_gates_the_admin_urls(self):
        self.assertIn("admin/", routes_for(admin_window=True))
        self.assertNotIn("admin/", routes_for(admin_window=False))

    def test_the_probe_is_registered_in_every_mode(self):
        """Whatever the auth mode, the container probe has to answer."""
        for flags in (
            {},
            {"keycloak": True},
            {"oauth_v1": True},
            {"oauth_v2": True},
        ):
            with self.subTest(**flags):
                routes = routes_for(**flags)
                self.assertIn("api/healthz", routes)
                self.assertIn("healthz", routes)

    def test_login_path_is_registered_twice_in_external_modes(self):
        """KNOWN DEFECT (plan item 5), pinned on purpose.

        ``system/urls.py`` registers ``api/login`` for username/password and
        ``authorization/urls.py`` registers the same path for the OAuth code
        exchange.  ``config/urls.py`` appends the authorization routes *before*
        the system ones, and Django resolves the first match, so
        ``OrganisationLogin`` is unreachable in every external-auth mode --
        while ``sse_lib.login()`` still posts username/password there.

        Update this test when the collision is fixed, not before.
        """
        for flags in ({"keycloak": True}, {"oauth_v1": True}, {"oauth_v2": True}):
            with self.subTest(**flags):
                self.assertEqual(routes_for(**flags).count("api/login"), 2)

    def test_token_mode_registers_login_once(self):
        self.assertEqual(routes_for().count("api/login"), 1)

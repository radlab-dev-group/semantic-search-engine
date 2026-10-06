import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import jwt
import requests
from django.conf import settings
from django.test import SimpleTestCase, RequestFactory, TestCase
from django.contrib.auth.models import User

from authorization.core.handlers import (
    RdlAuthGrantAccTokenHandler,
    RdlAuthStateHandler,
)
from authorization.core.authentication import GATokenAuthentication
from authorization.utils.config import RdlAuthConfig
from authorization.models import Token, SessionState


class IntrospectionPersistenceTests(TestCase):
    def setUp(self):
        self.mode = patch.object(settings.SYSTEM_HANDLER, "use_introspect", True)
        self.mode.start()
        self.addCleanup(self.mode.stop)
        self.handler = object.__new__(RdlAuthGrantAccTokenHandler)
        self.handler.logger = Mock()
        self.handler.rdl_auth_state_handler = object.__new__(RdlAuthStateHandler)
        self.handler.rdl_auth_state_handler.logger = Mock()
        self.handler.rdl_auth_state_handler.rdl_auth_config = SimpleNamespace(
            auth_host="https://idp",
            realm="realm",
            client_id="client",
            client_secret="secret",
            request_timeout=(1, 2),
            public_sign_key="",
            algorithms=[],
            audience="",
        )

    @patch("authorization.core.handlers.requests.post")
    def test_active_opaque_token_creates_user_and_revalidates_without_cache(
        self, post
    ):
        claims = {
            "active": True,
            "email": "user@example.test",
            "email_verified": True,
            "given_name": "Test",
            "family_name": "User",
        }
        post.return_value = Mock(ok=True, json=Mock(return_value=claims))
        user = self.handler.introspect_user_token("opaque-token")
        self.assertIsInstance(user, User)
        token = Token.objects.get(auth_user=user)
        self.assertEqual(token.decoded_token, claims)
        self.assertEqual(SessionState.objects.count(), 1)
        fresh = {**claims, "scope": "changed"}
        post.return_value.json.return_value = fresh
        self.assertEqual(self.handler.verify_and_decode_token(token, None), fresh)
        post.return_value.json.return_value = {"active": False}
        self.assertIsNone(self.handler.verify_and_decode_token(token, None))

    @patch("authorization.core.handlers.requests.post")
    def test_rejected_provider_claims_do_not_create_records(self, post):
        for claims in [
            {},
            {"active": False},
            {"active": True, "email": "user@example.test", "email_verified": False},
            {"active": True, "email": [], "email_verified": True},
        ]:
            post.return_value = Mock(ok=True, json=Mock(return_value=claims))
            self.assertIsNone(self.handler.introspect_user_token("opaque-token"))
            self.assertEqual(User.objects.count(), 0)
            self.assertEqual(Token.objects.count(), 0)
            self.assertEqual(SessionState.objects.count(), 0)


class AuthenticationSecurityTests(SimpleTestCase):
    def setUp(self):
        self.handler = object.__new__(RdlAuthGrantAccTokenHandler)
        self.handler.logger = Mock()
        self.config = SimpleNamespace(
            public_sign_key="test-secret-long-enough-for-hs256",
            algorithms=["HS256"],
            audience="",
            request_timeout=(1.5, 4),
            accepted_user_roles=["admin"],
            user_role_main_scope="resource_access",
            user_role_scope_variable="account",
            client_id="client",
            client_secret="secret",
        )
        self.handler.rdl_auth_state_handler = Mock(rdl_auth_config=self.config)
        self.mode = patch.object(settings.SYSTEM_HANDLER, "use_introspect", False)
        self.mode.start()
        self.addCleanup(self.mode.stop)

    def token(self, **claims):
        return jwt.encode(
            {"exp": time.time() + 60, **claims},
            self.config.public_sign_key,
            algorithm="HS256",
        )

    def test_local_verification_fails_closed(self):
        valid = self.token(email="user@example.test")
        self.assertIsNotNone(self.handler.verify_and_decode_token(None, valid))
        for key, algorithms in [
            ("", ["HS256"]),
            (None, ["HS256"]),
            (self.config.public_sign_key, []),
            (self.config.public_sign_key, ["none"]),
            (self.config.public_sign_key, "HS256"),
        ]:
            with self.subTest(key=key, algorithms=algorithms):
                self.config.public_sign_key, self.config.algorithms = key, algorithms
                self.assertIsNone(self.handler.verify_and_decode_token(None, valid))

    def test_bad_signature_algorithm_and_expiry(self):
        for token in [
            jwt.encode(
                {"exp": time.time() + 60},
                "different-secret-long-enough",
                algorithm="HS256",
            ),
            jwt.encode({}, self.config.public_sign_key, algorithm="HS512"),
            self.token(exp=time.time() - 60),
            "invalid",
        ]:
            self.assertIsNone(self.handler.verify_and_decode_token(None, token))

    def test_fresh_claims_replace_cache(self):
        token = SimpleNamespace(
            acc_token=self.token(email="fresh"), decoded_token={"email": "cached"}
        )
        self.assertEqual(
            self.handler.verify_and_decode_token(token, None)["email"], "fresh"
        )

    @patch("authorization.core.handlers.requests.post")
    def test_introspection_requires_literal_active_true(self, post):
        settings.SYSTEM_HANDLER.use_introspect = True
        prepare = "_RdlAuthGrantAccTokenHandler__prepare_token_introspect_request"
        with patch.object(
            self.handler, prepare, return_value=("https://idp/introspect", {})
        ):
            for body in [
                {},
                {"active": False},
                {"active": "true"},
                {"active": 1},
                [],
                {"active": True, "error": "secret"},
            ]:
                post.return_value = Mock(ok=True, json=Mock(return_value=body))
                self.assertIsNone(
                    self.handler.verify_and_decode_token(None, "opaque")
                )
                self.assertIsNone(self.handler.introspect_user_token("opaque"))

    @patch("authorization.core.handlers.requests.post")
    def test_provider_errors_are_controlled_and_redacted(self, post):
        settings.SYSTEM_HANDLER.use_introspect = True
        prepare = "_RdlAuthGrantAccTokenHandler__prepare_token_introspect_request"
        with patch.object(
            self.handler, prepare, return_value=("https://idp/introspect", {})
        ):
            for response in [
                requests.Timeout("secret"),
                Mock(ok=False, text="secret"),
                Mock(ok=True, json=Mock(side_effect=ValueError("secret"))),
                Mock(ok=True, json=Mock(return_value=[])),
            ]:
                post.side_effect = (
                    response if isinstance(response, Exception) else None
                )
                post.return_value = response
                self.assertIsNone(self.handler.introspect_user_token("secret"))
                self.assertEqual(post.call_args.kwargs["timeout"], (1.5, 4))
        self.assertNotIn("secret", str(self.handler.logger.mock_calls))

    def test_unverified_email_returns_none_without_creating_user(self):
        with (
            patch.object(
                self.handler,
                "_RdlAuthGrantAccTokenHandler__call_introspection_ep",
                return_value={
                    "active": True,
                    "email": "user",
                    "email_verified": False,
                },
            ),
            patch.object(
                self.handler,
                "_RdlAuthGrantAccTokenHandler__introspection_get_or_add_user",
            ) as create,
        ):
            settings.SYSTEM_HANDLER.use_introspect = True
            self.assertIsNone(self.handler.introspect_user_token("token"))
            create.assert_not_called()

    def test_introspection_uses_fresh_provider_claims_for_opaque_token(self):
        settings.SYSTEM_HANDLER.use_introspect = True
        claims = {"active": True, "email": "fresh", "email_verified": True}
        with patch.object(
            self.handler,
            "_RdlAuthGrantAccTokenHandler__call_introspection_ep",
            return_value=claims,
        ):
            self.assertEqual(
                self.handler.verify_and_decode_token(
                    SimpleNamespace(
                        acc_token="opaque", decoded_token={"email": "cached"}
                    ),
                    None,
                ),
                claims,
            )

    def test_authentication_rejects_revoked_roles(self):
        auth = object.__new__(GATokenAuthentication)
        auth._logger = Mock()
        auth.token_handler = self.handler
        token = SimpleNamespace(
            acc_token=self.token(resource_access={"account": {"roles": []}}),
            decoded_token={"resource_access": {"account": {"roles": ["admin"]}}},
            auth_user=SimpleNamespace(username="user"),
        )
        with patch.object(
            auth,
            "_GATokenAuthentication__find_user_token_for_token_str",
            return_value=token,
        ):
            self.assertEqual(
                auth.authenticate(
                    RequestFactory().get("/", HTTP_AUTHORIZATION="Bearer token")
                ),
                (None, ""),
            )

    def test_introspection_returns_user_and_stores_provider_claims(self):
        settings.SYSTEM_HANDLER.use_introspect = True
        claims = {
            "active": True,
            "email": "user",
            "email_verified": True,
            "session_state": "state",
        }
        user = SimpleNamespace(username="user")
        with (
            patch.object(
                self.handler,
                "_RdlAuthGrantAccTokenHandler__call_introspection_ep",
                return_value=claims,
            ),
            patch.object(
                self.handler,
                "_RdlAuthGrantAccTokenHandler__introspection_get_or_add_user",
                return_value=user,
            ),
            patch.object(
                self.handler,
                "_RdlAuthGrantAccTokenHandler__add_token_for_user",
                return_value=Mock(),
            ) as add,
        ):
            self.assertIs(self.handler.introspect_user_token("opaque"), user)
            self.assertEqual(add.call_args.kwargs["decoded_token"], claims)

    @patch("authorization.core.handlers.requests.post")
    def test_token_and_refresh_calls_have_timeout_and_control_errors(self, post):
        methods = [
            "_RdlAuthGrantAccTokenHandler__prepare_token_request_grant_code",
            "_RdlAuthGrantAccTokenHandler__request_token_with_refresh_token",
        ]
        for method, arguments in zip(
            methods,
            [
                {"grant_code": "code", "refresh_token": None},
                {"grant_code": None, "refresh_token": "refresh"},
            ],
        ):
            with patch.object(
                self.handler, method, return_value=("https://idp/token", {})
            ):
                for response in [
                    requests.ConnectionError("secret"),
                    Mock(ok=False, text="secret"),
                    Mock(ok=True, json=Mock(side_effect=ValueError("secret"))),
                    Mock(ok=True, json=Mock(return_value=[])),
                    Mock(ok=True, json=Mock(return_value={"error": "secret"})),
                ]:
                    post.side_effect = (
                        response if isinstance(response, Exception) else None
                    )
                    post.return_value = response
                    self.assertEqual(
                        self.handler.get_token_with_options(**arguments),
                        (None, None, []),
                    )
                    self.assertEqual(post.call_args.kwargs["timeout"], (1.5, 4))
                post.side_effect = None
                post.return_value = Mock(
                    ok=True,
                    json=Mock(
                        return_value={
                            "access_token": "access",
                            "refresh_token": "refresh",
                        }
                    ),
                )
                self.assertEqual(
                    self.handler.get_token_with_options(**arguments),
                    ("access", {"refresh_token": "refresh"}, []),
                )
        self.assertNotIn("secret", str(self.handler.logger.mock_calls))

    @patch("authorization.core.handlers.requests.post")
    def test_logout_disables_local_token_on_provider_failure(self, post):
        token = Mock(is_active=True, refresh_token="secret")
        with patch(
            "authorization.core.handlers.Token.objects.filter", return_value=[token]
        ):
            for failure in [
                requests.Timeout("secret"),
                Mock(ok=False, text="secret"),
            ]:
                post.side_effect = (
                    failure if isinstance(failure, Exception) else None
                )
                post.return_value = failure
                self.handler.disable_all_user_tokens(Mock())
                self.assertFalse(token.is_active)
                self.assertEqual(post.call_args.kwargs["timeout"], (1.5, 4))
        self.assertNotIn("secret", str(self.handler.logger.mock_calls))

    def test_missing_header_introspection_returns_none(self):
        auth = object.__new__(GATokenAuthentication)
        self.assertIsNone(auth.introspect_token(RequestFactory().get("/")))

    def test_middleware_revalidates_stored_token(self):
        with patch("authorization.core.authentication.GATokenAuthentication"):
            from authorization.core.middleware import AuthAuthenticationMiddleware
        middleware = object.__new__(AuthAuthenticationMiddleware)
        middleware.ga_token = Mock()
        middleware.logger = Mock()
        middleware.get_response = Mock()
        user = SimpleNamespace(username="user")
        middleware.ga_token.get_user_token_for_request.return_value = (
            SimpleNamespace(auth_user=user)
        )
        request = RequestFactory().get("/")
        middleware.ga_token.token_handler.verify_and_decode_token.return_value = None
        middleware.process_request(request)
        self.assertFalse(request.user.is_authenticated)
        middleware.ga_token.token_handler.verify_and_decode_token.return_value = {
            "email": "user"
        }
        middleware.process_request(request)
        self.assertIs(request.user, user)

    def test_timeout_config_defaults_overrides_and_invalid_values(self):
        import json
        from unittest.mock import mock_open

        config = {
            "realm": "realm",
            "client_secret": "secret",
            "client_id": "client",
            "grant_type": "authorization_code",
            "redirect_uri": "https://app",
            "auth_host": "https://idp",
        }
        for extra, expected in [
            ({}, (3.05, 10.0)),
            ({"connect_timeout": 1, "read_timeout": 2}, (1, 2)),
        ]:
            with patch(
                "builtins.open",
                mock_open(
                    read_data=json.dumps({"authorization": {**config, **extra}})
                ),
            ):
                self.assertEqual(RdlAuthConfig().request_timeout, expected)
        for value in [None, 0, -1, True, "10", float("inf"), float("nan")]:
            with patch(
                "builtins.open",
                mock_open(
                    read_data=json.dumps(
                        {"authorization": {**config, "read_timeout": value}}
                    )
                ),
            ):
                with self.assertRaises(ValueError):
                    RdlAuthConfig()

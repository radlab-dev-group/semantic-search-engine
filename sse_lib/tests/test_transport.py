"""
test_transport.py
-----------------

Tests of the HTTP layer: URL building, envelope handling, error mapping and the
"never replay a POST" retry policy.
"""

from __future__ import annotations

import unittest

import requests

from sse_lib import (
    SSEAPIError,
    SSEAuthenticationError,
    SSEConfigError,
    SSEHTTPError,
    SSETransportError,
    Transport,
)
from sse_lib.transport import build_base_url

from support import HOST, FakeSession, api_error, ok, response


class BuildBaseUrlTest(unittest.TestCase):
    def test_localhost_without_scheme_is_http(self):
        self.assertEqual(
            build_base_url("localhost:8271"), "http://localhost:8271/api"
        )

    def test_remote_host_without_scheme_is_https(self):
        self.assertEqual(
            build_base_url("search.example.com"), "https://search.example.com/api"
        )

    def test_trailing_slashes_are_removed(self):
        self.assertEqual(
            build_base_url("http://localhost:8271///", "/api/v1/"),
            "http://localhost:8271/api/v1",
        )

    def test_sub_path_of_the_host_is_kept(self):
        self.assertEqual(
            build_base_url("https://example.com/sse", "api/v2"),
            "https://example.com/sse/api/v2",
        )

    def test_empty_prefix_is_allowed(self):
        self.assertEqual(build_base_url("http://host:1", ""), "http://host:1")

    def test_missing_host_is_rejected(self):
        with self.assertRaises(SSEConfigError):
            build_base_url("   ")


class TransportTest(unittest.TestCase):
    def make(self, *responses, **kwargs):
        kwargs.setdefault("max_retries", 0)
        session = FakeSession(responses=list(responses))
        transport = Transport(HOST, session=session, retry_backoff=0.0, **kwargs)
        return transport, session

    def test_url_for_joins_prefix_and_endpoint(self):
        transport, _ = self.make()
        self.assertEqual(
            transport.url_for("search_with_options"),
            f"{HOST}/api/search_with_options",
        )

    def test_successful_envelope_returns_the_body(self):
        transport, _ = self.make(ok({"collections": [{"id": 1}]}))
        self.assertEqual(transport.get("collections"), {"collections": [{"id": 1}]})

    def test_unwrap_false_returns_the_whole_payload(self):
        transport, _ = self.make(ok({"token": "abc"}))
        self.assertEqual(
            transport.post("login", json_body={}, unwrap=False),
            {"status": True, "body": {"token": "abc"}},
        )

    def test_payload_without_envelope_is_returned_as_is(self):
        transport, _ = self.make(response({"token": "abc"}))
        self.assertEqual(transport.post("login", json_body={}), {"token": "abc"})

    def test_business_error_becomes_sseapi_error(self):
        transport, _ = self.make(
            api_error(
                [
                    {
                        "message": "Nie podano wymaganych parametrów!",
                        "error_code": "e__data_501",
                        "not_given_params": ["collection_name"],
                    }
                ]
            )
        )
        with self.assertRaises(SSEAPIError) as caught:
            transport.get("collections")
        error = caught.exception
        self.assertEqual(error.error_names, ["e__data_501"])
        self.assertEqual(error.not_given_params, ["collection_name"])
        self.assertEqual(error.endpoint, "collections")
        self.assertIn("collection_name", str(error))

    def test_unauthorized_becomes_authentication_error(self):
        transport, _ = self.make(
            response({"detail": "Invalid token."}, status_code=403)
        )
        with self.assertRaises(SSEAuthenticationError):
            transport.get("collections")

    def test_server_error_message_is_taken_from_the_body(self):
        transport, _ = self.make(
            api_error([{"message": "Collection not found!"}], status_code=500)
        )
        with self.assertRaises(SSEHTTPError) as caught:
            transport.post("search_with_options", json_body={})
        self.assertEqual(caught.exception.status_code, 500)
        self.assertIn("Collection not found!", str(caught.exception))

    def test_non_json_error_page_is_still_reported(self):
        transport, _ = self.make(
            response(status_code=502, text="<html>Bad Gateway</html>")
        )
        with self.assertRaises(SSEHTTPError) as caught:
            transport.get("collections")
        self.assertIn("Bad Gateway", caught.exception.response_text)

    def test_empty_body_is_returned_as_none(self):
        transport, _ = self.make(response(status_code=204))
        self.assertIsNone(transport.post("logout"))

    def test_get_is_retried_on_a_transient_status(self):
        transport, session = self.make(
            response("", status_code=503), ok({"collections": []}), max_retries=2
        )
        self.assertEqual(transport.get("collections"), {"collections": []})
        self.assertEqual(len(session.calls), 2)

    def test_post_is_never_retried(self):
        transport, session = self.make(
            response("", status_code=503), ok({}), max_retries=2
        )
        with self.assertRaises(SSEHTTPError):
            transport.post("upload_and_index_files", data={})
        self.assertEqual(len(session.calls), 1)

    def test_connection_error_is_wrapped(self):
        transport, session = self.make(
            requests.ConnectionError("refused"),
            requests.ConnectionError("refused"),
            max_retries=1,
        )
        with self.assertRaises(SSETransportError):
            transport.get("collections")
        self.assertEqual(len(session.calls), 2)

    def test_authorization_header_is_added(self):
        transport, session = self.make(ok([]))
        transport.set_token("secret", "Token")
        transport.get("collections")
        self.assertEqual(
            session.calls[0].headers.get("Authorization"), "Token secret"
        )

    def test_explicit_authorization_header_wins(self):
        transport, session = self.make(ok([]))
        transport.set_token("secret")
        transport.get("collections", headers={"Authorization": "Bearer other"})
        self.assertEqual(session.calls[0].headers["Authorization"], "Bearer other")

    def test_default_headers_are_applied_to_the_session(self):
        session = FakeSession()
        transport = Transport(HOST, session=session, headers={"X-Tenant": "acme"})
        self.assertIs(transport.session, session)
        self.assertEqual(session.headers.get("X-Tenant"), "acme")
        self.assertIn("sse-lib", session.headers.get("User-Agent", ""))

    def test_none_params_are_pruned(self):
        transport, session = self.make(ok([]))
        transport.get("documents", params={"collection_name": "docs", "lang": None})
        self.assertEqual(session.calls[0].params, {"collection_name": "docs"})

    def test_invalid_timeout_is_rejected(self):
        with self.assertRaises(SSEConfigError):
            Transport(HOST, session=FakeSession(), timeout=0)

    def test_close_only_closes_own_sessions(self):
        session = FakeSession()
        Transport(HOST, session=session).close()
        self.assertFalse(session.closed)

    def test_absolute_url_is_used_as_is(self):
        transport, _ = self.make(ok({}))
        self.assertEqual(
            transport.url_for("https://other.host/api/login"),
            "https://other.host/api/login",
        )


if __name__ == "__main__":
    unittest.main()

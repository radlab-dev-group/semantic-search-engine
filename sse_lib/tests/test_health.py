"""
test_health.py
--------------

``health()`` / ``is_healthy()`` talk to the public probe: no token, no lazy
login, and a payload that is *not* the usual ``{"status": ..., "body": ...}``
envelope.  The last point matters, because an envelope-shaped ``status`` key
would be parsed as a failed API call.

Failing cases answer through a ``handler`` rather than a canned response,
because a 503 (and a connection error) on a GET is retried, so the fake
answer has to survive every attempt.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import requests
from support import HOST, build_client, response

import sse_lib


class HealthEndpointTests(unittest.TestCase):
    def test_health_reads_the_probe(self):
        client = build_client(
            responses=[response({"healthy": True, "checks": {"database": True}})]
        )
        self.assertEqual(
            client.health(), {"healthy": True, "checks": {"database": True}}
        )

    def test_url_has_the_api_prefix_and_no_trailing_slash(self):
        client = build_client(responses=[response({"healthy": True})])
        client.health()
        call = client.fake_session.calls[0]
        self.assertEqual(call.method, "GET")
        self.assertEqual(call.url, f"{HOST}/api/healthz")

    def test_probe_is_requested_without_a_token(self):
        client = build_client(responses=[response({"healthy": True})], token=None)
        client.health()
        self.assertNotIn("Authorization", client.fake_session.calls[0].headers)

    def test_probe_never_triggers_a_login(self):
        """A probe has no credentials, so it must not log in first."""
        with patch.dict(os.environ, {}, clear=True):
            client = build_client(
                responses=[response({"healthy": True})],
                token=None,
                username="user",
                password="secret",
            )
        client.health()
        urls = [call.url for call in client.fake_session.calls]
        self.assertEqual(urls, [f"{HOST}/api/healthz"])

    def test_unhealthy_probe_raises_rather_than_returning_a_body(self):
        # A handler instead of a canned response: 503 on a GET is retried, so
        # the fake answer has to survive every attempt.
        client = build_client(
            handler=lambda *_: response({"healthy": False}, status_code=503)
        )
        with self.assertRaises(sse_lib.SSETransportError):
            client.health()

    def test_unreachable_server_raises(self):
        def refuse(*_):
            raise requests.ConnectionError("connection refused")

        client = build_client(handler=refuse)
        with self.assertRaises(sse_lib.SSETransportError):
            client.health()


class IsHealthyTests(unittest.TestCase):
    def test_true_when_the_probe_is_green(self):
        client = build_client(responses=[response({"healthy": True})])
        self.assertTrue(client.is_healthy())

    def test_false_when_a_dependency_is_down(self):
        client = build_client(
            handler=lambda *_: response({"healthy": False}, status_code=503)
        )
        self.assertFalse(client.is_healthy())

    def test_false_when_the_server_is_unreachable(self):
        def refuse(*_):
            raise requests.ConnectionError("no route")

        client = build_client(handler=refuse)
        self.assertFalse(client.is_healthy())

    def test_false_on_an_unparsable_answer(self):
        client = build_client(responses=[response(None, text="not json")])
        self.assertFalse(client.is_healthy())


if __name__ == "__main__":
    unittest.main()

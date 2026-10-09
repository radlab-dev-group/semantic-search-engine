"""
support.py
----------

Test doubles: a ``requests.Session`` replacement that records the outgoing
calls and answers with canned responses, so no socket is ever opened.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import requests

from sse_lib import SSEClient, Transport

HOST = "http://sse.test:8271"


class Call:
    """One recorded request."""

    def __init__(self, method: str, url: str, kwargs: Dict[str, Any]) -> None:
        self.method = method
        self.url = url
        self.params = kwargs.get("params")
        self.json = kwargs.get("json")
        self.data = kwargs.get("data")
        self.files = kwargs.get("files")
        self.headers = kwargs.get("headers") or {}

    def __repr__(self) -> str:
        return f"Call(method={self.method!r}, url={self.url!r})"


class FakeSession:
    """Answers with the responses given to the constructor or ``handler``."""

    def __init__(self, responses=None, handler=None) -> None:
        self.responses = list(responses or [])
        self.handler = handler
        self.headers: Dict[str, str] = {}
        self.verify: Any = True
        self.closed = False
        self.calls: List[Call] = []

    def request(self, method: str, url: str, **kwargs):
        self.calls.append(Call(method, url, kwargs))
        if self.handler is not None:
            return self.handler(method, url, kwargs)
        if not self.responses:
            raise AssertionError(f"Unexpected request: {method} {url}")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self) -> None:
        self.closed = True


def response(payload=None, status_code=200, text: Optional[str] = None, url=HOST):
    """Build a real :class:`requests.Response` for ``FakeSession``."""
    built = requests.Response()
    built.status_code = status_code
    built.url = url
    if payload is not None:
        built._content = json.dumps(payload).encode("utf-8")
        built.headers["Content-Type"] = "application/json"
    else:
        built._content = (text or "").encode("utf-8")
    return built


def ok(body=None, **kwargs):
    """Successful envelope: ``{"status": true, "body": ...}``."""
    return response({"status": True, "body": body}, **kwargs)


def api_error(errors, status_code=200, **kwargs):
    """Business error envelope: ``{"status": false, "errors": [...]}``."""
    return response(
        {"status": False, "errors": errors}, status_code=status_code, **kwargs
    )


# Fields the backend insists on: ``data`` runs indexing_options through
# identifier()/number() and ``engine`` runs query_options through
# generation_options(), all of which reject null.  Tests that assert on request
# *shape* need valid values so validation is not what they trip over.
INDEXING = {"max_tokens_in_chunk": 128, "number_of_overlap_tokens": 0}
GENERATIVE = {"generative_model": "m", "percentage_rank_mass": 40}


def build_client(
    responses=None,
    handler=None,
    token: Optional[str] = "test-token",
    **client_kwargs,
) -> SSEClient:
    """``SSEClient`` wired to a :class:`FakeSession` (no sockets involved)."""
    session = FakeSession(responses=responses, handler=handler)
    transport = Transport(HOST, session=session, retry_backoff=0.0)
    client = SSEClient(HOST, transport=transport, token=token, **client_kwargs)
    client.fake_session = session
    return client

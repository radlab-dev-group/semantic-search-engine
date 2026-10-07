"""
transport.py
------------

Thin HTTP layer used by :class:`sse_lib.SSEClient`.

Responsibilities:

* build URLs (``<scheme>://<host>/<prefix>/<endpoint>``, no trailing slash -
  that is how Django registers the routes),
* attach the ``Authorization`` header and default headers,
* translate the two error channels of the backend into exceptions: HTTP status
  codes and the ``{"status": false, "errors": [...]}`` envelope,
* retry idempotent requests on connection errors and transient statuses.  A
  ``POST`` is never replayed: indexing a document twice is a side effect.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Optional

import requests

from sse_lib.endpoints import DEFAULT_API_PREFIX
from sse_lib.exceptions import (
    SSEAPIError,
    SSEAuthenticationError,
    SSEConfigError,
    SSEHTTPError,
    SSETransportError,
)

DEFAULT_TIMEOUT = 60.0
DEFAULT_USER_AGENT = (
    "sse-lib/+https://github.com/radlab-dev-group/semantic-search-engine"
)
DEFAULT_TOKEN_TYPE = "Token"
LOCAL_HOSTNAMES = ("localhost", "127.0.0.1", "0.0.0.0", "::1")
RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})
IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
MAX_RESPONSE_SNIPPET = 500


def build_base_url(api_host: str, api_prefix: str = DEFAULT_API_PREFIX) -> str:
    """Return ``scheme://host`` + prefix without any trailing slash.

    ``api_host`` may be given with or without a scheme; when it is missing,
    ``http`` is used for loopback hosts and ``https`` for everything else.  A
    path component of ``api_host`` is kept, which allows serving the API behind
    a sub-path.
    """
    host = (api_host or "").strip()
    if not host:
        raise SSEConfigError("api_host is required, e.g. 'http://localhost:8271'")
    if "://" not in host:
        host = f"{'http' if _is_local(host) else 'https'}://{host}"

    parts = requests.compat.urlsplit(host)
    if not parts.netloc:
        raise SSEConfigError(f"'{api_host}' is not a valid API host")

    prefix = (api_prefix or "").strip().strip("/")
    base = f"{parts.scheme}://{parts.netloc}{_clean_path(parts.path)}"
    if prefix:
        base = f"{base}/{prefix}"
    return base


def _is_local(host: str) -> bool:
    return host.split("://", 1)[-1].split("/")[0].split(":")[0] in LOCAL_HOSTNAMES


def _clean_path(path: str) -> str:
    path = (path or "").strip("/")
    return f"/{path}" if path else ""


class Transport:
    """Sends requests to the SSE REST API and returns unwrapped bodies."""

    def __init__(
        self,
        api_host: str,
        *,
        api_prefix: str = DEFAULT_API_PREFIX,
        timeout: float = DEFAULT_TIMEOUT,
        verify: bool = True,
        session: Optional[requests.Session] = None,
        headers: Optional[Mapping[str, str]] = None,
        user_agent: str = DEFAULT_USER_AGENT,
        max_retries: int = 2,
        retry_backoff: float = 0.5,
    ) -> None:
        if timeout is None or float(timeout) <= 0:
            raise SSEConfigError("timeout has to be a positive number of seconds")
        self.base_url = build_base_url(api_host, api_prefix)
        self.timeout = float(timeout)
        self.verify = verify
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff = float(retry_backoff)
        self._token: Optional[str] = None
        self._token_type = DEFAULT_TOKEN_TYPE
        self._owns_session = session is None
        self.session = session or requests.Session()
        self.session.verify = verify
        default_headers = {"User-Agent": user_agent, "Accept": "application/json"}
        if headers:
            default_headers.update(dict(headers))
        self.session.headers.update(
            {key: value for key, value in default_headers.items() if value}
        )

    # --- configuration -------------------------------------------------------
    @property
    def token(self) -> Optional[str]:
        """Currently used bearer/token value, if any."""
        return self._token

    def set_token(
        self, token: Optional[str], token_type: Optional[str] = None
    ) -> None:
        """Set (or clear with ``None``) the value of ``Authorization``."""
        self._token = token or None
        self._token_type = token_type or DEFAULT_TOKEN_TYPE

    def url_for(self, endpoint: str) -> str:
        """Absolute URL of an endpoint name (or of a path with a leading ``/``)."""
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            return endpoint
        path = endpoint.strip().strip("/")
        if not path:
            return f"{self.base_url}/"
        return f"{self.base_url}/{path}"

    # --- requests ------------------------------------------------------------
    def request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json_body: Optional[Any] = None,
        data: Optional[Any] = None,
        files: Optional[Any] = None,
        headers: Optional[Mapping[str, str]] = None,
        unwrap: bool = True,
    ) -> Any:
        """Perform a single call and return the unwrapped ``body`` value.

        ``json_body`` is sent as a JSON document, ``data``/``files`` as a
        multipart form (uploading files).  Raises the exceptions documented in
        :mod:`sse_lib.exceptions`.
        """
        method = method.upper()
        url = self.url_for(endpoint)
        kwargs: Dict[str, Any] = {
            "params": _prune(params),
            "timeout": self.timeout,
            "headers": self._build_headers(headers),
        }
        if json_body is not None:
            kwargs["json"] = json_body
        if data is not None:
            kwargs["data"] = data
        if files is not None:
            kwargs["files"] = files

        attempts = self.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                response = self.session.request(method, url, **kwargs)
            except requests.RequestException as exc:
                can_retry = (
                    method in IDEMPOTENT_METHODS and attempt <= self.max_retries
                )
                if can_retry:
                    self._sleep_before_retry(attempt)
                    continue
                raise SSETransportError(
                    f"Cannot reach the SSE API at {url}: {exc}"
                ) from exc

            if (
                response.status_code in RETRYABLE_STATUS_CODES
                and method in IDEMPOTENT_METHODS
                and attempt <= self.max_retries
            ):
                self._sleep_before_retry(attempt)
                continue
            return self._handle_response(method, url, endpoint, response, unwrap)

        raise SSETransportError(f"Cannot reach the SSE API at {url}")

    def get(
        self, endpoint: str, params: Optional[Mapping[str, Any]] = None, **kwargs
    ):
        """Shortcut for ``request("GET", ...)``."""
        return self.request("GET", endpoint, params=params, **kwargs)

    def post(
        self,
        endpoint: str,
        json_body: Optional[Any] = None,
        params: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ):
        """Shortcut for ``request("POST", ...)``."""
        return self.request(
            "POST", endpoint, params=params, json_body=json_body, **kwargs
        )

    def close(self) -> None:
        """Close the underlying session when this transport created it."""
        if self._owns_session:
            self.session.close()

    def __enter__(self) -> "Transport":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # --- internals -----------------------------------------------------------
    def _build_headers(self, headers: Optional[Mapping[str, str]]) -> Dict[str, str]:
        prepared = dict(headers or {})
        if self._token and "Authorization" not in prepared:
            prepared["Authorization"] = f"{self._token_type} {self._token}"
        return prepared

    def _sleep_before_retry(self, attempt: int) -> None:
        time.sleep(self.retry_backoff * attempt)

    def _handle_response(
        self,
        method: str,
        url: str,
        endpoint: str,
        response: requests.Response,
        unwrap: bool,
    ) -> Any:
        payload, text = _decode_payload(response)

        if response.status_code >= 400:
            self._raise_for_status(method, url, response, payload, text)

        if not unwrap:
            return payload if payload is not None else text

        if isinstance(payload, dict) and "status" in payload:
            if payload["status"] is True:
                return payload.get("body")
            raise SSEAPIError(
                payload.get("errors"),
                endpoint=endpoint,
                status_code=response.status_code,
            )
        # Endpoints outside the envelope (``login`` returns {"token": "..."}).
        return payload if payload is not None else text

    def _raise_for_status(
        self,
        method: str,
        url: str,
        response: requests.Response,
        payload: Any,
        text: Optional[str],
    ) -> None:
        status = response.status_code
        server_message = _first_error_message(payload)
        reason = server_message or response.reason or "HTTP error"
        snippet = (text or "")[:MAX_RESPONSE_SNIPPET] if text else None
        common = {
            "method": method,
            "url": url,
            "status_code": status,
            "response_text": snippet,
        }
        if status in (401, 403):
            raise SSEAuthenticationError(
                f"Authentication failed: {reason}", **common
            )
        raise SSEHTTPError(f"SSE API error: {reason}", **common)


def _prune(params: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not params:
        return None
    return {key: value for key, value in params.items() if value is not None}


def _decode_payload(response: requests.Response):
    text = response.text
    if not text:
        return None, None
    try:
        return response.json(), text
    except ValueError:
        return None, text


def _first_error_message(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    errors = payload.get("errors")
    if isinstance(errors, list) and errors and isinstance(errors[0], dict):
        return errors[0].get("message")
    return None

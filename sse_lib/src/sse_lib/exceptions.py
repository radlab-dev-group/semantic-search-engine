"""
exceptions.py
-------------

Errors raised by :class:`sse_lib.SSEClient`.

The backend answers with an HTTP 200 plus ``{"status": false, "errors": [...]}``
for business errors (see ``sse_api.core.response.response_with_status``), so a
plain ``raise_for_status()`` is not enough: those become
:class:`SSEAPIError`.  Real transport/HTTP failures keep their status code and
become :class:`SSEHTTPError`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

MESSAGE_KEY = "message"
ERROR_CODE_KEY = "error_code"
NOT_GIVEN_PARAMS_KEY = "not_given_params"


class SSEError(Exception):
    """Base class of every error raised by this library."""


class SSEConfigError(SSEError):
    """The client is misconfigured (unusable host, missing credentials, ...)."""


class SSEValueError(SSEError, ValueError):
    """An argument is rejected before the request is sent to the API."""


class SSETransportError(SSEError):
    """No HTTP response was received (connection refused, DNS, timeout)."""


class SSEHTTPError(SSETransportError):
    """The API answered with a non ``2xx`` status code."""

    def __init__(
        self,
        message: str,
        *,
        method: Optional[str] = None,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        response_text: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.method = method
        self.url = url
        self.status_code = status_code
        self.response_text = response_text

    def __str__(self) -> str:
        if self.status_code is None:
            return self.message
        return f"{self.message} (status={self.status_code}, url={self.url})"


class SSEAuthenticationError(SSEHTTPError):
    """401/403: the token is missing, expired, or lacks the permission."""


class SSEAPIError(SSEError):
    """The API returned ``{"status": false, "errors": [...]}``.

    ``errors`` keeps the server payloads (``message``, ``error_code``,
    ``required_params``, ``not_given_params``, ...) so callers can react to a
    concrete error name instead of parsing a string.
    """

    def __init__(
        self,
        errors: Optional[List[Dict[str, Any]]] = None,
        *,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
    ) -> None:
        self.errors: List[Dict[str, Any]] = [
            error for error in (errors or []) if isinstance(error, dict)
        ]
        self.endpoint = endpoint
        self.status_code = status_code
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        if not self.errors:
            return f"API call failed{self._endpoint_suffix()}"
        parts = []
        for error in self.errors:
            text = str(error.get(MESSAGE_KEY) or "Unknown API error")
            code = error.get(ERROR_CODE_KEY)
            parts.append(f"{text} [{code}]" if code else text)
            missing = error.get(NOT_GIVEN_PARAMS_KEY)
            if missing:
                parts[-1] = f"{parts[-1]}: missing {', '.join(map(str, missing))}"
        return "; ".join(parts) + self._endpoint_suffix()

    def _endpoint_suffix(self) -> str:
        return f" (endpoint={self.endpoint})" if self.endpoint else ""

    @property
    def error_names(self) -> List[str]:
        """``error_code`` of every reported error, in server order."""
        return [str(error.get(ERROR_CODE_KEY)) for error in self.errors]

    @property
    def not_given_params(self) -> List[str]:
        """Union of the parameters reported as missing."""
        params: List[str] = []
        for error in self.errors:
            for param in error.get(NOT_GIVEN_PARAMS_KEY) or []:
                if param not in params:
                    params.append(str(param))
        return params

from typing import Any, Dict, Optional

import requests

from src.config import get_search_url, get_timeout_seconds
from src.mcp_app import mcp


def _normalize_response(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, list):
        return {"results": payload}
    if isinstance(payload, dict):
        return payload
    return {"data": payload}


@mcp.tool()
def semantic_search(
    query: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Wyszukiwarka semantyczna z dokumentacją framework Medusa.
    Używaj jeżeli:
     - klient chce w bazie wiedzy wyszukać informacje
     - chce dowiedzieć jak zakodować coś zgodnie z dokumentacją
    """

    if not query or not query.strip():
        return {"error": "query is empty"}

    body: Dict[str, Any] = {"query": query, "top_k": int(top_k)}
    if filters:
        body["filters"] = filters

    search_url = get_search_url()
    timeout = get_timeout_seconds()

    try:
        resp = requests.post(
            search_url,
            json=body,
            timeout=timeout,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()

        try:
            data = resp.json()
        except ValueError:
            return {"raw_text": resp.text}

        return _normalize_response(data)

    except requests.Timeout:
        return {
            "error": "timeout",
            "details": f"Request to {search_url} exceeded {timeout}s",
        }
    except requests.HTTPError as e:
        return {
            "error": "http_error",
            "status_code": getattr(e.response, "status_code", None),
            "details": "Non-2xx response from semantic search API",
        }
    except requests.RequestException:
        return {
            "error": "request_failed",
            "details": f"Could not connect to {search_url}",
        }

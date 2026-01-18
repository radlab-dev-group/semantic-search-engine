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

    # return {
    #     "results": [
    #         {
    #             "id": "doc:kb:12345",
    #             "score": 0.9321,
    #             "title": "Jak skonfigurować logowanie SSO",
    #             "snippet": "Aby włączyć SSO, przejdź do Ustawienia → Bezpieczeństwo → SSO i wklej metadane IdP...",
    #             "url": "https://docs.example.com/kb/sso-setup",
    #             "metadata": {
    #                 "source": "knowledge_base",
    #                 "lang": "pl",
    #                 "product": "core-app",
    #                 "tags": ["sso", "security", "configuration"],
    #                 "created_at": "2025-01-10T12:00:00Z",
    #                 "updated_at": "2025-12-03T09:30:00Z",
    #             },
    #         },
    #         {
    #             "id": "doc:faq:778",
    #             "score": 0.8476,
    #             "title": "Reset hasła użytkownika",
    #             "snippet": "Reset hasła wykonasz z poziomu panelu administratora w sekcji Użytkownicy → Akcje → Resetuj hasło...",
    #             "url": "https://docs.example.com/faq/password-reset",
    #             "metadata": {
    #                 "source": "faq",
    #                 "lang": "pl",
    #                 "product": "admin-panel",
    #                 "tags": ["password", "account"],
    #                 "audience": "admin",
    #             },
    #         },
    #     ]
    # }

    if not query or not query.strip():
        return {"error": "query is empty"}
    #
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

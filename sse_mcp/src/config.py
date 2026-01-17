import os


def get_sse_api_host() -> str:
    return os.environ.get("SSE_API_HOST", "http://<HOST>:<PORT>")


def get_search_url() -> str:
    return f"{get_sse_api_host().rstrip('/')}/api/search"


def get_timeout_seconds() -> float:
    return float(os.environ.get("SSE_API_TIMEOUT", "20"))

"""Shared utility functions used across the Django application."""

import hashlib


def compute_text_hash(text_str: str) -> str:
    """Return the MD5 hex digest of *text_str*.

    Used for document deduplication — not security-sensitive.
    """
    return hashlib.md5(text_str.encode("utf8")).hexdigest()

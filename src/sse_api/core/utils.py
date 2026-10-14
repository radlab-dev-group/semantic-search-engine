import hashlib


def compute_text_hash(text: str) -> str:
    """
    Computes MD5 hash of the given text.
    """
    if text is None:
        text = ""
    return hashlib.md5(text.encode("utf-8")).hexdigest()

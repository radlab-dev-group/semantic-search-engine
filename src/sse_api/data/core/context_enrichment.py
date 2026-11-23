"""
context_enrichment.py
---------------------

Building text representations which carry the document's identity.

A chunk on its own is context-blind: *"Wnioski należy składać do 30 kwietnia"*
does not say which regulation, which programme or which company it belongs
to, and a query naming the programme never reaches it.  A modern embedding
model attends over the whole input, so prepending a small structured header
(document name, category, path, section) moves those concepts into every
token of the chunk and the vector finally contains them.

Two rules keep this safe:

* the enrichment is for the **embedder only** — PostgreSQL stores the raw
  text, the API displays the raw text, and the reranker builds its own
  enriched candidate; nothing a human reads ever shows the header;
* the header is **bounded** — a pathological document name or path cannot
  outgrow the chunk it explains.
"""

from typing import Optional

#: Label of the header line; also the marker which tells an enriched text
#: from a raw one (re-indexing must never enrich a text twice).
HEADER_PREFIX = "Dokument:"

#: Separator between the header and the chunk body.
HEADER_SEPARATOR = "\n\n"

#: Maximum length of the header; longer values are cut with an ellipsis.
MAX_HEADER_CHARS = 400

#: Markdown headings which name the section a chunk starts with.
_HEADING_MARKERS = ("#", "##", "###")


def extract_section_title(text: Optional[str]) -> Optional[str]:
    """
    Section title a chunk starts with, ``None`` when it has none.

    Recognises a Markdown heading on the first non-empty line (``## Sekcja``)
    and a "title line" — a short single line without terminal punctuation
    followed by the body, which is how a PDF paragraph under a heading often
    arrives.
    """
    if not text:
        return None
    lines = text.split("\n")
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        for marker in _HEADING_MARKERS:
            if stripped.startswith(marker + " "):
                title = stripped[len(marker) :].strip(" #\t")
                return title or None
        # A title line: short, no sentence-ending punctuation, and the text
        # continues underneath it.
        rest_has_body = any(line.strip() for line in lines[index + 1 :])
        if (
            rest_has_body
            and len(stripped) <= 80
            and not stripped.endswith((".", ":", ";", ",", "!", "?"))
        ):
            return stripped
        return None
    return None


def build_context_header(
    document_name: str,
    category: Optional[str] = None,
    relative_path: Optional[str] = None,
    section_title: Optional[str] = None,
) -> str:
    """
    One-line context header, e.g. ``Dokument: a.pdf | Kategoria: finanse``.

    Missing optional fields are left out; the document name alone is still
    worth prepending.
    """
    headers = [f"{HEADER_PREFIX} {document_name}"]
    if category:
        headers.append(f"Kategoria: {category}")
    if relative_path:
        headers.append(f"Ścieżka: {relative_path}")
    if section_title:
        headers.append(f"Sekcja: {section_title}")
    header = " | ".join(headers)
    if len(header) > MAX_HEADER_CHARS:
        header = header[: MAX_HEADER_CHARS - 1].rstrip() + "…"
    return header


def build_enriched_chunk_text(
    chunk_text: str,
    document_name: str,
    category: Optional[str] = None,
    relative_path: Optional[str] = None,
    section_title: Optional[str] = None,
) -> str:
    """
    Text for the embedder: a context header above the chunk's own content.

    Parameters
    ----------
    chunk_text : str
        The chunk as it is stored (cleared version when there is one).
    document_name : str
        Name of the document the chunk belongs to.
    category : str, optional
        Category of the document, when it has one.
    relative_path : str, optional
        Path the document was read from, when it has one.
    section_title : str, optional
        Section heading inside the document.  When it is not given it is
        extracted from the chunk itself (a Markdown heading or a title line).

    Returns
    -------
    str
        ``header`` + blank line + chunk.  An empty chunk, a chunk which is
        already enriched, or one without a usable document name comes back
        unchanged.
    """
    if not chunk_text or not chunk_text.strip():
        return chunk_text or ""
    if not document_name:
        return chunk_text
    body = chunk_text.strip()
    if body.startswith(HEADER_PREFIX):
        # Re-indexing an already enriched text must not stack headers.
        return chunk_text

    header = build_context_header(
        document_name=document_name,
        category=category,
        relative_path=relative_path,
        section_title=(
            section_title
            if section_title is not None
            else extract_section_title(body)
        ),
    )
    return f"{header}{HEADER_SEPARATOR}{body}"


def is_enriched(text: Optional[str]) -> bool:
    """Whether ``text`` carries a context header already."""
    return bool(text) and text.strip().startswith(HEADER_PREFIX)


def raw_chunk_text(text: Optional[str]) -> Optional[str]:
    """
    The chunk without its context header.

    A text stored in Milvus before the separation of "text to embed" from
    "text to store" may hold the header; this is what a reranker or a
    response should read instead.
    """
    if not is_enriched(text):
        return text
    _, _, body = text.partition(HEADER_SEPARATOR)
    return body if body.strip() else text

"""Coordinates in the relational preview, not in an unavailable original file.

Pages retain the existing preview representation: nonempty chunks of one type,
in reading order, separated by two newlines. Overlap and repeated occurrences
are intentionally retained. Offsets are zero-based Unicode code-point indices,
with an exclusive end; no normalization or approximate substring matching is
performed.
"""


def page_content_with_offsets(chunks) -> tuple[str, dict]:
    """Build one page and locate each stored occurrence in that exact text."""
    parts = []
    offsets = {}
    position = 0
    for chunk in sorted(chunks, key=lambda row: (row.text_number, row.pk)):
        text = chunk.text_str
        if not text:
            offsets[chunk.pk] = (None, None)
            continue
        if parts:
            position += 2
        offsets[chunk.pk] = (position, position + len(text))
        position += len(text)
        parts.append(text)
    return "\n\n".join(parts), offsets


def chunk_offsets_for_texts(chunks) -> dict:
    """Batch coordinates for loaded DocumentPageText rows, using one query.

    Search adapters can attach each tuple as ``char_start`` / ``char_end`` to
    their result text dict. A stale, changed, empty, or unmappable chunk gets
    ``(None, None)`` rather than a guessed position.
    """
    from sse_api.data.models import DocumentPageText

    chunks = list(chunks)
    offsets = {chunk.pk: (None, None) for chunk in chunks}
    if not chunks:
        return offsets
    groups = {(chunk.page_id, chunk.text_chunk_type) for chunk in chunks}
    stored = DocumentPageText.objects.filter(
        page_id__in={page_id for page_id, _ in groups},
        text_chunk_type__in={chunk_type for _, chunk_type in groups},
    ).order_by("page_id", "text_chunk_type", "text_number", "id")
    pages = {}
    by_id = {}
    for chunk in stored:
        key = (chunk.page_id, chunk.text_chunk_type)
        if key in groups:
            pages.setdefault(key, []).append(chunk)
            by_id[chunk.pk] = chunk
    positions = {}
    for page in pages.values():
        _, ranges = page_content_with_offsets(page)
        positions.update(ranges)
    for chunk in chunks:
        original = by_id.get(chunk.pk)
        if (
            original is not None
            and original.page_id == chunk.page_id
            and original.text_chunk_type == chunk.text_chunk_type
            and original.text_number == chunk.text_number
            and original.text_str == chunk.text_str
        ):
            offsets[chunk.pk] = positions[chunk.pk]
    return offsets

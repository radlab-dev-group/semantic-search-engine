"""
preview.py
----------

Whole content of one document of a collection, as the engine has it
(``GET document_content``) — the preview of a row of the documents tab.

The text of a document lives in ``DocumentPage`` and ``DocumentPageText``:
every page carries the chunks it was split into, and one page may carry the
chunks of several *chunk types* (raw text, cleared text, the same text split
with another chunk size, ...).  A preview shows one type at a time — by
default the type of the chunks which were stored last, i.e. the form the
document was indexed with most recently.  Nothing is read from the vector
database and no model is loaded: this is the relational text only.

    controller = ContentPreviewController()
    preview = controller.document_content(document=document, from_page=1)
    print(preview.as_dict())
"""

from dataclasses import dataclass, field
from typing import List

from sse_api.data.core.constants import PREVIEW_MAX_PAGES, PREVIEW_MAX_PAGES_LIMIT
from sse_api.data.models import Document, DocumentPage, DocumentPageText

from sse_api.engine.controllers.database.relational_db import RelationalDBController


@dataclass
class DocumentPreview:
    """
    Content of a document, prepared for the API response.

    ``pages`` keeps the page numbers of the document, ``content`` is the same
    text with the pages joined into one string, and ``truncated`` says that
    the document has more pages than this window (ask again with
    ``next_page``).
    """

    document: Document
    chunk_type: str = ""
    chunk_types: List[str] = field(default_factory=list)
    from_page: int = 1
    pages: List[dict] = field(default_factory=list)
    total_pages: int = 0
    truncated: bool = False

    @property
    def content(self) -> str:
        """Text of the window, the pages joined the way the document reads."""
        return "\n\n".join(page["content"] for page in self.pages if page["content"])

    @property
    def chunks(self) -> int:
        return sum(page["chunks"] for page in self.pages)

    @property
    def next_page(self) -> int | None:
        """First page of the following window, ``None`` when there is none."""
        if not self.truncated or not self.pages:
            return None
        return self.pages[-1]["page"] + 1

    def as_dict(self) -> dict:
        """
        Preview prepared for the API response.
        :return: Dictionary with the document, the chunk type the text comes
        from, its pages, the whole text of the window and the limits of it
        """
        content = self.content
        return {
            "document_id": self.document.pk,
            "document_name": self.document.name,
            "relative_path": self.document.relative_path,
            "category": self.document.category,
            "use_in_search": self.document.use_in_search,
            "chunk_type": self.chunk_type,
            "chunk_types": list(self.chunk_types),
            "pages": self.pages,
            "content": content,
            "characters": len(content),
            "chunks": self.chunks,
            "from_page": self.from_page,
            "total_pages": self.total_pages,
            "truncated": self.truncated,
            "next_page": self.next_page,
        }


class ContentPreviewController:
    """
    Reads the whole text of a document out of the relational database.
    """

    def find_document(
        self,
        collection,
        document_id: int | None = None,
        document_name: str | None = None,
    ) -> Document | None:
        """
        One document of the collection, addressed by its id or its name.

        A name is unique per (collection, path, hash), so the same name may
        be used by more than one document of a collection; the most recently
        indexed one is shown, the same way the search of a re-indexed document
        works on its newest form.
        :param collection: Collection to search in
        :param document_id: id of the document, preferred over the name
        :param document_name: Name of the document
        :return: Document or None when the collection has no such document
        """
        if document_id is not None:
            documents = RelationalDBController.get_collection_documents_by_ids(
                collection=collection, document_ids=[document_id]
            )
        else:
            documents = RelationalDBController.get_collection_documents_by_names(
                collection=collection, document_names=[document_name]
            )
        return documents.order_by("-id").first()

    def document_content(
        self,
        document: Document,
        from_page: int = 1,
        max_pages: int = PREVIEW_MAX_PAGES,
        chunk_type: str | None = None,
    ) -> DocumentPreview:
        """
        Text of one window of the pages of a document.
        :param document: Document to read
        :param from_page: First page of the window (pages are numbered from 1)
        :param max_pages: How many pages the window holds at most
        :param chunk_type: Chunk type to read, ``None`` for the newest one
        :return: DocumentPreview
        """
        max_pages = self.__bounded_pages(max_pages)
        from_page = max(1, int(from_page or 1))
        chunk_types = self.chunk_types(document)
        chosen = self.__resolve_chunk_type(document, chunk_types, chunk_type)

        total_pages = DocumentPage.objects.filter(document=document).count()
        pages = list(
            DocumentPage.objects.filter(
                document=document, page_number__gte=from_page
            ).order_by("page_number")[: max_pages + 1]
        )
        truncated = len(pages) > max_pages
        window = pages[:max_pages]

        texts = self.__chunk_texts(window, chosen)
        rows = [
            {
                "page": page.page_number,
                "content": "\n\n".join(texts.get(page.pk, [])),
                "chunks": len(texts.get(page.pk, [])),
            }
            for page in window
        ]
        return DocumentPreview(
            document=document,
            chunk_type=chosen,
            chunk_types=chunk_types,
            from_page=from_page,
            pages=rows,
            total_pages=total_pages,
            truncated=truncated,
        )

    @staticmethod
    def chunk_types(document: Document) -> List[str]:
        """
        Chunk types the text of the document is stored in.
        :param document: Document to inspect
        :return: Sorted list of ``text_chunk_type`` values, empty when the
        document has no text at all
        """
        return sorted(
            set(
                DocumentPageText.objects.filter(page__document=document)
                .values_list("text_chunk_type", flat=True)
                .distinct()
            )
        )

    @staticmethod
    def __chunk_texts(pages, chunk_type: str) -> dict:
        """
        Text of every page of the window: its chunks of one type, joined in
        the order the document reads them.
        :param pages: DocumentPage rows of the window
        :param chunk_type: Chunk type to read
        :return: ``{page_id: text}``
        """
        if not len(pages) or not chunk_type:
            return {}
        chunks = DocumentPageText.objects.filter(
            page__in=pages, text_chunk_type=chunk_type
        ).order_by("page_id", "text_number", "id")
        texts: dict = {}
        for page_id, text in chunks.values_list("page_id", "text_str"):
            if text:
                texts.setdefault(page_id, []).append(text)
        return texts

    @staticmethod
    def __resolve_chunk_type(
        document: Document, chunk_types: List[str], chunk_type: str | None
    ) -> str:
        """
        Chunk type to read: the one the client asked for when the document
        has it, otherwise the type of the chunks which were stored last.
        :param document: Document to read
        :param chunk_types: Types the document is stored in
        :param chunk_type: Type asked for, ``None`` when the client has no
        preference
        :return: Name of the type to read, ``""`` when there is none
        """
        if not len(chunk_types):
            return ""
        if chunk_type and chunk_type in chunk_types:
            return chunk_type
        newest = (
            DocumentPageText.objects.filter(page__document=document)
            .order_by("-id")
            .values_list("text_chunk_type", flat=True)
            .first()
        )
        return newest or chunk_types[-1]

    @staticmethod
    def __bounded_pages(max_pages) -> int:
        """
        Window of pages the server is willing to answer with.
        :param max_pages: Number asked for
        :return: ``max_pages`` kept inside ``[1, PREVIEW_MAX_PAGES_LIMIT]``,
        the default when the value is not a number
        """
        try:
            pages = int(max_pages)
        except (TypeError, ValueError):
            return PREVIEW_MAX_PAGES
        return max(1, min(pages, PREVIEW_MAX_PAGES_LIMIT))

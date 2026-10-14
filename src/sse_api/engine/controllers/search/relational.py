from collections import defaultdict

from django.contrib.postgres.search import SearchVector, SearchQuery, SearchRank
from sse_api.data.models import CollectionOfDocuments, DocumentPageText, Document


class DBTextSearchController:
    """
    Helper controller that fetches raw text fragments from the relational
    database and builds the result structures expected by the semantic
    search controller.
    """

    def __init__(self):
        """
        Initialise the ``DBTextSearchController``.  No internal state is
        required at the moment.
        """
        pass

    def get_texts(
        self, texts_ids: list, texts_scores: list, surrounding_chunks: int = 0
    ) -> list:
        """
        Retrieve ``DocumentPageText`` objects for the given IDs and attach
        their scores, efficiently fetching surrounding context in batches.

        Parameters
        ----------
        texts_ids : list
            List of primary keys of ``DocumentPageText`` records.
        texts_scores : list
            Corresponding relevance scores.
        surrounding_chunks : int, default 0
            Number of neighbouring text chunks to include as context.

        Returns
        -------
        list
            List of dictionaries containing the main text and its context.
        """
        document_pages = list(
            DocumentPageText.objects.filter(id__in=texts_ids).select_related(
                "page__document"
            )
        )
        # Map IDs to objects to maintain order from texts_ids
        pages_map = {page.id: page for page in document_pages}

        context_map = {}
        if surrounding_chunks > 0 and document_pages:
            context_map = self._get_surrounding_chunks(
                document_pages, surrounding_chunks=surrounding_chunks
            )

        doc_results = []
        for text_id, score in zip(texts_ids, texts_scores):
            doc_page = pages_map.get(text_id)
            if doc_page:
                doc_results.append(
                    self._prepare_document_page_with_map(
                        doc_page,
                        score=float(score),
                        context_map=context_map,
                    )
                )
        return doc_results

    @staticmethod
    def _get_surrounding_chunks(
        document_pages: list, surrounding_chunks: int
    ) -> dict:
        """
        Find the neighbouring chunks of every text in ``document_pages``.

        Neighbours are taken from the reading order of the whole document
        (page number, then chunk number) among chunks of the same
        ``text_chunk_type``. Readers without page metadata (txt, md, html)
        store every chunk on its own ``DocumentPage``, so looking for
        neighbours on the same page only would always return nothing.

        Returns
        -------
        dict
            ``{text_id: (left_context, right_context)}`` where both are lists
            of ``{"text_number": ..., "text_str": ...}`` ordered as in the
            document (the nearest left chunk is the last one, the nearest
            right chunk is the first one).
        """
        document_ids = {page.page.document_id for page in document_pages}
        ordered = DocumentPageText.objects.filter(
            page__document_id__in=document_ids
        ).order_by("page__document_id", "page__page_number", "text_number", "id")

        sequences = defaultdict(list)  # (document_id, chunk_type) -> [(id, tn)]
        positions = {}  # text_id -> (sequence key, index)
        for text_id, document_id, chunk_type, text_number in ordered.values_list(
            "id", "page__document_id", "text_chunk_type", "text_number"
        ):
            key = (document_id, chunk_type)
            positions[text_id] = (key, len(sequences[key]))
            sequences[key].append((text_id, text_number))

        neighbours = {}
        needed_ids = set()
        for page in document_pages:
            if page.id not in positions:
                continue
            key, index = positions[page.id]
            sequence = sequences[key]
            left = sequence[max(index - surrounding_chunks, 0) : index]
            right = sequence[index + 1 : index + 1 + surrounding_chunks]
            neighbours[page.id] = (left, right)
            needed_ids.update(text_id for text_id, _ in left + right)

        texts = dict(
            DocumentPageText.objects.filter(id__in=needed_ids).values_list(
                "id", "text_str"
            )
        )

        def as_context(chunks: list) -> list:
            return [
                {"text_number": text_number, "text_str": texts[text_id]}
                for text_id, text_number in chunks
                if text_id in texts
            ]

        return {
            text_id: (as_context(left), as_context(right))
            for text_id, (left, right) in neighbours.items()
        }

    def _prepare_document_page_with_map(
        self,
        doc_page: DocumentPageText,
        score: float,
        context_map: dict,
    ) -> dict:
        """
        Build the result dictionary using a pre-fetched context map
        (see ``_get_surrounding_chunks``).
        """
        main_text = {
            "score": score,
            "document_name": doc_page.page.document.name,
            "relative_path": doc_page.page.document.relative_path,
            "page_number": doc_page.page.page_number,
            "text_number": doc_page.text_number,
            "language": doc_page.language,
            "text_str": doc_page.text_str,
        }

        left_context, right_context = context_map.get(doc_page.id, ([], []))

        return {
            "result": {
                "left_context": left_context,
                "text": main_text,
                "right_context": right_context,
            }
        }

    @staticmethod
    def get_all_categories(collection: CollectionOfDocuments):
        """
        Return the distinct categories present in ``collection``.

        Parameters
        ----------
        collection : CollectionOfDocuments
            The collection to inspect.

        Returns
        -------
        QuerySet
            Distinct category values.
        """
        categories = (
            Document.objects.filter(collection=collection)
            .values_list("category", flat=True)
            .distinct()
        )
        return categories

    @staticmethod
    def get_all_documents(collection: CollectionOfDocuments):
        """
        Return the distinct document names present in ``collection``.

        Parameters
        ----------
        collection : CollectionOfDocuments
            The collection to inspect.

        Returns
        -------
        QuerySet
            Distinct document names.
        """
        documents = (
            Document.objects.filter(collection=collection)
            .values_list("name", flat=True)
            .distinct()
        )
        return documents

    @staticmethod
    def documents_names_from_categories(
        collection: CollectionOfDocuments,
        categories: list,
        only_used_to_search: bool = True,
    ):
        """
        Retrieve document names that belong to any of the supplied
        ``categories``.

        Parameters
        ----------
        collection : CollectionOfDocuments
            The collection to query.
        categories : list
            List of category strings.
        only_used_to_search : bool, default True
            If ``True`` limit to documents marked ``use_in_search=True``.

        Returns
        -------
        QuerySet
            Document names matching the categories.
        """
        opts = {"collection": collection.pk, "category__in": categories}
        if only_used_to_search:
            opts["use_in_search"] = True
        return (
            Document.objects.filter(**opts).values_list("name", flat=True).distinct()
        )

    @staticmethod
    def document_names_relative_path_contains(
        collection: CollectionOfDocuments,
        texts: list[str],
        only_used_to_search: bool = True,
    ) -> list[str]:
        """
                Find document names whose relative path contains any of the given
                substrings.

                Parameters
                ----------
                collection : CollectionOfDocuments
                    The collection to search.
                texts : list[str]
                    Substrings to look for in ``relative_path``.
                only_used_to_search : bool, default True
                    Restrict to documents marked ``use_in_search=True``.

                Returns
        -------
                list[str]
                    Unique document names matching at least one substring.
        """
        all_doc_contains = []
        for text in texts:
            opts = {"collection": collection.pk, "relative_path__contains": text}
            if only_used_to_search:
                opts["use_in_search"] = True
            doc_contains = (
                Document.objects.filter(**opts)
                .values_list("name", flat=True)
                .distinct()
            )
            if len(doc_contains):
                all_doc_contains.extend(doc_contains)
        return list(set(all_doc_contains))

    def search_texts(
        self,
        query_str: str,
        collection: CollectionOfDocuments,
        max_results: int = 100,
        filters: dict = None,
        language: str = None,
    ) -> list:
        """
        Perform a full-text search in PostgreSQL using FTS functionality.

        Parameters
        ----------
        query_str : str
            The query string to search for.
        collection : CollectionOfDocuments
            The collection to search within.
        max_results : int, default 100
            Maximum number of results to return.
        filters : dict, optional
            Additional filters (e.g., document_names, relative_paths).
        language : str, optional
            Language for the FTS configuration (e.g., 'polish', 'english').

        Returns
        -------
        list
            List of tuples (text_id, rank).
        """
        queryset = DocumentPageText.objects.filter(
            page__document__collection=collection
        )

        if filters:
            if "document_names" in filters and filters["document_names"]:
                queryset = queryset.filter(
                    page__document__name__in=filters["document_names"]
                )
            if "relative_paths" in filters and filters["relative_paths"]:
                queryset = queryset.filter(
                    page__document__relative_path__in=filters["relative_paths"]
                )
            if "language" in filters and filters["language"]:
                queryset = queryset.filter(language=filters["language"])
        elif language:
            queryset = queryset.filter(language=language)

        # Map language code to Postgres FTS config name
        lang_config = "simple"
        if language == "pl":
            lang_config = "polish"
        elif language == "en":
            lang_config = "english"

        vector = SearchVector(
            "text_str", weight="A", config=lang_config
        ) + SearchVector("text_str_clear", weight="B", config=lang_config)

        query = SearchQuery(query_str, config=lang_config)

        results = (
            queryset.annotate(rank=SearchRank(vector, query))
            .filter(rank__gt=0.001)
            .order_by("-rank")[:max_results]
        )

        return [(res.id, res.rank) for res in results]

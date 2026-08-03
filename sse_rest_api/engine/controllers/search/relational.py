from data.models import CollectionOfDocuments, DocumentPageText, Document


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

        # Identify all needed context chunks
        context_map = {}
        if surrounding_chunks > 0:
            page_ids = set()
            needed_context = []  # List of (page_id, text_number)
            for page in document_pages:
                page_ids.add(page.page_id)
                tn = page.text_number
                # Left context
                for i in range(max(tn - surrounding_chunks, 0), tn):
                    needed_context.append((page.page_id, i))
                # Right context
                for i in range(tn + 1, tn + surrounding_chunks + 1):
                    needed_context.append((page.page_id, i))

            # Batch fetch all potential context chunks for these pages
            if page_ids:
                all_possible_contexts = DocumentPageText.objects.filter(
                    page_id__in=page_ids
                ).values("page_id", "text_number", "text_str")
                for ctx in all_possible_contexts:
                    context_map[(ctx["page_id"], ctx["text_number"])] = ctx["text_str"]

        doc_results = []
        for text_id, score in zip(texts_ids, texts_scores):
            doc_page = pages_map.get(text_id)
            if doc_page:
                doc_results.append(
                    self._prepare_document_page_with_map(
                        doc_page,
                        score=float(score),
                        surrounding_chunks=surrounding_chunks,
                        context_map=context_map,
                    )
                )
        return doc_results

    def _prepare_document_page_with_map(
        self,
        doc_page: DocumentPageText,
        score: float,
        surrounding_chunks: int,
        context_map: dict,
    ) -> dict:
        """
        Build the result dictionary using a pre-fetched context map.
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

        left_context = []
        right_context = []

        if surrounding_chunks > 0:
            tn = doc_page.text_number
            pid = doc_page.page_id
            # Left
            for i in range(max(tn - surrounding_chunks, 0), tn):
                val = context_map.get((pid, i))
                if val is not None:
                    left_context.append({"text_number": i, "text_str": val})
            # Right
            for i in range(tn + 1, tn + surrounding_chunks + 1):
                val = context_map.get((pid, i))
                if val is not None:
                    right_context.append({"text_number": i, "text_str": val})

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


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
        their scores.

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
            List of dictionaries produced by ``_prepare_document_page``.
        """
        document_pages = DocumentPageText.objects.filter(id__in=texts_ids)
        doc_results = []
        for idx, doc_page in enumerate(document_pages):
            text_score = float(texts_scores[idx])
            doc_results.append(
                self._prepare_document_page(
                    doc_page, score=text_score, surrounding_chunks=surrounding_chunks
                )
            )
        return doc_results

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

    def _prepare_document_page(
        self, doc_page: DocumentPageText, score: float, surrounding_chunks: int
    ) -> dict:
        """
        Build the result dictionary for a single ``DocumentPageText``
        instance, including surrounding context if requested.

        Parameters
        ----------
        doc_page : DocumentPageText
            The text fragment to process.
        score : float
            Relevance score associated with this fragment.
        surrounding_chunks : int
            Number of adjacent chunks to include as left/right context.

        Returns
        -------
        dict
            Structured representation of the fragment and its context.
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
        left_context = self._prepare_text_context(
            doc_page, surrounding_chunks, context="left"
        )
        right_context = self._prepare_text_context(
            doc_page, surrounding_chunks, context="right"
        )
        return {
            "result": {
                "left_context": left_context,
                "text": main_text,
                "right_context": right_context,
            }
        }

    def _prepare_text_context(self, doc_page, surrounding_chunks, context) -> list:
        """
        Retrieve surrounding text chunks for ``doc_page`` either to the
        ``left`` or ``right`` of the current fragment.

        Parameters
        ----------
        doc_page : DocumentPageText
            Reference fragment.
        surrounding_chunks : int
            Number of neighbouring chunks to fetch.
        context : str
            Either ``"left"`` or ``"right"``.

        Returns
        -------
        list
            List of dictionaries ``{'text_number': int, 'text_str': str}``
            representing the surrounding context.
        """
        text_num = doc_page.text_number
        if context == "left":
            beg_position = max(text_num - surrounding_chunks, 0)
            ctx_nums = [x for x in range(beg_position, text_num)]
        elif context == "right":
            ctx_nums = [
                x for x in range(text_num + 1, text_num + surrounding_chunks + 1)
            ]
        else:
            raise Exception(f"Unknown context type {context}")

        context_res = []
        doc_pages = DocumentPageText.objects.filter(
            text_number__in=ctx_nums, page=doc_page.page
        )
        for d_page in doc_pages:
            context_res.append(
                {
                    "text_number": d_page.text_number,
                    "text_str": d_page.text_str,
                }
            )
        return context_res

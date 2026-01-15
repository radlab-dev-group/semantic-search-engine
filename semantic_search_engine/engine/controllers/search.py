"""
search.py
---------

Utility module providing high‑level search functionality built on top of
Milvus vector store, Django ORM models and optional transformer‑based
embedder/reranker models.  It defines three main controller classes:

* ``SearchQueryController`` – orchestrates creation of a user query and
  retrieval of the corresponding response.
* ``DBSemanticSearchController`` – wraps low‑level indexing and search
  operations, handling filtering, metadata queries and result
  post‑processing.
* ``DBTextSearchController`` – fetches raw text fragments from the
  relational database and formats them for display.

The module is used by the web application to answer user questions
against a collection of documents.
"""

import math
import tqdm
import json
import pandas as pd
from django.db.models import QuerySet
from transformers import AutoTokenizer

from radlab_data.text.utils import TextUtils

from main.src.constants import get_logger
from data.models import (
    Document,
    DocumentPageText,
    CollectionOfDocuments,
    QueryTemplate,
)

from engine.controllers.relational_db import RelationalDBController
from data.controllers.constants import NORMALIZE_EMBEDDINGS
from data.controllers.query_templates import QueryTemplateController

from engine.models import UserQuery, UserQueryResponse
from system.models import OrganisationUser

from engine.controllers.milvus import MilvusHandler
from engine.controllers.embedders_rerankers import EmbeddingModelsConfig


class SearchQueryController:
    """
    High‑level controller that creates a ``UserQuery`` record,
    runs the semantic search and stores the resulting ``UserQueryResponse``.
    """

    def __init__(self):
        """
        Initialise a ``SearchQueryController`` instance.

        Currently no internal state is required; the constructor is kept
        for future extensibility.
        """
        pass

    @staticmethod
    def new_query(
        query_str: str,
        search_options_dict: dict,
        collection: CollectionOfDocuments,
        organisation_user: OrganisationUser,
        sse_engin_config_path: str,
        ignore_question_lang_detect: bool = False,
    ):
        """
        Create a new user query, execute the search and persist the response.

        Parameters
        ----------
        query_str : str
            The raw text entered by the user.
        search_options_dict : dict
            Dictionary of search options (filters, ranking flags, etc.).
        collection : CollectionOfDocuments
            The document collection against which the search is performed.
        organisation_user : OrganisationUser
            The user performing the query.
        sse_engin_config_path : str
            Path to the semantic search engine configuration file.
        ignore_question_lang_detect : bool, optional
            If ``True`` the language detection step is skipped.

        Returns
        -------
        dict
            A dictionary containing the search ``results`` and the
            ``query_response_id``.  If template prompts were generated they
            are included under the ``template_prompts`` key.
        """
        new_query_obj = UserQuery.objects.create(
            organisation_user=organisation_user,
            collection=collection,
            query_str_prompt=query_str,
            query_options=search_options_dict,
        )

        sem_db_controller = (
            DBSemanticSearchController.prepare_controller_for_collection(
                collection=collection, sse_engin_config_path=sse_engin_config_path
            )
        )

        # template_prompts
        results, structured_results, template_prompts = (
            sem_db_controller.search_with_options(
                question_str=query_str,
                search_params=search_options_dict,
                convert_to_pd=False,
                reformat_to_display=True,
                ignore_question_lang_detect=ignore_question_lang_detect,
                organisation_user=organisation_user,
                collection=collection,
                user_query=new_query_obj,
            )
        )

        query_response = UserQueryResponse.objects.create(
            user_query=new_query_obj,
            general_stats_json=results.get("stats", {}),
            detailed_results_json=results.get("detailed_results", {}),
            structured_results=structured_results,
        )

        query_response_result = {
            "results": results,
            "query_response_id": query_response.pk,
        }
        if len(template_prompts):
            query_response_result["template_prompts"] = template_prompts

        return query_response_result

    @staticmethod
    def get_user_response_by_id(query_response_id) -> UserQueryResponse | None:
        """
        Retrieve a ``UserQueryResponse`` instance by its primary key.

        Parameters
        ----------
        query_response_id : int
            Primary key of the ``UserQueryResponse`` to fetch.

        Returns
        -------
        UserQueryResponse | None
            The matching response object, or ``None`` if it does not exist.
        """
        try:
            return UserQueryResponse.objects.get(id=query_response_id)
        except UserQueryResponse.DoesNotExist:
            return None


class DBSemanticSearchController:
    """
    Controller that handles indexing of document texts into Milvus and
    executing vector‑based searches with optional metadata filtering,
    reranking and result post‑processing.
    """

    POSSIBLE_OPERATORS = ["in", "eq", "ne", "gt", "lt", "gte", "lte", "hse"]
    """
    `hse` operation --> has same elem:
     list_a = [1, 2] list_b = [2, 5, 6, 7] --> return True
     list_a = [] list_b = [2, 5, 6, 7] --> return False
     list_a = [2] list_b = [2, 5, 6, 7] --> return True
     list_a = [1, 2, 3, 5] list_b = [6, 7] --> return False
     list_a = [1, 2, 3, 5] list_b = [5] --> return True
    """

    def __init__(
        self,
        jsonl_config_path: str,
        collection_name: str,
        index_name: str | None,
        batch_size: int = 10,
        embedder_model: str | None = None,
        cross_encoder_model: str | None = None,
    ):
        """
        Initialise the controller with configuration for Milvus, embedder
        and optional reranker models.

        Parameters
        ----------
        jsonl_config_path : str
            Path to the JSONL configuration for Milvus collections.
        collection_name : str
            Name of the Milvus collection.
        index_name : str | None
            Optional name of the Milvus index to use.
        batch_size : int, default 10
            Number of documents processed per batch during indexing.
        embedder_model : str | None, optional
            Identifier of the transformer model used for embedding texts.
        cross_encoder_model : str | None, optional
            Identifier of the cross‑encoder model used for reranking.
        """
        self._logger = get_logger()

        self.batch_size = batch_size
        self.max_tokens_in_text = 508

        embedder_device = ""
        embedder_model_path = None
        embedder_vector_size = -1
        self.emb_tokenizer = None
        if embedder_model is not None and len(embedder_model):
            embedder_model_path = EmbeddingModelsConfig.get_embedder_path(
                embedder_model
            )
            embedder_vector_size = EmbeddingModelsConfig.get_embedder_vector_size(
                embedder_model
            )
            embedder_device = EmbeddingModelsConfig.get_embedder_device(
                embedder_model
            )
            self.emb_tokenizer = AutoTokenizer.from_pretrained(embedder_model_path)

        reranker_device = ""
        reranker_model_path = None
        if cross_encoder_model is not None and len(cross_encoder_model):
            reranker_model_path = EmbeddingModelsConfig.get_reranker_path(
                cross_encoder_model
            )
            reranker_device = EmbeddingModelsConfig.get_reranker_device(
                cross_encoder_model
            )

        self._milvus_handler = MilvusHandler(
            jsonl_config_path=jsonl_config_path,
            collection_name=collection_name,
            collection_description=collection_name,
            embedder_model_path=embedder_model_path,
            embedding_size=embedder_vector_size,
            index_name=index_name,
            reranker_model_path=reranker_model_path,
            embedder_device=embedder_device,
            reranker_device=reranker_device,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
        )

        self._text_db_controller = RelationalDBController()
        self._template_controller = QueryTemplateController()

    @staticmethod
    def prepare_controller_for_collection(
        collection: CollectionOfDocuments, sse_engin_config_path: str
    ):
        """
        Factory method that creates a ``DBSemanticSearchController`` instance
        pre‑configured for the supplied collection.

        Parameters
        ----------
        collection : CollectionOfDocuments
            The collection for which the controller should be prepared.
        sse_engin_config_path : str
            Path to the semantic‑search engine configuration file.

        Returns
        -------
        DBSemanticSearchController
            Configured controller ready to index or search the collection.
        """
        return DBSemanticSearchController(
            jsonl_config_path=sse_engin_config_path,
            collection_name=collection.name,
            index_name=collection.embedder_index_type,
            batch_size=10,
            embedder_model=collection.model_embedder,
            cross_encoder_model=collection.model_reranker,
        )

    def index_texts(self, from_collection: CollectionOfDocuments) -> None:
        """
        Index all text fragments belonging to ``from_collection`` into
        Milvus.

        Parameters
        ----------
        from_collection : CollectionOfDocuments
            The source collection whose texts will be indexed.
        """
        all_doc_pages_texts = self._text_db_controller.get_all_texts_from_collection(
            from_collection
        )

        return self.index_texts_from_list(all_doc_pages_texts, from_collection)

    def index_texts_from_list(
        self, all_texts: list | QuerySet, collection: CollectionOfDocuments
    ) -> None:
        """
        Index a list (or ``QuerySet``) of ``DocumentPageText`` objects.

        Parameters
        ----------
        all_texts : list | QuerySet
            Iterable containing text objects to be indexed.
        collection : CollectionOfDocuments
            The collection to which the texts belong.
        """
        with tqdm.tqdm(total=len(all_texts), desc="Indexing documents") as pbar:
            # batched_texts = []
            for text in all_texts:
                str_text_to_index = text.text_str
                if text.text_str_clear and len(text.text_str_clear):
                    str_text_to_index = text.text_str_clear

                if len(str_text_to_index) < 10:
                    pbar.update()
                    continue

                text_metadata = {
                    "external_text_id": str(text.id),
                    "external_document_id": str(text.page.document.pk),
                    "external_collection_id": collection.pk,
                    "text_language": text.language,
                    "filename": text.page.document.name,
                    "relative_path": text.page.document.relative_path,
                }

                text_tokens = self.emb_tokenizer(str_text_to_index).input_ids
                if len(text_tokens) > self.max_tokens_in_text:
                    str_text_to_index = self.emb_tokenizer.decode(
                        token_ids=text_tokens[: self.max_tokens_in_text],
                        skip_special_tokens=True,
                    )

                self._milvus_handler.add_single_text(
                    text=str_text_to_index, metadata=text_metadata
                )
                pbar.update()
        return None

    def search(
        self,
        search_text: str,
        max_results: int,
        rerank_results: bool = True,
        language: str = None,
        additional_output_fields: list | None = None,
        return_with_factored_fields: bool = False,
        search_in_documents: list = None,
        relative_paths: list = None,
    ) -> []:
        """
        Perform a vector search in Milvus with optional metadata filters.

        Parameters
        ----------
        search_text : str
            The query string to embed and search for.
        max_results : int
            Maximum number of results to return.
        rerank_results : bool, default True
            Whether to apply the reranker model to the raw results.
        language : str, optional
            Language filter – only vectors with this language are considered.
        additional_output_fields : list | None, optional
            Extra fields to retrieve from Milvus.
        return_with_factored_fields : bool, default False
            If ``True`` the returned objects include additional factor fields.
        search_in_documents : list, optional
            List of document names to restrict the search to.
        relative_paths : list, optional
            List of relative file paths to restrict the search to.

        Returns
        -------
        list
            List of search hits returned by Milvus.
        """
        metadata_filter = {}
        if language is not None and len(language):
            metadata_filter["text_language"] = language
        if search_in_documents is not None and len(search_in_documents):
            metadata_filter["filenames"] = search_in_documents
        if relative_paths is not None and len(relative_paths):
            metadata_filter["relative_paths"] = relative_paths

        post_search_options = {
            "rerank_results": rerank_results,
            "return_with_factored_fields": return_with_factored_fields,
        }

        milvus_search = self._milvus_handler.search(
            search_text=search_text,
            max_results=max_results,
            additional_output_fields=additional_output_fields,
            metadata_filter=metadata_filter,
            post_search_options=post_search_options,
        )
        return milvus_search

    def search_with_options(
        self,
        question_str: str,
        search_params: dict,
        convert_to_pd: bool = False,
        reformat_to_display: bool = False,
        ignore_question_lang_detect: bool = False,
        organisation_user: OrganisationUser = None,
        collection: CollectionOfDocuments = None,
        user_query: UserQuery = None,
    ):
        """
        High‑level search method that interprets ``search_params`` (filters,
        ranking options, etc.), performs the vector search and formats the
        output.

        Sample search_params
        {
          "categories": [
            "Sport",
            "Rozrywka"
          ],
          "documents": [],
          "relative_paths": [],
          "templates": 8,
          "only_template_documents": true,
          "max_results": 40,
          "rerank_results": false,
          "return_with_factored_fields": false,
          "relative_path_contains": [
            "https://cam.waw.pl"
          ]
        }

        Parameters
        ----------
        question_str : str
            The user query.
        search_params : dict
            Dictionary containing search options (categories, documents,
            templates, pagination, etc.).
        convert_to_pd : bool, default False
            If ``True`` the statistics are returned as a pandas DataFrame.
        reformat_to_display : bool, default False
            If ``True`` the detailed results are reformatted for UI display.
        ignore_question_lang_detect : bool, default False
            Skip language detection for the query.
        organisation_user : OrganisationUser, optional
            The user performing the search.
        collection : CollectionOfDocuments, optional
            The collection to search within.
        user_query : UserQuery, optional
            The ``UserQuery`` instance to associate with the search.

        Returns
        -------
        tuple
            ``(results, structured_results, template_prompts)`` where
            * ``results`` is a dict with ``query``, ``stats`` and
              ``detailed_results``,
            * ``structured_results`` is a list of structured documents
              (or empty list),
            * ``template_prompts`` is a list of system prompts generated
              from matched templates.
        """

        doc_names_from_cat = []
        textual_controller = DBTextSearchController()

        use_and_operator = search_params.get("use_and_operator", False)
        if use_and_operator is None:
            use_and_operator = False

        categories = search_params.get("categories", [])
        categories = [] if categories is None else categories

        if len(categories):
            doc_names_from_cat = textual_controller.documents_names_from_categories(
                collection=collection,
                categories=categories,
                only_used_to_search=True,
            )

        # Directly given document names
        document_names = search_params.get("documents", [])
        document_names = [] if document_names is None else document_names

        # Directly given relatives paths
        relative_paths = search_params.get("relative_paths", [])
        relative_paths = [] if relative_paths is None else relative_paths

        was_filter_option = False
        # Names of documents which contains given phrase (if given)
        relative_doc_names = []
        relative_path_contains = search_params.get("relative_path_contains", [])
        if relative_path_contains is not None and len(relative_path_contains):
            relative_doc_names = (
                textual_controller.document_names_relative_path_contains(
                    collection=collection,
                    texts=relative_path_contains,
                    only_used_to_search=True,
                )
            )
            was_filter_option = True
            if not len(relative_doc_names) and use_and_operator:
                return {}, {}, []

        # Names of documents which match the template (if given)
        q_document_names = []
        query_templates = self._template_controller.prepare_templates_for_user(
            organisation_user=organisation_user,
            templates=search_params.get("templates", []),
            return_only_data_connector=False,
        )
        template_prompts = []
        if len(query_templates):
            if user_query is not None:
                for template in query_templates:
                    if type(template) in [dict]:
                        continue
                    user_query.query_templates.add(template)

                    if template.system_prompt is not None and len(
                        template.system_prompt.strip()
                    ):
                        template_prompts.append(template.system_prompt)

            template_doc_names = self.__get_documents_based_on_templates(
                query_templates=query_templates,
                collection=collection,
                return_documents_names=True,
            )

            only_templates = search_params.get("only_template_documents", False)
            if only_templates:
                doc_names_from_cat = []
                document_names = []
                relative_paths = []

            was_filter_option = True
            q_document_names.extend(template_doc_names)
            if not len(q_document_names) and only_templates:
                self._logger.warning("No documents found from template")
                return {}, {}, []

            if not len(q_document_names) and use_and_operator:
                return {}, {}, []

        # Names of documents which match the metadata filter (if given)
        metadata_doc_names = []
        metadata_filters = search_params.get("metadata_filters", [])
        if metadata_filters is not None and len(metadata_filters):
            if type(metadata_filters) in [dict]:
                metadata_filters = [metadata_filters]

            metadata_doc_names = self.__filter_documents_based_on_metadata(
                collection=collection,
                metadata_filters=metadata_filters,
                use_and_operator=False,
            )
            was_filter_option = True

            if not len(metadata_doc_names) and use_and_operator:
                return {}, {}, []

        # Prepare OR/AND operator between all doc names from:
        #  - document_names
        #  - doc_names_from_cat
        #  - q_document_names
        #  - metadata_doc_names
        #  - relative_doc_names
        # At the beginning only non-empty documents names lists will be used
        filter_names_lists = []
        for doc_names_list in [
            doc_names_from_cat,
            document_names,
            q_document_names,
            metadata_doc_names,
            relative_doc_names,
        ]:
            if len(doc_names_list):
                filter_names_lists.append(doc_names_list)
        # Using non-empty lists names prepare intersection between all lists
        docs_to_search = []
        if len(filter_names_lists):
            and_or_doc_names = set(filter_names_lists[0])
            for n_list in filter_names_lists[1:]:
                if use_and_operator:
                    and_or_doc_names = set(n_list) & and_or_doc_names
                else:
                    and_or_doc_names = set(n_list) | and_or_doc_names
            docs_to_search = list(and_or_doc_names)

        # self._logger.debug(
        #     json.dumps(docs_to_search, indent=2, ensure_ascii=False)
        # )

        # When categories are selected but no documents after filtering
        if not len(docs_to_search) and len(categories):
            self._logger.warning("No documents to search!")
            return {}, {}, []

        if was_filter_option and not len(relative_paths) and not len(docs_to_search):
            self._logger.warning(
                "was_filter_option is set to True and no documents found!"
            )
            return {}, {}, []

        # if user passed language then only embeddings with
        # given language will be filtered to calculate similarity
        lang_str = (
            None
            if ignore_question_lang_detect
            else TextUtils.text_language(question_str)
        )

        # Call search method
        query_results = self.search(
            search_text=question_str,
            max_results=int(search_params.get("max_results", 50)),
            rerank_results=bool(search_params.get("rerank_results", False)),
            language=lang_str,
            return_with_factored_fields=bool(
                search_params.get("return_with_factored_fields", False)
            ),
            search_in_documents=docs_to_search,
            relative_paths=relative_paths,
        )[0]
        if not len(query_results):
            self._logger.warning("query_results is empty!")
            return {}, {}, []

        # Prepare results to presents for user
        texts_ids = []
        texts_scores = []
        for text_res in query_results:
            texts_ids.append(int(text_res["metadata"]["external_text_id"]))
            texts_scores.append(text_res["score"])

        postgres_docs = textual_controller.get_texts(
            texts_ids=texts_ids, texts_scores=texts_scores, surrounding_chunks=2
        )

        results = {
            "query": question_str,
            "stats": self.filter_stats_to_display_results(
                doc_stats=self.prepare_documents_stats(postgres_docs),
                remove_under_hits=1,
                remove_under_pages=1,
            ),
            "detailed_results": postgres_docs,
        }

        structured_results = None
        if len(query_templates):
            structured_results = self.__prepare_structured_results(
                results=postgres_docs,
                collection=collection,
                query_templates=query_templates,
            )
        if structured_results is None or not len(structured_results):
            structured_results = []
        results["structured_results"] = structured_results

        if reformat_to_display:
            results["detailed_results"] = self.reformat_search_results_to_display(
                search_results=results["detailed_results"]
            )
        if convert_to_pd:
            results["stats"] = pd.DataFrame.transpose(
                pd.DataFrame.from_dict(results["stats"])
            ).sort_values("score_weighted", ascending=False)

            ref_display = results["detailed_results"]
            if not reformat_to_display:
                ref_display = self.reformat_search_results_to_display(
                    search_results=results["detailed_results"]
                )

            results["detailed_results"] = pd.DataFrame.transpose(
                pd.DataFrame.from_dict(ref_display)
            ).sort_values("score", ascending=False)
        return results, structured_results, template_prompts

    def __filter_documents_based_on_metadata(
        self,
        collection: CollectionOfDocuments,
        metadata_filters: list,
        use_and_operator: bool,
    ) -> list[Document]:
        """
        Filter collection of documents based on metadata filters.

        Sample single metadata filters:
        [
            {
              "operator": "in",
              "field":
                "deep_labels": {
                  "0": ["kategoria"]
                }
              }
            },
            {
              "operator": "eq",
              "field": {
                  "main_category": "kategoria"
              }
            },
            {
                "operator": "lt",
                "field": {
                    "kategoria": {
                        "gdzie_wartosc": {
                            "jest_bardzo_gleboko": 100
                        }
                    }
                }
            }
        ]

        Possible operators: ["in", "eq", "ne", "gt", "lt", "gte", "lte", "hse"]

        Field may be any nested, and is stored into Document.metadata_json field.

        Parameters
        ----------
        collection : CollectionOfDocuments
            Collection to filter.
        metadata_filters : list
            List of dictionaries (single metadata filter).
        use_and_operator : bool
            Whether to combine filters with ``AND`` (True) or ``OR`` (False).

        Returns
        -------
        list[Document]
            List of filtered Documents.
        """
        if not len(metadata_filters):
            return []

        self._logger.info("Filtering documents based on metadata filters")
        self._logger.debug(
            json.dumps(metadata_filters, indent=2, ensure_ascii=False)
        )

        all_doc_names = []
        for m_d_filter in metadata_filters:
            docs_with_md_filter = (
                self.__get_documents_with_metadata_filter_expression(
                    collection=collection,
                    expression=m_d_filter,
                )
            )
            if len(docs_with_md_filter):
                all_doc_names.append([d.name for d in docs_with_md_filter])

        if not len(all_doc_names):
            self._logger.info("No documents after filtering")
            return []

        # Prepare AND operator for all_doc_names
        docs_names_and_or = set(all_doc_names[0])
        for d in all_doc_names[1:]:
            if use_and_operator:
                docs_names_and_or = set(d) & docs_names_and_or
            else:
                docs_names_and_or = set(d) | docs_names_and_or
        docs_names_and_or = list(docs_names_and_or)

        self._logger.info(
            f"Number of unique documents after "
            f"metadata filtering: {len(docs_names_and_or)}"
        )

        return docs_names_and_or

    def __get_documents_with_metadata_filter_expression(
        self, collection: CollectionOfDocuments, expression: dict
    ) -> list[Document]:
        """
        Retrieve documents from ``collection`` that satisfy a single
        ``expression`` (operator + field definition).

        Parameters
        ----------
        collection : CollectionOfDocuments
            The collection to search.
        expression : dict
            Dictionary describing the operator and field to filter on.

        Returns
        -------
        list[Document]
            Documents matching the expression.
        """
        c_docs = self._text_db_controller.get_all_documents_from_collection(
            collection=collection
        )
        filtered_documents = []
        for doc in c_docs:
            if self.__is_document_covered_with_metadata_expression(
                document=doc, expression=expression
            ):
                filtered_documents.append(doc)
        return filtered_documents

    def __is_document_covered_with_metadata_expression(
        self, document: Document, expression: dict
    ) -> bool:
        """
        Check whether a single ``document`` satisfies the provided
        ``expression``.

        Parameters
        ----------
        document : Document
            Document instance to test.
        expression : dict
            Expression containing ``operator`` and ``field``.

        Returns
        -------
        bool
            ``True`` if the document matches the expression, ``False`` otherwise.
        """
        if document.metadata_json is None or not len(document.metadata_json):
            return False

        assert self.__is__proper__expression(
            expression=expression
        ), "Expression is not valid!"

        return self.__is_properly_matched_expression_with_metadata(
            expr_operator=expression["operator"],
            expr_dict=expression["field"],
            metadata_dict=document.metadata_json,
        )

    def __is_properly_matched_expression_with_metadata(
        self, expr_operator: str, expr_dict: dict, metadata_dict: dict
    ) -> bool:
        """
        Recursively evaluate a nested metadata expression against a
        document's metadata.

        Parameters
        ----------
        expr_operator : str
            One of the supported operators (``in``, ``eq`` … ``hse``).
        expr_dict : dict
            The field definition (may be nested).
        metadata_dict : dict
            The document's ``metadata_json`` dictionary.

        Returns
        -------
        bool
            ``True`` if the expression matches, ``False`` otherwise.
        """
        for k, v in expr_dict.items():
            if k not in metadata_dict:
                return False

            if type(v) in [dict]:
                return self.__is_properly_matched_expression_with_metadata(
                    expr_operator=expr_operator,
                    expr_dict=v,
                    metadata_dict=metadata_dict[k],
                )

            return self.__check_expression_operator_value(
                expr_operator=expr_operator,
                expr_value=v,
                metadata_value=metadata_dict[k],
            )
        return False

    def __check_expression_operator_value(
        self, expr_operator: str, expr_value, metadata_value
    ) -> bool:
        """
        Evaluate a primitive comparison between ``expr_value`` and
        ``metadata_value`` according to ``expr_operator``.

        Parameters
        ----------
        expr_operator : str
            Operator name (e.g., ``in``, ``eq``).
        expr_value : Any
            Value from the filter expression.
        metadata_value : Any
            Corresponding value from the document metadata.

        Returns
        -------
        bool
            Result of the comparison.
        """
        expr_operator = expr_operator.lower().strip()
        if expr_operator == "in":
            return expr_value in metadata_value
        elif expr_operator == "eq":
            return expr_value == metadata_value
        elif expr_operator == "ne":
            return expr_value != metadata_value
        elif expr_operator == "gt":
            return expr_value > metadata_value
        elif expr_operator == "lt":
            return expr_value < metadata_value
        elif expr_operator == "ge":
            return expr_value >= metadata_value
        elif expr_operator == "le":
            return expr_value <= metadata_value
        elif expr_operator == "hse":
            # Dict is not hashable, return False if expr_value
            # or metadata_value has dict type
            if type(expr_value) in [dict] or type(metadata_value) in [dict]:
                return False
            # If expression/metadata value is not list,
            # then convert value as list(value)
            if type(expr_value) not in [list]:
                expr_value = list(expr_value)
            if type(metadata_value) not in [list]:
                metadata_value = list([metadata_value])
            # self._logger.debug(f"{expr_value} {expr_operator} {metadata_value}")
            # Check intersection size between two lists of elements
            return len(set(expr_value).intersection(set(metadata_value))) > 0
        else:
            self._logger.error(
                f"Unknown expression operator: {expr_operator} "
                f"choose one of {self.POSSIBLE_OPERATORS}"
            )
        return False

    def __is__proper__expression(self, expression: dict) -> bool:
        """
        Validate that ``expression`` contains required keys and a supported
        operator.

        Parameters
        ----------
        expression : dict
            Expression dictionary to validate.

        Returns
        -------
        bool
            ``True`` if the expression is well‑formed, ``False`` otherwise.
        """
        if None in [
            expression.get("operator", None),
            expression.get("field", None),
        ]:
            return False

        if expression["operator"] not in self.POSSIBLE_OPERATORS:
            return False

        return True

    @staticmethod
    def __prepare_structured_results(
        results: list[dict],
        collection: CollectionOfDocuments,
        query_templates: list[QueryTemplate],
    ):
        """
        Build a list of structured document dictionaries based on the first
        template that requests a structured response.

        Parameters
        ----------
        results : list[dict]
            Raw search results.
        collection : CollectionOfDocuments
            Collection from which the documents originate.
        query_templates : list[QueryTemplate]
            Templates used in the query.

        Returns
        -------
        list[dict] | None
            List of structured documents or ``None`` if no template requires it.
        """
        template_to_use = None
        for qt in query_templates:
            if qt.structured_response_if_exists:
                template_to_use = qt
                break
        if template_to_use is None:
            return None

        metadata_fields = template_to_use.structured_response_data_fields
        if metadata_fields is None or not len(metadata_fields):
            return None

        structured_docs = []
        structured_docs_ids = []
        for result in results:
            document = (
                RelationalDBController.get_document_from_col_by_name_and_rel_path(
                    collection=collection,
                    document_name=result["result"]["text"]["document_name"],
                    relative_path=result["result"]["text"]["relative_path"],
                )
            )
            if document is None or document.pk in structured_docs_ids:
                continue
            structured_docs.append(document)
            structured_docs_ids.append(document.pk)

        structured_docs_out = []
        for doc in structured_docs:
            if doc.metadata_json is None:
                continue
            struct_doc = {
                "name": doc.name,
                "path": doc.path,
                "relative_path": doc.relative_path,
            }
            for md_field in metadata_fields:
                md_val = doc.metadata_json.get(md_field, None)
                if md_val is not None:
                    struct_doc[md_field] = md_val
            if len(struct_doc) > 3:
                structured_docs_out.append(struct_doc)
        return structured_docs_out

    @staticmethod
    def get_accumulated_docs_by_rank_perc(
        results: dict, perc_rank_gen_qa: float
    ) -> list:
        """
        Return document names whose cumulative weighted score reaches the
        given percentage of the total.

        Parameters
        ----------
        results : dict
            Search results containing a ``stats`` mapping.
        perc_rank_gen_qa : float
            Target cumulative percentage (e.g., ``0.2`` for 20 %).

        Returns
        -------
        list
            Ordered list of document names.
        """
        doc_stats = results["stats"]
        sorted_doc_names = sorted(
            list(doc_stats.keys()),
            key=lambda x: (doc_stats[x]["score_weighted_scaled"]),
            reverse=True,
        )

        perc_max = perc_rank_gen_qa
        if perc_rank_gen_qa > 1:
            perc_max = float(perc_rank_gen_qa / 100)

        ret_docs = []
        accum_perc = 0.0
        for doc_name in sorted_doc_names:
            if accum_perc >= perc_max:
                break
            accum_perc += results["stats"][doc_name]["score_weighted_scaled"]
            ret_docs.append(doc_name)
        return ret_docs

    @staticmethod
    def convert_search_results_to_doc2answer(
        search_results: dict,
        which_docs: list | None,
        use_doc_names_in_response: bool = False,
    ) -> dict:
        """
        Convert raw search results into a mapping ``{document_name: [answers]}``.

        Parameters
        ----------
        search_results : dict
            Mapping of result identifiers to result dictionaries.
        which_docs : list | None
            If provided, only include results from these documents.
        use_doc_names_in_response : bool, default False
            Prefix each answer with its document name.

        Returns
        -------
        dict
            ``{document_name: [answer strings]}``.
        """
        doc2answers = {}
        for _, result in search_results.items():
            doc_name = result["document_name"]
            if (
                which_docs is not None
                and len(which_docs)
                and doc_name not in which_docs
            ):
                continue

            if doc_name not in doc2answers:
                doc2answers[doc_name] = []
            text_str = result["text_str"]
            if use_doc_names_in_response:
                text_str = f"{doc_name}: {text_str}"
            doc2answers[doc_name].append(text_str)
        return doc2answers

    @staticmethod
    def prepare_documents_stats(
        postgres_docs, smooth_factor: float = 0.0001
    ) -> dict:
        """
        Compute per‑document statistics (hits, average score, weighted
        score, etc.) from a list of PostgreSQL document fragments.

        Parameters
        ----------
        postgres_docs : list
            List of document fragment dictionaries returned by
            ``_prepare_document_page``.
        smooth_factor : float, default 0.0001
            Minimum value used when scaling weighted scores.

        Returns
        -------
        dict
            Mapping ``{document_name: stats_dict}``.
        """
        doc_stats = dict()
        for result in postgres_docs:
            result = result["result"]
            doc_name = result["text"]["document_name"]
            doc_score = result["text"]["score"]
            doc_page_number = result["text"]["page_number"]
            if doc_name not in doc_stats:
                doc_stats[doc_name] = {
                    "score": 0.0,
                    "score_weighted": 0.0,
                    "hits": 0,
                    "pages": [],
                    "pages_count": 0,
                    "relative_path": result["text"]["relative_path"],
                }
            doc_stats[doc_name]["hits"] += 1
            doc_stats[doc_name]["score"] += doc_score
            doc_stats[doc_name]["pages"].append(doc_page_number)

        w_scores = []
        for doc, res in doc_stats.items():
            res["score"] = res["score"] / res["hits"]
            res["pages"] = sorted(set(res["pages"]))
            res["pages_count"] = len(res["pages"])
            res["score_weighted"] = math.log(
                float(res["score"] * res["hits"] * res["pages_count"])
            )
            w_scores.append(res["score_weighted"])
        # percentage-like scaling with a smooth factor
        min_w_scores = abs(min(w_scores))
        sum_w_scores = sum([s + min_w_scores for s in w_scores])
        for doc, res in doc_stats.items():
            v = res["score_weighted"]
            if sum_w_scores > 0.0:
                doc_stats[doc]["score_weighted_scaled"] = max(
                    smooth_factor, (v + min_w_scores) / sum_w_scores
                )
            else:
                doc_stats[doc]["score_weighted_scaled"] = 1.0
        return doc_stats

    @staticmethod
    def filter_stats_to_display_results(
        doc_stats: dict, remove_under_hits: int = 5, remove_under_pages: int = 3
    ) -> dict:
        """
        Filter out documents that do not meet minimal hit or page count
        thresholds.

        Parameters
        ----------
        doc_stats : dict
            Statistics dictionary produced by ``prepare_documents_stats``.
        remove_under_hits : int, default 5
            Minimum number of hits required.
        remove_under_pages : int, default 3
            Minimum number of distinct pages required.

        Returns
        -------
        dict
            Filtered statistics dictionary.
        """
        filter_result = dict()
        for doc, res in doc_stats.items():
            if (
                res["hits"] >= remove_under_hits
                and res["pages_count"] >= remove_under_pages
            ):
                filter_result[doc] = res
        return filter_result

    @staticmethod
    def reformat_search_results_to_display(
        search_results: list | dict,
    ) -> dict | list:
        """
        Transform raw search results into a UI‑friendly structure that
        contains scores, context snippets and document identifiers.

        Parameters
        ----------
        search_results : list | dict
            Raw results as returned by ``search``.

        Returns
        -------
        dict | list
            Reformatted results ready for presentation.
        """
        ans_dict = {}
        for idx, answer in enumerate(search_results):
            a = answer["result"]
            left_ctx_str, right_ctx_str = "", ""
            if len(a["left_context"]):
                left_ctx_str = a["left_context"][-1]["text_str"]
            if len(a["right_context"]):
                right_ctx_str = a["right_context"][-1]["text_str"]
            ans_dict[idx] = {
                "score": a["text"]["score"],
                "document_name": a["text"]["document_name"],
                "relative_filepath": a["text"]["relative_path"],
                "page_number": a["text"]["page_number"],
                "text_number": a["text"]["text_number"],
                "language": a["text"]["language"],
                "text_str": a["text"]["text_str"],
                "left_context": left_ctx_str,
                "right_context": right_ctx_str,
            }
        return ans_dict

    def __get_documents_based_on_templates(
        self,
        query_templates: list[QueryTemplate],
        collection: CollectionOfDocuments,
        return_documents_names: bool = False,
    ) -> list[str | Document]:
        """
        Retrieve documents that match any of the supplied ``query_templates``.

        Parameters
        ----------
        query_templates : list[QueryTemplate]
            Templates defining data‑connector constraints.
        collection : CollectionOfDocuments
            Collection from which to fetch documents.
        return_documents_names : bool, default False
            If ``True`` return only document names; otherwise return full
            ``Document`` objects.

        Returns
        -------
        list[str] | list[Document]
            List of matching document names or objects.
        """
        data_filter = {}
        for template in query_templates:
            for dc_name, dc_value in template.data_connector.items():
                data_filter[f"metadata_json__{dc_name}"] = dc_value

        # NOTE:
        # In case when no data_connector is defined, then this `filter`
        # returns all documents from database
        template_docs = Document.objects.filter(
            collection=collection, use_in_search=True, **data_filter
        )
        print("1 len(template_docs)=", len(template_docs))
        template_docs = self._template_controller.filter_documents(
            documents=template_docs, query_templates=query_templates
        )
        print("2 len(template_docs)=", len(template_docs))
        if return_documents_names:
            template_docs = [d.name for d in template_docs]
        return template_docs


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

import os
import json
import tqdm

from typing import List, Dict, Any
from django.db.models import Q, QuerySet
from django.apps import apps
from django.db import transaction

from radlab_data.text.reader import DirectoryFileReader
from radlab_data.text.document import Document as InputTextDocument

from sse_api.system.models import OrganisationUser, OrganisationGroup
from sse_api.data.models import (
    CollectionOfDocuments,
    UploadedDocuments,
    Document,
    DocumentPage,
    DocumentPageText,
    QueryTemplate,
)
from sse_api.data.controllers.denoiser import DenoiserController
from sse_api.engine.controllers.models_logic.embedders_rerankers import (
    EmbeddingModelsConfig,
)
from sse_api.core.utils import compute_text_hash


class RelationalDBController:
    """
    Text data controller.
    """

    def __init__(self, store_to_db: bool = True):
        self.dir_reader = None
        self.store_to_db = store_to_db
        self._denoiser_controller = None

    @staticmethod
    def get_visible_collections(user: OrganisationUser, check_in_organisation=True):
        if user is None:
            return CollectionOfDocuments.objects.none()
        visibility = Q(created_by=user)
        if check_in_organisation:
            groups = user.user_groups.filter(organisation=user.organisation)
            visibility |= Q(visible_to_groups__in=groups)
        return CollectionOfDocuments.objects.filter(visibility).distinct()

    @staticmethod
    def get_user_collections(
        user: OrganisationUser, semantic_collections: List[str]
    ):
        filter_opts = {"created_by": user}
        if len(semantic_collections):
            filter_opts["name__in"] = semantic_collections

        db_collections = CollectionOfDocuments.objects.filter(**filter_opts)
        return db_collections

    @staticmethod
    def get_user_organisation_collections(
        user: OrganisationUser, semantic_collections: List[Any]
    ):
        return RelationalDBController.get_visible_collections(user).filter(
            name__in=semantic_collections
        )

    def add_uploaded_documents_to_db(
        self,
        collection_name: str,
        organisation_user: OrganisationUser,
        uploaded_document: UploadedDocuments,
        prepare_proper_pages: bool,
        merge_document_pages: bool = False,
        clear_texts: bool = True,
        use_text_denoiser: bool = False,
        max_tokens_in_chunk: int = None,
        number_of_overlap_tokens: int = None,
        check_text_language: bool = True,
        number_of_process: int = 1,
    ) -> (UploadedDocuments, list):
        """
        Adds previously uploaded files to database.
        :param collection_name: Name of collection to add documents
        :param organisation_user: User who uploaded documents
        :param uploaded_document: UploadedDocuments
        :param prepare_proper_pages: Prepare proper pages (if set to True)
        :param merge_document_pages: Merge whole documents into single page
        :param clear_texts: Clear texts (if set to True)
        :param use_text_denoiser: Use text denoiser (if set to True)
        :param max_tokens_in_chunk: Number of tokens to keep in
        single chunk (if given)
        :param number_of_overlap_tokens: Number of overlapped tokens
        between chunks (if given)
        :param check_text_language: Check text language (if set to True)
        :param number_of_process: Number of processes to use (default is 1)
        :return: UploadedDocuments
        """
        self.dir_reader = DirectoryFileReader(
            main_dir_path=uploaded_document.dir_path,
            read_sub_dirs=True,
            processes_count=number_of_process,
        )

        self.dir_reader.load(
            prepare_proper_pages=prepare_proper_pages,
            merge_document_pages=merge_document_pages,
            clear_texts=clear_texts,
            use_text_denoiser=use_text_denoiser,
            max_tokens_in_chunk=max_tokens_in_chunk,
            number_of_overlap_tokens=number_of_overlap_tokens,
            check_text_language=check_text_language,
        )

        collection = self.get_add_collection(
            collection_name=collection_name,
            created_by=organisation_user,
            collection_display_name="",
            collection_description="",
            model_embedder="",
            model_reranker="",
            embedder_index_type="",
            embedder_index_params="",
            embedder_search_params="",
        )

        added_document_chunks = []
        with tqdm.tqdm(
            total=len(self.dir_reader.documents), desc="Adding documents to database"
        ) as pbar:
            for doc in self.dir_reader.documents:
                doc_as_dict = doc.as_dict
                if self.exists_document_in_db(
                    doc_as_dict=doc_as_dict, collection=collection
                ):
                    print("Skipped document, already in database")
                    pbar.update(1)
                    continue

                _, document_pages_chunks = self.add_document_from_dict(
                    collection=collection,
                    document_dict=doc_as_dict,
                    organisation_user=organisation_user,
                    upload_process=uploaded_document,
                    use_text_denoiser=use_text_denoiser,
                )
                added_document_chunks.extend(document_pages_chunks)
                pbar.update()
        uploaded_document.number_of_indexed_documents_rel_db = len(
            self.dir_reader.documents
        )
        if self.store_to_db:
            uploaded_document.save()

        return uploaded_document, added_document_chunks

    def add_documents_from_list_of_doc_as_dict(
        self,
        collection,
        texts: List[Dict],
        organisation_user: OrganisationUser,
        indexing_options: Dict,
    ):
        indexed_documents = []
        for doc_as_dict in texts:
            for page in doc_as_dict["pages"]:
                doc_to_db = InputTextDocument(
                    file_path=doc_as_dict["filepath"] + ".input_text",
                    relative_file_path=doc_as_dict["relative_filepath"],
                    file_name=doc_as_dict["filepath"],
                    prepare_proper_pages=indexing_options["prepare_proper_pages"],
                    merge_document_pages=indexing_options["merge_document_pages"],
                    clear_texts=indexing_options["clear_text"],
                    use_text_denoiser=indexing_options["use_text_denoiser"],
                    max_tokens_in_chunk=indexing_options["max_tokens_in_chunk"],
                    number_of_overlap_tokens=indexing_options[
                        "number_of_overlap_tokens"
                    ],
                    check_text_language=indexing_options["check_text_lang"],
                    document_category=doc_as_dict["category"],
                    document_content=page["page_content"],
                )
                doc_to_db.load()

                d_as_dict = doc_to_db.as_dict
                if self.exists_document_in_db(
                    doc_as_dict=d_as_dict, collection=collection
                ):
                    print("Skipped document, already in database")
                    continue

                metadata_dict = doc_as_dict.get("options", {})
                if metadata_dict is not None:
                    if "options" not in d_as_dict:
                        d_as_dict["options"] = {}
                    for k, v in metadata_dict.items():
                        d_as_dict["options"][k] = v

                document, document_pages_chunks = self.add_document_from_dict(
                    collection=collection,
                    document_dict=d_as_dict,
                    organisation_user=organisation_user,
                    upload_process=None,
                    use_text_denoiser=False,
                )

                indexed_documents.append(
                    {"document": document, "pages_chunks": document_pages_chunks}
                )
        return indexed_documents

    def add_document_from_dict(
        self,
        collection: CollectionOfDocuments,
        document_dict: dict,
        organisation_user: OrganisationUser,
        upload_process: UploadedDocuments = None,
        use_text_denoiser: bool = False,
    ) -> (Document, list):
        """
        Adds a document from dictionary to collection of documents.
        When collection doesn't exist it will be created.
        :param organisation_user:
        :param collection: Is the collection to store document
        :param document_dict: The document in dictionary format
        :param upload_process: If given then document will be connected with upload proces
        :param use_text_denoiser: Text will be denoised if true
        :return: Document
        """
        metadata = document_dict["options"]
        metadata_as_str = json.dumps(metadata, indent=2)

        try:
            document = Document.objects.get(
                name=os.path.basename(document_dict["filepath"]),
                collection=collection,
                path=document_dict["filepath"],
                document_hash=compute_text_hash(
                    metadata_as_str + document_dict["filepath"]
                ),
            )
            cr = False
        except Document.DoesNotExist:
            document, cr = Document.objects.get_or_create(
                name=os.path.basename(document_dict["filepath"]),
                document_hash=compute_text_hash(
                    metadata_as_str + document_dict["filepath"]
                ),
                collection=collection,
                path=document_dict["filepath"],
                relative_path=document_dict["relative_filepath"],
                category=document_dict["category"],
                upload_process=upload_process,
                metadata_json=metadata,
                added_by=organisation_user,
            )

        if cr and document.upload_process is not None:
            document.category = self._prepare_document_category(document)
            document.save()

        document_pages_chunks = []
        for page_number, doc_page in enumerate(document_dict["pages"], start=1):
            d_page, d_page_text = self.get_add_document_page(
                document,
                doc_page,
                use_text_denoiser=use_text_denoiser,
                page_number=page_number,
            )
            document_pages_chunks.append(d_page_text)
        return document, document_pages_chunks

    @staticmethod
    def _prepare_document_category(document: Document) -> str:
        if document.upload_process is None:
            return document.category

        upload_dir = document.upload_process.dir_path
        clear_upl_dir = upload_dir[2:] if upload_dir.startswith("./") else upload_dir
        document_category = (
            document.path.replace(document.name, "")
            .replace(upload_dir, "")
            .replace(clear_upl_dir, "")
            .strip("/")
            .strip("\\")
        )

        return document_category

    def get_add_document_page(
        self,
        document: Document,
        doc_page: dict,
        use_text_denoiser: bool = False,
        page_number: int = 1,
    ) -> (DocumentPage, DocumentPageText):
        """
        Adds a document when document doesn't exist, otherwise it will be returned
        :param document: Document to add
        :param doc_page: Document page (as dict)
        :param use_text_denoiser: Use text denoiser if is True
        :param page_number: Fallback page number when reader metadata omits it
        :return: DocumentPage
        """
        metadata = doc_page["metadata"]
        reader_page_number = metadata.get("page")
        if reader_page_number is None:
            reader_page_number = page_number
        page, _ = DocumentPage.objects.get_or_create(
            document=document, page_number=int(reader_page_number)
        )
        doc_page_text_chunk = self.add_document_chunk(
            page, doc_page, use_text_denoiser=use_text_denoiser
        )

        return page, doc_page_text_chunk

    def _run_denoiser(self, text_to_denoiser: str) -> str:
        if self._denoiser_controller is None:
            self._denoiser_controller = DenoiserController(load_model=True)
        return self._denoiser_controller.denoise_text(text=text_to_denoiser)

    def add_document_chunk(
        self, page: DocumentPage, doc_page: dict, use_text_denoiser: bool = False
    ) -> DocumentPageText:
        metadata = doc_page["metadata"]
        text_hash = compute_text_hash(doc_page["page_content"])
        denoised_page_content = None
        if use_text_denoiser:
            denoised_page_content = self._run_denoiser(
                text_to_denoiser=doc_page["page_content"]
            ).replace("\x00", "")

        text_chunk, _ = DocumentPageText.objects.get_or_create(
            page=page,
            text_number=int(metadata["chunk"]),
            text_str=doc_page["page_content"].replace("\x00", ""),
            text_str_clear=denoised_page_content,
            text_hash=text_hash,
            text_chunk_type=metadata["text_chunk_type"],
            language=metadata.get("language", ""),
            metadata_json=metadata,
        )
        return text_chunk

    @staticmethod
    def get_all_texts_from_collection(
        collection: CollectionOfDocuments,
    ) -> QuerySet[DocumentPageText]:
        return DocumentPageText.objects.filter(
            page__document__collection=collection,
        )

    @staticmethod
    def get_all_documents_from_collection(
        collection: CollectionOfDocuments,
    ) -> QuerySet[Document]:
        return Document.objects.filter(collection=collection.pk)

    @staticmethod
    def get_all_categories_from_collection(collection: CollectionOfDocuments):
        return (
            Document.objects.filter(collection=collection, use_in_search=True)
            .values_list("category", flat=True)
            .distinct("category")
        )

    @staticmethod
    def get_collection(
        created_by: OrganisationUser,
        collection_name: str,
        check_in_organisation: bool = True,
        owner_only: bool = False,
    ) -> CollectionOfDocuments | None:
        """
        Find the collection by name.
        :param created_by: User looking for the collection
        :param collection_name: Name of the collection to find
        :param check_in_organisation: Include collections shared with the
        groups of the user organisation
        :param owner_only: Return only the collections created by the user,
        used for the operations which change the collection (f.e. deletion)
        :return: CollectionOfDocuments or None when the collection doesn't
        exist or is not visible to the user
        """
        if owner_only:
            return CollectionOfDocuments.objects.filter(
                created_by=created_by, name=collection_name
            ).first()
        return (
            RelationalDBController.get_visible_collections(
                created_by, check_in_organisation=check_in_organisation
            )
            .filter(name=collection_name)
            .first()
        )

    @staticmethod
    def get_collection_from_user_group(
        organisation_user: OrganisationUser, collection_name: str
    ) -> CollectionOfDocuments:
        return (
            RelationalDBController.get_visible_collections(organisation_user)
            .filter(name=collection_name)
            .first()
        )

    @staticmethod
    def get_organisation_templates(
        organisation: OrganisationUser,
    ) -> QuerySet[QueryTemplate]:
        return QueryTemplate.objects.filter(
            template_grammar__template_collection__organisation=organisation,
            is_active=True,
        )

    @staticmethod
    def get_documents_to_search_from_collection(
        collection: CollectionOfDocuments,
    ) -> QuerySet[Document]:
        return Document.objects.filter(collection=collection, use_in_search=True)

    @staticmethod
    def set_documents_use_in_search(
        documents,
        use_in_search: bool,
    ) -> int:
        """
        Enable or disable the use of documents in the search
        (``Document.use_in_search``).
        :param documents: Documents to change (iterable of Document)
        :param use_in_search: Value to set on the documents
        :return: Count of the updated documents
        """
        document_ids = [document.pk for document in documents]
        if not document_ids:
            return 0
        return Document.objects.filter(pk__in=document_ids).update(
            use_in_search=use_in_search
        )

    @staticmethod
    def get_document_from_col_by_name_and_rel_path(
        collection: CollectionOfDocuments,
        document_name: str,
        relative_path: str,
    ) -> Document or None:
        docs = Document.objects.filter(
            collection=collection,
            name=document_name,
            relative_path=relative_path,
        )
        if len(docs) > 0:
            return docs[0]
        return None

    @staticmethod
    def get_add_collection(
        collection_name: str,
        created_by: OrganisationUser,
        collection_display_name: str,
        collection_description: str,
        model_embedder: str,
        model_reranker: str,
        embedder_index_type: str,
        embedder_index_params: str = None,
        embedder_search_params: str = None,
        add_to_group: OrganisationGroup = None,
        metadata_json: dict = None,
    ) -> CollectionOfDocuments:
        collection, created = CollectionOfDocuments.objects.get_or_create(
            name=collection_name, created_by=created_by
        )
        if created:
            model_embedder_vector_size = -1
            if model_embedder is not None and len(model_embedder):
                model_embedder_vector_size = (
                    EmbeddingModelsConfig.get_embedder_vector_size(model_embedder)
                )
            collection.display_name = collection_display_name
            collection.description = collection_description
            collection.model_embedder = model_embedder
            collection.model_reranker = model_reranker
            collection.model_embedder_vector_size = model_embedder_vector_size
            collection.embedder_index_type = embedder_index_type
            collection.embedder_index_params = embedder_index_params
            collection.embedder_search_params = embedder_search_params
            collection.metadata_json = metadata_json
            if add_to_group is not None:
                if not CollectionOfDocuments.objects.filter(
                    pk=collection.pk, visible_to_groups__in=[add_to_group]
                ).exists():
                    collection.visible_to_groups.add(add_to_group)
            collection.save()
        return collection

    @staticmethod
    def upload_documents_to_dir(destination_dir):
        return destination_dir

    @staticmethod
    def exists_document_in_db(
        doc_as_dict: dict, collection: CollectionOfDocuments
    ) -> bool:
        name = doc_as_dict.get("name", None)
        if name is None:
            name = os.path.basename(doc_as_dict["filepath"])
        return Document.objects.filter(collection=collection, name=name).exists()

    # ------------------------------------------------------------------
    # Deleting collection content
    # ------------------------------------------------------------------

    @staticmethod
    def get_collection_documents_by_ids(
        collection: CollectionOfDocuments, document_ids
    ) -> QuerySet[Document]:
        """
        Documents of the collection with given ids.
        :param collection: Collection to search in
        :param document_ids: ids of the documents to find
        :return: QuerySet of Document
        """
        return Document.objects.filter(
            collection=collection, pk__in=[int(pk) for pk in document_ids]
        )

    @staticmethod
    def get_collection_documents_by_names(
        collection: CollectionOfDocuments, document_names
    ) -> QuerySet[Document]:
        """
        Documents of the collection with given names.
        :param collection: Collection to search in
        :param document_names: Names of the documents to find
        :return: QuerySet of Document
        """
        return Document.objects.filter(
            collection=collection, name__in=list(document_names)
        )

    @staticmethod
    def get_collection_chunks_by_ids(
        collection: CollectionOfDocuments, chunk_ids
    ) -> QuerySet[DocumentPageText]:
        """
        Text chunks of the collection with given ids.
        :param collection: Collection to search in
        :param chunk_ids: ids of the text chunks to find
        :return: QuerySet of DocumentPageText
        """
        return DocumentPageText.objects.filter(
            pk__in=[int(pk) for pk in chunk_ids],
            page__document__collection=collection,
        )

    @staticmethod
    def chunks_to_delete(chunk_ids) -> List[int]:
        """
        Expand the ids of the chunks to delete with the chunks referencing
        them through "paged_text" (chunks of a merged document point back to
        the chunks of a single page), which are protected against removal.
        :param chunk_ids: ids of the text chunks to delete
        :return: Sorted list of all chunk ids that have to be removed
        """
        all_chunk_ids = {int(chunk_id) for chunk_id in chunk_ids}
        current_chunk_ids = list(all_chunk_ids)
        while len(current_chunk_ids):
            current_chunk_ids = [
                int(chunk_id)
                for chunk_id in DocumentPageText.objects.filter(
                    paged_text__in=current_chunk_ids
                ).values_list("pk", flat=True)
                if chunk_id not in all_chunk_ids
            ]
            all_chunk_ids.update(current_chunk_ids)
        return sorted(all_chunk_ids)

    @staticmethod
    def delete_chunks(chunk_ids) -> Dict[str, int]:
        """
        Delete text chunks (and pages left without any chunk) from the db.
        :param chunk_ids: ids of the text chunks to delete
        :return: Number of deleted texts, pages and documents
        """
        deleted = {"texts": 0, "pages": 0, "documents": 0}
        chunk_ids = RelationalDBController.chunks_to_delete(chunk_ids)
        if not chunk_ids:
            return deleted

        with transaction.atomic():
            page_ids = list(
                DocumentPageText.objects.filter(pk__in=chunk_ids)
                .values_list("page_id", flat=True)
                .distinct()
            )
            result = DocumentPageText.objects.filter(pk__in=chunk_ids).delete()
            deleted["texts"] += _deleted_rows(result, DocumentPageText)

            result = DocumentPage.objects.filter(
                pk__in=page_ids, documentpagetext__isnull=True
            ).delete()
            deleted["pages"] += _deleted_rows(result, DocumentPage)
        return deleted

    @staticmethod
    def delete_documents(documents) -> Dict[str, int]:
        """
        Delete documents with their pages and text chunks from the db.
        :param documents: Documents (iterable of Document) to delete
        :return: Number of deleted texts, pages and documents
        """
        deleted = {"texts": 0, "pages": 0, "documents": 0}
        document_ids = [document.pk for document in documents]
        if not document_ids:
            return deleted

        with transaction.atomic():
            result = DocumentPageText.objects.filter(
                page__document__in=document_ids
            ).delete()
            deleted["texts"] += _deleted_rows(result, DocumentPageText)

            result = DocumentPage.objects.filter(document__in=document_ids).delete()
            deleted["pages"] += _deleted_rows(result, DocumentPage)

            result = Document.objects.filter(pk__in=document_ids).delete()
            deleted["documents"] += _deleted_rows(result, Document)
        return deleted

    @staticmethod
    def delete_collection(collection: CollectionOfDocuments) -> Dict[str, int]:
        """
        Delete the collection with all its documents, pages and text chunks.
        Chats and search history referencing the collection are removed too,
        because they are protected against cascading deletion.
        :param collection: Collection to delete
        :return: Number of deleted texts, pages, documents, chats, queries
        and collection
        """
        with transaction.atomic():
            deleted = RelationalDBController.delete_documents(
                Document.objects.filter(collection=collection)
            )
            deleted["chats"] = 0
            deleted["queries"] = 0
            deleted["collection"] = 0

            deleted["chats"] = _delete_collection_chats(collection)
            deleted["queries"] = _delete_collection_queries(collection)

            result = CollectionOfDocuments.objects.filter(pk=collection.pk).delete()
            deleted["collection"] += _deleted_rows(result, CollectionOfDocuments)
        return deleted


class PublicRelationDBController:

    @staticmethod
    def get_collection(collection_name: str) -> CollectionOfDocuments | None:
        filter_opts = {
            "name": collection_name,
        }
        collections = CollectionOfDocuments.objects.filter(**filter_opts)
        if len(collections) > 0:
            return collections[0]
        return None

    @staticmethod
    def get_collection_by_id(collection_id) -> CollectionOfDocuments | None:
        filter_opts = {
            "pk": collection_id,
        }
        collections = CollectionOfDocuments.objects.filter(**filter_opts)
        if len(collections) > 0:
            return collections[0]
        return None

    @staticmethod
    def get_collection_chunks(
        collection: CollectionOfDocuments, min_chunk_char_len: int = 300
    ) -> List[DocumentPageText]:
        filtered_pages = []
        all_doc_pages = DocumentPageText.objects.filter(
            page__document__collection=collection
        )
        for d_page in all_doc_pages:
            if d_page.text_str is None or len(d_page.text_str) < min_chunk_char_len:
                continue
            filtered_pages.append(d_page)
        return filtered_pages


# ----------------------------------------------------------------------------
# Deletion helpers
# ----------------------------------------------------------------------------


def _deleted_rows(delete_result, model) -> int:
    """
    Number of rows of the model deleted by QuerySet.delete().
    :param delete_result: Result of QuerySet.delete()
    :param model: Model which rows were deleted
    :return: Number of deleted rows
    """
    # QuerySet.delete() returns (number_of_deleted_rows, counts_by_model)
    if isinstance(delete_result, tuple) and len(delete_result) == 2:
        delete_result = delete_result[1]
    if not isinstance(delete_result, dict):
        return 0
    # The counts are keyed by "<app>.<Model>".
    for label in (model._meta.label, model.__name__):
        if label in delete_result:
            return int(delete_result[label])
    return 0


def _delete_collection_chats(collection: CollectionOfDocuments) -> int:
    """
    Delete chats referencing the collection. Models of other apps are
    resolved dynamically to avoid a circular import (chat imports data).
    :param collection: Collection referenced by the chats
    :return: Number of deleted chats
    """
    chat_model = apps.get_model("chat", "Chat")
    result = chat_model.objects.filter(collection=collection).delete()
    return _deleted_rows(result, chat_model)


def _delete_collection_queries(collection: CollectionOfDocuments) -> int:
    """
    Delete search history (answers, responses and queries) referencing the
    collection, in the order which respects the protected foreign keys.
    :param collection: Collection referenced by the search history
    :return: Number of deleted user queries
    """
    user_query_model = apps.get_model("engine", "UserQuery")
    user_response_model = apps.get_model("engine", "UserQueryResponse")
    user_answer_model = apps.get_model("engine", "UserQueryResponseAnswer")

    user_answer_model.objects.filter(
        user_response__user_query__collection=collection
    ).delete()
    user_response_model.objects.filter(user_query__collection=collection).delete()
    result = user_query_model.objects.filter(collection=collection).delete()
    return _deleted_rows(result, user_query_model)

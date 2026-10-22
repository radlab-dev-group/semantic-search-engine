from rest_framework.views import APIView

from sse_api.engine.controllers.database.milvus import INDEX_QUERY_PARAMS

from sse_api.core.decorators import required_params_exists, get_default_language
from sse_api.core.response import response_with_status
from sse_api.core.validation import (
    request_params,
    required_text,
    optional_text,
    identifier,
    identifiers,
    text_list,
    boolean,
    chunking,
    indexing_options,
    texts,
)
from sse_api.core.errors import error_response
from sse_api.core.errors_list import INVALID_PARAMS


from sse_api.system.core.errors import GROUP_NAME_NOT_EXIST
from sse_api.system.core.decorators import get_organisation_user
from sse_api.system.controllers import SystemController

from sse_api.data.serializers import (
    CollectionOfDocumentsSerializer,
    UploadedDocumentsSerializer,
    SimpleDocumentSerializer,
    DocumentVisibilitySerializer,
    SimpleQueryTemplateSerializer,
)
from sse_api.data import indexing_queue
from sse_api.data.core import indexing_progress
from sse_api.data.core.file_metadata import FileMetadataError, resolve_file_metadata
from sse_api.data.controllers.upload import UploadDocumentsController, UploadRejected
from sse_api.data import indexing_pipeline
from sse_api.data.controllers.deletion import (
    ContentDeletionController,
    ContentProtected,
)
from sse_api.data.controllers.visibility import ContentVisibilityController
from sse_api.data.controllers.preview import ContentPreviewController
from sse_api.data.core.errors import (
    UPLOAD_REJECTED,
    CHUNK_NOT_FOUND,
    DOCUMENT_NOT_FOUND,
    DELETE_PROTECTED,
    CHUNKING_TOO_LONG,
    CHUNKING_MISMATCH,
    INDEXING_JOB_NOT_FOUND,
)
from sse_api.data.core.constants import (
    DEFAULT_MAX_TOKENS_IN_CHUNK,
    DEFAULT_NUMBER_OF_OVERLAP_TOKENS,
    INDEXING_JOB_FILES,
    INDEXING_JOB_TEXTS,
    MAX_STATUS_DOCUMENTS,
    MILVUS_CONFIG_PATH,
    PREVIEW_MAX_PAGES,
)
from sse_api.data.controllers.chunking import (
    ChunkingMismatch,
    ChunkingTooLong,
    InvalidChunking,
    check_chunk_size,
    resolve_chunking,
)


from sse_api.chat.core.errors import COLLECTION_NOT_FOUND

from sse_api.engine.controllers.database.semantic_db import SemanticDBController
from sse_api.engine.controllers.database.relational_db import RelationalDBController

from sse_api.data.models import Document, CollectionOfDocuments, IndexingJob


def resolve_request_chunking(
    collection: CollectionOfDocuments,
    indexing_options: dict,
    language: str,
):
    """
    Resolve the chunking of one indexing call against the collection.

    Returns
    -------
    tuple
        ``(options, None)`` with the options to index with, or
        ``(None, response)`` carrying the error to send back to the client.
    """
    try:
        options = resolve_chunking(collection, indexing_options)
    except ChunkingMismatch:
        return None, error_response(CHUNKING_MISMATCH, language)
    except ChunkingTooLong:
        return None, error_response(CHUNKING_TOO_LONG, language)
    except InvalidChunking:
        return None, error_response(INVALID_PARAMS, language)
    return options, None


def get_owned_collection(
    collection_name: str, organisation_user
) -> CollectionOfDocuments | None:
    """
    Collection created by the user. Deleting is allowed only for the owner of
    the collection, collections shared with the groups of the user are not
    enough.
    :param collection_name: Name of the collection to find
    :param organisation_user: User who is asking for the collection
    :return: CollectionOfDocuments or None
    """
    return RelationalDBController.get_collection(
        collection_name=collection_name,
        created_by=organisation_user,
        owner_only=True,
    )


class NewCollection(APIView):
    required_params = [
        "collection_name",
        "collection_display_name",
        "collection_description",
        "model_embedder",
        "model_reranker",
        "embedder_index_type",
    ]
    optional_params = ["group_name", "chunking", "strict_chunking"]
    input_validators = {key: required_text for key in required_params}
    input_validators["group_name"] = optional_text
    input_validators["chunking"] = chunking
    input_validators["strict_chunking"] = boolean

    @required_params_exists(
        required_params=required_params, optional_params=optional_params
    )
    @get_organisation_user
    @get_default_language
    def post(self, language, organisation_user, request, *args, **kwargs):
        params = request_params(request)
        embedder_index_type = params.get("embedder_index_type")
        if embedder_index_type not in INDEX_QUERY_PARAMS:
            return error_response(INVALID_PARAMS, language)
        embedder_index_params = INDEX_QUERY_PARAMS[embedder_index_type][
            "INDEX_PARAMS"
        ]
        embedder_search_params = INDEX_QUERY_PARAMS[embedder_index_type][
            "QUERY_PARAMS"
        ]

        collection_name = params.get("collection_name").replace(" ", "_")

        org_group = None
        group_name = params.get("group_name") or ""
        if len(group_name.strip()):
            org_group = SystemController.get_organisation_group(
                organisation=organisation_user.organisation, group_name=group_name
            )
            if org_group is None:
                return response_with_status(
                    status=False,
                    language=language,
                    error_name=GROUP_NAME_NOT_EXIST,
                )

        # The embedder of the new collection has to fit the declared chunks;
        # a chunk which would be cut off by the model is refused here, where
        # the choice of the model is made.
        collection_draft = CollectionOfDocuments()
        if "chunking" in params:
            try:
                check_chunk_size(
                    params["chunking"]["max_tokens_in_chunk"],
                    embedder_model=params.get("model_embedder"),
                    reranker_model=params.get("model_reranker"),
                )
            except (ChunkingTooLong, InvalidChunking):
                return error_response(CHUNKING_TOO_LONG, language)
            collection_draft.set_chunking(
                params["chunking"]["max_tokens_in_chunk"],
                params["chunking"]["number_of_overlap_tokens"],
            )
        if params.get("strict_chunking"):
            collection_draft.strict_chunking = True

        text_data_db_controller = RelationalDBController()
        semantic_db_controller = SemanticDBController(
            milvus_config_path="configs/milvus_config.json", prepare_db=False
        )
        collection = text_data_db_controller.get_add_collection(
            collection_name=collection_name,
            created_by=organisation_user,
            collection_display_name=request.data.get("collection_display_name"),
            collection_description=request.data.get("collection_description"),
            model_embedder=request.data.get("model_embedder"),
            model_reranker=request.data.get("model_reranker"),
            embedder_index_type=embedder_index_type,
            embedder_index_params=embedder_index_params,
            embedder_search_params=embedder_search_params,
            add_to_group=org_group,
            metadata_json=collection_draft.metadata_json,
        )

        semantic_db_controller.get_add_collection(
            collection_name=collection_name,
            collection_description=request.data.get("collection_description"),
            model_embedder=request.data.get("model_embedder"),
        )

        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body={
                "collection_id": collection.pk,
                "chunking": collection.chunking_with_defaults(),
                "strict_chunking": collection.strict_chunking,
            },
        )


class ListCollections(APIView):

    @get_organisation_user
    @get_default_language
    def get(self, language, organisation_user, request, *args, **kwargs):
        db_controller = RelationalDBController()
        semantic_db_controller = SemanticDBController(
            milvus_config_path="configs/milvus_config.json", prepare_db=False
        )
        sem_collections = semantic_db_controller.collections()
        # User collections
        user_collections = db_controller.get_user_collections(
            organisation_user, semantic_collections=sem_collections
        )
        # User Organisation collections
        org_collections = db_controller.get_user_organisation_collections(
            organisation_user, semantic_collections=sem_collections
        )

        collections = self.__merge_collections(
            collections_lists=[user_collections, org_collections],
        )

        s_col = CollectionOfDocumentsSerializer(collections, many=True)
        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body={"collections": s_col.data},
        )

    @staticmethod
    def __merge_collections(collections_lists):
        """
        Merge the collections to single flat list.
        :param collections_lists: List of list (or QuerySet) where the single
        list (QuerySet) element is the CollectionOfDocuments object.
        :return: List of CollectionOfDocuments
        """
        collections_ids = []
        merged_collections = []
        for collections in collections_lists:
            for collection in collections:
                if collection is None or collection.pk in collections_ids:
                    continue
                merged_collections.append(collection)
                collections_ids.append(collection.pk)
        return merged_collections


def _uploaded_documents_body(upl_doc) -> dict:
    """
    The upload process as the body of an answer, ready for extra keys.

    ``indexing_job`` (and ``async`` when the work was only queued) is added on
    top of it, so that one client handles the queue and the request-mode
    indexing the same way.
    """
    return dict(UploadedDocumentsSerializer(upl_doc).data)


class UploadAndIndexFilesToCollection(APIView):
    required_params = ["files[]", "collection_name", "indexing_options"]
    input_validators = {
        "collection_name": required_text,
        "indexing_options": indexing_options,
    }

    @required_params_exists(required_params=required_params)
    @get_organisation_user
    @get_default_language
    def post(self, language, organisation_user, request, *args, **kwargs):
        files = request.FILES.getlist("files[]")
        collection_name = request.data.get("collection_name")
        indexing_options = request_params(request).get("indexing_options")
        if not files:
            return error_response(INVALID_PARAMS, language)

        try:
            file_metadata = resolve_file_metadata(
                request.data.get("file_metadata"), [file.name for file in files]
            )
        except (TypeError, ValueError):
            return error_response(UPLOAD_REJECTED, language)

        collection = RelationalDBController.get_collection(
            collection_name=collection_name, created_by=organisation_user
        )
        if collection is None:
            return response_with_status(
                status=False,
                language=language,
                error_name=COLLECTION_NOT_FOUND,
                response_body=None,
            )

        indexing_options, error = resolve_request_chunking(
            collection=collection,
            indexing_options=indexing_options,
            language=language,
        )
        if error is not None:
            return error

        upl_controller = UploadDocumentsController(upload_dir="./upload_sse")
        job = indexing_progress.create_job(
            collection=collection,
            organisation_user=organisation_user,
            kind=INDEXING_JOB_FILES,
            indexing_options=indexing_options,
            request_payload=(
                {"file_metadata": file_metadata} if file_metadata is not None else None
            ),
        )

        if indexing_queue.async_indexing():
            try:
                upl_doc = upl_controller.store_uploaded_files(
                    organisation_user=organisation_user,
                    files=files,
                    collection=collection,
                )
            except UploadRejected:
                indexing_progress.fail_job(job, "upload rejected")
                return error_response(UPLOAD_REJECTED, language)

            job.upload_process = upl_doc
            job.save(update_fields=["upload_process"])
            indexing_queue.dispatch_indexing_job(job)

            response_body = _uploaded_documents_body(upl_doc)
            response_body["async"] = True
            response_body["indexing_job"] = indexing_progress.job_payload(job)
            return response_with_status(
                status=True,
                language=language,
                error_name=None,
                response_body=response_body,
            )

        try:
            upl_doc = upl_controller.store_and_index_files_rel_db_post_request(
                organisation_user=organisation_user,
                files=files,
                collection=collection,
                indexing_options=indexing_options,
                semantic_config_path=MILVUS_CONFIG_PATH,
                indexing_job=job,
            )
        except (UploadRejected, FileMetadataError):
            indexing_progress.fail_job(job, "upload rejected")
            return error_response(UPLOAD_REJECTED, language)

        job.refresh_from_db()
        response_body = _uploaded_documents_body(upl_doc)
        response_body["indexing_job"] = indexing_progress.job_payload(job)
        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body=response_body,
        )


class AddAndIndexTextsFromEP(APIView):
    required_params = ["texts[]", "collection_name", "indexing_options"]
    input_validators = {
        "collection_name": required_text,
        "indexing_options": indexing_options,
        "texts[]": texts,
    }

    @required_params_exists(required_params=required_params)
    @get_organisation_user
    @get_default_language
    def post(self, language, organisation_user, request, *args, **kwargs):

        texts = request.data.get("texts[]")
        collection_name = request.data.get("collection_name")
        indexing_options = request_params(request).get("indexing_options")

        rel_db_controller = RelationalDBController(store_to_db=True)
        collection = rel_db_controller.get_collection(
            collection_name=collection_name, created_by=organisation_user
        )
        if collection is None:
            return response_with_status(
                status=False,
                language=language,
                error_name=COLLECTION_NOT_FOUND,
                response_body=None,
            )

        indexing_options, error = resolve_request_chunking(
            collection=collection,
            indexing_options=indexing_options,
            language=language,
        )
        if error is not None:
            return error

        job = indexing_progress.create_job(
            collection=collection,
            organisation_user=organisation_user,
            kind=INDEXING_JOB_TEXTS,
            indexing_options=indexing_options,
            request_payload={"texts": texts},
        )

        if indexing_queue.async_indexing():
            indexing_queue.dispatch_indexing_job(job)
            return response_with_status(
                status=True,
                language=language,
                error_name=None,
                response_body={
                    "async": True,
                    "indexing_job": indexing_progress.job_payload(job),
                },
            )

        indexing_pipeline.run_job(job, jsonl_config_path=MILVUS_CONFIG_PATH)
        job.refresh_from_db()

        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body={
                "indexed_documents": job.documents.count(),
                "indexed_chunks": indexing_pipeline.count_job_chunks(job),
                "indexing_job": indexing_progress.job_payload(job),
            },
        )


class ListCategoriesFromCollection(APIView):
    required_params = ["collection_name"]
    input_validators = {"collection_name": required_text}

    @required_params_exists(required_params=required_params)
    @get_organisation_user
    @get_default_language
    def get(self, language, organisation_user, request):
        collection_name = request_params(request).get("collection_name")
        collection = RelationalDBController.get_collection(
            collection_name=collection_name, created_by=organisation_user
        )
        if collection is None:
            return response_with_status(
                status=False,
                language=language,
                error_name=COLLECTION_NOT_FOUND,
                response_body=None,
            )

        categories = RelationalDBController.get_all_categories_from_collection(
            collection=collection
        )

        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body={"categories": categories},
        )


class ListDocumentsFromCollection(APIView):
    """
    Documents of the collection used in the search.  With
    ``with_disabled=true`` the documents excluded from the search
    (``use_in_search=False``) are listed as well, each one with its
    ``id`` and ``use_in_search`` flag (management of the collection).
    """

    required_params = ["collection_name"]
    optional_params = ["with_disabled"]
    input_validators = {"collection_name": required_text, "with_disabled": boolean}

    @required_params_exists(
        required_params=required_params, optional_params=optional_params
    )
    @get_organisation_user
    @get_default_language
    def get(self, language, organisation_user, request):
        params = request_params(request)
        collection_name = params.get("collection_name")
        collection = RelationalDBController.get_collection(
            collection_name=collection_name, created_by=organisation_user
        )
        if collection is None:
            return response_with_status(
                status=False, language=language, error_name=COLLECTION_NOT_FOUND
            )
        if params.get("with_disabled", False):
            documents = RelationalDBController.get_all_documents_from_collection(
                collection=collection,
            ).order_by("name")
            serialized = DocumentVisibilitySerializer(documents, many=True).data
        else:
            documents = (
                RelationalDBController.get_documents_to_search_from_collection(
                    collection=collection,
                )
            )
            serialized = SimpleDocumentSerializer(documents, many=True).data
        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body={"documents": serialized},
        )


class GetDocumentContent(APIView):
    """
    Whole text of one document of a collection (``GET document_content``),
    which is what the preview of a row of the documents tab shows.

    The text is read from the relational database — the chunks the document
    was indexed with, page by page — so neither the original file nor the
    vector database is needed.  A document longer than the window of pages
    (``max_pages``, default 200) is answered with ``truncated`` and the
    ``next_page`` to ask for.
    """

    required_params = ["collection_name"]
    optional_params = [
        "document_name",
        "document_id",
        "chunk_type",
        "from_page",
        "max_pages",
    ]
    input_validators = {
        "collection_name": required_text,
        "document_name": optional_text,
        "document_id": identifier,
        "chunk_type": optional_text,
        "from_page": identifier,
        "max_pages": identifier,
    }

    @required_params_exists(
        required_params=required_params, optional_params=optional_params
    )
    @get_organisation_user
    @get_default_language
    def get(self, language, organisation_user, request):
        params = request_params(request)
        collection = RelationalDBController.get_collection(
            collection_name=params.get("collection_name"),
            created_by=organisation_user,
        )
        if collection is None:
            return response_with_status(
                status=False, language=language, error_name=COLLECTION_NOT_FOUND
            )

        document_id = params.get("document_id")
        document_name = params.get("document_name")
        if document_id is None and not str(document_name or "").strip():
            return error_response(INVALID_PARAMS, language)

        controller = ContentPreviewController()
        document = controller.find_document(
            collection=collection,
            document_id=document_id,
            document_name=document_name,
        )
        if document is None:
            return response_with_status(
                status=False, language=language, error_name=DOCUMENT_NOT_FOUND
            )

        preview = controller.document_content(
            document=document,
            from_page=params.get("from_page") or 1,
            max_pages=params.get("max_pages") or PREVIEW_MAX_PAGES,
            chunk_type=params.get("chunk_type"),
        )
        return response_with_status(
            status=True, language=language, response_body=preview.as_dict()
        )


class IndexingStatus(APIView):
    """
    Progress of the indexing: how much is done and what is still waiting.

    ``job_id`` reports one indexing request, ``collection_name`` the newest
    request of the collection together with the documents of the collection
    which are still waiting for their embeddings.  The documents of the job are
    listed with their own status, so a client can show which document is already
    indexed and how many are left.
    """

    required_params = []
    optional_params = ["job_id", "collection_name"]
    input_validators = {"job_id": identifier, "collection_name": required_text}

    @required_params_exists(
        required_params=required_params, optional_params=optional_params
    )
    @get_organisation_user
    @get_default_language
    def get(self, language, organisation_user, request):
        params = request_params(request)
        job_id = params.get("job_id")
        collection_name = params.get("collection_name")
        if not job_id and not collection_name:
            return error_response(INVALID_PARAMS, language)

        job = None
        if job_id:
            job = (
                IndexingJob.objects.filter(pk=job_id)
                .select_related("collection")
                .first()
            )
            if job is None:
                return error_response(INDEXING_JOB_NOT_FOUND, language)
            collection = RelationalDBController.get_collection(
                collection_name=job.collection.name, created_by=organisation_user
            )
            if collection is None or collection.pk != job.collection_id:
                return error_response(INDEXING_JOB_NOT_FOUND, language)
        else:
            collection = RelationalDBController.get_collection(
                collection_name=collection_name, created_by=organisation_user
            )
            if collection is None:
                return response_with_status(
                    status=False,
                    language=language,
                    error_name=COLLECTION_NOT_FOUND,
                    response_body=None,
                )
            job = indexing_progress.latest_job(collection)

        documents = Document.objects.none()
        if job is not None:
            documents = job.documents.order_by("pk")[:MAX_STATUS_DOCUMENTS]
        elif collection is not None:
            documents = Document.objects.filter(
                collection=collection,
                indexing_status__in=indexing_progress.RUNNING_DOCUMENT_STATUSES,
            ).order_by("pk")[:MAX_STATUS_DOCUMENTS]

        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body={
                "job": indexing_progress.job_payload(job),
                "documents": [
                    {
                        "id": document.pk,
                        "name": document.name,
                        "indexing_status": document.indexing_status,
                    }
                    for document in documents
                ],
                "pending_documents": indexing_progress.pending_documents(collection),
            },
        )


class ListQuestionTemplates(APIView):

    @get_organisation_user
    @get_default_language
    def get(self, language, organisation_user, request):
        org_templates = RelationalDBController.get_organisation_templates(
            organisation=organisation_user.organisation
        )

        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body={
                "templates": self.prepare_ep_out(query_templates=org_templates)
            },
        )

    @staticmethod
    def prepare_ep_out(query_templates):
        qt_dict = {}
        for qt in query_templates:
            qt_name = qt.template_grammar.template_collection.name
            if qt_name not in qt_dict:
                qt_dict[qt_name] = []
            qt_dict[qt_name].append(
                SimpleQueryTemplateSerializer(qt, many=False).data
            )
        return qt_dict


class ListFilteringOptions(APIView):
    @get_organisation_user
    @get_default_language
    def get(self, language, organisation_user, request):
        org_templates = RelationalDBController.get_organisation_templates(
            organisation=organisation_user.organisation
        )

        # URLs are not modelled in this system yet — use empty list.
        # Categories come from visible documents (use_in_search=True).
        collections = RelationalDBController.get_visible_collections(
            organisation_user
        )
        categories = list(
            Document.objects.filter(
                collection__in=collections,
                use_in_search=True,
                category__isnull=False,
            )
            .values_list("category", flat=True)
            .distinct()
        )

        templates = ListQuestionTemplates.prepare_ep_out(
            query_templates=org_templates
        )

        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body={
                "templates": templates,
                "urls": [],
                "categories": categories,
                "chunking_defaults": {
                    "max_tokens_in_chunk": DEFAULT_MAX_TOKENS_IN_CHUNK,
                    "number_of_overlap_tokens": DEFAULT_NUMBER_OF_OVERLAP_TOKENS,
                },
            },
        )


class DeleteTextsFromCollection(APIView):
    """
    Delete text chunks of the collection from the relational database and
    their embeddings from Milvus.
    """

    required_params = ["collection_name", "chunk_ids[]"]
    input_validators = {"collection_name": required_text, "chunk_ids[]": identifiers}

    @required_params_exists(required_params=required_params)
    @get_organisation_user
    @get_default_language
    def post(self, language, organisation_user, request, *args, **kwargs):
        params = request_params(request)
        collection = get_owned_collection(
            collection_name=params.get("collection_name"),
            organisation_user=organisation_user,
        )
        if collection is None:
            return response_with_status(
                status=False, language=language, error_name=COLLECTION_NOT_FOUND
            )

        controller = ContentDeletionController(milvus_config_path=MILVUS_CONFIG_PATH)
        try:
            result = controller.delete_texts(
                collection=collection, chunk_ids=params.get("chunk_ids[]")
            )
        except ContentProtected:
            return response_with_status(
                status=False, language=language, error_name=DELETE_PROTECTED
            )

        if not result.deleted.get("texts"):
            return response_with_status(
                status=False, language=language, error_name=CHUNK_NOT_FOUND
            )

        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body=result.as_dict(),
        )

    delete = post


class DeleteDocumentsFromCollection(APIView):
    """
    Delete documents, with their pages and text chunks, from the relational
    database and their embeddings from Milvus. Documents may be given by
    names (``documents[]``) and/or by ids (``document_ids[]``).
    """

    required_params = ["collection_name"]
    optional_params = ["documents[]", "document_ids[]"]
    input_validators = {
        "collection_name": required_text,
        "documents[]": text_list,
        "document_ids[]": identifiers,
    }

    @required_params_exists(
        required_params=required_params, optional_params=optional_params
    )
    @get_organisation_user
    @get_default_language
    def post(self, language, organisation_user, request, *args, **kwargs):
        params = request_params(request)
        collection = get_owned_collection(
            collection_name=params.get("collection_name"),
            organisation_user=organisation_user,
        )
        if collection is None:
            return response_with_status(
                status=False, language=language, error_name=COLLECTION_NOT_FOUND
            )

        document_names = params.get("documents[]")
        document_ids = params.get("document_ids[]")
        if not len(document_names or []) and not len(document_ids or []):
            return error_response(INVALID_PARAMS, language)

        controller = ContentDeletionController(milvus_config_path=MILVUS_CONFIG_PATH)
        try:
            result = controller.delete_documents(
                collection=collection,
                document_names=document_names,
                document_ids=document_ids,
            )
        except ContentProtected:
            return response_with_status(
                status=False, language=language, error_name=DELETE_PROTECTED
            )

        if not result.deleted.get("documents"):
            return response_with_status(
                status=False,
                language=language,
                error_name=DOCUMENT_NOT_FOUND,
            )

        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body=result.as_dict(),
        )

    delete = post


class DeleteCollection(APIView):
    """
    Delete the whole collection: its documents, pages and text chunks from
    the relational database, and the collection itself from Milvus.
    """

    required_params = ["collection_name"]
    input_validators = {"collection_name": required_text}

    @required_params_exists(required_params=required_params)
    @get_organisation_user
    @get_default_language
    def post(self, language, organisation_user, request, *args, **kwargs):
        params = request_params(request)
        collection = get_owned_collection(
            collection_name=params.get("collection_name"),
            organisation_user=organisation_user,
        )
        if collection is None:
            return response_with_status(
                status=False, language=language, error_name=COLLECTION_NOT_FOUND
            )

        controller = ContentDeletionController(milvus_config_path=MILVUS_CONFIG_PATH)
        try:
            result = controller.delete_collection(collection=collection)
        except ContentProtected:
            return response_with_status(
                status=False, language=language, error_name=DELETE_PROTECTED
            )

        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body=result.as_dict(),
        )

    delete = post


class SetDocumentsUseInSearch(APIView):
    """
    Enable or disable the use of documents in the search of the collection:
    sets ``Document.use_in_search`` on the documents given by names
    (``documents[]``) and/or by ids (``document_ids[]``).  Nothing is removed
    — the pages, text chunks and the embeddings of a disabled document stay
    in both databases, so the document may be enabled again without any
    re-indexing.
    """

    required_params = ["collection_name"]
    optional_params = ["documents[]", "document_ids[]", "use_in_search"]
    input_validators = {
        "collection_name": required_text,
        "documents[]": text_list,
        "document_ids[]": identifiers,
        "use_in_search": boolean,
    }

    @required_params_exists(
        required_params=required_params, optional_params=optional_params
    )
    @get_organisation_user
    @get_default_language
    def post(self, language, organisation_user, request, *args, **kwargs):
        params = request_params(request)
        collection = get_owned_collection(
            collection_name=params.get("collection_name"),
            organisation_user=organisation_user,
        )
        if collection is None:
            return response_with_status(
                status=False, language=language, error_name=COLLECTION_NOT_FOUND
            )

        document_names = params.get("documents[]")
        document_ids = params.get("document_ids[]")
        if not len(document_names or []) and not len(document_ids or []):
            return error_response(INVALID_PARAMS, language)

        use_in_search = params.get("use_in_search", False)

        controller = ContentVisibilityController(
            milvus_config_path=MILVUS_CONFIG_PATH
        )
        result = controller.set_documents_use_in_search(
            collection=collection,
            use_in_search=use_in_search,
            document_names=document_names,
            document_ids=document_ids,
        )

        if not result.updated_documents:
            return response_with_status(
                status=False,
                language=language,
                error_name=DOCUMENT_NOT_FOUND,
            )

        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body=result.as_dict(),
        )

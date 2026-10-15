from rest_framework.views import APIView

from sse_api.engine.controllers.database.milvus import INDEX_QUERY_PARAMS

from sse_api.core.decorators import required_params_exists, get_default_language
from sse_api.core.response import response_with_status
from sse_api.core.validation import (
    request_params,
    required_text,
    optional_text,
    identifiers,
    text_list,
    boolean,
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
from sse_api.data.controllers.upload import UploadDocumentsController, UploadRejected
from sse_api.data.controllers.deletion import (
    ContentDeletionController,
    ContentProtected,
)
from sse_api.data.controllers.visibility import ContentVisibilityController
from sse_api.data.core.errors import (
    UPLOAD_REJECTED,
    CHUNK_NOT_FOUND,
    DOCUMENT_NOT_FOUND,
    DELETE_PROTECTED,
)


from sse_api.chat.core.errors import COLLECTION_NOT_FOUND

from sse_api.engine.controllers.search.semantic import DBSemanticSearchController
from sse_api.engine.controllers.database.semantic_db import SemanticDBController
from sse_api.engine.controllers.database.relational_db import RelationalDBController

from sse_api.data.models import Document, CollectionOfDocuments

# Connection configuration of the vector database (Milvus).
MILVUS_CONFIG_PATH = "./configs/milvus_config.json"


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
    optional_params = ["group_name"]
    input_validators = {key: required_text for key in required_params}
    input_validators["group_name"] = optional_text

    @required_params_exists(
        required_params=required_params, optional_params=optional_params
    )
    @get_organisation_user
    @get_default_language
    def post(self, language, organisation_user, request, *args, **kwargs):
        embedder_index_type = request.data.get("embedder_index_type")
        if embedder_index_type not in INDEX_QUERY_PARAMS:
            return error_response(INVALID_PARAMS, language)
        embedder_index_params = INDEX_QUERY_PARAMS[embedder_index_type][
            "INDEX_PARAMS"
        ]
        embedder_search_params = INDEX_QUERY_PARAMS[embedder_index_type][
            "QUERY_PARAMS"
        ]

        collection_name = request.data.get("collection_name").replace(" ", "_")

        org_group = None
        group_name = request.data.get("group_name") or ""
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
            response_body={"collection_id": collection.pk},
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

        upl_controller = UploadDocumentsController(upload_dir="./upload_sse")
        try:
            upl_doc = upl_controller.store_and_index_files_rel_db_post_request(
                organisation_user=organisation_user,
                files=files,
                collection=collection,
                indexing_options=indexing_options,
                semantic_config_path="./configs/milvus_config.json",
            )
        except UploadRejected:
            return error_response(UPLOAD_REJECTED, language)

        upl_doc_serialized = UploadedDocumentsSerializer(upl_doc)

        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body=upl_doc_serialized.data,
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

        results_doc = rel_db_controller.add_documents_from_list_of_doc_as_dict(
            collection=collection,
            texts=texts,
            organisation_user=organisation_user,
            indexing_options=indexing_options,
        )
        l_r_doc = len(results_doc)

        if l_r_doc:
            sem_db_controller = DBSemanticSearchController(
                collection_name=collection.name,
                index_name=collection.embedder_index_type,
                batch_size=100,
                embedder_model=collection.model_embedder,
                cross_encoder_model=collection.model_reranker,
                jsonl_config_path="./configs/milvus_config.json",
            )
            for r_doc in results_doc:
                pages_chunks = r_doc["pages_chunks"]
                sem_db_controller.index_texts_from_list(
                    all_texts=pages_chunks, collection=collection
                )

        l_i_chunks = (
            sum(len(pc["pages_chunks"]) for pc in results_doc) if l_r_doc else 0
        )

        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body={
                "indexed_documents": l_r_doc,
                "indexed_chunks": l_i_chunks,
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

        controller = ContentVisibilityController()
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

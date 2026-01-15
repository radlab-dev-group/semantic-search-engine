import json
from rest_framework.views import APIView

from engine.controllers.milvus import INDEX_QUERY_PARAMS

from main.src.decorators import required_params_exists, get_default_language
from main.src.response import response_with_status

from chat.core.errors import COLLECTION_NOT_FOUND
from system.core.errors import GROUP_NAME_NOT_EXIST
from system.core.decorators import get_organisation_user

from data.serializers import (
    CollectionOfDocumentsSerializer,
    UploadedDocumentsSerializer,
    SimpleDocumentSerializer,
    SimpleQueryTemplateSerializer,
)

from system.controllers import SystemController
from data.controllers.upload import UploadDocumentsController
from engine.controllers.semantic_db import SemanticDBController
from engine.controllers.relational_db import RelationalDBController
from engine.controllers.search import DBSemanticSearchController


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

    text_data_db_controller = RelationalDBController()
    semantic_db_controller = SemanticDBController(
        milvus_config_path="configs/milvus_config.json", prepare_db=False
    )

    @required_params_exists(
        required_params=required_params, optional_params=optional_params
    )
    @get_organisation_user
    @get_default_language
    def post(self, language, organisation_user, request, *args, **kwargs):
        embedder_index_type = request.data.get("embedder_index_type")
        embedder_index_params = INDEX_QUERY_PARAMS[embedder_index_type][
            "INDEX_PARAMS"
        ]
        embedder_search_params = INDEX_QUERY_PARAMS[embedder_index_type][
            "QUERY_PARAMS"
        ]

        collection_name = request.data.get("collection_name").replace(" ", "_")

        org_group = None
        group_name = request.data.get("group_name", "")
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

        collection = self.text_data_db_controller.get_add_collection(
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

        self.semantic_db_controller.get_add_collection(
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
    db_controller = RelationalDBController()
    semantic_db_controller = SemanticDBController(
        milvus_config_path="configs/milvus_config.json", prepare_db=False
    )

    @get_organisation_user
    @get_default_language
    def get(self, language, organisation_user, request, *args, **kwargs):
        sem_collections = self.semantic_db_controller.collections()
        # User collections
        user_collections = self.db_controller.get_user_collections(
            organisation_user, semantic_collections=sem_collections
        )
        # User Organisation collections
        org_collections = self.db_controller.get_user_organisation_collections(
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

    upl_controller = UploadDocumentsController(upload_dir="./upload_sse")

    @required_params_exists(required_params=required_params)
    @get_organisation_user
    @get_default_language
    def post(self, language, organisation_user, request, *args, **kwargs):
        files = request.FILES.getlist("files[]")
        collection_name = request.data.get("collection_name")
        indexing_options = json.loads(request.data.get("indexing_options"))

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

        upl_doc = self.upl_controller.store_and_index_files_rel_db_post_request(
            organisation_user=organisation_user,
            files=files,
            collection=collection,
            indexing_options=indexing_options,
            semantic_config_path="./configs/milvus_config.json",
        )

        upl_doc_serialized = UploadedDocumentsSerializer(upl_doc)

        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body=upl_doc_serialized.data,
        )


class AddAndIndexTextsFromEP(APIView):
    required_params = ["texts[]", "collection_name", "indexing_options"]

    @required_params_exists(required_params=required_params)
    @get_organisation_user
    @get_default_language
    def post(self, language, organisation_user, request, *args, **kwargs):

        texts = request.data.get("texts[]")
        collection_name = request.data.get("collection_name")
        indexing_options = request.data.get("indexing_options")

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

    @required_params_exists(required_params=required_params)
    @get_organisation_user
    @get_default_language
    def get(self, language, organisation_user, request):
        pass
        collection_name = request.data.get("collection_name")
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
    required_params = ["collection_name"]

    @required_params_exists(required_params=required_params)
    @get_organisation_user
    @get_default_language
    def get(self, language, organisation_user, request):
        collection_name = request.data.get("collection_name")
        collection = RelationalDBController.get_collection(
            collection_name=collection_name, created_by=organisation_user
        )
        documents = RelationalDBController.get_documents_to_search_from_collection(
            collection=collection,
        )
        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body={
                "documents": SimpleDocumentSerializer(documents, many=True).data
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

        mock_urls = [
            {"id": 0, "url": "https://cam.waw.pl", "description": "Strona CAM"},
            {
                "id": 1,
                "url": "https://waw4free.pl",
                "description": "Strona waw4free",
            },
        ]

        mock_categories = [
            {
                "id": 0,
                "name": "Sport",
                "description": "Wydarzenia z kategorii sport",
            },
            {
                "id": 1,
                "name": "Kultura",
                "description": "Wydarzenia z kategorii kultura",
            },
            {
                "id": 2,
                "name": "Edukacja",
                "description": "Wydarzenia z kategorii edukacja",
            },
        ]

        templates = ListQuestionTemplates.prepare_ep_out(
            query_templates=org_templates
        )

        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body={
                "templates": templates,
                "urls": mock_urls,
                "categories": mock_categories,
            },
        )

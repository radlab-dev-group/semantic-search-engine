import json
import os

from rest_framework.response import Response
from rest_framework.views import APIView

from main.src.response import response_with_status
from main.src.decorators import required_params_exists, get_default_language
from main.src.constants import CONFIG_DIR

from system.core.decorators import get_organisation_user
from engine.controllers.search.query import SearchQueryController
from engine.controllers.database.relational_db import RelationalDBController
from engine.controllers.system_logic.system import EngineSystemController
from engine.controllers.models_logic.generative import (
    GenerativeModelController,
    GenerativeModelControllerApi,
)
from engine.controllers.models_logic.embedders_rerankers import EmbeddingModelsConfig


class SearchWithOptions(APIView):
    """
    sample_options = {
        "categories": [],
        "documents_names": [],
        "max_results": 50,
        "rerank_results": False,
        "return_with_factored_fields": False,
    }
    """

    required_params = ["collection_name", "query_str", "options"]
    optional_params = ["ignore_question_lang_detect"]

    @required_params_exists(
        required_params=required_params, optional_params=optional_params
    )
    @get_organisation_user
    @get_default_language
    def post(self, language, organisation_user, request):
        query_str = request.data.get("query_str")
        collection_name = request.data.get("collection_name")
        options_dict = json.loads(request.data.get("options"))

        ignore_question_lang_detect = bool(
            request.data.get("ignore_question_lang_detect", False)
        )

        collection = RelationalDBController.get_collection(
            collection_name=collection_name, created_by=organisation_user
        )
        if collection is None:
            return response_with_status(
                status=False,
                language=language,
                error_name="Collection not found or access denied!",
                response_body={},
            )

        sse_engin_config_path = os.path.join(CONFIG_DIR, "milvus_config.json")
        results = SearchQueryController.new_query(
            query_str=query_str,
            search_options_dict=options_dict,
            collection=collection,
            organisation_user=organisation_user,
            sse_engin_config_path=sse_engin_config_path,
            ignore_question_lang_detect=ignore_question_lang_detect,
        )

        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body=results,
        )


class GenerativeAnswerForQuestion(APIView):
    """
    sample_options = {
        "generative_model": "",
        "percentage_rank_mass": 40,
        "answer_language": "",
        "translate_answer": False
    }
    """

    required_params = ["query_response_id", "query_options"]
    optional_params = ["system_prompt"]

    @required_params_exists(required_params=required_params)
    @get_organisation_user
    @get_default_language
    def post(self, language, organisation_user, request):
        query_response_id = request.data.get("query_response_id")
        query_options = json.loads(request.data.get("query_options"))

        query_instruction = request.data.get("query_instruction", "")

        system_prompt = request.data.get("system_prompt", None)
        if system_prompt is not None and len(system_prompt.strip()):
            system_prompt = system_prompt.strip()
        else:
            system_prompt = None

        user_response = SearchQueryController.get_user_response_by_id(
            query_response_id=query_response_id, organisation_user=organisation_user
        )

        if user_response is None:
            return response_with_status(
                status=False,
                language=language,
                error_name="Response not found or access denied!",
                response_body={},
            )

        gen_model_controller = GenerativeModelController(store_to_db=True)
        query_response = gen_model_controller.generative_answer_for_response(
            user_response=user_response,
            query_instruction=query_instruction,
            query_options=query_options,
            system_prompt=system_prompt,
        )

        if query_response is None:
            which_key = "DEEPL_AUTH_KEY"
            if "openai" in query_options["generative_model"]:
                which_key = "OPENAI_API_KEY"
            return response_with_status(
                status=False,
                language=language,
                error_name=f"{which_key} is not set or model error occurred!",
                response_body={},
            )

        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body={
                "response_id": query_response.pk,
                "answer": query_response.generated_answer,
                "answer_translated": query_response.generated_answer_translated,
                "generation_time": query_response.generation_time,
            },
        )


class ListGenerativeModels(APIView):
    @get_default_language
    def get(self, language, request):
        gam_controller = GenerativeModelControllerApi(deepl_api_key="")
        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body=gam_controller.models_config.active_local_models_hosts,
        )


class ListEmbeddersModels(APIView):
    @get_default_language
    def get(self, language, request):
        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body={"models": EmbeddingModelsConfig.embedders()},
        )


class ListRerankersModels(APIView):
    @get_default_language
    def get(self, language, request):
        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body={"models": EmbeddingModelsConfig.rerankers()},
        )


class SetRateForQueryResponseAnswer(APIView):
    required_params = ["answer_response_id", "rate_value", "rate_value_max"]
    optional_params = ["rate_comment"]

    @required_params_exists(
        required_params=required_params, optional_params=optional_params
    )
    @get_organisation_user
    @get_default_language
    def post(self, language, organisation_user, request):
        answer_response_id = request.data["answer_response_id"]
        rate_value = request.data["rate_value"]
        rate_value_max = request.data["rate_value_max"]
        comment = request.data.get("rate_comment", None)

        query_response_answer = (
            GenerativeModelController.get_user_query_response_answer(
                user_query_response_id=answer_response_id,
                organisation_user=organisation_user,
            )
        )

        if query_response_answer is None:
            return response_with_status(
                status=False,
                language=language,
                error_name="Answer not found or access denied!",
                response_body={},
            )

        engine_controller = EngineSystemController(store_to_db=True)
        engine_controller.set_rating(
            query_response_answer,
            rating_value=rate_value,
            rating_value_max=rate_value_max,
            comment=comment,
        )

        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body={},
        )

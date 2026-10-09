import os

from rest_framework.views import APIView

from sse_api.core.response import response_with_status
from sse_api.core.decorators import required_params_exists, get_default_language
from sse_api.core.constants import CONFIG_DIR
from sse_api.core.validation import (request_params, required_text, optional_text,
                                     identifier, boolean, search_options, generation_options, rating_integer)
from sse_api.core.errors import error_response
from sse_api.core.errors_list import INVALID_PARAMS
from sse_api.engine.core.errors import (COLLECTION_ACCESS_DENIED, RESPONSE_ACCESS_DENIED,
                                        ANSWER_ACCESS_DENIED, GENERATION_FAILED)

from sse_api.system.core.decorators import get_organisation_user
from sse_api.engine.controllers.search.query import SearchQueryController
from sse_api.engine.controllers.database.relational_db import RelationalDBController
from sse_api.engine.controllers.system_logic.system import EngineSystemController
from sse_api.engine.controllers.models_logic.generative import (
    GenerativeModelController,
    GenerativeModelControllerApi,
)
from sse_api.engine.controllers.models_logic.embedders_rerankers import (
    EmbeddingModelsConfig,
)


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
    input_validators = {"collection_name": required_text, "query_str": required_text,
                        "options": search_options, "ignore_question_lang_detect": boolean}

    @required_params_exists(
        required_params=required_params, optional_params=optional_params
    )
    @get_organisation_user
    @get_default_language
    def post(self, language, organisation_user, request):
        params = request_params(request)
        query_str = params.get("query_str")
        collection_name = params.get("collection_name")
        options_dict = params.get("options")
        ignore_question_lang_detect = params.get("ignore_question_lang_detect", False)

        collection = RelationalDBController.get_collection(
            collection_name=collection_name, created_by=organisation_user
        )
        if collection is None:
            return response_with_status(
                status=False,
                language=language,
                error_name=COLLECTION_ACCESS_DENIED,
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
    input_validators = {"query_response_id": identifier, "query_options": generation_options,
                        "query_instruction": optional_text, "system_prompt": optional_text}

    @required_params_exists(required_params=required_params)
    @get_organisation_user
    @get_default_language
    def post(self, language, organisation_user, request):
        params = request_params(request)
        query_response_id = params.get("query_response_id")
        query_options = params.get("query_options")

        query_instruction = request.data.get("query_instruction") or ""

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
                error_name=RESPONSE_ACCESS_DENIED,
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
            return response_with_status(
                status=False,
                language=language,
                error_name=GENERATION_FAILED,
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
    input_validators = {"answer_response_id": identifier, "rate_value": rating_integer,
                        "rate_value_max": rating_integer, "rate_comment": optional_text}

    @required_params_exists(
        required_params=required_params, optional_params=optional_params
    )
    @get_organisation_user
    @get_default_language
    def post(self, language, organisation_user, request):
        params = request_params(request)
        answer_response_id = params["answer_response_id"]
        rate_value = params["rate_value"]
        rate_value_max = params["rate_value_max"]
        comment = params.get("rate_comment", None)
        if rate_value_max <= 0 or not 0 <= rate_value <= rate_value_max:
            return error_response(INVALID_PARAMS, language)

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
                error_name=ANSWER_ACCESS_DENIED,
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

import json
import logging
import os

from rest_framework.views import APIView

from sse_api.core.response import response_with_status
from sse_api.core.decorators import required_params_exists, get_default_language
from sse_api.core.constants import CONFIG_DIR
from sse_api.core.validation import (
    request_params,
    required_text,
    optional_text,
    conversation_history,
    identifier,
    boolean,
    identifiers,
    search_options,
    generation_options,
    rating_integer,
)
from sse_api.core.errors import error_response
from sse_api.core.errors_list import INVALID_PARAMS
from sse_api.engine.core.errors import (
    COLLECTION_ACCESS_DENIED,
    RESPONSE_ACCESS_DENIED,
    ANSWER_ACCESS_DENIED,
    GENERATION_FAILED,
    QUERY_NOT_FOUND,
    DELETE_PROTECTED,
)

from sse_api.system.core.decorators import get_organisation_user
from sse_api.engine.controllers.search.query import SearchQueryController
from sse_api.engine.controllers.database.relational_db import RelationalDBController
from sse_api.engine.controllers.history.deletion import (
    SearchHistoryDeletionController,
    HistoryDeletionProtected,
)
from sse_api.engine.controllers.system_logic.system import EngineSystemController
from sse_api.engine.controllers.models_logic.generative import (
    GenerativeModelController,
    GenerativeModelControllerApi,
)
from sse_api.engine.controllers.models_logic.embedders_rerankers import (
    EmbeddingModelsConfig,
)
from sse_api.engine.models import UserQuery, UserQueryResponse
from sse_api.engine.controllers.search.semantic import DBSemanticSearchController


def optional_collection_names(value):
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            raise ValueError("Expected list of collection names")
    if not isinstance(value, (list, tuple)):
        raise ValueError("Expected list of collection names")
    parsed = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("Expected nonempty string collection name")
        parsed.append(item.strip())
    return parsed


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
    optional_params = ["collection_names", "ignore_question_lang_detect"]
    input_validators = {
        "collection_name": required_text,
        "collection_names": optional_collection_names,
        "query_str": required_text,
        "options": search_options,
        "ignore_question_lang_detect": boolean,
    }

    @required_params_exists(
        required_params=required_params, optional_params=optional_params
    )
    @get_organisation_user
    @get_default_language
    def post(self, language, organisation_user, request):
        params = request_params(request)
        query_str = params.get("query_str")
        collection_name = params.get("collection_name")
        collection_names_param = params.get("collection_names")
        options_dict = params.get("options")
        ignore_question_lang_detect = params.get(
            "ignore_question_lang_detect", False
        )

        multi_coll_mode = options_dict.get("multi_collection_mode", "global_limit")
        if multi_coll_mode not in ("per_collection", "global_limit"):
            multi_coll_mode = "global_limit"

        if collection_names_param:
            target_collection_names = collection_names_param
        elif collection_name:
            target_collection_names = [collection_name]
        else:
            target_collection_names = []

        if not target_collection_names:
            return response_with_status(
                status=False,
                language=language,
                error_name=COLLECTION_ACCESS_DENIED,
                response_body={},
            )

        collections = []
        for cname in target_collection_names:
            coll = RelationalDBController.get_collection(
                collection_name=cname, created_by=organisation_user
            )
            if coll is None:
                return response_with_status(
                    status=False,
                    language=language,
                    error_name=COLLECTION_ACCESS_DENIED,
                    response_body={},
                )
            collections.append(coll)

        sse_engin_config_path = os.path.join(CONFIG_DIR, "milvus_config.json")

        if len(collections) == 1:
            collection = collections[0]
            if hasattr(collection, "apply_search_defaults"):
                options_dict = collection.apply_search_defaults(options_dict)

            results = SearchQueryController.new_query(
                query_str=query_str,
                search_options_dict=options_dict,
                collection=collection,
                organisation_user=organisation_user,
                sse_engin_config_path=sse_engin_config_path,
                ignore_question_lang_detect=ignore_question_lang_detect,
            )
            if isinstance(results, dict) and "results" in results:
                detailed = results["results"].get("detailed_results", {})
                if isinstance(detailed, dict):
                    for item in detailed.values():
                        if isinstance(item, dict):
                            item.setdefault("collection_name", collection.name)
                            item.setdefault(
                                "collection_display_name",
                                collection.display_name or collection.name,
                            )
                            item.setdefault(
                                "original_document_name",
                                item.get("document_name"),
                            )

            return response_with_status(
                status=True,
                language=language,
                error_name=None,
                response_body=results,
            )

        # Multi-collection search
        primary_collection = collections[0]
        new_query_obj = UserQuery.objects.create(
            organisation_user=organisation_user,
            collection=primary_collection,
            query_str_prompt=query_str,
            query_options=options_dict,
        )

        all_detailed = []
        all_stats = {}
        all_template_prompts = []
        all_structured_results = []
        max_results = options_dict.get("max_results") or 50
        try:
            max_results = int(max_results)
        except (ValueError, TypeError):
            max_results = 50

        for coll in collections:
            coll_opts = dict(options_dict)
            if hasattr(coll, "apply_search_defaults"):
                coll_opts = coll.apply_search_defaults(coll_opts)

            sem_db_controller = (
                DBSemanticSearchController.prepare_controller_for_collection(
                    collection=coll, sse_engin_config_path=sse_engin_config_path
                )
            )
            coll_results, coll_structured, coll_template_prompts = (
                sem_db_controller.search_with_options(
                    question_str=query_str,
                    search_params=coll_opts,
                    convert_to_pd=False,
                    reformat_to_display=True,
                    ignore_question_lang_detect=ignore_question_lang_detect,
                    organisation_user=organisation_user,
                    collection=coll,
                    user_query=new_query_obj,
                )
            )
            if coll_template_prompts:
                all_template_prompts.extend(coll_template_prompts)
            if coll_structured:
                all_structured_results.extend(coll_structured)

            coll_detailed = coll_results.get("detailed_results", {})
            items = (
                list(coll_detailed.values())
                if isinstance(coll_detailed, dict)
                else list(coll_detailed or [])
            )
            for item in items:
                if not isinstance(item, dict):
                    continue
                orig_doc = item.get("document_name", "")
                annotated_doc = (
                    f"[{coll.display_name or coll.name}] {orig_doc}"
                )
                item_copy = dict(item)
                item_copy["original_document_name"] = orig_doc
                item_copy["document_name"] = annotated_doc
                item_copy["collection_name"] = coll.name
                item_copy["collection_display_name"] = (
                    coll.display_name or coll.name
                )
                all_detailed.append(item_copy)

            coll_stats = coll_results.get("stats", {})
            if isinstance(coll_stats, dict):
                for doc_name, stat_val in coll_stats.items():
                    if isinstance(stat_val, dict):
                        annotated_doc = (
                            f"[{coll.display_name or coll.name}] {doc_name}"
                        )
                        stat_copy = dict(stat_val)
                        stat_copy["original_document_name"] = doc_name
                        stat_copy["document_name"] = annotated_doc
                        stat_copy["collection_name"] = coll.name
                        stat_copy["collection_display_name"] = (
                            coll.display_name or coll.name
                        )
                        all_stats[annotated_doc] = stat_copy

        if multi_coll_mode == "global_limit":
            all_detailed.sort(
                key=lambda x: x.get("score")
                if isinstance(x.get("score"), (int, float))
                else 0.0,
                reverse=True,
            )
            final_detailed_list = all_detailed[:max_results]
            kept_doc_names = {
                x["document_name"]
                for x in final_detailed_list
                if "document_name" in x
            }
            final_stats = {
                k: v for k, v in all_stats.items() if k in kept_doc_names
            }
        else:
            final_detailed_list = all_detailed
            final_stats = all_stats

        total_score_weighted = sum(
            v.get("score_weighted", 0.0)
            for v in final_stats.values()
            if isinstance(v.get("score_weighted"), (int, float))
        )
        if total_score_weighted > 0:
            for v in final_stats.values():
                if isinstance(v.get("score_weighted"), (int, float)):
                    v["score_weighted_scaled"] = (
                        v["score_weighted"] / total_score_weighted
                    )

        final_detailed_dict = {
            str(idx): item for idx, item in enumerate(final_detailed_list)
        }

        merged_results = {
            "stats": final_stats,
            "detailed_results": final_detailed_dict,
        }
        if all_structured_results:
            merged_results["structured_results"] = all_structured_results

        query_response = UserQueryResponse.objects.create(
            user_query=new_query_obj,
            general_stats_json=final_stats,
            detailed_results_json=final_detailed_dict,
            structured_results=all_structured_results or None,
        )

        query_response_result = {
            "results": merged_results,
            "query_response_id": query_response.pk,
        }
        if all_template_prompts:
            query_response_result["template_prompts"] = all_template_prompts

        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body=query_response_result,
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
    input_validators = {
        "query_response_id": identifier,
        "query_options": generation_options,
        "query_instruction": optional_text,
        "system_prompt": optional_text,
    }

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

        # The collection answers with the model and the sampling parameters
        # saved on it, unless the request names its own.
        collection = getattr(
            getattr(user_response, "user_query", None), "collection", None
        )
        if collection is not None and hasattr(
            collection, "apply_generation_defaults"
        ):
            query_options = collection.apply_generation_defaults(query_options)

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
                "citations": (query_response.generative_qa_result_json or {}).get(
                    "citations", []
                ),
            },
        )


class RewriteQuery(APIView):
    required_params = ["query_str", "generative_model"]
    optional_params = ["historical_messages"]
    input_validators = {
        "query_str": required_text,
        "generative_model": required_text,
        "historical_messages": conversation_history,
    }

    @required_params_exists(
        required_params=required_params, optional_params=optional_params
    )
    @get_organisation_user
    @get_default_language
    def post(self, language, organisation_user, request):
        params = request_params(request)
        try:
            controller = GenerativeModelControllerApi(deepl_api_key="")
            rewritten = controller.rewrite_query(
                query=params["query_str"],
                history=params.get("historical_messages", []),
                model_name=params["generative_model"],
                lang=language,
            )
            if not isinstance(rewritten, str) or not rewritten.strip():
                raise ValueError("Router returned an empty rewritten query")
        except Exception:
            logging.exception("Query rewriting failed")
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
            response_body={"rewritten_query": rewritten},
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
    """
    Configured embedders.

    ``models`` lists the names, ``models_info`` describes each of them:
    ``vector_size`` (fixed for the whole collection) and
    ``max_model_length`` (the largest single input text, in tokens, which
    limits ``max_tokens_in_chunk`` of a collection).
    """

    @get_default_language
    def get(self, language, request):
        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body={
                "models": EmbeddingModelsConfig.embedders(),
                "models_info": EmbeddingModelsConfig.embedders_info(),
            },
        )


class ListRerankersModels(APIView):
    """
    Configured rerankers.

    ``max_model_length`` of a reranker is the limit of a ``query + text``
    pair, so a chunk which fits the embedder may still be cut here.
    """

    @get_default_language
    def get(self, language, request):
        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body={
                "models": EmbeddingModelsConfig.rerankers(),
                "models_info": EmbeddingModelsConfig.rerankers_info(),
            },
        )


class SetRateForQueryResponseAnswer(APIView):
    required_params = ["answer_response_id", "rate_value", "rate_value_max"]
    optional_params = ["rate_comment"]
    input_validators = {
        "answer_response_id": identifier,
        "rate_value": rating_integer,
        "rate_value_max": rating_integer,
        "rate_comment": optional_text,
    }

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


class ListOfSearchHistory(APIView):
    """
    Return the search history for the authenticated user.
    Optionally filter by collection_name or query_response_id.
    """

    optional_params = ["collection_name", "query_response_id"]
    input_validators = {
        "collection_name": optional_text,
        "query_response_id": identifier,
    }

    @required_params_exists(required_params=[], optional_params=optional_params)
    @get_organisation_user
    @get_default_language
    def get(self, language, organisation_user, request):
        params = request_params(request)
        collection_name = params.get("collection_name")
        query_response_id = params.get("query_response_id")

        if query_response_id is not None:
            user_response = SearchQueryController.get_user_response_by_id(
                query_response_id=query_response_id,
                organisation_user=organisation_user,
            )
            if user_response is None:
                return response_with_status(
                    status=False,
                    language=language,
                    error_name=RESPONSE_ACCESS_DENIED,
                    response_body={},
                )
            history = [
                SearchQueryController.format_search_history_entry(user_response)
            ]
        else:
            responses = SearchQueryController.get_user_search_history(
                organisation_user=organisation_user,
                collection_name=collection_name,
            )
            history = [
                SearchQueryController.format_search_history_entry(resp)
                for resp in responses
            ]

        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body={"history": history},
        )


class GetSearchHistoryById(APIView):
    """
    Retrieve a specific search history entry by query_response_id.
    """

    required_params = ["query_response_id"]
    input_validators = {
        "query_response_id": identifier,
    }

    @required_params_exists(required_params=required_params)
    @get_organisation_user
    @get_default_language
    def get(self, language, organisation_user, request):
        params = request_params(request)
        query_response_id = params.get("query_response_id")

        user_response = SearchQueryController.get_user_response_by_id(
            query_response_id=query_response_id,
            organisation_user=organisation_user,
        )
        if user_response is None:
            return response_with_status(
                status=False,
                language=language,
                error_name=RESPONSE_ACCESS_DENIED,
                response_body={},
            )

        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body=SearchQueryController.format_search_history_entry(
                user_response
            ),
        )


class DeleteSearchHistory(APIView):
    """
    Delete the search history of the authenticated user.

    Entries can be given by ``query_ids[]`` and/or by ``query_response_ids[]``
    (at least one of them is required); the single ``query_id`` and
    ``query_response_id`` are accepted as aliases.  These are the identifiers
    which the rest of the history API (``search_history``,
    ``get_search_history``) serves.  The responses, the generated answers and
    the RAG states of the chats which reference them are removed with them;
    nothing is removed from the collections themselves.
    """

    required_params = []
    optional_params = [
        "query_ids[]",
        "query_response_ids[]",
        "query_id",
        "query_response_id",
    ]
    input_validators = {
        "query_ids[]": identifiers,
        "query_response_ids[]": identifiers,
        "query_id": identifier,
        "query_response_id": identifier,
    }

    @required_params_exists(
        required_params=required_params, optional_params=optional_params
    )
    @get_organisation_user
    @get_default_language
    def post(self, language, organisation_user, request, *args, **kwargs):
        params = request_params(request)
        query_ids = list(params.get("query_ids[]") or [])
        query_response_ids = list(params.get("query_response_ids[]") or [])
        if params.get("query_id") is not None:
            query_ids.append(params["query_id"])
        if params.get("query_response_id") is not None:
            query_response_ids.append(params["query_response_id"])

        if not query_ids and not query_response_ids:
            return error_response(INVALID_PARAMS, language)

        controller = SearchHistoryDeletionController()
        try:
            result = controller.delete_queries(
                organisation_user=organisation_user,
                query_ids=query_ids,
                query_response_ids=query_response_ids,
            )
        except HistoryDeletionProtected:
            return response_with_status(
                status=False, language=language, error_name=DELETE_PROTECTED
            )

        if not result.deleted.get("queries"):
            return response_with_status(
                status=False,
                language=language,
                error_name=QUERY_NOT_FOUND,
            )

        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body=result.as_dict(),
        )

    delete = post

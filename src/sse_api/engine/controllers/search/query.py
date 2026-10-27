from typing import List, Optional

from sse_api.system.models import OrganisationUser

from sse_api.data.models import CollectionOfDocuments

from sse_api.engine.models import (
    UserQuery,
    UserQueryResponse,
    UserQueryResponseAnswer,
)
from sse_api.engine.controllers.search.semantic import DBSemanticSearchController


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
    def get_user_response_by_id(
        query_response_id: int, organisation_user: OrganisationUser
    ) -> UserQueryResponse | None:
        """
        Retrieve a ``UserQueryResponse`` instance by its primary key and verify owner.

        Parameters
        ----------
        query_response_id : int
            Primary key of the ``UserQueryResponse`` to fetch.
        organisation_user : OrganisationUser
            The user whose ownership is verified.

        Returns
        -------
        UserQueryResponse | None
            The matching response object, or ``None`` if it does not exist or access denied.
        """
        try:
            return UserQueryResponse.objects.get(
                id=query_response_id, user_query__organisation_user=organisation_user
            )
        except UserQueryResponse.DoesNotExist:
            return None

    @staticmethod
    def get_user_search_history(
        organisation_user: OrganisationUser,
        collection: Optional[CollectionOfDocuments] = None,
        collection_name: Optional[str] = None,
    ) -> List[UserQueryResponse]:
        """
        Retrieve all ``UserQueryResponse`` instances belonging to the given user,
        optionally filtered by collection.

        Parameters
        ----------
        organisation_user : OrganisationUser
            The user whose query history is fetched.
        collection : CollectionOfDocuments | None, optional
            Collection instance to filter by.
        collection_name : str | None, optional
            Collection name to filter by.

        Returns
        -------
        List[UserQueryResponse]
            List of responses ordered from newest to oldest.
        """
        qs = UserQueryResponse.objects.filter(
            user_query__organisation_user=organisation_user
        ).select_related("user_query", "user_query__collection")
        if collection is not None:
            qs = qs.filter(user_query__collection=collection)
        elif collection_name is not None and collection_name != "":
            qs = qs.filter(user_query__collection__name=collection_name)
        return list(qs.order_by("-id"))

    @staticmethod
    def format_search_history_entry(user_response: UserQueryResponse) -> dict:
        """
        Format a ``UserQueryResponse`` into a comprehensive dictionary containing
        the full query metadata, complete search results, and all generated answers.

        Parameters
        ----------
        user_response : UserQueryResponse
            The response object to format.

        Returns
        -------
        dict
            Structured dictionary with search history details.
        """
        user_query = user_response.user_query
        answers_qs = UserQueryResponseAnswer.objects.filter(
            user_response=user_response
        ).order_by("id")
        answers = []
        for ans in answers_qs:
            answers.append(
                {
                    "id": ans.pk,
                    "response_id": ans.pk,
                    "is_generative": ans.is_generative,
                    "answer_options": ans.answer_options or {},
                    "query_instruction_prompt": ans.query_instruction_prompt,
                    "generated_answer": ans.generated_answer,
                    "generated_answer_translated": ans.generated_answer_translated,
                    "rate_value": ans.rate_value,
                    "rate_value_max": ans.rate_nax_value,
                    "rate_comment": ans.rate_comment,
                    "generation_time": (
                        ans.generation_time.total_seconds()
                        if ans.generation_time is not None
                        else None
                    ),
                    "extractive_qa_result_json": ans.extractive_qa_result_json,
                    "generative_qa_result_json": ans.generative_qa_result_json,
                }
            )
        latest_answer = answers[-1] if answers else None

        return {
            "query_id": user_query.pk,
            "query_response_id": user_response.pk,
            "query": user_query.query_str_prompt,
            "query_str": user_query.query_str_prompt,
            # when the search was made; a client which listed the history
            # before this field existed treats it as an unknown date
            "created_on": (
                user_query.created_on.isoformat()
                if user_query.created_on is not None
                else None
            ),
            "collection_id": (
                user_query.collection.pk if user_query.collection else None
            ),
            "collection_name": (
                user_query.collection.name if user_query.collection else None
            ),
            "query_options": user_query.query_options or {},
            "results": {
                "query": user_query.query_str_prompt,
                "stats": user_response.general_stats_json or {},
                "detailed_results": user_response.detailed_results_json or {},
                "structured_results": user_response.structured_results or [],
            },
            "stats": user_response.general_stats_json or {},
            "detailed_results": user_response.detailed_results_json or {},
            "structured_results": user_response.structured_results or [],
            "answers": answers,
            "generated_answer": (
                latest_answer.get("generated_answer") if latest_answer else None
            ),
            "generated_answer_translated": (
                latest_answer.get("generated_answer_translated")
                if latest_answer
                else None
            ),
            "answer": (
                latest_answer.get("generated_answer") if latest_answer else None
            ),
        }

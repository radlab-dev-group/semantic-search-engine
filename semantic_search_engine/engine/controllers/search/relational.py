from system.models import OrganisationUser

from data.models import CollectionOfDocuments

from engine.models import UserQuery, UserQueryResponse
from engine.controllers.search.semantic import DBSemanticSearchController


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

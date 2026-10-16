"""
deletion.py
-----------

Deletion of the search history of a user: the stored queries, their
responses, the generated answers and the RAG states of the chats which
reference them.

The history rows are related through ``PROTECT`` foreign keys, so they have
to be removed in the order which respects those keys: chat states first,
then answers, responses, and only then the queries themselves.  The states
of the chat messages are not deleted - the message keeps its life and only
loses the reference (``state`` is set to ``None``).  Everything runs in one
transaction, so a partial failure leaves the history untouched::

    controller = SearchHistoryDeletionController()
    result = controller.delete_queries(
        organisation_user=organisation_user, query_ids=[1, 2]
    )
    print(result.as_dict())
"""

from dataclasses import dataclass, field
from typing import Any, List

from django.db import transaction
from django.db.models.deletion import ProtectedError  # type: ignore[import-untyped]
from django.apps import apps

from sse_api.system.models import OrganisationUser


def _deleted_rows(delete_result: Any, model: Any) -> int:
    """
    Number of rows of the model deleted by QuerySet.delete().
    :param delete_result: Result of QuerySet.delete()
    :param model: Model which rows were deleted
    :return: Number of deleted rows
    """
    if isinstance(delete_result, tuple) and len(delete_result) == 2:
        delete_result = delete_result[1]
    if not isinstance(delete_result, dict):
        return 0
    for label in (model._meta.label, model.__name__):
        if label in delete_result:
            return int(delete_result[label])
    return 0


class HistoryDeletionProtected(Exception):
    """
    Raised when other rows are protected against the removal of the history.
    """


@dataclass
class HistoryDeletionResult:
    """
    Outcome of a single deletion of search history: what was removed and
    which of the requested queries were not found.
    """

    deleted: dict = field(default_factory=dict)
    not_found: List[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        """
        Deletion outcome prepared for the API response: the number of the
        deleted rows and the ids which were not found.
        :return: Dictionary with ``deleted`` and ``not_found``
        """
        return {"deleted": self.deleted, "not_found": self.not_found}


class SearchHistoryDeletionController:
    """
    Removes the search history of a user: queries, responses and answers.
    """

    @staticmethod
    def delete_queries(
        organisation_user: OrganisationUser,
        query_ids,
    ) -> HistoryDeletionResult:
        """
        Delete the queries of the user given by their ids, together with
        their responses, answers and the RAG states which reference them.
        The ``state`` of the chat messages is cleared instead of removed.
        :param organisation_user: User who the history belongs to
        :param query_ids: Ids of the queries to delete
        :return: HistoryDeletionResult
        """
        wanted_ids = [
            int(query_id) for query_id in query_ids if str(query_id).strip()
        ]
        query_model = apps.get_model("engine", "UserQuery")
        query_response_model = apps.get_model("engine", "UserQueryResponse")
        query_answer_model = apps.get_model("engine", "UserQueryResponseAnswer")
        rag_state_model = apps.get_model("chat", "RAGMessageState")
        message_state_model = apps.get_model("chat", "MessageState")

        queries = list(
            query_model.objects.filter(
                organisation_user=organisation_user, pk__in=wanted_ids
            )
        )
        query_ids_of_user = [query.pk for query in queries]

        deleted = {
            "answers": 0,
            "responses": 0,
            "queries": 0,
            "rag_states": 0,
        }
        if not query_ids_of_user:
            return HistoryDeletionResult(
                deleted=deleted, not_found=sorted(wanted_ids)
            )

        try:
            with transaction.atomic():
                rag_states = list(
                    rag_state_model.objects.filter(
                        sse_query__pk__in=query_ids_of_user
                    )
                )
                rag_state_ids = [state.pk for state in rag_states]

                # The states of the chat messages are detached instead of
                # removed: the message keeps its life and only loses the
                # reference, which is what PROTECT is there for.
                message_state_model.objects.filter(
                    rag_message_state__pk__in=rag_state_ids
                ).update(rag_message_state=None)

                result = rag_state_model.objects.filter(
                    pk__in=rag_state_ids
                ).delete()
                deleted["rag_states"] = _deleted_rows(result, rag_state_model)

                responses = list(
                    query_response_model.objects.filter(
                        user_query__pk__in=query_ids_of_user
                    )
                )
                response_ids = [response.pk for response in responses]
                result = query_answer_model.objects.filter(
                    user_response__pk__in=response_ids
                ).delete()
                deleted["answers"] += _deleted_rows(result, query_answer_model)
                result = query_response_model.objects.filter(
                    pk__in=response_ids
                ).delete()
                deleted["responses"] += _deleted_rows(result, query_response_model)

                result = query_model.objects.filter(
                    pk__in=query_ids_of_user
                ).delete()
                deleted["queries"] += _deleted_rows(result, query_model)
        except ProtectedError as error:
            raise HistoryDeletionProtected(str(error)) from error

        return HistoryDeletionResult(
            deleted=deleted,
            not_found=sorted(set(wanted_ids) - set(query_ids_of_user)),
        )

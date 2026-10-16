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
transaction, so a partial failure leaves the history untouched.

Entries can be addressed by their ``query_id`` or by their
``query_response_id``: the two are the identifiers which the rest of the
history API (``search_history``, ``get_search_history``) serves, so either
of them can be given.  When both are given, the union is deleted::

    controller = SearchHistoryDeletionController()
    result = controller.delete_queries(
        organisation_user=organisation_user,
        query_ids=[1, 2],
        query_response_ids=[7],
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


def _as_ids(values) -> List[int]:
    """
    Normalise a parameter which may be a single value, a list, or ``None``
    into a de-duplicated list of positive integers.
    :param values: The raw parameter value
    :return: List of integers
    """
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        items = list(values)
    else:
        items = [values]
    ids = set()
    for item in items:
        if item is None or str(item).strip() == "":
            continue
        try:
            value = int(item)
        except (TypeError, ValueError):
            continue
        if value > 0:
            ids.add(value)
    return sorted(ids)


class HistoryDeletionProtected(Exception):
    """
    Raised when other rows are protected against the removal of the history.
    """


@dataclass
class HistoryDeletionResult:
    """
    Outcome of a single deletion of search history: what was removed and
    which of the requested ids were not found.
    """

    deleted: dict = field(default_factory=dict)
    not_found: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """
        Deletion outcome prepared for the API response: the number of the
        deleted rows and the ids which were not found, per identifier type.
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
        query_ids=None,
        query_response_ids=None,
    ) -> HistoryDeletionResult:
        """
        Delete the queries of the user, addressed by their ``query_id`` and/or
        their ``query_response_id``, together with their responses, answers and
        the RAG states which reference them.  The ``state`` of the chat
        messages is cleared instead of removed.
        :param organisation_user: User who the history belongs to
        :param query_ids: Ids of the queries to delete
        :param query_response_ids: Ids of the query responses to delete
        :return: HistoryDeletionResult
        """
        wanted_query_ids = _as_ids(query_ids)
        wanted_response_ids = _as_ids(query_response_ids)

        query_model = apps.get_model("engine", "UserQuery")
        query_response_model = apps.get_model("engine", "UserQueryResponse")
        query_answer_model = apps.get_model("engine", "UserQueryResponseAnswer")
        rag_state_model = apps.get_model("chat", "RAGMessageState")
        message_state_model = apps.get_model("chat", "MessageState")

        # Resolve the ids of the queries which are actually owned by the user,
        # from both the query ids and the response ids which were given.
        query_ids_of_user = set()
        if wanted_query_ids:
            query_ids_of_user |= {
                query.pk
                for query in query_model.objects.filter(
                    organisation_user=organisation_user,
                    pk__in=wanted_query_ids,
                )
            }
        if wanted_response_ids:
            query_ids_of_user |= {
                response.user_query_id
                for response in query_response_model.objects.filter(
                    user_query__organisation_user=organisation_user,
                    pk__in=wanted_response_ids,
                )
            }
        query_ids_of_user = sorted(query_ids_of_user)

        # Report, per identifier type, which of the requested ids were not
        # found (or did not belong to the user).
        not_found = {"query_ids": [], "query_response_ids": []}
        if wanted_query_ids:
            not_found["query_ids"] = sorted(
                set(wanted_query_ids) - set(query_ids_of_user)
            )
        if wanted_response_ids:
            found_response_ids = set()
            if query_ids_of_user:
                found_response_ids = {
                    response.pk
                    for response in query_response_model.objects.filter(
                        user_query__organisation_user=organisation_user,
                        user_query__pk__in=query_ids_of_user,
                    )
                }
            # Only the requested response ids count as found.
            not_found["query_response_ids"] = sorted(
                set(wanted_response_ids) - found_response_ids
            )

        deleted = {
            "answers": 0,
            "responses": 0,
            "queries": 0,
            "rag_states": 0,
        }
        if not query_ids_of_user:
            return HistoryDeletionResult(deleted=deleted, not_found=not_found)

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
            not_found=not_found,
        )

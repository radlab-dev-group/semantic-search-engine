"""
deletion.py
-----------

Deletion of the chats of a user: the chats themselves, their messages and
the message states (RAG and content-supervisor states) which belong to them.

The rows of a chat are related through ``PROTECT`` foreign keys, so they have
to be removed in the order which respects those keys: messages first, then
``MessageState``, then ``RAGMessageState`` and ``ContentSupervisorState``, and
finally the ``Chat`` rows themselves.  The underlying search queries,
responses and answers referenced by ``RAGMessageState`` are not removed —
search history keeps its own life.  Everything runs in one transaction, so a
partial failure leaves the chats untouched.
"""

from dataclasses import dataclass, field
from typing import Any, List

from django.apps import apps
from django.db import transaction
from django.db.models.deletion import ProtectedError  # type: ignore[import-untyped]

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


class ChatDeletionProtected(Exception):
    """
    Raised when other rows are protected against the removal of a chat.
    """


@dataclass
class ChatDeletionResult:
    """
    Outcome of a single deletion of chats: what was removed and which of the
    requested ids were not found.
    """

    deleted: dict = field(default_factory=dict)
    not_found: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """
        Deletion outcome prepared for the API response: the number of the
        deleted rows and the ids which were not found.
        :return: Dictionary with ``deleted`` and ``not_found``
        """
        return {"deleted": self.deleted, "not_found": self.not_found}


def delete_chats_by_ids(chat_ids: List[int]) -> dict:
    """
    Delete the given chats together with their messages and message states,
    respecting the ``PROTECT`` foreign keys.
    :param chat_ids: Primary keys of the chats to remove
    :return: Dictionary with counts of deleted rows per model
    """
    chat_model = apps.get_model("chat", "Chat")
    message_model = apps.get_model("chat", "Message")
    message_state_model = apps.get_model("chat", "MessageState")
    rag_state_model = apps.get_model("chat", "RAGMessageState")
    cs_state_model = apps.get_model("chat", "ContentSupervisorState")

    deleted = {
        "chats": 0,
        "messages": 0,
        "message_states": 0,
        "rag_states": 0,
        "content_supervisor_states": 0,
    }
    if not chat_ids:
        return deleted

    with transaction.atomic():
        state_ids = list(
            message_model.objects.filter(chat__pk__in=chat_ids, state__isnull=False)
            .values_list("state_id", flat=True)
            .distinct()
        )
        rag_state_ids = []
        cs_state_ids = []
        if state_ids:
            rag_state_ids = list(
                message_state_model.objects.filter(
                    pk__in=state_ids, rag_message_state__isnull=False
                )
                .values_list("rag_message_state_id", flat=True)
                .distinct()
            )
            cs_state_ids = list(
                message_state_model.objects.filter(
                    pk__in=state_ids, content_supervisor_state__isnull=False
                )
                .values_list("content_supervisor_state_id", flat=True)
                .distinct()
            )

        result = message_model.objects.filter(chat__pk__in=chat_ids).delete()
        deleted["messages"] = _deleted_rows(result, message_model)

        if state_ids:
            result = message_state_model.objects.filter(
                pk__in=state_ids, message__isnull=True
            ).delete()
            deleted["message_states"] = _deleted_rows(result, message_state_model)

        if rag_state_ids:
            result = rag_state_model.objects.filter(
                pk__in=rag_state_ids, messagestate__isnull=True
            ).delete()
            deleted["rag_states"] = _deleted_rows(result, rag_state_model)

        if cs_state_ids:
            result = cs_state_model.objects.filter(
                pk__in=cs_state_ids, messagestate__isnull=True
            ).delete()
            deleted["content_supervisor_states"] = _deleted_rows(
                result, cs_state_model
            )

        result = chat_model.objects.filter(pk__in=chat_ids).delete()
        deleted["chats"] = _deleted_rows(result, chat_model)

    return deleted


class ChatDeletionController:
    """
    Removes the chats of a user together with their messages and states.
    """

    @staticmethod
    def delete_chats(
        organisation_user: OrganisationUser,
        chat_ids=None,
    ) -> ChatDeletionResult:
        """
        Delete the chats of the user addressed by ``chat_ids``, together with
        their messages and message states.
        :param organisation_user: User who the chats belong to
        :param chat_ids: Ids of the chats to delete
        :return: ChatDeletionResult
        """
        wanted_chat_ids = _as_ids(chat_ids)
        chat_model = apps.get_model("chat", "Chat")

        chat_ids_of_user = []
        if wanted_chat_ids:
            chat_ids_of_user = sorted(
                chat.pk
                for chat in chat_model.objects.filter(
                    organisation_user=organisation_user,
                    pk__in=wanted_chat_ids,
                )
            )

        not_found = {
            "chat_ids": sorted(set(wanted_chat_ids) - set(chat_ids_of_user))
        }
        deleted = {
            "chats": 0,
            "messages": 0,
            "message_states": 0,
            "rag_states": 0,
            "content_supervisor_states": 0,
        }
        if not chat_ids_of_user:
            return ChatDeletionResult(deleted=deleted, not_found=not_found)

        try:
            deleted = delete_chats_by_ids(chat_ids_of_user)
        except ProtectedError as error:
            raise ChatDeletionProtected(str(error)) from error

        return ChatDeletionResult(
            deleted=deleted,
            not_found=not_found,
        )

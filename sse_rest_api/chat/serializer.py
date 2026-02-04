from rest_framework import serializers

from engine.models import UserQuery, UserQueryResponse, UserQueryResponseAnswer
from chat.models import (
    Chat,
    Message,
    MessageState,
    ContentSupervisorState,
    RAGMessageState,
)


class ChatSerializer(serializers.ModelSerializer):
    """
    Serializer for the ``Chat`` model.

    Exposes the primary identifier, owning organization user, optional
    collection, creation timestamp, and any generation ``options`` that were
    supplied when the chat was created.
    """

    class Meta:
        model = Chat
        fields = ["id", "organisation_user", "collection", "created_at", "options"]


class MessageSerializer(serializers.ModelSerializer):
    """
    Serializer for the ``Message`` model (lightweight view).

    Includes basic identification and content fields as well as the message
    order (``number``), timestamps, and the generation duration.
    """

    class Meta:
        model = Message
        fields = [
            "id",
            "chat",
            "role",
            "text",
            "text_translated",
            "number",
            "date_time",
            "generation_time",
        ]


class ContentSupervisorStateSerializer(serializers.ModelSerializer):
    """
    Serializer for ``ContentSupervisorState``.

    Returns the full set of fields (``state_type`` and optional JSON
    ``www_content``) because the API may need the complete supervision payload.
    """

    class Meta:
        model = ContentSupervisorState
        fields = "__all__"


class UserQuerySerializer(serializers.ModelSerializer):
    """
    Serializer for ``engine.models.UserQuery``.

    Exposes all model fields; the serializer is primarily used as a nested read‑only
    representation inside ``RAGStateSerializer``.
    """

    class Meta:
        model = UserQuery
        fields = "__all__"


class UserQueryResponseSerializer(serializers.ModelSerializer):
    """
    Serializer for ``engine.models.UserQueryResponse``.

    Provides a full representation of a search‑engine response; used as a nested
    read‑only field in ``RAGStateSerializer``.
    """

    class Meta:
        model = UserQueryResponse
        fields = "__all__"


class UserQueryResponseAnswerSerializer(serializers.ModelSerializer):
    """
    Serializer for ``engine.models.UserQueryResponseAnswer``.

    Returns the complete answer object; nested inside ``RAGStateSerializer``.
    """

    class Meta:
        model = UserQueryResponseAnswer
        fields = "__all__"


class RAGStateSerializer(serializers.ModelSerializer):
    """
    Serializer for ``RAGMessageState`` with nested read‑only query objects.

    The nested ``UserQuery``, ``UserQueryResponse`` and
    ``UserQueryResponseAnswer`` serializers are marked ``read_only=True`` to
    prevent accidental writes through this endpoint.
    """

    sse_query = UserQuerySerializer(read_only=True)
    sse_response = UserQueryResponseSerializer(read_only=True)
    sse_answer = UserQueryResponseAnswerSerializer(read_only=True)

    class Meta:
        model = RAGMessageState
        fields = "__all__"


class MessageStateSerializer(serializers.ModelSerializer):
    """
    Serializer for ``MessageState`` linking RAG and content‑supervisor data.

    Both related objects are included as nested read‑only serializers.
    """

    content_supervisor_state = ContentSupervisorStateSerializer(read_only=True)
    rag_message_state = RAGStateSerializer(read_only=True)

    class Meta:
        model = MessageState
        fields = "__all__"


class DeepMessageSerializer(serializers.ModelSerializer):
    """
    Full serializer for ``Message`` including its ``MessageState`` and
    navigation links.

    This serializer is used when the API needs to return the complete chat
    history with state information and pointers to the previous/next messages.
    """

    state = MessageStateSerializer(read_only=True)

    class Meta:
        model = Message
        fields = [
            "id",
            "chat",
            "role",
            "text",
            "text_translated",
            "number",
            "date_time",
            "state",
            "prev_message",
            "next_message",
            "generation_time",
        ]

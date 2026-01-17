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
    class Meta:
        model = Chat
        fields = ["id", "organisation_user", "collection", "created_at", "options"]


class MessageSerializer(serializers.ModelSerializer):
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

    class Meta:
        model = ContentSupervisorState
        fields = "__all__"


class UserQuerySerializer(serializers.ModelSerializer):
    class Meta:
        model = UserQuery
        fields = "__all__"


class UserQueryResponseSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserQueryResponse
        fields = "__all__"


class UserQueryResponseAnswerSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserQueryResponseAnswer
        fields = "__all__"


class RAGStateSerializer(serializers.ModelSerializer):
    sse_query = UserQuerySerializer(read_only=True)
    sse_response = UserQueryResponseSerializer(read_only=True)
    sse_answer = UserQueryResponseAnswerSerializer(read_only=True)

    class Meta:
        model = RAGMessageState
        fields = "__all__"


class MessageStateSerializer(serializers.ModelSerializer):
    content_supervisor_state = ContentSupervisorStateSerializer(read_only=True)
    rag_message_state = RAGStateSerializer(read_only=True)

    class Meta:
        model = MessageState
        fields = "__all__"


class DeepMessageSerializer(serializers.ModelSerializer):
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

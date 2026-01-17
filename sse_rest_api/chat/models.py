from django.db import models

from engine.models import UserQuery, UserQueryResponse, UserQueryResponseAnswer
from data.models import OrganisationUser, CollectionOfDocuments


class Chat(models.Model):
    # pk: auto-id

    organisation_user = models.ForeignKey(
        OrganisationUser, null=False, on_delete=models.PROTECT
    )
    collection = models.ForeignKey(
        CollectionOfDocuments, null=True, on_delete=models.PROTECT
    )

    created_at = models.DateTimeField(auto_now_add=True)
    options = models.JSONField(null=True)
    search_options = models.JSONField(null=True)

    hash = models.TextField(null=True)
    is_saved = models.BooleanField(default=False, null=False)
    read_only = models.BooleanField(default=False, null=False)


class ContentSupervisorState(models.Model):
    # pk: auto-id

    state_type = models.TextField(null=False)

    www_content = models.JSONField(null=True)


class RAGMessageState(models.Model):
    # pk: auto-id

    contains_query = models.BooleanField(null=False)
    contains_instruction = models.BooleanField(null=False)

    extracted_query = models.TextField(null=True)
    extracted_instruction = models.TextField(null=True)

    sse_query = models.ForeignKey(
        UserQuery,
        null=True,
        on_delete=models.PROTECT,
    )
    sse_response = models.ForeignKey(
        UserQueryResponse, null=True, on_delete=models.PROTECT
    )
    sse_answer = models.ForeignKey(
        UserQueryResponseAnswer, null=True, on_delete=models.PROTECT
    )

    system_prompt = models.TextField(null=True)


class MessageState(models.Model):
    # pk: auto-id

    rag_message_state = models.ForeignKey(
        RAGMessageState, null=True, on_delete=models.PROTECT
    )

    content_supervisor_state = models.ForeignKey(
        ContentSupervisorState, null=True, on_delete=models.PROTECT
    )


class Message(models.Model):
    # pk: auto-id

    chat = models.ForeignKey(Chat, null=False, on_delete=models.PROTECT)
    role = models.CharField(max_length=32, null=False)

    text = models.TextField(null=False)
    text_translated = models.TextField(null=True)
    number = models.IntegerField(null=False)

    date_time = models.DateTimeField(auto_now_add=True)

    state = models.ForeignKey(MessageState, null=True, on_delete=models.PROTECT)
    options = models.JSONField(null=True)
    search_options = models.JSONField(null=True)

    prev_message = models.ForeignKey(
        "Message",
        null=True,
        on_delete=models.CASCADE,
        related_name="msg_prev_message",
    )

    next_message = models.ForeignKey(
        "Message",
        null=True,
        on_delete=models.CASCADE,
        related_name="msg_next_message",
    )

    generation_time = models.DurationField(null=True)

    class Meta:
        unique_together = ("chat", "role", "number")

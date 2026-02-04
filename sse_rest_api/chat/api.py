"""
Top‑level module providing API endpoints for chat operations.

This module defines several Django REST Framework ``APIView`` subclasses that
handle creating chats, adding user messages, retrieving saved chats, and
listing a user's chats.  The views delegate business logic to a ``ChatController``
and rely on custom decorators for request validation, organisation resolution,
and language selection.

All responses are wrapped with ``response_with_status`` to provide a uniform
JSON payload that includes a status flag, optional error information and the
actual response body.
"""

from rest_framework.views import APIView

from main.src.response import response_with_status
from main.src.decorators import required_params_exists, get_default_language

from system.core.decorators import get_organisation_user

from chat.models import Chat, Message
from chat.controllers.chat import ChatController
from chat.core.errors import (
    COLLECTION_NOT_FOUND,
    CHAT_ID_NOT_FOUND,
    USER_DENIED_TO_CHAT,
    CANNOT_ADD_MESSAGE_CHAT_RO,
)
from chat.serializer import ChatSerializer, MessageSerializer, MessageStateSerializer

from engine.controllers.database.relational_db import RelationalDBController


class NewChat(APIView):
    """
    Create a new chat session.

    The endpoint expects optional ``options``, a ``collection_name`` and
    ``search_options`` in the request body.  If a collection name is supplied
    the corresponding collection is fetched from the relational database; a
    missing collection results in an error response.

    The view returns a JSON response containing the serialized newly created
    ``Chat`` object.
    """

    required_params = []
    optional_params = ["options", "collection_name", "search_options"]

    chat_controller = ChatController()

    @required_params_exists(
        required_params=required_params, optional_params=optional_params
    )
    @get_organisation_user
    @get_default_language
    def post(self, language, organisation_user, request):
        options_dict = {}
        if "options" in request.data and len(request.data.get("options")):
            options_dict = request.data.get("options")

        collection = None
        collection_name = request.data.get("collection_name", None)
        if collection_name is not None and len(collection_name.strip()):
            collection = RelationalDBController.get_collection(
                collection_name=collection_name, created_by=organisation_user
            )
            if collection is None:
                return response_with_status(
                    status=False,
                    language=language,
                    error_name=COLLECTION_NOT_FOUND,
                    response_body=None,
                )

        search_options_dict = None
        search_options = request.data.get("search_options", None)
        if search_options is not None and len(search_options):
            search_options_dict = request.data.get("search_options")

        new_chat = self.chat_controller.new_chat(
            organisation_user=organisation_user,
            collection=collection,
            options=options_dict,
            search_options=search_options_dict,
        )

        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body={"chat": ChatSerializer(new_chat, many=False).data},
        )


class AddUserMessageToChatWithSystemResponse(APIView):
    """
    Add a user message to an existing chat and generate a system response.

    This view performs several steps:
    1. Validates required and optional parameters.
    2. Retrieves the target ``Chat`` and checks ownership/read‑only status.
    3. Optionally resolves a collection for Retrieval‑Augmented Generation (RAG).
    4. Persists the user message via ``ChatController.add_user_message``.
    5. Generates an assistant reply using ``ChatController.generate_assistant_message_cs_rag``.
    6. Returns the generated message, timing information, and updated history.
    """

    LAST_QUESTIONS_TO_QUERY = 4
    chat_controller = ChatController(add_to_db=True)

    required_params = ["chat_id", "user_message", "options"]
    optional_params = ["collection_name", "search_options", "system_prompt"]

    @required_params_exists(
        required_params=required_params, optional_params=optional_params
    )
    @get_organisation_user
    @get_default_language
    def post(self, language, organisation_user, request):
        """Process a user message and return the assistant's reply.

        The request payload must contain:

        - ``chat_id`` (int): Identifier of the target chat.
        - ``user_message`` (str): The message sent by the user.
        - ``options`` (dict): Generation options for the language model.
        - ``collection_name`` (optional, str): Name of the collection used for
          RAG.
        - ``search_options`` (optional, dict): Parameters controlling the
          vector‑store search.
        - ``system_prompt`` (optional, str): Custom system prompt that, if
          provided, overrides the default prompt.

        The method validates ownership, read‑only status and collection
        existence, then delegates to ``ChatController`` for the heavy lifting.
        It finally returns a JSON payload with:

        * ``generation_time`` – time taken by the model,
        * ``history`` – serialized list of all messages in the chat,
        * ``last_state`` – optional serialized ``MessageState``,
        * ``generated_assistant_message`` – the new assistant reply.
        """
        """
        {
            "chat_id": pk,
            "user_message": "user_message",
            "options": genai_options,
            "collection_name": collection["name"],
            "search_options": sse_options
            "system_prompt": system_prompt | None
        }
        """
        chat_id = request.data.get("chat_id")
        user_message = request.data.get("user_message")
        options_dict = request.data.get("options")

        system_prompt = request.data.get("system_prompt", None)
        if system_prompt is not None and len(system_prompt.strip()):
            system_prompt = system_prompt.strip()
        else:
            system_prompt = None

        search_options_dict = {}
        if "search_options" in request.data and len(request.data["search_options"]):
            search_options_dict = request.data.get("search_options")
        collection_name = request.data.get("collection_name", None)

        """
        search options dict
        {
          "categories": [
            "Sport",
            "Rozrywka"
          ],
          "documents": [],
          "relative_paths": [],
          "templates": 8,
          "only_template_documents": true,
          "max_results": 40,
          "rerank_results": false,
          "return_with_factored_fields": false,
          "relative_path_contains": [
            "https://cam.waw.pl"
          ]
        }
        """

        chat_obj = self.chat_controller.get_chat_by_id(chat_id=chat_id)
        if chat_obj is None:
            return response_with_status(
                status=False,
                language=language,
                error_name=CHAT_ID_NOT_FOUND,
                response_body=None,
            )
        if chat_obj.organisation_user != organisation_user:
            return response_with_status(
                status=False,
                language=language,
                error_name=USER_DENIED_TO_CHAT,
                response_body=None,
            )

        if chat_obj.read_only:
            return response_with_status(
                status=False,
                language=language,
                error_name=CANNOT_ADD_MESSAGE_CHAT_RO,
                response_body=None,
            )

        collection = RelationalDBController.get_collection(
            collection_name=collection_name, created_by=organisation_user
        )

        if collection is None:
            return response_with_status(
                status=False,
                language=language,
                error_name=COLLECTION_NOT_FOUND,
                response_body=None,
            )

        history, last_user_message = self.chat_controller.add_user_message(
            chat=chat_obj,
            message=user_message,
            options=options_dict,
            search_options=search_options_dict,
        )

        assistant_msg, history, message_state, generation_time = (
            self.chat_controller.generate_assistant_message_cs_rag(
                chat=chat_obj,
                collection=collection,
                last_user_message=last_user_message,
                history=history,
                options=options_dict,
                organisation_user=organisation_user,
                sse_engin_config_path="configs/milvus_config.json",
                last_questions_to_query=self.LAST_QUESTIONS_TO_QUERY,
                system_prompt=system_prompt,
            )
        )

        response_body = {
            "generation_time": generation_time,
            "history": MessageSerializer(history, many=True).data,
            "last_state": (
                MessageStateSerializer([message_state], many=True).data
                if message_state is not None
                else None
            ),
            "generated_assistant_message": assistant_msg,
        }

        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body=response_body,
        )


class SetChatStateAsSaved(APIView):
    """
    Mark a chat as saved (read‑only) or make it editable again.

    The endpoint expects ``chat_id`` and a boolean ``read_only`` flag.  It
    validates that the requesting user owns the chat and then updates the
    ``read_only`` attribute via ``ChatController.set_chat_as_saved``.  The
    response contains the new ``chat_hash`` that can be used to retrieve the
    saved chat later.
    """

    required_params = ["chat_id", "read_only"]

    chat_controller = ChatController(add_to_db=True)

    @required_params_exists(required_params=required_params)
    @get_organisation_user
    @get_default_language
    def post(self, language, organisation_user, request):
        chat_id = request.data.get("chat_id")
        read_only = request.data.get("read_only")
        chat_obj = self.chat_controller.get_chat_by_id(chat_id=chat_id)
        if chat_obj is None:
            raise Exception("Chat object not found!")
        if chat_obj.organisation_user != organisation_user:
            raise Exception(
                "Chat organisation user is different than "
                "message organisation user!"
            )

        chat_hash = self.chat_controller.set_chat_as_saved(
            chat=chat_obj, read_only=read_only
        )

        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body={"chat_hash": chat_hash},
        )


class GetSavedChatByHash(APIView):
    """
    Retrieve a previously saved (read‑only) chat using its hash.

    The view only returns chats that have been marked as saved.  If the chat
    does not exist, an empty history is returned.  Ownership is verified to
    prevent leaking another organisation's data.
    """

    required_params = ["chat_hash"]

    chat_controller = ChatController(add_to_db=True)

    @required_params_exists(required_params=required_params)
    @get_organisation_user
    @get_default_language
    def get(self, language, organisation_user, request):
        chat_hash = request.data.get("chat_hash")

        chat_obj = self.chat_controller.get_chat_by_chat_hash(
            chat_hash=chat_hash, only_saved=True
        )
        if chat_obj is None:
            chat_messages = []
        else:
            if chat_obj.organisation_user != organisation_user:
                raise Exception(
                    "Chat organisation user is different from "
                    "message organisation user!"
                )
            chat_messages = self.chat_controller.get_chat_messages(chat=chat_obj)

        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body=self.prepare_response_body(
                chat=chat_obj, chat_messages=chat_messages
            ),
        )

    @staticmethod
    def prepare_response_body(chat: Chat, chat_messages: list[Message]):
        """
        Create a consistent response dictionary for a chat.

        Parameters
        ----------
        chat : Chat | None
            The ``Chat`` instance, or ``None`` if not found.
        chat_messages : list[Message]
            List of ``Message`` objects belonging to the chat.

        Returns
        -------
        dict
            Mapping with keys:

            * ``chat_id`` – primary key of the chat or ``None``,
            * ``is_read_only`` – ``chat.read_only`` flag or ``None``,
            * ``chat_history`` – serialized list of messages.
        """
        return {
            "chat_id": chat.pk if chat is not None else None,
            "is_read_only": chat.read_only if chat else None,
            "chat_history": MessageSerializer(chat_messages, many=True).data,
        }


class ListOfUserChats(APIView):
    """
    Return a list of all chats belonging to the authenticated user.

    Each entry in the returned ``history`` list mirrors the structure produced
    by :meth:`GetSavedChatByHash.prepare_response_body`, i.e. it contains the
    chat identifier, its read‑only status and the full message history.
    """

    chat_controller = ChatController(add_to_db=True)

    @get_organisation_user
    @get_default_language
    def get(self, language, organisation_user, request):
        chat_objs = self.chat_controller.get_list_of_user_chats(
            user=organisation_user,
        )

        history = []
        for chat_obj in chat_objs:
            chat_messages = self.chat_controller.get_chat_messages(chat=chat_obj)
            single_chat_body = GetSavedChatByHash.prepare_response_body(
                chat=chat_obj, chat_messages=chat_messages
            )
            history.append(single_chat_body)

        return response_with_status(
            status=True,
            language=language,
            error_name=None,
            response_body={"history": history},
        )

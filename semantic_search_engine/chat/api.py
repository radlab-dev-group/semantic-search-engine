import json
import logging

from rest_framework.views import APIView

from main.src.response import response_with_status
from main.src.decorators import required_params_exists, get_default_language

from chat.models import Chat, Message
from chat.controllers import ChatController
from chat.core.errors import (
    COLLECTION_NOT_FOUND,
    CHAT_ID_NOT_FOUND,
    USER_DENIED_TO_CHAT,
    CANNOT_ADD_MESSAGE_CHAT_RO,
)
from chat.serializer import ChatSerializer, MessageSerializer, MessageStateSerializer
from system.core.decorators import get_organisation_user
from data.controllers.relational_db import RelationalDBController


class NewChat(APIView):
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
        return {
            "chat_id": chat.pk if chat is not None else None,
            "is_read_only": chat.read_only if chat else None,
            "chat_history": MessageSerializer(chat_messages, many=True).data,
        }


class ListOfUserChats(APIView):
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

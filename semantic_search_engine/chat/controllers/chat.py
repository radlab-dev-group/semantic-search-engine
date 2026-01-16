import string
import random
import logging

from django.db.models import QuerySet

from chat.models import Chat
from data.models import OrganisationUser, CollectionOfDocuments


class ChatLogicController:
    def __init__(self, add_to_db: bool = True, new_chat_hash_length: int = 128):
        self._add_to_db = add_to_db
        self._new_chat_hash_length = new_chat_hash_length

    def new_chat(
        self,
        organisation_user: OrganisationUser,
        collection: CollectionOfDocuments | None,
        options: dict | None,
        search_options: dict | None = None,
    ) -> Chat:
        """
        Create a new chat.
        :param organisation_user: Chat user
        :param collection: Collection of documents to conversation
        :param options: Dictionary with chat options.
        :param search_options: Dictionary with search options.
        :return: Created chat object
        """
        chat_hash = self._generate_chat_hash()
        logging.info(f"Created new chat with hash: {chat_hash}")

        new_chat = Chat(
            organisation_user=organisation_user,
            collection=collection,
            options=options,
            search_options=search_options,
            hash=chat_hash,
        )
        if self._add_to_db:
            new_chat.save()
        return new_chat

    def set_chat_as_saved(self, chat: Chat, read_only: bool) -> str:
        """
        Return hash of saved chat
        :param chat:
        :param read_only:
        :return:
        """
        chat.is_saved = True
        chat.read_only = read_only
        if self._add_to_db:
            chat.save()
        return chat.hash

    @staticmethod
    def get_list_of_user_chats(user: OrganisationUser) -> QuerySet[Chat]:
        return Chat.objects.filter(organisation_user=user)

    @staticmethod
    def get_chat_by_chat_hash(
        chat_hash: str, only_saved: bool = True
    ) -> Chat | None:
        try:
            return Chat.objects.get(is_saved=only_saved, hash=chat_hash)
        except Chat.DoesNotExist:
            return None

    def _generate_chat_hash(self) -> str:
        chat_hash = "".join(
            random.SystemRandom().choice(
                string.ascii_lowercase + string.ascii_uppercase + string.digits
            )
            for _ in range(self._new_chat_hash_length)
        )
        return chat_hash

    @staticmethod
    def get_chat_by_id(chat_id) -> Chat | None:
        try:
            return Chat.objects.get(id=chat_id)
        except Chat.DoesNotExist:
            return None

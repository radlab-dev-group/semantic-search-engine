from django.urls import path

from chat.api import (
    NewChat,
    AddUserMessageToChatWithSystemResponse,
    SetChatStateAsSaved,
    GetSavedChatByHash,
    ListOfUserChats,
)
from main.src.constants import prepare_api_url

urlpatterns = [
    path(
        prepare_api_url("new_chat"),
        NewChat.as_view(),
        name="new_chat",
    ),
    path(
        prepare_api_url("save_chat"),
        SetChatStateAsSaved.as_view(),
        name="save_chat",
    ),
    path(
        prepare_api_url("get_chat_by_hash"),
        GetSavedChatByHash.as_view(),
        name="get_chat_by_hash",
    ),
    path(
        prepare_api_url("add_user_message"),
        AddUserMessageToChatWithSystemResponse.as_view(),
        name="add_user_message",
    ),
    path(
        prepare_api_url("chats"),
        ListOfUserChats.as_view(),
        name="chats",
    ),
]

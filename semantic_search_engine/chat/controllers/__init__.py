# from chat.controllers.chat import ChatLogicController
# from chat.controllers.message import MessageLogicController
#
#
# class ChatController(ChatLogicController, MessageLogicController):
#     def __init__(self, add_to_db: bool = True, new_chat_hash_length: int = 128):
#         ChatLogicController.__init__(
#             self, add_to_db=add_to_db, new_chat_hash_length=new_chat_hash_length
#         )
#         MessageLogicController.__init__(self, add_to_db=add_to_db)

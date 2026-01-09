### Overview

The **chat** package implements a conversational interface on top of the semantic search engine. It defines models for
chat sessions, message states, and content‑supervision, along with API endpoints for creating chats, adding user
messages, persisting chats, and retrieving chat history.

### Package Layout

```
chat/
├─ core/
│   ├─ __init__.py
│   ├─ constants.py               # Error namespace marker
│   └─ errors.py                  # Chat‑specific error definitions
├─ migrations/
│   ├─ 0001_initial.py
│   └─ __init__.py
├─ __init__.py
├─ api.py                         # REST endpoints for chat operations
├─ apps.py                        # Django AppConfig
├─ controllers.py                 # High‑level chat controller (state handling, RAG, etc.)
├─ models.py                      # Django ORM models for Chat, Message, states, etc.
├─ serializer.py                  # DRF serializers for chat objects
└─ urls.py                        # URL routing for chat‑related endpoints
```

### Core Models (`models.py`)

| Model                      | Description                                                                                                                                                                                                            |
|----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Chat**                   | Represents a conversation. Links to an `OrganisationUser`, optional `CollectionOfDocuments`, stores JSON `options`/`search_options`, a unique `hash`, and flags `is_saved` & `read_only`.                              |
| **Message**                | Individual utterance. Fields: `role` (`user`, `assistant`, `system`), `text`, optional translated text, ordering `number`, timestamps, optional `MessageState`, and navigation links (`prev_message`, `next_message`). |
| **MessageState**           | Holds optional `RAGMessageState` and `ContentSupervisorState`.                                                                                                                                                         |
| **ContentSupervisorState** | Stores the type of supervision (`state_type`) and any extracted web content (`www_content`).                                                                                                                           |
| **RAGMessageState**        | Records whether a message triggered a query, the extracted query/instruction, and foreign keys to the Milvus search results (`sse_query`, `sse_response`, `sse_answer`). Also stores an optional system prompt.        |

### Controllers (`controllers.py`)

- **ChatController** – Central orchestrator for chat lifecycle:
    - `new_chat()` creates a `Chat` with a random hash.
    - `add_user_message()` stores a user message, updates linked list pointers, and returns the updated history.
    - `generate_assistant_message_cs_rag()` runs the RAG pipeline: extracts a question, optionally prepares a SSE query,
      calls the semantic search controller, builds a `RAGMessageState`, and finally invokes the generative model to
      produce an assistant reply.
    - Helper methods for content supervision, message state creation, and history conversion.
- **MessageState creation** – Depending on request options (`use_rag_supervisor`, `use_content_supervisor`), the
  controller creates the appropriate state objects and links them to the message.

### Serializers (`serializer.py`)

- **ChatSerializer**, **MessageSerializer** – Basic serializers exposing primary fields.
- **DeepMessageSerializer** – Includes nested `MessageState` for full message‑history retrieval.
- **ContentSupervisorStateSerializer**, **RAGStateSerializer**, **MessageStateSerializer** – Provide detailed state
  information when needed.

### API (`api.py`)

| Endpoint             | HTTP Method | Purpose                                                                                                                                                                  |
|----------------------|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **new_chat**         | `POST`      | Create a fresh chat session (`NewChat`). Optional `options`, `collection_name`, `search_options`.                                                                        |
| **add_user_message** | `POST`      | Append a user message, run RAG + generative answer, and return updated history, generation time, and final assistant message (`AddUserMessageToChatWithSystemResponse`). |
| **save_chat**        | `POST`      | Mark a chat as saved (`SetChatStateAsSaved`), optionally make it read‑only, and return the chat hash.                                                                    |
| **get_chat_by_hash** | `GET`       | Retrieve a saved chat (including full message history) using its hash (`GetSavedChatByHash`).                                                                            |
| **chats**            | `GET`       | List all chats belonging to the authenticated user (`ListOfUserChats`).                                                                                                  |

All endpoints use the common decorators:

- `@required_params_exists` – validates request payload.
- `@get_organisation_user` – injects the `OrganisationUser` based on the authenticated Django user.
- `@get_default_language` – resolves the language for error messages and responses.

### Typical Conversation Flow

1. **Start a Chat** – `POST /api/<version>/new_chat/` → receives a new `chat_id` (or hash after saving).
2. **User Message** – `POST /api/<version>/add_user_message/` with `chat_id`, `user_message`, and optional generation
   options (`search_options`, `system_prompt`). The controller: <br>a) Persists the user message. <br>b) Calls
   `generate_assistant_message_cs_rag()` which: <br>– Determines whether the message should trigger a semantic search (
   RAG). <br>– Executes the search via `SearchQueryController`. <br>– Creates a `RAGMessageState` (and optional
   `ContentSupervisorState`). <br>– Generates an assistant reply using the selected generative model. <br>c) Stores the
   assistant message and returns the full updated history. |
3. **Save / Retrieve** – When the user wants to keep the conversation, `POST /api/<version>/save_chat/` marks it as
   saved and optionally read‑only. Later, `GET /api/<version>/get_chat_by_hash/` fetches the stored conversation. |
4. **List Chats** – `GET /api/<version>/chats/` returns a summary of all chats for the user, each with its own message
   history.

### Extending the Chat Functionality

- **Add a new message role** – Extend the `Message.role` choices (e.g., `system_notice`) and update
  `ChatController.generate_assistant_message_cs_rag()` to handle the new role.
- **Custom RAG pipelines** – Override `ChatController._prepare_rag_supervisor_state()` to inject additional
  query‑generation logic, or add new search options to the request payload.
- **Persist additional state** – Create new fields on `MessageState` (e.g., sentiment analysis) and update the
  controller to populate them after each user message.
- **Streaming responses** – Replace the synchronous `generative_answer_local_api_model()` call with an async generator
  that streams partial answers back to the client.

### Important Constants

- `ERROR_MARK_CHAT = "__chat"` – Namespace marker used when registering chat‑specific error codes in `chat.errors`.
"""
test_client.py
--------------

Wire-format tests of :class:`sse_lib.SSEClient`.

They pin the parts of the API contract that are easy to get wrong by hand:
which option field is a JSON string, which one is a JSON object, how the
multipart upload is named, and how the server errors surface in Python.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest

import requests

from sse_lib import (
    SSEAPIError,
    SSEClient,
    SSEConfigError,
    SSEValueError,
    SearchOptions,
    text_document,
)
from sse_lib.client import ENV_API_HOST, ENV_API_TOKEN

from support import api_error, build_client, ok, response


class ClientConfigurationTest(unittest.TestCase):
    def test_defaults_of_a_client(self):
        client = SSEClient("http://localhost:8271", token="t")
        self.assertEqual(client.base_url, "http://localhost:8271/api")
        self.assertEqual(client.language, "pl")
        self.assertEqual(client.token, "t")

    def test_host_may_be_taken_from_the_environment(self):
        os.environ[ENV_API_HOST] = "http://env.host:1234"
        os.environ[ENV_API_TOKEN] = "env-token"
        try:
            client = SSEClient()
            self.assertEqual(client.base_url, "http://env.host:1234/api")
            self.assertEqual(client.token, "env-token")
        finally:
            del os.environ[ENV_API_HOST]
            del os.environ[ENV_API_TOKEN]

    def test_missing_host_is_reported_early(self):
        os.environ.pop(ENV_API_HOST, None)
        with self.assertRaises(SSEConfigError):
            SSEClient()

    def test_unsupported_language_is_rejected(self):
        with self.assertRaises(SSEValueError):
            SSEClient("http://localhost:8271", token="t", language="de")

    def test_versioned_prefix_is_configurable(self):
        client = SSEClient("http://localhost:8271", token="t", api_prefix="api/v1")
        self.assertEqual(
            client.transport.url_for("collections"),
            "http://localhost:8271/api/v1/collections",
        )


class AuthenticationTest(unittest.TestCase):
    def test_login_stores_the_returned_token(self):
        client = build_client([response({"token": "abc123"})], token=None)
        self.assertEqual(client.login("user", "Secret01"), "abc123")
        call = client.fake_session.calls[0]
        self.assertEqual(call.url, "http://sse.test:8271/api/login")
        self.assertEqual(call.json["username"], "user")
        self.assertEqual(call.json["password"], "Secret01")
        self.assertEqual(client.token, "abc123")

    def test_login_happens_lazily_and_only_once(self):
        client = build_client(
            [
                response({"token": "abc123"}),
                ok({"collections": []}),
                ok({"collections": []}),
            ],
            token=None,
            username="user",
            password="Secret01",
        )
        client.list_collections()
        client.list_collections()
        self.assertEqual(len(client.fake_session.calls), 3)
        self.assertEqual(
            client.fake_session.calls[2].headers.get("Authorization"), "Token abc123"
        )

    def test_login_without_credentials_is_reported(self):
        client = build_client([], token=None)
        with self.assertRaises(SSEConfigError):
            client.login()

    def test_login_without_token_in_the_response_is_an_error(self):
        client = build_client([response({})], token=None)
        with self.assertRaises(SSEAPIError):
            client.login("user", "Secret01")

    def test_oauth_code_exchange_stores_both_tokens(self):
        client = build_client(
            [ok({"token": "access", "refresh_token": "refresh"})], token=None
        )
        self.assertEqual(client.login_with_code("code", "state"), "access")
        self.assertEqual(client.fake_session.calls[0].json["code"], "code")
        client._refresh_token = "refresh"
        self.assertEqual(client._refresh_token, "refresh")

    def test_refresh_uses_the_stored_refresh_token(self):
        client = build_client(
            [
                ok({"token": "access", "refresh_token": "refresh"}),
                ok({"token": "access2", "refresh_token": "refresh2"}),
            ],
            token=None,
        )
        client.login_with_code("code", "state")
        self.assertEqual(client.refresh_access_token(), "access2")
        self.assertEqual(
            client.fake_session.calls[1].json, {"refresh_token": "refresh"}
        )

    def test_logout_clears_the_token(self):
        client = build_client([ok({})])
        client.logout()
        self.assertIsNone(client.token)


class CollectionsTest(unittest.TestCase):
    def test_create_collection_sends_the_server_field_names(self):
        client = build_client([ok({"collection_id": 7})])
        collection = client.create_collection(
            "my docs",
            description="Test collection",
            embedder="radlab/polish-bi-encoder-mean",
            reranker="radlab/polish-cross-encoder",
            index_type="HNSW",
        )
        call = client.fake_session.calls[0]
        self.assertEqual(call.url, "http://sse.test:8271/api/new_collection")
        self.assertEqual(call.json["collection_name"], "my docs")
        self.assertEqual(call.json["collection_display_name"], "my docs")
        self.assertEqual(
            call.json["model_embedder"], "radlab/polish-bi-encoder-mean"
        )
        self.assertEqual(call.json["model_reranker"], "radlab/polish-cross-encoder")
        self.assertEqual(call.json["embedder_index_type"], "HNSW")
        self.assertNotIn("group_name", call.json)
        self.assertEqual(collection.id, 7)
        # The server replaces spaces with underscores.
        self.assertEqual(collection.name, "my_docs")

    def test_create_collection_with_group(self):
        client = build_client([ok({"collection_id": 1})])
        client.create_collection(
            "docs",
            "Docs",
            "Description",
            embedder="e",
            reranker="r",
            group_name="analysts",
        )
        self.assertEqual(client.fake_session.calls[0].json["group_name"], "analysts")

    def test_empty_description_is_rejected_before_the_call(self):
        client = build_client([])
        with self.assertRaises(SSEValueError):
            client.create_collection(
                "docs", description="  ", embedder="e", reranker="r"
            )
        self.assertEqual(client.fake_session.calls, [])

    def test_unknown_index_type_is_rejected_before_the_call(self):
        client = build_client([])
        with self.assertRaises(SSEValueError):
            client.create_collection(
                "docs",
                description="d",
                embedder="e",
                reranker="r",
                index_type="DISKANN",
            )

    def test_list_collections(self):
        client = build_client(
            [ok({"collections": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]})]
        )
        collections = client.list_collections()
        self.assertEqual([collection.name for collection in collections], ["a", "b"])
        self.assertEqual(client.fake_session.calls[0].params, {"lang": "pl"})

    def test_get_collection_and_exists(self):
        client = build_client([ok({"collections": [{"id": 1, "name": "a"}]})] * 3)
        self.assertEqual(client.get_collection("a").name, "a")
        self.assertTrue(client.collection_exists("a"))
        self.assertIsNone(client.get_collection("missing"))

    def test_list_categories_sends_the_collection_name(self):
        client = build_client([ok({"categories": ["Law", "Sport"]})])
        self.assertEqual(client.list_categories("my_docs"), ["Law", "Sport"])
        self.assertEqual(
            client.fake_session.calls[0].params,
            {"collection_name": "my_docs", "lang": "pl"},
        )

    def test_list_documents_returns_plain_names(self):
        client = build_client(
            [ok({"documents": [{"name": "a.pdf"}, {"name": "b.pdf"}]})]
        )
        self.assertEqual(client.list_documents("docs"), ["a.pdf", "b.pdf"])

    def test_templates_and_filter_options(self):
        client = build_client(
            [
                ok(
                    {
                        "templates": {
                            "invoices": [{"id": 1, "name": "n", "display": "d"}]
                        }
                    }
                ),
                ok({"templates": {}, "urls": [], "categories": ["Law"]}),
            ]
        )
        templates = client.list_question_templates()
        self.assertEqual(list(templates), ["invoices"])
        self.assertEqual(client.list_filter_options()["categories"], ["Law"])

    def test_business_error_is_raised(self):
        client = build_client(
            [api_error([{"message": "Collection not found or access denied!"}])]
        )
        with self.assertRaises(SSEAPIError):
            client.list_categories("nope")


class IndexingTest(unittest.TestCase):
    def test_upload_files_builds_a_multipart_request(self):
        client = build_client([ok({"dir_hash": "h", "is_indexed": True})])
        result = client.upload_files(
            "my_docs",
            [("file.txt", io.BytesIO(b"content"))],
            indexing_options={"max_tokens_in_chunk": 200},
        )
        call = client.fake_session.calls[0]
        self.assertEqual(call.url, "http://sse.test:8271/api/upload_and_index_files")
        self.assertIsNone(call.json)
        self.assertEqual(call.data["collection_name"], "my_docs")
        # The server json.loads() this field, so it has to be a string.
        self.assertIsInstance(call.data["indexing_options"], str)
        self.assertEqual(
            json.loads(call.data["indexing_options"])["max_tokens_in_chunk"], 200
        )
        self.assertEqual(call.data["lang"], "pl")
        field_name, (uploaded_name, uploaded, content_type) = call.files[0]
        self.assertEqual(field_name, "files[]")
        self.assertEqual(uploaded_name, "file.txt")
        self.assertEqual(content_type, "text/plain")
        self.assertEqual(uploaded.read(), b"content")
        self.assertTrue(result.is_indexed)

    def test_upload_files_opens_and_closes_paths(self):
        client = build_client([ok({})])
        handle = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        handle.write(b"data")
        handle.close()
        try:
            client.upload_files("docs", handle.name)
        finally:
            os.unlink(handle.name)
        uploaded = client.fake_session.calls[0].files[0][1]
        self.assertEqual(uploaded[0], os.path.basename(handle.name))
        self.assertTrue(uploaded[1].closed)

    def test_upload_without_files_is_rejected(self):
        client = build_client([])
        with self.assertRaises(SSEValueError):
            client.upload_files("docs", [])

    def test_index_documents_converts_plain_strings(self):
        client = build_client([ok({"indexed_documents": 2, "indexed_chunks": 3})])
        result = client.index_documents(
            "my_docs", ["first text", "second text"], category="Notes"
        )
        payload = client.fake_session.calls[0].json
        self.assertEqual(
            payload["texts[]"],
            [
                text_document("first text", category="Notes"),
                text_document("second text", category="Notes"),
            ],
        )
        # Here the server reads the options as an object, not as a JSON string.
        self.assertIsInstance(payload["indexing_options"], dict)
        self.assertEqual(payload["collection_name"], "my_docs")
        self.assertEqual(result.indexed_documents, 2)
        self.assertEqual(result.indexed_chunks, 3)

    def test_index_documents_keeps_prepared_documents(self):
        client = build_client([ok({"indexed_documents": 1, "indexed_chunks": 1})])
        prepared = text_document("page", name="report", category="Finance")
        client.index_documents("docs", prepared)
        self.assertEqual(client.fake_session.calls[0].json["texts[]"], [prepared])

    def test_index_documents_accepts_indexing_keywords(self):
        client = build_client([ok({})])
        client.index_documents("docs", "text", max_tokens_in_chunk=128)
        options = client.fake_session.calls[0].json["indexing_options"]
        self.assertEqual(options["max_tokens_in_chunk"], 128)
        self.assertEqual(
            set(options),
            {
                "prepare_proper_pages",
                "merge_document_pages",
                "clear_text",
                "use_text_denoiser",
                "max_tokens_in_chunk",
                "number_of_overlap_tokens",
                "check_text_lang",
            },
        )


class SearchTest(unittest.TestCase):
    SEARCH_BODY = {
        "results": {
            "query": "data retention",
            "stats": {"a.pdf": {"score_weighted": 0.8}},
            "detailed_results": {
                "0": {
                    "score": 0.91,
                    "document_name": "a.pdf",
                    "relative_filepath": "/docs/a.pdf",
                    "page_number": 3,
                    "text_number": 1,
                    "language": "en",
                    "text_str": "Retention policy ...",
                    "left_context": "",
                    "right_context": "",
                }
            },
            "structured_results": [],
        },
        "query_response_id": 42,
    }

    def test_search_sends_options_as_a_json_string(self):
        client = build_client([ok(self.SEARCH_BODY)])
        response_body = client.search(
            "my_docs",
            "data retention",
            SearchOptions(max_results=20, rerank_results=True),
        )
        call = client.fake_session.calls[0]
        self.assertEqual(call.url, "http://sse.test:8271/api/search_with_options")
        self.assertEqual(call.json["collection_name"], "my_docs")
        self.assertEqual(call.json["query_str"], "data retention")
        self.assertIsInstance(call.json["options"], str)
        self.assertEqual(
            json.loads(call.json["options"]),
            {"max_results": 20, "rerank_results": True},
        )
        self.assertEqual(response_body.query_response_id, 42)
        self.assertEqual(response_body.query, "data retention")
        hit = response_body.hits[0]
        self.assertEqual(hit.document_name, "a.pdf")
        self.assertEqual(hit.score, 0.91)
        self.assertEqual(hit.text, "Retention policy ...")

    def test_search_options_may_be_plain_keywords(self):
        client = build_client([ok(self.SEARCH_BODY), ok(self.SEARCH_BODY)])
        client.search("docs", "q", max_results=5, hybrid_search=False)
        sent = json.loads(client.fake_session.calls[0].json["options"])
        self.assertEqual(sent, {"max_results": 5, "hybrid_search": False})

    def test_client_defaults_are_merged_with_the_call_options(self):
        client = build_client(
            [ok(self.SEARCH_BODY)],
            default_search_options=SearchOptions(
                max_results=100, hybrid_search=False
            ),
        )
        client.search("docs", "q", SearchOptions(max_results=5))
        sent = json.loads(client.fake_session.calls[0].json["options"])
        self.assertEqual(sent, {"max_results": 5, "hybrid_search": False})

    def test_language_detection_flag_is_optional(self):
        client = build_client([ok(self.SEARCH_BODY), ok(self.SEARCH_BODY)])
        client.search("docs", "q")
        self.assertNotIn(
            "ignore_question_lang_detect", client.fake_session.calls[0].json
        )
        client.search("docs", "q", ignore_question_lang_detect=True)
        self.assertTrue(
            client.fake_session.calls[1].json["ignore_question_lang_detect"]
        )

    def test_empty_query_is_rejected(self):
        client = build_client([])
        with self.assertRaises(SSEValueError):
            client.search("docs", "   ")

    def test_template_prompts_are_exposed(self):
        body = dict(self.SEARCH_BODY)
        body["template_prompts"] = ["answer like a table"]
        client = build_client([ok(body)])
        self.assertEqual(
            client.search("docs", "q").template_prompts, ["answer like a table"]
        )


class GenerationTest(unittest.TestCase):
    def test_generate_answer_sends_query_options_as_a_string(self):
        client = build_client(
            [ok({"response_id": 5, "answer": "42", "generation_time": 1.5})]
        )
        answer = client.generate_answer(
            42,
            {
                "generative_model": "radlab/pLLama-3-8B-DPO-L",
                "percentage_rank_mass": 0.3,
            },
            system_prompt="Answer briefly.",
        )
        call = client.fake_session.calls[0]
        self.assertEqual(call.url, "http://sse.test:8271/api/generative_answer")
        self.assertEqual(call.json["query_response_id"], 42)
        self.assertIsInstance(call.json["query_options"], str)
        self.assertEqual(
            json.loads(call.json["query_options"])["generative_model"],
            "radlab/pLLama-3-8B-DPO-L",
        )
        self.assertEqual(call.json["system_prompt"], "Answer briefly.")
        self.assertEqual(answer.answer, "42")
        self.assertEqual(answer.response_id, 5)
        self.assertEqual(answer.generation_time, 1.5)

    def test_empty_prompts_are_not_sent(self):
        client = build_client([ok({"answer": "x"})])
        client.generate_answer(1, {"generative_model": "m"})
        self.assertNotIn("system_prompt", client.fake_session.calls[0].json)
        self.assertNotIn("query_instruction", client.fake_session.calls[0].json)

    def test_rate_answer_sends_all_required_fields(self):
        client = build_client([ok({}), ok({})])
        client.rate_answer(9, 4, 5)
        payload = client.fake_session.calls[0].json
        self.assertEqual(payload["answer_response_id"], 9)
        self.assertEqual(payload["rate_value"], 4)
        self.assertEqual(payload["rate_value_max"], 5)
        self.assertNotIn("rate_comment", payload)
        client.rate_answer(9, 4, 5, comment="good")
        self.assertEqual(client.fake_session.calls[1].json["rate_comment"], "good")

    def test_model_listings(self):
        client = build_client(
            [
                ok({"models": ["radlab/polish-bi-encoder-mean"]}),
                ok({"models": ["radlab/polish-cross-encoder"]}),
                ok({"radlab/model": {"host": "http://llm:8080"}}),
            ]
        )
        self.assertEqual(client.list_embedders(), ["radlab/polish-bi-encoder-mean"])
        self.assertEqual(client.list_rerankers(), ["radlab/polish-cross-encoder"])
        self.assertEqual(
            client.list_generative_models(),
            {"radlab/model": {"host": "http://llm:8080"}},
        )


class ChatTest(unittest.TestCase):
    def test_new_chat_sends_objects_not_json_strings(self):
        client = build_client([ok({"chat": {"id": 3, "collection": 1}})])
        chat = client.new_chat(
            "my_docs",
            options={"generative_model": "m"},
            search_options={"max_results": 10},
        )
        payload = client.fake_session.calls[0].json
        self.assertEqual(payload["collection_name"], "my_docs")
        self.assertEqual(payload["options"], {"generative_model": "m"})
        self.assertEqual(payload["search_options"], {"max_results": 10})
        self.assertEqual(chat.id, 3)

    def test_send_chat_message_parses_the_reply(self):
        client = build_client(
            [
                ok(
                    {
                        "generation_time": 2.0,
                        "history": [{"id": 1, "role": "user", "text": "q"}],
                        "last_state": None,
                        "generated_assistant_message": "a",
                    }
                )
            ]
        )
        reply = client.send_chat_message(
            3, "q", {"generative_model": "m"}, collection="docs"
        )
        payload = client.fake_session.calls[0].json
        self.assertEqual(payload["chat_id"], 3)
        self.assertEqual(payload["user_message"], "q")
        self.assertEqual(payload["options"], {"generative_model": "m"})
        self.assertEqual(payload["collection_name"], "docs")
        self.assertEqual(reply.assistant_message, "a")
        self.assertEqual([message.text for message in reply.history], ["q"])

    def test_send_chat_message_builds_options_from_keywords(self):
        client = build_client([ok({"generated_assistant_message": "a"})])
        client.send_chat_message(
            1, "q", collection="docs", generative_model="m", answer_language="pl"
        )
        self.assertEqual(
            client.fake_session.calls[0].json["collection_name"], "docs"
        )
        self.assertEqual(
            client.fake_session.calls[0].json["options"],
            {"generative_model": "m", "answer_language": "pl"},
        )

    def test_empty_message_is_rejected(self):
        client = build_client([])
        with self.assertRaises(SSEValueError):
            client.send_chat_message(1, "")

    def test_save_chat_returns_the_hash(self):
        client = build_client([ok({"chat_hash": "deadbeef"})])
        self.assertEqual(client.save_chat(1), "deadbeef")
        self.assertEqual(client.fake_session.calls[0].json["read_only"], True)

    def test_get_chat_by_hash_uses_query_parameters(self):
        client = build_client(
            [
                ok(
                    {
                        "chat_id": 1,
                        "is_read_only": True,
                        "chat_history": [{"text": "hi"}],
                    }
                )
            ]
        )
        history = client.get_chat_by_hash("deadbeef")
        self.assertEqual(
            client.fake_session.calls[0].params,
            {"chat_hash": "deadbeef", "lang": "pl"},
        )
        self.assertEqual([message.text for message in history.messages], ["hi"])
        self.assertTrue(history.is_read_only)

    def test_list_chats(self):
        client = build_client([ok({"history": [{"chat_id": 1}, {"chat_id": 2}]})])
        self.assertEqual(
            [history.chat_id for history in client.list_chats()], [1, 2]
        )


class TransportErrorsTest(unittest.TestCase):
    def test_connection_problem_is_wrapped(self):
        client = build_client([requests.ConnectionError("refused")])
        with self.assertRaises(Exception) as caught:
            client.list_collections()
        self.assertIn("sse.test", str(caught.exception))

    def test_http_error_keeps_the_status_code(self):
        from sse_lib import SSEHTTPError

        client = build_client([response({"detail": "no"}, status_code=404)])
        with self.assertRaises(SSEHTTPError) as caught:
            client.list_collections()
        self.assertEqual(caught.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()

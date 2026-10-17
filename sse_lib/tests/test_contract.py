"""
test_contract.py
----------------

Guards against the library drifting away from the Django routes.

The server side is the source of truth: every route registered with
``prepare_api_url("...")`` in ``sse_api/*/urls.py`` has to be known by
``sse_lib.endpoints`` and reachable through the client.  The check is skipped
when the client is used outside of the server repository (installed wheel),
because then the Django tree is simply not there.
"""

from __future__ import annotations

import inspect
import re
import unittest
from pathlib import Path

from dataclasses import fields

from sse_lib import (
    ChatOptions,
    GenerativeOptions,
    IndexingOptions,
    SearchOptions,
    SSEClient,
    endpoints,
)

SERVER_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "sse_api"
ROUTE_PATTERN = re.compile(r'prepare_api_url\("([^"]+)"\)')


def server_endpoints() -> set:
    names = set()
    for urls_file in SERVER_ROOT.glob("*/urls.py"):
        names |= set(ROUTE_PATTERN.findall(urls_file.read_text(encoding="utf-8")))
    return names


def _consumed_keys(*patterns: str, only: tuple = ()) -> set:
    """Option keys the server reads from its controllers.

    A key is *known* to the backend as soon as a controller calls
    ``<name>.get("<key>"`` or ``<name>["<key>"]`` on it.  That is the source
    of truth for the parameter-level contract: the validators of the views
    tolerate a couple of legacy keys (``documents_names``) which nothing
    consumes, and those must not grow into options of the library.

    ``only`` restricts the scan to the controllers which own a given
    option dictionary: a bare ``options.get(...)`` pattern also matches
    unrelated settings (management commands, the content supervisor), so the
    chat switches are scanned in the chat controller only.
    """
    keys = set()
    for pattern in patterns:
        for path in SERVER_ROOT.rglob("*.py"):
            if only and not any(str(path).endswith(part) for part in only):
                continue
            keys |= set(re.findall(pattern, path.read_text(encoding="utf-8")))
    return keys


def _option_fields(cls) -> set:
    return {meta.name for meta in fields(cls) if meta.name != "extra"}


@unittest.skipUnless(
    SERVER_ROOT.is_dir(), "sse_api is not available (installed package)"
)
class ApiContractTest(unittest.TestCase):
    def test_every_registered_route_is_known(self):
        self.assertEqual(
            server_endpoints() - endpoints.ALL_ENDPOINTS,
            set(),
            "sse_lib.endpoints is missing routes registered by the backend",
        )

    def test_the_library_invents_no_routes(self):
        self.assertEqual(
            endpoints.ALL_ENDPOINTS - server_endpoints(),
            set(),
            "sse_lib.endpoints knows endpoints the backend does not expose",
        )

    def test_every_endpoint_is_reachable_from_the_client(self):
        source = Path(inspect.getsourcefile(SSEClient) or "").read_text(
            encoding="utf-8"
        )
        constants = {
            name
            for name, value in vars(endpoints).items()
            if name.isupper() and value in endpoints.ALL_ENDPOINTS
        }
        self.assertEqual(len(constants), len(endpoints.ALL_ENDPOINTS))
        for name in sorted(constants):
            self.assertIn(
                f"endpoints.{name}",
                source,
                f"{name} is not used by any SSEClient method",
            )

    def test_public_facade_has_the_documented_methods(self):
        for method in (
            "login",
            "create_collection",
            "list_collections",
            "list_documents",
            "list_document_details",
            "upload_files",
            "index_documents",
            "search",
            "generate_answer",
            "rate_answer",
            "list_search_history",
            "get_search_history",
            "delete_search_history",
            "new_chat",
            "send_chat_message",
            "save_chat",
            "get_chat_by_hash",
            "list_chats",
            "list_embedders",
            "list_rerankers",
            "embedders_info",
            "rerankers_info",
            "list_generative_models",
        ):
            self.assertTrue(callable(getattr(SSEClient, method)), method)

    def test_search_options_match_the_server_pipeline(self):
        # The hybrid search of ``engine.controllers.search.semantic`` is the
        # one path the public API goes through; the keys are all distinct
        # enough to be found across the server tree.
        consumed = _consumed_keys(r'search_params\.get\("([a-z_]+)"')
        self.assertEqual(
            _option_fields(SearchOptions),
            consumed,
            "SearchOptions drifted away from the options the server reads",
        )

    def test_generative_options_match_the_server(self):
        consumed = _consumed_keys(
            r'query_options\.get\("([a-z_]+)"',
            r'query_options\["([a-z_]+)"\]',
        )
        self.assertEqual(
            _option_fields(GenerativeOptions),
            consumed,
            "GenerativeOptions drifted away from the options the server reads",
        )

    def test_indexing_options_match_the_server(self):
        consumed = _consumed_keys(
            r'indexing_options\["([a-z_]+)"\]',
            only=(
                "data/controllers/upload.py",
                "engine/controllers/database/relational_db.py",
            ),
        )
        self.assertEqual(
            _option_fields(IndexingOptions),
            consumed,
            "IndexingOptions drifted away from the options the server reads",
        )

    def test_chat_options_match_the_server(self):
        consumed = _consumed_keys(
            r'options\.get\("([a-z_]+)"',
            only=("chat/controllers/message.py",),
        )
        self.assertEqual(
            _option_fields(ChatOptions),
            consumed,
            "ChatOptions drifted away from the switches the chat reads",
        )


class PublicApiTest(unittest.TestCase):
    def test_everything_exported_is_importable(self):
        import sse_lib

        for name in sse_lib.__all__:
            self.assertTrue(hasattr(sse_lib, name), name)

    def test_new_public_names_are_exported(self):
        import sse_lib

        for name in ("DocumentRef", "ChatOptions"):
            self.assertIn(name, sse_lib.__all__)
            self.assertTrue(hasattr(sse_lib, name), name)

    def test_version_looks_semantic(self):
        import sse_lib

        self.assertRegex(sse_lib.__version__, r"^\d+\.\d+\.\d+")


if __name__ == "__main__":
    unittest.main()

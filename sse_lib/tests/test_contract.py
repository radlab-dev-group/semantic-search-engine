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

from sse_lib import SSEClient, endpoints

SERVER_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "sse_api"
ROUTE_PATTERN = re.compile(r'prepare_api_url\("([^"]+)"\)')


def server_endpoints() -> set:
    names = set()
    for urls_file in SERVER_ROOT.glob("*/urls.py"):
        names |= set(ROUTE_PATTERN.findall(urls_file.read_text(encoding="utf-8")))
    return names


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
            "upload_files",
            "index_documents",
            "search",
            "generate_answer",
            "rate_answer",
            "new_chat",
            "send_chat_message",
            "save_chat",
            "get_chat_by_hash",
            "list_chats",
            "list_embedders",
            "list_rerankers",
            "list_generative_models",
        ):
            self.assertTrue(callable(getattr(SSEClient, method)), method)


class PublicApiTest(unittest.TestCase):
    def test_everything_exported_is_importable(self):
        import sse_lib

        for name in sse_lib.__all__:
            self.assertTrue(hasattr(sse_lib, name), name)

    def test_version_looks_semantic(self):
        import sse_lib

        self.assertRegex(sse_lib.__version__, r"^\d+\.\d+\.\d+")


if __name__ == "__main__":
    unittest.main()

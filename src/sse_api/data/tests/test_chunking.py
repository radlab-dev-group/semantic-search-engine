"""
Chunking of a collection.

How the texts of a collection are split is decided once, when the collection
is created, and every indexing call is resolved against it:

* the request wins when it declares the values,
* otherwise the settings of the collection are used,
* otherwise the global defaults, which are then remembered on the collection,
* a strict collection refuses indexing options which differ from its own.

A chunk also has to fit the embedder of the collection, see
``CHUNKING_TOO_LONG``.  Tests use a real (in memory) relational database;
Milvus and the file handling are never touched.
"""

import json
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.test import SimpleTestCase, TestCase
from rest_framework.test import APIRequestFactory, force_authenticate

from sse_api.core.validation import chunking as chunking_validator
from sse_api.core.validation import indexing_options as indexing_options_validator
from sse_api.data.api import (
    AddAndIndexTextsFromEP,
    ListCollections,
    ListFilteringOptions,
    NewCollection,
    UploadAndIndexFilesToCollection,
)
from sse_api.data.controllers.chunking import (
    ChunkingMismatch,
    ChunkingTooLong,
    InvalidChunking,
    check_chunk_size,
    declared_chunking,
    plan_chunking_fix,
    resolve_chunking,
)
from sse_api.data.core.constants import (
    CHUNKING_METADATA_KEY,
    DEFAULT_MAX_TOKENS_IN_CHUNK,
    DEFAULT_NUMBER_OF_OVERLAP_TOKENS,
    MAX_TOKENS_IN_CHUNK_KEY,
    MAX_TOKENS_IN_CHUNK_LIMIT,
    NUMBER_OF_OVERLAP_TOKENS_KEY,
    STRICT_CHUNKING_METADATA_KEY,
)
from sse_api.data.management.commands import check_collection_chunking
from sse_api.data.models import (
    CollectionOfDocuments,
    Document,
    DocumentPage,
    DocumentPageText,
)
from sse_api.system.models import Organisation, OrganisationUser

INDEXING_BASE = {
    "prepare_proper_pages": True,
    "merge_document_pages": False,
    "clear_text": True,
    "use_text_denoiser": False,
    "check_text_lang": False,
}
TEXT_DOCUMENT = [
    {
        "filepath": "text",
        "relative_filepath": "text",
        "category": "Notes",
        "pages": [{"page_content": "Some text to index."}],
    }
]


class ChunkingTestCase(TestCase):
    """Organisation, user, collection and the model input length patches."""

    def setUp(self):
        self.factory = APIRequestFactory()
        self.organisation = Organisation.objects.create(name="chunking org")
        self.user = User.objects.create_user(username="chunking-user")
        self.organisation_user = OrganisationUser.objects.create(
            auth_user=self.user, organisation=self.organisation
        )
        self.collection = self.make_collection("collection")
        self._patchers = []

    def tearDown(self):
        for patcher in reversed(self._patchers):
            patcher.stop()
        super().tearDown()

    def make_collection(self, name, chunking=None, strict=False, **kwargs):
        metadata = {}
        if chunking is not None:
            metadata[CHUNKING_METADATA_KEY] = {
                MAX_TOKENS_IN_CHUNK_KEY: chunking[0],
                NUMBER_OF_OVERLAP_TOKENS_KEY: chunking[1],
            }
        if strict:
            metadata[STRICT_CHUNKING_METADATA_KEY] = True
        return CollectionOfDocuments.objects.create(
            name=name,
            display_name=name.title(),
            created_by=self.organisation_user,
            metadata_json=metadata or None,
            **kwargs,
        )

    def pretend_model_lengths(self, embedder=None, reranker=None):
        """Declare the input length of the models used by the resolver."""
        for name, value in (
            ("get_embedder_max_model_length", embedder),
            ("get_reranker_max_model_length", reranker),
        ):
            patcher = patch(
                f"sse_api.data.controllers.chunking.EmbeddingModelsConfig.{name}",
                return_value=value,
            )
            patcher.start()
            self._patchers.append(patcher)

    def invoke(self, view, payload=None, method="post"):
        if method == "get":
            request = self.factory.get("/", payload or {})
        else:
            request = self.factory.post(
                "/",
                payload,
                format=(
                    "multipart"
                    if payload
                    and any(hasattr(value, "read") for value in payload.values())
                    else "json"
                ),
            )
        force_authenticate(request, user=self.user)
        return view.as_view()(request)

    def reload(self, collection):
        return CollectionOfDocuments.objects.get(pk=collection.pk)


class CollectionModelChunkingTests(ChunkingTestCase):
    def test_collection_without_settings_reads_as_empty(self):
        self.assertEqual(self.collection.chunking(), {})
        self.assertFalse(self.collection.strict_chunking)

    def test_defaults_are_reported_without_touching_the_collection(self):
        self.assertEqual(
            self.collection.chunking_with_defaults(),
            {
                MAX_TOKENS_IN_CHUNK_KEY: DEFAULT_MAX_TOKENS_IN_CHUNK,
                NUMBER_OF_OVERLAP_TOKENS_KEY: DEFAULT_NUMBER_OF_OVERLAP_TOKENS,
            },
        )
        self.assertEqual(self.reload(self.collection).metadata_json, None)

    def test_set_chunking_keeps_the_other_metadata(self):
        self.collection.metadata_json = {"created_from": "UI"}
        self.collection.set_chunking(300, 30)
        self.collection.strict_chunking = True
        self.assertEqual(
            self.collection.chunking(),
            {MAX_TOKENS_IN_CHUNK_KEY: 300, NUMBER_OF_OVERLAP_TOKENS_KEY: 30},
        )
        self.assertTrue(self.collection.strict_chunking)
        self.assertEqual(self.collection.metadata_json["created_from"], "UI")

    def test_broken_metadata_is_not_a_chunking(self):
        for metadata in ({"chunking": "many"}, [1, 2], {"chunking": {}}):
            with self.subTest(metadata=metadata):
                collection = CollectionOfDocuments(metadata_json=metadata)
                self.assertEqual(collection.chunking(), {})
                self.assertFalse(collection.strict_chunking)


class ResolveChunkingTests(ChunkingTestCase):
    def setUp(self):
        super().setUp()
        self.pretend_model_lengths(embedder=2048, reranker=512)

    def test_request_wins_over_the_collection(self):
        collection = self.make_collection("declared", chunking=(500, 50))
        options = resolve_chunking(
            collection,
            {
                **INDEXING_BASE,
                MAX_TOKENS_IN_CHUNK_KEY: 200,
                NUMBER_OF_OVERLAP_TOKENS_KEY: 20,
            },
        )
        self.assertEqual(options[MAX_TOKENS_IN_CHUNK_KEY], 200)
        self.assertEqual(options[NUMBER_OF_OVERLAP_TOKENS_KEY], 20)
        # A non-strict collection is not rewritten by a single request.
        self.assertEqual(
            self.reload(collection).chunking(),
            {MAX_TOKENS_IN_CHUNK_KEY: 500, NUMBER_OF_OVERLAP_TOKENS_KEY: 50},
        )

    def test_settings_of_the_collection_are_used(self):
        collection = self.make_collection("declared", chunking=(500, 50))
        options = resolve_chunking(collection, dict(INDEXING_BASE))
        self.assertEqual(options[MAX_TOKENS_IN_CHUNK_KEY], 500)
        self.assertEqual(options[NUMBER_OF_OVERLAP_TOKENS_KEY], 50)

    def test_defaults_are_bound_onto_a_fresh_collection(self):
        options = resolve_chunking(self.collection, dict(INDEXING_BASE))
        self.assertEqual(
            options[MAX_TOKENS_IN_CHUNK_KEY], DEFAULT_MAX_TOKENS_IN_CHUNK
        )
        self.assertEqual(
            options[NUMBER_OF_OVERLAP_TOKENS_KEY],
            DEFAULT_NUMBER_OF_OVERLAP_TOKENS,
        )
        self.assertEqual(
            self.reload(self.collection).chunking(),
            {
                MAX_TOKENS_IN_CHUNK_KEY: DEFAULT_MAX_TOKENS_IN_CHUNK,
                NUMBER_OF_OVERLAP_TOKENS_KEY: DEFAULT_NUMBER_OF_OVERLAP_TOKENS,
            },
        )

    def test_binding_can_be_switched_off(self):
        resolve_chunking(self.collection, dict(INDEXING_BASE), bind_defaults=False)
        self.assertEqual(self.reload(self.collection).metadata_json, None)

    def test_request_declared_chunking_becomes_the_setting_of_the_collection(self):
        options = resolve_chunking(
            self.collection,
            {**INDEXING_BASE, MAX_TOKENS_IN_CHUNK_KEY: 640},
        )
        self.assertEqual(options[MAX_TOKENS_IN_CHUNK_KEY], 640)
        self.assertEqual(
            self.reload(self.collection).chunking()[MAX_TOKENS_IN_CHUNK_KEY], 640
        )

    def test_overlap_alone_keeps_the_chunk_of_the_collection(self):
        collection = self.make_collection("declared", chunking=(500, 50))
        options = resolve_chunking(
            collection, {**INDEXING_BASE, NUMBER_OF_OVERLAP_TOKENS_KEY: 25}
        )
        self.assertEqual(options[MAX_TOKENS_IN_CHUNK_KEY], 500)
        self.assertEqual(options[NUMBER_OF_OVERLAP_TOKENS_KEY], 25)

    def test_strict_collection_refuses_other_values(self):
        collection = self.make_collection("strict", chunking=(500, 50), strict=True)
        with self.assertRaises(ChunkingMismatch) as caught:
            resolve_chunking(
                collection,
                {**INDEXING_BASE, MAX_TOKENS_IN_CHUNK_KEY: 400},
            )
        self.assertEqual(caught.exception.declared[MAX_TOKENS_IN_CHUNK_KEY], 500)
        self.assertEqual(caught.exception.requested[MAX_TOKENS_IN_CHUNK_KEY], 400)

    def test_strict_collection_accepts_its_own_values(self):
        collection = self.make_collection("strict", chunking=(500, 50), strict=True)
        options = resolve_chunking(
            collection,
            {
                **INDEXING_BASE,
                MAX_TOKENS_IN_CHUNK_KEY: 500,
                NUMBER_OF_OVERLAP_TOKENS_KEY: 50,
            },
        )
        self.assertEqual(options[MAX_TOKENS_IN_CHUNK_KEY], 500)

    def test_strict_collection_refuses_an_overlap_alone(self):
        collection = self.make_collection("strict", chunking=(500, 50), strict=True)
        with self.assertRaises(ChunkingMismatch):
            resolve_chunking(
                collection, {**INDEXING_BASE, NUMBER_OF_OVERLAP_TOKENS_KEY: 0}
            )

    def test_chunk_which_does_not_fit_the_embedder_is_refused(self):
        collection = self.make_collection(
            "declared", chunking=(4096, 0), model_embedder="test-embedder"
        )
        with self.assertRaises(ChunkingTooLong) as caught:
            resolve_chunking(collection, dict(INDEXING_BASE))
        self.assertEqual(caught.exception.max_model_length, 2048)

    def test_sanity_limit_applies_without_a_model_limit(self):
        self.pretend_model_lengths(embedder=None, reranker=None)
        with self.assertRaises(ChunkingTooLong):
            resolve_chunking(
                self.collection,
                {
                    **INDEXING_BASE,
                    MAX_TOKENS_IN_CHUNK_KEY: MAX_TOKENS_IN_CHUNK_LIMIT + 1,
                },
                bind_defaults=False,
            )

    def test_overlap_of_the_collection_must_stay_inside_the_chunk(self):
        collection = self.make_collection("impossible", chunking=(100, 200))
        with self.assertRaises(InvalidChunking):
            resolve_chunking(collection, dict(INDEXING_BASE), bind_defaults=False)

    def test_values_of_a_broken_collection_are_read_as_defaults(self):
        collection = self.make_collection("broken")
        collection.metadata_json = {
            CHUNKING_METADATA_KEY: {
                MAX_TOKENS_IN_CHUNK_KEY: "wide",
                NUMBER_OF_OVERLAP_TOKENS_KEY: None,
            }
        }
        options = resolve_chunking(collection, dict(INDEXING_BASE))
        self.assertEqual(
            options[MAX_TOKENS_IN_CHUNK_KEY], DEFAULT_MAX_TOKENS_IN_CHUNK
        )

    def test_declared_chunking_reads_only_what_is_given(self):
        self.assertEqual(declared_chunking(dict(INDEXING_BASE)), (None, None))
        self.assertEqual(
            declared_chunking(
                {MAX_TOKENS_IN_CHUNK_KEY: 128, NUMBER_OF_OVERLAP_TOKENS_KEY: 8}
            ),
            (128, 8),
        )
        self.assertEqual(declared_chunking(None), (None, None))


class CheckChunkSizeTests(ChunkingTestCase):
    def test_chunk_above_the_embedder_input_is_refused(self):
        self.pretend_model_lengths(embedder=512)
        with self.assertRaises(ChunkingTooLong):
            check_chunk_size(513, embedder_model="model")

    def test_unknown_model_length_is_no_limit(self):
        check_chunk_size(100000, embedder_model="model-without-config")

    def test_reranker_budget_is_only_reported(self):
        self.pretend_model_lengths(embedder=8192, reranker=512)
        with self.assertLogs(level="WARNING") as logs:
            check_chunk_size(600, embedder_model="model", reranker_model="reranker")
        self.assertIn("reranker", "\n".join(logs.output))

    def test_chunk_fitting_the_reranker_is_quiet(self):
        self.pretend_model_lengths(embedder=8192, reranker=512)
        try:
            with self.assertNoLogs(level="WARNING"):
                check_chunk_size(300, embedder_model="model", reranker_model="r")
        except AttributeError:  # Python < 3.10
            check_chunk_size(300, embedder_model="model", reranker_model="r")

    def test_not_a_number_is_an_invalid_chunk(self):
        for value in (0, -1, "500", True, None, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(InvalidChunking):
                    check_chunk_size(value)


class ChunkingValidatorTests(SimpleTestCase):
    def test_complete_chunking(self):
        self.assertEqual(
            chunking_validator(
                {"max_tokens_in_chunk": 400, "number_of_overlap_tokens": 40}
            ),
            {"max_tokens_in_chunk": 400, "number_of_overlap_tokens": 40},
        )

    def test_overlap_defaults_to_none(self):
        self.assertEqual(
            chunking_validator({"max_tokens_in_chunk": 400}),
            {"max_tokens_in_chunk": 400, "number_of_overlap_tokens": 0},
        )

    def test_json_string_is_accepted(self):
        self.assertEqual(
            chunking_validator(json.dumps({"max_tokens_in_chunk": 20}))[
                "max_tokens_in_chunk"
            ],
            20,
        )

    def test_values_which_are_not_a_split_are_refused(self):
        for value in (
            {},
            {"max_tokens_in_chunk": 0},
            {"max_tokens_in_chunk": True},
            {"max_tokens_in_chunk": 100, "number_of_overlap_tokens": 100},
            {"max_tokens_in_chunk": 100, "number_of_overlap_tokens": -1},
            {"max_tokens_in_chunk": 100, "number_of_overlap_tokens": 1.5},
            "[]",
            7,
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    chunking_validator(value)

    def test_indexing_options_may_leave_chunking_out(self):
        self.assertEqual(
            indexing_options_validator(dict(INDEXING_BASE)), dict(INDEXING_BASE)
        )

    def test_indexing_options_still_check_what_is_given(self):
        for options in (
            {**INDEXING_BASE, "max_tokens_in_chunk": 0},
            {**INDEXING_BASE, "max_tokens_in_chunk": True},
            {
                **INDEXING_BASE,
                "max_tokens_in_chunk": 100,
                "number_of_overlap_tokens": 100,
            },
            {**INDEXING_BASE, "number_of_overlap_tokens": "wide"},
        ):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    indexing_options_validator(options)

    def test_overlap_without_a_chunk_size_is_left_to_the_collection(self):
        options = indexing_options_validator(
            {**INDEXING_BASE, "number_of_overlap_tokens": 5}
        )
        self.assertEqual(options["number_of_overlap_tokens"], 5)
        self.assertNotIn("max_tokens_in_chunk", options)


class NewCollectionEndpointTests(ChunkingTestCase):
    def base_payload(self, **kwargs):
        payload = {
            "collection_name": "new-collection",
            "collection_display_name": "New collection",
            "collection_description": "Chunking tests",
            "model_embedder": "test-embedder",
            "model_reranker": "test-reranker",
            "embedder_index_type": "HNSW",
        }
        payload.update(kwargs)
        return payload

    def create(self, **kwargs):
        with (
            patch("sse_api.data.api.SemanticDBController"),
            patch(
                "sse_api.engine.controllers.database.relational_db."
                "EmbeddingModelsConfig.get_embedder_vector_size",
                return_value=8,
            ),
        ):
            return self.invoke(NewCollection, self.base_payload(**kwargs))

    def test_declared_chunking_is_stored_and_reported(self):
        response = self.create(
            chunking={"max_tokens_in_chunk": 400, "number_of_overlap_tokens": 40},
            strict_chunking=True,
        )
        self.assertTrue(response.data["status"])
        self.assertEqual(
            response.data["body"]["chunking"],
            {"max_tokens_in_chunk": 400, "number_of_overlap_tokens": 40},
        )
        self.assertTrue(response.data["body"]["strict_chunking"])
        collection = self.reload(
            CollectionOfDocuments.objects.get(name="new-collection")
        )
        self.assertEqual(
            collection.chunking(),
            {"max_tokens_in_chunk": 400, "number_of_overlap_tokens": 40},
        )
        self.assertTrue(collection.strict_chunking)

    def test_collection_without_chunking_keeps_metadata_empty(self):
        response = self.create()
        self.assertTrue(response.data["status"])
        # The defaults are reported, but only the first indexing call binds
        # them onto the collection.
        self.assertEqual(
            response.data["body"]["chunking"],
            {
                "max_tokens_in_chunk": DEFAULT_MAX_TOKENS_IN_CHUNK,
                "number_of_overlap_tokens": DEFAULT_NUMBER_OF_OVERLAP_TOKENS,
            },
        )
        collection = CollectionOfDocuments.objects.get(name="new-collection")
        self.assertEqual(collection.metadata_json, None)

    def test_chunk_which_does_not_fit_the_embedder_is_refused(self):
        self.pretend_model_lengths(embedder=512)
        response = self.create(chunking={"max_tokens_in_chunk": 600})
        self.assertFalse(response.data["status"])
        self.assertEqual(response.data["errors"][0]["error_code"], "000005_DATA")
        self.assertFalse(
            CollectionOfDocuments.objects.filter(name="new-collection").exists()
        )

    def test_broken_chunking_is_a_validation_error(self):
        for chunking in (
            {"number_of_overlap_tokens": 4},
            "500",
            {"max_tokens_in_chunk": 0},
        ):
            with self.subTest(chunking=chunking):
                response = self.create(chunking=chunking)
                self.assertFalse(response.data["status"])
        self.assertFalse(
            CollectionOfDocuments.objects.filter(name="new-collection").exists()
        )


class IndexingEndpointsChunkingTests(ChunkingTestCase):
    def test_upload_uses_the_chunking_of_the_collection(self):
        self.make_collection("upload-target", chunking=(321, 21))
        with patch("sse_api.data.api.UploadDocumentsController") as upload:
            upload.return_value.store_and_index_files_rel_db_post_request.return_value = (
                None
            )
            request = self.factory.post(
                "/",
                {
                    "files[]": SimpleUploadedFile("test.txt", b"text"),
                    "collection_name": "upload-target",
                    "indexing_options": json.dumps(dict(INDEXING_BASE)),
                },
                format="multipart",
            )
            force_authenticate(request, user=self.user)
            UploadAndIndexFilesToCollection.as_view()(request)
        options = upload.return_value.store_and_index_files_rel_db_post_request.call_args.kwargs[
            "indexing_options"
        ]
        self.assertEqual(options[MAX_TOKENS_IN_CHUNK_KEY], 321)
        self.assertEqual(options[NUMBER_OF_OVERLAP_TOKENS_KEY], 21)

    def test_texts_of_a_strict_collection_are_refused(self):
        self.make_collection("strict", chunking=(500, 0), strict=True)
        response = self.invoke(
            AddAndIndexTextsFromEP,
            {
                "texts[]": TEXT_DOCUMENT,
                "collection_name": "strict",
                "indexing_options": {**INDEXING_BASE, MAX_TOKENS_IN_CHUNK_KEY: 128},
            },
        )
        self.assertFalse(response.data["status"])
        self.assertEqual(response.data["errors"][0]["error_code"], "000006_DATA")

    def test_texts_without_chunking_are_indexed_with_the_collection(self):
        collection = self.make_collection("texts", chunking=(240, 16))
        with patch(
            "sse_api.engine.controllers.database.relational_db."
            "RelationalDBController.add_documents_from_list_of_doc_as_dict",
            return_value=[],
        ) as add_documents:
            response = self.invoke(
                AddAndIndexTextsFromEP,
                {
                    "texts[]": TEXT_DOCUMENT,
                    "collection_name": "texts",
                    "indexing_options": dict(INDEXING_BASE),
                },
            )
        self.assertTrue(response.data["status"])
        options = add_documents.call_args.kwargs["indexing_options"]
        self.assertEqual(options[MAX_TOKENS_IN_CHUNK_KEY], 240)
        self.assertEqual(options[NUMBER_OF_OVERLAP_TOKENS_KEY], 16)
        self.assertEqual(
            self.reload(collection).chunking(),
            {MAX_TOKENS_IN_CHUNK_KEY: 240, NUMBER_OF_OVERLAP_TOKENS_KEY: 16},
        )


class CollectionSettingsReportTests(ChunkingTestCase):
    def test_collections_expose_their_settings(self):
        self.make_collection("with-settings", chunking=(512, 32), strict=True)
        with (
            patch("sse_api.data.api.SemanticDBController") as semantic,
            patch(
                "sse_api.engine.controllers.database.relational_db."
                "RelationalDBController.get_user_collections",
                return_value=CollectionOfDocuments.objects.all(),
            ),
            patch(
                "sse_api.engine.controllers.database.relational_db."
                "RelationalDBController.get_user_organisation_collections",
                return_value=CollectionOfDocuments.objects.none(),
            ),
        ):
            semantic.return_value.collections.return_value = ["with-settings"]
            response = self.invoke(ListCollections, method="get")
        entries = {
            entry["name"]: entry for entry in response.data["body"]["collections"]
        }
        self.assertEqual(
            entries["with-settings"]["metadata_json"][CHUNKING_METADATA_KEY],
            {
                MAX_TOKENS_IN_CHUNK_KEY: 512,
                NUMBER_OF_OVERLAP_TOKENS_KEY: 32,
            },
        )
        self.assertTrue(
            entries["with-settings"]["metadata_json"][STRICT_CHUNKING_METADATA_KEY]
        )
        # A collection which has never been indexed has no settings at all.
        self.assertIsNone(entries["collection"]["metadata_json"])

    def test_filtering_options_report_the_defaults(self):
        response = self.invoke(ListFilteringOptions, method="get")
        self.assertTrue(response.data["status"])
        self.assertEqual(
            response.data["body"]["chunking_defaults"],
            {
                "max_tokens_in_chunk": DEFAULT_MAX_TOKENS_IN_CHUNK,
                "number_of_overlap_tokens": DEFAULT_NUMBER_OF_OVERLAP_TOKENS,
            },
        )


class CountingTokenizer:
    """Tokenizer stand-in: every token of the chunk is one character."""

    def __call__(self, text):
        from types import SimpleNamespace

        return SimpleNamespace(input_ids=list(range(len(text))))


class CheckCollectionChunkingCommandTests(ChunkingTestCase):
    def setUp(self):
        super().setUp()
        self.declared = self.make_collection("declared-collection", chunking=(11, 1))
        self.add_chunk(self.declared, "tiny")
        self.add_chunk(self.declared, "a much longer chunk of text")

    def add_chunk(self, collection, text):
        document = Document.objects.create(
            name=f"{collection.name}-{text[:5]}.txt",
            collection=collection,
            path=f"/tmp/{text[:5]}",
            relative_path=text[:5],
            document_hash=f"{collection.name}-{len(text)}",
            added_by=self.organisation_user,
        )
        page = DocumentPage.objects.create(document=document, page_number=1)
        return DocumentPageText.objects.create(
            page=page,
            text_number=1,
            text_str=text,
            text_hash=text,
            text_chunk_type="plain",
        )

    def run_command(self, *args, **kwargs):
        from io import StringIO

        out = StringIO()
        call_command("check_collection_chunking", *args, stdout=out, **kwargs)
        return out.getvalue()

    def test_report_names_the_declared_settings(self):
        output = self.run_command()
        self.assertIn("declared-collection", output)
        self.assertIn("strict=False", output)
        # A collection which has never been configured falls back to defaults.
        self.assertIn("not declared (defaults apply)", output)

    def test_only_requested_collections_are_reported(self):
        report = json.loads(
            self.run_command("--collection", "declared-collection", "--json")
        )
        self.assertEqual(
            [entry["collection"] for entry in report], ["declared-collection"]
        )

    def test_json_report(self):
        report = json.loads(self.run_command("--json"))
        entries = {entry["collection"]: entry for entry in report}
        self.assertEqual(
            entries["declared-collection"]["declared_chunking"][
                MAX_TOKENS_IN_CHUNK_KEY
            ],
            11,
        )
        self.assertEqual(entries["declared-collection"]["chunks"], 2)
        self.assertIsNone(entries["collection"]["declared_chunking"])

    def test_verify_measures_the_chunks(self):
        with patch.object(
            check_collection_chunking,
            "load_tokenizer",
            return_value=CountingTokenizer(),
        ):
            report = json.loads(self.run_command("--verify", "--json"))
        measured = {entry["collection"]: entry["measured"] for entry in report}[
            "declared-collection"
        ]
        self.assertEqual(measured["chunks_checked"], 2)
        self.assertEqual(measured["above_declared"], 1)

    def test_chunk_longer_than_the_embedder_input_is_reported(self):
        with patch.object(
            check_collection_chunking, "embedder_max_length", return_value=5
        ):
            report = json.loads(self.run_command("--json"))
        entry = {e["collection"]: e for e in report}["declared-collection"]
        self.assertTrue(entry["chunk_too_long_for_embedder"])

    def test_without_a_tokenizer_nothing_is_measured(self):
        with patch.object(
            check_collection_chunking, "load_tokenizer", return_value=None
        ):
            report = json.loads(self.run_command("--verify", "--json"))
        entry = {e["collection"]: e for e in report}["declared-collection"]
        self.assertIsNone(entry["measured"])


class PlanChunkingFixTests(ChunkingTestCase):
    def test_collection_without_chunking_gets_the_defaults(self):
        plan = plan_chunking_fix(self.collection)
        self.assertEqual(
            plan["to"],
            {
                MAX_TOKENS_IN_CHUNK_KEY: DEFAULT_MAX_TOKENS_IN_CHUNK,
                NUMBER_OF_OVERLAP_TOKENS_KEY: DEFAULT_NUMBER_OF_OVERLAP_TOKENS,
            },
        )
        self.assertEqual(plan["from"], {})

    def test_collection_with_a_sane_chunking_needs_no_fix(self):
        collection = self.make_collection("sane", chunking=(300, 30))
        self.assertIsNone(plan_chunking_fix(collection, embedder_limit=2048))

    def test_chunk_above_the_embedder_input_is_clamped(self):
        collection = self.make_collection("too-long", chunking=(4096, 128))
        plan = plan_chunking_fix(collection, embedder_limit=2048)
        self.assertEqual(plan["to"][MAX_TOKENS_IN_CHUNK_KEY], 2048)
        self.assertEqual(plan["to"][NUMBER_OF_OVERLAP_TOKENS_KEY], 128)

    def test_overlap_is_brought_back_inside_the_chunk(self):
        for name, chunking in (
            ("overlap-too-big", (100, 200)),
            ("overlap-negative", (100, -3)),
        ):
            with self.subTest(chunking=chunking):
                collection = self.make_collection(name, chunking=chunking)
                plan = plan_chunking_fix(collection)
                self.assertEqual(
                    plan["to"][NUMBER_OF_OVERLAP_TOKENS_KEY],
                    DEFAULT_NUMBER_OF_OVERLAP_TOKENS,
                )

    def test_values_which_are_not_numbers_are_normalized(self):
        collection = self.make_collection("junk")
        collection.metadata_json = {
            CHUNKING_METADATA_KEY: {
                MAX_TOKENS_IN_CHUNK_KEY: "wide",
                NUMBER_OF_OVERLAP_TOKENS_KEY: 0,
            }
        }
        plan = plan_chunking_fix(collection)
        self.assertEqual(
            plan["to"][MAX_TOKENS_IN_CHUNK_KEY], DEFAULT_MAX_TOKENS_IN_CHUNK
        )
        self.assertTrue(plan["reasons"])


class CheckCollectionChunkingFixTests(CheckCollectionChunkingCommandTests):
    def test_dry_run_writes_nothing(self):
        output = self.run_command("--collection", "collection", "--fix", "--dry-run")
        self.assertIn("fix (dry run)", output)
        self.assertEqual(self.reload(self.collection).metadata_json, None)

    def test_fix_binds_the_defaults(self):
        output = self.run_command("--collection", "collection", "--fix")
        self.assertIn("fix applied", output)
        self.assertEqual(
            self.reload(self.collection).chunking(),
            {
                MAX_TOKENS_IN_CHUNK_KEY: DEFAULT_MAX_TOKENS_IN_CHUNK,
                NUMBER_OF_OVERLAP_TOKENS_KEY: DEFAULT_NUMBER_OF_OVERLAP_TOKENS,
            },
        )

    def test_fix_clamps_a_chunk_which_does_not_fit_the_embedder(self):
        with patch.object(
            check_collection_chunking, "embedder_max_length", return_value=5
        ):
            output = self.run_command(
                "--collection", "declared-collection", "--fix", "--json"
            )
        entry = {e["collection"]: e for e in json.loads(output)}[
            "declared-collection"
        ]
        self.assertTrue(entry["fix"]["changed"])
        self.assertTrue(entry["fix"]["applied"])
        self.assertEqual(entry["fix"]["to"][MAX_TOKENS_IN_CHUNK_KEY], 5)
        self.assertEqual(
            self.reload(self.declared).chunking()[MAX_TOKENS_IN_CHUNK_KEY], 5
        )

    def test_sane_collection_is_left_alone(self):
        self.declared.set_chunking(64, 4)
        self.declared.save(update_fields=["metadata_json"])
        report = json.loads(
            self.run_command(
                "--collection", "declared-collection", "--fix", "--json"
            )
        )
        entry = {e["collection"]: e for e in report}["declared-collection"]
        self.assertFalse(entry["fix"]["changed"])

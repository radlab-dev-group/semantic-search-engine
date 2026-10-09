import io
import json
import stat
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.exceptions import ImproperlyConfigured
from django.test import SimpleTestCase, TestCase, override_settings
from rest_framework.test import force_authenticate

from sse_api.data.api import UploadAndIndexFilesToCollection
from sse_api.data.controllers.upload import (
    UploadBudget,
    UploadDocumentsController,
    UploadRejected,
)
from sse_api.tests.input_support import EndpointInputMixin, INDEXING_OPTIONS


def archive(entries, compression=zipfile.ZIP_STORED):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=compression) as zipped:
        for name, content in entries:
            zipped.writestr(name, content)
    return SimpleUploadedFile("documents.zip", output.getvalue())


class ZipUploadTests(SimpleTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def extract(self, upload):
        return UploadDocumentsController.unzip_uploaded_zip_file(self.root, upload)

    def rejected(self, upload):
        with self.assertRaises(UploadRejected):
            self.extract(upload)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_nested_regular_files_and_no_archive(self):
        paths = self.extract(archive([("nested/", b""), ("nested/a.txt", b"a"), ("b.txt", b"b")]))
        self.assertEqual(set(paths), {str(self.root / "nested/a.txt"), str(self.root / "b.txt")})
        self.assertTrue(all(Path(path).is_file() for path in paths))
        self.assertFalse(list(self.root.rglob("*.zip")))

    def test_corrupt_archive(self):
        self.rejected(SimpleUploadedFile("bad.zip", b"not a zip"))

    def test_unsafe_paths(self):
        for name in ("../escape", "/absolute", "a/../../escape", "C:/escape", "a\\escape", "a/./b", "a//b"):
            with self.subTest(name=name):
                self.rejected(archive([("good.txt", b"ok"), (name, b"bad")]))

    def test_symlink_and_special_file(self):
        for mode in (stat.S_IFLNK, stat.S_IFIFO):
            entry = zipfile.ZipInfo("link")
            entry.create_system = 3
            entry.external_attr = (mode | 0o777) << 16
            self.rejected(archive([(entry, b"../escape")]))

    def test_collisions(self):
        for names in (("same", "same"), ("A.txt", "a.txt"), ("a", "a/b"), ("A/x", "a/y")):
            with self.subTest(names=names):
                self.rejected(archive([(name, b"x") for name in names]))

    def test_existing_file_and_symlink_not_overwritten(self):
        target = self.root / "existing"
        target.write_bytes(b"original")
        with self.assertRaises(UploadRejected):
            self.extract(archive([("existing", b"replacement")]))
        self.assertEqual(target.read_bytes(), b"original")
        (self.root / "link").symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(UploadRejected):
            self.extract(archive([("link/new", b"bad")]))
        self.assertFalse((self.root / "new").exists())

    @override_settings(UPLOAD_MAX_FILES=1)
    def test_header_file_limit(self):
        self.rejected(archive([("a", b"1"), ("b", b"2")]))

    @override_settings(UPLOAD_MAX_UNCOMPRESSED_BYTES=1)
    def test_header_byte_limit(self):
        self.rejected(archive([("a", b"12")]))

    @override_settings(UPLOAD_MAX_COMPRESSION_RATIO=2)
    def test_header_ratio_limit(self):
        self.rejected(archive([("a", b"a" * 1000)], zipfile.ZIP_DEFLATED))

    @override_settings(UPLOAD_MAX_INPUT_BYTES=10)
    def test_input_limit(self):
        self.rejected(archive([("a", b"1")]))

    def test_bounded_reads(self):
        upload = archive([("a", b"a" * 100)])
        original = upload.read

        def bounded(size=-1):
            self.assertGreater(size, 0)
            self.assertLessEqual(size, 64 * 1024)
            return original(size)
        wrapper = SimpleNamespace(name=upload.name, read=bounded)
        self.extract(wrapper)

    def test_crc_failure_cleans_previously_extracted_files(self):
        upload = archive([("first", b"one"), ("nested/second", b"unique payload")])
        damaged = upload.read().replace(b"unique payload", b"broken payload")
        self.rejected(SimpleUploadedFile("bad.zip", damaged))

    def test_stream_limits_and_cleanup(self):
        for config in ({"UPLOAD_MAX_UNCOMPRESSED_BYTES": 3}, {"UPLOAD_MAX_COMPRESSION_RATIO": 2}):
            with self.subTest(config=config), override_settings(**config):
                upload = archive([("nested/a", b"x")])
                with patch.object(zipfile.ZipFile, "open", return_value=io.BytesIO(b"abcdefgh")):
                    self.rejected(upload)

    @override_settings(UPLOAD_MAX_UNCOMPRESSED_BYTES=2)
    def test_plain_stream_limit_cleanup(self):
        controller = UploadDocumentsController(store_to_db=False)
        with self.assertRaises(UploadRejected):
            controller._store_single_file_to_upload_dir(self.root, SimpleUploadedFile("plain", b"long"))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_stream_file_count_budget(self):
        with override_settings(UPLOAD_MAX_FILES=1):
            budget = UploadBudget()
            budget.copy(io.BytesIO(b"a"), io.BytesIO())
            with self.assertRaises(UploadRejected):
                budget.copy(io.BytesIO(b"b"), io.BytesIO())

    def test_invalid_limits_fail_closed(self):
        for config in ({"UPLOAD_MAX_FILES": 0}, {"UPLOAD_MAX_UNCOMPRESSED_BYTES": -1},
                       {"UPLOAD_MAX_COMPRESSION_RATIO": float("inf")}, {"UPLOAD_MAX_COMPRESSION_RATIO": float("nan")},
                       {"UPLOAD_MAX_INPUT_BYTES": "100"}):
            with self.subTest(config=config), override_settings(**config):
                with self.assertRaises(ImproperlyConfigured):
                    UploadBudget()

    def test_cross_upload_collision(self):
        controller = UploadDocumentsController(store_to_db=False)
        controller._store_single_file_to_upload_dir(self.root, SimpleUploadedFile("A", b"original"))
        for upload in (SimpleUploadedFile("a", b"new"), archive([("a", b"new")])):
            with self.assertRaises(UploadRejected):
                controller._store_single_file_to_upload_dir(self.root, upload)
        self.assertEqual((self.root / "A").read_bytes(), b"original")
        self.assertEqual(len(list(self.root.iterdir())), 1)


class UploadPipelineTests(EndpointInputMixin, TestCase):
    def setUp(self):
        super().setUp()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.controller = UploadDocumentsController(upload_dir=self.temp.name, store_to_db=False)
        self.collection = SimpleNamespace(name="collection")

    def run_upload(self, files):
        return self.controller.store_and_index_files_rel_db_post_request(
            self.profile, files, self.collection, INDEXING_OPTIONS, "unused.json",
        )

    def test_count_documents_and_archive_is_not_scanned(self):
        record = SimpleNamespace()
        with patch.object(self.controller, "get_add_uploaded_documents", return_value=record), patch.object(
            self.controller.rel_db_controller, "add_uploaded_documents_to_db", return_value=(record, [])
        ), patch.object(self.controller, "_index_in_semantic_db"):
            result = self.run_upload([archive([("nested/a", b"1"), ("b", b"2")]), SimpleUploadedFile("c", b"3")])
        self.assertEqual(result.number_of_uploaded_documents, 3)
        self.assertEqual(len([path for path in self.root.rglob("*") if path.is_file()]), 3)
        self.assertFalse(list(self.root.rglob("*.zip")))

    @override_settings(UPLOAD_MAX_FILES=1)
    def test_request_limit_cleanup_no_record_or_index(self):
        with patch.object(self.controller, "get_add_uploaded_documents") as record, patch.object(
            self.controller.rel_db_controller, "add_uploaded_documents_to_db"
        ) as relational, patch.object(self.controller, "_index_in_semantic_db") as semantic:
            with self.assertRaises(UploadRejected):
                self.run_upload([SimpleUploadedFile("first", b"a"), archive([("second", b"b")])])
        record.assert_not_called()
        relational.assert_not_called()
        semantic.assert_not_called()
        self.assertFalse([p for p in self.root.rglob("*") if p.is_file()])
        self.assertFalse(list((self.root / self.profile.organisation.name / self.user.username / "collection").iterdir()))

    def test_endpoint_rejection_envelope_and_no_index(self):
        with patch("sse_api.data.api.RelationalDBController.get_collection", return_value=self.collection), patch(
            "sse_api.data.api.UploadDocumentsController", return_value=self.controller
        ), patch.object(self.controller.rel_db_controller, "add_uploaded_documents_to_db") as relational, patch.object(
            self.controller, "_index_in_semantic_db"
        ) as semantic:
            request = self.factory.post("/", {
                "files[]": SimpleUploadedFile("bad.zip", b"broken"),
                "collection_name": "collection", "indexing_options": json.dumps(INDEXING_OPTIONS),
            }, format="multipart")
            force_authenticate(request, user=self.user)
            response = UploadAndIndexFilesToCollection.as_view()(request)
        self.assert_denial(response)
        self.assertEqual(response.data["errors"][0]["error_code"], "000001_DATA")
        relational.assert_not_called()
        semantic.assert_not_called()
        self.assertFalse([p for p in self.root.rglob("*") if p.is_file()])
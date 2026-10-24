import copy
import json
import tempfile
import unittest
from pathlib import Path

from sse_api.data.core.file_metadata import apply_file_metadata, resolve_file_metadata


class FileMetadataTests(unittest.TestCase):
    def setUp(self):
        self.name = "doc:20261022:abc.txt"
        self.entry = {
            "category": "doc",
            "options": {
                "source_file": "docs/guide.md",
                "scope": "project",
                "doc_type": "markdown",
                "nested": {"tags": ["żółć"], "enabled": True, "empty": None},
            },
        }

    def test_worker_metadata_json_round_trip(self):
        metadata = {self.name: self.entry}
        self.assertEqual(resolve_file_metadata(json.dumps(metadata), [self.name]), metadata)

    def test_exact_and_basename_mapping(self):
        for key in (self.name, "docs/" + self.name, "docs\\" + self.name):
            with self.subTest(key=key):
                self.assertEqual(
                    resolve_file_metadata({key: self.entry}, [self.name]),
                    {self.name: self.entry},
                )

    def test_omission_preserves_legacy_uploads(self):
        self.assertIsNone(resolve_file_metadata(None, ["old.zip", "old.txt"]))
        document = {"category": "legacy", "options": {"reader": True}}
        self.assertEqual(apply_file_metadata([document], None, "/unused"), [(document, False)])

    def test_metadata_and_zip_can_share_a_batch(self):
        self.assertEqual(
            resolve_file_metadata({self.name: self.entry}, ["old.ZIP", self.name]),
            {self.name: self.entry},
        )
        self.assertEqual(resolve_file_metadata({}, ["old.zip"]), {})

    def test_invalid_mapping_and_entries(self):
        for value in (
            "{", "null", "[]", [], 1, "{}",
            {self.name: None}, {self.name: []},
            {self.name: {"category": "doc"}},
            {self.name: {"category": 1, "options": {}}},
            {self.name: {"category": "doc", "options": []}},
            {self.name: {**self.entry, "unknown": True}},
            {self.name: {"category": "doc", "options": {"x": float("nan")}}},
            {"": self.entry}, {"directory/": self.entry}, {12: self.entry},
        ):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                resolve_file_metadata(value, [self.name])

    def test_non_finite_and_duplicate_json_keys(self):
        for raw in (
            '{"a.txt":{"category":"doc","options":{"x":NaN}}}',
            '{"a.txt":{"category":"doc","options":{"x":Infinity}}}',
            '{"a.txt":{"category":"doc","options":{"x":1e999}}}',
            '{"a.txt":{},"a.txt":{}}',
            '{"a.txt":{"category":"a","category":"b","options":{}}}',
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                resolve_file_metadata(raw, ["a.txt"])

    def test_missing_extra_duplicate_and_ambiguous_targets(self):
        for metadata, names in (
            ({}, [self.name]),
            ({self.name: self.entry}, [self.name, "other.txt"]),
            ({"missing.txt": self.entry}, [self.name]),
            ({self.name: self.entry}, [self.name, self.name]),
            ({self.name: self.entry, "dir/" + self.name: self.entry}, [self.name]),
            ({"a.txt": self.entry}, ["one/a.txt", "two/a.txt"]),
            ({"archive.zip": self.entry}, ["archive.zip"]),
            ({"nested/a.txt": self.entry}, ["archive.zip"]),
        ):
            with self.subTest(metadata=metadata, names=names), self.assertRaises(ValueError):
                resolve_file_metadata(metadata, names)

    def test_exact_name_wins_over_basename(self):
        metadata = {"one/a.txt": self.entry, "two/a.txt": self.entry}
        self.assertEqual(resolve_file_metadata(metadata, list(metadata)), metadata)

    def test_apply_merges_before_storage_without_mutating_reader_or_zip_members(self):
        with tempfile.TemporaryDirectory() as root:
            document = {
                "filepath": str(Path(root) / self.name),
                "category": "reader",
                "options": {"reader": True, "scope": "old"},
                "pages": [{"page_content": "original"}],
            }
            zipped = {**document, "filepath": str(Path(root) / "nested" / self.name)}
            original = copy.deepcopy(document)
            prepared = apply_file_metadata([document, zipped], {self.name: self.entry}, root)
            updated, explicit = prepared[0]
            self.assertTrue(explicit)
            self.assertEqual(updated["category"], "doc")
            self.assertEqual(updated["options"], {"reader": True, **self.entry["options"]})
            self.assertEqual(updated["pages"], document["pages"])
            self.assertEqual(document, original)
            self.assertEqual(prepared[1], (zipped, False))

    def test_reader_missing_duplicate_and_outside_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            document = {"filepath": str(Path(root) / self.name), "options": {}}
            for documents in ([], [document, document], [{"filepath": "/outside.txt"}]):
                with self.subTest(documents=documents), self.assertRaises(ValueError):
                    apply_file_metadata(documents, {self.name: self.entry}, root)

    def test_empty_explicit_category_is_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            document = {"filepath": str(Path(root) / self.name), "category": "reader"}
            result = apply_file_metadata(
                [document], {self.name: {"category": "", "options": {}}}, root
            )
            self.assertEqual(result[0][0]["category"], "")
            self.assertTrue(result[0][1])
"""
test_context_enrichment.py
--------------------------

Unit tests of the context‑enrichment helpers which build the text sent to
the embedder: a small structured header (document, category, path, section)
above the raw chunk.

The enrichment is for the embedder only: the raw text is what is stored and
displayed.  These tests are pure, they need neither Django nor a database.
"""

from sse_api.data.core.context_enrichment import (
    HEADER_PREFIX,
    build_context_header,
    build_enriched_chunk_text,
    extract_section_title,
    is_enriched,
    raw_chunk_text,
)

import unittest


class BuildContextHeaderTest(unittest.TestCase):
    def test_document_name_only(self):
        header = build_context_header("regulamin.pdf")
        self.assertEqual(header, f"{HEADER_PREFIX} regulamin.pdf")

    def test_all_fields(self):
        header = build_context_header(
            "regulamin.pdf",
            category="finanse",
            relative_path="docs/regulamin.pdf",
            section_title="Wnioski",
        )
        self.assertEqual(
            header,
            f"{HEADER_PREFIX} regulamin.pdf | Kategoria: finanse | "
            "Ścieżka: docs/regulamin.pdf | Sekcja: Wnioski",
        )

    def test_missing_optional_fields_are_left_out(self):
        header = build_context_header("a.txt", category=None, section_title="")
        self.assertEqual(header, f"{HEADER_PREFIX} a.txt")

    def test_path_longer_than_the_limit_is_cut(self):
        path = "very/long/path/" + "segment/" * 100 + "plik.pdf"
        header = build_context_header("plik.pdf", relative_path=path)
        self.assertLessEqual(len(header), 400)
        self.assertTrue(header.endswith("…"))


class ExtractSectionTitleTest(unittest.TestCase):
    def test_markdown_heading_on_the_first_line(self):
        text = "## Warunki gwarancji\n\nGwarancja wynosi 24 miesiące."
        self.assertEqual(extract_section_title(text), "Warunki gwarancji")

    def test_markdown_heading_after_blank_lines(self):
        text = "\n\n### Sekcja 3\nTreść pod nagłówkiem."
        self.assertEqual(extract_section_title(text), "Sekcja 3")

    def test_title_line_followed_by_a_body(self):
        text = "Warunki gwarancji\nGwarancja wynosi 24 miesiące i obejmuje naprawy."
        self.assertEqual(extract_section_title(text), "Warunki gwarancji")

    def test_no_section_when_the_text_is_a_single_sentence(self):
        text = "Jedno zdanie bez nagłówka."
        self.assertIsNone(extract_section_title(text))

    def test_title_line_with_terminal_punctuation_is_not_a_title(self):
        text = "Pierwsze zdanie kończy się kropką.\nDruga linia tekstu."
        self.assertIsNone(extract_section_title(text))

    def test_empty_and_none(self):
        self.assertIsNone(extract_section_title(""))
        self.assertIsNone(extract_section_title(None))

    def test_long_first_line_is_not_a_title(self):
        long_line = ("Słowo " * 40).strip()
        self.assertIsNone(extract_section_title(f"{long_line}\nTreść."))


class BuildEnrichedChunkTextTest(unittest.TestCase):
    def test_header_and_blank_line_above_the_chunk(self):
        enriched = build_enriched_chunk_text("Treść chunka.", "dok.pdf")
        self.assertEqual(f"{HEADER_PREFIX} dok.pdf\n\nTreść chunka.", enriched)

    def test_all_context_fields(self):
        enriched = build_enriched_chunk_text(
            "Treść.",
            "dok.pdf",
            category="prawo",
            relative_path="docs/dok.pdf",
            section_title="Sekcja",
        )
        self.assertEqual(
            enriched,
            f"{HEADER_PREFIX} dok.pdf | Kategoria: prawo | "
            "Ścieżka: docs/dok.pdf | Sekcja: Sekcja\n\nTreść.",
        )

    def test_section_title_extracted_from_the_chunk(self):
        enriched = build_enriched_chunk_text("## Wnioski\nTreść.", "dok.pdf")
        self.assertIn("Sekcja: Wnioski", enriched)

    def test_empty_chunk_comes_back_empty(self):
        self.assertEqual(build_enriched_chunk_text("", "dok.pdf"), "")
        self.assertEqual(build_enriched_chunk_text("   ", "dok.pdf"), "   ")

    def test_missing_document_name_keeps_the_chunk(self):
        self.assertEqual(build_enriched_chunk_text("Treść.", ""), "Treść.")
        self.assertEqual(build_enriched_chunk_text("Treść.", None), "Treść.")

    def test_an_already_enriched_chunk_is_not_enriched_twice(self):
        enriched = build_enriched_chunk_text("Treść.", "dok.pdf")
        again = build_enriched_chunk_text(enriched, "inne.pdf")
        self.assertEqual(enriched, again)
        self.assertNotIn("inne.pdf", again)

    def test_the_chunk_body_is_stripped(self):
        enriched = build_enriched_chunk_text("  Treść.  \n", "dok.pdf")
        self.assertEqual(f"{HEADER_PREFIX} dok.pdf\n\nTreść.", enriched)


class EnrichmentMarkersTest(unittest.TestCase):
    def test_is_enriched(self):
        self.assertTrue(is_enriched(f"{HEADER_PREFIX} dok.pdf\n\nTreść."))
        self.assertTrue(is_enriched(f"  {HEADER_PREFIX} dok.pdf\n\nTreść."))
        self.assertFalse(is_enriched("Treść bez nagłówka."))
        self.assertFalse(is_enriched(""))
        self.assertFalse(is_enriched(None))

    def test_raw_chunk_text_removes_the_header(self):
        enriched = build_enriched_chunk_text("Treść chunka.", "dok.pdf")
        self.assertEqual(raw_chunk_text(enriched), "Treść chunka.")

    def test_raw_chunk_text_keeps_plain_text(self):
        self.assertEqual(raw_chunk_text("Zwykły tekst."), "Zwykły tekst.")
        self.assertIsNone(raw_chunk_text(None))

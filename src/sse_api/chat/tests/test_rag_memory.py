from types import SimpleNamespace
from unittest import TestCase

from sse_api.chat.controllers.memory import (
    build_memory,
    is_history_transformation,
    merge_sources,
    scoped_history,
)


def source(n, text_id, text="Evidence"):
    return {
        "n": n,
        "text_id": text_id,
        "document_id": 1,
        "document_name": "report",
        "page_number": 1,
        "text_str": text,
    }


def turn(question, answer, citations, registry=None):
    metadata = {"citations": citations}
    if registry is not None:
        metadata["citation_registry"] = registry
    rag = SimpleNamespace(
        sse_answer=SimpleNamespace(generative_qa_result_json=metadata)
    )
    return [
        SimpleNamespace(
            role="user", text=question, state=SimpleNamespace(rag_message_state=rag)
        ),
        SimpleNamespace(role="assistant", text=answer, state=None),
    ]


class RagMemoryTests(TestCase):
    def test_numbers_survive_new_turn_and_repeated_source(self):
        registry = [source(1, 10), source(2, 20)]
        merged = merge_sources([source(1, 20), source(2, 30)], registry)
        self.assertEqual([s["n"] for s in merged], [2, 3])
        self.assertEqual([s["text_id"] for s in registry], [10, 20, 30])

    def test_legacy_conflicts_are_remapped_per_turn(self):
        history = turn("First", "First [1]", [source(1, 10)])
        history += turn("Second", "Second [1]", [source(1, 20)])
        turns, registry, sources = build_memory(
            history, max_turns=8, max_chars=24000
        )
        self.assertEqual(turns[-1]["assistant"], "Second [2]")
        self.assertEqual([s["n"] for s in sources], [1, 2])
        self.assertEqual(len(registry), 2)

    def test_registry_rehydrates_even_when_old_turn_outside_window(self):
        history = turn("First", "A [1]", [source(1, 10)])
        history += turn("Second", "B [2]", [source(2, 20)])
        turns, registry, sources = build_memory(
            history, max_turns=1, max_chars=24000
        )
        self.assertEqual(len(turns), 1)
        self.assertEqual([s["n"] for s in registry], [1, 2])
        self.assertEqual([s["n"] for s in sources], [2])

    def test_changed_text_is_not_mistaken_for_old_evidence(self):
        registry = [source(1, 10, "old")]
        self.assertEqual(merge_sources([source(1, 10, "new")], registry)[0]["n"], 2)

    def test_conservative_transformation_detection(self):
        for text in (
            "Zrób z tego tabelę",
            "skróć",
            "przetłumacz na angielski",
            "Make it a table",
        ):
            self.assertTrue(is_history_transformation(text), text)
        for text in (
            "Jakie koszty?",
            "Zrób tabelę kosztów z nowych raportów",
            "Skróć i znajdź nowe dane",
        ):
            self.assertFalse(is_history_transformation(text), text)

    def test_character_budget_does_not_cut_a_turn_in_half(self):
        turns, _, _ = build_memory(turn("Long question", "Long answer", []), 8, 3)
        self.assertEqual(turns, [])

    def test_three_turns_resume_with_stable_sources(self):
        import json

        history = turn("Obowiązki AI Act?", "Ewidencja [1].", [source(1, 10)])
        _, registry, _ = build_memory(history)
        sources = merge_sources([source(1, 10), source(2, 20)], registry)
        history += turn(
            "A które dla małych firm?",
            "Dla MŚP: ewidencja [1], dokumentacja [2].",
            sources,
            json.loads(json.dumps(registry)),
        )
        turns, restored, sources = build_memory(history)
        self.assertIn("MŚP", turns[-1]["assistant"])
        self.assertTrue(is_history_transformation("Zrób z tego tabelę"))
        self.assertEqual([s["n"] for s in sources], [1, 2])
        table = "| Obowiązek |\n|---|\n| Ewidencja [1] |\n| Dokumentacja [2] |"
        history += turn("Zrób z tego tabelę", table, sources, restored)
        resumed, final_registry, _ = build_memory(history, max_turns=1)
        self.assertEqual(resumed[-1]["assistant"], table)
        self.assertEqual(final_registry, restored)

    def test_unknown_legacy_references_are_not_rebound(self):
        history = turn("First", "Supported [1], unknown [2].", [source(1, 10)])
        history += turn("Second", "Other [1].", [source(1, 20)])
        turns, _, _ = build_memory(history)
        self.assertEqual(turns[0]["assistant"], "Supported [1], unknown .")

    def test_collection_or_document_scope_change_excludes_old_turn(self):
        history = turn("First", "First [1]", [source(1, 10)])
        history[0].search_options = {"documents": ["old.pdf"]}
        history[0].state.rag_message_state.sse_query = SimpleNamespace(
            collection_id=1
        )
        self.assertEqual(
            scoped_history(history, 2, history[0].search_options), ([], None)
        )
        self.assertEqual(
            scoped_history(history, 1, {"documents": ["new.pdf"]}), ([], None)
        )
        selected, previous = scoped_history(history, 1, history[0].search_options)
        self.assertEqual(selected, history)
        self.assertIs(previous, history[0].state.rag_message_state)

    def test_enriched_metadata_does_not_change_source_number(self):
        registry = [source(1, 10)]
        enriched = dict(source(1, 10), char_start=3, char_end=11)
        self.assertEqual(merge_sources([enriched], registry)[0]["n"], 1)

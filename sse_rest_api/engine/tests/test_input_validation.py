from unittest.mock import patch

from django.test import TestCase

from engine.api import SearchWithOptions, GenerativeAnswerForQuestion, SetRateForQueryResponseAnswer
from tests.input_support import EndpointInputMixin, GENERATION_OPTIONS


class EngineInputTests(EndpointInputMixin, TestCase):
    def test_bad_search_input_has_no_query_or_lookup(self):
        payload = {"collection_name": "docs", "query_str": "question", "options": {}}
        cases = [("collection_name", None), ("collection_name", []), ("query_str", False),
                 ("query_str", {}), ("options", "{"), ("options", "[]"),
                 ("options", []), ("options", None), ("options", 4),
                 ("ignore_question_lang_detect", "maybe"),
                 ("options", {"max_results": "bad"}),
                 ("options", {"rerank_results": []})]
        with patch("engine.api.SearchQueryController.new_query") as query, \
                patch("engine.api.RelationalDBController.get_collection") as lookup:
            for key, value in cases:
                with self.subTest(key=key, value=value):
                    self.assert_denial(self.invoke(SearchWithOptions, {**payload, key: value}))
            query.assert_not_called()
            lookup.assert_not_called()

    def test_object_options_and_false_are_preserved(self):
        with patch("engine.api.SearchQueryController.new_query", return_value={}) as query, \
                patch("engine.api.RelationalDBController.get_collection", return_value=object()):
            for options in ({}, "{}", {"rerank_results": "false"}):
                for flag in (False, "false", 0):
                    with self.subTest(options=options, flag=flag):
                        response = self.invoke(SearchWithOptions, {
                            "collection_name": "docs", "query_str": "question",
                            "options": options, "ignore_question_lang_detect": flag,
                        })
                        self.assertTrue(response.data["status"])
                        self.assertIs(query.call_args.kwargs["ignore_question_lang_detect"], False)
                        self.assertIsInstance(query.call_args.kwargs["search_options_dict"], dict)
                        if isinstance(options, dict) and "rerank_results" in options:
                            self.assertIs(query.call_args.kwargs["search_options_dict"]["rerank_results"], False)

    def test_generation_input_before_lookup_or_model(self):
        payload = {"query_response_id": 1, "query_options": GENERATION_OPTIONS}
        with patch("engine.api.SearchQueryController.get_user_response_by_id") as lookup, \
                patch("engine.api.GenerativeModelController") as model:
            for key, value in (("query_response_id", "bad"), ("query_response_id", True),
                               ("query_response_id", -1), ("query_response_id", []),
                               ("query_options", "[]"), ("query_options", {}),
                               ("system_prompt", 2), ("query_instruction", [])):
                with self.subTest(key=key, value=value):
                    self.assert_denial(self.invoke(GenerativeAnswerForQuestion, {**payload, key: value}))
            lookup.assert_not_called()
            model.assert_not_called()

    def test_generation_denial_and_model_failure_use_registered_codes(self):
        payload = {"query_response_id": "1", "query_options": GENERATION_OPTIONS}
        with patch("engine.api.SearchQueryController.get_user_response_by_id", return_value=None), \
                patch("engine.api.GenerativeModelController") as model:
            self.assert_denial(self.invoke(GenerativeAnswerForQuestion, payload))
            model.assert_not_called()
        with patch("engine.api.SearchQueryController.get_user_response_by_id", return_value=object()), \
                patch("engine.api.GenerativeModelController") as model:
            model.return_value.generative_answer_for_response.return_value = None
            self.assert_denial(self.invoke(GenerativeAnswerForQuestion, payload))

    def test_invalid_ratings_do_not_lookup_or_write(self):
        payload = {"answer_response_id": 1, "rate_value": 0, "rate_value_max": 5}
        with patch("engine.api.GenerativeModelController.get_user_query_response_answer") as lookup, \
                patch("engine.api.EngineSystemController") as writer:
            for key, value in (("answer_response_id", "bad"), ("answer_response_id", 1.5),
                               ("rate_value", -1), ("rate_value", 6), ("rate_value", True),
                               ("rate_value", 1.5), ("rate_value_max", 2147483648),
                               ("rate_value", "nan"), ("rate_value_max", 0),
                               ("rate_value_max", "inf"), ("rate_comment", [])):
                with self.subTest(key=key, value=value):
                    self.assert_denial(self.invoke(SetRateForQueryResponseAnswer, {**payload, key: value}))
            lookup.assert_not_called()
            writer.assert_not_called()

    def test_zero_rating_valid_and_missing_answer_denied(self):
        payload = {"answer_response_id": "1", "rate_value": 0, "rate_value_max": 5}
        with patch("engine.api.GenerativeModelController.get_user_query_response_answer", return_value=object()), \
                patch("engine.api.EngineSystemController") as writer:
            self.assertTrue(self.invoke(SetRateForQueryResponseAnswer, payload).data["status"])
            self.assertEqual(writer.return_value.set_rating.call_args.kwargs["rating_value"], 0)
        with patch("engine.api.GenerativeModelController.get_user_query_response_answer", return_value=None), \
                patch("engine.api.EngineSystemController") as writer:
            self.assert_denial(self.invoke(SetRateForQueryResponseAnswer, payload))
            writer.assert_not_called()
"""Extractive question‑answering over the retrieved passages.

The engine does not own the QA weights: an answer is extracted by the model
service (``POST /v1/extractive_qa`` of ``sse_models``), which is what
``SSE_MODEL_SERVICE_ENABLED`` switches on.  ``transformers`` is imported by
neither this module nor anything it imports at module level, so the engine
stays importable — and deployable — without the model stack.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sse_api.core.constants import get_logger


class ExtractiveQAController:
    """
    Answer a question with a span of every retrieved passage.

    Three ways of answering, tried in this order:

    * a ``qa_pipeline`` (or an ``answerer`` callable) handed to the
      constructor — a test double, or a caller which owns its model;
    * the **model service**, the production path: one HTTP call for the whole
      candidate set, no torch in this process;
    * the in‑process pipeline of
      :mod:`sse_api.data.controllers.extractive_qa_local`, the historical path
      for deployments which do not run the service.

    A passage the model cannot be run over answers with an empty string and a
    score of ``0.0``: a failed answer is a missing answer, not a failed search.
    """

    model_path = "radlab/polish-qa-v2"

    def __init__(
        self,
        model_path: Optional[str] = None,
        qa_pipeline: Any = None,
        device: str = "cpu",
        answerer: Optional[
            Callable[[str, Sequence[str]], List[Tuple[str, float]]]
        ] = None,
    ) -> None:
        """
        Initialise the controller.

        Parameters
        ----------
        model_path : str | None
            Model used by :meth:`load_model`. Nothing is loaded by the
            constructor: without an injected pipeline the question is answered
            by the model service, or by the in‑process pipeline on first use.
        qa_pipeline : object | None
            An already‑initialised HuggingFace question‑answering pipeline.
        device : str
            Device :meth:`load_model` asks for (``"cpu"``, ``"cuda"``).
        answerer : callable | None
            ``(question, contexts) -> [(answer, score), ...]`` override, used
            instead of the service.
        """
        self.device = device
        self.model_path = model_path or self.model_path
        self.question_answerer = qa_pipeline
        self._answerer = answerer
        self._logger = get_logger()

    # ------------------------------------------------------------------
    def load_model(self, model_path: Optional[str] = None):
        """
        Load the in‑process HuggingFace question‑answering pipeline.

        Kept for callers which really want the weights in their own process;
        the service path never calls it.
        """
        from sse_api.data.controllers.extractive_qa_local import load_qa_pipeline

        self.question_answerer = load_qa_pipeline(
            model_path or self.model_path, device=self.device
        )
        return self.question_answerer

    # ------------------------------------------------------------------
    def run_extractive_qa(self, question_str: str, search_results: dict) -> dict:
        """
        Perform extractive QA on a set of retrieved passages.

        Parameters
        ----------
        question_str : str
            The user's question.
        search_results : dict
            Dictionary produced by the semantic‑search controller. Must contain a
            ``"results"`` list where each entry holds a ``"result"`` dict with
            ``"text"`` sub‑fields.

        Returns
        -------
        dict
            Mapping of document name to a dict containing page number, text
            number, the extracted answer and its confidence score.
        """
        passages = [
            (
                answer["result"]["text"]["text_str"],
                answer["result"]["text"]["document_name"],
                answer["result"]["text"]["page_number"],
                answer["result"]["text"]["text_number"],
            )
            for answer in search_results["results"]
        ]

        answers = self._answer(question_str, [passage[0] for passage in passages])

        document_answers: Dict[str, Any] = {}
        for (answer, score), (_text, document_name, page, text_number) in zip(
            answers, passages
        ):
            document_answers[document_name] = {
                "page_number": page,
                "text_number": text_number,
                "answer": answer,
                "score": score,
            }
        return document_answers

    # ------------------------------------------------------------------
    def _answer(
        self, question: str, contexts: Sequence[str]
    ) -> List[Tuple[str, float]]:
        """Answer over ``contexts``, one ``(answer, score)`` pair each."""
        if not contexts:
            return []
        if not question.strip():
            return [("", 0.0)] * len(contexts)

        if self._answerer is not None:
            return list(self._answerer(question, contexts))

        if self.question_answerer is not None:
            answers: List[Tuple[str, float]] = []
            for context in contexts:
                if not context.strip():
                    answers.append(("", 0.0))
                    continue
                result = self.question_answerer(question=question, context=context)
                answers.append(
                    (
                        str(result.get("answer") or ""),
                        float(result.get("score") or 0.0),
                    )
                )
            return answers

        service_answerer = self._service_answerer()
        if service_answerer is not None:
            try:
                return list(service_answerer(question, contexts))
            except Exception as exc:  # dead service: fall through to local
                self._logger.warning(
                    "Model service extractive QA failed (%s); trying the local "
                    "QA pipeline",
                    exc,
                )

        from sse_api.data.controllers.extractive_qa_local import answer_locally

        try:
            return answer_locally(question, list(contexts))
        except Exception as exc:  # no weights / no torch on this host
            self._logger.warning(
                "Extractive QA is unavailable (%s); answering with no answer",
                exc,
            )
            return [("", 0.0)] * len(contexts)

    @staticmethod
    def _service_answerer():
        """The ``POST /v1/extractive_qa`` client, ``None`` when not configured."""
        from sse_api.engine.core.models.backend import (
            build_extractive_qa_client,
        )

        return build_extractive_qa_client()

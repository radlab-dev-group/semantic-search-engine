"""
extractive_qa.py
----------------

Extractive question-answering inference for the model service.

The engine no longer owns the QA weights: it posts the question together with
the retrieved passages to ``POST /v1/extractive_qa`` and this module does the
forward pass.  A passage is answered with the span the model extracts from it
— verbatim, plus its confidence — which is the shape the engine's former
``ExtractiveQAController`` produced, so moving the work changes no answer.

``transformers`` is imported inside :func:`load_qa_pipeline`, so importing
this module (and the whole service) stays lightweight: a service which never
answers the endpoint never loads the model stack.

Why the forward pass is written here rather than asked for as a pipeline:
transformers v5 **removed** the ``question-answering`` pipeline while keeping
the ``*ForQuestionAnswering`` model classes.  Extractive QA is a small, fixed
computation — tokenise ``[CLS] question [SEP] context [SEP]``, score every
position as a span start and as a span end, keep the best ``start <= end``
pair and read that span back out of the context — so it is done directly
against the model and runs on every release, instead of pinning the service to
an old one.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Longest span (in tokens) the model may quote from one passage.
DEFAULT_MAX_ANSWER_LENGTH = 512
#: Model input window; this family of QA models takes 512 tokens.
MODEL_MAX_INPUT_LENGTH = 512


def _resolve_device(device: str) -> str:
    """A requested CUDA device which is not present becomes CPU."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return str(device)


def load_qa_pipeline(model: str, device: str = "cpu") -> Dict[str, Any]:
    """
    Load a HuggingFace question-answering model for ``model``.

    ``model`` is an identifier or a path, exactly what the engine passed to
    ``pipeline("question-answering", ...)`` before the move.  ``device`` is a
    torch device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``); a requested
    CUDA device which is not available falls back to CPU instead of failing
    the request.

    Returns the ``{"model", "tokenizer", "path"}`` bundle the registry caches.
    """
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    resolved = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    loaded = AutoModelForQuestionAnswering.from_pretrained(model)
    loaded.to(resolved)
    loaded.eval()
    return {"model": loaded, "tokenizer": tokenizer, "path": model}


def answer(
    bundle: Dict[str, Any],
    question: str,
    contexts: List[str],
    max_answer_length: int = DEFAULT_MAX_ANSWER_LENGTH,
) -> List[Tuple[str, float]]:
    """
    Answer ``question`` with a span of every context, in input order.

    Returns one ``(answer, score)`` pair per context.  A blank context
    answers with an empty string and a score of ``0.0`` without calling the
    model: that is all a blank passage can offer, and a question the model
    cannot ground must not invent an answer.
    """
    inference = QAInference(bundle, max_answer_length=max_answer_length)

    answers: List[Tuple[str, float]] = []
    for context in contexts:
        if not context.strip():
            answers.append(("", 0.0))
            continue
        result = inference(question, context)
        answers.append((result["answer"], result["score"]))
    return answers


class QAInference:
    """
    Answer a question with a span of one context.

    ``bundle`` is what :func:`load_qa_pipeline` returns — a
    ``{"model", "tokenizer"}`` mapping.  A callable is accepted as well (it is
    then asked for ``question=`` / ``context=`` and answers with
    ``{"answer", "score"}``), which is what the tests hand over: the HTTP
    contract is then exercisable without torch.
    """

    def __init__(
        self, bundle: Any, max_answer_length: int = DEFAULT_MAX_ANSWER_LENGTH
    ) -> None:
        self._bundle = bundle
        self._max_answer_length = max_answer_length

    def __call__(self, question: str, context: str) -> Dict[str, Any]:
        if callable(self._bundle):
            result = self._bundle(question=question, context=context)
            return {
                "answer": str(result.get("answer") or ""),
                "score": float(result.get("score") or 0.0),
            }
        return self._extract(question, context)

    # ------------------------------------------------------------------
    def _extract(self, question: str, context: str) -> Dict[str, Any]:
        import torch

        model = self._bundle["model"]
        tokenizer = self._bundle["tokenizer"]

        encoding, offsets = self._encode(tokenizer, question, context)
        device = next(model.parameters()).device
        with torch.no_grad():
            outputs = model(
                **{
                    key: (value.to(device) if hasattr(value, "to") else value)
                    for key, value in encoding.items()
                }
            )

        start_scores = torch.softmax(outputs.start_logits[0].float(), dim=0)
        end_scores = torch.softmax(outputs.end_logits[0].float(), dim=0)

        best = self._best_span(start_scores, end_scores)
        if best is None:
            return {"answer": "", "score": 0.0}

        start, end, score = best
        return {
            "answer": _context_span(context, offsets, start, end, tokenizer),
            "score": score,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _encode(tokenizer, question: str, context: str):
        """
        Tokenise the ``question`` / ``context`` pair for the model.

        The character offsets are asked for with the pair, because they are
        what turns a token span back into the passage's own text, and they
        are then taken out of the encoding: they tell the model nothing, and
        passing them as a model input would be a type error.  A tokeniser
        without offset support (a slow one) simply yields no offsets, and the
        span is decoded instead.
        """
        try:
            encoding = tokenizer(
                question,
                context,
                return_tensors="pt",
                truncation=True,
                max_length=MODEL_MAX_INPUT_LENGTH,
                return_offsets_mapping=True,
            )
            offsets = encoding.pop("offset_mapping")[0].tolist()
            return dict(encoding), offsets
        except Exception:
            encoding = tokenizer(
                question,
                context,
                return_tensors="pt",
                truncation=True,
                max_length=MODEL_MAX_INPUT_LENGTH,
            )
            return dict(encoding), []

    # ------------------------------------------------------------------
    def _best_span(self, start_scores, end_scores):
        """
        The most probable ``start <= end`` pair, as ``(start, end, score)``.

        The probability of a span is the product of its two position
        probabilities, which is how the task is trained.  A span is at most
        ``max_answer_length`` tokens long.
        """
        import torch

        limit = min(len(start_scores), len(end_scores))
        best_score = -1.0
        best_span: Optional[Tuple[int, int, float]] = None
        for start in range(limit):
            start_probability = float(start_scores[start])
            if start_probability <= 0.0:
                continue
            end_limit = min(limit, start + self._max_answer_length)
            if end_limit <= start:
                continue
            window = end_scores[start:end_limit]
            offset = int(torch.argmax(window))
            score = start_probability * float(window[offset])
            if score > best_score:
                best_score = score
                best_span = (start, start + offset, score)
        return best_span


def _context_span(
    context: str, offsets: Sequence[Sequence[int]], start: int, end: int, tokenizer
) -> str:
    """
    The text of the token span ``[start, end]``, read back from ``context``.

    Offsets of a pair input are reported against the text they came from — the
    context for the second half of the sequence, the question for the first —
    and a special token reports ``(0, 0)``.  A span whose end is not after its
    start therefore came from outside the context (or from an unsupported
    tokeniser) and is decoded instead of sliced.  A cut which lands mid-word
    is the model's, not the text's: the span is then closed on the word
    boundary it had already crossed, so the answer reads as the passage does.
    """
    if offsets:
        try:
            start_offset = int(offsets[start][0])
            end_offset = int(offsets[end][1])
        except (IndexError, TypeError, ValueError):
            start_offset = end_offset = -1
        if 0 <= start_offset < end_offset <= len(context):
            return _close_word(
                context[start_offset:end_offset], context, end_offset
            ).strip()

    return tokenizer.decode(
        tokenizer(context, truncation=True, max_length=MODEL_MAX_INPUT_LENGTH)[
            "input_ids"
        ][start : end + 1],
        skip_special_tokens=True,
    ).strip()


def _close_word(text: str, context: str, cut: int) -> str:
    """
    Finish the last word of ``text`` when the cut ran through it.

    ``text`` is a slice of ``context`` which ends at character ``cut``.  A
    character-position cut stops wherever the model's token ended, which can
    split a word in the middle; when the character right after the cut
    continues that same word, the rest of the word belongs to the answer and
    is appended.  Whitespace after the cut closes the span where it is.
    """
    if not text or cut >= len(context):
        return text

    following = context[cut]
    if not following.isalnum() or not text[-1].isalnum():
        return text

    end = cut
    while end < len(context) and context[end].isalnum():
        end += 1
    return text + context[cut:end]

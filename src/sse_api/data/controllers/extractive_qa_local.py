"""
Utility module for extractive question‑answering in the engine process.

The engine no longer owns the QA weights: extracting an answer is a
model‑service operation, and the engine posts the question together with the
retrieved passages to the ``POST /v1/extractive_qa`` endpoint of
``sse_models``.  This module is the historical in‑process fallback for
deployments which do not run the model service: it loads a HuggingFace
question‑answering model from the ``extractive_qa`` section of
``configs/models.json`` and answers with it.

``transformers`` is imported only when the model is really loaded, so
importing the engine stays independent of the model stack.
"""

import json
import logging
import os
from typing import Any, Dict, List, Tuple

from sse_api.core.constants import CONFIG_DIR

CACHED_MODELS = {}


def read_qa_config(config_path: str | None = None) -> dict:
    """Read the ``extractive_qa`` section of ``configs/models.json``."""
    path = config_path or os.path.join(CONFIG_DIR, "models.json")
    with open(path, "rt") as handle:
        config = json.load(handle)
    section = config.get("extractive_qa") or {}
    if not (section.get("model_path") or section.get("model_name")):
        raise ValueError(
            f"extractive_qa section of {path} has no model_path (or model_name)"
        )
    return section


def load_qa_pipeline(model: str, device: str = "cpu") -> Dict[str, Any]:
    """
    Load the in‑process QA model for ``model``.

    A requested CUDA device which is not available falls back to CPU instead
    of failing the call, so a CPU worker keeps answering (more slowly) rather
    than raising on every question.  The result is a
    ``{"model", "tokenizer"}`` bundle, the same shape the service uses, so a
    deployment that later turns on the service answers with the same numbers.
    """
    try:
        import torch
    except ImportError:
        torch = None

    if (
        torch is not None
        and str(device).startswith("cuda")
        and not torch.cuda.is_available()
    ):
        device = "cpu"

    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    loaded = AutoModelForQuestionAnswering.from_pretrained(model)
    loaded.to(device)
    loaded.eval()
    return {"model": loaded, "tokenizer": tokenizer, "path": model}


def _local_bundle() -> Dict[str, Any]:
    """The cached in‑process QA bundle, loaded on first use."""
    if CACHED_MODELS.get("qa_bundle") is not None:
        return CACHED_MODELS["qa_bundle"]

    config = read_qa_config()
    bundle = load_qa_pipeline(
        config.get("model_path") or config.get("model_name"),
        device=config.get("device") or "cpu",
    )
    CACHED_MODELS["qa_bundle"] = bundle
    return bundle


def answer_locally(
    question: str, contexts: List[str], bundle: Dict[str, Any] | None = None
) -> List[Tuple[str, float]]:
    """
    Answer ``question`` over ``contexts`` with the in‑process model.

    Returns one ``(answer, score)`` pair per context, in input order; an
    empty context answers with an empty string and a score of ``0.0`` without
    calling the model.  The extraction itself lives in the model service's
    ``sse_models.extractive_qa``: both paths answer with the same span and
    the same probability, so switching the work between the two changes no
    answer.  ``bundle`` is injectable for tests.
    """
    from sse_models.extractive_qa import answer as service_answer

    logger = logging.getLogger(__name__)
    if bundle is None:
        bundle = _local_bundle()

    answers: List[Tuple[str, float]] = []
    for context in contexts:
        if not context.strip():
            answers.append(("", 0.0))
            continue
        try:
            result = service_answer(bundle, question, [context])[0]
            answers.append(result)
        except Exception as exc:  # a broken passage must not lose the others
            logger.warning("Extractive QA failed for one passage: %s", exc)
            answers.append(("", 0.0))
    return answers

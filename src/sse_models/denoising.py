"""
denoising.py
------------

Text-cleaning inference for the model service (T5 encoder-decoder).

The logic is the engine's former ``DenoiserInference``: normalise the input,
split it into paragraphs, cut every paragraph into model-size chunks that end
at a sentence boundary, generate with the T5 weights and stitch the result
back together.  It lives here now because the engine no longer carries torch:
the engine posts the text to ``POST /v1/denoise`` and this module does the
forward pass.

``Levenshtein`` is the only non-stdlib import and it is imported lazily, so a
service without the metric installed still denoises (the correctness score
falls back to ``None`` -> reported as ``1.0`` by the caller).
"""

from typing import List, Optional

#: Tokens reserved for the prompt prefix and the model's own specials.
ADDITIONAL_TOKENS_COUNT = 20
#: Generation cap for one chunk; a denoiser rewrites, it does not compose.
MAX_NEW_TOKENS = 1024
#: Prompt the denoiser models are trained with.
DEFAULT_INPUT_MSG = "denoise"


def correctness_score(text_raw: str, text_clear: str) -> float:
    """``1 - (levenshtein / len(raw))`` in [0, 1]; 1.0 when identical."""
    if text_raw == text_clear:
        return 1.0
    if not text_raw:
        return 0.0
    try:
        import Levenshtein
    except ImportError:  # metric not installed on this service host
        return 1.0
    distance = Levenshtein.distance(text_raw, text_clear)
    return max(0.0, 1.0 - (distance / len(text_raw)))


def prepare_tokens_chunks(text_tokens, split_size: int):
    """Yield successive token-id slices of at most ``split_size`` tokens."""
    for i in range(0, len(text_tokens), split_size):
        yield text_tokens[i : i + split_size]


def reformat_chunks_to_proper_end(chunks_to_reformat: List[str]) -> List[str]:
    """Move a sentence split across two chunks onto the earlier chunk's tail.

    A chunk which ends mid-sentence makes the denoiser invent an ending;
    pushing the trailing fragment onto the next chunk (or pulling the head of
    the next chunk back to the last period) keeps every chunk self-contained.
    """
    new_chunks = []
    partial_chunk = None
    for chunk in chunks_to_reformat:
        if partial_chunk is not None:
            dot_position = chunk.find(".")
            if dot_position > -1:
                dot_position += 1
                partial_chunk += " " + chunk[:dot_position]
                new_chunks.append(partial_chunk)
                chunk = chunk[dot_position:]
            else:
                new_chunks.append(partial_chunk + " " + chunk)
                chunk = None
            partial_chunk = None

        if chunk is None:
            continue

        dot_position = chunk.rfind(".")
        if dot_position > len(chunk) / 2:
            dot_position += 1
            new_chunks.append(chunk[: dot_position + 1])
            partial_chunk = chunk[dot_position + 1 :]
        else:
            new_chunks.append(chunk)

    return new_chunks


class DenoisingInference:
    """
    Denoise text with a loaded ``{"model", "tokenizer"}`` bundle.

    ``bundle`` is what :meth:`sse_models.registry.ModelRegistry.denoiser`
    returns; keeping the weights outside means this class can be tested with
    a stub tokenizer and a stub generate().
    """

    def __init__(self, bundle: dict, batch_size: int = 50) -> None:
        self._model = bundle["model"]
        self._tokenizer = bundle["tokenizer"]
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    def split_text_to_max_chunks_len(
        self, text: str, input_msg: str, max_model_input: int
    ) -> List[str]:
        """Cut ``text`` into chunks of at most ``max_model_input`` tokens."""
        prefix = f"{input_msg}: "
        prefix_tokens = self._tokenizer.encode(prefix)
        max_tokens_in_chunk = max_model_input - len(prefix_tokens)
        whole_text_tokens = self._tokenizer.encode(text)
        if len(whole_text_tokens) < max_model_input:
            return [text]

        tokens_after_chunking = [
            self._tokenizer.decode(chunk)
            for chunk in prepare_tokens_chunks(
                text_tokens=whole_text_tokens, split_size=max_tokens_in_chunk
            )
        ]
        return reformat_chunks_to_proper_end(tokens_after_chunking)

    def _split_to_batches(self, text_chunks):
        for i in range(0, len(text_chunks), self.batch_size):
            yield text_chunks[i : i + self.batch_size]

    # ------------------------------------------------------------------
    def run(self, text: str, input_msg: str, max_model_input: int) -> str:
        """Denoise a multi-paragraph string, paragraph by paragraph."""
        normalized = "\n".join(
            line.strip() for line in text.strip().split("\n")
        ).strip()

        denoised_text = ""
        for paragraph in normalized.split("\n\n"):
            denoised_text += (
                self._run_paragraph(
                    text=paragraph,
                    input_msg=input_msg,
                    max_model_input=max_model_input // 2,
                ).strip()
                + "\n\n"
            )
        return denoised_text.replace("\n\n\n", "\n\n").strip()

    def _run_paragraph(
        self, text: str, input_msg: str, max_model_input: int
    ) -> str:
        import torch

        max_model_input -= ADDITIONAL_TOKENS_COUNT
        text_chunks = self.split_text_to_max_chunks_len(
            text=text, input_msg=input_msg, max_model_input=max_model_input
        )

        corrected_sentences = []
        for chunks_batch in self._split_to_batches(text_chunks):
            input_texts = [f"{input_msg}: {chunk}" for chunk in chunks_batch]
            inputs = self._tokenizer(input_texts, return_tensors="pt", padding=True)
            device = next(self._model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                corrected_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=2,
                    early_stopping=True,
                )
            for idx, ids in enumerate(corrected_ids):
                corrected_sentences.append(
                    {
                        "raw": chunks_batch[idx],
                        "clear": self._tokenizer.decode(
                            ids, skip_special_tokens=True
                        ),
                    }
                )

        out_text = ""
        for entry in corrected_sentences:
            raw = entry["raw"]
            if not len(raw.strip()):
                continue
            if raw[0].isspace():
                out_text += " " + entry["clear"]
            elif raw[-1].isspace():
                out_text += entry["clear"] + " "
            else:
                out_text += entry["clear"]
        out_text = out_text.strip()
        if out_text.endswith("<"):
            out_text = out_text[:-1]
        return out_text.strip()


def denoise_text(
    bundle: dict,
    text: str,
    max_model_input: int = 256,
    batch_size: Optional[int] = None,
) -> (str, float):
    """Denoise ``text``; returns ``(cleaned_text, correctness_score)``."""
    inference = DenoisingInference(
        bundle, batch_size=batch_size or 50
    )
    cleaned = inference.run(
        text=text, input_msg=DEFAULT_INPUT_MSG, max_model_input=max_model_input
    )
    return cleaned, correctness_score(text, cleaned)

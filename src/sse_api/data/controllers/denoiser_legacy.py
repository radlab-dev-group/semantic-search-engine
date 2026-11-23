"""
denoiser_legacy.py
------------------

The historical in‑process T5 denoiser, kept for deployments which do not run
the model service.

``torch`` and ``transformers`` are imported only when the model is actually
loaded (inside :meth:`DenoiserInference.__init__`), so importing this module
— and with it the engine — costs nothing.  The clean entry point of the engine
is :class:`~sse_api.data.controllers.denoiser.DenoiserController`; this module
is its local fallback.
"""

import json
import logging
import os
from typing import List

from sse_api.core.constants import CONFIG_DIR


def read_denoiser_config(config_path: str | None = None) -> dict:
    """Read the ``denoiser`` section of ``configs/models.json``."""
    path = config_path or os.path.join(CONFIG_DIR, "models.json")
    with open(path, "rt") as handle:
        config = json.load(handle)
    section = config.get("denoiser") or {}
    if not section.get("model_path"):
        raise ValueError(f"denoiser section of {path} has no model_path")
    return section


class DenoiserInference:
    """
    Performs inference with a pretrained T5 model to “denoise” text.

        The class is responsible for:
        * loading the model and tokenizer,
        * splitting long inputs into model‑compatible chunks,
        * re‑formatting those chunks so sentences are not broken mid‑point,
        * batching the chunks for efficient GPU/CPU execution,
        * post‑processing the generated output back into a single cleaned string.

        Parameters
        ----------
        model_path: str
            Path (or model identifier) that ``transformers`` can use to load the
            ``T5ForConditionalGeneration`` model and its tokenizer.
        batch_size: int, default ``50``
            Number of text chunks to process in a single forward pass.
        device: str | None, default ``None``
            ``'cpu'`` or ``'cuda'``. If omitted the best available device is chosen
            automatically.
    """

    def __init__(
        self, model_path: str, batch_size: int = 50, device: str | None = None
    ):
        """
        Create a new inference instance.

        The constructor loads the model and moves it to the selected device.
        """
        import torch
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        self.batch_size = batch_size
        self.device = device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)

    @staticmethod
    def prepare_tokens_chunks(text_tokens, split_size):
        """
        Yield successive token chunks of a given size.

        Parameters
        ----------
        text_tokens : list[int]
            Token IDs produced by the tokenizer.
        split_size : int
            Desired maximum size of each chunk.

        Yields
        ------
        list[int]
            A slice of ``text_tokens`` with length ``<= split_size``.
        """
        for i in range(0, len(text_tokens), split_size):
            yield text_tokens[i : i + split_size]

    @staticmethod
    def calculate_correctness(text_raw: str, text_clear: str) -> float:
        """
        Return a similarity score based on Levenshtein distance.

        The score is ``1 - (distance / len(text_raw))`` and therefore lies in the
        interval ``[0, 1]`` where ``1`` means the strings are identical.

        Parameters
        ----------
        text_raw : str
            Original (noisy) text.
        text_clear : str
            Denoised text.

        Returns
        -------
        float
            Normalised correctness metric.
        """
        l_dist = Levenshtein.distance(text_clear, text_raw)
        bad_distance = 0.0
        if l_dist > 0:
            bad_distance = float(l_dist) / len(text_raw)
        return 1.0 - bad_distance

    @staticmethod
    def calculate_measure_name() -> str:
        """
        Return the human‑readable name of the correctness metric.
        """
        return "1.0 - (levenstein / len(text_raw))"

    @staticmethod
    def reformat_chunks_to_proper_end(chunks_to_reformat: List[str]):
        """
        Ensure each chunk ends at a sentence boundary.

        If a sentence is split across two chunks, the method merges the split part
        into a new chunk so that every returned chunk ends with a period (or the
        original text end).

        Parameters
        ----------
        chunks_to_reformat : List[str]
            List of raw text chunks produced by ``prepare_tokens_chunks``.

        Returns
        -------
        List[str]
            List of re‑formatted chunks where sentence boundaries are respected.
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
                new_chunk = chunk[: dot_position + 1]
                partial_chunk = chunk[dot_position + 1 :]
                new_chunks.append(new_chunk)
            else:
                new_chunks.append(chunk)

        return new_chunks

    def split_text_to_max_chunks_len(
        self, text, input_msg: str, max_model_input: int
    ) -> List[str]:
        """
        Split a long text into model‑size‑compatible chunks.

        The method tokenizes ``text``, reserves space for the ``input_msg`` prefix,
        and then yields a list of strings that fit within ``max_model_input`` tokens.

        Parameters
        ----------
        text : str
            Full input text that may exceed the model’s maximum token length.
        input_msg : str
            Prefix that will be added to each chunk (e.g. ``'denoise'``).
        max_model_input : int
            Maximum number of tokens the model can accept, including the prefix.

        Returns
        -------
        List[str]
            List of chunk strings ready for inference.
        """
        input_msg = f"{input_msg}: "
        input_msg_tokens = self.tokenizer.encode(input_msg, return_tensors="pt")[0]
        max_tokens_in_chunk = max_model_input - len(input_msg_tokens)
        whole_text_tokens = self.tokenizer.encode(text, return_tensors="pt")[0]
        if len(whole_text_tokens) < max_model_input:
            return [text]

        tokens_after_chunking = [
            self.tokenizer.decode(ch)
            for ch in self.prepare_tokens_chunks(
                text_tokens=whole_text_tokens, split_size=max_tokens_in_chunk
            )
        ]

        return self.reformat_chunks_to_proper_end(tokens_after_chunking)

    def _split_to_batches(self, text_chunks):
        """
        Yield successive batches of ``text_chunks`` according to ``batch_size``.
        """
        for i in range(0, len(text_chunks), self.batch_size):
            yield text_chunks[i : i + self.batch_size]

    def do_inference(self, text, input_msg: str, max_model_input: int):
        """
        High‑level entry point that denoises a multi‑paragraph string.

        The method normalizes whitespace, splits the text into paragraph blocks,
        runs each block through ``_do_inference`` and finally stitches the results
        back together.

        Parameters
        ----------
        text : str
            Raw text possibly containing several paragraphs.
        input_msg : str
            Prompt prefix (e.g. ``'denoise'``).
        max_model_input : int
            Maximum token length accepted by the model.

        Returns
        -------
        str
            Denoised version of the input text.
        """
        n_text = ""
        for t_lin in text.strip().split("\n"):
            n_text += t_lin.strip() + "\n"
        text = n_text.strip()

        denoised_text = ""
        for text_spl in text.strip().split("\n\n"):
            text_spl_denoised = self._do_inference(
                text=text_spl,
                input_msg=input_msg,
                max_model_input=max_model_input // 2,
            )
            denoised_text += text_spl_denoised.strip() + "\n\n"
        return denoised_text.replace("\n\n\n", "\n\n").strip()

    def _do_inference(self, text, input_msg: str, max_model_input: int):
        """
        Core inference routine that works on a single paragraph.

        It performs chunking, batching, model generation, and post‑processing.

        Parameters
        ----------
        text : str
            Paragraph to denoise.
        input_msg : str
            Prompt prefix.
        max_model_input : int
            Token budget for the model (reduced internally for safety).

        Returns
        -------
        str
            Cleaned paragraph.
        """
        # denoise: + <s>
        additional_tokens_count = 20
        max_model_input -= additional_tokens_count

        text_chunks = self.split_text_to_max_chunks_len(
            text=text,
            input_msg=input_msg,
            max_model_input=max_model_input,
        )

        all_corrected_ids = []
        text_chunks_batches = self._split_to_batches(text_chunks)
        for chunks_batch in text_chunks_batches:
            input_texts = [f"{input_msg}: {chunk_str}" for chunk_str in chunks_batch]
            inputs_texts = self.tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                # max_length=max_model_input,
            )
            inputs_texts.to(self.device)

            import torch

            with torch.no_grad():
                corrected_ids = self.model.generate(
                    **inputs_texts,
                    max_new_tokens=1024,
                    num_beams=2,
                    early_stopping=True,
                )
            all_corrected_ids.append(corrected_ids)

        corrected_sentences = []
        for corrected_ids in all_corrected_ids:
            for idx, ci in enumerate(corrected_ids):
                corrected_sentences.append(
                    {
                        "raw": text_chunks[idx],
                        "clear": self.tokenizer.decode(ci, skip_special_tokens=True),
                    }
                )
        out_text = ""
        for text in corrected_sentences:
            if not len(text["raw"].strip()):
                continue
            add_ws_beg, add_ws_end = False, False
            if text["raw"][0].isspace():
                add_ws_beg = True
            elif text["raw"][-1].isspace():
                add_ws_end = True
            if add_ws_beg:
                out_text += " " + text["clear"]
            elif add_ws_end:
                out_text += text["clear"] + " "
            else:
                out_text += text["clear"]
        out_text = out_text.strip()
        if out_text.endswith("<"):
            out_text = out_text[:-1]
        return out_text.strip()


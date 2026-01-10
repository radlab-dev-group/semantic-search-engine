import torch
import Levenshtein
from typing import List
from transformers import T5ForConditionalGeneration, T5Tokenizer


class DenoiserInference:
    def __init__(
        self, model_path: str, batch_size: int = 50, device: str | None = None
    ):
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
        Split text tokens into chunks of size split_size
        :param text_tokens: List of tokens to split
        :param split_size: Size of single chunk
        :return: Yielding lists of tokens
        """
        for i in range(0, len(text_tokens), split_size):
            yield text_tokens[i : i + split_size]

    @staticmethod
    def calculate_correctness(text_raw: str, text_clear: str) -> float:
        l_dist = Levenshtein.distance(text_clear, text_raw)
        bad_distance = 0.0
        if l_dist > 0:
            bad_distance = float(l_dist) / len(text_raw)
        return 1.0 - bad_distance

    @staticmethod
    def calculate_measure_name() -> str:
        return "1.0 - (levenstein / len(text_raw))"

    @staticmethod
    def reformat_chunks_to_proper_end(chunks_to_reformat: List[str]):
        """
        Reformat chunks to proper sentence end. In case when sentence is split
        between two chunks, then will be extracted as additional chunk.

        :param chunks_to_reformat: List of chunks to reformat
        :return: List of reformatted chunks (strings)
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
        for i in range(0, len(text_chunks), self.batch_size):
            yield text_chunks[i : i + self.batch_size]

    def do_inference(self, text, input_msg: str, max_model_input: int):
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

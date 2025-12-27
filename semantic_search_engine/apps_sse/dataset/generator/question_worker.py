import json
import queue
import threading


from radlab_data.utils.threads import ThreadWorker
from radlab_data.utils.api_handler import APIHandler


class LLamaHandler(ThreadWorker):
    def __init__(
        self,
        base_api_url: str,
        dataset_read_queue: queue.Queue,
        results_write_queue: queue.Queue,
        batch_size: int = 8,
    ) -> None:
        self._batch_size = batch_size
        self._dataset_read_queue = dataset_read_queue
        self._results_write_queue = results_write_queue

        self._api = APIHandler(base_api_url=base_api_url)
        super().__init__()

    def _task(self):
        while self._enabled:
            doc_pages = []
            for _ in range(self._batch_size):
                try:
                    document_page = self._dataset_read_queue.get(timeout=10)
                    doc_pages.append(document_page)
                    self._dataset_read_queue.task_done()
                except queue.Empty:
                    break
            if not doc_pages:
                continue

            request_data = {
                "number_of_questions": 3,
                "texts": [d.text_str for d in doc_pages],
                "model_name": "radlab/pLLama3.1-8B-DPO-L-30k-lora32_16-8bit",
                "proper_input": True,
                "post_proc_llama_output": True,
                "top_k": 50,
                "top_p": 0.95,
                "max_new_tokens": None,
                "temperature": 0.65,
                "typical_p": 1.0,
                "repetition_penalty": 1.05,
            }

            response_data = self._api.call(
                endpoint="generate_questions",
                data=request_data,
                method="POST",
            )

            if "response" not in response_data:
                print("error")
                print(response_data)
                continue

            for text_response in response_data["response"]:
                data = {
                    "text": text_response["text"],
                    "label": text_response["questions"],
                }
                self._results_write_queue.put(data)


file_write_lock = threading.Lock()


class FileWriter(ThreadWorker):
    def __init__(
        self, output_file_path: str, file_writer_queue: queue.Queue
    ) -> None:
        self._file_writer_queue = file_writer_queue
        self._output_file_path = output_file_path
        super().__init__()

    def _task(self) -> None:
        while self._enabled:
            try:
                entry = self._file_writer_queue.get(timeout=10)
            except queue.Empty:
                continue
            with file_write_lock, open(self._output_file_path, "a") as outfile:
                print("writing to file", outfile)
                data = json.dumps(entry, ensure_ascii=False)
                outfile.write(data + "\n")
                outfile.flush()
                self._file_writer_queue.task_done()

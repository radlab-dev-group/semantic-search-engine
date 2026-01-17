from transformers import pipeline


class ExtractiveQAController:
    """
    Controller that runs extractive question‑answering using a HuggingFace pipeline.
    """

    model_path = "radlab/polish-qa-v2"

    def __init__(
        self, model_path: str | None, qa_pipeline=None, device: str = "cpu"
    ):
        """
        Initialise the controller.

        Parameters
        ----------
        model_path : str | None
            Path or identifier of the model to load if ``qa_pipeline`` is not
            provided.
        qa_pipeline : object | None
            An already‑initialised ``pipeline`` object. If supplied, ``model_path``
            is ignored.
        device : str
            Device to run the model on (e.g., ``"cpu"`` or ``"cuda"``).
        """
        assert model_path is not None or qa_pipeline is not None

        self.device = device
        if qa_pipeline is not None:
            self.question_answerer = qa_pipeline
        else:
            self.question_answerer = self.load_model(model_path)

    def load_model(self, model_path):
        """
        Load a HuggingFace “question‑answering” pipeline.

        Parameters
        ----------
        model_path : str
            Model identifier or path.

        Returns
        -------
        pipeline
            A ready‑to‑use question‑answering pipeline.
        """
        return pipeline("question-answering", model=model_path, device=self.device)

    def run_extractive_qa(self, question_str: str, search_results: dict):
        """
        Perform extractive QA on a set of retrieved passages.

        Parameters
        ----------
        question_str : str
            The user’s question.
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
        document_answers = {}
        for answer in search_results["results"]:
            answer_str = answer["result"]["text"]["text_str"]
            document_name = answer["result"]["text"]["document_name"]
            page_number = answer["result"]["text"]["page_number"]
            text_number = answer["result"]["text"]["text_number"]

            q_res = self.question_answerer(question=question_str, context=answer_str)
            document_answers[document_name] = {
                "page_number": page_number,
                "text_number": text_number,
                "answer": q_res["answer"],
                "score": q_res["score"],
            }
        return document_answers

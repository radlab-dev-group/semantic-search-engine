import json
import logging
from radlab_cleaner.cleaner.t5_inference import DenoiserInference


CACHED_MODELS = {}


class DenoiserController:
    JSON_FIELD_DEVICE = "device"
    JSON_FIELD_DENOISER = "denoiser"
    JSON_FIELD_DENOISER_MODEL_NAME = "model_name"
    JSON_FIELD_DENOISER_MODEL_PATH = "model_path"

    def __init__(
        self, load_model: bool = True, config_path: str = "configs/models.json"
    ):
        assert config_path is not None and len(config_path)
        self._config_path = config_path

        self._device = None
        self._model_inference = None
        self._model_config = None
        if load_model:
            self._load_model()

    def load(self, model_name_or_path: str | None = None):
        if self.JSON_FIELD_DENOISER in CACHED_MODELS:
            logging.info("Loading denoiser model from cache")
            self._model_inference = CACHED_MODELS[self.JSON_FIELD_DENOISER]

        return self._load_model(path=model_name_or_path)

    def denoise_text(self, text: str) -> str:
        assert self._model_inference is not None
        if not len(text.strip()):
            return text
        return self._model_inference.do_inference(
            text=text, input_msg="denoise", max_model_input=256
        )

    def _load_model(self, path: str | None = None):
        self._model_inference = None
        self._model_config = json.load(open(self._config_path, "rt"))

        model_path = None
        if path is not None:
            model_path = model_path
        else:
            model_path = self._model_config[self.JSON_FIELD_DENOISER][
                self.JSON_FIELD_DENOISER_MODEL_PATH
            ]

        logging.info(f"Loading denoiser model from {model_path}")

        self._device = self._model_config[self.JSON_FIELD_DENOISER][
            self.JSON_FIELD_DEVICE
        ]

        self._model_inference = DenoiserInference(
            model_path=model_path, device=self._device
        )
        CACHED_MODELS[self.JSON_FIELD_DENOISER] = self._model_inference

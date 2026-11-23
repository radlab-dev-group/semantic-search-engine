"""
Utility module for text "denoising" (cleaning noisy or error‑prone input).

The engine no longer owns the T5 weights: text cleaning is a model‑service
operation, and :class:`DenoiserController` posts the text to the
``POST /v1/denoise`` endpoint of ``sse_models`` (or to any model service that
implements it) through the engine's ``ModelBackend`` abstraction.  When the
model service is not configured the controller degrades gracefully — it logs
a clear warning and returns the original text, so indexing never fails on a
missing T5 model.

The historical in‑process T5 path is kept behind
:func:`denoise_text_locally`; it imports torch / transformers lazily, so a
process that never cleans a text never loads them.
"""

import json
import logging
import os

from typing import List

from sse_api.core.constants import CONFIG_DIR

CACHED_MODELS = {}


def denoise_text_locally(text: str) -> str:
    """
    Clean ``text`` with the in‑process T5 denoiser (the pre‑service path).

    ``torch`` and ``transformers`` are imported inside this function, so a
    module import of the engine stays lightweight and a deployment that
    delegates cleaning to the model service never touches the model stack.
    """
    if not len(text.strip()):
        return text
    inference = _local_inference()
    return inference.do_inference(
        text=text, input_msg="denoise", max_model_input=256
    )


def _local_inference():
    """The cached local :class:`DenoiserInference`, loaded on first use."""
    if CACHED_MODELS.get("denoiser") is not None:
        return CACHED_MODELS["denoiser"]

    from sse_api.data.controllers.denoiser_legacy import (  # noqa: PLC0415
        DenoiserInference,
        read_denoiser_config,
    )

    config = read_denoiser_config()
    inference = DenoiserInference(
        model_path=config["model_path"],
        device=config.get("device"),
    )
    CACHED_MODELS["denoiser"] = inference
    return inference


class DenoiserController:
    """
    High‑level façade for text denoising.

    The controller is deliberately thin:

    * with the model service configured (``SSE_MODEL_SERVICE_ENABLED``) the
      text is cleaned over HTTP by ``sse_models``;
    * when the service is unavailable a warning is logged and the original
      text is returned, so a missing denoiser never breaks an indexing job;
    * :meth:`denoise_text` stays the single public entry point used by the
      ingestion pipeline.
    """

    JSON_FIELD_DEVICE = "device"
    JSON_FIELD_DENOISER = "denoiser"
    JSON_FIELD_DENOISER_MODEL_NAME = "model_name"
    JSON_FIELD_DENOISER_MODEL_PATH = "model_path"

    def __init__(
        self,
        load_model: bool = True,
        config_path: str = os.path.join(CONFIG_DIR, "models.json"),
    ):
        """
        Initialise the controller.

        Parameters
        ----------
        load_model : bool, default ``True``
            Kept for call‑site compatibility. Nothing is loaded eagerly: the
            model service is stateless per request and the local T5 (when
            used) loads on the first ``denoise_text`` call.
        config_path : str, default ``configs/models.json``
            Path to the JSON configuration that contains model information.
        """
        assert config_path is not None and len(config_path)
        self._config_path = config_path
        self._device = None
        self._model_inference = None
        self._model_config = None
        self._logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    def load(self, model_name_or_path: str | None = None):
        """
        Pre‑warm the local denoiser (no‑op for the HTTP path).

        Returns the loaded inference object when the local model could be
        loaded; ``None`` when the model is not available (the caller should
        then expect :meth:`denoise_text` to return the input unchanged).
        """
        try:
            return _local_inference()
        except Exception as exc:  # missing weights, no torch, bad config
            self._logger.warning(
                "Local denoiser is not available (%s); denoising will be "
                "delegated to the model service or skipped",
                exc,
                exc_info=False,
            )
            return None

    def denoise_text(self, text: str) -> str:
        """
        Clean ``text``.

        Order of preference: the model service (when enabled), then the
        in‑process T5 (when its weights are available). When neither answers
        the original text is returned with a warning, so the caller's
        indexing flow continues.

        Parameters
        ----------
        text : str
            Input string to be cleaned. Empty or whitespace‑only strings
            are returned unchanged.

        Returns
        -------
        str
            Denoised output (or the input, when nothing could clean it).
        """
        if not len(text.strip()):
            return text

        from sse_api.engine.core.models.backend import (  # noqa: PLC0415
            build_denoise_client,
        )

        service_client = build_denoise_client()
        if service_client is not None:
            try:
                return service_client(text)
            except Exception as exc:  # dead service: fall through to local
                self._logger.warning(
                    "Model service denoising failed (%s); trying the local "
                    "denoiser",
                    exc,
                    exc_info=False,
                )

        try:
            return _local_inference().do_inference(
                text=text, input_msg="denoise", max_model_input=256
            )
        except Exception as exc:  # no T5 weights / no torch on this host
            self._logger.warning(
                "Text denoising is unavailable (model service off and no "
                "local T5 model: %s); the original text is used",
                exc,
                exc_info=False,
            )
            return text

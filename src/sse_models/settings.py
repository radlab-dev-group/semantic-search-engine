"""
settings.py
-----------

Environment-driven configuration for the standalone model service.

Every knob is an environment variable so the same image can be pointed at a
different host / port / model set without a rebuild.  The service reads the
*same* ``configs/embedders.json`` / ``configs/rerankers.json`` files the main
engine uses (via ``SSE_CONFIG_DIR``), so a model that the engine knows about is
automatically available here.
"""

import os


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class ModelServiceSettings:
    """Runtime settings for the model service (one instance per process)."""

    def __init__(self, env: dict | None = None) -> None:
        env = env if env is not None else os.environ

        self.host = env.get("SSE_MODELS_HOST", "0.0.0.0")
        self.port = _as_int(env.get("SSE_MODELS_PORT"), 8280)

        # Which role this instance serves.  ``all`` loads both embedder and
        # reranker models on demand; ``embed``/``rerank`` only serve their
        # endpoint (the other returns 503).  This is what lets the same image
        # be deployed as a cheap embedder fleet and a beefier reranker fleet.
        self.role = env.get("SSE_MODELS_ROLE", "all").strip().lower()
        # ``denoise`` adds the T5 text-cleaning model, which is only loaded
        # when that endpoint is asked for.
        if self.role not in {"all", "embed", "rerank", "denoise"}:
            self.role = "all"

        # Optional shared secret.  When set, /v1/* requests must carry
        # ``Authorization: Bearer <token>``.  Health endpoints stay open.
        self.api_token = env.get("SSE_MODELS_API_TOKEN") or ""

        # Where the model catalog (embedders.json / rerankers.json) lives.
        self.config_dir = env.get("SSE_CONFIG_DIR") or os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "configs",
        )

        # Preload every active model at startup (useful behind a load
        # balancer that expects an instant ready signal).  Default off so a
        # fresh boot is fast; models load lazily on first use.
        self.warm_models = _as_bool(env.get("SSE_MODELS_WARM_MODELS"), False)

        # Inference batch size.  Unset means "whatever the model defaults to",
        # which keeps the service numerically identical to the engine calling
        # the model itself; some tokenizers have no pad token and fail on any
        # explicit batch, so this is not set unless asked for.
        self.batch_size = _as_int(env.get("SSE_MODELS_BATCH_SIZE"), 0)

        # Max number of documents a single /v1/rerank call may score.
        self.rerank_max_documents = _as_int(
            env.get("SSE_MODELS_RERANK_MAX_DOCUMENTS"), 256
        )

        # Token window the denoiser feeds to T5 (the model's own limit; the
        # engine's former value, so behaviour does not change when the work
        # moves from the engine process to this service).
        self.denoise_max_model_input = _as_int(
            env.get("SSE_MODELS_DENOISE_MAX_MODEL_INPUT"), 256
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @property
    def embed_enabled(self) -> bool:
        return self.role in {"all", "embed"}

    @property
    def rerank_enabled(self) -> bool:
        return self.role in {"all", "rerank"}

    @property
    def denoise_enabled(self) -> bool:
        """Text denoising (T5); off for the two pure inference roles."""
        return self.role in {"all", "denoise"}

    def embedders_config_path(self) -> str:
        return os.path.join(self.config_dir, "embedders.json")

    def rerankers_config_path(self) -> str:
        return os.path.join(self.config_dir, "rerankers.json")

    def token_required(self) -> bool:
        return bool(self.api_token)


def get_settings() -> ModelServiceSettings:
    """Return a process-wide settings object (created once)."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = ModelServiceSettings()
    return _SETTINGS


_SETTINGS: ModelServiceSettings | None = None

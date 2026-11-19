"""
__main__.py
-----------

Entrypoint for the model service: ``python -m sse_models`` (or ``sse-models``).

Configures logging, applies the startup model warm-up policy and hands over to
uvicorn.  All knobs come from the environment — see ``sse_models.settings``.
"""

import logging
import os
import sys

import uvicorn

from sse_models.app import create_app
from sse_models.registry import ModelRegistry
from sse_models.settings import ModelServiceSettings


def _configure_logging() -> None:
    level = os.environ.get("SSE_MODELS_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        stream=sys.stdout,
    )


def main(argv=None) -> int:
    _configure_logging()
    logger = logging.getLogger("sse_models")

    settings = ModelServiceSettings()
    registry = ModelRegistry(settings)

    logger.info(
        "model service role=%s config_dir=%s warm_models=%s",
        settings.role,
        settings.config_dir,
        settings.warm_models,
    )

    if settings.warm_models:
        # Warm-up runs before uvicorn binds the port, so the first request
        # after the process reports ready never pays a model load.
        for key, state in registry.warm().items():
            logger.info("warm %-45s %s", key, state)

    app = create_app(settings=settings, registry=registry)

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=os.environ.get("SSE_MODELS_LOG_LEVEL", "info").lower(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

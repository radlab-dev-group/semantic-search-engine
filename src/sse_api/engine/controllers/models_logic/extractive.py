# Extractive QA has one implementation, in :mod:`sse_api.engine.core.
# generative.extractive`; it answers through the model service
# (``POST /v1/extractive_qa``) and loads no model in this process unless a
# caller injects its own pipeline.  The controller is re-exported here so the
# ``controllers`` package keeps the import path it has always had.

from sse_api.engine.core.generative.extractive import ExtractiveQAController

__all__ = [
    "ExtractiveQAController",
]

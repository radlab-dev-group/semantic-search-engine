"""
engine/controllers/database
---------------------------

The controllers of the relational database of the engine, reachable from
the package itself::

    from sse_api.engine.controllers.database import (
        PublicRelationDBController,
        RelationalDBController,
    )

The package used to be empty, which left the tools of the repository
importing names it did not export.  The controllers are imported lazily
(on the first attribute access), because reading the module of the
relational controllers pulls in the document readers and the models of the
engine: a consumer of ``database.milvus`` should not pay for that, and a
deployment which decoupled the ML stack from the engine core keeps it
decoupled.
"""

import importlib

#: The controllers of this package and the module which defines each of them.
_EXPORTS = {
    "RelationalDBController": "sse_api.engine.controllers.database.relational_db",
    "PublicRelationDBController": (
        "sse_api.engine.controllers.database.relational_db"
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

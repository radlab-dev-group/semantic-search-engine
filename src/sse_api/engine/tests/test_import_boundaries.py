"""
The engine core must import without the model stack.

Plan 01 of the decoupling roadmap: an API or worker host which delegates
inference to ``sse_models`` installs ``requirements.txt`` only — no ``torch``,
no ``transformers``, no ``sentence-transformers``.  That holds as long as no
module of the engine imports them at module level, which is what these tests
guard: they import the engine's packages in a subprocess in which those
modules raise ``ImportError`` on purpose.

A subprocess, because the rest of the test suite runs with the model stack
installed (and imports it), and an already imported module cannot be
un-imported here.
"""

import os
import subprocess
import sys
import textwrap

import pytest

#: The heavy modules an inference-free engine host does not have.
BLOCKED = ("torch", "transformers", "sentence_transformers")

#: Packages and modules which have to import with those missing.
#:
#: The list is the engine's own surface — the API-facing controllers, the
#: core, the model backend — not every file of the repository: the in-process
#: fallbacks (``denoiser_legacy``, ``extractive_qa_local``) may and do use the
#: model stack, but only from inside a function.
ENGINE_IMPORTS = (
    "sse_api.engine",
    "sse_api.engine.core",
    "sse_api.engine.core.models.backend",
    "sse_api.engine.core.generative.extractive",
    "sse_api.engine.controllers",
    "sse_api.engine.controllers.search.semantic",
    "sse_api.engine.controllers.database.relational_db",
    "sse_api.engine.controllers.database.milvus",
    "sse_api.engine.controllers.models_logic.extractive",
    "sse_api.engine.core.search_engine",
    "sse_api.engine.database.vector_store_adapter",
    "sse_api.data.controllers",
    "sse_api.data.controllers.denoiser",
    "sse_api.data.controllers.chunking",
    "sse_api.data.controllers.extractive_qa_local",
)

_PROBE = textwrap.dedent("""
    import builtins, sys

    BLOCKED = set(sys.argv[1].split(","))
    _real_import = builtins.__import__

    def _guarded(name, *args, **kwargs):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(
                f"{name} is not installed (an inference-free engine host)"
            )
        return _real_import(name, *args, **kwargs)

    builtins.__import__ = _guarded

    import django
    django.setup()

    failed = []
    for module in sys.argv[2].split(","):
        try:
            __import__(module)
        except Exception as exc:  # noqa: BLE001 - reported per module
            failed.append(f"{module}: {type(exc).__name__}: {exc}")

    loaded = sorted(
        name
        for name in sys.modules
        if name.split(".")[0] in BLOCKED
    )
    if loaded:
        failed.append("model stack imported anyway: " + ", ".join(loaded))
    if failed:
        print("\\n".join(failed))
        sys.exit(1)
    """)


@pytest.mark.parametrize("module", ENGINE_IMPORTS)
def test_engine_module_imports_without_the_model_stack(module):
    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )  # .../src
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        os.path.join(root, "src") + os.pathsep + environment.get("PYTHONPATH", "")
    )
    environment["DJANGO_SETTINGS_MODULE"] = "sse_api.tests.settings_django"

    result = subprocess.run(
        [sys.executable, "-c", _PROBE, ",".join(BLOCKED), module],
        capture_output=True,
        text=True,
        env=environment,
        cwd=root,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"{module} cannot be imported without {', '.join(BLOCKED)}\n"
        f"{result.stdout}\n{result.stderr[-2000:]}"
    )

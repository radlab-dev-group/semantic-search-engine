"""
conftest.py
-----------

Makes ``src/sse_lib`` importable without installing the package, so the tests
run from a plain checkout (``python -m pytest sse_lib`` from the repository
root as well as ``python -m unittest`` inside ``sse_lib``).
"""

import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parent.parent / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

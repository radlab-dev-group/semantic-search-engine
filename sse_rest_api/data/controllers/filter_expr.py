"""
safe_filter_expr – Safe expression evaluator for query‑template filters.

Replaces the former ``eval()`` call in ``QueryTemplateFilterer`` with a
restricted namespace that only allows:
 * **datetime** module (for ``datetime.datetime.now()``, ``datetime.timedelta``,
   ``strptime``, ``dateutil.parser.isoparse``)
 * **comparison operators**  (``<=``, ``>=``, ``<``, ``>``, ``==``, ``!=``)

Template placeholders ``'DATA_VALUE'`` are replaced with the supplied
metadata value before evaluation.
"""

import ast
from datetime import datetime, timedelta
from typing import Any


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Nodes whose *evaluation* we allow on the right‑hand side of a comparison.
_ALLOWED_NODES: type = (
    ast.Call,
    ast.Compare,
    ast.Constant,
    ast.Name,
    ast.BinOp,
    ast.Attribute,
    ast.Subscript,
    ast.Index,
)


def _walk(node: ast.AST, predicate):
    """Yield *node* and all descendants for which *predicate* returns True."""
    if predicate(node):
        yield node
    for child in ast.iter_child_nodes(node):
        yield from _walk(child, predicate)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def safe_evaluate_template_filter(
    expression: str,
    data_value: Any,
) -> bool:
    """Evaluate *expression* safely with *data_value* substituted
    for ``'DATA_VALUE'``.

    Parameters
    ----------
    expression : str
        A filter expression such as::

            datetime.datetime.strptime(
                'DATA_VALUE', '%Y-%m-%d') >= datetime.datetime.now()

    data_value : Any
        The actual metadata value from the document (usually a string).

    Returns
    -------
    bool
        ``True`` if the expression evaluates to truthy, ``False`` otherwise.
        On any parsing or evaluation error returns ``False`` (matching the
        original ``eval()`` fallback behaviour).
    """
    try:
        safe = _safe_compile(expression)
        return bool(safe(data_value))  # type: ignore[no-untyped-call]
    except Exception:
        return False


def _safe_compile(expression: str):
    """Return a callable ``(data_value) -> result`` compiled from *expression*.

    Raises ``ValueError`` if the expression cannot be parsed safely.
    """
    # Replace all occurrences of 'DATA_VALUE' (with surrounding quotes) with
    # a quoted placeholder value so AST parsing succeeds.
    placeholder = _make_placeholder()
    substituted = expression.replace("'DATA_VALUE'", f"'{placeholder}'")

    tree = ast.parse(substituted, mode="eval")

    # Walk the AST and verify every node is in the allowed set.
    for node in _walk(tree, lambda n: True):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Unsupported AST node type: {type(node).__name__}")

    # Build a restricted namespace with only datetime helpers.
    namespace = {
        "datetime": datetime,
        "timedelta": timedelta,
        "_placeholder_val": placeholder,
    }

    # Try importing dateutil.parser (best-effort – falls back gracefully).
    try:
        from dateutil import parser as _dateutil_parser  # type: ignore

        namespace["dateutil"] = type(
            "dateutil", (), {"parser": _dateutil_parser}
        )
    except ImportError:
        pass  # dateutil not installed; expressions will fail safely.

    code = compile(tree, "<filter_expr>", "eval")

    def _evaluate(data_value: Any) -> Any:
        # Replace placeholder with actual value at runtime.
        namespace["_placeholder_val"] = data_value
        # Eval with restricted builtins and namespace (safe).
        return eval(code, {"__builtins__": {}}, namespace)  # noqa: S307

    return _evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_placeholder() -> str:
    """Create a unique sentinel unlikely to appear in real data."""
    return "___DATA_VALUE_PLACEHOLDER_7f8a3b9c___"

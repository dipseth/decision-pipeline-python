"""
The expression seam for JSON specs (spec.py) and `expr` metrics (hypothesis.py).

An engine compiles a source string once (raising on a syntax error, so a bad
spec fails at load) into something that evaluates against a root value plus
named bindings. `decision_pipeline.jsonata` is the JSONata engine; any
engine with this shape works.
"""

from __future__ import annotations

from typing import Any, Awaitable, Mapping, Optional, Protocol, Union, runtime_checkable


@runtime_checkable
class CompiledExpression(Protocol):
    def evaluate(self, root: Any, bindings: Mapping[str, Any]) -> Union[Any, Awaitable[Any]]: ...

    def roots(self) -> Optional[list[str]]:
        """
        Top-level names the expression reads off its ROOT (`bundle` in
        `bundle.title`), when the engine can tell. Conservative: a name it
        cannot place is left out. Return None when the engine cannot tell at all.
        """
        ...


@runtime_checkable
class ExpressionEngine(Protocol):
    def compile(self, source: str) -> CompiledExpression:
        """Raises on a syntax error — so a bad expression fails at load, not mid-run."""
        ...

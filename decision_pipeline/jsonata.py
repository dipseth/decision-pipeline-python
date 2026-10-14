"""
The JSONata `ExpressionEngine` for ./spec and `expr` metrics — an optional
module (`pip install decision-pipeline[jsonata]`), so the core's only runtime
dependency stays pydantic and a host that never compiles a spec never loads
JSONata. Built on `jsonata-python`.

Why JSONata: one language for predicates (`row.confidence >= $t.gate`) AND
reshaping (`{ "slug": top[0].slug }`); no I/O. JSON Logic cannot map; CEL
predicates well but reshapes poorly.

Two JSONata habits a spec author (or model) should know:
  - a path over an array of ONE element yields the element, not the array;
    `cuisines[]` keeps it an array. Plumbing belongs in `inputs`, whose
    resolver has no such rule.
  - a missing path is `undefined` (None here), never an error; `$exists(x)` tests it.

Every evaluation is timeboxed (time and depth) through JSONata's evaluate
entry/exit hooks, which live on the evaluation FRAME. A compiled pipeline is
shared by concurrent requests, where counters on one shared expression would
be summed across them. So `compile` parses once to fail a bad spec at load,
and each evaluation parses its own copy with its own frame and counters.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from jsonata import Jsonata  # type: ignore[import-untyped]
from jsonata.jexception import JException  # type: ignore[import-untyped]

from .expressions import CompiledExpression, ExpressionEngine


def _as_error(err: BaseException, source: str) -> BaseException:
    """A JSONata failure as `<code>: <message> at <position> in \\`<source>\\``; any other error as-is."""
    if not isinstance(err, JException):
        return err
    code = getattr(err, "error", None) or "JSONata"
    location = getattr(err, "location", None)
    at = "" if location is None or location < 0 else f" at {location}"
    return Exception(f"{code}: {err}{at} in `{source}`")


def _plain(v: Any) -> Any:
    """
    JSONata marks the arrays it builds as sequences (a list subclass). Hand back
    plain lists and dicts so nothing downstream — pydantic, a record, a
    Langfuse payload — ever sees the marker.
    """
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    return v


def _root_names(node: Any, out: dict[str, None]) -> None:
    """
    Names read off the ROOT. Descends only where JSONata keeps the root as the
    context (operators, conditions, blocks, object/array constructors, function
    arguments, the first step of a path). Later path steps, filters, sorts and
    lambda bodies are relative to something else, so their names are fields,
    not roots, and are left alone — the walk can miss a root, never invent one.
    """
    if node is None or isinstance(node, (str, int, float, bool)):
        return
    t = getattr(node, "type", None)

    def walk(x: Any) -> None:
        _root_names(x, out)

    if t == "path":
        steps = getattr(node, "steps", None) or []
        first = steps[0] if steps else None
        if first is not None and getattr(first, "type", None) == "name" and isinstance(getattr(first, "value", None), str):
            out[first.value] = None
        else:
            walk(first)
    elif t == "name":
        v = getattr(node, "value", None)
        if isinstance(v, str):
            out[v] = None
    elif t == "binary":
        walk(getattr(node, "lhs", None))
        walk(getattr(node, "rhs", None))
    elif t == "unary":
        value = getattr(node, "value", None)
        pairs = getattr(node, "lhs_object", None)
        if value == "{" and isinstance(pairs, (list, tuple)):
            for pair in pairs:
                walk(pair[0])
                walk(pair[1])
        elif value == "[":
            for e in getattr(node, "expressions", None) or []:
                walk(e)
        else:
            walk(getattr(node, "expression", None))
    elif t == "condition":
        walk(getattr(node, "condition", None))
        walk(getattr(node, "then", None))
        walk(getattr(node, "_else", None))
    elif t == "block":
        for e in getattr(node, "expressions", None) or []:
            walk(e)
    elif t in ("function", "partial"):
        for e in getattr(node, "arguments", None) or []:
            walk(e)
    elif t == "bind":
        walk(getattr(node, "rhs", None))
    elif t == "apply":
        walk(getattr(node, "lhs", None))
        walk(getattr(node, "rhs", None))


class _ExpressionLimit(Exception):
    """Raised from the entry/exit hooks when an evaluation runs too deep or too long."""


@dataclass
class _CompiledJsonata:
    source: str
    ast: Any
    timeout_ms: float
    max_depth: int

    def roots(self) -> Optional[list[str]]:
        out: dict[str, None] = {}
        _root_names(self.ast, out)
        return list(out)

    def evaluate(self, root: Any, bindings: Mapping[str, Any]) -> Any:
        source = self.source
        started = time.monotonic() * 1000
        depth = 0
        max_depth = self.max_depth
        timeout_ms = self.timeout_ms

        def check() -> None:
            if depth > max_depth:
                raise _ExpressionLimit(f"expression exceeded depth {max_depth}: `{source}`")
            if time.monotonic() * 1000 - started > timeout_ms:
                raise _ExpressionLimit(f"expression exceeded {_ms(timeout_ms)} ms: `{source}`")

        def entry(_exp: Any, _input: Any, env: Any) -> None:
            nonlocal depth
            if getattr(env, "is_parallel_call", False):
                return
            depth += 1
            check()

        def exit_(_exp: Any, _input: Any, env: Any, _res: Any) -> None:
            nonlocal depth
            if getattr(env, "is_parallel_call", False):
                return
            depth -= 1
            check()

        try:
            expression = Jsonata(source)
            frame = expression.create_frame()
            for name, value in (bindings or {}).items():
                frame.bind(name, Jsonata.JLambda(value) if callable(value) else value)
            frame.set_evaluate_entry_callback(entry)
            frame.set_evaluate_exit_callback(exit_)
            out = expression.evaluate(root, frame)
            return _plain(out)
        except Exception as err:  # noqa: BLE001 - re-raised, formatted like TS
            formatted = _as_error(err, source)
            if formatted is err:
                raise
            raise formatted from err


def _ms(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else str(v)


@dataclass
class _JsonataEngine:
    timeout_ms: float = 250
    max_depth: int = 500

    def compile(self, source: str) -> CompiledExpression:
        try:
            ast = Jsonata(source).ast
        except Exception as err:  # noqa: BLE001
            formatted = _as_error(err, source)
            if formatted is err:
                raise
            raise formatted from err
        return _CompiledJsonata(source=source, ast=ast, timeout_ms=self.timeout_ms, max_depth=self.max_depth)


def jsonata_engine(timeout_ms: float = 250, max_depth: int = 500) -> ExpressionEngine:
    """
    `timeout_ms`: wall-clock budget per evaluation — these are small expressions
    over one run's state. `max_depth`: evaluation depth ceiling (runaway recursion).
    """
    return _JsonataEngine(timeout_ms=timeout_ms, max_depth=max_depth)

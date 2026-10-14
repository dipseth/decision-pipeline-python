"""
Small helpers that stand in for what TypeScript and zod gave the original for
free: an "absent" sentinel distinct from None, sync-or-async calls, schema
adaptors over pydantic, JS-compatible timestamps and number formatting.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Protocol, TypeVar, Union, runtime_checkable

T = TypeVar("T")


class _Missing:
    """TS `undefined` where it has to differ from `null`. Internal: never returned to a caller."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "MISSING"

    def __bool__(self) -> bool:
        return False


MISSING: Any = _Missing()


async def maybe_await(value: Union[T, Awaitable[T]]) -> T:
    """Await `value` when it is awaitable; return it otherwise. Lets a host write sync or async ports."""
    if inspect.isawaitable(value):
        return await value  # type: ignore[no-any-return]
    return value  # type: ignore[return-value]


def error_message(err: BaseException) -> str:
    """`err instanceof Error ? err.message : String(err)`."""
    return str(err)


# ---------------------------------------------------------------------------
# Schemas (zod → pydantic)
# ---------------------------------------------------------------------------


@runtime_checkable
class Schema(Protocol):
    """What the core needs from a schema: parse (raise on failure) and describe."""

    def parse(self, value: Any) -> Any: ...

    def json_schema(self) -> dict[str, Any]: ...


class SchemaError(ValueError):
    """A value did not match a schema. `problems` is the flattened issue list."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


class _PydanticSchema:
    __slots__ = ("adapter", "model")

    def __init__(self, model: Any, adapter: Any):
        self.model = model
        self.adapter = adapter

    def parse(self, value: Any) -> Any:
        from pydantic import ValidationError

        try:
            return self.adapter.validate_python(value)
        except ValidationError as err:
            raise SchemaError(pydantic_problems(err)) from err

    def json_schema(self, mode: str = "validation") -> dict[str, Any]:
        return self.adapter.json_schema(mode=mode)  # type: ignore[no-any-return]

    def __repr__(self) -> str:  # pragma: no cover
        return f"Schema({self.model!r})"


class _CallableSchema:
    __slots__ = ("fn",)

    def __init__(self, fn: Callable[[Any], Any]):
        self.fn = fn

    def parse(self, value: Any) -> Any:
        return self.fn(value)

    def json_schema(self) -> dict[str, Any]:
        return {}


def is_schema(value: Any) -> bool:
    """True when `as_schema(value)` will succeed."""
    try:
        as_schema(value)
        return True
    except TypeError:
        return False


def as_schema(value: Any) -> Any:
    """
    Accepts a pydantic `BaseModel` subclass, a `TypeAdapter`, a dataclass type /
    typing annotation (wrapped in a TypeAdapter), or any object with `parse()`.
    Returns something with `parse(value)` and `json_schema()`.
    """
    if value is None:
        raise TypeError("a schema is required")
    if hasattr(value, "parse") and callable(getattr(value, "parse")) and hasattr(value, "json_schema"):
        return value
    from pydantic import BaseModel, TypeAdapter

    if isinstance(value, TypeAdapter):
        return _PydanticSchema(value, value)
    if inspect.isclass(value) and issubclass(value, BaseModel):
        return _PydanticSchema(value, TypeAdapter(value))
    if inspect.isclass(value) and dataclasses.is_dataclass(value):
        return _PydanticSchema(value, TypeAdapter(value))
    if hasattr(value, "parse") and callable(getattr(value, "parse")):
        return _CallableSchema(value.parse)
    # A typing annotation such as `dict[str, int]` or `list[str]`.
    try:
        adapter = TypeAdapter(value)
    except Exception as err:  # pragma: no cover - pydantic's own message is fine
        raise TypeError(f"not a schema: {value!r}") from err
    return _PydanticSchema(value, adapter)


def pydantic_problems(error: Any) -> list[str]:
    """
    Flatten a pydantic `ValidationError` into the `string[]` shape the contract
    checks use: `path.to.field: message`, `(root): message` at the root.
    """
    out: list[str] = []
    for issue in error.errors():
        loc = [str(p) for p in issue.get("loc", ())]
        path = ".".join(loc) if loc else "(root)"
        out.append(f"{path}: {issue.get('msg', 'invalid')}")
    return out


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def now_ms() -> float:
    """`Date.now()`: milliseconds since the epoch."""
    return datetime.now(timezone.utc).timestamp() * 1000


def iso_from_ms(ms: float) -> str:
    """`new Date(ms).toISOString()`: `2023-11-14T22:13:20.000Z`."""
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(round(ms)) % 1000:03d}Z"


def ms_from_iso(iso: str) -> float:
    """`Date.parse(iso)` for the ISO forms this package writes and reads. NaN when unparseable."""
    try:
        s = iso.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp() * 1000
    except (ValueError, AttributeError):
        return math.nan


# ---------------------------------------------------------------------------
# Numbers as JavaScript prints them
# ---------------------------------------------------------------------------


def js_number(v: float) -> str:
    """`String(v)` for a finite number: `1`, `1.5`, `1e+21`, `1e-7`."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if v != v or v in (math.inf, -math.inf):
        raise ValueError("js_number: not finite")
    if v == 0:
        return "0"
    if v == int(v) and abs(v) < 2**53:
        return str(int(v))
    r = repr(v)  # shortest round-trip digits, like JS
    if "e" not in r:
        return r[:-2] if r.endswith(".0") else r
    mant, exp = r.split("e")
    e = int(exp)
    neg = mant.startswith("-")
    mant = mant.lstrip("-")
    if 0 <= e < 21:
        # JS prints up to 1e21 in plain digits, padding the shortest mantissa with zeros.
        whole, _, frac = mant.partition(".")
        digits = whole + frac
        point = len(whole) + e
        plain = digits[:point] + "0" * (point - len(digits)) if point >= len(digits) else digits[:point] + "." + digits[point:]
        return ("-" if neg else "") + plain
    if -7 < e < 0:
        whole, _, frac = mant.partition(".")
        return ("-" if neg else "") + "0." + "0" * (-e - 1) + whole + frac
    sign = "-" if e < 0 else "+"
    return ("-" if neg else "") + f"{mant}e{sign}{abs(e)}"


def to_precision(v: float, p: int) -> str:
    """`Number.prototype.toPrecision(p)`: JS rounds ties away from zero."""
    from decimal import ROUND_HALF_UP, Decimal

    if v != v or v in (math.inf, -math.inf):
        return js_number(v) if v == v else "NaN"
    if v == 0:
        return "0" if p == 1 else "0." + "0" * (p - 1)
    d = Decimal(v)  # the exact binary value, as JS rounds it
    exp = d.adjusted()
    q = d.scaleb(-exp).quantize(Decimal(1).scaleb(-(p - 1)), rounding=ROUND_HALF_UP)
    if q.adjusted() != 0:  # rounding carried into the next power of ten (9.99 → 10.0)
        exp += 1
        q = d.scaleb(-exp).quantize(Decimal(1).scaleb(-(p - 1)), rounding=ROUND_HALF_UP)
    if exp < -7 or exp >= p:
        mant = f"{q:f}"
        sign = "-" if exp < 0 else "+"
        return f"{mant}e{sign}{abs(exp)}"
    return f"{q.scaleb(exp):f}" if exp < p - 1 else f"{q.scaleb(exp).quantize(Decimal(1)):f}"


def to_fixed(v: float, n: int) -> str:
    """`Number.prototype.toFixed(n)`: JS rounds ties away from zero and prints -0 as 0."""
    from decimal import ROUND_HALF_UP, Decimal

    if v != v or v in (math.inf, -math.inf) or abs(v) >= 1e21:
        return js_number(v) if math.isfinite(v) else ("NaN" if v != v else ("Infinity" if v > 0 else "-Infinity"))
    if v == 0:
        v = 0.0
    out = f"{Decimal(v).quantize(Decimal(1).scaleb(-n), rounding=ROUND_HALF_UP):f}"
    return "0" + out[2:] if out.startswith("-0") and float(out) == 0 else out


def js_round_precision(v: float, p: int) -> float:
    """`Number(v.toPrecision(p))`."""
    return float(to_precision(v, p))


# ---------------------------------------------------------------------------
# Plain data
# ---------------------------------------------------------------------------


def to_plain(value: Any) -> Any:
    """
    Pydantic models, dataclasses, tuples and mapping proxies → plain dicts and
    lists, recursively. What an expression engine or a JSON encoder sees.
    """
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, dict):
        return {k: to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return [to_plain(v) for v in value]
    from collections.abc import Mapping as _Mapping

    if isinstance(value, _Mapping):
        return {k: to_plain(v) for k, v in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    dump = getattr(value, "model_dump", None)
    if callable(dump) and not isinstance(value, type):
        try:
            return to_plain(dump(by_alias=True))
        except TypeError:
            return to_plain(dump())
    return value

"""
Threshold resolution. A gate body never reads the environment — the runtime
resolves every threshold once, hands the NUMBERS to the body, and records
them on the span and the run record. That recording is the whole point:
a replay re-tunes a threshold offline with zero model calls.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Optional, cast

from ._compat import maybe_await
from .types import Threshold


@dataclass(kw_only=True)
class ThresholdSource:
    """
    Where a named override actually comes from. The host owns both lookups; any
    object with these attributes works, and a plain dict with the same keys too.
    """

    #: Process env, or whatever stands in for it. `(name) -> str | None`.
    env: Optional[Callable[[str], Optional[str]]] = None
    #: A Langfuse prompt-config key. Langfuse wins over a literal, as today.
    #: `(key) -> float | None`, sync or async.
    prompt_config: Optional[Callable[[str], Any]] = None


@dataclass(kw_only=True)
class ResolvedThreshold:
    value: float
    source: Literal["literal", "env", "prompt-config", "default"]
    #: The name that was consulted, for `env` / `prompt-config`.
    ref: Optional[str] = None


_DECIMAL = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_RADIX = re.compile(r"0[xX][0-9a-fA-F]+|0[bB][01]+|0[oO][0-7]+")


def _js_number(raw: str) -> float:
    """`Number(raw)` for a string; NaN when JS would give NaN. Blank is handled by the caller."""
    s = raw.strip()
    if _DECIMAL.fullmatch(s):
        return float(s)
    if _RADIX.fullmatch(s):
        return float(int(s, 0))
    if s in ("Infinity", "+Infinity"):
        return math.inf
    if s == "-Infinity":
        return -math.inf
    return math.nan


def _number_or_none(raw: Any) -> Optional[float]:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return raw if math.isfinite(raw) else None
    if not isinstance(raw, str) or raw.strip() == "":
        return None
    n = _js_number(raw)
    return n if math.isfinite(n) else None


def _hook(source: Any, *names: str) -> Optional[Callable[[str], Any]]:
    if source is None:
        return None
    for name in names:
        fn = source.get(name) if isinstance(source, Mapping) else getattr(source, name, None)
        if fn is not None:
            return fn  # type: ignore[no-any-return]
    return None


async def resolve_threshold(threshold: Threshold, source: Any = None) -> ResolvedThreshold:
    if isinstance(threshold, (int, float)) and not isinstance(threshold, bool):
        return ResolvedThreshold(value=threshold, source="literal")

    threshold = cast(Mapping[str, Any], threshold)
    if "env" in threshold:
        env = _hook(source, "env")
        raw = env(threshold["env"]) if env else None
        parsed = _number_or_none(raw)
        if parsed is None:
            return ResolvedThreshold(value=threshold["default"], source="default", ref=threshold["env"])
        return ResolvedThreshold(value=parsed, source="env", ref=threshold["env"])

    key = threshold["promptConfig"] if "promptConfig" in threshold else threshold["prompt_config"]
    prompt_config = _hook(source, "prompt_config", "promptConfig")
    raw = await maybe_await(prompt_config(key)) if prompt_config else None
    parsed = _number_or_none(raw)
    if parsed is None:
        return ResolvedThreshold(value=threshold["default"], source="default", ref=key)
    return ResolvedThreshold(value=parsed, source="prompt-config", ref=key)


async def resolve_thresholds(
    thresholds: Mapping[str, Threshold], source: Any = None
) -> dict[str, ResolvedThreshold]:
    out: dict[str, ResolvedThreshold] = {}
    for name, threshold in thresholds.items():
        out[name] = await resolve_threshold(threshold, source)
    return out


def threshold_values(resolved: Mapping[str, ResolvedThreshold]) -> dict[str, float]:
    return {name: r.value for name, r in resolved.items()}

"""
Hashing for the run record. Two things depend on it being STABLE across
processes and orderings: `input_hash` (per-node cache key + offline replay)
and the composite `<fact>_version`.

`stable_stringify` produces the same string as the TypeScript package for
JSON-like data, so hashes, cache keys, fingerprints and experiment
assignments agree across the two implementations.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from datetime import date, datetime, timezone
from enum import Enum
from collections.abc import Mapping
from typing import Any

from ._compat import js_number


def _json_string(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def stable_stringify(value: Any) -> str:
    """
    JSON with object keys sorted at every depth, so `{a,b}` and `{b,a}` hash the
    same. Arrays keep their order (an ingredient list is not a set). Values JSON
    cannot carry are tagged rather than silently dropped, because a node input
    that hashes the same as a different input is a replay bug, not a nuisance.
    """
    seen: set[int] = set()

    def walk(v: Any) -> str:
        if v is None:
            return "null"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            return js_number(v) if math.isfinite(v) else '"__nonfinite__"'
        if isinstance(v, str):
            return _json_string(v)
        if isinstance(v, bytes):
            return _json_string("__bytes__" + v.hex())
        if isinstance(v, Enum):
            return walk(v.value)
        if isinstance(v, datetime):
            dt = v if v.tzinfo else v.replace(tzinfo=timezone.utc)
            dt = dt.astimezone(timezone.utc)
            return _json_string(dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z")
        if isinstance(v, date):
            return _json_string(v.isoformat())
        oid = id(v)
        if oid in seen:
            return '"__cycle__"'
        seen.add(oid)
        try:
            if isinstance(v, (list, tuple)):
                return "[" + ",".join(walk(x) for x in v) + "]"
            if isinstance(v, (set, frozenset)):
                return "[" + ",".join(sorted(walk(x) for x in v)) + "]"
            if isinstance(v, Mapping):
                items = sorted(((str(k), val) for k, val in v.items()), key=lambda kv: kv[0])
                return "{" + ",".join(f"{_json_string(k)}:{walk(val)}" for k, val in items) + "}"
            if dataclasses.is_dataclass(v) and not isinstance(v, type):
                return walk({f.name: getattr(v, f.name) for f in dataclasses.fields(v)})
            dump = getattr(v, "model_dump", None)
            if callable(dump) and not isinstance(v, type):
                # Wire names (aliases), so a model with `recipeId` hashes like the TS object it mirrors.
                try:
                    return walk(dump(by_alias=True))
                except TypeError:
                    return walk(dump())
            if callable(v):
                return _json_string(f"__fn__{getattr(v, '__name__', '') or 'anonymous'}")
            if hasattr(v, "__dict__"):
                return walk({k: val for k, val in vars(v).items() if not k.startswith("_")})
            return _json_string(f"__{type(v).__name__}__{v!r}")
        finally:
            seen.discard(oid)

    return walk(value)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_hash(value: Any) -> str:
    """16 hex chars — 64 bits, plenty for a cache key, short enough for a span attribute."""
    return sha256_hex(stable_stringify(value))[:16]


def hash_to_unit(text: str) -> float:
    """
    Deterministic 0..1 for experiment assignment. Recomputable from the run
    record, which is what makes an assignment auditable rather than a guess.
    """
    hex13 = sha256_hex(text)[:13]  # 52 bits, exact in a double
    return int(hex13, 16) / 2**52

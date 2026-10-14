"""
Named inputs — a node says WHAT it reads instead of reaching into
`args.from_[<node_id>]`.

Every node body used to open by digging its inputs out of `from_` by node id.
That makes node ids load-bearing inside bodies: rename or move a node and some
other node's code breaks, with no load-time error. A binding moves that dig
into the manifest, where `define_pipeline` can check it:

    inputs: {"row": "evidence.row", "taxonomy": "taxonomy", "iteration": "$input.iteration"}

and the body reads `args.in_["row"]`. A ref is

    <node>[.<field>|[<n>]]*     an output of a DIRECT inbound node, or a part of it
    $input[.<field>|[<n>]]*     the pipeline's validated input
    <ref> | <ref> | ...         the first that resolves (`maps | retry_maps`)
    ... | null / 0 / true / "x"   a JSON literal as the LAST alternative, the
                                default when nothing before it resolves

"Resolves" means the node ran and the path is absent from the data — `None` is
a value (a gate that routes on `row = None` needs to see it). A skipped node is
absent from `from_`, so it falls through to the next alternative, and a ref
with no alternative left binds `None`.

Direct inbound only, the same rule as `from_` and `probes.from_`: a binding to
a non-adjacent ancestor would read an output the edge guards never vouched for
on this path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Collection, Mapping, Optional, Union

from ._compat import MISSING

INPUT_REF = "$input"

_IDENT = "[A-Za-z_][A-Za-z0-9_]*"
_REF_RE = re.compile(rf"(\$input|{_IDENT})((?:\.{_IDENT}|\[[0-9]+\])*)")
_SEGMENT_RE = re.compile(rf"\.({_IDENT})|\[([0-9]+)\]")
_NAME_RE = re.compile(_IDENT)
_LITERAL_RE = re.compile(r'null|true|false|-?[0-9]+(?:\.[0-9]+)?|"[^"\\]*"')


@dataclass(kw_only=True)
class RefSource:
    #: A node id, or `$input`.
    source: str
    path: list[Union[str, int]]


@dataclass(kw_only=True)
class RefLiteral:
    literal: Union[None, bool, int, float, str]


ParsedRef = Union[RefSource, RefLiteral]


def parse_input_ref(ref: str) -> Union[list[ParsedRef], str]:
    """A ref's alternatives, or the reason it does not parse."""
    alternatives = [s.strip() for s in ref.split("|")]
    parsed: list[ParsedRef] = []
    for i, alt in enumerate(alternatives):
        if _LITERAL_RE.fullmatch(alt):
            if i == 0:
                return f'"{alt}" is a literal — a binding reads something first, a literal is only its default'
            if i != len(alternatives) - 1:
                return f'literal "{alt}" must be the last alternative — nothing after a default is ever read'
            parsed.append(RefLiteral(literal=json.loads(alt)))
            continue
        m = _REF_RE.fullmatch(alt)
        if not m:
            return f'"{alt}" is not <node>[.field|[n]]* or $input[.field|[n]]*'
        path: list[Union[str, int]] = []
        for s in _SEGMENT_RE.finditer(m.group(2)):
            path.append(s.group(1) if s.group(1) is not None else int(s.group(2)))
        parsed.append(RefSource(source=m.group(1), path=path))
    return parsed


def binding_sources(inputs: Mapping[str, str]) -> list[str]:
    """The node ids a set of bindings reads from — for edge checks and diagrams."""
    out: dict[str, None] = {}
    for ref in inputs.values():
        parsed = parse_input_ref(ref)
        if isinstance(parsed, str):
            continue
        for p in parsed:
            if isinstance(p, RefSource) and p.source != INPUT_REF:
                out[p.source] = None
    return list(out)


def binding_problems(node_id: str, inputs: Mapping[str, str], direct_inbound: Collection[str]) -> list[str]:
    """Load-time checks: names are identifiers, refs parse, every source is a direct inbound node."""
    problems: list[str] = []
    for name, ref in inputs.items():
        if not _NAME_RE.fullmatch(name):
            problems.append(f'node "{node_id}": input name "{name}" is not an identifier')
        parsed = parse_input_ref(ref)
        if isinstance(parsed, str):
            problems.append(f'node "{node_id}": input "{name}": {parsed}')
            continue
        for p in parsed:
            if isinstance(p, RefSource) and p.source != INPUT_REF and p.source not in direct_inbound:
                problems.append(
                    f'node "{node_id}": input "{name}" reads "{p.source}", which has no edge '
                    f"{p.source} -> {node_id} — a node binds only what flows into it"
                )
    return problems


_ATOMS = (str, bytes, int, float, bool)


def _step(v: Any, seg: Union[str, int]) -> Any:
    """One path segment; MISSING when it walks off the data."""
    if v is None or isinstance(v, _ATOMS):
        return MISSING
    if isinstance(v, Mapping):
        if seg in v:
            return v[seg]
        if isinstance(seg, int) and str(seg) in v:
            return v[str(seg)]
        return MISSING
    if isinstance(v, (list, tuple)):
        if isinstance(seg, int) and 0 <= seg < len(v):
            return v[seg]
        return MISSING
    # Objects (pydantic models, dataclasses): attribute access for field names.
    if isinstance(seg, str) and not seg.startswith("_"):
        return getattr(v, seg, MISSING)
    return MISSING


def _walk(value: Any, path: list[Union[str, int]]) -> Any:
    v = value
    for seg in path:
        v = _step(v, seg)
        if v is MISSING:
            return MISSING
    return v


def resolve_inputs(
    inputs: Optional[Mapping[str, str]],
    from_: Mapping[str, Any],
    input: Any,
) -> dict[str, Any]:
    """
    Bindings -> values, for one node on one run. Assumes `binding_problems`
    passed at load; an unparseable ref here binds `None` rather than raise.
    """
    out: dict[str, Any] = {}
    if not inputs:
        return out
    for name, ref in inputs.items():
        parsed = parse_input_ref(ref)
        value: Any = MISSING
        if not isinstance(parsed, str):
            for p in parsed:
                if isinstance(p, RefLiteral):
                    value = p.literal
                    break
                if p.source != INPUT_REF and p.source not in from_:
                    continue
                value = _walk(input if p.source == INPUT_REF else from_[p.source], p.path)
                if value is not MISSING:
                    break
        out[name] = None if value is MISSING else value
    return out

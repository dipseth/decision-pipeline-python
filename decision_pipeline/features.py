"""
Pipeline-agnostic feature extraction for the run record.

The shared layer between pipelines is the record schema and THESE — not the
weights. Anything derivable from a run without knowing what the pipeline
does belongs here; anything that needs domain knowledge goes in the
manifest's own `features()`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_finite(v: Any) -> bool:
    return _is_num(v) and math.isfinite(v)


def normalized_entropy(p: Sequence[float]) -> Optional[float]:
    """Shannon entropy in nats, normalised to 0..1 against a uniform distribution."""
    values = [v for v in p if _is_finite(v) and v > 0]
    if len(values) <= 1:
        return 0.0
    total = sum(values)
    if total <= 0:
        return None
    h = 0.0
    for v in values:
        q = v / total
        h -= q * math.log(q)
    return h / math.log(len(values))


def top_margin(p: Sequence[float]) -> Optional[float]:
    """Gap between the top two probabilities — how nearly the gate could have gone the other way."""
    ordered = sorted((v for v in p if _is_finite(v)), reverse=True)
    if not ordered:
        return None
    first = ordered[0]
    second = ordered[1] if len(ordered) > 1 else 0
    return first - second


@dataclass(kw_only=True)
class DistributionFeatureInput:
    node_id: str
    distributions: dict[str, list[float]]


def distribution_features(inputs: Sequence[Any]) -> dict[str, float]:
    """
    `<node_id>.<question>.margin` and `.entropy` for every distribution a decide
    node produced. Flat keys, because the record's `features` is flat and the
    training set is a join.
    """
    out: dict[str, float] = {}
    for item in inputs:
        if isinstance(item, Mapping):
            node_id = item.get("node_id", item.get("nodeId"))
            distributions = item["distributions"]
        else:
            node_id, distributions = item.node_id, item.distributions
        for question, p in distributions.items():
            margin = top_margin(p)
            if margin is not None:
                out[f"{node_id}.{question}.margin"] = margin
            entropy = normalized_entropy(p)
            if entropy is not None:
                out[f"{node_id}.{question}.entropy"] = entropy
            finite = [v for v in p if _is_finite(v)]
            if finite:
                out[f"{node_id}.{question}.top"] = max(finite)
    return out

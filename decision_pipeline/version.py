"""
The composite version. Never hand-set: the runtime hashes what actually
resolved at run start, so `<fact>_version` names a real configuration rather
than whatever someone remembered to bump.

The ARM is folded in — otherwise two arms of an experiment collide under one
`<fact>_version` and the records are unusable.

Invalidation is per NODE (`node_version + input_hash`), not per pipeline: a
threshold bump on the gate must not re-run the decide call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .hash import short_hash


@dataclass(kw_only=True)
class VersionInputs:
    pipeline: str
    node_versions: dict[str, str] = field(default_factory=dict)
    #: Resolved Langfuse prompt versions, generate + decide. None when unresolved.
    prompt_versions: dict[str, Optional[str]] = field(default_factory=dict)
    #: Resolved threshold VALUES, by node then name.
    thresholds: dict[str, dict[str, float]] = field(default_factory=dict)
    knowledge_version: Optional[str] = None
    #: experiment id -> arm.
    arms: dict[str, str] = field(default_factory=dict)


def composite_version(inputs: VersionInputs) -> str:
    # The hashed keys are the TS camelCase names, so versions agree across implementations.
    return short_hash(
        {
            "pipeline": inputs.pipeline,
            "nodeVersions": inputs.node_versions,
            "promptVersions": inputs.prompt_versions,
            "thresholds": inputs.thresholds,
            "knowledgeVersion": inputs.knowledge_version,
            "arms": inputs.arms,
        }
    )


def node_cache_key(pipeline: str, node_id: str, node_version: str, input_hash: str) -> str:
    """Per-node cache key. The node version is IN the key, so a bump invalidates by itself."""
    return f"{pipeline}:{node_id}:{node_version}:{input_hash}"

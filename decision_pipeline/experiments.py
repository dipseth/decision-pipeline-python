"""
Experiments — a variant is a PATCH over the declarative half of a manifest.

That falls straight out of "manifest, not interpreter": if a variant needs a
new `run` function it is a new NODE VERSION, not a variant. Structural A/B
and config A/B collapse into one mechanism, and the patch diff is the arm
diff, for free.

Arms resolve in the runtime alongside `ScopeContext`, before node 0 — the
same position, for the same reason. An arm decidable mid-run from mid-run
state could never be replayed offline, and replay is the whole point.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Sequence

from .hash import hash_to_unit
from .types import EdgeGuard, ToolSelectable, ToolSpec, scope_index, tool_spec_from_dict

ExperimentMode = Literal["replay", "replay-redecide", "shadow", "split"]
"""
replay           Free: re-derive from the recorded distributions. No model calls, no exposure.
replay-redecide  Model calls only: re-run the decide node against the recorded input.
shadow           ~2x cost: both arms run, ONE ships. The challenger must not write the fact.
split            1x cost, real exposure, waits for labels. The last resort, never the default.
"""


@dataclass(kw_only=True)
class GuardPatch:
    """Edge guard patch. `when=None` removes a guard; a missing edge is a validation error."""

    from_: str
    to: str
    when: Optional[EdgeGuard] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_,
            "to": self.to,
            "when": None if self.when is None else {"gate": self.when.gate, "branch": self.when.branch},
        }


def _guard_patch_from_dict(d: Any) -> GuardPatch:
    if isinstance(d, GuardPatch):
        return d
    when = d.get("when")
    if when is not None and not isinstance(when, EdgeGuard):
        when = EdgeGuard(gate=when["gate"], branch=when["branch"])
    return GuardPatch(from_=d["from"] if "from" in d else d["from_"], to=d["to"], when=when)


def _tool_spec_to_dict(spec: ToolSpec) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if spec.static is not None:
        out["static"] = spec.static
    if spec.selectable is not None:
        s = spec.selectable
        sel: dict[str, Any] = {"candidates": list(s.candidates), "from": s.from_, "gate": s.gate}
        if s.threshold is not None:
            sel["threshold"] = s.threshold
        out["selectable"] = sel
    if spec.allow is not None:
        out["allow"] = list(spec.allow)
    return out


@dataclass(kw_only=True)
class ManifestPatch:
    """Only the declarative half. There is deliberately no way to patch a node body."""

    #: node id -> threshold name -> literal value.
    thresholds: Optional[dict[str, dict[str, float]]] = None
    #: node id -> Langfuse prompt ref (generate nodes).
    prompts: Optional[dict[str, str]] = None
    #: node id -> Jev question prompt name (decide nodes).
    questions: Optional[dict[str, str]] = None
    #: node id -> model route (generate nodes).
    routes: Optional[dict[str, str]] = None
    #: node id -> ToolSpec (generate nodes).
    tools: Optional[dict[str, ToolSpec]] = None
    #: Edge guard patch. `when=None` removes a guard; a missing edge is a validation error.
    guards: Optional[list[GuardPatch]] = None

    def to_dict(self) -> dict[str, Any]:
        """The TS JSON shape (a wire format). Unset fields are omitted."""
        out: dict[str, Any] = {}
        if self.thresholds is not None:
            out["thresholds"] = {n: dict(v) for n, v in self.thresholds.items()}
        if self.prompts is not None:
            out["prompts"] = dict(self.prompts)
        if self.questions is not None:
            out["questions"] = dict(self.questions)
        if self.routes is not None:
            out["routes"] = dict(self.routes)
        if self.tools is not None:
            out["tools"] = {n: _tool_spec_to_dict(t) for n, t in self.tools.items()}
        if self.guards is not None:
            out["guards"] = [g.to_dict() for g in self.guards]
        return out


def manifest_patch_from_dict(d: Any) -> ManifestPatch:
    """Accepts the TS JSON shape (`from`/`when` guards, camelCase tool keys). A `ManifestPatch` passes through."""
    if isinstance(d, ManifestPatch):
        return d
    d = d or {}
    tools = d.get("tools")
    guards = d.get("guards")
    return ManifestPatch(
        thresholds=None if d.get("thresholds") is None else {n: dict(v) for n, v in d["thresholds"].items()},
        prompts=None if d.get("prompts") is None else dict(d["prompts"]),
        questions=None if d.get("questions") is None else dict(d["questions"]),
        routes=None if d.get("routes") is None else dict(d["routes"]),
        tools=None if tools is None else {n: tool_spec_from_dict(t) for n, t in tools.items()},  # type: ignore[misc]
        guards=None if guards is None else [_guard_patch_from_dict(g) for g in guards],
    )


def is_empty_patch(p: ManifestPatch) -> bool:
    """True when the patch changes nothing (every field unset or empty)."""
    return not (p.thresholds or p.prompts or p.questions or p.routes or p.tools or p.guards)


@dataclass(kw_only=True)
class ExperimentSpec:
    id: str
    #: Assign at the WIDEST scope any feedback edge writes to (usually `user`).
    #: Then feedback contamination between arms is impossible by construction: a
    #: row written under arm B can only ever be read by arm-B runs.
    unit: str
    mode: ExperimentMode
    #: The arm that ships when nothing is assigned. Must be a key of `arms`.
    champion: str
    #: Arm name -> patch. The champion's patch is normally empty.
    arms: dict[str, ManifestPatch]
    #: `split` only. Defaults to equal weight across arms.
    weights: Optional[dict[str, float]] = None
    #: Declared is not running. Default false.
    enabled: Optional[bool] = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "unit": self.unit,
            "mode": self.mode,
            "champion": self.champion,
            "arms": {n: p.to_dict() for n, p in self.arms.items()},
        }
        if self.weights is not None:
            out["weights"] = dict(self.weights)
        if self.enabled is not None:
            out["enabled"] = self.enabled
        return out


def experiment_spec_from_dict(d: Any) -> ExperimentSpec:
    if isinstance(d, ExperimentSpec):
        return d
    return ExperimentSpec(
        id=d["id"],
        unit=d["unit"],
        mode=d["mode"],
        champion=d["champion"],
        arms={n: manifest_patch_from_dict(p) for n, p in (d.get("arms") or {}).items()},
        weights=d.get("weights"),
        enabled=d.get("enabled"),
    )


@dataclass(kw_only=True)
class ArmAssignment:
    experiment: str
    arm: str
    unit: Optional[str]
    #: Hash of the assigning id, so an assignment is auditable and recomputable.
    hash: Optional[str]
    #: True when the arm ships; false for a shadow challenger, which must not write the fact.
    ships: bool
    reason: Literal["disabled", "no-unit-id", "champion", "assigned", "shadow-challenger"]


def _arm_names(spec: ExperimentSpec) -> list[str]:
    return sorted(spec.arms.keys())


def assign_arm(spec: ExperimentSpec, unit_id: Optional[str]) -> ArmAssignment:
    """
    Deterministic, recomputable, and independent per experiment — so two
    concurrent experiments stay orthogonal for measurement even though their
    user-visible COMBINATION may still be bad. Recording `arms` as a map is what
    keeps that detectable.
    """
    champion = ArmAssignment(
        experiment=spec.id, arm=spec.champion, unit=spec.unit, hash=None, ships=True, reason="champion"
    )

    if spec.enabled is not True:
        return dataclasses.replace(champion, reason="disabled")
    if spec.mode in ("replay", "replay-redecide"):
        # Offline modes never touch a live run.
        return dataclasses.replace(champion, reason="champion")
    if not unit_id:
        return dataclasses.replace(champion, reason="no-unit-id")

    hash_ = f"{spec.id}:{spec.unit}:{unit_id}"

    if spec.mode == "shadow":
        # Both arms run; the champion is what ships. The challenger is resolved by
        # the caller running the pipeline a second time with `force_arm`.
        return dataclasses.replace(champion, hash=hash_, reason="champion")

    names = _arm_names(spec)
    weights = [max(0, (spec.weights or {}).get(n, 1)) for n in names]
    total = sum(weights)
    if total <= 0:
        return dataclasses.replace(champion, hash=hash_, reason="champion")

    cursor = hash_to_unit(hash_) * total
    for i, name in enumerate(names):
        cursor -= weights[i]
        if cursor < 0:
            return ArmAssignment(
                experiment=spec.id, arm=name, unit=spec.unit, hash=hash_, ships=True, reason="assigned"
            )
    return dataclasses.replace(champion, hash=hash_, reason="champion")


def shadow_challenger(spec: ExperimentSpec, arm: str, unit_id: Optional[str]) -> ArmAssignment:
    """The shadow challenger's assignment — it runs, it is recorded, it must NOT write the fact."""
    return ArmAssignment(
        experiment=spec.id,
        arm=arm,
        unit=spec.unit,
        hash=f"{spec.id}:{spec.unit}:{unit_id}" if unit_id else None,
        ships=False,
        reason="shadow-challenger",
    )


def _field(obj: Any, name: str) -> Any:
    return obj[name] if isinstance(obj, Mapping) else getattr(obj, name)


def widest_direct_feedback_scope(feedback: Sequence[Any]) -> Optional[str]:
    """The widest scope any `override`/`feature` feedback edge writes to. Items need `.form` and `.scope` (or keys)."""
    widest: Optional[str] = None
    for edge in feedback:
        if _field(edge, "form") not in ("override", "feature"):
            continue
        scope = _field(edge, "scope")
        if widest is None or scope_index(scope) > scope_index(widest):
            widest = scope
    return widest


def merge_patches(patches: Sequence[ManifestPatch]) -> ManifestPatch:
    """Merge patches left to right. Later wins per key; `guards` concatenate."""
    out = ManifestPatch()
    for p in patches:
        if p.thresholds is not None:
            merged = dict(out.thresholds or {})
            for node_id, values in p.thresholds.items():
                merged[node_id] = {**merged.get(node_id, {}), **values}
            out.thresholds = merged
        if p.prompts is not None:
            out.prompts = {**(out.prompts or {}), **p.prompts}
        if p.questions is not None:
            out.questions = {**(out.questions or {}), **p.questions}
        if p.routes is not None:
            out.routes = {**(out.routes or {}), **p.routes}
        if p.tools is not None:
            out.tools = {**(out.tools or {}), **p.tools}
        if p.guards is not None:
            out.guards = [*(out.guards or []), *p.guards]
    return out

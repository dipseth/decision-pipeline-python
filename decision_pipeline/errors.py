"""Errors the manifest layer and the runtime raise. All carry the pipeline id."""

from __future__ import annotations

from typing import Any


class PipelineContractError(Exception):
    def __init__(self, pipeline: str, problems: list[str]):
        super().__init__(
            f'Pipeline "{pipeline}" violates the decision-pipeline contract:\n'
            + "\n".join(f"  - {p}" for p in problems)
        )
        self.pipeline = pipeline
        self.problems = list(problems)


class MissingPortError(Exception):
    """A node needed a port the host did not supply. Named, so the fix is obvious."""

    def __init__(self, port: str, node_id: str, kind: str):
        super().__init__(f'No "{port}" port supplied, but node "{node_id}" is a {kind} node that needs one.')
        self.port = port
        self.node_id = node_id
        self.kind = kind


class NodeFailedError(Exception):
    """A node body failed and its `on_failure` was `fail` (the default)."""

    def __init__(self, node_id: str, cause: Any):
        message = str(cause)
        super().__init__(f'Node "{node_id}" failed: {message}')
        self.node_id = node_id
        self.cause = cause


class NodeTimeoutError(Exception):
    """A node ran past its declared `timeout_ms`. Its `on_failure` decides the rest."""

    def __init__(self, node_id: str, timeout_ms: float):
        super().__init__(f'node "{node_id}" exceeded its {_ms(timeout_ms)} ms deadline')
        self.node_id = node_id
        self.timeout_ms = timeout_ms


class NodeSkipped(Exception):
    """
    A node body that deliberately did not run — a capability switched off, not
    a failure. Raise it from a body or port: the runtime records the node as
    skipped with this message, whatever its `on_failure`, and the node span
    closes normally, so a switched-off node never reads as an error in the trace.
    """

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _ms(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else str(v)


class UnknownBranchError(Exception):
    """A gate body returned a branch it never declared."""

    def __init__(self, node_id: str, branch: str, branches: list[str]):
        super().__init__(
            f'Gate "{node_id}" returned branch "{branch}", which is not one of [{", ".join(branches)}].'
        )
        self.node_id = node_id
        self.branch = branch
        self.branches = list(branches)

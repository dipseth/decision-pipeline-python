"""
Jev wire helpers — pure, no HTTP. The decide port stays the host's; these
are the parts every Jev-backed port was writing by hand.

TypeSafe's "danger zone" features (Oct 2026) are opt-in per request through
headers, and are dreamer-ready rather than production-ready: format and
accuracy may change between releases.
 - Images work on every Jev, `jev-1.13.0` included, behind
   `TypeSafe-Danger-Zone-Images: true`. An image is an asset whose id can
   appear anywhere the JSON does: state, instructions, even criteria.
 - The `box` question type needs `jev-1.14.0` (or `jev-preview`), and the
   same images header.
 - Longer context than production allows, behind
   `TypeSafe-Danger-Zone-Long-Context: true`. TypeSafe still recommends
   keeping state short; so does this package.
`jev-latest` moves to 1.14.0 a week after release: pin the exact version
once a threshold is tuned.

Requests and answers are plain dicts in the API's own (snake_case) keys, or
the official `typesafe-sdk` objects (`Noul`, `Box`, `SystemOneResponse`, ...):
every helper reads both, and the package never imports the SDK. With the SDK,
the port is `client.system_one(...)` plus `jev_extra_headers` (the SDK does not
send the danger-zone headers itself) and `jev_response_to_decide_result`.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Optional

from .box import BOX_CONFIDENCE_OPTION
from .ports import DecideResult

__all__ = [
    "JEV_MODEL",
    "JEV_PREVIEW_MODEL",
    "JEV_ENDPOINT",
    "JEV_IMAGES_HEADER",
    "JEV_LONG_CONTEXT_HEADER",
    "jev_supports_box",
    "jev_needs_images",
    "check_jev_request",
    "jev_headers",
    "jev_extra_headers",
    "jev_answers_to_decide_result",
    "jev_response_to_decide_result",
]

#: The default this package documents and tests against.
JEV_MODEL = "jev-1.14.0"
#: Moves with TypeSafe's releases; never tune a threshold on it.
JEV_PREVIEW_MODEL = "jev-preview"
JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
#: Required on any request with an image asset or a Box question.
JEV_IMAGES_HEADER = "TypeSafe-Danger-Zone-Images"
#: Opt-in to context beyond the production limit. Less reliable; prefer a smaller state.
JEV_LONG_CONTEXT_HEADER = "TypeSafe-Danger-Zone-Long-Context"

_VERSION = re.compile(r"^jev-(\d+)\.(\d+)\.(\d+)$")


def _plain(v: Any) -> Any:
    """An SDK (pydantic) object as the API's JSON dict; anything else as it is."""
    dump = getattr(v, "model_dump", None)
    return dump(mode="json", exclude_none=True) if callable(dump) else v


def _qtype(q: Any) -> Any:
    return q.get("type") if isinstance(q, Mapping) else getattr(q, "type", None)


def _is_sdk_asset(v: Any) -> bool:
    # `typesafe_sdk.Image` (or any future asset class): matched by module so the SDK stays optional.
    return type(v).__module__.startswith("typesafe_sdk") and not callable(getattr(v, "model_dump", None))


def _has_sdk_asset(v: Any, depth: int = 0) -> bool:
    if depth > 32:
        return False
    if _is_sdk_asset(v):
        return True
    if isinstance(v, Mapping):
        return any(_has_sdk_asset(x, depth + 1) for x in v.values())
    if isinstance(v, (list, tuple)):
        return any(_has_sdk_asset(x, depth + 1) for x in v)
    if callable(getattr(v, "model_dump", None)):  # an SDK question: its instructions or criteria may hold an image
        return any(_has_sdk_asset(getattr(v, f, None), depth + 1) for f in ("instructions", "criteria"))
    return False


def jev_supports_box(model: str) -> bool:
    """
    Whether `model` answers Box questions: `jev-preview`, or a pinned version
    from 1.14.0. An alias this package cannot read (`jev-latest`) is given the
    benefit of the doubt. Images alone need no particular version.
    """
    if model == JEV_PREVIEW_MODEL:
        return True
    m = _VERSION.match(model)
    if not m:
        return True
    major, minor = int(m.group(1)), int(m.group(2))
    return major > 1 or (major == 1 and minor >= 14)


def jev_needs_images(body: Mapping[str, Any]) -> bool:
    """
    True when the request needs the images header: an image asset or a Box
    question. Questions may be SDK objects, and an SDK `Image` anywhere in
    `state` or a question counts as an image asset.
    """
    questions = body.get("questions") or {}
    assets = body.get("assets") or {}
    return (
        any(_qtype(q) == "box" for q in questions.values())
        or any((isinstance(a, Mapping) and a.get("type") == "image") or _is_sdk_asset(a) for a in assets.values())
        or _has_sdk_asset(body.get("state"))
        or _has_sdk_asset(list(questions.values()))
    )


def check_jev_request(body: Mapping[str, Any]) -> list[str]:
    """Every problem with a request before it is sent, all at once. Empty means send it."""
    problems: list[str] = []
    has_box = False
    for key, q in (body.get("questions") or {}).items():
        if _qtype(q) != "box":
            continue
        has_box = True
        instructions = q.get("instructions") if isinstance(q, Mapping) else getattr(q, "instructions", None)
        if isinstance(q, Mapping) and "criteria" in q:
            problems.append(f'question "{key}": a box takes no criteria')
        # Instructions may be an image, a list or an object (SDK content): only a missing or blank one is flagged.
        if instructions is None or (isinstance(instructions, str) and not instructions.strip()):
            problems.append(f'question "{key}": a box needs instructions naming the target and the image')
    model = str(body.get("model") or "")
    if has_box and not jev_supports_box(model):
        problems.append(f'model "{model}" does not answer box questions; send "{JEV_MODEL}" or "{JEV_PREVIEW_MODEL}"')
    return problems


def jev_headers(api_key: str, body: Mapping[str, Any], *, long_context: bool = False) -> dict[str, str]:
    """
    Auth, the images header when the request needs it, and the long-context
    header only when asked for. Content type is the caller's (JSON or multipart).
    """
    headers = {"authorization": f"Bearer {api_key}"}
    if jev_needs_images(body):
        headers[JEV_IMAGES_HEADER] = "true"
    if long_context:
        headers[JEV_LONG_CONTEXT_HEADER] = "true"
    return headers


def jev_extra_headers(
    questions: Mapping[str, Any],
    state: Any = None,
    *,
    assets: Optional[Mapping[str, Any]] = None,
    images: bool = False,
    long_context: bool = False,
) -> dict[str, str]:
    """
    The danger-zone headers alone, for the SDK's `extra_headers=` (the client
    sends auth). Images header for a Box, an SDK `Image` in `state`, a
    question or `assets` (the same mapping passed to `system_one`), or `images=True`.

        res = await client.system_one(state=state, questions=qs, model=JEV_MODEL,
                                      extra_headers=jev_extra_headers(qs, state))
    """
    headers: dict[str, str] = {}
    if images or jev_needs_images({"questions": questions, "state": state, "assets": assets}):
        headers[JEV_IMAGES_HEADER] = "true"
    if long_context:
        headers[JEV_LONG_CONTEXT_HEADER] = "true"
    return headers


def jev_answers_to_decide_result(
    answers: Mapping[str, Mapping[str, Any]],
    *,
    usage: Optional[Mapping[str, Any]] = None,
    cost_usd: Optional[Callable[[Mapping[str, Any]], float]] = None,
) -> DecideResult:
    """
    Jev answers -> the decide port's result. Noul -> `[p]` / `["true"]`;
    Choice and Score -> one probability per option; Box -> `boxes[key]` (the
    runtime checks it and mirrors its confidence into `distributions`). An
    answer with none of these is left out. Answers and usage may be SDK
    objects; a usage count the API left out (`None`) is dropped. A Box with
    `confidence: None` (the SDK allows it) is passed through, and the runtime
    fails the node on purpose: a gate needs a number, so `on_failure` decides.
    """
    answers = {k: _plain(a) for k, a in answers.items()}
    if usage is not None:
        usage = {k: v for k, v in dict(_plain(usage)).items() if v is not None}
    distributions: dict[str, list[float]] = {}
    options: dict[str, list[str]] = {}
    boxes: dict[str, Any] = {}
    for key, a in answers.items():
        if a.get("type") == "box" or a.get("box") is not None:
            boxes[key] = {"box": a.get("box"), "confidence": a.get("confidence")}
            distributions[key] = [a.get("confidence")]  # type: ignore[list-item]
            options[key] = [BOX_CONFIDENCE_OPTION]
        elif isinstance(a.get("noul"), (int, float)):
            distributions[key] = [a["noul"]]
            options[key] = ["true"]
        elif a.get("probabilities"):
            options[key] = list(a["probabilities"].keys())
            distributions[key] = list(a["probabilities"].values())
    return DecideResult(
        distributions=distributions,
        distribution_options=options,
        boxes=boxes or None,
        cost_usd=cost_usd(usage) if usage is not None and cost_usd is not None else None,
    )


def jev_response_to_decide_result(
    response: Any,
    *,
    cost_usd: Optional[Callable[[Mapping[str, Any]], float]] = None,
) -> DecideResult:
    """
    A `typesafe_sdk.SystemOneResponse` (or the raw response dict) -> the decide
    port's result. Answer kinds the SDK skipped are not seen; read
    `response.raw_http_response.json()["answers"]` if you need them.
    """
    if isinstance(response, Mapping):
        return jev_answers_to_decide_result(response.get("answers") or {}, usage=response.get("usage"), cost_usd=cost_usd)
    return jev_answers_to_decide_result(response.answers, usage=getattr(response, "usage", None), cost_usd=cost_usd)

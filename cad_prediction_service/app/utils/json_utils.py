import json
from typing import Any, Optional

from app.schemas import PredictResponse, QueryDesignResponse, QueryResponse
from app.utils.logging_utils import log


def extract_json(text: str) -> Optional[dict[str, Any]]:
    """Extract the first JSON object from raw LLM output.

    Strategy:
      1. Direct parse.
      2. Balanced-brace scan — finds the first '{' and walks character-by-character
         tracking depth and string state. This correctly handles markdown fences,
         trailing text, and nested objects with '}' inside string values.
    Returns None if no valid JSON object is found.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Balanced-brace scan — robust against greedy-regex false positives.
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as exc:
                    log.debug("extract_json: balanced-brace candidate failed: %s", exc)
                    return None

    log.debug("extract_json: no balanced JSON object found in %d-char text", len(text))
    return None


def validate_response(data: dict[str, Any]) -> Optional[PredictResponse]:
    """Validate a raw dict against PredictResponse. Returns None on failure."""
    try:
        return PredictResponse(**data)
    except Exception as exc:
        log.debug("validate_response failed: %s", exc)
        return None


def validate_query_response(data: dict[str, Any]) -> Optional[QueryResponse]:
    """Validate a raw dict against QueryResponse. Returns None on failure."""
    try:
        return QueryResponse(**data)
    except Exception as exc:
        log.debug("validate_query_response failed: %s", exc)
        return None


def validate_query_design_response(data: dict[str, Any]) -> Optional[QueryDesignResponse]:
    """Validate a raw dict against QueryDesignResponse. Returns None on failure."""
    try:
        return QueryDesignResponse(**data)
    except Exception as exc:
        log.debug("validate_query_design_response failed: %s", exc)
        return None


def fallback_query_response(detail: Optional[str] = None) -> QueryResponse:
    """Safe fallback returned when the query LLM output cannot be parsed."""
    reasoning = "Could not parse model output."
    if detail:
        reasoning = f"{reasoning} Detail: {detail}"
    return QueryResponse(todo=[], commands=[], reasoning=reasoning)


def fallback_query_design_response(detail: Optional[str] = None) -> QueryDesignResponse:
    """Safe fallback returned when the query-design LLM output cannot be parsed."""
    reasoning = "Could not parse model output."
    if detail:
        reasoning = f"{reasoning} Detail: {detail}"
    return QueryDesignResponse(
        todo=[],
        commands=[],
        reasoning=reasoning,
        desiredCommands=[],
        designFeedback=[],
    )


def fallback_response(model: str, provider: str, detail: Optional[str] = None) -> PredictResponse:
    """Safe fallback returned when the LLM output cannot be parsed or validated."""
    desc = "Could not parse model output."
    if detail:
        desc = f"{desc} Detail: {detail}"
    return PredictResponse(
        task_summary={"label": "unknown", "description": desc},
        context_update={"current_goal": "unknown", "active_object_types": [], "pattern_detected": None},
        predictions=[],
        meta={"model": model, "provider": provider, "prompt_version": "v1"},
    )

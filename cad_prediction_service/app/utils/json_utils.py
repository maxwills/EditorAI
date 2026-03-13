import json
import re
from typing import Any, Optional

from app.schemas import PredictResponse


def extract_json(text: str) -> Optional[dict[str, Any]]:
    """Extract the first JSON object from raw LLM output.

    Tries direct parse first, then falls back to regex scanning.
    Returns None if no valid JSON object is found.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Some models wrap the JSON in markdown fences or add extra text.
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def validate_response(data: dict[str, Any]) -> Optional[PredictResponse]:
    """Validate a raw dict against PredictResponse. Returns None on failure."""
    try:
        return PredictResponse(**data)
    except Exception:
        return None


def fallback_response(model: str, provider: str) -> PredictResponse:
    """Safe fallback returned when the LLM output cannot be parsed or validated."""
    return PredictResponse(
        task_summary={"label": "unknown", "description": "Could not parse model output."},
        context_update={"current_goal": "unknown", "active_object_types": [], "pattern_detected": None},
        predictions=[],
        meta={"model": model, "provider": provider, "prompt_version": "v1"},
    )

import json
from app.schemas import PredictRequest

PROMPT_VERSION = "v1"

#: Hardcoded action vocabulary. Extend this list as new editor actions are added.
ALLOWED_ACTIONS: list[str] = [
    "create_tube",
    "duplicate_selection",
    "move_object",
    "rotate_object",
    "scale_object",
    "delete_selection",
    "group_selection",
    "ungroup_selection",
    "extrude_face",
    "boolean_union",
    "boolean_subtract",
    "apply_material",
    "change_tool",
]

# Schema template embedded in prompt so the LLM knows the exact expected shape.
_RESPONSE_SCHEMA_TEMPLATE = {
    "task_summary": {
        "label": "<snake_case label>",
        "description": "<short description of what the user is doing>",
    },
    "context_update": {
        "current_goal": "<inferred user goal>",
        "active_object_types": ["<object type string>"],
        "pattern_detected": "<detected pattern or null>",
    },
    "predictions": [
        {
            "label": "<label from ALLOWED_ACTIONS>",
            "params": {},
            "score": 0.0,
        }
    ],
    "meta": {
        "model": "<model name>",
        "provider": "<provider>",
        "prompt_version": "v1",
    },
}


def build_prompt(request: PredictRequest, model: str, provider: str, top_k: int) -> str:
    """Build the full LLM prompt for a prediction request."""
    payload = request.model_dump()
    return f"""You are an AI assistant embedded in a 3D CAD editor.

Analyze the recent user actions and predict the next {top_k} most likely actions.

## Allowed action labels
{json.dumps(ALLOWED_ACTIONS, indent=2)}

## Required response JSON schema
{json.dumps(_RESPONSE_SCHEMA_TEMPLATE, indent=2)}

## Rules
- Return ONLY valid JSON. No markdown. No explanations outside the JSON object.
- predictions must use labels from the allowed list only.
- Return exactly {top_k} predictions, ordered by score descending (1.0 = highest confidence).
- meta.model must be "{model}", meta.provider must be "{provider}", meta.prompt_version must be "v1".

## Request payload
{json.dumps(payload, indent=2)}
"""

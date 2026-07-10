from typing import Any

from app.predictor.base import PredictorAdapter


#: Fixed pool of mock predictions — ordered by score so slicing gives top_k best.
_MOCK_PREDICTIONS = [
    {"label": "duplicate_selection", "params": {"count": 1},           "score": 0.82},
    {"label": "move_object",         "params": {"axis": "x", "delta": 120}, "score": 0.64},
    {"label": "rotate_object",       "params": {"axis": "z", "angle": 90},  "score": 0.41},
    {"label": "scale_object",        "params": {"factor": 1.5},             "score": 0.30},
    {"label": "delete_selection",    "params": {},                          "score": 0.18},
]


class MockPredictor(PredictorAdapter):
    """Deterministic predictor for testing without a running LLM."""

    def query(self, payload: dict[str, Any], attachments: list | None = None, system_prefix: str = "") -> dict[str, Any]:
        return {
            "todo": [{"id": 1, "description": "Mock query — no LLM used.", "status": "done"}],
            "commands": [],
            "reasoning": "Mock predictor active.",
        }

    def query_design(self, payload: dict[str, Any], attachments: list | None = None, system_prefix: str = "") -> dict[str, Any]:
        return {
            "todo": [{"id": 1, "description": "Mock query-design — no LLM used.", "status": "done"}],
            "commands": [],
            "reasoning": "Mock predictor active.",
            "desiredCommands": [],
            "designFeedback": [],
        }

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        top_k: int = payload.get("options", {}).get("top_k", 3)
        return {
            "task_summary": {
                "label": "mock_task",
                "description": "Mock predictor active — no LLM is used.",
            },
            "context_update": {
                "current_goal": "Mock goal",
                "active_object_types": ["tube"],
                "pattern_detected": "mock_pattern",
            },
            "predictions": _MOCK_PREDICTIONS[:top_k],
            "meta": {
                "model": "mock",
                "provider": "mock",
                "prompt_version": "v1",
            },
        }

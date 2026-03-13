from typing import Any, Optional
from pydantic import BaseModel


# ── Request models ─────────────────────────────────────────────────────────────

class Selection(BaseModel):
    count: int
    types: list[str]


class EditorContext(BaseModel):
    mode: str
    current_tool: str
    selection: Optional[Selection] = None


class PreviousContext(BaseModel):
    current_goal: Optional[str] = None
    # add more inferred state fields here as the system evolves


class Action(BaseModel):
    label: str
    params: dict[str, Any] = {}


class Options(BaseModel):
    top_k: int = 3


class PredictRequest(BaseModel):
    editor_context: EditorContext
    previous_context: Optional[PreviousContext] = None
    recent_actions: list[Action]
    options: Options = Options()


# ── Response models ────────────────────────────────────────────────────────────

class TaskSummary(BaseModel):
    label: str
    description: str


class ContextUpdate(BaseModel):
    current_goal: str
    active_object_types: list[str] = []
    pattern_detected: Optional[str] = None


class PredictedAction(BaseModel):
    label: str
    params: dict[str, Any] = {}
    score: float


class Meta(BaseModel):
    model: str
    provider: str
    prompt_version: str = "v1"


class PredictResponse(BaseModel):
    task_summary: TaskSummary
    context_update: ContextUpdate
    predictions: list[PredictedAction]
    meta: Meta

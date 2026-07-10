from typing import Any, Literal, Optional
from pydantic import BaseModel

ProviderName = Literal["mock", "ollama", "claude"]
#: "default" is the placeholder for ollama until proper model naming is added.
ModelName = Literal["claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-5", "default"]


class Options(BaseModel):
    top_k: int = 3
    #: Backend provider to use. None → MockPredictor.
    provider: Optional[ProviderName] = None
    model: Optional[ModelName] = None
    #: Include the full raw LLM response text in the reply (useful for debugging).
    include_full_answer: bool = False
    #: Include the prompt that was sent to the LLM in the reply (useful for prompt tuning).
    include_prompt: bool = False


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
    #: Raw LLM output — only populated when options.include_full_answer is True.
    llm_raw_response: Optional[str] = None
    #: Prompt sent to the LLM — only populated when options.include_prompt is True.
    llm_prompt: Optional[str] = None


# ── /query endpoint models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    #: Arbitrary JSON payload forwarded verbatim into the prompt.
    payload: dict[str, Any]
    options: Options = Options()


class QueryResponse(BaseModel):
    todo: list[Any] = []
    commands: list[Any] = []
    reasoning: Optional[str] = None
    #: Raw LLM output — only populated when options.include_full_answer is True.
    llm_raw_response: Optional[str] = None
    #: Prompt sent to the LLM — only populated when options.include_prompt is True.
    llm_prompt: Optional[str] = None


# ── /query-design endpoint models ──────────────────────────────────────────

class QueryDesignResponse(QueryResponse):
    """Extended response for /query-design — includes design analysis fields."""
    desiredCommands: list[Any] = []
    designFeedback: list[Any] = []
    notes: Optional[str] = None

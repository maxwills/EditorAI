import asyncio
import json
import re
import time
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from starlette.types import ASGIApp, Receive, Scope, Send

from app.milestone_01_mock import (
    keyword_present as _milestone_01_present,
    run_milestone_01,
    run_milestone_01_design,
    strip_keyword as _strip_milestone_01_keyword,
)
from app.predictor.base import PredictorAdapter
from app.predictor.claude_predictor import CLAUDE_MODEL, ClaudePredictor
from app.predictor.mock_predictor import MockPredictor
from app.predictor.ollama_predictor import OllamaPredictor
from app.prompts import DEV_MODE_KEYWORD, DEV_MODE_SYSTEM_PREFIX
from app.schemas import Options, PredictRequest, PredictResponse, QueryDesignResponse, QueryRequest, QueryResponse
from app.utils.attachment_utils import RawAttachment
from app.utils.json_utils import fallback_query_design_response, fallback_query_response, fallback_response, validate_query_design_response, validate_query_response, validate_response
from app.utils.logging_utils import log

app = FastAPI(title="CAD Prediction Service")

_mock = MockPredictor()


# ── Logging middleware ──────────────────────────────────────────────────────────
#: Pure ASGI middleware — avoids BaseHTTPMiddleware which deadlocks with async handlers.

class _LoggingMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.monotonic()
        method: str = scope.get("method", "")
        path: str = scope.get("path", "")

        # Capture request body chunks as they stream in.
        req_chunks: list[bytes] = []

        async def _receive() -> dict:
            message = await receive()
            if message["type"] == "http.request":
                req_chunks.append(message.get("body", b""))
            return message

        # Capture response status + body chunks as they stream out.
        status_code = 500
        resp_chunks: list[bytes] = []

        async def _send(message: dict) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            elif message["type"] == "http.response.body":
                resp_chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    _log_pair(method, path, status_code,
                               b"".join(req_chunks), b"".join(resp_chunks),
                               (time.monotonic() - start) * 1000)
            await send(message)

        await self.app(scope, _receive, _send)


def _summarise(obj: Any, max_chars: int = 3000) -> str:
    """Compact JSON summary of a dict/list, truncated to max_chars."""
    try:
        s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        s = str(obj)
    return s[:max_chars] + ("…" if len(s) > max_chars else "")


def _extract_multipart_text(raw: str, max_field_chars: int = 2000) -> str:
    """Extract text form-field values from a raw multipart/form-data body.

    Skips binary fields (file uploads). Falls back to a plain truncation when
    no named parts are found (e.g. non-multipart or malformed body).
    """
    parts = re.findall(r'name="([^"]+)"\r?\n\r?\n(.*?)(?=\r?\n--|$)', raw, re.DOTALL)
    if not parts:
        return raw[:max_field_chars]
    lines = []
    for name, value in parts:
        if name == "file":
            lines.append("file=<binary>")
        else:
            v = value.strip()
            lines.append(f"{name}={v[:max_field_chars]}{'…' if len(v) > max_field_chars else ''}")
    return "  ".join(lines)


def _log_pair(method: str, path: str, status: int,
              req_body: bytes, resp_body: bytes, elapsed_ms: float) -> None:
    try:
        req_preview = _summarise(json.loads(req_body.decode("utf-8")))
    except Exception:
        req_preview = _extract_multipart_text(req_body.decode("utf-8", errors="replace"))

    try:
        resp_preview = _summarise(json.loads(resp_body.decode("utf-8")))
    except Exception:
        resp_preview = resp_body.decode("utf-8", errors="replace")[:3000]

    log.info("→ %s %s  %s", method, path, req_preview)
    level = log.error if status >= 400 else log.info
    level("← %s %s  status=%d  elapsed=%.0fms  %s", method, path, status, elapsed_ms, resp_preview)


def _parse_form_json(field_name: str, value: str) -> Any:
    """Parse a JSON string from a form field; raises HTTP 422 on invalid input."""
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"'{field_name}' must be a valid JSON string. {exc}",
        )


app.add_middleware(_LoggingMiddleware)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_predictor(provider: str | None, model: str | None) -> tuple[PredictorAdapter, str, str]:
    """Return (predictor, model_name, provider_name) from request options.

    Dispatch table — add new providers here:
      "ollama"    → OllamaPredictor  (local Ollama)
      "claude"    → ClaudePredictor  (Anthropic Claude)
      "openai"    → OpenAIPredictor  (future)
      None / "mock" / unknown → MockPredictor
    """
    if provider == "ollama":
        if not model:
            log.warning("[main] provider=ollama but no model specified, falling back to mock.")
            return _mock, "mock", "mock"
        return OllamaPredictor(model=model), model, "ollama"

    if provider == "claude":
        resolved_model = model or CLAUDE_MODEL
        return ClaudePredictor(model=resolved_model), resolved_model, "claude"

    # Future providers go here:
    # if provider == "openai":
    #     return OpenAIPredictor(model=model or "gpt-4o"), model or "gpt-4o", "openai"

    return _mock, "mock", "mock"


def _check_dev_mode(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return (system_prefix, cleaned_payload).

    If DEV_MODE_KEYWORD is found anywhere in the serialized payload, returns
    DEV_MODE_SYSTEM_PREFIX and a copy of the payload with the keyword stripped
    from userText (wherever it lives). Otherwise returns ("", payload) unchanged.
    """
    if DEV_MODE_KEYWORD not in json.dumps(payload):
        return "", payload

    log.info("[dev] DEV_MODE_KEYWORD detected — activating developer override.")
    # Strip keyword from userText; check taskContext first (GUI path), then root (bare payload).
    tc = payload.get("taskContext") or {}
    if DEV_MODE_KEYWORD in (tc.get("userText") or ""):
        cleaned_tc = {**tc, "userText": tc["userText"].replace(DEV_MODE_KEYWORD, "").strip()}
        payload = {**payload, "taskContext": cleaned_tc}
    elif DEV_MODE_KEYWORD in (payload.get("userText") or ""):
        payload = {**payload, "userText": payload["userText"].replace(DEV_MODE_KEYWORD, "").strip()}
    return DEV_MODE_SYSTEM_PREFIX, payload


def _try_parse_payload(payload: str) -> Any | None:
    """Best-effort JSON parse for keyword scanning; None if invalid."""
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _pop_error(raw: dict[str, Any]) -> str | None:
    """Extract the _error sentinel (if any) from a predictor result dict."""
    return raw.pop("_error", None)


def _strip_debug_fields(raw: dict[str, Any], options) -> None:
    if not options.include_full_answer:
        raw.pop("llm_raw_response", None)
    if not options.include_prompt:
        raw.pop("llm_prompt", None)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Main prediction endpoint."""
    predictor, model_name, provider_name = _get_predictor(
        request.options.provider, request.options.model
    )

    raw = predictor.predict(request.model_dump())

    if raw is not None:
        error_detail = _pop_error(raw)
        _strip_debug_fields(raw, request.options)
        validated = validate_response(raw)
        if validated is not None:
            return validated
        log.error("[/predict] validation failed. error_detail=%s", error_detail)
        return fallback_response(model=model_name, provider=provider_name, detail=error_detail)

    log.error("[/predict] predictor returned None (API/network error).")
    return fallback_response(model=model_name, provider=provider_name)


@app.post("/query", response_model=QueryResponse)
async def query(
    payload: str = Form(..., description="Task payload as a JSON string."),
    options: str = Form("{}", description="Options as a JSON string (provider, model, etc.)."),
    files: Annotated[list[UploadFile], File(description="Optional files to attach (image, PDF, Excel, DXF, etc.).")] = [],
) -> QueryResponse:
    """CAD modelling assistant endpoint."""
    parsed_payload = _try_parse_payload(payload)
    if _milestone_01_present(payload, options, parsed_payload):
        #: Skip _get_predictor entirely — not Claude, not Ollama, not MockPredictor.
        parsed_payload = _strip_milestone_01_keyword(parsed_payload)
        log.info("[/query] MILESTONE_01_MOCK detected — skipping LLM, running emit CLI.")
        return await asyncio.to_thread(run_milestone_01)

    parsed_payload = _parse_form_json("payload", payload)
    system_prefix, parsed_payload = _check_dev_mode(parsed_payload)
    request = QueryRequest(payload=parsed_payload, options=Options(**_parse_form_json("options", options)))
    predictor, _, _ = _get_predictor(request.options.provider, request.options.model)

    attachments = [RawAttachment(filename=f.filename, data=await f.read()) for f in files] or None
    raw = predictor.query(request.payload, attachments, system_prefix=system_prefix)

    if raw is not None:
        error_detail = _pop_error(raw)
        _strip_debug_fields(raw, request.options)
        validated = validate_query_response(raw)
        if validated is not None:
            return validated
        log.error("[/query] validation failed. error_detail=%s", error_detail)
        return fallback_query_response(detail=error_detail)

    log.error("[/query] predictor returned None (API/network error).")
    return fallback_query_response()


@app.post("/query-design", response_model=QueryDesignResponse)
async def query_design(
    payload: str = Form(..., description="Task payload as a JSON string."),
    options: str = Form("{}", description="Options as a JSON string (provider, model, etc.)."),
    files: Annotated[list[UploadFile], File(description="Optional files to attach (image, PDF, Excel, DXF, etc.).")] = [],
) -> QueryDesignResponse:
    """Design-analysis endpoint.

    Same prompt as /query but prefixed with design-analysis instructions.
    Commands are not dispatched; the LLM analyses gaps and produces desiredCommands / designFeedback.
    """
    parsed_payload = _try_parse_payload(payload)
    if _milestone_01_present(payload, options, parsed_payload):
        parsed_payload = _strip_milestone_01_keyword(parsed_payload)
        log.info("[/query-design] MILESTONE_01_MOCK detected — skipping LLM, running emit CLI.")
        return await asyncio.to_thread(run_milestone_01_design)

    parsed_payload = _parse_form_json("payload", payload)
    system_prefix, parsed_payload = _check_dev_mode(parsed_payload)
    request = QueryRequest(payload=parsed_payload, options=Options(**_parse_form_json("options", options)))
    predictor, _, _ = _get_predictor(request.options.provider, request.options.model)

    attachments = [RawAttachment(filename=f.filename, data=await f.read()) for f in files] or None
    raw = predictor.query_design(request.payload, attachments, system_prefix=system_prefix)

    if raw is not None:
        error_detail = _pop_error(raw)
        _strip_debug_fields(raw, request.options)
        validated = validate_query_design_response(raw)
        if validated is not None:
            return validated
        log.error("[/query-design] validation failed. error_detail=%s", error_detail)
        return fallback_query_design_response(detail=error_detail)

    log.error("[/query-design] predictor returned None (API/network error).")
    return fallback_query_design_response()

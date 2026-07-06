import json
import time
from typing import Any

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.predictor.base import PredictorAdapter
from app.predictor.claude_predictor import CLAUDE_MODEL, ClaudePredictor
from app.predictor.mock_predictor import MockPredictor
from app.predictor.ollama_predictor import OllamaPredictor
from app.schemas import PredictRequest, PredictResponse, QueryDesignResponse, QueryRequest, QueryResponse
from app.utils.json_utils import fallback_query_design_response, fallback_query_response, fallback_response, validate_query_design_response, validate_query_response, validate_response
from app.utils.logging_utils import log

app = FastAPI(title="CAD Prediction Service")

_mock = MockPredictor()


# ── Logging middleware ──────────────────────────────────────────────────────────

class _LoggingMiddleware(BaseHTTPMiddleware):
    """Logs every request (body summary + timing) and its response status."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.monotonic()

        # Cache and preview request body (FastAPI re-reads from cache downstream).
        body_bytes = await request.body()
        try:
            body_preview = _summarise(json.loads(body_bytes.decode("utf-8")))
        except Exception:
            body_preview = body_bytes.decode("utf-8", errors="replace")[:300]

        log.info("→ %s %s  %s", request.method, request.url.path, body_preview)

        response = await call_next(request)
        status_code = response.status_code

        resp_bytes = b""
        async for chunk in response.body_iterator:
            resp_bytes += chunk

        elapsed_ms = (time.monotonic() - start) * 1000

        try:
            resp_preview = _summarise(json.loads(resp_bytes.decode("utf-8")))
        except Exception:
            resp_preview = resp_bytes.decode("utf-8", errors="replace")[:300]

        level = log.error if status_code >= 400 else log.info
        level("← %s %s  status=%d  elapsed=%.0fms  %s",
              request.method, request.url.path, status_code, elapsed_ms, resp_preview)

        return Response(
            content=resp_bytes,
            status_code=status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )


def _summarise(obj: Any, max_chars: int = 300) -> str:
    """Compact JSON summary of a dict/list, truncated to max_chars."""
    try:
        s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        s = str(obj)
    return s[:max_chars] + ("…" if len(s) > max_chars else "")


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
def query(request: QueryRequest) -> QueryResponse:
    """CAD modelling assistant endpoint."""
    predictor, _, _ = _get_predictor(request.options.provider, request.options.model)

    raw = predictor.query(request.payload)

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
def query_design(request: QueryRequest) -> QueryDesignResponse:
    """Design-analysis endpoint.

    Same prompt as /query but prefixed with design-analysis instructions.
    Commands are not dispatched; the LLM analyses gaps and produces desiredCommands / designFeedback.
    """
    predictor, _, _ = _get_predictor(request.options.provider, request.options.model)

    raw = predictor.query_design(request.payload)

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

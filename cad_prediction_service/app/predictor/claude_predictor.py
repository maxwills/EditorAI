import json
from typing import Any

from anthropic import Anthropic, APIError
from dotenv import load_dotenv

from app.predictor.base import PredictorAdapter
from app.prompts import (
    PROMPT_VERSION, QUERY_DESIGN_PROMPT_VERSION, QUERY_PROMPT_VERSION,
    build_prompt, build_query_design_prompt, build_query_design_system_prompt,
    build_query_prompt, build_query_system_prompt, build_query_user_content,
)
from app.schemas import PredictRequest
from app.utils.attachment_utils import to_content_blocks
from app.utils.json_utils import extract_json
from app.utils.logging_utils import log

load_dotenv()

# Available Claude models.
CLAUDE_HAIKU = "claude-haiku-4-5"
CLAUDE_SONNET = "claude-sonnet-5" # default effort is high with adative reasoning. claude-sonnet-4-5
CLAUDE_OPUS = "claude-opus-5"

#: Default model used for Claude predictions.
CLAUDE_MODEL = CLAUDE_SONNET


def _message_text(content: Any) -> str:
    """Join text blocks from a Claude response. Skip thinking / tool_use blocks."""
    parts: list[str] = []
    for block in content or []:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


def _usage_bits(usage: Any) -> str:
    if usage is None:
        return "n/a"
    parts: list[str] = []
    for key in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
        val = getattr(usage, key, None)
        if val:
            parts.append(f"{key}={val}")
    return ", ".join(parts) or "n/a"


def _block_bits(content: Any) -> str:
    labels: list[str] = []
    for block in content or []:
        btype = getattr(block, "type", type(block).__name__)
        if btype == "text":
            labels.append(f"text({len(block.text or '')}c)")
        elif btype == "thinking":
            labels.append(f"thinking({len(getattr(block, 'thinking', '') or '')}c)")
        else:
            labels.append(str(btype))
    return ", ".join(labels) or "none"


def _log_claude_message(message: Any, log_tag: str, raw_text: str) -> None:
    """INFO summary of a Claude reply so empty-text / max_tokens cases are visible in the terminal."""
    stop_reason = getattr(message, "stop_reason", None)
    log.info(
        "[ClaudePredictor] %s stop_reason=%s model=%s usage={%s} blocks=[%s] text_chars=%d",
        log_tag,
        stop_reason,
        getattr(message, "model", None),
        _usage_bits(getattr(message, "usage", None)),
        _block_bits(message.content),
        len(raw_text),
    )
    if raw_text:
        return
    for block in message.content or []:
        if getattr(block, "type", None) != "thinking":
            continue
        thinking = getattr(block, "thinking", "") or ""
        preview = thinking[:2000] + ("…" if len(thinking) > 2000 else "")
        log.warning(
            "[ClaudePredictor] %s empty text (stop_reason=%s). Thinking preview (%d chars):\n%s",
            log_tag, stop_reason, len(thinking), preview,
        )


class ClaudePredictor(PredictorAdapter):
    """Calls the Anthropic Claude API to generate predictions."""

    def __init__(self, model: str = CLAUDE_MODEL) -> None:
        self.model = model
        self.provider = "claude"
        self.client = Anthropic(timeout=120.0)  # reads ANTHROPIC_API_KEY from env

    def predict(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        request = PredictRequest(**payload)
        top_k: int = payload.get("options", {}).get("top_k", 3)

        prompt = build_prompt(request, self.model, self.provider, top_k)

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
        except APIError as exc:
            log.error("[ClaudePredictor] predict API error: %s", exc)
            return None

        raw_text: str = _message_text(message.content)
        _log_claude_message(message, "predict", raw_text)
        log.debug("[ClaudePredictor.predict] LLM raw:\n%s", raw_text)
        parsed = extract_json(raw_text)

        if parsed is None:
            log.error("[ClaudePredictor] predict: could not extract JSON. Full raw output:\n%s", raw_text)
            return {"_error": f"JSON parse failed. Raw output: {raw_text}", "llm_raw_response": raw_text, "llm_prompt": prompt}

        parsed.setdefault("meta", {})
        parsed["meta"].update({
            "model": self.model,
            "provider": self.provider,
            "prompt_version": PROMPT_VERSION,
        })

        parsed["llm_raw_response"] = raw_text
        parsed["llm_prompt"] = prompt

        return parsed

    def _call_claude(
        self,
        system: str,
        user_content: Any,
        max_tokens: int = 16384,
        log_tag: str = "query",
    ) -> tuple[str | None, str | None]:
        """Send a single system+user turn to Claude.

        Returns (text, stop_reason). text is None on API error.
        """
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_content}],
            )
        except APIError as exc:
            log.error("[ClaudePredictor] %s API error: %s", log_tag, exc)
            return None, None
        raw_text = _message_text(message.content)
        _log_claude_message(message, log_tag, raw_text)
        return raw_text, getattr(message, "stop_reason", None)

    def query(self, payload: dict[str, Any], attachments: list | None = None, system_prefix: str = "") -> dict[str, Any] | None:
        system = system_prefix + build_query_system_prompt()
        user_content = to_content_blocks(build_query_user_content(payload), attachments or [])
        prompt_logged = build_query_prompt(payload)

        raw_text, stop_reason = self._call_claude(system, user_content, log_tag="query")
        if raw_text is None:
            return None

        log.debug("[ClaudePredictor.query] LLM raw:\n%s", raw_text)
        parsed = extract_json(raw_text)

        if parsed is None:
            log.error("[ClaudePredictor] query: could not extract JSON (stop_reason=%s, text_chars=%d). Raw:\n%s",
                      stop_reason, len(raw_text), raw_text)
            detail = (
                f"Claude returned no text (stop_reason={stop_reason}). "
                "Thinking likely consumed max_tokens."
                if not raw_text.strip() else
                f"JSON parse failed (stop_reason={stop_reason}). Raw output: {raw_text}"
            )
            return {"_error": detail, "llm_raw_response": raw_text, "llm_prompt": prompt_logged}

        parsed["llm_raw_response"] = raw_text
        parsed["llm_prompt"] = prompt_logged
        parsed.setdefault("prompt_version", QUERY_PROMPT_VERSION)

        return parsed

    def query_design(self, payload: dict[str, Any], attachments: list | None = None, system_prefix: str = "") -> dict[str, Any] | None:
        system = system_prefix + build_query_design_system_prompt()
        user_content = to_content_blocks(build_query_user_content(payload), attachments or [])
        prompt_logged = build_query_design_prompt(payload)

        raw_text, stop_reason = self._call_claude(system, user_content, log_tag="query_design")
        if raw_text is None:
            return None

        log.debug("[ClaudePredictor.query_design] LLM raw:\n%s", raw_text)
        parsed = extract_json(raw_text)

        if parsed is None:
            log.error("[ClaudePredictor] query_design: could not extract JSON (stop_reason=%s, text_chars=%d). Raw:\n%s",
                      stop_reason, len(raw_text), raw_text)
            detail = (
                f"Claude returned no text (stop_reason={stop_reason}). "
                "Thinking likely consumed max_tokens."
                if not raw_text.strip() else
                f"JSON parse failed (stop_reason={stop_reason}). Raw output: {raw_text}"
            )
            return {"_error": detail, "llm_raw_response": raw_text, "llm_prompt": prompt_logged}

        parsed["llm_raw_response"] = raw_text
        parsed["llm_prompt"] = prompt_logged
        parsed.setdefault("prompt_version", QUERY_DESIGN_PROMPT_VERSION)
        parsed.setdefault("desiredCommands", [])
        parsed.setdefault("designFeedback", [])

        return parsed

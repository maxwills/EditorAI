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
CLAUDE_SONNET = "claude-sonnet-4-5"
CLAUDE_OPUS = "claude-opus-4-5"

#: Default model used for Claude predictions.
CLAUDE_MODEL = CLAUDE_HAIKU


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

        raw_text: str = message.content[0].text if message.content else ""
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
        max_tokens: int = 4096,
        log_tag: str = "query",
    ) -> str | None:
        """Send a single system+user turn to Claude and return the raw text, or None on error."""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_content}],
            )
        except APIError as exc:
            log.error("[ClaudePredictor] %s API error: %s", log_tag, exc)
            return None
        return message.content[0].text if message.content else ""

    def query(self, payload: dict[str, Any], attachments: list | None = None, system_prefix: str = "") -> dict[str, Any] | None:
        system = system_prefix + build_query_system_prompt()
        user_content = to_content_blocks(build_query_user_content(payload), attachments or [])
        prompt_logged = build_query_prompt(payload)

        raw_text = self._call_claude(system, user_content, log_tag="query")
        if raw_text is None:
            return None

        log.debug("[ClaudePredictor.query] LLM raw:\n%s", raw_text)
        parsed = extract_json(raw_text)

        if parsed is None:
            log.error("[ClaudePredictor] query: could not extract JSON. Full raw output:\n%s", raw_text)
            return {"_error": f"JSON parse failed. Raw output: {raw_text}", "llm_raw_response": raw_text, "llm_prompt": prompt_logged}

        parsed["llm_raw_response"] = raw_text
        parsed["llm_prompt"] = prompt_logged
        parsed.setdefault("prompt_version", QUERY_PROMPT_VERSION)

        return parsed

    def query_design(self, payload: dict[str, Any], attachments: list | None = None, system_prefix: str = "") -> dict[str, Any] | None:
        system = system_prefix + build_query_design_system_prompt()
        user_content = to_content_blocks(build_query_user_content(payload), attachments or [])
        prompt_logged = build_query_design_prompt(payload)

        raw_text = self._call_claude(system, user_content, log_tag="query_design")
        if raw_text is None:
            return None

        log.debug("[ClaudePredictor.query_design] LLM raw:\n%s", raw_text)
        parsed = extract_json(raw_text)

        if parsed is None:
            log.error("[ClaudePredictor] query_design: could not extract JSON. Full raw output:\n%s", raw_text)
            return {"_error": f"JSON parse failed. Raw output: {raw_text}", "llm_raw_response": raw_text, "llm_prompt": prompt_logged}

        parsed["llm_raw_response"] = raw_text
        parsed["llm_prompt"] = prompt_logged
        parsed.setdefault("prompt_version", QUERY_DESIGN_PROMPT_VERSION)
        parsed.setdefault("desiredCommands", [])
        parsed.setdefault("designFeedback", [])

        return parsed

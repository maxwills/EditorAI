from typing import Any

import httpx

from app.predictor.base import PredictorAdapter
from app.prompts import PROMPT_VERSION, QUERY_DESIGN_PROMPT_VERSION, QUERY_PROMPT_VERSION, build_prompt, build_query_design_prompt, build_query_prompt
from app.schemas import PredictRequest
from app.utils.json_utils import extract_json
from app.utils.logging_utils import log

#: Ollama local API endpoint. Change if Ollama runs on a different host/port.
OLLAMA_URL = "http://localhost:11434/api/generate"


class OllamaPredictor(PredictorAdapter):
    """Calls a locally running Ollama instance to generate predictions."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.provider = "ollama"

    def predict(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        request = PredictRequest(**payload)
        top_k: int = payload.get("options", {}).get("top_k", 3)

        prompt = build_prompt(request, self.model, self.provider, top_k)

        try:
            response = httpx.post(
                OLLAMA_URL,
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=60.0,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            log.error("[OllamaPredictor] predict HTTP error: %s", exc)
            return None

        raw_text: str = response.json().get("response", "")
        log.debug("[OllamaPredictor.predict] LLM raw:\n%s", raw_text)
        parsed = extract_json(raw_text)

        if parsed is None:
            log.error("[OllamaPredictor] predict: could not extract JSON. Full raw output:\n%s", raw_text)
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

    def query(self, payload: dict[str, Any], attachments: list | None = None, system_prefix: str = "") -> dict[str, Any] | None:
        if attachments:
            log.warning("[OllamaPredictor] attachments are not supported — %d file(s) ignored.", len(attachments))
        prompt = system_prefix + build_query_prompt(payload)

        try:
            response = httpx.post(
                OLLAMA_URL,
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=120.0,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            log.error("[OllamaPredictor] query HTTP error: %s", exc)
            return None

        raw_text: str = response.json().get("response", "")
        log.debug("[OllamaPredictor.query] LLM raw:\n%s", raw_text)
        parsed = extract_json(raw_text)

        if parsed is None:
            log.error("[OllamaPredictor] query: could not extract JSON. Full raw output:\n%s", raw_text)
            return {"_error": f"JSON parse failed. Raw output: {raw_text}", "llm_raw_response": raw_text, "llm_prompt": prompt}

        parsed["llm_raw_response"] = raw_text
        parsed["llm_prompt"] = prompt
        parsed.setdefault("prompt_version", QUERY_PROMPT_VERSION)

        return parsed

    def query_design(self, payload: dict[str, Any], attachments: list | None = None, system_prefix: str = "") -> dict[str, Any] | None:
        if attachments:
            log.warning("[OllamaPredictor] attachments are not supported — %d file(s) ignored.", len(attachments))
        prompt = system_prefix + build_query_design_prompt(payload)

        try:
            response = httpx.post(
                OLLAMA_URL,
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=120.0,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            log.error("[OllamaPredictor] query_design HTTP error: %s", exc)
            return None

        raw_text: str = response.json().get("response", "")
        log.debug("[OllamaPredictor.query_design] LLM raw:\n%s", raw_text)
        parsed = extract_json(raw_text)

        if parsed is None:
            log.error("[OllamaPredictor] query_design: could not extract JSON. Full raw output:\n%s", raw_text)
            return {"_error": f"JSON parse failed. Raw output: {raw_text}", "llm_raw_response": raw_text, "llm_prompt": prompt}

        parsed["llm_raw_response"] = raw_text
        parsed["llm_prompt"] = prompt
        parsed.setdefault("prompt_version", QUERY_DESIGN_PROMPT_VERSION)
        parsed.setdefault("desiredCommands", [])
        parsed.setdefault("designFeedback", [])

        return parsed

from typing import Any

import httpx

from app.predictor.base import PredictorAdapter
from app.prompts import PROMPT_VERSION, build_prompt
from app.schemas import PredictRequest
from app.utils.json_utils import extract_json

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
            # Network or HTTP error — caller will use fallback
            print(f"[OllamaPredictor] HTTP error: {exc}")
            return None

        raw_text: str = response.json().get("response", "")
        parsed = extract_json(raw_text)

        if parsed is None:
            print(f"[OllamaPredictor] Could not extract JSON from model output:\n{raw_text[:300]}")
            return None

        # Enforce meta fields regardless of what the model returned
        parsed.setdefault("meta", {})
        parsed["meta"].update({
            "model": self.model,
            "provider": self.provider,
            "prompt_version": PROMPT_VERSION,
        })

        return parsed

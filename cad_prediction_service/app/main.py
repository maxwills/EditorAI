from fastapi import FastAPI

from app.predictor.base import PredictorAdapter
from app.schemas import PredictRequest, PredictResponse
from app.utils.json_utils import fallback_response, validate_response

# ── Configuration ──────────────────────────────────────────────────────────────
# To switch provider: change MODEL_PROVIDER to "ollama" and set OLLAMA_MODEL.
MODEL_PROVIDER = "mock"       # "mock" | "ollama"
OLLAMA_MODEL   = "deepseek-r1:7b"
# ──────────────────────────────────────────────────────────────────────────────


def _load_predictor() -> PredictorAdapter:
    if MODEL_PROVIDER == "ollama":
        from app.predictor.ollama_predictor import OllamaPredictor
        return OllamaPredictor(model=OLLAMA_MODEL)
    from app.predictor.mock_predictor import MockPredictor
    return MockPredictor()


app = FastAPI(title="CAD Prediction Service")
predictor: PredictorAdapter = _load_predictor()


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Main prediction endpoint.

    Flow: request → predictor → validate schema → response (or safe fallback).
    """
    _model  = OLLAMA_MODEL if MODEL_PROVIDER == "ollama" else "mock"
    _provider = MODEL_PROVIDER

    raw = predictor.predict(request.model_dump())

    if raw is not None:
        validated = validate_response(raw)
        if validated is not None:
            return validated

    return fallback_response(model=_model, provider=_provider)

from abc import ABC, abstractmethod
from typing import Any


class PredictorAdapter(ABC):
    """Abstract base for all prediction backends.

    Implement predict() to plug in any LLM provider.
    Returns a raw dict that main.py will validate against PredictResponse.
    Returning None signals that the predictor failed; main.py will use the fallback.
    """

    @abstractmethod
    def predict(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        raise NotImplementedError

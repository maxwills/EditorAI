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

    @abstractmethod
    def query(self, payload: dict[str, Any], attachments: list | None = None, system_prefix: str = "") -> dict[str, Any] | None:
        """Send the query prompt (built from payload) to the LLM and return the parsed response.

        attachments: optional list of FileAttachment objects to include in the request.
        system_prefix: prepended verbatim to the system/top-of-prompt before the normal instructions.
        Returns None on any error so the caller can fall back gracefully.
        """
        raise NotImplementedError

    @abstractmethod
    def query_design(self, payload: dict[str, Any], attachments: list | None = None, system_prefix: str = "") -> dict[str, Any] | None:
        """Send the query-design prompt to the LLM and return the parsed response.

        Same as query() but uses the design-analysis prefix. Returns None on error.
        """
        raise NotImplementedError

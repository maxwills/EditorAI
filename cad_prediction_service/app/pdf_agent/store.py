"""Bounded in-memory storage for resumable PDF drawing sessions."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass

from app.pdf_agent.models import PdfDrawingState


@dataclass
class _Stored:
    state: PdfDrawingState
    touched_at: float


class PdfSessionStore:
    """Keep uploaded PDFs only while an interactive drawing request is active."""

    def __init__(self, ttl_seconds: int = 1800, max_sessions: int = 32) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_sessions = max_sessions
        self._items: dict[str, _Stored] = {}
        self._lock = threading.Lock()

    def create(self, filename: str, pdf_bytes: bytes, payload: dict) -> PdfDrawingState:
        """Create and retain a new state."""
        with self._lock:
            self._prune_locked()
            if len(self._items) >= self._max_sessions:
                oldest = min(self._items, key=lambda key: self._items[key].touched_at)
                del self._items[oldest]
            session_id = uuid.uuid4().hex
            state = PdfDrawingState(session_id, filename, pdf_bytes, payload)
            self._items[session_id] = _Stored(state, time.monotonic())
            return state

    def get(self, session_id: str) -> PdfDrawingState | None:
        """Return and refresh one live session."""
        with self._lock:
            self._prune_locked()
            item = self._items.get(session_id)
            if item is None:
                return None
            item.touched_at = time.monotonic()
            return item.state

    def delete(self, session_id: str) -> None:
        """Forget one completed or cancelled session."""
        with self._lock:
            self._items.pop(session_id, None)

    def _prune_locked(self) -> None:
        cutoff = time.monotonic() - self._ttl_seconds
        expired = [key for key, item in self._items.items() if item.touched_at < cutoff]
        for key in expired:
            del self._items[key]

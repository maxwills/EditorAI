"""Typed state for the PDF tube-drawing agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DrawingInputs:
    """Engineering values resolved from the drawing or the user."""

    scale_denominator: float | None = None
    outer_diameter_m: float | None = None
    wall_thickness_m: float | None = None
    page: int | None = None
    view_box_pdf: list[float] | None = None
    used_likely_defaults: bool = False


@dataclass
class PdfDrawingState:
    """Short-lived state retained while the agent asks a clarification."""

    session_id: str
    filename: str
    pdf_bytes: bytes
    payload: dict[str, Any]
    inputs: DrawingInputs = field(default_factory=DrawingInputs)
    inspection: dict[str, Any] | None = None
    tube_set: dict[str, Any] | None = None
    plan: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

"""Deterministic tools used by the PDF drawing session."""

from __future__ import annotations

import base64
import hashlib
import importlib
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

import fitz

from app.pdf_agent.models import DrawingInputs

class CalibrationInputError(ValueError):
    """Engineering inputs need correction before geometry can be emitted."""


Tool = Callable[..., dict[str, Any]]


class ToolRegistry:
    """Small typed-by-convention registry for independently testable tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, tool: Tool) -> None:
        """Register one named tool."""
        self._tools[name] = tool

    def run(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """Execute one tool or fail with an explicit name."""
        if name not in self._tools:
            raise KeyError(f"Unknown PDF agent tool: {name}")
        return self._tools[name](**kwargs)

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered names in insertion order."""
        return tuple(self._tools)


def inspect_pdf(pdf_bytes: bytes, filename: str = "drawing.pdf") -> dict[str, Any]:
    """Inspect pages, text, and vector density without interpreting geometry."""
    digest = hashlib.sha256(pdf_bytes).hexdigest()
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            if document.needs_pass:
                raise ValueError("PDF is password protected")
            pages = []
            for index, page in enumerate(document):
                drawings = page.get_drawings()
                pages.append({
                    "page": index + 1,
                    "widthPdf": float(page.rect.width),
                    "heightPdf": float(page.rect.height),
                    "rotation": int(page.rotation),
                    "vectorDrawingCount": len(drawings),
                    "text": page.get_text("text")[:12000],
                    "hasVectorGeometry": bool(drawings),
                })
    except Exception as exc:
        raise ValueError(f"Invalid PDF: {exc}") from exc
    if not pages:
        raise ValueError("PDF contains no pages")
    return {"filename": filename, "sha256": digest, "pages": pages}


def _candidate_detector_roots() -> list[Path]:
    """Return configured and workspace-local detector roots."""
    roots: list[Path] = []
    configured = os.environ.get("DETECTA_TUBOS_ROOT")
    if configured:
        roots.append(Path(configured))
    iqs_root = Path(__file__).resolve().parents[4]
    roots.extend([
        iqs_root / "detecta_tubos 2" / "detecta_tubos",
        iqs_root / "detecta_tubos",
    ])
    return roots


def _load_detector_module() -> Any:
    """Load the reusable detector package or workspace module."""
    module_names = ("tube_detector", "detector", "detecta_tubos")
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "detect_pdf_bytes"):
                return module
        except ImportError:
            continue
    for root in _candidate_detector_roots():
        if not root.is_dir():
            continue
        root_s = str(root)
        if root_s not in sys.path:
            sys.path.insert(0, root_s)
        for module_name in module_names:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, "detect_pdf_bytes"):
                    return module
            except ImportError:
                continue
    raise RuntimeError(
        "detecta_tubos library is unavailable; install it or set DETECTA_TUBOS_ROOT"
    )


def detect_tubes(
    pdf_bytes: bytes,
    filename: str,
    page: int | None = None,
    view_box_pdf: list[float] | None = None,
    strategy: str = "auto",
) -> dict[str, Any]:
    """Run detecta_tubos and return its normalized TubeSet dictionary."""
    detector = _load_detector_module()
    kwargs: dict[str, Any] = {"filename": filename, "strategy": strategy}
    if page is not None:
        kwargs["page"] = page
    if view_box_pdf is not None:
        kwargs["view_box_pdf"] = view_box_pdf
    result = detector.detect_pdf_bytes(pdf_bytes, **kwargs)
    if hasattr(result, "to_dict"):
        result = result.to_dict()
    if not isinstance(result, dict):
        raise ValueError("detecta_tubos returned a non-object TubeSet")
    return result


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _points_pdf(tube: dict[str, Any]) -> list[list[float]]:
    raw = tube.get("pointsPdf") or tube.get("puntos_pdf") or tube.get("points")
    if not isinstance(raw, list):
        return []
    points: list[list[float]] = []
    for item in raw:
        if isinstance(item, dict):
            x, y = _number(item.get("x")), _number(item.get("y"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            x, y = _number(item[0]), _number(item[1])
        else:
            continue
        if x is not None and y is not None:
            points.append([x, y])
    return points


def validate_detected_tube_set(tube_set: dict[str, Any]) -> dict[str, Any]:
    """Validate detector output before calibration or scene command emission."""
    tubes = tube_set.get("tubes")
    if not isinstance(tubes, list):
        tubes = tube_set.get("recorridos")
    errors: list[str] = []
    valid = []
    for index, tube in enumerate(tubes or []):
        if not isinstance(tube, dict):
            errors.append(f"tube[{index}] is not an object")
            continue
        points = _points_pdf(tube)
        if len(points) < 2:
            errors.append(f"tube[{index}] has fewer than two points")
            continue
        if any(
            math.hypot(b[0] - a[0], b[1] - a[1]) <= 1e-9
            for a, b in zip(points, points[1:])
        ):
            errors.append(f"tube[{index}] has duplicate consecutive points")
            continue
        valid.append(tube)
    incomplete = (
        tube_set.get("quality", {}).get("incompleteCount")
        if isinstance(tube_set.get("quality"), dict)
        else tube_set.get("recorridos_con_extremos_interiores")
    )
    return {
        "valid": bool(valid) and not errors,
        "tubeCount": len(valid),
        "incompleteCount": int(incomplete or 0),
        "errors": errors,
    }


def _source_view_box(tube_set: dict[str, Any], inspection: dict[str, Any]) -> list[float]:
    source = tube_set.get("source") if isinstance(tube_set.get("source"), dict) else {}
    box = source.get("viewBoxPdf") or tube_set.get("recorte_pdf_xyxy")
    if isinstance(box, list) and len(box) == 4:
        values = [_number(value) for value in box]
        if all(value is not None for value in values):
            return [float(value) for value in values]
    page = inspection["pages"][0]
    return [0.0, 0.0, page["widthPdf"], page["heightPdf"]]


def calibrate_tube_set(
    tube_set: dict[str, Any],
    inspection: dict[str, Any],
    inputs: DrawingInputs,
) -> dict[str, Any]:
    """Map PDF points to world metres and attach manufacturing dimensions."""
    tubes = tube_set.get("tubes")
    if not isinstance(tubes, list):
        tubes = tube_set.get("recorridos") or []
    box = _source_view_box(tube_set, inspection)
    x0, y0, _x1, y1 = box

    diameter_pdf = _number(tube_set.get("diameterPdf"))
    if diameter_pdf is None:
        diameter_pdf = _number(tube_set.get("diametro_estimado_pt"))
    factor = None
    factor_source = None
    if inputs.outer_diameter_m and diameter_pdf and diameter_pdf > 0:
        factor = inputs.outer_diameter_m / diameter_pdf
        factor_source = (
            "likely_outer_diameter"
            if inputs.used_likely_defaults
            else "user_outer_diameter"
        )
    elif inputs.scale_denominator and inputs.scale_denominator > 0:
        factor = inputs.scale_denominator * 0.0254 / 72.0
        factor_source = "drawing_scale"
    calibration = tube_set.get("calibration")
    if factor is None and isinstance(calibration, dict):
        factor = _number(calibration.get("worldMetresPerPdfPoint"))
        factor_source = calibration.get("source") or "detector"

    if factor is None or factor <= 0:
        raise CalibrationInputError("Drawing scale or tube outer diameter is required")

    world_tubes = []
    for index, tube in enumerate(tubes):
        points = _points_pdf(tube)
        if len(points) < 2:
            continue
        tube_diameter_pdf = _number(tube.get("outerDiameterPdf")) or diameter_pdf
        outer_diameter = inputs.outer_diameter_m
        if outer_diameter is None and tube_diameter_pdf:
            outer_diameter = tube_diameter_pdf * factor
        if outer_diameter is None or outer_diameter <= 0:
            raise CalibrationInputError("Tube outer diameter is required")
        thickness = inputs.wall_thickness_m
        if thickness is None or not 0 < thickness < outer_diameter / 2:
            raise CalibrationInputError(
                f"Wall thickness must be positive and less than half the tube outside diameter "
                f"({outer_diameter * 1000:g} mm). Check the view scale, outside diameter, and thickness."
            )
        points_world = [
            {"x": (point[0] - x0) * factor, "y": (y1 - point[1]) * factor, "z": 0.0}
            for point in points
        ]
        radius_pdf = _number(tube.get("bendRadiusPdf"))
        bend_radius = radius_pdf * factor if radius_pdf and radius_pdf > 0 else None
        if bend_radius:
            for point in points_world[1:-1]:
                point["bendRadius"] = bend_radius
        world_tubes.append({
            "id": str(tube.get("id") or tube.get("name") or f"tube_{index + 1}"),
            "pointsWorld": points_world,
            "outerDiameter": outer_diameter,
            "wallThickness": thickness,
            "confidence": _number(tube.get("confidence")),
            "provenance": tube.get("provenance") or {"detector": "detecta_tubos"},
        })

    source = tube_set.get("source") if isinstance(tube_set.get("source"), dict) else {}
    diagnostics = tube_set.get("diagnostics") if isinstance(tube_set.get("diagnostics"), list) else []
    quality = tube_set.get("quality") if isinstance(tube_set.get("quality"), dict) else {}
    return {
        "schemaVersion": "1.0",
        "source": {
            **source,
            "sha256": source.get("sha256") or inspection["sha256"],
            "filename": source.get("filename") or inspection["filename"],
            "viewBoxPdf": box,
        },
        "coordinateSpace": "world_m",
        "calibration": {
            "worldMetresPerPdfPoint": factor,
            "source": factor_source,
            "scaleDenominator": (
                inputs.scale_denominator
                if factor_source == "drawing_scale"
                else None
            ),
            "usedLikelyDefaults": inputs.used_likely_defaults,
        },
        "tubes": world_tubes,
        "diagnostics": diagnostics,
        "quality": {
            **quality,
            "tubeCount": len(world_tubes),
        },
    }


def render_reference_image(
    pdf_bytes: bytes,
    calibrated: dict[str, Any],
    dpi: int = 150,
) -> dict[str, Any]:
    """Render the detected source view with its exact world transform."""
    source = calibrated["source"]
    box = source["viewBoxPdf"]
    page_number = int(source.get("page") or 1)
    factor = float(calibrated["calibration"]["worldMetresPerPdfPoint"])
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        if not 1 <= page_number <= len(document):
            raise ValueError(f"Source page {page_number} is outside the PDF")
        page = document[page_number - 1]
        clip = fitz.Rect(*box) & page.rect
        zoom = dpi / 72.0
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
    world_width = clip.width * factor
    world_height = clip.height * factor
    return {
        "name": f"{source.get('filename') or 'PDF'} p{page_number}",
        "pngBase64": base64.b64encode(pixmap.tobytes("png")).decode("ascii"),
        "position": {"x": world_width / 2, "y": world_height / 2, "z": -0.001},
        "worldScale": factor / zoom,
        "source": {
            "sha256": source.get("sha256"),
            "page": page_number,
            "viewBoxPdf": list(clip),
        },
    }


def default_registry() -> ToolRegistry:
    """Build the production tool registry."""
    registry = ToolRegistry()
    registry.register("inspect_pdf", inspect_pdf)
    registry.register("detect_tubes", detect_tubes)
    registry.register("validate_tube_set", validate_detected_tube_set)
    registry.register("calibrate_tube_set", calibrate_tube_set)
    registry.register("render_reference_image", render_reference_image)
    return registry

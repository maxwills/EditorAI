"""Resumable orchestration for PDF-to-TubeV2 requests."""

from __future__ import annotations

import re
from typing import Any

from app.pdf_agent.models import DrawingInputs, PdfDrawingState
from app.pdf_agent.store import PdfSessionStore
from app.pdf_agent.tools import CalibrationInputError, ToolRegistry, default_registry
from app.utils.attachment_utils import RawAttachment
from app.utils.logging_utils import log

_DRAW_WORDS = re.compile(
    r"\b(draw|model|create|build|trace|tube|pipe|coil|panel|dibuja|modela|crea|traza|tubo|serpent[ií]n)\w*",
    re.IGNORECASE,
)
_SCALE = re.compile(r"(?:scale|escala)?\s*1\s*[:/]\s*(\d+(?:[.,]\d+)?)", re.IGNORECASE)
_OD = re.compile(
    r"(?:outer\s*diameter|outside\s*diameter|\bOD\b|di[aá]metro(?:\s+exterior)?|[Ø⌀])"
    r"\s*[:=]?\s*(\d+(?:[.,]\d+)?)\s*(mm|cm|m)?",
    re.IGNORECASE,
)
_THICKNESS = re.compile(
    r"(?:wall\s*thickness|thickness|espesor|pared)\s*[:=]?\s*(\d+(?:[.,]\d+)?)\s*(mm|cm|m)?",
    re.IGNORECASE,
)
_BARE_MEASURE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(mm|cm|m)\b", re.IGNORECASE)
_PAGE = re.compile(r"(?:page|p[aá]gina)\s*[:#]?\s*(\d+)", re.IGNORECASE)
_BOX = re.compile(
    r"(?:box|crop|roi|recorte)\s*[:=]?\s*"
    r"(\d+(?:[.,]\d+)?)\s*[, ]+\s*(\d+(?:[.,]\d+)?)\s*[, ]+\s*"
    r"(\d+(?:[.,]\d+)?)\s*[, ]+\s*(\d+(?:[.,]\d+)?)",
    re.IGNORECASE,
)
_LIKELY_OUTER_DIAMETER_M = 0.0635
_LIKELY_WALL_THICKNESS_M = 0.006


def _user_text(payload: dict[str, Any]) -> str:
    task_context = payload.get("taskContext")
    if isinstance(task_context, dict):
        return str(task_context.get("userText") or "")
    return str(payload.get("userText") or "")


def _session_id(payload: dict[str, Any]) -> str | None:
    direct = payload.get("agentSessionId")
    if isinstance(direct, str) and direct:
        return direct
    task_context = payload.get("taskContext")
    nested = task_context.get("agentSessionId") if isinstance(task_context, dict) else None
    return nested if isinstance(nested, str) and nested else None


def _is_pdf(attachment: RawAttachment) -> bool:
    return attachment.data.startswith(b"%PDF") or (attachment.filename or "").lower().endswith(".pdf")


def is_pdf_drawing_request(
    payload: dict[str, Any],
    attachments: list[RawAttachment] | None,
) -> bool:
    """Return whether this request belongs to the PDF tube agent."""
    if _session_id(payload):
        return True
    task_context = payload.get("taskContext")
    if isinstance(task_context, dict):
        micro = task_context.get("microUT")
        if isinstance(micro, dict) and micro.get("id") == "draw_tubes_from_pdf":
            return True
        if task_context.get("mode") == "pdf_tube_drawing":
            return True
    has_pdf = any(_is_pdf(item) for item in attachments or [])
    return has_pdf and bool(_DRAW_WORDS.search(_user_text(payload)))


def _to_metres(value: str, unit: str | None) -> float:
    number = float(value.replace(",", "."))
    normalized = (unit or "mm").lower()
    return number if normalized == "m" else number / 100.0 if normalized == "cm" else number / 1000.0


def apply_user_inputs(text: str, inputs: DrawingInputs) -> None:
    """Parse explicit engineering values from one user turn."""
    changed = False
    scale = _SCALE.search(text)
    if scale:
        inputs.scale_denominator = float(scale.group(1).replace(",", "."))
        changed = True
    diameter = _OD.search(text)
    if diameter:
        inputs.outer_diameter_m = _to_metres(diameter.group(1), diameter.group(2))
        changed = True
    thickness = _THICKNESS.search(text)
    if thickness:
        inputs.wall_thickness_m = _to_metres(thickness.group(1), thickness.group(2))
        changed = True
    elif text.strip():
        # Accept concise replies such as `1:50, 6 mm` after the labelled prompt.
        remaining = text
        if diameter:
            remaining = remaining[:diameter.start()] + remaining[diameter.end():]
        bare = _BARE_MEASURE.findall(remaining)
        if len(bare) == 1:
            inputs.wall_thickness_m = _to_metres(*bare[0])
            changed = True
    page = _PAGE.search(text)
    if page:
        inputs.page = int(page.group(1))
        changed = True
    box = _BOX.search(text)
    if box:
        x, y, width, height = [float(value.replace(",", ".")) for value in box.groups()]
        inputs.view_box_pdf = [x, y, x + width, y + height]
        changed = True
    if changed:
        inputs.used_likely_defaults = False


def _todo(state: PdfDrawingState, status: str) -> list[dict[str, Any]]:
    names = [step["tool"] for step in state.plan]
    stages = [
        ("inspect_pdf", "Inspect PDF structure and candidate pages"),
        ("detect_tubes", "Extract tube centerlines with the selected deterministic strategy"),
        ("validate_tube_set", "Validate topology and completeness"),
        ("calibrate_tube_set", "Resolve scale and manufacturing dimensions"),
        ("render_reference_image", "Render the source view with the same world transform"),
        ("emit_editor_command", "Create editable TubeV2 geometry"),
    ]
    result = []
    for index, (name, description) in enumerate(stages, 1):
        if name in names:
            item_status = "done"
        elif status == "needs_input" and name in ("calibrate_tube_set", "emit_editor_command"):
            item_status = "blocked"
        else:
            item_status = "pending"
        result.append({"id": index, "description": description, "status": item_status})
    return result


def _response(
    state: PdfDrawingState,
    status: str,
    commands: list[dict[str, Any]],
    reasoning: str,
) -> dict[str, Any]:
    return {
        "todo": _todo(state, status),
        "commands": commands,
        "reasoning": reasoning,
        "agentSessionId": state.session_id,
        "agentStatus": status,
        "diagnostics": state.diagnostics,
    }


class PdfDrawingAgent:
    """Plan and run deterministic PDF tools, pausing for missing inputs."""

    def __init__(
        self,
        store: PdfSessionStore | None = None,
        tools: ToolRegistry | None = None,
        max_pdf_bytes: int = 50 * 1024 * 1024,
    ) -> None:
        self.store = store or PdfSessionStore()
        self.tools = tools or default_registry()
        self.max_pdf_bytes = max_pdf_bytes

    def handle(
        self,
        payload: dict[str, Any],
        attachments: list[RawAttachment] | None = None,
    ) -> dict[str, Any] | None:
        """Handle a PDF drawing turn; return None for normal chat."""
        if not is_pdf_drawing_request(payload, attachments):
            return None
        state = self._resolve_state(payload, attachments or [])
        if isinstance(state, dict):
            return state
        user_text = _user_text(payload)
        use_likely_defaults = bool(_session_id(payload)) and not user_text.strip()
        if not use_likely_defaults:
            apply_user_inputs(user_text, state.inputs)
        try:
            if state.inspection is None:
                state.inspection = self._run(state, "inspect_pdf", pdf_bytes=state.pdf_bytes, filename=state.filename)
            if state.tube_set is None:
                strategy = self._strategy(state.inspection)
                state.tube_set = self._run(
                    state,
                    "detect_tubes",
                    pdf_bytes=state.pdf_bytes,
                    filename=state.filename,
                    page=state.inputs.page,
                    view_box_pdf=state.inputs.view_box_pdf,
                    strategy=strategy,
                )
                validation = self._run(state, "validate_tube_set", tube_set=state.tube_set)
                if not validation["valid"]:
                    state.diagnostics.append({
                        "severity": "error",
                        "code": "invalid_detection",
                        "message": "; ".join(validation["errors"]) or "No complete tube paths were detected",
                    })
                    return _response(
                        state,
                        "failed",
                        [],
                        "The deterministic detector did not produce valid tube paths. "
                        "Use a clearer vector view or provide a page and ROI.",
                    )
                if validation["incompleteCount"]:
                    state.diagnostics.append({
                        "severity": "warning",
                        "code": "incomplete_paths",
                        "message": f"{validation['incompleteCount']} detected paths may be incomplete",
                    })

            if use_likely_defaults:
                self._apply_likely_parameters(state)
            missing = self._missing_inputs(state)
            if missing:
                question = self._question(state, missing)
                return _response(
                    state,
                    "needs_input",
                    [{"command": "task.askUser", "params": {"question": question}}],
                    "Geometry was detected, but fabrication dimensions are ambiguous.",
                )

            calibrated = self._run(
                state,
                "calibrate_tube_set",
                tube_set=state.tube_set,
                inspection=state.inspection,
                inputs=state.inputs,
            )
            existing_codes = {
                item.get("code")
                for item in calibrated["diagnostics"]
                if isinstance(item, dict)
            }
            calibrated["diagnostics"].extend(
                item for item in state.diagnostics
                if item.get("code") not in existing_codes
            )
            commands: list[dict[str, Any]] = []
            if "render_reference_image" in self.tools.names:
                reference = self._run(
                    state,
                    "render_reference_image",
                    pdf_bytes=state.pdf_bytes,
                    calibrated=calibrated,
                )
                commands.append({
                    "command": "scene.addReferenceImages",
                    "params": {"images": [reference]},
                })
            state.plan.append({"tool": "emit_editor_command"})
            commands.append({
                "command": "scene.createDetectedTubeSet",
                "params": {"tubeSet": calibrated},
            })
            response = _response(
                state,
                "completed",
                commands,
                f"Detected and validated {len(calibrated['tubes'])} tube paths; dimensions are in world metres."
                + (" Likely boiler-tube dimensions were assumed and recorded." if state.inputs.used_likely_defaults else ""),
            )
            self.store.delete(state.session_id)
            return response
        except CalibrationInputError as exc:
            # Keep the uploaded PDF and detection so a corrected answer can resume.
            question = self._question(state, ["invalid_values"], prefix=str(exc))
            return _response(
                state, "needs_input",
                [{"command": "task.askUser", "params": {"question": question}}],
                "The supplied dimensions could not be calibrated; please correct them.",
            )
        except Exception as exc:
            log.exception("[pdf-agent] session %s failed", state.session_id)
            state.diagnostics.append({
                "severity": "error",
                "code": "tool_failure",
                "message": str(exc),
            })
            return _response(state, "failed", [], f"PDF tube extraction failed: {exc}")

    def _resolve_state(
        self,
        payload: dict[str, Any],
        attachments: list[RawAttachment],
    ) -> PdfDrawingState | dict[str, Any]:
        session_id = _session_id(payload)
        if session_id:
            state = self.store.get(session_id)
            if state is not None:
                return state
            ghost = PdfDrawingState(session_id, "", b"", payload)
            return _response(
                ghost,
                "expired",
                [],
                "The PDF drawing session expired. Attach the PDF again to restart.",
            )
        pdfs = [item for item in attachments if _is_pdf(item)]
        if len(pdfs) != 1:
            ghost = PdfDrawingState("none", "", b"", payload)
            return _response(
                ghost,
                "failed",
                [],
                "Attach exactly one PDF for the tube drawing task.",
            )
        pdf = pdfs[0]
        if len(pdf.data) > self.max_pdf_bytes:
            ghost = PdfDrawingState("none", pdf.filename or "drawing.pdf", b"", payload)
            return _response(ghost, "failed", [], "The PDF exceeds the 50 MB upload limit.")
        return self.store.create(pdf.filename or "drawing.pdf", pdf.data, payload)

    def _run(self, state: PdfDrawingState, tool: str, **kwargs: Any) -> dict[str, Any]:
        result = self.tools.run(tool, **kwargs)
        state.plan.append({"tool": tool})
        return result

    @staticmethod
    def _strategy(inspection: dict[str, Any]) -> str:
        pages = inspection.get("pages") or []
        return "auto" if any(page.get("hasVectorGeometry") for page in pages) else "raster"

    @staticmethod
    def _missing_inputs(state: PdfDrawingState) -> list[str]:
        missing = []
        calibration = state.tube_set.get("calibration") if isinstance(state.tube_set, dict) else None
        try:
            has_factor = (
                isinstance(calibration, dict)
                and float(calibration.get("worldMetresPerPdfPoint")) > 0
            )
        except (TypeError, ValueError):
            has_factor = False
        diameter = state.tube_set.get("diameterPdf") if isinstance(state.tube_set, dict) else None
        if not has_factor and not state.inputs.scale_denominator and not (
            state.inputs.outer_diameter_m and diameter
        ):
            missing.append("scale_or_outer_diameter")
        if state.inputs.wall_thickness_m is None:
            missing.append("wall_thickness")
        return missing

    @staticmethod
    def _likely_parameters(state: PdfDrawingState) -> tuple[float, float]:
        """Choose conservative boiler-tube dimensions from available evidence."""
        diameter = state.inputs.outer_diameter_m
        diameter_pdf = (
            state.tube_set.get("diameterPdf")
            if isinstance(state.tube_set, dict)
            else None
        )
        if diameter is None and state.inputs.scale_denominator and diameter_pdf:
            inferred = float(diameter_pdf) * state.inputs.scale_denominator * 0.0254 / 72.0
            if 0.02 <= inferred <= 0.15:
                diameter = inferred
        if diameter is None or not 0.02 <= diameter <= 0.15:
            diameter = _LIKELY_OUTER_DIAMETER_M
        thickness = state.inputs.wall_thickness_m
        if thickness is None or not 0 < thickness < diameter / 2:
            likely_mm = max(1.0, round(diameter * 1000 * 0.1))
            thickness = min(likely_mm / 1000.0, diameter * 0.2)
        return diameter, thickness

    def _apply_likely_parameters(self, state: PdfDrawingState) -> None:
        """Apply and record defaults selected by an empty clarification answer."""
        diameter, thickness = self._likely_parameters(state)
        state.inputs.outer_diameter_m = diameter
        state.inputs.wall_thickness_m = thickness
        state.inputs.used_likely_defaults = True
        state.diagnostics = [
            item for item in state.diagnostics
            if item.get("code") != "assumed_dimensions"
        ]
        state.diagnostics.append({
            "severity": "warning",
            "code": "assumed_dimensions",
            "message": (
                f"Used likely boiler-tube dimensions: OD {diameter * 1000:g} mm, "
                f"wall thickness {thickness * 1000:g} mm."
            ),
        })

    def _question(
        self,
        state: PdfDrawingState,
        missing: list[str],
        prefix: str | None = None,
    ) -> str:
        requests = []
        if "scale_or_outer_diameter" in missing:
            requests.append("the view scale (for example `scale 1:10`) or tube outside diameter (`OD 63.5 mm`)")
        if "wall_thickness" in missing:
            requests.append("the tube wall thickness (for example `thickness 6 mm`)")
        if "invalid_values" in missing:
            requests.append(
                "corrected values with labels and units, for example "
                "`OD 63.5 mm; thickness 6 mm`"
            )
        diameter, thickness = self._likely_parameters(state)
        request_text = "Please provide " + " and ".join(requests) + "."
        default_text = (
            " Send an empty answer to use likely boiler-tube parameters: "
            f"OD {diameter * 1000:g} mm and wall thickness {thickness * 1000:g} mm. "
            "The assumption will be recorded in the model provenance."
        )
        return (f"{prefix} " if prefix else "") + request_text + default_text

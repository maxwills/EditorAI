"""Contract tests for the resumable PDF tube-drawing agent."""

from __future__ import annotations

import json
from io import BytesIO

import fitz
from fastapi.testclient import TestClient

from app import main as main_module
from app.main import app
from app.pdf_agent.session import PdfDrawingAgent, apply_user_inputs, is_pdf_drawing_request
from app.pdf_agent.tools import (
    ToolRegistry,
    calibrate_tube_set,
    inspect_pdf,
    render_reference_image,
    validate_detected_tube_set,
)
from app.utils.attachment_utils import RawAttachment


def _pdf_bytes() -> bytes:
    with fitz.open() as document:
        page = document.new_page(width=200, height=300)
        page.draw_line((20, 20), (20, 260))
        page.draw_line((25, 20), (25, 260))
        return document.tobytes()


def _detected() -> dict:
    return {
        "schemaVersion": "1.0",
        "source": {"page": 1, "viewBoxPdf": [0, 0, 200, 300]},
        "diameterPdf": 5.0,
        "tubes": [{
            "id": "coil_1",
            "pointsPdf": [[22.5, 20], [22.5, 260]],
            "confidence": 0.98,
        }],
        "diagnostics": [],
        "quality": {"incompleteCount": 0},
    }


def _agent() -> PdfDrawingAgent:
    tools = ToolRegistry()
    tools.register("inspect_pdf", inspect_pdf)
    tools.register("detect_tubes", lambda **_kwargs: _detected())
    tools.register("validate_tube_set", validate_detected_tube_set)
    tools.register("calibrate_tube_set", calibrate_tube_set)
    tools.register("render_reference_image", render_reference_image)
    return PdfDrawingAgent(tools=tools)


def _payload(text: str, session_id: str | None = None) -> dict:
    task_context = {
        "microUT": {"id": "draw_tubes_from_pdf"},
        "userText": text,
    }
    if session_id:
        task_context["agentSessionId"] = session_id
    return {"taskContext": task_context}


def test_inspection_reports_hash_page_and_vectors():
    result = inspect_pdf(_pdf_bytes(), "synthetic.pdf")
    assert result["filename"] == "synthetic.pdf"
    assert len(result["sha256"]) == 64
    assert result["pages"][0]["hasVectorGeometry"] is True


def test_input_parser_uses_explicit_units():
    from app.pdf_agent.models import DrawingInputs

    inputs = DrawingInputs()
    apply_user_inputs("scale 1:10, OD 63.5 mm, wall thickness 6 mm, page 2", inputs)
    assert inputs.scale_denominator == 10
    assert inputs.outer_diameter_m == 0.0635
    assert inputs.wall_thickness_m == 0.006
    assert inputs.page == 2


def test_input_parser_accepts_scale_and_bare_thickness_reply():
    from app.pdf_agent.models import DrawingInputs

    inputs = DrawingInputs()
    apply_user_inputs("1:50, 6 mm", inputs)
    assert inputs.scale_denominator == 50
    assert inputs.wall_thickness_m == 0.006


def test_pdf_request_requires_draw_intent_without_explicit_task():
    attachment = RawAttachment("drawing.pdf", _pdf_bytes())
    assert is_pdf_drawing_request({"taskContext": {"userText": "draw the tube coil"}}, [attachment])
    assert not is_pdf_drawing_request({"taskContext": {"userText": "summarize this document"}}, [attachment])


def test_session_asks_then_resumes_without_reupload():
    agent = _agent()
    first = agent.handle(
        _payload("draw this tube"),
        [RawAttachment("synthetic.pdf", _pdf_bytes())],
    )
    assert first["agentStatus"] == "needs_input"
    assert first["commands"][0]["command"] == "task.askUser"
    assert "empty answer" in first["commands"][0]["params"]["question"]
    assert "OD 63.5 mm" in first["commands"][0]["params"]["question"]
    session_id = first["agentSessionId"]

    second = agent.handle(_payload("scale 1:10 and thickness 6 mm", session_id), [])
    assert second["agentStatus"] == "completed"
    assert second["commands"][0]["command"] == "scene.addReferenceImages"
    assert second["commands"][0]["params"]["images"][0]["worldScale"] > 0
    command = second["commands"][1]
    assert command["command"] == "scene.createDetectedTubeSet"
    tube_set = command["params"]["tubeSet"]
    assert tube_set["coordinateSpace"] == "world_m"
    assert tube_set["tubes"][0]["wallThickness"] == 0.006
    assert tube_set["tubes"][0]["pointsWorld"][0]["y"] > tube_set["tubes"][0]["pointsWorld"][1]["y"]


def test_empty_answer_uses_recorded_likely_parameters():
    agent = _agent()
    first = agent.handle(
        _payload("draw this tube"),
        [RawAttachment("synthetic.pdf", _pdf_bytes())],
    )
    second = agent.handle(_payload("", first["agentSessionId"]), [])
    assert second["agentStatus"] == "completed"
    command = next(
        item for item in second["commands"]
        if item["command"] == "scene.createDetectedTubeSet"
    )
    tube_set = command["params"]["tubeSet"]
    assert tube_set["calibration"]["source"] == "likely_outer_diameter"
    assert tube_set["calibration"]["usedLikelyDefaults"] is True
    assert tube_set["tubes"][0]["outerDiameter"] == 0.0635
    assert tube_set["tubes"][0]["wallThickness"] == 0.006
    assert any(item["code"] == "assumed_dimensions" for item in tube_set["diagnostics"])


def test_invalid_detector_result_never_emits_geometry():
    tools = ToolRegistry()
    tools.register("inspect_pdf", inspect_pdf)
    tools.register("detect_tubes", lambda **_kwargs: {"tubes": [{"pointsPdf": [[0, 0]]}]})
    tools.register("validate_tube_set", validate_detected_tube_set)
    tools.register("calibrate_tube_set", calibrate_tube_set)
    result = PdfDrawingAgent(tools=tools).handle(
        _payload("draw"),
        [RawAttachment("synthetic.pdf", _pdf_bytes())],
    )
    assert result["agentStatus"] == "failed"
    assert result["commands"] == []
    assert result["diagnostics"][0]["code"] == "invalid_detection"


def test_query_endpoint_preserves_server_session(monkeypatch):
    monkeypatch.setattr(main_module, "_pdf_agent", _agent())
    client = TestClient(app)
    first = client.post(
        "/query",
        data={"payload": json.dumps(_payload("draw this tube")), "options": "{}"},
        files={"files": ("synthetic.pdf", BytesIO(_pdf_bytes()), "application/pdf")},
    )
    assert first.status_code == 200
    first_body = first.json()
    assert first_body["agentStatus"] == "needs_input"

    second = client.post(
        "/query",
        data={
            "payload": json.dumps(_payload(
                "scale 1:10; thickness 6 mm",
                first_body["agentSessionId"],
            )),
            "options": "{}",
        },
    )
    assert second.status_code == 200
    body = second.json()
    assert body["agentStatus"] == "completed"
    assert body["commands"][0]["command"] == "scene.addReferenceImages"
    assert body["commands"][1]["command"] == "scene.createDetectedTubeSet"


def test_autoplot_requires_upload_and_uses_same_agent(monkeypatch):
    monkeypatch.setattr(main_module, "_pdf_agent", _agent())
    client = TestClient(app)
    missing = client.post("/autoplot", data={"text": "draw tubes"})
    assert missing.status_code == 422

    response = client.post(
        "/autoplot",
        data={"text": "draw tubes"},
        files={"file": ("synthetic.pdf", BytesIO(_pdf_bytes()), "application/pdf")},
    )
    assert response.status_code == 200
    assert response.json()["agentStatus"] == "needs_input"


def test_non_pdf_query_keeps_normal_predictor_path():
    response = TestClient(app).post(
        "/query",
        data={
            "payload": json.dumps({"taskContext": {"userText": "rename the selected object"}}),
            "options": json.dumps({"provider": "mock"}),
        },
    )
    assert response.status_code == 200
    assert response.json()["commands"] == []
    assert response.json()["reasoning"] == "Mock predictor active."


def test_inconsistent_dimensions_ask_for_correction_then_resume():
    agent = _agent()
    first = agent.handle(
        _payload("draw; scale 1:1; thickness 6 mm"),
        [RawAttachment("synthetic.pdf", _pdf_bytes())],
    )
    assert first["agentStatus"] == "needs_input"
    question = first["commands"][0]["params"]["question"]
    assert "less than half" in question
    assert "outside diameter" in question
    state = agent.store.get(first["agentSessionId"])
    assert state is not None
    assert state.diagnostics == []
    second = agent.handle(_payload("OD 63.5 mm; thickness 6 mm", first["agentSessionId"]), [])
    assert second["agentStatus"] == "completed"
    assert sum(step["tool"] == "detect_tubes" for step in state.plan) == 1
    tube_set = second["commands"][-1]["params"]["tubeSet"]
    assert second["commands"][-1]["command"] == "scene.createDetectedTubeSet"
    assert tube_set["calibration"]["source"] == "user_outer_diameter"
    assert tube_set["calibration"]["worldMetresPerPdfPoint"] == 0.0635 / 5.0

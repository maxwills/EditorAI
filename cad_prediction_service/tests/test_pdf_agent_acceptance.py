"""Opt-in acceptance test for the local CUR1000650 drawing."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.pdf_agent.session import PdfDrawingAgent
from app.utils.attachment_utils import RawAttachment

_PDF = os.environ.get("CUR1000650_PDF")


@pytest.mark.integration
@pytest.mark.skipif(not _PDF, reason="Set CUR1000650_PDF to run the local acceptance test")
def test_cur1000650_emits_editable_tube_set():
    path = Path(_PDF)
    result = PdfDrawingAgent().handle(
        {
            "taskContext": {
                "microUT": {"id": "draw_tubes_from_pdf"},
                    "userText": "Draw the tube coil; OD 63.5 mm; thickness 6 mm",
            }
        },
        [RawAttachment(path.name, path.read_bytes())],
    )
    assert result is not None
    assert result["agentStatus"] == "completed", result
    command = next(
        item for item in result["commands"]
        if item["command"] == "scene.createDetectedTubeSet"
    )
    assert command["command"] == "scene.createDetectedTubeSet"
    tube_set = command["params"]["tubeSet"]
    assert len(tube_set["tubes"]) == 4
    assert tube_set["quality"]["incompleteCount"] == 0
    assert all(tube["pointsWorld"] for tube in tube_set["tubes"])
    assert all(tube["outerDiameter"] == pytest.approx(0.0635) for tube in tube_set["tubes"])

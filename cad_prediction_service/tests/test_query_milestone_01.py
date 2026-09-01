"""POST /query MILESTONE_01_MOCK: emit CLI → scene.createSweptPipe, never an LLM."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.main import app
from app.milestone_01_mock import (
    MILESTONE_01_KEYWORD,
    keyword_present,
    strip_keyword,
    tubes_to_commands,
    world_fillet_radius,
)

SAMPLE_EMIT = {
    "unit": "plot",
    "scale": "1:10",
    "plotToWorld": 0.001,
    "n": 2,
    "tubes": [
        {
            "name": "s25_2904_2629",
            "points": [[2629.0, 2903.5], [2683.0, 2903.5]],
            "D": 47.0,
            "R": 0,
            "filletRadius": 0,
            "autoFillet": False,
            "layer": "Symbols_25",
        },
        {
            "name": "ghd1_100_200",
            "points": [[100.0, 200.0], [180.0, 200.0]],
            "D": 46.0,
            "R": 0,
            "filletRadius": 0,
            "autoFillet": False,
            "layer": "GHD-1",
        },
    ],
}

#: 2-pt straight + 3-pt 90 + 4-pt 180. R is plot mm; world fillet = R * 0.001.
CURVE_EMIT = {
    "unit": "plot",
    "scale": "1:10",
    "plotToWorld": 0.001,
    "n": 3,
    "tubes": [
        {
            "name": "s25_straight",
            "points": [[2629.0, 2903.5], [2683.0, 2903.5]],
            "D": 47.0,
            "R": 0,
            "filletRadius": 0,
            "autoFillet": False,
            "layer": "Symbols_25",
        },
        {
            "name": "s25_90",
            "points": [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]],
            "D": 47.0,
            "R": 47.0,
            "autoFillet": False,
            "layer": "Symbols_25",
        },
        {
            "name": "s25_180",
            "points": [[0.0, 0.0], [80.0, 0.0], [80.0, 94.0], [0.0, 94.0]],
            "D": 47.0,
            "R": 47.0,
            "filletRadius": 0.047,
            "autoFillet": False,
            "layer": "Symbols_25",
        },
    ],
}


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def blueprint_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    root = tmp_path / "BlueprintProcessing"
    (root / "src").mkdir(parents=True)
    pdf_dir = root / "data" / "samples"
    pdf_dir.mkdir(parents=True)
    pdf = pdf_dir / "CUR1000650-primario-1.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")
    monkeypatch.setenv("BLUEPRINT_PROCESSING_ROOT", str(root))
    monkeypatch.setenv("MILESTONE_01_PDF", str(pdf))
    monkeypatch.setenv("MILESTONE_01_PYTHON", "python")
    return {"root": root, "pdf": pdf}


def _fake_emit_run(emit: dict):
    def _run(cmd, cwd=None, env=None, capture_output=False, encoding=None, errors=None, timeout=None):
        assert cmd[1:4] == ["-m", "blueprint_processing", "emit"]
        Path(cmd[5]).write_text(json.dumps(emit), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return _run


def _post_query(client: TestClient, payload: dict, options: dict | None = None):
    data = {"payload": json.dumps(payload)}
    if options is not None:
        data["options"] = json.dumps(options)
    return client.post("/query", data=data)


def test_keyword_detected_in_user_text_and_options():
    assert keyword_present(
        json.dumps({"taskContext": {"userText": f"go {MILESTONE_01_KEYWORD}"}}),
        "{}",
        {"taskContext": {"userText": f"go {MILESTONE_01_KEYWORD}"}},
    )
    assert keyword_present("{}", '{"note": "MILESTONE_01_MOCK"}', {})
    assert not keyword_present('{"taskContext": {"userText": "hello"}}', "{}", {"taskContext": {"userText": "hello"}})


def test_strip_keyword_from_task_context_user_text():
    cleaned = strip_keyword({"taskContext": {"userText": f"build CUR1000650 {MILESTONE_01_KEYWORD}"}})
    assert MILESTONE_01_KEYWORD not in cleaned["taskContext"]["userText"]
    assert "CUR1000650" in cleaned["taskContext"]["userText"]


def test_tubes_map_to_scene_create_swept_pipe_plot_units():
    cmds = tubes_to_commands(SAMPLE_EMIT)
    assert len(cmds) == 2
    first = cmds[0]
    assert first["command"] == "scene.createSweptPipe"
    assert first["params"]["points"] == [[2629.0, 2903.5], [2683.0, 2903.5]]
    assert first["params"]["diameter"] == 47.0
    assert first["params"]["id"] == "s25_2904_2629"
    assert first["params"]["plotToWorld"] == 0.001
    assert first["params"]["autoFillet"] is False
    assert first["params"]["filletRadius"] == 0
    assert "createSweptPipe" != first["command"]  # must have scene. prefix
    assert all(c["command"] != "scene.createTube" for c in cmds)


def test_two_pt_fillet_stays_zero_even_if_r_set():
    tube = {
        "name": "straight",
        "points": [[0.0, 0.0], [50.0, 0.0]],
        "D": 47.0,
        "R": 47.0,
        "filletRadius": 0.047,
    }
    assert world_fillet_radius(tube, 2, 0.001) == 0.0
    cmds = tubes_to_commands({"plotToWorld": 0.001, "tubes": [tube]})
    assert cmds[0]["params"]["filletRadius"] == 0
    assert cmds[0]["params"]["autoFillet"] is False


def test_three_pt_r_converts_plot_mm_to_world_metres():
    tube = CURVE_EMIT["tubes"][1]
    assert len(tube["points"]) == 3
    assert tube["R"] == 47.0
    assert "filletRadius" not in tube
    cmds = tubes_to_commands({"plotToWorld": 0.001, "tubes": [tube]})
    fr = cmds[0]["params"]["filletRadius"]
    assert fr != 0
    assert fr == pytest.approx(47.0 * 0.001)
    assert cmds[0]["params"]["autoFillet"] is False
    assert cmds[0]["params"]["diameter"] == 47.0
    assert cmds[0]["params"]["points"] == tube["points"]
    assert cmds[0]["command"] == "scene.createSweptPipe"


def test_four_pt_uses_emit_fillet_radius_when_already_world():
    tube = CURVE_EMIT["tubes"][2]
    assert len(tube["points"]) == 4
    cmds = tubes_to_commands({"plotToWorld": 0.001, "tubes": [tube]})
    assert cmds[0]["params"]["filletRadius"] == pytest.approx(0.047)
    assert cmds[0]["params"]["autoFillet"] is False
    assert cmds[0]["params"]["id"] == "s25_180"


def test_three_pt_zero_placeholder_fillet_falls_back_to_r():
    #: Emit schema always has filletRadius; 0 means unset, R is the plot-mm bend.
    tube = {
        "name": "s25_90",
        "points": [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]],
        "D": 47.0,
        "R": 47.0,
        "filletRadius": 0,
        "autoFillet": False,
    }
    cmds = tubes_to_commands({"plotToWorld": 0.001, "tubes": [tube]})
    assert cmds[0]["params"]["filletRadius"] == pytest.approx(0.047)
    assert cmds[0]["params"]["filletRadius"] != 0


def test_three_pt_does_not_invent_r_when_missing():
    tube = {"name": "no_r", "points": [[0, 0], [10, 0], [10, 10]], "D": 47.0}
    assert world_fillet_radius(tube, 3, 0.001) == 0.0


def test_mixed_emit_maps_one_command_per_tube():
    cmds = tubes_to_commands(CURVE_EMIT)
    assert [c["command"] for c in cmds] == ["scene.createSweptPipe"] * 3
    assert cmds[0]["params"]["filletRadius"] == 0
    assert cmds[1]["params"]["filletRadius"] == pytest.approx(0.047)
    assert cmds[2]["params"]["filletRadius"] == pytest.approx(0.047)
    assert all(c["params"]["autoFillet"] is False for c in cmds)


def test_query_with_keyword_skips_predictor_and_returns_swept_pipes(client, blueprint_env):
    payload = {"taskContext": {"userText": f"place CUR1000650 1:10 {MILESTONE_01_KEYWORD}"}}
    options = {"provider": "claude", "model": "claude-haiku-4-5"}

    with (
        patch("app.main._get_predictor") as get_predictor,
        patch("app.predictor.claude_predictor.ClaudePredictor.query") as claude_query,
        patch("app.predictor.ollama_predictor.OllamaPredictor.query") as ollama_query,
        patch("app.predictor.mock_predictor.MockPredictor.query") as mock_query,
        patch("app.milestone_01_mock.subprocess.run", side_effect=_fake_emit_run(SAMPLE_EMIT)) as run,
    ):
        resp = _post_query(client, payload, options)

    assert resp.status_code == 200
    body = resp.json()
    get_predictor.assert_not_called()
    claude_query.assert_not_called()
    ollama_query.assert_not_called()
    mock_query.assert_not_called()
    run.assert_called_once()
    cmd = run.call_args[0][0]
    assert cmd[1:4] == ["-m", "blueprint_processing", "emit"]
    assert run.call_args.kwargs["cwd"] == str(blueprint_env["root"])
    assert str(blueprint_env["root"] / "src") in run.call_args.kwargs["env"]["PYTHONPATH"]

    commands = body["commands"]
    assert len(commands) == 2
    assert all(c["command"] == "scene.createSweptPipe" for c in commands)
    assert commands[0]["params"]["points"] == [[2629.0, 2903.5], [2683.0, 2903.5]]
    assert commands[0]["params"]["diameter"] == 47.0
    assert commands[1]["params"]["diameter"] == 46.0
    assert all(c["params"]["autoFillet"] is False for c in commands)
    assert all(c["params"]["plotToWorld"] == 0.001 for c in commands)
    assert "MILESTONE_01_MOCK" in body["reasoning"]
    assert "no LLM" in body["reasoning"].lower() or "skipped LLM" in body["reasoning"]
    assert "n=2" in body["reasoning"]
    assert body["todo"]


def test_query_curve_emit_skips_llm_and_maps_fillets(client, blueprint_env):
    payload = {"taskContext": {"userText": f"place CUR1000650 curves {MILESTONE_01_KEYWORD}"}}
    options = {"provider": "claude", "model": "claude-haiku-4-5"}

    with (
        patch("app.main._get_predictor") as get_predictor,
        patch("app.predictor.claude_predictor.ClaudePredictor.query") as claude_query,
        patch("app.predictor.mock_predictor.MockPredictor.query") as mock_query,
        patch("app.milestone_01_mock.subprocess.run", side_effect=_fake_emit_run(CURVE_EMIT)),
    ):
        resp = _post_query(client, payload, options)

    assert resp.status_code == 200
    get_predictor.assert_not_called()
    claude_query.assert_not_called()
    mock_query.assert_not_called()
    commands = resp.json()["commands"]
    assert len(commands) == 3
    assert commands[0]["params"]["filletRadius"] == 0
    assert len(commands[1]["params"]["points"]) == 3
    assert commands[1]["params"]["filletRadius"] != 0
    assert commands[1]["params"]["filletRadius"] == pytest.approx(0.047)
    assert len(commands[2]["params"]["points"]) == 4
    assert commands[2]["params"]["filletRadius"] == pytest.approx(0.047)
    assert all(c["params"]["autoFillet"] is False for c in commands)
    assert all(c["command"] == "scene.createSweptPipe" for c in commands)


def test_query_without_keyword_uses_normal_predictor(client, blueprint_env):
    payload = {"taskContext": {"userText": "place CUR1000650 1:10"}}

    with (
        patch.object(main_module, "_get_predictor", wraps=main_module._get_predictor) as get_predictor,
        patch("app.milestone_01_mock.subprocess.run") as run,
    ):
        resp = _post_query(client, payload, {"provider": "mock"})

    assert resp.status_code == 200
    body = resp.json()
    get_predictor.assert_called_once()
    run.assert_not_called()
    assert body["commands"] == []
    assert body["reasoning"] == "Mock predictor active."


def test_query_cli_nonzero_exit_returns_empty_commands_and_stderr(client, blueprint_env):
    payload = {"taskContext": {"userText": MILESTONE_01_KEYWORD}}

    def _fail(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="emit: missing Symbols_25 layer")

    with (
        patch("app.main._get_predictor") as get_predictor,
        patch("app.milestone_01_mock.subprocess.run", side_effect=_fail),
    ):
        resp = _post_query(client, payload)

    assert resp.status_code == 200
    body = resp.json()
    get_predictor.assert_not_called()
    assert body["commands"] == []
    assert "missing Symbols_25 layer" in body["reasoning"]
    assert "MILESTONE_01_MOCK" in body["reasoning"]
    assert "exit 2" in body["reasoning"]


def test_query_missing_pdf_does_not_invent_tubes_or_call_llm(client, tmp_path, monkeypatch):
    root = tmp_path / "BlueprintProcessing"
    (root / "src").mkdir(parents=True)
    monkeypatch.setenv("BLUEPRINT_PROCESSING_ROOT", str(root))
    monkeypatch.setenv("MILESTONE_01_PDF", str(root / "missing.pdf"))
    monkeypatch.setenv("MILESTONE_01_PYTHON", "python")

    with (
        patch("app.main._get_predictor") as get_predictor,
        patch("app.milestone_01_mock.subprocess.run") as run,
    ):
        resp = _post_query(client, {"taskContext": {"userText": MILESTONE_01_KEYWORD}})

    assert resp.status_code == 200
    get_predictor.assert_not_called()
    run.assert_not_called()
    body = resp.json()
    assert body["commands"] == []
    assert "not found" in body["reasoning"].lower()
    assert "MILESTONE_01_MOCK" in body["reasoning"]


def test_query_runs_real_python_m_emit_module(client, tmp_path, monkeypatch):
    """End-to-end CLI: python -m blueprint_processing emit writes JSON; no subprocess stub."""
    root = tmp_path / "BlueprintProcessing"
    pkg = root / "src" / "blueprint_processing"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "__main__.py").write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        f"EMIT = {SAMPLE_EMIT!r}\n"
        "def main():\n"
        "    args = sys.argv[1:]\n"
        "    if not args or args[0] != 'emit' or len(args) != 3:\n"
        "        print('usage: emit <pdf> <json>', file=sys.stderr)\n"
        "        sys.exit(2)\n"
        "    Path(args[2]).write_text(json.dumps(EMIT), encoding='utf-8')\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    pdf = root / "data" / "samples" / "CUR1000650-primario-1.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4 dummy")
    monkeypatch.setenv("BLUEPRINT_PROCESSING_ROOT", str(root))
    monkeypatch.setenv("MILESTONE_01_PDF", str(pdf))
    monkeypatch.setenv("MILESTONE_01_PYTHON", sys.executable)

    with patch("app.main._get_predictor") as get_predictor:
        resp = _post_query(
            client,
            {"taskContext": {"userText": f"CUR1000650 {MILESTONE_01_KEYWORD}"}},
            {"provider": "claude"},
        )

    assert resp.status_code == 200
    get_predictor.assert_not_called()
    body = resp.json()
    assert len(body["commands"]) == 2
    assert body["commands"][0]["command"] == "scene.createSweptPipe"
    assert body["commands"][0]["params"]["points"][0] == [2629.0, 2903.5]
    assert body["commands"][0]["params"]["autoFillet"] is False
    assert "blueprint_processing emit" in body["reasoning"]


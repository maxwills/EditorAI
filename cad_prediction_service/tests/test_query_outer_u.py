"""POST /query and /query-design OUTER_U_MOCK: emit-outer-u --preunion-glue → scene.createSweptPipe, never an LLM."""

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
    OUTER_U_KEYWORD,
    keyword_present,
    strip_keyword,
    tubes_to_commands,
)

#: One outer U at 1:50. R is plot mm; world fillet = R * 0.001.
SAMPLE_OUTER_U_EMIT = {
    "unit": "plot",
    "scale": "1:50",
    "plotToWorld": 0.001,
    "n": 1,
    "tubes": [
        {
            "name": "outer_u",
            "points": [[0.0, 0.0], [0.0, 200.0], [80.0, 200.0], [80.0, 0.0]],
            "D": 47.0,
            "R": 47.0,
            "filletRadius": 0,
            "autoFillet": False,
            "layer": "Symbols_25",
        },
    ],
}

SAMPLE_EMIT_SHORT = {
    "unit": "plot",
    "scale": "1:10",
    "plotToWorld": 0.001,
    "n": 20,
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


def _fake_cli_run(emit: dict, subcommand: str):
    def _run(cmd, cwd=None, env=None, capture_output=False, encoding=None, errors=None, timeout=None):
        assert cmd[1:4] == ["-m", "blueprint_processing", subcommand]
        if subcommand == "emit-outer-u":
            assert cmd[6:] == ["--preunion-glue"]
        else:
            assert "--preunion-glue" not in cmd
        Path(cmd[5]).write_text(json.dumps(emit), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return _run


def _post_query(client: TestClient, payload: dict, options: dict | None = None):
    data = {"payload": json.dumps(payload)}
    if options is not None:
        data["options"] = json.dumps(options)
    return client.post("/query", data=data)


def _post_query_design(client: TestClient, payload: dict, options: dict | None = None):
    data = {"payload": json.dumps(payload)}
    if options is not None:
        data["options"] = json.dumps(options)
    return client.post("/query-design", data=data)


def test_keyword_detected_in_user_text_and_options():
    assert keyword_present(
        json.dumps({"taskContext": {"userText": f"go {OUTER_U_KEYWORD}"}}),
        "{}",
        {"taskContext": {"userText": f"go {OUTER_U_KEYWORD}"}},
        OUTER_U_KEYWORD,
    )
    assert keyword_present("{}", '{"note": "OUTER_U_MOCK"}', {}, OUTER_U_KEYWORD)
    assert not keyword_present(
        '{"taskContext": {"userText": "hello"}}',
        "{}",
        {"taskContext": {"userText": "hello"}},
        OUTER_U_KEYWORD,
    )
    assert not keyword_present(
        json.dumps({"taskContext": {"userText": MILESTONE_01_KEYWORD}}),
        "{}",
        {"taskContext": {"userText": MILESTONE_01_KEYWORD}},
        OUTER_U_KEYWORD,
    )


def test_strip_keyword_from_task_context_user_text():
    cleaned = strip_keyword(
        {"taskContext": {"userText": f"build CUR1000650 {OUTER_U_KEYWORD}"}},
        OUTER_U_KEYWORD,
    )
    assert OUTER_U_KEYWORD not in cleaned["taskContext"]["userText"]
    assert "CUR1000650" in cleaned["taskContext"]["userText"]


def test_tubes_map_fillet_r_times_plot_to_world():
    cmds = tubes_to_commands(SAMPLE_OUTER_U_EMIT)
    assert len(cmds) == 1
    first = cmds[0]
    assert first["command"] == "scene.createSweptPipe"
    assert first["params"]["points"] == SAMPLE_OUTER_U_EMIT["tubes"][0]["points"]
    assert first["params"]["diameter"] == 47.0
    assert first["params"]["id"] == "outer_u"
    assert first["params"]["plotToWorld"] == 0.001
    assert first["params"]["autoFillet"] is False
    assert first["params"]["filletRadius"] == pytest.approx(47.0 * 0.001)


def test_query_with_keyword_skips_predictor_and_runs_emit_outer_u(client, blueprint_env):
    payload = {"taskContext": {"userText": f"place CUR1000650 1:50 {OUTER_U_KEYWORD}"}}
    options = {"provider": "claude", "model": "claude-haiku-4-5"}

    with (
        patch("app.main._get_predictor") as get_predictor,
        patch("app.predictor.claude_predictor.ClaudePredictor.query") as claude_query,
        patch("app.predictor.ollama_predictor.OllamaPredictor.query") as ollama_query,
        patch("app.predictor.mock_predictor.MockPredictor.query") as mock_query,
        patch(
            "app.milestone_01_mock.subprocess.run",
            side_effect=_fake_cli_run(SAMPLE_OUTER_U_EMIT, "emit-outer-u"),
        ) as run,
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
    assert cmd[1:4] == ["-m", "blueprint_processing", "emit-outer-u"]
    assert cmd[3] != "emit"
    assert cmd[6:] == ["--preunion-glue"]
    assert run.call_args.kwargs["cwd"] == str(blueprint_env["root"])
    assert str(blueprint_env["root"] / "src") in run.call_args.kwargs["env"]["PYTHONPATH"]

    commands = body["commands"]
    assert len(commands) == 1
    assert commands[0]["command"] == "scene.createSweptPipe"
    assert commands[0]["params"]["points"] == SAMPLE_OUTER_U_EMIT["tubes"][0]["points"]
    assert commands[0]["params"]["diameter"] == 47.0
    assert commands[0]["params"]["autoFillet"] is False
    assert commands[0]["params"]["plotToWorld"] == 0.001
    assert commands[0]["params"]["filletRadius"] == pytest.approx(0.047)
    assert "OUTER_U_MOCK" in body["reasoning"]
    assert "emit-outer-u" in body["reasoning"]
    assert "--preunion-glue" in body["reasoning"]
    assert "skipped LLM" in body["reasoning"]
    assert "n=1" in body["reasoning"]
    assert "n=20" not in body["reasoning"]
    assert body["todo"]


def test_query_design_with_keyword_skips_predictor_and_runs_emit_outer_u(client, blueprint_env):
    payload = {"taskContext": {"userText": OUTER_U_KEYWORD}}
    options = {"provider": "claude", "model": "claude-haiku-4-5"}

    with (
        patch("app.main._get_predictor") as get_predictor,
        patch("app.predictor.claude_predictor.ClaudePredictor.query_design") as claude_query,
        patch("app.predictor.mock_predictor.MockPredictor.query_design") as mock_query,
        patch(
            "app.milestone_01_mock.subprocess.run",
            side_effect=_fake_cli_run(SAMPLE_OUTER_U_EMIT, "emit-outer-u"),
        ) as run,
    ):
        resp = _post_query_design(client, payload, options)

    assert resp.status_code == 200
    body = resp.json()
    get_predictor.assert_not_called()
    claude_query.assert_not_called()
    mock_query.assert_not_called()
    cmd = run.call_args[0][0]
    assert cmd[1:4] == ["-m", "blueprint_processing", "emit-outer-u"]
    assert cmd[6:] == ["--preunion-glue"]
    assert len(body["commands"]) == 1
    assert body["commands"][0]["command"] == "scene.createSweptPipe"
    assert "OUTER_U_MOCK" in body["reasoning"]
    assert body["desiredCommands"] == []
    assert body["designFeedback"] == []


def test_both_keywords_outer_u_wins_one_u_not_short_1_10(client, blueprint_env):
    payload = {
        "taskContext": {
            "userText": f"place CUR1000650 {MILESTONE_01_KEYWORD} {OUTER_U_KEYWORD}",
        }
    }

    with (
        patch("app.main._get_predictor") as get_predictor,
        patch(
            "app.milestone_01_mock.subprocess.run",
            side_effect=_fake_cli_run(SAMPLE_OUTER_U_EMIT, "emit-outer-u"),
        ) as run,
    ):
        resp = _post_query(client, payload, {"provider": "claude"})

    assert resp.status_code == 200
    get_predictor.assert_not_called()
    cmd = run.call_args[0][0]
    assert cmd[3] == "emit-outer-u"
    assert cmd[3] != "emit"
    assert cmd[6:] == ["--preunion-glue"]
    body = resp.json()
    assert len(body["commands"]) == 1
    assert "OUTER_U_MOCK" in body["reasoning"]
    assert "emit-outer-u" in body["reasoning"]
    assert "n=1" in body["reasoning"]
    assert "MILESTONE_01_MOCK:" not in body["reasoning"]


def test_milestone_01_still_runs_emit_not_outer_u(client, blueprint_env):
    payload = {"taskContext": {"userText": f"place CUR1000650 1:10 {MILESTONE_01_KEYWORD}"}}

    with (
        patch("app.main._get_predictor") as get_predictor,
        patch(
            "app.milestone_01_mock.subprocess.run",
            side_effect=_fake_cli_run(SAMPLE_EMIT_SHORT, "emit"),
        ) as run,
    ):
        resp = _post_query(client, payload, {"provider": "claude"})

    assert resp.status_code == 200
    get_predictor.assert_not_called()
    cmd = run.call_args[0][0]
    assert cmd[1:4] == ["-m", "blueprint_processing", "emit"]
    assert cmd[3] != "emit-outer-u"
    assert "--preunion-glue" not in cmd
    body = resp.json()
    assert "MILESTONE_01_MOCK" in body["reasoning"]
    assert "emit n=" in body["reasoning"]
    assert "emit-outer-u" not in body["reasoning"]
    assert "--preunion-glue" not in body["reasoning"]
    assert "OUTER_U_MOCK" not in body["reasoning"]


def test_query_keyword_in_options_triggers_outer_u(client, blueprint_env):
    payload = {"taskContext": {"userText": "place CUR1000650"}}
    options = {"note": OUTER_U_KEYWORD, "provider": "claude"}

    with (
        patch("app.main._get_predictor") as get_predictor,
        patch(
            "app.milestone_01_mock.subprocess.run",
            side_effect=_fake_cli_run(SAMPLE_OUTER_U_EMIT, "emit-outer-u"),
        ) as run,
    ):
        resp = _post_query(client, payload, options)

    assert resp.status_code == 200
    get_predictor.assert_not_called()
    cmd = run.call_args[0][0]
    assert cmd[3] == "emit-outer-u"
    assert cmd[6:] == ["--preunion-glue"]
    assert "OUTER_U_MOCK" in resp.json()["reasoning"]


def test_query_without_keyword_uses_normal_predictor(client, blueprint_env):
    payload = {"taskContext": {"userText": "place CUR1000650 1:50 outer U"}}

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
    payload = {"taskContext": {"userText": OUTER_U_KEYWORD}}

    def _fail(cmd, **kwargs):
        assert cmd[3] == "emit-outer-u"
        assert cmd[6:] == ["--preunion-glue"]
        return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="emit-outer-u: missing overlay")

    with (
        patch("app.main._get_predictor") as get_predictor,
        patch("app.milestone_01_mock.subprocess.run", side_effect=_fail),
    ):
        resp = _post_query(client, payload)

    assert resp.status_code == 200
    body = resp.json()
    get_predictor.assert_not_called()
    assert body["commands"] == []
    assert "missing overlay" in body["reasoning"]
    assert "OUTER_U_MOCK" in body["reasoning"]
    assert "exit 2" in body["reasoning"]
    assert "emit-outer-u" in body["reasoning"]


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
        resp = _post_query(client, {"taskContext": {"userText": OUTER_U_KEYWORD}})

    assert resp.status_code == 200
    get_predictor.assert_not_called()
    run.assert_not_called()
    body = resp.json()
    assert body["commands"] == []
    assert "not found" in body["reasoning"].lower()
    assert "OUTER_U_MOCK" in body["reasoning"]


def test_query_runs_real_python_m_emit_outer_u_module(client, tmp_path, monkeypatch):
    """End-to-end CLI: python -m blueprint_processing emit-outer-u --preunion-glue writes JSON; no subprocess stub."""
    root = tmp_path / "BlueprintProcessing"
    pkg = root / "src" / "blueprint_processing"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "__main__.py").write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        f"EMIT = {SAMPLE_OUTER_U_EMIT!r}\n"
        "def main():\n"
        "    args = sys.argv[1:]\n"
        "    if not args or args[0] != 'emit-outer-u' or len(args) != 4 or args[3] != '--preunion-glue':\n"
        "        print('usage: emit-outer-u <pdf> <json> --preunion-glue', file=sys.stderr)\n"
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
            {"taskContext": {"userText": f"CUR1000650 {OUTER_U_KEYWORD}"}},
            {"provider": "claude"},
        )

    assert resp.status_code == 200
    get_predictor.assert_not_called()
    body = resp.json()
    assert len(body["commands"]) == 1
    assert body["commands"][0]["command"] == "scene.createSweptPipe"
    assert body["commands"][0]["params"]["points"][0] == [0.0, 0.0]
    assert body["commands"][0]["params"]["autoFillet"] is False
    assert body["commands"][0]["params"]["filletRadius"] == pytest.approx(0.047)
    assert "blueprint_processing emit-outer-u" in body["reasoning"]
    assert "--preunion-glue" in body["reasoning"]

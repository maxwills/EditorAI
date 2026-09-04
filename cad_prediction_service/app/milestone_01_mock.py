"""Keyword mocks: run Blub's emit CLI instead of any LLM and map tubes to Agent commands.

MILESTONE_01_MOCK → `emit` (short 1:10, n=20). OUTER_U_MOCK → `emit-outer-u --preunion-glue`
(opt-in growth tubes, n=14 = locked 12 + yup_r115_0/1; not the signed fallback n=12).
OUTER_U_MOCK wins when both keywords appear.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from app.schemas import QueryDesignResponse, QueryResponse
from app.utils.logging_utils import log

#: Presence anywhere in the /query body (payload string, options string, or
#: serialized parsed payload) skips Claude/Ollama/MockPredictor and runs emit.
MILESTONE_01_KEYWORD = "MILESTONE_01_MOCK"

#: Same detection style; runs emit-outer-u --preunion-glue (n=14 = locked 12 + yup_r115_0/1), not default emit.
OUTER_U_KEYWORD = "OUTER_U_MOCK"

#: Without this flag emit-outer-u returns the signed fallback n=12, not yup_r115_0/1 growth tubes.
_OUTER_U_CLI_ARGS = ["--preunion-glue"]

#: Max's local blueprint-processing checkout (emit lives on branch emit-tubev2-1-10-straights).
_DEFAULT_ROOT = r"D:\Max\Docs\BlueprintProcessing\BlueprintProcessing"
_DEFAULT_PDF_REL = Path("data") / "samples" / "CUR1000650-primario-1.pdf"
_EMIT_TIMEOUT_S = 300
_PLOT_TO_WORLD = 0.001
_STDERR_CAP = 2000


def keyword_present(
    payload_str: str,
    options_str: str,
    parsed_payload: Any,
    keyword: str = MILESTONE_01_KEYWORD,
) -> bool:
    """True if keyword appears anywhere in the request body fields."""
    if keyword in (payload_str or "") or keyword in (options_str or ""):
        return True
    if parsed_payload is None:
        return False
    try:
        return keyword in json.dumps(parsed_payload)
    except (TypeError, ValueError):
        return False


def strip_keyword(payload: Any, keyword: str = MILESTONE_01_KEYWORD) -> Any:
    """Remove the keyword from userText (taskContext first, then root) like DEV_MODE."""
    if not isinstance(payload, dict):
        return payload
    tc = payload.get("taskContext") or {}
    if isinstance(tc, dict) and keyword in (tc.get("userText") or ""):
        cleaned_tc = {**tc, "userText": tc["userText"].replace(keyword, "").strip()}
        return {**payload, "taskContext": cleaned_tc}
    if keyword in (payload.get("userText") or ""):
        return {**payload, "userText": payload["userText"].replace(keyword, "").strip()}
    return payload


def _clip(text: str) -> str:
    text = (text or "").strip()
    if len(text) <= _STDERR_CAP:
        return text
    return text[:_STDERR_CAP] + "…"


def _resolve_root() -> Path:
    return Path(os.environ.get("BLUEPRINT_PROCESSING_ROOT", _DEFAULT_ROOT))


def _resolve_pdf(root: Path) -> Path:
    override = os.environ.get("MILESTONE_01_PDF")
    if override:
        return Path(override)
    return root / _DEFAULT_PDF_REL


def _resolve_python(root: Path) -> str:
    override = os.environ.get("MILESTONE_01_PYTHON")
    if override:
        return override
    win = root / ".venv" / "Scripts" / "python.exe"
    if win.is_file():
        return str(win)
    posix = root / ".venv" / "bin" / "python"
    if posix.is_file():
        return str(posix)
    return "python"


def _is_missing_executable(python: str) -> bool:
    """Bare command names (`python`) are resolved by the OS; path-like values must exist."""
    if os.path.sep in python or (os.path.altsep and os.path.altsep in python):
        return not Path(python).is_file()
    if python.lower().endswith(".exe"):
        return not Path(python).is_file()
    return False


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def world_fillet_radius(tube: dict[str, Any], n_points: int, plot_to_world: float) -> float:
    #: COMMAND_REGISTRY filletRadius is world metres; emit R is plot mm (world = R * plotToWorld).
    #: Prefer emit filletRadius when already world (non-zero); else convert R. Never invent R.
    #: 2-pt straights: always 0. autoFillet stays false at the command layer.
    if n_points <= 2:
        return 0.0
    fr = tube.get("filletRadius")
    if _is_number(fr) and fr != 0:
        return float(fr)
    r = tube.get("R")
    if _is_number(r) and r != 0:
        return float(r) * float(plot_to_world)
    return 0.0


def tubes_to_commands(emit: dict[str, Any]) -> list[dict[str, Any]]:
    """One scene.createSweptPipe per emit tube. Plot units; no glue; no invented families."""
    plot_to_world = emit.get("plotToWorld", _PLOT_TO_WORLD)
    if not isinstance(plot_to_world, (int, float)):
        plot_to_world = _PLOT_TO_WORLD

    commands: list[dict[str, Any]] = []
    for tube in emit.get("tubes") or []:
        if not isinstance(tube, dict):
            continue
        points = tube.get("points")
        diameter = tube.get("D")
        if not isinstance(points, list) or len(points) < 2 or diameter is None:
            continue
        params: dict[str, Any] = {
            "points": points,
            "diameter": diameter,
            "plotToWorld": plot_to_world,
            "autoFillet": False,
            "filletRadius": world_fillet_radius(tube, len(points), plot_to_world),
        }
        name = tube.get("name")
        if isinstance(name, str) and name:
            params["id"] = name
        commands.append({"command": "scene.createSweptPipe", "params": params})
    return commands


def _error_response(
    reason: str,
    cli_used: str = "",
    keyword: str = MILESTONE_01_KEYWORD,
    subcommand: str = "emit",
) -> QueryResponse:
    extra = f" CLI: {cli_used}." if cli_used else ""
    return QueryResponse(
        todo=[
            {"id": 1, "description": f"Detect {keyword}", "status": "done"},
            {"id": 2, "description": f"Run blueprint_processing {subcommand} (no LLM)", "status": "blocked"},
        ],
        commands=[],
        reasoning=f"{keyword}: skipped LLM (no Claude, no Ollama). {reason}{extra}",
    )


def _success_response(
    emit: dict[str, Any],
    commands: list[dict[str, Any]],
    cli_used: str,
    keyword: str = MILESTONE_01_KEYWORD,
    subcommand: str = "emit",
) -> QueryResponse:
    n = emit.get("n", len(commands))
    return QueryResponse(
        todo=[
            {"id": 1, "description": f"Detect {keyword}", "status": "done"},
            {"id": 2, "description": f"Run blueprint_processing {subcommand} (no LLM)", "status": "done"},
            {"id": 3, "description": f"Map {len(commands)} tubes to scene.createSweptPipe", "status": "done"},
        ],
        commands=commands,
        reasoning=(
            f"{keyword}: skipped LLM (no Claude, no Ollama). "
            f"{subcommand} n={n}, mapped {len(commands)} scene.createSweptPipe commands. "
            f"CLI: {cli_used}."
        ),
    )


def run_emit_cli(
    subcommand: str = "emit",
    extra_cli_args: list[str] | None = None,
) -> tuple[dict[str, Any] | None, str, str]:
    """Run `python -m blueprint_processing <subcommand> <pdf> <json> [extra]`.

    Returns (parsed_json_or_None, error_or_empty, cli_summary).
    Never invents tubes: CLI failure → (None, reason, cli).
    """
    extra_cli_args = list(extra_cli_args or [])
    extra_s = (" " + " ".join(extra_cli_args)) if extra_cli_args else ""
    root = _resolve_root()
    pdf = _resolve_pdf(root)
    python = _resolve_python(root)
    cli_used = (
        f"{python} -m blueprint_processing {subcommand} {pdf} <temp.json>"
        f"{extra_s} (cwd={root}, PYTHONPATH=src)"
    )

    if not root.is_dir():
        return None, f"BLUEPRINT_PROCESSING_ROOT not found: {root}", cli_used
    if not pdf.is_file():
        return None, f"{subcommand} PDF not found: {pdf}", cli_used
    if _is_missing_executable(python):
        return None, f"{subcommand} python not found: {python}", cli_used

    out_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            out_path = tmp.name

        env = os.environ.copy()
        src = str(root / "src")
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = src if not existing else src + os.pathsep + existing

        cmd = [python, "-m", "blueprint_processing", subcommand, str(pdf), out_path, *extra_cli_args]
        log.info("[%s] running: %s", subcommand, cli_used)
        result = subprocess.run(
            cmd,
            cwd=str(root),
            env=env,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=_EMIT_TIMEOUT_S,
        )
        if result.returncode != 0:
            stderr = _clip(result.stderr or "")
            stdout = _clip(result.stdout or "")
            detail = stderr or stdout or "(no stderr)"
            return None, f"{subcommand} CLI failed (exit {result.returncode}): {detail}", cli_used

        out = Path(out_path)
        if not out.is_file() or out.stat().st_size == 0:
            return None, f"{subcommand} CLI produced no JSON file", cli_used
        try:
            data = json.loads(out.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return None, f"{subcommand} JSON parse failed: {exc}", cli_used
        if not isinstance(data, dict):
            return None, f"{subcommand} JSON is not an object", cli_used
        return data, "", cli_used
    except FileNotFoundError as exc:
        return None, f"{subcommand} CLI not found: {exc}", cli_used
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", "replace")
        return None, f"{subcommand} CLI timed out after {_EMIT_TIMEOUT_S}s. stderr: {_clip(stderr)}", cli_used
    except Exception as exc:
        return None, f"{subcommand} error: {exc}", cli_used
    finally:
        if out_path:
            Path(out_path).unlink(missing_ok=True)


def _run_keyword_cli(
    subcommand: str,
    keyword: str,
    extra_cli_args: list[str] | None = None,
) -> QueryResponse:
    """Skip all predictors; CLI tubes → scene.createSweptPipe QueryResponse."""
    try:
        data, err, cli_used = run_emit_cli(subcommand, extra_cli_args)
        if err or data is None:
            log.error("[%s] %s", keyword, err)
            return _error_response(err or f"{subcommand} returned no data", cli_used, keyword, subcommand)
        if "tubes" in data and not isinstance(data.get("tubes"), list):
            return _error_response(f"{subcommand} JSON has no tubes array", cli_used, keyword, subcommand)
        commands = tubes_to_commands(data)
        return _success_response(data, commands, cli_used, keyword, subcommand)
    except Exception as exc:
        log.exception("[%s] unexpected error", keyword)
        return _error_response(f"unexpected error: {exc}", keyword=keyword, subcommand=subcommand)


def run_milestone_01() -> QueryResponse:
    """Skip all predictors; emit tubes → scene.createSweptPipe QueryResponse."""
    return _run_keyword_cli("emit", MILESTONE_01_KEYWORD)


def run_outer_u() -> QueryResponse:
    """Skip all predictors; emit-outer-u --preunion-glue (n=14) → scene.createSweptPipe QueryResponse."""
    return _run_keyword_cli("emit-outer-u", OUTER_U_KEYWORD, extra_cli_args=_OUTER_U_CLI_ARGS)


def run_milestone_01_design() -> QueryDesignResponse:
    """Same emit path for /query-design so the keyword never hits an LLM there either."""
    return QueryDesignResponse(**run_milestone_01().model_dump())


def run_outer_u_design() -> QueryDesignResponse:
    """Same emit-outer-u --preunion-glue path for /query-design so OUTER_U_MOCK never hits an LLM."""
    return QueryDesignResponse(**run_outer_u().model_dump())

"""AutoPlot image workflow. The route starts this; the inner step is the v8 loop. Tubes stay disconnected."""

import base64
import hashlib
import json
import math
from pathlib import Path

import fitz

from app.predictor.claude_predictor import ClaudePredictor
from app.utils.attachment_utils import RawAttachment, to_content_blocks
from app.utils.json_utils import extract_json
from app.utils.logging_utils import log

#: Workspace copy of CUR1000650 primario 1. Do not commit the PDF.
PDF_PATH = Path(__file__).resolve().parents[1] / "local" / "input" / "CUR1000650-Model-primario-1.pdf"
EXPECTED_SHA256 = "9454070f939cf9002b23b8003930ed41a9a4ba5ce1fcbcc224d9505cce61766a"
DPI = 150
SCALE = 0.01
GAP = 0.5
#: Positive value turns thinking on. sonnet-5 uses adaptive effort, not this token count.
THINKING_BUDGET = 8192
MAX_TOKENS = 16384
#: Claude replies, not counting the one thinking-off retry.
MAX_CALLS = 8
_ACTIONS = ("crop", "record", "done")

#: Fixed objective. The editor text is not read.
OBJECTIVE = (
    "Focus on the 1:10 view of the superheater. "
    "Choose one coil. Record that coil on that view only, from one open end through every change of direction to the other open end. "
    "Do not record the 1:50 view. Do not record every coil. Do not emit a pipe list. Do not draw."
)
NUDGE_PLAN = (
    "Lay a plan first. Reply with JSON only: "
    '{"coil":"one sentence naming the coil",'
    '"view":"one sentence naming the 1:10 view",'
    '"plan":[{"action":"crop","why":"one sentence"},{"action":"record","why":"one sentence"},{"action":"done"}]}'
    ". Name one coil. The view sentence must contain 1:10. "
    "Each action is crop, record, or done. Do not crop or record in this reply."
)
NUDGE_NEXT = "Reply with one JSON action: crop, record, or done."
NUDGE_UNKNOWN = "That action is not available. " + NUDGE_NEXT
NUDGE_BAD_CROP = (
    "That crop box is not on the page. "
    "box x, y, width, and height are fractions from 0 to 1, origin top-left. "
    + NUDGE_NEXT
)
NUDGE_NEED_CROP = "Crop the 1:10 view before recording it. " + NUDGE_NEXT
NUDGE_NO_MORE = "That view was already replaced once. Reply with done."
NUDGE_EARLY_DONE = "The 1:10 view is not recorded yet. Record that coil on it, then done."

SYSTEM_PROMPT = (
    OBJECTIVE + " "
    "The first reply is a plan only. Later replies are exactly one action. "
    "Actions: crop, record, done. "
    'Plan JSON: {"coil":"one sentence naming the coil","view":"one sentence naming the 1:10 view",'
    '"plan":[{"action":"crop","why":"one sentence"},{"action":"record","why":"one sentence"},{"action":"done"}]}. '
    "coil names the one coil. The view sentence must contain 1:10. "
    'crop: {"action":"crop","box":{"x":0,"y":0,"width":1,"height":1}}. '
    "Crop the whole 1:10 view before recording. "
    "box x, y, width, and height are fractions of the page from 0 to 1. Origin is the top-left. "
    'record: {"action":"record","view":"the 1:10 view sentence from the plan","box":{"x":0,"y":0,"width":1,"height":1},'
    '"centerline":[{"x":0,"y":0},{"x":1,"y":0}],"bendRadius":0.01,"callout":""}. '
    "view is that same 1:10 sentence. Repeating it replaces the record. Do not name another view. "
    "The record box is the 1:10 view on the page. "
    "centerline points are fractions of that box, not of the page, origin top-left, in travel order along that coil. "
    "Start at one open end. Put a point at each change of direction and along each straight run. Stop at the other open end. "
    "Do not send the outer U. Do not send only the two ends. "
    "At least two points. There is no point cap. "
    "bendRadius is a fraction of the box width. callout is the radius text printed on the sheet, or empty. "
    'done: {"action":"done"}. '
    "No diameter, no thickness, no world coordinates."
)


class AutoPlotRegionsError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _record_nudge(coil: str | None, view: str | None) -> str:
    """Reject a record and name the coil and the 1:10 view the plan already chose."""
    name = coil.strip() if isinstance(coil, str) and coil.strip() else "the coil named in the plan"
    label = view.strip() if isinstance(view, str) and view.strip() else "the 1:10 view named in the plan"
    return (
        f'Record "{name}" on "{label}" only. '
        "That record needs that view sentence, the page box of the 1:10 view, "
        "one centerline of at least two points in fractions of that box, in travel order along that coil, "
        'and a positive bendRadius as a fraction of the box width. callout is the printed radius text or "". '
        "Start at one open end. Put a point at each change of direction and along each straight run. "
        "Stop at the other open end. Do not send the outer U. Do not send only the two ends. "
        + NUDGE_NEXT
    )


def _review_nudge(count: int) -> str:
    """One chance to replace a trace that misses a bend or stops short of an end."""
    return (
        f"Recorded {count} points of the 1:10 view. "
        "If any inflection is missing, or either end stops short, "
        "record that same view again with the complete path. Otherwise done."
    )


def _wrong_view_nudge(view: str) -> str:
    """Reject a record that is not the 1:10 sentence from the plan."""
    return f'Only the 1:10 view from the plan ("{view}"). Repeat that sentence. ' + NUDGE_NEXT


def run(text: str) -> dict:
    """Start the image workflow. `text` is required and not interpreted. Returns the 1:10 view. Tubes stay disconnected."""
    if not text or not text.strip():
        raise AutoPlotRegionsError(422, "text is required")
    pdf = _read_local_pdf()
    pix = _open_page(pdf)
    plan, coil, view, turns, extractions = _agent_loop(ClaudePredictor(), pix, pdf)
    for item in extractions:
        local = item.pop("centerlineCrop", None)
        log.info(
            "[autoplot] extraction %s points=%d crop=%s page=%s",
            item["view"],
            len(item["centerline"]),
            json.dumps(local),
            json.dumps(item["centerline"]),
        )
    return {
        "commands": [
            {
                "command": "scene.addReferenceImages",
                "params": {"images": _images_from_extractions(pix, extractions)},
            }
        ],
        "analysis": {
            "part": "superheater",
            "coil": coil,
            "view": view,
            "plan": plan,
            "turns": turns,
            "extractions": extractions,
        },
    }


def _agent_loop(predictor: ClaudePredictor, pix: fitz.Pixmap, pdf: bytes) -> tuple[list, str, str, list, list]:
    """Plan one coil and the 1:10 view, then crop / record / done. Succeeds with that one view after review or one replacement."""
    messages: list[dict] = [_opening_user(pdf)]
    plan = None
    coil = None
    view = None
    views: list[str] = []
    by_key: dict[str, dict] = {}
    turns: list[dict] = []
    thinking_budget: int | None = THINKING_BUDGET
    #: One replay rejection retries with thinking off. It is not one of the eight replies.
    rewound = False
    cropped = False
    last_png: str | None = None
    record_count = 0
    reviewed = False
    stopped = False
    calls = 0

    while calls < MAX_CALLS:
        raw_text, stop_reason, _thinking, blocks = predictor.call_messages(
            SYSTEM_PROMPT,
            messages,
            max_tokens=MAX_TOKENS,
            log_tag="autoplot",
            thinking_budget=thinking_budget,
        )
        if raw_text is None:
            if _rewind_thinking(stop_reason, messages, rewound, thinking_budget):
                rewound = True
                thinking_budget = 0
                messages = _strip_thinking(messages)
                log.warning("[autoplot] thinking replay rejected; continuing with thinking off")
                continue
            raise AutoPlotRegionsError(502, f"Claude API error: {stop_reason}")
        calls += 1
        if not raw_text.strip():
            raise AutoPlotRegionsError(502, f"Claude returned no text (stop_reason={stop_reason})")
        if not blocks:
            blocks = [{"type": "text", "text": raw_text}]

        stored_points = len(by_key[views[0].casefold()]["centerline"]) if views else 0
        step = _classify(
            extract_json(raw_text), plan is not None, views, coil, view,
            cropped, record_count, reviewed, stored_points,
        )
        png = _png_b64(pix, step["box"]) if step["accepted"] == "crop" else None
        if step["accepted"] == "crop" and png is None:
            step = {"accepted": "rejected", "nudge": NUDGE_BAD_CROP}

        turn = {"accepted": step["accepted"]}
        if step["accepted"] == "rejected":
            turn["reason"] = step["nudge"]
            turn["reply"] = raw_text[:500]
        elif step["accepted"] == "plan":
            plan = step["plan"]
            coil = step["coil"]
            view = step["view"]
            turn["plan"] = plan
            turn["coil"] = coil
            turn["view"] = view
        elif step["accepted"] == "crop":
            turn["box"] = step["box"]
        elif step["accepted"] == "record":
            turn["view"] = step["extraction"]["view"]
            turn["points"] = len(step["extraction"]["centerline"])
        turns.append(turn)
        log.info("[autoplot] turn %d %s", calls, step["accepted"])

        messages.append({"role": "assistant", "content": _assistant_content(blocks, raw_text, thinking_budget)})
        if step.get("stop"):
            stopped = True
            break
        if step["accepted"] == "record":
            extraction = step["extraction"]
            _store_view(views, by_key, extraction)
            record_count += 1
            log.info(
                "[autoplot] record %s points=%d crop=%s page=%s",
                extraction["view"],
                len(extraction["centerline"]),
                json.dumps(extraction.get("centerlineCrop")),
                json.dumps(extraction["centerline"]),
            )
            if record_count == 1:
                note = f'Recorded the 1:10 view ({len(extraction["centerline"])} points). ' + NUDGE_NEXT
            else:
                note = f'Replaced the 1:10 view ({len(extraction["centerline"])} points). Reply with done.'
            messages.append({"role": "user", "content": note})
        elif step["accepted"] == "crop":
            cropped = True
            last_png = png
            messages.append({"role": "user", "content": _crop_content(png)})
        elif step.get("review"):
            reviewed = True
            messages.append({"role": "user", "content": _review_content(last_png, step["nudge"])})
        else:
            messages.append({"role": "user", "content": step["nudge"]})

    extractions = [by_key[name.casefold()] for name in views]
    if plan is None or not coil or not view or len(extractions) != 1 or not stopped:
        log.error(
            "[autoplot] incomplete coil=%s view=%s plan=%s turns=%s",
            coil, view, json.dumps(plan), json.dumps(turns),
        )
        raise AutoPlotRegionsError(422, "Claude did not record the 1:10 view")
    return plan, coil, view, turns, extractions


def _opening_user(pdf: bytes) -> dict:
    content = to_content_blocks(
        OBJECTIVE + " Lay a plan first. Name one coil and the 1:10 view. Do not crop or record in this reply.",
        [RawAttachment(filename=PDF_PATH.name, data=pdf)],
    )
    return {"role": "user", "content": content}


def _crop_content(png_b64: str) -> list[dict]:
    return [
        {"type": "text", "text": "This is that crop. " + NUDGE_NEXT},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": png_b64},
        },
    ]


def _review_content(png_b64: str | None, text: str) -> list[dict]:
    """Show the 1:10 crop again with the one replacement offer."""
    blocks: list[dict] = [{"type": "text", "text": text}]
    if png_b64:
        blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": png_b64},
        })
    return blocks


def _rewind_thinking(error: str | None, messages: list[dict], rewound: bool, thinking_budget: int | None) -> bool:
    """True when the next turn was rejected for replaying thinking blocks."""
    if rewound or thinking_budget == 0:
        return False
    if not any(message.get("role") == "assistant" for message in messages):
        return False
    return "thinking" in (error or "").lower()


def _assistant_content(blocks: list[dict], raw_text: str, thinking_budget: int | None) -> list[dict]:
    """Replay thinking blocks only while thinking is still on."""
    if thinking_budget != 0:
        return blocks
    text = [block for block in blocks if isinstance(block, dict) and block.get("type") == "text"]
    return text or [{"type": "text", "text": raw_text}]


def _strip_thinking(messages: list[dict]) -> list[dict]:
    """Drop thinking blocks so the loop can continue with thinking off."""
    stripped = []
    for message in messages:
        content = message.get("content")
        if message.get("role") != "assistant" or not isinstance(content, list):
            stripped.append(message)
            continue
        text = [block for block in content if isinstance(block, dict) and block.get("type") == "text"]
        stripped.append({"role": "assistant", "content": text or [{"type": "text", "text": ""}]})
    return stripped


def _classify(
    parsed: object,
    have_plan: bool,
    views: list[str],
    coil: str | None,
    view: str | None,
    cropped: bool,
    record_count: int,
    reviewed: bool,
    stored_points: int,
) -> dict:
    """One reply becomes a plan, one action, or a rejection. Does not store anything."""
    if not have_plan:
        laid = _plan_steps(parsed)
        if laid is None:
            return {"accepted": "rejected", "nudge": NUDGE_PLAN}
        return {
            "accepted": "plan",
            "plan": laid["steps"],
            "coil": laid["coil"],
            "view": laid["view"],
            "nudge": NUDGE_NEXT,
        }

    action = parsed.get("action") if isinstance(parsed, dict) else None
    if action == "crop":
        box = _fraction_box(parsed.get("box")) if isinstance(parsed, dict) else None
        if box is None:
            return {"accepted": "rejected", "nudge": NUDGE_BAD_CROP}
        return {"accepted": "crop", "box": box}
    if action == "record":
        if not cropped:
            return {"accepted": "rejected", "nudge": NUDGE_NEED_CROP}
        if record_count >= 2:
            return {"accepted": "rejected", "nudge": NUDGE_NO_MORE}
        extraction = _extraction(parsed) if isinstance(parsed, dict) else None
        if extraction is None:
            return {"accepted": "rejected", "nudge": _record_nudge(coil, view)}
        if not view or extraction["view"].casefold() != view.casefold():
            return {"accepted": "rejected", "nudge": _wrong_view_nudge(view or "the 1:10 view")}
        return {"accepted": "record", "extraction": extraction}
    if action == "done":
        if len(views) == 1 and (reviewed or record_count >= 2):
            return {"accepted": "done", "stop": True}
        if len(views) == 1:
            return {"accepted": "rejected", "nudge": _review_nudge(stored_points), "review": True}
        return {"accepted": "rejected", "nudge": NUDGE_EARLY_DONE}
    return {"accepted": "rejected", "nudge": NUDGE_UNKNOWN}


def _plan_steps(parsed: object) -> dict | None:
    """Accept a plan only when it names one coil and a view sentence that contains 1:10."""
    if not isinstance(parsed, dict):
        return None
    coil = parsed.get("coil")
    view = parsed.get("view")
    if not isinstance(coil, str) or not coil.strip():
        return None
    if not isinstance(view, str) or "1:10" not in view:
        return None
    steps = parsed.get("plan")
    if not isinstance(steps, list) or not steps:
        return None
    ready = []
    for step in steps:
        if not isinstance(step, dict) or step.get("action") not in _ACTIONS:
            return None
        item = {"action": step["action"]}
        why = step.get("why")
        if isinstance(why, str) and why.strip():
            item["why"] = why.strip()
        ready.append(item)
    return {"coil": coil.strip(), "view": view.strip(), "steps": ready}


def _extraction(parsed: dict) -> dict | None:
    """Accept a crop-fraction centerline and convert it onto the page."""
    view = parsed.get("view")
    if not isinstance(view, str) or not view.strip():
        return None
    box = _fraction_box(parsed.get("box"))
    local = _centerline(parsed.get("centerline"))
    radius = _bend_radius(parsed.get("bendRadius"))
    callout = parsed.get("callout", "")
    if box is None or local is None or radius is None or not isinstance(callout, str):
        return None
    page_radius = radius * box["width"]
    if not math.isfinite(page_radius) or page_radius <= 0:
        return None
    return {
        "view": view.strip(),
        "box": box,
        "centerline": _to_page(box, local),
        "centerlineCrop": local,
        "bendRadius": page_radius,
        "callout": callout.strip(),
    }


def _to_page(box: dict, local: list[dict]) -> list[dict]:
    """Map fractions of the record box onto page fractions."""
    points = []
    for point in local:
        x = box["x"] + point["x"] * box["width"]
        y = box["y"] + point["y"] * box["height"]
        points.append({"x": min(1.0, max(0.0, x)), "y": min(1.0, max(0.0, y))})
    return points


def _centerline(value: object) -> list[dict] | None:
    if not isinstance(value, list) or len(value) < 2:
        return None
    points = []
    for point in value:
        if not isinstance(point, dict):
            return None
        try:
            x = float(point["x"])
            y = float(point["y"])
        except (KeyError, TypeError, ValueError):
            return None
        if x < 0 or y < 0 or x > 1 or y > 1:
            return None
        points.append({"x": x, "y": y})
    return points


def _bend_radius(value: object) -> float | None:
    try:
        radius = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(radius) or radius <= 0:
        return None
    return radius


def _store_view(views: list[str], by_key: dict[str, dict], extraction: dict) -> None:
    """Insert a view, or replace one that uses the same sentence."""
    key = extraction["view"].casefold()
    if key not in by_key:
        views.append(extraction["view"])
    else:
        for index, existing in enumerate(views):
            if existing.casefold() == key:
                views[index] = extraction["view"]
                break
    by_key[key] = extraction


def _read_local_pdf() -> bytes:
    if not PDF_PATH.is_file():
        raise AutoPlotRegionsError(404, "local PDF missing")
    data = PDF_PATH.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_SHA256:
        raise AutoPlotRegionsError(422, "local PDF hash mismatch")
    return data


def _fraction_box(box: object) -> dict | None:
    """Accept a top-left fraction rectangle. Clamp the far edges to the page."""
    if not isinstance(box, dict):
        return None
    try:
        x = float(box["x"])
        y = float(box["y"])
        w = float(box["width"])
        h = float(box["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if x < 0 or y < 0 or x >= 1 or y >= 1 or w <= 0 or h <= 0:
        return None
    if x + w > 1:
        w = 1 - x
    if y + h > 1:
        h = 1 - y
    if w <= 0 or h <= 0:
        return None
    return {"x": x, "y": y, "width": w, "height": h}


def _open_page(data: bytes) -> fitz.Pixmap:
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        page = doc[0]
        zoom = DPI / 72
        return page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    finally:
        doc.close()


def _tile(pix: fitz.Pixmap, box: dict) -> tuple[int, int, int, int] | None:
    px = int(round(box["x"] * pix.width))
    py = int(round(box["y"] * pix.height))
    pw = max(1, int(round(box["width"] * pix.width)))
    ph = max(1, int(round(box["height"] * pix.height)))
    if px >= pix.width or py >= pix.height:
        return None
    pw = min(pw, pix.width - px)
    ph = min(ph, pix.height - py)
    if pw < 1 or ph < 1:
        return None
    return px, py, pw, ph


def _png_b64(pix: fitz.Pixmap, box: dict) -> str | None:
    tile = _tile(pix, box)
    if tile is None:
        return None
    crop = _crop(pix, *tile)
    return base64.b64encode(crop.tobytes("png")).decode("ascii")


def _image_name(view: str, callout: str) -> str:
    """View sentence, plus the printed radius text when the sheet has one."""
    callout = callout.strip()
    if not callout:
        return view
    return f"{view} {callout}"


def _images_from_extractions(pix: fitz.Pixmap, extractions: list[dict]) -> list[dict]:
    tiles = []
    for item in extractions:
        tile = _tile(pix, item["box"])
        if tile is None:
            raise AutoPlotRegionsError(422, "recorded views do not fit the page")
        tiles.append((_image_name(item["view"], item["callout"]), *tile))
    positions = _row_positions([(w, h) for _, _, _, w, h in tiles])
    images = []
    for (name, x, y, tile_w, tile_h), position in zip(tiles, positions):
        crop = _crop(pix, x, y, tile_w, tile_h)
        images.append({
            "name": name,
            "pngBase64": base64.b64encode(crop.tobytes("png")).decode("ascii"),
            "position": position,
        })
    return images


def _crop(pix: fitz.Pixmap, x: int, y: int, w: int, h: int) -> fitz.Pixmap:
    """Copy a rectangle out of a pixmap. Pixmap(src, clip) mis-dispatches on this PyMuPDF."""
    n = pix.n
    stride = pix.width * n
    src = pix.samples
    buf = bytearray(w * h * n)
    row_bytes = w * n
    for row in range(h):
        src_off = (y + row) * stride + x * n
        dst_off = row * row_bytes
        buf[dst_off:dst_off + row_bytes] = src[src_off:src_off + row_bytes]
    return fitz.Pixmap(fitz.csRGB, w, h, bytes(buf), pix.alpha)


def _row_positions(sizes: list[tuple[int, int]]) -> list[dict]:
    """Lay tiles in one row around the origin. Size is pixels times SCALE. Position is the center."""
    widths = [w * SCALE for w, _h in sizes]
    total = sum(widths) + GAP * (len(sizes) - 1)
    cursor = -total / 2
    positions = []
    for width in widths:
        positions.append({"x": cursor + width / 2, "y": 0, "z": 0})
        cursor += width + GAP
    return positions

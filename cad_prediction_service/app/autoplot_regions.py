"""AutoPlot image workflow. The route starts this; the split step is replaceable."""

import base64
import hashlib
from pathlib import Path

import fitz

#: Workspace copy of CUR1000650 primario 1. Do not commit the PDF.
PDF_PATH = Path(__file__).resolve().parents[1] / "local" / "input" / "CUR1000650-Model-primario-1.pdf"
EXPECTED_SHA256 = "9454070f939cf9002b23b8003930ed41a9a4ba5ce1fcbcc224d9505cce61766a"
DPI = 150
SCALE = 0.01
GAP = 0.5
# Unequal on purpose, so the four images are not v3's equal quadrants.
WIDTH_FRACTION = 0.30
HEIGHT_FRACTION = 0.62


class AutoPlotRegionsError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def run(text: str) -> dict:
    """Start the image workflow. `text` is required and not interpreted. Returns one command."""
    if not text or not text.strip():
        raise AutoPlotRegionsError(422, "text is required")
    images = split_local_pdf()
    return {
        "commands": [
            {
                "command": "scene.addReferenceImages",
                "params": {"images": images},
            }
        ]
    }


def split_local_pdf() -> list[dict]:
    """Rasterize the local PDF and return four unequal images with world positions."""
    if not PDF_PATH.is_file():
        raise AutoPlotRegionsError(404, "local PDF missing")
    data = PDF_PATH.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_SHA256:
        raise AutoPlotRegionsError(422, "local PDF hash mismatch")
    return _split_page(data)


def _split_page(data: bytes) -> list[dict]:
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        page = doc[0]
        zoom = DPI / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    finally:
        doc.close()

    width, height = pix.width, pix.height
    left_w = max(1, int(width * WIDTH_FRACTION))
    top_h = max(1, int(height * HEIGHT_FRACTION))
    if left_w >= width:
        left_w = width - 1
    if top_h >= height:
        top_h = height - 1
    right_w = width - left_w
    bot_h = height - top_h

    tiles = [
        ("top-left", 0, 0, left_w, top_h),
        ("top-right", left_w, 0, right_w, top_h),
        ("bottom-left", 0, top_h, left_w, bot_h),
        ("bottom-right", left_w, top_h, right_w, bot_h),
    ]
    images = []
    for name, x, y, tile_w, tile_h in tiles:
        crop = _crop(pix, x, y, tile_w, tile_h)
        images.append({
            "name": name,
            "pngBase64": base64.b64encode(crop.tobytes("png")).decode("ascii"),
            "position": _position(name, tile_w, tile_h),
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


def _position(name: str, tile_w: int, tile_h: int) -> dict:
    """Place one tile in a 2×2 around the origin. Size is pixels times SCALE."""
    half_w = tile_w * SCALE / 2
    half_h = tile_h * SCALE / 2
    gap = GAP / 2
    if name == "top-left":
        x, y = -(gap + half_w), gap + half_h
    elif name == "top-right":
        x, y = gap + half_w, gap + half_h
    elif name == "bottom-left":
        x, y = -(gap + half_w), -(gap + half_h)
    else:
        x, y = gap + half_w, -(gap + half_h)
    return {"x": x, "y": y, "z": 0}

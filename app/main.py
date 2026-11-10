from __future__ import annotations

import io
import os
import uuid
from typing import Any

import cv2
import easyocr
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "15"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
OCR_LANGUAGES = [x.strip() for x in os.getenv("OCR_LANGUAGES", "en,hi").split(",") if x.strip()]
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*").split(",")

app = FastAPI(
    title="PixelText AI API",
    description="OCR-assisted photo text editing prototype. Images are processed in memory.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

_reader = None


def get_reader():
    """Load EasyOCR only when OCR is first requested."""
    global _reader
    if _reader is None:
        try:
            _reader = easyocr.Reader(OCR_LANGUAGES, gpu=False, verbose=False)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"OCR model could not initialize: {exc}",
            ) from exc
    return _reader


async def read_image(upload: UploadFile) -> tuple[Image.Image, bytes]:
    if upload.content_type and not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file.")
    raw = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Image must be smaller than {MAX_UPLOAD_MB} MB.")
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")
    try:
        image = Image.open(io.BytesIO(raw))
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Unsupported or damaged image.") from exc
    if image.width * image.height > 30_000_000:
        raise HTTPException(status_code=413, detail="Image is too large (maximum 30 megapixels).")
    return image, raw


def pil_to_cv(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def cv_to_png_bytes(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode processed image.")
    return encoded.tobytes()


@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": "PixelText AI",
        "ocr_languages": OCR_LANGUAGES,
        "ocr_loaded": _reader is not None,
    }


@app.post("/api/ocr")
async def detect_text(file: UploadFile = File(...)):
    """Detect text and return polygon + rectangle coordinates in original image pixels."""
    image, _ = await read_image(file)
    reader = get_reader()
    results = reader.readtext(np.asarray(image), detail=1, paragraph=False)

    regions: list[dict[str, Any]] = []
    for polygon, text, confidence in results:
        points = [[int(round(p[0])), int(round(p[1]))] for p in polygon]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        if not text.strip():
            continue
        regions.append({
            "id": str(uuid.uuid4()),
            "text": text,
            "confidence": round(float(confidence), 4),
            "polygon": points,
            "box": {"x": x0, "y": y0, "width": max(1, x1 - x0), "height": max(1, y1 - y0)},
        })

    return {
        "width": image.width,
        "height": image.height,
        "count": len(regions),
        "regions": regions,
    }


@app.post("/api/inpaint")
async def inpaint_text(
    file: UploadFile = File(...),
    regions_json: str = Form(...),
    radius: int = Form(5),
):
    """
    Remove selected OCR regions with OpenCV inpainting.
    This is a classical inpainting baseline, not generative AI.
    """
    image, _ = await read_image(file)
    try:
        import json
        regions = json.loads(regions_json)
        if not isinstance(regions, list) or not regions:
            raise ValueError("regions must be a non-empty list")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid regions_json.") from exc

    radius = max(1, min(int(radius), 20))
    bgr = pil_to_cv(image)
    mask = np.zeros((image.height, image.width), dtype=np.uint8)

    for region in regions:
        polygon = region.get("polygon")
        if not isinstance(polygon, list) or len(polygon) < 3:
            box = region.get("box", {})
            x, y = int(box.get("x", 0)), int(box.get("y", 0))
            w, h = int(box.get("width", 0)), int(box.get("height", 0))
            polygon = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        pts = np.array(
            [[
                max(0, min(image.width - 1, int(p[0]))),
                max(0, min(image.height - 1, int(p[1]))),
            ] for p in polygon],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [pts], 255)

    # Slight dilation helps cover antialiased glyph edges without expanding too far.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    restored = cv2.inpaint(bgr, mask, radius, cv2.INPAINT_TELEA)
    return Response(
        content=cv_to_png_bytes(restored),
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )

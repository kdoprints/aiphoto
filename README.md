# PixelText AI

A GitHub-ready starter project for a mobile + desktop photo text editor.

## What this version does

- Upload JPG, PNG or WebP images (15 MB / 30 megapixel limits).
- Detect text with EasyOCR (`en` + `hi` by default).
- Return text, confidence, polygon and bounding-box coordinates.
- Select OCR regions in the browser.
- Remove a selected region using OpenCV Telea inpainting.
- Add editable text layers, change font/size/colour/alignment, drag them.
- Export a PNG at the uploaded image's original pixel dimensions.
- Images are processed in memory; this project does not save uploads to disk.

## Important limitations

This is an OCR-assisted MVP, **not a finished exact-font AI editor**.

1. OpenCV Telea inpainting is classical image restoration, not generative AI. It is best on relatively smooth backgrounds. Text over printed fabric, gradients, faces, detailed artwork or overlapping objects can leave visible artifacts.
2. OCR detects text and its position; it does not know the original font with certainty.
3. The replacement text is rendered as a new editable layer. The user may need to adjust its font, size, colour and position.
4. Hindi recognition quality depends on the EasyOCR model and image quality. Validate with your own sample images before production.
5. The first OCR request downloads model weights. Allow several minutes and sufficient disk/network access on first startup.
6. This prototype has no user accounts, payments, permanent storage, rate limiting, or production abuse protection.

## Run locally

Requires Python 3.10 or 3.11. OCR dependencies (PyTorch/EasyOCR) can be large.

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000

API health: http://localhost:8000/api/health

## Run with Docker

```bash
docker compose up --build
```

Then open http://localhost:8000

The first OCR request downloads model weights into the container environment. For a persistent deployment, mount a volume for the model cache and plan sufficient RAM/CPU. CPU OCR may be slow.

## API

### `POST /api/ocr`

Multipart form:
- `file`: image file

Returns:
```json
{
  "width": 1200,
  "height": 800,
  "count": 1,
  "regions": [
    {
      "id": "uuid",
      "text": "Example",
      "confidence": 0.98,
      "polygon": [[10, 10], [150, 10], [150, 50], [10, 50]],
      "box": {"x": 10, "y": 10, "width": 140, "height": 40}
    }
  ]
}
```

### `POST /api/inpaint`

Multipart form:
- `file`: original image
- `regions_json`: JSON array containing OCR region objects
- `radius`: optional integer 1–20 (default 5)

Returns a PNG.

## Deployment notes

- Deploy behind HTTPS.
- Set `MAX_UPLOAD_MB` to your preferred limit.
- Set `ALLOW_ORIGINS` to your exact frontend origin(s), not `*`, when frontend and API are hosted separately.
- Add authentication, per-user rate limits, request timeouts and queueing before opening a public service.
- Do not promise perfect font matching or perfect background restoration.
- If hosting on a low-memory server, run OCR in a worker process rather than blocking the API process.

## Suggested next upgrade

Replace the OpenCV inpainting endpoint with a separately hosted LaMa/IOPaint service, preserving the current `/api/inpaint` contract. Then add font classification and a text-style adjustment UI. Test on at least 30 real photos, including Hindi/English text, fabric patterns, logos and business cards.

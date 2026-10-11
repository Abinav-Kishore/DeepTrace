"""FastAPI inference service.

POST /predict accepts a video upload, validates size/type, extracts frames,
runs the temporal model, and returns verdict + probabilities. Errors are
handled gracefully: 400 for corrupt/empty/no-decodable-frame videos and
oversized files, 415 for non-video media types, 503 when the checkpoint
is missing.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, Query

from deeptrace.config import FRAMES_PER_VIDEO

app = FastAPI(title="DeepTrace API", version="0.1.0")

MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB cap (protects the free-tier host)
ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
ALLOWED_NEGATIVE_TIP = 0.0


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "deeptrace"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Query("temporal", pattern="^(temporal|baseline)$"),
) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES and (file.content_type or "").split("/")[0] != "video":
        raise HTTPException(
            status_code=415, detail=f"Unsupported file type: {suffix or file.content_type}"
        )

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File too large (>100MB).")

    with tempfile.NamedTemporaryFile(suffix=suffix or ".mp4", delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    from deeptrace.inference import predict_video, frames_from_video

    try:
        # Fail fast on something cv2 can't decode before running the model.
        frames_from_video(tmp_path)  # raises ValueError on corrupt/empty video
        result = predict_video(tmp_path, kind=model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Video could not be processed: {e}")
    except RuntimeError as e:
        # Checkpoint missing etc.
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:  # noqa: BLE001 - return a clean error to the client
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    return result

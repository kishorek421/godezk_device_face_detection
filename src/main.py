"""
GoDezk Person & Face Detection Service
Fast CPU/GPU-ready YOLOv8 Person Detector for Cash Counters & Station Guards.
Accepts raw JPEG/PNG or JSON {image / frame_base64, camera_id?, frame_id?}
Outputs both 'detections' and 'results'/'faces' for full compatibility with PerceptionGate.
"""
import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO

# ── Config ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = os.environ.get("MODEL_PATH", str(BASE_DIR / "models" / "yolov8n.pt"))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.45"))
PORT = int(os.environ.get("PORT", "8002"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("person_detection")

# ── App & Model ────────────────────────────────────────────────────
app = FastAPI(title="GoDezk Person & Face Detection Service", version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    safe_errors = []
    for err in exc.errors():
        err_copy = dict(err)
        if isinstance(err_copy.get("input"), (bytes, bytearray)):
            err_copy["input"] = f"<binary data: {len(err_copy['input'])} bytes>"
        safe_errors.append(err_copy)
    return JSONResponse(status_code=422, content={"detail": safe_errors})


_model: Optional[YOLO] = None


@app.on_event("startup")
async def startup_event():
    global _model
    logger.info(f"Loading YOLO person detection model from: {MODEL_PATH}")
    try:
        _model = YOLO(MODEL_PATH)
        logger.info("✅ YOLO model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load YOLO model: {e}")
        raise e


# ── Schemas ────────────────────────────────────────────────────────
class DetectionItem(BaseModel):
    class_name: str
    confidence: float
    bbox: List[int]             # [x1, y1, x2, y2] pixel coords
    normalized_bbox: List[float] # [x1, y1, x2, y2] normalized 0-1


class DetectResponse(BaseModel):
    success: bool
    frame_id: Optional[str] = None
    camera_id: Optional[str] = None
    person_count: int
    persons_detected: int
    detections: List[Dict[str, Any]]
    results: Optional[List[Dict[str, Any]]] = None
    faces: Optional[List[Dict[str, Any]]] = None
    processing_time_ms: float


class DetectRequest(BaseModel):
    image: Optional[str] = None
    frame_base64: Optional[str] = None
    data: Optional[str] = None
    camera_id: Optional[str] = None
    frame_id: Optional[str] = None
    confidence_threshold: Optional[float] = None


# ── Helpers ────────────────────────────────────────────────────────
def _run_person_inference(frame: np.ndarray, conf_threshold: float) -> List[Dict[str, Any]]:
    global _model
    if _model is None:
        raise HTTPException(status_code=503, detail="YOLO model not loaded")

    h, w = frame.shape[:2]
    results = _model(frame, conf=conf_threshold, verbose=False)
    detections = []

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0].item())
            cls_name = _model.names.get(cls_id, "")
            
            # Filter strictly for person (COCO class 0)
            if cls_name.lower() != "person" and cls_id != 0:
                continue

            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(v) for v in xyxy]

            # Safe clamp
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))

            norm_box = [
                round(x1 / w, 4) if w > 0 else 0.0,
                round(y1 / h, 4) if h > 0 else 0.0,
                round(x2 / w, 4) if w > 0 else 0.0,
                round(y2 / h, 4) if h > 0 else 0.0,
            ]

            detections.append({
                "class": "person",
                "label": "person",
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
                "normalized_bbox": norm_box,
            })

    return detections


# ── Routes ─────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "person_detection",
        "model": "yolov8n",
        "port": PORT,
    }


@app.post("/detect", response_model=DetectResponse)
@app.post("/predict", response_model=DetectResponse)
@app.post("/recognize", response_model=DetectResponse)
async def detect(request: Request):
    start = time.time()
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty request body")

    frame = None
    frame_id = request.headers.get("x-frame-id") or request.query_params.get("frame_id")
    camera_id = request.headers.get("x-camera-id") or request.query_params.get("camera_id")
    conf = CONFIDENCE_THRESHOLD

    # 1. Raw binary image (JPEG starts with 0xFF 0xD8, PNG starts with 0x89 PNG)
    if body.startswith(bytes([0xFF, 0xD8])) or body.startswith(bytes([0x89, 0x50, 0x4E, 0x47])) or request.headers.get("content-type", "").startswith("image/"):
        np_arr = np.frombuffer(body, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    else:
        # 2. Try JSON payload
        try:
            data = json.loads(body.decode("utf-8", errors="replace"))
            raw_b64 = data.get("image") or data.get("frame_base64") or data.get("data")
            camera_id = data.get("camera_id", camera_id)
            frame_id = data.get("frame_id", frame_id)
            if data.get("confidence_threshold"):
                conf = float(data["confidence_threshold"])
            if raw_b64:
                if "," in raw_b64:
                    raw_b64 = raw_b64.split(",", 1)[1]
                img_bytes = base64.b64decode(raw_b64)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception:
            # Fallback: maybe raw image bytes without content-type header
            np_arr = np.frombuffer(body, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image from request body")

    detections = _run_person_inference(frame, conf)
    duration_ms = round((time.time() - start) * 1000, 2)

    # Format into face/person items compatible with PerceptionGate _decideFace
    results = [
        {
            "matched": False,
            "person_id": f"person_{i+1}",
            "track_id": str(i+1),
            "unknown_track_id": f"UNK-{i+1}",
            "name": d.get("label", "person"),
            "confidence": d.get("confidence", 0.0),
            "det_score": d.get("confidence", 0.0),
            "similarity": d.get("confidence", 0.0),
            "bbox": d.get("bbox"),
            "normalized_bbox": d.get("normalized_bbox"),
        }
        for i, d in enumerate(detections)
    ]

    return DetectResponse(
        success=True,
        frame_id=frame_id,
        camera_id=camera_id,
        person_count=len(detections),
        persons_detected=len(detections),
        detections=detections,
        results=results,
        faces=results,
        processing_time_ms=duration_ms,
    )


if __name__ == "__main__":
    import sys
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

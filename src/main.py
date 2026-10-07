"""
GoDezk Person Detection Service (formerly face_detection)
Fast CPU/GPU-ready YOLOv8 Person Detector for Cash Counters & Station Guards.
Accepts raw JPEG/PNG or JSON {image / frame_base64, camera_id?, frame_id?}
"""
import base64
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = os.environ.get("MODEL_PATH", str(BASE_DIR / "models" / "yolov8n.pt"))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.35"))
PORT = int(os.environ.get("PORT", "8002"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("person_detection")

# ── App & Model ───────────────────────────────────────────────
app = FastAPI(title="GoDezk Person Detection Service", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


# ── Schemas ───────────────────────────────────────────────────
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
    processing_time_ms: float


class DetectRequest(BaseModel):
    image: Optional[str] = None
    frame_base64: Optional[str] = None
    data: Optional[str] = None
    camera_id: Optional[str] = None
    frame_id: Optional[str] = None
    confidence_threshold: Optional[float] = None


# ── Helpers ───────────────────────────────────────────────────
def _decode_frame(req_data: DetectRequest) -> np.ndarray:
    raw_b64 = req_data.image or req_data.frame_base64 or req_data.data
    if not raw_b64:
        raise HTTPException(status_code=400, detail="Missing image base64 data")

    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(raw_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("cv2.imdecode returned None")
        return frame
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image encoding: {e}")


def _run_person_inference(frame: np.ndarray, conf_threshold: float) -> List[Dict[str, Any]]:
    global _model
    if _model is None:
        raise HTTPException(status_code=503, detail="YOLO model not loaded")

    h, w = frame.shape[:2]
    # Run YOLO with verbose=False for fast execution
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


# ── Routes ────────────────────────────────────────────────────
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
def detect(req: DetectRequest):
    start = time.time()
    frame = _decode_frame(req)
    conf = req.confidence_threshold or CONFIDENCE_THRESHOLD

    detections = _run_person_inference(frame, conf)
    duration_ms = round((time.time() - start) * 1000, 2)

    return DetectResponse(
        success=True,
        frame_id=req.frame_id,
        camera_id=req.camera_id,
        person_count=len(detections),
        persons_detected=len(detections),
        detections=detections,
        processing_time_ms=duration_ms,
    )


# Compatibility endpoint for legacy callers expecting /recognize
@app.post("/recognize")
def recognize_compat(req: DetectRequest):
    return detect(req)


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=PORT, log_level="info")

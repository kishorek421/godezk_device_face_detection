"""
GoDezk Person & Facial Expression Recognition Service
Fast CPU/GPU-ready YOLOv8 Person Detector + OpenCV YuNet Face Localizer + DDAMFN Emotion Classifier.
Accepts raw JPEG/PNG binary or JSON {image / frame_base64 / data, camera_id?, frame_id?, confidence_threshold?, detect_expression?}
Outputs 'detections', 'results', and 'faces' for full compatibility with PerceptionGate and Counter Agitation Workflows.
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
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO

# ── Config ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

YOLO_MODEL_PATH = os.environ.get("MODEL_PATH", str(MODELS_DIR / "yolov8n.pt"))
MODEL_PATH = YOLO_MODEL_PATH
YUNET_MODEL_PATH = os.environ.get("YUNET_MODEL_PATH", str(MODELS_DIR / "face_detection_yunet_2023mar.onnx"))
FER_MODEL_PATH = os.environ.get("FER_MODEL_PATH", str(MODELS_DIR / "affecnet7.script"))

CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.45"))
FACE_CONFIDENCE_THRESHOLD = float(os.environ.get("FACE_CONFIDENCE_THRESHOLD", "0.60"))
ANGER_THRESHOLD = float(os.environ.get("ANGER_THRESHOLD", "0.45"))
ENABLE_FER_DEFAULT = os.environ.get("ENABLE_FER", "true").lower() in ("true", "1", "yes")
PORT = int(os.environ.get("PORT", "8002"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("person_fer_detection")

# ── AffectNet-7 Emotion Labels ─────────────────────────────────────
EMOTION_LABELS = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "anger"]

# ── App ────────────────────────────────────────────────────────────
app = FastAPI(title="GoDezk Person & Facial Expression Service", version="2.2.0")
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


# ── Global Model Singletons ────────────────────────────────────────
_yolo_model: Optional[YOLO] = None
_face_detector: Optional[cv2.FaceDetectorYN] = None
_fer_model: Optional[torch.jit.ScriptModule] = None


@app.on_event("startup")
async def startup_event():
    global _yolo_model, _face_detector, _fer_model
    logger.info("Initializing Person & Expression Recognition pipeline...")

    # 1. Load YOLO Person Detector
    logger.info(f"Loading YOLO person detection model from: {YOLO_MODEL_PATH}")
    try:
        _yolo_model = YOLO(YOLO_MODEL_PATH)
        logger.info("✅ YOLO model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load YOLO model: {e}")
        raise e

    # 2. Load YuNet Face Detector
    if os.path.exists(YUNET_MODEL_PATH):
        try:
            logger.info(f"Loading YuNet face detector from: {YUNET_MODEL_PATH}")
            _face_detector = cv2.FaceDetectorYN.create(
                model=YUNET_MODEL_PATH,
                config="",
                input_size=(320, 320),
                score_threshold=FACE_CONFIDENCE_THRESHOLD,
                nms_threshold=0.3,
                top_k=5000,
            )
            logger.info("✅ YuNet face detector loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load YuNet face detector: {e}")
            _face_detector = None
    else:
        logger.warning(f"⚠️ YuNet model file not found at {YUNET_MODEL_PATH}")

    # 3. Load DDAMFN Emotion Classifier
    if os.path.exists(FER_MODEL_PATH):
        try:
            logger.info(f"Loading DDAMFN facial expression model from: {FER_MODEL_PATH}")
            _fer_model = torch.jit.load(FER_MODEL_PATH, map_location="cpu")
            _fer_model.eval()
            logger.info("✅ DDAMFN expression model loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load DDAMFN emotion model: {e}")
            _fer_model = None
    else:
        logger.warning(f"⚠️ DDAMFN emotion model not found at {FER_MODEL_PATH}")


# ── Schemas ────────────────────────────────────────────────────────
class EmotionDetail(BaseModel):
    label: str
    confidence: float
    anger_score: float
    is_anger: bool
    scores: Dict[str, float]


class FaceDetail(BaseModel):
    detected: bool
    bbox: Optional[List[int]] = None
    normalized_bbox: Optional[List[float]] = None
    confidence: Optional[float] = None
    emotion: Optional[EmotionDetail] = None


class PersonDetection(BaseModel):
    class_name: str
    label: str = "person"
    confidence: float
    bbox: List[int]
    normalized_bbox: List[float]
    face: Optional[FaceDetail] = None


class DetectResponse(BaseModel):
    success: bool
    frame_id: Optional[str] = None
    camera_id: Optional[str] = None
    person_count: int
    persons_detected: int
    face_count: int = 0
    anger_detected: bool = False
    max_anger_confidence: float = 0.0
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
    detect_expression: Optional[bool] = None


# ── Pipeline Helpers ───────────────────────────────────────────────
def _detect_faces_yunet(person_crop: np.ndarray, offset_x: int, offset_y: int) -> List[Dict[str, Any]]:
    global _face_detector
    if _face_detector is None:
        return []

    ch, cw = person_crop.shape[:2]
    if ch < 20 or cw < 20:
        return []

    _face_detector.setInputSize((cw, ch))
    _, faces = _face_detector.detect(person_crop)

    if faces is None or len(faces) == 0:
        return []

    results = []
    for face in faces:
        fx, fy, fw, fh = face[0:4].astype(int)
        conf = float(face[-1])
        if conf < FACE_CONFIDENCE_THRESHOLD:
            continue

        x1 = max(0, offset_x + fx)
        y1 = max(0, offset_y + fy)
        x2 = offset_x + fx + fw
        y2 = offset_y + fy + fh

        results.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "confidence": round(conf, 3),
            "crop": person_crop[max(0, fy):min(ch, fy + fh), max(0, fx):min(cw, fx + fw)],
        })

    return results


def _classify_emotion(face_crop: np.ndarray) -> Optional[Dict[str, Any]]:
    global _fer_model
    if _fer_model is None or face_crop is None or face_crop.size == 0:
        return None

    try:
        resized = cv2.resize(face_crop, (112, 112))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        tensor = tensor / 255.0
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            logits = _fer_model(tensor)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        emotion_scores = {EMOTION_LABELS[i]: round(float(probs[i]), 3) for i in range(len(EMOTION_LABELS))}
        dominant_idx = int(np.argmax(probs))
        dominant_label = EMOTION_LABELS[dominant_idx]
        dominant_conf = round(float(probs[dominant_idx]), 3)

        anger_score = emotion_scores.get("anger", 0.0)
        disgust_score = emotion_scores.get("disgust", 0.0)
        is_anger = (anger_score >= ANGER_THRESHOLD) or (dominant_label == "anger") or (disgust_score >= 0.55 and dominant_label == "disgust")

        return {
            "expression": dominant_label,
            "expression_confidence": dominant_conf,
            "emotions": emotion_scores,
            "anger_score": anger_score,
            "is_anger": is_anger,
        }
    except Exception as e:
        logger.warning(f"Expression inference failed: {e}")
        return None


def _run_pipeline(
    frame: np.ndarray,
    conf_threshold: float,
    detect_expression: bool = True
) -> tuple[List[Dict[str, Any]], bool, float, int]:
    global _yolo_model, _face_detector
    if _yolo_model is None:
        raise HTTPException(status_code=503, detail="YOLO model not loaded")

    h, w = frame.shape[:2]
    results = _yolo_model(frame, conf=conf_threshold, verbose=False)
    detections: List[Dict[str, Any]] = []

    anger_detected = False
    max_anger_conf = 0.0
    face_count = 0

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0].item())
            cls_name = _yolo_model.names.get(cls_id, "")

            if cls_name.lower() != "person" and cls_id != 0:
                continue

            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(v) for v in xyxy]

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

            det_item: Dict[str, Any] = {
                "class": "person",
                "label": "person",
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
                "normalized_bbox": norm_box,
                "face": None,
            }

            if detect_expression and _face_detector is not None:
                pw = x2 - x1
                ph = y2 - y1
                face_search_h = int(ph * 0.70)
                upper_crop = frame[y1:y1 + face_search_h, x1:x2]

                if upper_crop.size > 0:
                    faces = _detect_faces_yunet(upper_crop, offset_x=x1, offset_y=y1)
                    if faces:
                        best_face = max(faces, key=lambda f: f["confidence"])
                        face_count += 1
                        fx1, fy1, fx2, fy2 = best_face["bbox"]
                        fx1, fy1 = max(0, fx1), max(0, fy1)
                        fx2, fy2 = min(w, fx2), min(h, fy2)

                        norm_face_box = [
                            round(fx1 / w, 4) if w > 0 else 0.0,
                            round(fy1 / h, 4) if h > 0 else 0.0,
                            round(fx2 / w, 4) if w > 0 else 0.0,
                            round(fy2 / h, 4) if h > 0 else 0.0,
                        ]

                        face_info: Dict[str, Any] = {
                            "detected": True,
                            "bbox": [fx1, fy1, fx2, fy2],
                            "normalized_bbox": norm_face_box,
                            "confidence": best_face["confidence"],
                        }

                        fer_res = _classify_emotion(best_face["crop"])
                        if fer_res:
                            face_info.update(fer_res)
                            if fer_res["is_anger"]:
                                anger_detected = True
                                max_anger_conf = max(max_anger_conf, fer_res["anger_score"])

                        det_item["face"] = face_info

            detections.append(det_item)

    return detections, anger_detected, round(max_anger_conf, 3), face_count


# ── Routes ─────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "person_and_facial_expression_recognition",
        "yolo_loaded": _yolo_model is not None,
        "face_detector_loaded": _face_detector is not None,
        "fer_model_loaded": _fer_model is not None,
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
    detect_expression = ENABLE_FER_DEFAULT

    # 1. Raw binary image stream (JPEG starts with 0xFF 0xD8, PNG starts with 0x89 PNG)
    if body.startswith(bytes([0xFF, 0xD8])) or body.startswith(bytes([0x89, 0x50, 0x4E, 0x47])) or request.headers.get("content-type", "").startswith("image/"):
        np_arr = np.frombuffer(body, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    else:
        # 2. JSON payload
        try:
            data = json.loads(body.decode("utf-8", errors="replace"))
            raw_b64 = data.get("image") or data.get("frame_base64") or data.get("data")
            camera_id = data.get("camera_id", camera_id)
            frame_id = data.get("frame_id", frame_id)
            if data.get("confidence_threshold") is not None:
                conf = float(data["confidence_threshold"])
            if data.get("detect_expression") is not None:
                detect_expression = bool(data["detect_expression"])

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

    detections, anger_detected, max_anger_conf, face_count = _run_pipeline(frame, conf, detect_expression)
    duration_ms = round((time.time() - start) * 1000, 2)

    # Format into face/person items compatible with PerceptionGate _decideFace
    results = []
    for i, d in enumerate(detections):
        f = d.get("face")
        item = {
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
            "expression": f.get("expression") if f else "unknown",
            "expression_confidence": f.get("expression_confidence") if f else 0.0,
            "is_anger": f.get("is_anger", False) if f else False,
            "anger_score": f.get("anger_score", 0.0) if f else 0.0,
            "face_bbox": f.get("bbox") if f else None,
        }
        results.append(item)

    return DetectResponse(
        success=True,
        frame_id=frame_id,
        camera_id=camera_id,
        person_count=len(detections),
        persons_detected=len(detections),
        face_count=face_count,
        anger_detected=anger_detected,
        max_anger_confidence=max_anger_conf,
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

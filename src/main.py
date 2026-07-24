"""
GoDezk Face Detection Service
RetinaFace-based face detector — standalone HTTP API, workflow node
Accepts raw JPEG/PNG or JSON {frame_base64, camera_id?, frame_id?}
"""
import asyncio
import base64
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import concurrent.futures
import threading

import cv2
import numpy as np
import onnxruntime as ort
import supervision as sv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from insightface.model_zoo.attribute import Attribute
from insightface.model_zoo.retinaface import RetinaFace
from insightface.utils import face_align
from pydantic import BaseModel

try:
    from prometheus_client import Counter, Histogram, generate_latest

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(BASE_DIR / "models" / "det_10g.onnx"),
)
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.70"))
DET_SIZE = int(os.environ.get("DET_SIZE", "640"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
PORT = int(os.environ.get("PORT", "8008"))

GENDERAGE_MODEL_PATH = os.environ.get(
    "GENDERAGE_MODEL_PATH",
    str(BASE_DIR / "models" / "genderage.onnx"),
)
TRACK_ACTIVATION_THRESHOLD = float(os.environ.get("TRACK_ACTIVATION_THRESHOLD", "0.35"))
LOST_TRACK_BUFFER = int(os.environ.get("LOST_TRACK_BUFFER", "30"))
TRACK_MATCHING_THRESHOLD = float(os.environ.get("TRACK_MATCHING_THRESHOLD", "0.80"))
GENDER_CONFIDENCE_THRESHOLD = float(os.environ.get("GENDER_CONFIDENCE_THRESHOLD", "0.80"))
GENDER_REFRESH_SECONDS = float(os.environ.get("GENDER_REFRESH_SECONDS", "0.30"))
GENDER_HISTORY_SIZE = int(os.environ.get("GENDER_HISTORY_SIZE", "7"))
GENDER_LOCK_THRESHOLD = float(os.environ.get("GENDER_LOCK_THRESHOLD", "0.75"))
GENDER_LOCK_MIN_HISTORY = int(os.environ.get("GENDER_LOCK_MIN_HISTORY", "3"))
TRACK_STATE_TTL_SECONDS = float(os.environ.get("TRACK_STATE_TTL_SECONDS", "5.0"))
MIN_FACE_SIZE = int(os.environ.get("MIN_FACE_SIZE", "50"))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Metrics ───────────────────────────────────────────────────
if _PROMETHEUS_AVAILABLE:
    detection_requests = Counter("face_detection_requests_total", "Total face detection requests")
    face_detections = Counter("face_detections_total", "Total faces detected")
    detection_latency = Histogram("face_detection_latency_seconds", "Face detection latency")
else:
    class _DummyMetric:
        def inc(self, amount: int = 1) -> None:
            pass

        def observe(self, amount: float) -> None:
            pass

    detection_requests = _DummyMetric()
    face_detections = _DummyMetric()
    detection_latency = _DummyMetric()

# ── App ───────────────────────────────────────────────────────
_face_detector = None
_gender_model = None
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
_track_lock = threading.Lock()
_trackers: dict = {}


class Face(BaseModel):
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    det_score: float
    track_id: int
    gender: Optional[str] = None
    face_image_b64: Optional[str] = None


class DetectRequest(BaseModel):
    frame_base64: str
    camera_id: Optional[str] = None
    frame_id: Optional[str] = None


class DetectResponse(BaseModel):
    success: bool
    frame_id: Optional[str] = None
    camera_id: Optional[str] = None
    face_count: int
    count: int
    faces: List[Face]
    results: List[Face]
    has_face: bool
    processing_time_ms: float


class RecognizeRequest(BaseModel):
    image_b64: str
    camera_id: Optional[str] = None
    frame_id: Optional[str] = None
    org_id: Optional[str] = None
    track_id: Optional[str] = None
    event_key: Optional[str] = None


class RecognizedFace(BaseModel):
    matched: bool = False
    status: str = "unknown"
    person_id: Optional[str] = None
    employee_id: Optional[str] = None
    name: Optional[str] = None
    similarity: float = 0.0
    det_score: float
    bbox: List[int]
    face_image_b64: Optional[str] = None
    gender: Optional[str] = None
    track_id: int


class RecognizeResponse(BaseModel):
    success: bool
    results: List[RecognizedFace]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: float


# ── Helpers ───────────────────────────────────────────────────

def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    return img


class FaceTracker:
    """Per-camera ByteTrack tracker with in-memory gender metadata."""

    def __init__(self) -> None:
        self.byte = sv.ByteTrack(
            track_activation_threshold=TRACK_ACTIVATION_THRESHOLD,
            lost_track_buffer=LOST_TRACK_BUFFER,
            minimum_matching_threshold=TRACK_MATCHING_THRESHOLD,
        )
        self.tracks: dict = {}
        self.lock = threading.Lock()

    def update(self, bboxes, now: float) -> list:
        with self.lock:
            if bboxes is None or bboxes.shape[0] == 0:
                dets = sv.Detections.empty()
            else:
                dets = sv.Detections(
                    xyxy=bboxes[:, :4].astype(np.float32),
                    confidence=bboxes[:, 4].astype(np.float32),
                    class_id=np.zeros(bboxes.shape[0], dtype=int),
                )
            tracked = self.byte.update_with_detections(dets)
            active_ids = set()
            out = []
            if tracked.tracker_id is not None:
                for bbox, track_id, conf in zip(tracked.xyxy, tracked.tracker_id, tracked.confidence):
                    tid = int(track_id)
                    active_ids.add(tid)
                    meta = self.tracks.setdefault(
                        tid,
                        {
                            "first_seen": now,
                            "last_gender_attempt": 0.0,
                            "gender_history": deque(maxlen=GENDER_HISTORY_SIZE),
                            "locked_gender": None,
                            "locked_confidence": 0.0,
                            "emitted": False,
                        },
                    )
                    meta["last_seen"] = now
                    out.append((tid, bbox.tolist(), float(conf), meta))
            for tid, meta in list(self.tracks.items()):
                if tid not in active_ids and now - meta.get("last_seen", now) > TRACK_STATE_TTL_SECONDS:
                    del self.tracks[tid]
            return out


def _tracker_for(camera_id: Optional[str]) -> FaceTracker:
    cid = camera_id or "__default__"
    with _track_lock:
        if cid not in _trackers:
            _trackers[cid] = FaceTracker()
        return _trackers[cid]


def _crop_face_b64(img_bgr: np.ndarray, bbox: list) -> Optional[str]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img_bgr[y1:y2, x1:x2]
    ok, buf = cv2.imencode(".jpg", crop)
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _estimate_gender(img_bgr: np.ndarray, bbox: list) -> tuple[Optional[str], float]:
    x1, y1, x2, y2 = bbox
    width, height = float(x2 - x1), float(y2 - y1)
    center = (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)
    scale = _gender_model.input_size[0] / (max(width, height) * 1.5)
    aimg, _ = face_align.transform(img_bgr, center, _gender_model.input_size[0], scale, 0)
    if aimg is None or aimg.size == 0:
        return None, 0.0
    input_size = tuple(aimg.shape[0:2][::-1])
    blob = cv2.dnn.blobFromImage(
        aimg,
        1.0 / _gender_model.input_std,
        input_size,
        (_gender_model.input_mean,) * 3,
        swapRB=True,
    )
    raw = _gender_model.session.run(
        _gender_model.output_names, {_gender_model.input_name: blob}
    )[0]
    if raw.ndim != 2 or raw.shape[0] != 1 or raw.shape[1] < 2:
        return None, 0.0
    pred = raw[0, :2]
    probs = np.exp(pred - np.max(pred))
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return None, 0.0
    probs /= total
    gender_val = int(np.argmax(probs))
    confidence = float(probs[gender_val])
    gender = "Male" if gender_val == 1 else "Female"
    return gender, confidence


def _get_track_gender(
    img_bgr: np.ndarray,
    bbox: list,
    meta: dict,
    now: float,
) -> Optional[str]:
    if meta.get("emitted"):
        return None
    if meta.get("locked_gender"):
        meta["emitted"] = True
        return meta["locked_gender"]
    if now - meta.get("last_gender_attempt", 0.0) < GENDER_REFRESH_SECONDS:
        return None
    meta["last_gender_attempt"] = now
    gender, confidence = _estimate_gender(img_bgr, bbox)
    if gender is None or confidence < GENDER_CONFIDENCE_THRESHOLD:
        logger.debug(f"[gender] low confidence or failed: {confidence:.2f}")
        return None
    history = meta["gender_history"]
    history.append((gender, confidence))
    logger.debug(f"[gender] added prediction: {gender} ({confidence:.2f}), history_len={len(history)}")
    if len(history) < GENDER_LOCK_MIN_HISTORY:
        return None
    weights = {"Male": 0.0, "Female": 0.0}
    for label, score in history:
        weights[label] += score
    total_weight = weights["Male"] + weights["Female"]
    if total_weight <= 0.0:
        return None
    locked_gender = max(weights, key=weights.get)
    stability = weights[locked_gender] / total_weight
    logger.debug(f"[gender] stability: {stability:.2f} for {locked_gender}")
    if stability < GENDER_LOCK_THRESHOLD:
        return None
    winning_scores = [score for label, score in history if label == locked_gender]
    meta["locked_gender"] = locked_gender
    meta["locked_confidence"] = sum(winning_scores) / len(winning_scores)
    meta["emitted"] = True
    return locked_gender


def _load_model() -> None:
    global _face_detector, _gender_model
    model_file = Path(MODEL_PATH)
    if not model_file.is_file():
        raise FileNotFoundError(f"Face detection model not found: {MODEL_PATH}")

    logger.info("Loading face detection model from %s", MODEL_PATH)
    session = ort.InferenceSession(
        str(model_file),
        providers=["CPUExecutionProvider"],
    )
    detector = RetinaFace(model_file=str(model_file), session=session)
    detector.prepare(
        ctx_id=-1,
        input_size=(DET_SIZE, DET_SIZE),
        det_thresh=CONFIDENCE_THRESHOLD,
    )
    _face_detector = detector

    gender_file = Path(GENDERAGE_MODEL_PATH)
    if not gender_file.is_file():
        raise FileNotFoundError(f"Gender model not found: {GENDERAGE_MODEL_PATH}")
    logger.info("Loading gender model from %s", GENDERAGE_MODEL_PATH)
    gender_session = ort.InferenceSession(
        str(gender_file),
        providers=["CPUExecutionProvider"],
    )
    _gender_model = Attribute(model_file=str(gender_file), session=gender_session)
    _gender_model.prepare(ctx_id=-1)

    logger.info(
        "Face Detection Service ready (input_size=%s, det_thresh=%s)",
        DET_SIZE,
        CONFIDENCE_THRESHOLD,
    )


def _process_frame(img_bgr: np.ndarray, camera_id: Optional[str]) -> List[dict]:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    bboxes, _ = _face_detector.detect(rgb, max_num=0, metric="default")
    logger.info(f"[_process_frame] camera_id={camera_id}, raw_detections={len(bboxes) if bboxes is not None else 0}")

    tracker = _tracker_for(camera_id)
    now = time.time()
    tracked = tracker.update(bboxes, now)
    logger.info(f"[_process_frame] camera_id={camera_id}, tracked_faces={len(tracked)}")

    faces = []
    for tid, bbox, conf, meta in tracked:
        x1, y1, x2, y2 = bbox
        width, height = x2 - x1, y2 - y1
        logger.info(f"[track_id={tid}] bbox={bbox}, size={width}x{height}, conf={conf:.2f}")
        if min(width, height) < MIN_FACE_SIZE:
            logger.info(f"[track_id={tid}] skipped: face too small ({width}x{height})")
            continue
        if conf < CONFIDENCE_THRESHOLD:
            logger.info(f"[track_id={tid}] skipped: confidence too low ({conf:.2f})")
            continue

        gender = _get_track_gender(img_bgr, bbox, meta, now)
        if gender is None:
            logger.info(f"[track_id={tid}] gender not locked yet (history_len={len(meta['gender_history'])}, emitted={meta.get('emitted')})")
            continue
        logger.info(f"[track_id={tid}] gender locked: {gender} (history_len={len(meta['gender_history'])})")
        faces.append(
            {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(conf),
                "det_score": float(conf),
                "track_id": tid,
                "gender": gender,
                "face_image_b64": None,
            }
        )
    return faces


@asynccontextmanager
async def lifespan(application: FastAPI):
    global _face_detector
    logger.info("Loading face detection model...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_thread_pool, _load_model)
    logger.info("Face Detection Service ready")
    yield
    _thread_pool.shutdown(wait=True)


app = FastAPI(title="GoDezk Face Detection Service", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if _face_detector is not None else "starting",
        model_loaded=_face_detector is not None,
        timestamp=time.time(),
    )


@app.get("/alive")
async def alive():
    if _face_detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "alive"}


@app.get("/ready")
async def ready():
    if _face_detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


if _PROMETHEUS_AVAILABLE:

    @app.get("/metrics")
    async def metrics():
        return Response(content=generate_latest(), media_type="text/plain")

else:

    @app.get("/metrics")
    async def metrics():
        return {"status": "metrics disabled", "reason": "prometheus-client not installed"}


@app.post("/detect", response_model=DetectResponse)
async def detect(request: Request):
    """Detect faces in an image.

    Accepts binary JPEG/PNG (Content-Type: image/*) or JSON
    {frame_base64, camera_id?, frame_id?}.
    """
    detection_requests.inc()
    start = time.time()

    try:
        content_type = request.headers.get("content-type", "application/json")

        if content_type.startswith("image/"):
            raw = await request.body()
            img = _decode_image(raw)
            camera_id = request.headers.get("x-camera-id")
            frame_id = request.headers.get("x-frame-id", str(time.time()))
        else:
            body = await request.json()
            req = DetectRequest(**body)
            img = _decode_image(base64.b64decode(req.frame_base64))
            camera_id = req.camera_id
            frame_id = req.frame_id or str(time.time())

        loop = asyncio.get_event_loop()
        faces = await loop.run_in_executor(_thread_pool, _process_frame, img, camera_id)

        elapsed = time.time() - start
        detection_latency.observe(elapsed)
        face_detections.inc(len(faces))

        logger.info(
            "frame=%s faces=%s max_conf=%.3f latency=%.1fms",
            frame_id,
            len(faces),
            max((f["confidence"] for f in faces), default=0.0),
            elapsed * 1000,
        )

        return DetectResponse(
            success=True,
            frame_id=frame_id,
            camera_id=camera_id,
            face_count=len(faces),
            count=len(faces),
            faces=faces,
            results=faces,
            has_face=len(faces) > 0,
            processing_time_ms=elapsed * 1000,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Face detection failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recognize", response_model=RecognizeResponse)
async def recognize(request: Request):
    """Backdoor-compatible endpoint that returns face tracking + gender.

    Accepts binary JPEG/PNG (Content-Type: image/*) with x-* headers
    or JSON {image_b64, camera_id?, frame_id?, org_id?, event_key?}.
    """
    content_type = request.headers.get("content-type", "application/json")
    if content_type.startswith("image/"):
        raw = await request.body()
        img = _decode_image(raw)
        camera_id = request.headers.get("x-camera-id")
        frame_id = request.headers.get("x-frame-id", str(time.time()))
    else:
        body = await request.json()
        req = RecognizeRequest(**body)
        img = _decode_image(base64.b64decode(req.image_b64))
        camera_id = req.camera_id
        frame_id = req.frame_id or str(time.time())

    loop = asyncio.get_event_loop()
    logger.info(f"[/recognize] camera_id={camera_id}, image_shape={img.shape if img is not None else 'None'}")
    faces = await loop.run_in_executor(_thread_pool, _process_frame, img, camera_id)
    logger.info(f"[/recognize] camera_id={camera_id}, returned_locked_faces={len(faces)}")

    results = []
    for face in faces:
        bbox_int = [int(v) for v in face["bbox"]]
        results.append(
            RecognizedFace(
                matched=False,
                status="unknown",
                person_id=None,
                employee_id=None,
                name=None,
                similarity=0.0,
                det_score=face["det_score"],
                bbox=bbox_int,
                face_image_b64=face.get("face_image_b64"),
                gender=face.get("gender"),
                track_id=face["track_id"],
            )
        )

    return RecognizeResponse(
        success=True,
        results=results,
        count=len(results),
    )


if __name__ == "__main__":
    import uvicorn

    try:
        import uvloop

        uvloop.install()
    except ImportError:
        pass

    uvicorn.run(app, host="0.0.0.0", port=PORT, workers=1)

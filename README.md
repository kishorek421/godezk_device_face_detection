# GoDezk Face Detection Service

## Overview

Standalone RetinaFace-based face detection HTTP API.

Supports:
- Image (JPEG/PNG) via raw binary `POST /detect`
- Base64 image via `POST /detect` JSON body

## Endpoints

- `GET /health` — service health and model load status
- `GET /alive` — liveness probe (503 until model is loaded)
- `GET /ready` — readiness probe (503 until model is loaded)
- `GET /metrics` — Prometheus metrics
- `POST /detect` — detect faces, returns bounding boxes and confidence

## Model

Place the RetinaFace ONNX model at:

```
models/det_10g.onnx
```

or set `MODEL_PATH` to the desired checkpoint.

## Configuration

Copy `.env.example` to `.env` and adjust values.

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/det_10g.onnx` | Path to RetinaFace ONNX model |
| `CONFIDENCE_THRESHOLD` | `0.70` | Minimum face confidence |
| `DET_SIZE` | `640` | Detection input size |
| `PORT` | `8008` | HTTP server port |

## Run locally

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/main.py
```

## Run with Docker

```bash
docker build -t face-detection .
docker run -p 8008:8008 face-detection
```

## Example request

```bash
curl -X POST http://localhost:8008/detect \\
  -H "Content-Type: application/json" \\
  -d '{"frame_base64": "<base64-image>", "camera_id": "cam-01", "frame_id": "frame-001"}'
```

Response:

```json
{
  "success": true,
  "frame_id": "frame-001",
  "camera_id": "cam-01",
  "face_count": 1,
  "faces": [
    {
      "bbox": [120.0, 80.0, 300.0, 300.0],
      "confidence": 0.96
    }
  ],
  "has_face": true,
  "processing_time_ms": 42.5
}
```
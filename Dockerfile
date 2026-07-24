FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download InsightFace buffalo_l pack so the RetinaFace detector is
# available if it is not already provided in models/ at build time.
RUN python -c "
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
fa.prepare(ctx_id=-1, det_size=(640, 640))
print('InsightFace buffalo_l cached')
" || true

COPY src/ ./src/
COPY models/ ./models/

# If the model was not supplied in models/, copy it from the cache
RUN mkdir -p /app/models && \
    if [ ! -f /app/models/det_10g.onnx ]; then \
        cp /root/.insightface/models/buffalo_l/det_10g.onnx /app/models/det_10g.onnx; \
    fi

# Verify model is in place
RUN ls -lah /app/models

EXPOSE 8008

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8008/health', timeout=5)" || exit 1

CMD ["python", "src/main.py"]

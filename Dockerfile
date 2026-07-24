# Use lightweight Python 3.10 base image for smaller final image size
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for OpenCV and YOLO-World
# - git: for cloning CLIP repository
# - libglib2.0-0, libsm6, libxext6, libxrender-dev: OpenCV GUI dependencies
# - libgomp1: OpenMP support for parallel processing
# - libgl1: OpenGL support for visualization
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install CLIP library required by YOLO-World for open-vocabulary detection
# YOLO-World uses CLIP for text-image embeddings to detect custom classes
RUN pip install --no-cache-dir git+https://github.com/ultralytics/CLIP.git

# Copy source code into container
COPY src/ ./src/

# Expose port 8011 for the FastAPI inference server
EXPOSE 8011

# Health check to ensure the service is responsive
# Checks the /health endpoint every 30s after 60s startup grace period
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8011/health', timeout=5)"

# Start the face detection HTTP server
CMD ["python", "src/face_detection_server.py"]

# ─────────────────────────────────────────────
# Vertex AI Custom Training Container
# ─────────────────────────────────────────────
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY configs/ configs/

# Entry point for Vertex AI training job
ENTRYPOINT ["python", "-m", "src.training.trainer"]

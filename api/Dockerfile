# ─── Crop Disease Detection API — Cloud Run ───

FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (cached layer)
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# Copy source code
COPY src/ /app/src/
COPY api/ /app/api/
COPY configs/ /app/configs/

# Create results dir for checkpoints
RUN mkdir -p /app/results

# Expose port (Cloud Run injects PORT env)
EXPOSE 8080
ENV PORT=8080

# Env vars (override at deploy time)
ENV GCS_BUCKET=crop-disease-detection-1
ENV MODELS_TO_DOWNLOAD=resnet_50,vit,mobilenet_v3,swin_base
ENV CHECKPOINTS_DIR=results

# Start server immediately; models download/load lazily on first request
CMD uvicorn api.main:app --host 0.0.0.0 --port 8080 --workers 1

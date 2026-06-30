# ─── Crop Disease Detection API — Cloud Run (CPU) ───

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

COPY src/ /app/src/
COPY api/ /app/api/
COPY configs/ /app/configs/

RUN mkdir -p /app/results

EXPOSE 8080
ENV PORT=8080
ENV GCS_BUCKET=crop-disease-detection-1
ENV MODELS_TO_DOWNLOAD=resnet_50,mobilenet_v3,resnet_152,vit,swin_base
ENV CHECKPOINTS_DIR=results

CMD uvicorn api.main:app --host 0.0.0.0 --port 8080 --workers 1

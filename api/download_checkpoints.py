"""
Download model checkpoints from GCS at container startup.

This avoids baking large checkpoint files into the Docker image.
Checkpoints are downloaded to /app/results/{model_name}/models/best_model_fold_1.pth
"""

import os
import logging
from pathlib import Path

from google.cloud import storage

logger = logging.getLogger(__name__)

GCS_BUCKET = os.environ.get("GCS_BUCKET", "crop-disease-detection-1")
MODELS_TO_DOWNLOAD = os.environ.get(
    "MODELS_TO_DOWNLOAD",
    "resnet_50,resnet_152,vit,mobilenet_v3,swin_base,efficientnet_b4,vgg_16,cnn_baseline"
).split(",")

LOCAL_RESULTS_DIR = Path(os.environ.get("CHECKPOINTS_DIR", "results"))


def download_checkpoints():
    """Download model checkpoints from GCS bucket."""
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)

    for model_name in MODELS_TO_DOWNLOAD:
        model_name = model_name.strip()
        if not model_name:
            continue

        gcs_path = f"results/{model_name}/models/best_model_fold_1.pth"
        local_path = LOCAL_RESULTS_DIR / model_name / "models" / "best_model_fold_1.pth"

        if local_path.exists():
            logger.info(f"Already exists: {local_path}")
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)

        blob = bucket.blob(gcs_path)
        if blob.exists():
            logger.info(f"Downloading: gs://{GCS_BUCKET}/{gcs_path} → {local_path}")
            blob.download_to_filename(str(local_path))
            logger.info(f"Downloaded: {model_name} ({local_path.stat().st_size / 1e6:.1f} MB)")
        else:
            logger.warning(f"Not found in GCS: gs://{GCS_BUCKET}/{gcs_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_checkpoints()

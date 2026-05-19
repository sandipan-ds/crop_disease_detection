"""
Submit a test Custom Training Job to Vertex AI.

Uses vertex_ai_training_test.py — picks 20 random classes,
~5000 train samples, runs the full pipeline on a GPU.

Usage:
    python scripts/submit_vertex_job_test.py
    python scripts/submit_vertex_job_test.py --gpu T4 --region asia-south1 --epochs 10
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
if cred_path and not os.path.isabs(cred_path):
    cred_path = str(PROJECT_ROOT / cred_path)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

from google.cloud import aiplatform


PROJECT_ID = os.getenv("GCP_PROJECT_ID", "crop-disease-detection-496608")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "crop-disease-detection-1")
REGION = os.getenv("GCP_REGION", "asia-south1")


def container_uri(region):
    prefix = region.split("-")[0]
    return f"{prefix}-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-2.py310:latest"

GPU_OPTIONS = {
    "T4":    {"machine": "n1-standard-8",  "gpu": "NVIDIA_TESLA_T4",   "count": 1},
    "V100":  {"machine": "n1-standard-8",  "gpu": "NVIDIA_TESLA_V100", "count": 1},
    "A100":  {"machine": "a2-highgpu-1g",  "gpu": "NVIDIA_TESLA_A100", "count": 1},
}

# Vertex AI TensorBoard (only in asia-south1)
TENSORBOARD_ASIA = "projects/1049249498032/locations/asia-south1/tensorboards/4509369864393064448"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Submit Vertex AI Test Job")
    parser.add_argument("--gpu", type=str, default="T4", choices=GPU_OPTIONS.keys())
    parser.add_argument("--region", type=str, default=REGION)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gpu_config = GPU_OPTIONS[args.gpu]
    image = container_uri(args.region)
    job_name = f"crop-disease-cnn-test-{args.gpu.lower()}-{TIMESTAMP}"
    tensorboard = TENSORBOARD_ASIA if args.region.startswith("asia") else None

    print("=" * 60)
    print("  VERTEX AI — TEST JOB SUBMISSION")
    print("=" * 60)
    print(f"\n  Job name:       {job_name}")
    print(f"  Project:        {PROJECT_ID}")
    print(f"  Region:         {args.region}")
    print(f"  Container:      {image}")
    print(f"  Machine:        {gpu_config['machine']}")
    print(f"  GPU:            {gpu_config['gpu']} × {gpu_config['count']}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Learning rate:  {args.lr}")
    print(f"  GCS bucket:     gs://{BUCKET_NAME}/")
    print(f"  TensorBoard:    {'enabled' if tensorboard else 'N/A (use asia-south1)'}")
    print(f"  Script:         vertex_ai_training_test.py")

    if args.dry_run:
        print("\n  [DRY RUN] Job not submitted.")
        return

    aiplatform.init(
        project=PROJECT_ID,
        location=args.region,
        staging_bucket=f"gs://{BUCKET_NAME}",
    )

    job = aiplatform.CustomJob.from_local_script(
        display_name=job_name,
        script_path=str(PROJECT_ROOT / "scripts" / "vertex_ai_training_test.py"),
        container_uri=image,
        requirements=[
            "scikit-learn>=1.3.0",
            "seaborn>=0.12.0",
            "tqdm>=4.65.0",
            "google-cloud-storage>=2.10.0",
            "Pillow>=10.0.0",
        ],
        args=[
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
        ],
        environment_variables={
            "GCS_BUCKET_NAME": BUCKET_NAME,
        },
        machine_type=gpu_config["machine"],
        accelerator_type=gpu_config["gpu"],
        accelerator_count=gpu_config["count"],
        base_output_dir=f"gs://{BUCKET_NAME}/results/{job_name}",
    )

    print(f"\n  Submitting job: {job_name}")
    print("  This may take 5-10 minutes to provision...")

    job.run(
        service_account=f"crop-disease-detection@{PROJECT_ID}.iam.gserviceaccount.com",
        tensorboard=tensorboard,
        sync=True,
    )

    print(f"\n{'='*60}")
    print(f"  JOB COMPLETE")
    print(f"{'='*60}")
    print(f"  Results: gs://{BUCKET_NAME}/results/{job_name}/")
    print(f"  View in console: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")


if __name__ == "__main__":
    main()

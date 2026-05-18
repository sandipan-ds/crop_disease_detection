"""
Submit a Custom Training Job to Vertex AI.

This script packages your training code and submits it to Vertex AI
using a pre-built PyTorch container (no Docker needed).

Prerequisites:
    1. GCS bucket with data uploaded (run upload_to_gcs.py first)
    2. Service account with Vertex AI User + Storage Object Admin roles
    3. Vertex AI API enabled
    4. GPU quota available in the target region

Usage:
    python scripts/submit_vertex_job.py
    python scripts/submit_vertex_job.py --gpu T4 --region asia-south1
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Resolve service account key path (relative → absolute)
cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
if cred_path and not os.path.isabs(cred_path):
    cred_path = str(PROJECT_ROOT / cred_path)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

from google.cloud import aiplatform


# =========================================================
# CONFIG
# =========================================================

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "crop-disease-detection-496608")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "crop-disease-detection-1")
REGION = os.getenv("GCP_REGION", "asia-south1")  # Mumbai (closest to India)

# Pre-built PyTorch containers from Google
# See: https://cloud.google.com/vertex-ai/docs/training/pre-built-containers
PYTORCH_CONTAINER = "asia-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-2:latest"

# GPU options
GPU_OPTIONS = {
    "T4":    {"machine": "n1-standard-8",  "gpu": "NVIDIA_TESLA_T4",   "count": 1},
    "V100":  {"machine": "n1-standard-8",  "gpu": "NVIDIA_TESLA_V100", "count": 1},
    "A100":  {"machine": "a2-highgpu-1g",  "gpu": "NVIDIA_TESLA_A100", "count": 1},
}

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# =========================================================
# MAIN
# =========================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Submit Vertex AI Training Job")
    parser.add_argument("--gpu", type=str, default="T4", choices=GPU_OPTIONS.keys(),
                        help="GPU type (default: T4)")
    parser.add_argument("--region", type=str, default=REGION,
                        help=f"GCP region (default: {REGION})")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print job config without submitting")
    args = parser.parse_args()

    gpu_config = GPU_OPTIONS[args.gpu]
    job_name = f"crop-disease-cnn-{args.gpu.lower()}-{TIMESTAMP}"

    print("=" * 60)
    print("  VERTEX AI JOB SUBMISSION")
    print("=" * 60)
    print(f"\n  Job name:       {job_name}")
    print(f"  Project:        {PROJECT_ID}")
    print(f"  Region:         {args.region}")
    print(f"  Container:      {PYTORCH_CONTAINER}")
    print(f"  Machine:        {gpu_config['machine']}")
    print(f"  GPU:            {gpu_config['gpu']} × {gpu_config['count']}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Learning rate:  {args.lr}")
    print(f"  GCS bucket:     gs://{BUCKET_NAME}/")

    if args.dry_run:
        print("\n  [DRY RUN] Job not submitted.")
        return

    # --- Initialize Vertex AI ---
    aiplatform.init(
        project=PROJECT_ID,
        location=args.region,
        staging_bucket=f"gs://{BUCKET_NAME}",
    )

    # --- Create and submit custom training job ---
    job = aiplatform.CustomJob.from_local_script(
        display_name=job_name,

        # Training script and source code
        script_path=str(PROJECT_ROOT / "scripts" / "vertex_ai_training.py"),
        container_uri=PYTORCH_CONTAINER,

        # Pass the src/ directory so imports work
        requirements=[
            "scikit-learn>=1.3.0",
            "seaborn>=0.12.0",
            "tqdm>=4.65.0",
            "google-cloud-storage>=2.10.0",
            "Pillow>=10.0.0",
        ],

        # Command line args to the training script
        args=[
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
        ],

        # Environment variables
        environment_variables={
            "GCS_BUCKET_NAME": BUCKET_NAME,
        },

        # Machine configuration
        machine_type=gpu_config["machine"],
        accelerator_type=gpu_config["gpu"],
        accelerator_count=gpu_config["count"],

        # Model output directory in GCS
        model_serving_container_image_uri=None,
        base_output_dir=f"gs://{BUCKET_NAME}/results/{job_name}",
    )

    print(f"\n  Submitting job: {job_name}")
    print("  This may take 5-10 minutes to provision...")

    # Submit and monitor
    job.run(
        service_account=f"crop-disease-detection@{PROJECT_ID}.iam.gserviceaccount.com",
        tensorboard=None,  # Set this if you create a Vertex AI TensorBoard instance
        sync=True,         # Wait for job to complete
    )

    print(f"\n{'='*60}")
    print(f"  JOB COMPLETE")
    print(f"{'='*60}")
    print(f"  Results: gs://{BUCKET_NAME}/results/{job_name}/")
    print(f"  View in console: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")


if __name__ == "__main__":
    main()

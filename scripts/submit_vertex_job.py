"""
Submit a Full Custom Training Job to Vertex AI.

Uses vertex_ai_training.py — full dataset (~42K train + ~9.5K val + ~9.5K test),
class-balanced via WeightedRandomSampler + weighted CrossEntropyLoss.

Usage:
    python scripts/submit_vertex_job.py
    python scripts/submit_vertex_job.py --gpu T4 --region asia-south1 --epochs 30
"""

import os
import sys
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
if cred_path and not os.path.isabs(cred_path):
    cred_path = str(PROJECT_ROOT / cred_path)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

from google.cloud import aiplatform, storage


PROJECT_ID = os.getenv("GCP_PROJECT_ID", "crop-disease-detection-496608")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "crop-disease-detection-1")
REGION = os.getenv("GCP_REGION", "us-central1")


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


# ─────────────────────────────────────────────────────────────
#  Package builder: bundles training script + src/ modules
# ─────────────────────────────────────────────────────────────

def _build_and_upload_package(project_root, bucket_name):
    """
    Build a Python source distribution containing:
      - trainer/task.py  (vertex_ai_training.py)
      - src/             (dataset, model, trainer, augmentations)

    Uploads the sdist .tar.gz to GCS and returns the gs:// URI.
    """
    tmpdir_path = tempfile.mkdtemp(prefix="vertex_pkg_")
    try:
        tmpdir = Path(tmpdir_path)

        # ── trainer/ package (wraps the training script) ──
        trainer_dir = tmpdir / "trainer"
        trainer_dir.mkdir()
        (trainer_dir / "__init__.py").write_text("")
        shutil.copy2(
            project_root / "scripts" / "vertex_ai_training.py",
            trainer_dir / "task.py",
        )

        # ── src/ package (only the modules the script imports) ──
        src_dir = tmpdir / "src"
        src_dir.mkdir()
        shutil.copy2(project_root / "src" / "__init__.py", src_dir / "__init__.py")
        for module in ["dataset.py", "model.py", "trainer.py", "augmentations.py"]:
            shutil.copy2(project_root / "src" / module, src_dir / module)

        # ── setup.py ──
        (tmpdir / "setup.py").write_text(textwrap.dedent("""\
            from setuptools import setup, find_packages
            setup(
                name="crop-disease-trainer",
                version="0.1.0",
                packages=find_packages(),
                install_requires=[
                    "scikit-learn>=1.3.0",
                    "seaborn>=0.12.0",
                    "tqdm>=4.65.0",
                    "google-cloud-storage>=2.10.0",
                    "Pillow>=10.0.0",
                    "tensorboard>=2.13.0",
                    "python-json-logger>=2.0.0",
                    "pandas>=2.0.0",
                    "psutil>=5.9.0",
                ],
            )
        """))

        # ── Build sdist ──
        print("  Building source distribution...")
        result = subprocess.run(
            [sys.executable, "setup.py", "sdist", "--formats=gztar"],
            cwd=str(tmpdir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  [ERROR] sdist build failed:\n{result.stderr}")
            raise RuntimeError("Failed to build source distribution")

        sdist_path = next((tmpdir / "dist").glob("*.tar.gz"))

        # ── Upload to GCS ──
        gcs_key = f"packages/{sdist_path.name}"
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_key)
        blob.upload_from_filename(str(sdist_path))

        gcs_uri = f"gs://{bucket_name}/{gcs_key}"
        print(f"  Package uploaded → {gcs_uri}")
        return gcs_uri

    finally:
        shutil.rmtree(tmpdir_path, ignore_errors=True)


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Submit Vertex AI Full Training Job")
    parser.add_argument("--model", type=str, default="cnn_baseline",
                        choices=["cnn_baseline", "resnet_50", "vgg_16", "vit"],
                        help="Model architecture to train")
    parser.add_argument("--gpu", type=str, default="T4", choices=GPU_OPTIONS.keys())
    parser.add_argument("--region", type=str, default=REGION)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--folds", type=int, default=1,
                        help="Number of CV folds (1=single split, 5=full CV)")
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="Disable Vertex AI managed TensorBoard (logs still saved to GCS)")
    parser.add_argument("--timeout", type=int, default=86400,
                        help="Max job duration in seconds (default: 86400 = 24h)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gpu_config = GPU_OPTIONS[args.gpu]
    image = container_uri(args.region)
    job_name = f"crop-disease-{args.model}-{args.gpu.lower()}-{TIMESTAMP}"

    # Disable TensorBoard sidecar if requested (suspected crash cause)
    if args.no_tensorboard:
        tensorboard = None
    else:
        tensorboard = TENSORBOARD_ASIA if args.region.startswith("asia") else None

    print("=" * 60)
    print("  VERTEX AI — FULL TRAINING JOB SUBMISSION")
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
    print(f"  CV folds:       {args.folds} ({'single split' if args.folds == 1 else f'{args.folds}-fold CV'})")
    print(f"  GCS bucket:     gs://{BUCKET_NAME}/")
    print(f"  TensorBoard:    {'DISABLED' if args.no_tensorboard else ('enabled' if tensorboard else 'N/A')}")
    print(f"  Timeout:        {args.timeout}s ({args.timeout//3600}h)")
    print(f"  Model:          {args.model}")
    print(f"  Dataset:        FULL (~42K train + ~9.5K val + ~9.5K test, class-balanced)")
    print(f"  Script:         vertex_ai_training.py (packaged as trainer.task)")

    if args.dry_run:
        print("\n  [DRY RUN] Job not submitted.")
        return

    # ── Build & upload Python package with src/ modules ──
    print("\n  Packaging training script + src/ modules...")
    package_uri = _build_and_upload_package(PROJECT_ROOT, BUCKET_NAME)

    aiplatform.init(
        project=PROJECT_ID,
        location=args.region,
        staging_bucket=f"gs://{BUCKET_NAME}",
    )

    # ── CustomPythonPackageTrainingJob (bundles src/ with the script) ──
    job = aiplatform.CustomPythonPackageTrainingJob(
        display_name=job_name,
        python_package_gcs_uri=package_uri,
        python_module_name="trainer.task",
        container_uri=image,
    )

    print(f"\n  Submitting job: {job_name}")
    print("  This may take 5-10 minutes to provision...")
    print("  Full training will take several hours.")

    job.run(
        args=[
            "--model", args.model,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--folds", str(args.folds),
        ],
        environment_variables={
            "GCS_BUCKET_NAME": BUCKET_NAME,
        },
        machine_type=gpu_config["machine"],
        accelerator_type=gpu_config["gpu"],
        accelerator_count=gpu_config["count"],
        base_output_dir=f"gs://{BUCKET_NAME}/results/{job_name}",
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


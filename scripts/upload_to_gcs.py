"""
Upload processed dataset to Google Cloud Storage.

Usage:
    python scripts/upload_to_gcs.py
"""

import os
from pathlib import Path
from google.cloud import storage
from tqdm import tqdm
from dotenv import load_dotenv

# =========================================================
# CONFIG (loaded from .env)
# =========================================================

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
LOCATION = os.getenv("GCS_LOCATION", "asia")

# Resolve service account key path (relative → absolute)
cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
if cred_path and not os.path.isabs(cred_path):
    cred_path = str(PROJECT_ROOT / cred_path)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

# What to upload  (local_path, gcs_prefix)
UPLOAD_DIRS = [
    (PROJECT_ROOT / "data" / "processed" / "combined_train", "data/processed/combined_train"),
    (PROJECT_ROOT / "data" / "processed" / "combined_test",  "data/processed/combined_test"),
]

UPLOAD_FILES = [
    (PROJECT_ROOT / "notebook" / "train.csv",           "data/train.csv"),
    (PROJECT_ROOT / "notebook" / "test.csv",            "data/test.csv"),
    (PROJECT_ROOT / "configs" / "label_mapping.json",   "configs/label_mapping.json"),
]


# =========================================================
# HELPERS
# =========================================================

def get_client():
    """Authenticate and return a storage client (uses ADC)."""
    client = storage.Client(project=PROJECT_ID)
    return client


def get_or_create_bucket(client):
    """Get a reference to the bucket (must already exist)."""
    bucket = client.bucket(BUCKET_NAME)
    print(f"Using bucket: gs://{BUCKET_NAME}/")
    return bucket


def upload_file(bucket, local_path, gcs_path):
    """Upload a single file to GCS."""
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(str(local_path))


def upload_directory(bucket, local_dir, gcs_prefix):
    """Upload an entire directory to GCS with progress bar."""

    local_dir = Path(local_dir)

    # Collect all files first
    all_files = []
    for root, dirs, files in os.walk(local_dir):
        for filename in files:
            local_path = Path(root) / filename
            relative = local_path.relative_to(local_dir)
            gcs_path = f"{gcs_prefix}/{relative.as_posix()}"
            all_files.append((local_path, gcs_path))

    print(f"\nUploading {len(all_files)} files from {local_dir.name}/")

    for local_path, gcs_path in tqdm(all_files, desc=f"  {gcs_prefix}"):
        upload_file(bucket, local_path, gcs_path)

    return len(all_files)


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("  GCS UPLOAD SCRIPT")
    print("=" * 60)

    print(f"Project:         {PROJECT_ID}")
    print(f"Bucket:          gs://{BUCKET_NAME}/")

    client = get_client()
    bucket = get_or_create_bucket(client)

    total_uploaded = 0

    # ---------------------------------------------------------
    # Upload directories (images)
    # ---------------------------------------------------------
    for local_dir, gcs_prefix in UPLOAD_DIRS:
        if local_dir.exists():
            count = upload_directory(bucket, local_dir, gcs_prefix)
            total_uploaded += count
        else:
            print(f"\nDirectory not found: {local_dir}")

    # ---------------------------------------------------------
    # Upload individual files (CSVs, configs)
    # ---------------------------------------------------------
    print("\nUploading config files...")
    for local_path, gcs_path in UPLOAD_FILES:
        if local_path.exists():
            upload_file(bucket, local_path, gcs_path)
            print(f"  Done: {gcs_path}")
            total_uploaded += 1
        else:
            print(f"  Not found: {local_path}")

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  UPLOAD COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total files uploaded: {total_uploaded}")
    print(f"  Bucket:              gs://{BUCKET_NAME}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

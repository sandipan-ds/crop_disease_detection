"""
Vertex AI training script for crop disease detection (CNN baseline).

Runs on Vertex AI Custom Training with full dataset.
- Downloads data from GCS
- Train + 5-fold CV on full ~42K images
- Test evaluation on full ~19K images
- TensorBoard logs synced to Vertex AI
- Best model saved back to GCS

Usage (local test):
    python scripts/vertex_ai_training.py --local-test

Vertex AI runs this automatically via the job submission script.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path (handles both local and container execution)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from google.cloud import storage

from src.dataset import CropDiseaseDataset
from src.model import build_model
from src.trainer import Trainer
from src.augmentations import get_train_transforms, get_eval_transforms


# =========================================================
# CONFIG
# =========================================================

CONFIG = {
    # Model
    "num_classes": 102,
    "img_size": 224,
    "dropout_conv": 0.25,
    "dropout_fc": 0.5,

    # Training
    "epochs": 30,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "min_lr": 1e-6,
    "n_folds": 5,

    # DataLoader
    "num_workers": 4,

    # Augmentation
    "min_aug": 0,
    "max_aug": 5,
}

# GCS settings (loaded from env or defaults)
GCS_BUCKET = os.getenv("GCS_BUCKET_NAME", "crop-disease-detection-1")
GCS_DATA_PREFIX = "data"


# =========================================================
# GCS HELPERS
# =========================================================

def download_from_gcs(bucket_name, gcs_prefix, local_dir):
    """Download a directory from GCS to local filesystem."""
    local_dir = Path(local_dir)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = list(bucket.list_blobs(prefix=gcs_prefix))
    print(f"  Downloading {len(blobs)} files from gs://{bucket_name}/{gcs_prefix}/")

    from tqdm import tqdm
    for blob in tqdm(blobs, desc=f"  {gcs_prefix}"):
        if blob.name.endswith("/"):
            continue

        # Preserve directory structure
        relative_path = blob.name[len(gcs_prefix):].lstrip("/")
        local_path = local_dir / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))


def upload_to_gcs(local_path, bucket_name, gcs_path):
    """Upload a single file to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(str(local_path))
    print(f"  Uploaded: gs://{bucket_name}/{gcs_path}")


def upload_dir_to_gcs(local_dir, bucket_name, gcs_prefix):
    """Upload an entire directory to GCS."""
    local_dir = Path(local_dir)
    for file_path in local_dir.rglob("*"):
        if file_path.is_file():
            relative = file_path.relative_to(local_dir)
            gcs_path = f"{gcs_prefix}/{relative.as_posix()}"
            upload_to_gcs(str(file_path), bucket_name, gcs_path)


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Crop Disease CNN Training")
    parser.add_argument("--local-test", action="store_true",
                        help="Run locally instead of on Vertex AI (uses local data)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    # Override config from args
    if args.epochs:
        CONFIG["epochs"] = args.epochs
    if args.batch_size:
        CONFIG["batch_size"] = args.batch_size
    if args.lr:
        CONFIG["learning_rate"] = args.lr

    print("=" * 60)
    print("  VERTEX AI TRAINING — CNN BASELINE")
    print("  (Full dataset: ~42K train, ~19K test)")
    print("=" * 60)

    # --- Device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print("\n  WARNING: No GPU detected — training on CPU")

    # --- Data directory setup ---
    if args.local_test:
        # Use local data paths
        data_root = str(PROJECT_ROOT)
        train_csv = str(PROJECT_ROOT / "notebook" / "train.csv")
        test_csv = str(PROJECT_ROOT / "notebook" / "test.csv")
        log_dir = str(PROJECT_ROOT / "runs" / "vertex_local_test")
        save_dir = str(PROJECT_ROOT / "models" / "saved" / "vertex_local_test")
    else:
        # On Vertex AI: download from GCS to /tmp
        data_root = "/tmp/crop_data"
        train_csv = f"{data_root}/train.csv"
        test_csv = f"{data_root}/test.csv"

        # Vertex AI provides AIP_TENSORBOARD_LOG_DIR for managed TensorBoard
        log_dir = os.getenv("AIP_TENSORBOARD_LOG_DIR", "/tmp/runs")

        # Vertex AI provides AIP_MODEL_DIR for model output
        save_dir = os.getenv("AIP_MODEL_DIR", "/tmp/model_output")

        print(f"\n  Downloading data from gs://{GCS_BUCKET}/{GCS_DATA_PREFIX}/")

        # Download training images
        download_from_gcs(GCS_BUCKET, f"{GCS_DATA_PREFIX}/processed/combined_train",
                          f"{data_root}/data/processed/combined_train")

        # Download test images
        download_from_gcs(GCS_BUCKET, f"{GCS_DATA_PREFIX}/processed/combined_test",
                          f"{data_root}/data/processed/combined_test")

        # Download CSVs
        download_from_gcs(GCS_BUCKET, f"{GCS_DATA_PREFIX}/train.csv",
                          f"{data_root}")
        download_from_gcs(GCS_BUCKET, f"{GCS_DATA_PREFIX}/test.csv",
                          f"{data_root}")

        # Fix paths in CSV: the CSVs reference paths relative to project root
        # On Vertex AI, data_root IS the project root
        print("  Data download complete.")

    # --- Transforms ---
    train_transform = get_train_transforms(
        img_size=CONFIG["img_size"],
        min_aug=CONFIG["min_aug"],
        max_aug=CONFIG["max_aug"],
    )
    eval_transform = get_eval_transforms(img_size=CONFIG["img_size"])

    # --- Datasets ---
    print("\n  Loading datasets...")
    train_dataset = CropDiseaseDataset(
        csv_path=train_csv,
        data_root=data_root,
        transform=train_transform,
    )
    test_dataset = CropDiseaseDataset(
        csv_path=test_csv,
        data_root=data_root,
        transform=eval_transform,
    )

    class_names = train_dataset.get_class_names()
    CONFIG["num_classes"] = train_dataset.num_classes

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples:  {len(test_dataset)}")
    print(f"  Num classes:   {train_dataset.num_classes}")

    # --- Model factory ---
    def model_fn():
        return build_model(
            num_classes=CONFIG["num_classes"],
            dropout_conv=CONFIG["dropout_conv"],
            dropout_fc=CONFIG["dropout_fc"],
        )

    # Print model summary
    model_temp = model_fn()
    total_params = sum(p.numel() for p in model_temp.parameters())
    print(f"\n  Model params: {total_params:,}")
    del model_temp

    # --- Trainer ---
    trainer = Trainer(
        model_fn=model_fn,
        config=CONFIG,
        log_dir=log_dir,
        save_dir=save_dir,
        device=device,
        class_names=class_names,
    )

    # --- Run 5-fold CV ---
    fold_results, avg_metrics = trainer.run_cv(
        dataset=train_dataset,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )

    # --- Final test evaluation ---
    best_fold = max(range(1, CONFIG["n_folds"] + 1),
                    key=lambda f: fold_results[f - 1]["f1_macro"])
    print(f"\n  Using fold {best_fold} for test evaluation (highest CV F1)")

    trainer.evaluate_test(
        test_dataset=test_dataset,
        eval_transform=eval_transform,
        fold_to_use=best_fold,
    )

    # --- Upload results to GCS (if on Vertex AI) ---
    if not args.local_test:
        print(f"\n  Uploading results to gs://{GCS_BUCKET}/results/")
        upload_dir_to_gcs(save_dir, GCS_BUCKET, "results/models")
        upload_dir_to_gcs(log_dir, GCS_BUCKET, "results/logs")

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")

    if args.local_test:
        print(f"\n  TensorBoard: tensorboard --logdir {log_dir}")
        print(f"  Model checkpoints: {save_dir}")


if __name__ == "__main__":
    main()

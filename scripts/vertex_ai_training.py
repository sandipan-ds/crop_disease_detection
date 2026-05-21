"""
Vertex AI training script for crop disease detection (CNN baseline).

Runs on Vertex AI Custom Training with full dataset.
- Downloads data from GCS (parallel, fast)
- Balances classes via oversampling to median class count
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
import numpy as np
import pandas as pd
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
    "epochs": 200,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "min_lr": 1e-6,
    "n_folds": 5,

    # LR scheduling
    "use_reduce_lr_on_plateau": True,
    "lr_factor": 0.5,
    "lr_patience": 3,

    # Early stopping
    "early_stop_patience": 6,

    # DataLoader (2 workers to stay within n1-standard-8's 30GB RAM)
    "num_workers": 2,

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
    """Download a directory from GCS to local filesystem (sequential)."""
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


def download_selected_files_parallel(bucket_name, image_paths, data_root, max_workers=32):
    """Download specific files from GCS in parallel using ThreadPoolExecutor."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    data_root = Path(data_root)

    def _download_one(gcs_path):
        local_path = data_root / gcs_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob = bucket.blob(gcs_path)
        blob.download_to_filename(str(local_path))

    # Deduplicate and normalize paths
    unique_paths = list(set(p.replace("\\", "/") for p in image_paths))
    print(f"  Downloading {len(unique_paths)} files with {max_workers} parallel workers...")

    failed = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_one, p): p for p in unique_paths}
        for future in tqdm(as_completed(futures), total=len(futures), desc="  Downloading"):
            try:
                future.result()
            except Exception as e:
                failed.append((futures[future], str(e)))

    if failed:
        print(f"  WARNING: {len(failed)} files failed to download")
        for path, err in failed[:5]:
            print(f"    {path}: {err}")


def download_file_from_gcs(bucket_name, gcs_path, local_path):
    """Download a single file from GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.download_to_filename(str(local_path))
    print(f"  Downloaded: gs://{bucket_name}/{gcs_path} -> {local_path}")


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
# CLASS BALANCING
# =========================================================

def balance_classes_by_oversampling(df, target_col="label"):
    """
    Balance classes by oversampling underrepresented classes to the median
    class count. Classes above the median are left untouched.

    This ensures the DataLoader sees a roughly equal number of samples per
    class. The actual augmentation (random transforms) applied during training
    ensures oversampled copies are visually diverse.

    Returns a new DataFrame with oversampled rows appended.
    """
    class_counts = df[target_col].value_counts()
    median_count = int(class_counts.median())

    print(f"\n  Class balance statistics:")
    print(f"    Min class count:    {class_counts.min()}")
    print(f"    Max class count:    {class_counts.max()}")
    print(f"    Median class count: {median_count}")
    print(f"    Std:                {class_counts.std():.1f}")

    balanced_dfs = []
    oversampled_count = 0

    for label_val in sorted(df[target_col].unique()):
        class_df = df[df[target_col] == label_val]
        current_count = len(class_df)

        if current_count >= median_count:
            # Class already at or above median — keep as-is
            balanced_dfs.append(class_df)
        else:
            # Oversample to reach median
            shortage = median_count - current_count
            oversampled = class_df.sample(n=shortage, replace=True, random_state=42)
            balanced_dfs.append(pd.concat([class_df, oversampled], ignore_index=True))
            oversampled_count += shortage

    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    print(f"    Oversampled rows:   {oversampled_count}")
    print(f"    Original size:      {len(df)}")
    print(f"    Balanced size:      {len(balanced_df)}")

    return balanced_df


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Crop Disease CNN Training — Full Dataset")
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
    print("  (Full dataset: ~42K train, ~19K test, class-balanced)")
    print("=" * 60)

    # --- Device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
        dprops = torch.cuda.get_device_properties(0)
        vram_gb = getattr(dprops, "total_memory", getattr(dprops, "total_mem", 0)) / 1e9
        print(f"  VRAM: {vram_gb:.1f} GB")
    else:
        print("\n  WARNING: No GPU detected — training on CPU")

    # --- Data directory setup ---
    if args.local_test:
        # Use local data paths
        data_root = str(PROJECT_ROOT)
        train_csv = str(PROJECT_ROOT / "notebook" / "train.csv")
        test_csv = str(PROJECT_ROOT / "notebook" / "test.csv")
        log_dir = str(PROJECT_ROOT / "runs" / "vertex_full")
        save_dir = str(PROJECT_ROOT / "models" / "saved" / "vertex_full")
    else:
        # On Vertex AI: download from GCS to /tmp
        data_root = "/tmp/crop_data"
        train_csv = f"{data_root}/train.csv"
        test_csv = f"{data_root}/test.csv"

        # Vertex AI provides AIP_TENSORBOARD_LOG_DIR for managed TensorBoard
        log_dir = os.getenv("AIP_TENSORBOARD_LOG_DIR", "/tmp/runs")

        # Vertex AI provides AIP_MODEL_DIR for model output
        save_dir = os.getenv("AIP_MODEL_DIR", "/tmp/model_output")

        # Step 1: Download only the CSV manifests (lightweight)
        print(f"\n  Downloading CSV manifests from gs://{GCS_BUCKET}/{GCS_DATA_PREFIX}/")
        Path(data_root).mkdir(parents=True, exist_ok=True)
        download_file_from_gcs(GCS_BUCKET, f"{GCS_DATA_PREFIX}/train.csv", train_csv)
        download_file_from_gcs(GCS_BUCKET, f"{GCS_DATA_PREFIX}/test.csv", test_csv)

    # --- Load and balance training data ---
    print("\n  Loading CSV manifests...")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    print(f"  Original train: {len(train_df)} samples, {train_df['label'].nunique()} classes")
    print(f"  Test:           {len(test_df)} samples, {test_df['label'].nunique()} classes")

    # Balance training classes by oversampling to median
    train_df_balanced = balance_classes_by_oversampling(train_df, target_col="label")

    # Step 2 (Vertex AI only): Download all images in parallel
    if not args.local_test:
        # Collect unique image paths from both train (balanced) and test
        all_image_paths = list(set(
            train_df_balanced["image_path"].values.tolist() +
            test_df["image_path"].values.tolist()
        ))
        print(f"\n  Downloading {len(all_image_paths)} unique images (parallel)...")
        download_selected_files_parallel(GCS_BUCKET, all_image_paths, data_root)
        print("  Data download complete.")

    # Save balanced train CSV to temp location
    temp_dir = Path("/tmp/crop_balanced" if not args.local_test else str(PROJECT_ROOT / "data" / "temp_balanced"))
    temp_dir.mkdir(parents=True, exist_ok=True)
    balanced_train_csv = str(temp_dir / "train_balanced.csv")
    train_df_balanced.to_csv(balanced_train_csv, index=False)

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
        csv_path=balanced_train_csv,
        data_root=data_root,
    )
    test_dataset = CropDiseaseDataset(
        csv_path=test_csv,
        data_root=data_root,
    )

    class_names = train_dataset.get_class_names()
    CONFIG["num_classes"] = train_dataset.num_classes

    print(f"  Train samples (balanced): {len(train_dataset)}")
    print(f"  Test samples:             {len(test_dataset)}")
    print(f"  Num classes:              {train_dataset.num_classes}")

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

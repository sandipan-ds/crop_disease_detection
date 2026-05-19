"""
Vertex AI training test — validates the pipeline with a small subset.

Randomly picks 20 unique classes, filters train.csv and test.csv to
only those classes (up to 5000 training samples), then runs the full
training pipeline: CV folds, train, validate, test, metrics, TensorBoard.

This is meant to be a quick smoke test on Vertex AI before the
full 40K-sample run.

Usage (local test):
    python scripts/vertex_ai_training_test.py --local-test

Usage on Vertex AI:
    (Submitted automatically by submit_vertex_job.py)
"""

import os
import sys
import argparse
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from google.cloud import storage

from src.dataset import CropDiseaseDataset
from src.model import build_model
from src.trainer import Trainer
from src.augmentations import get_train_transforms, get_eval_transforms


CONFIG = {
    "num_classes": 20,
    "img_size": 224,
    "dropout_conv": 0.25,
    "dropout_fc": 0.5,
    "epochs": 10,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "min_lr": 1e-6,
    "n_folds": 3,
    "num_workers": 2,
    "min_aug": 0,
    "max_aug": 5,
    "use_reduce_lr_on_plateau": True,
    "lr_factor": 0.5,
    "lr_patience": 2,
    "early_stop_patience": 4,
}

GCS_BUCKET = os.getenv("GCS_BUCKET_NAME", "crop-disease-detection-1")
GCS_DATA_PREFIX = "data"

SUBSET_CLASSES = 20
SUBSET_TRAIN_SAMPLES = 5000
SEED = 42


def download_from_gcs(bucket_name, gcs_prefix, local_dir):
    local_dir = Path(local_dir)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=gcs_prefix))
    print(f"  Downloading {len(blobs)} files from gs://{bucket_name}/{gcs_prefix}/")
    from tqdm import tqdm
    for blob in tqdm(blobs, desc=f"  {gcs_prefix}"):
        if blob.name.endswith("/"):
            continue
        relative_path = blob.name[len(gcs_prefix):].lstrip("/")
        local_path = local_dir / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))


def download_file_from_gcs(bucket_name, gcs_path, local_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.download_to_filename(str(local_path))
    print(f"  Downloaded: gs://{bucket_name}/{gcs_path} -> {local_path}")


def upload_to_gcs(local_path, bucket_name, gcs_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(str(local_path))
    print(f"  Uploaded: gs://{bucket_name}/{gcs_path}")


def upload_dir_to_gcs(local_dir, bucket_name, gcs_prefix):
    local_dir = Path(local_dir)
    for file_path in local_dir.rglob("*"):
        if file_path.is_file():
            relative = file_path.relative_to(local_dir)
            gcs_path = f"{gcs_prefix}/{relative.as_posix()}"
            upload_to_gcs(str(file_path), bucket_name, gcs_path)


def _filter_csv(csv_path, selected_labels, sample_n=None):
    """Filter CSV to only selected labels, optionally sample N total."""
    df = pd.read_csv(csv_path)
    df = df[df["label"].isin(selected_labels)].reset_index(drop=True)
    if sample_n is not None and sample_n < len(df):
        per_class = max(1, sample_n // len(selected_labels))
        sampled = []
        for label in selected_labels:
            class_df = df[df["label"] == label]
            n = min(len(class_df), per_class)
            sampled.append(class_df.sample(n=n, random_state=SEED))
        df = pd.concat(sampled, ignore_index=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Crop Disease CNN — Vertex AI Test")
    parser.add_argument("--local-test", action="store_true",
                        help="Run locally instead of on Vertex AI")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    if args.epochs:
        CONFIG["epochs"] = args.epochs
    if args.batch_size:
        CONFIG["batch_size"] = args.batch_size
    if args.lr:
        CONFIG["learning_rate"] = args.lr

    random.seed(SEED)

    print("=" * 60)
    print("  VERTEX AI TRAINING — TEST RUN")
    print(f"  ({SUBSET_CLASSES} classes, {SUBSET_TRAIN_SAMPLES} train samples)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
        dprops = torch.cuda.get_device_properties(0)
vram_gb = getattr(dprops, "total_memory", getattr(dprops, "total_mem", 0)) / 1e9
print(f"  VRAM: {vram_gb:.1f} GB")
    else:
        print("\n  WARNING: No GPU detected — training on CPU")

    # Data directory setup
    if args.local_test:
        data_root = str(PROJECT_ROOT)
        train_csv = str(PROJECT_ROOT / "notebook" / "train.csv")
        test_csv = str(PROJECT_ROOT / "notebook" / "test.csv")
        log_dir = str(PROJECT_ROOT / "runs" / "vertex_test")
        save_dir = str(PROJECT_ROOT / "models" / "saved" / "vertex_test")
    else:
        data_root = "/tmp/crop_data"
        train_csv = f"{data_root}/train.csv"
        test_csv = f"{data_root}/test.csv"
        log_dir = os.getenv("AIP_TENSORBOARD_LOG_DIR", "/tmp/runs")
        save_dir = os.getenv("AIP_MODEL_DIR", "/tmp/model_output")

        print(f"\n  Downloading data from gs://{GCS_BUCKET}/{GCS_DATA_PREFIX}/")
        download_from_gcs(GCS_BUCKET, f"{GCS_DATA_PREFIX}/processed/combined_train",
                          f"{data_root}/data/processed/combined_train")
        download_from_gcs(GCS_BUCKET, f"{GCS_DATA_PREFIX}/processed/combined_test",
                          f"{data_root}/data/processed/combined_test")
        download_file_from_gcs(GCS_BUCKET, f"{GCS_DATA_PREFIX}/train.csv", train_csv)
        download_file_from_gcs(GCS_BUCKET, f"{GCS_DATA_PREFIX}/test.csv", test_csv)
        print("  Data download complete.")

    # Read CSVs and select random subset of classes
    print("\n  Selecting random subset of classes...")
    full_train = pd.read_csv(train_csv)
    all_labels = sorted(full_train["label"].unique().tolist())
    selected_labels = sorted(random.sample(all_labels, min(SUBSET_CLASSES, len(all_labels))))
    print(f"  Selected {len(selected_labels)} labels: {selected_labels}")

    # Filter both CSVs
    train_subset = _filter_csv(train_csv, selected_labels, sample_n=SUBSET_TRAIN_SAMPLES)
    test_subset = _filter_csv(test_csv, selected_labels)

    # Save filtered CSVs to temp files
    temp_dir = Path("/tmp/crop_subset" if not args.local_test else str(PROJECT_ROOT / "data" / "temp_subset"))
    temp_dir.mkdir(parents=True, exist_ok=True)
    train_subset_path = str(temp_dir / "train_subset.csv")
    test_subset_path = str(temp_dir / "test_subset.csv")
    train_subset.to_csv(train_subset_path, index=False)
    test_subset.to_csv(test_subset_path, index=False)

    print(f"  Train subset: {len(train_subset)} samples, {train_subset['label'].nunique()} classes")
    print(f"  Test subset:  {len(test_subset)} samples, {test_subset['label'].nunique()} classes")

    # Re-label to 0..N-1
    le = {old: new for new, old in enumerate(sorted(train_subset["label"].unique()))}
    train_subset["label"] = train_subset["label"].map(le)
    test_subset["label"] = test_subset["label"].map(le)
    train_subset.to_csv(train_subset_path, index=False)
    test_subset.to_csv(test_subset_path, index=False)

    # Transforms
    train_transform = get_train_transforms(
        img_size=CONFIG["img_size"],
        min_aug=CONFIG["min_aug"],
        max_aug=CONFIG["max_aug"],
    )
    eval_transform = get_eval_transforms(img_size=CONFIG["img_size"])

    # Datasets
    print("\n  Loading datasets...")
    train_dataset = CropDiseaseDataset(
        csv_path=train_subset_path,
        data_root=data_root,
    )
    test_dataset = CropDiseaseDataset(
        csv_path=test_subset_path,
        data_root=data_root,
    )

    CONFIG["num_classes"] = train_dataset.num_classes
    class_names = train_dataset.get_class_names()

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples:  {len(test_dataset)}")
    print(f"  Num classes:   {train_dataset.num_classes}")

    def model_fn():
        return build_model(
            num_classes=CONFIG["num_classes"],
            dropout_conv=CONFIG["dropout_conv"],
            dropout_fc=CONFIG["dropout_fc"],
        )

    model_temp = model_fn()
    total_params = sum(p.numel() for p in model_temp.parameters())
    print(f"\n  Model params: {total_params:,}")
    del model_temp

    trainer = Trainer(
        model_fn=model_fn,
        config=CONFIG,
        log_dir=log_dir,
        save_dir=save_dir,
        device=device,
        class_names=class_names,
    )

    fold_results, avg_metrics = trainer.run_cv(
        dataset=train_dataset,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )

    best_fold = max(range(1, CONFIG["n_folds"] + 1),
                    key=lambda f: fold_results[f - 1]["f1_macro"])
    print(f"\n  Using fold {best_fold} for test evaluation (highest CV F1)")

    trainer.evaluate_test(
        test_dataset=test_dataset,
        eval_transform=eval_transform,
        fold_to_use=best_fold,
    )

    if not args.local_test:
        print(f"\n  Uploading results to gs://{GCS_BUCKET}/results/")
        upload_dir_to_gcs(save_dir, GCS_BUCKET, "results/models_test")
        upload_dir_to_gcs(log_dir, GCS_BUCKET, "results/logs_test")

    print(f"\n{'='*60}")
    print(f"  TEST RUN COMPLETE")
    print(f"{'='*60}")

    if args.local_test:
        print(f"\n  TensorBoard: tensorboard --logdir {log_dir}")
        print(f"  Model checkpoints: {save_dir}")


if __name__ == "__main__":
    main()

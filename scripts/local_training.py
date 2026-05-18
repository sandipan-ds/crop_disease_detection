"""
Local training script for crop disease detection (CNN baseline).

Runs on your local GPU (RTX 2060) with a subset of data for quick testing.
- Train + 5-fold CV on ~5,000 images
- Test evaluation on ~1,000 images
- TensorBoard logs saved to runs/local/

Usage:
    python scripts/local_training.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.dataset import CropDiseaseDataset
from src.model import build_model
from src.trainer import Trainer
from src.augmentations import get_train_transforms, get_eval_transforms


# =========================================================
# CONFIG
# =========================================================

CONFIG = {
    # Data
    "train_csv": str(PROJECT_ROOT / "notebook" / "train.csv"),
    "test_csv":  str(PROJECT_ROOT / "notebook" / "test.csv"),
    "data_root": str(PROJECT_ROOT),
    "train_subset": 5000,       # sample 5K for local testing
    "test_subset":  1000,       # sample 1K for local testing

    # Model
    "num_classes": 102,
    "img_size": 224,
    "dropout_conv": 0.25,
    "dropout_fc": 0.5,

    # Training
    "epochs": 10,               # fewer epochs for local testing
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


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("  LOCAL TRAINING — CNN BASELINE")
    print("  (Subset: 5K train, 1K test)")
    print("=" * 60)

    # --- Device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print("\n  ⚠ No GPU detected — training on CPU (will be slow)")

    # --- Transforms ---
    train_transform = get_train_transforms(
        img_size=CONFIG["img_size"],
        min_aug=CONFIG["min_aug"],
        max_aug=CONFIG["max_aug"],
    )
    eval_transform = get_eval_transforms(img_size=CONFIG["img_size"])

    print(f"\n  Train transforms: {train_transform}")
    print(f"  Eval transforms:  {eval_transform}")

    # --- Datasets ---
    print("\n  Loading datasets...")
    train_dataset = CropDiseaseDataset(
        csv_path=CONFIG["train_csv"],
        data_root=CONFIG["data_root"],
        transform=train_transform,
        sample_n=CONFIG["train_subset"],
    )
    test_dataset = CropDiseaseDataset(
        csv_path=CONFIG["test_csv"],
        data_root=CONFIG["data_root"],
        transform=eval_transform,
        sample_n=CONFIG["test_subset"],
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
    trainable_params = sum(p.numel() for p in model_temp.parameters() if p.requires_grad)
    print(f"\n  Model params: {total_params:,} total | {trainable_params:,} trainable")
    del model_temp

    # --- Directories ---
    log_dir = str(PROJECT_ROOT / "runs" / "local")
    save_dir = str(PROJECT_ROOT / "models" / "saved" / "local")

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
    # Use the best fold (highest F1)
    best_fold = max(range(1, CONFIG["n_folds"] + 1),
                    key=lambda f: fold_results[f - 1]["f1_macro"])
    print(f"\n  Using fold {best_fold} for test evaluation (highest CV F1)")

    trainer.evaluate_test(
        test_dataset=test_dataset,
        eval_transform=eval_transform,
        fold_to_use=best_fold,
    )

    # --- TensorBoard instructions ---
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"\n  To view metrics in TensorBoard:")
    print(f"    tensorboard --logdir {log_dir}")
    print(f"  Then open: http://localhost:6006")
    print(f"\n  Model checkpoints: {save_dir}")


if __name__ == "__main__":
    main()

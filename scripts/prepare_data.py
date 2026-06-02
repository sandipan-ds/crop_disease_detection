"""
Data preparation pipeline for crop disease detection.

Converts raw organized image folders into:
- Renamed, consistently-formatted images
- CSV manifests (image_path, image_name, target)
- label_mapping.json (class index mappings)

Usage:
    python scripts/prepare_data.py \
        --train-dir data/processed/combined_train \
        --test-dir data/processed/combined_test \
        --output-dir data/csv

This script is the automated extraction of the previously manual notebook cell
that processed datasets, renamed images, and generated CSV manifests.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder


# =========================================================
# CONFIG
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "combined_train"
DEFAULT_TEST_DIR = PROJECT_ROOT / "data" / "processed" / "combined_test"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


# =========================================================
# HELPERS
# =========================================================

def format_target_name(folder_name: str) -> str:
    """
    Convert folder names into clean, consistent labels.

    Example:
        "Apple Apple Scab"     -> "Apple_Apple_Scab"
        "Grape_Esca (Black Measles)" -> "Grape_Esca_Black_Measles"
    """
    folder_name = folder_name.replace(" ", "_")
    folder_name = folder_name.replace("(", "_")
    folder_name = folder_name.replace(")", "_")
    folder_name = folder_name.replace("-", "_")
    while "__" in folder_name:
        folder_name = folder_name.replace("__", "_")
    return folder_name.strip("_")


def process_dataset(dataset_folder: Path, base_dir: Path = PROJECT_ROOT) -> pd.DataFrame:
    """
    Rename images to a consistent scheme and build a manifest DataFrame.

    Renaming scheme: {target_label}_{idx:04d}.{ext}
        e.g. Apple_Apple_Scab_0001.jpg

    Returns DataFrame with columns:
        - image_name: renamed filename
        - image_path: path relative to project root (forward slashes)
        - target:     string class label
    """
    dataset_records = []
    print(f"\n  Processing: {dataset_folder}")

    # Iterate through class folders
    class_folders = sorted([d for d in dataset_folder.iterdir() if d.is_dir()])
    if not class_folders:
        raise ValueError(f"No class folders found in {dataset_folder}")

    for class_folder in class_folders:
        target_label = format_target_name(class_folder.name)

        # Collect image files
        image_files = sorted([
            img for img in class_folder.iterdir()
            if img.suffix.lower() in IMAGE_EXTENSIONS and img.is_file()
        ])

        print(f"    {class_folder.name:50s} -> {len(image_files):4d} images")

        # Rename images to consistent scheme
        for idx, image_path in enumerate(image_files, start=1):
            extension = image_path.suffix.lower()
            new_name = f"{target_label}_{idx:04d}{extension}"
            new_path = class_folder / new_name

            if image_path.name != new_name:
                try:
                    shutil.move(str(image_path), str(new_path))
                except PermissionError:
                    print(f"      Skipped locked file: {image_path.name}")
                    continue

            # Relative path from project root, always forward slashes
            relative_path = new_path.relative_to(base_dir).as_posix()

            dataset_records.append({
                "image_name": new_name,
                "image_path": relative_path,
                "target": target_label,
            })

    df = pd.DataFrame(dataset_records)
    return df


def generate_label_mapping(train_df: pd.DataFrame, output_path: Path) -> dict:
    """
    Fit a LabelEncoder on training targets and save the mapping to JSON.

    Returns mapping dict with keys:
        - num_classes
        - label_to_class:  {str(index): class_name}
        - class_to_label:  {class_name: index}
    """
    le = LabelEncoder()
    le.fit(train_df["target"])

    class_names = le.classes_.tolist()
    num_classes = len(class_names)

    label_to_class = {str(idx): name for idx, name in enumerate(class_names)}
    class_to_label = {name: idx for idx, name in enumerate(class_names)}

    mapping = {
        "num_classes": num_classes,
        "label_to_class": label_to_class,
        "class_to_label": class_to_label,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=4)

    print(f"\n  Label mapping saved: {output_path}")
    print(f"  Number of classes:   {num_classes}")
    return mapping


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare crop disease dataset: rename images and generate CSV manifests."
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=DEFAULT_TRAIN_DIR,
        help=f"Path to training image folder (default: {DEFAULT_TRAIN_DIR})",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=DEFAULT_TEST_DIR,
        help=f"Path to test image folder (default: {DEFAULT_TEST_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write CSVs and label_mapping.json (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-rename",
        action="store_true",
        help="Skip image renaming (useful if images are already renamed)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  DATA PREPARATION PIPELINE")
    print("=" * 60)

    # Validate input directories exist
    if not args.train_dir.exists():
        print(f"\n  ERROR: Train directory does not exist: {args.train_dir}")
        sys.exit(1)
    if not args.test_dir.exists():
        print(f"\n  ERROR: Test directory does not exist: {args.test_dir}")
        sys.exit(1)

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Process train dataset
    # -----------------------------------------------------
    print("\n  [1/4] Processing TRAIN dataset...")
    train_df = process_dataset(args.train_dir)
    train_csv_path = args.output_dir / "train.csv"
    train_df.to_csv(train_csv_path, index=False)
    print(f"\n  Train CSV saved: {train_csv_path}")
    print(f"  Total train samples: {len(train_df)}")

    # -----------------------------------------------------
    # Process test dataset
    # -----------------------------------------------------
    print("\n  [2/4] Processing TEST dataset...")
    test_df = process_dataset(args.test_dir)
    test_csv_path = args.output_dir / "test.csv"
    test_df.to_csv(test_csv_path, index=False)
    print(f"\n  Test CSV saved: {test_csv_path}")
    print(f"  Total test samples: {len(test_df)}")

    # -----------------------------------------------------
    # Label encoding & mapping
    # -----------------------------------------------------
    print("\n  [3/4] Building label mapping...")
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["target"])

    train_df["label"] = label_encoder.transform(train_df["target"])
    test_df["label"] = label_encoder.transform(test_df["target"])

    # Save updated CSVs with label column
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    print(f"  Updated CSVs with 'label' column")

    # Generate label_mapping.json
    configs_dir = PROJECT_ROOT / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = configs_dir / "label_mapping.json"
    generate_label_mapping(train_df, mapping_path)

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------
    print("\n  [4/4] Summary")
    print("  " + "-" * 50)
    print(f"  Train samples:       {len(train_df):,}")
    print(f"  Test samples:        {len(test_df):,}")
    print(f"  Classes:             {train_df['target'].nunique()}")
    print(f"  Output CSVs:         {args.output_dir}")
    print(f"  Label mapping:       {mapping_path}")
    print("  " + "-" * 50)
    print("\n  Data preparation complete.")
    print("  Next: run `python scripts/validate_data.py` to verify.\n")


if __name__ == "__main__":
    main()

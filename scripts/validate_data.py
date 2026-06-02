"""
Lightweight data validation for crop disease detection datasets.

Validates CSV manifests and their referenced images without heavy frameworks.
Run after prepare_data.py and before training.

Usage:
    python scripts/validate_data.py \
        --train-csv data/csv/train.csv \
        --test-csv data/csv/test.csv \
        --data-root . \
        --label-mapping configs/label_mapping.json

Exit code 0 = all checks passed.
Exit code 1 = one or more checks failed.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


# =========================================================
# CHECKERS
# =========================================================

def check_file_exists(path: Path, description: str) -> bool:
    """Verify a file exists."""
    if not path.exists():
        print(f"  [FAIL] {description} not found: {path}")
        return False
    print(f"  [PASS] {description} exists: {path}")
    return True


def check_required_columns(df: pd.DataFrame, required: list, context: str) -> bool:
    """Verify DataFrame has all required columns."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  [FAIL] {context} missing columns: {missing}")
        return False
    print(f"  [PASS] {context} has all required columns: {required}")
    return True


def check_images_exist(df: pd.DataFrame, data_root: Path, context: str) -> tuple:
    """
    Verify every image_path in the DataFrame resolves to an existing file.
    Returns (ok_count, missing_count).
    """
    missing = []
    for idx, row in df.iterrows():
        img_path = data_root / row["image_path"]
        if not img_path.exists():
            missing.append((idx, row["image_path"]))

    if missing:
        print(f"  [FAIL] {context}: {len(missing)} / {len(df)} images MISSING")
        for idx, path in missing[:5]:
            print(f"         row {idx}: {path}")
        if len(missing) > 5:
            print(f"         ... and {len(missing) - 5} more")
        return len(df) - len(missing), len(missing)

    print(f"  [PASS] {context}: all {len(df)} images exist")
    return len(df), 0


def check_no_duplicates(df: pd.DataFrame, column: str, context: str) -> bool:
    """Verify no duplicate values in a column."""
    dups = df[column].duplicated().sum()
    if dups > 0:
        print(f"  [FAIL] {context}: {dups} duplicate '{column}' entries")
        return False
    print(f"  [PASS] {context}: no duplicate '{column}' entries")
    return True


def check_class_consistency(train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    """Verify train and test share the same class vocabulary."""
    train_classes = set(train_df["target"].unique())
    test_classes = set(test_df["target"].unique())

    missing_in_test = train_classes - test_classes
    missing_in_train = test_classes - train_classes

    ok = True
    if missing_in_test:
        print(f"  [WARN] Classes in train but missing in test: {sorted(missing_in_test)}")
        ok = False
    if missing_in_train:
        print(f"  [FAIL] Classes in test but missing in train: {sorted(missing_in_train)}")
        ok = False

    if ok:
        print(f"  [PASS] Train and test share the same {len(train_classes)} classes")
    return ok


def check_label_mapping_consistency(train_df: pd.DataFrame, mapping_path: Path) -> bool:
    """Verify label_mapping.json matches the classes in train.csv."""
    with open(mapping_path) as f:
        mapping = json.load(f)

    expected_classes = set(train_df["target"].unique())
    mapped_classes = set(mapping.get("class_to_label", {}).keys())

    missing = expected_classes - mapped_classes
    extra = mapped_classes - expected_classes

    ok = True
    if missing:
        print(f"  [FAIL] Classes in train.csv missing from label_mapping: {sorted(missing)}")
        ok = False
    if extra:
        print(f"  [WARN] Extra classes in label_mapping not in train.csv: {sorted(extra)}")
    if ok:
        print(f"  [PASS] label_mapping.json consistent with train.csv ({len(expected_classes)} classes)")
    return ok


def check_path_format(df: pd.DataFrame, context: str) -> bool:
    """Verify image_path uses forward slashes (cross-platform)."""
    bad = df["image_path"].str.contains(r"\\", regex=True).sum()
    if bad > 0:
        print(f"  [WARN] {context}: {bad} paths contain backslashes (should use '/')")
        return False
    print(f"  [PASS] {context}: all paths use forward slashes")
    return True


def check_class_distribution(df: pd.DataFrame, context: str, min_images: int = 10) -> bool:
    """Flag classes with very few samples."""
    counts = df["target"].value_counts()
    low = counts[counts < min_images]
    if not low.empty:
        print(f"  [WARN] {context}: {len(low)} classes have < {min_images} images")
        for cls, cnt in low.items():
            print(f"         {cls}: {cnt}")
        return False
    print(f"  [PASS] {context}: all classes have >= {min_images} images")
    return True


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Validate crop disease dataset manifests.")
    parser.add_argument("--train-csv", type=Path, default="data/csv/train.csv")
    parser.add_argument("--test-csv", type=Path, default="data/csv/test.csv")
    parser.add_argument("--data-root", type=Path, default=".",
                        help="Project root that image_path is relative to")
    parser.add_argument("--label-mapping", type=Path, default="configs/label_mapping.json")
    parser.add_argument("--min-class-size", type=int, default=10,
                        help="Minimum images per class before warning")
    args = parser.parse_args()

    print("=" * 60)
    print("  DATA VALIDATION")
    print("=" * 60)

    all_passed = True

    # --- File existence ---
    print("\n[1/7] File existence checks")
    all_passed &= check_file_exists(args.train_csv, "train.csv")
    all_passed &= check_file_exists(args.test_csv, "test.csv")
    all_passed &= check_file_exists(args.label_mapping, "label_mapping.json")

    if not all_passed:
        print("\n  Aborting: required files missing.\n")
        sys.exit(1)

    # --- Load CSVs ---
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    # --- Column checks ---
    print("\n[2/7] Schema checks")
    required = ["image_path", "target"]
    all_passed &= check_required_columns(train_df, required, "train.csv")
    all_passed &= check_required_columns(test_df, required, "test.csv")

    # --- Image existence ---
    print("\n[3/7] Image existence checks")
    ok_train, miss_train = check_images_exist(train_df, args.data_root, "train.csv")
    ok_test, miss_test = check_images_exist(test_df, args.data_root, "test.csv")
    if miss_train > 0 or miss_test > 0:
        all_passed = False

    # --- Duplicate checks ---
    print("\n[4/7] Duplicate checks")
    all_passed &= check_no_duplicates(train_df, "image_path", "train.csv")
    all_passed &= check_no_duplicates(test_df, "image_path", "test.csv")

    # --- Class consistency ---
    print("\n[5/7] Class consistency checks")
    all_passed &= check_class_consistency(train_df, test_df)

    # --- Label mapping consistency ---
    print("\n[6/7] Label mapping consistency")
    all_passed &= check_label_mapping_consistency(train_df, args.label_mapping)

    # --- Path format ---
    print("\n[7/7] Path format & distribution checks")
    check_path_format(train_df, "train.csv")   # warn only
    check_path_format(test_df, "test.csv")     # warn only
    check_class_distribution(train_df, "train.csv", args.min_class_size)
    check_class_distribution(test_df, "test.csv", args.min_class_size)

    # --- Summary ---
    print("\n" + "=" * 60)
    if all_passed:
        print("  RESULT: ALL CRITICAL CHECKS PASSED")
        print(f"  Train: {len(train_df):,} samples, {train_df['target'].nunique()} classes")
        print(f"  Test:  {len(test_df):,} samples, {test_df['target'].nunique()} classes")
        print("=" * 60 + "\n")
        sys.exit(0)
    else:
        print("  RESULT: ONE OR MORE CRITICAL CHECKS FAILED")
        print("  Fix the issues above before proceeding to training.")
        print("=" * 60 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

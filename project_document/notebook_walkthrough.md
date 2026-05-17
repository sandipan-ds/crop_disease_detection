# Notebook Walkthrough: `crop_disease_detection.ipynb`

## Overview

This Jupyter notebook implements a **complete data preparation and exploratory analysis pipeline** for a **multi-class crop disease image classification** project. The goal is to build a dataset of crop leaf images (healthy and diseased) that can later be used to train a deep learning model (likely PyTorch-based, given the GPU/CUDA checks) to detect diseases across **multiple plant species**.

---

## Step-by-Step Breakdown

### 1. Environment Setup & GPU Verification
The notebook begins by importing core libraries (`numpy`, `pandas`, `matplotlib`, `seaborn`, `torch`, `tqdm`, etc.) and verifying that **CUDA/GPU** is available.

- **Hardware detected:** NVIDIA GeForce RTX 2060, PyTorch 2.12.0+cu132

This confirms the project is set up for GPU-accelerated training.

---

### 2. Data Preparation — Merging Multiple Source Datasets
> *Markdown header: "1. Data preparation"*

The first code cell (commented out, already executed) **consolidates images from two separate source datasets** into a unified train/test split:

| Source Dataset | Description |
|---|---|
| `plant_dataset_2` | Covers crops like Bean, BlackGram, Cotton, Cucumber, Onion, Pumpkin, Ragi, Rice, Sugarcane, Wheat |
| `plantvillage dataset` | Covers Apple, Cherry, Corn, Grape, Orange, Peach, PepperBell, Potato, Soybean, Strawberry, Tomato |

**Splitting strategy (per class):**
- **Train:** `min(500, 80% of total images)`
- **Test:** `min(300, remaining images)`

Images are copied into `data/processed/combined_train/` and `data/processed/combined_test/` directories, organized by class name.

**Result:** 108 total classes across both datasets, ~42,548 training samples.

---

### 3. Statistical Quality Control — PAC/Hoeffding Bound Analysis
The second code cell applies **Hoeffding's inequality (PAC learning theory)** to determine a **minimum sample threshold** for each class, ensuring that empirical training error stays within an acceptable bound of the true generalization error.

**Formula used:**
```
n_min = log(2/δ) / (2 × ε²)
```

**Working threshold selected:** `ε = 0.10, δ = 0.05` → **185 training samples minimum**

This means any class with **fewer than 185 training samples** is statistically unreliable and should be discarded.

| Metric | Value |
|---|---|
| Total classes before filtering | 108 |
| Classes kept (≥ 185 train) | 96 |
| Classes discarded (< 185 train) | 12 |
| Train samples retained | 40,996 (96.4%) |
| Train samples discarded | 1,552 (3.6%) |

A CSV with per-class epsilon bounds is saved to `data/pac_analysis_results.csv`.

---

### 4. Removing Under-Represented Classes (with exceptions)
The third code cell **physically removes** classes with fewer than 170 training samples from the processed directories — **except for healthy classes**, which are preserved regardless of size (e.g., `Potato_Healthy` with only 121 samples is kept).

> [!IMPORTANT]
> The threshold used here (170) is slightly different from the PAC-derived 185, making this a more lenient pass. Some borderline classes like `BlackGram_Healthy` (176) and `BlackGram_Anthracnose` (184) survive this step.

**Classes removed:** 6 non-healthy classes
**Healthy classes preserved despite low count:** 1 (`Potato_Healthy`)

**Final dataset:** 102 classes across 20 plant types

---

### 5. Image Renaming & CSV Manifest Creation
The fourth code cell:

1. **Renames all images** using a consistent naming convention: `{ClassName}_{NNNN}.{ext}` (e.g., `Apple_Apple_Scab_0001.jpg`)
2. **Creates CSV manifest files** (`train.csv` and `test.csv`) with columns:
   - `image_name` — the renamed filename
   - `image_path` — relative path from project root
   - `target` — the class label

| Split | Total Samples |
|---|---|
| Train | 42,006 |
| Test | 19,167 |

---

### 6. Exploratory Visualization — Class Distribution Charts
> *Markdown header: "2. Visualizing the number of samples for healthy and diseased fruits and vegetable plants"*

The final code cell generates **bar charts** showing the number of training and test samples per class, grouped by plant type. These appear as embedded images in the notebook output (4 separate charts covering different plant groups).

---

## Summary

| Stage | What It Does |
|---|---|
| **GPU check** | Confirms CUDA availability for future model training |
| **Data merge** | Combines two source datasets into a unified train/test split with caps (500 train, 300 test per class) |
| **PAC analysis** | Uses Hoeffding's inequality to identify statistically unreliable classes |
| **Class pruning** | Removes small non-healthy classes; preserves healthy classes even if small |
| **CSV creation** | Standardizes filenames and creates train/test manifest CSVs |
| **Visualization** | Bar charts of per-class sample counts |

> [!NOTE]
> All code cells after the first are **commented out** with their outputs preserved, indicating these steps have already been executed and the notebook is being kept as documentation of the pipeline.

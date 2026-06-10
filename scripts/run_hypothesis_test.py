"""
Offline Hypothesis Testing — Compare Trained Models Statistically.

Loads saved model checkpoints (NO retraining), runs inference on the test
set, and performs rigorous pairwise statistical comparisons:

    1. McNemar's Test:  Tests if two models have significantly different
                        error rates using per-image correctness.
    2. Bootstrap CI:    Estimates 95% confidence interval for the F1-score
                        difference via 1,000× resampling.

With 5 models, there are C(5,2) = 10 pairwise comparisons.
Bonferroni-corrected significance threshold: 0.05 / 10 = 0.005.

All results are logged to Weights & Biases (W&B) for visualization.

Usage:
    # Full test set (all 5 models — ~15-20 min on GPU, ~2h on CPU)
    python scripts/run_hypothesis_test.py \\
        --models resnet_50 mobilenet_v3 resnet_152 vit swin_base

    # Quick smoke test (100 images — ~1 min)
    python scripts/run_hypothesis_test.py \\
        --models resnet_50 mobilenet_v3 \\
        --limit 100

    # Compare just two models
    python scripts/run_hypothesis_test.py \\
        --models resnet_50 resnet_152

Prerequisites:
    pip install wandb scipy
    wandb login  # one-time setup
"""

import os
import sys
import json
import glob
from datetime import datetime
import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Tuple

# Fix Windows console encoding (cp1252 → UTF-8)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env (for WANDB_API_KEY)
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    with open(_env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

from src.model import get_model
from src.dataset import CropDiseaseDataset
from src.augmentations import get_eval_transforms


# ─────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────

RESULTS_DIR = PROJECT_ROOT / "results"
TEST_CSV = PROJECT_ROOT / "data" / "csv" / "test.csv"
WANDB_PROJECT = "crop-disease-detection"
NUM_CLASSES = 102
IMAGE_SIZE = 224
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42

# Map model names to their checkpoint directories
# (handles the ViT folder name quirk: "ViT (B-16)/16)/models/")
CHECKPOINT_MAP = {
    "resnet_50":      RESULTS_DIR / "resnet_50" / "models" / "best_model_fold_1.pth",
    "resnet_152":     RESULTS_DIR / "resnet_152" / "models" / "best_model_fold_1.pth",
    "mobilenet_v3":   RESULTS_DIR / "mobilenet_v3" / "models" / "best_model_fold_1.pth",
    "vit":            RESULTS_DIR / "ViT (B-16)" / "16)" / "models" / "best_model_fold_1.pth",
    "swin_base":      RESULTS_DIR / "swin_base" / "models" / "best_model_fold_1.pth",
    "efficientnet_b4": RESULTS_DIR / "efficientnet_b4" / "models" / "best_model_fold_1.pth",
    "vgg_16":         RESULTS_DIR / "vgg_16" / "models" / "best_model_fold_1.pth",
    "cnn_baseline":   RESULTS_DIR / "cnn_baseline" / "models" / "best_model_fold_1.pth",
}

# F1 scores from documentation (for display reference)
MODEL_F1_REFERENCE = {
    "resnet_50": 0.9360, "resnet_152": 0.9519, "mobilenet_v3": 0.9231,
    "vit": 0.9177, "swin_base": 0.9271, "efficientnet_b4": 0.8942,
    "vgg_16": 0.8708, "cnn_baseline": 0.8309,
}


# ─────────────────────────────────────────────────────
# Model Loading (from saved checkpoints — NO retraining)
# ─────────────────────────────────────────────────────

def load_model_from_checkpoint(model_name: str, device: torch.device) -> torch.nn.Module:
    """
    Load a trained model from its saved checkpoint.

    This does NOT retrain. It:
      1. Creates the model architecture with pretrained=False (no ImageNet download).
      2. Loads the saved weights from the .pth file.
      3. Sets the model to eval mode (no gradients, no dropout).
    """
    checkpoint_path = CHECKPOINT_MAP.get(model_name)
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found for '{model_name}' at {checkpoint_path}\n"
            f"Available checkpoints: {[k for k, v in CHECKPOINT_MAP.items() if v.exists()]}"
        )

    print(f"    Loading {model_name} from {checkpoint_path.relative_to(PROJECT_ROOT)}...")

    # Create model architecture (pretrained=False → no ImageNet download)
    model = get_model(model_name, num_classes=NUM_CLASSES, pretrained=False)

    # Load saved weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Disable inplace ReLU (required for some torch operations)
    for module in model.modules():
        if hasattr(module, "inplace"):
            module.inplace = False

    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"    [OK] Loaded {model_name}: {param_count / 1e6:.1f}M parameters")

    return model


# ─────────────────────────────────────────────────────
# Inference — Get per-image predictions
# ─────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    eval_transform: torch.nn.Module,
    device: torch.device,
    model_name: str,
) -> np.ndarray:
    """
    Run model inference on the entire test set.

    Returns:
        predictions: np.ndarray of shape (N,) with predicted class indices.
    """
    all_preds = []

    for images, _ in tqdm(dataloader, desc=f"    {model_name}", unit="batch"):
        images = images.to(device)
        images = eval_transform(images)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)

    return np.concatenate(all_preds)


# ─────────────────────────────────────────────────────
# Statistical Tests
# ─────────────────────────────────────────────────────

def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> Dict:
    """
    Perform McNemar's test on per-image correctness arrays.

    The 2x2 contingency table:
        |             | B correct | B wrong |
        |-------------|-----------|---------|
        | A correct   |   n00     |   n01   |
        | A wrong     |   n10     |   n11   |

    McNemar's test examines n01 vs n10 (the disagreement cells).
    Null hypothesis: both models have the same error rate.

    Args:
        correct_a: boolean array, True if model A got image i correct
        correct_b: boolean array, True if model B got image i correct

    Returns:
        dict with contingency table, chi2 statistic, and p-value
    """
    from scipy.stats import chi2

    # Build contingency table
    n00 = int(np.sum(correct_a & correct_b))       # both correct
    n01 = int(np.sum(correct_a & ~correct_b))      # only A correct
    n10 = int(np.sum(~correct_a & correct_b))      # only B correct
    n11 = int(np.sum(~correct_a & ~correct_b))     # both wrong

    # McNemar's chi-squared statistic (with continuity correction)
    if (n01 + n10) == 0:
        chi2_stat = 0.0
        p_value = 1.0
    else:
        chi2_stat = ((abs(n01 - n10) - 1) ** 2) / (n01 + n10)
        p_value = 1 - chi2.cdf(chi2_stat, df=1)

    return {
        "contingency_table": {
            "both_correct": n00,
            "only_A_correct": n01,
            "only_B_correct": n10,
            "both_wrong": n11,
        },
        "chi2_statistic": round(chi2_stat, 4),
        "p_value": p_value,
        "disagreements": n01 + n10,
    }


def bootstrap_f1_difference(
    true_labels: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    n_bootstrap: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> Dict:
    """
    Bootstrap confidence interval for F1(A) - F1(B).

    Resamples the test set n_bootstrap times with replacement,
    computes macro F1 for each model on each resample, and
    returns the distribution of differences.

    Args:
        true_labels: ground truth class indices
        preds_a: Model A predicted class indices
        preds_b: Model B predicted class indices
        n_bootstrap: number of bootstrap samples
        seed: random seed for reproducibility

    Returns:
        dict with mean difference, 95% CI, and full distribution
    """
    from sklearn.metrics import f1_score

    rng = np.random.RandomState(seed)
    n = len(true_labels)
    differences = []

    for _ in range(n_bootstrap):
        # Resample indices with replacement
        idx = rng.choice(n, size=n, replace=True)
        y_true = true_labels[idx]
        y_a = preds_a[idx]
        y_b = preds_b[idx]

        f1_a = f1_score(y_true, y_a, average="macro", zero_division=0)
        f1_b = f1_score(y_true, y_b, average="macro", zero_division=0)
        differences.append(f1_a - f1_b)

    differences = np.array(differences)
    ci_lower = float(np.percentile(differences, 2.5))
    ci_upper = float(np.percentile(differences, 97.5))
    mean_diff = float(np.mean(differences))
    ci_excludes_zero = (ci_lower > 0) or (ci_upper < 0)

    return {
        "mean_difference": round(mean_diff, 6),
        "ci_95_lower": round(ci_lower, 6),
        "ci_95_upper": round(ci_upper, 6),
        "ci_excludes_zero": ci_excludes_zero,
        "distribution": differences.tolist(),
    }


def compute_model_metrics(true_labels: np.ndarray, preds: np.ndarray) -> Dict:
    """Compute classification metrics for a single model."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {
        "accuracy": round(float(accuracy_score(true_labels, preds)), 6),
        "f1_macro": round(float(f1_score(true_labels, preds, average="macro", zero_division=0)), 6),
        "f1_weighted": round(float(f1_score(true_labels, preds, average="weighted", zero_division=0)), 6),
        "precision_macro": round(float(precision_score(true_labels, preds, average="macro", zero_division=0)), 6),
        "recall_macro": round(float(recall_score(true_labels, preds, average="macro", zero_division=0)), 6),
    }


# ─────────────────────────────────────────────────────
# W&B Logging
# ─────────────────────────────────────────────────────

def log_to_wandb(
    model_names: List[str],
    model_metrics: Dict[str, Dict],
    pairwise_results: List[Dict],
    n_comparisons: int,
    alpha_corrected: float,
    n_test_images: int,
):
    """Log all hypothesis testing results to W&B."""
    import wandb
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Initialize W&B run
    run = wandb.init(
        project=WANDB_PROJECT,
        name="hypothesis-test",
        tags=["hypothesis-test", "offline", "statistical-comparison"],
        job_type="hypothesis-test",
        config={
            "models_compared": model_names,
            "n_models": len(model_names),
            "n_comparisons": n_comparisons,
            "bonferroni_alpha": alpha_corrected,
            "bootstrap_samples": BOOTSTRAP_N,
            "test_set_size": n_test_images,
        },
    )

    # ── 1. Model Summary Table ──
    summary_table = wandb.Table(columns=[
        "Model", "Accuracy", "F1 (Macro)", "F1 (Weighted)",
        "Precision (Macro)", "Recall (Macro)",
    ])
    for name in model_names:
        m = model_metrics[name]
        summary_table.add_data(
            name, m["accuracy"], m["f1_macro"], m["f1_weighted"],
            m["precision_macro"], m["recall_macro"],
        )
    wandb.log({"model_summary": summary_table})

    # ── 2. Pairwise Comparison Table ──
    comparison_table = wandb.Table(columns=[
        "Model A", "Model B",
        "F1(A)", "F1(B)", "F1 Diff",
        "McNemar χ²", "p-value", "Significant?",
        "Bootstrap CI Lower", "Bootstrap CI Upper", "CI Excludes 0?",
        "Both Correct", "Only A Correct", "Only B Correct", "Both Wrong",
    ])

    for result in pairwise_results:
        mc = result["mcnemar"]
        bs = result["bootstrap"]
        ct = mc["contingency_table"]
        comparison_table.add_data(
            result["model_a"], result["model_b"],
            result["f1_a"], result["f1_b"],
            round(result["f1_a"] - result["f1_b"], 6),
            mc["chi2_statistic"], round(mc["p_value"], 6),
            "✅ Yes" if mc["p_value"] < alpha_corrected else "❌ No",
            bs["ci_95_lower"], bs["ci_95_upper"],
            "✅ Yes" if bs["ci_excludes_zero"] else "❌ No",
            ct["both_correct"], ct["only_A_correct"],
            ct["only_B_correct"], ct["both_wrong"],
        )
    wandb.log({"pairwise_comparisons": comparison_table})

    # ── 3. Bootstrap Distribution Histograms ──
    n_pairs = len(pairwise_results)
    cols = min(3, n_pairs)
    rows = (n_pairs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    if n_pairs == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for i, result in enumerate(pairwise_results):
        ax = axes[i // cols, i % cols]
        dist = result["bootstrap"]["distribution"]
        bs = result["bootstrap"]
        mc = result["mcnemar"]

        ax.hist(dist, bins=50, alpha=0.7, color="#4e79a7", edgecolor="white", linewidth=0.5)
        ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="No difference")
        ax.axvline(x=bs["mean_difference"], color="#59a14f", linestyle="-", linewidth=1.5, label=f"Mean: {bs['mean_difference']:.4f}")
        ax.axvline(x=bs["ci_95_lower"], color="orange", linestyle=":", linewidth=1.2)
        ax.axvline(x=bs["ci_95_upper"], color="orange", linestyle=":", linewidth=1.2, label=f"95% CI")

        sig_marker = "★" if mc["p_value"] < alpha_corrected else ""
        ax.set_title(f"{result['model_a']} vs {result['model_b']} {sig_marker}\np={mc['p_value']:.4f}", fontsize=10)
        ax.set_xlabel("F1(A) − F1(B)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=8)

    # Hide unused subplots
    for i in range(n_pairs, rows * cols):
        axes[i // cols, i % cols].set_visible(False)

    plt.tight_layout()
    wandb.log({"bootstrap_distributions": wandb.Image(fig)})
    plt.close(fig)

    # ── 4. Summary Metrics ──
    for result in pairwise_results:
        key = f"{result['model_a']}_vs_{result['model_b']}"
        wandb.summary[f"{key}/p_value"] = result["mcnemar"]["p_value"]
        wandb.summary[f"{key}/chi2"] = result["mcnemar"]["chi2_statistic"]
        wandb.summary[f"{key}/significant"] = result["mcnemar"]["p_value"] < alpha_corrected
        wandb.summary[f"{key}/f1_diff"] = result["f1_a"] - result["f1_b"]
        wandb.summary[f"{key}/ci_lower"] = result["bootstrap"]["ci_95_lower"]
        wandb.summary[f"{key}/ci_upper"] = result["bootstrap"]["ci_95_upper"]

    run_url = run.get_url()
    run.finish()
    return run_url


# ─────────────────────────────────────────────────────
# Local File Saving
# ─────────────────────────────────────────────────────

HYPOTHESIS_RESULTS_DIR = PROJECT_ROOT / "hypothesis_testing"
JSON_RESULTS_DIR = HYPOTHESIS_RESULTS_DIR / "json_results"
BOOTSTRAP_PLOTS_DIR = HYPOTHESIS_RESULTS_DIR / "bootstrap_distributions"
MODEL_RANKING_DIR = HYPOTHESIS_RESULTS_DIR / "model_ranking"
PVALUE_HEATMAP_DIR = HYPOTHESIS_RESULTS_DIR / "pvalue_heatmap"


def _json_default(obj):
    """Handle numpy types during JSON serialization."""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_results_locally(
    model_names: List[str],
    model_metrics: Dict[str, Dict],
    pairwise_results: List[Dict],
    n_comparisons: int,
    alpha_corrected: float,
    n_test_images: int,
) -> Path:
    """
    Save hypothesis testing results as a JSON file.

    Naming convention:
        hypothesis_test_{YYYYMMDD_HHMMSS}_run_{N}.json

    The run number N auto-increments based on existing files.
    """
    # Create directory
    JSON_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Determine run number
    existing = sorted(JSON_RESULTS_DIR.glob("hypothesis_test_*.json"))
    run_number = len(existing) + 1

    # Build filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hypothesis_test_{timestamp}_run_{run_number}.json"
    filepath = JSON_RESULTS_DIR / filename

    # Build results dict (strip bootstrap distribution to keep file small)
    pairwise_clean = []
    for result in pairwise_results:
        pair = {
            "model_a": result["model_a"],
            "model_b": result["model_b"],
            "f1_a": result["f1_a"],
            "f1_b": result["f1_b"],
            "f1_difference": round(result["f1_a"] - result["f1_b"], 6),
            "mcnemar": result["mcnemar"],
            "bootstrap": {
                "mean_difference": result["bootstrap"]["mean_difference"],
                "ci_95_lower": result["bootstrap"]["ci_95_lower"],
                "ci_95_upper": result["bootstrap"]["ci_95_upper"],
                "ci_excludes_zero": result["bootstrap"]["ci_excludes_zero"],
                # Omit full distribution array to keep file small
            },
            "significant": result["mcnemar"]["p_value"] < alpha_corrected,
        }
        pairwise_clean.append(pair)

    results_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "run_number": run_number,
            "test_set_size": n_test_images,
            "n_models": len(model_names),
            "n_comparisons": n_comparisons,
            "bonferroni_alpha": alpha_corrected,
            "bootstrap_samples": BOOTSTRAP_N,
        },
        "models_compared": model_names,
        "model_metrics": model_metrics,
        "pairwise_comparisons": pairwise_clean,
        "summary": {
            "significant_pairs": sum(
                1 for r in pairwise_results
                if r["mcnemar"]["p_value"] < alpha_corrected
            ),
            "total_pairs": n_comparisons,
            "ranking": [
                {"rank": i + 1, "model": name, "f1_macro": m["f1_macro"]}
                for i, (name, m) in enumerate(
                    sorted(model_metrics.items(),
                           key=lambda x: x[1]["f1_macro"], reverse=True)
                )
            ],
        },
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False, default=_json_default)

    return filepath


def save_plots_locally(
    model_names: List[str],
    model_metrics: Dict[str, Dict],
    pairwise_results: List[Dict],
    alpha_corrected: float,
    run_label: str,
) -> List[Path]:
    """
    Generate and save visualization plots as PNG files.

    Generates 3 charts:
      1. Model Ranking Bar Chart (F1 + Accuracy)
      2. Pairwise P-value Heatmap
      3. Bootstrap Distribution Histograms

    Returns:
        List of saved file paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    MODEL_RANKING_DIR.mkdir(parents=True, exist_ok=True)
    PVALUE_HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    BOOTSTRAP_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    saved_files = []

    # Color palette
    COLORS = {
        "primary": "#4e79a7",
        "secondary": "#f28e2b",
        "accent": "#e15759",
        "green": "#59a14f",
        "purple": "#b07aa1",
        "grid": "#e0e0e0",
    }

    # ── 1. Model Ranking Bar Chart ──
    ranked = sorted(model_metrics.items(), key=lambda x: x[1]["f1_macro"], reverse=True)
    names = [n for n, _ in ranked]
    f1_scores = [m["f1_macro"] for _, m in ranked]
    acc_scores = [m["accuracy"] for _, m in ranked]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, f1_scores, width, label="F1 (Macro)",
                   color=COLORS["primary"], edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width/2, acc_scores, width, label="Accuracy",
                   color=COLORS["secondary"], edgecolor="white", linewidth=0.8)

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.002, f"{h:.4f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.002, f"{h:.4f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Performance Ranking", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(min(min(f1_scores), min(acc_scores)) - 0.05, 1.02)
    ax.grid(axis="y", alpha=0.3, color=COLORS["grid"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    path1 = MODEL_RANKING_DIR / f"plot_model_ranking_{run_label}.png"
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_files.append(path1)

    # ── 2. Pairwise P-value Heatmap ──
    n = len(model_names)
    p_matrix = np.ones((n, n))
    sig_matrix = np.zeros((n, n), dtype=bool)

    for result in pairwise_results:
        i = model_names.index(result["model_a"])
        j = model_names.index(result["model_b"])
        p_val = result["mcnemar"]["p_value"]
        p_matrix[i, j] = p_val
        p_matrix[j, i] = p_val
        is_sig = p_val < alpha_corrected
        sig_matrix[i, j] = is_sig
        sig_matrix[j, i] = is_sig

    fig, ax = plt.subplots(figsize=(8, 7))

    # Custom colormap: green (significant) to red (not significant)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "sig_cmap", ["#2d6a4f", "#40916c", "#95d5b2", "#ffd166", "#ef476f"], N=256
    )

    im = ax.imshow(p_matrix, cmap=cmap, vmin=0, vmax=0.1, aspect="equal")

    # Add text annotations
    for i in range(n):
        for j in range(n):
            if i == j:
                text = "-"
                color = "gray"
            else:
                p_val = p_matrix[i, j]
                text = f"{p_val:.4f}"
                if sig_matrix[i, j]:
                    text += "\n*SIG*"
                color = "white" if p_val < 0.05 else "black"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=8, fontweight="bold" if sig_matrix[i, j] else "normal",
                    color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(model_names, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(model_names, fontsize=9)
    ax.set_title(f"McNemar's Test P-values\n(Bonferroni a = {alpha_corrected:.4f})",
                 fontsize=13, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("p-value", fontsize=10)
    plt.tight_layout()

    path2 = PVALUE_HEATMAP_DIR / f"plot_pvalue_heatmap_{run_label}.png"
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_files.append(path2)

    # ── 3. Bootstrap Distribution Histograms ──
    n_pairs = len(pairwise_results)
    cols = min(3, n_pairs)
    rows = max(1, (n_pairs + cols - 1) // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    if n_pairs == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for i, result in enumerate(pairwise_results):
        ax = axes[i // cols, i % cols]
        dist = result["bootstrap"]["distribution"]
        bs = result["bootstrap"]
        mc = result["mcnemar"]

        ax.hist(dist, bins=50, alpha=0.75, color=COLORS["primary"],
                edgecolor="white", linewidth=0.5)
        ax.axvline(x=0, color=COLORS["accent"], linestyle="--",
                   linewidth=2, label="No difference")
        ax.axvline(x=bs["mean_difference"], color=COLORS["green"],
                   linestyle="-", linewidth=2,
                   label=f"Mean: {bs['mean_difference']:.4f}")
        ax.axvspan(bs["ci_95_lower"], bs["ci_95_upper"],
                   alpha=0.15, color=COLORS["secondary"], label="95% CI")
        ax.axvline(x=bs["ci_95_lower"], color=COLORS["secondary"],
                   linestyle=":", linewidth=1.5)
        ax.axvline(x=bs["ci_95_upper"], color=COLORS["secondary"],
                   linestyle=":", linewidth=1.5)

        sig_marker = "  [SIGNIFICANT]" if mc["p_value"] < alpha_corrected else ""
        ax.set_title(
            f"{result['model_a']} vs {result['model_b']}{sig_marker}\n"
            f"p={mc['p_value']:.4f}  |  CI=[{bs['ci_95_lower']:.4f}, {bs['ci_95_upper']:.4f}]",
            fontsize=9, fontweight="bold"
        )
        ax.set_xlabel("F1(A) - F1(B)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused subplots
    for i in range(n_pairs, rows * cols):
        axes[i // cols, i % cols].set_visible(False)

    plt.suptitle("Bootstrap F1 Difference Distributions",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    path3 = BOOTSTRAP_PLOTS_DIR / f"plot_bootstrap_distributions_{run_label}.png"
    fig.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_files.append(path3)

    return saved_files


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Offline Hypothesis Testing — Compare trained models statistically",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all 5 production models (full test set)
  python scripts/run_hypothesis_test.py \\
      --models resnet_50 mobilenet_v3 resnet_152 vit swin_base

  # Quick smoke test (100 images)
  python scripts/run_hypothesis_test.py \\
      --models resnet_50 mobilenet_v3 --limit 100
        """,
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Model names to compare (e.g., resnet_50 mobilenet_v3 resnet_152 vit swin_base)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit test set to N images (for quick smoke tests). Default: use full test set.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for inference. Default: 64.",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Skip W&B logging (print results to console only).",
    )
    parser.add_argument(
        "--save-plots", action="store_true",
        help="Save visualization plots (model ranking, p-value heatmap, bootstrap histograms) as PNGs.",
    )
    args = parser.parse_args()

    # Validate model names
    for name in args.models:
        if name not in CHECKPOINT_MAP:
            print(f"ERROR: Unknown model '{name}'. Available: {list(CHECKPOINT_MAP.keys())}")
            sys.exit(1)

    if len(args.models) < 2:
        print("ERROR: Need at least 2 models to compare.")
        sys.exit(1)

    # Pairwise combinations
    pairs = list(itertools.combinations(args.models, 2))
    n_comparisons = len(pairs)
    alpha_corrected = 0.05 / n_comparisons  # Bonferroni correction

    print("=" * 65)
    print("  OFFLINE HYPOTHESIS TESTING -- Crop Disease Detection")
    print("=" * 65)
    print(f"\n  Models:       {', '.join(args.models)}")
    print(f"  Comparisons:  {n_comparisons} pairwise")
    print(f"  Bonferroni a: {alpha_corrected:.4f} (0.05 / {n_comparisons})")
    print(f"  Bootstrap:    {BOOTSTRAP_N} resamples")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device:       {device}")
    if device.type == "cuda":
        print(f"  GPU:          {torch.cuda.get_device_name(0)}")

    # ── Step 1: Load test dataset ──
    print(f"\n  Loading test dataset from {TEST_CSV.relative_to(PROJECT_ROOT)}...")
    test_dataset = CropDiseaseDataset(
        csv_path=str(TEST_CSV),
        data_root=str(PROJECT_ROOT),
        sample_n=args.limit,
    )
    n_test = len(test_dataset)
    print(f"  Test images: {n_test}")

    dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # Ground truth labels
    true_labels = np.array(test_dataset.get_labels())

    # Eval transform (GPU pipeline)
    eval_transform = get_eval_transforms(img_size=IMAGE_SIZE).to(device)

    # ── Step 2: Load models and run inference ──
    print(f"\n  Loading checkpoints and running inference...")
    predictions: Dict[str, np.ndarray] = {}
    model_metrics: Dict[str, Dict] = {}

    for model_name in args.models:
        print(f"\n  -- {model_name} --")
        model = load_model_from_checkpoint(model_name, device)
        preds = run_inference(model, dataloader, eval_transform, device, model_name)
        predictions[model_name] = preds
        model_metrics[model_name] = compute_model_metrics(true_labels, preds)

        m = model_metrics[model_name]
        print(f"    Accuracy: {m['accuracy']:.4f}  |  F1 (macro): {m['f1_macro']:.4f}")

        # Free GPU memory
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Step 3: Pairwise statistical tests ──
    print(f"\n  Running {n_comparisons} pairwise statistical tests...")
    pairwise_results = []

    for model_a, model_b in pairs:
        preds_a = predictions[model_a]
        preds_b = predictions[model_b]
        correct_a = (preds_a == true_labels)
        correct_b = (preds_b == true_labels)

        # McNemar's test
        mc = mcnemar_test(correct_a, correct_b)

        # Bootstrap CI for F1 difference
        bs = bootstrap_f1_difference(true_labels, preds_a, preds_b)

        f1_a = model_metrics[model_a]["f1_macro"]
        f1_b = model_metrics[model_b]["f1_macro"]
        significant = mc["p_value"] < alpha_corrected

        pairwise_results.append({
            "model_a": model_a,
            "model_b": model_b,
            "f1_a": f1_a,
            "f1_b": f1_b,
            "mcnemar": mc,
            "bootstrap": bs,
        })

        sig_mark = "* SIGNIFICANT" if significant else "  not significant"
        print(f"\n    {model_a} vs {model_b}:")
        print(f"      F1: {f1_a:.4f} vs {f1_b:.4f} (delta = {f1_a - f1_b:+.4f})")
        print(f"      McNemar's chi2 = {mc['chi2_statistic']:.4f}, p = {mc['p_value']:.6f}  {sig_mark}")
        print(f"      Bootstrap 95% CI: [{bs['ci_95_lower']:.4f}, {bs['ci_95_upper']:.4f}]")
        ct = mc["contingency_table"]
        print(f"      Contingency: both_ok={ct['both_correct']}, onlyA_ok={ct['only_A_correct']}, "
              f"onlyB_ok={ct['only_B_correct']}, both_wrong={ct['both_wrong']}")

    # ── Step 4: Log to W&B ──
    if not args.no_wandb:
        print(f"\n  Logging results to W&B (project: {WANDB_PROJECT})...")
        try:
            run_url = log_to_wandb(
                model_names=args.models,
                model_metrics=model_metrics,
                pairwise_results=pairwise_results,
                n_comparisons=n_comparisons,
                alpha_corrected=alpha_corrected,
                n_test_images=n_test,
            )
            print(f"  [OK] W&B run: {run_url}")
        except Exception as e:
            print(f"  [WARN] W&B logging failed: {e}")
            print("     Results were printed above. Use --no-wandb to skip W&B.")
    else:
        print("\n  Skipping W&B logging (--no-wandb flag).")

    # ── Step 5: Save results locally ──
    print(f"\n  Saving results locally...")
    try:
        result_file = save_results_locally(
            model_names=args.models,
            model_metrics=model_metrics,
            pairwise_results=pairwise_results,
            n_comparisons=n_comparisons,
            alpha_corrected=alpha_corrected,
            n_test_images=n_test,
        )
        print(f"  [OK] Saved: {result_file.relative_to(PROJECT_ROOT)}")
    except Exception as e:
        print(f"  [WARN] Local save failed: {e}")

    # -- Step 6: Save plots --
    if args.save_plots:
        print(f"\n  Generating visualization plots...")
        try:
            # Use same run label as JSON for matching
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            existing_plots = sorted(MODEL_RANKING_DIR.glob("plot_model_ranking_*.png"))
            plot_run = len(existing_plots) + 1
            run_label = f"{timestamp}_run_{plot_run}"

            plot_files = save_plots_locally(
                model_names=args.models,
                model_metrics=model_metrics,
                pairwise_results=pairwise_results,
                alpha_corrected=alpha_corrected,
                run_label=run_label,
            )
            for pf in plot_files:
                print(f"  [OK] Plot: {pf.relative_to(PROJECT_ROOT)}")
        except Exception as e:
            print(f"  [WARN] Plot generation failed: {e}")

    # ── Summary ──
    print(f"\n{'=' * 65}")
    print(f"  HYPOTHESIS TESTING COMPLETE")
    print(f"{'=' * 65}")
    print(f"\n  Models tested:   {len(args.models)}")
    print(f"  Test images:     {n_test}")
    print(f"  Pairwise tests:  {n_comparisons}")
    print(f"  Bonferroni a:    {alpha_corrected:.4f}")

    sig_count = sum(1 for r in pairwise_results if r["mcnemar"]["p_value"] < alpha_corrected)
    print(f"  Significant:     {sig_count} / {n_comparisons}")

    print(f"\n  Model Rankings (by F1 Macro):")
    ranked = sorted(model_metrics.items(), key=lambda x: x[1]["f1_macro"], reverse=True)
    for rank, (name, m) in enumerate(ranked, 1):
        print(f"    {rank}. {name:20s}  F1={m['f1_macro']:.4f}  Acc={m['accuracy']:.4f}")


if __name__ == "__main__":
    main()

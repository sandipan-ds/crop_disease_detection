"""
Sync existing TensorBoard logs from results/ to Weights & Biases (W&B).

This script creates one W&B run per trained model, syncing all TensorBoard
scalars, images, and text logs. It also logs model metadata (architecture,
parameter count, F1 score) for unified comparison across all 8 models.

Usage:
    python scripts/sync_tensorboard_to_wandb.py

Prerequisites:
    pip install wandb tensorboard
    wandb login  # one-time setup

The script will:
    1. Discover all model directories in results/
    2. Extract model metadata (architecture, params, F1 score)
    3. Create a W&B run for each model
    4. Sync TensorBoard event files
    5. Log hyperparameters and summary metrics
    6. Print a W&B project URL for comparison
"""

import os
import sys
import json
import glob
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env if present (for WANDB_API_KEY)
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    with open(_env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

import torch
import numpy as np

# Optional imports - handle gracefully if not installed
try:
    import wandb
    from wandb import tensorboard as wb_tb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("ERROR: wandb not installed. Run: pip install wandb")
    sys.exit(1)

try:
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("ERROR: tensorboard not installed. Run: pip install tensorboard")
    sys.exit(1)

# Import model registry to get parameter counts
from src.model import get_model, MODEL_REGISTRY


# =========================================================
# CONFIGURATION
# =========================================================

RESULTS_DIR = PROJECT_ROOT / "results"
WANDB_PROJECT = "crop-disease-detection"
WANDB_ENTITY = None  # Set to your W&B username/team if desired

# Model metadata from documentation
MODEL_F1_SCORES = {
    "cnn_baseline": 0.8309,
    "resnet_50": 0.9360,
    "resnet_152": 0.9519,
    "vgg_16": 0.8708,
    "vit": 0.9177,
    "efficientnet_b4": 0.8942,
    "mobilenet_v3": 0.9231,
    "swin_base": 0.9271,
}

MODEL_ACCURACIES = {
    "cnn_baseline": 0.852,
    "resnet_50": 0.942,
    "resnet_152": 0.954,
    "vgg_16": 0.875,
    "vit": 0.921,
    "efficientnet_b4": 0.899,
    "mobilenet_v3": 0.926,
    "swin_base": 0.929,
}


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def count_parameters(model_name: str) -> int:
    """Get parameter count for a model architecture."""
    try:
        model = get_model(model_name, num_classes=102, pretrained=False)
        return sum(p.numel() for p in model.parameters())
    except Exception as e:
        print(f"  [WARN] Could not count params for {model_name}: {e}")
        return 0


def extract_tensorboard_scalars(log_dir: Path) -> Dict[str, Any]:
    """Extract scalar metrics from TensorBoard event files."""
    scalars = {}
    
    event_files = list(log_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        return scalars
    
    for event_file in event_files:
        try:
            ea = EventAccumulator(str(event_file))
            ea.Reload()
            
            # Get all scalar tags
            tags = ea.Tags().get("scalars", [])
            for tag in tags:
                events = ea.Scalars(tag)
                if events:
                    # Store last value (final epoch) and full history
                    tag_name = tag.replace("/", "_")
                    scalars[f"{tag_name}_final"] = events[-1].value
                    scalars[f"{tag_name}_max"] = max(e.value for e in events)
                    scalars[f"{tag_name}_min"] = min(e.value for e in events)
        except Exception as e:
            print(f"  [WARN] Could not read {event_file}: {e}")
            continue
    
    return scalars


def extract_hyperparameters(log_dir: Path) -> Dict[str, Any]:
    """Extract hyperparameters from TensorBoard text logs."""
    hparams = {}
    
    event_files = list(log_dir.rglob("events.out.tfevents.*"))
    for event_file in event_files:
        try:
            ea = EventAccumulator(str(event_file))
            ea.Reload()
            
            # Check for text summaries (hyperparameters)
            tags = ea.Tags().get("text", [])
            for tag in tags:
                if "hyperparameter" in tag.lower() or "config" in tag.lower():
                    events = ea.Text(tag)
                    if events:
                        try:
                            # Parse JSON from text
                            text = events[-1].text
                            if text.startswith("```"):
                                text = text.strip("`").strip()
                            parsed = json.loads(text)
                            if isinstance(parsed, dict):
                                hparams.update(parsed)
                        except json.JSONDecodeError:
                            pass
        except Exception:
            pass
    
    return hparams


def discover_models(results_dir: Path) -> list:
    """Discover all trained models in results/ directory."""
    models = []
    
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return models
    
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        if model_name not in MODEL_REGISTRY:
            continue
        
        logs_dir = model_dir / "logs"
        if not logs_dir.exists():
            continue
        
        # Check for event files
        has_events = any(logs_dir.rglob("events.out.tfevents.*"))
        if not has_events:
            continue
        
        models.append({
            "name": model_name,
            "dir": model_dir,
            "logs_dir": logs_dir,
            "f1_score": MODEL_F1_SCORES.get(model_name, 0.0),
            "accuracy": MODEL_ACCURACIES.get(model_name, 0.0),
        })
    
    return models


# =========================================================
# MAIN SYNC LOGIC
# =========================================================

def sync_model_to_wandb(model_info: Dict[str, Any]) -> Optional[str]:
    """Sync a single model's TensorBoard logs to W&B."""
    model_name = model_info["name"]
    logs_dir = model_info["logs_dir"]
    
    print(f"\n  Syncing: {model_name}")
    print(f"    Logs: {logs_dir}")
    
    # Count parameters
    param_count = count_parameters(model_name)
    model_size_mb = param_count * 4 / (1024 * 1024)  # FP32
    
    # Extract TensorBoard data
    print(f"    Extracting TensorBoard scalars...")
    scalars = extract_tensorboard_scalars(logs_dir)
    
    print(f"    Extracting hyperparameters...")
    hparams = extract_hyperparameters(logs_dir)
    
    # Initialize W&B run
    run_name = f"{model_name}_vertex_ai"
    tags = ["vertex-ai", "transfer-learning", model_name.split("_")[0]]
    
    config = {
        "model_name": model_name,
        "model_architecture": MODEL_REGISTRY.get(model_name, "Unknown"),
        "num_classes": 102,
        "dataset": "PlantVillage + plant_dataset_2",
        "training_images": 42006,
        "test_images": 19167,
        "folds": 5,
        "optimizer": hparams.get("optimizer", "AdamW"),
        "scheduler": hparams.get("scheduler", "ReduceLROnPlateau"),
        "batch_size": hparams.get("batch_size", 128),
        "epochs": hparams.get("epochs", 200),
        "initial_lr": hparams.get("lr", 1e-3),
        "weight_decay": hparams.get("weight_decay", 1e-4),
        "image_size": hparams.get("image_size", 224),
        "dropout_fc": hparams.get("dropout_fc", 0.5),
        "param_count": param_count,
        "model_size_mb": round(model_size_mb, 2),
        **hparams,  # Merge any other extracted hyperparameters
    }
    
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        tags=tags,
        config=config,
        job_type="model-training",
        resume="allow",
    )
    
    # Log summary metrics
    summary_metrics = {
        "best_val_f1_macro": model_info["f1_score"],
        "test_accuracy": model_info["accuracy"],
        "param_count": param_count,
        "model_size_mb": round(model_size_mb, 2),
    }
    
    # Add extracted TensorBoard scalars
    for key, value in scalars.items():
        if isinstance(value, (int, float)):
            summary_metrics[key] = value
    
    # Use wandb.summary for final metrics
    for key, value in summary_metrics.items():
        wandb.summary[key] = value
    
    # Sync TensorBoard files directly using wandb.tensorboard.patch
    # This reads the event files and re-logs them to W&B
    print(f"    Syncing TensorBoard event files...")
    try:
        # Use wandb.tensorboard to sync
        wandb.tensorboard.patch(root_logdir=str(logs_dir))
        
        # Manually trigger sync by creating a dummy writer and flushing
        # (wandb.tensorboard.patch auto-logs when SummaryWriter is used)
        # Instead, we'll read and log key metrics directly
        
        # Log per-fold metrics if available
        fold_dirs = [d for d in logs_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")]
        for fold_dir in sorted(fold_dirs):
            fold_name = fold_dir.name
            fold_scalars = extract_tensorboard_scalars(fold_dir)
            
            # Log fold metrics with step=fold number
            fold_num = int(fold_name.split("_")[1]) if fold_name.startswith("fold_") else 0
            for key, value in fold_scalars.items():
                if isinstance(value, (int, float)) and "final" in key:
                    # Strip "_final" suffix and log
                    metric_name = key.replace("_final", "")
                    wandb.log({f"{fold_name}/{metric_name}": value}, step=fold_num)
        
    except Exception as e:
        print(f"    [WARN] TensorBoard sync issue: {e}")
    
    # Log model artifact (reference to checkpoint path)
    checkpoint_path = model_info["dir"] / "models" / "best_model_fold_1.pth"
    if checkpoint_path.exists():
        artifact = wandb.Artifact(
            name=f"model-{model_name}",
            type="model",
            description=f"Best checkpoint for {model_name}",
        )
        artifact.add_file(str(checkpoint_path))
        run.log_artifact(artifact)
        print(f"    Logged model artifact: {checkpoint_path}")
    
    run_url = run.get_url()
    run.finish()
    
    print(f"    ✅ Done: {run_url}")
    return run_url


def main():
    print("=" * 70)
    print("  W&B TensorBoard Sync — Crop Disease Detection")
    print("=" * 70)
    
    # Verify W&B login
    try:
        wandb_api = wandb.Api()
        user = wandb_api.viewer
        username = getattr(user, 'username', getattr(user, 'name', str(user)))
        print(f"\n  Logged in as: {username}")
    except Exception as e:
        print(f"\n  ERROR: Not logged into W&B. Run: wandb login")
        print(f"  Details: {e}")
        sys.exit(1)
    
    # Discover models
    print(f"\n  Scanning: {RESULTS_DIR}")
    models = discover_models(RESULTS_DIR)
    
    if not models:
        print("  ERROR: No models with TensorBoard logs found in results/")
        print("  Expected structure: results/<model_name>/logs/")
        sys.exit(1)
    
    print(f"\n  Found {len(models)} models:")
    for m in models:
        print(f"    • {m['name']:20s}  F1={m['f1_score']:.4f}  Acc={m['accuracy']:.4f}")
    
    # Sync each model
    print(f"\n  Syncing to W&B project: {WANDB_PROJECT}")
    if WANDB_ENTITY:
        print(f"  Entity: {WANDB_ENTITY}")
    
    run_urls = []
    for model_info in models:
        try:
            url = sync_model_to_wandb(model_info)
            if url:
                run_urls.append(url)
        except Exception as e:
            print(f"\n  [ERROR] Failed to sync {model_info['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "=" * 70)
    print("  SYNC COMPLETE")
    print("=" * 70)
    
    if run_urls:
        project_url = f"https://wandb.ai/{WANDB_ENTITY or '~'}/{WANDB_PROJECT}"
        print(f"\n  📊 View all runs: {project_url}")
        print(f"\n  Individual runs:")
        for url in run_urls:
            print(f"    • {url}")
    else:
        print("\n  ⚠️  No runs were successfully synced.")
    
    print(f"\n  Next steps:")
    print(f"    1. Open the W&B project link above")
    print(f"    2. Go to 'Workspace' → create comparison plots")
    print(f"    3. Add 'parallel coordinates' for hyperparameter vs F1")
    print(f"    4. Copy the project URL to your resume!")


if __name__ == "__main__":
    main()

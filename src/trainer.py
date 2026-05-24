"""
Training engine for crop disease detection.

Features:
    - 5-fold Stratified Cross-Validation
    - TensorBoard logging (loss, accuracy, precision, recall, F1, LR, confusion matrix)
    - Hyperparameter logging
    - Best model saving per fold
    - Final test set evaluation
"""

import time
import json
import gc
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from tqdm import tqdm

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# =========================================================
# METRIC COMPUTATION
# =========================================================

def compute_metrics(targets, predictions):
    """
    Compute all classification metrics.

    Returns dict with:
        accuracy, precision/recall/f1 (macro, micro, weighted)
    """
    return {
        "accuracy": accuracy_score(targets, predictions),

        "precision_macro":    precision_score(targets, predictions, average="macro",    zero_division=0),
        "precision_micro":    precision_score(targets, predictions, average="micro",    zero_division=0),
        "precision_weighted": precision_score(targets, predictions, average="weighted", zero_division=0),

        "recall_macro":    recall_score(targets, predictions, average="macro",    zero_division=0),
        "recall_micro":    recall_score(targets, predictions, average="micro",    zero_division=0),
        "recall_weighted": recall_score(targets, predictions, average="weighted", zero_division=0),

        "f1_macro":    f1_score(targets, predictions, average="macro",    zero_division=0),
        "f1_micro":    f1_score(targets, predictions, average="micro",    zero_division=0),
        "f1_weighted": f1_score(targets, predictions, average="weighted", zero_division=0),
    }


def plot_confusion_matrix(targets, predictions, class_names, title="Confusion Matrix"):
    """
    Generate a confusion matrix figure for TensorBoard logging.

    For 102 classes, creates a large heatmap with readable labels.
    """
    cm = confusion_matrix(targets, predictions)
    num_classes = len(class_names)

    # Scale figure size based on number of classes
    fig_size = max(20, num_classes // 4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    sns.heatmap(
        cm,
        annot=False,               # too many classes for numbers
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=14)

    # Rotate labels for readability
    plt.xticks(rotation=90, fontsize=4)
    plt.yticks(rotation=0, fontsize=4)
    plt.tight_layout()

    return fig


# =========================================================
# TRAINER
# =========================================================

class Trainer:
    """
    Training engine with 5-fold CV, TensorBoard logging, and full metrics.

    Args:
        model_fn:       callable that returns a fresh model instance
        config:         dict with training hyperparameters
        log_dir:        directory for TensorBoard logs
        save_dir:       directory for saving model checkpoints
        device:         'cuda' or 'cpu'
        class_names:    list of class name strings (for confusion matrix)
    """

    def __init__(self, model_fn, config, log_dir, save_dir, device, class_names,
                 on_checkpoint=None):
        self.model_fn = model_fn
        self.config = config
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.device = torch.device(device)
        self.class_names = class_names
        self.on_checkpoint = on_checkpoint  # callback(list_of_saved_files)

        # Create save directory
        from pathlib import Path
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        # Log hyperparameters
        self._log_hyperparameters()

    def _log_hyperparameters(self):
        """Save hyperparameters to JSON for reproducibility."""
        from pathlib import Path
        hp_path = Path(self.save_dir) / "hyperparameters.json"
        with open(hp_path, "w") as f:
            json.dump(self.config, f, indent=4)
        print(f"Hyperparameters saved: {hp_path}")

    def _build_optimizer(self, model):
        """Build optimizer from config."""
        return torch.optim.Adam(
            model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

    def _build_scheduler(self, optimizer):
        """Build learning rate scheduler.
        
        Uses ReduceLROnPlateau when config["use_reduce_lr_on_plateau"] is True,
        otherwise falls back to CosineAnnealingLR.
        """
        if self.config.get("use_reduce_lr_on_plateau", False):
            return ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=self.config.get("lr_factor", 0.5),
                patience=self.config.get("lr_patience", 3),
                min_lr=self.config.get("min_lr", 1e-6),
            )
        return CosineAnnealingLR(
            optimizer,
            T_max=self.config["epochs"],
            eta_min=self.config.get("min_lr", 1e-6),
        )

    def _train_one_epoch(self, model, dataloader, optimizer, criterion, epoch, writer, fold, transform=None):
        """Train for one epoch. Returns average loss, all targets, and all predictions."""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        # Pre-allocate lists with estimated capacity
        num_samples = len(dataloader.dataset)
        all_targets = np.empty(num_samples, dtype=np.int64)
        all_predictions = np.empty(num_samples, dtype=np.int64)
        idx_offset = 0

        pbar = tqdm(dataloader, desc=f"  Fold {fold} | Epoch {epoch+1} [TRAIN]", leave=False)

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            if transform is not None:
                images = transform(images)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            total += batch_size
            correct += predicted.eq(labels).sum().item()

            # Store predictions efficiently (no Python list overhead)
            labels_np = labels.cpu().numpy()
            preds_np = predicted.cpu().numpy()
            all_targets[idx_offset:idx_offset + batch_size] = labels_np
            all_predictions[idx_offset:idx_offset + batch_size] = preds_np
            idx_offset += batch_size

            # Update progress bar (every 10 batches to reduce log spam)
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100. * correct / total:.1f}%",
                })

        avg_loss = running_loss / total

        # Trim to actual size (drop_last=True may reduce count)
        return avg_loss, all_targets[:idx_offset], all_predictions[:idx_offset]

    @torch.no_grad()
    def _validate(self, model, dataloader, criterion, epoch, fold, transform=None):
        """Validate model. Returns loss, accuracy, all targets, all predictions."""
        model.eval()
        running_loss = 0.0
        all_targets = []
        all_predictions = []

        pbar = tqdm(dataloader, desc=f"  Fold {fold} | Epoch {epoch+1} [VAL]  ", leave=False)

        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            if transform is not None:
                images = transform(images)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)

            all_targets.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        avg_loss = running_loss / len(all_targets)
        return avg_loss, np.array(all_targets), np.array(all_predictions)

    def train_fold(self, fold, train_dataset, val_dataset, train_transform, eval_transform):
        """
        Train one fold of cross-validation.

        Returns: best validation metrics dict
        """
        from pathlib import Path

        print(f"\n{'='*60}")
        print(f"  FOLD {fold} / {self.config['n_folds']}")
        print(f"{'='*60}")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples:   {len(val_dataset)}")

        # --- Compute class weights from training labels (for balanced training) ---
        # Get labels from the training subset
        if hasattr(train_dataset, 'dataset'):
            # It's a Subset — get labels from the underlying dataset
            train_labels = [train_dataset.dataset.labels[i] for i in train_dataset.indices]
        else:
            train_labels = train_dataset.get_labels()

        train_labels_np = np.array(train_labels)
        class_counts = np.bincount(train_labels_np, minlength=self.config["num_classes"])
        class_counts = np.maximum(class_counts, 1)  # avoid division by zero

        # WeightedRandomSampler: oversample minority classes during training
        # Each sample gets weight = 1 / count_of_its_class
        sample_weights = 1.0 / class_counts[train_labels_np]
        sample_weights = torch.DoubleTensor(sample_weights)

        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),  # one full epoch = original dataset size
            replacement=True,
        )

        # Class-weighted CrossEntropyLoss: penalize rare-class errors more
        class_weights = 1.0 / class_counts.astype(np.float32)
        class_weights = class_weights / class_weights.sum() * len(class_weights)  # normalize
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)

        # Print class balance info
        print(f"  Class counts: min={class_counts.min()}, max={class_counts.max()}, "
              f"mean={class_counts.mean():.0f}, median={np.median(class_counts):.0f}")
        print(f"  Using WeightedRandomSampler + class-weighted CrossEntropyLoss")

        # DataLoaders (sampler replaces shuffle for training)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            sampler=sampler,  # oversamples minority classes
            num_workers=self.config.get("num_workers", 2),
            prefetch_factor=2 if self.config.get("num_workers", 2) > 0 else None,
            pin_memory=True,
            drop_last=True,
            persistent_workers=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config.get("num_workers", 2),
            prefetch_factor=2 if self.config.get("num_workers", 2) > 0 else None,
            pin_memory=True,
            persistent_workers=False,
        )

        # Fresh model for each fold
        model = self.model_fn().to(self.device)
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        # Move GPU augmentation pipelines to device
        gpu_train_transform = train_transform.to(self.device)
        gpu_eval_transform = eval_transform.to(self.device)

        # TensorBoard writer for this fold
        writer = SummaryWriter(log_dir=f"{self.log_dir}/fold_{fold}")

        # Log hyperparameters
        writer.add_text("hyperparameters", json.dumps(self.config, indent=2), 0)

        best_val_f1 = 0.0
        best_metrics = {}
        early_stop_counter = 0
        start_epoch = 0

        # --- Resume from checkpoint if available ---
        checkpoint_path = Path(self.save_dir) / f"checkpoint_fold_{fold}.pth"
        if checkpoint_path.exists():
            print(f"  [RESUME] Found checkpoint: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"]
            best_val_f1 = ckpt.get("best_val_f1", 0.0)
            best_metrics = ckpt.get("best_metrics", {})
            early_stop_counter = ckpt.get("early_stop_counter", 0)
            print(f"  [RESUME] Resuming from epoch {start_epoch + 1}, best F1: {best_val_f1:.4f}")

        for epoch in range(start_epoch, self.config["epochs"]):
            start_time = time.time()

            # Memory monitoring
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if HAS_PSUTIL:
                mem = psutil.virtual_memory()
                gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                print(f"  [MEM] RAM: {mem.used/1e9:.1f}/{mem.total/1e9:.1f} GB ({mem.percent}%) | "
                      f"GPU: {gpu_mem:.2f} GB allocated")
                if mem.percent > 90:
                    print("  [WARNING] RAM usage above 90%! Risk of OOM kill.")

            # --- Train ---
            train_loss, train_targets, train_preds = self._train_one_epoch(
                model, train_loader, optimizer, criterion, epoch, writer, fold,
                transform=gpu_train_transform,
            )
            train_metrics = compute_metrics(train_targets, train_preds)

            # --- Validate ---
            val_loss, val_targets, val_preds = self._validate(
                model, val_loader, criterion, epoch, fold,
                transform=gpu_eval_transform,
            )
            val_metrics = compute_metrics(val_targets, val_preds)

            # --- Learning rate step ---
            current_lr = optimizer.param_groups[0]["lr"]
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics["f1_macro"])
            else:
                scheduler.step()

            elapsed = time.time() - start_time

            # --- Log to TensorBoard ---
            writer.add_scalar(f"loss/train", train_loss, epoch)
            writer.add_scalar(f"loss/val", val_loss, epoch)
            writer.add_scalar(f"learning_rate", current_lr, epoch)
            writer.add_scalar(f"weight_decay", self.config["weight_decay"], epoch)

            # Log all train metrics
            for metric_name, metric_value in train_metrics.items():
                writer.add_scalar(f"train/{metric_name}", metric_value, epoch)

            # Log all val metrics
            for metric_name, metric_value in val_metrics.items():
                writer.add_scalar(f"val/{metric_name}", metric_value, epoch)

            # --- Print epoch summary ---
            print(f"\n  {'─'*90}")
            print(
                f"  Epoch {epoch+1:>3}/{self.config['epochs']} | "
                f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s"
            )
            print(f"  {'─'*90}")
            print(
                f"  TRAIN │ Loss: {train_loss:.4f} │ "
                f"Acc: {train_metrics['accuracy']:.4f} │ "
                f"P(m): {train_metrics['precision_macro']:.4f}  P(w): {train_metrics['precision_weighted']:.4f} │ "
                f"R(m): {train_metrics['recall_macro']:.4f}  R(w): {train_metrics['recall_weighted']:.4f}"
            )
            print(
                f"        │                │ "
                f"F1(macro): {train_metrics['f1_macro']:.4f} │ "
                f"F1(weighted): {train_metrics['f1_weighted']:.4f} │ "
                f"F1(micro): {train_metrics['f1_micro']:.4f}"
            )
            print(
                f"  VAL   │ Loss: {val_loss:.4f} │ "
                f"Acc: {val_metrics['accuracy']:.4f} │ "
                f"P(m): {val_metrics['precision_macro']:.4f}  P(w): {val_metrics['precision_weighted']:.4f} │ "
                f"R(m): {val_metrics['recall_macro']:.4f}  R(w): {val_metrics['recall_weighted']:.4f}"
            )
            print(
                f"        │                │ "
                f"F1(macro): {val_metrics['f1_macro']:.4f} │ "
                f"F1(weighted): {val_metrics['f1_weighted']:.4f} │ "
                f"F1(micro): {val_metrics['f1_micro']:.4f}"
            )

            # --- Save best model ---
            if val_metrics["f1_macro"] > best_val_f1:
                best_val_f1 = val_metrics["f1_macro"]
                best_metrics = val_metrics.copy()
                best_metrics["epoch"] = epoch + 1
                best_metrics["train_loss"] = train_loss
                best_metrics["val_loss"] = val_loss
                # Also store train metrics for the best epoch
                for k, v in train_metrics.items():
                    best_metrics[f"train_{k}"] = v

                save_path = Path(self.save_dir) / f"best_model_fold_{fold}.pth"
                torch.save({
                    "fold": fold,
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": best_metrics,
                    "config": self.config,
                }, save_path)
                print(f"  * New best model saved (F1: {best_val_f1:.4f})")

                # Reset early stopping counter on improvement
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                early_stop_patience = self.config.get("early_stop_patience", 0)
                if early_stop_patience > 0 and early_stop_counter >= early_stop_patience:
                    print(f"  [STOP] Early stopping triggered (no improvement for {early_stop_patience} epochs)")
                    break

            # --- Epoch checkpoint (for crash recovery) ---
            checkpoint_path = Path(self.save_dir) / f"checkpoint_fold_{fold}.pth"
            torch.save({
                "fold": fold,
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_f1": best_val_f1,
                "best_metrics": best_metrics,
                "early_stop_counter": early_stop_counter,
                "config": self.config,
            }, checkpoint_path)

            # Sync checkpoint + best model to remote storage (GCS)
            if self.on_checkpoint:
                files_to_sync = [str(checkpoint_path)]
                best_model_path = Path(self.save_dir) / f"best_model_fold_{fold}.pth"
                if best_model_path.exists():
                    files_to_sync.append(str(best_model_path))
                try:
                    self.on_checkpoint(files_to_sync)
                except Exception as e:
                    print(f"  [WARN] Checkpoint sync failed: {e}")

            # Flush TensorBoard and free memory
            writer.flush()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # --- Log confusion matrix for best epoch ---
        fig = plot_confusion_matrix(val_targets, val_preds, self.class_names, f"Fold {fold} - Confusion Matrix")
        writer.add_figure("confusion_matrix", fig, self.config["epochs"])
        plt.close(fig)

        writer.flush()
        writer.close()

        # Clean up checkpoint after fold completes successfully
        checkpoint_path = Path(self.save_dir) / f"checkpoint_fold_{fold}.pth"
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"  Fold {fold} checkpoint cleaned up.")

        print(f"\n  Fold {fold} best: Acc={best_metrics['accuracy']:.4f} | F1(macro)={best_metrics['f1_macro']:.4f} | Epoch={best_metrics['epoch']}")

        return best_metrics

    def run_cv(self, dataset, train_transform, eval_transform):
        """
        Run stratified cross-validation with crash resume support.

        - n_folds=1: single stratified 80/20 split (all classes in both sets)
        - n_folds>=2: full k-fold stratified CV

        Completed folds are saved to fold_results.json and skipped on resume.

        Returns: list of best metrics per fold, and averaged metrics.
        """
        from pathlib import Path
        from sklearn.model_selection import train_test_split

        n_folds = self.config["n_folds"]
        labels = dataset.get_labels()

        # Generate fold splits
        if n_folds == 1:
            # Single stratified 70/30 split — guarantees all classes in both sets
            all_idx = list(range(len(dataset)))
            train_idx, val_idx = train_test_split(
                all_idx, test_size=0.3, random_state=42, stratify=labels
            )
            fold_splits = [(train_idx, val_idx)]
            print(f"\n  Single stratified split: {len(train_idx)} train (70%), {len(val_idx)} val (30%)")
        else:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            fold_splits = list(skf.split(range(len(dataset)), labels))

        # Load previously completed fold results (for crash resume)
        fold_results_path = Path(self.save_dir) / "fold_results.json"
        completed_folds = {}
        if fold_results_path.exists():
            with open(fold_results_path, "r") as f:
                completed_folds = json.load(f)
            print(f"\n  [RESUME] Found {len(completed_folds)} completed fold(s): {list(completed_folds.keys())}")

        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(fold_splits, 1):
            fold_key = str(fold)

            # Skip already-completed folds
            if fold_key in completed_folds:
                print(f"\n  [SKIP] Fold {fold} already completed (F1: {completed_folds[fold_key].get('f1_macro', 'N/A')})")
                fold_results.append(completed_folds[fold_key])
                continue

            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            fold_metrics = self.train_fold(
                fold=fold,
                train_dataset=train_subset,
                val_dataset=val_subset,
                train_transform=train_transform,
                eval_transform=eval_transform,
            )
            fold_results.append(fold_metrics)

            # Save completed fold results (for crash resume)
            completed_folds[fold_key] = {
                k: (float(v) if isinstance(v, (np.floating, float)) else v)
                for k, v in fold_metrics.items()
            }
            with open(fold_results_path, "w") as f:
                json.dump(completed_folds, f, indent=4)

            # Sync fold results file to GCS
            if self.on_checkpoint:
                try:
                    self.on_checkpoint([str(fold_results_path)])
                except Exception as e:
                    print(f"  [WARN] Fold results sync failed: {e}")

        # --- Compute average metrics across folds ---
        avg_metrics = {}
        metric_keys = [k for k in fold_results[0].keys() if isinstance(fold_results[0][k], (float, np.floating))]
        for key in metric_keys:
            values = [f[key] for f in fold_results]
            avg_metrics[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

        # --- Print CV summary ---
        print(f"\n{'='*60}")
        print(f"  CROSS-VALIDATION SUMMARY ({n_folds} folds)")
        print(f"{'='*60}")
        for key in ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]:
            if key in avg_metrics:
                print(f"  {key:>20}: {avg_metrics[key]['mean']:.4f} ± {avg_metrics[key]['std']:.4f}")

        # --- Log CV summary to TensorBoard ---
        writer = SummaryWriter(log_dir=f"{self.log_dir}/cv_summary")
        for key, val in avg_metrics.items():
            writer.add_scalar(f"cv_mean/{key}", val["mean"], 0)
            writer.add_scalar(f"cv_std/{key}", val["std"], 0)
        writer.close()

        # --- Save CV results to JSON ---
        cv_results_path = Path(self.save_dir) / "cv_results.json"
        with open(cv_results_path, "w") as f:
            json.dump({
                "fold_results": [{k: (v if not isinstance(v, np.floating) else float(v))
                                  for k, v in fr.items()} for fr in fold_results],
                "average_metrics": avg_metrics,
            }, f, indent=4)
        print(f"\n  CV results saved: {cv_results_path}")

        return fold_results, avg_metrics

    def evaluate_test(self, test_dataset, eval_transform, fold_to_use=1):
        """
        Final evaluation on the held-out test set using the best model from a fold.

        Args:
            test_dataset:   CropDiseaseDataset for test split
            eval_transform: deterministic eval transforms
            fold_to_use:    which fold's best model to load (default: 1)
        """
        from pathlib import Path

        print(f"\n{'='*60}")
        print(f"  FINAL TEST EVALUATION (using best model from fold {fold_to_use})")
        print(f"{'='*60}")

        # Load best model
        model_path = Path(self.save_dir) / f"best_model_fold_{fold_to_use}.pth"
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        model = self.model_fn().to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])

        gpu_eval_transform = eval_transform.to(self.device)

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True,
        )

        # Evaluate
        criterion = nn.CrossEntropyLoss()
        test_loss, test_targets, test_preds = self._validate(
            model, test_loader, criterion, epoch=0, fold="test",
            transform=gpu_eval_transform,
        )
        test_metrics = compute_metrics(test_targets, test_preds)

        # --- Print test results ---
        print(f"\n  {'='*70}")
        print(f"  TEST SET RESULTS")
        print(f"  {'='*70}")
        print(f"  Loss:                  {test_loss:.4f}")
        print(f"  {'─'*70}")
        print(f"  Accuracy:              {test_metrics['accuracy']:.4f}")
        print(f"  {'─'*70}")
        print(f"  Precision (macro):     {test_metrics['precision_macro']:.4f}")
        print(f"  Precision (micro):     {test_metrics['precision_micro']:.4f}")
        print(f"  Precision (weighted):  {test_metrics['precision_weighted']:.4f}")
        print(f"  {'─'*70}")
        print(f"  Recall (macro):        {test_metrics['recall_macro']:.4f}")
        print(f"  Recall (micro):        {test_metrics['recall_micro']:.4f}")
        print(f"  Recall (weighted):     {test_metrics['recall_weighted']:.4f}")
        print(f"  {'─'*70}")
        print(f"  F1-score (macro):      {test_metrics['f1_macro']:.4f}")
        print(f"  F1-score (micro):      {test_metrics['f1_micro']:.4f}")
        print(f"  F1-score (weighted):   {test_metrics['f1_weighted']:.4f}")
        print(f"  {'='*70}")

        # --- Log to TensorBoard ---
        writer = SummaryWriter(log_dir=f"{self.log_dir}/test")
        for key, val in test_metrics.items():
            writer.add_scalar(f"test/{key}", val, 0)
        writer.add_scalar("test/loss", test_loss, 0)

        # Confusion matrix
        fig = plot_confusion_matrix(test_targets, test_preds, self.class_names, "Test Set - Confusion Matrix")
        writer.add_figure("test/confusion_matrix", fig, 0)
        plt.close(fig)

        # Classification report (with overall accuracy and all summary metrics)
        report = classification_report(
            test_targets, test_preds,
            target_names=self.class_names,
            zero_division=0,
            digits=4,
        )
        overall_acc = accuracy_score(test_targets, test_preds)
        report_header = (
            f"{'='*70}\n"
            f"  CLASSIFICATION REPORT\n"
            f"{'='*70}\n"
            f"  Accuracy:           {overall_acc:.4f} ({int(overall_acc * len(test_targets))}/{len(test_targets)})\n"
            f"  Precision (macro):  {test_metrics['precision_macro']:.4f}  (micro): {test_metrics['precision_micro']:.4f}  (weighted): {test_metrics['precision_weighted']:.4f}\n"
            f"  Recall    (macro):  {test_metrics['recall_macro']:.4f}  (micro): {test_metrics['recall_micro']:.4f}  (weighted): {test_metrics['recall_weighted']:.4f}\n"
            f"  F1-score  (macro):  {test_metrics['f1_macro']:.4f}  (micro): {test_metrics['f1_micro']:.4f}  (weighted): {test_metrics['f1_weighted']:.4f}\n"
            f"{'='*70}\n"
        )
        full_report = report_header + report
        writer.add_text("test/classification_report", f"```\n{full_report}\n```", 0)
        print(f"\n{full_report}")

        writer.close()

        # Save test results
        test_results_path = Path(self.save_dir) / "test_results.json"
        with open(test_results_path, "w") as f:
            json.dump({
                "test_loss": test_loss,
                "metrics": test_metrics,
                "model_used": str(model_path),
            }, f, indent=4)
        print(f"  Test results saved: {test_results_path}")

        return test_metrics

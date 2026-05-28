"""
Model loading and inference service.

Loads all model checkpoints at startup with pretrained=False (no ImageNet download).
Provides prediction and GradCAM explainability.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

# Add project root to path for model imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model import get_model

logger = logging.getLogger(__name__)

# ─── Constants ───
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 102
IMAGE_SIZE = 224

# ─── Model Metadata ───
MODEL_METADATA = {
    "resnet_50": {
        "display_name": "ResNet-50",
        "type": "Transfer Learning",
        "params": "~25.6M",
        "speed": "medium",
    },
    "mobilenet_v3": {
        "display_name": "MobileNetV3-Large",
        "type": "Transfer Learning",
        "params": "~5.4M",
        "speed": "fast",
    },
}


class InferenceService:
    """Manages model loading and prediction for all supported architectures."""

    def __init__(self, checkpoints_dir: str, label_mapping_path: str):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.models: dict = {}
        self.label_mapping: dict = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load label mapping
        with open(label_mapping_path) as f:
            mapping = json.load(f)
        self.label_mapping = mapping.get("label_to_class", mapping)

        logger.info(f"Inference device: {self.device}")
        logger.info(f"Loaded {len(self.label_mapping)} class labels")

    def load_model(self, model_name: str) -> bool:
        """Load a single model from checkpoint. Returns True if successful."""
        checkpoint_path = self.checkpoints_dir / model_name / "models" / "best_model_fold_1.pth"

        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return False

        try:
            model = get_model(model_name, num_classes=NUM_CLASSES, pretrained=False)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            self._disable_inplace_ops(model)
            model.to(self.device)
            model.eval()

            # Store model + metadata from checkpoint
            self.models[model_name] = {
                "model": model,
                "epoch": checkpoint.get("epoch", "N/A"),
                "val_metrics": checkpoint.get("val_metrics", {}),
            }

            f1 = checkpoint.get("val_metrics", {}).get("f1_macro", "N/A")
            logger.info(f"Loaded {model_name} (epoch {checkpoint.get('epoch')}, F1: {f1})")
            return True

        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            return False

    def load_all_models(self):
        """Load all available model checkpoints."""
        loaded = 0
        for model_name in MODEL_METADATA.keys():
            if self.load_model(model_name):
                loaded += 1
        logger.info(f"Loaded {loaded}/{len(MODEL_METADATA)} models")

    def _disable_inplace_ops(self, model):
        for module in model.modules():
            if hasattr(module, "inplace"):
                module.inplace = False

    def get_available_models(self) -> list[dict]:
        """Return metadata for all loaded models."""
        available = []
        for name, meta in MODEL_METADATA.items():
            if name in self.models:
                info = {
                    "model_name": name,
                    **meta,
                    "epoch": self.models[name]["epoch"],
                    "val_f1_macro": self.models[name]["val_metrics"].get("f1_macro"),
                    "val_accuracy": self.models[name]["val_metrics"].get("accuracy"),
                }
                available.append(info)
        return available

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess PIL image for inference."""
        image = image.convert("RGB")
        image = image.resize((256, 256))
        img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        img_tensor = v2.functional.center_crop(img_tensor, [IMAGE_SIZE, IMAGE_SIZE])
        img_tensor = v2.functional.normalize(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return img_tensor.unsqueeze(0)  # Add batch dimension

    def predict(self, image: Image.Image, model_name: str, top_k: int = 5) -> dict:
        """
        Run prediction on a single image.

        Returns:
            {
                "prediction": str,
                "confidence": float,
                "top_k": [{"class": str, "confidence": float}, ...],
                "model_used": str,
            }
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded. Available: {list(self.models.keys())}")

        model = self.models[model_name]["model"]
        input_tensor = self.preprocess_image(image).to(self.device)

        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        # Top-K predictions
        top_k_probs, top_k_indices = torch.topk(probabilities, k=min(top_k, NUM_CLASSES))

        top_k_results = []
        for prob, idx in zip(top_k_probs.tolist(), top_k_indices.tolist()):
            class_name = self.label_mapping.get(str(idx), f"class_{idx}")
            top_k_results.append({"class": class_name, "confidence": round(prob, 4)})

        return {
            "prediction": top_k_results[0]["class"],
            "confidence": top_k_results[0]["confidence"],
            "top_k": top_k_results,
            "model_used": model_name,
        }

    def get_gradcam(self, image: Image.Image, model_name: str) -> Optional[np.ndarray]:
        """
        Generate GradCAM heatmap for the predicted class.

        Returns:
            numpy array (H, W) with values 0-1, or None on failure.
        """
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded.")

        model = self.models[model_name]["model"]
        input_tensor = self.preprocess_image(image).to(self.device)

        # Determine target layer based on model architecture
        target_layer = self._get_target_layer(model_name, model)
        if target_layer is None:
            return None

        # Determine reshape transform
        reshape_transform = self._get_reshape_transform(model_name)

        # Get prediction first
        with torch.no_grad():
            logits = model(input_tensor)
            pred_idx = logits.argmax(dim=1).item()

        # Generate GradCAM
        targets = [ClassifierOutputTarget(pred_idx)]
        with GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

        return grayscale_cam[0, :]

    def _get_target_layer(self, model_name: str, model):
        """Get the appropriate target layer for GradCAM."""
        try:
            if model_name == "resnet_50":
                return model.backbone.layer4[-1]
            elif model_name == "mobilenet_v3":
                return model.backbone.features[-1]
            else:
                return None
        except Exception:
            return None

    def _get_reshape_transform(self, model_name: str):
        """Get reshape transform for transformer models."""
        return None

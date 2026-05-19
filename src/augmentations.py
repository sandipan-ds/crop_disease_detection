"""
GPU-accelerated image augmentations using torchvision v2 transforms.

All heavy ops (RandomResizedCrop, rotation, warps, blur, color jitter,
RandomErasing) run on GPU via torchvision v2 nn.Module pipelines.
The Dataset only performs minimal I/O: PIL read + uint8 tensor conversion.
"""

import random
import torch.nn as nn
from torchvision.transforms import v2


# =========================================================
# ImageNet normalization (used by all pretrained backbones)
# =========================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =========================================================
# Random Subset Augmentation (GPU-compatible)
# =========================================================

class RandomSubsetAugment(nn.Module):
    """
    Randomly selects k augmentations from a pool and applies them.

    k is sampled uniformly from [min_aug, max_aug] per image.
    When k=0, the image passes through with no augmentation (raw).

    Operates on GPU tensors (torchvision v2 modules).
    """

    def __init__(self, augmentation_pool, min_aug=0, max_aug=5):
        super().__init__()
        self.pool = nn.ModuleList(augmentation_pool)
        self.min_aug = min_aug
        self.max_aug = min(max_aug, len(augmentation_pool))

    def forward(self, img):
        k = random.randint(self.min_aug, self.max_aug)
        if k == 0:
            return img
        indices = random.sample(range(len(self.pool)), k)
        for i in indices:
            img = self.pool[i](img)
        return img

    def __repr__(self):
        return (
            f"RandomSubsetAugment("
            f"pool_size={len(self.pool)}, "
            f"k=[{self.min_aug}, {self.max_aug}])"
        )


# =========================================================
# Augmentation pool (v2 modules — GPU-compatible)
# =========================================================

def _build_gpu_augmentation_pool():
    """
    Pool of individual v2 augmentations (GPU-compatible nn.Modules).

    Index | Augmentation        | Real-world condition
    ──────┼─────────────────────┼────────────────────────────────────
      0   | HorizontalFlip      | Leaf orientation is arbitrary
      1   | VerticalFlip        | Phone held upside down
      2   | Rotation (±30°)     | Tilted phone, angled shots
      3   | Perspective warp    | Non-perpendicular camera angle
      4   | ColorJitter         | Weather + camera quality variation
      5   | GaussianBlur        | Shaky hands, cheap lens
      6   | Grayscale           | Forces texture learning
    ──────┴─────────────────────┴────────────────────────────────────
    """
    return nn.ModuleList([
        v2.RandomHorizontalFlip(p=1.0),
        v2.RandomVerticalFlip(p=1.0),
        v2.RandomRotation(degrees=30, fill=0),
        v2.RandomPerspective(distortion_scale=0.15, p=1.0),
        v2.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.30, hue=0.08),
        v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        v2.RandomGrayscale(p=1.0),
    ])


# =========================================================
# GPU Pipeline Modules
# =========================================================

class GPUTrainPipeline(nn.Module):
    """
    Full training augmentation pipeline that runs on GPU.

    Accepts uint8 tensor batches (B, C, H, W) on GPU, applies:
    1. RandomResizedCrop
    2. Random 0-5 subset augmentations from pool
    3. Normalize (auto-casts uint8 → float / 255)
    4. RandomErasing (15% chance)

    Call .to(device) to move all internal transforms to GPU.
    """

    def __init__(self, img_size=224, min_aug=0, max_aug=5):
        super().__init__()
        pool = _build_gpu_augmentation_pool()
        self.transform = nn.Sequential(
            v2.RandomResizedCrop(
                size=(img_size, img_size),
                scale=(0.6, 1.0),
                ratio=(0.8, 1.2),
            ),
            RandomSubsetAugment(pool, min_aug, max_aug),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            v2.RandomErasing(
                p=0.15,
                scale=(0.02, 0.15),
                ratio=(0.3, 3.3),
                value="random",
            ),
        )

    def forward(self, images):
        return self.transform(images)


class GPUEvalPipeline(nn.Module):
    """
    Evaluation pipeline that runs on GPU.

    Accepts uint8 tensor batches (B, C, H, W) on GPU, applies:
    1. Resize to (img_size + 32, img_size + 32)
    2. CenterCrop to (img_size, img_size)
    3. Normalize (auto-casts uint8 → float / 255)

    Call .to(device) to move all internal transforms to GPU.
    """

    def __init__(self, img_size=224):
        super().__init__()
        self.transform = nn.Sequential(
            v2.Resize(size=(img_size + 32, img_size + 32)),
            v2.CenterCrop(size=(img_size, img_size)),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        )

    def forward(self, images):
        return self.transform(images)


# =========================================================
# Public API (unchanged signatures — drop-in replacement)
# =========================================================

def get_train_transforms(img_size=224, min_aug=0, max_aug=5):
    """Return a GPU train augmentation pipeline (callable nn.Module)."""
    return GPUTrainPipeline(img_size, min_aug, max_aug)


def get_eval_transforms(img_size=224):
    """Return a GPU eval augmentation pipeline (callable nn.Module)."""
    return GPUEvalPipeline(img_size)

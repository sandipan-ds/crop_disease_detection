"""
Image augmentation transforms for crop disease detection.

Designed to simulate real-world farm field conditions:
- Varying phone camera quality and brands
- Indian weather: harsh sun, monsoon overcast, partial shade
- Photos taken at arbitrary angles, distances, orientations
- Partial occlusion from overlapping leaves, hands, tools

Key design:
- Each image randomly receives 0-5 augmentations from a pool
- 0 augmentations = raw image (just resized + normalized)
- This mimics reality: some field photos are clean, others have
  multiple distortions stacked together

Usage:
    from src.augmentations import get_train_transforms, get_eval_transforms

    train_transform = get_train_transforms(img_size=224)
    eval_transform  = get_eval_transforms(img_size=224)
"""

import random
from torchvision import transforms


# =========================================================
# ImageNet normalization (used by all pretrained backbones)
# =========================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class RandomSubsetAugment:
    """
    Randomly selects k augmentations from a pool and applies them.

    k is sampled uniformly from [min_aug, max_aug] per image.
    When k=0, the image passes through with no augmentation (raw).
    """

    def __init__(self, augmentation_pool, min_aug=0, max_aug=5):
        self.pool = augmentation_pool
        self.min_aug = min_aug
        self.max_aug = min(max_aug, len(augmentation_pool))

    def __call__(self, img):
        k = random.randint(self.min_aug, self.max_aug)

        if k == 0:
            return img

        selected = random.sample(self.pool, k)

        for aug in selected:
            img = aug(img)

        return img

    def __repr__(self):
        return (
            f"RandomSubsetAugment("
            f"pool_size={len(self.pool)}, "
            f"k=[{self.min_aug}, {self.max_aug}])"
        )


def _build_augmentation_pool():
    """
    Pool of individual augmentations mimicking farm field conditions.

    Index | Augmentation        | Real-world condition
    ──────┼─────────────────────┼────────────────────────────────────
      0   | HorizontalFlip      | Leaf orientation is arbitrary
      1   | VerticalFlip        | Phone held upside down
      2   | Rotation (±30°)     | Tilted phone, angled shots
      3   | Perspective warp    | Non-perpendicular camera angle
      4   | ColorJitter         | Weather + camera quality variation
      5   | GaussianBlur        | Shaky hands, cheap lens
      6   | Grayscale           | Forces texture learning
      7   | RandomErasing       | Partial occlusion (leaves, fingers)
    ──────┴─────────────────────┴────────────────────────────────────
    """

    return [
        # 0 — Horizontal flip
        transforms.RandomHorizontalFlip(p=1.0),

        # 1 — Vertical flip
        transforms.RandomVerticalFlip(p=1.0),

        # 2 — Rotation: tilted phone
        transforms.RandomRotation(degrees=30, fill=0),

        # 3 — Perspective: non-perpendicular angle
        transforms.RandomPerspective(distortion_scale=0.15, p=1.0),

        # 4 — Color jitter: weather + camera differences
        transforms.ColorJitter(
            brightness=0.35,
            contrast=0.35,
            saturation=0.30,
            hue=0.08,
        ),

        # 5 — Gaussian blur: shaky hands, cheap lens
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),

        # 6 — Grayscale: force texture learning
        transforms.RandomGrayscale(p=1.0),

        # 7 — Random erasing: partial occlusion
        # (applied after ToTensor, so handled separately below)
    ]


def get_train_transforms(img_size: int = 224, min_aug: int = 0, max_aug: int = 5):
    """
    Training transforms with random subset augmentation.

    Args:
        img_size:  target image size for the model
        min_aug:   minimum number of augmentations per image (0 = raw)
        max_aug:   maximum number of augmentations per image

    Pipeline:
        1. RandomResizedCrop  (always — required for consistent size)
        2. Random 0-5 augmentations from pool
        3. ToTensor + Normalize (always — required for model)
        4. RandomErasing (15% chance — operates on tensors)
    """

    pool = _build_augmentation_pool()

    return transforms.Compose([

        # --- Always applied: resize to model input ---
        transforms.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.6, 1.0),
            ratio=(0.8, 1.2),
        ),

        # --- Random subset of augmentations ---
        RandomSubsetAugment(
            augmentation_pool=pool,
            min_aug=min_aug,
            max_aug=max_aug,
        ),

        # --- Always applied: tensor conversion ---
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

        # --- Occlusion (15% chance, works on tensors) ---
        transforms.RandomErasing(
            p=0.15,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3),
            value="random",
        ),
    ])


def get_eval_transforms(img_size: int = 224):
    """
    Evaluation transforms: deterministic, no augmentation.
    Only resize, center crop, and normalize.
    """

    return transforms.Compose([
        transforms.Resize(size=(img_size + 32, img_size + 32)),
        transforms.CenterCrop(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

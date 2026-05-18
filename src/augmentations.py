"""
Image augmentation transforms for crop disease detection.

Designed to simulate real-world farm field conditions:
- Varying phone camera quality and brands
- Indian weather: harsh sun, monsoon overcast, partial shade
- Photos taken at arbitrary angles, distances, orientations
- Partial occlusion from overlapping leaves, hands, tools

Usage:
    from src.augmentations import get_train_transforms, get_eval_transforms

    train_transform = get_train_transforms(img_size=224)
    eval_transform  = get_eval_transforms(img_size=224)
"""

from torchvision import transforms


# =========================================================
# ImageNet normalization (used by all pretrained backbones)
# =========================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 224):
    """
    Training augmentations that mimic real farm field conditions.

    Augmentation rationale:
    ─────────────────────────────────────────────────────────
    RandomResizedCrop    → farmer holds phone at varying distances
    RandomHorizontalFlip → leaf orientation is arbitrary
    RandomVerticalFlip   → phone may be upside down
    RandomRotation       → tilted phone, angled shots
    RandomPerspective    → non-perpendicular viewing angle
    ColorJitter
      - brightness       → harsh Indian sun vs monsoon overcast
      - contrast         → shadow under tree canopy vs open field
      - saturation       → cheap vs premium phone camera sensors
      - hue              → white balance differences across brands
    GaussianBlur         → out-of-focus shots, shaky hands
    RandomGrayscale      → forces model to use texture, not just color
    RandomErasing        → partial occlusion (other leaves, fingers)
    ─────────────────────────────────────────────────────────
    """

    return transforms.Compose([

        # --- Geometry ---

        # Crop and resize: simulates different distances (60-100% of image)
        transforms.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.6, 1.0),       # zoom range
            ratio=(0.8, 1.2),       # slight aspect ratio variation
        ),

        # Flips: leaves have no fixed orientation
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),

        # Rotation: phone held at different angles
        transforms.RandomRotation(
            degrees=30,
            fill=0,
        ),

        # Perspective warp: non-perpendicular camera angle
        transforms.RandomPerspective(
            distortion_scale=0.15,
            p=0.3,
        ),

        # --- Photometric ---

        # Weather + camera quality variations
        transforms.ColorJitter(
            brightness=0.35,    # harsh sun to overcast
            contrast=0.35,      # shade vs direct light
            saturation=0.30,    # sensor quality differences
            hue=0.08,           # white balance across phone brands
        ),

        # Blur: simulates shaky hands, motion blur, cheap lens
        transforms.RandomApply([
            transforms.GaussianBlur(
                kernel_size=5,
                sigma=(0.1, 2.0),
            )
        ], p=0.25),

        # Grayscale: forces texture learning, not just color
        transforms.RandomGrayscale(p=0.05),

        # --- Tensor conversion ---
        transforms.ToTensor(),

        # ImageNet normalization (required for pretrained backbones)
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),

        # Random erasing: simulates partial occlusion
        # (applied AFTER ToTensor because it operates on tensors)
        transforms.RandomErasing(
            p=0.15,
            scale=(0.02, 0.15),     # erase 2-15% of image
            ratio=(0.3, 3.3),
            value="random",         # fill with random noise
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
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])

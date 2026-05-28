"""
Model definitions for crop disease classification.

Supports:
  - cnn_baseline:      Custom 5-block CNN (~4.9M params)
  - resnet_50:         Pretrained ResNet-50 (ImageNet) with custom head
  - resnet_152:        Pretrained ResNet-152 (ImageNet) with custom head
  - vgg_16:            Pretrained VGG-16 (ImageNet) with custom head
  - vit:               Pretrained ViT-B/16 (ImageNet) with custom head
  - efficientnet_b4:   Pretrained EfficientNet-B4 (ImageNet) with custom head
  - mobilenet_v3:      Pretrained MobileNetV3-Large (ImageNet) with custom head
  - swin_base:         Pretrained Swin-Base (ImageNet) with custom head

Usage:
    model = get_model("resnet_50", num_classes=102)
"""

import torch.nn as nn
import torchvision.models as models


class ConvBlock(nn.Module):
    """Double convolution block: Conv → BN → ReLU → Conv → BN → ReLU → MaxPool → Dropout."""

    def __init__(self, in_channels, out_channels, dropout=0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x):
        return self.block(x)


class CropDiseaseCNN(nn.Module):
    """
    CNN baseline for 102-class crop disease classification.

    Architecture:
        Input:  3 × 224 × 224
        Block1: 3  → 32  (112×112)
        Block2: 32 → 64  (56×56)
        Block3: 64 → 128 (28×28)
        Block4: 128→ 256 (14×14)
        Block5: 256→ 512 (7×7)
        GAP:    512
        FC:     512 → 256 → num_classes

    Total params: ~7.5M (trainable)
    """

    def __init__(self, num_classes=102, dropout_conv=0.25, dropout_fc=0.5):
        super().__init__()

        # Feature extractor: 5 conv blocks
        self.features = nn.Sequential(
            ConvBlock(3,   32,  dropout=dropout_conv),
            ConvBlock(32,  64,  dropout=dropout_conv),
            ConvBlock(64,  128, dropout=dropout_conv),
            ConvBlock(128, 256, dropout=dropout_conv),
            ConvBlock(256, 512, dropout=dropout_conv),
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


def build_model(num_classes=102, dropout_conv=0.25, dropout_fc=0.5):
    """Factory function to create the CNN model."""
    return CropDiseaseCNN(
        num_classes=num_classes,
        dropout_conv=dropout_conv,
        dropout_fc=dropout_fc,
    )


# =========================================================
# PRETRAINED MODELS (Transfer Learning)
# =========================================================

class ResNet50Transfer(nn.Module):
    """ResNet-50 with frozen early layers + custom classifier head."""

    def __init__(self, num_classes=102, dropout_fc=0.5, freeze_backbone=True, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        # Freeze backbone layers (unfreeze later for fine-tuning)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the final FC layer
        in_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(512, num_classes),
        )

        # Unfreeze layer4 + fc for training
        if freeze_backbone:
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


class ResNet152Transfer(nn.Module):
    """ResNet-152 with frozen early layers + custom classifier head."""

    def __init__(self, num_classes=102, dropout_fc=0.5, freeze_backbone=True, pretrained=True):
        super().__init__()
        weights = models.ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet152(weights=weights)

        # Freeze backbone layers
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the final FC layer
        in_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(512, num_classes),
        )

        # Unfreeze layer4 + fc for training
        if freeze_backbone:
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


class VGG16Transfer(nn.Module):
    """VGG-16 with frozen features + custom classifier head."""

    def __init__(self, num_classes=102, dropout_fc=0.5, freeze_backbone=True, pretrained=True):
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.vgg16(weights=weights)

        # Freeze feature extractor
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        # Replace classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


class ViTTransfer(nn.Module):
    """Vision Transformer (ViT-B/16) with custom classifier head."""

    def __init__(self, num_classes=102, dropout_fc=0.5, freeze_backbone=True, pretrained=True):
        super().__init__()
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.vit_b_16(weights=weights)

        # Freeze all layers except the head
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace classification head
        in_features = self.backbone.heads.head.in_features  # 768
        self.backbone.heads.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(512, num_classes),
        )

        # Unfreeze last 2 encoder blocks for fine-tuning
        if freeze_backbone:
            for block in self.backbone.encoder.layers[-2:]:
                for param in block.parameters():
                    param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


class EfficientNetB4Transfer(nn.Module):
    """EfficientNet-B4 with frozen features + custom classifier head."""

    def __init__(self, num_classes=102, dropout_fc=0.5, freeze_backbone=True, pretrained=True):
        super().__init__()
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b4(weights=weights)

        # Freeze all backbone layers
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features  # 1792
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_fc),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc * 0.5),
            nn.Linear(512, num_classes),
        )

        # Unfreeze last 2 blocks of features for fine-tuning
        if freeze_backbone:
            for param in self.backbone.features[-2:].parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


class SwinBaseTransfer(nn.Module):
    """Swin Transformer Base with frozen backbone + custom head."""

    def __init__(self, num_classes=102, dropout_fc=0.5, freeze_backbone=True, pretrained=True):
        super().__init__()
        weights = models.Swin_B_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.swin_b(weights=weights)

        # Freeze all layers
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace classification head
        in_features = self.backbone.head.in_features  # 1024
        self.backbone.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(512, num_classes),
        )

        # Unfreeze last 2 stages for fine-tuning
        if freeze_backbone:
            for param in self.backbone.features[-2:].parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


class MobileNetV3Transfer(nn.Module):
    """MobileNetV3-Large with frozen features + custom classifier head."""

    def __init__(self, num_classes=102, dropout_fc=0.4, freeze_backbone=True, pretrained=True):
        super().__init__()
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.mobilenet_v3_large(weights=weights)

        # Freeze all backbone layers
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace classifier head (original: Linear(960, 1280) → ReLU → Dropout → Linear(1280, 1000))
        in_features = self.backbone.classifier[0].in_features  # 960
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(512, num_classes),
        )

        # Unfreeze last 3 inverted residual blocks for fine-tuning
        if freeze_backbone:
            for param in self.backbone.features[-3:].parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


# =========================================================
# UNIFIED MODEL FACTORY
# =========================================================

MODEL_REGISTRY = {
    "cnn_baseline": "CropDiseaseCNN",
    "resnet_50": "ResNet50Transfer",
    "resnet_152": "ResNet152Transfer",
    "vgg_16": "VGG16Transfer",
    "vit": "ViTTransfer",
    "efficientnet_b4": "EfficientNetB4Transfer",
    "mobilenet_v3": "MobileNetV3Transfer",
    "swin_base": "SwinBaseTransfer",
}


def get_model(model_name, num_classes=102, dropout_fc=0.5, pretrained=True, **kwargs):
    """
    Unified factory to create any supported model by name.

    Args:
        model_name: One of 'cnn_baseline', 'resnet_50', 'resnet_152', 'vgg_16',
                    'vit', 'efficientnet_b4', 'mobilenet_v3', 'swin_base'
        num_classes: Number of output classes
        dropout_fc: Dropout rate for FC layers
        pretrained: If True, load ImageNet weights (needed for training).
                    Set to False for inference from checkpoint (no download).

    Returns:
        nn.Module
    """
    if model_name == "cnn_baseline":
        return CropDiseaseCNN(num_classes=num_classes, dropout_fc=dropout_fc, **kwargs)
    elif model_name == "resnet_50":
        return ResNet50Transfer(num_classes=num_classes, dropout_fc=dropout_fc, pretrained=pretrained)
    elif model_name == "resnet_152":
        return ResNet152Transfer(num_classes=num_classes, dropout_fc=dropout_fc, pretrained=pretrained)
    elif model_name == "vgg_16":
        return VGG16Transfer(num_classes=num_classes, dropout_fc=dropout_fc, pretrained=pretrained)
    elif model_name == "vit":
        return ViTTransfer(num_classes=num_classes, dropout_fc=dropout_fc, pretrained=pretrained)
    elif model_name == "efficientnet_b4":
        return EfficientNetB4Transfer(num_classes=num_classes, dropout_fc=dropout_fc, pretrained=pretrained)
    elif model_name == "mobilenet_v3":
        return MobileNetV3Transfer(num_classes=num_classes, dropout_fc=dropout_fc, pretrained=pretrained)
    elif model_name == "swin_base":
        return SwinBaseTransfer(num_classes=num_classes, dropout_fc=dropout_fc, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_REGISTRY.keys())}")



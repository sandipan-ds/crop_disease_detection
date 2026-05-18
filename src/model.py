"""
CNN baseline model for crop disease classification.

Architecture: 5 conv blocks → Global Average Pooling → FC classifier
Designed as a baseline before moving to transfer learning.
"""

import torch.nn as nn


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

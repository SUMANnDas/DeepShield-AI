"""CIFAKE model architectures, all initialized from scratch."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import efficientnet_b0, resnet50
from torchvision.models.vision_transformer import VisionTransformer


def build_model(architecture: str, num_classes: int = 2) -> nn.Module:
    name = architecture.strip().lower().replace("_", "-")
    if name in {"resnet50", "resnet-50"}:
        model = resnet50(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name in {"efficientnet-b0", "efficientnetb0"}:
        model = efficientnet_b0(weights=None)
        first_conv = model.features[0][0]
        if not isinstance(first_conv, nn.Conv2d):
            raise RuntimeError("Unexpected EfficientNet-B0 stem layout")
        model.features[0][0] = nn.Conv2d(
            first_conv.in_channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=1,
            padding=first_conv.padding,
            dilation=first_conv.dilation,
            groups=first_conv.groups,
            bias=first_conv.bias is not None,
            padding_mode=first_conv.padding_mode,
        )
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if name in {"vit-small", "vit"}:
        return VisionTransformer(
            image_size=32,
            patch_size=4,
            num_layers=6,
            num_heads=3,
            hidden_dim=192,
            mlp_dim=768,
            dropout=0.0,
            attention_dropout=0.0,
            num_classes=num_classes,
        )
    raise ValueError(f"Unsupported image architecture: {architecture}")

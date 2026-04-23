"""Model factory wrappers around timm and torchvision."""
from __future__ import annotations

from typing import Optional

import timm
import torch
import torch.nn as nn

# Supported model name aliases. Keep the catalog small so experiments are reproducible.
SUPPORTED_BACKBONES = {
    "scratch_cnn": None,  # custom small CNN built below
    "mobilenetv3_small_100": "mobilenetv3_small_100",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b3": "efficientnet_b3",
    "resnet50": "resnet50",
    "convnext_tiny": "convnext_tiny",
    "convnext_small": "convnext_small",
    "convnext_base": "convnext_base",
    "regnety_008": "regnety_008",
    "deit_small_patch16_224": "deit_small_patch16_224",
    "swin_tiny_patch4_window7_224": "swin_tiny_patch4_window7_224",
    "convnextv2_base_22k": "convnextv2_base.fcmae_ft_in22k_in1k",
    "convnextv2_large_22k": "convnextv2_large.fcmae_ft_in22k_in1k",
    "eva02_base_448": "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
    "swin_large_384": "swin_large_patch4_window12_384.ms_in22k_ft_in1k",
    "beit_large_224": "beit_large_patch16_224.in22k_ft_in22k_in1k",
    "deit3_large_384": "deit3_large_patch16_384.fb_in22k_ft_in1k",
    "eva02_large_448": "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
}


class TinyCNN(nn.Module):
    """Small from-scratch CNN baseline (~1.2M params) for the A1 experiment."""

    def __init__(self, num_classes: int, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    drop_rate: float = 0.2,
    drop_path_rate: float = 0.0,
) -> nn.Module:
    """Build a model by alias. Falls back to TinyCNN for ``scratch_cnn``."""
    if name == "scratch_cnn":
        return TinyCNN(num_classes=num_classes)
    if name not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unknown backbone {name!r}. Supported: {list(SUPPORTED_BACKBONES)}")

    timm_name = SUPPORTED_BACKBONES[name]
    kwargs = dict(pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
    if drop_path_rate > 0.0:
        kwargs["drop_path_rate"] = drop_path_rate
    model = timm.create_model(timm_name, **kwargs)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the classifier head (works for timm models)."""
    head = model.get_classifier() if hasattr(model, "get_classifier") else None
    head_params = set(map(id, head.parameters())) if head is not None else set()
    for p in model.parameters():
        if id(p) not in head_params:
            p.requires_grad = False


def unfreeze_last_n(model: nn.Module, n_blocks: int) -> None:
    """Unfreeze the last ``n_blocks`` named modules (useful for partial fine-tune)."""
    modules = list(model.named_modules())
    to_unfreeze = modules[-n_blocks:] if n_blocks > 0 else []
    targets = {name for name, _ in to_unfreeze}
    for name, p in model.named_parameters():
        if any(name.startswith(t) for t in targets):
            p.requires_grad = True


def differential_lr_param_groups(
    model: nn.Module, head_lr: float, backbone_lr_factor: float = 0.1
) -> list[dict]:
    """Return parameter groups assigning a smaller LR to the backbone."""
    head = model.get_classifier() if hasattr(model, "get_classifier") else None
    head_param_ids = set(map(id, head.parameters())) if head is not None else set()
    head_params = [p for p in model.parameters() if id(p) in head_param_ids and p.requires_grad]
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids and p.requires_grad]
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": head_lr * backbone_lr_factor})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr})
    return groups

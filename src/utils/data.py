"""Dataset, transforms and stratified split helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "dataset"
SEED = 42

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SplitIndex:
    """Container for train/val/test image paths and labels."""

    train_paths: list[Path]
    train_labels: list[int]
    val_paths: list[Path]
    val_labels: list[int]
    test_paths: list[Path]
    test_labels: list[int]
    classes: list[str]

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.classes)}


def discover_dataset(root: Path = DATASET_DIR) -> tuple[list[Path], list[str], list[str]]:
    """Walk training+validation, return paths, class names per file, and class list."""
    paths: list[Path] = []
    labels: list[str] = []
    classes: set[str] = set()
    for split in ("training", "validation"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            classes.add(class_dir.name)
            for img in class_dir.iterdir():
                if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    paths.append(img)
                    labels.append(class_dir.name)
    return paths, labels, sorted(classes)


def stratified_split(
    paths: Sequence[Path],
    labels: Sequence[str],
    classes: Sequence[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = SEED,
) -> SplitIndex:
    """Stratified 70/15/15 split (test = 1-train-val)."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[label] for label in labels])
    paths_arr = np.array([str(p) for p in paths])

    train_paths, temp_paths, train_y, temp_y = train_test_split(
        paths_arr, y, test_size=1.0 - train_ratio, stratify=y, random_state=seed
    )
    rel_val = val_ratio / (1.0 - train_ratio)
    val_paths, test_paths, val_y, test_y = train_test_split(
        temp_paths, temp_y, test_size=1.0 - rel_val, stratify=temp_y, random_state=seed
    )

    return SplitIndex(
        train_paths=[Path(p) for p in train_paths],
        train_labels=train_y.tolist(),
        val_paths=[Path(p) for p in val_paths],
        val_labels=val_y.tolist(),
        test_paths=[Path(p) for p in test_paths],
        test_labels=test_y.tolist(),
        classes=list(classes),
    )


class SceneImageDataset(Dataset):
    """Lightweight image dataset that always returns RGB tensors."""

    def __init__(
        self,
        paths: Sequence[Path],
        labels: Sequence[int],
        transform: transforms.Compose,
    ) -> None:
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path = self.paths[idx]
        with Image.open(path) as im:
            img = im.convert("RGB")
        return self.transform(img), int(self.labels[idx])


def build_transforms(
    image_size: int = 224,
    augment: bool = True,
    strong: bool = False,
) -> transforms.Compose:
    """Standard ImageNet preprocessing plus optional augmentations for training.

    ``strong=True`` swaps the light pipeline for TrivialAugmentWide + a
    higher RandomErasing probability — useful when the model overfits.
    """
    if augment and strong:
        return transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.TrivialAugmentWide(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(p=0.4, scale=(0.02, 0.2)),
            ]
        )
    if augment:
        return transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_loaders(
    split: SplitIndex,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    strong_augment: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders from a stratified split."""
    train_tfm = build_transforms(image_size, augment=True, strong=strong_augment)
    eval_tfm = build_transforms(image_size, augment=False)

    train_ds = SceneImageDataset(split.train_paths, split.train_labels, train_tfm)
    val_ds = SceneImageDataset(split.val_paths, split.val_labels, eval_tfm)
    test_ds = SceneImageDataset(split.test_paths, split.test_labels, eval_tfm)

    common = dict(num_workers=num_workers, pin_memory=pin_memory)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader

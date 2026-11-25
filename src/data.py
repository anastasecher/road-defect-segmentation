from __future__ import annotations

import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import IMAGE_EXTS, IMAGENET_MEAN, IMAGENET_STD

SPLIT_ALIASES = {"train": "train", "training": "train", "val": "val", "valid": "val",
                 "validation": "val", "test": "test", "testing": "test"}


def find_pairs(root: str | Path) -> list[tuple[Path, Path | None]]:
    root = Path(root)
    images, labels = [], {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            p = Path(dirpath) / f
            ext = p.suffix.lower()
            if ext in IMAGE_EXTS:
                images.append(p)
            elif ext == ".txt" and p.stem.lower() not in {"classes", "readme", "notes"}:
                labels.setdefault(p.stem, []).append(p)

    pairs = []
    for img in sorted(images):
        cands = labels.get(img.stem, [])
        lab = None
        if len(cands) == 1:
            lab = cands[0]
        elif len(cands) > 1:
            mirror = Path(str(img.with_suffix(".txt")).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"))
            lab = mirror if mirror in cands else cands[0]
        pairs.append((img, lab))
    return pairs


def detect_split(path: Path) -> str | None:
    for part in path.parts:
        s = SPLIT_ALIASES.get(part.lower())
        if s:
            return s
    return None


def make_split(n: int, val_frac: float, test_frac: float, seed: int) -> list[str]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    split = ["train"] * n
    for i in idx[:n_test]:
        split[i] = "test"
    for i in idx[n_test:n_test + n_val]:
        split[i] = "val"
    return split


def get_transforms(img_h: int, img_w: int, train: bool):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    if train:
        aug = [
            A.Resize(img_h, img_w),
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.85, 1.2), translate_percent=(-0.05, 0.05), rotate=(-7, 7), p=0.6),
            A.RandomBrightnessContrast(0.25, 0.25, p=0.6),
            A.HueSaturationValue(8, 15, 10, p=0.4),
            A.OneOf([A.MotionBlur(blur_limit=5), A.GaussianBlur(blur_limit=(3, 5))], p=0.2),
            A.CLAHE(p=0.1),
        ]
    else:
        aug = [A.Resize(img_h, img_w)]
    aug += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(aug)


class RoadDefectDataset:

    def __init__(self, index_csv: str | Path, split: str, transform=None, root: str | Path | None = None):
        df = pd.read_csv(index_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.transform = transform
        self.root = Path(root) if root else Path(index_csv).parent

    def __len__(self):
        return len(self.df)

    def _resolve(self, p: str) -> Path:
        p = Path(p)
        return p if p.is_absolute() else self.root / p

    def load_raw(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.df.iloc[i]
        img = cv2.imread(str(self._resolve(row["image_path"])), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self._resolve(row["mask_path"])), cv2.IMREAD_GRAYSCALE)
        return img, mask

    def __getitem__(self, i: int):
        import torch
        img, mask = self.load_raw(i)
        if self.transform is not None:
            out = self.transform(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
        return img, torch.as_tensor(np.asarray(mask), dtype=torch.long)

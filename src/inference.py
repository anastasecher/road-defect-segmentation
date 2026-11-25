from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .config import IMAGENET_MEAN, IMAGENET_STD
from .model import build_model


def load_model(ckpt_path: str | Path, device: torch.device | None = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    model = build_model(meta["arch"], meta["encoder"], meta["num_classes"], encoder_weights=None)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, meta, device


def preprocess(image_rgb: np.ndarray, img_h: int, img_w: int) -> torch.Tensor:
    x = cv2.resize(image_rgb, (img_w, img_h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1))


def remove_small_regions(mask: np.ndarray, min_area: int, class_ids) -> tuple[np.ndarray, dict[int, int]]:
    mask = mask.copy()
    counts = {}
    for c in class_ids:
        n, lab, st, _ = cv2.connectedComponentsWithStats((mask == c).astype(np.uint8))
        small = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] < min_area]
        if small:
            mask[np.isin(lab, small)] = 0
        counts[c] = n - 1 - len(small)
    return mask, counts


@torch.no_grad()
def predict_probs(model, batch: torch.Tensor, device, tta: bool = False) -> torch.Tensor:
    batch = batch.to(device)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        probs = model(batch).float().softmax(1)
        if tta:
            probs = (probs + model(batch.flip(-1)).float().softmax(1).flip(-1)) / 2
    return probs


def predict_mask(model, meta, device, image_rgb: np.ndarray, tta: bool = False) -> tuple[np.ndarray, np.ndarray]:
    h, w = image_rgb.shape[:2]
    x = preprocess(image_rgb, meta["img_h"], meta["img_w"]).unsqueeze(0)
    probs = predict_probs(model, x, device, tta)[0]
    probs = torch.nn.functional.interpolate(probs[None], size=(h, w), mode="bilinear", align_corners=False)[0]
    conf, mask = probs.max(0)
    return mask.cpu().numpy().astype(np.uint8), conf.cpu().numpy()

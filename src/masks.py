from __future__ import annotations

import cv2
import numpy as np

from .labels import Annotation


def polygon_mask(ann: Annotation, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(ann.polygon).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def compose_semantic_mask(object_masks: list[tuple[int, np.ndarray]], h: int, w: int) -> np.ndarray:
    sem = np.zeros((h, w), dtype=np.uint8)
    for cls_idx, m in sorted(object_masks, key=lambda t: -int(t[1].sum())):
        sem[m] = cls_idx
    return sem


class SamRefiner:

    def __init__(self, weights: str = "sam2.1_b.pt", device: str | None = None,
                 dilate_px: int = 6, min_ratio: float = 0.03):
        from ultralytics import SAM
        self.model = SAM(weights)
        self.device = device
        self.dilate_px = dilate_px
        self.min_ratio = min_ratio

    def refine(self, image_bgr: np.ndarray, anns: list[Annotation]) -> list[tuple[np.ndarray, str]]:
        h, w = image_bgr.shape[:2]
        poly_masks = [polygon_mask(a, h, w) for a in anns]
        if not anns:
            return []
        boxes = [list(a.bbox) for a in anns]
        kwargs = {"bboxes": boxes, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        try:
            res = self.model(image_bgr, **kwargs)[0]
            sam_masks = res.masks.data.cpu().numpy() if res.masks is not None else None
        except Exception as exc:
            print(f"  SAM error: {exc}; используется рамка")
            sam_masks = None

        kernel = np.ones((2 * self.dilate_px + 1, 2 * self.dilate_px + 1), np.uint8)
        out = []
        for i, pm in enumerate(poly_masks):
            if sam_masks is None or i >= len(sam_masks):
                out.append((pm, "box"))
                continue
            sm = sam_masks[i]
            if sm.shape != (h, w):
                sm = cv2.resize(sm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            region = cv2.dilate(pm.astype(np.uint8), kernel).astype(bool)
            refined = sm.astype(bool) & region
            ratio = refined.sum() / max(pm.sum(), 1)
            if ratio < self.min_ratio:
                out.append((pm, "box"))
            else:
                out.append((refined, "sam"))
        return out


_FONT_CACHE: dict = {}


def put_text(image_rgb: np.ndarray, text: str, xy: tuple[int, int], color=(255, 255, 255),
             size: int = 18) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont
    if size not in _FONT_CACHE:
        try:
            from matplotlib import font_manager
            _FONT_CACHE[size] = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
        except Exception:
            _FONT_CACHE[size] = ImageFont.load_default()
    im = Image.fromarray(image_rgb)
    ImageDraw.Draw(im).text(xy, text, fill=tuple(color), font=_FONT_CACHE[size],
                            stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(im).copy()


def colorize(mask: np.ndarray, colors: dict[int, tuple[int, int, int]]) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for k, c in colors.items():
        rgb[mask == k] = c
    return rgb


def overlay(image_rgb: np.ndarray, mask: np.ndarray, colors: dict[int, tuple[int, int, int]],
            alpha: float = 0.5) -> np.ndarray:
    out = image_rgb.copy()
    col = colorize(mask, colors)
    fg = mask > 0
    out[fg] = (alpha * col[fg] + (1 - alpha) * out[fg]).astype(np.uint8)
    return out

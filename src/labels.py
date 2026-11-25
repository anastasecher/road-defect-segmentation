from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Annotation:
    cls_id: int
    polygon: np.ndarray

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xs, ys = self.polygon[:, 0], self.polygon[:, 1]
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

    @property
    def area(self) -> float:
        x, y = self.polygon[:, 0], self.polygon[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def parse_line(line: str, img_w: int, img_h: int) -> Annotation | None:
    parts = line.strip().split()
    if not parts:
        return None
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None
    cls_id = int(values[0])
    coords = values[1:]

    if len(coords) == 4:
        xc, yc, w, h = coords
        pts = np.array([
            [xc - w / 2, yc - h / 2], [xc + w / 2, yc - h / 2],
            [xc + w / 2, yc + h / 2], [xc - w / 2, yc + h / 2],
        ], dtype=np.float32)
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        pts = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
    else:
        return None

    pts = np.clip(pts, 0.0, 1.0)
    pts[:, 0] *= img_w
    pts[:, 1] *= img_h
    ann = Annotation(cls_id=cls_id, polygon=pts)
    if ann.area < 1.0:
        return None
    return ann


def read_label_file(path: str | Path, img_w: int, img_h: int,
                    valid_classes: set[int] | None = None) -> tuple[list[Annotation], int]:
    anns: list[Annotation] = []
    skipped = 0
    path = Path(path)
    if not path.exists():
        return anns, 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        ann = parse_line(line, img_w, img_h)
        if ann is None or (valid_classes is not None and ann.cls_id not in valid_classes):
            skipped += 1
            continue
        anns.append(ann)
    return anns, skipped

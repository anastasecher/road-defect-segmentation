from __future__ import annotations

import numpy as np


class SegMetrics:
    def __init__(self, num_classes: int, class_names: list[str] | None = None,
                 defect_ids: list[int] | None = None):
        self.n = num_classes
        self.names = class_names or [str(i) for i in range(num_classes)]
        self.defect_ids = defect_ids if defect_ids is not None else list(range(1, num_classes))
        self.reset()

    def reset(self):
        self.cm = np.zeros((self.n, self.n), dtype=np.int64)

    def update(self, pred, target):
        if hasattr(pred, "detach"):
            pred = pred.detach().cpu().numpy()
        if hasattr(target, "detach"):
            target = target.detach().cpu().numpy()
        p = np.asarray(pred).reshape(-1).astype(np.int64)
        t = np.asarray(target).reshape(-1).astype(np.int64)
        valid = (t >= 0) & (t < self.n)
        self.cm += np.bincount(self.n * t[valid] + p[valid], minlength=self.n ** 2).reshape(self.n, self.n)

    @staticmethod
    def _safe(num, den):
        return float(num / den) if den > 0 else float("nan")

    def per_class(self) -> dict[str, dict[str, float]]:
        res = {}
        total = self.cm.sum()
        for c in range(self.n):
            tp = self.cm[c, c]
            fp = self.cm[:, c].sum() - tp
            fn = self.cm[c, :].sum() - tp
            res[self.names[c]] = {
                "iou": self._safe(tp, tp + fp + fn),
                "dice": self._safe(2 * tp, 2 * tp + fp + fn),
                "precision": self._safe(tp, tp + fp),
                "recall": self._safe(tp, tp + fn),
                "support_px": int(tp + fn),
                "share_px": self._safe(tp + fn, total),
            }
        return res

    def binary(self) -> dict[str, float]:
        d = self.defect_ids
        tp = self.cm[np.ix_(d, d)].sum()
        fp = self.cm[0, d].sum()
        fn = self.cm[d, 0].sum()
        return {
            "iou": self._safe(tp, tp + fp + fn),
            "dice": self._safe(2 * tp, 2 * tp + fp + fn),
            "precision": self._safe(tp, tp + fp),
            "recall": self._safe(tp, tp + fn),
        }

    def summary(self) -> dict:
        pc = self.per_class()
        defect_names = [self.names[i] for i in self.defect_ids]

        def mean_over(key, names):
            vals = [pc[n][key] for n in names if not np.isnan(pc[n][key])]
            return float(np.mean(vals)) if vals else float("nan")

        return {
            "pixel_accuracy": self._safe(np.trace(self.cm), self.cm.sum()),
            "mIoU": mean_over("iou", defect_names),
            "mDice": mean_over("dice", defect_names),
            "mPrecision": mean_over("precision", defect_names),
            "mRecall": mean_over("recall", defect_names),
            "mIoU_with_bg": mean_over("iou", self.names),
            "binary_defect": self.binary(),
            "per_class": pc,
            "confusion_matrix": self.cm.tolist(),
        }

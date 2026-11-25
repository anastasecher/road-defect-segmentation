from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CLASS_COLORS, CLASS_NAMES, CLASS_NAMES_RU, DEFECT_CLASS_IDS, NUM_CLASSES
from src.data import RoadDefectDataset
from src.inference import load_model, predict_probs, preprocess
from src.masks import overlay, put_text
from src.metrics import SegMetrics


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="prepared")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None, help="по умолчанию <папка чекпойнта>/eval_<split>")
    ap.add_argument("--tta", action="store_true", help="усреднение с отражённым изображением")
    ap.add_argument("--n-vis", type=int, default=12)
    ap.add_argument("--full-res", action="store_true",
                    help="считать метрики в исходном разрешении (иначе — во входном разрешении сети)")
    return ap.parse_args()


def save_confusion(cm: np.ndarray, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES), CLASS_NAMES_RU)
    ax.set_yticks(range(NUM_CLASSES), CLASS_NAMES_RU)
    ax.set_xlabel("Прогноз"); ax.set_ylabel("Истина")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Нормированная матрица ошибок (по пикселям)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_bars(table: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = table[table["class"] != "background"]
    keys = ["iou", "dice", "precision", "recall"]
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, key in enumerate(keys):
        ax.bar(x + (k - 1.5) * 0.2, t[key], 0.2, label=key)
    ax.set_xticks(x, t["class_ru"]); ax.set_ylim(0, 1); ax.legend(ncols=4); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Метрики сегментации по классам (тест)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    args = parse_args()
    model, meta, device = load_model(args.ckpt)
    out = Path(args.out) if args.out else Path(args.ckpt).parent / f"eval_{args.split}"
    (out / "vis").mkdir(parents=True, exist_ok=True)
    ds = RoadDefectDataset(Path(args.data) / "index.csv", args.split, transform=None)
    print(f"{args.split}: {len(ds)} изображений, модель {meta['arch']}/{meta['encoder']}, устройство {device}")

    met = SegMetrics(NUM_CLASSES, CLASS_NAMES, DEFECT_CLASS_IDS)
    per_image, times = [], []
    for i in tqdm(range(len(ds)), desc="тест"):
        img, gt = ds.load_raw(i)
        x = preprocess(img, meta["img_h"], meta["img_w"]).unsqueeze(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        probs = predict_probs(model, x, device, args.tta)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

        if args.full_res:
            probs = F.interpolate(probs, size=gt.shape, mode="bilinear", align_corners=False)
            target = gt
        else:
            target = cv2.resize(gt, (meta["img_w"], meta["img_h"]), interpolation=cv2.INTER_NEAREST)
        pred = probs.argmax(1)[0].cpu().numpy().astype(np.uint8)
        met.update(pred, target)

        im_met = SegMetrics(NUM_CLASSES, CLASS_NAMES, DEFECT_CLASS_IDS)
        im_met.update(pred, target)
        b = im_met.binary()
        per_image.append({"image": Path(ds.df.iloc[i]["image_path"]).name, "defect_iou": b["iou"],
                          "gt_defect_px": int((target > 0).sum()), "pred_defect_px": int((pred > 0).sum())})

        if i < args.n_vis:
            small = cv2.resize(img, (pred.shape[1], pred.shape[0]))
            panel = np.concatenate([small, overlay(small, target, CLASS_COLORS, 0.6),
                                    overlay(small, pred, CLASS_COLORS, 0.6)], axis=1)
            for k, title in enumerate(["Изображение", "Разметка", "Прогноз"]):
                panel = put_text(panel, title, (k * small.shape[1] + 8, 6))
            cv2.imwrite(str(out / "vis" / f"{i:03d}.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    s = met.summary()
    warm = times[3:] if len(times) > 5 else times
    s["inference_ms_per_image"] = float(np.mean(warm) * 1000)
    s["fps"] = float(1.0 / np.mean(warm))
    s["tta"] = args.tta
    s["split"] = args.split
    s["n_images"] = len(ds)
    s["model"] = f"{meta['arch']}/{meta['encoder']}"
    (out / "metrics.json").write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = [{"class": n, "class_ru": CLASS_NAMES_RU[k], **{m: s["per_class"][n][m] for m in
             ("iou", "dice", "precision", "recall", "share_px")}} for k, n in enumerate(CLASS_NAMES)]
    rows.append({"class": "mean_defects", "class_ru": "среднее (дефекты)", "iou": s["mIoU"], "dice": s["mDice"],
                 "precision": s["mPrecision"], "recall": s["mRecall"], "share_px": np.nan})
    rows.append({"class": "defect_vs_bg", "class_ru": "дефект/фон", **s["binary_defect"], "share_px": np.nan})
    table = pd.DataFrame(rows)
    table.to_csv(out / "metrics_table.csv", index=False, float_format="%.4f")
    pd.DataFrame(per_image).to_csv(out / "per_image.csv", index=False, float_format="%.4f")
    save_confusion(np.array(s["confusion_matrix"]), out / "confusion_matrix.png")
    save_bars(table, out / "metrics_by_class.png")

    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Pixel accuracy: {s['pixel_accuracy']:.4f} | время: {s['inference_ms_per_image']:.1f} мс/кадр "
          f"({s['fps']:.1f} FPS)")
    print(f"Результаты: {out}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CLASS_COLORS, CLASS_NAMES, NUM_CLASSES, SEED, SOURCE_CLASSES
from src.data import detect_split, find_pairs, make_split
from src.labels import read_label_file
from src.masks import SamRefiner, compose_semantic_mask, overlay, polygon_mask


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="папка датасета (ищется рекурсивно)")
    ap.add_argument("--out", default="prepared", help="куда сохранить маски и index.csv")
    ap.add_argument("--mask-mode", choices=["box", "sam"], default="sam",
                    help="box — залитые рамки; sam — уточнение рамок моделью SAM 2")
    ap.add_argument("--sam-weights", default="sam2.1_b.pt")
    ap.add_argument("--device", default=None, help="cuda / cpu для SAM")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--use-existing-split", action="store_true",
                    help="использовать папки train/val/test датасета, если они есть")
    ap.add_argument("--keep-unlabeled", action="store_true",
                    help="изображения без txt включать как «без дефектов»")
    ap.add_argument("--vis", type=int, default=12, help="сколько примеров разметки сохранить")
    ap.add_argument("--limit", type=int, default=0, help="обработать только N изображений (отладка)")
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    (out / "examples").mkdir(exist_ok=True)

    pairs = find_pairs(args.data_root)
    print(f"Найдено изображений: {len(pairs)}; с разметкой: {sum(l is not None for _, l in pairs)}")
    if not args.keep_unlabeled:
        pairs = [(i, l) for i, l in pairs if l is not None]
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        sys.exit("Нет пар изображение+разметка. Проверьте --data-root.")

    refiner = SamRefiner(args.sam_weights, device=args.device) if args.mask_mode == "sam" else None
    valid = set(SOURCE_CLASSES)

    rows, used_stems = [], Counter()
    obj_counts = Counter()
    src_counts = Counter()
    bad_lines = 0
    for k, (img_path, lab_path) in enumerate(tqdm(pairs, desc="маски")):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  не читается: {img_path}")
            continue
        h, w = img.shape[:2]
        anns, skipped = read_label_file(lab_path, w, h, valid) if lab_path else ([], 0)
        bad_lines += skipped

        if refiner is not None and anns:
            masks_src = refiner.refine(img, anns)
        else:
            masks_src = [(polygon_mask(a, h, w), "box") for a in anns]
        obj_masks = []
        for a, (m, src) in zip(anns, masks_src):
            obj_masks.append((a.cls_id + 1, m))
            obj_counts[a.cls_id + 1] += 1
            src_counts[(a.cls_id + 1, src)] += 1
        sem = compose_semantic_mask(obj_masks, h, w)

        stem = img_path.stem
        used_stems[stem] += 1
        name = stem if used_stems[stem] == 1 else f"{stem}__{used_stems[stem]}"
        mask_rel = f"masks/{name}.png"
        cv2.imwrite(str(out / mask_rel), sem)

        px = np.bincount(sem.reshape(-1), minlength=NUM_CLASSES)
        row = {"image_path": str(img_path.resolve()), "mask_path": mask_rel,
               "width": w, "height": h, "orig_split": detect_split(img_path) or ""}
        for c in range(1, NUM_CLASSES):
            row[f"n_{CLASS_NAMES[c]}"] = sum(1 for cid, _ in obj_masks if cid == c)
        for c in range(NUM_CLASSES):
            row[f"px_{CLASS_NAMES[c]}"] = int(px[c])
        rows.append(row)

        if k < args.vis:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            vis = overlay(rgb, sem, CLASS_COLORS, alpha=0.55)
            for a in anns:
                cv2.polylines(vis, [np.round(a.polygon).astype(np.int32)], True, (255, 255, 255), 1)
            both = np.concatenate([rgb, vis], axis=1)
            cv2.imwrite(str(out / "examples" / f"{name}.jpg"), cv2.cvtColor(both, cv2.COLOR_RGB2BGR))

    df = pd.DataFrame(rows)
    has_orig = args.use_existing_split and (df["orig_split"] != "").all() and df["orig_split"].nunique() >= 2
    if has_orig:
        df["split"] = df["orig_split"]
        if "val" not in set(df["split"]):
            tr = df.index[df["split"] == "train"].tolist()
            sub = make_split(len(tr), args.val / (1 - args.test), 0.0, args.seed)
            for i, s in zip(tr, sub):
                df.at[i, "split"] = s
    else:
        df["split"] = make_split(len(df), args.val, args.test, args.seed)
    df.to_csv(out / "index.csv", index=False)

    stats = {"mask_mode": args.mask_mode, "images": len(df), "skipped_label_lines": bad_lines,
             "split_sizes": df["split"].value_counts().to_dict(), "objects": {}, "mask_source": {},
             "pixels_train": {}}
    for c in range(1, NUM_CLASSES):
        stats["objects"][CLASS_NAMES[c]] = int(obj_counts[c])
        stats["mask_source"][CLASS_NAMES[c]] = {s: int(src_counts[(c, s)]) for s in ("sam", "box")}
    per_split = defaultdict(dict)
    for s, g in df.groupby("split"):
        for c in range(1, NUM_CLASSES):
            per_split[s][CLASS_NAMES[c]] = int(g[f"n_{CLASS_NAMES[c]}"].sum())
    stats["objects_per_split"] = per_split
    tr = df[df["split"] == "train"]
    stats["pixels_train"] = {CLASS_NAMES[c]: int(tr[f"px_{CLASS_NAMES[c]}"].sum()) for c in range(NUM_CLASSES)}
    (out / "dataset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.config import CLASS_NAMES_RU
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        names = CLASS_NAMES_RU[1:]
        ax[0].bar(names, [obj_counts[c] for c in range(1, NUM_CLASSES)],
                  color=[np.array(CLASS_COLORS[c]) / 255 for c in range(1, NUM_CLASSES)])
        ax[0].set_title("Число объектов по классам")
        px = [stats["pixels_train"][CLASS_NAMES[c]] for c in range(NUM_CLASSES)]
        ax[1].bar(CLASS_NAMES_RU, np.array(px) / max(sum(px), 1) * 100,
                  color=[np.array(CLASS_COLORS[c]) / 255 if c else (0.6, 0.6, 0.6) for c in range(NUM_CLASSES)])
        ax[1].set_yscale("log")
        ax[1].set_title("Доля пикселей классов, % (train, лог. шкала)")
        fig.tight_layout()
        fig.savefig(out / "class_distribution.png", dpi=150)
    except Exception as exc:
        print("График не построен:", exc)
    print(f"Готово: {out / 'index.csv'}")


if __name__ == "__main__":
    main()

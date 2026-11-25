from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CLASS_COLORS, CLASS_NAMES, CLASS_NAMES_RU, DEFECT_CLASS_IDS, IMAGE_EXTS
from src.inference import load_model, predict_mask, remove_small_regions
from src.knowledge import InferenceEngine, KnowledgeBase, extract_facts
from src.masks import overlay, put_text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="изображение или папка")
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--kb", default=None, help="файл базы знаний")
    ap.add_argument("--min-area", type=int, default=30, help="мин. площадь области в пикселях")
    args = ap.parse_args()

    model, meta, device = load_model(args.ckpt)
    kb = KnowledgeBase(args.kb) if args.kb else KnowledgeBase()
    engine = InferenceEngine(kb)
    inp = Path(args.input)
    files = [inp] if inp.is_file() else sorted(p for p in inp.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    out = Path(args.out)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    (out / "overlay").mkdir(exist_ok=True)

    rows = []
    for p in files:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask, conf = predict_mask(model, meta, device, rgb, args.tta)

        mask, regions = remove_small_regions(mask, args.min_area, DEFECT_CLASS_IDS)
        row = {"image": p.name}
        for c in DEFECT_CLASS_IDS:
            row[f"{CLASS_NAMES[c]}_regions"] = regions[c]
            row[f"{CLASS_NAMES[c]}_area_%"] = round(100 * float((mask == c).mean()), 3)
        res = engine.run(extract_facts(mask, conf, kb.params.get("near_zone_from", 0.6)))
        row["state"] = res.level_name
        row["priority"] = res.priority
        row["rules"] = ",".join(f["id"] for f in res.fired)
        row["recommendations"] = " | ".join(res.recommendations)
        row["mean_conf_defect"] = round(float(conf[mask > 0].mean()), 3) if (mask > 0).any() else None
        rows.append(row)

        cv2.imwrite(str(out / "masks" / f"{p.stem}.png"), mask)
        vis = overlay(rgb, mask, CLASS_COLORS, 0.55)
        vis = put_text(vis, f"Состояние: {res.level_name}", (8, vis.shape[0] - 26), (255, 255, 255), 17)
        y = 6
        for c in DEFECT_CLASS_IDS:
            if row[f"{CLASS_NAMES[c]}_regions"]:
                txt = (f"{CLASS_NAMES_RU[c]}: {row[f'{CLASS_NAMES[c]}_regions']} обл., "
                       f"{row[f'{CLASS_NAMES[c]}_area_%']}% кадра")
                vis = put_text(vis, txt, (8, y), CLASS_COLORS[c], 16)
                y += 22
        cv2.imwrite(str(out / "overlay" / f"{p.stem}.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    print(df.to_string(index=False))
    print("Легенда:", ", ".join(f"{CLASS_NAMES_RU[c]} — {CLASS_NAMES[c]}" for c in DEFECT_CLASS_IDS))
    print(f"Сохранено в {out}")


if __name__ == "__main__":
    main()

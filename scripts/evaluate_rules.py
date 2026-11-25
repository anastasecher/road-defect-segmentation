from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DEFECT_CLASS_IDS
from src.data import RoadDefectDataset
from src.inference import load_model, predict_mask, remove_small_regions
from src.knowledge import InferenceEngine, KnowledgeBase, extract_facts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="prepared")
    ap.add_argument("--split", default="test")
    ap.add_argument("--kb", default=None)
    ap.add_argument("--min-area", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    kb = KnowledgeBase(args.kb) if args.kb else KnowledgeBase()
    engine = InferenceEngine(kb)
    nz = kb.params.get("near_zone_from", 0.6)
    model, meta, device = load_model(args.ckpt)
    ds = RoadDefectDataset(Path(args.data) / "index.csv", args.split)
    out = Path(args.out) if args.out else Path(args.ckpt).parent / f"rules_{args.split}"
    out.mkdir(parents=True, exist_ok=True)

    rows, fired_gt, fired_pred = [], Counter(), Counter()
    for i in tqdm(range(len(ds)), desc="правила"):
        img, gt = ds.load_raw(i)
        pred, conf = predict_mask(model, meta, device, img)
        pred, _ = remove_small_regions(pred, args.min_area, DEFECT_CLASS_IDS)
        gt_c, _ = remove_small_regions(gt, args.min_area, DEFECT_CLASS_IDS)
        r_gt = engine.run(extract_facts(gt_c, None, nz))
        r_pr = engine.run(extract_facts(pred, conf, nz))
        fired_gt.update(f["id"] for f in r_gt.fired)
        fired_pred.update(f["id"] for f in r_pr.fired)
        rows.append({"image": Path(ds.df.iloc[i]["image_path"]).name,
                     "level_gt": r_gt.level, "level_pred": r_pr.level,
                     "state_gt": r_gt.level_name, "state_pred": r_pr.level_name,
                     "warning_gt": bool(r_gt.facts.get("driver_warning")),
                     "warning_pred": bool(r_pr.facts.get("driver_warning"))})

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_image.csv", index=False)
    levels = sorted(kb.levels)
    names = [kb.levels[l]["name"] for l in levels]
    cm = pd.crosstab(pd.Categorical(df.level_gt, levels), pd.Categorical(df.level_pred, levels), dropna=False)
    cm.index, cm.columns = [f"эталон: {n}" for n in names], [f"прогноз: {n}" for n in names]
    cm.to_csv(out / "confusion_levels.csv")

    diff = (df.level_gt - df.level_pred).abs()
    urgent_gt, urgent_pred = df.level_gt >= 3, df.level_pred >= 3
    tp = int((urgent_gt & urgent_pred).sum())
    summary = {
        "n_images": len(df),
        "exact_agreement": float((diff == 0).mean()),
        "within_one_level": float((diff <= 1).mean()),
        "repair_needed_precision": tp / max(int(urgent_pred.sum()), 1),
        "repair_needed_recall": tp / max(int(urgent_gt.sum()), 1),
        "driver_warning_agreement": float((df.warning_gt == df.warning_pred).mean()),
        "distribution_gt": df.state_gt.value_counts().to_dict(),
        "distribution_pred": df.state_pred.value_counts().to_dict(),
        "rules_fired_gt": dict(sorted(fired_gt.items())),
        "rules_fired_pred": dict(sorted(fired_pred.items())),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(cm.to_string())
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("rules")}, ensure_ascii=False, indent=2))
    print(f"Сохранено в {out}")


if __name__ == "__main__":
    main()

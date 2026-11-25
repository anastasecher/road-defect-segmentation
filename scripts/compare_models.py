from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="папки запусков (внутри eval_test/metrics.json)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="runs/comparison.csv")
    args = ap.parse_args()

    rows = []
    for r in args.runs:
        r = Path(r)
        mfile = r / f"eval_{args.split}" / "metrics.json"
        if not mfile.exists():
            print("нет", mfile)
            continue
        m = json.loads(mfile.read_text(encoding="utf-8"))
        cfg = json.loads((r / "config.json").read_text(encoding="utf-8")) if (r / "config.json").exists() else {}
        row = {"модель": m["model"], "параметров, млн": round(cfg.get("params_M", float("nan")), 1),
               "mIoU": m["mIoU"], "mDice": m["mDice"], "mPrecision": m["mPrecision"], "mRecall": m["mRecall"],
               "IoU выбоина": m["per_class"]["pothole"]["iou"], "IoU трещина": m["per_class"]["crack"]["iou"],
               "IoU люк": m["per_class"]["manhole"]["iou"], "IoU дефект/фон": m["binary_defect"]["iou"],
               "мс/кадр": m["inference_ms_per_image"]}
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("mIoU", ascending=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, float_format="%.4f")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (CLASS_NAMES, DEFAULT_IMG_H, DEFAULT_IMG_W, DEFECT_CLASS_IDS,
                        NUM_CLASSES, SEED)
from src.data import RoadDefectDataset, get_transforms
from src.metrics import SegMetrics
from src.model import DiceCELoss, build_model, compute_class_weights


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="prepared", help="папка с index.csv (результат prepare_dataset.py)")
    ap.add_argument("--out", default="runs/unet_resnet34")
    ap.add_argument("--arch", default="unet", help="unet | unetplusplus | deeplabv3plus | fpn | pspnet")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--no-pretrained", action="store_true", help="не загружать веса ImageNet")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--img-h", type=int, default=DEFAULT_IMG_H)
    ap.add_argument("--img-w", type=int, default=DEFAULT_IMG_W)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=10, help="ранняя остановка по val mIoU")
    ap.add_argument("--class-weights", choices=["auto", "none"], default="auto")
    ap.add_argument("--max-steps", type=int, default=0, help="ограничить число шагов в эпохе (отладка)")
    ap.add_argument("--seed", type=int, default=SEED)
    return ap.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_loader(model, loader, loss_fn, device, max_steps=0):
    model.eval()
    met = SegMetrics(NUM_CLASSES, CLASS_NAMES, DEFECT_CLASS_IDS)
    total, n = 0.0, 0
    for step, (x, y) in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x)
        total += loss_fn(logits.float(), y).item() * x.size(0)
        n += x.size(0)
        met.update(logits.argmax(1), y)
    return total / max(n, 1), met.summary()


def plot_history(hist: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(hist.epoch, hist.train_loss, label="train")
    ax[0].plot(hist.epoch, hist.val_loss, label="val")
    ax[0].set_title("Функция потерь"); ax[0].legend(); ax[0].set_xlabel("эпоха")
    ax[1].plot(hist.epoch, hist.val_mIoU, label="mIoU")
    ax[1].plot(hist.epoch, hist.val_mDice, label="mDice")
    ax[1].set_title("Валидация (среднее по дефектам)"); ax[1].legend(); ax[1].set_xlabel("эпоха")
    for c in DEFECT_CLASS_IDS:
        ax[2].plot(hist.epoch, hist[f"val_iou_{CLASS_NAMES[c]}"], label=CLASS_NAMES[c])
    ax[2].set_title("IoU по классам (val)"); ax[2].legend(); ax[2].set_xlabel("эпоха")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Устройство:", device)

    index = Path(args.data) / "index.csv"
    train_ds = RoadDefectDataset(index, "train", get_transforms(args.img_h, args.img_w, True))
    val_ds = RoadDefectDataset(index, "val", get_transforms(args.img_h, args.img_w, False))
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers,
                          pin_memory=device.type == "cuda", drop_last=len(train_ds) > args.batch_size)
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers)

    model = build_model(args.arch, args.encoder, NUM_CLASSES,
                        None if args.no_pretrained else "imagenet").to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Модель {args.arch}/{args.encoder}: {n_params / 1e6:.1f} млн параметров")

    if args.class_weights == "auto":
        df = train_ds.df
        counts = [int(df[f"px_{n}"].sum()) for n in CLASS_NAMES]
        weights = compute_class_weights(counts)
    else:
        weights = torch.ones(NUM_CLASSES)
    print("Веса классов:", dict(zip(CLASS_NAMES, [round(float(w), 2) for w in weights])))
    loss_fn = DiceCELoss(NUM_CLASSES, weights).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = min(len(train_dl), args.max_steps) if args.max_steps else len(train_dl)
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup = min(200, total_steps // 10)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    meta = {"arch": args.arch, "encoder": args.encoder, "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES, "img_h": args.img_h, "img_w": args.img_w}
    (out / "config.json").write_text(json.dumps({**vars(args), **meta,
                                                 "class_weights": weights.tolist(),
                                                 "params_M": n_params / 1e6}, indent=2, ensure_ascii=False))

    history, best, bad_epochs = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        pbar = tqdm(train_dl, desc=f"эпоха {epoch}/{args.epochs}", leave=False)
        for step, (x, y) in enumerate(pbar):
            if args.max_steps and step >= args.max_steps:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
            loss = loss_fn(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * x.size(0)
            seen += x.size(0)
            pbar.set_postfix(loss=f"{run_loss / seen:.4f}")

        val_loss, s = evaluate_loader(model, val_dl, loss_fn, device, args.max_steps)
        rec = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": run_loss / max(seen, 1),
               "val_loss": val_loss, "val_mIoU": s["mIoU"], "val_mDice": s["mDice"],
               "val_mPrecision": s["mPrecision"], "val_mRecall": s["mRecall"],
               "val_binary_iou": s["binary_defect"]["iou"], "time_s": time.time() - t0}
        for c in DEFECT_CLASS_IDS:
            rec[f"val_iou_{CLASS_NAMES[c]}"] = s["per_class"][CLASS_NAMES[c]]["iou"]
        history.append(rec)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(f"эпоха {epoch:3d} | train {rec['train_loss']:.4f} | val {val_loss:.4f} | "
              f"mIoU {s['mIoU']:.4f} | mDice {s['mDice']:.4f} | "
              + " ".join(f"{CLASS_NAMES[c]}={rec[f'val_iou_{CLASS_NAMES[c]}']:.3f}" for c in DEFECT_CLASS_IDS)
              + f" | {rec['time_s']:.0f} c")

        ckpt = {"model": model.state_dict(), "meta": meta, "epoch": epoch, "val": s}
        torch.save(ckpt, out / "last.pt")
        score = s["mIoU"] if not np.isnan(s["mIoU"]) else 0.0
        if score > best:
            best, bad_epochs = score, 0
            torch.save(ckpt, out / "best.pt")
            print(f"   ↑ лучшая модель сохранена (val mIoU = {best:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Ранняя остановка: {args.patience} эпох без улучшения")
                break

    plot_history(pd.DataFrame(history), out / "training_curves.png")
    print(f"Лучший val mIoU: {best:.4f}. Файлы: {out}")


if __name__ == "__main__":
    main()

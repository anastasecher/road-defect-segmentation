from __future__ import annotations

import csv
import html
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import CLASS_COLORS, CLASS_NAMES, CLASS_NAMES_RU, DEFECT_CLASS_IDS

LEVEL_COLORS = {0: "#22A447", 1: "#7CB518", 2: "#E0B000", 3: "#F07A10", 4: "#E02424"}


def _to_deg(value) -> float:
    d, m, s = (float(x) for x in value)
    return d + m / 60 + s / 3600


def read_gps(path: str | Path) -> tuple[float, float] | None:
    try:
        from PIL import Image
        with Image.open(path) as im:
            gps = im.getexif().get_ifd(0x8825)
        if not gps or 2 not in gps or 4 not in gps:
            return None
        lat, lon = _to_deg(gps[2]), _to_deg(gps[4])
        if gps.get(1, "N") in ("S", b"S"):
            lat = -lat
        if gps.get(3, "E") in ("W", b"W"):
            lon = -lon
        if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat == 0 and lon == 0):
            return None
        return round(lat, 6), round(lon, 6)
    except Exception:
        return None


def parse_coords(text: str | None) -> tuple[float, float] | None:
    if not text or not text.strip():
        return None
    try:
        parts = [float(x) for x in text.replace(";", ",").split(",")]
        if len(parts) == 2 and -90 <= parts[0] <= 90 and -180 <= parts[1] <= 180:
            return round(parts[0], 6), round(parts[1], 6)
    except ValueError:
        pass
    return None


@dataclass
class FrameResult:
    name: str
    overlay: np.ndarray
    mask: np.ndarray
    level: int
    state: str
    priority: str
    areas: dict
    regions: dict
    messages: list
    recommendations: list
    rules: list
    gps: tuple[float, float] | None = None
    time_s: float | None = None

    @property
    def defect_area(self) -> float:
        return self.areas.get("pothole", 0.0) + self.areas.get("crack", 0.0)

    def record(self, source: str) -> dict:
        r = {"время": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "источник": source, "кадр": self.name,
             "состояние": self.state, "уровень": self.level, "приоритет": self.priority}
        for c in DEFECT_CLASS_IDS:
            r[f"{CLASS_NAMES_RU[c]}, областей"] = self.regions[CLASS_NAMES[c]]
            r[f"{CLASS_NAMES_RU[c]}, % кадра"] = self.areas[CLASS_NAMES[c]]
        r["широта"] = self.gps[0] if self.gps else ""
        r["долгота"] = self.gps[1] if self.gps else ""
        r["правила"] = ", ".join(self.rules)
        r["рекомендации"] = " | ".join(self.recommendations)
        return r


class Journal:

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, records: list[dict]):
        if not records:
            return
        new = not self.path.exists() or self.path.stat().st_size == 0
        fields = list(records[0].keys())
        if not new:
            with self.path.open(encoding="utf-8-sig", newline="") as f:
                fields = next(csv.reader(f), fields)
        with self.path.open("a", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerows(records)

    def load(self):
        import pandas as pd
        if not self.path.exists() or self.path.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(self.path, encoding="utf-8-sig")

    def clear(self):
        if self.path.exists():
            self.path.unlink()


def compare_before_after(before: FrameResult, after: FrameResult, max_ratio: float = 0.3) -> dict:
    a0, a1 = before.defect_area, after.defect_area
    out = {"area_before": round(a0, 2), "area_after": round(a1, 2), "level_before": before.level,
           "level_after": after.level, "ratio": None}
    if a0 <= 0:
        out.update(verdict="сравнение не требуется", detail="На снимке «до» дефекты не обнаружены.")
        return out
    ratio = a1 / a0
    out["ratio"] = round(ratio, 3)
    potholes_after = after.regions.get("pothole", 0)
    if after.level <= 1 or (ratio <= max_ratio and potholes_after == 0):
        out.update(verdict="ремонт выполнен",
                   detail=f"Площадь дефектов уменьшилась с {a0:.2f} % до {a1:.2f} % кадра, выбоин не осталось.")
    elif ratio < 1:
        out.update(verdict="ремонт выполнен частично",
                   detail=f"Площадь дефектов уменьшилась с {a0:.2f} % до {a1:.2f} % кадра "
                          f"({ratio * 100:.0f} % от исходной), выбоин осталось: {potholes_after}.")
    else:
        out.update(verdict="ремонт не выполнен",
                   detail=f"Площадь дефектов не уменьшилась: {a0:.2f} % → {a1:.2f} % кадра.")
    return out


def sample_video(path: str | Path, step_s: float = 1.0, max_frames: int = 300):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Не удалось открыть видео")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(step_s * fps)))
    idx, taken = 0, 0
    try:
        while taken < max_frames and (n == 0 or idx < n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            yield idx / fps, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            taken += 1
            idx += step
    finally:
        cap.release()


def video_timeline_png(results: list[FrameResult], out_path: str | Path, theme_dark=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = [r.time_s for r in results]
    fg, bg, grid = ("#e5e5e5", "#0b0b0b", "#2a2a2a") if theme_dark else ("#111111", "#FFFFFF", "#E4E4E1")
    fig, ax = plt.subplots(2, 1, figsize=(10, 4.2), sharex=True, facecolor=bg,
                           gridspec_kw={"height_ratios": [1, 1.2]})
    for a in ax:
        a.set_facecolor(bg); a.tick_params(colors=fg); a.grid(color=grid, lw=0.6)
        for s in a.spines.values():
            s.set_color(grid)
    ax[0].scatter(t, [r.level for r in results], c=[LEVEL_COLORS[r.level] for r in results], s=26, zorder=3)
    ax[0].step(t, [r.level for r in results], where="mid", color="#111111", lw=1.2, alpha=0.5)
    ax[0].set_yticks(range(5)); ax[0].set_ylim(-0.5, 4.5)
    ax[0].set_ylabel("уровень", color=fg)
    for c, col in ((1, CLASS_COLORS[1]), (2, CLASS_COLORS[2]), (3, CLASS_COLORS[3])):
        ax[1].plot(t, [r.areas[CLASS_NAMES[c]] for r in results], color=np.array(col) / 255, lw=1.6,
                   label=CLASS_NAMES_RU[c])
    ax[1].set_ylabel("% кадра", color=fg); ax[1].set_xlabel("время видео, с", color=fg)
    leg = ax[1].legend(facecolor=bg, edgecolor=grid, labelcolor=fg, fontsize=8, ncols=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=bg)
    plt.close(fig)


def map_html(points: list[dict], height: int = 520) -> str:
    if not points:
        return ("<div style='padding:40px;text-align:center;color:#5B5B57;border:2px dashed #C9C9C4;border-radius:28px'>"
                "Нет кадров с координатами. Координаты берутся из GPS-данных фото (EXIF) "
                "или вводятся вручную на вкладке «Кадр».</div>")
    lat = sum(p["lat"] for p in points) / len(points)
    lon = sum(p["lon"] for p in points) / len(points)
    data = json.dumps([{**p, "color": LEVEL_COLORS.get(int(p["level"]), "#999")} for p in points], ensure_ascii=False)
    fallback = _offline_svg(points, height)
    page = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#m{{height:100%;margin:0;background:#F3F3F1;font-family:Manrope,system-ui,sans-serif}}
.leaflet-tile-pane{{filter:grayscale(1) brightness(1.05)}}
.leaflet-popup-content-wrapper,.leaflet-popup-tip{{background:#111111;color:#F3F3F1;border-radius:14px}}
#fb{{display:none;height:100%}}</style>
</head><body><div id="m"></div><div id="fb">{fallback}</div><script>
var pts={data};
if (typeof L === 'undefined') {{ document.getElementById('m').style.display='none';
  document.getElementById('fb').style.display='block'; }}
else {{ var map=L.map('m').setView([{lat},{lon}],14);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{maxZoom:19,attribution:'© OpenStreetMap'}}).addTo(map);
var g=[];pts.forEach(function(p){{var c=L.circleMarker([p.lat,p.lon],{{radius:9,color:p.color,weight:2,fillColor:p.color,fillOpacity:.55}})
.bindPopup('<b>'+p.state+'</b><br>'+p.name+'<br>'+p.time).addTo(map);g.push([p.lat,p.lon]);}});
if(g.length>1)map.fitBounds(g,{{padding:[30,30]}}); }}
</script></body></html>"""
    return (f"<iframe style='width:100%;height:{height}px;border:none;border-radius:28px' "
            f"srcdoc=\"{html.escape(page, quote=True)}\"></iframe>")


def _offline_svg(points: list[dict], height: int) -> str:
    W, H, pad = 1000, height, 60
    lats = [p["lat"] for p in points]; lons = [p["lon"] for p in points]
    la0, la1 = min(lats), max(lats); lo0, lo1 = min(lons), max(lons)
    span = max(la1 - la0, (lo1 - lo0) * math.cos(math.radians(sum(lats) / len(lats))), 1e-4)
    k = min((W - 2 * pad), (H - 2 * pad)) / span
    cx, cy = (lo0 + lo1) / 2, (la0 + la1) / 2
    coslat = math.cos(math.radians(cy))
    def xy(la, lo):
        return W / 2 + (lo - cx) * coslat * k, H / 2 - (la - cy) * k
    grid = "".join(f"<line x1='{x}' y1='0' x2='{x}' y2='{H}' stroke='#E4E4E1'/>" for x in range(0, W, 50)) + \
        "".join(f"<line x1='0' y1='{y}' x2='{W}' y2='{y}' stroke='#E4E4E1'/>" for y in range(0, H, 50))
    marks = ""
    for p in points:
        x, y = xy(p["lat"], p["lon"]); c = LEVEL_COLORS.get(int(p["level"]), "#999")
        marks += (f"<circle cx='{x:.1f}' cy='{y:.1f}' r='16' fill='{c}' opacity='.18'/>"
                  f"<circle cx='{x:.1f}' cy='{y:.1f}' r='7' fill='{c}' stroke='#FFFFFF' stroke-width='2'/>"
                  f"<text x='{x + 14:.1f}' y='{y - 10:.1f}' fill='#111111' font-size='13'>{html.escape(str(p['state']))}</text>"
                  f"<text x='{x + 14:.1f}' y='{y + 6:.1f}' fill='#5B5B57' font-size='11'>{p['lat']:.4f}, {p['lon']:.4f}</text>")
    note = ("<text x='20' y='28' fill='#111111' font-size='12' font-weight='700' letter-spacing='2'>СХЕМА БЕЗ ПОДЛОЖКИ · НЕТ ДОСТУПА К КАРТЕ</text>")
    return (f"<svg viewBox='0 0 {W} {H}' width='100%' height='100%' preserveAspectRatio='xMidYMid meet' "
            f"style='background:#FFFFFF'>{grid}{note}{marks}</svg>")


def pdf_report(results: list[FrameResult], out_path: str | Path, title: str = "Отчет о состоянии покрытия"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    A4 = (8.27, 11.69)
    ordered = sorted(results, key=lambda r: -r.level)
    with PdfPages(out_path) as pdf:
        fig = plt.figure(figsize=A4)
        fig.text(0.08, 0.95, title, fontsize=17, weight="bold")
        fig.text(0.08, 0.925, f"Сформирован {datetime.now():%d.%m.%Y %H:%M} · кадров: {len(results)}", fontsize=9,
                 color="#555")
        counts = {}
        for r in results:
            counts[r.state] = counts.get(r.state, 0) + 1
        y = 0.89
        for st, n in sorted(counts.items(), key=lambda kv: -max(r.level for r in results if r.state == kv[0])):
            fig.text(0.08, y, f"{st}: {n}", fontsize=10); y -= 0.02
        rows = [[r.name[:28], r.state, r.priority, f"{r.areas['pothole']:.2f}", f"{r.areas['crack']:.2f}",
                 (f"{r.gps[0]:.5f}, {r.gps[1]:.5f}" if r.gps else "—")] for r in ordered[:40]]
        ax = fig.add_axes([0.06, 0.05, 0.88, y - 0.07]); ax.axis("off")
        if rows:
            t = ax.table(cellText=rows, colLabels=["Кадр", "Состояние", "Приоритет", "Выбоины, %", "Трещины, %",
                                                   "Координаты"], loc="upper center", cellLoc="left")
            t.auto_set_font_size(False); t.set_fontsize(7.5); t.scale(1, 1.25)
            for (i, j), cell in t.get_celld().items():
                cell.set_edgecolor("#bbb")
                if i == 0:
                    cell.set_facecolor("#efefef")
        pdf.savefig(fig); plt.close(fig)
        for r in [r for r in ordered if r.level >= 1][:30]:
            fig = plt.figure(figsize=A4)
            fig.text(0.08, 0.95, r.name, fontsize=13, weight="bold")
            fig.text(0.08, 0.925, f"Состояние: {r.state} · приоритет: {r.priority}"
                     + (f" · координаты: {r.gps[0]:.5f}, {r.gps[1]:.5f}" if r.gps else "")
                     + (f" · время видео: {r.time_s:.1f} с" if r.time_s is not None else ""), fontsize=9)
            ax = fig.add_axes([0.08, 0.5, 0.84, 0.4]); ax.imshow(r.overlay); ax.axis("off")
            y = 0.46
            for c in DEFECT_CLASS_IDS:
                n = CLASS_NAMES[c]
                fig.text(0.08, y, f"{CLASS_NAMES_RU[c]}: областей {r.regions[n]}, {r.areas[n]:.2f} % кадра", fontsize=10)
                y -= 0.022
            y -= 0.01
            for head, items in (("Выводы:", r.messages), ("Рекомендации:", r.recommendations)):
                if items:
                    fig.text(0.08, y, head, fontsize=10, weight="bold"); y -= 0.022
                    for it in items:
                        for line in _wrap(it, 95):
                            fig.text(0.1, y, line, fontsize=9); y -= 0.019
                    y -= 0.01
            fig.text(0.08, y, "Сработавшие правила: " + ", ".join(r.rules), fontsize=8, color="#555")
            pdf.savefig(fig); plt.close(fig)
    return str(out_path)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines or [""]

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import CLASS_COLORS, CLASS_NAMES, CLASS_NAMES_RU, DEFECT_CLASS_IDS, NUM_CLASSES
from src.inference import load_model, predict_mask, remove_small_regions
from src.knowledge import InferenceEngine, KnowledgeBase, extract_facts
from src.masks import colorize, overlay
from src.metrics import SegMetrics
from src.monitoring import (LEVEL_COLORS, FrameResult, Journal, compare_before_after, map_html, parse_coords,
                            pdf_report, read_gps, sample_video, video_timeline_png)

ACCENT = "#F5EE0A"
INK = "#111111"
OUT_DIR = Path(tempfile.gettempdir()) / "road_defect_app"
OUT_DIR.mkdir(exist_ok=True)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="веса модели (best.pt)")
    ap.add_argument("--data", default="prepared", help="папка с index.csv — для примеров и эталонных масок")
    ap.add_argument("--split", default="test")
    ap.add_argument("--examples-dir", default=None, help="папка с кадрами-примерами (вместо выборки из index.csv)")
    ap.add_argument("--n-examples", type=int, default=24)
    ap.add_argument("--kb", default=None, help="файл базы знаний (по умолчанию configs/knowledge_base.json)")
    ap.add_argument("--journal", default="logs/journal.csv", help="файл журнала проверок")
    ap.add_argument("--share", action="store_true", help="создать публичную ссылку gradio.live")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--inbrowser", action="store_true", help="сразу открыть интерфейс в браузере")
    return ap.parse_args()


def confidence_map(rgb: np.ndarray, conf: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB).astype(np.float32) * 0.55 + 90
    unc = np.clip((1.0 - conf) / 0.5, 0, 1)[..., None]
    heat = np.array([245, 238, 10], np.float32) * (1 - unc) + np.array([224, 36, 36], np.float32) * unc
    a = np.clip(unc * 1.6, 0, 0.85)
    return (gray * (1 - a) + heat * a).clip(0, 255).astype(np.uint8)


def _path(f):
    return f if isinstance(f, str) else getattr(f, "name", None) or getattr(f, "path", None)


class Segmenter:
    def __init__(self, ckpt: str, data: str, split: str, kb_path: str | None = None,
                 journal: str = "logs/journal.csv"):
        self.model, self.meta, self.device = load_model(ckpt)
        self.kb = KnowledgeBase(kb_path) if kb_path else KnowledgeBase()
        self.engine = InferenceEngine(self.kb)
        self.journal = Journal(journal)
        self.gt_by_image: dict[str, Path] = {}
        self.examples: list[str] = []
        index = Path(data) / "index.csv"
        if index.exists():
            df = pd.read_csv(index)
            for _, r in df.iterrows():
                self.gt_by_image[Path(r["image_path"]).name] = Path(data) / r["mask_path"]
            sub = df[df["split"] == split]
            sub = sub.assign(_k=(sub.filter(like="n_") > 0).sum(axis=1)).sort_values("_k", ascending=False)
            self.examples = [p for p in sub["image_path"] if Path(p).exists()]

    def analyze(self, rgb: np.ndarray, name: str, min_area: int = 30, tta: bool = False, alpha: float = 0.55,
                gps=None, time_s=None):
        mask, conf = predict_mask(self.model, self.meta, self.device, rgb, tta)
        mask, regions = remove_small_regions(mask, int(min_area), DEFECT_CLASS_IDS)
        res = self.engine.run(extract_facts(mask, conf, self.kb.params.get("near_zone_from", 0.6)))
        fr = FrameResult(
            name=name, overlay=overlay(rgb, mask, CLASS_COLORS, alpha), mask=mask, level=res.level,
            state=res.level_name, priority=res.priority,
            areas={CLASS_NAMES[c]: round(100 * float((mask == c).mean()), 2) for c in DEFECT_CLASS_IDS},
            regions={CLASS_NAMES[c]: regions[c] for c in DEFECT_CLASS_IDS},
            messages=res.messages, recommendations=res.recommendations, rules=[f["id"] for f in res.fired],
            gps=gps, time_s=time_s)
        return fr, res, conf

    @staticmethod
    def _read(path):
        bgr = cv2.imread(path) if path else None
        return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def segment(self, image_path, alpha, min_area, tta, coords_text):
        import gradio as gr
        empty = (None, None, gr.update(value=None, visible=False), pd.DataFrame(), "", "", "", None)
        if not image_path:
            return empty[:4] + ("Загрузите изображение или выберите кадр из списка.",) + empty[5:]
        rgb = self._read(image_path)
        if rgb is None:
            return empty[:4] + ("⚠️ Файл не удалось прочитать как изображение. Поддерживаются JPG, PNG, BMP.",) + empty[5:]
        gps = parse_coords(coords_text) or read_gps(image_path)
        t0 = time.perf_counter()
        fr, res, conf = self.analyze(rgb, Path(image_path).name, min_area, tta, alpha, gps)
        dt = (time.perf_counter() - t0) * 1000
        self.journal.add([fr.record("кадр")])

        rows = []
        for c in DEFECT_CLASS_IDS:
            sel = fr.mask == c
            rows.append({"Класс": CLASS_NAMES_RU[c], "Областей": fr.regions[CLASS_NAMES[c]],
                         "Площадь, % кадра": fr.areas[CLASS_NAMES[c]],
                         "Ср. уверенность": round(float(conf[sel].mean()), 3) if sel.any() else None})
        found = [f"{r['Класс']} ({r['Областей']})" for r in rows if r["Областей"]]
        md = [f"**Обнаружено:** {', '.join(found)}" if found else "**Дефекты не обнаружены.**",
              f"Время обработки: {dt:.0f} мс · {self.meta['arch']}/{self.meta['encoder']} · {self.device.type.upper()}",
              f"Координаты: {gps[0]:.6f}, {gps[1]:.6f}" if gps else "Координаты: нет (можно указать вручную)"]
        gt_vis = None
        gt_path = self.gt_by_image.get(Path(image_path).name)
        if gt_path is not None and gt_path.exists():
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            gt_vis = overlay(rgb, gt, CLASS_COLORS, alpha)
            m = SegMetrics(NUM_CLASSES, CLASS_NAMES, DEFECT_CLASS_IDS)
            m.update(fr.mask, gt)
            b = m.binary()
            md.append(f"Сравнение с разметкой: IoU «дефект/фон» = **{b['iou']:.3f}**, "
                      f"точность = {b['precision']:.3f}, полнота = {b['recall']:.3f}")
        report = pdf_report([fr], OUT_DIR / f"заявка_{Path(image_path).stem}.pdf", "Заявка на обследование участка")
        import gradio as gr
        return (fr.overlay, confidence_map(rgb, conf), gr.update(value=gt_vis, visible=gt_vis is not None),
                pd.DataFrame(rows), "\n\n".join(md), self.format_assessment(res), res.explanation(), report)

    def format_assessment(self, res) -> str:
        color = LEVEL_COLORS.get(res.level, "#888")
        out = [f"<div class='state' style='border-color:{color}'><span class='dot' style='background:{color}'></span>"
               f"<b>{res.level_name.upper()}</b><span class='prio'>приоритет: {res.priority}</span></div>"]
        if res.facts.get("driver_warning"):
            out.append("<p class='warn'>▲ Выбоина в ближней зоне перед автомобилем</p>")
        if res.messages:
            out.append("<p class='h'>Выводы</p><ul>" + "".join(f"<li>{m}</li>" for m in res.messages) + "</ul>")
        if res.recommendations:
            out.append("<p class='h'>Рекомендации</p><ul>" + "".join(f"<li>{m}</li>" for m in res.recommendations) + "</ul>")
        return "".join(out)

    def batch(self, files, min_area, tta):
        if not files:
            return [], pd.DataFrame(), None, None, map_html([])
        gallery, rows, results, records = [], [], [], []
        for f in files:
            path = _path(f)
            rgb = self._read(path)
            if rgb is None:
                rows.append({"Кадр": Path(path).name, "Состояние": "ошибка: файл не является изображением"})
                continue
            fr, res, _ = self.analyze(rgb, Path(path).name, min_area, tta, gps=read_gps(path))
            results.append(fr)
            rec = fr.record("пакет")
            records.append(rec)
            rows.append({k: v for k, v in rec.items() if k not in ("время", "источник", "уровень", "рекомендации")})
            gallery.append((fr.overlay, f"{fr.name} — {fr.state}"))
        self.journal.add(records)
        df = pd.DataFrame(rows).rename(columns={"кадр": "Кадр", "состояние": "Состояние"})
        csv_path = OUT_DIR / "пакет.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        pdf_path = pdf_report(results, OUT_DIR / "отчет_пакет.pdf") if results else None
        pts = [{"lat": r.gps[0], "lon": r.gps[1], "state": r.state, "level": r.level, "name": r.name, "time": ""}
               for r in results if r.gps]
        return gallery, df, str(csv_path), pdf_path, map_html(pts, 420)

    def video(self, file, step_s, max_frames, min_area, progress=None):
        path = _path(file)
        if not path:
            return None, [], pd.DataFrame(), None, None
        results = []
        try:
            for t, rgb in sample_video(path, float(step_s), int(max_frames)):
                fr, _, _ = self.analyze(rgb, f"{Path(path).stem} @ {t:.1f} с", min_area, time_s=round(t, 2))
                results.append(fr)
        except ValueError as e:
            raise RuntimeError(f"Видео не удалось прочитать: {e}")
        if not results:
            raise RuntimeError("В видео не найдено ни одного кадра")
        self.journal.add([r.record("видео") for r in results])
        png = OUT_DIR / "видео_график.png"
        video_timeline_png(results, png)
        worst = sorted(results, key=lambda r: (-r.level, -r.defect_area))[:6]
        gallery = [(r.overlay, f"{r.time_s:.1f} с — {r.state}") for r in worst]
        df = pd.DataFrame([{"время, с": r.time_s, "состояние": r.state, "приоритет": r.priority,
                            "выбоины, %": r.areas["pothole"], "трещины, %": r.areas["crack"],
                            "люки": r.regions["manhole"], "правила": ", ".join(r.rules)} for r in results])
        csv_path = OUT_DIR / "видео.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        pdf_path = pdf_report(results, OUT_DIR / "отчет_видео.pdf", "Отчет по видеозаписи участка")
        return str(png), gallery, df, str(csv_path), pdf_path

    def before_after(self, before_path, after_path, ratio_pct, min_area):
        if not before_path or not after_path:
            return None, None, "<p>Загрузите оба снимка участка.</p>", pd.DataFrame()
        a, b = self._read(before_path), self._read(after_path)
        if a is None or b is None:
            return None, None, "<p>⚠️ Один из файлов не является изображением.</p>", pd.DataFrame()
        fa, _, _ = self.analyze(a, "до", min_area)
        fb, _, _ = self.analyze(b, "после", min_area)
        v = compare_before_after(fa, fb, float(ratio_pct) / 100)
        color = {"ремонт выполнен": "#4ade80", "ремонт выполнен частично": "#facc15",
                 "ремонт не выполнен": "#ef4444"}.get(v["verdict"], "#9ca3af")
        html = (f"<div class='state' style='border-color:{color}'><span class='dot' style='background:{color}'></span>"
                f"<b>{v['verdict'].upper()}</b></div><p>{v['detail']}</p>")
        df = pd.DataFrame([{"Снимок": "до", "Состояние": fa.state, "Выбоины, %": fa.areas["pothole"],
                            "Трещины, %": fa.areas["crack"], "Выбоин": fa.regions["pothole"]},
                           {"Снимок": "после", "Состояние": fb.state, "Выбоины, %": fb.areas["pothole"],
                            "Трещины, %": fb.areas["crack"], "Выбоин": fb.regions["pothole"]}])
        return fa.overlay, fb.overlay, html, df

    def journal_view(self, state_filter):
        df = self.journal.load()
        if df.empty:
            return df, None
        if state_filter and state_filter != "все":
            df = df[df["состояние"] == state_filter]
        df = df.iloc[::-1].reset_index(drop=True)
        path = OUT_DIR / "журнал.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return df, str(path)

    def journal_clear(self):
        self.journal.clear()
        return pd.DataFrame(), None

    def map_view(self):
        df = self.journal.load()
        if df.empty or "широта" not in df:
            return map_html([]), "Записей с координатами: 0"
        df = df[pd.to_numeric(df["широта"], errors="coerce").notna()]
        df = df.drop_duplicates(subset=["кадр", "широта", "долгота"], keep="last")
        pts = [{"lat": float(r["широта"]), "lon": float(r["долгота"]), "state": r["состояние"],
                "level": int(r["уровень"]), "name": str(r["кадр"]), "time": str(r["время"])} for _, r in df.iterrows()]
        return map_html(pts), f"Записей с координатами: {len(pts)}"


LOGO = """<div class="brand"><svg width="34" height="34" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg" aria-label="ROADSEG">
<path d="M5 37 L15.5 3 M35 37 L24.5 3" stroke="#111111" stroke-width="5" stroke-linecap="butt"/>
<path d="M20 6 V10 M20 15 V20 M20 25 V31" stroke="#111111" stroke-width="2.6"/>
<rect x="22.5" y="20.5" width="9" height="8" fill="#F5EE0A" stroke="#111111" stroke-width="2"/>
</svg><span class="word">ROADSEG</span></div>"""

CSS = """
.gradio-container { background:#FFFFFF !important; max-width:1360px !important; }
footer { display:none !important; }
#hdr { display:flex; align-items:center; gap:22px; padding:16px 4px 16px; border-bottom:1px solid #E6E6E6; margin-bottom:6px; }
#hdr .brand { display:flex; align-items:center; gap:10px; }
#hdr .word { font-size:25px; font-weight:800; letter-spacing:-.01em; color:#111111; line-height:1; }
#hdr .sep { width:1px; height:26px; background:#D9D9D9; }
#hdr .title { font-size:15px; color:#555555; }
#hdr .legend { margin-left:auto; color:#333333; font-size:13px; }
.legend span.sw { display:inline-block; width:10px; height:10px; border-radius:50%; margin:0 6px 0 16px; vertical-align:middle; }
button[role=tab] { font-size:14px; color:#666666 !important; }
button[role=tab][aria-selected=true] { color:#111111 !important; border-bottom:2px solid #111111 !important; font-weight:600; }
.block, .form { border-radius:2px !important; box-shadow:none !important; }
.state { display:flex; align-items:center; gap:12px; padding:12px 16px; border:1px solid #E6E6E6; border-left:5px solid;
  border-radius:2px; background:#FFFFFF; font-size:16px; color:#111111; }
.state b { color:#111111 !important; font-weight:700; }
.state .dot { display:none; }
.state .prio { margin-left:auto; color:#666666; font-size:14px; }
p.h { color:#111111; font-size:14px; font-weight:700; margin:14px 0 4px; }
p.warn { color:#E02424; font-weight:700; margin-top:10px; }
.prose, .md { padding-left:6px !important; }
table, table * { font-family:inherit !important; }
"""


def build_theme():
    import gradio as gr
    return gr.themes.Base(
        primary_hue=gr.themes.Color(c50="#fffde0", c100="#fffbb8", c200="#fff88a", c300="#fbf45a", c400="#f8f130",
                                    c500="#F5EE0A", c600="#d6cf00", c700="#a8a300", c800="#7a7600", c900="#4d4a00",
                                    c950="#2b2900"),
        neutral_hue="stone",
        font=[gr.themes.Font("Inter"), gr.themes.Font("Helvetica Neue"), gr.themes.Font("Arial"), gr.themes.Font("sans-serif")],
        font_mono=[gr.themes.Font("Inter"), gr.themes.Font("Helvetica Neue"), gr.themes.Font("Arial"), gr.themes.Font("sans-serif")],
        radius_size="sm",
    ).set(
        body_background_fill="#FFFFFF", body_background_fill_dark="#FFFFFF",
        body_text_color=INK, body_text_color_dark=INK,
        block_background_fill="#FFFFFF", block_background_fill_dark="#FFFFFF",
        block_border_color="#E6E6E6", block_border_color_dark="#E6E6E6",
        block_label_background_fill="#FFFFFF", block_label_background_fill_dark="#FFFFFF",
        block_label_text_color=INK, block_label_text_color_dark=INK,
        block_title_text_color=INK, block_title_text_color_dark=INK,
        input_background_fill="#FAFAFA", input_background_fill_dark="#FAFAFA",
        border_color_primary="#E6E6E6", border_color_primary_dark="#E6E6E6",
        button_primary_background_fill=ACCENT, button_primary_background_fill_dark=ACCENT,
        button_primary_background_fill_hover="#E6DF00", button_primary_background_fill_hover_dark="#E6DF00",
        button_primary_text_color=INK, button_primary_text_color_dark=INK,
        button_secondary_background_fill=INK, button_secondary_background_fill_dark=INK,
        button_secondary_background_fill_hover="#333333", button_secondary_background_fill_hover_dark="#333333",
        button_secondary_text_color="#FFFFFF", button_secondary_text_color_dark="#FFFFFF",
        button_large_radius="2px", button_small_radius="2px",
        slider_color=INK, slider_color_dark=INK,
        color_accent_soft="#FFF9B0", color_accent_soft_dark="#FFF9B0",
        table_even_background_fill="#FFFFFF", table_even_background_fill_dark="#FFFFFF",
        table_odd_background_fill="#FAFAFA", table_odd_background_fill_dark="#FAFAFA",
        background_fill_primary="#FFFFFF", background_fill_primary_dark="#FFFFFF",
        background_fill_secondary="#FAFAFA", background_fill_secondary_dark="#FAFAFA",
    )


def header_html() -> str:
    legend = "".join(f"<span class='sw' style='background:rgb{CLASS_COLORS[c]}'></span>{CLASS_NAMES_RU[c]}"
                     for c in DEFECT_CLASS_IDS)
    return (f"<div id='hdr'>{LOGO}<div class='sep'></div>"
            f"<div class='title'>Сегментация дефектов дорожного покрытия</div>"
            f"<div class='legend'>{legend}</div></div>")


def _gradio_major() -> int:
    import gradio as gr
    try:
        return int(gr.__version__.split(".")[0])
    except ValueError:
        return 5


def style_kwargs() -> dict:
    return {"theme": build_theme(), "css": CSS}


def build_ui(seg: Segmenter):
    import gradio as gr

    blocks_kw = {} if _gradio_major() >= 6 else style_kwargs()
    with gr.Blocks(title="Сегментация дефектов дорожного покрытия", **blocks_kw) as demo:
        gr.HTML(header_html())
        with gr.Tab("Кадр"):
            with gr.Row():
                with gr.Column(scale=1):
                    inp = gr.Image(type="filepath", label="Исходный кадр", height=280)
                    coords = gr.Textbox(label="Координаты (если в фото нет GPS)", placeholder="52.0976, 23.7341")
                    alpha = gr.Slider(0.1, 1.0, 0.55, step=0.05, label="Прозрачность маски")
                    min_area = gr.Slider(0, 500, 30, step=10, label="Мин. площадь области, пикс.")
                    tta = gr.Checkbox(False, label="TTA (усреднение с отраженным кадром)")
                    btn = gr.Button("Обновить с новыми настройками", variant="secondary")
                    report = gr.File(label="Заявка на обследование (PDF)")
                with gr.Column(scale=2):
                    out_overlay = gr.Image(label="Результат", height=360)
                    with gr.Row():
                        out_mask = gr.Image(label="Карта уверенности модели", height=190)
                        out_gt = gr.Image(label="Эталонная разметка", height=190, visible=False)
                    out_assess = gr.HTML()
                    with gr.Accordion("Объяснение вывода: сработавшие правила базы знаний", open=False):
                        out_expl = gr.Textbox(lines=8, show_label=False)
                    out_md = gr.Markdown()
                    out_table = gr.Dataframe(label="Сводка по классам", interactive=False)
            outs = [out_overlay, out_mask, out_gt, out_table, out_md, out_assess, out_expl, report]
            args = [inp, alpha, min_area, tta, coords]
            btn.click(seg.segment, args, outs)
            inp.change(seg.segment, args, outs)
            if seg.examples:
                gr.Examples(examples=[[p] for p in seg.examples], inputs=[inp], label="Кадры для проверки",
                            examples_per_page=12)
        with gr.Tab("Пакет"):
            files = gr.File(file_count="multiple", file_types=["image"], label="Кадры маршрута")
            with gr.Row():
                b_min = gr.Slider(0, 500, 30, step=10, label="Мин. площадь области, пикс.")
                b_tta = gr.Checkbox(False, label="TTA")
                b_btn = gr.Button("Обработать", variant="primary")
            gal = gr.Gallery(label="Результаты", columns=4, height="auto")
            b_table = gr.Dataframe(label="Сводка по кадрам", interactive=False)
            with gr.Row():
                b_csv = gr.File(label="Таблица (CSV)")
                b_pdf = gr.File(label="Отчет (PDF)")
            b_map = gr.HTML()
            b_btn.click(seg.batch, [files, b_min, b_tta], [gal, b_table, b_csv, b_pdf, b_map])
        with gr.Tab("Видео"):
            vfile = gr.File(file_types=["video"], label="Видео с регистратора (MP4, AVI, MOV)")
            with gr.Row():
                v_step = gr.Slider(0.2, 5, 1.0, step=0.2, label="Шаг между кадрами, с")
                v_max = gr.Slider(10, 600, 120, step=10, label="Не более кадров")
                v_min = gr.Slider(0, 500, 30, step=10, label="Мин. площадь области, пикс.")
                v_btn = gr.Button("Разобрать видео", variant="primary")
            v_plot = gr.Image(label="Состояние покрытия по времени", height=300)
            v_gal = gr.Gallery(label="Самые проблемные моменты", columns=3, height="auto")
            v_table = gr.Dataframe(label="Кадры", interactive=False, max_height=320)
            with gr.Row():
                v_csv = gr.File(label="Таблица (CSV)")
                v_pdf = gr.File(label="Отчет (PDF)")
            v_btn.click(seg.video, [vfile, v_step, v_max, v_min], [v_plot, v_gal, v_table, v_csv, v_pdf])
        with gr.Tab("До / после"):
            gr.Markdown("Загрузите два снимка одного участка: до ремонта и после. Система сравнит площадь выбоин и "
                        "трещин и сделает вывод о выполнении ремонта. Снимки желательно делать с одной точки.")
            with gr.Row():
                im_b = gr.Image(type="filepath", label="До ремонта", height=240)
                im_a = gr.Image(type="filepath", label="После ремонта", height=240)
            with gr.Row():
                ba_ratio = gr.Slider(5, 80, 30, step=5, label="Остаток площади дефектов, при котором ремонт засчитан, %")
                ba_min = gr.Slider(0, 500, 30, step=10, label="Мин. площадь области, пикс.")
                ba_btn = gr.Button("Сравнить", variant="primary")
            with gr.Row():
                ba_o1 = gr.Image(label="До", height=260)
                ba_o2 = gr.Image(label="После", height=260)
            ba_html = gr.HTML()
            ba_table = gr.Dataframe(interactive=False)
            ba_btn.click(seg.before_after, [im_b, im_a, ba_ratio, ba_min], [ba_o1, ba_o2, ba_html, ba_table])
        with gr.Tab("Карта"):
            with gr.Row():
                m_btn = gr.Button("Обновить карту", variant="primary", scale=0)
                m_info = gr.Markdown()
            m_html = gr.HTML()
            m_btn.click(seg.map_view, None, [m_html, m_info])
        with gr.Tab("Журнал"):
            with gr.Row():
                j_filter = gr.Dropdown(["все"] + [seg.kb.levels[k]["name"] for k in sorted(seg.kb.levels)],
                                       value="все", label="Состояние")
                j_btn = gr.Button("Показать", variant="primary")
                j_clear = gr.Button("Очистить журнал", variant="stop")
            j_table = gr.Dataframe(interactive=False, max_height=520)
            j_csv = gr.File(label="Журнал (CSV)")
            j_btn.click(seg.journal_view, [j_filter], [j_table, j_csv])
            j_clear.click(seg.journal_clear, None, [j_table, j_csv])
        with gr.Tab("База знаний"):
            gr.Markdown("Пороги и продукционные правила хранятся в `configs/knowledge_base.json` и меняются без "
                        "правки кода. Машина вывода применяет правила прямым выводом до неподвижной точки; итоговое "
                        "состояние — максимальный уровень сработавших правил.")
            desc = seg.kb.param_descriptions
            gr.Dataframe(pd.DataFrame([{"Параметр": k, "Значение": v, "Смысл": desc.get(k, "")}
                                       for k, v in seg.kb.params.items()]),
                         interactive=False, label="Параметры (пороги)", wrap=True, column_widths=["22%", "10%", "68%"])
            gr.Dataframe(pd.DataFrame([{
                "id": r["id"], "Правило": r["name"],
                "ЕСЛИ": " И ".join(f"{a} {o} {b}" for a, o, b in r["if"]).replace("True", "да"),
                "ТО": "; ".join(filter(None, [
                    f"уровень {r['then']['severity']} ({seg.kb.levels[r['then']['severity']]['name']})"
                    if "severity" in r["then"] else "",
                    ", ".join(f"{k} = да" if v is True else f"{k} = {v}" for k, v in r["then"].get("facts", {}).items()),
                    "" if ("severity" in r["then"] or r["then"].get("facts")) else "сообщение и рекомендация"]))}
                for r in seg.kb.rules]), interactive=False, label="Продукционные правила", wrap=True,
                max_height=900, column_widths=["6%", "22%", "44%", "28%"])
    return demo


def main():
    args = parse_args()
    seg = Segmenter(args.ckpt, args.data, args.split, args.kb, args.journal)
    if args.examples_dir:
        seg.examples = sorted(str(p.resolve()) for p in Path(args.examples_dir).iterdir()
                              if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    seg.examples = seg.examples[: args.n_examples]
    print(f"Модель загружена ({seg.meta['arch']}/{seg.meta['encoder']}, {seg.device}); "
          f"примеров: {len(seg.examples)}; журнал: {seg.journal.path}")
    launch_kw = style_kwargs() if _gradio_major() >= 6 else {}
    build_ui(seg).launch(share=args.share, server_port=args.port, show_error=True, inbrowser=args.inbrowser,
                         allowed_paths=sorted({str(Path(p).parent) for p in seg.examples} | {str(OUT_DIR)}),
                         **launch_kw)


if __name__ == "__main__":
    main()

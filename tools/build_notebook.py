from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
FILES = ["src/__init__.py", "src/config.py", "src/labels.py", "src/masks.py", "src/data.py", "src/metrics.py",
         "src/model.py", "src/inference.py", "scripts/prepare_dataset.py", "scripts/train.py",
         "scripts/evaluate.py", "scripts/predict.py", "scripts/compare_models.py", "scripts/evaluate_rules.py",
         "src/knowledge.py", "src/monitoring.py", "tests/test_monitoring.py", "configs/knowledge_base.json", "tests/test_core.py", "tests/test_knowledge.py", "app.py"]

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

def code_cells():
    out = [md("## 0. Код проекта\nЯчейки ниже создают файлы проекта в `/kaggle/working/road_defect_seg`."),
           code("import os\nos.makedirs('/kaggle/working/road_defect_seg', exist_ok=True)\n"
                "%cd /kaggle/working/road_defect_seg\nfor d in ['src', 'scripts', 'tests', 'configs']:\n"
                "    os.makedirs(d, exist_ok=True)")]
    for f in FILES:
        out.append(code(f"%%writefile {f}\n" + (ROOT / f).read_text(encoding="utf-8")))
    return out


ZIP_SMALL = ("!cd /kaggle/working && rm -f {name} && zip -qr {name} road_defect_seg/runs "
             "road_defect_seg/prepared/dataset_stats.json road_defect_seg/prepared/class_distribution.png "
             "road_defect_seg/prepared/examples road_defect_seg/predictions -x '*.pt' && ls -lh {name}")

cells = [
    md("# Сегментация дефектов дорожного покрытия на основе СНС\n"
       "Курсовая работа, Черноиван А. Н., гр. ИИ-26.\n\n"
       "**Перед запуском:** справа в панели *Input* добавьте датасет "
       "`lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes`; в *Settings* включите "
       "**Accelerator: GPU** и **Internet: On**. Затем *Run All*. Полный прогон занимает ~1,5–2 ч."),
] + code_cells()

cells += [
    md("## 1. Окружение"),
    code("!pip install -q segmentation-models-pytorch ultralytics\n"
         "import torch; print(torch.__version__, torch.cuda.is_available(), "
         "torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\n"
         "!python -m pytest -q tests"),
    code("DATA_ROOT = '/kaggle/input'\n"
         "MASK_MODE = 'sam'\n"
         "EPOCHS = 40\nBATCH = 8\nWORKERS = 4\n"
         "MODELS = [('unet', 'resnet34'), ('deeplabv3plus', 'resnet34'), ('fpn', 'resnet34')]\n"
         "RUN_BOX_ABLATION = False"),
    md("## 2. Формирование набора данных\nРамки разметки преобразуются в попиксельные маски, "
       "данные делятся на train/val/test = 70/15/15."),
    code("!python scripts/prepare_dataset.py --data-root {DATA_ROOT} --out prepared --mask-mode {MASK_MODE} --vis 16"),
    code("from IPython.display import Image, display\nimport glob\n"
         "display(Image('prepared/class_distribution.png'))\n"
         "for p in sorted(glob.glob('prepared/examples/*.jpg'))[:6]:\n    display(Image(p))"),
    md("## 3. Обучение моделей"),
    code("for arch, enc in MODELS:\n"
         "    !python scripts/train.py --data prepared --arch {arch} --encoder {enc} --epochs {EPOCHS} "
         "--batch-size {BATCH} --workers {WORKERS} --out runs/{arch}_{enc}"),
    code("if RUN_BOX_ABLATION:\n"
         "    !python scripts/prepare_dataset.py --data-root {DATA_ROOT} --out prepared_box --mask-mode box --vis 0\n"
         "    !python scripts/train.py --data prepared_box --arch unet --encoder resnet34 --epochs {EPOCHS} "
         "--batch-size {BATCH} --workers {WORKERS} --out runs/unet_resnet34_boxmasks"),
    code("for arch, enc in MODELS:\n    print(arch, enc); display(Image(f'runs/{arch}_{enc}/training_curves.png'))"),
    md("## 4. Тестирование и метрики (IoU, Dice, точность, полнота)"),
    code("for arch, enc in MODELS:\n"
         "    !python scripts/evaluate.py --ckpt runs/{arch}_{enc}/best.pt --data prepared --split test --tta\n"
         "runs = ' '.join(f'runs/{a}_{e}' for a, e in MODELS)\n"
         "!python scripts/compare_models.py {runs} --out runs/comparison.csv"),
    code("import pandas as pd\nbest = pd.read_csv('runs/comparison.csv').iloc[0]\nprint('Лучшая модель:', best['модель'])\n"
         "BEST_RUN = 'runs/' + best['модель'].replace('/', '_')\n"
         "display(pd.read_csv(f'{BEST_RUN}/eval_test/metrics_table.csv'))\n"
         "display(Image(f'{BEST_RUN}/eval_test/confusion_matrix.png'))\n"
         "display(Image(f'{BEST_RUN}/eval_test/metrics_by_class.png'))\n"
         "for p in sorted(glob.glob(f'{BEST_RUN}/eval_test/vis/*.jpg'))[:8]:\n    display(Image(p))"),
    md("## 5. Применение к новым кадрам"),
    code("import pandas as pd\ntest_imgs = pd.read_csv('prepared/index.csv').query(\"split=='test'\").image_path.head(10)\n"
         "os.makedirs('demo_in', exist_ok=True)\nimport shutil\n"
         "for p in test_imgs: shutil.copy(p, 'demo_in/')\n"
         "!python scripts/predict.py --ckpt {BEST_RUN}/best.pt --input demo_in --out predictions\n"
         "for p in sorted(glob.glob('predictions/overlay/*.jpg'))[:5]:\n    display(Image(p))"),
    md("## 6. Оценка состояния покрытия (база знаний + машина вывода)\n"
       "Правила применяются к эталонной и к предсказанной маске каждого тестового кадра; "
       "сравниваются итоговые оценки состояния."),
    code("!python scripts/evaluate_rules.py --ckpt {BEST_RUN}/best.pt --data prepared --split test"),
    md("## 7. Архив результатов\nСкачайте из *Output*: `results_small.zip` (метрики, графики, примеры — несколько МБ) "
       "и `best_model.pt` (веса лучшей модели)."),
    code(ZIP_SMALL.format(name="results_small.zip") + "\n"
         "import shutil; shutil.copy(f'{BEST_RUN}/best.pt', '/kaggle/working/best_model.pt')"),
    md("## 8. Веб-интерфейс\nЯчейка выводит публичную ссылку `https://….gradio.live` (действует ~72 ч, пока работает ноутбук). "
       "Откройте её: можно загрузить своё фото или выбрать кадр из тестовой выборки — рядом покажется эталонная разметка."),
    code("!pip install -q gradio\n"
         "from app import Segmenter, build_ui, style_kwargs, _gradio_major\n"
         "seg = Segmenter(f'{BEST_RUN}/best.pt', 'prepared', 'test')\n"
         "seg.examples = seg.examples[:24]\n"
         "demo = build_ui(seg)\n"
         "demo.launch(share=True, allowed_paths=sorted({os.path.dirname(p) for p in seg.examples}),\n"
         "            **(style_kwargs() if _gradio_major() >= 6 else {}))"),
]

cells2 = [
    md("# Оценка состояния покрытия и веб-интерфейс (без повторного обучения)\n"
       "Курсовая работа, Черноиван А. Н., гр. ИИ-26.\n\n"
       "**Перед запуском** (панель справа, *Add Input*):\n"
       "1. Датасет `lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes`.\n"
       "2. Ваши веса: *Add Input → Upload → New Dataset*, перетащите `best.pt` из "
       "`runs/deeplabv3plus_resnet34/` архива результатов, назовите, например, `road-defect-weights`, *Create*.\n\n"
       "*Settings*: **GPU** и **Internet: On**, затем *Run All* (~10–15 мин). "
       "В последней ячейке появится ссылка на интерфейс."),
] + code_cells() + [
    md("## 1. Окружение и веса"),
    code("!pip install -q segmentation-models-pytorch ultralytics gradio\n"
         "!python -m pytest -q tests\n"
         "import glob, os, torch\n"
         "cands = [p for p in glob.glob('/kaggle/input/**/*.pt', recursive=True) if 'sam' not in os.path.basename(p).lower()]\n"
         "assert cands, 'Не найден файл весов .pt — добавьте его через Add Input → Upload'\n"
         "CKPT = cands[0]\n"
         "meta = torch.load(CKPT, map_location='cpu', weights_only=False)['meta']\n"
         "print('Веса:', CKPT, '| модель:', meta['arch'], meta['encoder'])\n"
         "BEST_RUN = f\"runs/{meta['arch']}_{meta['encoder']}\"\n"
         "os.makedirs(BEST_RUN, exist_ok=True)\n"
         "import shutil; shutil.copy(CKPT, f'{BEST_RUN}/best.pt')"),
    md("## 2. Тестовая выборка\nМаски строятся тем же способом (SAM 2) и с тем же разбиением (seed 42), "
       "что и при обучении, поэтому метрики должны совпасть с первым прогоном."),
    code("!python scripts/prepare_dataset.py --data-root /kaggle/input --out prepared --mask-mode sam --vis 0"),
    code("!python scripts/evaluate.py --ckpt {BEST_RUN}/best.pt --data prepared --split test --tta"),
    md("## 3. Оценка состояния покрытия (база знаний + машина вывода)"),
    code("!python scripts/evaluate_rules.py --ckpt {BEST_RUN}/best.pt --data prepared --split test"),
    code("import pandas as pd, os, shutil\nos.makedirs('demo_in', exist_ok=True)\n"
         "for p in pd.read_csv('prepared/index.csv').query(\"split=='test'\").image_path.head(12): shutil.copy(p, 'demo_in/')\n"
         "!python scripts/predict.py --ckpt {BEST_RUN}/best.pt --input demo_in --out predictions\n"
         "from IPython.display import Image, display\nimport glob\n"
         "for p in sorted(glob.glob('predictions/overlay/*.jpg'))[:4]: display(Image(p))"),
    md("## 4. Архив результатов\nСкачайте `results_rules.zip` из *Output* (несколько МБ)."),
    code(ZIP_SMALL.format(name="results_rules.zip")),
    md("## 5. Веб-интерфейс\nОткройте ссылку `https://….gradio.live`. Она работает, пока запущен ноутбук."),
    code("from app import Segmenter, build_ui, style_kwargs, _gradio_major\n"
         "seg = Segmenter(f'{BEST_RUN}/best.pt', 'prepared', 'test')\n"
         "seg.examples = seg.examples[:24]\n"
         "demo = build_ui(seg)\n"
         "demo.launch(share=True, allowed_paths=sorted({os.path.dirname(p) for p in seg.examples}),\n"
         "            **(style_kwargs() if _gradio_major() >= 6 else {}))"),
]

for name, cs in (("kaggle_road_defect_segmentation.ipynb", cells), ("kaggle_rules_and_interface.ipynb", cells2)):
    nb = nbf.v4.new_notebook(cells=cs)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = ROOT / "notebooks" / name
    out.parent.mkdir(exist_ok=True)
    nbf.write(nb, out)
    print("ноутбук:", out)

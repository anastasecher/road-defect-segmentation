import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CLASS_NAMES, NUM_CLASSES
from src.data import find_pairs, make_split
from src.labels import parse_line, read_label_file
from src.masks import compose_semantic_mask, polygon_mask
from src.metrics import SegMetrics
from src.model import DiceCELoss, build_model, compute_class_weights


def test_parse_obb_line():
    a = parse_line("1 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6", 100, 50)
    assert a.cls_id == 1
    assert a.polygon.shape == (4, 2)
    assert a.bbox == pytest.approx((10, 10, 50, 30))
    assert a.area == pytest.approx(40 * 20)


def test_parse_yolo_box_line():
    a = parse_line("0 0.5 0.5 0.2 0.4", 200, 100)
    assert a.bbox == pytest.approx((80, 30, 120, 70))


def test_parse_polygon_line():
    a = parse_line("2 0.1 0.1 0.3 0.1 0.3 0.3 0.2 0.4 0.1 0.3", 100, 100)
    assert a.polygon.shape == (5, 2)


@pytest.mark.parametrize("line", ["", "abc", "1 0.1 0.2", "0 0.5 0.5 0 0"])
def test_parse_bad_lines(line):
    assert parse_line(line, 100, 100) is None


def test_coords_are_clipped():
    a = parse_line("0 -0.2 -0.2 1.5 -0.2 1.5 1.5 -0.2 1.5", 10, 10)
    assert a.polygon.min() >= 0 and a.polygon.max() <= 10


def test_read_label_file_filters_classes(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n7 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\nbroken\n")
    anns, skipped = read_label_file(f, 100, 100, {0, 1, 2})
    assert len(anns) == 1 and skipped == 2


def test_polygon_mask_area():
    a = parse_line("0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5", 100, 100)
    m = polygon_mask(a, 100, 100)
    assert abs(int(m.sum()) - 41 * 41) <= 82


def test_small_objects_drawn_on_top():
    big = np.zeros((20, 20), bool); big[:, :] = True
    small = np.zeros((20, 20), bool); small[5:8, 5:8] = True
    sem = compose_semantic_mask([(3, big), (2, small)], 20, 20)
    assert sem[6, 6] == 2 and sem[0, 0] == 3


def test_make_split_proportions():
    s = make_split(100, 0.15, 0.15, 0)
    assert s.count("test") == 15 and s.count("val") == 15 and s.count("train") == 70
    assert s == make_split(100, 0.15, 0.15, 0)


def test_find_pairs(tmp_path):
    (tmp_path / "images").mkdir(); (tmp_path / "labels").mkdir()
    from PIL import Image
    Image.new("RGB", (8, 8)).save(tmp_path / "images" / "x.jpg")
    Image.new("RGB", (8, 8)).save(tmp_path / "images" / "y.jpg")
    (tmp_path / "labels" / "x.txt").write_text("0 0.5 0.5 0.2 0.2")
    (tmp_path / "labels" / "classes.txt").write_text("pothole")
    pairs = dict((p.name, l) for p, l in find_pairs(tmp_path))
    assert pairs["x.jpg"].name == "x.txt" and pairs["y.jpg"] is None


def test_metrics_perfect():
    t = np.random.randint(0, NUM_CLASSES, (2, 16, 16))
    m = SegMetrics(NUM_CLASSES, CLASS_NAMES)
    m.update(t, t)
    s = m.summary()
    assert s["mIoU"] == pytest.approx(1.0) and s["pixel_accuracy"] == pytest.approx(1.0)


def test_metrics_known_values():
    gt = np.array([[1, 1, 0, 0]])
    pr = np.array([[1, 0, 1, 0]])
    m = SegMetrics(NUM_CLASSES, CLASS_NAMES)
    m.update(pr, gt)
    pc = m.per_class()["pothole"]
    assert pc["iou"] == pytest.approx(1 / 3)
    assert pc["dice"] == pytest.approx(0.5)
    assert pc["precision"] == pytest.approx(0.5)
    assert pc["recall"] == pytest.approx(0.5)
    assert np.isnan(m.per_class()["crack"]["iou"])
    assert m.summary()["mIoU"] == pytest.approx(1 / 3)


def test_binary_merges_defect_classes():
    gt = np.array([[1, 2, 0]])
    pr = np.array([[2, 1, 0]])
    m = SegMetrics(NUM_CLASSES, CLASS_NAMES)
    m.update(pr, gt)
    assert m.binary()["iou"] == pytest.approx(1.0)
    assert m.per_class()["pothole"]["iou"] == pytest.approx(0.0)


def test_class_weights_monotonic():
    w = compute_class_weights([1_000_000, 10_000, 1_000, 100_000])
    assert w[0] == pytest.approx(1.0) and w[2] >= w[1] >= w[3] > 1


def test_loss_decreases_for_correct_prediction():
    y = torch.randint(0, NUM_CLASSES, (2, 8, 8))
    good = torch.nn.functional.one_hot(y, NUM_CLASSES).permute(0, 3, 1, 2).float() * 10
    bad = torch.randn(2, NUM_CLASSES, 8, 8)
    loss = DiceCELoss(NUM_CLASSES)
    assert loss(good, y) < loss(bad, y)


@pytest.mark.parametrize("arch", ["unet", "deeplabv3plus", "fpn"])
def test_model_output_shape(arch):
    m = build_model(arch, "resnet18", NUM_CLASSES, encoder_weights=None).eval()
    with torch.no_grad():
        out = m(torch.randn(1, 3, 64, 128))
    assert out.shape == (1, NUM_CLASSES, 64, 128)

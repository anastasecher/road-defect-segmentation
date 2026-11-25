import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.knowledge import DEFAULT_KB, InferenceEngine, KnowledgeBase, assess, extract_facts

H, W = 100, 200


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase()


def blank():
    return np.zeros((H, W), np.uint8)


def ids(res):
    return {f["id"] for f in res.fired}


def test_kb_loads_and_validates(kb):
    assert len(kb.rules) >= 10 and "crack_major_pct" in kb.params


def test_bad_kb_rejected(tmp_path):
    data = json.loads(DEFAULT_KB.read_text(encoding="utf-8"))
    data["rules"][0]["if"][0][2] = "$нет_такого"
    p = tmp_path / "kb.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        KnowledgeBase(p)


def test_facts_extraction():
    m = blank()
    m[10:20, 10:30] = 1
    m[80:90, 100:120] = 1
    m[40:60, 50:150] = 2
    f = extract_facts(m)
    assert f["pothole_count"] == 2 and f["pothole_area_pct"] == pytest.approx(2.0)
    assert f["pothole_max_pct"] == pytest.approx(1.0)
    assert f["pothole_near_max_pct"] == pytest.approx(1.0)
    assert f["crack_area_pct"] == pytest.approx(10.0)
    assert f["manhole_count"] == 0


def test_no_defects(kb):
    r = assess(blank(), kb=kb)
    assert r.level == 0 and "R0" in ids(r)


def test_manhole_is_not_defect(kb):
    m = blank(); m[10:30, 10:30] = 3
    r = assess(m, kb=kb)
    assert r.level == 0 and {"R0", "R9"} <= ids(r)


def test_medium_cracks(kb):
    m = blank(); m[10:30, 10:30] = 2
    r = assess(m, kb=kb)
    assert r.level == 2 and "R2" in ids(r)


def test_large_crack_zone(kb):
    m = blank(); m[0:50, 0:40] = 2
    r = assess(m, kb=kb)
    assert r.level == 3 and "R3" in ids(r)


def test_small_pothole(kb):
    m = blank(); m[10:20, 10:20] = 1
    r = assess(m, kb=kb)
    assert r.level == 3 and "R4" in ids(r) and not r.facts.get("driver_warning")


def test_large_pothole_is_emergency(kb):
    m = blank(); m[10:40, 10:30] = 1
    r = assess(m, kb=kb)
    assert r.level == 4 and "R5" in ids(r)


def test_many_potholes(kb):
    m = blank()
    for x in (10, 60, 110):
        m[10:15, x:x + 10] = 1
    r = assess(m, kb=kb)
    assert r.level == 4 and "R6" in ids(r)


def test_forward_chaining_progressive_damage(kb):
    m = blank()
    m[10:30, 10:30] = 2
    m[15:20, 15:20] = 1
    r = assess(m, kb=kb)
    assert {"D1", "D2", "D3", "R7"} <= ids(r)
    assert r.facts["progressive_damage"] is True
    order = [f["id"] for f in r.fired]
    assert order.index("D3") > order.index("D1") and order.index("D3") > order.index("D2")


def test_driver_warning(kb):
    m = blank(); m[85:95, 90:110] = 1
    r = assess(m, kb=kb)
    assert r.facts.get("driver_warning") is True and "R8" in ids(r)


def test_low_confidence_flag(kb):
    m = blank(); m[10:30, 10:30] = 2
    conf = np.full((H, W), 0.4, np.float32)
    r = assess(m, conf, kb=kb)
    assert r.facts.get("needs_review") is True


def test_each_rule_fires_once(kb):
    m = blank(); m[0:50, 0:40] = 2; m[5:10, 5:10] = 1
    r = InferenceEngine(kb).run(extract_facts(m))
    order = [f["id"] for f in r.fired]
    assert len(order) == len(set(order))


def test_explanation_mentions_facts(kb):
    m = blank(); m[10:40, 10:30] = 1
    text = assess(m, kb=kb).explanation()
    assert "R5" in text and "pothole_max_pct" in text

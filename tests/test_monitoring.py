import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.monitoring import (FrameResult, Journal, compare_before_after, map_html, parse_coords,
                            pdf_report, read_gps, sample_video)


def fr(level=3, pothole=0.5, crack=2.0, n_pot=1, gps=None, name="k.jpg"):
    return FrameResult(name=name, overlay=np.zeros((36, 64, 3), np.uint8), mask=np.zeros((36, 64), np.uint8),
                       level=level, state=f"уровень {level}", priority="p",
                       areas={"pothole": pothole, "crack": crack, "manhole": 0.0},
                       regions={"pothole": n_pot, "crack": 1 if crack else 0, "manhole": 0},
                       messages=["сообщение"], recommendations=["рекомендация"], rules=["D1", "R4"], gps=gps)


def test_read_gps_from_exif(tmp_path):
    from PIL import Image
    im = Image.new("RGB", (32, 32))
    exif = Image.Exif()
    gps = exif.get_ifd(0x8825)
    gps.update({1: "N", 2: (52.0, 5.0, 51.36), 3: "E", 4: (23.0, 44.0, 2.76)})
    p = tmp_path / "g.jpg"
    im.save(p, exif=exif)
    lat, lon = read_gps(p)
    assert lat == pytest.approx(52.0976, abs=1e-4) and lon == pytest.approx(23.7341, abs=1e-4)


def test_read_gps_absent(tmp_path):
    from PIL import Image
    p = tmp_path / "n.jpg"
    Image.new("RGB", (8, 8)).save(p)
    assert read_gps(p) is None and read_gps(tmp_path / "missing.jpg") is None


@pytest.mark.parametrize("text,expected", [("52.0976, 23.7341", (52.0976, 23.7341)), ("52.1;23.7", (52.1, 23.7)),
                                           ("", None), ("abc", None), ("95, 10", None)])
def test_parse_coords(text, expected):
    assert parse_coords(text) == expected


def test_journal_roundtrip(tmp_path):
    j = Journal(tmp_path / "logs" / "j.csv")
    j.add([fr().record("кадр")])
    j.add([fr(level=4, gps=(52.1, 23.7)).record("пакет"), fr(level=0, pothole=0, crack=0, n_pot=0).record("пакет")])
    df = j.load()
    assert len(df) == 3 and set(df["источник"]) == {"кадр", "пакет"}
    assert df["широта"].notna().sum() == 1
    j.clear()
    assert j.load().empty


def test_before_after_repaired():
    v = compare_before_after(fr(pothole=1.0, crack=3.0), fr(level=0, pothole=0, crack=0, n_pot=0))
    assert v["verdict"] == "ремонт выполнен"


def test_before_after_partial():
    v = compare_before_after(fr(pothole=1.0, crack=3.0, n_pot=2), fr(pothole=0.4, crack=1.0, n_pot=1))
    assert v["verdict"] == "ремонт выполнен частично" and v["ratio"] == pytest.approx(0.35)


def test_before_after_not_done():
    v = compare_before_after(fr(pothole=1.0, crack=3.0), fr(pothole=1.2, crack=3.5))
    assert v["verdict"] == "ремонт не выполнен"


def test_before_after_nothing_before():
    v = compare_before_after(fr(level=0, pothole=0, crack=0, n_pot=0), fr())
    assert v["verdict"] == "сравнение не требуется"


def test_sample_video(tmp_path):
    import cv2
    p = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(50):
        w.write(np.full((48, 64, 3), i * 5, np.uint8))
    w.release()
    frames = list(sample_video(p, step_s=1.0))
    assert len(frames) == 5 and frames[1][0] == pytest.approx(1.0) and frames[0][1].shape == (48, 64, 3)
    assert len(list(sample_video(p, step_s=0.5, max_frames=3))) == 3


def test_sample_video_bad_file(tmp_path):
    p = tmp_path / "bad.mp4"
    p.write_text("not a video")
    with pytest.raises(ValueError):
        list(sample_video(p))


def test_pdf_report(tmp_path):
    out = pdf_report([fr(gps=(52.1, 23.7)), fr(level=0, pothole=0, crack=0, n_pot=0, name="b.jpg")], tmp_path / "r.pdf")
    data = Path(out).read_bytes()
    assert data[:4] == b"%PDF" and data.count(b"/Type /Page") >= 2


def test_map_html():
    assert "Нет кадров" in map_html([])
    h = map_html([{"lat": 52.1, "lon": 23.7, "state": "аварийное", "level": 4, "name": "k", "time": ""}])
    assert "leaflet" in h and "52.1" in h

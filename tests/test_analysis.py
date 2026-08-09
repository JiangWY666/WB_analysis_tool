from pathlib import Path

import cv2
import numpy as np

from wb_analyzer.analysis import analyze_array, analyze_image
from wb_analyzer.export import export_result


def synthetic_blot() -> np.ndarray:
    image = np.full((180, 260), 225, dtype=np.uint8)
    for x in (35, 105, 175):
        cv2.rectangle(image, (x, 45), (x + 42, 57), 45, -1)
        cv2.rectangle(image, (x, 112), (x + 42, 127), 75, -1)
    return image


def test_detects_lanes_and_bands() -> None:
    result = analyze_array(synthetic_blot())
    assert result.dark_bands
    assert len({item.lane for item in result.measurements}) == 3
    assert len(result.measurements) == 6
    assert all(item.integrated_density > 0 for item in result.measurements)


def test_export_and_unicode_path(tmp_path: Path) -> None:
    source = tmp_path / "样品.png"
    assert cv2.imencode(".png", synthetic_blot())[1].tofile(source) is None
    result = analyze_image(source)
    csv_path, preview_path = export_result(result, source, tmp_path / "结果")
    assert csv_path.exists() and preview_path.exists()
    assert csv_path.read_text(encoding="utf-8-sig").count("\n") == 7


def test_rejects_tiny_image() -> None:
    try:
        analyze_array(np.zeros((10, 10), dtype=np.uint8))
    except ValueError as exc:
        assert "尺寸过小" in str(exc)
    else:
        raise AssertionError("tiny images must be rejected")

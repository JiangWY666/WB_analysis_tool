from pathlib import Path

import cv2
import numpy as np
import pytest

from wb_analyzer.analysis import (
    AnalysisConfig,
    analyze_array,
    analyze_image,
    rebuild_result,
)
from wb_analyzer.cli import build_parser, main as cli_main
from wb_analyzer.export import export_result


def synthetic_blot(noise: float = 0.0, uneven: bool = False, invert: bool = False) -> np.ndarray:
    if uneven:
        image = np.linspace(180, 240, 180 * 260, dtype=np.float32).reshape(180, 260)
    else:
        image = np.full((180, 260), 225, dtype=np.float32)
    for x in (35, 105, 175):
        image[45:58, x : x + 43] = 45
        image[112:128, x : x + 43] = 75
    if noise:
        rng = np.random.default_rng(0)
        image = image + rng.normal(0.0, noise, image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)
    if invert:
        return 255 - image
    return image


def test_detects_lanes_and_bands() -> None:
    result = analyze_array(synthetic_blot())
    assert result.dark_bands
    assert len({item.lane for item in result.measurements}) == 3
    assert len(result.measurements) == 6
    assert all(item.integrated_density > 0 for item in result.measurements)


def test_detects_light_bands_on_dark_background() -> None:
    result = analyze_array(synthetic_blot(invert=True), AnalysisConfig(background_mode="auto"))
    assert not result.dark_bands
    assert len(result.measurements) == 6


def test_detects_with_uneven_background_and_noise() -> None:
    result = analyze_array(synthetic_blot(noise=8.0, uneven=True), AnalysisConfig(sensitivity=0.85))
    assert len({item.lane for item in result.measurements}) == 3
    assert len(result.measurements) >= 5


def test_manual_background_override() -> None:
    image = synthetic_blot()
    forced_light = analyze_array(image, AnalysisConfig(background_mode="light_bands"))
    assert not forced_light.dark_bands


def test_manual_rebuild_add_and_delete() -> None:
    result = analyze_array(synthetic_blot())
    assert result.gray_image is not None
    kept = result.measurements[:-1]
    reduced = rebuild_result(result.gray_image, kept, dark_bands=result.dark_bands)
    assert len(reduced.measurements) == len(kept)
    restored = rebuild_result(
        result.gray_image,
        [*reduced.measurements, result.measurements[-1]],
        dark_bands=result.dark_bands,
    )
    assert len(restored.measurements) == 6


def test_export_and_unicode_path(tmp_path: Path) -> None:
    source = tmp_path / "样品.png"
    assert cv2.imencode(".png", synthetic_blot())[1].tofile(source) is None
    result = analyze_image(source)
    csv_path, preview_path = export_result(result, source, tmp_path / "结果")
    assert csv_path.exists() and preview_path.exists()
    text = csv_path.read_text(encoding="utf-8-sig")
    assert text.count("\n") == 7
    assert "saturated_fraction" in text.splitlines()[0]


def test_rejects_tiny_image() -> None:
    with pytest.raises(ValueError, match="尺寸过小"):
        analyze_array(np.zeros((10, 10), dtype=np.uint8))


def test_cli_parser_and_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "demo.png"
    cv2.imencode(".png", synthetic_blot())[1].tofile(source)
    parser = build_parser()
    args = parser.parse_args([str(source), "-o", str(tmp_path / "out"), "--sensitivity", "0.9"])
    assert args.sensitivity == 0.9
    cli_main([str(source), "-o", str(tmp_path / "out"), "--background", "dark_bands"])
    captured = capsys.readouterr().out
    assert "检测到 6 个条带" in captured
    assert (tmp_path / "out" / "demo_bands.csv").exists()

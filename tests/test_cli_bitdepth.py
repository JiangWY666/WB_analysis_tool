"""End-to-end: the CLI must quantify 16-bit TIFFs at their native depth.

``analyze_image`` used to read every file through ``cv2.IMREAD_COLOR``, which
collapses a 16-bit exposure to 256 grey levels before a single number is
measured. Detection still needs 8-bit (CLAHE and Otsu do), so the pipeline
detects on the display layer and re-measures on the raw plane - and these tests
pin down that the reported numbers come from the raw plane.
"""

import csv

import cv2
import numpy as np
import pytest

from wb_analyzer.analysis import analyze_array, analyze_image
from wb_analyzer.cli import main as cli_main
from wb_analyzer.imageio import load_blot
from wb_analyzer.quant import measure_box_native

LANE_WIDTH = 120
HEIGHT = 300
DEPTHS = [24000.0, 12000.0]


def chemi_16bit(background: float = 65132.0, noise: float = 20.0) -> np.ndarray:
    """Dark bands on a near-full-scale 16-bit film background."""
    width = LANE_WIDTH * len(DEPTHS)
    xs = np.arange(width, dtype=float)
    ys = np.arange(HEIGHT, dtype=float)
    raw = np.full((HEIGHT, width), background)
    for index, depth in enumerate(DEPTHS):
        centre = LANE_WIDTH * index + LANE_WIDTH / 2
        raw -= (
            depth
            * np.exp(-0.5 * ((ys - HEIGHT / 2) / 18.0) ** 2)[:, None]
            * np.exp(-0.5 * ((xs - centre) / 16.0) ** 2)[None, :]
        )
    raw += np.random.default_rng(3).normal(0.0, noise, raw.shape)
    return np.clip(raw, 0, 65535).astype(np.uint16)


def write_tiff(path, image: np.ndarray) -> None:
    ok, buffer = cv2.imencode(".tif", image)
    assert ok
    buffer.tofile(str(path))


def test_analyze_image_measures_on_the_raw_plane(tmp_path) -> None:
    path = tmp_path / "blot16.tif"
    write_tiff(path, chemi_16bit())
    result = analyze_image(path)
    assert result.measurements
    assert result.dark_bands
    # An 8-bit read caps a band of this size near 1e5; the native-depth answer
    # is ~256x larger because the values themselves are 256x larger.
    assert max(m.integrated_density for m in result.measurements) > 1e6


def test_reported_numbers_come_from_the_raw_plane(tmp_path) -> None:
    """Every measurement must reproduce exactly when re-measured on ``raw``."""
    path = tmp_path / "blot16.tif"
    write_tiff(path, chemi_16bit())
    blot = load_blot(path)
    assert blot.white_level == 65535.0
    assert blot.bit_depth == 16

    result = analyze_image(path)
    for item in result.measurements:
        native = measure_box_native(
            blot.raw, item.x, item.y, item.width, item.height,
            dark_bands=result.dark_bands, white_level=blot.white_level,
        )
        assert item.integrated_density == pytest.approx(native.integrated_density)
        assert item.mean_intensity == pytest.approx(native.mean_intensity)


def test_stronger_band_still_reads_stronger(tmp_path) -> None:
    path = tmp_path / "blot16.tif"
    write_tiff(path, chemi_16bit())
    result = analyze_image(path)
    by_lane: dict[int, float] = {}
    for item in result.measurements:
        by_lane[item.lane] = by_lane.get(item.lane, 0.0) + item.integrated_density
    assert len(by_lane) == len(DEPTHS)
    strong, weak = (by_lane[lane] for lane in sorted(by_lane))
    assert strong > weak


def test_native_depth_is_256x_the_downscaled_8bit_reading(tmp_path) -> None:
    """Direct evidence the raw plane is used: mean intensity keeps its scale.

    ``mean_intensity`` is background-subtracted and area-normalised, so it is
    comparable across the two paths even though their detected boxes differ
    slightly. A 16-bit reading must come out ~257x an 8-bit one, not equal to it.
    """
    image = chemi_16bit()
    tif = tmp_path / "blot16.tif"
    write_tiff(tif, image)
    eight_bit = (image.astype(np.float64) / 257.0).round().astype(np.uint8)

    def peak_mean(result) -> float:
        return max(item.mean_intensity for item in result.measurements)

    native = peak_mean(analyze_image(tif))
    downscaled = peak_mean(analyze_array(cv2.cvtColor(eight_bit, cv2.COLOR_GRAY2BGR)))
    assert native / downscaled == pytest.approx(257.0, rel=0.25)


def test_analyze_array_rejects_non_uint8_with_a_readable_error() -> None:
    with pytest.raises(ValueError, match="8 位"):
        analyze_array(chemi_16bit())
    with pytest.raises(ValueError, match="8 位"):
        analyze_array(chemi_16bit().astype(np.float32))


def test_cli_writes_native_depth_numbers(tmp_path, capsys) -> None:
    path = tmp_path / "血迹图.tif"  # also covers non-ASCII paths on Windows
    write_tiff(path, chemi_16bit())
    out = tmp_path / "results"
    cli_main([str(path), "--output", str(out)])
    assert "检测到" in capsys.readouterr().out

    csv_path = next(out.glob("*.csv"))
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    densities = [float(row["integrated_density"]) for row in rows]
    assert max(densities) > 1e6
    assert all(np.isfinite(value) for value in densities)

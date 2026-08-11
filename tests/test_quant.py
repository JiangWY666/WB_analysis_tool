"""Tests for lane-profile quantification with baseline subtraction."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from wb_analyzer.analysis import AnalysisResult
from wb_analyzer.imageio import blot_from_array
from wb_analyzer.normalize import AnalysisSession, QuantEntry, export_normalized_csv
from wb_analyzer.quant import (
    compute_lane_profiles,
    measure_box_native,
    quantify_lanes,
    remeasure_native,
    resolve_dark_bands_native,
    roi_column_profile,
    rubber_band_baseline,
    segment_bands,
    valley_baseline,
)
from wb_analyzer.roi import RoiRect


def synthetic_blot16(
    band_depths: list[float],
    *,
    bg: float = 60000.0,
    gradient: float = 0.0,
    band_y: float = 120.0,
    sigma: float = 8.0,
) -> np.ndarray:
    """16-bit light-background blot with one Gaussian band per lane."""
    h, w = 240, 20 + len(band_depths) * 95
    yy = np.arange(h, dtype=np.float64)
    raw = np.full((h, w), bg, dtype=np.float64)
    if gradient:
        raw += gradient * (yy[:, None] / h - 0.5) * 2.0
    lane_w = 80
    for index, depth in enumerate(band_depths):
        x0 = 20 + index * 95
        profile = depth * np.exp(-0.5 * ((yy - band_y) / sigma) ** 2)
        raw[:, x0 : x0 + lane_w] -= profile[:, None]
    return np.clip(raw, 0, 65535).astype(np.uint16)


def test_rubber_band_baseline_recovers_slope() -> None:
    x = np.arange(300, dtype=float)
    baseline_true = 50.0 + 0.3 * x
    peak = 80.0 * np.exp(-0.5 * ((x - 150) / 10.0) ** 2)
    profile = baseline_true + peak
    baseline = rubber_band_baseline(profile)
    corrected = np.maximum(profile - baseline, 0)
    true_area = float(peak.sum())
    assert abs(float(corrected.sum()) - true_area) / true_area < 0.06
    assert abs(baseline[20] - baseline_true[20]) < 4.0
    assert abs(baseline[280] - baseline_true[280]) < 4.0


def test_valley_baseline_connects_minima() -> None:
    x = np.arange(300, dtype=float)
    baseline_true = np.full(300, 40.0)
    peaks = 90.0 * np.exp(-0.5 * ((x - 90) / 9.0) ** 2)
    peaks += 60.0 * np.exp(-0.5 * ((x - 210) / 9.0) ** 2)
    profile = baseline_true + peaks
    baseline = valley_baseline(profile)
    corrected = np.maximum(profile - baseline, 0)
    true_area = float(peaks.sum())
    assert abs(float(corrected.sum()) - true_area) / true_area < 0.08
    assert abs(baseline[150] - 40.0) < 3.0  # valley between the two peaks


def test_segment_bands_splits_two_peaks() -> None:
    x = np.arange(200, dtype=float)
    two = 100.0 * np.exp(-0.5 * ((x - 60) / 9.0) ** 2)
    two += 70.0 * np.exp(-0.5 * ((x - 130) / 9.0) ** 2)
    segments = segment_bands(two)
    assert len(segments) == 2
    (a1, a2), (b1, b2) = segments
    assert a1 < 60 < a2 <= b1 < 130 < b2


def test_compute_lane_profiles_ratio_accuracy_16bit() -> None:
    depths = [24000.0, 12000.0, 6000.0]
    image = synthetic_blot16(depths, gradient=1500.0)
    blot = blot_from_array(image)
    roi = RoiRect(10, 40, blot.width - 20, 170)
    separators = [10 + 95.0 + 10, 10 + 190.0 + 10]
    lanes = compute_lane_profiles(
        blot.raw,
        roi,
        separators,
        dark_bands=True,
        white_level=blot.white_level,
    )
    assert len(lanes) == 3
    totals = [lane.total_density for lane in lanes]
    assert all(v > 0 for v in totals)
    # Integrated ratios must match the injected depth ratios closely.
    assert abs(totals[0] / totals[1] - 2.0) < 0.1
    assert abs(totals[1] / totals[2] - 2.0) < 0.12
    assert [len(lane.segments) for lane in lanes] == [1, 1, 1]
    for lane in lanes:
        assert lane.profile.shape == (roi.height,)
        assert lane.baseline.shape == (roi.height,)
        assert len(lane.band_densities) == len(lane.segments)
        seg_start, seg_stop = lane.segments[0]
        # Band segment hugs the Gaussian at y=120 (absolute).
        assert lane.y0 + seg_start < 120 < lane.y0 + seg_stop


def test_quantify_lanes_returns_analysis_result() -> None:
    depths = [20000.0, 10000.0]
    image = synthetic_blot16(depths)
    blot = blot_from_array(image)
    roi = RoiRect(10, 40, blot.width - 20, 170)
    result = quantify_lanes(
        blot.raw,
        roi,
        [10 + 95.0 + 10],
        dark_bands=True,
        white_level=blot.white_level,
        display_gray=blot.display_gray,
    )
    assert isinstance(result, AnalysisResult)
    assert result.dark_bands
    assert [(m.lane, m.band) for m in result.measurements] == [(1, 1), (2, 1)]
    assert result.measurements[0].integrated_density > result.measurements[1].integrated_density
    assert result.annotated_image.shape[:2] == blot.raw.shape[:2]
    assert result.annotated_image.ndim == 3
    assert all(m.saturated_fraction < 0.05 for m in result.measurements)


def test_quantify_lanes_empty_lane_falls_back_to_whole_lane() -> None:
    # Lane 2 has no band at all; it must still emit one whole-lane record.
    image = synthetic_blot16([20000.0, 0.0])
    blot = blot_from_array(image)
    roi = RoiRect(10, 40, blot.width - 20, 170)
    result = quantify_lanes(
        blot.raw,
        roi,
        [10 + 95.0 + 10],
        dark_bands=True,
        white_level=blot.white_level,
    )
    lanes_present = sorted({m.lane for m in result.measurements})
    assert lanes_present == [1, 2]
    lane2 = [m for m in result.measurements if m.lane == 2]
    assert len(lane2) == 1
    assert lane2[0].height == roi.height  # whole-lane fallback
    assert lane2[0].integrated_density < result.measurements[0].integrated_density * 0.1


def test_quantify_lanes_baseline_modes() -> None:
    image = synthetic_blot16([18000.0])
    blot = blot_from_array(image)
    roi = RoiRect(10, 40, 90, 170)
    for mode in ("rubber", "valley", "none"):
        lanes = compute_lane_profiles(
            blot.raw, roi, [], dark_bands=True, white_level=blot.white_level, baseline_mode=mode
        )
        assert lanes[0].total_density > 0
    with pytest.raises(ValueError, match="基线模式"):
        compute_lane_profiles(
            blot.raw, roi, [], dark_bands=True, white_level=blot.white_level, baseline_mode="bogus"
        )
    # "none" keeps the full background signal, so it integrates much higher.
    rubber = compute_lane_profiles(
        blot.raw, roi, [], dark_bands=True, white_level=blot.white_level, baseline_mode="rubber"
    )
    none = compute_lane_profiles(
        blot.raw, roi, [], dark_bands=True, white_level=blot.white_level, baseline_mode="none"
    )
    assert none[0].total_density > rubber[0].total_density


def test_roi_column_profile_semantics() -> None:
    image = synthetic_blot16([20000.0, 5000.0])
    blot = blot_from_array(image)
    roi = RoiRect(10, 40, blot.width - 20, 170)
    profile = roi_column_profile(
        blot.raw, roi, dark_bands=True, white_level=blot.white_level
    )
    assert profile.shape == (roi.width,)
    # Signal is higher inside the strong first lane than in the gap after it.
    lane1_center = 20 + 40 - roi.x
    gap_center = 20 + 80 + 7 - roi.x
    assert profile[lane1_center] > profile[gap_center]


def test_measure_and_remeasure_native_16bit() -> None:
    image = synthetic_blot16([16000.0, 8000.0])
    blot = blot_from_array(image)
    m1 = measure_box_native(
        blot.raw, 20, 95, 80, 50, dark_bands=True, white_level=blot.white_level
    )
    m2 = measure_box_native(
        blot.raw, 115, 95, 80, 50, dark_bands=True, white_level=blot.white_level
    )
    assert m1.integrated_density > 0 and m2.integrated_density > 0
    assert abs(m1.integrated_density / m2.integrated_density - 2.0) < 0.15
    result = remeasure_native(
        [m1, m2],
        blot.raw,
        dark_bands=True,
        white_level=blot.white_level,
        display_gray=blot.display_gray,
    )
    assert [(m.lane, m.band) for m in result.measurements] == [(1, 1), (2, 1)]
    assert result.annotated_image.ndim == 3


def test_resolve_dark_bands_native() -> None:
    light_bg = np.full((60, 80), 60000, dtype=np.float32)  # bright film
    dark_bg = np.full((60, 80), 4000, dtype=np.float32)  # chemiluminescence display
    assert resolve_dark_bands_native(light_bg, 65535.0) is True
    assert resolve_dark_bands_native(dark_bg, 65535.0) is False
    assert resolve_dark_bands_native(dark_bg, 65535.0, mode="dark_bands") is True
    assert resolve_dark_bands_native(light_bg, 65535.0, mode="light_bands") is False


def test_saturated_band_flagged_16bit() -> None:
    image = synthetic_blot16([70000.0])  # depth exceeds range -> clipped at 0
    blot = blot_from_array(image)
    roi = RoiRect(10, 40, 90, 170)
    result = quantify_lanes(
        blot.raw, roi, [], dark_bands=True, white_level=blot.white_level
    )
    assert result.measurements[0].saturated_fraction > 0.0
    assert result.saturated_band_count == 1


def test_normalize_fold_change(tmp_path: Path) -> None:
    session = AnalysisSession()
    session.upsert(
        QuantEntry(
            kind="loading_control",
            name="Actin",
            group="G",
            intensities=[100.0, 100.0, 100.0],
        )
    )
    session.upsert(
        QuantEntry(kind="target", name="pAKT", group="G", intensities=[50.0, 100.0, 25.0])
    )
    rows = session.normalize_all(control_lane=1)
    assert [row.fold_change for row in rows] == [1.0, 2.0, 0.5]
    rows_lane2 = session.normalize_all(control_lane=2)
    assert rows_lane2[0].fold_change == 0.5
    out = export_normalized_csv(rows, tmp_path / "norm.csv")
    header = out.read_text(encoding="utf-8-sig").splitlines()[0]
    assert "fold_change" in header


def test_demo_image_quantifies_8_lanes() -> None:
    demo = Path(__file__).resolve().parents[1] / "pad4_001(1)_signal.tif"
    if not demo.exists():
        pytest.skip("demo image not in repo checkout")
    from wb_analyzer.imageio import load_blot
    from wb_analyzer.roi import separators_from_profile

    blot = load_blot(demo)
    roi = RoiRect(430, 400, 560, 60)
    profile = roi_column_profile(
        blot.raw, roi, dark_bands=True, white_level=blot.white_level
    )
    separators = separators_from_profile(profile, roi, expected_lanes=8)
    assert len(separators) == 7
    lanes = compute_lane_profiles(
        blot.raw, roi, separators, dark_bands=True, white_level=blot.white_level
    )
    assert len(lanes) == 8
    assert all(lane.total_density > 0 for lane in lanes)
    result = quantify_lanes(
        blot.raw,
        roi,
        separators,
        dark_bands=True,
        white_level=blot.white_level,
        display_gray=blot.display_gray,
    )
    assert len({m.lane for m in result.measurements}) == 8

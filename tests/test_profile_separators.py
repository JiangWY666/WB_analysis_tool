from pathlib import Path

import numpy as np
import pytest

from wb_analyzer.roi import RoiRect, find_profile_peaks, separators_from_profile


def test_separators_follow_profile_valleys() -> None:
    profile = np.zeros(200, dtype=float)
    for center, height in ((30, 20.0), (80, 12.0), (130, 18.0), (170, 10.0)):
        profile[center - 6 : center + 7] += height
    roi = RoiRect(100, 40, 200, 50)
    separators = separators_from_profile(profile, roi, expected_lanes=4)
    assert len(separators) == 3
    rel = [s - roi.x for s in separators]
    assert 30 < rel[0] < 80
    assert 80 < rel[1] < 130
    assert 130 < rel[2] < 170


def test_find_profile_peaks_respects_expected_count() -> None:
    profile = np.array([0, 1, 5, 1, 0, 1, 8, 1, 0, 1, 6, 1, 0, 1, 3, 1, 0], dtype=float)
    peaks = find_profile_peaks(profile, expected_lanes=3, min_distance=3)
    assert len(peaks) == 3


def comb_profile(size: int, centers: list[float], heights: list[float], sigma: float = 5.0) -> np.ndarray:
    """合成列剖面：每个条带一个高斯峰。"""
    x = np.arange(size, dtype=float)
    profile = np.zeros(size, dtype=float)
    for center, height in zip(centers, heights, strict=True):
        profile += height * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    return profile


def shoulder_comb_profile(size: int, lane_w: float, strong: float, weak: float) -> np.ndarray:
    """强弱交替 8 泳道剖面，每个强峰右侧带一个肩部假峰（模拟 16 位原生强带）。"""
    centers = [lane_w / 2 + lane_w * i for i in range(8)]
    heights = [strong if i % 2 == 0 else weak for i in range(8)]
    profile = comb_profile(size, centers, heights)
    x = np.arange(size, dtype=float)
    for i in range(0, 8, 2):
        shoulder_center = centers[i] + 9.0
        profile += strong * 0.55 * np.exp(-0.5 * ((x - shoulder_center) / 3.0) ** 2)
    return profile


def test_expected_lanes_grid_immune_to_shoulder_peaks() -> None:
    # 强带肩部假峰曾让贪心选峰错位、分隔线切进条带；等宽网格+谷位吸附不受影响。
    lane_w = 40.0
    profile = shoulder_comb_profile(320, lane_w, strong=100.0, weak=18.0)
    roi = RoiRect(50, 0, 320, 40)
    separators = separators_from_profile(profile, roi, expected_lanes=8)
    assert len(separators) == 7
    for index, sep in enumerate(separators, start=1):
        rel = sep - roi.x
        ideal = lane_w * index
        assert abs(rel - ideal) <= 0.3 * lane_w  # 每条分隔都贴近等分边界
        # 分隔线严格落在相邻条带中心之间，不切穿任何条带。
        assert lane_w / 2 + lane_w * (index - 1) < rel < lane_w / 2 + lane_w * index


def test_auto_mode_equal_width_prior_on_alternating_lanes() -> None:
    # 强弱交替时自相关主峰落在 2×泳道间距上；峰证据消歧应恢复真实泳道数。
    lane_w = 40.0
    profile = shoulder_comb_profile(320, lane_w, strong=100.0, weak=18.0)
    roi = RoiRect(0, 0, 320, 40)
    separators = separators_from_profile(profile, roi, expected_lanes=None)
    assert len(separators) == 7  # 8 泳道
    for index, sep in enumerate(separators, start=1):
        assert abs(sep - lane_w * index) <= 0.3 * lane_w


def test_grid_separators_snap_to_offset_valleys() -> None:
    # 条带中心偏离等分中心时，分隔线应在窗口内吸附到真实谷位。
    profile = comb_profile(240, [25.0, 95.0, 150.0, 215.0], [50.0, 50.0, 50.0, 50.0], sigma=8.0)
    roi = RoiRect(0, 0, 240, 40)
    separators = separators_from_profile(profile, roi, expected_lanes=4)
    assert len(separators) == 3
    # 真实谷位是相邻高斯中心的中点：60 / 122.5 / 182.5。
    for sep, valley in zip(separators, (60.0, 122.5, 182.5), strict=True):
        assert abs(sep - valley) <= 4.0


def test_auto_mode_sparse_peaks_fall_back_to_valley() -> None:
    # 无周期性的稀疏双峰剖面：自动模式仍应给出一条落在两峰之间的分隔。
    profile = comb_profile(200, [50.0, 150.0], [100.0, 60.0], sigma=6.0)
    roi = RoiRect(10, 0, 200, 40)
    separators = separators_from_profile(profile, roi, expected_lanes=None)
    assert len(separators) == 1
    assert 60.0 < separators[0] - roi.x < 140.0


def test_flat_profile_expected_lanes_gives_exact_grid() -> None:
    profile = np.full(200, 7.0)
    roi = RoiRect(0, 0, 200, 40)
    separators = separators_from_profile(profile, roi, expected_lanes=4)
    assert separators == [50.0, 100.0, 150.0]
    assert separators_from_profile(profile, roi, expected_lanes=None) == []


def test_demo_profile_split_on_real_roi() -> None:
    demo = Path(__file__).resolve().parents[1] / "pad4_001(1)_signal.tif"
    if not demo.exists():
        pytest.skip("demo blot image not present")
    from wb_analyzer.roi import load_bgr, roi_intensity_profile, to_gray

    gray = to_gray(load_bgr(demo))
    roi = RoiRect(430, 400, 560, 60)
    profile, _ = roi_intensity_profile(gray, roi, background_mode="dark_bands")
    separators = separators_from_profile(profile, roi, expected_lanes=8)
    assert len(separators) == 7

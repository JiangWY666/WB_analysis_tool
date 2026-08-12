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


BAND_HALF_WIDTH = 20.0
SMILE_DIP = 0.30
SMILE_SIGMA = 7.0


def smiling_comb(size: int, centers: list[float], heights: list[float]) -> np.ndarray:
    """合成列剖面：平顶宽条带，中间带一个"微笑"浅凹。

    真实化学发光膜上的条带常常是弯的：同一行像素只切到条带的左右两端，列
    剖面里一条条带因此出现两个局部峰、中间一个 10~30% 的浅凹（样例图实测
    6%~27%），两个鼓包相距约半个泳道间距。这是分隔线放置最容易被骗的地
    方——浅凹和真正的泳道间谷（掉 50%~70%）长得很像，间距还刚好一半。
    """
    x = np.arange(size, dtype=float)
    profile = np.zeros(size, dtype=float)
    for center, height in zip(centers, heights, strict=True):
        body = np.exp(-(((x - center) / BAND_HALF_WIDTH) ** 4))
        smile = 1.0 - SMILE_DIP * np.exp(-0.5 * ((x - center) / SMILE_SIGMA) ** 2)
        profile += height * body * smile
    return profile


def lane_spans(roi: RoiRect, separators: list[float]) -> list[tuple[float, float]]:
    edges = [float(roi.x), *sorted(separators), float(roi.x + roi.width)]
    return list(zip(edges, edges[1:], strict=False))


def assert_one_band_per_lane(roi: RoiRect, separators: list[float], centers: list[float]) -> None:
    spans = lane_spans(roi, separators)
    assert len(spans) == len(centers), f"泳道数 {len(spans)} ≠ 条带数 {len(centers)}"
    for index, center in enumerate(centers):
        left, right = spans[index]
        assert left < roi.x + center < right, f"第 {index + 1} 条条带没落在自己的泳道里"


def test_tight_neighbours_are_split_even_when_one_is_faint() -> None:
    # 用户真实样图的核心症状：一条弱带紧挨着一条强带（间距与其余泳道相同），
    # 旧的等宽网格把两者并成一个双倍宽的泳道。分隔线必须落在两者之间。
    centers = [45.0 + 47.0 * i for i in range(8)]
    heights = [30.0, 100.0, 95.0, 70.0, 60.0, 90.0, 100.0, 65.0]
    profile = smiling_comb(400, centers, heights)
    roi = RoiRect(0, 0, 400, 40)
    assert_one_band_per_lane(
        roi, separators_from_profile(profile, roi, expected_lanes=None), centers
    )


def test_blank_gap_gets_one_separator_and_no_extra_lane() -> None:
    # ROI 中段整整空出两个泳道的宽度、两侧留白还不等宽：那里不能凭空多出泳道，
    # 分隔线要落在空白中间，两侧泳道各分一半（收紧足迹后谁都不会把空白算进积分）。
    left = [75.0 + 47.0 * i for i in range(4)]
    right = [75.0 + 47.0 * i for i in range(6, 10)]
    centers = left + right
    profile = smiling_comb(560, centers, [80.0] * len(centers))
    roi = RoiRect(0, 0, 560, 40)
    separators = separators_from_profile(profile, roi, expected_lanes=None)
    assert_one_band_per_lane(roi, separators, centers)
    gap_center = (left[-1] + right[0]) / 2.0
    in_gap = [s for s in separators if abs(s - gap_center) <= 15.0]
    assert len(in_gap) == 1, f"空白正中应恰好一条分隔线，实得 {separators}"


def test_non_uniform_spacing_follows_the_bands() -> None:
    # 泳道间距不必相等（这里是两两成组、组间稍宽）：分隔线应跟着条带走，
    # 而不是把 ROI 均分。
    centers = [40.0, 86.0, 132.0, 196.0, 242.0, 305.0, 351.0, 397.0]
    profile = smiling_comb(450, centers, [70.0] * len(centers))
    roi = RoiRect(200, 0, 450, 40)
    separators = separators_from_profile(profile, roi, expected_lanes=None)
    assert_one_band_per_lane(roi, separators, centers)
    for sep, (left, right) in zip(
        separators, zip(centers, centers[1:], strict=False), strict=True
    ):
        # 落在两条带的中点附近，而不是贴着某一条
        assert abs((sep - roi.x) - (left + right) / 2.0) <= 0.25 * (right - left)


def test_smiling_bands_are_not_cut_in_half() -> None:
    # 微笑形条带自身的中间凹陷间距约等于半个泳道间距，最容易被误判成泳道间谷。
    centers = [40.0 + 48.0 * i for i in range(8)]
    profile = smiling_comb(400, centers, [90.0] * 8)
    roi = RoiRect(0, 0, 400, 40)
    for expected in (None, 8):
        separators = separators_from_profile(profile, roi, expected_lanes=expected)
        assert len(separators) == 7, f"expected_lanes={expected} 切出了 {len(separators)} 条分隔线"
        for sep in separators:
            nearest = min(abs(sep - center) for center in centers)
            assert nearest >= BAND_HALF_WIDTH, f"分隔线 {sep} 切进了条带"


def test_blank_roi_margin_does_not_move_the_separators() -> None:
    # 旧的等宽网格把相位钉死在 ROI 两端，左边多框一点空白整排分隔线就跟着挪
    # （评审实测：左侧加 40px 空白泳道数从 8 变 9，加 120px 变 10）。
    centers = [40.0 + 48.0 * i for i in range(6)]
    base = smiling_comb(320, centers, [80.0] * 6)
    reference = separators_from_profile(base, RoiRect(0, 0, 320, 40), expected_lanes=None)
    assert len(reference) == 5
    for margin in (40, 120):
        padded = np.concatenate((np.zeros(margin), base))
        separators = separators_from_profile(
            padded, RoiRect(0, 0, padded.size, 40), expected_lanes=None
        )
        assert len(separators) == len(reference), f"左侧留白 {margin}px 改变了泳道数"
        for shifted, plain in zip(separators, reference, strict=True):
            assert abs((shifted - margin) - plain) <= 4.0


def test_crowded_lanes_on_a_wide_roi_are_all_split() -> None:
    # 高分辨率扫描 + 满梳 24 泳道：相邻泳道之间只掉 25%，而峰检测的最小间距
    # 启发式按 ROI 宽度缩放（size/20），到这个密度已经分不开相邻两条了。
    pitch = 60.0
    centers = [pitch * (index + 1) for index in range(24)]
    profile = smiling_comb(1500, centers, [80.0] * len(centers))
    roi = RoiRect(0, 0, 1500, 40)
    assert_one_band_per_lane(
        roi, separators_from_profile(profile, roi, expected_lanes=None), centers
    )


def test_band_clipped_by_the_roi_edge_still_gets_a_lane() -> None:
    # 用户常把 ROI 贴着最外侧条带画，边上那条会被切掉一半——它仍然是一条泳道。
    centers = [30.0, 78.0, 126.0, 174.0, 222.0]
    profile = smiling_comb(235, centers, [80.0] * len(centers))  # 末条带被右边界切断
    roi = RoiRect(0, 0, 235, 40)
    assert_one_band_per_lane(
        roi, separators_from_profile(profile, roi, expected_lanes=None), centers
    )


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

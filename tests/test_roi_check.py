"""ROI 质量自检：它该拦下什么、该放行什么。

背景是一次实测调查：一块被剪成窄条的膜（只有 y 336-426 有膜，框外是相机暗
电平），ROI 的上下边一旦越过膜边，橡皮筋基线的弦就掉到暗电平上，整块膜的
本底被当成蛋白积进去——最强泳道总积分虚涨 60%、相对定量最差错 38%，而条带
数看上去完全正常。这些测试用合成膜复现同一批几何，钉住两条判据各自能查出
什么，以及它们如何随条带高度自适应。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from wb_analyzer import check_roi_quality as exported_check
from wb_analyzer.analysis import BandMeasurement
from wb_analyzer.quant import compute_lane_profiles, quantify_lanes
from wb_analyzer.roi import RoiRect
from wb_analyzer.roi_check import (
    ISSUE_BAND_COUNT,
    ISSUE_EDGE_BOTTOM,
    ISSUE_EDGE_TOP,
    ISSUE_EMPTY_LANE,
    ISSUE_FORCED_COUNT,
    check_roi_quality,
    check_roi_quality_from_measurements,
)

LANE_COUNT = 6
LANE_WIDTH = 40
ROI_X = 10
IMAGE_HEIGHT = 200
IMAGE_WIDTH = 2 * ROI_X + LANE_COUNT * LANE_WIDTH
MEMBRANE = (30, 120)
"""合成图里"有膜"的行范围；框外是暗电平，跟真实的窄条膜一样。"""

DARK_LEVEL = 2.0
MEMBRANE_LEVEL = 12.0
AMPLITUDES = (90.0, 70.0, 50.0, 35.0, 25.0, 15.0)
SEPARATORS = [float(ROI_X + i * LANE_WIDTH) for i in range(1, LANE_COUNT)]


def synthetic_membrane(
    *,
    centers: Sequence[float] = (60.0, 82.0),
    sigma: float = 5.0,
    amplitudes: Sequence[float] = AMPLITUDES,
    seed: int = 0,
) -> np.ndarray:
    """一条窄膜，每个泳道 ``len(centers)`` 条高斯带，膜外是暗电平。"""
    rng = np.random.default_rng(seed)
    rows = np.arange(IMAGE_HEIGHT, dtype=float)
    raw = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), DARK_LEVEL, dtype=np.float64)
    raw[MEMBRANE[0] : MEMBRANE[1], :] = MEMBRANE_LEVEL
    for index, amplitude in enumerate(amplitudes):
        x0 = ROI_X + index * LANE_WIDTH
        column = np.zeros(IMAGE_HEIGHT)
        for center in centers:
            column += amplitude * np.exp(-0.5 * ((rows - center) / sigma) ** 2)
        raw[:, x0 + 6 : x0 + LANE_WIDTH - 6] += column[:, None]
    raw += rng.normal(0.0, 0.25, raw.shape)
    return raw.astype(np.float32)


def roi_of(top: int, bottom: int) -> RoiRect:
    return RoiRect(ROI_X, top, LANE_COUNT * LANE_WIDTH, bottom - top)


def profiles(raw: np.ndarray, roi: RoiRect, *, expected_bands: int | None = None):
    return compute_lane_profiles(
        raw,
        roi,
        SEPARATORS,
        dark_bands=False,
        white_level=255.0,
        expected_bands=expected_bands,
    )


def boxes(
    roi: RoiRect, per_lane: Sequence[Sequence[tuple[int, int]]]
) -> list[BandMeasurement]:
    """Band measurements from ``(top, bottom)`` offsets relative to the ROI top."""
    items: list[BandMeasurement] = []
    for lane, bands in enumerate(per_lane, start=1):
        for band, (top, bottom) in enumerate(bands, start=1):
            items.append(
                BandMeasurement(
                    lane=lane,
                    band=band,
                    x=roi.x,
                    y=roi.y + top,
                    width=LANE_WIDTH,
                    height=bottom - top,
                    area=LANE_WIDTH * (bottom - top),
                    mean_intensity=100.0,
                    background=10.0,
                    integrated_density=1000.0,
                )
            )
    return items


# --- 放行 --------------------------------------------------------------------


def test_a_roi_inside_the_membrane_passes() -> None:
    report = check_roi_quality(profiles(synthetic_membrane(), roi_of(35, 115)))
    assert report.ok
    assert report.severity is None
    assert report.issues == ()
    assert (report.lane_count, report.band_count) == (LANE_COUNT, 2 * LANE_COUNT)
    assert (report.bands_per_lane, report.bands_per_lane_source) == (2, "mode")


def test_a_passing_summary_states_the_geometry_it_checked() -> None:
    """状态栏那行字得说清楚它查了什么，否则"通过"是句空话。"""
    report = check_roi_quality(profiles(synthetic_membrane(), roi_of(35, 115)))
    summary = report.summary()
    assert "通过" in summary
    assert f"{LANE_COUNT} 泳道" in summary
    assert f"{report.required_clearance:.0f} px" in summary


def test_diagnosing_never_touches_the_numbers() -> None:
    """只诊断不拒绝，也不改结果——用户完全可能有正当理由那样画。"""
    lanes = profiles(synthetic_membrane(), roi_of(55, 115))
    before = [(tuple(l.segments), l.total_density, tuple(l.band_densities)) for l in lanes]
    assert not check_roi_quality(lanes).ok
    after = [(tuple(l.segments), l.total_density, tuple(l.band_densities)) for l in lanes]
    assert before == after


# --- 判据 2：条带框贴边 ------------------------------------------------------


def test_top_edge_cutting_into_a_band_is_caught() -> None:
    """判据 1 的盲区：顶边切进条带时条带数仍然是对的。"""
    report = check_roi_quality(profiles(synthetic_membrane(), roi_of(55, 115)))
    assert report.band_count == 2 * LANE_COUNT  # 条带数毫无异常
    assert not report.of_kind(ISSUE_BAND_COUNT)  # 判据 1 一声不吭
    (issue,) = report.of_kind(ISSUE_EDGE_TOP)
    assert issue.severity == "error"
    assert issue.lanes == tuple(range(1, LANE_COUNT + 1))
    assert "上边" in issue.detail and "往上挪" in issue.detail
    assert report.top_clearance is not None
    assert report.top_clearance < report.required_clearance


def test_bottom_edge_cutting_into_a_band_is_caught() -> None:
    report = check_roi_quality(profiles(synthetic_membrane(), roi_of(35, 95)))
    (issue,) = report.of_kind(ISSUE_EDGE_BOTTOM)
    assert issue.severity == "error"
    assert "下边" in issue.detail and "往下挪" in issue.detail
    assert not report.of_kind(ISSUE_EDGE_TOP)


def test_the_named_bands_are_the_ones_near_the_edge() -> None:
    """界面要靠这个把具体的条带框高亮出来，所以编号必须对得上。"""
    roi = roi_of(0, 60)
    report = check_roi_quality_from_measurements(
        boxes(roi, [[(2, 18), (30, 46)], [(20, 36), (40, 56)]]), roi
    )
    (issue,) = report.of_kind(ISSUE_EDGE_TOP)
    assert issue.bands == ((1, 1),)  # 只有第 1 道第 1 条贴着上边
    assert issue.lanes == (1,)


def test_required_clearance_scales_with_the_measured_band_height() -> None:
    """"半个条带高度"由实际检出的条带算出来，不是写死的像素数。

    同一个 ROI、同一批泳道，只把条带做胖一倍：要求的留白跟着长，原本够用的
    ROI 就不够用了。
    """
    roi = roi_of(35, 115)
    thin = check_roi_quality(profiles(synthetic_membrane(), roi))
    fat = check_roi_quality(
        profiles(synthetic_membrane(centers=(56.0, 90.0), sigma=10.0), roi)
    )
    assert fat.band_height > thin.band_height * 1.4
    assert fat.required_clearance == pytest.approx(fat.band_height * 0.5)
    assert thin.ok and not fat.ok
    assert fat.of_kind(ISSUE_EDGE_TOP) and fat.of_kind(ISSUE_EDGE_BOTTOM)


def test_a_band_short_of_half_a_height_is_a_warning_not_an_error() -> None:
    """留白不足分两档：压在拖尾上（几个百分点）和整条被切断（两位数）。"""
    roi = roi_of(0, 100)
    bands = [[(top, top + 20), (60, 80)] for top in (7, 8, 9)]
    warning = check_roi_quality_from_measurements(boxes(roi, bands), roi)
    (issue,) = warning.of_kind(ISSUE_EDGE_TOP)
    assert issue.severity == "warning"  # 7 px ≥ 20 × 0.25
    severe = check_roi_quality_from_measurements(
        boxes(roi, [[(4, 24), (60, 80)]] * 3), roi
    )
    (issue,) = severe.of_kind(ISSUE_EDGE_TOP)
    assert issue.severity == "error"  # 4 px < 20 × 0.25


# --- 判据 1：条带数 ----------------------------------------------------------


def test_one_lane_with_an_extra_band_is_named() -> None:
    """没指定每道几条时，"每道条带数"取各泳道的众数，不要求用户填。"""
    raw = synthetic_membrane()
    rows = np.arange(IMAGE_HEIGHT, dtype=float)
    x0 = ROI_X + 2 * LANE_WIDTH
    raw[:, x0 + 6 : x0 + LANE_WIDTH - 6] += (
        40.0 * np.exp(-0.5 * ((rows - 104.0) / 5.0) ** 2)
    )[:, None].astype(np.float32)
    report = check_roi_quality(profiles(raw, roi_of(35, 115)))
    (issue,) = report.of_kind(ISSUE_BAND_COUNT)
    assert report.bands_per_lane == 2
    assert report.bands_per_lane_source == "mode"
    assert issue.severity == "error"
    assert issue.lanes == (3,)
    assert "第 3 道多切了（3 条）" in issue.detail
    assert "少切" not in issue.detail


def test_a_tie_in_the_lane_counts_backs_the_smaller_one() -> None:
    """并列时押"更少"：多切出来的段几乎总是背景，不是条带。"""
    roi = roi_of(0, 100)
    report = check_roi_quality_from_measurements(
        boxes(roi, [[(30, 45), (50, 65)], [(30, 45), (50, 65), (70, 85)]]), roi
    )
    assert report.bands_per_lane == 2
    (issue,) = report.of_kind(ISSUE_BAND_COUNT)
    assert issue.lanes == (2,)


def test_a_lane_short_of_the_mode_gets_its_own_advice() -> None:
    """少切一段和多切一段是两回事，措辞不能混。"""
    roi = roi_of(0, 100)
    report = check_roi_quality_from_measurements(
        boxes(roi, [[(30, 45), (55, 70)]] * 3 + [[(30, 45)]]), roi
    )
    (issue,) = report.of_kind(ISSUE_BAND_COUNT)
    assert issue.lanes == (4,)
    assert "第 4 道少切了（1 条）" in issue.detail
    assert "多切" not in issue.detail


def test_wildly_uneven_lane_counts_admit_the_premise_is_shaky() -> None:
    """众数只覆盖一半泳道时，"每道几条"这个前提本身就该被质疑。"""
    roi = roi_of(0, 100)
    report = check_roi_quality_from_measurements(
        boxes(roi, [[(20, 35), (50, 65), (75, 90)]] * 3 + [[(20, 35), (50, 65)]] * 2), roi
    )
    (issue,) = report.of_kind(ISSUE_BAND_COUNT)
    assert "条带数本身就很散" in issue.detail
    assert "3 条 3 道" in issue.detail and "2 条 2 道" in issue.detail


def test_a_consistent_lane_count_says_nothing_about_the_count() -> None:
    """所有泳道一致时判据 1 无话可说——即使那个数是错的（见模块文档的盲区）。"""
    roi = roi_of(0, 100)
    report = check_roi_quality_from_measurements(
        boxes(roi, [[(30, 45), (50, 65), (70, 85)]] * 4), roi
    )
    assert report.bands_per_lane == 3
    assert not report.of_kind(ISSUE_BAND_COUNT)


# --- expected_bands：判据 1 被架空的时候 -------------------------------------


TRIPLET = {"centers": (60.0, 74.0, 88.0), "sigma": 6.0}
"""三条挨在一起的带：整段是一个连续的 run，强制切成 2 段是切得动的。"""


def test_forcing_two_bands_onto_three_looks_normal_but_is_caught() -> None:
    """最危险的那一档：``expected_bands`` 把判据 1 变成同义反复。

    真实样图上 ROI 画到 y 340-470（明显过高）时自动分段给 30 条带，一眼看出
    不对；填上"每道 2 条"之后给出整整 20 条带、看上去完全正常，而最大泳道
    误差是 34%。这里是同一件事的合成版：6 泳道 × 2 = 12 条，数字挑不出毛病。
    """
    raw = synthetic_membrane(**TRIPLET)
    roi = roi_of(34, 116)
    forced = check_roi_quality(profiles(raw, roi, expected_bands=2), expected_bands=2)
    assert forced.band_count == 2 * LANE_COUNT  # "6 泳道 × 每道 2 条"，完全正常
    assert not forced.of_kind(ISSUE_BAND_COUNT)  # 判据 1 果然一声不吭
    assert forced.forced_split and forced.auto_band_count == 3 * LANE_COUNT
    (issue,) = forced.of_kind(ISSUE_FORCED_COUNT)
    assert issue.severity == "error"
    assert "不能作为 ROI 画对的证据" in issue.detail
    assert f"自动分段会给出 {3 * LANE_COUNT} 条带" in issue.detail
    assert not forced.ok


def test_the_same_roi_passes_when_nobody_forces_the_count() -> None:
    """对照组：不填 expected_bands 时这个 ROI 本来就是干净的。"""
    report = check_roi_quality(profiles(synthetic_membrane(**TRIPLET), roi_of(34, 116)))
    assert report.ok
    assert report.band_count == 3 * LANE_COUNT
    assert report.auto_band_count is None  # 没强制就不用重跑自动分段


def test_a_matching_expected_bands_leaves_only_a_caveat() -> None:
    """填对了也要说一句：这个条带数是切出来的，不构成 ROI 画对的证据。"""
    raw = synthetic_membrane(**TRIPLET)
    report = check_roi_quality(
        profiles(raw, roi_of(34, 116), expected_bands=3), expected_bands=3
    )
    (issue,) = report.of_kind(ISSUE_FORCED_COUNT)
    assert issue.severity == "info"
    assert "不能作为 ROI 画对的证据" in issue.detail
    assert report.ok  # info 不算问题，不打断流程
    assert report.severity is None


def test_the_cross_check_degrades_to_a_caveat_without_profiles() -> None:
    """只有测量表时重跑不了自动分段，就明说查不了，别装作查过。"""
    roi = roi_of(0, 100)
    report = check_roi_quality_from_measurements(
        boxes(roi, [[(30, 45), (50, 65)]] * 3), roi, expected_bands=2
    )
    (issue,) = report.of_kind(ISSUE_FORCED_COUNT)
    assert issue.severity == "info"
    assert report.auto_band_count is None
    assert "无法与自动分段对照" in issue.detail


# --- 空泳道 ------------------------------------------------------------------


def test_unsegmented_lanes_say_so_instead_of_guessing() -> None:
    """``split_bands=False`` 时每个泳道只有一条整体记录，两条判据都无从说起。"""
    raw = synthetic_membrane()
    roi = roi_of(35, 115)
    lanes = compute_lane_profiles(
        raw, roi, SEPARATORS, dark_bands=False, white_level=255.0, split_bands=False
    )
    report = check_roi_quality(lanes)
    (issue,) = report.issues
    assert issue.kind == ISSUE_EMPTY_LANE
    assert issue.lanes == tuple(range(1, LANE_COUNT + 1))
    assert "split_bands" in issue.detail
    assert report.band_height == 0.0
    assert (report.top_clearance, report.bottom_clearance) == (None, None)


def test_an_empty_lane_is_reported_on_its_own_terms() -> None:
    """没检出条带的泳道退回"整条一条记录"，那条假条带铺满 ROI。

    不摘出来的话，它既贴上边又贴下边，判据 2 会把一条空泳道报成"条带被切断"，
    条带高度的中位数也会被这条假条带拉高。
    """
    raw = synthetic_membrane(amplitudes=(90.0, 70.0, 0.0, 35.0, 25.0, 15.0))
    report = check_roi_quality(profiles(raw, roi_of(35, 115)))
    (issue,) = report.of_kind(ISSUE_EMPTY_LANE)
    assert issue.severity == "warning"
    assert issue.lanes == (3,)
    assert not report.of_kind(ISSUE_BAND_COUNT)
    assert not report.of_kind(ISSUE_EDGE_TOP)
    assert not report.of_kind(ISSUE_EDGE_BOTTOM)
    assert report.band_height == pytest.approx(21.5, abs=3.0)  # 没被 80 px 的假条带带偏


# --- 两个入口一致 ------------------------------------------------------------


@pytest.mark.parametrize(
    ("top", "bottom", "expected_bands"),
    [(35, 115, None), (55, 115, None), (35, 95, None), (34, 116, 3)],
)
def test_both_entry_points_agree(top: int, bottom: int, expected_bands: int | None) -> None:
    raw = synthetic_membrane(**(TRIPLET if expected_bands else {}))
    roi = roi_of(top, bottom)
    lanes = profiles(raw, roi, expected_bands=expected_bands)
    from_profiles = check_roi_quality(lanes, expected_bands=expected_bands)
    result = quantify_lanes(
        raw, roi, SEPARATORS, dark_bands=False, white_level=255.0, lanes=lanes
    )
    from_boxes = check_roi_quality_from_measurements(
        result.measurements, roi, expected_bands=expected_bands
    )
    assert from_boxes.ok == from_profiles.ok
    assert from_boxes.band_count == from_profiles.band_count
    assert from_boxes.top_clearance == from_profiles.top_clearance
    assert from_boxes.bottom_clearance == from_profiles.bottom_clearance
    assert {i.kind for i in from_boxes.issues} == {i.kind for i in from_profiles.issues}


# --- 退化输入 ----------------------------------------------------------------


def test_no_lanes_at_all() -> None:
    report = check_roi_quality([])
    assert report.ok
    assert report.lane_count == 0
    assert "没有可评估" in report.summary()


def test_one_lane_cannot_disagree_with_itself() -> None:
    roi = roi_of(0, 100)
    report = check_roi_quality_from_measurements(boxes(roi, [[(30, 45), (55, 70)]]), roi)
    assert report.bands_per_lane == 2
    assert not report.of_kind(ISSUE_BAND_COUNT)
    assert report.ok


def test_a_roi_barely_taller_than_its_bands_is_reported_not_rejected() -> None:
    """ROI 紧贴条带是合法的画法（膜上就那么窄一条），只报警，不抛异常。"""
    report = check_roi_quality(profiles(synthetic_membrane(), roi_of(48, 96)))
    assert not report.ok
    assert report.of_kind(ISSUE_EDGE_TOP) or report.of_kind(ISSUE_EDGE_BOTTOM)


def test_issues_come_worst_first() -> None:
    """界面按顺序显示，最该先看的必须排在最前面。"""
    raw = synthetic_membrane(**TRIPLET)
    report = check_roi_quality(profiles(raw, roi_of(34, 116), expected_bands=2), expected_bands=2)
    ranks = ["error", "warning", "info"]
    positions = [ranks.index(issue.severity) for issue in report.issues]
    assert positions == sorted(positions)
    assert report.issues[0].severity == "error"


def test_the_public_name_is_the_same_function() -> None:
    assert exported_check is check_roi_quality

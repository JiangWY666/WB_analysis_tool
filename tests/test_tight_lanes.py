"""条带足迹收紧（tight）定量测试：泳道内空白不进积分、弱带不被误杀。"""

import itertools

import numpy as np
import pytest

from wb_analyzer.imageio import blot_from_array
from wb_analyzer.quant import (
    MIN_NOISE_SAMPLES,
    _column_span,
    _strip_background,
    compute_lane_profiles,
    quantify_lanes,
)
from wb_analyzer.roi import RoiRect


def gaussian_band_blot16(
    centers_x: list[float],
    depths: list[float],
    *,
    width: int,
    height: int = 200,
    band_y: float = 100.0,
    sigma_x: float = 7.0,
    sigma_y: float = 9.0,
    bg: float = 60000.0,
    noise: float = 30.0,
    seed: int = 7,
    artifact: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """16 位亮底图：x 方向孤立的高斯条带，可叠加噪声与远处伪影斑点。"""
    rng = np.random.default_rng(seed)
    xx = np.arange(width, dtype=np.float64)
    yy = np.arange(height, dtype=np.float64)
    raw = np.full((height, width), bg, dtype=np.float64)
    for center_x, depth in zip(centers_x, depths, strict=True):
        gx = np.exp(-0.5 * ((xx - center_x) / sigma_x) ** 2)
        gy = np.exp(-0.5 * ((yy - band_y) / sigma_y) ** 2)
        raw -= depth * gy[:, None] * gx[None, :]
    if artifact is not None:
        ax, ay, adepth = artifact
        gx = np.exp(-0.5 * ((xx - ax) / 6.0) ** 2)
        gy = np.exp(-0.5 * ((yy - ay) / 6.0) ** 2)
        raw -= adepth * gy[:, None] * gx[None, :]
    if noise:
        raw += rng.normal(0.0, noise, raw.shape)
    return np.clip(raw, 0, 65535).astype(np.uint16)


def total_density(blot, roi: RoiRect, separators: list[float], *, tight: bool = True) -> float:
    result = quantify_lanes(
        blot.raw, roi, separators, dark_bands=True, white_level=blot.white_level, tight=tight
    )
    return sum(m.integrated_density for m in result.measurements)


def test_wide_blank_roi_matches_tight_roi_density() -> None:
    # 用户诉求：同一条带，两侧留大量空白的宽泳道 vs 紧贴条带的窄泳道，
    # 积分应基本一致；空白区的伪影斑点不得再混进条带积分。
    image = gaussian_band_blot16([100.0], [20000.0], width=200, artifact=(40.0, 60.0, 4000.0))
    blot = blot_from_array(image)
    wide = RoiRect(20, 20, 160, 160)  # 条带两侧各留 ~60px 空白，含伪影
    narrow = RoiRect(80, 20, 40, 160)  # 紧贴条带（±20px ≈ ±2.9σ）
    dens_wide = total_density(blot, wide, [])
    dens_narrow = total_density(blot, narrow, [])
    assert abs(dens_wide - dens_narrow) / dens_narrow < 0.06
    # 旧全宽行为会把空白区伪影计入宽 ROI 的积分，显著偏高。
    dens_wide_full = total_density(blot, wide, [], tight=False)
    assert dens_wide_full > dens_wide * 1.05
    # 标注框（x/width）收紧到条带足迹而不是横跨整个泳道。
    lanes = compute_lane_profiles(
        blot.raw, wide, [], dark_bands=True, white_level=blot.white_level
    )
    assert lanes[0].x1 <= 100 <= lanes[0].x2
    assert lanes[0].x2 - lanes[0].x1 < 60


def test_weak_band_not_killed_by_tightening() -> None:
    image = gaussian_band_blot16([100.0], [800.0], width=200)
    blot = blot_from_array(image)
    wide = RoiRect(20, 20, 160, 160)
    lanes = compute_lane_profiles(
        blot.raw, wide, [], dark_bands=True, white_level=blot.white_level
    )
    assert len(lanes) == 1
    lane = lanes[0]
    assert lane.x1 <= 100 <= lane.x2  # 足迹覆盖弱带中心
    assert lane.x2 - lane.x1 < 80  # 确实收紧了（不是整条 160px）
    assert lane.total_density > 0
    # 与紧贴条带的窄 ROI 一致。原本这里放到 15%，理由写的是"弱带噪声地板占比
    # 大"——但两边收紧后的足迹都是 ~35px，噪声偏置在相除时基本抵消，这个容差
    # 从来没被用到过：10 个噪声种子实测最差 3.25%（扣噪声地板后 2.93%），本用
    # 例的种子是 2.91%。收到 5%，让它真的能拦住东西。
    narrow = RoiRect(80, 20, 40, 160)
    dens_narrow = total_density(blot, narrow, [])
    assert abs(lane.total_density - dens_narrow) / dens_narrow < 0.05


def test_empty_lane_falls_back_full_width_and_near_zero() -> None:
    image = gaussian_band_blot16([60.0], [20000.0], width=240)
    blot = blot_from_array(image)
    roi = RoiRect(10, 20, 220, 160)
    lanes = compute_lane_profiles(
        blot.raw, roi, [120.0], dark_bands=True, white_level=blot.white_level
    )
    assert len(lanes) == 2
    # 空白泳道无可靠条带峰：回退全宽（保持旧行为），积分接近 0。
    assert (lanes[1].x1, lanes[1].x2) == (120, 230)
    assert lanes[1].total_density < lanes[0].total_density * 0.02


def test_tight_false_keeps_legacy_full_width() -> None:
    image = gaussian_band_blot16([100.0], [20000.0], width=200)
    blot = blot_from_array(image)
    roi = RoiRect(20, 20, 160, 160)
    lanes = compute_lane_profiles(
        blot.raw, roi, [], dark_bands=True, white_level=blot.white_level, tight=False
    )
    assert (lanes[0].x1, lanes[0].x2) == (20, 180)


def test_tight_reduces_separator_placement_sensitivity() -> None:
    # 协同效应：分隔线偏进邻带尾部时，被"切入"一侧的泳道（lane1）在全宽
    # 模式下会把邻带尾部质量算进自己；收紧后 lane1 的足迹仍锚定在自己的
    # 条带上，邻带尾部不进积分。
    image = gaussian_band_blot16([60.0, 140.0], [20000.0, 10000.0], width=200)
    blot = blot_from_array(image)
    roi = RoiRect(10, 20, 180, 160)

    def lane1_total(separator: float, tight: bool) -> float:
        lanes = compute_lane_profiles(
            blot.raw,
            roi,
            [separator],
            dark_bands=True,
            white_level=blot.white_level,
            tight=tight,
        )
        return lanes[0].total_density

    def drift(tight: bool) -> float:
        base = lane1_total(100.0, tight)
        off = lane1_total(126.0, tight)  # 分隔线切进 lane2 条带的左侧尾部
        return abs(off - base) / base

    drift_tight = drift(True)
    drift_full = drift(False)
    assert drift_tight < 0.005
    assert drift_full > drift_tight


def _soft_window(values: np.ndarray, low: float, high: float, edge: float = 5.0) -> np.ndarray:
    """[low, high) 的软边窗口，边缘用 logistic 过渡（模拟真实条带的边界）。"""
    return 1.0 / (1.0 + np.exp(-(values - low) / edge)) - 1.0 / (
        1.0 + np.exp(-(values - high) / edge)
    )


SOFT_EDGE = 5.0
"""生成图上条带边界的 logistic 尺度：过渡在 ±3×SOFT_EDGE 内基本完成。"""

SMEAR_X = (30, 250)
BAND_X = (252, 380)
SIGNAL_Y = (40, 200)


def plateau(span: tuple[int, int]) -> tuple[int, int]:
    """软边之内、信号确定满幅的那一段列范围。

    足迹按峰高的 ``TIGHT_EDGE_FRAC`` 截断，本来就允许啃掉条带的软边；断言
    只钉"平台上的列一列都不许丢"，才不至于把正常的边缘修剪也判成漏带。
    """
    margin = int(round(3.0 * SOFT_EDGE))
    return span[0] + margin, span[1] - margin


def smeared_lane_blot16(
    *,
    width: int = 460,
    height: int = 240,
    smear_level: float = 2200.0,
    band_level: float = 4700.0,
    core_depth: float = 34000.0,
    core_y: float = 120.0,
    core_sigma_y: float = 7.0,
    bg: float = 60000.0,
    noise: float = 30.0,
    seed: int = 11,
) -> np.ndarray:
    """16 位亮底图：左边一大片弱拖尾、右边一条强带，中间不回落到背景。

    复刻一张真实膜（用户私有数据，未入库）上把 tight 打崩的那条泳道的几何
    比例，三个比例都是从那张图上量出来的：

    * 弱拖尾的列均值只有强带峰的 0.26 倍（真图 10.6 / 40.2）；
    * 两者之间**没有**降回背景的空隙，是一路抬上去的连续信号；
    * 有信号的行占 ROI 高度 73%（真图 80%），所以纵向还剩得下背景行。

    关键在于分隔线切在强带里，使得泳道条几乎被信号占满、只剩百分之几的
    空白列——这正是"列剖面 5 分位数当背景"这个假设失效的条件。
    """
    rng = np.random.default_rng(seed)
    xx = np.arange(width, dtype=np.float64)
    yy = np.arange(height, dtype=np.float64)
    rows = _soft_window(yy, *SIGNAL_Y)
    columns = smear_level * _soft_window(xx, *SMEAR_X) + band_level * _soft_window(xx, *BAND_X)
    core = core_depth * _soft_window(xx, BAND_X[0] + 4, BAND_X[1] - 4)
    signal = rows[:, None] * columns[None, :]
    signal += np.exp(-0.5 * ((yy - core_y) / core_sigma_y) ** 2)[:, None] * core[None, :]
    raw = bg - signal + rng.normal(0.0, noise, (height, width))
    return np.clip(raw, 0, 65535).astype(np.uint16)


def lane_pair(blot, roi: RoiRect, separators: list[float]):
    """同一 ROI 的 (tight, full-width) 两套剖面，按泳道配好对。"""
    common = dict(dark_bands=True, white_level=blot.white_level)
    tight = compute_lane_profiles(blot.raw, roi, separators, tight=True, **common)
    full = compute_lane_profiles(blot.raw, roi, separators, tight=False, **common)
    return list(zip(tight, full, strict=True))


def assert_footprint_keeps_the_signal(
    tight, full, *, signal_x: tuple[int, int], min_keep: float, tag: str = ""
) -> None:
    """通用健全性断言：收紧只能去掉空白，不能去掉信号。

    只对"泳道内的信号是**一整片连着的**"这种几何成立——tight 正是为这种
    几何设计的：把条带两侧的空白挡在积分外面。它不适用于泳道里另有一块
    孤立污点（那时丢掉污点才是对的，见
    :func:`test_wide_blank_roi_matches_tight_roi_density`），也不适用于一条
    泳道里塞了两条中间断开的条带（那是分泳道没切对，tight 无从分辨）。

    两条口径同时钉：

    * **几何**——足迹必须盖住这条泳道里信号满幅的那些列（``signal_x`` 给的
      是 :func:`plateau` 口径，软边允许被修掉）；
    * **积分**——tight 的积分不得低于全宽积分的 ``min_keep``。它拦得住足迹
      "盖住了平台但两侧啃掉一大截"的情形。
    """
    lo, hi = max(signal_x[0], full.x1), min(signal_x[1], full.x2)
    assert tight.x1 <= lo + 2, f"{tag}: 足迹左边界 {tight.x1} 把 {lo} 起的信号列切掉了"
    assert tight.x2 >= hi - 2, f"{tag}: 足迹右边界 {tight.x2} 把 {hi} 止的信号列切掉了"
    keep = tight.total_density / full.total_density
    assert keep >= min_keep, f"{tag}: 收紧后只剩全宽积分的 {keep:.1%}（下限 {min_keep:.0%}）"


def test_fully_covered_lane_keeps_its_weak_half() -> None:
    """泳道几乎被信号占满时，弱的那半边不许被足迹丢掉。

    真实膜上的表现：250 px 宽的泳道只收到右边 86 px，积分从 692896 掉到
    358283（少 48%），两条泳道的比值从 1.0178 变成 0.5272——而界面上看不出
    任何异常。根因是列剖面的 5 分位数被当成了背景：这条泳道只有 7 列
    （2.8%）是空白，5 分位数落在弱带自己的平台上，``edge_frac`` 的相对切点
    于是从一个高于弱带的电平起算，整片弱信号被判在阈值以下。
    """
    blot = blot_from_array(smeared_lane_blot16())
    roi = RoiRect(24, 10, 400, 220)
    (tight, full), (tight2, full2) = lane_pair(blot, roi, [290.0])

    lane_width = full.x2 - full.x1
    blank_columns = SMEAR_X[0] - full.x1
    assert blank_columns / lane_width < 0.05, "本例的前提就是空白列不足 5%"

    assert_footprint_keeps_the_signal(
        tight, full, signal_x=plateau(SMEAR_X), min_keep=0.97, tag="弱拖尾泳道"
    )
    assert tight.x2 - tight.x1 > lane_width * 0.9
    # 相邻泳道（强带那侧）本来就没坏，顺带钉住它没被改坏。
    assert_footprint_keeps_the_signal(
        tight2, full2, signal_x=plateau(BAND_X), min_keep=0.97, tag="强带泳道"
    )
    # 泳道比值必须与全宽口径一致：这才是用户真正会看的数字。
    ratio_tight = tight.total_density / tight2.total_density
    ratio_full = full.total_density / full2.total_density
    assert abs(ratio_tight - ratio_full) / ratio_full < 0.02


def test_footprint_does_not_flip_on_a_few_pixels_of_roi_margin() -> None:
    """ROI 左边缘挪几个像素，足迹不许在"对"和"漏半条泳道"之间跳变。

    这是同一个缺陷最难发现的表现形式：真图上 ROI.x 从 820 挪到 823（三个
    像素，全落在空白上）就让 lane1 的积分从 692955 掉到 358283。原因是空白
    列的占比正好跨过 5%，5 分位数从空白跳到了弱带平台上。所以这里扫的不是
    某一个 ROI，而是这个跳变本身。
    """
    blot = blot_from_array(smeared_lane_blot16())
    covered: list[float] = []
    keeps: list[float] = []
    for roi_x in range(14, 40, 2):
        roi = RoiRect(roi_x, 10, 400, 220)
        tight, full = lane_pair(blot, roi, [290.0])[0]
        assert_footprint_keeps_the_signal(
            tight, full, signal_x=plateau(SMEAR_X), min_keep=0.97, tag=f"ROI.x={roi_x}"
        )
        covered.append((tight.x2 - tight.x1) / (full.x2 - full.x1))
        keeps.append(tight.total_density / full.total_density)
    # 这条泳道从头到尾都被信号占着，足迹就该占满整条；出问题时只剩 22%。
    assert min(covered) > 0.9, f"足迹只占泳道的 {min(covered):.1%}"
    assert max(keeps) - min(keeps) < 0.02, f"保留比例在 {min(keeps):.1%}~{max(keeps):.1%} 之间跳"


@pytest.mark.parametrize("smear_level", [900.0, 2200.0, 4200.0])
def test_tight_keeps_the_signal_across_geometries(smear_level: float) -> None:
    """健全性不变量：信号连成一片的泳道，收紧后不许丢信号。

    扫一遍会改变"空白列占比"和"强弱对比度"的几何：分隔线位置（决定泳道
    条被信号占满到什么程度）、ROI 左边缘、ROI 高度、噪声种子，共 54 组。

    下限取 0.95 而不是贴着实测值：这 54×2 条泳道实测最差保留 98.0%，而缺陷
    发作时是 41%~54%——0.95 落在这中间四十多个百分点的空档里，两头都有大把
    余量，挪到 0.6 或 0.97 结论都不变。
    """
    for separator, roi_x, roi_h, seed in itertools.product(
        (270.0, 290.0, 310.0), (14, 24, 36), (200, 220), (11, 12, 13)
    ):
        blot = blot_from_array(smeared_lane_blot16(smear_level=smear_level, seed=seed))
        roi = RoiRect(roi_x, 10, 400, roi_h)
        tag = f"smear={smear_level} sep={separator} x={roi_x} h={roi_h} seed={seed}"
        pairs = lane_pair(blot, roi, [separator])
        assert_footprint_keeps_the_signal(
            pairs[0][0], pairs[0][1], signal_x=plateau(SMEAR_X), min_keep=0.95, tag=f"{tag} L1"
        )
        assert_footprint_keeps_the_signal(
            pairs[1][0], pairs[1][1], signal_x=plateau(BAND_X), min_keep=0.95, tag=f"{tag} L2"
        )


def test_strip_background_reads_the_blank_rows_not_the_darkest_column() -> None:
    """背景电平量在条带段之外的行上，与条带横向占了多宽无关。"""
    height, width = 120, 200
    signal = np.full((height, width), 7.0)  # 背景
    signal[30:90, :] += 120.0  # 每一列都被条带占满，一列空白都没有
    background = _strip_background(signal, [(30, 90)])
    assert background == pytest.approx(7.0)
    # 5 分位数在这种泳道上会落到条带自己身上，正是缺陷的来源。
    assert float(np.percentile(signal.mean(axis=0), 5)) > 60.0


def test_strip_background_ignores_a_blot_sitting_in_the_blank_rows() -> None:
    """条带段之外的一块污点不该抬高背景估计：跨列取中位数，少数列拉不动。"""
    height, width = 120, 200
    signal = np.full((height, width), 7.0)
    signal[30:90, :] += 120.0
    signal[5:25, 10:40] += 400.0  # 空白行里的污点，占 15% 的列
    assert _strip_background(signal, [(30, 90)]) == pytest.approx(7.0)


def test_strip_background_declines_when_there_is_no_blank_row() -> None:
    """没有条带段、或者空白行少到不够量，就不给估计值，退回原来的口径。"""
    signal = np.full((40, 60), 5.0)
    assert _strip_background(signal, None) is None
    assert _strip_background(signal, []) is None
    short = MIN_NOISE_SAMPLES - 1
    assert _strip_background(signal, [(0, 40 - short)]) is None
    assert _strip_background(signal, [(0, 40 - MIN_NOISE_SAMPLES)]) == pytest.approx(5.0)


def test_a_high_background_estimate_cannot_shrink_the_footprint() -> None:
    """背景估计只允许把 floor 往低了修，估高了必须被 5 分位数挡回去。

    ROI 下沿压到膜边的辉光上时，条带段之外的行量出来会偏高；那时收窄足迹
    正是要防的事，所以两个估计取更低者。
    """
    xx = np.arange(300, dtype=np.float64)
    column = 100.0 + 900.0 * np.exp(-0.5 * ((xx - 150.0) / 20.0) ** 2)
    plain = _column_span(column)
    assert plain is not None
    assert _column_span(column, background=1e6) == plain
    lowered = _column_span(column, background=0.0)
    assert lowered is not None
    assert lowered[0] <= plain[0] and lowered[1] >= plain[1]

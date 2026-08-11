"""条带足迹收紧（tight）定量测试：泳道内空白不进积分、弱带不被误杀。"""

import numpy as np

from wb_analyzer.imageio import blot_from_array
from wb_analyzer.quant import compute_lane_profiles, quantify_lanes
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
    # 与紧贴条带的窄 ROI 一致（弱带信噪比低，噪声地板占比大，容差放宽）。
    narrow = RoiRect(80, 20, 40, 160)
    dens_narrow = total_density(blot, narrow, [])
    assert abs(lane.total_density - dens_narrow) / dens_narrow < 0.15


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

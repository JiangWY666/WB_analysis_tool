"""噪声地板扣除与检出限：积分不再随框的大小长个儿。

`max(profile - baseline, 0)` 在空白泳道上不会平均到 0——橡皮筋是下包络，本来
就压在背景均值之下，再把负向涨落全截掉，剩下的正向残渣按样本数累加。于是同一
条带，用户把 ROI 拉高一点数值就变大：修复前弱带在 ROI 高 60/200/500 三档上漂移
了 62.6%。这组用例盯的就是这个漂移，外加「强带别被误伤」和「弱到测不出来要说
出来」两条边界。

注意这里量的是**橡皮筋基线之上的截断偏置**。基线本身也会随剖面变长而下沉，那
是另一回事（凸包跟不上弯曲的背景），合成图上不明显、真实扫描图上是主项，见
`compute_lane_profiles` 的 Known limitation。
"""

import numpy as np

from wb_analyzer.imageio import blot_from_array
from wb_analyzer.quant import (
    LOD_SIGMA,
    compute_lane_profiles,
    detection_limit,
    measure_box_native,
    quantify_lanes,
    segment_bands,
)
from wb_analyzer.roi import RoiRect

LANE_PITCH = 95
BAND_Y = 300.0


def noisy_lanes(
    depths: list[float],
    *,
    height: int = 600,
    noise: float = 60.0,
    sigma_x: float = 10.0,
    sigma_y: float = 9.0,
    band_y: float = BAND_Y,
    bg: float = 60000.0,
    seed: int = 11,
) -> np.ndarray:
    """16 位浅底图：每个泳道一条高斯带，叠加高斯噪声。"""
    rng = np.random.default_rng(seed)
    width = 20 + len(depths) * LANE_PITCH
    xx = np.arange(width, dtype=float)
    yy = np.arange(height, dtype=float)
    raw = np.full((height, width), bg)
    for index, depth in enumerate(depths):
        center = 20 + index * LANE_PITCH + LANE_PITCH / 2
        raw -= (
            depth
            * np.exp(-0.5 * ((yy - band_y) / sigma_y) ** 2)[:, None]
            * np.exp(-0.5 * ((xx - center) / sigma_x) ** 2)[None, :]
        )
    raw += rng.normal(0.0, noise, raw.shape)
    return np.clip(raw, 0, 65535).astype(np.uint16)


def separators(count: int) -> list[float]:
    return [float(20 + LANE_PITCH * i) for i in range(1, count)]


def lane_totals(blot, roi: RoiRect, seps: list[float]) -> dict[int, float]:
    result = quantify_lanes(
        blot.raw, roi, seps, dark_bands=True, white_level=blot.white_level
    )
    totals: dict[int, float] = {}
    for m in result.measurements:
        totals[int(m.lane)] = totals.get(int(m.lane), 0.0) + float(m.integrated_density)
    return totals


def roi_of_height(blot, height: int) -> RoiRect:
    """以条带为中心、给定高度的 ROI（三档高度都完整包住条带）。"""
    return RoiRect(10, int(BAND_Y) - height // 2, blot.width - 20, height)


def spread(values: list[float]) -> float:
    return (max(values) - min(values)) / min(values)


def mean_weak_ratio(height: int, seeds: range) -> float:
    """多个噪声实现下弱/强比值的平均。

    单次测量里，弱带本身的散粒噪声就有百分之一二的抖动（信号只比噪声高一个
    量级），系统性漂移会被它盖住。要问的是"平均值随 ROI 高度变不变"，所以先
    对噪声实现取平均，再看三档之间的差。
    """
    ratios = []
    for seed in seeds:
        blot = blot_from_array(noisy_lanes([40000.0, 300.0], seed=seed))
        totals = lane_totals(blot, roi_of_height(blot, height), separators(2))
        ratios.append(totals[2] / totals[1])
    return sum(ratios) / len(ratios)


def test_weak_band_ratio_survives_roi_height() -> None:
    # 验收指标 1：同一条弱带，ROI 高度 60 / 200 / 500 三档，弱/强比值波动 < 5%。
    # 修复前这三档是 0.00817 / 0.01022 / 0.01329（同样口径的多种子平均），
    # 漂了 62.6%——纯粹因为 ROI 越高噪声残渣积得越多，弱带本身的信号量没变。
    # 修复后是 0.00758 / 0.00729 / 0.00746，真值 300/40000 = 0.0075。
    seeds = range(11, 19)
    ratios = [mean_weak_ratio(height, seeds) for height in (60, 200, 500)]
    assert spread(ratios) < 0.05, f"ROI 高度依赖未消除：{ratios}"
    # 顺带守住量级：过度扣除会把弱带压没，真值比是 300/40000 = 0.0075。
    assert 0.006 < min(ratios) and max(ratios) < 0.009


def old_pipeline_total(blot, roi: RoiRect, seps: list[float], lane_number: int) -> float:
    """修复前的口径：分段不看噪声，积分在截断过的 corrected 上做，不扣地板。"""
    for lane in compute_lane_profiles(
        blot.raw, roi, seps, dark_bands=True, white_level=blot.white_level
    ):
        if int(lane.lane) != lane_number:
            continue
        corrected = np.maximum(lane.profile - lane.baseline, 0.0)
        segments = segment_bands(corrected) or [(0, int(corrected.size))]
        return float(sum(corrected[s:e].sum() for s, e in segments))
    raise AssertionError(f"没有泳道 {lane_number}")


def test_strong_band_is_left_alone() -> None:
    # 验收指标 2：强条带几乎不受影响，变化 < 3.5%。扣掉的是"样本数 × 每样本
    # 残渣"，对强带积分来说是个零头。这里量的是修复前后端到端的差，因此也覆盖
    # 了"积分改用未截断残差"和"分段改成噪声感知"这两处口径变化。
    blot = blot_from_array(noisy_lanes([40000.0, 300.0]))
    for height in (60, 200, 500):
        roi = roi_of_height(blot, height)
        before = old_pipeline_total(blot, roi, separators(2), 1)
        after = lane_totals(blot, roi, separators(2))[1]
        change = abs(after - before) / before
        assert change < 0.035, f"ROI 高 {height}：强带变了 {change:.2%}（{before} → {after}）"


def test_two_fold_dilution_stays_accurate() -> None:
    # 验收指标 4：2 倍梯度稀释的精度不能退化。
    blot = blot_from_array(
        noisy_lanes([20000.0, 10000.0, 5000.0, 2500.0], height=300, band_y=150.0, seed=5)
    )
    totals = lane_totals(blot, RoiRect(10, 60, blot.width - 20, 180), separators(4))
    measured = [totals[n] / totals[1] for n in (1, 2, 3, 4)]
    for value, expected in zip(measured, [1.0, 0.5, 0.25, 0.125], strict=True):
        assert abs(value - expected) / expected < 0.02, measured


def test_pure_noise_lane_integrates_to_nothing() -> None:
    # 空白泳道原本能积出六位数（n × 0.9σ，恒为正）。扣完偏置后剩下的是随机
    # 涨落之和，量级只有 σ√n——也就是它自己的检出限之内，正是「测不到东西」
    # 应有的样子。
    rng = np.random.default_rng(3)
    flat = np.full((400, 120), 60000.0) + rng.normal(0.0, 40.0, (400, 120))
    blot = blot_from_array(np.clip(flat, 0, 65535).astype(np.uint16))
    lane = compute_lane_profiles(
        blot.raw, RoiRect(10, 10, 100, 380), [], dark_bands=True, white_level=blot.white_level
    )[0]
    n = int(lane.profile.size)
    assert lane.noise_sigma > 0.0
    assert lane.total_density < detection_limit(lane.noise_sigma, n)
    # 修复前这里是 0.9σn 量级；现在连 0.1σn 都不到。
    assert lane.total_density < 0.1 * lane.noise_sigma * n


def test_noiseless_image_is_untouched() -> None:
    # 噪声为零时不该凭空扣掉什么，也不该给出检出限判定（0 是「未评估」）。
    yy = np.arange(200, dtype=float)
    xx = np.arange(120, dtype=float)
    raw = np.full((200, 120), 60000.0)
    raw -= (
        20000.0
        * np.exp(-0.5 * ((yy - 100.0) / 9.0) ** 2)[:, None]
        * np.exp(-0.5 * ((xx - 60.0) / 10.0) ** 2)[None, :]
    )
    blot = blot_from_array(raw.astype(np.uint16))
    lane = compute_lane_profiles(
        blot.raw, RoiRect(10, 10, 100, 180), [], dark_bands=True, white_level=blot.white_level
    )[0]
    assert lane.noise_floor == 0.0
    assert lane.noise_sigma == 0.0
    assert all(limit == 0.0 for limit in lane.band_detection_limits)
    assert lane.total_density > 0.0


def test_weak_lane_is_flagged_below_detection() -> None:
    # 检出限报告：噪声里的弱带要被点名，真条带不能被误杀。
    blot = blot_from_array(noisy_lanes([40000.0, 60.0], noise=120.0))
    result = quantify_lanes(
        blot.raw,
        RoiRect(10, 200, blot.width - 20, 200),
        separators(2),
        dark_bands=True,
        white_level=blot.white_level,
    )
    by_lane = {int(m.lane): m for m in result.measurements}
    assert not by_lane[1].below_detection, "强带不该被判成低于检出限"
    assert by_lane[1].snr > LOD_SIGMA
    assert by_lane[2].below_detection, "噪声量级的弱带必须报低于检出限"
    assert by_lane[2].snr < LOD_SIGMA
    assert by_lane[2].detection_limit > 0.0


def test_detection_limit_grows_with_the_square_root_of_the_window() -> None:
    # 逐样本噪声在积分窗口上按平方根累加，所以窗口越宽，能被认出来的最小
    # 信号越大——4 倍宽的窗口，门槛翻倍。
    assert detection_limit(10.0, 100) == LOD_SIGMA * 10.0 * 10.0
    assert detection_limit(10.0, 400) == 2 * detection_limit(10.0, 100)
    assert detection_limit(0.0, 400) == 0.0
    assert detection_limit(10.0, 0) == 0.0


def test_hand_box_integral_does_not_grow_with_the_box() -> None:
    # 环形中值那条路径（自动检测 / 手绘框）有同样的截断偏置：背景取的是中值，
    # 截掉负半边后每像素留下 0.399σ，于是框拉大数值就变大。
    rng = np.random.default_rng(17)
    yy = np.arange(240, dtype=float)
    xx = np.arange(160, dtype=float)
    raw = np.full((240, 160), 60000.0)
    raw -= (
        8000.0
        * np.exp(-0.5 * ((yy - 120.0) / 4.0) ** 2)[:, None]
        * np.exp(-0.5 * ((xx - 80.0) / 6.0) ** 2)[None, :]
    )
    raw += rng.normal(0.0, 50.0, raw.shape)
    image = np.clip(raw, 0, 65535).astype(np.uint16)
    values = [
        measure_box_native(
            image, 60, 120 - h // 2, 40, h, dark_bands=True, white_level=65535.0
        ).integrated_density
        for h in (30, 50, 80)
    ]
    assert spread(values) < 0.05, f"手绘框积分随框高变化：{values}"


def test_blank_box_is_reported_below_detection() -> None:
    rng = np.random.default_rng(23)
    flat = np.full((200, 200), 60000.0) + rng.normal(0.0, 50.0, (200, 200))
    image = np.clip(flat, 0, 65535).astype(np.uint16)
    m = measure_box_native(image, 80, 80, 40, 40, dark_bands=True, white_level=65535.0)
    assert m.below_detection
    assert m.integrated_density < m.detection_limit

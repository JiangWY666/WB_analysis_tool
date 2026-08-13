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


def _local_maxima(profile: np.ndarray, lo: int, hi: int) -> list[int]:
    """``[lo, hi]`` 内的局部极大值位置；等值平台取中点。"""
    found: list[int] = []
    index = lo
    while index <= hi:
        stop = index
        while stop + 1 <= hi and profile[stop + 1] == profile[index]:
            stop += 1
        rising = index == 0 or profile[index] > profile[index - 1]
        falling = stop == profile.size - 1 or profile[stop] > profile[stop + 1]
        if rising and falling:
            found.append((index + stop) // 2)
        index = stop + 1
    return found


def band_bodies(profile: np.ndarray, centers: list[float]) -> list[tuple[int, int]]:
    """每条条带的"信号显著区"：该条带自己的半高区间（闭区间，列索引）。

    标尺是**这条条带自己的高度**，不是整条剖面的动态范围——弱带因此和强带
    一样量得出一个像样的信号区，这正是全局电平量不出来的东西。做法：在半个
    泳道间距的窗口里找出属于这条条带的所有鼓包（弯条带的微笑双峰、强带上的
    纹路都算），从最外侧的鼓包一路下坡走到坡底当作这条条带的地盘，再取地盘
    内高过"峰顶与坡底中点"的最外两列。下坡走到坡底这一步是关键：弱带右侧若
    紧跟着一片比它还高的膜辉光，走到谷底就停，辉光不会被算进弱带的身子。
    """
    profile = np.asarray(profile, dtype=float)
    size = int(profile.size)
    bodies: list[tuple[int, int]] = []
    for index, center in enumerate(centers):
        left = center - centers[index - 1] if index else 2.0 * center
        right = centers[index + 1] - center if index + 1 < len(centers) else 2.0 * (size - center)
        lo = max(0, int(np.ceil(center - left / 2.0)))
        hi = min(size - 1, int(np.floor(center + right / 2.0)))
        apexes = _local_maxima(profile, lo, hi) or [lo + int(np.argmax(profile[lo : hi + 1]))]
        start, stop = apexes[0], apexes[-1]
        while start > 0 and profile[start - 1] <= profile[start]:
            start -= 1
        while stop < size - 1 and profile[stop + 1] <= profile[stop]:
            stop += 1
        peak = float(profile[lo : hi + 1].max())
        level = (peak + max(float(profile[start]), float(profile[stop]))) / 2.0
        above = np.flatnonzero(profile[start : stop + 1] > level)
        bodies.append((start + int(above[0]), start + int(above[-1])))
    return bodies


def assert_no_separator_inside_a_band(
    profile: np.ndarray, roi: RoiRect, separators: list[float], centers: list[float]
) -> None:
    """通用不变量：分隔线不许落在任何一条条带的信号显著区内。

    "分隔线切在条带里"整整一类 bug 的不变量都是这一条。它比"分隔线离条带
    中心至少多远"更贴题：宽窄不同的条带各按自己的身宽算，不必给每种合成
    几何各挑一个容差。切在条带里的直接后果是两侧泳道都被同一条条带的信号
    占满，积分与比值双双失真（真实膜实测：分隔线切进强带时两条泳道的积分
    比是 1.0178，看上去毫无异常，实际上量的是同一条带的左右两半）。
    """
    bodies = band_bodies(profile, centers)
    for separator in separators:
        rel = separator - roi.x
        for index, (start, stop) in enumerate(bodies):
            assert not start <= rel <= stop, (
                f"分隔线 {separator} 切进了第 {index + 1} 条条带的信号区 [{start}, {stop}]"
            )


def assert_one_band_per_lane(
    profile: np.ndarray, roi: RoiRect, separators: list[float], centers: list[float]
) -> None:
    spans = lane_spans(roi, separators)
    assert len(spans) == len(centers), f"泳道数 {len(spans)} ≠ 条带数 {len(centers)}"
    for index, center in enumerate(centers):
        left, right = spans[index]
        assert left < roi.x + center < right, f"第 {index + 1} 条条带没落在自己的泳道里"
    assert_no_separator_inside_a_band(profile, roi, separators, centers)


def test_tight_neighbours_are_split_even_when_one_is_faint() -> None:
    # 用户真实样图的核心症状：一条弱带紧挨着一条强带（间距与其余泳道相同），
    # 旧的等宽网格把两者并成一个双倍宽的泳道。分隔线必须落在两者之间。
    centers = [45.0 + 47.0 * i for i in range(8)]
    heights = [30.0, 100.0, 95.0, 70.0, 60.0, 90.0, 100.0, 65.0]
    profile = smiling_comb(400, centers, heights)
    roi = RoiRect(0, 0, 400, 40)
    assert_one_band_per_lane(
        profile, roi, separators_from_profile(profile, roi, expected_lanes=None), centers
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
    assert_one_band_per_lane(profile, roi, separators, centers)
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
    assert_one_band_per_lane(profile, roi, separators, centers)
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
        profile, roi, separators_from_profile(profile, roi, expected_lanes=None), centers
    )


def test_band_clipped_by_the_roi_edge_still_gets_a_lane() -> None:
    # 用户常把 ROI 贴着最外侧条带画，边上那条会被切掉一半——它仍然是一条泳道。
    centers = [30.0, 78.0, 126.0, 174.0, 222.0]
    profile = smiling_comb(235, centers, [80.0] * len(centers))  # 末条带被右边界切断
    roi = RoiRect(0, 0, 235, 40)
    assert_one_band_per_lane(
        profile, roi, separators_from_profile(profile, roi, expected_lanes=None), centers
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


# --- ROI 高度敏感性 ---------------------------------------------------------
#
# 同一块膜、同一批泳道，只因为 ROI 画高了 29px 就丢掉 3 条分隔线：条带计数原先
# 走的是幅度判据（剖面 30 分位数基线 + 8% 动态范围的峰检测），ROI 往上多框十几
# 行，两个 0.02 深的肩峰消失，计数从 8 掉到 6、单位宽度虚高 34%。所以下面的用例
# 一律**扫一遍 ROI 高度**，只钉一个高度是钉不住这个 bug 的。

BLOT_PITCH = 66.0
BLOT_HALF_WIDTH = 25.0
BLOT_SMILE_DIP = 0.40
BLOT_SMILE_SIGMA = 11.0
BLOT_ARC = 6.0
BLOT_ROW_SIGMA = 8.0
BLOT_ROW_CENTER = 160.0
BLOT_MARGIN = 30.0
BLOT_AMPLITUDES = (1.00, 0.45, 0.95, 0.26, 0.90, 0.50, 1.00, 0.34)
BLOT_ROWS = 340


def curved_blot() -> tuple[np.ndarray, list[float]]:
    """合成一张 8 泳道的浅底深带膜，条带是**向上弯的弧**。

    比 :func:`smiling_comb` 多两样真实膜才有的东西，缺了哪样都复刻不出这个 bug：

    * **二维**。列剖面是 ROI 行范围内的平均，ROI 多框几行才谈得上"高度敏感"。
    * **弯曲**。条带弯了，"某一行切到条带的哪一段"就随 x 变化，列剖面因此不是
      行方向的可分离积分——ROI 上下边一动，剖面形状跟着变（不只是整体缩放）。
      条带若是笔直的，加多少行都只是仿射变换，什么判据都不会出错。

    几何对着样例图定：泳道间距 66px、条带足迹 50px（相邻条带几乎贴着，足迹连成
    一片），微笑凹陷实测相对深度 0.19（样例图实测 0.006~0.208），强弱交替。
    """
    centers = [BLOT_MARGIN + BLOT_HALF_WIDTH + BLOT_PITCH * i for i in range(8)]
    width = int(centers[-1] + BLOT_HALF_WIDTH + BLOT_MARGIN)
    x = np.arange(width, dtype=float)
    y = np.arange(BLOT_ROWS, dtype=float)[:, None]
    signal = np.zeros((BLOT_ROWS, width), dtype=float)
    for center, amplitude in zip(centers, BLOT_AMPLITUDES, strict=True):
        offset = x - center
        body = np.exp(-((offset / BLOT_HALF_WIDTH) ** 6))  # 平顶
        smile = 1.0 - BLOT_SMILE_DIP * np.exp(-0.5 * (offset / BLOT_SMILE_SIGMA) ** 2)
        arc = 2.0 * np.clip(offset / BLOT_HALF_WIDTH, -1.0, 1.0) ** 2 - 1.0
        ridge = BLOT_ROW_CENTER + BLOT_ARC * arc
        signal += (
            215.0 * amplitude * body * smile
            * np.exp(-0.5 * ((y - ridge) / BLOT_ROW_SIGMA) ** 2)
        )
    return np.clip(255.0 - signal, 0, 255).astype(np.uint8), centers


BLOT_TOPS = (90, 110, 130, 145, 150)
BLOT_HEIGHTS = (40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200)


def test_curved_blot_split_is_the_same_at_every_roi_height() -> None:
    # ROI 高度从 40 扫到 200、顶边再挪 5 个位置：同一张图上分隔线不能因为 ROI
    # 画得高一点就消失或多出来，位置也得钉住。
    from wb_analyzer.roi import roi_intensity_profile

    image, centers = curved_blot()
    width = image.shape[1] - 40
    reference: list[float] | None = None
    for top in BLOT_TOPS:
        for height in BLOT_HEIGHTS:
            roi = RoiRect(20, top, width, height)
            profile, _ = roi_intensity_profile(image, roi, background_mode="dark_bands")
            separators = separators_from_profile(profile, roi, expected_lanes=None)
            where = f"ROI 顶边 {top}、高 {height}"
            assert len(separators) == 7, f"{where} 切出了 {len(separators)} 条分隔线"
            assert_one_band_per_lane(profile, roi, separators, [c - roi.x for c in centers])
            if reference is None:
                reference = separators
            for moved, plain in zip(separators, reference, strict=True):
                assert abs(moved - plain) <= 4.0, f"{where} 把分隔线挪了 {moved - plain:.1f}px"


def test_curved_blot_separators_land_between_the_bands() -> None:
    # 数量对不代表位置对：每条分隔线都要落在相邻两条带的中点附近，而不是切进去。
    from wb_analyzer.roi import roi_intensity_profile

    image, centers = curved_blot()
    roi = RoiRect(20, 130, image.shape[1] - 40, 89)
    profile, _ = roi_intensity_profile(image, roi, background_mode="dark_bands")
    separators = separators_from_profile(profile, roi, expected_lanes=None)
    midpoints = [(a + b) / 2.0 for a, b in zip(centers, centers[1:], strict=False)]
    for separator, midpoint in zip(separators, midpoints, strict=True):
        assert abs(separator - midpoint) <= 0.15 * BLOT_PITCH


def test_noise_ripples_on_a_flat_topped_band_are_not_two_bands() -> None:
    # 宽度判据只看峰在哪、不看峰多高，所以一条平顶宽带顶上的噪声波纹也会被它当成
    # 可以分开的鼓包。真实症状：16 位合成图上一条 80px 宽的平顶带，顶上三个 0.005
    # 深的波纹被数成 2 条条带，单位宽度腰斩，指定 2 条泳道时那条带被从中间劈开、
    # 空白对照泳道跟着丢掉"低于检出限"的判定。噪声级谷深封顶挡的就是这个。
    rng = np.random.default_rng(4)
    profile = np.zeros(180, dtype=float)
    profile[5:85] = 100.0  # 一条 80px 宽的平顶带，右边整片空白
    profile += rng.normal(0.0, 0.8, profile.size)  # 顶上的波纹相对深度约 0.005~0.011
    roi = RoiRect(15, 60, 180, 130)
    separators = separators_from_profile(profile, roi, expected_lanes=2)
    assert len(separators) == 1
    assert separators[0] - roi.x > 85, "分隔线切进了唯一的那条条带"


def deep_smile_comb(size: int, centers: list[float], dip: float) -> np.ndarray:
    """微笑凹陷可调的平顶宽条带；``dip`` 越大凹得越深。"""
    x = np.arange(size, dtype=float)
    profile = np.zeros(size, dtype=float)
    for center in centers:
        body = np.exp(-(((x - center) / BAND_HALF_WIDTH) ** 4))
        smile = 1.0 - dip * np.exp(-0.5 * ((x - center) / SMILE_SIGMA) ** 2)
        profile += 90.0 * body * smile
    return profile


@pytest.mark.parametrize("dip", (0.45, 0.55, 0.65))
def test_deep_smiling_bands_are_not_cut_in_half(dip: float) -> None:
    # 凹得比"泳道间隙"还深的微笑条带（实测相对深度 0.20 / 0.29 / 0.40）。样例图上
    # 同一条条带的微笑凹陷实测 0.200、而紧邻两条泳道之间的真谷实测 0.080——两拨值
    # 互相穿插，靠谷深阈值分不开，只有"这两个鼓包离得够不够远"分得开。
    centers = [40.0 + 48.0 * i for i in range(8)]
    profile = deep_smile_comb(400, centers, dip)
    roi = RoiRect(0, 0, 400, 40)
    separators = separators_from_profile(profile, roi, expected_lanes=None)
    assert len(separators) == 7, f"微笑凹陷 {dip} 切出了 {len(separators)} 条分隔线"
    for separator in separators:
        nearest = min(abs(separator - center) for center in centers)
        assert nearest >= BAND_HALF_WIDTH, f"分隔线 {separator} 切进了条带"


DEMO_STANDARD_ROI = RoiRect(430, 400, 560, 60)
DEMO_STANDARD_SEPARATORS = [499.0, 559.0, 640.0, 702.0, 784.0, 844.0, 927.0]


def demo_profile(roi: RoiRect):
    demo = Path(__file__).resolve().parents[1] / "pad4_001(1)_signal.tif"
    if not demo.exists():
        pytest.skip("demo blot image not present")
    from wb_analyzer.roi import load_bgr, roi_intensity_profile, to_gray

    gray = to_gray(load_bgr(demo))
    profile, _ = roi_intensity_profile(gray, roi, background_mode="dark_bands")
    return profile


def test_demo_standard_roi_separators_are_unchanged() -> None:
    # 标准 ROI 是所有既有回归的基准，逐值钉死：任何改动都不许挪动它。
    profile = demo_profile(DEMO_STANDARD_ROI)
    for expected in (None, 8):
        assert (
            separators_from_profile(profile, DEMO_STANDARD_ROI, expected_lanes=expected)
            == DEMO_STANDARD_SEPARATORS
        )


def test_demo_tall_roi_finds_the_same_lanes_as_the_standard_roi() -> None:
    # 复现报告里的 ROI：比标准 ROI 高 29px、左右各多框几列，泳道一条没变。
    # 修复前这里只切出 4 条分隔线，L3 只剩全宽积分的 54%、泳道比值错 40~51%。
    roi = RoiRect(421, 386, 567, 89)
    separators = separators_from_profile(demo_profile(roi), roi, expected_lanes=None)
    assert len(separators) == 7
    for tall, standard in zip(separators, DEMO_STANDARD_SEPARATORS, strict=True):
        assert abs(tall - standard) <= 2.0


@pytest.mark.parametrize("top", (390, 395, 400, 405, 410))
def test_demo_blot_split_is_the_same_at_every_roi_height(top: int) -> None:
    # 真实样图上的高度扫描：ROI 高从 50 到 200、顶边挪 5 个位置，泳道数都得是 8。
    # 修复前只有顶边压在 400 以下、或者 ROI 很矮时才对，其余大片区域退化成 4~5 条。
    for height in range(50, 201, 10):
        roi = RoiRect(430, top, 560, height)
        separators = separators_from_profile(demo_profile(roi), roi, expected_lanes=None)
        assert len(separators) == 7, f"ROI 顶边 {top}、高 {height} 切出了 {len(separators)} 条"
        for moved, standard in zip(separators, DEMO_STANDARD_SEPARATORS, strict=True):
            assert abs(moved - standard) <= 4.0


# --- 动态范围很大的膜：弱带、辉光、强带被全局电平连成一片 ----------------------
#
# 真实膜上的症状：一条弱带与一条强带相差 4 倍，强带周围一圈弥散辉光
# 把两者之间的空隙填到全局电平以上，`smooth > floor + 0.12 × span` 因此切出**一个**
# 连通区，条带足迹总量虚高一倍。但真正让分隔线切进强带的不是这个：足迹哪怕量得
# 完全准确，把宽强带从中间劈开仍然更便宜——劈完两半各自都像一条装了一条条带的
# 正常泳道（实测代价 1.9），而切在真空隙上要留一条只装了小弱带的窄泳道（5.9）。
# 足迹先验按定义看不出"这两半是同一条条带"，只有条带的**身宽**看得出。

WIDE_RANGE_SIZE = 800
WIDE_RANGE_WEAK_CENTER = 130.0
WIDE_RANGE_WEAK_HALF = 18.0
WIDE_RANGE_STRONG_CENTER = 440.0
WIDE_RANGE_STRONG_HALF = 110.0
WIDE_RANGE_STRONG_AMPLITUDE = 100.0
WIDE_RANGE_WEAK_AMPLITUDE = 25.0  # 强弱相差正好 4 倍
WIDE_RANGE_GLOW_AMPLITUDE = 34.0
WIDE_RANGE_GAP = (
    WIDE_RANGE_WEAK_CENTER + WIDE_RANGE_WEAK_HALF,
    WIDE_RANGE_STRONG_CENTER - WIDE_RANGE_STRONG_HALF,
)  # 两条条带的平顶边缘之间，真正空着的那一段


def wide_dynamic_range_profile() -> np.ndarray:
    """一弱一强两条条带 + 强带周围的弥散膜辉光。

    * 强带：220px 宽的平顶带，顶上有两道 10% 的纹路（真实强带都不是光板一块，
      这两道纹路正是旧代码切下去的地方）。
    * 弱带：幅度只有强带的四分之一，足迹 36px。
    * 辉光：以强带为中心的宽缓底座，在两条条带之间的空隙处仍高过全局电平——
      也就是把弱带、空隙、强带焊成一个连通区的那一层。
    """
    x = np.arange(WIDE_RANGE_SIZE, dtype=float)
    strong = WIDE_RANGE_STRONG_AMPLITUDE * np.exp(
        -(((x - WIDE_RANGE_STRONG_CENTER) / WIDE_RANGE_STRONG_HALF) ** 6)
    )
    for offset in (-45.0, 45.0):
        strong *= 1.0 - 0.10 * np.exp(
            -0.5 * ((x - WIDE_RANGE_STRONG_CENTER - offset) / 16.0) ** 2
        )
    weak = WIDE_RANGE_WEAK_AMPLITUDE * np.exp(
        -(((x - WIDE_RANGE_WEAK_CENTER) / WIDE_RANGE_WEAK_HALF) ** 4)
    )
    glow = WIDE_RANGE_GLOW_AMPLITUDE * np.exp(
        -(((x - WIDE_RANGE_STRONG_CENTER) / 330.0) ** 4)
    )
    return strong + weak + glow


def test_wide_dynamic_range_bands_form_one_blob_above_the_global_level() -> None:
    # 前置条件：这条剖面确实复刻了"全局电平把弱带和强带连成一片"的几何。
    # 断言的是合成数据本身，钉住它才知道下面那条用例测的是这个 bug。
    profile = wide_dynamic_range_profile()
    floor, span = float(profile.min()), float(profile.max() - profile.min())
    level = floor + 0.12 * span  # BAND_LEVEL_FRAC
    inside_gap = profile[int(WIDE_RANGE_GAP[0]) : int(WIDE_RANGE_GAP[1])]
    assert inside_gap.min() > level, "空隙没有被辉光填到全局电平以上，复刻失败"
    above = profile > level
    edges = int(np.count_nonzero(above[1:] != above[:-1]))
    assert edges == 2, f"电平以上应当只有一个连通区，实得 {edges // 2 + 1} 个"


@pytest.mark.parametrize("expected", (None, 2))
def test_faint_band_beside_a_strong_one_is_split_in_the_real_gap(expected: int | None) -> None:
    # 修复前：分隔线落在 392（强带左侧那道纹路上），两条泳道各拿到强带的一半，
    # 积分比看上去很正常（真实膜实测 1.0178），其实量的是同一条条带的左右两半。
    profile = wide_dynamic_range_profile()
    roi = RoiRect(0, 0, WIDE_RANGE_SIZE, 60)
    centers = [WIDE_RANGE_WEAK_CENTER, WIDE_RANGE_STRONG_CENTER]
    separators = separators_from_profile(profile, roi, expected_lanes=expected)
    assert len(separators) == 1, f"两条条带应当切成两道，实得 {separators}"
    assert_no_separator_inside_a_band(profile, roi, separators, centers)
    low, high = WIDE_RANGE_GAP
    assert low < separators[0] - roi.x < high, (
        f"分隔线 {separators[0]} 没落在真空隙 [{low}, {high}] 内"
    )


def test_separators_never_land_inside_a_band() -> None:
    """通用不变量：任何合成几何上，分隔线都不许落在条带的信号显著区内。

    比"钉住某张图的分隔线坐标"值钱的地方在于它对**整类** bug 有效：等宽网格
    错位、微笑凹陷被当成泳道间隙、宽强带被从纹路上劈开——症状各不相同，坏掉
    的不变量都是这一条。
    """
    cases: list[tuple[str, np.ndarray, RoiRect, list[float], int | None]] = []
    even = [40.0 + 48.0 * i for i in range(8)]
    cases.append(("等间距微笑条带", smiling_comb(400, even, [90.0] * 8), RoiRect(0, 0, 400, 40), even, None))
    cases.append(("指定 8 泳道", smiling_comb(400, even, [90.0] * 8), RoiRect(0, 0, 400, 40), even, 8))
    for dip in (0.45, 0.55, 0.65):
        cases.append(
            (f"深微笑凹陷 {dip}", deep_smile_comb(400, even, dip), RoiRect(0, 0, 400, 40), even, None)
        )
    faint = [45.0 + 47.0 * i for i in range(8)]
    cases.append(
        (
            "弱带紧邻强带",
            smiling_comb(400, faint, [30.0, 100.0, 95.0, 70.0, 60.0, 90.0, 100.0, 65.0]),
            RoiRect(0, 0, 400, 40),
            faint,
            None,
        )
    )
    crowded = [60.0 * (i + 1) for i in range(24)]
    cases.append(
        ("满梳 24 泳道", smiling_comb(1500, crowded, [80.0] * 24), RoiRect(0, 0, 1500, 40), crowded, None)
    )
    shoulder_centers = [20.0 + 40.0 * i for i in range(8)]
    shoulder = shoulder_comb_profile(320, 40.0, strong=100.0, weak=18.0)
    cases.append(("强带带肩峰", shoulder, RoiRect(50, 0, 320, 40), shoulder_centers, 8))
    wide = wide_dynamic_range_profile()
    cases.append(
        (
            "弱带 + 辉光 + 强带",
            wide,
            RoiRect(0, 0, WIDE_RANGE_SIZE, 60),
            [WIDE_RANGE_WEAK_CENTER, WIDE_RANGE_STRONG_CENTER],
            None,
        )
    )
    image, blot_centers = curved_blot()
    for top, height in ((130, 89), (110, 160), (150, 200)):
        from wb_analyzer.roi import roi_intensity_profile

        roi = RoiRect(20, top, image.shape[1] - 40, height)
        curved, _ = roi_intensity_profile(image, roi, background_mode="dark_bands")
        cases.append(
            (f"弯曲膜 顶边 {top} 高 {height}", curved, roi, [c - roi.x for c in blot_centers], None)
        )

    for tag, profile, roi, centers, expected in cases:
        separators = separators_from_profile(profile, roi, expected_lanes=expected)
        try:
            assert_no_separator_inside_a_band(profile, roi, separators, centers)
        except AssertionError as exc:  # noqa: PERF203 - 报错要说清是哪一张图
            raise AssertionError(f"{tag}：{exc}") from exc

"""ROI geometry, lane separator placement, rotation helpers, and intensity profiles."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .analysis import BackgroundMode, _resolve_dark_bands


@dataclass(frozen=True)
class RoiRect:
    x: int
    y: int
    width: int
    height: int

    def clipped(self, gray: np.ndarray) -> RoiRect:
        h, w = gray.shape[:2]
        x1 = int(np.clip(self.x, 0, max(0, w - 1)))
        y1 = int(np.clip(self.y, 0, max(0, h - 1)))
        x2 = int(np.clip(x1 + max(1, self.width), x1 + 1, w))
        y2 = int(np.clip(y1 + max(1, self.height), y1 + 1, h))
        return RoiRect(x1, y1, x2 - x1, y2 - y1)


def load_bgr(path) -> np.ndarray:
    """Load an image as BGR, with TIFF-friendly fallbacks for Windows paths."""
    from pathlib import Path

    from PIL import Image

    path = Path(path)
    try:
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is not None:
            return image
    except OSError:
        pass

    # Some TIFF variants fail in cv2.imdecode; Pillow is more tolerant.
    try:
        with Image.open(path) as pil_image:
            pil_image = pil_image.convert("RGB")
            rgb = np.array(pil_image)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except OSError as exc:
        raise ValueError(f"不支持或已损坏的图片：{path}") from exc


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def angle_to_horizontal(x0: float, y0: float, x1: float, y1: float) -> float:
    """Return the segment tilt in degrees relative to horizontal (image coords)."""
    return float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))


MAX_STRAIGHTEN_ANGLE = 45.0
"""How far the fine-rotation control may straighten a blot, in degrees.

Straightening resamples the image, so every degree costs a little sharpness;
past 45° the interpolation blur starts to matter more than the tilt it removes,
and a blot that far off is better fixed with a 90° quarter turn first. The
single source of truth for both the core geometry helpers and the GUI spin box.
"""


def clamp_straighten_angle(angle_deg: float) -> float:
    """Clamp a straightening angle to ±:data:`MAX_STRAIGHTEN_ANGLE`."""
    return float(np.clip(float(angle_deg), -MAX_STRAIGHTEN_ANGLE, MAX_STRAIGHTEN_ANGLE))


def normalize_angle(angle_deg: float) -> float:
    """Map angle into (-90, 90] for blot straightening."""
    angle = ((float(angle_deg) + 180.0) % 360.0) - 180.0
    if angle > 90.0:
        angle -= 180.0
    elif angle <= -90.0:
        angle += 180.0
    return angle


def rotate_image(
    image: np.ndarray,
    angle_deg: float,
    *,
    border_value: float | None = None,
) -> np.ndarray:
    """Rotate image around center.

    Positive degrees follow OpenCV/image-coordinate convention: they reduce a
    down-right (positive atan2) tilt, i.e. lift the right side.

    ``border_value=None`` fills the expanded corners with the median tone of
    the image border, so dark-background blots don't get white wedges that
    would skew background estimates.
    """
    if abs(angle_deg) < 1e-6:
        return image.copy()
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]
    if border_value is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        edges = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
        border_value = float(np.median(edges))
    border = (border_value,) * 3 if image.ndim == 3 else border_value
    return cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def default_separators(roi: RoiRect, lane_count: int) -> list[float]:
    """Absolute X positions of internal lane separators (equal split fallback)."""
    n = max(1, int(lane_count))
    if n <= 1:
        return []
    return [roi.x + roi.width * i / n for i in range(1, n)]


def _smooth_profile(values: np.ndarray, radius: int = 4) -> np.ndarray:
    """Moving average with edge padding.

    Zero padding would drop the first and last ``radius`` samples toward zero,
    creating fake valleys at the ROI edges that separator placement and the
    autocorrelation lane-count estimate would both take seriously.
    """
    radius = max(1, int(radius))
    kernel = np.ones(radius * 2 + 1, dtype=float)
    kernel /= kernel.sum()
    padded = np.pad(np.asarray(values, dtype=float), radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def find_profile_peaks(
    profile: np.ndarray,
    *,
    expected_lanes: int | None = None,
    min_distance: int | None = None,
) -> list[int]:
    """Return peak indices on an ROI intensity profile."""
    if profile.size < 8:
        return []
    smooth = _smooth_profile(profile, max(2, profile.size // 80))
    baseline = float(np.percentile(smooth, 30))
    span = float(smooth.max() - baseline)
    if span <= 1e-6:
        return []
    # Keep spacing below typical strong/weak pair gaps; flat tops are handled by
    # plateau centering below rather than an oversized min-distance.
    if min_distance is not None:
        distance = max(3, int(min_distance))
    elif expected_lanes and expected_lanes > 1:
        distance = max(8, profile.size // max(expected_lanes * 3, 16))
    else:
        distance = max(8, profile.size // 20)
    win = max(2, distance // 4)
    min_prom = span * 0.08
    plateau_eps = max(1e-6, span * 0.02)
    candidates: list[int] = []
    index = win
    while index < len(smooth) - win:
        neighborhood = smooth[index - win : index + win + 1]
        if smooth[index] >= float(neighborhood.max()) - 1e-12 and smooth[index] - baseline >= min_prom:
            left = index
            while left > 0 and abs(smooth[left - 1] - smooth[index]) <= plateau_eps:
                left -= 1
            right = index
            while right < len(smooth) - 1 and abs(smooth[right + 1] - smooth[index]) <= plateau_eps:
                right += 1
            candidates.append((left + right) // 2)
            index = right + 1
            continue
        index += 1
    # 肩部假峰不在这里合并：真实样图基准扫描证明任何谷深合并阈值都会误合并
    # 多膜拼贴图的真峰（合并越激进正确率越低）。分隔线放置不再依赖本函数，
    # 它用 _fit_lane_separators 的间距先验来区分肩峰与真泳道间谷。
    candidates = sorted(set(candidates), key=lambda i: smooth[i], reverse=True)
    peaks: list[int] = []
    for peak in candidates:
        if all(abs(peak - other) >= distance for other in peaks):
            peaks.append(peak)
        if expected_lanes and len(peaks) >= max(1, expected_lanes):
            break
    return sorted(peaks)


GRID_SNAP_WINDOW_FRAC = 0.25
"""网格分隔线的谷位吸附半窗，占单泳道宽度的比例。"""

AUTO_GRID_MAX_SEPARATOR_HEIGHT = 0.55
"""自动网格分隔线允许的最大相对高度（0=谷底 1=最高峰），超出则回退峰谷。"""


def _grid_valley_positions(
    smooth: np.ndarray,
    lanes: int,
    *,
    window_frac: float = GRID_SNAP_WINDOW_FRAC,
    center_pull: float = 0.03,
) -> list[int]:
    """等宽网格初始化 + 谷位吸附后的分隔位（相对 ROI 的列索引）。

    泳道由梳子成型、物理上近似等宽：先按 ``lanes`` 等分给出 n-1 个初始
    分隔位，再让每根线在 ±window_frac×泳道宽 内吸附到平滑剖面的局部最低
    点。``center_pull`` 给一个朝网格位的微小拉力，让大片空白（平坦谷）里
    的分隔线停在网格附近而不是漂到窗口边缘。窗口 ≤ 半泳道宽保证相邻
    分隔线不会交叉，额外再做单调钳制兜底。
    """
    size = int(smooth.size)
    lane_w = size / float(lanes)
    span = float(smooth.max() - smooth.min())
    if span <= 1e-9:
        # 平坦剖面（空白 ROI）：直接用精确等分位。
        return [int(round(lane_w * i)) for i in range(1, lanes)]
    half = lane_w * float(min(window_frac, 0.49))
    positions: list[int] = []
    prev = 0
    for i in range(1, lanes):
        center = lane_w * i
        lo = max(int(np.ceil(center - half)), prev + 2, 1)
        hi = min(int(np.floor(center + half)), size - 2)
        if hi < lo:
            pos = int(np.clip(round(center), prev + 1, size - 1))
        else:
            idx = np.arange(lo, hi + 1)
            pull = span * float(center_pull) * np.abs(idx - center) / max(half, 1.0)
            pos = int(idx[int(np.argmin(smooth[lo : hi + 1] + pull))])
        positions.append(pos)
        prev = pos
    return positions


def _autocorr_lag(
    smooth: np.ndarray,
    *,
    ac_min: float = 0.25,
    sub_min: float = 0.10,
    max_lanes: int = 24,
) -> float | None:
    """剖面自相关的主周期（像素）；周期性不足返回 None。

    强弱交替的泳道布局会让自相关主峰落在 2×真实间距上（强带与强带对齐）；
    只要半周期处也存在幅度 ≥ ``sub_min`` 的局部峰（说明半间距上真有条带），
    就取更细的周期。

    ``min_lag`` 的下界同时挡住了另一件事：单条条带自身有二三十像素宽，它与
    自己错开小半个身位仍高度重叠，小 lag 上的自相关因此天然偏高（真实样图上
    lag=19 的自相关 0.585 比真实间距 lag=47 的 0.407 还大）。把搜索下限压在
    ``size / max_lanes`` 才不会把"条带有宽度"误当成"泳道很密"。
    """
    size = int(smooth.size)
    if size < 24:
        return None
    x = smooth - float(smooth.mean())
    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        return None
    ac = np.correlate(x, x, mode="full")[size - 1 :] / denom
    min_lag = max(8, size // max_lanes)
    max_lag = size // 2
    if max_lag <= min_lag + 1:
        return None
    best_lag = None
    best_val = float(ac_min)
    for lag in range(min_lag, max_lag + 1):
        left = float(ac[lag - 1])
        right = float(ac[lag + 1]) if lag + 1 < ac.size else -np.inf
        if ac[lag] >= left and ac[lag] >= right and float(ac[lag]) >= best_val:
            best_lag, best_val = lag, float(ac[lag])
    if best_lag is None:
        return None
    lag = best_lag
    while lag // 2 >= min_lag:
        lo = max(min_lag, lag // 2 - 2)
        hi = min(max_lag, lag // 2 + 3)
        j = lo + int(np.argmax(ac[lo : hi + 1]))
        is_peak = (
            0 < j < ac.size - 1
            and float(ac[j]) >= float(ac[j - 1])
            and float(ac[j]) >= float(ac[j + 1])
            and float(ac[j]) >= float(sub_min)
        )
        if not is_peak:
            break
        lag = j
    return float(lag)


def _estimate_lane_count(
    smooth: np.ndarray,
    *,
    ac_min: float = 0.25,
    sub_min: float = 0.10,
    max_lanes: int = 24,
) -> int | None:
    """用剖面自相关的主周期估计泳道数（等宽先验）；周期性不足返回 None。"""
    lag = _autocorr_lag(smooth, ac_min=ac_min, sub_min=sub_min, max_lanes=max_lanes)
    if lag is None:
        return None
    lanes = int(round(int(smooth.size) / lag))
    if lanes < 2 or lanes > max_lanes:
        return None
    return lanes


def _pick_lane_count(size: int, lanes: int, peaks: list[int], *, max_lanes: int = 24) -> int:
    """在 ``lanes`` 与 ``2×lanes`` 之间用峰证据消歧。

    强弱交替布局（强-弱-强-弱）会让自相关主峰落在 2×真实泳道间距上（弱带
    太弱时半周期处甚至没有自相关峰）。判据：把 ROI 等分成候选数量的格子，
    统计每格内检出峰的数量——正确的泳道数应让大多数格子恰好 1 峰；格子里
    挤了 ≥2 个峰说明划分过粗。空格子不罚分（空白泳道合法）。
    """
    candidates = [lanes]
    doubled = lanes * 2
    if doubled <= max_lanes and size / doubled >= 8:
        candidates.append(doubled)
    if len(candidates) == 1 or not peaks:
        return lanes

    def score(n: int) -> float:
        counts = [0] * n
        cell_w = size / n
        for peak in peaks:
            counts[min(n - 1, int(peak / cell_w))] += 1
        exact = sum(1 for c in counts if c == 1)
        crowded = sum(1 for c in counts if c >= 2)
        return exact - 0.5 * crowded

    return max(candidates, key=score)


NOISE_VALLEY_FRAC = 0.005
"""持久性剪枝阈值：谷深不足剖面动态范围这个比例时，相邻两峰算同一个峰。

只用来清掉空白区里那种起伏不到千分之几的噪声"峰"。阈值必须压得很低：真实
样图上强带紧挨弱带时，两者之间的真分隔谷相对深度只有 0.08，剪得稍狠一点
（3%）就会把弱带整个并进强带，那条线之后再也切不出来。多留下来的浅谷只是
DP 的候选，代价高自然选不上，比漏掉候选安全得多。
"""

SEPARATOR_QUALITY_WEIGHT = 1.6
"""动态规划目标里"谷质量"项相对"宽度先验"项的权重。

调大 = 更信剖面（宁可泳道宽窄不均也要切在深谷上）；调小 = 更信等间距先验。
"""

BAND_LEVEL_FRAC = 0.12
"""判定"这一列上有条带"的电平，从剖面地板起算，占动态范围的比例。"""

BAND_COUNT_DEPTH = 0.40
"""条带数的起点：谷深到这个程度，两侧一定是两条不同的条带。"""

BAND_COUNT_DEPTH_LOOSE = 0.20
"""整条剖面一个深谷都没有时，退而求其次的起点谷深。

只在"没有任何谷深过 :data:`BAND_COUNT_DEPTH`"（泳道挨得极紧的膜）时才用得上。
它**不再决定条带数**——真实样图上同一条条带的微笑凹陷深到 0.20、而紧邻两条
泳道之间的真谷浅到 0.08，谷深根本分不开这两件事，任何取值都是掷硬币。条带数
改由 :func:`_resolved_band_count` 按"宽度"判定。
"""

BAND_NOISE_DEPTH = 0.05
"""条带数的硬上界：谷浅到这个程度，两侧只能是噪声或同一条条带的纹理。

宽度判据只看峰在哪，不看峰有多高，所以一条平顶宽带上的噪声起伏也会被它当成
可以分开的鼓包（真实症状：16 位合成图上一条 80px 宽的平顶带，顶上三个 0.005
深的噪声波纹被数成 2 条条带，单位宽度腰斩，指定 2 条泳道时那条带被从中间劈
开）。这里按谷深封顶——0.005 量级的谷是噪声，这个判断没有歧义，不像"微笑凹陷
还是泳道间隙"那样两拨值互相穿插。

窗口很宽：下边界是噪声波纹的 0.008，上边界是样例图上必须保住的第 7 条真谷
0.197，取 0.05 离两头都有 4~6 倍的余量。
"""

BAND_MERGE_FRAC = 0.57
"""两个局部峰相距不足这个比例的条带宽时，判定它们是同一条条带的两个鼓包。

这是条带计数唯一的尺度先验，取的是**宽度**而不是**幅度**：两条不同泳道的条带
中心至少隔一整个条带宽（否则它们就重叠了），而同一条条带的微笑双峰、肩峰只隔
半个身位。样例图上真实泳道的最近峰距 25px、条带宽 49px（0.51 倍），同一条条带
的两个鼓包相距 17~23px（0.35~0.47 倍）。

上下界都由回归用例钉住，可用区间实测为 [0.55, 0.60]，这里取几何中点：
* 调到 0.61 以上，样例图上紧邻的两条泳道被并成一条（
  ``test_demo_blot_split_is_the_same_at_every_roi_height`` 变红）；
* 调到 0.54 以下，微笑凹陷 0.19 的弯条带被从每条条带正中劈开（
  ``test_curved_blot_split_is_the_same_at_every_roi_height`` 变红）。
"""

LANE_NARROW_TOLERANCE = 0.45
"""泳道足迹比单位宽度窄多少算"一个标准差"。窄一半就要付 ~1.2 的代价。"""

LANE_WIDE_TOLERANCE = 0.30
"""泳道足迹比单位宽度宽多少算"一个标准差"。

宽度代价拿**条带足迹**（泳道内电平以上的列数）而不是泳道的几何宽度去比，
这是整个方案能同时应付"空白间隙"和"紧邻条带"的关键：泳道旁边空多少无所
谓，装了几条条带才有所谓。中间的空白不进足迹，天然免罚；而两条泳道没切
开时足迹是 2 个单位，代价超过 11，压得过任何一条浅谷的质量代价——真实样
图里相对深度只有 0.08 的那条分隔线就是这样切下来的。

两侧不对称：偏窄比偏宽宽容一些，因为弱带的足迹本来就会被电平削掉一圈，
不该因此被并进邻道。
"""

EMPTY_LANE_COST = 1.5
"""偏窄一侧的代价上限，也就是"这条泳道里一条条带都没有"要付多少。

必须封顶。用户手动指定泳道数、而 ROI 里的条带比这个数少时（例如指定 2 条
泳道去测"一条强带 + 一条空白对照"），不封顶的话把强带从中间劈开反而比留
一条空泳道便宜。封在 1.5：自动模式下仍然贵到不会凭空造出空泳道，指定泳道
数时又允许它老老实实空着一条。
"""

MIN_LANE_FRAC = 0.45
"""允许的最窄泳道，占单位宽度的比例。低于此几何宽度的切法直接不进 DP。"""

BLANK_LANE_FRAC = 0.15
"""泳道内最高信号低于"各泳道中位高度"的这个比例时，判定该泳道是空白。"""

BLANK_LANE_SPAN_FRAC = 0.08
"""空白判定的另一条上限，占剖面动态范围的比例；与上一条取更小者。"""

FLAT_BASIN_FRAC = 0.03
"""判定"平坦谷底"的高度容差，占动态范围的比例。"""


def _plateau_extrema(smooth: np.ndarray) -> list[int]:
    """局部极大值的位置；等值平台取中点。

    剖面两端按"外面更低"处理：ROI 边缘常常正好切在一条条带上（用户框选时
    习惯贴着最外侧的条带），那条被切掉一半的条带在数组端点上，不这么算就
    整条丢掉，条带计数和条带区范围都会跟着错。
    """
    size = int(smooth.size)
    runs: list[tuple[int, float]] = []
    start = 0
    while start < size:
        stop = start
        while stop + 1 < size and smooth[stop + 1] == smooth[start]:
            stop += 1
        runs.append(((start + stop) // 2, float(smooth[start])))
        start = stop + 1
    last = len(runs) - 1
    return [
        runs[k][0]
        for k in range(len(runs))
        if (k == 0 or runs[k][1] > runs[k - 1][1])
        and (k == last or runs[k][1] > runs[k + 1][1])
    ]


def _merge_noise_peaks(smooth: np.ndarray, peaks: list[int], min_depth: float) -> list[int]:
    """持久性剪枝：谷深不足 ``min_depth`` 的相邻峰合并，保留较高的那个。"""
    peaks = list(peaks)
    while len(peaks) > 1:
        depths = [
            min(float(smooth[a]), float(smooth[b])) - float(smooth[a : b + 1].min())
            for a, b in zip(peaks, peaks[1:], strict=False)
        ]
        index = int(np.argmin(depths))
        if depths[index] >= min_depth:
            break
        a, b = peaks[index], peaks[index + 1]
        peaks.pop(index if smooth[a] <= smooth[b] else index + 1)
    return peaks


def _valleys_between(smooth: np.ndarray, peaks: list[int]) -> list[int]:
    return [
        a + int(np.argmin(smooth[a : b + 1]))
        for a, b in zip(peaks, peaks[1:], strict=False)
    ]


def _valley_depths(
    smooth: np.ndarray, peaks: list[int], valleys: list[int], floor: float
) -> list[float]:
    """每个谷的相对深度：相对较矮那侧峰顶，向剖面最低点下探了多少（0..1）。

    用"较矮的那侧峰"作分母，弱带旁边的真分隔线才不会因为邻居是强带而被
    稀释成一个很小的比值。
    """
    depths: list[float] = []
    for (left, right), valley in zip(
        zip(peaks, peaks[1:], strict=False), valleys, strict=True
    ):
        low = min(float(smooth[left]), float(smooth[right]))
        head = low - floor
        if head <= 1e-12:
            depths.append(0.0)
            continue
        depths.append(float(np.clip((low - float(smooth[valley])) / head, 0.0, 1.0)))
    return depths


def _peaks_after_merge(smooth: np.ndarray, peaks: list[int], distance: float) -> int:
    """把相距不足 ``distance`` 的峰当作同一条条带归并后还剩几条。

    从最高的峰开始占位，后来的峰离已占位的峰太近就并进去。先高后低是为了让
    每条条带由它自己最高的那个鼓包代表，弱带不会被强邻居的肩峰顶掉。
    """
    kept: list[int] = []
    for peak in sorted(peaks, key=lambda index: -float(smooth[index])):
        if all(abs(peak - other) >= distance for other in kept):
            kept.append(peak)
    return len(kept)


def _band_spans(
    smooth: np.ndarray, peaks: list[int], level: float, count: int, unit: float
) -> list[tuple[int, int]]:
    """每条条带自己横跨的列区间（最左鼓包到最右鼓包）；分不清就返回空表。

    分隔线落进其中任何一个区间，就等于把一条条带劈成两半分给两个泳道，所以
    这些区间是分隔线的禁区。

    条带数 ``count`` 已经由 :func:`_band_count` 定下来了，分组因此不必再按
    半径贪心占位——直接把峰序列按**最大的 count-1 个间距**切开。这个划分与
    相位无关，而贪心占位不是：峰距均匀时（微笑凹陷深到与泳道间隙同宽）它会
    从某个位置起整体错半格，把条带内的凹陷与条带之间的空隙对调，禁区正好设
    反。

    划分只在两拨间距**真的分得开**时才作数，尺子就是既有的"同一条条带"归并
    距离（:data:`BAND_MERGE_FRAC` × 条带宽）：组内的每个间距都得短于它，组间
    的每个间距都得够得着它。达不到就返回空表，让 DP 回到只看足迹与谷深的老
    判据——禁区设错地方比不设禁区糟得多。

    这道门槛必须两边都卡死，只要求"组间比组内大"是不够的。样例图上 8 条条带
    共 14 个鼓包，组间最小间距 25px、组内最大 23px，两拨值实际上是连在一起
    的；ROI 换一个高度，25 和 23 的次序就翻过来，禁区跟着挪到隔壁，一条真分
    隔线就没了。
    """
    above = [peak for peak in peaks if smooth[peak] > level]
    if unit <= 0 or len(above) < 2 or not 1 <= count < len(above):
        return []
    gaps = [b - a for a, b in zip(above, above[1:], strict=False)]
    order = sorted(range(len(gaps)), key=lambda index: gaps[index])
    split = len(gaps) - (count - 1)
    inside, cuts = order[:split], sorted(order[split:])
    distance = BAND_MERGE_FRAC * unit
    if max(gaps[index] for index in inside) >= distance:
        return []
    if cuts and min(gaps[index] for index in cuts) < distance:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    for stop in (*cuts, len(above) - 1):
        if stop > start:
            spans.append((above[start], above[stop]))
        start = stop + 1
    return spans


def _resolved_band_count(
    smooth: np.ndarray,
    peaks: list[int],
    level: float,
    band_columns: float,
    seed: int,
    ceiling: int,
) -> int:
    """从保守起点出发，按"条带有多宽"把剖面上的峰归并成条带，返回条带数。

    归并半径取 :data:`BAND_MERGE_FRAC` × 当前假设的条带宽（= 电平以上的总列
    数 ÷ 当前条带数），所以判据是自洽的：条带数越多、每条越窄、归并半径越小、
    分得出的条带就越多。从 ``seed``（谷深给出的可靠下界）单调往上松弛到不动
    点为止——只增不减保证一定收敛，也保证深谷两侧永远算两条条带；``ceiling``
    （噪声级谷深给出的可靠上界）挡住噪声波纹把计数顶穿。

    这一步替掉了原先的峰检测计数。那条路线按**绝对幅度**（剖面 30 分位数基线
    + 8% 动态范围）取舍，ROI 往上多框十几行空白就会让两个 0.02 深的肩峰消失，
    样例图上条带数因此从 8 掉到 6、单位宽度虚高 34%，动态规划跟着丢掉 3 条
    分隔线。宽度判据只看峰的相对位置，ROI 高度变化对它是仿射变换，天然不敏感。
    """
    above = [peak for peak in peaks if smooth[peak] > level]
    if not above or band_columns <= 0:
        return seed
    ceiling = max(1, min(ceiling, len(above)))
    count = max(1, min(seed, ceiling))
    for _ in range(len(above)):
        merged = min(
            _peaks_after_merge(smooth, above, BAND_MERGE_FRAC * band_columns / count),
            ceiling,
        )
        if merged <= count:
            break
        count = merged
    return count


def _valley_group_count(
    smooth: np.ndarray,
    peaks: list[int],
    depths: list[float],
    level: float,
    min_depth: float,
) -> int:
    """按谷深把峰分组数条带：谷浅于 ``min_depth`` 的两侧算同一条条带。

    只数电平以上的组，空白区里的噪声峰不算条带。这条路线不看绝对幅度，弱带
    把 ROI 占满也数得对。
    """
    groups: list[bool] = []
    for index, peak in enumerate(peaks):
        above = bool(smooth[peak] > level)
        if index == 0 or depths[index - 1] >= min_depth:
            groups.append(above)
        elif above:
            groups[-1] = True
    return sum(groups)


def _band_count(
    smooth: np.ndarray,
    peaks: list[int],
    depths: list[float],
    level: float,
    band_columns: float,
) -> int:
    """剖面上有几条条带。

    谷深只负责圈出一个**两头都可靠**的区间：深过 :data:`BAND_COUNT_DEPTH` 的
    谷两侧一定是两条条带（下界），浅过 :data:`BAND_NOISE_DEPTH` 的一定不是
    （上界）。区间内由 :func:`_resolved_band_count` 按宽度定夺，它数得出被强
    邻居压住、谷又浅的弱带。

    区间中段不能再靠谷深定夺了：真实样图上同一条条带的微笑凹陷能深到 0.20，
    而紧邻两条泳道之间的真谷能浅到 0.08，两拨值互相穿插，没有哪个阈值分得开。
    """
    lower = _valley_group_count(smooth, peaks, depths, level, BAND_COUNT_DEPTH)
    if lower <= 1:
        # 整条剖面上没有一个谷深到"一定是泳道间隙"的程度，说明这块膜上泳道
        # 挨得很紧、深谷浅谷根本分不成两拨。放宽起点，免得从 1 起步松弛不动。
        loose = _valley_group_count(smooth, peaks, depths, level, BAND_COUNT_DEPTH_LOOSE)
        if loose >= 3:
            lower = loose
    ceiling = _valley_group_count(smooth, peaks, depths, level, BAND_NOISE_DEPTH)
    lower = max(1, lower)
    return _resolved_band_count(
        smooth, peaks, level, band_columns, lower, max(lower, ceiling)
    )


def _lane_unit_width(smooth: np.ndarray, count: int, band_columns: float) -> float | None:
    """估计"一条泳道该有多少条带足迹"（像素列数）。

    直接量：电平以上的总列数 ÷ 条带数。这个量与泳道怎么排布无关——泳道之间
    空多少、成不成组、间距均不均匀都不影响它，只取决于"一条条带有多宽"，而
    同一块膜上的条带宽度是很齐的。

    这正是自相关做不到的地方：自相关给的是**周期**，两条一组、组间留空的
    膜上它抓到的是"组"的周期（样例图 141，真实单位 47），强弱交替时又会锁
    在 2× 上。

    ``count`` 由 :func:`_band_count` 给出；一条都数不出来时才退回自相关周期。
    """
    if count and band_columns > 0:
        unit = band_columns / count
        return unit if unit >= 4.0 else None
    lag = _autocorr_lag(smooth)
    if lag is None or lag < 4.0:
        return None
    return float(lag)


def _collapse_blank_lanes(
    smooth: np.ndarray, edges: list[int], floor: float, min_height: float
) -> list[int]:
    """抹掉不含条带的泳道，返回过滤后的分隔线。

    ROI 中间整段空白时，等间距先验会把它硬掰成若干个空格子。这里把连续空格
    子整体收掉，只在空白正中留一条线：空白两侧各分一半，谁也没多出一个"泳道"
    ——收紧足迹后两边都不会把空白算进积分。贴着条带区两端的空白直接丢掉。
    """
    slots = list(zip(edges, edges[1:], strict=False))
    separators: list[int] = []
    previous_end: int | None = None
    for start, stop in slots:
        if float(smooth[start : stop + 1].max()) - floor < min_height:
            continue
        if previous_end is not None:
            separators.append((previous_end + start) // 2)
        previous_end = stop
    return separators


def _footprint_cost(footprint: float, unit: float) -> float:
    """泳道内条带足迹偏离"一条泳道该有多宽"的代价（不对称）。"""
    ratio = footprint / unit
    if ratio >= 1.0:
        return ((ratio - 1.0) / LANE_WIDE_TOLERANCE) ** 2
    return min(((1.0 - ratio) / LANE_NARROW_TOLERANCE) ** 2, EMPTY_LANE_COST)


def _center_in_flat_basin(
    smooth: np.ndarray, index: int, tolerance: float, low: int, high: int
) -> int:
    """把落在空白里的分隔位挪到那段平地的中点，范围限制在 ``[low, high]``。

    空白间隙的谷底是一整段平地，argmin 落在哪里由噪声决定；取中点才让分隔线
    落在空白正中，两侧泳道各分一半空白（收紧足迹后谁都不会把空白算进积分）。
    只对贴着剖面地板的谷这么做——普通 V 形谷的 argmin 本来就是对的，对它做
    "平地展宽"会顺着单侧斜坡一路爬进条带。
    """
    level = float(smooth[index]) + tolerance
    left = index
    while left > low and smooth[left - 1] <= level:
        left -= 1
    right = index
    while right < high and smooth[right + 1] <= level:
        right += 1
    return (left + right) // 2


def _dp_boundaries(
    nodes: list[int],
    node_cost: list[float],
    band_columns: np.ndarray,
    unit: float,
    lane_count: int | None,
) -> list[int] | None:
    """在候选谷位上选一组分隔线，最小化「谷质量 + 足迹宽度偏离先验」。

    ``nodes[0]`` / ``nodes[-1]`` 是条带区的左右端，固定入选且不计质量代价；
    中间节点入选一个就多一条分隔线。``band_columns`` 是"有条带的列"的前缀
    和，用来 O(1) 取任意一段的条带足迹。``lane_count`` 非空时强制切出恰好
    这么多个泳道。返回内部分隔线在剖面中的索引。
    """
    count = len(nodes)
    if count < 2:
        return None
    slots = lane_count if lane_count else 1
    min_width = max(3.0, MIN_LANE_FRAC * unit)
    inf = float("inf")
    best = [[inf] * (slots + 1) for _ in range(count)]
    prev = [[-1] * (slots + 1) for _ in range(count)]
    best[0][0] = 0.0
    for j in range(1, count):
        for i in range(j):
            if nodes[j] - nodes[i] < min_width:
                continue
            footprint = float(band_columns[nodes[j]] - band_columns[nodes[i]])
            step = _footprint_cost(footprint, unit) + node_cost[j]
            if lane_count:
                for k in range(1, slots + 1):
                    if best[i][k - 1] == inf:
                        continue
                    candidate = best[i][k - 1] + step
                    if candidate < best[j][k]:
                        best[j][k], prev[j][k] = candidate, i
            elif best[i][0] != inf and best[i][0] + step < best[j][0]:
                best[j][0], prev[j][0] = best[i][0] + step, i
    target = slots if lane_count else 0
    if best[count - 1][target] == inf:
        return None
    chosen: list[int] = []
    node, used = count - 1, target
    while node > 0:
        parent = prev[node][used]
        if parent < 0:
            return None
        if parent > 0:
            chosen.append(nodes[parent])
        node, used = parent, (used - 1 if lane_count else 0)
    return sorted(chosen)


def _fit_lane_separators(
    smooth: np.ndarray, *, lane_count: int | None = None
) -> list[int] | None:
    """按剖面地形直接放泳道分隔线；证据不足时返回 ``None`` 让调用方回退。

    流程：局部极值 → 持久性剪枝去掉空白区噪声峰 → 算每个谷相对两侧峰顶的
    下探深度 → 量出单条泳道的条带足迹宽度 → **划掉落在条带内部的候选谷** →
    在条带区内用动态规划挑一组分隔线 → 抹掉空白泳道 → 空白里的分隔线挪到
    空白正中。

    与旧的"等宽网格 + 谷位吸附"的本质区别有两条。一是网格的相位被硬钉在
    ROI 两端，ROI 左右留白不等就整体错位，之后 ±¼ 泳道宽的吸附窗救不回来；
    这里的位置完全由剖面决定。二是旧网格约束的是泳道的几何宽度，中间有
    空白时必然错，而这里约束的是泳道里装了多少条带足迹——空白不算数。

    足迹先验挡不住的是"把一条宽条带从中间劈开"：劈完两半各自都像一条装了
    一条条带的正常泳道，代价反而比留一宽一窄两道更低（真实膜实测：完美足迹
    下劈开代价 1.9，切在真空隙上 5.9）。所以候选谷先过一道
    :func:`_band_spans` —— 落在同一条条带两个鼓包之间的谷根本不进 DP。
    这一步不看谷有多深，只看"两个鼓包是不是同一条条带"，判据与条带计数同源。
    """
    size = int(smooth.size)
    span = float(smooth.max() - smooth.min())
    if size < 16 or span <= 1e-9:
        return None
    floor = float(smooth.min())
    peaks = _merge_noise_peaks(
        smooth, _plateau_extrema(smooth), span * NOISE_VALLEY_FRAC
    )
    if len(peaks) < 2:
        return None
    valleys = _valleys_between(smooth, peaks)
    depths = _valley_depths(smooth, peaks, valleys, floor)
    level = floor + span * BAND_LEVEL_FRAC
    band_columns = np.concatenate(([0], np.cumsum(smooth > level))).astype(float)
    count = _band_count(smooth, peaks, depths, level, float(band_columns[-1]))
    unit = _lane_unit_width(smooth, count, float(band_columns[-1]))
    if unit is None:
        if not lane_count or lane_count < 2:
            return None
        # 最后兜底：拿条带跨度按用户给的泳道数均分。
        unit = (peaks[-1] - peaks[0]) / max(lane_count - 1, 1)
    if unit < 4.0:
        return None
    half = unit / 2.0
    lo = max(0, int(round(peaks[0] - half)))
    hi = min(size - 1, int(round(peaks[-1] + half)))
    if hi - lo < max(8.0, unit):
        return None
    inner = [(v, d) for v, d in zip(valleys, depths, strict=True) if lo < v < hi]
    spans = _band_spans(smooth, peaks, level, count, unit)
    between = [(v, d) for v, d in inner if not any(a < v < b for a, b in spans)]

    def fit(candidates: list[tuple[int, float]]) -> list[int] | None:
        nodes = [lo, *(v for v, _ in candidates), hi]
        cost = [0.0, *(SEPARATOR_QUALITY_WEIGHT * (1.0 - d) for _, d in candidates), 0.0]
        return _dp_boundaries(nodes, cost, band_columns, unit, lane_count)

    chosen = fit(between)
    if chosen is None and len(between) < len(inner):
        # 只有"指定泳道数比条带数还多"时才可能凑不出这么多泳道。宁可切在条带
        # 里也要给够泳道数——用户说了有这么多道，界面上看得见、挪得动。
        chosen = fit(inner)
    if chosen is None:
        return None
    edges = [lo, *chosen, hi]
    if not lane_count:
        # 泳道数由用户指定时不抹空白泳道：他要的就是这么多条。
        heights = [
            float(smooth[a : b + 1].max()) - floor
            for a, b in zip(edges, edges[1:], strict=False)
        ]
        blank = min(BLANK_LANE_FRAC * float(np.median(heights)), BLANK_LANE_SPAN_FRAC * span)
        chosen = _collapse_blank_lanes(smooth, edges, floor, blank)
        edges = [lo, *chosen, hi]
    tolerance = span * FLAT_BASIN_FRAC
    blank_level = floor + span * BLANK_LANE_SPAN_FRAC
    return [
        _center_in_flat_basin(
            smooth,
            edges[k],
            tolerance,
            (edges[k - 1] + edges[k]) // 2,
            (edges[k] + edges[k + 1]) // 2,
        )
        if smooth[edges[k]] <= blank_level
        else edges[k]
        for k in range(1, len(edges) - 1)
    ]


def _max_relative_height(smooth: np.ndarray, positions: list[int]) -> float:
    """分隔位处平滑剖面的最大相对高度（0=基线 1=最高峰）。"""
    baseline = float(np.percentile(smooth, 30))
    span = float(smooth.max() - baseline)
    if span <= 1e-9 or not positions:
        return 0.0
    return max((float(smooth[p]) - baseline) / span for p in positions)


def separators_from_profile(
    profile: np.ndarray,
    roi: RoiRect,
    *,
    expected_lanes: int | None = None,
) -> list[float]:
    """Place lane separators on an ROI column profile.

    主路径 :func:`_fit_lane_separators` 直接读剖面地形：找峰、算每个谷相对
    两侧峰顶的下探深度、量出"一条泳道该有多少条带足迹"，再用动态规划在
    "谷够深"和"每个泳道装一条条带的量"之间取平衡。因为约束的是**条带足迹**
    而不是泳道的几何宽度，间距不均、中间整段空白、两条紧挨着的条带都能
    处理；条带自身的肩部/微笑凹陷则被挡住——在那里切一刀会切出两个只装了
    半条条带的泳道。

    ``expected_lanes`` 给了就强制切出这么多泳道（位置仍由剖面决定）。地形
    证据不足时回退到旧的等宽网格 / 峰-谷逻辑。
    """
    profile = np.asarray(profile, dtype=float)
    n_target = int(expected_lanes) if expected_lanes else 0
    if n_target == 1:
        return []
    if n_target >= 2 and profile.size < max(8, n_target * 2):
        return default_separators(roi, n_target)
    if profile.size < 8:
        return []
    smooth = _smooth_profile(profile, max(2, profile.size // 80))
    fitted = _fit_lane_separators(smooth, lane_count=n_target or None)
    if fitted is not None:
        return [float(roi.x + p) for p in fitted]
    if n_target >= 2:
        return [float(roi.x + p) for p in _grid_valley_positions(smooth, n_target)]
    peaks = find_profile_peaks(profile)
    lanes = _estimate_lane_count(smooth)
    if lanes is not None:
        lanes = _pick_lane_count(profile.size, lanes, peaks)
        if profile.size >= lanes * 2:
            positions = _grid_valley_positions(smooth, lanes)
            if positions and _max_relative_height(smooth, positions) <= AUTO_GRID_MAX_SEPARATOR_HEIGHT:
                return [float(roi.x + p) for p in positions]
    if len(peaks) < 2:
        return []
    separators: list[float] = []
    for left, right in zip(peaks, peaks[1:], strict=False):
        if right - left < 3:
            continue
        valley = left + int(np.argmin(smooth[left : right + 1]))
        # Keep valley away from peak centers.
        valley = int(np.clip(valley, left + 2, right - 2))
        separators.append(float(roi.x + valley))
    return separators


def lane_boundaries(roi: RoiRect, separators: list[float]) -> list[tuple[int, int]]:
    cleaned = sorted(float(s) for s in separators if roi.x < s < roi.x + roi.width)
    edges = [float(roi.x), *cleaned, float(roi.x + roi.width)]
    bounds: list[tuple[int, int]] = []
    for left, right in zip(edges, edges[1:], strict=False):
        x1, x2 = int(round(left)), int(round(right))
        if x2 > x1:
            bounds.append((x1, x2))
    return bounds


def roi_intensity_profile(
    gray: np.ndarray,
    roi: RoiRect,
    *,
    background_mode: BackgroundMode = "auto",
) -> tuple[np.ndarray, bool]:
    """Mean vertical intensity across ROI width; higher means stronger band signal."""
    roi = roi.clipped(gray)
    crop = gray[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
    dark_bands = _resolve_dark_bands(gray, background_mode)
    signal = (255.0 - crop.astype(float)) if dark_bands else crop.astype(float)
    return signal.mean(axis=0), dark_bands

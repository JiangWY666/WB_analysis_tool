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
    # 多膜拼贴图的真峰（合并越激进正确率越低）；肩峰问题由
    # separators_from_profile 的等宽网格路径结构性规避。
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


def _estimate_lane_count(
    smooth: np.ndarray,
    *,
    ac_min: float = 0.25,
    sub_min: float = 0.10,
    max_lanes: int = 24,
) -> int | None:
    """用剖面自相关的主周期估计泳道数（等宽先验）；周期性不足返回 None。

    强弱交替的泳道布局会让自相关主峰落在 2×真实间距上（强带与强带对齐）；
    只要半周期处也存在幅度 ≥ ``sub_min`` 的局部峰（说明半间距上真有条带），
    就取更细的周期。
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
    lanes = int(round(size / float(lag)))
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

    - ``expected_lanes`` 已知（≥2）：等宽网格初始化 + 谷位吸附，完全不依赖
      峰检测，强带的肩部假峰不会再让分隔线切进条带。
    - 未知（自动）：先用剖面自相关估计泳道间距（等宽先验），拟合等宽网格
      并吸附谷位；周期性不足或分隔线落在过高的剖面位置时，回退到原有的
      峰-谷逻辑（候选峰带肩部假峰合并）。
    """
    profile = np.asarray(profile, dtype=float)
    n_target = int(expected_lanes) if expected_lanes else 0
    if n_target == 1:
        return []
    if n_target >= 2:
        if profile.size < max(8, n_target * 2):
            return default_separators(roi, n_target)
        smooth = _smooth_profile(profile, max(2, profile.size // 80))
        positions = _grid_valley_positions(smooth, n_target)
        return [float(roi.x + p) for p in positions]
    if profile.size < 8:
        return []
    smooth = _smooth_profile(profile, max(2, profile.size // 80))
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

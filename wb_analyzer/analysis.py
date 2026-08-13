"""Image processing pipeline for automatic Western Blot quantification."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

BackgroundMode = Literal["auto", "dark_bands", "light_bands"]


@dataclass(frozen=True)
class AnalysisConfig:
    """Tunable detection settings. Lower sensitivity detects weaker bands."""

    sensitivity: float = 1.0
    kernel_size: int | None = None
    background_mode: BackgroundMode = "auto"
    saturate_level: int = 250

    def normalized(self) -> AnalysisConfig:
        sensitivity = float(np.clip(self.sensitivity, 0.35, 2.5))
        kernel_size = self.kernel_size
        if kernel_size is not None:
            kernel_size = max(3, int(kernel_size) | 1)
        mode = self.background_mode if self.background_mode in {"auto", "dark_bands", "light_bands"} else "auto"
        saturate_level = int(np.clip(self.saturate_level, 1, 255))
        return replace(self, sensitivity=sensitivity, kernel_size=kernel_size, background_mode=mode, saturate_level=saturate_level)


@dataclass(frozen=True)
class BandMeasurement:
    lane: int
    band: int
    x: int
    y: int
    width: int
    height: int
    area: int
    mean_intensity: float
    background: float
    integrated_density: float
    saturated_fraction: float = 0.0

    detection_limit: float = 0.0
    """3σ 检出限，与 ``integrated_density`` 同单位。

    条带的积分低于这个值时，它与「什么都没有」在统计上无法区分。0 是哨兵值，
    表示这条测量的噪声无法估计（噪声为零的合成图，或没有可用空白区的路径），
    此时不作检出限判定。
    """

    snr: float = 0.0
    """信噪比 = ``integrated_density`` / 积分窗口上的噪声幅度。

    与 :attr:`detection_limit` 同源，因此 ``snr < 3`` 与 :attr:`below_detection`
    等价；单独给出是为了让用户看到「差多远」而不只是「过没过」。
    """

    peak: float = 0.0
    """条带窗口内、扣除背景后的峰值（每像素灰度）。

    ROI 路径取剖面减基线后再除以条带宽度的最大值；局部背景路径取框内
    ``max(signal - background, 0)`` 的最大值。0 表示未测量（占位框、旧数据）。
    """

    @property
    def below_detection(self) -> bool:
        """积分没有越过 3σ 检出限——这个数值不能当成真实信号来用。"""
        return self.detection_limit > 0.0 and self.integrated_density < self.detection_limit

    def as_dict(self) -> dict[str, int | float]:
        data: dict[str, int | float] = asdict(self)
        data["below_detection"] = int(self.below_detection)
        return data


@dataclass(frozen=True)
class AnalysisResult:
    measurements: list[BandMeasurement]
    annotated_image: np.ndarray
    dark_bands: bool
    saturated_band_count: int = 0
    gray_image: np.ndarray | None = None


def _segments(mask: np.ndarray, minimum: int, max_gap: int = 2) -> list[tuple[int, int]]:
    """Return inclusive/exclusive runs of true values, merging tiny gaps."""
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    splits = np.flatnonzero(np.diff(indices) > max_gap) + 1
    groups = np.split(indices, splits)
    return [(int(g[0]), int(g[-1] + 1)) for g in groups if len(g) >= minimum]


def _adaptive_threshold(scores: np.ndarray, percentile: float, std_k: float, sensitivity: float) -> float:
    """Build a threshold that stays below the peak so strong lanes are never fully excluded."""
    if scores.size == 0:
        return 0.0
    mean = float(scores.mean())
    std = float(scores.std())
    peak = float(scores.max())
    if peak <= mean:
        return peak + 1.0  # no usable signal
    baseline = max(float(np.percentile(scores, percentile)), mean + std_k * std)
    # sensitivity↑ → stricter; keep threshold under the peak with a small margin
    thr = mean + max(0.0, baseline - mean) * sensitivity
    thr = min(thr, peak - max(1e-3, 0.02 * (peak - mean)))
    return thr


def _resolve_dark_bands(gray: np.ndarray, mode: BackgroundMode) -> bool:
    if mode == "dark_bands":
        return True
    if mode == "light_bands":
        return False
    # Use plate/border tone: bright film → dark bands; dark display → light bands.
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    border_med = float(np.median(border))
    if border_med >= 140:
        return True
    if border_med <= 115:
        return False
    return float(np.mean(gray)) >= 127.5


def _auto_kernel_size(gray: np.ndarray) -> int:
    # Prefer a kernel larger than typical band thickness but smaller than lane spacing.
    height, width = gray.shape
    guess = min(height, width) // 10
    return max(11, min(guess, 51) | 1)


def _local_background(gray: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    """Median of a ring around the band box.

    The band region is masked with NaN rather than 0: a ``> 0`` filter would
    also throw away genuinely black background, which on a dark-background
    chemiluminescence frame leaves only the bright specks behind and pushes
    the background estimate far too high.
    """
    pad_x = max(2, (x2 - x1) // 4)
    pad_y = max(2, (y2 - y1) // 2)
    left, top = max(0, x1 - pad_x), max(0, y1 - pad_y)
    right, bottom = min(gray.shape[1], x2 + pad_x), min(gray.shape[0], y2 + pad_y)
    ring = gray[top:bottom, left:right].astype(np.float64)
    ring[y1 - top : y2 - top, x1 - left : x2 - left] = np.nan
    values = ring[~np.isnan(ring)]
    return float(np.median(values)) if values.size else float(np.median(gray))


def measure_roi(
    gray: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    dark_bands: bool,
    lane: int = 0,
    band: int = 0,
    saturate_level: int = 250,
) -> BandMeasurement:
    """Quantify one band rectangle on a grayscale blot image."""
    h, w = gray.shape[:2]
    x1 = int(np.clip(x, 0, max(0, w - 1)))
    y1 = int(np.clip(y, 0, max(0, h - 1)))
    x2 = int(np.clip(x1 + max(1, width), x1 + 1, w))
    y2 = int(np.clip(y1 + max(1, height), y1 + 1, h))
    roi = gray[y1:y2, x1:x2]
    local_bg = _local_background(gray, x1, y1, x2, y2)
    roi_f = roi.astype(float)
    if dark_bands:
        integrated = float(np.maximum(local_bg - roi_f, 0).sum())
        saturated_fraction = float(np.mean(roi <= (255 - saturate_level)))
    else:
        integrated = float(np.maximum(roi_f - local_bg, 0).sum())
        saturated_fraction = float(np.mean(roi >= saturate_level))
    residue = np.maximum((local_bg - roi_f) if dark_bands else (roi_f - local_bg), 0.0)
    peak = round(float(residue.max()), 3) if residue.size else 0.0
    return BandMeasurement(
        lane=lane,
        band=band,
        x=x1,
        y=y1,
        width=x2 - x1,
        height=y2 - y1,
        area=int(roi.size),
        mean_intensity=round(float(roi.mean()), 3),
        background=round(local_bg, 3),
        integrated_density=round(integrated, 3),
        saturated_fraction=round(saturated_fraction, 4),
        peak=peak,
    )


def renumber_measurements(measurements: list[BandMeasurement]) -> list[BandMeasurement]:
    """Assign lane/band indices from left-to-right, top-to-bottom geometry."""
    if not measurements:
        return []
    ordered = sorted(measurements, key=lambda m: (m.x + m.width / 2.0, m.y + m.height / 2.0))
    centers = [m.x + m.width / 2.0 for m in ordered]
    widths = [m.width for m in ordered]
    merge_tol = max(8.0, float(np.median(widths)) * 0.55)
    lane_ids: list[int] = []
    lane_centers: list[float] = []
    for center in centers:
        assigned = None
        for idx, lane_center in enumerate(lane_centers):
            if abs(center - lane_center) <= merge_tol:
                assigned = idx
                lane_centers[idx] = (lane_centers[idx] + center) / 2.0
                break
        if assigned is None:
            assigned = len(lane_centers)
            lane_centers.append(center)
        lane_ids.append(assigned + 1)

    by_lane: dict[int, list[BandMeasurement]] = {}
    for item, lane in zip(ordered, lane_ids, strict=True):
        by_lane.setdefault(lane, []).append(item)

    renumbered: list[BandMeasurement] = []
    for lane in sorted(by_lane):
        bands = sorted(by_lane[lane], key=lambda m: m.y)
        for band_number, item in enumerate(bands, start=1):
            renumbered.append(replace(item, lane=lane, band=band_number))
    return renumbered


def annotate_measurements(gray: np.ndarray, measurements: list[BandMeasurement]) -> np.ndarray:
    annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for item in measurements:
        x2 = item.x + item.width - 1
        y2 = item.y + item.height - 1
        color = (0, 0, 255) if item.saturated_fraction < 0.05 else (0, 140, 255)
        cv2.rectangle(annotated, (item.x, item.y), (x2, y2), color, 1)
        cv2.putText(
            annotated,
            f"L{item.lane}-B{item.band}",
            (item.x, max(12, item.y - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def rebuild_result(
    gray: np.ndarray,
    measurements: list[BandMeasurement],
    *,
    dark_bands: bool,
    saturate_level: int = 250,
) -> AnalysisResult:
    remasured = [
        measure_roi(
            gray,
            m.x,
            m.y,
            m.width,
            m.height,
            dark_bands=dark_bands,
            saturate_level=saturate_level,
        )
        for m in measurements
    ]
    numbered = renumber_measurements(remasured)
    saturated = sum(1 for m in numbered if m.saturated_fraction >= 0.05)
    return AnalysisResult(
        measurements=numbered,
        annotated_image=annotate_measurements(gray, numbered),
        dark_bands=dark_bands,
        saturated_band_count=saturated,
        gray_image=gray.copy(),
    )


def _smooth_1d(values: np.ndarray, radius: int) -> np.ndarray:
    """Moving average with edge padding.

    Zero padding (``mode="same"`` without a pad) attenuates the first and last
    ``radius`` samples by up to 50%, which hides bands sitting against the
    edge of a crop from the projection detector.
    """
    radius = max(1, int(radius))
    kernel = np.ones(radius * 2 + 1, dtype=float)
    kernel /= kernel.sum()
    padded = np.pad(np.asarray(values, dtype=float), radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _content_bbox(gray: np.ndarray, dark_bands: bool) -> tuple[int, int, int, int]:
    """Largest membrane-like region, ignoring huge empty margins."""
    h, w = gray.shape
    mask = ((gray < 248) if dark_bands else (gray > 8)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return 0, 0, w, h
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, area = (int(v) for v in stats[index])
    if area < 0.01 * h * w:
        return 0, 0, w, h
    pad = 6
    return max(0, x - pad), max(0, y - pad), min(w, x + bw + pad), min(h, y + bh + pad)


def _prepare_signal(gray: np.ndarray, dark_bands: bool, kernel_size: int) -> np.ndarray:
    signal = 255 - gray if dark_bands else gray
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(signal)
    background = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size), np.uint8))
    corrected = cv2.subtract(enhanced, background)
    return cv2.GaussianBlur(corrected, (5, 5), 0)


def _split_bimodal_box(corr: np.ndarray, box: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    """Split a wide box if its column profile has two clear peaks."""
    x, y, w, h = box
    if w < 55:
        return [box]
    strip = corr[y : y + h, x : x + w].astype(float)
    if strip.size == 0:
        return [box]
    profile = _smooth_1d(strip.mean(axis=0), 3)
    peak_floor = float(profile.max()) * 0.35
    candidates = [
        i
        for i in range(3, len(profile) - 3)
        if profile[i] >= max(profile[i - 3 : i + 4]) and profile[i] >= peak_floor
    ]
    candidates = sorted(candidates, key=lambda i: profile[i], reverse=True)
    peaks: list[int] = []
    for idx in candidates:
        if all(abs(idx - other) >= max(12, w // 6) for other in peaks):
            peaks.append(idx)
        if len(peaks) == 2:
            break
    if len(peaks) < 2:
        return [box]
    left_peak, right_peak = sorted(peaks[:2])
    valley = left_peak + int(np.argmin(profile[left_peak : right_peak + 1]))
    if valley <= 6 or valley >= w - 6:
        return [box]
    return [(x, y, valley, h), (x + valley, y, w - valley, h)]


def _score_band_boxes(
    boxes: list[tuple[int, int, int, int]],
    height: int,
    width: int,
    corrected: np.ndarray | None = None,
) -> float:
    """Higher is better: prefer a plausible number of similarly sized band boxes."""
    count = len(boxes)
    if count == 0:
        return -1.0
    if count > 28:
        return 0.0
    widths = np.array([box[2] for box in boxes], dtype=float)
    heights = np.array([box[3] for box in boxes], dtype=float)
    med_w = float(np.median(widths))
    med_h = float(np.median(heights))

    if 4 <= count <= 14:
        score = 12.0
    elif 2 <= count <= 20:
        score = 6.0
    else:
        score = 1.0
    # Mild bias against missing bands, without rewarding noise splinters.
    score += min(count, 12) * 0.35

    if med_w > 0:
        score -= float(widths.std() / med_w) * 22.0
        score -= float(np.sum(widths < 0.45 * med_w)) * 7.0
        score -= float(np.sum(widths > 1.85 * med_w)) * 5.0
    if med_h > 0:
        score -= float(heights.std() / med_h) * 8.0

    # Same-row and density-median bonuses are meaningless for a single box (a
    # lone membrane-sized blob would trivially collect them), so gate on >= 2.
    centers_y = [y + h / 2.0 for _, y, _, h in boxes]
    if count >= 2 and max(centers_y) - min(centers_y) < 0.4 * height:
        score += 6.0
    # Noise blobs often cling to the top margin on uneven gradients.
    score -= sum(1 for _, y, _, h in boxes if y <= 2 and h < 0.22 * height) * 5.0

    if corrected is not None and count >= 2:
        dens = np.array(
            [float(corrected[y : y + h, x : x + w].sum()) for x, y, w, h in boxes],
            dtype=float,
        )
        ref = float(np.percentile(dens, 75))
        if ref > 0:
            score += float(np.median(dens)) / max(ref, 1.0) * 4.0
            # Real strong/weak pairs reach ~10x ratios; only punish boxes far
            # weaker than that (noise splinters), not genuine faint bands.
            score -= float(np.sum(dens < 0.08 * ref)) * 6.0

    if all(w < 0.4 * width for _, _, w, _ in boxes):
        score += 3.0
    if all(h < 0.65 * height for _, _, _, h in boxes):
        score += 2.0
    return score


def _filter_weak_component_boxes(
    boxes: list[tuple[int, int, int, int]],
    corrected: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """Drop low-signal fragments relative to the strongest components."""
    if len(boxes) < 2:
        return boxes
    dens = np.array(
        [float(corrected[y : y + h, x : x + w].sum()) for x, y, w, h in boxes],
        dtype=float,
    )
    ref = float(np.percentile(dens, 75))
    if ref <= 1e-6:
        return boxes
    # Keep genuine faint bands (strong/weak pairs reach ~10x); drop only
    # fragments far below the strong boxes.
    kept = [box for box, value in zip(boxes, dens, strict=True) if value >= 0.08 * ref]
    return kept or boxes


def _components_at_threshold(
    corrected: np.ndarray,
    threshold: float,
    *,
    min_area: float,
) -> list[tuple[int, int, int, int]]:
    binary = (corrected >= threshold).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 2), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    h, w = corrected.shape
    boxes: list[tuple[int, int, int, int]] = []
    for index in range(1, count):
        x, y, bw, bh, area = (int(v) for v in stats[index])
        if area < min_area or bw < 7 or bh < 3:
            continue
        if bw > 0.45 * w or bh > 0.7 * h:
            if bw <= 0.58 * w and bh <= 0.78 * h and area >= min_area:
                boxes.append((x, y, bw, bh))
            continue
        boxes.append((x, y, bw, bh))
    split: list[tuple[int, int, int, int]] = []
    for box in boxes:
        split.extend(_split_bimodal_box(corrected, box))
    return split


def _boxes_from_membrane_components(
    gray: np.ndarray,
    *,
    dark_bands: bool,
    sensitivity: float,
    kernel_size: int | None,
) -> list[tuple[int, int, int, int]]:
    """Detect bands as connected components inside the membrane crop."""
    x1, y1, x2, y2 = _content_bbox(gray, dark_bands)
    crop_gray = gray[y1:y2, x1:x2]
    if min(crop_gray.shape) < 16:
        return []
    local_kernel = kernel_size or max(15, (min(crop_gray.shape) // 5) | 1)
    corrected = _prepare_signal(crop_gray, dark_bands, local_kernel)
    h, w = corrected.shape
    min_area = max(28.0, 0.00055 * h * w / max(sensitivity, 0.5))
    base_pct = float(np.clip(90.0 + 3.0 * (sensitivity - 1.0), 84.0, 96.0))
    otsu_thr, _ = cv2.threshold(corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    peak_level = float(corrected.max())
    thresholds = [
        max(float(np.percentile(corrected, base_pct - 3.0)), 2.0),
        max(float(np.percentile(corrected, base_pct)), 2.0),
        max(float(np.percentile(corrected, min(base_pct + 2.0, 97.0))), 2.0),
        max(float(otsu_thr) * (0.45 + 0.25 * sensitivity), 2.0),
        # Peak-relative rungs: percentile/Otsu thresholds collapse to the noise
        # floor (or jump past the band range) on mostly-empty membranes, so the
        # correct mid-range candidates would never be generated without these.
        max(peak_level * 0.10, 2.0),
        max(peak_level * 0.18, 2.0),
        max(peak_level * 0.30, 2.0),
    ]

    best_boxes: list[tuple[int, int, int, int]] = []
    best_score = -1e9
    # Prefer stricter thresholds on ties so weak-threshold over-splits lose.
    for thr in sorted(set(thresholds), reverse=True):
        boxes = _filter_weak_component_boxes(
            _components_at_threshold(corrected, thr, min_area=min_area),
            corrected,
        )
        if not boxes:
            # An empty candidate must never outrank a real (if penalized)
            # detection; small membranes easily score below the old -1.0.
            continue
        score = _score_band_boxes(boxes, h, w, corrected)
        if score >= best_score:
            best_score = score
            best_boxes = boxes

    best_boxes = _filter_weak_component_boxes(best_boxes, corrected)

    pad_x, pad_y = 2, 3
    absolute: list[tuple[int, int, int, int]] = []
    for x, y, bw, bh in best_boxes:
        ax = max(0, x1 + x - pad_x)
        ay = max(0, y1 + y - pad_y)
        aw = min(gray.shape[1], x1 + x + bw + pad_x) - ax
        ah = min(gray.shape[0], y1 + y + bh + pad_y) - ay
        if aw >= 6 and ah >= 4:
            absolute.append((ax, ay, aw, ah))
    return sorted(absolute, key=lambda b: (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0))


def _projection_regions(scores: np.ndarray, sensitivity: float, min_width: int, max_gap: int) -> list[tuple[int, int]]:
    """Find signal runs from a 1D projection using a peak-relative threshold."""
    if scores.size == 0:
        return []
    smooth = _smooth_1d(scores, max(1, min_width // 3))
    baseline = float(np.percentile(smooth, 35))
    peak = float(smooth.max())
    if peak <= baseline + 1e-3:
        return []
    rel = float(np.clip(0.18 + 0.28 * sensitivity, 0.12, 0.75))
    thr = baseline + (peak - baseline) * rel
    thr = min(thr, peak - max(1e-3, 0.01 * (peak - baseline)))
    return _segments(smooth >= thr, minimum=min_width, max_gap=max_gap)


def _boxes_from_projection(corrected: np.ndarray, sensitivity: float) -> list[tuple[int, int, int, int]]:
    h, w = corrected.shape
    min_lane = max(4, w // 80)
    lanes = _projection_regions(corrected.mean(axis=0), sensitivity, min_lane, max_gap=max(3, w // 60))
    boxes: list[tuple[int, int, int, int]] = []
    for raw_x1, raw_x2 in lanes:
        pad = max(1, (raw_x2 - raw_x1) // 8)
        x1, x2 = max(0, raw_x1 - pad), min(w, raw_x2 + pad)
        min_band = max(2, h // 120)
        bands = _projection_regions(corrected[:, x1:x2].mean(axis=1), sensitivity, min_band, max_gap=max(2, h // 80))
        for raw_y1, raw_y2 in bands:
            y_pad = max(1, (raw_y2 - raw_y1) // 3)
            y1, y2 = max(0, raw_y1 - y_pad), min(h, raw_y2 + y_pad)
            if (y2 - y1) > 0.55 * h:
                continue
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes


def _boxes_from_contours(corrected: np.ndarray, sensitivity: float) -> list[tuple[int, int, int, int]]:
    h, w = corrected.shape
    otsu_thr, _ = cv2.threshold(corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = float(np.clip(otsu_thr * (0.65 + 0.35 * sensitivity), 1, 254))
    _, binary = cv2.threshold(corrected, thr, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(20.0, (h * w) * (0.00025 / max(sensitivity, 0.35)))
    max_area = (h * w) * 0.25
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = float(bw * bh)
        if area < min_area or area > max_area:
            continue
        if bw < 4 or bh < 2:
            continue
        if bw > 0.8 * w and bh > 0.8 * h:
            continue
        aspect = bw / max(bh, 1)
        if aspect < 0.6 and bh > 0.25 * h:
            continue
        if bh > 0.5 * h:
            continue
        boxes.append((x, y, bw, bh))
    return boxes


def _merge_boxes(boxes: list[tuple[int, int, int, int]], iou_thr: float = 0.35) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept: list[tuple[int, int, int, int]] = []

    def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax2, ay2 = a[0] + a[2], a[1] + a[3]
        bx2, by2 = b[0] + b[2], b[1] + b[3]
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = a[2] * a[3] + b[2] * b[3] - inter
        return inter / max(union, 1)

    for box in boxes:
        if any(iou(box, other) >= iou_thr for other in kept):
            continue
        kept.append(box)
    return kept


def analyze_array(image: np.ndarray, config: AnalysisConfig | None = None) -> AnalysisResult:
    """Analyze a BGR or grayscale image and return detected band measurements."""
    cfg = (config or AnalysisConfig()).normalized()
    if image is None or image.size == 0:
        raise ValueError("图片为空，无法分析")
    if image.dtype != np.uint8:
        raise ValueError(
            f"analyze_array 只接受 8 位图像（收到 {image.dtype}）。"
            "原生位深图片请用 imageio.load_blot() 读取后传入 BlotImage.display_bgr，"
            "再用 quant.remeasure_native() 在原始位深上复测——analyze_image() 已按这个流程实现。"
        )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    if min(gray.shape) < 20:
        raise ValueError("图片尺寸过小（宽和高至少为 20 像素）")

    dark_bands = _resolve_dark_bands(gray, cfg.background_mode)
    kernel_size = cfg.kernel_size or _auto_kernel_size(gray)

    # Primary path: membrane crop + connected components (works for sparse chemiluminescence).
    boxes = _boxes_from_membrane_components(
        gray,
        dark_bands=dark_bands,
        sensitivity=cfg.sensitivity,
        kernel_size=cfg.kernel_size,
    )

    # Fallback for dense/full-frame blots.
    if len(boxes) < 2:
        corrected = _prepare_signal(gray, dark_bands, kernel_size)
        proj_boxes = _boxes_from_projection(corrected, cfg.sensitivity)
        contour_boxes = _boxes_from_contours(corrected, cfg.sensitivity)
        boxes = _merge_boxes(proj_boxes + contour_boxes if len(proj_boxes) >= 2 else contour_boxes + proj_boxes)

    placeholders = [
        BandMeasurement(
            lane=0,
            band=0,
            x=x,
            y=y,
            width=width,
            height=height,
            area=0,
            mean_intensity=0.0,
            background=0.0,
            integrated_density=0.0,
        )
        for x, y, width, height in boxes
    ]
    return rebuild_result(gray, placeholders, dark_bands=dark_bands, saturate_level=cfg.saturate_level)


def analyze_image(path: str | Path, config: AnalysisConfig | None = None) -> AnalysisResult:
    """Read and analyze an image at its native bit depth.

    Band *detection* runs on the 8-bit display layer (CLAHE, Otsu and the
    morphology below all require uint8), but every reported number is then
    re-measured on the untouched ``raw`` plane. This is the same two-stage
    pipeline the GUI uses; going through ``cv2.IMREAD_COLOR`` instead would
    collapse a 16-bit exposure to 256 grey levels before anything is measured.
    """
    from .imageio import load_blot
    from .quant import remeasure_native

    blot = load_blot(path)
    result = analyze_array(blot.display_bgr, config=config)
    if not result.measurements:
        return result
    return remeasure_native(
        result.measurements,
        blot.raw,
        dark_bands=result.dark_bands,
        white_level=blot.white_level,
        display_gray=blot.display_gray,
    )

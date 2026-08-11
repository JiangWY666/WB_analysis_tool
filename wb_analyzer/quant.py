"""ImageJ-style lane-profile quantification on native bit-depth data.

The workflow mirrors the classic Gel Analyzer: each lane strip is collapsed
into a vertical intensity profile, a baseline (rubber-band lower convex hull
or valley-to-valley) is subtracted, and the area under each remaining peak is
the band's integrated density. All math runs on the raw (possibly 16-bit)
data; the 8-bit display layer is only used for annotation.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .analysis import AnalysisResult, BandMeasurement, renumber_measurements
from .imageio import saturation_fraction
from .roi import RoiRect, lane_boundaries

ROI_COLOR = (255, 229, 0)  # cyan-ish (BGR)
SEPARATOR_COLOR = (0, 180, 255)  # orange (BGR)
BAND_COLOR = (60, 60, 255)  # red (BGR)
BAND_SATURATED_COLOR = (0, 140, 255)  # orange warning (BGR)

SEGMENT_EDGE_FRAC = 0.02
"""Band segment edges are trimmed where the signal falls to ~2% of the peak."""


@dataclass(frozen=True)
class LaneProfile:
    """Per-lane vertical profile, baseline, and band segmentation."""

    lane: int
    """1-based lane number, left to right."""

    x1: int
    x2: int
    """Absolute pixel span actually integrated: [x1, x2).

    With ``tight=True``（默认）这是收紧后的条带横向足迹——泳道条内两侧的
    空白不参与积分；找不到可靠条带峰（空白/平坦泳道）或 ``tight=False``
    时退回整个泳道条 [分隔线, 分隔线)。
    """

    y0: int
    """ROI top; profile[i] corresponds to image row y0 + i."""

    profile: np.ndarray
    """Signal profile along y (sum across strip width per row)."""

    baseline: np.ndarray
    segments: list[tuple[int, int]]
    """Relative [start, end) y ranges, one per band."""

    total_density: float
    """sum(max(profile - baseline, 0)) over the whole lane."""

    band_densities: list[float]
    """Integral of each segment, aligned with ``segments``."""


def _smooth(values: np.ndarray, radius: int) -> np.ndarray:
    """Moving average with edge padding (no zero-pad droop at the ends)."""
    radius = max(1, int(radius))
    kernel = np.ones(radius * 2 + 1, dtype=float)
    kernel /= kernel.sum()
    padded = np.pad(values.astype(float), radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def resolve_dark_bands_native(
    raw: np.ndarray, white_level: float, mode: str = "auto"
) -> bool:
    """Decide band polarity on native-depth data (scaled 8-bit heuristics)."""
    if mode == "dark_bands":
        return True
    if mode == "light_bands":
        return False
    border = np.concatenate((raw[0], raw[-1], raw[:, 0], raw[:, -1]))
    border_med = float(np.median(border))
    if border_med >= white_level * (140.0 / 255.0):
        return True
    if border_med <= white_level * (115.0 / 255.0):
        return False
    return float(np.mean(raw)) >= white_level / 2.0


def rubber_band_baseline(profile: np.ndarray) -> np.ndarray:
    """Lower convex hull baseline (rubber band stretched under the profile)."""
    n = int(profile.size)
    if n == 0:
        return np.zeros(0, dtype=float)
    if n < 3:
        return profile.astype(float).copy()
    smooth = _smooth(profile, max(1, n // 100))
    hull: list[int] = []
    for i in range(n):
        while len(hull) >= 2:
            i0, i1 = hull[-2], hull[-1]
            # Keep only points forming a convex (upward-opening) lower chain.
            cross = (i1 - i0) * (smooth[i] - smooth[i0]) - (i - i0) * (smooth[i1] - smooth[i0])
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return np.interp(np.arange(n), hull, smooth[hull])


def valley_baseline(profile: np.ndarray) -> np.ndarray:
    """Valley-to-valley linear baseline (local minima joined by straight lines)."""
    n = int(profile.size)
    if n == 0:
        return np.zeros(0, dtype=float)
    if n < 5:
        return np.full(n, float(np.min(profile)))
    smooth = _smooth(profile, max(1, n // 100))
    win = max(3, n // 40)
    anchors: list[int] = [0]
    index = win
    while index < n - win:
        neighborhood = smooth[index - win : index + win + 1]
        if smooth[index] <= float(neighborhood.min()) + 1e-12:
            anchors.append(index)
            index += win
            continue
        index += 1
    anchors.append(n - 1)
    anchors = sorted(set(anchors))
    return np.interp(np.arange(n), anchors, smooth[anchors])


def _baseline_for(profile: np.ndarray, baseline_mode: str) -> np.ndarray:
    if baseline_mode == "rubber":
        return rubber_band_baseline(profile)
    if baseline_mode == "valley":
        return valley_baseline(profile)
    if baseline_mode == "none":
        return np.zeros(int(profile.size), dtype=float)
    raise ValueError(f"未知的基线模式：{baseline_mode}（可选 rubber / valley / none）")


def _local_maxima(values: np.ndarray, min_height: float, min_distance: int) -> list[int]:
    peaks: list[int] = []
    n = len(values)
    win = max(1, min_distance // 2)
    index = win
    while index < n - win:
        neighborhood = values[index - win : index + win + 1]
        if values[index] >= float(neighborhood.max()) - 1e-12 and values[index] >= min_height:
            # Center on plateaus.
            left = index
            while left > 0 and abs(values[left - 1] - values[index]) <= 1e-12:
                left -= 1
            right = index
            while right < n - 1 and abs(values[right + 1] - values[index]) <= 1e-12:
                right += 1
            peaks.append((left + right) // 2)
            index = right + win
            continue
        index += 1
    filtered: list[int] = []
    for peak in sorted(peaks, key=lambda i: values[i], reverse=True):
        if all(abs(peak - other) >= min_distance for other in filtered):
            filtered.append(peak)
    return sorted(filtered)


def _trim_segment(
    smooth: np.ndarray, start: int, stop: int, edge_frac: float = SEGMENT_EDGE_FRAC
) -> tuple[int, int]:
    """Shrink a segment to where the signal exceeds ``edge_frac`` of its peak."""
    window = smooth[start:stop]
    if window.size == 0:
        return start, stop
    floor = float(window.max()) * edge_frac
    above = np.flatnonzero(window > floor)
    if above.size == 0:
        return start, stop
    return start + int(above[0]), start + int(above[-1]) + 1


def segment_bands(
    corrected: np.ndarray,
    *,
    min_height_frac: float = 0.04,
    min_width: int = 3,
    valley_drop: float = 0.30,
) -> list[tuple[int, int]]:
    """Split a baseline-corrected lane profile into band segments.

    Returns half-open (start, stop) index ranges. Runs above a small relative
    threshold form candidates; runs containing two clearly separated peaks
    (valley dips at least ``valley_drop`` below the smaller peak) are split.
    Segment edges are trimmed where the signal falls to ~2% of the local peak.
    """
    n = int(corrected.size)
    if n == 0:
        return []
    smooth = _smooth(corrected, max(1, n // 120))
    peak_value = float(smooth.max())
    if peak_value <= 1e-9:
        return []
    threshold = peak_value * float(min_height_frac)
    mask = smooth > threshold
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start >= min_width:
                runs.append((start, index))
            start = None
    if start is not None and n - start >= min_width:
        runs.append((start, n))

    segments: list[tuple[int, int]] = []
    for run_start, run_stop in runs:
        window = smooth[run_start:run_stop]
        min_dist = max(3, (run_stop - run_start) // 8, n // 60)
        peaks = _local_maxima(window, threshold, min_dist)
        if len(peaks) <= 1:
            segments.append(_trim_segment(smooth, run_start, run_stop))
            continue
        # Split at valleys that dip clearly below both adjacent peaks.
        cuts: list[int] = []
        for left, right in zip(peaks, peaks[1:], strict=False):
            valley = left + int(np.argmin(window[left : right + 1]))
            smaller = min(window[left], window[right])
            if window[valley] <= smaller * (1.0 - valley_drop):
                cuts.append(valley)
        bounds = [run_start, *[run_start + c for c in cuts], run_stop]
        for seg_start, seg_stop in zip(bounds, bounds[1:], strict=False):
            if seg_stop - seg_start >= min_width:
                segments.append(_trim_segment(smooth, seg_start, seg_stop))
    return [(s, e) for s, e in segments if e - s >= min_width]


TIGHT_EDGE_FRAC = 0.08
"""条带足迹边缘阈值：列信号跌破峰高的 8% 处截断（5%~10% 区间的中值）。"""

TIGHT_MIN_WIDTH = 5
"""收紧后条带足迹的最小宽度（像素）。"""


def _tight_band_span(
    signal: np.ndarray,
    *,
    edge_frac: float = TIGHT_EDGE_FRAC,
    min_width: int = TIGHT_MIN_WIDTH,
) -> tuple[int, int] | None:
    """泳道条内条带的实际横向足迹 [x1, x2)（相对本条），不可靠时返回 None。

    以泳道条列剖面（每列信号均值）的最强列为锚，向两侧扩展到信号跌破
    ``floor + edge_frac × 峰高`` 为止。峰高相对本条列噪声不显著（无可靠
    条带，例如空白泳道）时返回 None，调用方回退全宽，保持旧行为。
    """
    width = int(signal.shape[1])
    if width <= min_width:
        return None
    column = signal.mean(axis=0)
    smooth = _smooth(column, max(1, width // 40))
    floor = float(np.percentile(smooth, 5))
    peak = int(np.argmax(smooth))
    height = float(smooth[peak]) - floor
    residual = column - smooth
    sigma = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    if height <= max(4.0 * sigma, 1e-9):
        return None
    threshold = floor + height * float(edge_frac)
    left = peak
    while left > 0 and smooth[left - 1] > threshold:
        left -= 1
    right = peak
    while right < width - 1 and smooth[right + 1] > threshold:
        right += 1
    x1, x2 = left, right + 1
    if x2 - x1 >= width - 2:
        return None  # 已接近全宽，收紧无意义
    if x2 - x1 < min_width:
        pad = (min_width - (x2 - x1) + 1) // 2
        x1 = max(0, x1 - pad)
        x2 = min(width, x2 + pad)
    return x1, x2


def compute_lane_profiles(
    raw: np.ndarray,
    roi: RoiRect,
    separators: list[float],
    *,
    dark_bands: bool,
    white_level: float = 255.0,
    baseline_mode: str = "rubber",
    split_bands: bool = True,
    tight: bool = True,
) -> list[LaneProfile]:
    """Profile, baseline, and band segmentation for every ROI lane.

    ``tight=True``（默认）时先在每个泳道条内定位条带的实际横向足迹，
    纵向剖面、基线与积分只在该足迹内进行——条带两侧的空白不再稀释
    积分值；找不到可靠条带的泳道回退全宽。``tight=False`` 保持旧的
    全宽行为。
    """
    roi = roi.clipped(raw)
    if roi.width < 3 or roi.height < 3:
        raise ValueError("ROI 过小，无法定量")
    y1, y2 = roi.y, roi.y + roi.height
    lanes: list[LaneProfile] = []
    for lane_index, (x1, x2) in enumerate(lane_boundaries(roi, separators), start=1):
        strip = raw[y1:y2, x1:x2].astype(np.float64)
        signal = (white_level - strip) if dark_bands else strip
        if tight:
            span = _tight_band_span(signal)
            if span is not None:
                x1, x2 = x1 + span[0], x1 + span[1]
                signal = signal[:, span[0] : span[1]]
        profile = signal.sum(axis=1)
        baseline = _baseline_for(profile, baseline_mode)
        corrected = np.maximum(profile - baseline, 0.0)
        total = float(corrected.sum())
        segments = segment_bands(corrected) if split_bands else [(0, int(profile.size))]
        if not segments:
            # No detectable peak: fall back to one whole-lane record so the
            # lane still lines up across proteins during normalization.
            segments = [(0, int(profile.size))]
        densities = [float(corrected[s:e].sum()) for s, e in segments]
        lanes.append(
            LaneProfile(
                lane=lane_index,
                x1=x1,
                x2=x2,
                y0=y1,
                profile=profile,
                baseline=baseline,
                segments=segments,
                total_density=total,
                band_densities=densities,
            )
        )
    return lanes


def _display_bgr_from(raw: np.ndarray, white_level: float, display_gray: np.ndarray | None) -> np.ndarray:
    if display_gray is not None:
        if display_gray.ndim == 3:
            return display_gray.copy()
        return cv2.cvtColor(display_gray, cv2.COLOR_GRAY2BGR)
    scale = 255.0 / max(white_level, 1e-9)
    return cv2.cvtColor(np.clip(raw * scale, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def annotate_on_display(
    display_bgr: np.ndarray,
    measurements: list[BandMeasurement],
    *,
    roi: RoiRect | None = None,
    separators: list[float] | None = None,
) -> np.ndarray:
    annotated = display_bgr.copy()
    if annotated.ndim == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
    if roi is not None:
        cv2.rectangle(
            annotated,
            (roi.x, roi.y),
            (roi.x + roi.width - 1, roi.y + roi.height - 1),
            ROI_COLOR,
            1,
            cv2.LINE_AA,
        )
        for sep in separators or []:
            x = int(round(sep))
            if roi.x < x < roi.x + roi.width:
                cv2.line(annotated, (x, roi.y), (x, roi.y + roi.height - 1), SEPARATOR_COLOR, 1, cv2.LINE_AA)
    for item in measurements:
        color = BAND_COLOR if item.saturated_fraction < 0.05 else BAND_SATURATED_COLOR
        cv2.rectangle(
            annotated,
            (item.x, item.y),
            (item.x + item.width - 1, item.y + item.height - 1),
            color,
            1,
            cv2.LINE_AA,
        )
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


def quantify_lanes(
    raw: np.ndarray,
    roi: RoiRect,
    separators: list[float],
    *,
    dark_bands: bool,
    white_level: float = 255.0,
    display_gray: np.ndarray | None = None,
    baseline_mode: str = "rubber",
    split_bands: bool = True,
    tight: bool = True,
) -> AnalysisResult:
    """Quantify ROI lanes via baseline-subtracted vertical profiles.

    Returns a standard :class:`AnalysisResult`; with ``split_bands=True`` each
    detected band becomes one measurement (lanes falling back to a single
    whole-lane record when no peak is found). Use
    :func:`compute_lane_profiles` for the plottable profiles and totals.

    ``tight=True``（默认）把每个泳道的积分与标注框收紧到条带的实际横向
    足迹，泳道内条带两侧的空白不再计入积分；``tight=False`` 保持旧的
    全宽行为。
    """
    roi = roi.clipped(raw)
    lanes = compute_lane_profiles(
        raw,
        roi,
        separators,
        dark_bands=dark_bands,
        white_level=white_level,
        baseline_mode=baseline_mode,
        split_bands=split_bands,
        tight=tight,
    )
    measurements: list[BandMeasurement] = []
    for lane in lanes:
        width = lane.x2 - lane.x1
        for band_index, ((seg_start, seg_stop), density) in enumerate(
            zip(lane.segments, lane.band_densities, strict=True), start=1
        ):
            box_raw = raw[lane.y0 + seg_start : lane.y0 + seg_stop, lane.x1 : lane.x2]
            baseline_per_pixel = float(lane.baseline[seg_start:seg_stop].mean()) / max(width, 1)
            display_bg = (white_level - baseline_per_pixel) if dark_bands else baseline_per_pixel
            measurements.append(
                BandMeasurement(
                    lane=lane.lane,
                    band=band_index,
                    x=lane.x1,
                    y=lane.y0 + seg_start,
                    width=width,
                    height=seg_stop - seg_start,
                    area=int(box_raw.size),
                    mean_intensity=round(float(box_raw.mean()), 3) if box_raw.size else 0.0,
                    background=round(display_bg, 3),
                    integrated_density=round(density, 3),
                    saturated_fraction=round(
                        saturation_fraction(box_raw, dark_bands=dark_bands, white_level=white_level),
                        4,
                    ),
                )
            )

    display_bgr = _display_bgr_from(raw, white_level, display_gray)
    annotated = annotate_on_display(display_bgr, measurements, roi=roi, separators=separators)
    saturated = sum(1 for m in measurements if m.saturated_fraction >= 0.05)
    return AnalysisResult(
        measurements=measurements,
        annotated_image=annotated,
        dark_bands=dark_bands,
        saturated_band_count=saturated,
        gray_image=cv2.cvtColor(display_bgr, cv2.COLOR_BGR2GRAY),
    )


def roi_column_profile(
    raw: np.ndarray,
    roi: RoiRect,
    *,
    dark_bands: bool,
    white_level: float = 255.0,
) -> np.ndarray:
    """Mean signal per column across the ROI (for lane separator placement)."""
    roi = roi.clipped(raw)
    crop = raw[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width].astype(np.float64)
    signal = (white_level - crop) if dark_bands else crop
    return signal.mean(axis=0)


def _local_background_native(raw: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    """Median of a surrounding ring, computed on native-depth data."""
    pad_x = max(2, (x2 - x1) // 4)
    pad_y = max(2, (y2 - y1) // 2)
    left, top = max(0, x1 - pad_x), max(0, y1 - pad_y)
    right, bottom = min(raw.shape[1], x2 + pad_x), min(raw.shape[0], y2 + pad_y)
    ring = raw[top:bottom, left:right].astype(np.float64).copy()
    ring[y1 - top : y2 - top, x1 - left : x2 - left] = np.nan
    values = ring[~np.isnan(ring)]
    return float(np.median(values)) if values.size else float(np.median(raw))


def measure_box_native(
    raw: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    dark_bands: bool,
    white_level: float = 255.0,
    lane: int = 0,
    band: int = 0,
) -> BandMeasurement:
    """Quantify one rectangle with ring-median local background (native depth)."""
    img_h, img_w = raw.shape[:2]
    x1 = int(np.clip(x, 0, max(0, img_w - 1)))
    y1 = int(np.clip(y, 0, max(0, img_h - 1)))
    x2 = int(np.clip(x1 + max(1, w), x1 + 1, img_w))
    y2 = int(np.clip(y1 + max(1, h), y1 + 1, img_h))
    box = raw[y1:y2, x1:x2].astype(np.float64)
    local_bg = _local_background_native(raw, x1, y1, x2, y2)
    if dark_bands:
        integrated = float(np.maximum(local_bg - box, 0.0).sum())
    else:
        integrated = float(np.maximum(box - local_bg, 0.0).sum())
    return BandMeasurement(
        lane=lane,
        band=band,
        x=x1,
        y=y1,
        width=x2 - x1,
        height=y2 - y1,
        area=int(box.size),
        mean_intensity=round(float(box.mean()), 3),
        background=round(local_bg, 3),
        integrated_density=round(integrated, 3),
        saturated_fraction=round(
            saturation_fraction(box, dark_bands=dark_bands, white_level=white_level), 4
        ),
    )


def remeasure_native(
    measurements: list[BandMeasurement],
    raw: np.ndarray,
    *,
    dark_bands: bool,
    white_level: float = 255.0,
    display_gray: np.ndarray | None = None,
) -> AnalysisResult:
    """Re-measure coarse 8-bit detection boxes on native-depth data."""
    remeasured = [
        measure_box_native(
            raw,
            m.x,
            m.y,
            m.width,
            m.height,
            dark_bands=dark_bands,
            white_level=white_level,
        )
        for m in measurements
    ]
    numbered = renumber_measurements(remeasured)
    display_bgr = _display_bgr_from(raw, white_level, display_gray)
    annotated = annotate_on_display(display_bgr, numbered)
    saturated = sum(1 for m in numbered if m.saturated_fraction >= 0.05)
    return AnalysisResult(
        measurements=numbered,
        annotated_image=annotated,
        dark_bands=dark_bands,
        saturated_band_count=saturated,
        gray_image=cv2.cvtColor(display_bgr, cv2.COLOR_BGR2GRAY),
    )

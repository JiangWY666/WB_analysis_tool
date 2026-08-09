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

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


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
    pad_x = max(2, (x2 - x1) // 4)
    pad_y = max(2, (y2 - y1) // 2)
    left, top = max(0, x1 - pad_x), max(0, y1 - pad_y)
    right, bottom = min(gray.shape[1], x2 + pad_x), min(gray.shape[0], y2 + pad_y)
    ring = gray[top:bottom, left:right].copy()
    ring[y1 - top : y2 - top, x1 - left : x2 - left] = 0
    values = ring[ring > 0]
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


def analyze_array(image: np.ndarray, config: AnalysisConfig | None = None) -> AnalysisResult:
    """Analyze a BGR or grayscale image and return detected band measurements."""
    cfg = (config or AnalysisConfig()).normalized()
    if image is None or image.size == 0:
        raise ValueError("图片为空，无法分析")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    if min(gray.shape) < 20:
        raise ValueError("图片尺寸过小（宽和高至少为 20 像素）")

    dark_bands = _resolve_dark_bands(gray, cfg.background_mode)
    signal = 255 - gray if dark_bands else gray

    kernel_size = cfg.kernel_size or _auto_kernel_size(gray)
    background = cv2.morphologyEx(signal, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size), np.uint8))
    corrected = cv2.subtract(signal, background)
    corrected = cv2.GaussianBlur(corrected, (3, 3), 0)

    column_score = corrected.mean(axis=0)
    lane_threshold = _adaptive_threshold(column_score, percentile=55, std_k=0.2, sensitivity=cfg.sensitivity)
    min_lane = max(3, gray.shape[1] // 120)
    lanes = _segments(column_score >= lane_threshold, min_lane, max_gap=max(2, gray.shape[1] // 80))

    boxes: list[tuple[int, int, int, int]] = []
    for raw_x1, raw_x2 in lanes:
        margin = max(1, (raw_x2 - raw_x1) // 5)
        x1, x2 = max(0, raw_x1 - margin), min(gray.shape[1], raw_x2 + margin)
        row_score = corrected[:, x1:x2].mean(axis=1)
        band_threshold = _adaptive_threshold(row_score, percentile=65, std_k=0.45, sensitivity=cfg.sensitivity)
        min_band = max(2, gray.shape[0] // 160)
        bands = _segments(row_score >= band_threshold, min_band, max_gap=max(2, gray.shape[0] // 100))
        for raw_y1, raw_y2 in bands:
            y_margin = max(1, (raw_y2 - raw_y1) // 2)
            y1, y2 = max(0, raw_y1 - y_margin), min(gray.shape[0], raw_y2 + y_margin)
            boxes.append((x1, y1, x2 - x1, y2 - y1))

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
    """Read and analyze an image while supporting non-ASCII Windows paths."""
    path = Path(path)
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"无法读取图片：{path}") from exc
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"不支持或已损坏的图片：{path}")
    return analyze_array(image, config=config)

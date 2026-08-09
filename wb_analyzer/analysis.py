"""Image processing pipeline for automatic Western Blot quantification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


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

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class AnalysisResult:
    measurements: list[BandMeasurement]
    annotated_image: np.ndarray
    dark_bands: bool


def _segments(mask: np.ndarray, minimum: int) -> list[tuple[int, int]]:
    """Return inclusive/exclusive runs of true values, merging tiny gaps."""
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    splits = np.flatnonzero(np.diff(indices) > 2) + 1
    groups = np.split(indices, splits)
    return [(int(g[0]), int(g[-1] + 1)) for g in groups if len(g) >= minimum]


def _local_background(gray: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    pad_x = max(2, (x2 - x1) // 4)
    pad_y = max(2, (y2 - y1) // 2)
    left, top = max(0, x1 - pad_x), max(0, y1 - pad_y)
    right, bottom = min(gray.shape[1], x2 + pad_x), min(gray.shape[0], y2 + pad_y)
    ring = gray[top:bottom, left:right].copy()
    ring[y1 - top : y2 - top, x1 - left : x2 - left] = 0
    values = ring[ring > 0]
    return float(np.median(values)) if values.size else float(np.median(gray))


def analyze_array(image: np.ndarray) -> AnalysisResult:
    """Analyze a BGR or grayscale image and return detected band measurements."""
    if image is None or image.size == 0:
        raise ValueError("图片为空，无法分析")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    if min(gray.shape) < 20:
        raise ValueError("图片尺寸过小（宽和高至少为 20 像素）")

    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    dark_bands = float(np.median(border)) >= float(np.median(gray))
    signal = 255 - gray if dark_bands else gray

    kernel_size = max(15, (min(gray.shape) // 6) | 1)
    background = cv2.morphologyEx(signal, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size), np.uint8))
    corrected = cv2.subtract(signal, background)
    corrected = cv2.GaussianBlur(corrected, (3, 3), 0)

    column_score = corrected.mean(axis=0)
    lane_threshold = max(float(np.percentile(column_score, 60)), float(column_score.mean() + 0.25 * column_score.std()))
    lanes = _segments(column_score > lane_threshold, max(3, gray.shape[1] // 100))

    measurements: list[BandMeasurement] = []
    annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for lane_number, (raw_x1, raw_x2) in enumerate(lanes, start=1):
        margin = max(1, (raw_x2 - raw_x1) // 5)
        x1, x2 = max(0, raw_x1 - margin), min(gray.shape[1], raw_x2 + margin)
        row_score = corrected[:, x1:x2].mean(axis=1)
        threshold = max(float(np.percentile(row_score, 70)), float(row_score.mean() + 0.6 * row_score.std()))
        bands = _segments(row_score > threshold, max(2, gray.shape[0] // 150))
        for band_number, (raw_y1, raw_y2) in enumerate(bands, start=1):
            y_margin = max(1, (raw_y2 - raw_y1) // 2)
            y1, y2 = max(0, raw_y1 - y_margin), min(gray.shape[0], raw_y2 + y_margin)
            roi = gray[y1:y2, x1:x2]
            local_bg = _local_background(gray, x1, y1, x2, y2)
            if dark_bands:
                integrated = float(np.maximum(local_bg - roi.astype(float), 0).sum())
            else:
                integrated = float(np.maximum(roi.astype(float) - local_bg, 0).sum())
            measurements.append(BandMeasurement(
                lane_number, band_number, x1, y1, x2 - x1, y2 - y1,
                int(roi.size), round(float(roi.mean()), 3), round(local_bg, 3), round(integrated, 3),
            ))
            cv2.rectangle(annotated, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 1)
            cv2.putText(annotated, f"L{lane_number}-B{band_number}", (x1, max(12, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1, cv2.LINE_AA)
    return AnalysisResult(measurements, annotated, dark_bands)


def analyze_image(path: str | Path) -> AnalysisResult:
    """Read and analyze an image while supporting non-ASCII Windows paths."""
    path = Path(path)
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"无法读取图片：{path}") from exc
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"不支持或已损坏的图片：{path}")
    return analyze_array(image)

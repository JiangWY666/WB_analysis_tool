"""ROI lane quantification, rotation helpers, and intensity profiles."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .analysis import (
    AnalysisConfig,
    AnalysisResult,
    BandMeasurement,
    BackgroundMode,
    _resolve_dark_bands,
    annotate_measurements,
)


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


def angle_to_vertical(x0: float, y0: float, x1: float, y1: float) -> float:
    """Return the segment tilt in degrees relative to vertical (image coords)."""
    return float(np.degrees(np.arctan2(y1 - y0, x1 - x0)) - 90.0)


def normalize_angle(angle_deg: float) -> float:
    """Map angle into (-90, 90] for blot straightening."""
    angle = ((float(angle_deg) + 180.0) % 360.0) - 180.0
    if angle > 90.0:
        angle -= 180.0
    elif angle <= -90.0:
        angle += 180.0
    return angle


def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image around center.

    Positive degrees follow OpenCV/image-coordinate convention: they reduce a
    down-right (positive atan2) tilt, i.e. lift the right side.
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
    border = (255, 255, 255) if image.ndim == 3 else 255
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
    radius = max(1, int(radius))
    kernel = np.ones(radius * 2 + 1, dtype=float)
    kernel /= kernel.sum()
    return np.convolve(values.astype(float), kernel, mode="same")


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
    candidates = sorted(set(candidates), key=lambda i: smooth[i], reverse=True)
    peaks: list[int] = []
    for peak in candidates:
        if all(abs(peak - other) >= distance for other in peaks):
            peaks.append(peak)
        if expected_lanes and len(peaks) >= max(1, expected_lanes):
            break
    return sorted(peaks)


def separators_from_profile(
    profile: np.ndarray,
    roi: RoiRect,
    *,
    expected_lanes: int | None = None,
) -> list[float]:
    """Place lane separators at valleys between profile peaks."""
    peaks = find_profile_peaks(profile, expected_lanes=expected_lanes or None)
    if len(peaks) < 2:
        if expected_lanes and expected_lanes > 1:
            return default_separators(roi, expected_lanes)
        return []
    smooth = _smooth_profile(profile, max(2, profile.size // 80))
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


def quantify_lane_section(
    gray: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    dark_bands: bool,
    lane: int,
    saturate_level: int = 250,
) -> BandMeasurement:
    """Quantify one full-height lane strip using per-lane P25 background."""
    h, w = gray.shape[:2]
    x1 = int(np.clip(x1, 0, max(0, w - 1)))
    y1 = int(np.clip(y1, 0, max(0, h - 1)))
    x2 = int(np.clip(x2, x1 + 1, w))
    y2 = int(np.clip(y2, y1 + 1, h))
    section = gray[y1:y2, x1:x2]
    signal = (255.0 - section.astype(float)) if dark_bands else section.astype(float)
    background = float(np.percentile(signal, 25))
    threshold = background + 0.2 * float(signal.std())
    mask = signal >= threshold
    corrected = np.maximum(signal - background, 0.0)
    integrated = float(corrected[mask].sum()) if mask.any() else 0.0
    if dark_bands:
        saturated_fraction = float(np.mean(section <= (255 - saturate_level)))
        display_bg = 255.0 - background
    else:
        saturated_fraction = float(np.mean(section >= saturate_level))
        display_bg = background
    return BandMeasurement(
        lane=lane,
        band=1,
        x=x1,
        y=y1,
        width=x2 - x1,
        height=y2 - y1,
        area=int(section.size),
        mean_intensity=round(float(section.mean()), 3),
        background=round(float(display_bg), 3),
        integrated_density=round(integrated, 3),
        saturated_fraction=round(saturated_fraction, 4),
    )


def quantify_roi_lanes(
    gray: np.ndarray,
    roi: RoiRect,
    separators: list[float],
    *,
    background_mode: BackgroundMode = "auto",
    saturate_level: int = 250,
) -> AnalysisResult:
    roi = roi.clipped(gray)
    if roi.width < 3 or roi.height < 3:
        raise ValueError("ROI 过小，无法定量")
    dark_bands = _resolve_dark_bands(gray, background_mode)
    measurements = [
        quantify_lane_section(
            gray,
            x1,
            roi.y,
            x2,
            roi.y + roi.height,
            dark_bands=dark_bands,
            lane=index,
            saturate_level=saturate_level,
        )
        for index, (x1, x2) in enumerate(lane_boundaries(roi, separators), start=1)
    ]
    annotated = annotate_measurements(gray, measurements)
    # Draw separators on top for ROI QC.
    for sep in separators:
        x = int(round(sep))
        if roi.x < x < roi.x + roi.width:
            cv2.line(annotated, (x, roi.y), (x, roi.y + roi.height - 1), (255, 180, 0), 1, cv2.LINE_AA)
    cv2.rectangle(
        annotated,
        (roi.x, roi.y),
        (roi.x + roi.width - 1, roi.y + roi.height - 1),
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )
    saturated = sum(1 for m in measurements if m.saturated_fraction >= 0.05)
    return AnalysisResult(
        measurements=measurements,
        annotated_image=annotated,
        dark_bands=dark_bands,
        saturated_band_count=saturated,
        gray_image=gray.copy(),
    )


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


def analyze_rotated_image(
    path,
    *,
    angle_deg: float = 0.0,
    config: AnalysisConfig | None = None,
) -> AnalysisResult:
    from .analysis import analyze_array

    image = rotate_image(load_bgr(path), angle_deg)
    return analyze_array(image, config=config)

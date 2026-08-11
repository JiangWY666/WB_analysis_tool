"""Native bit-depth image loading, display conversion, and rotation.

Chemiluminescence imagers commonly export 16-bit TIFFs. Quantification must
run on the native dynamic range; only the on-screen preview is reduced to
8-bit. This module keeps both layers together:

- ``raw``: float32 grayscale in the source value range (quantification layer)
- ``display_bgr``: 8-bit BGR with auto-contrast for >8-bit sources (view layer)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class BlotImage:
    """A blot photo carrying both the quantification and the display layer."""

    raw: np.ndarray
    """float32 grayscale, native value range [0, white_level]."""

    display_bgr: np.ndarray
    """uint8 BGR used for preview and coarse auto-detection."""

    white_level: float
    """Maximum representable value of the source (255.0, 65535.0, ...)."""

    bit_depth: int
    path: Path | None = None

    @property
    def display_gray(self) -> np.ndarray:
        return cv2.cvtColor(self.display_bgr, cv2.COLOR_BGR2GRAY)

    @property
    def height(self) -> int:
        return int(self.raw.shape[0])

    @property
    def width(self) -> int:
        return int(self.raw.shape[1])


def _infer_white_level(array: np.ndarray) -> tuple[float, int]:
    """Return (white_level, bit_depth) for a numpy image array."""
    if array.dtype == np.uint8:
        return 255.0, 8
    if array.dtype == np.uint16:
        return 65535.0, 16
    if array.dtype in (np.int32, np.uint32, np.int64):
        peak = float(array.max()) if array.size else 0.0
        if peak <= 255.0:
            return 255.0, 8
        if peak <= 65535.0:
            return 65535.0, 16
        return max(peak, 1.0), 32
    # Float sources: honor the observed range.
    finite = array[np.isfinite(array)]
    peak = float(finite.max()) if finite.size else 1.0
    if peak <= 1.0:
        return 1.0, 32
    if peak <= 255.0:
        return 255.0, 32
    if peak <= 65535.0:
        return 65535.0, 32
    return peak, 32


def _to_gray_float(array: np.ndarray) -> np.ndarray:
    """Collapse channels to float32 grayscale without changing value range."""
    if array.ndim == 2:
        return array.astype(np.float32)
    if array.shape[2] == 4:
        array = array[:, :, :3]
    # Rec. 601 luma keeps the native range for any bit depth.
    b = array[:, :, 0].astype(np.float32)
    g = array[:, :, 1].astype(np.float32)
    r = array[:, :, 2].astype(np.float32)
    return 0.114 * b + 0.587 * g + 0.299 * r


def display_gray_from_raw(raw: np.ndarray, white_level: float) -> np.ndarray:
    """8-bit preview. 8-bit sources pass through; deeper sources auto-contrast."""
    if white_level <= 255.0:
        return np.clip(raw, 0, 255).astype(np.uint8)
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        return np.zeros(raw.shape, dtype=np.uint8)
    lo = float(np.percentile(finite, 0.05))
    hi = float(np.percentile(finite, 99.85))
    if hi - lo < 1e-9:
        hi = lo + 1.0
    scaled = (raw.astype(np.float32) - lo) / (hi - lo)
    return np.clip(scaled * 255.0 + 0.5, 0, 255).astype(np.uint8)


def blot_from_array(array: np.ndarray, *, path: Path | None = None) -> BlotImage:
    """Build a BlotImage from an already-decoded numpy image."""
    if array is None or array.size == 0:
        raise ValueError("图片为空，无法读取")
    white_level, bit_depth = _infer_white_level(array)
    raw = _to_gray_float(array)
    if array.ndim == 3 and array.dtype == np.uint8:
        display_bgr = array[:, :, :3].copy()
    else:
        display_bgr = cv2.cvtColor(
            display_gray_from_raw(raw, white_level), cv2.COLOR_GRAY2BGR
        )
    return BlotImage(
        raw=raw,
        display_bgr=display_bgr,
        white_level=white_level,
        bit_depth=bit_depth,
        path=path,
    )


def load_blot(path: str | Path) -> BlotImage:
    """Load an image preserving native bit depth (Unicode-path friendly)."""
    path = Path(path)
    array: np.ndarray | None = None
    try:
        data = np.fromfile(path, dtype=np.uint8)
        array = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except OSError:
        array = None

    if array is None:
        # Some TIFF variants fail in cv2.imdecode; Pillow is more tolerant.
        try:
            from PIL import Image

            with Image.open(path) as pil_image:
                if pil_image.mode in ("I;16", "I;16B", "I;16L", "I;16N"):
                    array = np.array(pil_image, dtype=np.uint16)
                elif pil_image.mode in ("I", "F"):
                    array = np.array(pil_image)
                elif pil_image.mode in ("L", "P"):
                    array = np.array(pil_image.convert("L"))
                else:
                    rgb = np.array(pil_image.convert("RGB"))
                    array = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except OSError as exc:
            raise ValueError(f"不支持或已损坏的图片：{path}") from exc

    if array is None:
        raise ValueError(f"不支持或已损坏的图片：{path}")
    return blot_from_array(array, path=path)


def saturation_fraction(
    raw: np.ndarray,
    *,
    dark_bands: bool,
    white_level: float,
    margin: float = 0.005,
) -> float:
    """Fraction of pixels clipped at the signal end of the dynamic range."""
    if raw.size == 0:
        return 0.0
    if dark_bands:
        return float(np.mean(raw <= white_level * margin))
    return float(np.mean(raw >= white_level * (1.0 - margin)))


def _rotation_params(
    shape_hw: tuple[int, int], angle_deg: float
) -> tuple[np.ndarray, int, int]:
    """Affine matrix plus the expanded canvas size for a center rotation."""
    height, width = int(shape_hw[0]), int(shape_hw[1])
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(height * sin + width * cos)
    new_h = int(height * cos + width * sin)
    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]
    return matrix, new_w, new_h


def rotation_matrix(shape_hw: tuple[int, int], angle_deg: float) -> np.ndarray:
    """2x3 affine matrix (with expansion translation) used by :func:`rotate_blot`."""
    matrix, _, _ = _rotation_params(shape_hw, angle_deg)
    return matrix


def _border_value(image: np.ndarray) -> float:
    """Median of the outer border: fills rotation corners with background tone."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    return float(np.median(border))


def rotate_blot(blot: BlotImage, angle_deg: float) -> BlotImage:
    """Rotate both layers, filling corners with the estimated background tone."""
    if abs(angle_deg) < 1e-6:
        return replace(blot, raw=blot.raw.copy(), display_bgr=blot.display_bgr.copy())
    matrix, new_w, new_h = _rotation_params(blot.raw.shape[:2], angle_deg)
    raw_fill = _border_value(blot.raw)
    raw = cv2.warpAffine(
        blot.raw,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=raw_fill,
    )
    disp_fill = _border_value(blot.display_bgr)
    display_bgr = cv2.warpAffine(
        blot.display_bgr,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(disp_fill, disp_fill, disp_fill),
    )
    return replace(blot, raw=raw, display_bgr=display_bgr)


def remap_rotation(
    points_xy: np.ndarray,
    shape_hw: tuple[int, int],
    old_angle_deg: float,
    new_angle_deg: float,
) -> np.ndarray:
    """Map points between two rotated frames of the same original image.

    ``shape_hw`` is (height, width) of the unrotated image; ``points_xy`` is an
    (N, 2) array in the old rotated frame. Lets the UI keep an ROI and its lane
    separators in place while the user fine-tunes the rotation angle.
    """
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if points.size == 0:
        return points.copy()
    m_old, _, _ = _rotation_params(shape_hw, old_angle_deg)
    m_new, _, _ = _rotation_params(shape_hw, new_angle_deg)
    m_old_inv = cv2.invertAffineTransform(m_old)
    original = cv2.transform(points.reshape(-1, 1, 2), m_old_inv)
    mapped = cv2.transform(original, m_new)
    return mapped.reshape(-1, 2)

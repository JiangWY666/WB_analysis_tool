"""Tests for native bit-depth loading, display conversion, and rotation."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from wb_analyzer.imageio import (
    blot_from_array,
    load_blot,
    remap_rotation,
    rotate_blot,
    rotation_matrix,
)


def test_load_blot_preserves_16bit(tmp_path: Path) -> None:
    image = np.full((60, 90), 52000, dtype=np.uint16)
    image[20:40, 30:60] = 21000
    target = tmp_path / "样品16位.tif"
    ok, buffer = cv2.imencode(".tif", image)
    assert ok
    buffer.tofile(target)
    blot = load_blot(target)
    assert blot.white_level == 65535.0
    assert blot.bit_depth == 16
    assert blot.raw.dtype == np.float32
    assert float(blot.raw.max()) > 255.0  # native range survives loading
    assert blot.display_bgr.dtype == np.uint8
    assert blot.display_bgr.ndim == 3
    assert blot.display_gray.ndim == 2
    assert blot.path == target


def test_load_blot_8bit_display_passthrough(tmp_path: Path) -> None:
    image = np.full((40, 50), 200, dtype=np.uint8)
    image[10:20, 10:30] = 60
    target = tmp_path / "sample8.png"
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    buffer.tofile(target)
    blot = load_blot(target)
    assert blot.white_level == 255.0
    assert blot.bit_depth == 8
    # 8-bit sources keep their original look (no contrast stretch).
    assert int(blot.display_gray[15, 15]) == 60
    assert int(blot.display_gray[0, 0]) == 200


def test_display_autocontrast_for_low_range_16bit() -> None:
    # Underexposed 16-bit source: peak value only ~3000 / 65535.
    image = np.full((60, 80), 2800, dtype=np.uint16)
    image[20:40, 20:60] = 300
    blot = blot_from_array(image)
    gray = blot.display_gray
    # Auto contrast must spread the preview across most of the 8-bit range.
    assert int(gray.max()) - int(gray.min()) > 180


def test_load_blot_rejects_broken_file(tmp_path: Path) -> None:
    bogus = tmp_path / "broken.tif"
    bogus.write_bytes(b"not an image at all")
    with pytest.raises(ValueError):
        load_blot(bogus)


def test_rotate_blot_uses_background_fill() -> None:
    image = np.full((120, 200), 12, dtype=np.uint8)  # dark background
    blot = blot_from_array(image)
    rotated = rotate_blot(blot, 12.0)
    assert rotated.width > blot.width  # canvas expands
    # Expanded corners stay near the dark background tone (no white wedges).
    assert float(rotated.raw[0, 0]) < 40.0
    assert int(rotated.display_bgr[0, 0, 0]) < 40


def test_rotation_matrix_shape_and_identity() -> None:
    matrix = rotation_matrix((100, 200), 0.0)
    assert matrix.shape == (2, 3)
    point = matrix @ np.array([50.0, 30.0, 1.0])
    assert abs(point[0] - 50.0) < 1e-9
    assert abs(point[1] - 30.0) < 1e-9


def test_remap_rotation_roundtrip() -> None:
    shape_hw = (180, 320)
    points = np.array([[50.0, 40.0], [200.0, 120.0]])
    forward = remap_rotation(points, shape_hw, 0.0, 9.0)
    assert forward.shape == (2, 2)
    back = remap_rotation(forward, shape_hw, 9.0, 0.0)
    assert np.allclose(back, points, atol=1e-6)


def test_remap_rotation_tracks_rotated_pixel() -> None:
    # A bright pixel rotated by rotate_blot must land where remap_rotation says.
    image = np.zeros((120, 160), dtype=np.uint8)
    image[40, 100] = 255
    blot = blot_from_array(image)
    rotated = rotate_blot(blot, 20.0)
    mapped = remap_rotation(np.array([[100.0, 40.0]]), (120, 160), 0.0, 20.0)[0]
    ys, xs = np.where(rotated.raw > 120)
    assert len(ys) > 0
    assert abs(float(np.mean(xs)) - mapped[0]) < 2.0
    assert abs(float(np.mean(ys)) - mapped[1]) < 2.0

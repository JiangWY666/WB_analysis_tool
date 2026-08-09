from pathlib import Path

import cv2
import numpy as np

from wb_analyzer.normalize import AnalysisSession, QuantEntry, export_normalized_csv, lane_intensities_from_measurements
from wb_analyzer.export import export_figure
from wb_analyzer.roi import (
    RoiRect,
    angle_to_horizontal,
    default_separators,
    normalize_angle,
    quantify_roi_lanes,
    roi_intensity_profile,
    rotate_image,
    to_gray,
)  # rotate_image: positive degrees are clockwise
from wb_analyzer.analysis import analyze_array


def synthetic_lanes(n: int = 4) -> np.ndarray:
    image = np.full((160, 320), 220, dtype=np.uint8)
    width = 40
    gap = 30
    x = 25
    for _ in range(n):
        cv2.rectangle(image, (x, 55), (x + width, 95), 40, -1)
        x += width + gap
    return image


def test_rotate_expands_and_keeps_content() -> None:
    image = synthetic_lanes()
    rotated = rotate_image(image, 8.0)
    assert rotated.shape[0] >= image.shape[0]
    assert rotated.shape[1] >= image.shape[1]
    assert int((rotated < 80).sum()) > 0  # dark band pixels survive rotation


def test_roi_lane_quantification_and_profile() -> None:
    gray = synthetic_lanes(4)
    roi = RoiRect(20, 40, 280, 70)
    seps = default_separators(roi, 4)
    assert len(seps) == 3
    result = quantify_roi_lanes(gray, roi, seps, background_mode="dark_bands")
    assert len(result.measurements) == 4
    assert all(m.integrated_density > 0 for m in result.measurements)
    profile, dark = roi_intensity_profile(gray, roi, background_mode="dark_bands")
    assert dark
    assert profile.shape[0] == roi.width
    assert float(profile.max()) > float(profile.min())


def test_normalization_session_roundtrip(tmp_path: Path) -> None:
    session = AnalysisSession()
    session.upsert(
        QuantEntry(
            kind="loading_control",
            name="Actin",
            group="实验A",
            intensities=[100.0, 110.0, 90.0],
            backgrounds=[20.0, 21.0, 19.0],
            mean_intensities=[50.0, 52.0, 48.0],
        )
    )
    session.upsert(
        QuantEntry(
            kind="target",
            name="pAKT",
            group="实验A",
            intensities=[40.0, 55.0, 30.0],
        )
    )
    rows = session.normalize_all()
    assert len(rows) == 3
    assert rows[0].normalized == 0.4
    lc_path = session.export_loading_control_csv(tmp_path / "lc.csv")
    session2 = AnalysisSession()
    assert session2.import_loading_control_csv(lc_path) == 1
    session2.upsert(QuantEntry(kind="target", name="pAKT", group="实验A", intensities=[40.0, 55.0, 30.0]))
    out = export_normalized_csv(session2.normalize_all(), tmp_path / "norm.csv")
    text = out.read_text(encoding="utf-8-sig")
    assert "normalized" in text
    assert "0.4" in text


def test_auto_detect_lane_aggregation_for_normalize() -> None:
    image = np.full((180, 260), 225, dtype=np.uint8)
    for x in (35, 105, 175):
        cv2.rectangle(image, (x, 45), (x + 42, 57), 45, -1)
        cv2.rectangle(image, (x, 112), (x + 42, 127), 75, -1)
    result = analyze_array(image)
    intensities, _, _ = lane_intensities_from_measurements(result.measurements)
    assert len(intensities) == 3
    assert all(v > 0 for v in intensities)


def test_to_gray_accepts_gray_and_bgr() -> None:
    gray = synthetic_lanes()
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    assert to_gray(gray).ndim == 2
    assert to_gray(bgr).shape == gray.shape


def test_ruler_angle_and_figure_export(tmp_path: Path) -> None:
    assert abs(angle_to_horizontal(0, 0, 10, 0)) < 1e-6
    assert abs(normalize_angle(angle_to_horizontal(0, 0, 10, 10)) - 45.0) < 1e-6
    image = cv2.cvtColor(synthetic_lanes(), cv2.COLOR_GRAY2BGR)
    roi = RoiRect(20, 40, 200, 80)
    path = export_figure(image, tmp_path / "fig.png", roi=roi, pad=4)
    assert path.exists()
    exported = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert exported is not None
    assert exported.shape[1] <= 200 + 8 + 2


def test_rotate_positive_lifts_right_side() -> None:
    image = np.zeros((101, 101), dtype=np.uint8)
    image[50, 80] = 255  # point to the right of center
    rotated = rotate_image(image, 90.0)
    # Positive OpenCV/UI angle: right -> top (corrects down-right blot tilt).
    ys, xs = np.where(rotated > 200)
    assert len(ys) > 0
    assert abs(float(np.mean(xs)) - 50) < 4
    assert float(np.mean(ys)) < 30


def test_positive_rotation_reduces_down_right_tilt() -> None:
    image = np.zeros((201, 201), dtype=np.uint8)
    for t in range(-40, 41):
        image[100 + t // 4, 100 + t] = 255
    ys, xs = np.where(image > 0)
    tilt0 = float(np.degrees(np.arctan(np.polyfit(xs, ys, 1)[0])))
    rotated = rotate_image(image, tilt0)
    ys, xs = np.where(rotated > 0)
    tilt1 = float(np.degrees(np.arctan(np.polyfit(xs, ys, 1)[0])))
    assert tilt0 > 5
    assert abs(tilt1) < abs(tilt0) / 3

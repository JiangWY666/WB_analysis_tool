from pathlib import Path

import numpy as np

from wb_analyzer.dnd import parse_drop_paths
from wb_analyzer.roi import RoiRect, find_profile_peaks, separators_from_profile


def test_separators_follow_profile_valleys() -> None:
    profile = np.zeros(200, dtype=float)
    for center, height in ((30, 20.0), (80, 12.0), (130, 18.0), (170, 10.0)):
        profile[center - 6 : center + 7] += height
    roi = RoiRect(100, 40, 200, 50)
    separators = separators_from_profile(profile, roi, expected_lanes=4)
    assert len(separators) == 3
    rel = [s - roi.x for s in separators]
    assert 30 < rel[0] < 80
    assert 80 < rel[1] < 130
    assert 130 < rel[2] < 170


def test_find_profile_peaks_respects_expected_count() -> None:
    profile = np.array([0, 1, 5, 1, 0, 1, 8, 1, 0, 1, 6, 1, 0, 1, 3, 1, 0], dtype=float)
    peaks = find_profile_peaks(profile, expected_lanes=3, min_distance=3)
    assert len(peaks) == 3


def test_parse_drop_paths_handles_tif_and_file_uri() -> None:
    paths = parse_drop_paths(r"{C:\data\demo.tif} {D:\x\a.TIFF}")
    assert [p.suffix.lower() for p in paths] == [".tif", ".tiff"]
    uri_paths = parse_drop_paths("file:///C:/Users/me/pad4_001(1)_signal.tif")
    assert uri_paths[0].name == "pad4_001(1)_signal.tif"


def test_demo_profile_split_on_real_roi() -> None:
    demo = Path(__file__).resolve().parents[1] / "pad4_001(1)_signal.tif"
    if not demo.exists():
        return
    from wb_analyzer.roi import load_bgr, roi_intensity_profile, to_gray

    gray = to_gray(load_bgr(demo))
    roi = RoiRect(430, 400, 560, 60)
    profile, _ = roi_intensity_profile(gray, roi, background_mode="dark_bands")
    separators = separators_from_profile(profile, roi, expected_lanes=8)
    assert len(separators) == 7

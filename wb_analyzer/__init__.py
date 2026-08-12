"""Automatic Western Blot grayscale analysis."""

from .analysis import AnalysisConfig, AnalysisResult, BandMeasurement, analyze_image
from .imageio import BlotImage, load_blot, remap_rotation, rotate_blot, rotation_matrix
from .normalize import (
    AnalysisSession,
    LaneSeries,
    QuantEntry,
    lane_series_from_measurements,
)
from .quant import (
    LaneProfile,
    compute_lane_profiles,
    measure_box_native,
    quantify_lanes,
    remeasure_native,
    resolve_dark_bands_native,
    roi_column_profile,
)
from .roi import RoiRect, rotate_image

__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "AnalysisSession",
    "BandMeasurement",
    "BlotImage",
    "LaneProfile",
    "QuantEntry",
    "LaneSeries",
    "lane_series_from_measurements",
    "RoiRect",
    "analyze_image",
    "compute_lane_profiles",
    "load_blot",
    "measure_box_native",
    "quantify_lanes",
    "remap_rotation",
    "remeasure_native",
    "resolve_dark_bands_native",
    "roi_column_profile",
    "rotate_blot",
    "rotate_image",
    "rotation_matrix",
]
__version__ = "0.3.0"

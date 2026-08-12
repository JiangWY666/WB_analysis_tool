"""Export analysis results and presentation figures."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from .analysis import AnalysisResult
from .roi import RoiRect


def export_result(result: AnalysisResult, image_path: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    csv_path = output / f"{stem}_bands.csv"
    preview_path = output / f"{stem}_annotated.png"
    fields = [
        "lane",
        "band",
        "x",
        "y",
        "width",
        "height",
        "area",
        "mean_intensity",
        "background",
        "integrated_density",
        "saturated_fraction",
        "detection_limit",
        "snr",
        "below_detection",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(m.as_dict() for m in result.measurements)
    encoded, buffer = cv2.imencode(".png", result.annotated_image)
    if not encoded:
        raise OSError("无法生成标注预览图")
    buffer.tofile(preview_path)
    return csv_path, preview_path


def _save_bgr(path: Path, image: np.ndarray) -> Path:
    suffix = path.suffix.lower() or ".png"
    if suffix not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        path = path.with_suffix(".png")
        suffix = ".png"
    ext = ".jpg" if suffix in {".jpg", ".jpeg"} else suffix
    encoded, buffer = cv2.imencode(ext, image)
    if not encoded:
        raise OSError(f"无法导出图片：{path}")
    buffer.tofile(path)
    return path


def export_figure(
    image_bgr: np.ndarray,
    path: str | Path,
    *,
    roi: RoiRect | None = None,
    pad: int = 8,
) -> Path:
    """Export a presentation crop (ROI if provided, else full frame)."""
    path = Path(path)
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("没有可导出的图像")
    frame = image_bgr
    if roi is not None:
        clipped = roi.clipped(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr)
        x1 = max(0, clipped.x - pad)
        y1 = max(0, clipped.y - pad)
        x2 = min(image_bgr.shape[1], clipped.x + clipped.width + pad)
        y2 = min(image_bgr.shape[0], clipped.y + clipped.height + pad)
        frame = image_bgr[y1:y2, x1:x2].copy()
    return _save_bgr(path, frame)

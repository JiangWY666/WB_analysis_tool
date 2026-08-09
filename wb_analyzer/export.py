"""Export analysis results."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2

from .analysis import AnalysisResult


def export_result(result: AnalysisResult, image_path: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    csv_path = output / f"{stem}_bands.csv"
    preview_path = output / f"{stem}_annotated.png"
    fields = ["lane", "band", "x", "y", "width", "height", "area", "mean_intensity", "background", "integrated_density"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(m.as_dict() for m in result.measurements)
    encoded, buffer = cv2.imencode(".png", result.annotated_image)
    if not encoded:
        raise OSError("无法生成标注预览图")
    buffer.tofile(preview_path)
    return csv_path, preview_path

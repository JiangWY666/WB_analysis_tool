"""Export analysis results and presentation figures.

A figure leaving this module is meant to end up in a manuscript, so every
irreversible thing done to the pixels is both *chosen* by the caller and
*written down* next to the file. :func:`export_presentation_figure` renders
from :attr:`BlotImage.raw` rather than from the 8-bit preview, tags the file
with a physical resolution, and drops a ``*_export_info.txt`` sidecar that a
methods section can be copied out of.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .analysis import AnalysisResult, BandMeasurement
from .imageio import (
    MAPPING_AUTO,
    MAPPING_LINEAR,
    ROTATE_FAST,
    ROTATE_SHARP,
    BlotImage,
    DisplayMapping,
    MappingMode,
    apply_display_mapping,
    clipping_fractions,
    resolve_display_mapping,
)
from .normalize import ORIGIN_UNKNOWN, BackgroundOrigin, coerce_origin
from .roi import RoiRect

__all__ = [
    "DEFAULT_DPI",
    "LAYER_ALL",
    "LAYER_BAND_BOXES",
    "LAYER_BAND_LABELS",
    "LAYER_ROI",
    "LAYER_SEPARATORS",
    "LOSSLESS_SUFFIXES",
    "MM_PER_INCH",
    "FigureExport",
    "PrintScale",
    "dpi_for_width_mm",
    "export_figure",
    "export_presentation_figure",
    "export_presentation_figures",
    "export_result",
    "max_print_width_mm",
    "print_scale",
]

MM_PER_INCH = 25.4

DEFAULT_DPI = 300.0
"""Resolution tag used when the caller does not state a physical width.

Nothing is resampled to reach it: the tag only says how large the existing
pixels should be printed. :func:`print_scale` reports the width that implies.
"""

LOSSLESS_SUFFIXES = (".tif", ".tiff", ".png")
"""The only containers a figure may be written to. JPEG shifted grayscale
values by up to 6–7 levels on real blots, which is indistinguishable from a
real intensity difference in a printed panel."""

LAYER_BAND_BOXES = "band_boxes"
LAYER_BAND_LABELS = "band_labels"
LAYER_ROI = "roi"
LAYER_SEPARATORS = "separators"

LAYER_ALL: frozenset[str] = frozenset(
    {LAYER_BAND_BOXES, LAYER_BAND_LABELS, LAYER_ROI, LAYER_SEPARATORS}
)

_LAYER_LABELS = {
    LAYER_BAND_BOXES: "条带框",
    LAYER_BAND_LABELS: "文字标签",
    LAYER_ROI: "ROI 框",
    LAYER_SEPARATORS: "泳道分隔线",
}

_INTERPOLATION_LABELS = {
    cv2.INTER_NEAREST: "最近邻",
    ROTATE_FAST: "双线性",
    cv2.INTER_CUBIC: "双三次",
    cv2.INTER_AREA: "区域平均",
    ROTATE_SHARP: "Lanczos-4",
}

_BAND_COLOR = (0, 0, 255)
_SATURATED_COLOR = (0, 140, 255)
_ROI_COLOR = (0, 200, 255)
_SEPARATOR_COLOR = (255, 200, 0)


# --------------------------------------------------------------------- 结果表


def export_result(
    result: AnalysisResult,
    image_path: str | Path,
    output_dir: str | Path,
    *,
    origin: BackgroundOrigin | str = ORIGIN_UNKNOWN,
    dpi: float = DEFAULT_DPI,
) -> tuple[Path, Path]:
    """Write the band table and the annotated preview.

    ``origin`` states which background convention measured these numbers
    (README §9). It is a parameter rather than something read off the result
    because only the caller knows which path it took — and a file that does not
    say is a file whose numbers cannot be safely compared with anyone else's a
    month later.

    The preview here is the 8-bit working view, not a figure; use
    :func:`export_presentation_figure` for anything that will be published.
    """
    origin = coerce_origin(origin)
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
        "origin",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({**m.as_dict(), "origin": origin} for m in result.measurements)
    _write_image(preview_path, result.annotated_image, dpi=dpi, bgr=True)
    return csv_path, preview_path


# ----------------------------------------------------------------- 物理尺寸


@dataclass(frozen=True)
class PrintScale:
    """How large a figure prints, given that its pixel count is fixed.

    This software never interpolates a figure up to hit a target resolution,
    so dpi is not a setting — it is a consequence of the physical width you
    place the image at. Reporting it honestly is the whole point.
    """

    width_px: int
    height_px: int
    dpi: float

    @property
    def width_mm(self) -> float:
        return self.width_px / self.dpi * MM_PER_INCH

    @property
    def height_mm(self) -> float:
        return self.height_px / self.dpi * MM_PER_INCH


def dpi_for_width_mm(width_px: int, width_mm: float) -> float:
    """Resolution that ``width_px`` pixels reach when printed ``width_mm`` wide."""
    width_mm = float(width_mm)
    if width_mm <= 0:
        raise ValueError(f"打印宽度必须为正数，收到 {width_mm} mm")
    return int(width_px) / (width_mm / MM_PER_INCH)


def max_print_width_mm(width_px: int, dpi: float) -> float:
    """Widest print that still reaches ``dpi`` without interpolating."""
    dpi = float(dpi)
    if dpi <= 0:
        raise ValueError(f"分辨率必须为正数，收到 {dpi} dpi")
    return int(width_px) / dpi * MM_PER_INCH


def print_scale(
    width_px: int,
    height_px: int,
    *,
    width_mm: float | None = None,
    dpi: float | None = None,
) -> PrintScale:
    """Resolve the print geometry from either a physical width or a dpi."""
    if width_mm is not None and dpi is not None:
        raise ValueError("打印宽度与 dpi 只能给一个：给了宽度，dpi 就是算出来的")
    if width_mm is not None:
        dpi = dpi_for_width_mm(width_px, width_mm)
    elif dpi is None:
        dpi = DEFAULT_DPI
    dpi = float(dpi)
    if not np.isfinite(dpi) or dpi <= 0:
        raise ValueError(f"分辨率必须为有限正数，收到 {dpi}")
    return PrintScale(int(width_px), int(height_px), dpi)


# --------------------------------------------------------------------- 写文件


def _resolve_suffix(path: Path, *, strict: bool) -> Path:
    suffix = path.suffix.lower()
    if suffix in LOSSLESS_SUFFIXES:
        return path
    if strict:
        raise ValueError(
            f"展示图只能存为无损格式（{'/'.join(LOSSLESS_SUFFIXES)}），收到 {path.suffix or '无扩展名'}。"
            "JPEG 等有损格式在实测中会把灰度改动 6–7 级，无法用于定量展示。"
        )
    # Legacy callers (and an old file dialog that still offers JPEG) get
    # silently redirected rather than an error; the returned path tells them.
    return path.with_suffix(".png")


def _write_image(path: Path, array: np.ndarray, *, dpi: float, bgr: bool) -> Path:
    """Write uint8 gray / uint8 colour / uint16 gray with a resolution tag.

    OpenCV cannot attach pHYs or the TIFF resolution tags and refuses 16-bit
    colour, so the figure writer goes through Pillow (already a dependency).
    """
    from PIL import Image

    if array.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"不支持的导出位深：{array.dtype}")
    data = array
    if data.ndim == 3:
        if data.dtype != np.uint8:
            raise ValueError("彩色标注图只支持 8 位输出")
        data = cv2.cvtColor(data[:, :, :3], cv2.COLOR_BGR2RGB) if bgr else data[:, :, :3]
    elif data.ndim != 2:
        raise ValueError(f"不支持的图像维度：{array.shape}")
    image = Image.fromarray(np.ascontiguousarray(data))
    options: dict[str, object] = {"dpi": (dpi, dpi)}
    if path.suffix.lower() in (".tif", ".tiff"):
        options["compression"] = "tiff_lzw"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, **options)
    return path


# --------------------------------------------------------------------- 标注层


def _draw_layers(
    canvas: np.ndarray,
    *,
    offset: tuple[int, int],
    measurements: Sequence[BandMeasurement],
    roi: RoiRect | None,
    separators: Sequence[float],
    layers: frozenset[str],
    line_width: int,
    font_scale: float,
) -> None:
    """Paint the requested overlay layers onto an 8-bit BGR crop, in place."""
    dx, dy = offset
    if roi is not None and LAYER_ROI in layers:
        cv2.rectangle(
            canvas,
            (roi.x - dx, roi.y - dy),
            (roi.x + roi.width - 1 - dx, roi.y + roi.height - 1 - dy),
            _ROI_COLOR,
            line_width,
        )
    if separators and LAYER_SEPARATORS in layers:
        top = (roi.y - dy) if roi is not None else 0
        bottom = (roi.y + roi.height - 1 - dy) if roi is not None else canvas.shape[0] - 1
        for x in separators:
            column = int(round(float(x))) - dx
            cv2.line(canvas, (column, top), (column, bottom), _SEPARATOR_COLOR, line_width)
    for item in measurements:
        color = _BAND_COLOR if item.saturated_fraction < 0.05 else _SATURATED_COLOR
        x1, y1 = item.x - dx, item.y - dy
        x2, y2 = x1 + item.width - 1, y1 + item.height - 1
        if LAYER_BAND_BOXES in layers:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_width)
        if LAYER_BAND_LABELS in layers:
            cv2.putText(
                canvas,
                f"L{item.lane}-B{item.band}",
                (x1, max(12, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                line_width,
                cv2.LINE_AA,
            )


def _crop_bounds(
    shape_hw: tuple[int, int], roi: RoiRect | None, pad: int
) -> tuple[int, int, int, int]:
    height, width = int(shape_hw[0]), int(shape_hw[1])
    if roi is None:
        return 0, 0, width, height
    clipped = roi.clipped(np.empty(shape_hw[:2], dtype=np.uint8))
    pad = max(0, int(pad))
    x1 = max(0, clipped.x - pad)
    y1 = max(0, clipped.y - pad)
    x2 = min(width, clipped.x + clipped.width + pad)
    y2 = min(height, clipped.y + clipped.height + pad)
    return x1, y1, x2, y2


# -------------------------------------------------------------------- 展示图


@dataclass(frozen=True)
class FigureExport:
    """Everything a caption would need to say about one exported figure."""

    path: Path
    info_path: Path | None
    mapping: DisplayMapping
    scale: PrintScale
    bit_depth: int
    channels: int
    clipped_low: float
    clipped_high: float
    annotated: bool
    layers: tuple[str, ...]
    source_path: Path | None
    source_bit_depth: int
    source_size: tuple[int, int]
    crop: tuple[int, int, int, int] | None
    rotation_deg: float
    rotation_interpolation: int
    compression: str
    generated_at: str

    @property
    def clipped_fraction(self) -> float:
        return self.clipped_low + self.clipped_high

    def info_text(self) -> str:
        return _info_text(self)


def _rotation_line(export: FigureExport) -> str:
    if abs(export.rotation_deg) < 1e-6:
        return "未旋转（图像未做任何几何校正）"
    direction = "顺时针" if export.rotation_deg > 0 else "逆时针"
    interp = _INTERPOLATION_LABELS.get(export.rotation_interpolation, "未知插值")
    return f"已旋正 {abs(export.rotation_deg):.2f}°（{direction}，{interp} 插值）"


def _mapping_lines(export: FigureExport) -> list[str]:
    mapping = export.mapping
    out_max = 255 if export.bit_depth == 8 else 65535
    lines = [
        f"  映射方式    {mapping.label}",
        f"  黑点        {mapping.black_point:.6g}（原始灰度，映射到输出 0）",
        f"  白点        {mapping.white_point:.6g}（原始灰度，映射到输出 {out_max}）",
    ]
    if mapping.mode == MAPPING_AUTO:
        lines.append(
            f"  取点依据    第 {mapping.low_percentile:g} / {mapping.high_percentile:g} 百分位"
        )
    channels = "单通道灰度" if export.channels == 1 else f"{export.channels} 通道彩色"
    container = "TIFF" if export.path.suffix.lower() in (".tif", ".tiff") else "PNG"
    lines.append("  传递函数    线性（未做伽马、曲线或局部调整）")
    lines.append(
        f"  输出格式    {export.bit_depth} 位{channels}，{container}（{export.compression}）"
    )
    lines.append(
        f"  削波像素    纯黑 {export.clipped_low * 100:.2f}%，纯白 {export.clipped_high * 100:.2f}%"
        "（占输出像素）"
    )
    # Plain ASCII markers only: this file gets opened in Notepad, pasted into
    # Word, and echoed through GBK consoles, and a symbol that survives none of
    # those is worse than no symbol.
    if export.clipped_fraction > 0:
        lines.append("  注意        落在纯黑/纯白上的像素之间已无灰度差异，这部分信息在本图中不可恢复。")
    else:
        lines.append("  说明        没有任何像素被压到纯黑或纯白，全部灰度层次都保留在图中。")
    return lines


def _method_sentence(export: FigureExport) -> str:
    mapping = export.mapping
    channels = "单通道灰度" if export.channels == 1 else "RGB 彩色（叠加标注）"
    parts = [
        f"图像以无损格式导出为 {export.bit_depth} 位{channels}，"
        f"未做插值缩放，保持采集分辨率 {export.scale.width_px} × {export.scale.height_px} 像素。"
    ]
    if mapping.mode == MAPPING_LINEAR:
        parts.append(
            f"灰度为线性全量程映射（原始值 0 → 黑，{mapping.white_point:.6g}（传感器满量程）→ 白），"
            "未做对比度拉伸、亮度或伽马调整。"
        )
    elif mapping.mode == MAPPING_AUTO:
        parts.append(
            f"灰度经线性对比度拉伸：原始值 {mapping.black_point:.6g}（第 {mapping.low_percentile:g} 百分位）→ 黑，"
            f"{mapping.white_point:.6g}（第 {mapping.high_percentile:g} 百分位）→ 白，未做伽马调整。"
        )
    else:
        parts.append(
            f"灰度按操作者指定的黑白点线性映射（{mapping.black_point:.6g} → 黑，"
            f"{mapping.white_point:.6g} → 白），未做伽马调整。"
        )
    if export.clipped_fraction > 0:
        parts.append(
            f"映射后 {export.clipped_low * 100:.2f}% 的像素落在纯黑、{export.clipped_high * 100:.2f}% 落在纯白，"
            "这部分像素之间的灰度差异在本图中已被削平。"
        )
    else:
        parts.append("映射后无像素落在纯黑或纯白，全部灰度层次均得到保留。")
    if abs(export.rotation_deg) >= 1e-6:
        interp = _INTERPOLATION_LABELS.get(export.rotation_interpolation, "未知插值")
        parts.append(f"图像整体旋正 {abs(export.rotation_deg):.2f}°（{interp} 插值）。")
    if export.crop is not None:
        parts.append("按感兴趣区整体裁切。")
    parts.append("全图统一处理，未做任何局部修饰、擦除或拼接。")
    return "".join(parts)


def _info_text(export: FigureExport) -> str:
    scale = export.scale
    source = export.source_path.name if export.source_path else "（未知，图像来自内存）"
    lines: list[str] = [
        "Western Blot 展示图导出说明",
        "本文件由 wb-gray-analyzer 自动生成，记录了这张图从原始数据到成图的全部处理。",
        "",
        f"图像文件    {export.path.name}",
        f"生成时间    {export.generated_at}",
        f"源文件      {source}",
        "",
        "【源数据】",
        f"  存储位深    {export.source_bit_depth} 位",
        f"  满量程      {export.mapping.white_level:.6g}（传感器白电平，非容器上限）",
        f"  原始尺寸    {export.source_size[0]} × {export.source_size[1]} px",
        "",
        "【几何】",
        f"  旋正        {_rotation_line(export)}",
    ]
    if export.crop is not None:
        x, y, w, h = export.crop
        lines.append(f"  裁切        原图 (x={x}, y={y}, w={w}, h={h})")
    else:
        lines.append("  裁切        无（整图导出）")
    lines += [
        "  重采样      无（未做任何插值放大或缩小）",
        f"  输出尺寸    {scale.width_px} × {scale.height_px} px",
        "",
        "【灰度映射】",
    ]
    lines += _mapping_lines(export)
    lines += [
        "",
        "【打印尺寸】",
        f"  写入文件的分辨率标签  {scale.dpi:.1f} dpi",
        f"  此图 {scale.width_px} px 宽，印成 {scale.width_mm:.1f} mm 宽时相当于 {scale.dpi:.1f} dpi"
        f"（高 {scale.height_px} px → {scale.height_mm:.1f} mm）",
        f"  若要求 300 dpi，最大可印 {max_print_width_mm(scale.width_px, 300.0):.1f} mm 宽",
        f"  若要求 600 dpi，最大可印 {max_print_width_mm(scale.width_px, 600.0):.1f} mm 宽",
        "  本软件不做插值放大：dpi 是印刷尺寸的结果，不是可以调大的设置。",
        "",
    ]
    if export.annotated:
        drawn = "、".join(_LAYER_LABELS.get(name, name) for name in export.layers)
        lines += [
            "【标注】",
            f"  已绘制      {drawn or '无'}",
            "  标注仅覆盖在图像之上，未改动下方任何像素的灰度。",
            "",
        ]
    lines += [
        "【方法学建议写法】",
        f"  {_method_sentence(export)}",
        "",
    ]
    return "\n".join(lines)


def export_presentation_figure(
    blot: BlotImage,
    path: str | Path,
    *,
    roi: RoiRect | None = None,
    pad: int = 8,
    mapping: MappingMode | DisplayMapping = MAPPING_LINEAR,
    black_point: float | None = None,
    white_point: float | None = None,
    bit_depth: int | None = None,
    annotate: bool = False,
    measurements: Sequence[BandMeasurement] = (),
    separators: Sequence[float] = (),
    layers: frozenset[str] | Sequence[str] = LAYER_ALL,
    line_width: int = 1,
    font_scale: float = 0.38,
    print_width_mm: float | None = None,
    dpi: float | None = None,
    rotation_deg: float = 0.0,
    rotation_interpolation: int = ROTATE_FAST,
    write_info: bool = True,
) -> FigureExport:
    """Render a publication figure from ``blot.raw`` and document what was done.

    ``mapping`` decides the only irreversible step. :data:`MAPPING_LINEAR`
    (the default) divides by the sensor full scale, so nothing is stretched
    and nothing clips; :data:`MAPPING_AUTO` reproduces the on-screen preview's
    percentile stretch, and :data:`MAPPING_MANUAL` takes ``black_point`` and
    ``white_point``. Whichever is used, the resolved numbers land in the
    ``*_export_info.txt`` sidecar beside the image.

    ``rotation_deg`` and ``rotation_interpolation`` are *declarations*: this
    function does not rotate, it records the straightening the caller already
    applied so the sidecar can state it.
    """
    if blot is None or blot.raw is None or blot.raw.size == 0:
        raise ValueError("没有可导出的图像")
    path = _resolve_suffix(Path(path), strict=True)

    source_h, source_w = blot.raw.shape[:2]
    x1, y1, x2, y2 = _crop_bounds((source_h, source_w), roi, pad)
    crop_raw = blot.raw[y1:y2, x1:x2]
    if crop_raw.size == 0:
        raise ValueError("裁切区域为空，无法导出")
    crop = None if roi is None else (x1, y1, x2 - x1, y2 - y1)

    if isinstance(mapping, DisplayMapping):
        resolved = mapping
    else:
        # Percentiles come from the whole frame, not the crop, so that
        # MAPPING_AUTO reproduces exactly the preview the user was looking at
        # when they framed the ROI. Clipping is then measured on what actually
        # ends up in the file.
        resolved = resolve_display_mapping(
            blot.raw,
            blot.white_level,
            mode=mapping,
            black_point=black_point,
            white_point=white_point,
        )

    if bit_depth is None:
        bit_depth = 8 if blot.bit_depth <= 8 else 16
    if bit_depth not in (8, 16):
        raise ValueError(f"导出位深只能是 8 或 16，收到 {bit_depth}")
    if annotate and bit_depth != 8:
        # Annotations are coloured, and no mainstream writer (Pillow included)
        # produces 16-bit RGB. Saying so beats silently downgrading.
        raise ValueError("彩色标注只能导出 8 位图；干净版可以保持 16 位")

    dtype = np.uint8 if bit_depth == 8 else np.uint16
    gray = apply_display_mapping(crop_raw, resolved, dtype=dtype)
    clipped_low, clipped_high = clipping_fractions(gray)

    if annotate:
        selected = frozenset(layers)
        unknown = selected - LAYER_ALL
        if unknown:
            raise ValueError(f"未知的标注图层：{sorted(unknown)}")
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        _draw_layers(
            canvas,
            offset=(x1, y1),
            measurements=measurements,
            roi=roi,
            separators=separators,
            layers=selected,
            line_width=max(1, int(line_width)),
            font_scale=float(font_scale),
        )
        pixels, channels = canvas, 3
        used_layers = tuple(sorted(selected))
    else:
        pixels, channels = gray, 1
        used_layers = ()

    scale = print_scale(
        pixels.shape[1], pixels.shape[0], width_mm=print_width_mm, dpi=dpi
    )
    _write_image(path, pixels, dpi=scale.dpi, bgr=True)

    export = FigureExport(
        path=path,
        info_path=None,
        mapping=resolved,
        scale=scale,
        bit_depth=bit_depth,
        channels=channels,
        clipped_low=clipped_low,
        clipped_high=clipped_high,
        annotated=annotate,
        layers=used_layers,
        source_path=blot.path,
        source_bit_depth=blot.bit_depth,
        source_size=(source_w, source_h),
        crop=crop,
        rotation_deg=float(rotation_deg),
        rotation_interpolation=int(rotation_interpolation),
        compression="LZW 无损" if path.suffix.lower() in (".tif", ".tiff") else "Deflate 无损",
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    if not write_info:
        return export
    info_path = path.with_name(f"{path.stem}_export_info.txt")
    info_path.write_text(export.info_text(), encoding="utf-8-sig")
    return replace(export, info_path=info_path)


def export_presentation_figures(
    blot: BlotImage,
    path: str | Path,
    *,
    clean: bool = True,
    annotated: bool = False,
    **kwargs: Any,
) -> tuple[FigureExport, ...]:
    """Export the clean figure and the annotated one in a single call.

    The annotated file always gets an ``_annotated`` suffix, so the pair is
    recognisable in a folder no matter which of the two was asked for. Each
    file carries its own sidecar: a figure that travels without its provenance
    is a figure whose provenance is gone.
    """
    if not clean and not annotated:
        raise ValueError("至少要导出一种版本（干净版或标注版）")
    if "annotate" in kwargs:
        raise ValueError("请用 clean= / annotated= 选择版本，不要再传 annotate=")
    base = _resolve_suffix(Path(path), strict=True)
    results: list[FigureExport] = []
    if clean:
        results.append(export_presentation_figure(blot, base, annotate=False, **kwargs))
    if annotated:
        target = base.with_name(f"{base.stem}_annotated{base.suffix}")
        # Colour overlays force 8-bit; the clean copy above keeps the depth
        # the caller asked for.
        results.append(
            export_presentation_figure(
                blot, target, **{**kwargs, "annotate": True, "bit_depth": 8}
            )
        )
    return tuple(results)


# ------------------------------------------------------------ 旧接口（兼容）


def export_figure(
    image_bgr: np.ndarray,
    path: str | Path,
    *,
    roi: RoiRect | None = None,
    pad: int = 8,
    dpi: float = DEFAULT_DPI,
) -> Path:
    """Export a presentation crop from an already-rendered 8-bit image.

    Kept for callers that only hold a preview buffer. It cannot state what was
    done to the pixels, so :func:`export_presentation_figure` is the one to
    reach for whenever a :class:`BlotImage` is available. Lossy suffixes are
    redirected to PNG rather than refused; the returned path is authoritative.
    """
    path = _resolve_suffix(Path(path), strict=False)
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("没有可导出的图像")
    x1, y1, x2, y2 = _crop_bounds(image_bgr.shape[:2], roi, pad)
    frame = image_bgr[y1:y2, x1:x2]
    return _write_image(path, np.ascontiguousarray(frame), dpi=dpi, bgr=True)

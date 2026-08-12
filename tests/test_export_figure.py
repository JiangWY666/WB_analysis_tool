"""Tests for publication-grade figure export.

The through-line: a figure that leaves this software must carry its own
provenance, and the default path must not throw away grey levels.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from wb_analyzer.analysis import BandMeasurement
from wb_analyzer.export import (
    DEFAULT_DPI,
    LAYER_BAND_BOXES,
    LAYER_BAND_LABELS,
    LAYER_ROI,
    LAYER_SEPARATORS,
    dpi_for_width_mm,
    export_figure,
    export_presentation_figure,
    export_presentation_figures,
    export_result,
    max_print_width_mm,
    print_scale,
)
from wb_analyzer.imageio import (
    MAPPING_AUTO,
    MAPPING_LINEAR,
    MAPPING_MANUAL,
    ROTATE_FAST,
    ROTATE_SHARP,
    apply_display_mapping,
    blot_from_array,
    clipping_fractions,
    display_gray_from_raw,
    resolve_display_mapping,
    rotate_blot,
)
from wb_analyzer.roi import RoiRect


def wide_range_16bit() -> np.ndarray:
    """A 16-bit blot shaped like the real ones: bright film, most pixels near full scale.

    The auto stretch crushes the top of this histogram, which is exactly the
    behaviour the linear mapping has to avoid.
    """
    rng = np.random.default_rng(7)
    image = np.full((80, 120), 65100, dtype=np.uint16)
    image += rng.integers(0, 130, image.shape, dtype=np.uint16)
    image = np.minimum(image, 65535).astype(np.uint16)
    for column in (25, 60, 95):
        image[30:50, column - 8 : column + 8] = 8000
    image[0:3, 0:3] = 2737  # a few very dark pixels, as on a real film edge
    return image


def load_pixels(path: Path) -> tuple[Image.Image, np.ndarray]:
    with Image.open(path) as handle:
        return handle.copy(), np.array(handle)


# ------------------------------------------------------------ 映射：核心问题


def test_linear_mapping_loses_no_grey_level(tmp_path: Path) -> None:
    """The default export must be a bit-exact copy of a full-scale 16-bit source."""
    source = wide_range_16bit()
    blot = blot_from_array(source, path=tmp_path / "样品.tif")
    assert blot.white_level == 65535.0

    export = export_presentation_figure(blot, tmp_path / "fig.tif")

    image, pixels = load_pixels(export.path)
    assert image.mode == "I;16"
    assert pixels.dtype == np.uint16
    assert np.array_equal(pixels, source), "线性全量程映射必须是无损的"
    assert len(np.unique(pixels)) == len(np.unique(source))
    assert export.clipped_low == 0.0
    assert export.clipped_high == 0.0


def test_auto_mapping_is_what_the_preview_did_and_it_clips(tmp_path: Path) -> None:
    """AUTO reproduces the on-screen stretch — including the damage it does."""
    source = wide_range_16bit()
    blot = blot_from_array(source, path=tmp_path / "样品.tif")

    export = export_presentation_figure(
        blot, tmp_path / "auto.tif", mapping=MAPPING_AUTO, bit_depth=8
    )

    _, pixels = load_pixels(export.path)
    # Same bytes the preview layer would have produced.
    assert np.array_equal(pixels, display_gray_from_raw(blot.raw, blot.white_level))
    # The stretch formally excludes almost nothing, yet rounding still dumps
    # most of the film background onto pure white. That gap is the whole point
    # of measuring clipping on the output.
    assert float(np.mean(source > export.mapping.white_point)) < 0.01
    assert export.clipped_high > 0.5, "自动拉伸本来就该把这张图的背景压成一片纯白"
    assert export.clipped_high == pytest.approx(float(np.mean(pixels == 255)))
    # And the numbers that caused it are on record.
    assert export.mapping.low_percentile == 0.05
    assert export.mapping.high_percentile == 99.85
    assert export.mapping.black_point < export.mapping.white_point


def test_linear_beats_auto_on_clipping_and_levels(tmp_path: Path) -> None:
    source = wide_range_16bit()
    blot = blot_from_array(source, path=tmp_path / "样品.tif")

    linear = export_presentation_figure(blot, tmp_path / "lin.tif")
    auto = export_presentation_figure(
        blot, tmp_path / "auto.tif", mapping=MAPPING_AUTO, bit_depth=8
    )

    assert linear.clipped_fraction == 0.0
    assert auto.clipped_fraction > linear.clipped_fraction
    _, lin_pixels = load_pixels(linear.path)
    _, auto_pixels = load_pixels(auto.path)
    assert len(np.unique(lin_pixels)) > len(np.unique(auto_pixels))


def test_manual_mapping_uses_the_points_it_was_given(tmp_path: Path) -> None:
    source = wide_range_16bit()
    blot = blot_from_array(source, path=tmp_path / "样品.tif")

    export = export_presentation_figure(
        blot,
        tmp_path / "manual.tif",
        mapping=MAPPING_MANUAL,
        black_point=8000.0,
        white_point=65200.0,
    )

    assert export.mapping.mode == MAPPING_MANUAL
    assert export.mapping.black_point == 8000.0
    assert export.mapping.white_point == 65200.0
    _, pixels = load_pixels(export.path)
    assert pixels.min() == 0 and pixels.max() == 65535
    # Everything the operator put outside the window ends up on an extreme.
    assert export.clipped_low >= float(np.mean(source < 8000.0))
    assert export.clipped_high >= float(np.mean(source > 65200.0))
    assert "8000" in export.info_path.read_text(encoding="utf-8-sig")


def test_manual_mapping_rejects_a_missing_or_inverted_window() -> None:
    raw = np.linspace(0, 100, 64, dtype=np.float32).reshape(8, 8)
    with pytest.raises(ValueError, match="同时给出黑点和白点"):
        resolve_display_mapping(raw, 255.0, mode=MAPPING_MANUAL, black_point=1.0)
    with pytest.raises(ValueError, match="白点必须大于黑点"):
        resolve_display_mapping(
            raw, 255.0, mode=MAPPING_MANUAL, black_point=90.0, white_point=10.0
        )
    with pytest.raises(ValueError, match="未知的灰度映射方式"):
        resolve_display_mapping(raw, 255.0, mode="gamma")  # type: ignore[arg-type]


def test_display_gray_from_raw_is_unchanged_by_the_refactor() -> None:
    """The preview path must still be byte-for-byte what it always was."""
    source = wide_range_16bit()
    raw = source.astype(np.float32)
    lo = float(np.percentile(raw, 0.05))
    hi = float(np.percentile(raw, 99.85))
    expected = np.clip((raw - lo) / (hi - lo) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    assert np.array_equal(display_gray_from_raw(raw, 65535.0), expected)
    # 8-bit sources still pass straight through.
    small = np.array([[0, 7, 255]], dtype=np.float32)
    assert np.array_equal(display_gray_from_raw(small, 255.0), small.astype(np.uint8))


def test_mapping_survives_nan_and_a_degenerate_histogram() -> None:
    flat = np.full((6, 6), 500.0, dtype=np.float32)
    mapping = resolve_display_mapping(flat, 65535.0, mode=MAPPING_AUTO)
    assert mapping.white_point > mapping.black_point  # no divide by zero
    assert apply_display_mapping(flat, mapping).shape == flat.shape

    dirty = flat.copy()
    dirty[0, 0] = np.nan
    rendered = apply_display_mapping(dirty, mapping, dtype=np.uint16)
    assert np.isfinite(rendered).all()
    assert rendered[0, 0] == 0  # NaN lands on the black point, never on garbage
    # A histogram with no width has nothing to show: every pixel is pure black,
    # and the sidecar will say so rather than pretend the figure has contrast.
    assert clipping_fractions(rendered) == (1.0, 0.0)
    assert clipping_fractions(np.zeros((0, 0), dtype=np.uint8)) == (0.0, 0.0)


# -------------------------------------------------------------- 位深 / 通道


def test_16bit_source_exports_single_channel_16bit(tmp_path: Path) -> None:
    blot = blot_from_array(wide_range_16bit(), path=tmp_path / "s.tif")
    export = export_presentation_figure(blot, tmp_path / "fig.tif")
    assert export.bit_depth == 16
    assert export.channels == 1
    with Image.open(export.path) as handle:
        assert handle.mode == "I;16"
        assert len(handle.getbands()) == 1
        assert handle.tag_v2[258] == (16,)


def test_8bit_source_stays_8bit(tmp_path: Path) -> None:
    source = np.linspace(0, 255, 40 * 60, dtype=np.uint8).reshape(40, 60)
    blot = blot_from_array(source, path=tmp_path / "s.png")
    export = export_presentation_figure(blot, tmp_path / "fig.tif")
    assert export.bit_depth == 8
    assert export.channels == 1
    _, pixels = load_pixels(export.path)
    assert pixels.dtype == np.uint8
    assert np.array_equal(pixels, source)


def test_bit_depth_can_be_forced_down(tmp_path: Path) -> None:
    blot = blot_from_array(wide_range_16bit(), path=tmp_path / "s.tif")
    export = export_presentation_figure(blot, tmp_path / "fig.tif", bit_depth=8)
    assert export.bit_depth == 8
    _, pixels = load_pixels(export.path)
    assert pixels.dtype == np.uint8
    with pytest.raises(ValueError, match="导出位深只能是 8 或 16"):
        export_presentation_figure(blot, tmp_path / "bad.tif", bit_depth=12)


# ----------------------------------------------------------------- dpi 元数据


def test_tiff_carries_resolution_tags(tmp_path: Path) -> None:
    blot = blot_from_array(wide_range_16bit(), path=tmp_path / "s.tif")
    export = export_presentation_figure(blot, tmp_path / "fig.tif", print_width_mm=53.0)
    with Image.open(export.path) as handle:
        assert float(handle.tag_v2[282]) == pytest.approx(export.scale.dpi, rel=1e-6)
        assert float(handle.tag_v2[283]) == pytest.approx(export.scale.dpi, rel=1e-6)
        assert handle.tag_v2[296] == 2  # ResolutionUnit = inch


def test_png_carries_phys(tmp_path: Path) -> None:
    blot = blot_from_array(wide_range_16bit(), path=tmp_path / "s.tif")
    export = export_presentation_figure(blot, tmp_path / "fig.png", dpi=600.0)
    with Image.open(export.path) as handle:
        # pHYs stores pixels-per-metre as an integer, so allow the rounding.
        assert handle.info["dpi"][0] == pytest.approx(600.0, abs=0.1)


def test_print_scale_math_reports_reachable_dpi() -> None:
    assert dpi_for_width_mm(624, 53.0) == pytest.approx(299.0, abs=0.5)
    assert max_print_width_mm(624, 600.0) == pytest.approx(26.4, abs=0.1)
    scale = print_scale(624, 400, width_mm=53.0)
    assert scale.dpi == pytest.approx(299.0, abs=0.5)
    assert scale.width_mm == pytest.approx(53.0, rel=1e-9)
    assert print_scale(624, 400).dpi == DEFAULT_DPI
    with pytest.raises(ValueError, match="只能给一个"):
        print_scale(624, 400, width_mm=53.0, dpi=300.0)
    with pytest.raises(ValueError, match="打印宽度必须为正数"):
        dpi_for_width_mm(624, 0.0)


def test_dpi_follows_the_cropped_width_not_the_source(tmp_path: Path) -> None:
    blot = blot_from_array(wide_range_16bit(), path=tmp_path / "s.tif")
    export = export_presentation_figure(
        blot, tmp_path / "fig.tif", roi=RoiRect(20, 10, 60, 40), pad=0, print_width_mm=20.0
    )
    assert export.scale.width_px == 60
    assert export.scale.dpi == pytest.approx(dpi_for_width_mm(60, 20.0))


# -------------------------------------------------------------------- 无损格式


def test_lossy_formats_are_refused(tmp_path: Path) -> None:
    blot = blot_from_array(wide_range_16bit(), path=tmp_path / "s.tif")
    with pytest.raises(ValueError, match="无损格式"):
        export_presentation_figure(blot, tmp_path / "fig.jpg")


def test_legacy_export_figure_redirects_jpeg_to_png(tmp_path: Path) -> None:
    """The old file dialog still offers JPEG; nothing lossy may reach the disk."""
    image = np.full((40, 60, 3), 120, dtype=np.uint8)
    saved = export_figure(image, tmp_path / "fig.jpg")
    assert saved.suffix == ".png"
    assert saved.exists()
    assert not (tmp_path / "fig.jpg").exists()
    with Image.open(saved) as handle:
        assert handle.info["dpi"][0] == pytest.approx(DEFAULT_DPI, abs=0.5)


def test_legacy_export_figure_still_crops_to_roi(tmp_path: Path) -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[40:60, 20:120] = 200
    saved = export_figure(image, tmp_path / "fig.png", roi=RoiRect(20, 40, 100, 20), pad=4)
    exported = cv2.imdecode(np.fromfile(saved, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert exported.shape[:2] == (28, 108)


# ------------------------------------------------------------------ 标注分层


def band(lane: int, x: int) -> BandMeasurement:
    return BandMeasurement(
        lane=lane, band=1, x=x, y=30, width=16, height=20, area=320,
        mean_intensity=100.0, background=10.0, integrated_density=1000.0,
    )


def test_clean_and_annotated_come_out_of_one_call(tmp_path: Path) -> None:
    blot = blot_from_array(wide_range_16bit(), path=tmp_path / "样品.tif")
    clean, annotated = export_presentation_figures(
        blot,
        tmp_path / "fig.tif",
        clean=True,
        annotated=True,
        measurements=[band(1, 17), band(2, 52)],
        roi=RoiRect(5, 5, 110, 70),
    )
    assert clean.path.name == "fig.tif"
    assert annotated.path.name == "fig_annotated.tif"
    assert clean.bit_depth == 16 and clean.channels == 1
    assert annotated.bit_depth == 8 and annotated.channels == 3
    # Each file travels with its own provenance.
    assert clean.info_path.name == "fig_export_info.txt"
    assert annotated.info_path.name == "fig_annotated_export_info.txt"
    for export in (clean, annotated):
        assert export.info_path.exists()
    with Image.open(annotated.path) as handle:
        assert handle.mode == "RGB"


def test_layers_can_be_selected_independently(tmp_path: Path) -> None:
    blot = blot_from_array(wide_range_16bit(), path=tmp_path / "s.tif")
    common = {
        "measurements": [band(1, 17), band(2, 52)],
        "roi": RoiRect(5, 5, 110, 70),
        "separators": [40.0, 75.0],
        "annotate": True,
        "bit_depth": 8,
    }

    def pixels_for(layers: frozenset[str], name: str) -> np.ndarray:
        export = export_presentation_figure(
            blot, tmp_path / name, layers=layers, **common
        )
        return load_pixels(export.path)[1]

    nothing = pixels_for(frozenset(), "none.tif")
    boxes = pixels_for(frozenset({LAYER_BAND_BOXES}), "boxes.tif")
    labelled = pixels_for(frozenset({LAYER_BAND_BOXES, LAYER_BAND_LABELS}), "lab.tif")
    roi_only = pixels_for(frozenset({LAYER_ROI}), "roi.tif")
    seps = pixels_for(frozenset({LAYER_SEPARATORS}), "seps.tif")

    # An empty layer set is the clean image, just carried in three channels.
    assert nothing[:, :, 0].tolist() == nothing[:, :, 1].tolist()
    for drawn in (boxes, labelled, roi_only, seps):
        assert not np.array_equal(drawn, nothing)
    # Each layer adds strictly on top of the previous one.
    assert np.count_nonzero(labelled != nothing) > np.count_nonzero(boxes != nothing)


def test_unknown_layer_is_rejected(tmp_path: Path) -> None:
    blot = blot_from_array(wide_range_16bit(), path=tmp_path / "s.tif")
    with pytest.raises(ValueError, match="未知的标注图层"):
        export_presentation_figure(
            blot, tmp_path / "f.tif", annotate=True, bit_depth=8, layers=frozenset({"halo"})
        )


def test_annotations_cannot_be_smuggled_into_a_16bit_file(tmp_path: Path) -> None:
    blot = blot_from_array(wide_range_16bit(), path=tmp_path / "s.tif")
    with pytest.raises(ValueError, match="彩色标注只能导出 8 位图"):
        export_presentation_figure(blot, tmp_path / "f.tif", annotate=True, bit_depth=16)


def test_annotated_export_does_not_disturb_the_grey_underneath(tmp_path: Path) -> None:
    blot = blot_from_array(wide_range_16bit(), path=tmp_path / "s.tif")
    clean = export_presentation_figure(blot, tmp_path / "clean.tif", bit_depth=8)
    annotated = export_presentation_figure(
        blot,
        tmp_path / "ann.tif",
        annotate=True,
        bit_depth=8,
        measurements=[band(1, 17)],
        layers=frozenset({LAYER_BAND_BOXES}),
    )
    _, flat = load_pixels(clean.path)
    _, drawn = load_pixels(annotated.path)
    touched = np.any(drawn != flat[:, :, None], axis=2)
    assert touched.mean() < 0.05, "标注只该覆盖很小一块，不该改动整幅灰度"
    assert np.array_equal(drawn[~touched][:, 0], flat[~touched])


# --------------------------------------------------------------------- 旁注文件


def test_sidecar_states_everything_a_caption_needs(tmp_path: Path) -> None:
    blot = blot_from_array(wide_range_16bit(), path=tmp_path / "样品16位.tif")
    export = export_presentation_figure(
        blot,
        tmp_path / "fig.tif",
        roi=RoiRect(10, 10, 100, 60),
        pad=4,
        print_width_mm=53.0,
        rotation_deg=3.2,
        rotation_interpolation=ROTATE_SHARP,
    )
    text = export.info_path.read_text(encoding="utf-8-sig")
    for needle in (
        "样品16位.tif",         # 源文件
        "16 位",                 # 位深
        "线性全量程",            # 映射方式
        "65535",                 # 白点
        "旋正 3.20°",            # 旋转
        "Lanczos-4",             # 插值
        "重采样      无",        # 没有插值放大
        "dpi",                   # 可达分辨率
        "53.0 mm",               # 给定物理宽度
        "方法学建议写法",
        "未做对比度拉伸",
    ):
        assert needle in text, f"旁注文件里缺少「{needle}」"


def test_sidecar_admits_the_damage_when_auto_clips(tmp_path: Path) -> None:
    blot = blot_from_array(wide_range_16bit(), path=tmp_path / "s.tif")
    export = export_presentation_figure(
        blot, tmp_path / "fig.tif", mapping=MAPPING_AUTO, bit_depth=8
    )
    text = export.info_path.read_text(encoding="utf-8-sig")
    assert "自动对比度拉伸" in text
    assert "百分位" in text
    assert "削平" in text
    assert f"纯白 {export.clipped_high * 100:.2f}%" in text


def test_sidecar_can_be_turned_off(tmp_path: Path) -> None:
    blot = blot_from_array(wide_range_16bit(), path=tmp_path / "s.tif")
    export = export_presentation_figure(blot, tmp_path / "fig.tif", write_info=False)
    assert export.info_path is None
    assert not (tmp_path / "fig_export_info.txt").exists()
    assert export.info_text()  # still available in memory for the UI to show


def test_sidecar_says_so_when_nothing_was_rotated_or_cropped(tmp_path: Path) -> None:
    blot = blot_from_array(wide_range_16bit(), path=tmp_path / "s.tif")
    export = export_presentation_figure(blot, tmp_path / "fig.tif")
    text = export.info_path.read_text(encoding="utf-8-sig")
    assert "未旋转" in text
    assert "裁切        无（整图导出）" in text


# ------------------------------------------------------------------ 边界条件


def test_tiny_image_and_empty_roi(tmp_path: Path) -> None:
    blot = blot_from_array(np.array([[7]], dtype=np.uint16), path=tmp_path / "s.tif")
    export = export_presentation_figure(blot, tmp_path / "one.tif")
    assert export.scale.width_px == 1 and export.scale.height_px == 1

    bigger = blot_from_array(wide_range_16bit(), path=tmp_path / "s.tif")
    # An ROI entirely outside the frame is clamped, never turned into 0 pixels.
    export = export_presentation_figure(
        bigger, tmp_path / "off.tif", roi=RoiRect(500, 500, 40, 40), pad=0
    )
    assert export.scale.width_px >= 1 and export.scale.height_px >= 1


def test_export_refuses_an_empty_image(tmp_path: Path) -> None:
    blot = blot_from_array(np.ones((4, 4), dtype=np.uint8))
    empty = type(blot)(
        raw=np.zeros((0, 0), dtype=np.float32),
        display_bgr=blot.display_bgr,
        white_level=255.0,
        bit_depth=8,
    )
    with pytest.raises(ValueError, match="没有可导出的图像"):
        export_presentation_figure(empty, tmp_path / "f.tif")


def test_unicode_path_round_trips(tmp_path: Path) -> None:
    blot = blot_from_array(wide_range_16bit(), path=tmp_path / "样品.tif")
    export = export_presentation_figure(blot, tmp_path / "结果" / "展示图.tif")
    assert export.path.exists() and export.info_path.exists()
    assert "展示图.tif" in export.info_path.read_text(encoding="utf-8-sig")


def test_export_result_preview_now_carries_dpi(tmp_path: Path) -> None:
    from wb_analyzer.analysis import AnalysisResult

    result = AnalysisResult(
        measurements=[],
        annotated_image=np.zeros((20, 30, 3), dtype=np.uint8),
        dark_bands=True,
    )
    _, preview = export_result(result, tmp_path / "blot.tif", tmp_path / "out")
    with Image.open(preview) as handle:
        assert handle.info["dpi"][0] == pytest.approx(DEFAULT_DPI, abs=0.5)


# ------------------------------------------------------------- 旋转插值质量


def test_sharp_rotation_keeps_values_inside_the_source_range() -> None:
    """Lanczos-4 rings; a raw layer must never come back with negative intensity."""
    source = wide_range_16bit()
    blot = blot_from_array(source)
    sharp = rotate_blot(blot, 7.0, interpolation=ROTATE_SHARP)
    assert float(sharp.raw.min()) >= 0.0
    assert float(sharp.raw.max()) <= float(source.max())


def test_rotation_default_is_still_bilinear() -> None:
    blot = blot_from_array(wide_range_16bit())
    assert np.array_equal(
        rotate_blot(blot, 5.0).raw, rotate_blot(blot, 5.0, interpolation=ROTATE_FAST).raw
    )


def test_sharp_rotation_preserves_fine_detail_better_than_bilinear() -> None:
    """A round trip through Lanczos-4 gets closer to the original than bilinear."""
    rng = np.random.default_rng(3)
    source = rng.integers(4000, 60000, (160, 200)).astype(np.uint16)
    blot = blot_from_array(source)

    def roundtrip_error(flag: int) -> float:
        there = rotate_blot(blot, 6.0, interpolation=flag)
        back = rotate_blot(there, -6.0, interpolation=flag)
        pad_y = (back.raw.shape[0] - 160) // 2
        pad_x = (back.raw.shape[1] - 200) // 2
        inner = back.raw[pad_y + 30 : pad_y + 130, pad_x + 30 : pad_x + 170]
        return float(np.mean((inner - source[30:130, 30:170].astype(np.float32)) ** 2))

    assert roundtrip_error(ROTATE_SHARP) < roundtrip_error(ROTATE_FAST)

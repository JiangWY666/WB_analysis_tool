"""导出数据档位：默认精简 6 列，完整统计与历史长表都是可选项。

覆盖：
* 默认 ``*_bands.csv`` 只有 source / lane / band / sample / integrated_density /
  below_detection，且顺序固定；
* 勾了完整统计列之后前 15 列与旧版逐列一致；
* 仅导原始统计时不含 normalized / fold change；
* 重复上样（同名样品 N 次）原样保留为 N 行；
* raw_long.csv 默认不产生；勾选后才写，列档位跟随完整统计开关；
* 未做归一化时对话框里该项被禁用；
* 对话框勾选（含完整列 / 历史长表）写入 prefs，下次读回；
* 对话框 offscreen 冒烟；
* 手绘补画（ROI 基线口径）写出非零 peak。
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pytest

from wb_analyzer.analysis import AnalysisResult, BandMeasurement
from wb_analyzer.export import (
    BANDS_CSV_FIELDS,
    COMPACT_BANDS_FIELDS,
    COMPACT_SESSION_RAW_FIELDS,
    LEGACY_BANDS_FIELDS,
    SESSION_RAW_FIELDS,
    build_raw_stat_rows,
    export_result,
    export_selected_data,
    export_session_raw_stats,
    format_roi_id,
    normalization_export_available,
    planned_data_export_names,
    summary_export_available,
)
from wb_analyzer.normalize import (
    ORIGIN_ROI_BASELINE,
    AnalysisSession,
    LaneBand,
    QuantEntry,
)
from wb_analyzer.quant import quantify_lanes
from wb_analyzer.roi import RoiRect, default_separators

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def measurement(
    lane: int,
    density: float,
    *,
    band: int = 1,
    peak: float = 0.0,
) -> BandMeasurement:
    return BandMeasurement(
        lane=lane,
        band=band,
        x=lane * 10,
        y=4,
        width=8,
        height=6,
        area=48,
        mean_intensity=density / 10.0,
        background=1.5,
        integrated_density=density,
        saturated_fraction=0.0,
        detection_limit=3.0,
        snr=12.0,
        peak=peak or density / 8.0,
    )


def fake_result(*densities: float) -> AnalysisResult:
    measurements = [
        measurement(index, value) for index, value in enumerate(densities, start=1)
    ]
    canvas = np.zeros((20, 40, 3), dtype=np.uint8)
    return AnalysisResult(
        measurements=measurements, annotated_image=canvas, dark_bands=True
    )


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def replicate_session() -> AnalysisSession:
    """同一份 Ctrl 上了两道，第三道是处理组——导出不得把 Ctrl 合成一行。"""
    session = AnalysisSession()
    session.upsert(
        QuantEntry(
            kind="target",
            name="ProteinX",
            group="G",
            intensities=[10.0, 11.0, 20.0],
            lanes=[1, 2, 3],
            source="mem_signal.tif",
            membrane="mem_signal",
            origin=ORIGIN_ROI_BASELINE,
            bands=[
                LaneBand(lane=1, band=1, y=10, integrated_density=10.0),
                LaneBand(lane=2, band=1, y=10, integrated_density=11.0),
                LaneBand(lane=3, band=1, y=10, integrated_density=20.0),
            ],
        )
    )
    session.set_lane_label("G", 1, "Ctrl")
    session.set_lane_label("G", 2, "Ctrl")
    session.set_lane_label("G", 3, "Treat")
    return session


# --------------------------------------------------------------- 核心：原始表


def test_default_bands_csv_is_the_six_compact_columns(tmp_path: Path) -> None:
    csv_path, preview = export_result(
        fake_result(40.0, 50.0),
        tmp_path / "blot.tif",
        tmp_path / "out",
        origin=ORIGIN_ROI_BASELINE,
        sample_names={1: "A", 2: "B"},
    )
    header, rows = read_csv(csv_path)
    assert header == COMPACT_BANDS_FIELDS
    assert header == [
        "source",
        "lane",
        "band",
        "sample",
        "integrated_density",
        "below_detection",
    ]
    assert "normalized" not in header and "fold_change" not in header
    assert "origin" not in header and "peak" not in header
    assert preview is not None and preview.exists()
    assert [row["sample"] for row in rows] == ["A", "B"]
    assert [row["integrated_density"] for row in rows] == ["40.0", "50.0"]
    assert {row["below_detection"] for row in rows} == {"0"}


def test_full_stats_keeps_the_legacy_fifteen_column_prefix(tmp_path: Path) -> None:
    csv_path, _ = export_result(
        fake_result(40.0, 50.0),
        tmp_path / "blot.tif",
        tmp_path / "out",
        origin=ORIGIN_ROI_BASELINE,
        full_stats=True,
    )
    header, rows = read_csv(csv_path)
    assert header[: len(LEGACY_BANDS_FIELDS)] == LEGACY_BANDS_FIELDS
    assert header == BANDS_CSV_FIELDS
    assert {row["origin"] for row in rows} == {ORIGIN_ROI_BASELINE}


def test_raw_stats_keep_technical_replicates(tmp_path: Path) -> None:
    result = fake_result(10.0, 11.0, 20.0)
    csv_path, _ = export_result(
        result,
        tmp_path / "blot.tif",
        tmp_path / "out",
        origin=ORIGIN_ROI_BASELINE,
        sample_names={1: "Ctrl", 2: "Ctrl", 3: "Treat"},
    )
    header, rows = read_csv(csv_path)
    assert header == COMPACT_BANDS_FIELDS
    ctrl = [row for row in rows if row["sample"] == "Ctrl"]
    assert len(ctrl) == 2, "同名样品出现几次就该有几行，不得合并"
    assert {row["lane"] for row in ctrl} == {"1", "2"}
    assert {row["integrated_density"] for row in ctrl} == {"10.0", "11.0"}
    assert rows[2]["sample"] == "Treat"
    assert "normalized" not in header and "fold_change" not in header


def test_session_raw_dump_follows_the_column_switch(tmp_path: Path) -> None:
    session = replicate_session()
    compact = export_session_raw_stats(session, tmp_path / "compact.csv")
    header, rows = read_csv(compact)
    assert header == list(COMPACT_SESSION_RAW_FIELDS)
    assert "normalized" not in header and "origin" not in header
    ctrl = [row for row in rows if row["sample"] == "Ctrl"]
    assert len(ctrl) == 2
    assert [row["integrated_density"] for row in ctrl] == ["10.0", "11.0"]
    assert all(row["protein"] == "ProteinX" for row in rows)

    full = export_session_raw_stats(
        session, tmp_path / "full.csv", full_stats=True
    )
    full_header, _ = read_csv(full)
    assert full_header == list(SESSION_RAW_FIELDS)
    assert full_header[: len(LEGACY_BANDS_FIELDS)] == LEGACY_BANDS_FIELDS


def test_raw_rows_never_drop_a_duplicate_sample_name() -> None:
    rows = build_raw_stat_rows(
        [measurement(1, 4.0), measurement(2, 4.0), measurement(3, 9.0)],
        sample_names={1: "A", 2: "A", 3: "A"},
    )
    assert [row.sample for row in rows] == ["A", "A", "A"]
    assert [row.lane for row in rows] == [1, 2, 3]
    assert [row.integrated_density for row in rows] == [4.0, 4.0, 9.0]


def test_selected_export_does_not_write_raw_long_by_default(tmp_path: Path) -> None:
    session = replicate_session()
    bundle = export_selected_data(
        tmp_path,
        raw_stats=True,
        normalized=False,
        summary=False,
        result=fake_result(10.0, 11.0, 20.0),
        image_path=tmp_path / "blot.tif",
        origin=ORIGIN_ROI_BASELINE,
        sample_names={1: "Ctrl", 2: "Ctrl", 3: "Treat"},
        session=session,
    )
    assert bundle.bands_csv is not None
    assert bundle.session_raw_csv is None
    assert bundle.normalized_csv is None
    assert bundle.summary is None
    header, _ = read_csv(bundle.bands_csv)
    assert header == COMPACT_BANDS_FIELDS
    assert not (tmp_path / "raw_long.csv").exists()
    assert not (tmp_path / "normalized_results.csv").exists()
    names = planned_data_export_names(
        "blot",
        raw_stats=True,
        normalized=False,
        summary=False,
        has_session=True,
        include_session_raw=False,
    )
    assert "raw_long.csv" not in names
    assert "normalized_results.csv" not in names


def test_selected_export_writes_raw_long_only_when_asked(tmp_path: Path) -> None:
    session = replicate_session()
    bundle = export_selected_data(
        tmp_path,
        raw_stats=True,
        result=fake_result(10.0, 11.0, 20.0),
        image_path=tmp_path / "blot.tif",
        session=session,
        include_session_raw=True,
        full_stats=False,
    )
    assert bundle.session_raw_csv is not None
    header, _ = read_csv(bundle.session_raw_csv)
    assert header == list(COMPACT_SESSION_RAW_FIELDS)
    names = planned_data_export_names(
        "blot", has_session=True, include_session_raw=True
    )
    assert "raw_long.csv" in names


def test_normalization_option_is_blocked_without_a_pair() -> None:
    session = replicate_session()
    ok, reason = normalization_export_available(session)
    assert not ok
    assert "内参" in reason
    summary_ok, summary_reason = summary_export_available(session)
    assert not summary_ok
    assert "内参" in summary_reason
    with pytest.raises(ValueError, match="归一化"):
        export_selected_data(Path("."), raw_stats=False, normalized=True, session=session)


def test_format_roi_id_is_stable() -> None:
    assert format_roi_id(RoiRect(1, 2, 3, 4)) == "x=1;y=2;w=3;h=4"
    assert format_roi_id(None) == ""
    assert format_roi_id(" already ") == "already"


def test_quantify_lanes_fills_peak() -> None:
    image = np.full((80, 90), 220, dtype=np.uint8)
    image[30:42, 20:40] = 40
    roi = RoiRect(10, 10, 70, 50)
    result = quantify_lanes(image, roi, default_separators(roi, 1), dark_bands=True)
    assert result.measurements
    assert all(item.peak != 0.0 for item in result.measurements)


def test_measure_band_roi_style_fills_peak() -> None:
    """手绘补画走 ROI 基线口径时，peak 必须按剖面残差写上，不能停在默认 0。"""
    pytest.importorskip("PySide6")
    from wb_analyzer.imageio import blot_from_array
    from wb_analyzer.ui.main_window import measure_band_roi_style

    image = np.full((80, 90), 220, dtype=np.uint8)
    image[30:42, 20:40] = 40
    blot = blot_from_array(image)
    roi = RoiRect(10, 10, 70, 50)
    added = measure_band_roi_style(blot, roi, 20, 30, 20, 12, True)
    assert added is not None
    assert added.integrated_density > 0.0
    assert added.peak != 0.0


def test_empty_raw_stats_are_rejected() -> None:
    with pytest.raises(ValueError, match="至少勾选"):
        export_selected_data(Path("."), raw_stats=False, normalized=False, summary=False)
    with pytest.raises(ValueError, match="原始灰度"):
        export_selected_data(Path("."), raw_stats=True)


# --------------------------------------------------------------- 对话框


@pytest.fixture
def qt_app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


def test_dialog_disables_normalization_when_unavailable(qt_app) -> None:
    from wb_analyzer.ui.data_export_dialog import DataExportDialog, DataExportOptions

    dialog = DataExportDialog(
        options=DataExportOptions(raw_stats=True, normalized=True, summary_tables=True),
        can_export_raw=True,
        can_export_normalized=False,
        can_export_summary=False,
        can_export_session_raw=False,
        normalized_reason="还没有可导出的归一化结果：同一分组需要同时有内参与目标蛋白。",
        summary_reason="还没有可导出的跨膜汇总。",
    )
    qt_app.processEvents()
    assert dialog.raw_check.isEnabled()
    assert dialog.raw_check.isChecked()
    assert not dialog.normalized_check.isEnabled()
    assert not dialog.normalized_check.isChecked()
    assert "内参" in dialog.normalized_check.toolTip()
    assert not dialog.summary_check.isEnabled()
    assert not dialog.session_raw_check.isEnabled()
    assert not dialog.session_raw_check.isChecked()
    assert not dialog.full_stats_check.isChecked()
    assert dialog.ok_button.isEnabled()
    dialog.raw_check.setChecked(False)
    qt_app.processEvents()
    assert not dialog.ok_button.isEnabled(), "可写项全空就没有文件可写"
    dialog.deleteLater()


def test_dialog_remembers_choices(qt_app, tmp_path: Path) -> None:
    from PySide6.QtCore import QSettings

    from wb_analyzer.ui.data_export_dialog import DataExportDialog, DataExportOptions

    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)
    dialog = DataExportDialog(
        options=DataExportOptions(raw_stats=True, normalized=False, summary_tables=False),
        can_export_raw=True,
        can_export_normalized=True,
        can_export_summary=True,
        can_export_session_raw=True,
    )
    dialog.normalized_check.setChecked(True)
    dialog.summary_check.setChecked(True)
    dialog.full_stats_check.setChecked(True)
    dialog.session_raw_check.setChecked(True)
    saved = dialog.save_options(store)
    assert saved.normalized and saved.summary_tables
    assert saved.full_stats and saved.session_raw
    dialog.deleteLater()

    loaded = DataExportDialog.load_options(store)
    assert loaded.raw_stats is True
    assert loaded.normalized is True
    assert loaded.summary_tables is True
    assert loaded.full_stats is True
    assert loaded.session_raw is True

    again = DataExportDialog(
        options=loaded,
        can_export_raw=True,
        can_export_normalized=True,
        can_export_summary=True,
        can_export_session_raw=True,
    )
    qt_app.processEvents()
    assert again.normalized_check.isChecked()
    assert again.summary_check.isChecked()
    assert again.full_stats_check.isChecked()
    assert again.session_raw_check.isChecked()
    again.deleteLater()


def test_dialog_offscreen_smoke(qt_app) -> None:
    from wb_analyzer.ui.data_export_dialog import DataExportDialog

    dialog = DataExportDialog(can_export_raw=True)
    dialog.show()
    qt_app.processEvents()
    assert dialog.windowTitle() == "导出数据"
    assert dialog.raw_check.isChecked()
    assert not dialog.full_stats_check.isChecked()
    assert not dialog.session_raw_check.isChecked()
    assert "积分灰度" in dialog.hint.text()
    options = dialog.options()
    assert options.raw_stats
    assert not options.normalized
    assert not options.summary_tables
    assert not options.full_stats
    assert not options.session_raw
    dialog.close()
    dialog.deleteLater()

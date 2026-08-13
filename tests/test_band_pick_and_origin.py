"""Provenance that travels with the numbers: background origin and band pick.

Two facts used to be lost the moment a quantification became a history entry.
Which background convention produced it — the two are not comparable, so a
target measured one way divided by a control measured the other way is a
plausible-looking number with no meaning. And which band of a doublet the lane
value stands for — summing a specific band with a non-specific one is the same
error one level down.
"""

import csv

import cv2
import numpy as np
import pytest

from wb_analyzer.analysis import BandMeasurement
from wb_analyzer.export import LEGACY_BANDS_FIELDS, export_result
from wb_analyzer.normalize import (
    BAND_PICK_ALL,
    BAND_PICK_BOTTOM,
    BAND_PICK_TOP,
    ORIGIN_LOCAL_BACKGROUND,
    ORIGIN_ROI_BASELINE,
    ORIGIN_UNKNOWN,
    AnalysisSession,
    LaneSeries,
    QuantEntry,
    coerce_origin,
    export_normalized_csv,
    lane_series_from_measurements,
)
from wb_analyzer.quant import quantify_lanes
from wb_analyzer.roi import RoiRect, default_separators

# ---------------------------------------------------------------- 合成数据


def band(lane: int, density: float, *, number: int = 1, y: int = 0) -> BandMeasurement:
    return BandMeasurement(
        lane=lane,
        band=number,
        x=lane * 10,
        y=y,
        width=10,
        height=10,
        area=100,
        mean_intensity=density / 100.0,
        background=1.0,
        integrated_density=density,
        saturated_fraction=0.0,
    )


def entry(kind: str, name: str, values: dict[int, float], **kwargs) -> QuantEntry:
    return QuantEntry(
        kind=kind,  # type: ignore[arg-type]
        name=name,
        group="G",
        intensities=list(values.values()),
        lanes=list(values),
        **kwargs,
    )


UPPER_TONES = (40, 90, 150)
LOWER_TONES = (170, 110, 50)
"""Lane 3's lower band is the strong one: a pick by position must not follow
the intensity ordering, which is the tempting-but-wrong implementation."""


def doublet_blot() -> np.ndarray:
    """Three lanes, two stacked bands each, strength crossing over."""
    image = np.full((200, 240), 230, dtype=np.uint8)
    for index, x in enumerate((20, 100, 180)):
        cv2.rectangle(image, (x, 40), (x + 40, 55), UPPER_TONES[index], -1)
        cv2.rectangle(image, (x, 110), (x + 40, 125), LOWER_TONES[index], -1)
    return image


def doublet_measurements() -> list[BandMeasurement]:
    gray = doublet_blot()
    roi = RoiRect(10, 20, 220, 160)
    result = quantify_lanes(gray, roi, default_separators(roi, 3), dark_bands=True)
    return list(result.measurements)


# ---------------------------------------------------------------- 口径标识


def test_origin_defaults_to_unknown_not_to_a_guess() -> None:
    fresh = entry("target", "T", {1: 10.0})
    assert fresh.origin == ORIGIN_UNKNOWN


def test_gui_spellings_are_accepted_as_aliases() -> None:
    assert coerce_origin("roi") == ORIGIN_ROI_BASELINE
    assert coerce_origin("detect") == ORIGIN_LOCAL_BACKGROUND
    assert coerce_origin("") == ORIGIN_UNKNOWN
    with pytest.raises(ValueError, match="未知的背景口径"):
        coerce_origin("rio")


def test_upsert_rejects_an_unrecognised_origin() -> None:
    session = AnalysisSession()
    with pytest.raises(ValueError, match="未知的背景口径"):
        session.upsert(entry("target", "T", {1: 10.0}, origin="ring-median"))


def test_cross_origin_pairing_is_refused_lane_by_lane() -> None:
    """ROI baseline control ÷ local background target = no number at all."""
    session = AnalysisSession()
    session.upsert(
        entry("loading_control", "Actin", {1: 100.0, 2: 200.0}, origin=ORIGIN_ROI_BASELINE)
    )
    session.upsert(
        entry("target", "PAD4", {1: 50.0, 2: 60.0}, origin=ORIGIN_LOCAL_BACKGROUND)
    )
    rows = {row.lane: row for row in session.normalize_all()}

    assert sorted(rows) == [1, 2], "拒绝配对仍要成行，和只在一侧出现的泳道一个待遇"
    for lane in (1, 2):
        assert rows[lane].normalized is None
        assert rows[lane].fold_change is None
        assert "口径" in rows[lane].note
        # 原始测量值是真的，留着；没有意义的是它们的商
        assert rows[lane].target is not None
        assert rows[lane].loading_control is not None
    assert any("口径" in w and "拒绝配对" in w for w in session.last_warnings)


def test_same_origin_pairs_without_complaint() -> None:
    session = AnalysisSession()
    session.upsert(
        entry("loading_control", "Actin", {1: 100.0}, origin=ORIGIN_ROI_BASELINE)
    )
    session.upsert(entry("target", "PAD4", {1: 50.0}, origin=ORIGIN_ROI_BASELINE))
    rows = session.normalize_all()
    assert rows[0].normalized == pytest.approx(0.5)
    assert rows[0].note == ""
    assert not [w for w in session.last_warnings if "口径" in w]


def test_unknown_origin_is_allowed_but_flagged() -> None:
    """An imported CSV keeps working; it just says what it cannot vouch for."""
    session = AnalysisSession()
    session.upsert(entry("loading_control", "Actin", {1: 100.0}))  # 旧 CSV：口径未知
    session.upsert(entry("target", "PAD4", {1: 50.0}, origin=ORIGIN_ROI_BASELINE))
    rows = session.normalize_all()
    assert rows[0].normalized == pytest.approx(0.5), "不能因为口径未知就不给数"
    assert rows[0].note == "", "未知不是错误，不该在结果行上打标"
    assert any("没有记录背景口径" in w for w in session.last_warnings)


def test_two_unknown_origins_stay_silent() -> None:
    """The all-legacy case is the status quo; a warning there is just noise."""
    session = AnalysisSession()
    session.upsert(entry("loading_control", "Actin", {1: 100.0}))
    session.upsert(entry("target", "PAD4", {1: 50.0}))
    session.normalize_all()
    assert not [w for w in session.last_warnings if "口径" in w]


# ---------------------------------------------------------------- 双条带策略


def test_three_strategies_aggregate_by_position_not_by_strength() -> None:
    measurements = doublet_measurements()
    assert len(measurements) == 6, "前置条件：每个泳道应分出上下两条带"

    merged = lane_series_from_measurements(measurements, band_pick=BAND_PICK_ALL)
    top = lane_series_from_measurements(measurements, band_pick=BAND_PICK_TOP)
    bottom = lane_series_from_measurements(measurements, band_pick=BAND_PICK_BOTTOM)

    assert merged.lanes == top.lanes == bottom.lanes == [1, 2, 3]
    for index in range(3):
        assert merged.intensities[index] == pytest.approx(
            top.intensities[index] + bottom.intensities[index]
        )
    # 上面那条越来越淡、下面那条越来越浓：按位置取才会得到这两条相反的趋势
    assert top.intensities[0] > top.intensities[1] > top.intensities[2]
    assert bottom.intensities[0] < bottom.intensities[1] < bottom.intensities[2]
    # 泳道 3 的下带比上带强，"上面那条"仍须取上带——按强弱取就会在这里翻车
    assert top.intensities[2] < bottom.intensities[2]
    assert top.used_bands == [[1], [1], [1]]
    assert bottom.used_bands == [[2], [2], [2]]
    assert merged.used_bands == [[1, 2], [1, 2], [1, 2]]


def test_unused_bands_are_still_carried() -> None:
    series = lane_series_from_measurements(doublet_measurements(), band_pick=BAND_PICK_TOP)
    assert len(series.bands) == 6, "没被采用的条带照样留在数据里，界面还要显示它"
    assert [b.band for b in series.bands_of(1)] == [1, 2]
    assert series.with_band_pick(BAND_PICK_ALL).intensities == pytest.approx(
        lane_series_from_measurements(doublet_measurements()).intensities
    )


def test_a_series_without_band_detail_says_so() -> None:
    bare = LaneSeries(lanes=[1], intensities=[10.0], backgrounds=[0.0], mean_intensities=[0.0])
    with pytest.raises(ValueError, match="重新定量"):
        bare.with_band_pick(BAND_PICK_TOP)


def test_switching_strategy_needs_no_requantification() -> None:
    """Re-aggregation runs off the entry's own band detail, not off the image."""
    measurements = doublet_measurements()
    expected = {
        pick: lane_series_from_measurements(measurements, band_pick=pick).intensities
        for pick in (BAND_PICK_ALL, BAND_PICK_TOP, BAND_PICK_BOTTOM)
    }
    session = AnalysisSession()
    session.upsert(
        QuantEntry.from_series(
            lane_series_from_measurements(measurements),
            kind="target",
            name="PAD4",
            group="G",
            origin=ORIGIN_ROI_BASELINE,
        )
    )
    del measurements  # 图和测量结果都不在手边了，切策略照样得成立

    for pick in (BAND_PICK_TOP, BAND_PICK_BOTTOM, BAND_PICK_ALL):
        updated = session.set_band_pick(0, pick)
        assert updated.band_pick == pick
        assert updated.intensities == pytest.approx(expected[pick])
        assert updated.origin == ORIGIN_ROI_BASELINE, "换策略不该动口径"
        assert len(updated.bands) == 6, "明细必须留着，否则只能切一次"


def test_lane_with_one_band_keeps_it_under_every_strategy() -> None:
    """"这个泳道只跑出一条带"和"我不要这条带"是两回事。"""
    measurements = [
        band(1, 100.0, number=1, y=10),
        band(1, 30.0, number=2, y=90),
        band(2, 70.0, number=1, y=12),  # 单条带泳道
    ]
    for pick in (BAND_PICK_TOP, BAND_PICK_BOTTOM):
        series = lane_series_from_measurements(measurements, band_pick=pick)
        assert series.lanes == [1, 2]
        assert series.intensities[1] == 70.0
        assert series.used_bands[1] == [1]


def test_three_bands_take_only_the_outermost_one() -> None:
    measurements = [
        band(1, 100.0, number=1, y=10),
        band(1, 50.0, number=2, y=50),
        band(1, 20.0, number=3, y=90),
    ]
    top = lane_series_from_measurements(measurements, band_pick=BAND_PICK_TOP)
    bottom = lane_series_from_measurements(measurements, band_pick=BAND_PICK_BOTTOM)
    assert top.intensities == [100.0], "最靠上的一条，不是上面两条求和"
    assert bottom.intensities == [20.0]


def test_band_numbers_do_not_decide_which_band_is_on_top() -> None:
    """Position is read off ``y``; the band number is just a label."""
    measurements = [band(1, 10.0, number=1, y=200), band(1, 90.0, number=2, y=20)]
    top = lane_series_from_measurements(measurements, band_pick=BAND_PICK_TOP)
    assert top.intensities == [90.0]
    assert top.used_bands == [[2]]


def test_control_and_target_can_use_opposite_ends_of_their_doublets() -> None:
    measurements = doublet_measurements()
    session = AnalysisSession()
    session.upsert(
        QuantEntry.from_series(
            lane_series_from_measurements(measurements, band_pick=BAND_PICK_BOTTOM),
            kind="loading_control",
            name="Actin",
            group="G",
            origin=ORIGIN_ROI_BASELINE,
        )
    )
    session.upsert(
        QuantEntry.from_series(
            lane_series_from_measurements(measurements, band_pick=BAND_PICK_TOP),
            kind="target",
            name="PAD4",
            group="G",
            origin=ORIGIN_ROI_BASELINE,
        )
    )
    rows = {row.lane: row for row in session.normalize_all()}
    top = lane_series_from_measurements(measurements, band_pick=BAND_PICK_TOP)
    bottom = lane_series_from_measurements(measurements, band_pick=BAND_PICK_BOTTOM)
    for index, lane in enumerate(top.lanes):
        assert rows[lane].normalized == pytest.approx(
            round(top.intensities[index] / bottom.intensities[index], 6)
        )
        assert rows[lane].target_bands == (1,)
        assert rows[lane].loading_control_bands == (2,)
    assert rows[1].target_band_pick == BAND_PICK_TOP
    assert rows[1].loading_control_band_pick == BAND_PICK_BOTTOM


def test_imported_entry_refuses_to_change_strategy(tmp_path) -> None:
    """No per-band detail in a CSV, so the honest answer is "re-quantify"."""
    session = AnalysisSession()
    session.upsert(
        QuantEntry.from_series(
            lane_series_from_measurements(doublet_measurements()),
            kind="loading_control",
            name="Actin",
            group="G",
            origin=ORIGIN_ROI_BASELINE,
        )
    )
    exported = session.export_loading_control_csv(tmp_path / "lc.csv")
    restored = AnalysisSession()
    restored.import_loading_control_csv(exported)
    with pytest.raises(ValueError, match="重新定量"):
        restored.set_band_pick(0, BAND_PICK_TOP)


# ---------------------------------------------------------------- CSV 导出


def test_bands_csv_records_the_convention(tmp_path) -> None:
    gray = doublet_blot()
    roi = RoiRect(10, 20, 220, 160)
    result = quantify_lanes(gray, roi, default_separators(roi, 3), dark_bands=True)
    csv_path, _ = export_result(
        result,
        tmp_path / "blot.tif",
        tmp_path / "out",
        origin=ORIGIN_ROI_BASELINE,
        full_stats=True,
    )
    lines = csv_path.read_text(encoding="utf-8-sig").strip().splitlines()
    header = next(csv.reader([lines[0]]))
    assert header[: len(LEGACY_BANDS_FIELDS)] == LEGACY_BANDS_FIELDS
    assert header[len(LEGACY_BANDS_FIELDS) - 1] == "origin"
    rows = list(csv.DictReader(lines))
    assert rows and all(row["origin"] == ORIGIN_ROI_BASELINE for row in rows)


def test_bands_csv_says_unknown_when_the_caller_did_not(tmp_path) -> None:
    gray = doublet_blot()
    roi = RoiRect(10, 20, 220, 160)
    result = quantify_lanes(gray, roi, default_separators(roi, 3), dark_bands=True)
    csv_path, _ = export_result(
        result, tmp_path / "blot.tif", tmp_path / "out", full_stats=True
    )
    assert ORIGIN_UNKNOWN in csv_path.read_text(encoding="utf-8-sig").splitlines()[1]


def test_loading_control_csv_round_trips_origin_and_pick(tmp_path) -> None:
    session = AnalysisSession()
    session.upsert(
        QuantEntry.from_series(
            lane_series_from_measurements(doublet_measurements(), band_pick=BAND_PICK_BOTTOM),
            kind="loading_control",
            name="Actin",
            group="G",
            origin=ORIGIN_ROI_BASELINE,
        )
    )
    exported = session.export_loading_control_csv(tmp_path / "lc.csv")
    header = exported.read_text(encoding="utf-8-sig").splitlines()[0]
    assert "origin" in header and "band_pick" in header and "bands_used" in header

    restored = AnalysisSession()
    assert restored.import_loading_control_csv(exported) == 1
    entry_back = restored.entries[0]
    assert entry_back.origin == ORIGIN_ROI_BASELINE
    assert entry_back.band_pick == BAND_PICK_BOTTOM
    assert entry_back.used_bands == [[2], [2], [2]]
    assert entry_back.intensities == pytest.approx(session.entries[0].intensities)


def test_csv_without_an_origin_column_imports_as_unknown(tmp_path) -> None:
    path = tmp_path / "legacy.csv"
    path.write_text(
        "kind,name,group,lane,integrated_density,background,mean_intensity,source\n"
        "loading_control,Actin,G,1,100,0,0,x\n",
        encoding="utf-8-sig",
    )
    session = AnalysisSession()
    session.import_loading_control_csv(path)
    assert session.entries[0].origin == ORIGIN_UNKNOWN


def test_garbled_origin_column_degrades_instead_of_aborting(tmp_path) -> None:
    path = tmp_path / "garbled.csv"
    path.write_text(
        "kind,name,group,lane,integrated_density,origin\n"
        "loading_control,Actin,G,1,100,ring median??\n",
        encoding="utf-8-sig",
    )
    session = AnalysisSession()
    assert session.import_loading_control_csv(path) == 1
    assert session.entries[0].origin == ORIGIN_UNKNOWN


def test_normalized_csv_carries_provenance_and_the_reason(tmp_path) -> None:
    session = AnalysisSession()
    session.upsert(
        entry("loading_control", "Actin", {1: 100.0}, origin=ORIGIN_ROI_BASELINE)
    )
    session.upsert(
        entry("target", "PAD4", {1: 50.0}, origin=ORIGIN_LOCAL_BACKGROUND)
    )
    out = export_normalized_csv(session.normalize_all(), tmp_path / "norm.csv")
    header, row = out.read_text(encoding="utf-8-sig").strip().splitlines()

    columns = header.split(",")
    assert columns[:8] == [
        "group",
        "lane",
        "loading_control_name",
        "loading_control",
        "target_name",
        "target",
        "normalized",
        "fold_change",
    ], "新列一律追加在后面，按列序读这个文件的东西不能被挪位"
    values = dict(zip(columns, row.split(","), strict=True))
    assert values["loading_control_origin"] == ORIGIN_ROI_BASELINE
    assert values["target_origin"] == ORIGIN_LOCAL_BACKGROUND
    assert values["normalized"] == ""
    assert "口径" in values["note"], "空的比值必须在文件里自带原因"

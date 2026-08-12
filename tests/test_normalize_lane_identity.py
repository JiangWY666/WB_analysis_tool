"""Normalization must pair a target with its loading control by lane number.

Aggregating bands into a bare positional list loses which lane each value came
from. Delete the band in lane 3 and every later value silently slides onto its
neighbour's loading control, so lane 4 gets divided by lane 3's actin - with no
error, no warning, and a plausible-looking number.
"""

import pytest

from wb_analyzer.analysis import BandMeasurement
from wb_analyzer.normalize import (
    AnalysisSession,
    QuantEntry,
    lane_series_from_measurements,
)


def band(lane: int, density: float) -> BandMeasurement:
    return BandMeasurement(
        lane=lane,
        band=1,
        x=lane * 10,
        y=0,
        width=10,
        height=10,
        area=100,
        mean_intensity=density / 100.0,
        background=0.0,
        integrated_density=density,
        saturated_fraction=0.0,
    )


ALL_LANES = [band(n, 100.0 * n) for n in range(1, 6)]
MISSING_LANE_3 = [m for m in ALL_LANES if m.lane != 3]


def test_lane_series_keeps_the_gap_visible() -> None:
    series = lane_series_from_measurements(MISSING_LANE_3)
    assert series.lanes == [1, 2, 4, 5]
    assert series.intensities == [100.0, 200.0, 400.0, 500.0]
    assert len(series.backgrounds) == len(series.mean_intensities) == 4


def test_deleted_band_no_longer_shifts_the_pairing() -> None:
    """Lane 4 must be divided by lane 4 of the control, not lane 3."""
    session = AnalysisSession()
    session.upsert(
        QuantEntry.from_series(
            lane_series_from_measurements(ALL_LANES),
            kind="loading_control", name="Actin", group="G",
        )
    )
    session.upsert(
        QuantEntry.from_series(
            lane_series_from_measurements(MISSING_LANE_3),
            kind="target", name="Target", group="G",
        )
    )
    rows = {row.lane: row for row in session.normalize_all()}
    assert sorted(rows) == [1, 2, 3, 4, 5]
    for lane in (1, 2, 4, 5):
        assert rows[lane].target == 100.0 * lane
        assert rows[lane].loading_control == 100.0 * lane
        assert rows[lane].normalized == pytest.approx(1.0)


def test_unpairable_lane_is_reported_empty_and_warned() -> None:
    session = AnalysisSession()
    session.upsert(
        QuantEntry.from_series(
            lane_series_from_measurements(ALL_LANES),
            kind="loading_control", name="Actin", group="G",
        )
    )
    session.upsert(
        QuantEntry.from_series(
            lane_series_from_measurements(MISSING_LANE_3),
            kind="target", name="Target", group="G",
        )
    )
    rows = {row.lane: row for row in session.normalize_all()}
    assert rows[3].target is None
    assert rows[3].normalized is None
    assert rows[3].note
    assert any("3" in warning for warning in session.last_warnings)


def test_positional_entries_still_work_but_warn() -> None:
    """Legacy callers that never recorded lane numbers keep the old behaviour."""
    session = AnalysisSession()
    session.upsert(
        QuantEntry(kind="loading_control", name="Actin", group="G",
                   intensities=[100.0, 200.0, 300.0, 400.0, 500.0])
    )
    session.upsert(
        QuantEntry(kind="target", name="Target", group="G",
                   intensities=[100.0, 200.0, 400.0, 500.0])
    )
    rows = {row.lane: row for row in session.normalize_all()}
    # Old positional meaning: the 3rd value pairs with the 3rd control.
    assert rows[3].target == 400.0
    assert rows[3].loading_control == 300.0
    assert any("泳道号" in warning for warning in session.last_warnings)


def test_upsert_rejects_inconsistent_lane_numbers() -> None:
    session = AnalysisSession()
    with pytest.raises(ValueError, match="不一致"):
        session.upsert(
            QuantEntry(kind="target", name="T", group="G",
                       intensities=[1.0, 2.0], lanes=[1, 2, 3])
        )
    with pytest.raises(ValueError, match="重复"):
        session.upsert(
            QuantEntry(kind="target", name="T", group="G",
                       intensities=[1.0, 2.0], lanes=[2, 2])
        )


def test_loading_control_csv_roundtrip_survives_shuffling(tmp_path) -> None:
    """Row order in the file must not decide which lane a value belongs to."""
    path = tmp_path / "shuffled.csv"
    path.write_text(
        "kind,name,group,lane,integrated_density,background,mean_intensity,source\n"
        "loading_control,Actin,G,3,300,13,3,x\n"
        "loading_control,Actin,G,1,100,11,1,x\n"
        "loading_control,Actin,G,5,500,15,5,x\n"
        "loading_control,Actin,G,2,200,12,2,x\n"
        "loading_control,Actin,G,4,400,14,4,x\n",
        encoding="utf-8-sig",
    )
    session = AnalysisSession()
    assert session.import_loading_control_csv(path) == 1
    entry = session.entries[0]
    assert entry.lanes == [1, 2, 3, 4, 5]
    assert entry.intensities == [100.0, 200.0, 300.0, 400.0, 500.0]
    assert entry.backgrounds == [11.0, 12.0, 13.0, 14.0, 15.0]


def test_loading_control_csv_gap_is_kept_and_warned(tmp_path) -> None:
    path = tmp_path / "gap.csv"
    path.write_text(
        "kind,name,group,lane,integrated_density,background,mean_intensity,source\n"
        "loading_control,Actin,G,1,100,0,0,x\n"
        "loading_control,Actin,G,2,200,0,0,x\n"
        "loading_control,Actin,G,4,400,0,0,x\n",
        encoding="utf-8-sig",
    )
    session = AnalysisSession()
    session.import_loading_control_csv(path)
    assert session.entries[0].lanes == [1, 2, 4]
    assert any("3" in warning for warning in session.last_warnings)


def test_export_then_import_preserves_lane_numbers(tmp_path) -> None:
    session = AnalysisSession()
    session.upsert(
        QuantEntry(kind="loading_control", name="Actin", group="A",
                   intensities=[100.0, 90.0], backgrounds=[20.0, 19.0],
                   mean_intensities=[50.0, 48.0], lanes=[2, 5])
    )
    exported = session.export_loading_control_csv(tmp_path / "lc.csv")
    restored = AnalysisSession()
    assert restored.import_loading_control_csv(exported) == 1
    assert restored.entries[0].lanes == [2, 5]
    assert restored.entries[0].intensities == [100.0, 90.0]


def test_csv_without_a_lane_column_falls_back_to_file_order(tmp_path) -> None:
    path = tmp_path / "nolane.csv"
    path.write_text(
        "kind,name,group,integrated_density,background,mean_intensity\n"
        "loading_control,Actin,G,100,0,0\n"
        "loading_control,Actin,G,200,0,0\n",
        encoding="utf-8-sig",
    )
    session = AnalysisSession()
    session.import_loading_control_csv(path)
    assert session.entries[0].intensities == [100.0, 200.0]
    assert session.entries[0].lanes == [1, 2]

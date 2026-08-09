from pathlib import Path

import pytest

from wb_analyzer.analysis import AnalysisConfig, analyze_image

DEMO = Path(__file__).resolve().parents[1] / "pad4_001(1)_signal.tif"


@pytest.mark.skipif(not DEMO.exists(), reason="demo blot image not present")
def test_demo_signal_detects_eight_bands() -> None:
    result = analyze_image(DEMO, AnalysisConfig(background_mode="auto"))
    assert result.dark_bands
    assert len(result.measurements) == 8
    assert len({item.lane for item in result.measurements}) == 8

    ordered = sorted(result.measurements, key=lambda item: item.lane)
    dens = [item.integrated_density for item in ordered]
    # Four strong/weak pairs in one row.
    assert dens[0] > dens[1]
    assert dens[2] > dens[3]
    assert dens[4] > dens[5]
    assert dens[6] > dens[7]
    assert all(value > 0 for value in dens)

    centers_y = [item.y + item.height / 2 for item in ordered]
    assert max(centers_y) - min(centers_y) < 50

    # Boxes should stay separated left-to-right without giant merges.
    widths = [item.width for item in ordered]
    assert max(widths) < 90
    assert min(widths) >= 20

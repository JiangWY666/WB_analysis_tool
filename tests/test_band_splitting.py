"""Where ``segment_bands`` draws the line between one band and two.

The geometric 30% valley test was the only thing deciding this, and it had no
coverage at all: the synthetic doublet the older tests use sits 70 px apart
with a 21 px FWHM (d/FWHM = 3.3), so every ``valley_drop`` from 0.30 down to
0.03 passed them. These tests pin both edges of the real decision — an
overlapping pair at the Rayleigh limit that must split, and a weak single band
that must not — plus the ``expected_bands`` override.
"""

from __future__ import annotations

import numpy as np
import pytest

from wb_analyzer.imageio import blot_from_array
from wb_analyzer.quant import (
    compute_lane_profiles,
    quantify_lanes,
    rubber_band_baseline,
    segment_bands,
)
from wb_analyzer.roi import RoiRect

NOISE = 40.0
"""Per-sample noise of the synthetic profiles, in profile units."""


def gaussian(size: int, center: float, fwhm: float, amplitude: float) -> np.ndarray:
    sigma = fwhm / 2.3548
    y = np.arange(size, dtype=float)
    return amplitude * np.exp(-0.5 * ((y - center) / sigma) ** 2)


def corrected_profile(profile: np.ndarray) -> tuple[np.ndarray, float]:
    """Run a raw profile through the same preparation ``quant`` uses.

    Returns the baseline-corrected profile and the noise scale that goes with
    it, so the tests exercise the values :func:`compute_lane_profiles` would
    actually hand to :func:`segment_bands`.
    """
    from wb_analyzer.quant import _noise_scale

    residual = profile - rubber_band_baseline(profile)
    return np.maximum(residual, 0.0), _noise_scale(residual)


def overlapping_doublet(seed: int, size: int = 160) -> np.ndarray:
    """Two bands 12 px apart with a 10 px FWHM — d/FWHM = 1.2.

    That is the geometry measured on a real blot whose doublets the 30% test
    could not resolve: FWHM 8.5-11.3 px, centres 11-14 px apart, amplitude
    ratio 0.26-0.70. A pair this close is at the Rayleigh limit, where the
    valley only comes down to ~96% of the smaller peak.
    """
    rng = np.random.default_rng(seed)
    profile = 4000.0 + gaussian(size, 72.0, 10.0, 8000.0) + gaussian(size, 84.0, 10.0, 4000.0)
    return profile + rng.normal(0.0, NOISE, size)


@pytest.mark.parametrize("seed", range(6))
def test_overlapping_doublet_splits(seed: int) -> None:
    corrected, sigma = corrected_profile(overlapping_doublet(seed))
    segments = segment_bands(corrected, noise_sigma=sigma)
    assert len(segments) == 2
    (a1, a2), (b1, b2) = segments
    assert a1 < 72 < a2 <= b1 < 84 < b2


@pytest.mark.parametrize("seed", range(6))
def test_overlapping_doublet_needs_the_statistical_test(seed: int) -> None:
    """Geometry alone cannot resolve this pair, and must not pretend to.

    ``noise_sigma=0`` is exactly the geometry-only configuration, and it is
    what :func:`wb_analyzer.quant._vertical_segments` passes when it locates a
    band footprint. Splitting there on a zero sigma would make every dip
    infinitely significant.
    """
    corrected, sigma = corrected_profile(overlapping_doublet(seed))
    assert len(segment_bands(corrected)) == 1
    assert len(segment_bands(corrected, noise_sigma=sigma)) == 2


@pytest.mark.parametrize("amplitude", [600.0, 900.0, 1500.0, 3000.0])
@pytest.mark.parametrize("seed", range(5))
def test_weak_single_band_is_not_split(amplitude: float, seed: int) -> None:
    """The lower bound: one faint band stays one band.

    This is what a looser ``valley_drop`` costs. Dropping it to the 0.036 that
    would catch every doublet above turns these same profiles into two "bands"
    several percent of the time.
    """
    rng = np.random.default_rng(1000 + seed)
    profile = 4000.0 + gaussian(160, 72.0, 11.0, amplitude) + rng.normal(0.0, NOISE, 160)
    corrected, sigma = corrected_profile(profile)
    assert len(segment_bands(corrected, noise_sigma=sigma)) == 1


@pytest.mark.parametrize("order", [4, 6, 8, 10])
def test_flat_top_band_is_not_split(order: int) -> None:
    """A plateau band's ripples are not a doublet however flat its top is."""
    rng = np.random.default_rng(2000 + order)
    y = np.arange(160, dtype=float)
    profile = 4000.0 + 6000.0 * np.exp(-0.5 * np.abs((y - 80.0) / 12.0) ** order)
    corrected, sigma = corrected_profile(profile + rng.normal(0.0, NOISE, 160))
    assert len(segment_bands(corrected, noise_sigma=sigma)) == 1


@pytest.mark.parametrize("seed", range(8))
def test_blank_lane_has_no_bands(seed: int) -> None:
    rng = np.random.default_rng(3000 + seed)
    corrected, sigma = corrected_profile(4000.0 + rng.normal(0.0, NOISE, 160))
    assert segment_bands(corrected, noise_sigma=sigma) == []


@pytest.mark.parametrize("seed", range(4))
def test_significant_dip_still_needs_to_be_a_real_dip(seed: int) -> None:
    """:data:`~wb_analyzer.quant.SEGMENT_SPLIT_MAX_RATIO` in isolation.

    A 1.6% notch on a strong band is 26 sigma deep — overwhelmingly
    significant, and physically nothing. Only the relative floor stops the
    statistical test from cutting there.
    """
    rng = np.random.default_rng(4000 + seed)
    y = np.arange(200, dtype=float)
    plateau = 50000.0 * np.exp(-0.5 * np.abs((y - 100.0) / 18.0) ** 8)
    profile = plateau - 800.0 * np.exp(-0.5 * ((y - 100.0) / 3.0) ** 2)
    corrected, sigma = corrected_profile(profile + rng.normal(0.0, NOISE, 200))
    assert len(segment_bands(corrected, noise_sigma=sigma)) == 1
    # Deepen the notch past the 2% floor and it does become a split.
    deep = plateau - 1600.0 * np.exp(-0.5 * ((y - 100.0) / 3.0) ** 2)
    corrected, sigma = corrected_profile(deep + rng.normal(0.0, NOISE, 200))
    assert len(segment_bands(corrected, noise_sigma=sigma)) == 2


def three_equal_peaks() -> np.ndarray:
    """One run, three peaks, two valleys of very different depth.

    50 <-> 72 is 1.4 FWHM apart (valley at 55% of the peaks); 72 <-> 108 is
    2.25 FWHM apart (valley at 6%).
    """
    return (
        gaussian(200, 50.0, 16.0, 1000.0)
        + gaussian(200, 72.0, 16.0, 1000.0)
        + gaussian(200, 108.0, 16.0, 1000.0)
    )


@pytest.mark.parametrize("seed", range(4))
def test_expected_bands_forces_a_split_the_criteria_refuse(seed: int) -> None:
    """The override exists for doublets no automatic test can be asked to see.

    A 60-sample ROI wrapped tight around this pair is the hard case: the band
    fills half the window, so the successive-difference noise scale reads about
    three times high and the statistical test goes quiet with it. The user
    knows there are two bands; saying so is all it should take.
    """
    rng = np.random.default_rng(seed)
    profile = 5000.0 + gaussian(60, 24.0, 10.0, 8000.0) + gaussian(60, 36.0, 10.0, 4000.0)
    corrected, sigma = corrected_profile(profile + rng.normal(0.0, NOISE, 60))
    assert len(segment_bands(corrected, noise_sigma=sigma)) == 1
    forced = segment_bands(corrected, noise_sigma=sigma, expected_bands=2)
    assert len(forced) == 2
    (a1, a2), (b1, b2) = forced
    assert a1 < 24 < a2 <= b1 < 36 < b2


def test_expected_bands_cuts_the_deepest_valleys_first() -> None:
    profile = three_equal_peaks()
    assert len(segment_bands(profile)) == 3
    # One cut to make: it goes to the deep valley near 90, not the shallow one
    # near 61, so the two touching peaks stay together.
    (first, second) = segment_bands(profile, expected_bands=2)
    assert first[0] < 50 < 72 < first[1]
    assert second[0] < 108 < second[1]
    assert 85 < first[1] < 95
    assert len(segment_bands(profile, expected_bands=3)) == 3


def test_expected_bands_never_removes_a_split_it_cannot_reach() -> None:
    """It only adds cuts. Asking for fewer bands than there are runs is a no-op.

    Two bands separated by blank lane are two runs before any valley is looked
    at, and nothing in this function can glue them back together.
    """
    profile = three_equal_peaks()
    assert len(segment_bands(profile, expected_bands=1)) == 1  # one run here
    separated = gaussian(200, 50.0, 12.0, 1000.0) + gaussian(200, 150.0, 12.0, 1000.0)
    assert len(segment_bands(separated, expected_bands=1)) == 2


def test_expected_bands_saturates_when_there_are_no_more_valleys() -> None:
    profile = three_equal_peaks()
    assert len(segment_bands(profile, expected_bands=9)) == 3


def test_expected_bands_must_be_at_least_one() -> None:
    profile = three_equal_peaks()
    with pytest.raises(ValueError, match="≥ 1"):
        segment_bands(profile, expected_bands=0)


def doublet_blot16(lanes: int = 4, *, seed: int = 0, height: int = 140) -> np.ndarray:
    """16-bit light-background blot carrying one overlapping doublet per lane."""
    rng = np.random.default_rng(seed)
    width = 20 + lanes * 60
    raw = np.full((height, width), 52000.0)
    for index in range(lanes):
        x0 = 20 + index * 60
        depth = 9000.0 * (1.0 - 0.12 * index)
        column = gaussian(height, height * 0.42, 10.0, depth)
        column += gaussian(height, height * 0.42 + 12.0, 10.0, depth * 0.5)
        raw[:, x0 : x0 + 34] -= column[:, None]
    return np.clip(raw + rng.normal(0.0, 100.0, raw.shape), 0, 65535).astype(np.uint16)


def doublet_lane_setup(seed: int) -> tuple[np.ndarray, RoiRect, list[float], float]:
    blot = blot_from_array(doublet_blot16(seed=seed))
    roi = RoiRect(5, 5, blot.width - 10, blot.height - 10)
    separators = [20.0 + 60.0 * i - 13.0 for i in range(1, 4)]
    return blot.raw, roi, separators, blot.white_level


@pytest.mark.parametrize("seed", range(4))
def test_overlapping_doublets_survive_the_whole_lane_pipeline(seed: int) -> None:
    raw, roi, separators, white_level = doublet_lane_setup(seed)
    lanes = compute_lane_profiles(
        raw, roi, separators, dark_bands=True, white_level=white_level
    )
    assert [len(lane.segments) for lane in lanes] == [2, 2, 2, 2]
    for lane in lanes:
        # The injected ratio is 2.0 and the measured one lands near 2.3: a cut
        # at the valley is not a deconvolution, so on an overlapping pair each
        # band keeps some of its neighbour's flank and the stronger one keeps
        # more. Splitting recovers *which* bands are there, not their exact
        # share of the overlap.
        strong, weak = lane.band_densities
        assert 1.8 < strong / weak < 3.0


def test_expected_bands_reaches_compute_lane_profiles_and_quantify_lanes() -> None:
    raw, roi, separators, white_level = doublet_lane_setup(0)
    options = {"dark_bands": True, "white_level": white_level}
    merged = compute_lane_profiles(raw, roi, separators, expected_bands=1, **options)
    assert [len(lane.segments) for lane in merged] == [1, 1, 1, 1]
    result = quantify_lanes(raw, roi, separators, expected_bands=1, **options)
    assert len(result.measurements) == 4
    split = quantify_lanes(raw, roi, separators, expected_bands=2, **options)
    assert len(split.measurements) == 8
    assert [m.band for m in split.measurements] == [1, 2] * 4

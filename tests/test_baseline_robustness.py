"""Regression tests: a scratch must not drag the baseline out from under a band.

A convex-hull ("rubber band") baseline is anchored by the lowest points of the
profile, so a single dust shadow or scratch pulls the whole chord down and
inflates every band in the lane. These are golden tests against a synthetic
profile with an exactly known peak area.
"""

import numpy as np
import pytest

from wb_analyzer.quant import rubber_band_baseline, valley_baseline

PROFILE_LENGTH = 400
BAND_CENTRE = 200.0
BAND_SIGMA = 9.0
BAND_AMPLITUDE = 3000.0
BACKGROUND = 5000.0
SCRATCH_AT = 60


def synthetic_profile(noise: float = 0.0, seed: int = 0) -> tuple[np.ndarray, float]:
    """Flat background plus one Gaussian band; returns (profile, true band area)."""
    x = np.arange(PROFILE_LENGTH, dtype=float)
    band = BAND_AMPLITUDE * np.exp(-0.5 * ((x - BAND_CENTRE) / BAND_SIGMA) ** 2)
    profile = BACKGROUND + band
    if noise:
        profile = profile + np.random.default_rng(seed).normal(0.0, noise, PROFILE_LENGTH)
    return profile, float(band.sum())


def measured_area(profile: np.ndarray) -> float:
    return float(np.maximum(profile - rubber_band_baseline(profile), 0.0).sum())


@pytest.mark.parametrize("width", [1, 3, 5, 9])
def test_scratch_does_not_inflate_band_area(width: int) -> None:
    """A narrow dark scratch far from the band must not change its integral."""
    profile, true_area = synthetic_profile()
    profile[SCRATCH_AT : SCRATCH_AT + width] -= 2500.0
    error = abs(measured_area(profile) - true_area) / true_area
    assert error < 0.05, f"{width}px scratch skewed the integral by {error:.1%}"


def test_scratch_matches_the_clean_profile_exactly() -> None:
    """The 3-pixel case from the review: was +246%, must now be indistinguishable."""
    clean, true_area = synthetic_profile()
    scratched = clean.copy()
    scratched[SCRATCH_AT : SCRATCH_AT + 3] -= 2500.0
    assert measured_area(scratched) == pytest.approx(measured_area(clean), rel=1e-9)
    assert measured_area(scratched) == pytest.approx(true_area, rel=0.01)


@pytest.mark.parametrize("noise", [5.0, 25.0, 100.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_scratch_costs_nothing_beyond_the_noise_floor(noise: float, seed: int) -> None:
    """With noise present, a scratch must add no error over the clean control.

    The clean profile itself is biased upward at high noise (rectification of
    the noise floor by ``maximum(..., 0)``), which is a separate issue; what
    matters here is that the scratch contributes nothing on top of it.
    """
    clean, _ = synthetic_profile(noise=noise, seed=seed)
    scratched = clean.copy()
    scratched[SCRATCH_AT : SCRATCH_AT + 3] -= 2500.0
    clean_area = measured_area(clean)
    assert abs(measured_area(scratched) - clean_area) / clean_area < 0.02


def test_clean_sloped_baseline_stays_accurate() -> None:
    """The outlier veto must not disturb an ordinary sloped background."""
    x = np.arange(300, dtype=float)
    baseline_true = 50.0 + 0.3 * x
    peak = 80.0 * np.exp(-0.5 * ((x - 150) / 10.0) ** 2)
    profile = baseline_true + peak
    baseline = rubber_band_baseline(profile)
    corrected = np.maximum(profile - baseline, 0.0)
    assert abs(float(corrected.sum()) - float(peak.sum())) / float(peak.sum()) < 0.06
    assert abs(baseline[20] - baseline_true[20]) < 4.0
    assert abs(baseline[280] - baseline_true[280]) < 4.0


def test_clean_profile_is_bit_identical_to_the_plain_rubber_band() -> None:
    """No outliers means no veto: the hull must be exactly the classic one."""
    profile, _ = synthetic_profile(noise=20.0, seed=7)
    from wb_analyzer.quant import _lower_hull, _smooth

    radius = max(1, profile.size // 100)
    smooth = _smooth(profile, radius)
    hull = _lower_hull(smooth, np.arange(profile.size))
    plain = np.interp(np.arange(profile.size), hull, smooth[hull])
    assert np.array_equal(rubber_band_baseline(profile), plain)


def test_valley_baseline_also_rejects_the_scratch() -> None:
    """Local minima are as easy to fake with a scratch as hull vertices are."""
    profile, true_area = synthetic_profile()
    scratched = profile.copy()
    scratched[SCRATCH_AT : SCRATCH_AT + 3] -= 2500.0
    area = float(np.maximum(scratched - valley_baseline(scratched), 0.0).sum())
    assert abs(area - true_area) / true_area < 0.05

"""ImageJ-style lane-profile quantification on native bit-depth data.

The workflow mirrors the classic Gel Analyzer: each lane strip is collapsed
into a vertical intensity profile, a baseline (rubber-band lower convex hull
or valley-to-valley) is subtracted, and the area under each remaining peak is
the band's integrated density. All math runs on the raw (possibly 16-bit)
data; the 8-bit display layer is only used for annotation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .analysis import AnalysisResult, BandMeasurement, renumber_measurements
from .imageio import require_finite, saturation_fraction
from .roi import RoiRect, lane_boundaries

ROI_COLOR = (255, 229, 0)  # cyan-ish (BGR)
SEPARATOR_COLOR = (0, 180, 255)  # orange (BGR)
BAND_COLOR = (60, 60, 255)  # red (BGR)
BAND_SATURATED_COLOR = (0, 140, 255)  # orange warning (BGR)
BAND_WEAK_COLOR = (200, 160, 90)  # steel blue: below the detection limit (BGR)

SEGMENT_EDGE_FRAC = 0.02
"""Band segment edges are trimmed where the signal falls to ~2% of the peak."""


@dataclass(frozen=True)
class LaneProfile:
    """Per-lane vertical profile, baseline, and band segmentation."""

    lane: int
    """1-based lane number, left to right."""

    x1: int
    x2: int
    """Absolute pixel span actually integrated: [x1, x2).

    With ``tight=True``（默认）这是收紧后的条带横向足迹——泳道条内两侧的
    空白不参与积分；找不到可靠条带峰（空白/平坦泳道）或 ``tight=False``
    时退回整个泳道条 [分隔线, 分隔线)。
    """

    y0: int
    """ROI top; profile[i] corresponds to image row y0 + i."""

    profile: np.ndarray
    """Signal profile along y (sum across strip width per row)."""

    baseline: np.ndarray
    segments: list[tuple[int, int]]
    """Relative [start, end) y ranges, one per band."""

    total_density: float
    """Whole-lane integral: ``sum(profile - baseline - noise_floor)``, clamped at 0."""

    band_densities: list[float]
    """Integral of each segment, aligned with ``segments``, same correction."""

    noise_floor: float = 0.0
    """How far the baseline sits below the true background, per sample.

    Measured as the mean of ``profile - baseline`` over the rows with no detected
    band, and subtracted from every sample that goes into an integral. Without
    it the integral grows with ROI height on its own; see :func:`_lane_noise`.
    """

    noise_sigma: float = 0.0
    """Per-sample noise of the baseline residual, in profile units."""

    band_detection_limits: list[float] = field(default_factory=list)
    """3σ detection limit of each segment, aligned with ``band_densities``."""


def _smooth(values: np.ndarray, radius: int) -> np.ndarray:
    """Moving average with edge padding (no zero-pad droop at the ends)."""
    radius = max(1, int(radius))
    kernel = np.ones(radius * 2 + 1, dtype=float)
    kernel /= kernel.sum()
    padded = np.pad(values.astype(float), radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _median_filter(values: np.ndarray, size: int) -> np.ndarray:
    """Sliding-window median with edge padding.

    Gives each sample the level its neighbourhood would have without it, which
    is how narrow spikes (dust, scratches, hot pixels) are spotted: a median of
    width ``size`` ignores anything narrower than ``size // 2``, so the gap
    between a sample and this curve is exactly the spike's depth.
    """
    values = np.asarray(values, dtype=float)
    size = max(1, int(size)) | 1
    if size <= 1 or values.size < 3:
        return values.copy()
    size = min(size, values.size if values.size % 2 else values.size - 1)
    radius = size // 2
    padded = np.pad(values, radius, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, size)
    return np.median(windows, axis=-1)


TAIL_ASYMMETRY = 1.5
"""One distribution tail must be this much longer than the other to call polarity."""


def resolve_dark_bands_native(
    raw: np.ndarray, white_level: float, mode: str = "auto"
) -> bool:
    """Decide band polarity on native-depth data.

    The primary test is scale-free: bands live in one tail of the intensity
    distribution, so whichever tail is clearly longer tells us which way round
    the blot is. This works regardless of how much of the storage container
    the sensor actually filled. Only a frame with no usable contrast (both
    tails equal, e.g. a blank field) falls back to comparing the border tone
    against ``white_level``.
    """
    if mode == "dark_bands":
        return True
    if mode == "light_bands":
        return False
    median = float(np.median(raw))
    low = float(np.percentile(raw, 0.5))
    high = float(np.percentile(raw, 99.5))
    dark_tail = median - low
    light_tail = high - median
    if dark_tail > light_tail * TAIL_ASYMMETRY:
        return True
    if light_tail > dark_tail * TAIL_ASYMMETRY:
        return False
    # Featureless frame: nothing to do but read the absolute border tone.
    border = np.concatenate((raw[0], raw[-1], raw[:, 0], raw[:, -1]))
    border_med = float(np.median(border))
    if border_med >= white_level * (140.0 / 255.0):
        return True
    if border_med <= white_level * (115.0 / 255.0):
        return False
    return bool(float(np.mean(raw)) >= white_level / 2.0)


BASELINE_DESPIKE_FRAC = 20
"""Neighbourhood width for outlier detection, as a fraction 1/N of the profile.

Sets how wide an artifact can be and still be rejected: roughly half the
window. At 1/20 of a 400-sample profile that is a ~9 pixel scratch, comfortably
below band scale, and every clean-profile result is unchanged by it.
"""

BASELINE_DESPIKE_MAX = 31
"""Upper bound on that neighbourhood width, in samples."""


BASELINE_DESPIKE_SIGMA = 6.0
"""How far below its neighbourhood a sample must fall to be called an outlier."""

BASELINE_DESPIKE_RANGE = 0.05
"""...and how much of the profile's own band amplitude the dip must also span."""


def _noise_scale(profile: np.ndarray) -> float:
    """Per-sample noise of a profile, from the MAD of successive differences.

    Measuring the noise this way is immune to the profile's own shape (a band
    is smooth, so it barely contributes to neighbour-to-neighbour steps) and,
    unlike a MAD taken across the whole profile, it does not collapse to zero
    on a near-blank lane whose values are constant over half its length. That
    collapse is what would otherwise let ordinary quantisation noise be
    mistaken for scratches, swinging the residual integral of an empty lane by
    tens of percent.
    """
    if profile.size < 3:
        return 0.0
    steps = np.abs(np.diff(profile))
    return 1.4826 * float(np.median(steps)) / np.sqrt(2.0)


LOD_SIGMA = 3.0
"""How many noise sigmas an integral must clear to count as a real band."""

NOISE_BAND_MARGIN = 0.5
"""Blank rows start half a segment height away from each band.

``segment_bands`` cuts at 4% of the peak, so the rows just outside a segment
still carry the band's vertical tails. Counting those as "blank" would read the
tails as noise and over-estimate the floor.
"""

MIN_NOISE_SAMPLES = 12
"""Fewer blank rows than this and the floor is left at zero rather than guessed."""

FLOOR_SIGMA_CAP = 3.0
"""Hard ceiling on the noise floor, in sigmas.

Whatever the baseline does, noise alone cannot leave more than a few sigma per
sample behind. Anything above that in the "blank" rows is signal the segmenter
missed — a band's vertical tail, a smear — and subtracting it would be eating
real data. Capping also makes the noise-free case exact: sigma is 0 there, so
the floor is 0 and every number comes out bit-for-bit as before.

The cap does bind on real scans, where neighbouring rows are correlated and the
successive-difference sigma therefore reads low: on a chemiluminescence blot the
blank rows sat 14-53 units above the baseline while 3 sigma was 8-9, so only
part of the offset came off. That errs towards leaving the numbers alone, which
is the right way to be wrong here — and on those images the offset is not the
dominant ROI-height effect anyway (see :func:`compute_lane_profiles`).
"""

SEGMENT_MIN_SNR = 4.0
"""How far above the noise a smoothed peak must rise to be cut out as a band.

Without an absolute floor, ``segment_bands`` thresholds at 4% of whatever the
tallest thing in the lane happens to be — which on an empty lane is a noise
spike. It obligingly carved a blank lane into ten "bands", each one a positive
excursion selected precisely because it was positive. Those fake bands then hid
the blank rows the noise floor is measured on, so the correction switched off
exactly where it was needed most, and the lane total kept growing with the ROI.

The threshold is stated on the smoothed profile, where the noise is already
down by ``sqrt(2 * radius + 1)``. Four sigma there sits comfortably above the
largest excursion a few hundred smoothed samples produce, and still an order of
magnitude below any band worth reporting: a peak this small integrates to
several times its own detection limit. Real bands that do fall under it are not
silently dropped — the lane has no segments at all then, so it falls back to one
whole-lane record, which the detection limit duly flags.
"""

SEGMENT_SPLIT_SIGMA = 3.0
"""谷底比较小峰低这么多个 sigma 才算真的凹下去（在平滑剖面上陈述）。

``valley_drop`` 那条 30% 几何判据要求的分辨率比瑞利判据还高，重叠双带在
几何上就切不开：真实样图上一对 FWHM 8.5–11.3 px、中心间距 11–14 px 的双带
（双高斯拟合得 d/FWHM = 1.02–1.42，正压在瑞利极限上）谷比只降到
0.81–0.96，10 个泳道漏掉 8 个。纯把 ``valley_drop`` 放宽到能全部捞到
（≤0.036）会让合成弱带的假分裂率从 0.2% 涨到 4.8%（更弱的带 1.7% → 20.7%），
放宽 25 倍以上，所以这里并联一条**统计**判据而不是放松几何判据：谷相对
较小峰的下探量在噪声尺度上显著就切。

k=3 的依据：同一批泳道的谷深显著度实测 4.6σ（最难的那道）到 23.7σ，最难
的一道仍有 53% 余量；k=4 也全中但余量只剩 15%，k=6 会重新漏掉 2 条。
"""

SEGMENT_SPLIT_MAX_RATIO = 0.98
"""统计判据的相对下限：再显著也得比较小峰低 2%，否则平顶带的微小起伏也会被切。

只挡"统计上显著但物理上没意义"的那一档——强带上一个 0.5% 的起伏在高信噪
比下轻松超过 3σ。0.98 而不是 0.97：最难的那道实测谷比 0.9645，按 0.97 只
剩 0.0055 的余量；而 0.97 / 0.98 / 不设上限三档实测的假阳性率完全相同。
"""

BLANK_LEVEL_QUANTILE = 10.0
BLANK_LEVEL_Z = 1.2816
"""Where the blank rows sit, from a low quantile pushed back up to the mean.

The threshold has to be measured *from the blank level*, not from zero: clipping
at zero already lifts an empty profile to roughly 0.9 sigma, so "4 sigma above
nothing" is barely 2 sigma above where the noise actually lives, and noise walks
straight through it. The blank level cannot be the mean or the median either —
bands are part of both. A low quantile is untouched as long as bands cover under
90% of the lane; adding back the standard normal's 10% z-score turns it into the
mean of the blank population it came from.
"""


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, stop) runs of True."""
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    splits = np.flatnonzero(np.diff(indices) > 1) + 1
    return [(int(g[0]), int(g[-1]) + 1) for g in np.split(indices, splits)]


def _blank_rows(n: int, segments: list[tuple[int, int]]) -> np.ndarray:
    """Mask of the rows that carry no detected band, nor its tails.

    The margin is not negotiable when it leaves too little to measure. A ROI
    cropped tight around a band has no blank lane at all, and the rows just
    outside the segment are the band's own tails: on a 60-row ROI around a
    strong band they read tens of sigma above background, and taking them for
    noise would have the correction eat 5% of the band. The caller sees an
    almost-empty mask and declines to correct — which costs little, because a
    hull anchored that close to the peak has no room to sink far below the
    background in the first place.
    """
    blank = np.ones(n, dtype=bool)
    for start, stop in segments:
        margin = max(2, int(round((stop - start) * NOISE_BAND_MARGIN)))
        blank[max(0, start - margin) : min(n, stop + margin)] = False
    return blank


def _lane_noise(residual: np.ndarray, segments: list[tuple[int, int]]) -> tuple[float, float]:
    """Baseline offset and noise scale of one lane, measured on its blank rows.

    The rubber band is a *lower* envelope: it settles below the mean of the
    background, and the longer the profile, the further below it can reach. So
    ``profile - baseline`` averages to some positive ``d`` on an empty stretch
    of lane instead of to zero, and every integral picks up ``d`` per sample —
    an error proportional to ROI height and lane width, with nothing to do with
    how much protein is on the membrane. On real data a weak lane read 15-20%
    higher when the user merely dragged the ROI taller.

    ``d`` is measured, not modelled: the mean of the residual over the rows
    where no band was detected, so whatever the baseline actually did under this
    profile is what gets subtracted back off. Note this is the mean of the
    *unclipped* residual. The clipped ``max(residual, 0)`` averages higher —
    around 0.9σ against 0.35σ here — because it keeps every upward fluctuation
    and drops every downward one, and that rectification is not present inside a
    band, where the signal keeps the sum well clear of zero. Subtracting the
    clipped mean from a band would over-correct it by a factor of two or three;
    the integrals avoid the rectification instead of paying for it (see
    :func:`compute_lane_profiles`).

    ``sigma`` is the random part, from the MAD of successive differences inside
    each blank run (immune to slow drift, and it does not collapse on a
    near-constant blank lane).

    Returns ``(floor, sigma)`` per sample. ``floor`` is 0.0 when the bands leave
    too little blank lane to measure: not correcting is safer than correcting by
    a number invented from a handful of samples. It is clamped to non-negative
    and capped at :data:`FLOOR_SIGMA_CAP` sigmas, so anything the segmenter
    missed can distort the estimate only so far — and the noise-free case, where
    sigma is 0, comes out at exactly 0.
    """
    n = int(residual.size)
    blank = _blank_rows(n, segments)
    runs = [(s, e) for s, e in _true_runs(blank) if e - s >= 3]
    if int(blank.sum()) < MIN_NOISE_SAMPLES or not runs:
        return 0.0, _noise_scale(residual)
    steps = np.concatenate([np.abs(np.diff(residual[s:e])) for s, e in runs])
    sigma = 1.4826 * float(np.median(steps)) / np.sqrt(2.0)
    offset = float(np.clip(residual[blank].mean(), 0.0, FLOOR_SIGMA_CAP * sigma))
    return offset, sigma


def detection_limit(sigma: float, samples: int) -> float:
    """3σ limit for an integral over ``samples`` independent samples.

    Per-sample noise adds in quadrature along the integration window, so a taller
    band tolerates proportionally less signal per row before it stops being
    distinguishable from an empty lane.
    """
    if sigma <= 0.0 or samples <= 0:
        return 0.0
    return LOD_SIGMA * float(sigma) * float(np.sqrt(samples))


def _downward_outliers(profile: np.ndarray, radius: int) -> np.ndarray:
    """Mask of samples that dip implausibly far below their neighbourhood.

    Detection runs on the *raw* profile, because the moving average that
    follows widens a 3-sample scratch into a 2*radius+1 sample trough that no
    same-width median could reject any more. The mask is then dilated by
    ``radius`` so the smeared shoulders of the spike are disqualified too.

    Both tests must fire: the dip has to be significant against the profile's
    own noise *and* deep enough to matter against the band amplitude it would
    distort. Requiring both is what keeps a clean profile completely untouched
    — the baseline of a blot with no scratches comes out bit-for-bit as before,
    and a near-blank lane, where the integral is a small residual that a tiny
    baseline shift would swing wildly, is left alone.
    """
    n = int(profile.size)
    if n < 5:
        return np.zeros(n, dtype=bool)
    width = min(max(3, n // BASELINE_DESPIKE_FRAC), BASELINE_DESPIKE_MAX)
    depth = _median_filter(profile, width) - profile
    sigma = _noise_scale(profile)
    amplitude = float(np.max(profile) - np.median(profile))
    floor = max(BASELINE_DESPIKE_SIGMA * sigma, BASELINE_DESPIKE_RANGE * amplitude)
    if floor <= 0.0:
        return np.zeros(n, dtype=bool)
    mask = depth > floor
    if not mask.any():
        return mask
    kernel = np.ones(2 * max(1, int(radius)) + 1)
    return np.convolve(mask.astype(float), kernel, mode="same") > 0.0


def _lower_hull(values: np.ndarray, candidates: np.ndarray) -> list[int]:
    """Indices of the lower convex hull over ``candidates`` (already sorted)."""
    hull: list[int] = []
    for index in candidates:
        i = int(index)
        while len(hull) >= 2:
            i0, i1 = hull[-2], hull[-1]
            # Keep only points forming a convex (upward-opening) lower chain.
            cross = (i1 - i0) * (values[i] - values[i0]) - (i - i0) * (values[i1] - values[i0])
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return hull


def rubber_band_baseline(profile: np.ndarray) -> np.ndarray:
    """Lower convex hull baseline (rubber band stretched under the profile).

    A convex hull needs only one bad sample to drag the whole chord beneath
    the real background — a 3-pixel scratch across a lane was enough to more
    than triple the integrated density of a band nowhere near it. Samples that
    dip implausibly far below their neighbourhood are therefore barred from
    anchoring the hull; everything else, including the entire clean case, is
    the plain rubber band as before.
    """
    n = int(profile.size)
    if n == 0:
        return np.zeros(0, dtype=float)
    if n < 3:
        return profile.astype(float).copy()
    radius = max(1, n // 100)
    smooth = _smooth(profile, radius)
    candidates = np.flatnonzero(~_downward_outliers(profile, radius))
    if candidates.size < 2:
        candidates = np.arange(n)
    hull = _lower_hull(smooth, candidates)
    return np.interp(np.arange(n), hull, smooth[hull])


def valley_baseline(profile: np.ndarray) -> np.ndarray:
    """Valley-to-valley linear baseline (local minima joined by straight lines).

    Local minima are just as easy to fake with a scratch as hull vertices are,
    so the same outlier veto applies.
    """
    n = int(profile.size)
    if n == 0:
        return np.zeros(0, dtype=float)
    if n < 5:
        return np.full(n, float(np.min(profile)))
    radius = max(1, n // 100)
    smooth = _smooth(profile, radius)
    outliers = _downward_outliers(profile, radius)
    win = max(3, n // 40)
    anchors: list[int] = [0]
    index = win
    while index < n - win:
        neighborhood = smooth[index - win : index + win + 1]
        if not outliers[index] and smooth[index] <= float(neighborhood.min()) + 1e-12:
            anchors.append(index)
            index += win
            continue
        index += 1
    anchors.append(n - 1)
    anchors = sorted(set(anchors))
    return np.interp(np.arange(n), anchors, smooth[anchors])


def _baseline_for(profile: np.ndarray, baseline_mode: str) -> np.ndarray:
    if baseline_mode == "rubber":
        return rubber_band_baseline(profile)
    if baseline_mode == "valley":
        return valley_baseline(profile)
    if baseline_mode == "none":
        return np.zeros(int(profile.size), dtype=float)
    raise ValueError(f"未知的基线模式：{baseline_mode}（可选 rubber / valley / none）")


def _local_maxima(values: np.ndarray, min_height: float, min_distance: int) -> list[int]:
    peaks: list[int] = []
    n = len(values)
    win = max(1, min_distance // 2)
    index = win
    while index < n - win:
        neighborhood = values[index - win : index + win + 1]
        if values[index] >= float(neighborhood.max()) - 1e-12 and values[index] >= min_height:
            # Center on plateaus.
            left = index
            while left > 0 and abs(values[left - 1] - values[index]) <= 1e-12:
                left -= 1
            right = index
            while right < n - 1 and abs(values[right + 1] - values[index]) <= 1e-12:
                right += 1
            peaks.append((left + right) // 2)
            index = right + win
            continue
        index += 1
    filtered: list[int] = []
    for peak in sorted(peaks, key=lambda i: values[i], reverse=True):
        if all(abs(peak - other) >= min_distance for other in filtered):
            filtered.append(peak)
    return sorted(filtered)


def _trim_segment(
    smooth: np.ndarray,
    start: int,
    stop: int,
    edge_frac: float = SEGMENT_EDGE_FRAC,
    noise_floor: float = 0.0,
) -> tuple[int, int]:
    """Shrink a segment to where the signal exceeds ``edge_frac`` of its peak.

    ``noise_floor`` is an absolute lower bound on that cut: 2% of a faint peak
    can land inside the noise, which would stretch the segment out along a
    random walk and drag the integration window with it.
    """
    window = smooth[start:stop]
    if window.size == 0:
        return start, stop
    floor = max(float(window.max()) * edge_frac, float(noise_floor))
    above = np.flatnonzero(window > floor)
    if above.size == 0:
        return start, stop
    return start + int(above[0]), start + int(above[-1]) + 1


def segment_bands(
    corrected: np.ndarray,
    *,
    min_height_frac: float = 0.04,
    min_width: int = 3,
    valley_drop: float = 0.30,
    noise_sigma: float = 0.0,
    expected_bands: int | None = None,
) -> list[tuple[int, int]]:
    """Split a baseline-corrected lane profile into band segments.

    Returns half-open (start, stop) index ranges. Runs above the threshold form
    candidates, and a run holding several peaks is cut at the valleys between
    them. Segment edges are trimmed where the signal falls to ~2% of the local
    peak.

    A valley is cut when *either* test fires:

    * **geometric** — it dips at least ``valley_drop`` below the smaller of the
      two peaks, i.e. the pair is resolved in the classic sense;
    * **statistical** — the dip is at least :data:`SEGMENT_SPLIT_SIGMA` sigmas
      deep and still reaches :data:`SEGMENT_SPLIT_MAX_RATIO` of the smaller
      peak. Overlapping doublets sitting at the Rayleigh limit never satisfy
      the geometric test — 30% is a *higher* resolution demand than Rayleigh —
      but their valley is unmistakable against the noise.

    The statistical test needs a real noise scale, so it is off whenever
    ``noise_sigma`` is 0; without that guard a zero sigma would make every
    dip "infinitely significant" and any ripple would be cut.

    ``expected_bands`` is the user saying how many bands each lane carries.
    Given N, the deepest valleys are cut until the lane has N segments and
    both automatic tests are bypassed entirely — the reliable fallback for a
    membrane whose doublets no automatic criterion can be expected to resolve.
    Bands already separated into distinct runs count towards N, and N can only
    add cuts, never merge: a lane that already shows more than N runs keeps
    them.

    The run threshold is the larger of a relative part (``min_height_frac`` of
    the lane's own peak) and an absolute one (:data:`SEGMENT_MIN_SNR` sigmas
    above the noise, stated on the smoothed profile) and is not affected by
    ``expected_bands``. Pass ``noise_sigma=0`` — the default — for the purely
    relative behaviour, which is what the footprint locator wants; anything
    that reports numbers should pass the real noise, or a blank lane comes back
    carved into bands that are pure noise.
    """
    if expected_bands is not None and int(expected_bands) < 1:
        raise ValueError(f"每泳道条带数必须 ≥ 1（收到 {expected_bands}）")
    n = int(corrected.size)
    if n == 0:
        return []
    radius = max(1, n // 120)
    smooth = _smooth(corrected, radius)
    peak_value = float(smooth.max())
    if peak_value <= 1e-9:
        return []
    # Smoothing already divided the per-sample noise by sqrt(2 * radius + 1).
    sigma_smoothed = float(noise_sigma) / np.sqrt(2 * radius + 1)
    noise_threshold = 0.0
    if noise_sigma > 0.0:
        blank_level = (
            float(np.percentile(smooth, BLANK_LEVEL_QUANTILE)) + BLANK_LEVEL_Z * sigma_smoothed
        )
        noise_threshold = blank_level + SEGMENT_MIN_SNR * sigma_smoothed
    threshold = max(peak_value * float(min_height_frac), noise_threshold)
    mask = smooth > threshold
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start >= min_width:
                runs.append((start, index))
            start = None
    if start is not None and n - start >= min_width:
        runs.append((start, n))

    # One entry per valley between adjacent peaks: (absolute index, valley
    # height, height of the smaller peak it separates).
    candidates: list[tuple[int, float, float]] = []
    run_peaks: list[list[int]] = []
    for run_start, run_stop in runs:
        window = smooth[run_start:run_stop]
        min_dist = max(3, (run_stop - run_start) // 8, n // 60)
        peaks = _local_maxima(window, threshold, min_dist)
        run_peaks.append(peaks)
        for left, right in zip(peaks, peaks[1:], strict=False):
            valley = left + int(np.argmin(window[left : right + 1]))
            candidates.append(
                (run_start + valley, float(window[valley]), float(min(window[left], window[right])))
            )

    if expected_bands is not None:
        # Rank by how far the valley drops *relative* to the smaller peak, the
        # same quantity ``valley_drop`` thresholds, so a shallow dip next to a
        # strong band cannot outrank a real split next to a weak one.
        needed = int(expected_bands) - len(runs)
        ranked = sorted(
            candidates,
            key=lambda c: (c[2] - c[1]) / c[2] if c[2] > 0.0 else 0.0,
            reverse=True,
        )
        cut_at = {index for index, _, _ in ranked[: max(0, needed)]}
    else:
        cut_at = set()
        for index, value, smaller in candidates:
            resolved = value <= smaller * (1.0 - valley_drop)
            significant = (
                noise_sigma > 0.0
                and smaller - value >= SEGMENT_SPLIT_SIGMA * sigma_smoothed
                and value <= smaller * SEGMENT_SPLIT_MAX_RATIO
            )
            if resolved or significant:
                cut_at.add(index)

    segments: list[tuple[int, int]] = []
    for (run_start, run_stop), peaks in zip(runs, run_peaks, strict=True):
        if len(peaks) <= 1:
            segments.append(
                _trim_segment(smooth, run_start, run_stop, noise_floor=noise_threshold)
            )
            continue
        cuts = sorted(c for c in cut_at if run_start < c < run_stop)
        bounds = [run_start, *cuts, run_stop]
        for seg_start, seg_stop in zip(bounds, bounds[1:], strict=False):
            if seg_stop - seg_start >= min_width:
                segments.append(
                    _trim_segment(smooth, seg_start, seg_stop, noise_floor=noise_threshold)
                )
    return [(s, e) for s, e in segments if e - s >= min_width]


TIGHT_EDGE_FRAC = 0.08
"""条带足迹边缘阈值：列信号跌破峰高的 8% 处截断（5%~10% 区间的中值）。"""

TIGHT_MIN_WIDTH = 5
"""收紧后条带足迹的最小宽度（像素）。"""

TIGHT_ROW_COVERAGE = 0.7
"""条带行占 ROI 高度超过这个比例时，说明没有空白可排除，列剖面仍用全部行。"""


def _column_span(
    column: np.ndarray,
    *,
    edge_frac: float = TIGHT_EDGE_FRAC,
    background: float | None = None,
) -> tuple[int, int] | None:
    """列剖面上单个条带的横向足迹 [x1, x2)，峰高不显著时返回 None。

    以最强列为锚，向两侧扩展到信号跌破 ``floor + edge_frac × 峰高`` 为止。

    ``background`` 是从条带行之外量出来的背景电平（见
    :func:`_strip_background`）。给了就与列剖面 5 分位数取更低者当 ``floor``；
    两个估计都偏高时都会把足迹往里收，取小的那个才不会漏带，而估偏低只会
    让足迹变宽（宽到接近全宽就整体回退全宽，即旧行为）。
    """
    width = int(column.size)
    if width < 2:
        return None
    smooth = _smooth(column, max(1, width // 40))
    floor = float(np.percentile(smooth, 5))
    if background is not None:
        floor = min(floor, float(background))
    peak = int(np.argmax(smooth))
    height = float(smooth[peak]) - floor
    residual = column - smooth
    sigma = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    if height <= max(4.0 * sigma, 1e-9):
        return None
    threshold = floor + height * float(edge_frac)
    left = peak
    while left > 0 and smooth[left - 1] > threshold:
        left -= 1
    right = peak
    while right < width - 1 and smooth[right + 1] > threshold:
        right += 1
    return left, right + 1


def _strip_background(
    signal: np.ndarray, segments: list[tuple[int, int]] | None
) -> float | None:
    """泳道条的背景电平，量在条带段之外的行上；行数不够时返回 None。

    背景必须从**没有条带的行**上量，不能从"泳道里最暗的那几列"上量。列剖面
    的 5 分位数隐含假设"泳道至少 5% 的列是空白"——条带把泳道占满时这个假设
    直接不成立，5 分位数落到条带自己的平台上，之后 ``edge_frac`` 的相对切
    点就从一个高于弱带的电平起算，整条弱带被判在阈值以下切掉。真实样图上
    这就是一条 250 px 宽的泳道只收到右边 86 px、积分少掉 48% 的原因，而且
    ROI 左边缘挪 3 px 就能在"正确"和"漏一条带"之间跳变。

    条带段之外的行对**每一列**都是背景，与条带在横向占了多宽无关，所以这个
    估计不会随泳道被占满而失效。段边缘（``segment_bands`` 切在峰高 2~4%
    处）外侧还残着条带的纵向拖尾，这里用中位数而不是均值：拖尾只影响紧挨
    段边的少数几行，中位数不为所动，也就不必再留一圈会把空白行吃光的余量
    ——真实样图上条带段占了 ROI 高度的 80%，留半个段高的余量后一行不剩。
    """
    if not segments:
        return None
    height = int(signal.shape[0])
    blank = np.ones(height, dtype=bool)
    for start, stop in segments:
        blank[max(0, int(start)) : min(height, int(stop))] = False
    if int(blank.sum()) < MIN_NOISE_SAMPLES:
        return None
    return float(np.median(signal[blank]))


def _tight_band_span(
    signal: np.ndarray,
    segments: list[tuple[int, int]] | None = None,
    *,
    edge_frac: float = TIGHT_EDGE_FRAC,
    min_width: int = TIGHT_MIN_WIDTH,
) -> tuple[int, int] | None:
    """泳道条内条带的实际横向足迹 [x1, x2)（相对本条），不可靠时返回 None。

    ``segments`` 是先在全宽纵向剖面上分出的条带 y 区间，有两个用处：列剖面
    只在这些**条带行**（各段上下各留半个段高的余量）上统计，纯空白行不参与
    定位——ROI 画得比条带高很多时，空白行原本会把列剖面的对比度稀释掉，弱带
    因此被判为"不可靠"而退回全宽积分；剩下的行则用来量背景电平，见
    :func:`_strip_background`。

    找不到可靠条带峰（空白/平坦泳道）时返回 None，调用方回退全宽。
    """
    width = int(signal.shape[1])
    if width <= min_width:
        return None

    column = _band_row_mean(signal, segments)
    span = _column_span(
        column, edge_frac=edge_frac, background=_strip_background(signal, segments)
    )
    if span is None:
        return None
    x1, x2 = span
    if x2 - x1 >= width - 2:
        return None  # 已接近全宽，收紧无意义
    if x2 - x1 < min_width:
        pad = (min_width - (x2 - x1) + 1) // 2
        x1 = max(0, x1 - pad)
        x2 = min(width, x2 + pad)
    return x1, x2


def _band_row_mean(
    signal: np.ndarray, segments: list[tuple[int, int]] | None
) -> np.ndarray:
    """Column profile averaged over the band rows only (all rows if unknown)."""
    height = int(signal.shape[0])
    if not segments:
        return signal.mean(axis=0)
    rows = np.zeros(height, dtype=bool)
    for start, stop in segments:
        if stop - start < 2:
            continue
        # The segment is cut at 4% of the peak, so its immediate surroundings
        # still carry the band's vertical tails; keeping half a band height on
        # each side stops the footprint tightening past the edge_frac contract.
        margin = max(2, (stop - start) // 2)
        rows[max(0, start - margin) : min(height, stop + margin)] = True
    covered = int(rows.sum())
    if covered < 2 or covered > height * TIGHT_ROW_COVERAGE:
        # Bands already fill the ROI: there is no blank to exclude, and
        # trimming the little that is left would only shift a footprint that
        # was never diluted in the first place.
        return signal.mean(axis=0)
    return signal[rows].mean(axis=0)


def _vertical_segments(signal: np.ndarray, baseline_mode: str) -> list[tuple[int, int]]:
    """全宽纵向剖面上的条带 y 区间，用于给 tight 足迹定位。"""
    profile = signal.sum(axis=1)
    corrected = np.maximum(profile - _baseline_for(profile, baseline_mode), 0.0)
    return segment_bands(corrected)


def _tight_span_for_strip(signal: np.ndarray, baseline_mode: str) -> tuple[int, int] | None:
    """Locate the band footprint of one lane strip, aiming with its band rows."""
    return _tight_band_span(signal, _vertical_segments(signal, baseline_mode))


def compute_lane_profiles(
    raw: np.ndarray,
    roi: RoiRect,
    separators: list[float],
    *,
    dark_bands: bool,
    white_level: float = 255.0,
    baseline_mode: str = "rubber",
    split_bands: bool = True,
    tight: bool = True,
    expected_bands: int | None = None,
) -> list[LaneProfile]:
    """Profile, baseline, and band segmentation for every ROI lane.

    ``tight=True``（默认）时先在每个泳道条内定位条带的实际横向足迹，
    纵向剖面、基线与积分只在该足迹内进行——条带两侧的空白不再稀释
    积分值；找不到可靠条带的泳道回退全宽。``tight=False`` 保持旧的
    全宽行为。

    ``expected_bands`` 是"我这张膜每个泳道有几条带"，交给
    :func:`segment_bands` 强制切出这么多段（见那里的说明）。它只作用于
    定量用的分段，不参与足迹定位——足迹只关心哪些行有信号，切不切开
    是同一批行。

    Known limitation: the rubber band can only be anchored at the two ends of
    the ROI, so ``roi`` is not just a crop — its top and bottom edges *are* the
    statement of where the background is. What decides the error is therefore
    not how tall the ROI is but whether both edges sit on the same flat
    background as the bands, and it behaves as a **step**, not as a drift with
    height. Measured on a chemiluminescence blot cut into a narrow strip (only
    y 336-426 carries membrane, outside it is the camera dark level): while
    both edges stay on the membrane, height barely matters — 62 to 88 rows all
    land within 3.6% of the reference. Cross an edge and the chord drops onto
    the dark level, taking the whole membrane glow into the integral with it:
    the strongest lane's total inflates by 60% and relative quantification is
    off by up to 38%. That is a baseline-model problem — a local background
    anchored just outside each segment would fix it — and it is separate from
    the clipping bias :func:`_lane_noise` removes. Until then,
    :func:`wb_analyzer.roi_check.check_roi_quality` diagnoses the ROIs where it
    bites; see README methodology §5 and §11.
    """
    roi = roi.clipped(raw)
    if roi.width < 3 or roi.height < 3:
        raise ValueError("ROI 过小，无法定量")
    y1, y2 = roi.y, roi.y + roi.height
    require_finite(raw[y1:y2, roi.x : roi.x + roi.width], "ROI 区域")
    lanes: list[LaneProfile] = []
    for lane_index, (x1, x2) in enumerate(lane_boundaries(roi, separators), start=1):
        strip = raw[y1:y2, x1:x2].astype(np.float64)
        signal = (white_level - strip) if dark_bands else strip
        if tight:
            # Aim the footprint using the band rows only, so blank rows above
            # and below cannot dilute the contrast the footprint relies on.
            span = _tight_span_for_strip(signal, baseline_mode)
            if span is not None:
                x1, x2 = x1 + span[0], x1 + span[1]
                signal = signal[:, span[0] : span[1]]
        profile = signal.sum(axis=1)
        baseline = _baseline_for(profile, baseline_mode)
        residual = profile - baseline
        corrected = np.maximum(residual, 0.0)
        # Segmentation needs the noise scale to reject spikes, and the noise
        # floor needs the segments to know which rows are blank. The successive
        # difference MAD breaks the circle: it barely notices a band's smooth
        # shape, so it is a usable seed before anything has been segmented.
        # Segments are found whether or not the caller wants them reported
        # separately, so split_bands=False sees the same blank rows — and the
        # same correction — as the default path.
        detected = segment_bands(
            corrected,
            noise_sigma=_noise_scale(residual),
            expected_bands=expected_bands,
        )
        floor, sigma = _lane_noise(residual, detected)
        # Integrate the residual, not its clipped version. Clipping is what made
        # a blank stretch of lane sum to something positive: it keeps every
        # upward fluctuation and discards every downward one. On the raw
        # residual the fluctuations cancel, so once the measured baseline offset
        # is taken out the window integrates to zero on empty lane and to the
        # band alone under a band — at any ROI height. ``corrected`` stays as it
        # is for segmentation and for the profile plot, which need a curve that
        # sits on zero.
        adjusted = residual - floor
        total = max(0.0, float(adjusted.sum()))
        segments = detected if split_bands else [(0, int(profile.size))]
        if not segments:
            # No detectable peak: fall back to one whole-lane record so the
            # lane still lines up across proteins during normalization.
            segments = [(0, int(profile.size))]
        densities = [max(0.0, float(adjusted[s:e].sum())) for s, e in segments]
        limits = [detection_limit(sigma, e - s) for s, e in segments]
        lanes.append(
            LaneProfile(
                lane=lane_index,
                x1=x1,
                x2=x2,
                y0=y1,
                profile=profile,
                baseline=baseline,
                segments=segments,
                total_density=total,
                band_densities=densities,
                noise_floor=floor,
                noise_sigma=sigma,
                band_detection_limits=limits,
            )
        )
    return lanes


def _display_bgr_from(raw: np.ndarray, white_level: float, display_gray: np.ndarray | None) -> np.ndarray:
    if display_gray is not None:
        if display_gray.ndim == 3:
            return display_gray.copy()
        return cv2.cvtColor(display_gray, cv2.COLOR_GRAY2BGR)
    scale = 255.0 / max(white_level, 1e-9)
    return cv2.cvtColor(np.clip(raw * scale, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def annotate_on_display(
    display_bgr: np.ndarray,
    measurements: list[BandMeasurement],
    *,
    roi: RoiRect | None = None,
    separators: list[float] | None = None,
) -> np.ndarray:
    annotated = display_bgr.copy()
    if annotated.ndim == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
    if roi is not None:
        cv2.rectangle(
            annotated,
            (roi.x, roi.y),
            (roi.x + roi.width - 1, roi.y + roi.height - 1),
            ROI_COLOR,
            1,
            cv2.LINE_AA,
        )
        for sep in separators or []:
            x = int(round(sep))
            if roi.x < x < roi.x + roi.width:
                cv2.line(annotated, (x, roi.y), (x, roi.y + roi.height - 1), SEPARATOR_COLOR, 1, cv2.LINE_AA)
    for item in measurements:
        if item.below_detection:
            color = BAND_WEAK_COLOR
        else:
            color = BAND_COLOR if item.saturated_fraction < 0.05 else BAND_SATURATED_COLOR
        cv2.rectangle(
            annotated,
            (item.x, item.y),
            (item.x + item.width - 1, item.y + item.height - 1),
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"L{item.lane}-B{item.band}",
            (item.x, max(12, item.y - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def quantify_lanes(
    raw: np.ndarray,
    roi: RoiRect,
    separators: list[float],
    *,
    dark_bands: bool,
    white_level: float = 255.0,
    display_gray: np.ndarray | None = None,
    baseline_mode: str = "rubber",
    split_bands: bool = True,
    tight: bool = True,
    expected_bands: int | None = None,
    lanes: list[LaneProfile] | None = None,
) -> AnalysisResult:
    """Quantify ROI lanes via baseline-subtracted vertical profiles.

    Returns a standard :class:`AnalysisResult`; with ``split_bands=True`` each
    detected band becomes one measurement (lanes falling back to a single
    whole-lane record when no peak is found). Use
    :func:`compute_lane_profiles` for the plottable profiles and totals.

    ``lanes`` 让已经调用过 :func:`compute_lane_profiles` 的调用方把结果传回来
    复用：要画剖面的界面因此只算一遍，画出来的基线和表里的数字必定同源。
    传了 ``lanes`` 就意味着 ``baseline_mode`` / ``split_bands`` / ``tight`` /
    ``expected_bands`` 早已固化在这些剖面里，本函数不再读取它们。

    ``tight=True``（默认）把每个泳道的积分与标注框收紧到条带的实际横向
    足迹，泳道内条带两侧的空白不再计入积分；``tight=False`` 保持旧的
    全宽行为。
    """
    roi = roi.clipped(raw)
    if lanes is None:
        lanes = compute_lane_profiles(
            raw,
            roi,
            separators,
            dark_bands=dark_bands,
            white_level=white_level,
            baseline_mode=baseline_mode,
            split_bands=split_bands,
            tight=tight,
            expected_bands=expected_bands,
        )
    measurements: list[BandMeasurement] = []
    for lane in lanes:
        width = lane.x2 - lane.x1
        for band_index, ((seg_start, seg_stop), density, limit) in enumerate(
            zip(lane.segments, lane.band_densities, lane.band_detection_limits, strict=True),
            start=1,
        ):
            box_raw = raw[lane.y0 + seg_start : lane.y0 + seg_stop, lane.x1 : lane.x2]
            baseline_per_pixel = float(lane.baseline[seg_start:seg_stop].mean()) / max(width, 1)
            display_bg = (white_level - baseline_per_pixel) if dark_bands else baseline_per_pixel
            snr = (density / limit * LOD_SIGMA) if limit > 0.0 else 0.0
            measurements.append(
                BandMeasurement(
                    lane=lane.lane,
                    band=band_index,
                    x=lane.x1,
                    y=lane.y0 + seg_start,
                    width=width,
                    height=seg_stop - seg_start,
                    area=int(box_raw.size),
                    mean_intensity=round(float(box_raw.mean()), 3) if box_raw.size else 0.0,
                    background=round(display_bg, 3),
                    integrated_density=round(density, 3),
                    saturated_fraction=round(
                        saturation_fraction(box_raw, dark_bands=dark_bands, white_level=white_level),
                        4,
                    ),
                    detection_limit=round(limit, 3),
                    snr=round(snr, 3),
                )
            )

    display_bgr = _display_bgr_from(raw, white_level, display_gray)
    annotated = annotate_on_display(display_bgr, measurements, roi=roi, separators=separators)
    saturated = sum(1 for m in measurements if m.saturated_fraction >= 0.05)
    return AnalysisResult(
        measurements=measurements,
        annotated_image=annotated,
        dark_bands=dark_bands,
        saturated_band_count=saturated,
        gray_image=cv2.cvtColor(display_bgr, cv2.COLOR_BGR2GRAY),
    )


def roi_column_profile(
    raw: np.ndarray,
    roi: RoiRect,
    *,
    dark_bands: bool,
    white_level: float = 255.0,
) -> np.ndarray:
    """Mean signal per column across the ROI (for lane separator placement)."""
    roi = roi.clipped(raw)
    crop = raw[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width].astype(np.float64)
    signal = (white_level - crop) if dark_bands else crop
    return signal.mean(axis=0)


CLIPPED_HALF_NORMAL_MEAN = 1.0 / np.sqrt(2.0 * np.pi)
"""E[max(x, 0)] / sigma for zero-mean Gaussian noise, i.e. 0.3989."""


def _local_background_native(
    raw: np.ndarray, x1: int, y1: int, x2: int, y2: int
) -> tuple[float, float]:
    """Median and noise scale of the ring around a box, on native-depth data.

    Returns ``(median, sigma)``. The sigma is a MAD so that a neighbouring band
    bleeding into the ring inflates it only mildly — and inflating it is the
    safe direction, since it raises the detection limit rather than lowering it.
    """
    pad_x = max(2, (x2 - x1) // 4)
    pad_y = max(2, (y2 - y1) // 2)
    left, top = max(0, x1 - pad_x), max(0, y1 - pad_y)
    right, bottom = min(raw.shape[1], x2 + pad_x), min(raw.shape[0], y2 + pad_y)
    ring = raw[top:bottom, left:right].astype(np.float64).copy()
    ring[y1 - top : y2 - top, x1 - left : x2 - left] = np.nan
    values = ring[~np.isnan(ring)]
    if not values.size:
        values = raw.astype(np.float64).ravel()
    median = float(np.median(values))
    sigma = 1.4826 * float(np.median(np.abs(values - median)))
    return median, sigma


def measure_box_native(
    raw: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    dark_bands: bool,
    white_level: float = 255.0,
    lane: int = 0,
    band: int = 0,
) -> BandMeasurement:
    """Quantify one rectangle with ring-median local background (native depth)."""
    img_h, img_w = raw.shape[:2]
    x1 = int(np.clip(x, 0, max(0, img_w - 1)))
    y1 = int(np.clip(y, 0, max(0, img_h - 1)))
    x2 = int(np.clip(x1 + max(1, w), x1 + 1, img_w))
    y2 = int(np.clip(y1 + max(1, h), y1 + 1, img_h))
    box = raw[y1:y2, x1:x2].astype(np.float64)
    require_finite(box, "条带区域")
    local_bg, sigma = _local_background_native(raw, x1, y1, x2, y2)
    if dark_bands:
        raw_integral = float(np.maximum(local_bg - box, 0.0).sum())
    else:
        raw_integral = float(np.maximum(box - local_bg, 0.0).sum())
    # Same clipping bias as the lane profiles, one level simpler: the ring
    # median sits at the centre of the noise rather than under it, so the
    # residue left by max(..., 0) is the textbook 0.399σ per pixel and does not
    # have to be measured. Without this the integral grows with the box the
    # user drew, not with the protein inside it.
    integrated = max(0.0, raw_integral - CLIPPED_HALF_NORMAL_MEAN * sigma * box.size)
    limit = detection_limit(sigma, int(box.size))
    return BandMeasurement(
        lane=lane,
        band=band,
        x=x1,
        y=y1,
        width=x2 - x1,
        height=y2 - y1,
        area=int(box.size),
        mean_intensity=round(float(box.mean()), 3),
        background=round(local_bg, 3),
        integrated_density=round(integrated, 3),
        saturated_fraction=round(
            saturation_fraction(box, dark_bands=dark_bands, white_level=white_level), 4
        ),
        detection_limit=round(limit, 3),
        snr=round(integrated / limit * LOD_SIGMA, 3) if limit > 0.0 else 0.0,
    )


def remeasure_native(
    measurements: list[BandMeasurement],
    raw: np.ndarray,
    *,
    dark_bands: bool,
    white_level: float = 255.0,
    display_gray: np.ndarray | None = None,
) -> AnalysisResult:
    """Re-measure coarse 8-bit detection boxes on native-depth data."""
    remeasured = [
        measure_box_native(
            raw,
            m.x,
            m.y,
            m.width,
            m.height,
            dark_bands=dark_bands,
            white_level=white_level,
        )
        for m in measurements
    ]
    numbered = renumber_measurements(remeasured)
    display_bgr = _display_bgr_from(raw, white_level, display_gray)
    annotated = annotate_on_display(display_bgr, numbered)
    saturated = sum(1 for m in numbered if m.saturated_fraction >= 0.05)
    return AnalysisResult(
        measurements=numbered,
        annotated_image=annotated,
        dark_bands=dark_bands,
        saturated_band_count=saturated,
        gray_image=cv2.cvtColor(display_bgr, cv2.COLOR_BGR2GRAY),
    )

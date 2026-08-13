"""Loading-control normalization session and CSV reuse.

Lane identity is the load-bearing concept here. A quantified protein is not
just a list of numbers — each number belongs to a specific lane, and pairing a
target with its loading control has to happen *by lane number*. Aggregating to
a bare positional list (which is what the pre-0.4 API did) silently shifts
every value after a deleted band, so :class:`QuantEntry` now carries the lane
numbers alongside the intensities.

Two other facts travel with the numbers for the same reason: which background
convention produced them (:data:`BackgroundOrigin`, see the README methodology
§9 — the two conventions are not comparable and a ratio across them is
meaningless) and which bands of a multi-band lane were aggregated
(:data:`BandPick`).
"""

from __future__ import annotations

import csv
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Literal

Kind = Literal["loading_control", "target"]

BackgroundOrigin = Literal["unknown", "roi_baseline", "local_background"]
"""Which background convention a number was measured with.

``roi_baseline``
    ROI quantification: rubber-band lower convex hull under the lane profile
    plus the measured noise floor (:func:`wb_analyzer.quant.quantify_lanes`).
``local_background``
    Ring median around each box (:func:`wb_analyzer.quant.measure_box_native`),
    used by whole-image auto-detection and by boxes drawn without a ROI.
``unknown``
    Nobody said. CSV import and every pre-0.5 caller land here; it is a real
    third state, not a default guess, because guessing wrong is exactly the
    error this field exists to prevent.
"""

ORIGIN_UNKNOWN: BackgroundOrigin = "unknown"
ORIGIN_ROI_BASELINE: BackgroundOrigin = "roi_baseline"
ORIGIN_LOCAL_BACKGROUND: BackgroundOrigin = "local_background"

ORIGIN_LABELS: dict[str, str] = {
    ORIGIN_ROI_BASELINE: "ROI 基线",
    ORIGIN_LOCAL_BACKGROUND: "局部背景",
    ORIGIN_UNKNOWN: "未知",
}

_ORIGIN_ALIASES: dict[str, BackgroundOrigin] = {
    "": ORIGIN_UNKNOWN,
    "-": ORIGIN_UNKNOWN,
    "—": ORIGIN_UNKNOWN,
    "unknown": ORIGIN_UNKNOWN,
    "roi": ORIGIN_ROI_BASELINE,
    "roi_baseline": ORIGIN_ROI_BASELINE,
    "detect": ORIGIN_LOCAL_BACKGROUND,
    "local_background": ORIGIN_LOCAL_BACKGROUND,
}
"""Accepted spellings. ``roi`` / ``detect`` are what the GUI calls its two
result paths, so it can hand its own state straight over without a lookup
table that could drift out of step with this one."""


def coerce_origin(value: object, *, strict: bool = True) -> BackgroundOrigin:
    """Canonical origin for ``value``.

    ``strict`` (in-process callers) rejects anything unrecognised, so a typo
    cannot quietly become "unknown" and slip past the cross-origin guard.
    Foreign data — a hand-edited CSV — is read with ``strict=False``, where an
    unrecognised value degrades to "unknown" rather than aborting the import.
    """
    text = str(value or "").strip().lower()
    origin = _ORIGIN_ALIASES.get(text)
    if origin is None:
        if strict:
            raise ValueError(
                f"未知的背景口径：{value!r}"
                f"（可选 {ORIGIN_ROI_BASELINE} / {ORIGIN_LOCAL_BACKGROUND} / {ORIGIN_UNKNOWN}）"
            )
        return ORIGIN_UNKNOWN
    return origin


def origin_label(value: object) -> str:
    """Chinese label for an origin, for status bars and table cells."""
    return ORIGIN_LABELS[coerce_origin(value, strict=False)]


BandPick = Literal["all", "top", "bottom"]
"""Which bands of a multi-band lane are aggregated into the lane's value.

Doublets are usually two modification states of the same protein, so the pick
is by *position* (image ``y``, smaller is higher) rather than by intensity: the
molecular weight of the form you want does not change from lane to lane, its
brightness does.
"""

BAND_PICK_ALL: BandPick = "all"
BAND_PICK_TOP: BandPick = "top"
BAND_PICK_BOTTOM: BandPick = "bottom"

BAND_PICK_LABELS: dict[str, str] = {
    BAND_PICK_ALL: "合并全部",
    BAND_PICK_TOP: "只取上面那条",
    BAND_PICK_BOTTOM: "只取下面那条",
}

_BAND_PICK_ALIASES: dict[str, BandPick] = {
    "": BAND_PICK_ALL,
    "all": BAND_PICK_ALL,
    "sum": BAND_PICK_ALL,
    "top": BAND_PICK_TOP,
    "upper": BAND_PICK_TOP,
    "bottom": BAND_PICK_BOTTOM,
    "lower": BAND_PICK_BOTTOM,
}


def coerce_band_pick(value: object, *, strict: bool = True) -> BandPick:
    """Canonical band pick for ``value`` (same strictness rules as origins)."""
    text = str(value or "").strip().lower()
    pick = _BAND_PICK_ALIASES.get(text)
    if pick is None:
        if strict:
            raise ValueError(
                f"未知的条带取用策略：{value!r}"
                f"（可选 {BAND_PICK_ALL} / {BAND_PICK_TOP} / {BAND_PICK_BOTTOM}）"
            )
        return BAND_PICK_ALL
    return pick


def band_pick_label(value: object) -> str:
    """Chinese label for a band pick."""
    return BAND_PICK_LABELS[coerce_band_pick(value, strict=False)]


RoiCheckGrade = Literal["", "ok", "info", "warning", "error"]

ROI_CHECK_UNKNOWN: RoiCheckGrade = ""
ROI_CHECK_OK: RoiCheckGrade = "ok"
ROI_CHECK_INFO: RoiCheckGrade = "info"
ROI_CHECK_WARNING: RoiCheckGrade = "warning"
ROI_CHECK_ERROR: RoiCheckGrade = "error"

ROI_CHECK_LABELS: dict[str, str] = {
    ROI_CHECK_UNKNOWN: "未自检",
    ROI_CHECK_OK: "通过",
    ROI_CHECK_INFO: "提示",
    ROI_CHECK_WARNING: "警告",
    ROI_CHECK_ERROR: "严重",
}


@dataclass(frozen=True)
class RoiCheck:
    """What the ROI self-check concluded about **one membrane's** quantification.

    One per entry, not one per lane: the check is about where the ROI was
    drawn, and that question is asked once for the whole image. Carrying it on
    the entry is what lets a summary table months later say "this column came
    off a membrane whose ROI was flagged" — by then the report object, the
    image and the operator's memory are all gone.
    """

    grade: RoiCheckGrade = ROI_CHECK_UNKNOWN
    """Machine-readable verdict. Empty means nobody ran the check."""

    summary: str = ""
    """:meth:`~wb_analyzer.roi_check.RoiQualityReport.summary` verbatim."""

    @property
    def recorded(self) -> bool:
        return bool(self.grade or self.summary)

    @property
    def ok(self) -> bool:
        return self.grade == ROI_CHECK_OK

    @property
    def label(self) -> str:
        return ROI_CHECK_LABELS.get(self.grade, ROI_CHECK_LABELS[ROI_CHECK_UNKNOWN])

    @classmethod
    def from_report(cls, report: object) -> RoiCheck:
        """Snapshot a :class:`~wb_analyzer.roi_check.RoiQualityReport`.

        Duck-typed rather than imported: the quantification core has no reason
        to depend on the diagnostics module just to store its one-line verdict.
        """
        if report is None:
            return cls()
        severity = getattr(report, "severity", None)
        grade = ROI_CHECK_OK if severity is None else str(severity)
        if grade not in ROI_CHECK_LABELS:
            grade = ROI_CHECK_UNKNOWN
        text = getattr(report, "summary", "")
        summary = text() if callable(text) else str(text)
        return cls(grade=grade, summary=summary.strip())  # type: ignore[arg-type]

    def as_field(self) -> str:
        """One CSV cell, ``grade: summary``.

        The prefix is ASCII so the column can be filtered by machine; the
        sentence behind it already states the verdict in Chinese, and the
        Chinese sentences use a full-width colon, so splitting on the ASCII
        one cannot cut a summary in half.
        """
        if not self.recorded:
            return ""
        if not self.summary:
            return self.grade
        if not self.grade:
            return self.summary
        return f"{self.grade}: {self.summary}"

    @classmethod
    def parse(cls, value: object) -> RoiCheck:
        """Inverse of :meth:`as_field`; unrecognised text is kept, ungraded."""
        text = str(value or "").strip()
        if not text:
            return cls()
        head, sep, tail = text.partition(":")
        key = head.strip().lower()
        if sep and key and key in ROI_CHECK_LABELS:
            return cls(grade=key, summary=tail.strip())  # type: ignore[arg-type]
        return cls(grade=ROI_CHECK_UNKNOWN, summary=text)


def coerce_roi_check(value: object) -> RoiCheck:
    """Accept a :class:`RoiCheck`, a report, a serialized cell, or nothing."""
    if value is None:
        return RoiCheck()
    if isinstance(value, RoiCheck):
        return value
    if isinstance(value, str):
        return RoiCheck.parse(value)
    return RoiCheck.from_report(value)


def membrane_from_source(source: object) -> str:
    """Default membrane name: the file name of ``source``, without extension."""
    text = str(source or "").strip()
    if not text:
        return ""
    try:
        return Path(text).stem
    except (OSError, ValueError):  # pragma: no cover - defensive
        return text


@lru_cache(maxsize=512)
def _resolved_source(text: str) -> str:
    try:
        resolved = str(Path(text).expanduser().resolve())
    except (OSError, ValueError, RuntimeError):
        # A path the OS will not look at (a device name, a dead network share)
        # still has an identity; comparing the text beats refusing to compare.
        resolved = text
    # normcase folds case and separators on Windows and is a no-op on POSIX,
    # which is exactly the distinction that matters: one filesystem says
    # X.TIF and x.tif are one file, the other says they are two.
    return os.path.normcase(resolved)


def source_key(source: object) -> str:
    r"""Identity of the file behind ``source``, **for comparison only**.

    ``upsert(key_on_source=True)`` treats the source as part of a membrane's
    identity, so ``D:\a\x.tif`` and ``d:/a/x.tif`` have to be one membrane and
    not two — otherwise "keep both" quietly produces a second record of a
    membrane the user only added once. Relative paths (hand-edited CSVs) are
    resolved against the working directory for the same reason.

    Comparison only: entries keep the exact path string they were handed, so
    nothing that gets written to a CSV changes shape here.

    An empty source stays empty. An image built from an array has no file, and
    "no file" is a different question from "which file" — every such entry
    shares the one key, which is what made them interchangeable before too.
    """
    text = str(source or "").strip()
    return _resolved_source(text) if text else ""


@dataclass(frozen=True)
class LaneBand:
    """One band's contribution to its lane, kept so the pick can be redone.

    This is the minimum of a :class:`~wb_analyzer.analysis.BandMeasurement`
    that aggregation needs. Storing it on the entry is what makes switching
    between "合并全部" and "只取上面那条" a re-aggregation instead of a
    re-quantification — the image may not even be open any more.
    """

    lane: int
    band: int
    y: int
    """Top edge in image coordinates. Smaller is higher up: that is "上面那条"."""

    integrated_density: float
    background: float = 0.0
    mean_intensity: float = 0.0
    below_detection: bool = False

    @classmethod
    def from_measurement(cls, measurement) -> LaneBand:
        return cls(
            lane=int(measurement.lane),
            band=int(measurement.band),
            y=int(measurement.y),
            integrated_density=float(measurement.integrated_density),
            background=float(measurement.background),
            mean_intensity=float(measurement.mean_intensity),
            below_detection=bool(measurement.below_detection),
        )


def select_lane_bands(bands: Sequence[LaneBand], pick: BandPick) -> list[LaneBand]:
    """The bands of **one** lane that ``pick`` adopts, ordered top to bottom.

    A lane with a single band keeps that band whichever end was asked for:
    "this lane only ran one band" and "I don't want this band" are different
    statements, and only the first one is true here.
    """
    ordered = sorted(bands, key=lambda b: (b.y, b.band))
    if not ordered or coerce_band_pick(pick) == BAND_PICK_ALL:
        return ordered
    if coerce_band_pick(pick) == BAND_PICK_TOP:
        return [ordered[0]]
    return [ordered[-1]]


def _group_by_lane(bands: Iterable[LaneBand]) -> dict[int, list[LaneBand]]:
    by_lane: dict[int, list[LaneBand]] = {}
    for band in bands:
        by_lane.setdefault(int(band.lane), []).append(band)
    return by_lane


@dataclass(frozen=True)
class LaneSeries:
    """Per-lane quantification of one protein, with lane identity preserved."""

    lanes: list[int]
    intensities: list[float]
    backgrounds: list[float]
    mean_intensities: list[float]
    below_detection: list[bool] = field(default_factory=list)
    """Whether each lane's total failed to clear the 3σ detection limit."""

    bands: list[LaneBand] = field(default_factory=list)
    """**Every** band of every lane, adopted or not.

    Bands the pick left out stay here on purpose: they are still real
    measurements the user should see, and they are what a later change of
    strategy re-aggregates from.
    """

    used_bands: list[list[int]] = field(default_factory=list)
    """Band numbers that actually went into each value, aligned with :attr:`lanes`."""

    band_pick: BandPick = BAND_PICK_ALL

    def bands_of(self, lane: int) -> list[LaneBand]:
        """Every band of one lane, ordered top to bottom."""
        return sorted(
            (b for b in self.bands if int(b.lane) == int(lane)),
            key=lambda b: (b.y, b.band),
        )

    def with_band_pick(self, pick: BandPick) -> LaneSeries:
        """Re-aggregate under a different pick. No re-measurement involved."""
        if not self.bands:
            raise ValueError(
                "这组数据没有保留逐条带明细，无法改用别的条带策略——请重新定量。"
            )
        return aggregate_lane_bands(self.bands, pick)


def aggregate_lane_bands(
    bands: Iterable[LaneBand], pick: BandPick = BAND_PICK_ALL
) -> LaneSeries:
    """Collapse per-band records into one value per lane under ``pick``.

    Background and mean intensity are averaged over the *adopted* bands only,
    and a lane counts as undetected only when every adopted band is: a band
    that was excluded from the sum must not drag the lane's flags around either.
    """
    pick = coerce_band_pick(pick)
    by_lane = _group_by_lane(bands)
    lanes = sorted(by_lane)
    chosen = {lane: select_lane_bands(by_lane[lane], pick) for lane in lanes}
    return LaneSeries(
        lanes=lanes,
        intensities=[float(sum(b.integrated_density for b in chosen[n])) for n in lanes],
        backgrounds=[float(np_mean([b.background for b in chosen[n]])) for n in lanes],
        mean_intensities=[float(np_mean([b.mean_intensity for b in chosen[n]])) for n in lanes],
        below_detection=[all(b.below_detection for b in chosen[n]) for n in lanes],
        bands=[b for n in lanes for b in sorted(by_lane[n], key=lambda b: (b.y, b.band))],
        used_bands=[[int(b.band) for b in chosen[n]] for n in lanes],
        band_pick=pick,
    )


@dataclass(frozen=True)
class QuantEntry:
    kind: Kind
    name: str
    group: str
    intensities: list[float]
    backgrounds: list[float] = field(default_factory=list)
    mean_intensities: list[float] = field(default_factory=list)
    source: str = ""
    lanes: list[int] = field(default_factory=list)
    """Lane number of each intensity. Empty means "assume 1..n" (legacy)."""

    below_detection: list[bool] = field(default_factory=list)
    """Which lanes were below the detection limit when they were quantified.

    Aligned with :attr:`intensities`. Empty means the question was never asked
    (CSV import, or a caller that predates detection limits).
    """

    doc_version: int = -1
    """Which image state (file + orientation) these numbers were measured on.

    The GUI loads and rotates images on background threads, so "the picture on
    screen" and "the picture the numbers came from" can briefly differ. Stamping
    the entry lets a caller notice that an entry predates the current image
    instead of silently mixing two orientations in one normalization. ``-1``
    means unknown (CSV import, or any non-GUI caller).
    """

    origin: BackgroundOrigin = ORIGIN_UNKNOWN
    """Which background convention produced these numbers.

    Defaults to :data:`ORIGIN_UNKNOWN` rather than to either real convention:
    an entry that never said cannot be *assumed* to match the one it is about
    to be divided by, and that assumption is the whole failure mode this field
    exists to stop (see :meth:`AnalysisSession.normalize`).
    """

    band_pick: BandPick = BAND_PICK_ALL
    """How a lane with several bands was collapsed into one value."""

    bands: list[LaneBand] = field(default_factory=list)
    """Per-band detail behind :attr:`intensities`, including the unused bands.

    Empty means the detail was never recorded (CSV import), which is the one
    case where :meth:`with_band_pick` cannot help and the user has to
    re-quantify.
    """

    used_bands: list[list[int]] = field(default_factory=list)
    """Band numbers actually summed into each value, aligned with :attr:`lanes`."""

    membrane: str = ""
    """Human-readable name of the membrane these numbers were measured on.

    Defaults to the file stem of :attr:`source` (filled in by
    :meth:`AnalysisSession.upsert`) and can be renamed afterwards. Stored
    rather than derived on demand: one experiment routinely spans several
    membranes, and two of them can perfectly well be called ``actin.tif`` in
    two different folders — in a summary table that ambiguity is an identity
    error, not a cosmetic one.
    """

    roi_check: RoiCheck = field(default_factory=RoiCheck)
    """The ROI self-check verdict for *this membrane's* quantification.

    One per entry, not one per lane. Empty when nobody ran the check (whole
    image auto-detection, CSV import).
    """

    @property
    def lane_count(self) -> int:
        return len(self.intensities)

    @property
    def label(self) -> str:
        """``Actin（膜 actin1）`` — how warnings and tables name this entry."""
        return f"{self.name}（膜 {self.membrane}）" if self.membrane else self.name

    @property
    def has_lane_identity(self) -> bool:
        """True when the caller stated which lane each value came from."""
        return bool(self.lanes)

    @property
    def lane_numbers(self) -> list[int]:
        """Explicit lane numbers, falling back to positional 1..n."""
        if self.lanes:
            return list(self.lanes)
        return list(range(1, len(self.intensities) + 1))

    def value_by_lane(self) -> dict[int, float]:
        return dict(zip(self.lane_numbers, self.intensities, strict=False))

    def weak_lanes(self) -> set[int]:
        """Lane numbers whose intensity was below the detection limit."""
        return {
            lane
            for lane, weak in zip(self.lane_numbers, self.below_detection, strict=False)
            if weak
        }

    def used_bands_by_lane(self) -> dict[int, list[int]]:
        """Which band numbers each lane's value was summed from."""
        return {
            lane: list(used)
            for lane, used in zip(self.lane_numbers, self.used_bands, strict=False)
        }

    def bands_of(self, lane: int) -> list[LaneBand]:
        """Every band of one lane, adopted or not, ordered top to bottom."""
        return sorted(
            (b for b in self.bands if int(b.lane) == int(lane)),
            key=lambda b: (b.y, b.band),
        )

    def with_band_pick(self, pick: BandPick) -> QuantEntry:
        """Same measurement, different band strategy.

        Re-aggregates from :attr:`bands`, so switching between "合并全部" and
        "只取上面那条" costs nothing and needs neither the image nor a second
        pass over the pixels. Raises when the detail is missing (CSV import),
        because the alternative would be inventing a number.
        """
        pick = coerce_band_pick(pick)
        if pick == coerce_band_pick(self.band_pick):
            return self
        if not self.bands:
            raise ValueError(
                f"“{self.name}”是从 CSV 导入的，没有逐条带明细，"
                "改条带策略需要重新定量。"
            )
        series = aggregate_lane_bands(self.bands, pick)
        return replace(
            self,
            intensities=list(series.intensities),
            backgrounds=list(series.backgrounds),
            mean_intensities=list(series.mean_intensities),
            lanes=list(series.lanes),
            below_detection=list(series.below_detection),
            bands=list(series.bands),
            used_bands=[list(u) for u in series.used_bands],
            band_pick=pick,
        )

    @classmethod
    def from_series(
        cls,
        series: LaneSeries,
        *,
        kind: Kind,
        name: str,
        group: str,
        source: str = "",
        doc_version: int = -1,
        origin: object = ORIGIN_UNKNOWN,
        membrane: str = "",
        roi_check: object = None,
    ) -> QuantEntry:
        return cls(
            kind=kind,
            name=name,
            group=group,
            intensities=list(series.intensities),
            backgrounds=list(series.backgrounds),
            mean_intensities=list(series.mean_intensities),
            source=source,
            lanes=list(series.lanes),
            below_detection=list(series.below_detection),
            doc_version=int(doc_version),
            origin=coerce_origin(origin),
            band_pick=coerce_band_pick(series.band_pick),
            bands=list(series.bands),
            used_bands=[list(u) for u in series.used_bands],
            membrane=str(membrane or "").strip() or membrane_from_source(source),
            # A RoiQualityReport, a RoiCheck or nothing: the GUI hands over
            # whichever it is holding without a conversion step it could skip.
            roi_check=coerce_roi_check(roi_check),
        )


@dataclass(frozen=True)
class NormalizedRow:
    group: str
    lane: int
    loading_control_name: str
    loading_control: float | None
    target_name: str
    target: float | None
    normalized: float | None
    fold_change: float | None = None
    """Normalized value relative to the control lane (control lane = 1.0)."""

    note: str = ""
    """Why this row has no value, when it doesn't (empty on a healthy row)."""

    loading_control_origin: BackgroundOrigin = ORIGIN_UNKNOWN
    target_origin: BackgroundOrigin = ORIGIN_UNKNOWN
    """Background convention of each side, carried through to the CSV."""

    loading_control_band_pick: BandPick = BAND_PICK_ALL
    target_band_pick: BandPick = BAND_PICK_ALL
    loading_control_bands: tuple[int, ...] = ()
    target_bands: tuple[int, ...] = ()
    """Which bands of this lane each side actually summed."""

    loading_control_membrane: str = ""
    target_membrane: str = ""
    """Which membrane each side came off.

    Both sides, always: once a loading control and its target live on separate
    membranes, "which two images produced this ratio" is the smallest fact
    that makes the number reproducible, and a single membrane column would
    read as if the numerator and denominator shared one.
    """

    loading_control_below_detection: bool = False
    target_below_detection: bool = False
    """Whether each side cleared the detection limit in this lane.

    A column of its own rather than a phrase inside :attr:`note`: ``note``
    holds one reason at a time, so a lane that is both undetectable *and*
    divided by a zero control would otherwise lose one of the two facts — and
    no downstream script can read the Chinese either way.
    """

    loading_control_roi_check: RoiCheck = field(default_factory=RoiCheck)
    target_roi_check: RoiCheck = field(default_factory=RoiCheck)
    """Each side's membrane-level ROI verdict, carried through to the CSV."""

    control_lane: int = 0
    """Lane this row's :attr:`fold_change` is measured against (0 = unstated).

    Recorded per row because the reference lane belongs to the group, not to
    the session: two experiments in one file can legitimately use different
    ones, and a fold change whose denominator is unnamed is not interpretable.

    Always the lane that was *asked for*. Normalization never substitutes a
    different one — when the requested lane has nothing to pair,
    :attr:`fold_change` is empty and :attr:`note` says so, because switching
    the reference silently switches which sample counts as 1.
    """


@dataclass(frozen=True)
class MembraneStatus:
    """One membrane's contribution to a group, as the UI would announce it."""

    kind: Kind
    name: str
    membrane: str
    source: str
    lanes: tuple[int, ...]
    missing_lanes: tuple[int, ...]
    """Lanes other membranes of this group have and this one does not."""

    roi_check: RoiCheck = field(default_factory=RoiCheck)

    @property
    def label(self) -> str:
        return f"{self.name}（膜 {self.membrane}）" if self.membrane else self.name


@dataclass(frozen=True)
class GroupStatus:
    """Whether one group (= one experiment) is ready to be normalized.

    A group spans several membranes — one loading control plus N targets — and
    the questions the interface has to answer before the user closes an image
    ("is there an internal control yet?", "which lanes is this membrane
    missing?") all have to be answerable *without* running normalization,
    which is the only place that used to find out.
    """

    group: str
    membranes: tuple[MembraneStatus, ...] = ()
    lanes: tuple[int, ...] = ()
    """Every lane number appearing on any membrane of the group."""

    control_lane: int = 1
    lane_labels: dict[int, str] = field(default_factory=dict)

    @property
    def loading_controls(self) -> tuple[MembraneStatus, ...]:
        return tuple(m for m in self.membranes if m.kind == "loading_control")

    @property
    def targets(self) -> tuple[MembraneStatus, ...]:
        return tuple(m for m in self.membranes if m.kind == "target")

    @property
    def has_loading_control(self) -> bool:
        return bool(self.loading_controls)

    @property
    def target_count(self) -> int:
        return len(self.targets)

    @property
    def is_empty(self) -> bool:
        return not self.membranes

    @property
    def control_lane_missing(self) -> bool:
        """Whether the reference lane appears nowhere in this group.

        Narrower than :attr:`targets_without_fold_change`, and kept separate on
        purpose: "does this lane exist here at all" is a different question
        from "will anything lose its fold change".
        """
        return bool(self.lanes) and self.control_lane not in self.lanes

    @property
    def targets_without_fold_change(self) -> tuple[str, ...]:
        """Targets the reference lane cannot produce a fold change for.

        One field, one answer. Pairing happens per target against the loading
        control normalization would use (the last one added), so the union in
        :attr:`lanes` is too coarse to answer this: a reference lane that sits
        on the control membrane but is missing from one target still costs
        *that* target every fold change, and making the interface combine two
        sentences to notice is how it gets missed.

        Structural only — it reads lane numbers, not intensities, so a
        reference lane whose control integrates to zero is not predictable
        here. That one surfaces as a warning during normalization.
        """
        if not self.targets or not self.has_loading_control:
            return ()
        control = self.loading_controls[-1]
        if self.control_lane not in control.lanes:
            return tuple(m.name for m in self.targets)
        return tuple(m.name for m in self.targets if self.control_lane not in m.lanes)

    @property
    def unnamed_lanes(self) -> tuple[int, ...]:
        return tuple(n for n in self.lanes if not self.lane_labels.get(n, "").strip())

    @property
    def problems(self) -> tuple[str, ...]:
        """Everything worth telling the user right now, one sentence each."""
        issues: list[str] = []
        if self.is_empty:
            return (f"分组“{self.group}”里还没有任何数据。",)
        if not self.has_loading_control:
            names = "、".join(m.name for m in self.targets)
            issues.append(
                f"分组“{self.group}”已经有 {self.target_count} 个目标蛋白（{names}），"
                "但还没有内参：请打开内参膜定量并加为内参，否则这一组算不出比值。"
            )
        elif len(self.loading_controls) > 1:
            listed = "、".join(m.label for m in self.loading_controls)
            issues.append(
                f"分组“{self.group}”里有 {len(self.loading_controls)} 条内参（{listed}），"
                f"归一化默认取最后加入的“{self.loading_controls[-1].label}”。"
                "如果这两条内参各管一部分泳道，请把它们拆成两个分组。"
            )
        if self.has_loading_control and not self.targets:
            issues.append(f"分组“{self.group}”只有内参，还没有目标蛋白。")
        for membrane in self.membranes:
            if membrane.missing_lanes:
                missing = "、".join(str(n) for n in membrane.missing_lanes)
                issues.append(
                    f"{membrane.label}缺泳道 {missing}（本组共 {len(self.lanes)} 道），"
                    "这些泳道在这张膜上不会有数值。"
                )
        # The reference lane's consequence gets exactly one sentence, whichever
        # way it fails: nothing above says it, and the reader should not have to
        # assemble it out of two other lines.
        blocked = self.targets_without_fold_change
        if self.control_lane_missing:
            issues.append(
                f"基准泳道 {self.control_lane} 不在本组任何一张膜上"
                f"（本组泳道 {'、'.join(str(n) for n in self.lanes)}），"
                "这一组的 fold change 会全部留空——软件不会替你换一道当基准。"
            )
        elif blocked and self.control_lane not in self.loading_controls[-1].lanes:
            issues.append(
                f"内参“{self.loading_controls[-1].label}”上没有基准泳道 {self.control_lane}，"
                "这一组所有目标的 fold change 都会留空——"
                "请换一道基准，或补上内参膜这一道的定量。"
            )
        elif blocked:
            listed = "、".join(f"“{name}”" for name in blocked)
            issues.append(
                f"基准泳道 {self.control_lane} 不在{listed}上，"
                f"{listed}的 fold change 会全部留空——"
                "请换一道两边都有的基准泳道，或补上这张膜这一道的定量。"
            )
        unnamed = self.unnamed_lanes
        if unnamed:
            issues.append(
                f"泳道 {'、'.join(str(n) for n in unnamed)} 还没有样品名，"
                "汇总表的 sample 列会是空的。"
            )
        return tuple(issues)

    def summary(self) -> str:
        """One Chinese line for a status bar."""
        if self.is_empty:
            return f"分组“{self.group}”：还没有数据。"
        control = (
            "、".join(m.label for m in self.loading_controls)
            if self.has_loading_control
            else "缺内参"
        )
        targets = "、".join(m.name for m in self.targets) or "还没有目标蛋白"
        text = (
            f"分组“{self.group}”：内参 {control}；目标 {targets}；"
            f"{len(self.lanes)} 个泳道，基准泳道 {self.control_lane}。"
        )
        problems = self.problems
        if problems:
            text += f" ⚠ {len(problems)} 项待确认。"
        return text


class AnalysisSession:
    """In-memory history of quantified loading controls and targets.

    A *group* is one experiment, and an experiment spans several membranes:
    the loading control is photographed on its own blot, each target on
    another. What ties them together is the lane numbering — lane 1 is the
    same sample on every membrane of the experiment — which is exactly why
    :attr:`lane_labels` and :attr:`control_lanes` are keyed by group rather
    than by membrane. Filling a sample name once covers all N membranes, and a
    group whose membranes disagree about which lanes exist is a group with a
    real problem worth surfacing (see :meth:`group_status`).
    """

    def __init__(self) -> None:
        self.entries: list[QuantEntry] = []
        self.last_warnings: list[str] = []
        """Human-readable problems found by the most recent normalize() call."""

        self.lane_labels: dict[str, dict[int, str]] = {}
        """``group -> lane -> sample name``. Use the accessors, not the dict.

        Experimental data, not a preference: these names end up as the axis
        labels of a published figure, so they must never be restored from
        somewhere the user did not put them (a settings store would happily
        carry last week's samples into this week's experiment). They travel
        with the loading-control CSV instead, where importing them is an
        explicit act.
        """

        self.control_lanes: dict[str, int] = {}
        """``group -> reference lane`` for fold change. One per experiment."""

        self.default_control_lane: int = 1
        """Reference lane for groups that never set their own."""

    def clear(self) -> None:
        self.entries.clear()
        self.last_warnings.clear()
        # Sample names and reference lanes describe *this* experiment; leaving
        # them behind is how the next experiment silently inherits the wrong
        # sample names. The session-wide default is a setting, so it stays.
        self.lane_labels.clear()
        self.control_lanes.clear()

    # ------------------------------------------------------------ 分组数据

    @staticmethod
    def canonical_group(group: str) -> str:
        """The group key an empty or padded name resolves to."""
        return group.strip() or "默认"

    def groups(self) -> list[str]:
        """Groups that have data, in the order they first appeared."""
        seen: list[str] = []
        for entry in self.entries:
            if entry.group not in seen:
                seen.append(entry.group)
        return seen

    def group_entries(self, group: str) -> list[QuantEntry]:
        group = self.canonical_group(group)
        return [e for e in self.entries if e.group == group]

    def group_lanes(self, group: str) -> list[int]:
        """Every lane number appearing on any membrane of the group."""
        lanes: set[int] = set()
        for entry in self.group_entries(group):
            lanes.update(entry.lane_numbers)
        return sorted(lanes)

    def set_lane_label(self, group: str, lane: int, sample: str) -> None:
        """Name the sample in one lane of one experiment. Empty clears it."""
        group = self.canonical_group(group)
        lane = int(lane)
        sample = str(sample or "").strip()
        labels = self.lane_labels.setdefault(group, {})
        if sample:
            labels[lane] = sample
        else:
            labels.pop(lane, None)
        if not labels:
            self.lane_labels.pop(group, None)

    def set_lane_labels(
        self, group: str, labels: Mapping[int, str], *, replace: bool = True
    ) -> None:
        """Bulk form of :meth:`set_lane_label` (a table editor, a CSV import)."""
        group = self.canonical_group(group)
        if replace:
            self.lane_labels.pop(group, None)
        for lane, sample in labels.items():
            self.set_lane_label(group, lane, sample)

    def lane_label(self, group: str, lane: int) -> str:
        """Sample name of one lane, or ``""`` when nobody named it."""
        return self.lane_labels.get(self.canonical_group(group), {}).get(int(lane), "")

    def lane_labels_of(self, group: str) -> dict[int, str]:
        """Copy of one group's sample names, keyed by lane."""
        return dict(self.lane_labels.get(self.canonical_group(group), {}))

    def set_control_lane(self, group: str, lane: int) -> None:
        """Set the fold-change reference lane of one experiment."""
        lane = int(lane)
        if lane < 1:
            raise ValueError(f"基准泳道必须是正整数，收到 {lane}")
        self.control_lanes[self.canonical_group(group)] = lane

    def control_lane_of(self, group: str, default: int | None = None) -> int:
        """Reference lane in force for a group.

        The group's own setting wins; ``default`` (then
        :attr:`default_control_lane`) only covers groups that never set one.
        """
        fallback = self.default_control_lane if default is None else int(default)
        return int(self.control_lanes.get(self.canonical_group(group), fallback))

    def forget_group(self, group: str) -> None:
        """Drop a group's sample names and reference lane.

        Deliberately not wired to entry deletion: the GUI's undo restores the
        entry list wholesale, and labels quietly discarded by a delete would
        not come back with it.
        """
        group = self.canonical_group(group)
        self.lane_labels.pop(group, None)
        self.control_lanes.pop(group, None)

    def group_status(self, group: str) -> GroupStatus:
        """Everything the interface needs to nag about, without normalizing."""
        group = self.canonical_group(group)
        entries = self.group_entries(group)
        lanes = self.group_lanes(group)
        all_lanes = set(lanes)
        membranes = tuple(
            MembraneStatus(
                kind=entry.kind,
                name=entry.name,
                membrane=entry.membrane,
                source=entry.source,
                lanes=tuple(entry.lane_numbers),
                missing_lanes=tuple(sorted(all_lanes - set(entry.lane_numbers))),
                roi_check=entry.roi_check,
            )
            for entry in entries
        )
        return GroupStatus(
            group=group,
            membranes=membranes,
            lanes=tuple(lanes),
            control_lane=self.control_lane_of(group),
            lane_labels=self.lane_labels_of(group),
        )

    # ------------------------------------------------------------ 记录管理

    def conflicting_entries(self, entry: QuantEntry) -> list[QuantEntry]:
        """Existing records this one would overwrite **from another membrane**.

        Empty means :meth:`upsert` is unambiguous — either nothing matches, or
        what matches came off the same file and is simply being re-measured.
        A non-empty result is the interface's cue to ask the one question the
        core cannot answer: is this the same membrane exposed twice (overwrite)
        or a second membrane of the same experiment (keep both)?
        """
        key = (entry.kind, entry.name.strip(), self.canonical_group(entry.group))
        source = source_key(entry.source)
        return [
            existing
            for existing in self.entries
            if (existing.kind, existing.name, existing.group) == key
            and source_key(existing.source) != source
        ]

    def upsert(self, entry: QuantEntry, *, key_on_source: bool = False) -> None:
        """Store one membrane's quantification, replacing what it supersedes.

        The key is ``(kind, name, group)``: re-quantifying a membrane and
        adding it again updates the record in place, which is what a second
        exposure of the same blot (``actin1`` / ``actin1(1)``) should do.

        ``key_on_source`` puts the file path in the key as well, so a *second*
        membrane carrying the same protein in the same experiment is kept
        alongside the first instead of replacing it. Only the interface can
        tell those two situations apart, so it is a parameter and not a guess.
        Keeping both is only safe because :meth:`normalize` refuses to pick
        between two loading controls quietly — see the warning it raises there.
        """
        if not entry.name.strip():
            raise ValueError("蛋白名称不能为空")
        if not entry.intensities:
            raise ValueError("没有可保存的泳道强度")
        lanes = [int(v) for v in entry.lanes]
        if lanes:
            if len(lanes) != len(entry.intensities):
                raise ValueError(
                    f"泳道号数量（{len(lanes)}）与强度数量（{len(entry.intensities)}）不一致"
                )
            if len(set(lanes)) != len(lanes):
                raise ValueError(f"泳道号有重复：{sorted(lanes)}")
        used_bands = [[int(b) for b in used] for used in entry.used_bands]
        if used_bands and len(used_bands) != len(entry.intensities):
            raise ValueError(
                f"采用条带记录数量（{len(used_bands)}）与强度数量"
                f"（{len(entry.intensities)}）不一致"
            )
        key = (entry.kind, entry.name.strip(), self.canonical_group(entry.group))
        normalized = QuantEntry(
            kind=entry.kind,
            name=entry.name.strip(),
            group=self.canonical_group(entry.group),
            intensities=[float(v) for v in entry.intensities],
            backgrounds=[float(v) for v in entry.backgrounds],
            mean_intensities=[float(v) for v in entry.mean_intensities],
            source=entry.source,
            lanes=lanes,
            below_detection=[bool(v) for v in entry.below_detection],
            doc_version=int(entry.doc_version),
            origin=coerce_origin(entry.origin),
            band_pick=coerce_band_pick(entry.band_pick),
            bands=list(entry.bands),
            used_bands=used_bands,
            membrane=str(entry.membrane or "").strip() or membrane_from_source(entry.source),
            roi_check=coerce_roi_check(entry.roi_check),
        )
        matches = [
            index
            for index, existing in enumerate(self.entries)
            if (existing.kind, existing.name, existing.group) == key
        ]
        source = source_key(normalized.source)
        same_source = [i for i in matches if source_key(self.entries[i].source) == source]
        if key_on_source:
            matches = same_source
        elif len(matches) > 1:
            # Several membranes already carry this protein and the caller did
            # not say which one it is replacing. Overwriting the one that came
            # off the same file is the only reading that cannot invent a
            # record claiming to be from a membrane it never touched.
            matches = same_source or matches
        if matches:
            self.entries[matches[0]] = normalized
            return
        self.entries.append(normalized)

    def set_membrane(self, index: int, membrane: str) -> QuantEntry:
        """Rename one entry's membrane. The file it came from does not change."""
        if not 0 <= index < len(self.entries):
            raise IndexError(f"历史记录第 {index} 行不存在")
        entry = self.entries[index]
        updated = replace(
            entry, membrane=str(membrane or "").strip() or membrane_from_source(entry.source)
        )
        self.entries[index] = updated
        return updated

    def remove_at(self, index: int) -> None:
        if 0 <= index < len(self.entries):
            self.entries.pop(index)

    def set_band_pick(self, index: int, pick: BandPick) -> QuantEntry:
        """Re-aggregate one history entry under a different band strategy.

        Per entry, not per session: a loading control taken from the lower band
        of its doublet and a target taken from the upper one is a normal
        combination, not a mistake. Returns the updated entry; raises when it
        came from a CSV and has no per-band detail to re-aggregate.
        """
        if not 0 <= index < len(self.entries):
            raise IndexError(f"历史记录第 {index} 行不存在")
        updated = self.entries[index].with_band_pick(pick)
        self.entries[index] = updated
        return updated

    def loading_controls(self, group: str | None = None) -> list[QuantEntry]:
        items = [e for e in self.entries if e.kind == "loading_control"]
        if group is None:
            return items
        group = group.strip() or "默认"
        return [e for e in items if e.group == group]

    def targets(self, group: str | None = None) -> list[QuantEntry]:
        items = [e for e in self.entries if e.kind == "target"]
        if group is None:
            return items
        group = group.strip() or "默认"
        return [e for e in items if e.group == group]

    def latest_loading_control(self, group: str) -> QuantEntry | None:
        """The loading control :meth:`normalize` would use for this group.

        "Latest" is a tie-break, not a decision: when an experiment really did
        run two gels with a loading control each, only the user knows which
        one belongs to a given target. :meth:`normalize` says out loud which
        one it took whenever there is more than one to take.
        """
        matches = self.loading_controls(group)
        return matches[-1] if matches else None

    def normalize(
        self,
        target: QuantEntry,
        loading_control: QuantEntry | None = None,
        control_lane: int | None = None,
        *,
        warnings: list[str] | None = None,
    ) -> list[NormalizedRow]:
        """Pair a target with its loading control **by lane number**.

        Every lane appearing on either side gets a row; a lane that exists on
        only one side is emitted with empty values and a note rather than
        being dropped or, worse, silently shifted onto a neighbour's value.
        Entries without explicit lane numbers are treated as lanes 1..n
        (legacy behaviour) and flagged when that assumption is load-bearing.

        A pairing across two different background conventions is refused the
        same way: rows still come out, the ratio does not. Both numbers exist
        and both are correct — their quotient is not, because the two
        conventions subtract different backgrounds (README §9), and a fold
        change built from one is off by an unknown factor. Refusing here rather
        than warning is deliberate: a warning is something the user can click
        past, and there is no reading of this number that would be right.

        ``control_lane`` defaults to the group's own reference lane
        (:meth:`control_lane_of`); passing a number overrides it for this call.
        """
        sink = self.last_warnings if warnings is None else warnings
        if warnings is None:
            self.last_warnings = sink = []
        lc = loading_control or self.latest_loading_control(target.group)
        if lc is None:
            raise ValueError(f"分组“{target.group}”还没有内参（Loading Control）")

        # Two loading controls in one experiment means two gels, and a target
        # from gel A divided by the actin of gel B produces a number that looks
        # exactly like a real one. The core cannot tell that apart from the
        # same membrane exposed twice, so it refuses to be quiet about it: this
        # warning is the only thing standing between "keep both" and a silently
        # wrong denominator.
        #
        # Split in two on purpose. The group-level half is one sentence however
        # many targets get normalized against it — five near-identical
        # paragraphs is how a warning that matters ends up skipped — and each
        # target adds only the one fact that is its own.
        controls = self.loading_controls(target.group)
        if len(controls) > 1:
            listed = "、".join(e.label for e in controls)
            _append_once(
                sink,
                f"分组“{target.group}”里有 {len(controls)} 条内参：{listed}。"
                "归一化默认取最后加入的那一条。两块胶各带一条内参时它们各管一部分泳道，"
                "配错了分母数字照样算得出来——请确认每个目标配的是它所在那块胶的内参，"
                "或把两块胶拆成两个分组；如果这只是同一张膜的两次曝光，删掉多余的一条即可。",
            )
            sink.append(f"“{target.name}”用的是内参“{lc.label}”。")

        target_values = target.value_by_lane()
        lc_values = lc.value_by_lane()
        shared = sorted(set(target_values) & set(lc_values))
        if not shared:
            raise ValueError("目标与内参没有可对齐的泳道")

        target_origin = coerce_origin(target.origin, strict=False)
        lc_origin = coerce_origin(lc.origin, strict=False)
        known = {target_origin, lc_origin} - {ORIGIN_UNKNOWN}
        cross_origin = len(known) > 1
        origin_note = ""
        if cross_origin:
            origin_note = (
                f"背景口径不一致（内参 {ORIGIN_LABELS[lc_origin]}"
                f" / 目标 {ORIGIN_LABELS[target_origin]}）"
            )
            sink.append(
                f"“{target.name}”（{ORIGIN_LABELS[target_origin]}）与内参"
                f"“{lc.name}”（{ORIGIN_LABELS[lc_origin]}）的背景口径不同，"
                "两套数字不可比，已拒绝配对（README 方法学第 9 节）。"
                "请让两者走同一条路径重新定量。"
            )
        elif known and ORIGIN_UNKNOWN in (target_origin, lc_origin):
            # One side states its convention, the other never recorded one —
            # typically a CSV from an older version. Refusing would invalidate
            # every historical file at once, so the pairing goes ahead and says
            # what it does not know.
            unknown_side = target if target_origin == ORIGIN_UNKNOWN else lc
            known_side = lc if target_origin == ORIGIN_UNKNOWN else target
            known_origin = lc_origin if target_origin == ORIGIN_UNKNOWN else target_origin
            sink.append(
                f"“{unknown_side.name}”没有记录背景口径（多半是旧版 CSV 导入），"
                f"无法确认它与“{known_side.name}”（{ORIGIN_LABELS[known_origin]}）"
                "是否同源；比值仅供参考，要下结论请重新定量。"
            )

        legacy = not (target.has_lane_identity and lc.has_lane_identity)
        if legacy and len(target.intensities) != len(lc.intensities):
            sink.append(
                f"“{target.name}”({len(target.intensities)} 个泳道) 与内参"
                f"“{lc.name}”({len(lc.intensities)} 个泳道) 数量不一致，且没有记录泳道号，"
                "只能按顺序配对——如果中途删过条带，配对结果会整体错位。"
                "请重新定量并保留泳道号。"
            )
        for lane in sorted(set(target_values) - set(lc_values)):
            sink.append(f"泳道 {lane}：内参“{lc.name}”缺少该泳道，无法归一化。")
        for lane in sorted(set(lc_values) - set(target_values)):
            sink.append(f"泳道 {lane}：目标“{target.name}”缺少该泳道，无法归一化。")

        # A ratio between two numbers that are themselves indistinguishable from
        # an empty lane is arithmetic, not measurement. The row still gets its
        # value — suppressing it would only hide the problem — but both the
        # warning list and the row's own note say not to trust it.
        weak_target = target.weak_lanes()
        weak_lc = lc.weak_lanes()
        # ...unless there is no ratio in the first place. Saying "this ratio is
        # unreliable" about a pairing that was just refused buries the reason
        # the user can actually act on.
        for lane in [] if cross_origin else sorted((weak_target | weak_lc) & set(shared)):
            sides = [
                f"目标“{target.name}”" if lane in weak_target else "",
                f"内参“{lc.name}”" if lane in weak_lc else "",
            ]
            who = "、".join(s for s in sides if s)
            sink.append(f"泳道 {lane}：{who}低于检出限（S/N < 3），该比值不可信。")

        ratios: dict[int, float | None] = {}
        for lane in shared:
            if cross_origin:
                ratios[lane] = None
                continue
            lc_val = lc_values[lane]
            ratios[lane] = (target_values[lane] / lc_val) if lc_val > 0 else None
            if ratios[lane] is None:
                sink.append(f"泳道 {lane}：内参强度为 0，无法作分母。")

        control = (
            self.control_lane_of(target.group) if control_lane is None else int(control_lane)
        )
        control_ratio = ratios.get(control)
        # No guess to make here. "Lane 1" means "the WT sample" to whoever set
        # it; quietly using lane 2 instead swaps the denominator for a
        # different sample and every fold change in the group comes out
        # plausible and wrong — the one failure this module exists to prevent.
        # The ratios themselves are unaffected and still exported, so refusing
        # costs only the quantity that depends on a choice nobody validly made.
        fold_note = ""
        if control not in ratios:
            pairable = "、".join(str(n) for n in shared)
            sink.append(
                f"基准泳道 {control} 在“{target.name}”与内参“{lc.name}”之间没有可配对的数据"
                f"（可配对的是泳道 {pairable}），这一组的 fold change 全部留空。"
                "请把基准泳道改成其中一道，或补上这一道的定量——"
                "软件不会替你挑一道当基准，那等于替你换掉“以哪个样品为 1”。"
            )
            fold_note = f"基准泳道 {control} 没有可配对的数据，fold change 无法计算"
        elif control_ratio is None and not cross_origin:
            # The lane is there, its ratio is not: a control that integrates to
            # nothing. Same consequence for the whole group, different cause —
            # and the per-lane "内参强度为 0" warning does not say that the
            # consequence reaches every other lane's fold change.
            sink.append(
                f"基准泳道 {control} 的比值算不出来（内参“{lc.name}”在这一道上强度为 0），"
                f"“{target.name}”这一组的 fold change 全部留空。"
            )
            fold_note = f"基准泳道 {control} 没有比值，fold change 无法计算"

        # Only entries that actually know their lane numbers get rows for the
        # lanes that could not be paired; a legacy positional caller keeps the
        # old "intersection only, never empty" contract, because its consumers
        # (the GUI table among them) assume both values are present.
        reported = sorted(set(target_values) | set(lc_values)) if not legacy else shared
        target_used = target.used_bands_by_lane()
        lc_used = lc.used_bands_by_lane()

        rows: list[NormalizedRow] = []
        for lane in reported:
            ratio = ratios.get(lane)
            note = ""
            if lane not in target_values:
                note = f"目标“{target.name}”缺少泳道 {lane}"
            elif lane not in lc_values:
                note = f"内参“{lc.name}”缺少泳道 {lane}"
            elif cross_origin:
                # Ahead of every other note: the other problems are about one
                # lane, this one voids the whole comparison.
                note = origin_note
            elif lane in weak_target or lane in weak_lc:
                # Ahead of "内参强度为 0": a control that integrates to nothing is
                # usually below detection, and that is the fact the user can act
                # on. The missing ratio is already visible as an empty cell, and
                # the warning list states both reasons in full.
                note = "低于检出限"
            elif ratio is None:
                note = "内参强度为 0"
            elif fold_note:
                # Last: this row's ratio is fine, only the fold change is
                # missing. Rows that already carry a problem keep theirs — one
                # note, one reason, and the group-level fact is in the warnings.
                note = fold_note
            fold = None
            if ratio is not None and control_ratio is not None and control_ratio > 0:
                fold = round(ratio / control_ratio, 6)
            rows.append(
                NormalizedRow(
                    group=target.group,
                    lane=lane,
                    loading_control_name=lc.name,
                    loading_control=lc_values.get(lane),
                    target_name=target.name,
                    target=target_values.get(lane),
                    normalized=None if ratio is None else round(ratio, 6),
                    fold_change=fold,
                    note=note,
                    loading_control_origin=lc_origin,
                    target_origin=target_origin,
                    loading_control_band_pick=coerce_band_pick(lc.band_pick, strict=False),
                    target_band_pick=coerce_band_pick(target.band_pick, strict=False),
                    loading_control_bands=tuple(lc_used.get(lane, ())),
                    target_bands=tuple(target_used.get(lane, ())),
                    loading_control_membrane=lc.membrane,
                    target_membrane=target.membrane,
                    loading_control_below_detection=lane in weak_lc,
                    target_below_detection=lane in weak_target,
                    loading_control_roi_check=lc.roi_check,
                    target_roi_check=target.roi_check,
                    control_lane=control,
                )
            )
        return rows

    def normalize_all(self, control_lane: int | None = None) -> list[NormalizedRow]:
        """Normalize every target, each against its own group's reference lane.

        ``control_lane`` is only the default for groups that never set one of
        their own (:meth:`set_control_lane`). One reference lane for the whole
        session was defensible while a session held one membrane; across an
        experiment's worth of membranes the reference belongs to the group,
        which is the unit the lanes are paired in.
        """
        rows: list[NormalizedRow] = []
        collected: list[str] = []
        for target in self.targets():
            try:
                rows.extend(
                    self.normalize(
                        target,
                        control_lane=self.control_lane_of(target.group, control_lane),
                        warnings=collected,
                    )
                )
            except ValueError as exc:
                collected.append(f"“{target.name}”未能归一化：{exc}")
        self.last_warnings = collected
        return rows

    def export_loading_control_csv(self, path: str | Path, group: str | None = None) -> Path:
        """Write the loading controls, sample names included.

        ``sample`` rides along because lane naming is experimental data with
        nowhere else to live: the next experiment usually loads the samples in
        the same order, and this file is how that naming comes back — by an
        explicit import, never by a settings store that would resurrect it
        uninvited.
        """
        path = Path(path)
        controls = self.loading_controls(group)
        if not controls:
            raise ValueError("没有可导出的内参数据")
        fields = [
            "kind",
            "name",
            "group",
            "sample",
            "lane",
            "integrated_density",
            "background",
            "mean_intensity",
            "source",
            "membrane",
            "origin",
            "band_pick",
            "bands_used",
            "roi_check",
        ]
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for entry in controls:
                used_by_lane = entry.used_bands_by_lane()
                for index, (lane, intensity) in enumerate(
                    zip(entry.lane_numbers, entry.intensities, strict=False)
                ):
                    bg = entry.backgrounds[index] if index < len(entry.backgrounds) else ""
                    mean = (
                        entry.mean_intensities[index]
                        if index < len(entry.mean_intensities)
                        else ""
                    )
                    writer.writerow(
                        {
                            "kind": entry.kind,
                            "name": entry.name,
                            "group": entry.group,
                            "sample": self.lane_label(entry.group, lane),
                            "lane": lane,
                            "integrated_density": intensity,
                            "background": bg,
                            "mean_intensity": mean,
                            "source": entry.source,
                            "membrane": entry.membrane,
                            "origin": coerce_origin(entry.origin, strict=False),
                            "band_pick": coerce_band_pick(entry.band_pick, strict=False),
                            "bands_used": _format_bands(used_by_lane.get(lane, [])),
                            "roi_check": entry.roi_check.as_field(),
                        }
                    )
        return path

    def import_loading_control_csv(self, path: str | Path) -> int:
        """Restore quantified entries from a CSV, keyed on the ``lane`` column.

        Row order in the file is *not* trusted: a CSV that has been sorted in
        Excel, or one with a lane missing, used to be read back positionally
        and quietly reassigned every intensity to the wrong lane. Gaps stay
        gaps (the lane simply won't pair during normalization) and are
        reported through :attr:`last_warnings`.

        Records are grouped by ``(kind, name, group, source)`` — the same
        identity :meth:`upsert` uses. Two membranes of one protein in one file
        come back as two records instead of one record whose lanes overwrote
        each other, and a protein that appears as both a loading control and a
        target (one experiment's actin exported next to another's) stays two
        records instead of collapsing into whichever ``kind`` came first.
        ``source`` is compared through :func:`source_key`, so the same file
        spelled two ways is one membrane.

        ``sample`` names found in the file are applied to their group; anything
        the file does not have (an older export with no ``sample`` /
        ``membrane`` / ``roi_check`` column) simply stays empty.
        """
        path = Path(path)
        warnings: list[str] = []
        samples: dict[tuple[str, int], str] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "integrated_density" not in reader.fieldnames:
                raise ValueError("内参 CSV 缺少 integrated_density 列")
            has_lane_column = "lane" in reader.fieldnames
            buckets: dict[tuple[str, str, str, str], dict] = {}
            for line_no, row in enumerate(reader, start=2):
                kind = (row.get("kind") or "loading_control").strip()
                if kind not in {"loading_control", "target"}:
                    kind = "loading_control"
                name = (row.get("name") or "Loading Control").strip()
                group = (row.get("group") or "默认").strip()
                source = str(row.get("source") or path)
                bucket = buckets.setdefault(
                    (kind, name, group, source_key(source)),
                    {
                        "rows": [],
                        "source": source,
                        "origins": set(),
                        "picks": set(),
                        "membranes": set(),
                        "roi_checks": set(),
                    },
                )
                bucket["membranes"].add((row.get("membrane") or "").strip())
                bucket["roi_checks"].add((row.get("roi_check") or "").strip())
                try:
                    intensity = float(row["integrated_density"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"第 {line_no} 行的 integrated_density 不是数字："
                        f"{row.get('integrated_density')!r}"
                    ) from exc
                lane = _parse_lane(row.get("lane")) if has_lane_column else None
                if has_lane_column and lane is None:
                    warnings.append(f"第 {line_no} 行缺少有效的 lane 值，已按出现顺序编号。")
                sample = (row.get("sample") or "").strip()
                if sample and lane is not None:
                    previous = samples.get((group, lane))
                    if previous and previous != sample:
                        warnings.append(
                            f"分组“{group}”泳道 {lane} 在文件里有两个样品名"
                            f"（{previous} / {sample}），保留后出现的那个。"
                        )
                    samples[(group, lane)] = sample
                # A CSV is foreign data: an unreadable origin degrades to
                # "unknown" (which normalization then flags) instead of
                # aborting an import the user cannot easily repair.
                bucket["origins"].add(coerce_origin(row.get("origin"), strict=False))
                bucket["picks"].add(coerce_band_pick(row.get("band_pick"), strict=False))
                bucket["rows"].append(
                    (
                        lane,
                        intensity,
                        _parse_float(row.get("background")),
                        _parse_float(row.get("mean_intensity")),
                        _parse_bands(row.get("bands_used")),
                    )
                )

        count = 0
        membranes_per_protein: dict[tuple[str, str, str], int] = {}
        for kind, name, group, _source in buckets:
            key = (kind, name, group)
            membranes_per_protein[key] = membranes_per_protein.get(key, 0) + 1
        for (kind, name, group, _source), bucket in buckets.items():
            rows = bucket["rows"]
            lanes = [lane for lane, *_ in rows]
            if any(lane is None for lane in lanes):
                # Mixed or absent lane numbers: fall back to file order.
                if has_lane_column and any(lane is not None for lane in lanes):
                    warnings.append(
                        f"“{name}”的 lane 列不完整，该蛋白整体按文件行序编号。"
                    )
                ordered = [(i + 1, *values) for i, (_, *values) in enumerate(rows)]
            else:
                seen: dict[int, tuple] = {}
                for lane, *values in rows:
                    if lane in seen:
                        warnings.append(f"“{name}”的泳道 {lane} 出现多次，保留最后一条。")
                    seen[lane] = tuple(values)
                ordered = [(lane, *seen[lane]) for lane in sorted(seen)]
                numbers = [lane for lane, *_ in ordered]
                missing = sorted(set(range(min(numbers), max(numbers) + 1)) - set(numbers))
                if missing:
                    warnings.append(
                        f"“{name}”缺少泳道 {', '.join(str(m) for m in missing)}，"
                        "这些泳道将不参与归一化。"
                    )
            origins = set(bucket["origins"])
            picks = set(bucket["picks"])
            if len(origins) > 1:
                warnings.append(
                    f"“{name}”的 origin 列在不同行上不一致，已按“口径未知”处理。"
                )
                origins = {ORIGIN_UNKNOWN}
            if len(picks) > 1:
                warnings.append(
                    f"“{name}”的 band_pick 列在不同行上不一致，已按“合并全部”处理。"
                )
                picks = {BAND_PICK_ALL}
            membranes = sorted(m for m in bucket["membranes"] if m)
            checks = sorted(c for c in bucket["roi_checks"] if c)
            self.upsert(
                QuantEntry(
                    kind=kind,  # type: ignore[arg-type]
                    name=name,
                    group=group,
                    intensities=[value for _, value, _, _, _ in ordered],
                    backgrounds=[value for _, _, value, _, _ in ordered],
                    mean_intensities=[value for _, _, _, value, _ in ordered],
                    source=str(bucket["source"]),
                    lanes=[lane for lane, *_ in ordered],
                    origin=origins.pop() if origins else ORIGIN_UNKNOWN,
                    band_pick=picks.pop() if picks else BAND_PICK_ALL,
                    # Only the aggregate survives a CSV, so the per-band detail
                    # stays empty: the pick can be *reported*, not redone.
                    used_bands=[list(value) for _, _, _, _, value in ordered],
                    membrane=membranes[0] if membranes else "",
                    roi_check=RoiCheck.parse(checks[0]) if checks else RoiCheck(),
                ),
                # Two membranes of one protein in one file are two records; the
                # default key would let the second one overwrite the first and
                # lose half the experiment on the way in.
                key_on_source=membranes_per_protein[(kind, name, group)] > 1,
            )
            count += 1
        for (kind, name, group), count_of_membranes in sorted(membranes_per_protein.items()):
            if count_of_membranes < 2:
                continue
            head = (
                f"文件里内参“{name}”（分组 {group}）有 {count_of_membranes} 张膜的数据，已各存一条记录。"
                if kind == "loading_control"
                else f"文件里目标“{name}”（分组 {group}）有 {count_of_membranes} 张膜的数据，已各存一条记录。"
            )
            warnings.append(
                head
                + (
                    "归一化默认取最后加入的那一条，请确认这正是你要的那张膜。"
                    if kind == "loading_control"
                    else "汇总表的宽表会把这些格子留空（不做合并），理由写在长表的 note 里。"
                )
            )
        for (group, lane), sample in samples.items():
            previous = self.lane_label(group, lane)
            if previous and previous != sample:
                warnings.append(
                    f"分组“{group}”泳道 {lane} 的样品名由“{previous}”改为“{sample}”（来自导入文件）。"
                )
            self.set_lane_label(group, lane, sample)
        self.last_warnings = warnings
        return count


def _append_once(sink: list[str], text: str) -> None:
    """Append a warning unless that exact sentence is already in the list.

    For facts about a *group*: :meth:`AnalysisSession.normalize` runs once per
    target and they all share one warning list.
    """
    if text not in sink:
        sink.append(text)


def _parse_lane(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _parse_float(value: object) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _format_bands(bands: Iterable[int]) -> str:
    return ";".join(str(int(b)) for b in bands)


def _parse_bands(value: object) -> list[int]:
    text = str(value or "").strip()
    if not text:
        return []
    numbers: list[int] = []
    for part in text.replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            numbers.append(int(float(part)))
        except ValueError:
            continue
    return numbers


def export_normalized_csv(rows: list[NormalizedRow], path: str | Path) -> Path:
    """Write the normalization table, provenance included.

    The extra columns are appended after the original eight, so anything
    reading this file by column position keeps working. ``note`` matters as
    much as the numbers do now: with cross-convention pairings refused, an
    empty ``normalized`` cell needs to say why it is empty.
    """
    path = Path(path)
    fields = [
        "group",
        "lane",
        "loading_control_name",
        "loading_control",
        "target_name",
        "target",
        "normalized",
        "fold_change",
        "loading_control_origin",
        "loading_control_band_pick",
        "loading_control_bands",
        "target_origin",
        "target_band_pick",
        "target_bands",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "group": row.group,
                    "lane": row.lane,
                    "loading_control_name": row.loading_control_name,
                    "loading_control": "" if row.loading_control is None else row.loading_control,
                    "target_name": row.target_name,
                    "target": "" if row.target is None else row.target,
                    "normalized": "" if row.normalized is None else row.normalized,
                    "fold_change": "" if row.fold_change is None else row.fold_change,
                    "loading_control_origin": row.loading_control_origin,
                    "loading_control_band_pick": row.loading_control_band_pick,
                    "loading_control_bands": _format_bands(row.loading_control_bands),
                    "target_origin": row.target_origin,
                    "target_band_pick": row.target_band_pick,
                    "target_bands": _format_bands(row.target_bands),
                    "note": row.note,
                }
            )
    return path


def lane_series_from_measurements(
    measurements, *, band_pick: BandPick = BAND_PICK_ALL
) -> LaneSeries:
    """Aggregate detected bands by lane, keeping the lane numbers.

    The lane numbers are what makes normalization safe: if the user deleted the
    band in lane 3, the returned lanes are ``[1, 2, 4, 5]`` and lane 4 still
    lines up with lane 4 of the loading control instead of sliding onto lane 3.

    ``band_pick`` decides what a lane with several bands is worth: the sum of
    all of them (the default, and what every caller got before), or just the
    upper / lower one. Every band is kept in :attr:`LaneSeries.bands` either
    way, so the choice can be revisited later without touching the image.
    """
    return aggregate_lane_bands(
        (LaneBand.from_measurement(m) for m in measurements), band_pick
    )


def np_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

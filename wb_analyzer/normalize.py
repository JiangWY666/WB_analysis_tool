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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
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

    @property
    def lane_count(self) -> int:
        return len(self.intensities)

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


class AnalysisSession:
    """In-memory history of quantified loading controls and targets."""

    def __init__(self) -> None:
        self.entries: list[QuantEntry] = []
        self.last_warnings: list[str] = []
        """Human-readable problems found by the most recent normalize() call."""

    def clear(self) -> None:
        self.entries.clear()
        self.last_warnings.clear()

    def upsert(self, entry: QuantEntry) -> None:
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
        key = (entry.kind, entry.name.strip(), entry.group.strip() or "默认")
        normalized = QuantEntry(
            kind=entry.kind,
            name=entry.name.strip(),
            group=entry.group.strip() or "默认",
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
        )
        for index, existing in enumerate(self.entries):
            if (existing.kind, existing.name, existing.group) == key:
                self.entries[index] = normalized
                return
        self.entries.append(normalized)

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
        matches = self.loading_controls(group)
        return matches[-1] if matches else None

    def normalize(
        self,
        target: QuantEntry,
        loading_control: QuantEntry | None = None,
        control_lane: int = 1,
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
        """
        sink = self.last_warnings if warnings is None else warnings
        if warnings is None:
            self.last_warnings = sink = []
        lc = loading_control or self.latest_loading_control(target.group)
        if lc is None:
            raise ValueError(f"分组“{target.group}”还没有内参（Loading Control）")

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

        control = int(control_lane)
        if control not in ratios:
            fallback = shared[int(np_clip_index(control - 1, len(shared)))]
            sink.append(f"对照泳道 {control} 不在可配对泳道中，已改用泳道 {fallback}。")
            control = fallback
        control_ratio = ratios.get(control)

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
                )
            )
        return rows

    def normalize_all(self, control_lane: int = 1) -> list[NormalizedRow]:
        rows: list[NormalizedRow] = []
        collected: list[str] = []
        for target in self.targets():
            try:
                rows.extend(
                    self.normalize(target, control_lane=control_lane, warnings=collected)
                )
            except ValueError as exc:
                collected.append(f"“{target.name}”未能归一化：{exc}")
        self.last_warnings = collected
        return rows

    def export_loading_control_csv(self, path: str | Path, group: str | None = None) -> Path:
        path = Path(path)
        controls = self.loading_controls(group)
        if not controls:
            raise ValueError("没有可导出的内参数据")
        fields = [
            "kind",
            "name",
            "group",
            "lane",
            "integrated_density",
            "background",
            "mean_intensity",
            "source",
            "origin",
            "band_pick",
            "bands_used",
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
                            "lane": lane,
                            "integrated_density": intensity,
                            "background": bg,
                            "mean_intensity": mean,
                            "source": entry.source,
                            "origin": coerce_origin(entry.origin, strict=False),
                            "band_pick": coerce_band_pick(entry.band_pick, strict=False),
                            "bands_used": _format_bands(used_by_lane.get(lane, [])),
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
        """
        path = Path(path)
        warnings: list[str] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "integrated_density" not in reader.fieldnames:
                raise ValueError("内参 CSV 缺少 integrated_density 列")
            has_lane_column = "lane" in reader.fieldnames
            buckets: dict[tuple[str, str], dict] = {}
            for line_no, row in enumerate(reader, start=2):
                kind = (row.get("kind") or "loading_control").strip()
                if kind not in {"loading_control", "target"}:
                    kind = "loading_control"
                name = (row.get("name") or "Loading Control").strip()
                group = (row.get("group") or "默认").strip()
                bucket = buckets.setdefault(
                    (name, group),
                    {
                        "kind": kind,
                        "rows": [],
                        "source": row.get("source") or str(path),
                        "origins": set(),
                        "picks": set(),
                    },
                )
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
        for (name, group), bucket in buckets.items():
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
            self.upsert(
                QuantEntry(
                    kind=bucket["kind"],  # type: ignore[arg-type]
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
                )
            )
            count += 1
        self.last_warnings = warnings
        return count


def np_clip_index(index: int, count: int) -> int:
    return max(0, min(int(index), count - 1))


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

"""Loading-control normalization session and CSV reuse.

Lane identity is the load-bearing concept here. A quantified protein is not
just a list of numbers — each number belongs to a specific lane, and pairing a
target with its loading control has to happen *by lane number*. Aggregating to
a bare positional list (which is what the pre-0.4 API did) silently shifts
every value after a deleted band, so :class:`QuantEntry` now carries the lane
numbers alongside the intensities.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Kind = Literal["loading_control", "target"]


@dataclass(frozen=True)
class LaneSeries:
    """Per-lane quantification of one protein, with lane identity preserved."""

    lanes: list[int]
    intensities: list[float]
    backgrounds: list[float]
    mean_intensities: list[float]
    below_detection: list[bool] = field(default_factory=list)
    """Whether each lane's total failed to clear the 3σ detection limit."""


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
        )
        for index, existing in enumerate(self.entries):
            if (existing.kind, existing.name, existing.group) == key:
                self.entries[index] = normalized
                return
        self.entries.append(normalized)

    def remove_at(self, index: int) -> None:
        if 0 <= index < len(self.entries):
            self.entries.pop(index)

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
        for lane in sorted((weak_target | weak_lc) & set(shared)):
            sides = [
                f"目标“{target.name}”" if lane in weak_target else "",
                f"内参“{lc.name}”" if lane in weak_lc else "",
            ]
            who = "、".join(s for s in sides if s)
            sink.append(f"泳道 {lane}：{who}低于检出限（S/N < 3），该比值不可信。")

        ratios: dict[int, float | None] = {}
        for lane in shared:
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

        rows: list[NormalizedRow] = []
        for lane in reported:
            ratio = ratios.get(lane)
            note = ""
            if lane not in target_values:
                note = f"目标“{target.name}”缺少泳道 {lane}"
            elif lane not in lc_values:
                note = f"内参“{lc.name}”缺少泳道 {lane}"
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
        fields = ["kind", "name", "group", "lane", "integrated_density", "background", "mean_intensity", "source"]
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for entry in controls:
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
                    {"kind": kind, "rows": [], "source": row.get("source") or str(path)},
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
                bucket["rows"].append(
                    (
                        lane,
                        intensity,
                        _parse_float(row.get("background")),
                        _parse_float(row.get("mean_intensity")),
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
            self.upsert(
                QuantEntry(
                    kind=bucket["kind"],  # type: ignore[arg-type]
                    name=name,
                    group=group,
                    intensities=[value for _, value, _, _ in ordered],
                    backgrounds=[value for _, _, value, _ in ordered],
                    mean_intensities=[value for _, _, _, value in ordered],
                    source=str(bucket["source"]),
                    lanes=[lane for lane, *_ in ordered],
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


def export_normalized_csv(rows: list[NormalizedRow], path: str | Path) -> Path:
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
                }
            )
    return path


def lane_series_from_measurements(measurements) -> LaneSeries:
    """Aggregate auto-detected bands by lane, keeping the lane numbers.

    The lane numbers are what makes normalization safe: if the user deleted the
    band in lane 3, the returned lanes are ``[1, 2, 4, 5]`` and lane 4 still
    lines up with lane 4 of the loading control instead of sliding onto lane 3.
    """
    by_lane: dict[int, list] = {}
    for item in measurements:
        by_lane.setdefault(int(item.lane), []).append(item)
    lanes = sorted(by_lane)
    return LaneSeries(
        lanes=lanes,
        intensities=[float(sum(b.integrated_density for b in by_lane[n])) for n in lanes],
        backgrounds=[float(np_mean([b.background for b in by_lane[n]])) for n in lanes],
        mean_intensities=[float(np_mean([b.mean_intensity for b in by_lane[n]])) for n in lanes],
        # A lane is only called undetected when *every* band in it is: one solid
        # band plus a faint smear still adds up to a lane worth normalizing.
        below_detection=[all(b.below_detection for b in by_lane[n]) for n in lanes],
    )


def np_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

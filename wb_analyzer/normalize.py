"""Loading-control normalization session and CSV reuse."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Kind = Literal["loading_control", "target"]


@dataclass(frozen=True)
class QuantEntry:
    kind: Kind
    name: str
    group: str
    intensities: list[float]
    backgrounds: list[float] = field(default_factory=list)
    mean_intensities: list[float] = field(default_factory=list)
    source: str = ""

    @property
    def lane_count(self) -> int:
        return len(self.intensities)


@dataclass(frozen=True)
class NormalizedRow:
    group: str
    lane: int
    loading_control_name: str
    loading_control: float
    target_name: str
    target: float
    normalized: float | None
    fold_change: float | None = None
    """Normalized value relative to the control lane (control lane = 1.0)."""


class AnalysisSession:
    """In-memory history of quantified loading controls and targets."""

    def __init__(self) -> None:
        self.entries: list[QuantEntry] = []

    def clear(self) -> None:
        self.entries.clear()

    def upsert(self, entry: QuantEntry) -> None:
        if not entry.name.strip():
            raise ValueError("蛋白名称不能为空")
        if not entry.intensities:
            raise ValueError("没有可保存的泳道强度")
        key = (entry.kind, entry.name.strip(), entry.group.strip() or "默认")
        normalized = QuantEntry(
            kind=entry.kind,
            name=entry.name.strip(),
            group=entry.group.strip() or "默认",
            intensities=[float(v) for v in entry.intensities],
            backgrounds=[float(v) for v in entry.backgrounds],
            mean_intensities=[float(v) for v in entry.mean_intensities],
            source=entry.source,
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
    ) -> list[NormalizedRow]:
        lc = loading_control or self.latest_loading_control(target.group)
        if lc is None:
            raise ValueError(f"分组“{target.group}”还没有内参（Loading Control）")
        count = min(len(target.intensities), len(lc.intensities))
        if count == 0:
            raise ValueError("目标与内参没有可对齐的泳道")
        ratios: list[float | None] = []
        for index in range(count):
            lc_val = float(lc.intensities[index])
            target_val = float(target.intensities[index])
            ratios.append((target_val / lc_val) if lc_val > 0 else None)
        control_index = int(np_clip_index(control_lane - 1, count))
        control_ratio = ratios[control_index]
        rows: list[NormalizedRow] = []
        for index in range(count):
            ratio = ratios[index]
            fold = None
            if ratio is not None and control_ratio is not None and control_ratio > 0:
                fold = round(ratio / control_ratio, 6)
            rows.append(
                NormalizedRow(
                    group=target.group,
                    lane=index + 1,
                    loading_control_name=lc.name,
                    loading_control=float(lc.intensities[index]),
                    target_name=target.name,
                    target=float(target.intensities[index]),
                    normalized=None if ratio is None else round(ratio, 6),
                    fold_change=fold,
                )
            )
        return rows

    def normalize_all(self, control_lane: int = 1) -> list[NormalizedRow]:
        rows: list[NormalizedRow] = []
        for target in self.targets():
            try:
                rows.extend(self.normalize(target, control_lane=control_lane))
            except ValueError:
                continue
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
                for lane, intensity in enumerate(entry.intensities, start=1):
                    bg = entry.backgrounds[lane - 1] if lane - 1 < len(entry.backgrounds) else ""
                    mean = entry.mean_intensities[lane - 1] if lane - 1 < len(entry.mean_intensities) else ""
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
        path = Path(path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "integrated_density" not in reader.fieldnames:
                raise ValueError("内参 CSV 缺少 integrated_density 列")
            buckets: dict[tuple[str, str], dict[str, list]] = {}
            for row in reader:
                kind = (row.get("kind") or "loading_control").strip()
                if kind not in {"loading_control", "target"}:
                    kind = "loading_control"
                name = (row.get("name") or "Loading Control").strip()
                group = (row.get("group") or "默认").strip()
                key = (name, group)
                bucket = buckets.setdefault(
                    key,
                    {"kind": kind, "intensities": [], "backgrounds": [], "mean_intensities": [], "source": row.get("source") or str(path)},
                )
                bucket["intensities"].append(float(row["integrated_density"]))
                bucket["backgrounds"].append(float(row["background"]) if row.get("background") not in (None, "") else 0.0)
                bucket["mean_intensities"].append(
                    float(row["mean_intensity"]) if row.get("mean_intensity") not in (None, "") else 0.0
                )
        count = 0
        for (name, group), bucket in buckets.items():
            self.upsert(
                QuantEntry(
                    kind=bucket["kind"],  # type: ignore[arg-type]
                    name=name,
                    group=group,
                    intensities=bucket["intensities"],
                    backgrounds=bucket["backgrounds"],
                    mean_intensities=bucket["mean_intensities"],
                    source=str(bucket["source"]),
                )
            )
            count += 1
        return count


def np_clip_index(index: int, count: int) -> int:
    return max(0, min(int(index), count - 1))


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
                    "loading_control": row.loading_control,
                    "target_name": row.target_name,
                    "target": row.target,
                    "normalized": "" if row.normalized is None else row.normalized,
                    "fold_change": "" if row.fold_change is None else row.fold_change,
                }
            )
    return path


def lane_intensities_from_measurements(measurements) -> tuple[list[float], list[float], list[float]]:
    """Aggregate auto-detected bands by lane for normalization."""
    by_lane: dict[int, list] = {}
    for item in measurements:
        by_lane.setdefault(item.lane, []).append(item)
    intensities: list[float] = []
    backgrounds: list[float] = []
    means: list[float] = []
    for lane in sorted(by_lane):
        bands = by_lane[lane]
        intensities.append(float(sum(b.integrated_density for b in bands)))
        backgrounds.append(float(np_mean([b.background for b in bands])))
        means.append(float(np_mean([b.mean_intensity for b in bands])))
    return intensities, backgrounds, means


def np_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

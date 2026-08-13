"""Export analysis results and presentation figures.

A figure leaving this module is meant to end up in a manuscript, so every
irreversible thing done to the pixels is both *chosen* by the caller and
*written down* next to the file. :func:`export_presentation_figure` renders
from :attr:`BlotImage.raw` rather than from the 8-bit preview, tags the file
with a physical resolution, and drops a ``*_export_info.txt`` sidecar that a
methods section can be copied out of.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .analysis import AnalysisResult, BandMeasurement
from .imageio import (
    MAPPING_AUTO,
    MAPPING_LINEAR,
    ROTATE_FAST,
    ROTATE_SHARP,
    BlotImage,
    DisplayMapping,
    MappingMode,
    apply_display_mapping,
    clipping_fractions,
    resolve_display_mapping,
)
from . import __version__
from .normalize import (
    ORIGIN_UNKNOWN,
    AnalysisSession,
    BackgroundOrigin,
    BandPick,
    RoiCheck,
    _format_bands,
    coerce_origin,
)
from .roi import RoiRect

__all__ = [
    "DEFAULT_DPI",
    "LAYER_ALL",
    "LAYER_BAND_BOXES",
    "LAYER_BAND_LABELS",
    "LAYER_ROI",
    "LAYER_SEPARATORS",
    "LOSSLESS_SUFFIXES",
    "MM_PER_INCH",
    "SUMMARY_LONG_FIELDS",
    "SUMMARY_VALUE_FOLD",
    "SUMMARY_VALUE_NORMALIZED",
    "FigureExport",
    "PrintScale",
    "SummaryExport",
    "SummaryRow",
    "WideCellGap",
    "WideRow",
    "WideTable",
    "build_summary_rows",
    "build_wide_table",
    "dpi_for_width_mm",
    "export_figure",
    "export_presentation_figure",
    "export_presentation_figures",
    "export_result",
    "export_summary_tables",
    "max_print_width_mm",
    "print_scale",
]

MM_PER_INCH = 25.4

DEFAULT_DPI = 300.0
"""Resolution tag used when the caller does not state a physical width.

Nothing is resampled to reach it: the tag only says how large the existing
pixels should be printed. :func:`print_scale` reports the width that implies.
"""

LOSSLESS_SUFFIXES = (".tif", ".tiff", ".png")
"""The only containers a figure may be written to. JPEG shifted grayscale
values by up to 6–7 levels on real blots, which is indistinguishable from a
real intensity difference in a printed panel."""

LAYER_BAND_BOXES = "band_boxes"
LAYER_BAND_LABELS = "band_labels"
LAYER_ROI = "roi"
LAYER_SEPARATORS = "separators"

LAYER_ALL: frozenset[str] = frozenset(
    {LAYER_BAND_BOXES, LAYER_BAND_LABELS, LAYER_ROI, LAYER_SEPARATORS}
)

_LAYER_LABELS = {
    LAYER_BAND_BOXES: "条带框",
    LAYER_BAND_LABELS: "文字标签",
    LAYER_ROI: "ROI 框",
    LAYER_SEPARATORS: "泳道分隔线",
}

_INTERPOLATION_LABELS = {
    cv2.INTER_NEAREST: "最近邻",
    ROTATE_FAST: "双线性",
    cv2.INTER_CUBIC: "双三次",
    cv2.INTER_AREA: "区域平均",
    ROTATE_SHARP: "Lanczos-4",
}

_BAND_COLOR = (0, 0, 255)
_SATURATED_COLOR = (0, 140, 255)
_ROI_COLOR = (0, 200, 255)
_SEPARATOR_COLOR = (255, 200, 0)


# --------------------------------------------------------------------- 结果表


def export_result(
    result: AnalysisResult,
    image_path: str | Path,
    output_dir: str | Path,
    *,
    origin: BackgroundOrigin | str = ORIGIN_UNKNOWN,
    dpi: float = DEFAULT_DPI,
) -> tuple[Path, Path]:
    """Write the band table and the annotated preview.

    ``origin`` states which background convention measured these numbers
    (README §9). It is a parameter rather than something read off the result
    because only the caller knows which path it took — and a file that does not
    say is a file whose numbers cannot be safely compared with anyone else's a
    month later.

    The preview here is the 8-bit working view, not a figure; use
    :func:`export_presentation_figure` for anything that will be published.
    """
    origin = coerce_origin(origin)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    csv_path = output / f"{stem}_bands.csv"
    preview_path = output / f"{stem}_annotated.png"
    fields = [
        "lane",
        "band",
        "x",
        "y",
        "width",
        "height",
        "area",
        "mean_intensity",
        "background",
        "integrated_density",
        "saturated_fraction",
        "detection_limit",
        "snr",
        "below_detection",
        "origin",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({**m.as_dict(), "origin": origin} for m in result.measurements)
    _write_image(preview_path, result.annotated_image, dpi=dpi, bgr=True)
    return csv_path, preview_path


# --------------------------------------------------------------------- 汇总表


SUMMARY_VALUE_FOLD = "fold_change"
SUMMARY_VALUE_NORMALIZED = "normalized"

SUMMARY_LONG_FIELDS = [
    "group",
    "lane",
    "sample",
    "target_name",
    "target_membrane",
    "target_membrane_path",
    "target",
    "target_origin",
    "target_band_pick",
    "target_bands",
    "target_below_detection",
    "target_roi_check",
    "control_name",
    "control_membrane",
    "control_membrane_path",
    "loading_control",
    "loading_control_origin",
    "loading_control_band_pick",
    "loading_control_bands",
    "loading_control_below_detection",
    "control_roi_check",
    "normalized",
    "control_lane",
    "fold_change",
    "note",
    "exported_at",
    "app_version",
]
"""Column order of ``summary_long.csv``.

``group`` / ``lane`` / ``sample`` lead because this file gets read by a human
first. Both sides carry their own ``membrane`` and ``membrane_path``: once the
loading control and its target live on different blots, "which two images was
this ratio taken between" is the smallest fact that makes the number
reproducible, and one membrane column would read as if they shared one.
"""


@dataclass(frozen=True)
class SummaryRow:
    """One row of the long table: group × lane × target protein."""

    group: str
    lane: int
    sample: str
    target_name: str
    target_membrane: str
    target_membrane_path: str
    target: float | None
    target_origin: BackgroundOrigin
    target_band_pick: BandPick
    target_bands: tuple[int, ...]
    target_below_detection: bool
    target_roi_check: RoiCheck
    control_name: str
    control_membrane: str
    control_membrane_path: str
    loading_control: float | None
    loading_control_origin: BackgroundOrigin
    loading_control_band_pick: BandPick
    loading_control_bands: tuple[int, ...]
    loading_control_below_detection: bool
    control_roi_check: RoiCheck
    normalized: float | None
    control_lane: int
    fold_change: float | None
    note: str = ""

    @property
    def cell(self) -> tuple[str, int, str]:
        """Which square of the wide table this row belongs in."""
        return (self.group, self.lane, self.target_name)

    def value(self, value_field: str = SUMMARY_VALUE_FOLD) -> float | None:
        if value_field == SUMMARY_VALUE_FOLD:
            return self.fold_change
        if value_field == SUMMARY_VALUE_NORMALIZED:
            return self.normalized
        raise ValueError(
            f"汇总表的取值只能是 {SUMMARY_VALUE_FOLD} 或 {SUMMARY_VALUE_NORMALIZED}，"
            f"收到 {value_field!r}"
        )

    def as_dict(self, *, exported_at: str = "", app_version: str = "") -> dict[str, object]:
        return {
            "group": self.group,
            "lane": self.lane,
            "sample": self.sample,
            "target_name": self.target_name,
            "target_membrane": self.target_membrane,
            "target_membrane_path": self.target_membrane_path,
            "target": _number(self.target),
            "target_origin": self.target_origin,
            "target_band_pick": self.target_band_pick,
            "target_bands": _format_bands(self.target_bands),
            "target_below_detection": int(self.target_below_detection),
            "target_roi_check": self.target_roi_check.as_field(),
            "control_name": self.control_name,
            "control_membrane": self.control_membrane,
            "control_membrane_path": self.control_membrane_path,
            "loading_control": _number(self.loading_control),
            "loading_control_origin": self.loading_control_origin,
            "loading_control_band_pick": self.loading_control_band_pick,
            "loading_control_bands": _format_bands(self.loading_control_bands),
            "loading_control_below_detection": int(self.loading_control_below_detection),
            "control_roi_check": self.control_roi_check.as_field(),
            "normalized": _number(self.normalized),
            "control_lane": self.control_lane,
            "fold_change": _number(self.fold_change),
            "note": self.note,
            "exported_at": exported_at,
            "app_version": app_version,
        }


@dataclass(frozen=True)
class WideCellGap:
    """One empty square of the wide table, and why it is empty."""

    group: str
    lane: int
    target_name: str
    reason: str


@dataclass(frozen=True)
class WideRow:
    group: str
    lane: int
    sample: str
    values: dict[str, float | None] = field(default_factory=dict)
    """Target name → value; missing or ``None`` means an empty cell."""


@dataclass(frozen=True)
class WideTable:
    """Group × lane rows, one column per target protein.

    No loading-control column, deliberately: an integrated density from the
    actin blot cannot be compared with one from another blot, so plotting it
    would be plotting exposure differences. The numbers are still in the long
    table, where they are labelled with the membrane they came from.
    """

    targets: tuple[str, ...] = ()
    rows: tuple[WideRow, ...] = ()
    gaps: tuple[WideCellGap, ...] = ()
    value_field: str = SUMMARY_VALUE_FOLD

    @property
    def header(self) -> list[str]:
        return ["group", "lane", "sample", *self.targets]


@dataclass(frozen=True)
class SummaryExport:
    """Everything the two tables and their sidecar came out as."""

    long_path: Path
    wide_path: Path
    readme_path: Path | None
    rows: tuple[SummaryRow, ...]
    wide: WideTable
    groups: tuple[str, ...]
    warnings: tuple[str, ...]
    generated_at: str
    app_version: str

    @property
    def value_field(self) -> str:
        return self.wide.value_field

    def readme_text(self) -> str:
        return _summary_readme_text(self)


def _number(value: float | None) -> object:
    return "" if value is None else value


def build_summary_rows(
    session: AnalysisSession, *, groups: Sequence[str] | None = None
) -> list[SummaryRow]:
    """Normalize every target of every group into long-table rows.

    Each group is paired against its own loading control and its own reference
    lane, which is what makes a multi-membrane experiment addable to one file:
    the pairing unit is the group, not the session. Problems are collected in
    :attr:`AnalysisSession.last_warnings`, as they are after
    :meth:`~wb_analyzer.normalize.AnalysisSession.normalize_all`.
    """
    names = list(groups) if groups is not None else session.groups()
    collected: list[str] = []
    rows: list[SummaryRow] = []
    for name in names:
        group = session.canonical_group(name)
        control = session.latest_loading_control(group)
        targets = session.targets(group)
        if not session.group_entries(group):
            collected.append(f"分组“{group}”里没有数据。")
            continue
        if control is None:
            for target in targets:
                collected.append(
                    f"“{target.name}”未能归一化：分组“{group}”还没有内参（Loading Control）"
                )
            continue
        if not targets:
            collected.append(f"分组“{group}”只有内参，还没有目标蛋白，汇总表里不会有这一组。")
            continue
        for target in targets:
            try:
                paired = session.normalize(target, control, warnings=collected)
            except ValueError as exc:
                collected.append(f"“{target.name}”未能归一化：{exc}")
                continue
            for row in paired:
                rows.append(
                    SummaryRow(
                        group=row.group,
                        lane=row.lane,
                        sample=session.lane_label(row.group, row.lane),
                        target_name=row.target_name,
                        target_membrane=row.target_membrane,
                        target_membrane_path=target.source,
                        target=row.target,
                        target_origin=row.target_origin,
                        target_band_pick=row.target_band_pick,
                        target_bands=row.target_bands,
                        target_below_detection=row.target_below_detection,
                        target_roi_check=row.target_roi_check,
                        control_name=row.loading_control_name,
                        control_membrane=row.loading_control_membrane,
                        control_membrane_path=control.source,
                        loading_control=row.loading_control,
                        loading_control_origin=row.loading_control_origin,
                        loading_control_band_pick=row.loading_control_band_pick,
                        loading_control_bands=row.loading_control_bands,
                        loading_control_below_detection=row.loading_control_below_detection,
                        control_roi_check=row.loading_control_roi_check,
                        normalized=row.normalized,
                        control_lane=row.control_lane,
                        fold_change=row.fold_change,
                        note=row.note,
                    )
                )
    session.last_warnings = collected
    return _annotate_duplicate_cells(rows)


def _annotate_duplicate_cells(rows: Sequence[SummaryRow]) -> list[SummaryRow]:
    """Say in the long table's ``note`` why the wide table left a square blank.

    Two records for one group × lane × protein — the same target quantified
    off two membranes — is a question only the user can settle. The wide table
    leaves the square empty and this note is where the reason lives, because
    the wide table has nowhere to put one.
    """
    duplicates: dict[tuple[str, int, str], list[SummaryRow]] = {}
    for row in rows:
        duplicates.setdefault(row.cell, []).append(row)
    out: list[SummaryRow] = []
    for row in rows:
        peers = duplicates[row.cell]
        if len(peers) < 2:
            out.append(row)
            continue
        membranes = "、".join(
            dict.fromkeys(p.target_membrane or p.target_membrane_path or "未命名" for p in peers)
        )
        note = (
            f"宽表该格留空：分组“{row.group}”泳道 {row.lane} 的“{row.target_name}”"
            f"有 {len(peers)} 条记录（膜 {membranes}），宽表不做合并——"
            "要不要合并是实验判断，软件不代做。"
        )
        out.append(replace(row, note=f"{row.note}；{note}" if row.note else note))
    return out


def _blank_reason(row: SummaryRow, value: str) -> str:
    """Why one square has no number, in the terms of the column it is in."""
    if row.note:
        return row.note
    if value == SUMMARY_VALUE_FOLD and row.normalized is not None:
        # The ratio is fine, the reference it would be divided by is not — a
        # per-column failure that the row's own note has no reason to mention.
        return f"基准泳道 {row.control_lane} 本身没有比值，这一组的 fold change 都算不出来"
    return "没有可用的数值"


def build_wide_table(
    rows: Sequence[SummaryRow], *, value: str = SUMMARY_VALUE_FOLD
) -> WideTable:
    """Pivot long rows into group × lane rows with one column per target.

    The row key is ``group + lane``, never lane alone: two experiments both
    have a lane 1, and merging them would put unrelated membranes on the same
    line. Unrelated groups make the table sparse instead, which is honest and
    which Prism reads without complaint.
    """
    if value not in (SUMMARY_VALUE_FOLD, SUMMARY_VALUE_NORMALIZED):
        raise ValueError(
            f"汇总表的取值只能是 {SUMMARY_VALUE_FOLD} 或 {SUMMARY_VALUE_NORMALIZED}，收到 {value!r}"
        )
    targets: list[str] = list(dict.fromkeys(row.target_name for row in rows))
    # Groups in the order the experiments were entered, lanes in gel order —
    # a target whose lanes start at 3 must not decide where lane 1 sits.
    group_order: list[str] = list(dict.fromkeys(row.group for row in rows))
    keys: list[tuple[str, int]] = sorted(
        {(row.group, row.lane) for row in rows},
        key=lambda key: (group_order.index(key[0]), key[1]),
    )
    by_cell: dict[tuple[str, int, str], list[SummaryRow]] = {}
    samples: dict[tuple[str, int], str] = {}
    for row in rows:
        by_cell.setdefault(row.cell, []).append(row)
        if row.sample:
            samples.setdefault((row.group, row.lane), row.sample)

    wide_rows: list[WideRow] = []
    gaps: list[WideCellGap] = []
    for group, lane in keys:
        values: dict[str, float | None] = {}
        for target in targets:
            peers = by_cell.get((group, lane, target), [])
            if not peers:
                values[target] = None
                gaps.append(
                    WideCellGap(group, lane, target, f"分组“{group}”没有“{target}”的记录")
                )
                continue
            if len(peers) > 1:
                membranes = "、".join(
                    dict.fromkeys(p.target_membrane or "未命名" for p in peers)
                )
                values[target] = None
                gaps.append(
                    WideCellGap(
                        group,
                        lane,
                        target,
                        f"该格对应 {len(peers)} 条记录（膜 {membranes}），未做合并",
                    )
                )
                continue
            number = peers[0].value(value)
            values[target] = number
            if number is None:
                gaps.append(WideCellGap(group, lane, target, _blank_reason(peers[0], value)))
        wide_rows.append(
            WideRow(group=group, lane=lane, sample=samples.get((group, lane), ""), values=values)
        )
    return WideTable(
        targets=tuple(targets),
        rows=tuple(wide_rows),
        gaps=tuple(gaps),
        value_field=value,
    )


_SUMMARY_VALUE_LABELS = {
    SUMMARY_VALUE_FOLD: "fold change（同分组内相对基准泳道的倍数，基准泳道 = 1）",
    SUMMARY_VALUE_NORMALIZED: "normalized（目标 / 内参的比值，未再除以基准泳道）",
}


def _summary_readme_text(export: SummaryExport) -> str:
    """The wide table's provenance, since the table itself carries none."""
    wide = export.wide
    lines: list[str] = [
        "Western Blot 汇总表导出说明",
        "本文件由 wb-gray-analyzer 自动生成，说明宽表里的数字是什么、空格为什么空。",
        "",
        f"宽表文件    {export.wide_path.name}",
        f"长表文件    {export.long_path.name}",
        f"生成时间    {export.generated_at}",
        f"软件版本    {export.app_version}",
        "",
        "【表里的值】",
        f"  数值口径    {_SUMMARY_VALUE_LABELS[wide.value_field]}",
        "  计算方式    目标积分灰度 / 同泳道的内参积分灰度"
        + ("，再除以基准泳道的同一比值" if wide.value_field == SUMMARY_VALUE_FOLD else ""),
        "  行            一行 = 一个分组的一个泳道（分组 + 泳道才是行的身份，只有泳道号会把两次实验并成一行）",
        "  列            一列 = 一个目标蛋白",
        "  没有内参列  内参的绝对积分值不能跨膜比较，画出来没有意义；",
        "              内参数值逐行保留在长表里，连同它来自哪张膜。",
        "",
        "【各分组的基准泳道】",
    ]
    seen_groups: list[str] = []
    for row in export.rows:
        if row.group not in seen_groups:
            seen_groups.append(row.group)
    for group in seen_groups:
        # One lane per group: normalization never substitutes a different one,
        # so every row of a group carries the same requested lane.
        group_rows = [r for r in export.rows if r.group == group]
        control_lane = group_rows[0].control_lane
        at_reference = [r for r in group_rows if r.lane == control_lane]
        if not at_reference:
            # Not "nobody named this lane": there is no row here to carry a
            # sample name because the lane has no data in this group at all,
            # and that — not an empty sample column — is why every fold change
            # below is blank. Blaming the sample name sends the reader off to
            # fix the wrong thing.
            lines.append(
                f"  {group}    泳道 {control_lane}"
                "（这一道在本组里没有数据，拿不到分母，这一组的 fold change 全部留空）"
            )
            continue
        sample = next((r.sample for r in at_reference if r.sample), "")
        suffix = f"（样品 {sample}）" if sample else "（这一道还没有样品名）"
        lines.append(f"  {group}    泳道 {control_lane}{suffix}")
        blocked = [
            name
            for name in dict.fromkeys(r.target_name for r in group_rows)
            if all(r.normalized is None for r in at_reference if r.target_name == name)
        ]
        if blocked:
            listed = "、".join(f"“{name}”" for name in blocked)
            which = "这些列" if len(blocked) > 1 else "这一列"
            lines.append(
                f"  {group}    ↑ 这一道在{listed}上算不出比值，"
                f"{which}的 fold change 全部留空（逐格理由见下）。"
            )
    if not seen_groups:
        lines.append("  （本次没有可汇总的分组）")

    lines += ["", "【本次用到的膜】"]
    seen_membranes: set[tuple[str, str, str]] = set()
    for row in export.rows:
        for kind, name, membrane, check in (
            ("内参", row.control_name, row.control_membrane, row.control_roi_check),
            ("目标", row.target_name, row.target_membrane, row.target_roi_check),
        ):
            key = (row.group, kind, f"{name}@{membrane}")
            if key in seen_membranes:
                continue
            seen_membranes.add(key)
            verdict = check.summary or "（这次定量没有做 ROI 自检）"
            lines.append(f"  {row.group}  {kind}  {name}  膜 {membrane or '未命名'}  {verdict}")
    if not seen_membranes:
        lines.append("  （本次没有可汇总的膜）")

    lines += ["", "【空格是什么意思】"]
    if wide.gaps:
        lines += [
            "  空格 = 这一格没有可填的数值，原因逐条列在下面。",
            "  宽表不做任何聚合：一格对应多条记录时一律留空，合不合并是实验判断，软件不代做。",
        ]
        for gap in wide.gaps:
            lines.append(
                f"  {gap.group}  泳道 {gap.lane}  {gap.target_name}    {gap.reason}"
            )
    else:
        lines.append("  本表没有空格：每个分组 × 泳道 × 目标蛋白都有一个数值。")

    if export.warnings:
        lines += ["", "【归一化时的提示】"]
        lines += [f"  {text}" for text in export.warnings]
    lines += [
        "",
        "【溯源】",
        f"  每个数字的来源（两侧的膜名、膜文件路径、背景口径、条带取用、检出限）都在 {export.long_path.name} 里逐行记录。",
        "",
    ]
    return "\n".join(lines)


def export_summary_tables(
    session: AnalysisSession,
    output_dir: str | Path,
    *,
    groups: Sequence[str] | None = None,
    value: str = SUMMARY_VALUE_FOLD,
    prefix: str = "summary",
    write_readme: bool = True,
    generated_at: str | None = None,
) -> SummaryExport:
    """Write ``summary_long.csv``, ``summary_wide.csv`` and the wide sidecar.

    One experiment spans several membranes, so this is the table the whole
    session adds up to — not the one group currently on screen, which is what
    :func:`~wb_analyzer.normalize.export_normalized_csv` still writes.

    The long table is the record: one row per group × lane × target, with both
    sides' membrane, background convention, band pick and detection flag. The
    wide table is the deliverable: group × lane rows and one column per
    target, ready to paste into GraphPad / Prism. The sidecar exists because
    the wide table carries no provenance of its own.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = build_summary_rows(session, groups=groups)
    warnings = tuple(session.last_warnings)
    if not rows:
        raise ValueError(
            "没有可导出的汇总数据：每个分组都需要一条内参和至少一个目标蛋白。"
            + ("问题：" + "；".join(warnings) if warnings else "")
        )
    wide = build_wide_table(rows, value=value)
    stamp = generated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    long_path = output / f"{prefix}_long.csv"
    with long_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_LONG_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict(exported_at=stamp, app_version=__version__))

    wide_path = output / f"{prefix}_wide.csv"
    with wide_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(wide.header)
        for row in wide.rows:
            writer.writerow(
                [row.group, row.lane, row.sample]
                + [_number(row.values.get(name)) for name in wide.targets]
            )

    export = SummaryExport(
        long_path=long_path,
        wide_path=wide_path,
        readme_path=None,
        rows=tuple(rows),
        wide=wide,
        groups=tuple(dict.fromkeys(row.group for row in rows)),
        warnings=warnings,
        generated_at=stamp,
        app_version=__version__,
    )
    if not write_readme:
        return export
    readme_path = output / f"{prefix}_wide_readme.txt"
    readme_path.write_text(export.readme_text(), encoding="utf-8-sig")
    return replace(export, readme_path=readme_path)


# ----------------------------------------------------------------- 物理尺寸


@dataclass(frozen=True)
class PrintScale:
    """How large a figure prints, given that its pixel count is fixed.

    This software never interpolates a figure up to hit a target resolution,
    so dpi is not a setting — it is a consequence of the physical width you
    place the image at. Reporting it honestly is the whole point.
    """

    width_px: int
    height_px: int
    dpi: float

    @property
    def width_mm(self) -> float:
        return self.width_px / self.dpi * MM_PER_INCH

    @property
    def height_mm(self) -> float:
        return self.height_px / self.dpi * MM_PER_INCH


def dpi_for_width_mm(width_px: int, width_mm: float) -> float:
    """Resolution that ``width_px`` pixels reach when printed ``width_mm`` wide."""
    width_mm = float(width_mm)
    if width_mm <= 0:
        raise ValueError(f"打印宽度必须为正数，收到 {width_mm} mm")
    return int(width_px) / (width_mm / MM_PER_INCH)


def max_print_width_mm(width_px: int, dpi: float) -> float:
    """Widest print that still reaches ``dpi`` without interpolating."""
    dpi = float(dpi)
    if dpi <= 0:
        raise ValueError(f"分辨率必须为正数，收到 {dpi} dpi")
    return int(width_px) / dpi * MM_PER_INCH


def print_scale(
    width_px: int,
    height_px: int,
    *,
    width_mm: float | None = None,
    dpi: float | None = None,
) -> PrintScale:
    """Resolve the print geometry from either a physical width or a dpi."""
    if width_mm is not None and dpi is not None:
        raise ValueError("打印宽度与 dpi 只能给一个：给了宽度，dpi 就是算出来的")
    if width_mm is not None:
        dpi = dpi_for_width_mm(width_px, width_mm)
    elif dpi is None:
        dpi = DEFAULT_DPI
    dpi = float(dpi)
    if not np.isfinite(dpi) or dpi <= 0:
        raise ValueError(f"分辨率必须为有限正数，收到 {dpi}")
    return PrintScale(int(width_px), int(height_px), dpi)


# --------------------------------------------------------------------- 写文件


def _resolve_suffix(path: Path, *, strict: bool) -> Path:
    suffix = path.suffix.lower()
    if suffix in LOSSLESS_SUFFIXES:
        return path
    if strict:
        raise ValueError(
            f"展示图只能存为无损格式（{'/'.join(LOSSLESS_SUFFIXES)}），收到 {path.suffix or '无扩展名'}。"
            "JPEG 等有损格式在实测中会把灰度改动 6–7 级，无法用于定量展示。"
        )
    # Legacy callers (and an old file dialog that still offers JPEG) get
    # silently redirected rather than an error; the returned path tells them.
    return path.with_suffix(".png")


def _write_image(path: Path, array: np.ndarray, *, dpi: float, bgr: bool) -> Path:
    """Write uint8 gray / uint8 colour / uint16 gray with a resolution tag.

    OpenCV cannot attach pHYs or the TIFF resolution tags and refuses 16-bit
    colour, so the figure writer goes through Pillow (already a dependency).
    """
    from PIL import Image

    if array.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"不支持的导出位深：{array.dtype}")
    data = array
    if data.ndim == 3:
        if data.dtype != np.uint8:
            raise ValueError("彩色标注图只支持 8 位输出")
        data = cv2.cvtColor(data[:, :, :3], cv2.COLOR_BGR2RGB) if bgr else data[:, :, :3]
    elif data.ndim != 2:
        raise ValueError(f"不支持的图像维度：{array.shape}")
    image = Image.fromarray(np.ascontiguousarray(data))
    options: dict[str, object] = {"dpi": (dpi, dpi)}
    if path.suffix.lower() in (".tif", ".tiff"):
        options["compression"] = "tiff_lzw"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, **options)
    return path


# --------------------------------------------------------------------- 标注层


def _draw_layers(
    canvas: np.ndarray,
    *,
    offset: tuple[int, int],
    measurements: Sequence[BandMeasurement],
    roi: RoiRect | None,
    separators: Sequence[float],
    layers: frozenset[str],
    line_width: int,
    font_scale: float,
) -> None:
    """Paint the requested overlay layers onto an 8-bit BGR crop, in place."""
    dx, dy = offset
    if roi is not None and LAYER_ROI in layers:
        cv2.rectangle(
            canvas,
            (roi.x - dx, roi.y - dy),
            (roi.x + roi.width - 1 - dx, roi.y + roi.height - 1 - dy),
            _ROI_COLOR,
            line_width,
        )
    if separators and LAYER_SEPARATORS in layers:
        top = (roi.y - dy) if roi is not None else 0
        bottom = (roi.y + roi.height - 1 - dy) if roi is not None else canvas.shape[0] - 1
        for x in separators:
            column = int(round(float(x))) - dx
            cv2.line(canvas, (column, top), (column, bottom), _SEPARATOR_COLOR, line_width)
    for item in measurements:
        color = _BAND_COLOR if item.saturated_fraction < 0.05 else _SATURATED_COLOR
        x1, y1 = item.x - dx, item.y - dy
        x2, y2 = x1 + item.width - 1, y1 + item.height - 1
        if LAYER_BAND_BOXES in layers:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_width)
        if LAYER_BAND_LABELS in layers:
            cv2.putText(
                canvas,
                f"L{item.lane}-B{item.band}",
                (x1, max(12, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                line_width,
                cv2.LINE_AA,
            )


def _crop_bounds(
    shape_hw: tuple[int, int], roi: RoiRect | None, pad: int
) -> tuple[int, int, int, int]:
    height, width = int(shape_hw[0]), int(shape_hw[1])
    if roi is None:
        return 0, 0, width, height
    clipped = roi.clipped(np.empty(shape_hw[:2], dtype=np.uint8))
    pad = max(0, int(pad))
    x1 = max(0, clipped.x - pad)
    y1 = max(0, clipped.y - pad)
    x2 = min(width, clipped.x + clipped.width + pad)
    y2 = min(height, clipped.y + clipped.height + pad)
    return x1, y1, x2, y2


# -------------------------------------------------------------------- 展示图


@dataclass(frozen=True)
class FigureExport:
    """Everything a caption would need to say about one exported figure."""

    path: Path
    info_path: Path | None
    mapping: DisplayMapping
    scale: PrintScale
    bit_depth: int
    channels: int
    clipped_low: float
    clipped_high: float
    annotated: bool
    layers: tuple[str, ...]
    source_path: Path | None
    source_bit_depth: int
    source_size: tuple[int, int]
    crop: tuple[int, int, int, int] | None
    rotation_deg: float
    rotation_interpolation: int
    compression: str
    generated_at: str

    @property
    def clipped_fraction(self) -> float:
        return self.clipped_low + self.clipped_high

    def info_text(self) -> str:
        return _info_text(self)


def _rotation_line(export: FigureExport) -> str:
    if abs(export.rotation_deg) < 1e-6:
        return "未旋转（图像未做任何几何校正）"
    direction = "顺时针" if export.rotation_deg > 0 else "逆时针"
    interp = _INTERPOLATION_LABELS.get(export.rotation_interpolation, "未知插值")
    return f"已旋正 {abs(export.rotation_deg):.2f}°（{direction}，{interp} 插值）"


def _mapping_lines(export: FigureExport) -> list[str]:
    mapping = export.mapping
    out_max = 255 if export.bit_depth == 8 else 65535
    lines = [
        f"  映射方式    {mapping.label}",
        f"  黑点        {mapping.black_point:.6g}（原始灰度，映射到输出 0）",
        f"  白点        {mapping.white_point:.6g}（原始灰度，映射到输出 {out_max}）",
    ]
    if mapping.mode == MAPPING_AUTO:
        lines.append(
            f"  取点依据    第 {mapping.low_percentile:g} / {mapping.high_percentile:g} 百分位"
        )
    channels = "单通道灰度" if export.channels == 1 else f"{export.channels} 通道彩色"
    container = "TIFF" if export.path.suffix.lower() in (".tif", ".tiff") else "PNG"
    lines.append("  传递函数    线性（未做伽马、曲线或局部调整）")
    lines.append(
        f"  输出格式    {export.bit_depth} 位{channels}，{container}（{export.compression}）"
    )
    lines.append(
        f"  削波像素    纯黑 {export.clipped_low * 100:.2f}%，纯白 {export.clipped_high * 100:.2f}%"
        "（占输出像素）"
    )
    # Plain ASCII markers only: this file gets opened in Notepad, pasted into
    # Word, and echoed through GBK consoles, and a symbol that survives none of
    # those is worse than no symbol.
    if export.clipped_fraction > 0:
        lines.append("  注意        落在纯黑/纯白上的像素之间已无灰度差异，这部分信息在本图中不可恢复。")
    else:
        lines.append("  说明        没有任何像素被压到纯黑或纯白，全部灰度层次都保留在图中。")
    return lines


def _method_sentence(export: FigureExport) -> str:
    mapping = export.mapping
    channels = "单通道灰度" if export.channels == 1 else "RGB 彩色（叠加标注）"
    parts = [
        f"图像以无损格式导出为 {export.bit_depth} 位{channels}，"
        f"未做插值缩放，保持采集分辨率 {export.scale.width_px} × {export.scale.height_px} 像素。"
    ]
    if mapping.mode == MAPPING_LINEAR:
        parts.append(
            f"灰度为线性全量程映射（原始值 0 → 黑，{mapping.white_point:.6g}（传感器满量程）→ 白），"
            "未做对比度拉伸、亮度或伽马调整。"
        )
    elif mapping.mode == MAPPING_AUTO:
        parts.append(
            f"灰度经线性对比度拉伸：原始值 {mapping.black_point:.6g}（第 {mapping.low_percentile:g} 百分位）→ 黑，"
            f"{mapping.white_point:.6g}（第 {mapping.high_percentile:g} 百分位）→ 白，未做伽马调整。"
        )
    else:
        parts.append(
            f"灰度按操作者指定的黑白点线性映射（{mapping.black_point:.6g} → 黑，"
            f"{mapping.white_point:.6g} → 白），未做伽马调整。"
        )
    if export.clipped_fraction > 0:
        parts.append(
            f"映射后 {export.clipped_low * 100:.2f}% 的像素落在纯黑、{export.clipped_high * 100:.2f}% 落在纯白，"
            "这部分像素之间的灰度差异在本图中已被削平。"
        )
    else:
        parts.append("映射后无像素落在纯黑或纯白，全部灰度层次均得到保留。")
    if abs(export.rotation_deg) >= 1e-6:
        interp = _INTERPOLATION_LABELS.get(export.rotation_interpolation, "未知插值")
        parts.append(f"图像整体旋正 {abs(export.rotation_deg):.2f}°（{interp} 插值）。")
    if export.crop is not None:
        parts.append("按感兴趣区整体裁切。")
    parts.append("全图统一处理，未做任何局部修饰、擦除或拼接。")
    return "".join(parts)


def _info_text(export: FigureExport) -> str:
    scale = export.scale
    source = export.source_path.name if export.source_path else "（未知，图像来自内存）"
    lines: list[str] = [
        "Western Blot 展示图导出说明",
        "本文件由 wb-gray-analyzer 自动生成，记录了这张图从原始数据到成图的全部处理。",
        "",
        f"图像文件    {export.path.name}",
        f"生成时间    {export.generated_at}",
        f"源文件      {source}",
        "",
        "【源数据】",
        f"  存储位深    {export.source_bit_depth} 位",
        f"  满量程      {export.mapping.white_level:.6g}（传感器白电平，非容器上限）",
        f"  原始尺寸    {export.source_size[0]} × {export.source_size[1]} px",
        "",
        "【几何】",
        f"  旋正        {_rotation_line(export)}",
    ]
    if export.crop is not None:
        x, y, w, h = export.crop
        lines.append(f"  裁切        原图 (x={x}, y={y}, w={w}, h={h})")
    else:
        lines.append("  裁切        无（整图导出）")
    lines += [
        "  重采样      无（未做任何插值放大或缩小）",
        f"  输出尺寸    {scale.width_px} × {scale.height_px} px",
        "",
        "【灰度映射】",
    ]
    lines += _mapping_lines(export)
    lines += [
        "",
        "【打印尺寸】",
        f"  写入文件的分辨率标签  {scale.dpi:.1f} dpi",
        f"  此图 {scale.width_px} px 宽，印成 {scale.width_mm:.1f} mm 宽时相当于 {scale.dpi:.1f} dpi"
        f"（高 {scale.height_px} px → {scale.height_mm:.1f} mm）",
        f"  若要求 300 dpi，最大可印 {max_print_width_mm(scale.width_px, 300.0):.1f} mm 宽",
        f"  若要求 600 dpi，最大可印 {max_print_width_mm(scale.width_px, 600.0):.1f} mm 宽",
        "  本软件不做插值放大：dpi 是印刷尺寸的结果，不是可以调大的设置。",
        "",
    ]
    if export.annotated:
        drawn = "、".join(_LAYER_LABELS.get(name, name) for name in export.layers)
        lines += [
            "【标注】",
            f"  已绘制      {drawn or '无'}",
            "  标注仅覆盖在图像之上，未改动下方任何像素的灰度。",
            "",
        ]
    lines += [
        "【方法学建议写法】",
        f"  {_method_sentence(export)}",
        "",
    ]
    return "\n".join(lines)


def export_presentation_figure(
    blot: BlotImage,
    path: str | Path,
    *,
    roi: RoiRect | None = None,
    pad: int = 8,
    mapping: MappingMode | DisplayMapping = MAPPING_LINEAR,
    black_point: float | None = None,
    white_point: float | None = None,
    bit_depth: int | None = None,
    annotate: bool = False,
    measurements: Sequence[BandMeasurement] = (),
    separators: Sequence[float] = (),
    layers: frozenset[str] | Sequence[str] = LAYER_ALL,
    line_width: int = 1,
    font_scale: float = 0.38,
    print_width_mm: float | None = None,
    dpi: float | None = None,
    rotation_deg: float = 0.0,
    rotation_interpolation: int = ROTATE_FAST,
    write_info: bool = True,
) -> FigureExport:
    """Render a publication figure from ``blot.raw`` and document what was done.

    ``mapping`` decides the only irreversible step. :data:`MAPPING_LINEAR`
    (the default) divides by the sensor full scale, so nothing is stretched
    and nothing clips; :data:`MAPPING_AUTO` reproduces the on-screen preview's
    percentile stretch, and :data:`MAPPING_MANUAL` takes ``black_point`` and
    ``white_point``. Whichever is used, the resolved numbers land in the
    ``*_export_info.txt`` sidecar beside the image.

    ``rotation_deg`` and ``rotation_interpolation`` are *declarations*: this
    function does not rotate, it records the straightening the caller already
    applied so the sidecar can state it.
    """
    if blot is None or blot.raw is None or blot.raw.size == 0:
        raise ValueError("没有可导出的图像")
    path = _resolve_suffix(Path(path), strict=True)

    source_h, source_w = blot.raw.shape[:2]
    x1, y1, x2, y2 = _crop_bounds((source_h, source_w), roi, pad)
    crop_raw = blot.raw[y1:y2, x1:x2]
    if crop_raw.size == 0:
        raise ValueError("裁切区域为空，无法导出")
    crop = None if roi is None else (x1, y1, x2 - x1, y2 - y1)

    if isinstance(mapping, DisplayMapping):
        resolved = mapping
    else:
        # Percentiles come from the whole frame, not the crop, so that
        # MAPPING_AUTO reproduces exactly the preview the user was looking at
        # when they framed the ROI. Clipping is then measured on what actually
        # ends up in the file.
        resolved = resolve_display_mapping(
            blot.raw,
            blot.white_level,
            mode=mapping,
            black_point=black_point,
            white_point=white_point,
        )

    if bit_depth is None:
        bit_depth = 8 if blot.bit_depth <= 8 else 16
    if bit_depth not in (8, 16):
        raise ValueError(f"导出位深只能是 8 或 16，收到 {bit_depth}")
    if annotate and bit_depth != 8:
        # Annotations are coloured, and no mainstream writer (Pillow included)
        # produces 16-bit RGB. Saying so beats silently downgrading.
        raise ValueError("彩色标注只能导出 8 位图；干净版可以保持 16 位")

    dtype = np.uint8 if bit_depth == 8 else np.uint16
    gray = apply_display_mapping(crop_raw, resolved, dtype=dtype)
    clipped_low, clipped_high = clipping_fractions(gray)

    if annotate:
        selected = frozenset(layers)
        unknown = selected - LAYER_ALL
        if unknown:
            raise ValueError(f"未知的标注图层：{sorted(unknown)}")
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        _draw_layers(
            canvas,
            offset=(x1, y1),
            measurements=measurements,
            roi=roi,
            separators=separators,
            layers=selected,
            line_width=max(1, int(line_width)),
            font_scale=float(font_scale),
        )
        pixels, channels = canvas, 3
        used_layers = tuple(sorted(selected))
    else:
        pixels, channels = gray, 1
        used_layers = ()

    scale = print_scale(
        pixels.shape[1], pixels.shape[0], width_mm=print_width_mm, dpi=dpi
    )
    _write_image(path, pixels, dpi=scale.dpi, bgr=True)

    export = FigureExport(
        path=path,
        info_path=None,
        mapping=resolved,
        scale=scale,
        bit_depth=bit_depth,
        channels=channels,
        clipped_low=clipped_low,
        clipped_high=clipped_high,
        annotated=annotate,
        layers=used_layers,
        source_path=blot.path,
        source_bit_depth=blot.bit_depth,
        source_size=(source_w, source_h),
        crop=crop,
        rotation_deg=float(rotation_deg),
        rotation_interpolation=int(rotation_interpolation),
        compression="LZW 无损" if path.suffix.lower() in (".tif", ".tiff") else "Deflate 无损",
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    if not write_info:
        return export
    info_path = path.with_name(f"{path.stem}_export_info.txt")
    info_path.write_text(export.info_text(), encoding="utf-8-sig")
    return replace(export, info_path=info_path)


def export_presentation_figures(
    blot: BlotImage,
    path: str | Path,
    *,
    clean: bool = True,
    annotated: bool = False,
    **kwargs: Any,
) -> tuple[FigureExport, ...]:
    """Export the clean figure and the annotated one in a single call.

    The annotated file always gets an ``_annotated`` suffix, so the pair is
    recognisable in a folder no matter which of the two was asked for. Each
    file carries its own sidecar: a figure that travels without its provenance
    is a figure whose provenance is gone.
    """
    if not clean and not annotated:
        raise ValueError("至少要导出一种版本（干净版或标注版）")
    if "annotate" in kwargs:
        raise ValueError("请用 clean= / annotated= 选择版本，不要再传 annotate=")
    base = _resolve_suffix(Path(path), strict=True)
    results: list[FigureExport] = []
    if clean:
        results.append(export_presentation_figure(blot, base, annotate=False, **kwargs))
    if annotated:
        target = base.with_name(f"{base.stem}_annotated{base.suffix}")
        # Colour overlays force 8-bit; the clean copy above keeps the depth
        # the caller asked for.
        results.append(
            export_presentation_figure(
                blot, target, **{**kwargs, "annotate": True, "bit_depth": 8}
            )
        )
    return tuple(results)


# ------------------------------------------------------------ 旧接口（兼容）


def export_figure(
    image_bgr: np.ndarray,
    path: str | Path,
    *,
    roi: RoiRect | None = None,
    pad: int = 8,
    dpi: float = DEFAULT_DPI,
) -> Path:
    """Export a presentation crop from an already-rendered 8-bit image.

    Kept for callers that only hold a preview buffer. It cannot state what was
    done to the pixels, so :func:`export_presentation_figure` is the one to
    reach for whenever a :class:`BlotImage` is available. Lossy suffixes are
    redirected to PNG rather than refused; the returned path is authoritative.
    """
    path = _resolve_suffix(Path(path), strict=False)
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("没有可导出的图像")
    x1, y1, x2, y2 = _crop_bounds(image_bgr.shape[:2], roi, pad)
    frame = image_bgr[y1:y2, x1:x2]
    return _write_image(path, np.ascontiguousarray(frame), dpi=dpi, bgr=True)

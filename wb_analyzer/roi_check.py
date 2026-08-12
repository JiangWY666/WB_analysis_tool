"""ROI 画对了没有：定量结果的结构化自检。

橡皮筋基线只能锚在 ROI 的上下两端，所以**上下边落在哪里，就等于在告诉工具
"这里是背景"**。落在与条带同一片平整背景上时，ROI 有多高几乎不影响结果；一旦
越过膜边、相邻的另一行条带或扫描图的黑边，弦就一路沉到那个错误的电平，中间
整块膜的本底被当成蛋白积进去——实测最强泳道总积分虚涨 60%、相对定量最差错
38%（见 README 方法学第 5、11 节）。

危险的地方在于这类错误**没有任何外在症状**：条带数看上去正常，表格里也全是
正数。本模块把调查里验证过的两条判据做成诊断，只报告、不拒绝：

1. **条带数 = 泳道数 × 每道条带数**（:data:`ISSUE_BAND_COUNT`）。对不上说明有
   段是背景被误切出来的。"每道条带数"在用户没指定时取各泳道条带数的**众数**，
   不要求用户填。
2. **每个条带框离 ROI 上下边至少半个条带高度**（:data:`ISSUE_EDGE_TOP` /
   :data:`ISSUE_EDGE_BOTTOM`）。这条补第一条的盲区：上边切进条带时条带数仍然
   正确，只有"条带框贴边"查得出来。"半个条带高度"由**实际检出的条带高度**算
   出来，不是写死的像素数。

在真实样图的 1044 个 ROI 上，两条同时通过时最大泳道误差从未超过 5%，零漏报；
代价是偏保守，一部分其实还行的 ROI 也会被拦下。

``expected_bands``（每道条带数）会把第一条判据变成同义反复——段是按这个数强
制切出来的。此时本模块**重新跑一遍自动分段**去对照（:data:`ISSUE_FORCED_COUNT`）：
自动分段不同意，就说明多出来的段是背景，ROI 画错了，只是被强制切分抹平了。
实测 ROI 画到 y 340–470（明显过高）时，自动分段给 30 条带一眼看出不对，填上
"每道 2 条"之后给出整整 20 条带、看上去完全正常，而最大泳道误差是 34%。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .analysis import BandMeasurement
from .quant import LaneProfile, _noise_scale, segment_bands
from .roi import RoiRect

Severity = Literal["error", "warning", "info"]
"""``error`` 数值多半已经错了；``warning`` 可疑，值得回头看一眼；``info`` 解读
其它结论时需要知道的背景，本身不代表有问题。"""

SEVERITY_RANK: dict[str, int] = {"info": 0, "warning": 1, "error": 2}
SEVERITY_LABELS: dict[str, str] = {"error": "严重", "warning": "警告", "info": "提示"}

ISSUE_BAND_COUNT = "band_count"
"""条带数与「泳道数 × 每道条带数」不符。"""

ISSUE_EDGE_TOP = "edge_clearance_top"
ISSUE_EDGE_BOTTOM = "edge_clearance_bottom"
"""条带框离 ROI 上 / 下边不足半个条带高度。"""

ISSUE_FORCED_COUNT = "forced_band_count"
"""条带数是 ``expected_bands`` 强制切出来的，不能当成 ROI 画对的证据。"""

ISSUE_EMPTY_LANE = "empty_lane"
"""整条泳道没检出条带峰，退回了"一条泳道一条记录"。"""

EDGE_CLEARANCE_FRAC = 0.5
"""条带框到 ROI 边的最小留白，以条带高度为单位。

半个条带高度是「这一侧还有干净背景可锚」的下限：段是在 4% 峰高处切的，紧邻
段外的行还带着条带自己的纵向拖尾（``quant.NOISE_BAND_MARGIN`` 用的是同一个
比例），噪声地板估计也正是从这个距离之外才开始取样。真实样图上把它调低到
任何比例都救不回 y 350–410 那种「上边压着条带」的框——它离最矮的一条条带
（9 px）都只有 3 px，所以这里没有"再宽容一点就不误报"的余地，宁可误报。
"""

EDGE_CLEARANCE_SEVERE_FRAC = 0.25
"""留白低于这个比例（条带高度）时，判为条带很可能已经被 ROI 边切断。

留一档中间地带是有意义的：留白介于四分之一和二分之一条带高度之间时，锚点
多半落在条带的拖尾上（数值偏差个位数百分比），而低于四分之一时上一条条带
往往整个被切掉一截，误差跳到两位数。
"""

BAND_HEIGHT_MIN = 2
"""参与"条带高度"统计的最小段高（像素）。比这更矮的段是噪声毛刺，不是条带。"""


@dataclass(frozen=True)
class RoiIssue:
    """一条诊断。``kind`` 给程序看，``title`` / ``detail`` 给用户看。"""

    kind: str
    severity: Severity
    title: str
    detail: str
    """具体到用户知道该改什么——"把 ROI 上边往上挪 5 px"而不是"ROI 可能有问题"。"""

    lanes: tuple[int, ...] = ()
    """涉及的泳道号（1 起）。"""

    bands: tuple[tuple[int, int], ...] = ()
    """涉及的条带，``(泳道号, 条带号)``，两者都是 1 起，与 ``BandMeasurement`` 对齐。"""


@dataclass(frozen=True)
class RoiQualityReport:
    """一次 ROI 定量的自检结论。诊断而已，不拒绝任何结果。"""

    issues: tuple[RoiIssue, ...] = ()

    lane_count: int = 0
    band_count: int = 0

    bands_per_lane: int | None = None
    """判定用的"每道条带数"；无法确定时为 ``None``。"""

    bands_per_lane_source: Literal["expected", "mode", "unknown"] = "unknown"
    """这个数是用户给的（``expected``）还是从各泳道众数猜的（``mode``）。"""

    band_height: float = 0.0
    """检出条带高度的中位数（像素），判据 2 的标尺。"""

    required_clearance: float = 0.0
    """由 :data:`EDGE_CLEARANCE_FRAC` 换算出的最小留白（像素）。"""

    top_clearance: int | None = None
    bottom_clearance: int | None = None
    """实测最小留白：所有条带框到 ROI 上 / 下边的最短距离（像素）。"""

    forced_split: bool = False
    """本次定量是否用了 ``expected_bands`` 强制切分。"""

    auto_band_count: int | None = None
    """强制切分时，同一批剖面按自动判据会切出多少条带；没强制时为 ``None``。"""

    @property
    def ok(self) -> bool:
        """没有 ``error`` 也没有 ``warning``。``info`` 不算问题。"""
        return self.severity is None

    @property
    def severity(self) -> Severity | None:
        """最高严重度；只有 ``info`` 或没有诊断时为 ``None``。"""
        actionable = [i for i in self.issues if i.severity != "info"]
        if not actionable:
            return None
        return max(actionable, key=lambda i: SEVERITY_RANK[i.severity]).severity

    def of_kind(self, kind: str) -> tuple[RoiIssue, ...]:
        return tuple(i for i in self.issues if i.kind == kind)

    def summary(self) -> str:
        """一行中文结论，状态栏可以直接显示。"""
        if self.lane_count == 0:
            return "ROI 自检：没有可评估的泳道。"
        if self.ok:
            per = f" × 每道 {self.bands_per_lane} 条" if self.bands_per_lane else ""
            clearances = [c for c in (self.top_clearance, self.bottom_clearance) if c is not None]
            margin = (
                f"条带框离 ROI 上下边最近 {min(clearances)} px（要求 ≥ {self.required_clearance:.0f} px）"
                if clearances
                else "没有可评估的条带框"
            )
            return (
                f"ROI 自检通过：{self.lane_count} 泳道{per}，共 {self.band_count} 条带；{margin}。"
            )
        actionable = [i for i in self.issues if i.severity != "info"]
        label = SEVERITY_LABELS[self.severity or "warning"]
        return f"ROI 自检{label}（{len(actionable)} 项）：" + "；".join(i.title for i in actionable) + "。"


@dataclass(frozen=True)
class _LaneGeometry:
    """自检真正需要的那点信息，从剖面或测量结果里都能凑出来。"""

    lane: int
    bands: tuple[tuple[int, int], ...]
    """每条条带在 ROI 内的 ``[上, 下)`` 行范围。"""

    whole_lane: bool
    """这条"条带"其实是整条泳道的兜底记录（没检出峰，或没开分段）。"""

    auto_bands: int | None = None
    """同一条剖面按自动判据会切出几段；无从得知时为 ``None``。"""


def _mode_count(counts: Sequence[int]) -> int:
    """众数；并列时取较小的那个。

    多切出来的段几乎总是背景被误当成条带，所以并列时押"更少"这一边——这样
    分歧会落在多出来的那些泳道身上，正是要点名的那些。
    """
    tally = Counter(counts)
    best = max(tally.values())
    return min(count for count, times in tally.items() if times == best)


def _band_height(geometries: Sequence[_LaneGeometry]) -> float:
    """检出条带高度的中位数（像素）。

    用中位数而不是均值：ROI 画错时总有几个段把背景一起吞了进去，那些段高得
    离谱，均值会被拖走。整条泳道的兜底记录和几个像素的毛刺不参与统计。

    比中位数更低的分位数（25 分位）会宽容不少——真实样图上误报少掉四成——
    但 5%~6% 那一档的漏报翻倍。这里的取舍是"宁可误报"，所以留在中位数。
    """
    heights = [
        stop - start
        for geometry in geometries
        if not geometry.whole_lane
        for start, stop in geometry.bands
        if stop - start >= BAND_HEIGHT_MIN
    ]
    return float(np.median(heights)) if heights else 0.0


def _lane_list(lanes: Sequence[int]) -> str:
    return "、".join(str(lane) for lane in lanes)


CONSENSUS_FRAC = 0.7
"""众数得覆盖这么多泳道，"每道几条"才算站得住的前提。"""


def _count_issue(
    geometries: Sequence[_LaneGeometry], per_lane: int, source: str
) -> RoiIssue | None:
    """判据 1：条带数 = 泳道数 × 每道条带数。"""
    extra = [g for g in geometries if len(g.bands) > per_lane]
    missing = [g for g in geometries if len(g.bands) < per_lane]
    if not extra and not missing:
        return None
    total = sum(len(g.bands) for g in geometries)
    premise = (
        f"你指定了每道 {per_lane} 条带"
        if source == "expected"
        else f"多数泳道是每道 {per_lane} 条带"
    )
    parts = [
        f"{premise}，{len(geometries)} 个泳道应共 {per_lane * len(geometries)} 条，"
        f"实际切出 {total} 条。"
    ]
    if extra:
        parts.append(
            f"第 {_lane_list([g.lane for g in extra])} 道多切了"
            f"（{'、'.join(f'{len(g.bands)} 条' for g in extra)}）——多出来的段通常不是条带，"
            f"而是 ROI 上下边越过膜边、越过相邻的另一行条带或扫到黑边之后，"
            f"基线沉下去、被误当成条带的背景。"
        )
    if missing:
        parts.append(
            f"第 {_lane_list([g.lane for g in missing])} 道少切了"
            f"（{'、'.join(f'{len(g.bands)} 条' for g in missing)}）——要么那里真的空着，"
            f"要么两条带贴得太紧、自动判据切不开（后者可以用「每道条带数」强制切分，"
            f"但先确认 ROI 画对了）。"
        )
    tally = Counter(len(g.bands) for g in geometries)
    if tally[per_lane] < len(geometries) * CONSENSUS_FRAC:
        parts.append(
            f"各泳道的条带数本身就很散（{'、'.join(f'{c} 条 {n} 道' for c, n in sorted(tally.items()))}），"
            f"连「每道几条」都难说——这种散法本身就说明 ROI 的上下边多半没有落在同一片平整背景上。"
        )
    parts.append("请把 ROI 的上下边收回到与条带同一片平整背景上，再重新定量。")
    return RoiIssue(
        kind=ISSUE_BAND_COUNT,
        severity="error",
        title="条带数与泳道数对不上",
        detail="".join(parts),
        lanes=tuple(g.lane for g in sorted(extra + missing, key=lambda g: g.lane)),
    )


def _clearance_issue(
    geometries: Sequence[_LaneGeometry],
    roi_height: int,
    band_height: float,
    *,
    top: bool,
) -> tuple[RoiIssue | None, int | None]:
    """判据 2：条带框离 ROI 上 / 下边至少半个条带高度。

    返回 ``(诊断, 实测最小留白)``；没有可评估的条带时留白为 ``None``。
    """
    margins: list[tuple[int, int, int]] = []  # (留白, 泳道, 条带号)
    for geometry in geometries:
        if geometry.whole_lane:
            continue
        for index, (start, stop) in enumerate(geometry.bands, start=1):
            margins.append(((start if top else roi_height - stop), geometry.lane, index))
    if not margins:
        return None, None
    required = band_height * EDGE_CLEARANCE_FRAC
    closest = min(margin for margin, _, _ in margins)
    if closest >= required:
        return None, closest
    offenders = [entry for entry in margins if entry[0] < required]
    severe = closest < band_height * EDGE_CLEARANCE_SEVERE_FRAC
    edge = "上" if top else "下"
    direction = "往上" if top else "往下"
    move = int(np.ceil(required - closest))
    cut = (
        f"ROI {edge}边很可能已经切进条带"
        if severe
        else f"ROI {edge}边压在条带的纵向拖尾上"
    )
    return (
        RoiIssue(
            kind=ISSUE_EDGE_TOP if top else ISSUE_EDGE_BOTTOM,
            severity="error" if severe else "warning",
            title=f"条带贴着 ROI {edge}边",
            detail=(
                f"离 ROI {edge}边最近的条带框只剩 {closest} px 留白，不足半个条带高度"
                f"（要求 ≥ {required:.0f} px，条带高度中位数 {band_height:.0f} px）。"
                f"{cut}：这一侧没有干净背景可以锚基线，橡皮筋的端点落在条带身上，"
                f"积分会明显偏差，而条带数看上去仍然完全正常——这种错误没有别的症状。"
                f"请把 ROI {edge}边{direction}挪至少 {move} px，"
                f"前提是不要越过膜/胶的边缘或相邻的另一行条带；"
                f"两个条件在这张图上无法同时满足时，说明这一侧本来就没有可用的背景。"
                f"涉及第 {_lane_list(sorted({lane for _, lane, _ in offenders}))} 道。"
            ),
            lanes=tuple(sorted({lane for _, lane, _ in offenders})),
            bands=tuple(sorted((lane, band) for _, lane, band in offenders)),
        ),
        closest,
    )


def _forced_issue(geometries: Sequence[_LaneGeometry], per_lane: int) -> tuple[RoiIssue, int | None]:
    """``expected_bands`` 下的对照诊断：自动分段同不同意这个条带数。"""
    auto = [g.auto_bands for g in geometries if g.auto_bands is not None]
    caveat = (
        f"本次用了「每道 {per_lane} 条带」强制切分，所以"
        f"「条带数 = 泳道数 × 每道条带数」必然成立——它是切出来的，"
        f"不能作为 ROI 画对的证据。"
    )
    if not auto:
        return (
            RoiIssue(
                kind=ISSUE_FORCED_COUNT,
                severity="info",
                title="条带数是强制切出来的",
                detail=caveat + "本次拿不到剖面，无法与自动分段对照，"
                "ROI 画对没有请以「条带贴边」那一条为准。",
            ),
            None,
        )
    auto_total = sum(auto)
    disagreeing = [g for g in geometries if g.auto_bands is not None and g.auto_bands != per_lane]
    if not disagreeing:
        return (
            RoiIssue(
                kind=ISSUE_FORCED_COUNT,
                severity="info",
                title="条带数是强制切出来的",
                detail=caveat
                + f"不指定时自动分段会给出同样的 {auto_total} 条带，两者一致；"
                f"ROI 画对没有仍请以「条带贴边」那一条为准。",
            ),
            auto_total,
        )
    detail_counts = "、".join(f"第 {g.lane} 道 {g.auto_bands} 条" for g in disagreeing)
    return (
        RoiIssue(
            kind=ISSUE_FORCED_COUNT,
            severity="error",
            title="强制切分掩盖了分段异常",
            detail=(
                caveat
                + f"把这个数去掉，同一批剖面自动分段会给出 {auto_total} 条带"
                f"（{detail_counts}），与每道 {per_lane} 条不符。"
                f"多出来的段通常是 ROI 上下边越过膜边之后被误当成条带的背景——"
                f"不填这个数时它会明显表现为条带数不对，填了之后条带数被强行改回"
                f"正常值，错误却还在。请先把 ROI 的上下边收回到与条带同一片平整背景上，"
                f"确认自动分段本身就给得出正确的条带数，再决定要不要用这个参数。"
            ),
            lanes=tuple(g.lane for g in disagreeing),
        ),
        auto_total,
    )


def _empty_lane_issue(geometries: Sequence[_LaneGeometry]) -> RoiIssue | None:
    empty = [g.lane for g in geometries if g.whole_lane]
    if not empty:
        return None
    if len(empty) == len(geometries):
        # 也可能不是空泳道，而是这次定量根本没按条带分段。两条判据都建立在
        # 条带框上，那种情况下它们什么也说不了，说清楚比硬报一句强。
        return RoiIssue(
            kind=ISSUE_EMPTY_LANE,
            severity="warning",
            title="没有一个泳道切出条带",
            detail=(
                f"{len(empty)} 个泳道全都退回了「一条泳道一条记录」。"
                f"要么这个 ROI 里确实没有能检出的条带峰（曝光不足、框错了位置），"
                f"要么本次定量关掉了条带分段（``split_bands=False``）。"
                f"两条判据都建立在条带框上，这种情况下自检无法判断 ROI 画得对不对。"
            ),
            lanes=tuple(empty),
        )
    return RoiIssue(
        kind=ISSUE_EMPTY_LANE,
        severity="warning",
        title="有泳道没检出条带",
        detail=(
            f"第 {_lane_list(empty)} 道没有检出任何条带峰，整条泳道退回"
            f"「一条泳道一条记录」，其积分扣完噪声地板后约等于 0，并会被标为低于检出限。"
            f"这些泳道不参与上面的条带数比对。泳道真的空着就正常；"
            f"肉眼看得见条带却检不出来，多半是曝光不足或 ROI 没框住这一道。"
        ),
        lanes=tuple(empty),
    )


def _evaluate(
    geometries: Sequence[_LaneGeometry],
    roi_height: int,
    expected_bands: int | None,
) -> RoiQualityReport:
    forced = expected_bands is not None
    measurable = [g for g in geometries if not g.whole_lane]
    band_total = sum(len(g.bands) for g in geometries)
    if not geometries:
        return RoiQualityReport(forced_split=forced)

    issues: list[RoiIssue] = []
    auto_total: int | None = None
    if forced:
        forced_issue, auto_total = _forced_issue(geometries, int(expected_bands))
        issues.append(forced_issue)

    per_lane: int | None = None
    source: Literal["expected", "mode", "unknown"] = "unknown"
    if forced:
        per_lane, source = int(expected_bands), "expected"
    elif measurable:
        per_lane, source = _mode_count([len(g.bands) for g in measurable]), "mode"
    if per_lane is not None and measurable:
        count_issue = _count_issue(measurable, per_lane, source)
        if count_issue is not None:
            issues.append(count_issue)

    height = _band_height(geometries)
    required = height * EDGE_CLEARANCE_FRAC
    top_issue, top_clearance = _clearance_issue(geometries, roi_height, height, top=True)
    bottom_issue, bottom_clearance = _clearance_issue(geometries, roi_height, height, top=False)
    issues.extend(issue for issue in (top_issue, bottom_issue) if issue is not None)

    empty_issue = _empty_lane_issue(geometries)
    if empty_issue is not None:
        issues.append(empty_issue)

    issues.sort(key=lambda i: -SEVERITY_RANK[i.severity])
    return RoiQualityReport(
        issues=tuple(issues),
        lane_count=len(geometries),
        band_count=band_total,
        bands_per_lane=per_lane,
        bands_per_lane_source=source,
        band_height=height,
        required_clearance=required,
        top_clearance=top_clearance,
        bottom_clearance=bottom_clearance,
        forced_split=forced,
        auto_band_count=auto_total,
    )


def _auto_band_count(lane: LaneProfile) -> int | None:
    """同一条剖面按自动判据会切出几段。

    重跑而不是记录：``compute_lane_profiles`` 把 ``expected_bands`` 直接透传给
    ``segment_bands``，自动结果当时并不存在。剖面和基线都在 :class:`LaneProfile`
    里，按 ``compute_lane_profiles`` 的原样重算一遍即可，几百个样本的开销可以
    忽略。
    """
    profile = np.asarray(lane.profile, dtype=float)
    baseline = np.asarray(lane.baseline, dtype=float)
    if profile.size == 0 or profile.shape != baseline.shape:
        return None
    residual = profile - baseline
    segments = segment_bands(
        np.maximum(residual, 0.0), noise_sigma=_noise_scale(residual), expected_bands=None
    )
    return len(segments)


def check_roi_quality(
    lanes: Sequence[LaneProfile],
    *,
    expected_bands: int | None = None,
) -> RoiQualityReport:
    """ROI 画对了没有：给 :func:`~wb_analyzer.quant.compute_lane_profiles` 的结果做自检。

    ``expected_bands`` 必须与定量时传给
    :func:`~wb_analyzer.quant.compute_lane_profiles` 的**同一个值**——本函数靠它
    知道条带数是不是强制切出来的。传错会让诊断跟着错。

    只诊断，不拒绝：用户完全可能有正当理由那样画 ROI（比如膜上就只有那么窄一
    条），所以这里从不抛异常，也不改动任何定量结果。
    """
    roi_height = int(lanes[0].profile.size) if lanes else 0
    geometries = [
        _LaneGeometry(
            lane=lane.lane,
            bands=tuple((int(s), int(e)) for s, e in lane.segments),
            whole_lane=_is_whole_lane(lane.segments, int(lane.profile.size)),
            auto_bands=_auto_band_count(lane) if expected_bands is not None else None,
        )
        for lane in lanes
    ]
    return _evaluate(geometries, roi_height, expected_bands)


def check_roi_quality_from_measurements(
    measurements: Sequence[BandMeasurement],
    roi: RoiRect,
    *,
    expected_bands: int | None = None,
) -> RoiQualityReport:
    """同样的自检，输入换成条带测量表 + ROI。

    给手上只有 :class:`~wb_analyzer.analysis.BandMeasurement` 的调用方用：CSV
    重新读回来、或者用户在界面上删掉了误检条带之后重新自检。少一样东西——没有
    剖面就没法重跑自动分段，``expected_bands`` 下的对照诊断会退化成一条提示。
    有剖面时请优先用 :func:`check_roi_quality`。

    ``roi`` 必须是**定量时实际用的那个** ROI，也就是
    :meth:`~wb_analyzer.roi.RoiRect.clipped` 之后的那个——留白是拿条带框相对
    ROI 上下边算的，边错了留白就错了。
    """
    grouped: dict[int, list[BandMeasurement]] = {}
    for item in measurements:
        grouped.setdefault(int(item.lane), []).append(item)
    roi_height = int(roi.height)
    geometries = []
    for lane in sorted(grouped):
        bands = tuple(
            sorted(
                (int(m.y) - int(roi.y), int(m.y) - int(roi.y) + int(m.height))
                for m in grouped[lane]
            )
        )
        geometries.append(
            _LaneGeometry(
                lane=lane, bands=bands, whole_lane=_is_whole_lane(bands, roi_height)
            )
        )
    return _evaluate(geometries, roi_height, expected_bands)


def _is_whole_lane(segments: Sequence[tuple[int, int]], roi_height: int) -> bool:
    """这条泳道是不是"没检出条带、整条记一条"的兜底记录。

    兜底记录的框正好铺满 ROI，它既贴着上边也贴着下边；不摘出来的话，判据 2
    会把一条空泳道报成"条带被切断"，条带高度的中位数也会被这条假条带拉高。
    """
    if len(segments) != 1 or roi_height <= 0:
        return False
    start, stop = segments[0]
    return int(start) == 0 and int(stop) == roi_height

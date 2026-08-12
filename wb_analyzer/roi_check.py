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
3. **条带的横向足迹没有被泳道分界切断**（:data:`ISSUE_FOOTPRINT_CLIPPED` /
   :data:`ISSUE_SEPARATOR_IN_BAND`）。前两条只看纵向，从不碰
   :attr:`~wb_analyzer.quant.LaneProfile.x1` / ``x2``，所以横向足迹塌掉时它们
   一声不吭：实测有过一个上下留白 84 px（要求 25 px）、条带数完全正常的 ROI，
   足迹只剩泳道宽度的 18.4%、积分只剩全宽的 59.8%，而自检报"通过"。这条判据
   补的就是这个盲区，做法见 :func:`_footprint_issues`。

在真实样图的 1044 个 ROI 上，前两条同时通过时最大泳道误差从未超过 5%，零漏报；
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

ISSUE_FOOTPRINT_CLIPPED = "footprint_clipped"
"""条带的横向足迹一侧顶死在泳道分界上，而且明显窄于泳道间距。"""

ISSUE_SEPARATOR_IN_BAND = "separator_in_band"
"""某条分隔线两侧的足迹都紧贴着它，且至少一侧的足迹是被切窄的。"""

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

FOOTPRINT_WIDTH_FRAC = 0.5
"""条带横向足迹的最小宽度，以**泳道间距**为单位。

和判据 2 用的是同一种标尺：那里是"半个条带高度"，这里是"半个泳道宽"，两者
都由这批泳道自己量出来，不写死像素数。

半这个数是实测的拐点。真实样图上 1584 个 ROI、11087 条泳道里，把"一侧顶死
在分界上、另一侧还留着空白"的足迹按占泳道间距的比例分档，被分界切断的比例
是：<20% 87%、20~30% 87%、30~40% 85%、40~50% 79%、50~60% 50%、>80% 30%。
50% 以下每一档都稳定在 80% 以上，50% 往上一档直接掉到随机水平（这批样本的
基础率 52%），所以门槛就落在这里。
"""

FOOTPRINT_SEVERE_FRAC = 0.25
"""足迹窄到这个比例（泳道间距）以下时判为严重。

和 :data:`EDGE_CLEARANCE_SEVERE_FRAC` 一样是"再砍一半"。分档依据不是把握
——50% 以下各档的命中率其实差不多——而是**损失量**：足迹只剩泳道的两成时，
条带被切掉的那截通常比留下的还多，积分能少四成以上；剩下四成多时丢的往往
是一条边的拖尾，个位数百分比。
"""

FOOTPRINT_MIN_LANES = 2
"""少于这么多泳道就没有"泳道间距"可言，本条判据整个不参与。"""


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

    lane_pitch: float = 0.0
    """泳道间距（像素）：相邻条带足迹中心距的中位数，判据 3 的标尺。

    量的是足迹中心而不是分隔线，因为自检拿不到分隔线；两者在足迹正常的泳道
    上等价，实测中位偏差 0.0%、5~95 分位 -15.6%~+12.6%。
    """

    footprint_frac: float | None = None
    """最窄的条带足迹占泳道间距的比例；无法评估时为 ``None``。"""

    required_footprint: float = 0.0
    """由 :data:`FOOTPRINT_WIDTH_FRAC` 换算出的最小足迹宽度（像素）。"""

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
            # 横向那一句不能省：整条判据就是因为"通过"只讲了纵向、而足迹已经
            # 塌到 18.4% 才补上的，报通过时也要把查过的那一维说出来。
            lateral = (
                f"；没有被泳道分界切断的条带足迹"
                f"（最窄的一道占泳道间距 {self.footprint_frac:.0%}）"
                if self.footprint_frac is not None
                else ""
            )
            return (
                f"ROI 自检通过：{self.lane_count} 泳道{per}，共 {self.band_count} 条带；"
                f"{margin}{lateral}。"
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

    x1: int = 0
    x2: int = 0
    """条带横向足迹的绝对像素范围 ``[x1, x2)``，即实际参与积分的那几列。"""


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
        # 用户已经填过「每道条带数」时，再劝他去填一遍是自相矛盾的：强制切分是
        # 「有几个谷切几个」，切不开时既不报错也不硬切。
        remedy = (
            "——强制切分是「有几个谷切几个」，这几道连可切的谷都没有：要么本来"
            "就只有一条带，要么两条贴得中间没有谷。"
            if source == "expected"
            else "——要么那里真的空着，要么两条带贴得太紧、自动判据切不开"
            "（后者可以用「每道条带数」强制切分，但先确认 ROI 画对了）。"
        )
        parts.append(
            f"第 {_lane_list([g.lane for g in missing])} 道少切了"
            f"（{'、'.join(f'{len(g.bands)} 条' for g in missing)}）{remedy}"
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


@dataclass(frozen=True)
class _Clipped:
    """一条足迹被分界切断的泳道。"""

    lane: int
    width: int
    frac: float
    """足迹宽度占泳道间距的比例。"""

    side: str
    """顶死的那一侧，``"左"`` 或 ``"右"``。"""

    neighbour: int | None
    """分界另一侧的泳道号；顶在 ROI 自己的边上时为 ``None``。"""

    margin: int
    """另一侧留下的空白（像素）——正是它证明这条足迹确实收紧过。"""


def _footprint_layout(
    geometries: Sequence[_LaneGeometry],
) -> tuple[list[_LaneGeometry], list[int], float]:
    """按泳道号排好的足迹、相邻足迹之间的空隙、泳道间距。

    间距取相邻足迹**中心距的中位数**。分隔线的位置自检拿不到（
    :class:`~wb_analyzer.quant.LaneProfile` 只记了足迹 ``[x1, x2)``），但中心
    距在足迹正常的泳道上就等于泳道宽，而且是中位数——一两条塌掉的足迹把自己
    的中心拉偏，也拉不动这个数。真实样图上它对真实泳道宽的偏差中位数 0.0%、
    5~95 分位 -15.6%~+12.6%。

    兜底记录（整条泳道一条）也参与间距统计：它的足迹就是整条泳道条，中心正
    是泳道中心，是个好样本；它只是不参与被点名。
    """
    ordered = sorted(geometries, key=lambda g: g.lane)
    if len(ordered) < FOOTPRINT_MIN_LANES:
        return ordered, [], 0.0
    if any(g.x2 <= g.x1 for g in ordered):
        return ordered, [], 0.0
    centers = [(g.x1 + g.x2) / 2.0 for g in ordered]
    steps = np.diff(centers)
    pitch = float(np.median(steps)) if steps.size else 0.0
    gaps = [ordered[i + 1].x1 - ordered[i].x2 for i in range(len(ordered) - 1)]
    return ordered, gaps, max(pitch, 0.0)


def _clipped_lanes(
    ordered: Sequence[_LaneGeometry], gaps: Sequence[int], pitch: float
) -> list[_Clipped]:
    """足迹一侧顶死在分界上、另一侧还空着，而且整体不到半个泳道宽的那些泳道。

    三个条件缺一不可，每一条都在挡一类东西：

    * **窄**（< :data:`FOOTPRINT_WIDTH_FRAC` × 泳道间距）——只有窄了才谈得上
      丢信号。单看窄不行：真实样图上只按"窄"点名的准确率 0.47，比 0.52 的基础
      率还低，因为一条本来就细的带落在一条宽泳道里完全正常。
    * **一侧顶死**（与相邻足迹的空隙 ≤ 0）——空隙为 0 意味着两条足迹在同一个
      像素上交接，也就是双方都顶到了那条分隔线上。单看顶死更不行：实测 75% 的
      相邻足迹本来就是贴着的（膜的本底把泳道之间的空白抬到了收紧阈值以上，
      足迹于是铺满整条泳道），准确率 0.43 对基础率 0.37。
    * **另一侧还空着**——这是"这条足迹确实收紧过"的证明。足迹收不动时会整条
      退回泳道全宽，那种足迹**两侧都**贴着邻居；实测 5378 条两侧都贴的足迹里，
      收紧过的一条也没有（0/5378），而"一侧贴一侧空"的 1139 条里 94.2% 是真
      收紧过的。没有这一条，门槛 0.5 的准确率从 0.83 掉到 0.63。

    三条合起来实测准确率 0.83（341 真 / 68 假，1584 个真实 ROI、11087 条泳道）。
    """
    if pitch <= 0.0 or len(ordered) < FOOTPRINT_MIN_LANES:
        return []
    required = pitch * FOOTPRINT_WIDTH_FRAC
    clipped: list[_Clipped] = []
    for index, geometry in enumerate(ordered):
        if geometry.whole_lane:
            continue
        width = geometry.x2 - geometry.x1
        if width >= required:
            continue
        sides = [
            ("左", gaps[index - 1], ordered[index - 1].lane) if index > 0 else None,
            ("右", gaps[index], ordered[index + 1].lane) if index < len(gaps) else None,
        ]
        present = [side for side in sides if side is not None]
        pinned = [side for side in present if side[1] <= 0]
        free = [side for side in present if side[1] > 0]
        if not pinned or not free:
            continue
        name, _, neighbour = pinned[0]
        clipped.append(
            _Clipped(
                lane=geometry.lane,
                width=width,
                frac=width / pitch,
                side=name,
                neighbour=neighbour,
                margin=max(side[1] for side in free),
            )
        )
    return clipped


def _footprint_issues(
    ordered: Sequence[_LaneGeometry], gaps: Sequence[int], pitch: float
) -> list[RoiIssue]:
    """判据 3：条带的横向足迹有没有被泳道分界切断。

    判据 1 和判据 2 只读条带框的上下沿，横向那一维它们从来没看过，所以"分隔线
    从条带中间穿过去"这类错误对它们完全隐形：条带数照样对，上下留白照样够，
    表里全是正数，只是每条带的信号被劈成了两半算进两个泳道。

    这里**不**去看"足迹保住了全宽积分的百分之多少"。那个量的两侧分布是压在
    一起的——同一批样本上，足迹被切断的泳道保留量 5 分位 0.561、中位 1.000，
    没被切断的 5 分位 0.659、中位 1.000，中位数完全重合，按 0.5 切的准确率只有
    0.69 而召回 4%。判据换成几何：足迹在哪儿停下来的、旁边还剩多少空白。
    """
    clipped = _clipped_lanes(ordered, gaps, pitch)
    if not clipped:
        return []
    lanes = {item.lane for item in clipped}
    band_refs = tuple(
        sorted(
            (g.lane, index)
            for g in ordered
            if g.lane in lanes
            for index in range(1, len(g.bands) + 1)
        )
    )
    worst = min(clipped, key=lambda item: item.frac)
    severe = worst.frac < FOOTPRINT_SEVERE_FRAC
    entries: list[str] = []
    for item in clipped:
        neighbour = (
            f"与第 {item.neighbour} 道的分界" if item.neighbour is not None else "ROI 的边"
        )
        entries.append(
            f"第 {item.lane} 道足迹只有 {item.width} px（{item.frac:.0%} 泳道宽），"
            f"{item.side}侧贴着{neighbour}，{'右' if item.side == '左' else '左'}侧还留着"
            f" {item.margin} px 空白"
        )
    parts = [
        f"第 {_lane_list(sorted(lanes))} 道的条带足迹一边顶死在泳道分界上、"
        f"另一边还空着，而且不到半个泳道宽"
        f"（泳道间距 {pitch:.0f} px，要求足迹 ≥ {pitch * FOOTPRINT_WIDTH_FRAC:.0f} px）：",
        "；".join(entries),
        "。",
    ]
    parts.append(
        "足迹是每个泳道条里真正参与积分的那几列，两侧的空白本该被排除掉。"
        "一侧收得动、另一侧却正好停在分界上，说明条带在那里还没结束就被分界截断了——"
        "被截掉的那半要么算进了隔壁泳道，要么整个落在 ROI 外面，两边的积分都偏低。"
    )
    if severe:
        parts.append(
            f"最窄的一道只剩 {worst.frac:.0%} 泳道宽，被切掉的通常比留下的还多，"
            f"积分能少四成以上。"
        )
    parts.append(
        "条带数和上下留白对这种错误完全无感，表里也全是正数，没有别的症状。"
        "请把那条分界挪到两条条带之间的空隙里（顶着 ROI 边的就把 ROI 的左右边往外拉），"
        "让每条条带完整地落在一个泳道条里，再重新定量。"
    )
    issues = [
        RoiIssue(
            kind=ISSUE_FOOTPRINT_CLIPPED,
            severity="error" if severe else "warning",
            title="条带足迹被泳道分界切断",
            detail="".join(parts),
            lanes=tuple(sorted(lanes)),
            bands=band_refs,
        )
    ]
    separator = _separator_issue(ordered, gaps, clipped)
    if separator is not None:
        issues.append(separator)
    return issues


def _separator_issue(
    ordered: Sequence[_LaneGeometry], gaps: Sequence[int], clipped: Sequence[_Clipped]
) -> RoiIssue | None:
    """分隔线切在条带里的直接证据：两侧足迹都紧贴着同一条分隔线。

    单凭"两侧都紧贴"不成立——那也是两条足迹都退回全宽时的样子，实测 75% 的
    相邻足迹都贴着，准确率 0.43 几乎等于基础率。配上"其中至少一侧的足迹是被
    切窄的"（即 :func:`_clipped_lanes` 点过名的那条）才站得住：实测准确率
    0.886（171 真 / 22 假）。

    和 :data:`ISSUE_FOOTPRINT_CLIPPED` 说的是同一件事的两面，分开报是因为要
    修的东西不同——那条说"第 3 道的数不对"，这条直接指出"第 3 道和第 4 道之间
    的那条分隔线画错了位置"，而且它一次点掉两个泳道。
    """
    flagged = {item.lane for item in clipped}
    hits: list[tuple[int, int]] = []
    for index, gap in enumerate(gaps):
        if gap > 0:
            continue
        left, right = ordered[index].lane, ordered[index + 1].lane
        if left in flagged or right in flagged:
            hits.append((left, right))
    if not hits:
        return None
    where = "、".join(f"第 {left} 道与第 {right} 道之间" for left, right in hits)
    lanes = sorted({lane for pair in hits for lane in pair})
    return RoiIssue(
        kind=ISSUE_SEPARATOR_IN_BAND,
        severity="error",
        title="分隔线切在条带里",
        detail=(
            f"{where}的分隔线，两侧的条带足迹都紧贴着它，中间一个像素的空白都没留下，"
            f"而其中至少一侧的足迹明显偏窄。两条足迹在同一条线上交接，意味着这条线"
            f"是从一整条条带中间穿过去的：同一条带的左右两半被算成了两个泳道的信号，"
            f"两个泳道的定量结果都不能用，泳道之间的相对比较更是无从谈起。"
            f"请把这条分隔线拖到两条条带之间的空隙上——"
            f"如果那里根本没有空隙，说明这两个「泳道」本来就是同一条泳道，"
            f"应该把这条分隔线删掉。"
        ),
        lanes=tuple(lanes),
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
                f"要么本次定量关掉了条带分段（split_bands=False）。"
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

    ordered, gaps, pitch = _footprint_layout(geometries)
    issues.extend(_footprint_issues(ordered, gaps, pitch))
    narrowest = min(
        ((g.x2 - g.x1) / pitch for g in ordered if not g.whole_lane and pitch > 0.0),
        default=None,
    )

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
        lane_pitch=pitch,
        footprint_frac=narrowest,
        required_footprint=pitch * FOOTPRINT_WIDTH_FRAC,
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
            x1=int(lane.x1),
            x2=int(lane.x2),
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

    横向足迹取每个泳道所有条带框的并集 ``[min(x), max(x + width))``。同一泳道
    的条带框本来就共用
    :attr:`~wb_analyzer.quant.LaneProfile.x1` / ``x2``，所以这和
    :func:`check_roi_quality` 看到的是同一个足迹；用户删掉几条带也不影响，
    剩下的框仍然带着那个足迹。
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
                lane=lane,
                bands=bands,
                whole_lane=_is_whole_lane(bands, roi_height),
                x1=min(int(m.x) for m in grouped[lane]),
                x2=max(int(m.x) + int(m.width) for m in grouped[lane]),
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

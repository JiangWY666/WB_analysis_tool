"""强度剖面面板：左图 ROI 横向剖面（可拖分隔线），右图选中泳道纵向剖面。"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from . import theme

pg.setConfigOptions(antialias=True)

# 配色统一取自 theme，本模块不再自带一份
ACCENT = theme.ACCENT
ACCENT_DIM = theme.ACCENT_DIM
SEPARATOR = theme.SEPARATOR
BASELINE = theme.BASELINE
PLOT_BG = theme.CANVAS_BG
AXIS_PEN = theme.BORDER
AXIS_TEXT = theme.TEXT_DIM
TITLE_COLOR = theme.TEXT_BRIGHT
LANE_LABEL = theme.LANE_LABEL
REGION_BRUSHES = theme.PLOT_REGION_RGBA
FILL_BRUSH = theme.PLOT_FILL_RGBA
SEP_MARGIN = 2.0

COLUMN_TITLE = "ROI 横向剖面（拖橙线调分隔）"
LANE_TITLE_EMPTY = "泳道纵向剖面（先 ROI 定量）"
# 两张图纵轴的量纲不同：左图是 ROI 每一列的平均信号，右图是整条泳道宽度上
# 求和后的信号。以前都标"信号强度"，读数会被误解成同一口径。
COLUMN_Y_LABEL = "列平均信号（每像素灰度）"
LANE_Y_LABEL = "行累加信号（整条泳道宽度求和）"


def _style_plot(widget: pg.PlotWidget) -> None:
    widget.setBackground(PLOT_BG)
    plot_item = widget.getPlotItem()
    plot_item.showGrid(x=True, y=True, alpha=0.12)
    for name in ("left", "bottom"):
        axis = plot_item.getAxis(name)
        axis.setPen(pg.mkPen(AXIS_PEN))
        axis.setTextPen(pg.mkPen(AXIS_TEXT))
    view_box = plot_item.getViewBox()
    view_box.setMouseMode(pg.ViewBox.RectMode)
    plot_item.setMenuEnabled(False)


def _legend(plot_item: pg.PlotItem) -> pg.LegendItem:
    legend = plot_item.addLegend(offset=(-8, 8), labelTextColor=AXIS_TEXT)
    legend.setBrush(pg.mkBrush(38, 40, 44, 200))
    legend.setPen(pg.mkPen(AXIS_PEN))
    return legend


def _swatch(pen, fill_rgba=None) -> pg.PlotDataItem:
    """只用来当图例色块的空 PlotDataItem。

    图例样例必须有 ``opts``（LegendItem 直接读它画那 20×20 的小图），所以分隔线、
    泳道分区、积分填充这些非曲线元素不能把 InfiniteLine / FillBetweenItem 塞进
    图例——真正绘制时会抛 AttributeError 连带整张图画不出来。
    """
    item = pg.PlotDataItem(pen=pen)
    if fill_rgba is not None:
        item.setFillLevel(0)
        # 图例色块只有 16px 宽，照搬图内那么淡的填充等于看不见，提一点透明度
        item.setFillBrush(pg.mkBrush(fill_rgba[0], fill_rgba[1], fill_rgba[2], 150))
    return item


class ProfilePanel(QWidget):
    """底部 dock 内容。分隔线拖动实时发 separatorsChanged（绝对图像 x）。"""

    separatorsChanged = Signal(list)
    editCommitted = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._updating = False
        self._profile: np.ndarray | None = None
        self._roi_x = 0.0
        self._roi_w = 0.0
        self._sep_lines: list[pg.InfiniteLine] = []
        self._regions: list[pg.LinearRegionItem] = []
        self._labels: list[pg.TextItem] = []
        self._lanes: list = []
        self._measurements: list = []
        self._current_lane: int | None = None

        self._col_plot = pg.PlotWidget()
        _style_plot(self._col_plot)
        self._col_plot.getPlotItem().setTitle(
            COLUMN_TITLE, color=TITLE_COLOR, size="10pt"
        )
        self._col_plot.setLabel("bottom", "X（像素）")
        self._col_plot.setLabel("left", COLUMN_Y_LABEL)
        self._col_curve = self._col_plot.plot(
            pen=pg.mkPen(ACCENT, width=2), name="列平均信号"
        )
        col_legend = _legend(self._col_plot.getPlotItem())
        col_legend.addItem(self._col_curve, "列平均信号")
        # 图例条目要能说明"橙线是分隔线、浅底是泳道分区"
        self._col_samples = [
            _swatch(pg.mkPen(SEPARATOR, width=2)),
            _swatch(pg.mkPen(ACCENT_DIM, width=1), REGION_BRUSHES[0]),
        ]
        col_legend.addItem(self._col_samples[0], "泳道分隔线（可拖动）")
        col_legend.addItem(self._col_samples[1], "泳道分区（深浅相间）")

        self._lane_plot = pg.PlotWidget()
        _style_plot(self._lane_plot)
        self._lane_plot.getPlotItem().setTitle(
            LANE_TITLE_EMPTY, color=TITLE_COLOR, size="10pt"
        )
        self._lane_plot.setLabel("bottom", "Y（像素）")
        self._lane_plot.setLabel("left", LANE_Y_LABEL)
        # 图例一次建好：show_lane 每次都 clear()，而 clear() 只会摘掉真正加进图里的
        # item，样例 item 会一轮一轮堆在图例里
        lane_legend = _legend(self._lane_plot.getPlotItem())
        self._lane_samples = [
            _swatch(pg.mkPen(ACCENT, width=2)),
            _swatch(pg.mkPen(BASELINE, width=1.5, style=Qt.PenStyle.DashLine)),
            _swatch(pg.mkPen(ACCENT_DIM, width=1), FILL_BRUSH),
        ]
        for sample, label in zip(
            self._lane_samples, ("泳道信号", "扣除用基线", "积分区间（条带）")
        ):
            lane_legend.addItem(sample, label)

        # RectMode 下误框选后没有右键菜单可以 View All，角落那个 A 按钮又太小
        reset_button = QPushButton("复位视图")
        reset_button.setToolTip("把两张剖面图恢复为自动缩放（框选放大后用它退出）")
        reset_button.clicked.connect(self.reset_views)
        hint = QLabel("框选可放大，双击图内空白也能复位")
        hint.setObjectName("dimHint")
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(8)
        bar.addWidget(hint)
        bar.addStretch(1)
        bar.addWidget(reset_button)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._col_plot)
        splitter.addWidget(self._lane_plot)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(splitter, 1)
        layout.addLayout(bar)
        self.setMinimumHeight(200)
        for widget in (self._col_plot, self._lane_plot):
            widget.getPlotItem().getViewBox().installEventFilter(self)

    def eventFilter(self, obj, event):  # noqa: N802 - Qt API
        """双击图内空白 = View All：RectMode 下误框选后总得有条退路。"""
        if event.type() == QEvent.Type.GraphicsSceneMouseDoubleClick:
            self.reset_views()
            return True
        return super().eventFilter(obj, event)

    def reset_views(self) -> None:
        for widget in (self._col_plot, self._lane_plot):
            plot_item = widget.getPlotItem()
            plot_item.enableAutoRange()
            plot_item.getViewBox().autoRange()

    # ------------------------------------------------------------------ 左图：横向剖面

    def set_column_profile(
        self,
        profile: np.ndarray | None,
        roi_x: float,
        roi_width: float,
        separators: list[float],
    ) -> None:
        self._updating = True
        try:
            if profile is None or np.asarray(profile).size == 0:
                self.clear_column_profile()
                return
            self._profile = np.asarray(profile, dtype=float)
            self._roi_x = float(roi_x)
            self._roi_w = float(roi_width)
            x = self._roi_x + np.arange(self._profile.size, dtype=float)
            self._col_curve.setData(x, self._profile)
            self._rebuild_lines(separators)
            self._rebuild_regions()
            self._col_plot.getPlotItem().enableAutoRange()
        finally:
            self._updating = False

    def clear_column_profile(self) -> None:
        self._updating = True
        try:
            self._profile = None
            self._col_curve.setData([], [])
            self._rebuild_lines([])
            self._rebuild_regions()
        finally:
            self._updating = False

    def set_separators(self, separators: list[float]) -> None:
        """程序化同步分隔线位置（不发信号）。"""
        self._updating = True
        try:
            self._rebuild_lines(separators)
            self._rebuild_regions()
        finally:
            self._updating = False

    def separators(self) -> list[float]:
        return sorted(float(line.value()) for line in self._sep_lines)

    def _rebuild_lines(self, separators: list[float]) -> None:
        for line in self._sep_lines:
            self._col_plot.removeItem(line)
        self._sep_lines.clear()
        if self._profile is None:
            return
        lo = self._roi_x + 1.0
        hi = self._roi_x + self._roi_w - 2.0
        for sep in sorted(float(s) for s in separators):
            line = pg.InfiniteLine(
                pos=min(max(sep, lo), hi),
                angle=90,
                movable=True,
                pen=pg.mkPen(SEPARATOR, width=2),
                hoverPen=pg.mkPen("#ffd54f", width=3),
                bounds=[lo, hi],
            )
            line.setZValue(10)
            line.sigPositionChanged.connect(self._on_line_moved)
            line.sigPositionChangeFinished.connect(self._on_line_finished)
            self._col_plot.addItem(line)
            self._sep_lines.append(line)

    def _rebuild_regions(self) -> None:
        for region in self._regions:
            self._col_plot.removeItem(region)
        self._regions.clear()
        for label in self._labels:
            self._col_plot.removeItem(label)
        self._labels.clear()
        if self._profile is None or self._roi_w <= 0:
            return
        positions = sorted(float(line.value()) for line in self._sep_lines)
        bounds = [self._roi_x, *positions, self._roi_x + self._roi_w]
        y_top = float(self._profile.max()) if self._profile.size else 1.0
        for index in range(len(bounds) - 1):
            a, b = bounds[index], bounds[index + 1]
            region = pg.LinearRegionItem(
                values=(a, b),
                movable=False,
                brush=pg.mkBrush(*REGION_BRUSHES[index % 2]),
                pen=pg.mkPen(0, 0, 0, 0),
            )
            region.setZValue(-10)
            for region_line in region.lines:
                region_line.setMovable(False)
            self._col_plot.addItem(region)
            self._regions.append(region)
            label = pg.TextItem(text=f"L{index + 1}", color=LANE_LABEL, anchor=(0.5, 0.0))
            label.setPos((a + b) / 2.0, y_top)
            self._col_plot.addItem(label)
            self._labels.append(label)

    def _on_line_moved(self, line: pg.InfiniteLine) -> None:
        if self._updating:
            return
        try:
            index = self._sep_lines.index(line)
        except ValueError:
            return
        pos = float(line.value())
        lo = self._roi_x + 1.0
        hi = self._roi_x + self._roi_w - 2.0
        if index > 0:
            lo = max(lo, float(self._sep_lines[index - 1].value()) + SEP_MARGIN)
        if index < len(self._sep_lines) - 1:
            hi = min(hi, float(self._sep_lines[index + 1].value()) - SEP_MARGIN)
        clamped = min(max(pos, lo), hi)
        if abs(clamped - pos) > 1e-9:
            self._updating = True
            try:
                line.setValue(clamped)
            finally:
                self._updating = False
        self._update_region_bounds()
        self.separatorsChanged.emit(self.separators())

    def _on_line_finished(self, _line: pg.InfiniteLine) -> None:
        if not self._updating:
            self.editCommitted.emit()

    def _update_region_bounds(self) -> None:
        positions = sorted(float(line.value()) for line in self._sep_lines)
        bounds = [self._roi_x, *positions, self._roi_x + self._roi_w]
        if len(self._regions) != len(bounds) - 1:
            self._rebuild_regions()
            return
        for index, region in enumerate(self._regions):
            region.setRegion((bounds[index], bounds[index + 1]))
            if index < len(self._labels):
                pos = self._labels[index].pos()
                self._labels[index].setPos(
                    (bounds[index] + bounds[index + 1]) / 2.0, pos.y()
                )
        if self._labels:
            last = len(self._regions) - 1
            if last < len(self._labels):
                pos = self._labels[last].pos()
                self._labels[last].setPos(
                    (bounds[last] + bounds[last + 1]) / 2.0, pos.y()
                )

    # ------------------------------------------------------------------ 右图：泳道纵向剖面

    def set_lane_profiles(self, lanes: list, measurements: list | None = None) -> None:
        """lanes 为 :class:`wb_analyzer.quant.LaneProfile` 列表。

        measurements 是当前仍然存在的条带。给了它就按它画积分填充区、按它算
        标题里的合计——删掉一个条带后 lanes 不会重算，照搬 lane.segments 与
        lane.total_density 会把删掉那条继续画上、继续算进去。
        """
        self._lanes = list(lanes or [])
        self._measurements = list(measurements or [])
        if not self._lanes:
            self._current_lane = None
            self._lane_plot.clear()
            self._lane_plot.getPlotItem().setTitle(
                LANE_TITLE_EMPTY, color=TITLE_COLOR, size="10pt"
            )
            return
        wanted = self._current_lane
        available = [int(lane.lane) for lane in self._lanes]
        self.show_lane(wanted if wanted in available else available[0])

    def _lane_bands(self, target) -> list:
        """本泳道当前仍存在的条带。"""
        return [m for m in self._measurements if int(m.lane) == int(target.lane)]

    def _fill_segments(self, target, profile_size: int, y0: float) -> list:
        """积分填充区间：有测量结果就按结果的 y 范围，否则退回 lane.segments。"""
        if self._measurements:
            spans = []
            for m in self._lane_bands(target):
                start = int(round(float(m.y) - y0))
                stop = start + int(round(float(m.height)))
                spans.append((max(0, start), min(profile_size, stop)))
            return spans
        return [
            (int(max(0, start)), int(min(profile_size, stop)))
            for start, stop in target.segments
        ]

    def _lane_total(self, target) -> float:
        """标题里的合计：**现存**条带的积分之和，和填充区、结果表同一口径。

        lane.total_density 是整条泳道扣完基线的总积分，删掉误检条带后它不会
        变小，直接显示会比结果表里剩下的几行加起来大一截。
        """
        if self._measurements:
            return float(sum(float(m.integrated_density) for m in self._lane_bands(target)))
        return float(sum(target.band_densities))

    def show_lane(self, lane_number: int) -> None:
        target = None
        for lane in self._lanes:
            if int(lane.lane) == int(lane_number):
                target = lane
                break
        if target is None:
            return
        self._current_lane = int(lane_number)
        plot_item = self._lane_plot.getPlotItem()
        self._lane_plot.clear()
        profile = np.asarray(target.profile, dtype=float)
        # 画的是**实际的积分下界**：橡皮筋基线再抬高一个噪声地板。定量扣掉了
        # 这一层，只画裸基线的话填充面积会比标题和结果表里的数大一截。
        baseline = np.asarray(target.baseline, dtype=float) + float(target.noise_floor)
        if profile.size == 0:
            return
        y0 = float(target.y0)
        x = y0 + np.arange(profile.size, dtype=float)
        self._lane_plot.plot(x, profile, pen=pg.mkPen(ACCENT, width=2))
        self._lane_plot.plot(
            x,
            baseline,
            pen=pg.mkPen(BASELINE, width=1.5, style=Qt.PenStyle.DashLine),
        )
        for start, stop in self._fill_segments(target, int(profile.size), y0):
            if stop - start < 2:
                continue
            xs = x[start:stop]
            upper = self._lane_plot.plot(xs, profile[start:stop], pen=None)
            lower = self._lane_plot.plot(xs, baseline[start:stop], pen=None)
            fill = pg.FillBetweenItem(upper, lower, brush=pg.mkBrush(*FILL_BRUSH))
            self._lane_plot.addItem(fill)
        plot_item.setTitle(
            f"L{self._current_lane} 条带积分合计：{self._lane_total(target):,.1f}",
            color=TITLE_COLOR,
            size="10pt",
        )
        plot_item.enableAutoRange()

    def clear(self) -> None:
        self.clear_column_profile()
        self.set_lane_profiles([])

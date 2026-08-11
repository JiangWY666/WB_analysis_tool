"""强度剖面面板：左图 ROI 横向剖面（可拖分隔线），右图选中泳道纵向剖面。"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QSplitter, QWidget

pg.setConfigOptions(antialias=True)

ACCENT = "#4fc3f7"
SEPARATOR = "#ffb300"
BASELINE = "#ffb74d"
PLOT_BG = "#141518"
AXIS_PEN = "#3a3d43"
AXIS_TEXT = "#9aa0a6"
TITLE_COLOR = "#dfe3e8"
LANE_LABEL = "#bfe9ff"
REGION_BRUSHES = ((79, 195, 247, 26), (255, 255, 255, 8))
SEP_MARGIN = 2.0


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
        self._current_lane: int | None = None

        self._col_plot = pg.PlotWidget()
        _style_plot(self._col_plot)
        self._col_plot.getPlotItem().setTitle(
            "ROI 横向剖面（列均值信号，拖橙线调分隔）", color=TITLE_COLOR, size="10pt"
        )
        self._col_plot.setLabel("bottom", "X（像素）")
        self._col_plot.setLabel("left", "信号强度")
        self._col_curve = self._col_plot.plot(pen=pg.mkPen(ACCENT, width=2))

        self._lane_plot = pg.PlotWidget()
        _style_plot(self._lane_plot)
        self._lane_plot.getPlotItem().setTitle(
            "泳道纵向剖面（先 ROI 定量）", color=TITLE_COLOR, size="10pt"
        )
        self._lane_plot.setLabel("bottom", "Y（像素）")
        self._lane_plot.setLabel("left", "信号强度")

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._col_plot)
        splitter.addWidget(self._lane_plot)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(splitter)
        self.setMinimumHeight(180)

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

    def set_lane_profiles(self, lanes: list) -> None:
        """lanes 为 quant.LaneQuant（或契约 LaneProfile）列表。"""
        self._lanes = list(lanes or [])
        if not self._lanes:
            self._current_lane = None
            self._lane_plot.clear()
            self._lane_plot.getPlotItem().setTitle(
                "泳道纵向剖面（先 ROI 定量）", color=TITLE_COLOR, size="10pt"
            )
            return
        wanted = self._current_lane
        available = [int(lane.lane) for lane in self._lanes]
        self.show_lane(wanted if wanted in available else available[0])

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
        baseline = np.asarray(target.baseline, dtype=float)
        if profile.size == 0:
            return
        y0 = float(getattr(target, "y0", 0))
        x = y0 + np.arange(profile.size, dtype=float)
        self._lane_plot.plot(x, profile, pen=pg.mkPen(ACCENT, width=2))
        self._lane_plot.plot(
            x,
            baseline,
            pen=pg.mkPen(BASELINE, width=1.5, style=Qt.PenStyle.DashLine),
        )
        for start, stop in getattr(target, "segments", []) or []:
            start = int(max(0, start))
            stop = int(min(profile.size, stop))
            if stop - start < 2:
                continue
            xs = x[start:stop]
            upper = self._lane_plot.plot(xs, profile[start:stop], pen=None)
            lower = self._lane_plot.plot(xs, baseline[start:stop], pen=None)
            fill = pg.FillBetweenItem(upper, lower, brush=pg.mkBrush(79, 195, 247, 70))
            self._lane_plot.addItem(fill)
        total = float(getattr(target, "total_density", 0.0))
        plot_item.setTitle(
            f"L{self._current_lane} 积分值：{total:,.1f}", color=TITLE_COLOR, size="10pt"
        )
        plot_item.enableAutoRange()

    def clear(self) -> None:
        self.clear_column_profile()
        self.set_lane_profiles([])

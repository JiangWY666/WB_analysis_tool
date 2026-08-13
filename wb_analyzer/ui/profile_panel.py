"""强度剖面面板：左图 ROI 横向剖面（可拖分隔线），右图选中泳道纵向剖面。"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .canvas import HIT_PX as CANVAS_HIT_PX
from .canvas import MIN_ROI_PX as MIN_SEP_GAP

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
# 右键命中分隔线的判定半径（屏幕像素）。画布用的是 HIT_PX - 1，这里照抄，
# 让"右键线上删除"在两处需要的手感完全一样。
HIT_PX = CANVAS_HIT_PX - 1.0

COLUMN_TITLE = "ROI 横向剖面（双击加分隔线 · 拖橙线微调 · 右键删除）"
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


SWATCH_W = 22
SWATCH_H = 12


def _swatch_pixmap(color: str, fill_rgba=None, dashed: bool = False) -> QPixmap:
    """图例小色块：一段线，需要时在线下垫一层填充色。"""
    pixmap = QPixmap(SWATCH_W, SWATCH_H)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    mid = SWATCH_H // 2
    if fill_rgba is not None:
        # 图内那么淡的填充缩到 22px 宽等于看不见，提一点不透明度
        painter.fillRect(
            1, mid, SWATCH_W - 2, SWATCH_H - mid - 1, QColor(*fill_rgba[:3], 150)
        )
    pen = QPen(QColor(color))
    pen.setWidth(2)
    if dashed:
        pen.setStyle(Qt.PenStyle.DashLine)
    painter.setPen(pen)
    painter.drawLine(1, mid, SWATCH_W - 2, mid)
    painter.end()
    return pixmap


class LegendStrip(QWidget):
    """画在坐标区**外**的图例条。

    pyqtgraph 的 addLegend 固定压在绘图区右上角，泳道一多，最右边几个 L# 分区
    标签就被它盖死（9 泳道时 L8/L9 已经完全读不出来），泳道越多压得越狠。信息
    不能丢，那就把它整个挪出坐标区——横排一行小色块，无论多少泳道都不遮挡。

    顺带绕开了 LegendItem 的老坑：它画样例时直接读 item.opts，把 InfiniteLine /
    FillBetweenItem 塞进去会在 paint 里抛 AttributeError，连带整张图画不出来。
    这里的色块是自己画的 QPixmap，和绘图项没有任何耦合。
    """

    def __init__(self, entries, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(4)
        for color, fill_rgba, dashed, text in entries:
            swatch = QLabel()
            swatch.setPixmap(_swatch_pixmap(color, fill_rgba, dashed))
            swatch.setFixedSize(SWATCH_W, SWATCH_H)
            label = QLabel(text)
            label.setObjectName("dimHint")
            font = label.font()
            font.setPointSizeF(max(7.5, font.pointSizeF() - 1.0))
            label.setFont(font)
            layout.addWidget(swatch)
            layout.addWidget(label)
            layout.addSpacing(6)
        layout.addStretch(1)


def _with_legend(plot: pg.PlotWidget, entries) -> QWidget:
    """把绘图控件和它的图例条竖着摞成一个整体，供 splitter 使用。"""
    box = QWidget()
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)
    layout.addWidget(plot, 1)
    layout.addWidget(LegendStrip(entries))
    return box


def _confirm_delete_separator(parent: QWidget, screen_pos) -> bool:
    """右键删除前的确认小菜单，和画布上右键分隔线弹出的是同一个。"""
    menu = QMenu(parent)
    action = menu.addAction("删除此分隔线")
    return menu.exec(screen_pos) is action


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
        self._col_plot.setToolTip(
            "双击空白处添加分隔线；右键分隔线删除；拖动橙线微调。\n"
            "位置按图像像素对齐，和画布上的分隔线是同一条。"
        )
        col_entries = [
            (ACCENT, None, False, "列平均信号"),
            (SEPARATOR, None, False, "泳道分隔线（双击加 / 拖动调 / 右键删）"),
            (ACCENT_DIM, REGION_BRUSHES[0], False, "泳道分区（深浅相间）"),
        ]

        self._lane_plot = pg.PlotWidget()
        _style_plot(self._lane_plot)
        self._lane_plot.getPlotItem().setTitle(
            LANE_TITLE_EMPTY, color=TITLE_COLOR, size="10pt"
        )
        self._lane_plot.setLabel("bottom", "Y（像素）")
        self._lane_plot.setLabel("left", LANE_Y_LABEL)
        lane_entries = [
            (ACCENT, None, False, "泳道信号"),
            (BASELINE, None, True, "扣除用基线"),
            (ACCENT_DIM, FILL_BRUSH, False, "积分区间（条带）"),
        ]

        # 左图的双击让给"添加分隔线"了，复位只剩这个按钮（RectMode 下误框选
        # 之后总得有条退路，角落那个 A 按钮太小）
        reset_button = QPushButton("复位视图")
        reset_button.setToolTip("把两张剖面图恢复为自动缩放（框选放大后用它退出）")
        reset_button.clicked.connect(self.reset_views)
        hint = QLabel("左图：双击加分隔线 · 拖动微调 · 右键线上删除 · 框选放大后按右侧按钮复位")
        hint.setObjectName("dimHint")
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(8)
        bar.addWidget(hint)
        bar.addStretch(1)
        bar.addWidget(reset_button)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(_with_legend(self._col_plot, col_entries))
        splitter.addWidget(_with_legend(self._lane_plot, lane_entries))
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(splitter, 1)
        layout.addLayout(bar)
        self.setMinimumHeight(200)
        # 装在 scene 上而不是 ViewBox 上：分隔线（InfiniteLine）压在 ViewBox
        # 上面，装 ViewBox 会漏掉正好点在线上的那些事件，而右键删除偏偏只发生
        # 在线上。scene 是视图派发给任何图元之前的必经之路。
        self._col_scene = self._col_plot.scene()
        self._lane_scene = self._lane_plot.scene()
        self._col_scene.installEventFilter(self)
        self._lane_scene.installEventFilter(self)

    def eventFilter(self, obj, event):  # noqa: N802 - Qt API
        """左图：双击加分隔线、右键删分隔线；右图：双击 View All。

        分隔线的增 / 删 / 拖在这里**不受画布工具模式约束**。画布上双击加分隔线
        之所以只在"调整"模式生效，是因为框选 ROI / 补画条带都占着鼠标拖拽，不
        分模式就打架；剖面图里没有任何别的手势要抢，而且分隔线一直都是随时可
        拖的——只把"加 / 删"卡在调整模式，同一张图里三个动作两套规矩反而更怪。
        """
        etype = event.type()
        if obj is self._col_scene:
            if (
                etype == QEvent.Type.GraphicsSceneMouseDoubleClick
                and event.button() == Qt.MouseButton.LeftButton
                and self._add_separator_at(event.scenePos())
            ):
                event.accept()
                return True
            if (
                etype == QEvent.Type.GraphicsSceneMousePress
                and event.button() == Qt.MouseButton.RightButton
                and self._delete_separator_at(event.scenePos(), event.screenPos())
            ):
                event.accept()
                return True
        elif (
            obj is self._lane_scene
            and etype == QEvent.Type.GraphicsSceneMouseDoubleClick
        ):
            self.reset_views()
            return True
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------ 左图：增删分隔线

    def _col_view_box(self) -> pg.ViewBox:
        return self._col_plot.getPlotItem().getViewBox()

    def image_x_at(self, scene_pos) -> float | None:
        """场景坐标 → **图像像素 x**；落在绘图区外返回 None。

        左图横轴画的就是绝对图像列号（``set_column_profile`` 里 x 从 roi_x 起
        步、步长 1），所以 ViewBox 的数据坐标即图像坐标，不需要再换算。这条
        必须准到亚像素：泳道边界错一列，积分就跟着变。
        """
        view_box = self._col_view_box()
        if not view_box.sceneBoundingRect().contains(scene_pos):
            return None
        return float(view_box.mapSceneToView(QPointF(scene_pos)).x())

    def _x_tolerance(self, pixels: float) -> float:
        """把屏幕像素容差折算成横轴数据单位：放大缩小后手感都一样。"""
        try:
            scale = float(self._col_view_box().viewPixelSize()[0])
        except TypeError:  # 视图还没布局，pyqtgraph 的像素向量是 None
            scale = 0.0
        return pixels * scale if scale > 0 else 1.0

    def separator_at(self, scene_pos) -> int | None:
        """场景坐标附近（±HIT_PX 屏幕像素）最近那条分隔线的序号。"""
        x = self.image_x_at(scene_pos)
        if x is None or not self._sep_lines:
            return None
        best: int | None = None
        best_gap = self._x_tolerance(HIT_PX)
        for index, value in enumerate(self.separators()):
            gap = abs(value - x)
            if gap <= best_gap:
                best, best_gap = index, gap
        return best

    def add_separator(self, x: float) -> bool:
        """在图像像素 x 处加一条分隔线，判定与画布双击添加逐条对齐。"""
        if self._profile is None or self._roi_w <= 0:
            return False
        lo = self._roi_x + MIN_SEP_GAP
        hi = self._roi_x + self._roi_w - MIN_SEP_GAP
        if hi <= lo:
            return False
        pos = min(max(float(x), lo), hi)
        current = self.separators()
        if any(abs(pos - value) < MIN_SEP_GAP for value in current):
            return False
        self._commit_separators([*current, pos])
        return True

    def remove_separator(self, index: int) -> bool:
        current = self.separators()
        if not 0 <= index < len(current):
            return False
        del current[index]
        self._commit_separators(current)
        return True

    def _add_separator_at(self, scene_pos) -> bool:
        # 双击正落在已有分隔线上就什么都不做——画布上也是这么让路的
        if self.separator_at(scene_pos) is not None:
            return False
        x = self.image_x_at(scene_pos)
        return False if x is None else self.add_separator(x)

    def _delete_separator_at(self, scene_pos, screen_pos) -> bool:
        index = self.separator_at(scene_pos)
        if index is None:
            return False
        if _confirm_delete_separator(self, screen_pos):
            self.remove_separator(index)
        # 命中了就吃掉这次右键：别再让 ViewBox 拿去做右键缩放
        return True

    def _commit_separators(self, values: list[float]) -> None:
        """增删各算一次完整编辑：先同步位置，再入撤销栈并触发重新定量。"""
        self._updating = True
        try:
            self._rebuild_lines(values)
            self._rebuild_regions()
        finally:
            self._updating = False
        self.separatorsChanged.emit(self.separators())
        self.editCommitted.emit()

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

    def _drag_bounds(self) -> tuple[float, float]:
        return self._roi_x + SEP_MARGIN, self._roi_x + self._roi_w - SEP_MARGIN

    def _rebuild_lines(self, separators: list[float]) -> None:
        for line in self._sep_lines:
            self._col_plot.removeItem(line)
        self._sep_lines.clear()
        if self._profile is None:
            return
        # 拖动可达范围与画布 _drag_separator 用同一个内缩量，否则同一条线在两处
        # 能拖到的最边上位置差一像素
        lo, hi = self._drag_bounds()
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
        lo, hi = self._drag_bounds()
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

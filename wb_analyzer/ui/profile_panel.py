"""强度剖面面板：左图 ROI 横向剖面（可拖分隔线），右图选中泳道纵向剖面。"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QPointF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
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

COLUMN_TITLE = "ROI 横向剖面"
LANE_TITLE_EMPTY = "泳道纵向剖面"
# 两张图纵轴的量纲不同：左图是 ROI 每一列的平均信号，右图是整条泳道宽度上
# 求和后的信号。轴上只留短名把两者区分开，完整口径写在图的 tooltip 里——竖排
# 的轴标签每多一个字就多占一列宽度，解释性文字摆在那里最不划算。
COLUMN_Y_LABEL = "列平均信号"
LANE_Y_LABEL = "行累加信号"

COLUMN_TIP = (
    "双击空白处添加分隔线；右键分隔线删除；拖动橙线微调。\n"
    "位置按图像像素对齐，和画布上的分隔线是同一条。\n"
    "纵轴：ROI 每一列的平均灰度。"
)
LANE_TIP = (
    "点选条带后显示该泳道的纵向剖面，需要先做 ROI 定量。\n"
    "纵轴：整条泳道宽度上求和后的灰度，与左图不是同一口径。"
)


TITLE_PT = 10.0
LEGEND_PT_DROP = 1.0  # 图例比正文小一号
LEGEND_PT_FLOOR = 7.5
# 相邻泳道标签之间至少留出的空白（设备像素）。贴着算"不重叠"在视觉上仍然是
# 一团，留一点缝才读得出这是两个数。
LANE_LABEL_GAP = 5.0
# 摆不开时依次尝试的间隔："每 2 道标一个"、"每 3 道标一个"……都摆不开就整批
# 隐藏。间隔必须是规则的：用户能预期"标出来的是第 1、3、5 道"，而"能塞下的那
# 几个"没人能预期。
LANE_LABEL_STRIDES = (2, 3, 4, 6, 8, 12)
# 面板最矮多高。字号一放大就得跟着长高：坐标轴刻度是按像素高度排的，绘图区不
# 变而字变大，几个刻度会直接叠在一起（实测 14 pt 时 12000/10000/8000 糊成一团）。
MIN_PANEL_H = 200


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
        self._labels: list[QLabel] = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(4)
        for color, fill_rgba, dashed, text in entries:
            swatch = QLabel()
            swatch.setPixmap(_swatch_pixmap(color, fill_rgba, dashed))
            swatch.setFixedSize(SWATCH_W, SWATCH_H)
            label = QLabel(text)
            label.setObjectName("dimHint")
            self._labels.append(label)
            layout.addWidget(swatch)
            layout.addWidget(label)
            layout.addSpacing(6)
        layout.addStretch(1)
        self.refresh_font()

    def refresh_font(self) -> None:
        """跟着父级字体走。显式 setFont 过的控件收不到继承的字体变更。"""
        base = self.font().pointSizeF()
        for label in self._labels:
            font = label.font()
            font.setPointSizeF(max(LEGEND_PT_FLOOR, base - LEGEND_PT_DROP))
            label.setFont(font)


def _with_legend(plot: pg.PlotWidget, entries) -> tuple[QWidget, LegendStrip]:
    """把绘图控件和它的图例条竖着摞成一个整体，供 splitter 使用。"""
    box = QWidget()
    strip = LegendStrip(entries)
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)
    layout.addWidget(plot, 1)
    layout.addWidget(strip)
    return box, strip


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
        # 泳道标签的横坐标（数据单位）与当前降级档位，见 _relayout_lane_labels
        self._label_centers: list[float] = []
        self._labels_packed = True
        self._labels_compact = False
        self._label_stride = 1
        self._relaying_labels = False
        self._lanes: list = []
        self._measurements: list = []
        self._current_lane: int | None = None
        # 界面设置里的字号倍数。1.0 时下面所有字号与改动前逐字节相同。
        self._font_scale = 1.0

        self._col_plot = pg.PlotWidget()
        _style_plot(self._col_plot)
        self._col_plot.getPlotItem().setTitle(
            COLUMN_TITLE, color=TITLE_COLOR, size=self._title_size()
        )
        self._col_plot.setLabel("bottom", "X（像素）")
        self._col_plot.setLabel("left", COLUMN_Y_LABEL)
        self._col_curve = self._col_plot.plot(
            pen=pg.mkPen(ACCENT, width=2), name="列平均信号"
        )
        self._col_plot.setToolTip(COLUMN_TIP)
        # 图例只说"这个颜色是什么"。怎么加 / 怎么删挂在图的 tooltip 里了，图例里
        # 再写一遍就是同一句话在同一张图上出现两次。
        col_entries = [
            (ACCENT, None, False, "列平均信号"),
            (SEPARATOR, None, False, "泳道分隔线"),
            (ACCENT_DIM, REGION_BRUSHES[0], False, "泳道分区"),
        ]

        self._lane_plot = pg.PlotWidget()
        _style_plot(self._lane_plot)
        self._lane_plot.getPlotItem().setTitle(
            LANE_TITLE_EMPTY, color=TITLE_COLOR, size=self._title_size()
        )
        self._lane_plot.setLabel("bottom", "Y（像素）")
        self._lane_plot.setLabel("left", LANE_Y_LABEL)
        self._lane_plot.setToolTip(LANE_TIP)
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
        # 这里以前还有一行"双击加分隔线 · 拖动微调 · 右键删除"的说明，占掉一整行
        # 高度。操作说明统一收在左图的 tooltip 里，面板上不再常驻。
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(8)
        bar.addStretch(1)
        bar.addWidget(reset_button)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        col_box, col_legend = _with_legend(self._col_plot, col_entries)
        lane_box, lane_legend = _with_legend(self._lane_plot, lane_entries)
        self._legends = (col_legend, lane_legend)
        splitter.addWidget(col_box)
        splitter.addWidget(lane_box)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(splitter, 1)
        layout.addLayout(bar)
        self.setMinimumHeight(MIN_PANEL_H)
        # 装在 scene 上而不是 ViewBox 上：分隔线（InfiniteLine）压在 ViewBox
        # 上面，装 ViewBox 会漏掉正好点在线上的那些事件，而右键删除偏偏只发生
        # 在线上。scene 是视图派发给任何图元之前的必经之路。
        self._col_scene = self._col_plot.scene()
        self._lane_scene = self._lane_plot.scene()
        self._col_scene.installEventFilter(self)
        self._lane_scene.installEventFilter(self)
        # 泳道标签的屏幕尺寸恒定，能给它的像素却随控件宽度与缩放变化——这两件事
        # 一变就得重新评一次降级档位，否则窄图 / 放大后标签直接叠成一团。
        view_box = self._col_view_box()
        view_box.sigResized.connect(self._on_col_view_changed)
        view_box.sigRangeChanged.connect(self._on_col_view_changed)

    # ------------------------------------------------------------------ 界面设置

    def _title_size(self) -> str:
        return f"{TITLE_PT * self._font_scale:g}pt"

    def set_font_scale(self, scale: float) -> None:
        """跟随界面字体大小缩放图内文字。

        pyqtgraph 的文字是另一套：坐标轴刻度用画笔当前字体（跟着控件走），但
        标题与轴标签是内嵌 CSS 字号、图例条的小字是显式 setFont 过的，这三处
        都收不到 Qt 的字体继承。所以要在这里逐个推一遍，否则调大字号后图里只
        有刻度变大、标题和轴标签纹丝不动。
        """
        scale = max(0.5, float(scale))
        if abs(scale - self._font_scale) < 1e-6:
            return
        self._font_scale = scale
        self.setMinimumHeight(int(round(MIN_PANEL_H * scale)))
        tick_font = QFont(self.font())
        tick_font.setPointSizeF(max(6.0, self.font().pointSizeF()))
        label_style = {"color": AXIS_TEXT, "font-size": f"{self.font().pointSizeF():g}pt"}
        for widget, bottom, left in (
            (self._col_plot, "X（像素）", COLUMN_Y_LABEL),
            (self._lane_plot, "Y（像素）", LANE_Y_LABEL),
        ):
            plot_item = widget.getPlotItem()
            for name in ("left", "bottom"):
                plot_item.getAxis(name).setStyle(tickFont=tick_font)
            widget.setLabel("bottom", bottom, **label_style)
            widget.setLabel("left", left, **label_style)
        self._col_plot.getPlotItem().setTitle(
            COLUMN_TITLE, color=TITLE_COLOR, size=self._title_size()
        )
        if self._current_lane is None:
            self._lane_plot.getPlotItem().setTitle(
                LANE_TITLE_EMPTY, color=TITLE_COLOR, size=self._title_size()
            )
        else:
            self.show_lane(self._current_lane)
        for legend in self._legends:
            legend.refresh_font()
        # 泳道分区标签是 TextItem，字体同样不继承
        self._rebuild_regions()

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
        self._label_centers.clear()
        self._labels_packed = True
        self._labels_compact = False
        self._label_stride = 1
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
            self._apply_label_font(label)
            label.setPos((a + b) / 2.0, y_top)
            # ignoreBounds：标签的宽度不许参与自动量程。参与的话"改文案 → 量程变
            # → 每像素的数据单位变 → 再评一次降级"会绕成一个环。
            self._col_plot.addItem(label, ignoreBounds=True)
            self._labels.append(label)
            self._label_centers.append((a + b) / 2.0)
        self._relayout_lane_labels()

    def _apply_label_font(self, label) -> None:
        """L# 分区标签的字号跟着界面走（TextItem 不继承控件字体）。

        字号无论有没有缩放都显式推一遍：避碰判据读的是标签**真实的**外接矩形，
        而"没设过字体"时那个矩形取决于 pyqtgraph 内部默认，量出来的宽度和真正
        画出来的可能不是一回事。
        """
        try:
            font = QFont(self.font())
            # self.font() 已经被界面设置按倍数改过了，这里不能再乘一次
            font.setPointSizeF(max(6.0, self.font().pointSizeF()))
            label.textItem.setFont(font)
            label.updateTextPos()
        except AttributeError:  # pyqtgraph 换了内部结构：标签字号不值得连累绘图
            pass

    # ------------------------------------------------------------------ 泳道标签避碰

    def _on_col_view_changed(self, *_args) -> None:
        if self._relaying_labels:
            return
        self._relayout_lane_labels()

    def _view_px_per_unit(self) -> float:
        """横轴一个数据单位占几个设备像素；视图还没布局时返回 0。"""
        try:
            size = float(self._col_view_box().viewPixelSize()[0])
        except (TypeError, IndexError, ValueError):
            return 0.0
        return 1.0 / size if size > 0 else 0.0

    def _set_lane_label_texts(self, compact: bool) -> None:
        for index, label in enumerate(self._labels):
            label.setText(f"{index + 1}" if compact else f"L{index + 1}")
            # setText 会重建内部文档，字体得重新推一遍
            self._apply_label_font(label)

    def _lane_label_widths(self) -> list[float]:
        """当前文案下每个标签的实际宽度（设备像素）。

        读的是 TextItem 自己的外接矩形，而不是拿 QFontMetrics 估一个——判据必须
        和最终画出来的那个矩形逐像素同一个来源，否则"算得开、画出来却压住"。
        TextItem 带 ItemIgnoresTransformations，它的 boundingRect 就是设备像素。
        """
        return [float(label.boundingRect().width()) for label in self._labels]

    def _lane_labels_fit(self, widths: list[float], stride: int) -> bool:
        px = self._view_px_per_unit()
        if px <= 0:  # 还没布局，别急着降级——布局好了 sigResized 会再来一次
            return True
        shown = list(range(0, len(self._label_centers), stride))
        for first, second in zip(shown, shown[1:]):
            gap = (self._label_centers[second] - self._label_centers[first]) * px
            need = (widths[first] + widths[second]) / 2.0 + LANE_LABEL_GAP
            if gap < need:
                return False
        return True

    def _apply_lane_labels(self, compact: bool, stride: int) -> None:
        """stride <= 0 表示整批隐藏。"""
        self._set_lane_label_texts(compact)
        for index, label in enumerate(self._labels):
            label.setVisible(stride > 0 and index % stride == 0)
        self._labels_compact = compact
        self._label_stride = max(0, int(stride))
        self._labels_packed = stride == 1

    def _relayout_lane_labels(self) -> None:
        """按能给的像素宽逐级降级，绝不留"糊成一团"的中间态。

        和画布上的条带标签同一套契约：

        1. 完整文案 ``L3``，每道都标；
        2. 摆不开就去掉 ``L`` 只留数字 ``3``；
        3. 还摆不开就按规则间隔标（每 2 道一个、每 3 道一个……）；
        4. 最稀的间隔也摆不开时整批隐藏。

        只显示"恰好塞得下的那几个"是不做的：泳道号一旦跳着漏，用户读错的代价
        比读不到更大。
        """
        if not self._labels:
            self._labels_packed = True
            self._labels_compact = False
            self._label_stride = 1
            return
        self._relaying_labels = True
        try:
            for compact in (False, True):
                self._set_lane_label_texts(compact)
                if self._lane_labels_fit(self._lane_label_widths(), 1):
                    self._apply_lane_labels(compact, 1)
                    return
            # 到这里已经是纯数字文案，宽度不会再变，量一次给所有间隔用
            widths = self._lane_label_widths()
            for stride in LANE_LABEL_STRIDES:
                if stride >= len(self._labels):
                    break
                if self._lane_labels_fit(widths, stride):
                    self._apply_lane_labels(True, stride)
                    return
            self._apply_lane_labels(True, 0)
        finally:
            self._relaying_labels = False

    def lane_labels_packed(self) -> bool:
        """当前是不是每道都标、且互不重叠（没有任何降级）。"""
        return bool(self._labels_packed)

    def lane_labels_compact(self) -> bool:
        """显示的是不是紧凑文案（只有数字、没有 ``L``）。"""
        return bool(self._labels_compact)

    def lane_label_stride(self) -> int:
        """每几道标一个；0 表示整批隐藏。"""
        return int(self._label_stride)

    def lane_label_rects(self) -> list:
        """可见泳道标签在场景里的矩形，供重叠检查使用。"""
        return [
            label.mapRectToScene(label.boundingRect())
            for label in self._labels
            if label.isVisible()
        ]

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
                center = (bounds[index] + bounds[index + 1]) / 2.0
                self._labels[index].setPos(center, pos.y())
                self._label_centers[index] = center
        # 拖分隔线会把某一道挤窄：间距变了，降级档位得重新评一次
        self._relayout_lane_labels()

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
                LANE_TITLE_EMPTY, color=TITLE_COLOR, size=self._title_size()
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

    def current_lane(self) -> int | None:
        return self._current_lane

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
            size=self._title_size(),
        )
        plot_item.enableAutoRange()

    def clear(self) -> None:
        self.clear_column_profile()
        self.set_lane_profiles([])
        # 量程也要退回去：不复位的话空面板上还留着上一张膜的横轴刻度（"50…350"），
        # 看上去像是图还在、只是曲线没画出来。
        self.reset_views()

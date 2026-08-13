"""BlotCanvas：QGraphicsView 图像画布。

- 滚轮缩放 / 中键或空格拖拽平移
- ROI 框选（8 手柄缩放、整体拖动）、泳道分隔线拖动 / 添加 / 删除
- 标尺测角、补画条带、检测框结果叠加、中心参考虚线
坐标约定：scene 坐标 == 图像像素坐标（pixmap 置于原点）。
"""

from __future__ import annotations

import enum
import time

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QTransform,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QMenu,
    QStyle,
)

from . import errorlog, theme

# 配色统一取自 theme，本模块只做 QColor 转换
ROI_COLOR = QColor(theme.ROI_LINE)
ROI_FILL = QColor(*theme.ROI_FILL_RGBA)
HANDLE_BORDER = QColor(theme.HANDLE_BORDER)
SEPARATOR_COLOR = QColor(theme.SEPARATOR)
SEPARATOR_FOCUS_COLOR = QColor(theme.SEPARATOR_FOCUS)
RULER_COLOR = QColor(theme.RULER)
BAND_COLOR = QColor(theme.BAND_DRAW)
RESULT_COLOR = QColor(theme.RESULT_BOX)
RESULT_SAT_COLOR = QColor(theme.RESULT_BOX_SAT)
RESULT_WEAK_COLOR = QColor(theme.RESULT_BOX_WEAK)
BAND_SELECT_COLOR = QColor(theme.SELECT)
BAND_SELECT_FILL = QColor(*theme.SELECT_FILL_RGBA)
GUIDE_COLOR = QColor(*theme.GUIDE_RGBA)
CANVAS_BG_COLOR = QColor(theme.CANVAS_BG)
EMPTY_HINT_COLOR = QColor(theme.TEXT_FAINT)
EMPTY_HINT = "打开或拖入 WB 图片（PNG / JPG / BMP / TIF）"
LABEL_PLATE_COLOR = QColor(18, 20, 23, 215)
LABELS_HIDDEN_HINT = "条带标签已隐藏（放大、悬停或选中可查看）"
LABELS_COMPACT_HINT = "标签已简化为泳道号（放大可看条带号）"
UNUSED_BAND_TIP = (
    "当前条带策略未采用（虚线框）：它仍是真实测量，只是不计入该泳道合计。"
    "删除才会让它从图上和表里消失。"
)
CLEARANCE_FILL = QColor(theme.ROI_LINE)
CLEARANCE_FILL.setAlpha(70)
CLEARANCE_TIP = (
    "斜纹带 = ROI 自检要求的边缘留白。条带框伸进这条带子，"
    "说明这一侧没有干净背景可以锚基线——把 ROI 这条边往外挪到斜纹空出来为止。"
)

# 被自检点名的分隔线：与诊断条 error 级用同一个红，两处才对得上号
SEPARATOR_FLAG_COLOR = QColor(theme.ERROR)

# 画布标注的基准线宽（设备像素）。界面设置里调的是**倍数**，整族一起缩放，
# 族内的比例恒定——"被点名的分隔线更粗"、"选中的条带框更粗"这两个区分靠的就是
# 比例，做成可调的绝对像素会让它们在某些取值下失效。
ROI_WIDTH = 2.0
BAND_WIDTH = 1.5
BAND_HOVER_WIDTH = 2.5
BAND_SELECT_WIDTH = 3.0
SEPARATOR_WIDTH = 3.0
SEPARATOR_FOCUS_WIDTH = 4.0
SEPARATOR_FLAG_WIDTH = 5.0

# ROI 自检角标：屏幕像素，随缩放恒定
FLAG_PX = 11.0
FLAG_FONT_PT = 7

# 自动测角的参考线：图像像素（要跟着缩放走，否则"平不平行"就看不出来了）
TILT_GUIDE_OVERHANG = 12.0
TILT_GUIDE_CAP = 6.0

HANDLE_PX = 8.0
HIT_PX = 7.0
MIN_ROI_PX = 3.0
SEP_MARGIN = 2.0
EMIT_INTERVAL = 0.03

# 条带标签排布（全部是设备像素：标签用 ItemIgnoresTransformations，屏幕尺寸恒定）
LABEL_FONT_PT = 8
LABEL_PAD_X = 3.0
LABEL_PAD_Y = 1.0
LABEL_MARGIN = 4.0  # 标签与条带框之间的空隙
LABEL_ROW_GAP = 3.0  # 错行时上下两行标签之间留的空隙（行距 = 标签高 + 它）
MAX_LABEL_ROWS = 2  # 最多错两行，再挤就整批隐藏
LABEL_GAP = 3.0  # 同一行相邻标签之间的最小间隙
# 图元 Z 序：条带框 8 < ROI 10 < 分隔线 11 < 标签 11.5 < 手柄 12 < 强制显示的标签 13
LABEL_Z = 11.5
LABEL_Z_FORCED = 13.0
# 橡皮筋框选：拖过这么多设备像素才算"框选"而不是"点了一下空白"
RUBBER_MIN_PX = 3.0
# 分隔线吸附：只有落到谷底 SNAP_PX 图像像素以内才吸，明显想放别处时不硬拽
SNAP_PX = 6.0

# QPixmap 的边长与显存上限：超过就先降采样，否则 fromImage 直接返回空图（一片黑）
MAX_DISPLAY_EDGE = 16384
MAX_DISPLAY_PIXELS = 80_000_000


class Tool(enum.Enum):
    ROI = "roi"
    RULER = "ruler"
    BAND = "band"
    ADJUST = "adjust"


_HANDLE_IDS = ("tl", "t", "tr", "r", "br", "b", "bl", "l")

_HANDLE_CURSORS = {
    "tl": Qt.CursorShape.SizeFDiagCursor,
    "br": Qt.CursorShape.SizeFDiagCursor,
    "tr": Qt.CursorShape.SizeBDiagCursor,
    "bl": Qt.CursorShape.SizeBDiagCursor,
    "t": Qt.CursorShape.SizeVerCursor,
    "b": Qt.CursorShape.SizeVerCursor,
    "l": Qt.CursorShape.SizeHorCursor,
    "r": Qt.CursorShape.SizeHorCursor,
}

_TOOL_CURSORS = {
    Tool.ROI: Qt.CursorShape.CrossCursor,
    Tool.RULER: Qt.CursorShape.CrossCursor,
    Tool.BAND: Qt.CursorShape.CrossCursor,
    Tool.ADJUST: Qt.CursorShape.ArrowCursor,
}


def _handle_anchor(rect: QRectF, handle_id: str) -> QPointF:
    cx, cy = rect.center().x(), rect.center().y()
    return {
        "tl": QPointF(rect.left(), rect.top()),
        "t": QPointF(cx, rect.top()),
        "tr": QPointF(rect.right(), rect.top()),
        "r": QPointF(rect.right(), cy),
        "br": QPointF(rect.right(), rect.bottom()),
        "b": QPointF(cx, rect.bottom()),
        "bl": QPointF(rect.left(), rect.bottom()),
        "l": QPointF(rect.left(), cy),
    }[handle_id]


class BandLabelItem(QGraphicsSimpleTextItem):
    """条带标签：深色底板 + 彩色文字，缩放时屏幕尺寸恒定。

    没有底板的裸文字压在条带、分隔线和 ROI 填充上根本读不出来；底板还让
    "标签是否互相压住"变成一个可以量出来的矩形问题（见
    :meth:`BlotCanvas.band_label_rects`）。
    """

    def __init__(self, text: str, compact_text: str, color: QColor, font: QFont) -> None:
        super().__init__(text)
        self._full_text = text
        self._compact_text = compact_text
        self.compact = False
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setBrush(QBrush(color))
        self.setFont(font)
        self.setZValue(LABEL_Z)
        self._border = QColor(color)
        self._border.setAlpha(160)
        self._forced = False
        self.shown = False
        """这一轮排布是否给它找到了不与别人相交的位置——可见性只看这一个字段。"""

    def set_compact(self, compact: bool) -> None:
        """缩小到放不下"L2-B1"时，退化成只写泳道号"L2"。"""
        compact = bool(compact)
        if compact == self.compact:
            return
        self.compact = compact
        self.setText(self._compact_text if compact else self._full_text)

    def plate_rect(self) -> QRectF:
        """底板矩形（item 坐标 == 设备像素）。"""
        return (
            QGraphicsSimpleTextItem.boundingRect(self)
            .adjusted(-LABEL_PAD_X, -LABEL_PAD_Y, LABEL_PAD_X, LABEL_PAD_Y)
        )

    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt API
        return self.plate_rect()

    def set_forced(self, forced: bool) -> None:
        """悬停 / 选中的条带：优先显示，并抬到手柄之上。"""
        self._forced = bool(forced)
        self.setZValue(LABEL_Z_FORCED if forced else LABEL_Z)

    def is_forced(self) -> bool:
        return self._forced

    def paint(self, painter, option, widget=None) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(self._border, 1))
        painter.setBrush(QBrush(LABEL_PLATE_COLOR))
        painter.drawRoundedRect(self.plate_rect(), 2.5, 2.5)
        painter.restore()
        super().paint(painter, option, widget)


class BandFlagItem(QGraphicsItem):
    """条带框右上角的三角警示角标——ROI 自检点到的那些条带。

    刻意**不引入第五种颜色**：角标用所属条带框自己的描边色（正常红 / 饱和橙 /
    低于检出限钢蓝），区分靠的是"多了一个三角形"这个形状通道。画布上已有的四
    种状态各占一个别的通道——色相（红 / 橙 / 钢蓝）、线型与透明度（未采用是半
    透明虚线）——所以一个条带完全可以既是"饱和"又是"未采用"又"被自检点名"，
    三件事各说各的，不会互相盖掉。角标画在框外的右上角，避开左上角的标签锚点。
    """

    def __init__(self, color: QColor, scale: float = 1.0) -> None:
        super().__init__()
        self._color = QColor(color)
        self._scale = max(0.5, float(scale))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setZValue(LABEL_Z - 0.1)

    def set_color(self, color: QColor) -> None:
        self._color = QColor(color)
        self.update()

    def set_scale_factor(self, scale: float) -> None:
        """字号调大时角标跟着放大，否则那个 ! 会缩成一个点。"""
        scale = max(0.5, float(scale))
        if abs(scale - self._scale) < 1e-6:
            return
        self.prepareGeometryChange()
        self._scale = scale
        self.update()

    def _size(self) -> float:
        return FLAG_PX * self._scale

    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt API
        size = self._size()
        return QRectF(-size, -size, size + 1.0, size + 1.0)

    def paint(self, painter, option, widget=None) -> None:
        size = self._size()
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        path_points = [
            QPointF(0.0, -size),
            QPointF(-size, -size),
            QPointF(0.0, 0.0),
        ]
        painter.setPen(QPen(LABEL_PLATE_COLOR, 1))
        painter.setBrush(QBrush(self._color))
        painter.drawPolygon(path_points)
        font = QFont(painter.font())
        font.setPointSizeF(FLAG_FONT_PT * self._scale)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(LABEL_PLATE_COLOR))
        painter.drawText(
            QRectF(-size + 1.0, -size + 0.5, size - 2.0, size - 2.0),
            Qt.AlignmentFlag.AlignCenter,
            "!",
        )
        painter.restore()


class BandBoxItem(QGraphicsRectItem):
    """可直接操纵的条带结果框：悬停高亮、点击/Ctrl 多选、Delete 删除。"""

    def __init__(
        self,
        index: int,
        lane: int,
        base_color: QColor,
        rect: QRectF,
        line_scale: float = 1.0,
        flag_scale: float = 1.0,
    ) -> None:
        super().__init__(rect)
        self.index = int(index)
        self.lane = int(lane)
        self._base = QColor(base_color)
        self._hover = False
        self._adopted = True
        self._flag_tip = ""
        self._line_scale = max(0.1, float(line_scale))
        self._flag_scale = max(0.5, float(flag_scale))
        self.flag: BandFlagItem | None = None
        self.base_tooltip = ""
        self.label: BandLabelItem | None = None
        self.owner: BlotCanvas | None = None
        self.setZValue(8)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self._apply_style()

    def is_flagged(self) -> bool:
        return self.flag is not None

    def set_scales(self, line_scale: float, flag_scale: float) -> None:
        """界面设置改了线宽 / 字号：重画描边与角标，比例关系原样保留。"""
        self._line_scale = max(0.1, float(line_scale))
        self._flag_scale = max(0.5, float(flag_scale))
        if self.flag is not None:
            self.flag.set_scale_factor(self._flag_scale)
        self._apply_style()

    def set_flagged(self, flagged: bool, tip: str = "") -> None:
        """ROI 自检点到了这一条：右上角挂个三角角标。"""
        self._flag_tip = str(tip or "")
        if flagged and self.flag is None:
            self.flag = BandFlagItem(self._base, self._flag_scale)
            self.flag.setParentItem(self)
            self.flag.setPos(self.rect().right(), self.rect().top())
        elif not flagged and self.flag is not None:
            self.flag.setParentItem(None)
            if self.scene() is not None:
                self.scene().removeItem(self.flag)
            self.flag = None
        self._refresh_tooltip()

    def is_hovered(self) -> bool:
        return self._hover

    def is_adopted(self) -> bool:
        return self._adopted

    def set_adopted(self, adopted: bool) -> None:
        """当前条带策略采不采用这一条。不采用 = 虚线框，**不是**删除。"""
        adopted = bool(adopted)
        if adopted == self._adopted:
            return
        self._adopted = adopted
        self._apply_style()
        if self.label is not None:
            self.label.setOpacity(1.0 if adopted else 0.55)
        self._refresh_tooltip()

    def _refresh_tooltip(self) -> None:
        text = self.base_tooltip
        if not self._adopted:
            text += "\n⚠ " + UNUSED_BAND_TIP
        if self._flag_tip:
            text += "\n⚠ " + self._flag_tip
        self.setToolTip(text)

    def hoverEnterEvent(self, event) -> None:
        self._hover = True
        self._apply_style()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self._hover = False
        self._apply_style()
        super().hoverLeaveEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._apply_style()
        return super().itemChange(change, value)

    def paint(self, painter, option, widget=None) -> None:
        # 屏蔽 Qt 默认的虚线选中框，用自定义选中样式
        option.state = option.state & ~QStyle.StateFlag.State_Selected
        super().paint(painter, option, widget)

    def _apply_style(self) -> None:
        scale = self._line_scale
        if self.isSelected():
            pen = QPen(BAND_SELECT_COLOR, BAND_SELECT_WIDTH * scale)
            brush = QBrush(BAND_SELECT_FILL)
        elif self._hover:
            pen = QPen(self._base.lighter(150), BAND_HOVER_WIDTH * scale)
            brush = QBrush(QColor(255, 255, 255, 18))
        else:
            pen = QPen(self._base, BAND_WIDTH * scale)
            brush = QBrush(Qt.BrushStyle.NoBrush)
        if not self._adopted:
            # 虚线 = "还在，只是不计入合计"。删除是框直接消失，两者一眼可分。
            pen.setStyle(Qt.PenStyle.DashLine)
            color = QColor(pen.color())
            color.setAlpha(150)
            pen.setColor(color)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(brush)
        # 缩小到标签整批隐藏时，选中 / 悬停的这一个仍要能看清是谁
        if self.label is not None:
            self.label.set_forced(self.isSelected() or self._hover)
        if self.owner is not None:
            self.owner.request_label_relayout()


class BlotCanvas(QGraphicsView):
    """图像画布，交互结果通过信号上报（图像像素坐标）。"""

    roiChanged = Signal(float, float, float, float)
    roiEditFinished = Signal()
    roiClearRequested = Signal()
    resegmentRequested = Signal()
    separatorsChanged = Signal(list)
    separatorsEditFinished = Signal()
    rulerMeasured = Signal(float, float, float)  # angle, 终点 scene x, scene y
    bandDrawn = Signal(float, float, float, float)
    bandSelectionChanged = Signal(list)
    bandsDeleteRequested = Signal(list)
    zoomChanged = Signal(float)

    MIN_SCALE = 0.05
    MAX_SCALE = 40.0

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setBackgroundBrush(QBrush(CANVAS_BG_COLOR))
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        # 拖放交给主窗口处理
        self.setAcceptDrops(False)
        self.viewport().setAcceptDrops(False)

        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setZValue(0)
        self._pixmap_item.setVisible(False)
        self._scene.addItem(self._pixmap_item)
        self._image_ref: np.ndarray | None = None
        self._display_ref: np.ndarray | None = None  # 上屏用的（可能降采样的）缓冲
        self._display_step_used = 1
        self._image_w = 0
        self._image_h = 0

        guide_pen = QPen(GUIDE_COLOR, 1, Qt.PenStyle.DashLine)
        guide_pen.setCosmetic(True)
        self._guide_h = QGraphicsLineItem()
        self._guide_v = QGraphicsLineItem()
        for item in (self._guide_h, self._guide_v):
            item.setPen(guide_pen)
            item.setZValue(5)
            item.setVisible(False)
            self._scene.addItem(item)
        self._guides_on = True

        # 界面设置里的线宽 / 字号倍数；1.0 = 出厂值，此时一处像素都不动
        self._roi_scale = 1.0
        self._band_scale = 1.0
        self._sep_scale = 1.0
        self._font_scale = 1.0

        roi_pen = QPen(ROI_COLOR, ROI_WIDTH)
        roi_pen.setCosmetic(True)
        self._roi_item = QGraphicsRectItem()
        self._roi_item.setPen(roi_pen)
        self._roi_item.setBrush(QBrush(ROI_FILL))
        self._roi_item.setZValue(10)
        self._roi_item.setVisible(False)
        self._scene.addItem(self._roi_item)

        # ROI 自检的"边缘留白不足"提示：贴着 ROI 上 / 下边的斜纹带
        self._clearance: tuple[float, float] = (0.0, 0.0)
        self._clearance_items: list[QGraphicsRectItem] = []
        for _ in range(2):
            strip = QGraphicsRectItem()
            strip.setPen(QPen(Qt.PenStyle.NoPen))
            strip.setBrush(QBrush(CLEARANCE_FILL, Qt.BrushStyle.BDiagPattern))
            strip.setZValue(9.5)  # 压在条带框之上、ROI 边框之下
            strip.setVisible(False)
            strip.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            strip.setToolTip(CLEARANCE_TIP)
            self._scene.addItem(strip)
            self._clearance_items.append(strip)

        self._handles: dict[str, QGraphicsRectItem] = {}
        for handle_id in _HANDLE_IDS:
            handle = QGraphicsRectItem(
                -HANDLE_PX / 2, -HANDLE_PX / 2, HANDLE_PX, HANDLE_PX
            )
            handle.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            handle.setPen(QPen(HANDLE_BORDER, 1))
            handle.setBrush(QBrush(ROI_COLOR))
            handle.setZValue(12)
            handle.setVisible(False)
            self._scene.addItem(handle)
            self._handles[handle_id] = handle

        self._roi: QRectF | None = None
        self._separators: list[float] = []
        self._sep_items: list[QGraphicsLineItem] = []
        self._flagged_seps: set[int] = set()
        self._flagged_sep_tip = ""
        # 键盘微调的目标：None / ("roi", -1) / ("sep", 序号)。最后一次鼠标操作
        # 决定它，方向键就作用在"刚才动过的那个"上
        self._nudge_target: tuple[str, int] | None = None
        self._snap_targets: list[float] = []
        self._snap_enabled = True

        self._tool = Tool.ROI
        self._drag: dict | None = None
        self._space_down = False
        self._last_emit = 0.0
        self._temp_item: QGraphicsRectItem | QGraphicsLineItem | None = None
        self._ruler_items: list[QGraphicsLineItem] = []
        self._band_items: list[BandBoxItem] = []
        self._band_labels: list[BandLabelItem] = []
        self._labels_packed = True
        self._labels_compact = False
        self._label_roi_key: tuple | None = None
        self._suspend_label_relayout = False
        self._label_relayout_pending = False
        self._updating_selection = False
        self._scene.selectionChanged.connect(self._on_scene_selection_changed)
        self._apply_tool_cursor()

    # ------------------------------------------------------------------ 图像

    def set_image(self, display_bgr: np.ndarray) -> bool:
        """显示 uint8 BGR 图像；scene 坐标始终等于原图像素坐标。

        超大图先按整数倍降采样再上屏，并给 pixmap item 一个补偿缩放——
        QPixmap 有边长与显存上限，直接塞进去只会得到一片黑。返回 False 表示
        连降采样后都无法生成 pixmap（数据仍可正常定量，只是没有预览）。
        """
        array = np.ascontiguousarray(display_bgr)
        if array.ndim == 2:
            array = np.ascontiguousarray(np.dstack([array] * 3))
        height, width = array.shape[:2]
        self._image_ref = array  # 持引用防悬垂
        self._image_w, self._image_h = width, height

        pixmap = QPixmap()
        step = self._display_step(width, height)
        while True:
            sample = array if step == 1 else np.ascontiguousarray(array[::step, ::step])
            image = QImage(
                sample.data,
                sample.shape[1],
                sample.shape[0],
                sample.strides[0],
                QImage.Format.Format_BGR888,
            )
            pixmap = QPixmap.fromImage(image)
            if not pixmap.isNull():
                self._display_ref = sample
                break
            if min(sample.shape[0], sample.shape[1]) // 2 < 1 or step > 64:
                errorlog.log_message(
                    f"无法为 {width}×{height} 的图片生成预览（降采样步长 {step}）"
                )
                break
            step *= 2

        if pixmap.isNull():
            self._pixmap_item.setPixmap(QPixmap())
            self._pixmap_item.setVisible(False)
        else:
            self._pixmap_item.setPixmap(pixmap)
            self._pixmap_item.setScale(float(step))
            self._pixmap_item.setVisible(True)
        self._display_step_used = step
        self._scene.setSceneRect(0, 0, width, height)
        self._guide_h.setLine(0, height / 2.0, width, height / 2.0)
        self._guide_v.setLine(width / 2.0, 0, width / 2.0, height)
        self._update_guides()
        self.clear_ruler()
        self.viewport().update()
        return not pixmap.isNull()

    def clear_image(self) -> None:
        """退回"没有图"的空画布。

        多文档改造里"关闭当前图片" / "切走当前文档"都走这里，所以它必须把
        set_image 建立起来的**每一样**东西都撤掉：像素图、图像尺寸、场景矩形、
        参考线、ROI、条带框、标尺、缩放变换。漏一样就会留下"图没了但还能拖 ROI"
        之类的鬼影状态。
        """
        self._image_ref = None
        self._display_ref = None
        self._image_w = 0
        self._image_h = 0
        self._display_step_used = 1
        self._pixmap_item.setPixmap(QPixmap())
        self._pixmap_item.setScale(1.0)
        self._pixmap_item.setVisible(False)
        self.clear_roi()
        self.clear_result_boxes()
        self.clear_ruler()
        self._nudge_target = None
        self._snap_targets = []
        self._guide_h.setLine(0, 0, 0, 0)
        self._guide_v.setLine(0, 0, 0, 0)
        self._update_guides()
        self._scene.setSceneRect(0, 0, 0, 0)
        self.setTransform(QTransform())
        self.viewport().update()
        self.zoomChanged.emit(self.current_scale())

    @staticmethod
    def _display_step(width: int, height: int) -> int:
        step = 1
        while (
            max(width // step, height // step) > MAX_DISPLAY_EDGE
            or (width // step) * (height // step) > MAX_DISPLAY_PIXELS
        ):
            step += 1
        return step

    def has_image(self) -> bool:
        return self._image_ref is not None

    def is_interacting(self) -> bool:
        return self._drag is not None

    def fit_view(self) -> None:
        if self._image_ref is None:
            return
        if self._pixmap_item.pixmap().isNull():
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        else:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._on_scale_changed()

    def zoom_100(self) -> None:
        self.setTransform(QTransform())
        self._on_scale_changed()

    def zoom_in(self) -> None:
        self._zoom_by(1.25)

    def zoom_out(self) -> None:
        self._zoom_by(1.0 / 1.25)

    def _zoom_by(self, factor: float) -> None:
        if self._image_ref is None:
            return
        current = self.current_scale()
        target = min(max(current * factor, self.MIN_SCALE), self.MAX_SCALE)
        if abs(target - current) > 1e-9:
            self.scale(target / current, target / current)
            self._on_scale_changed()

    def zoom_to_roi(self) -> None:
        """把视图缩放定位到当前 ROI（带少量留白）。"""
        if self._roi is None or self._image_ref is None:
            return
        margin = max(self._roi.width(), self._roi.height()) * 0.06 + 8.0
        self.fitInView(
            self._roi.adjusted(-margin, -margin, margin, margin),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        self._on_scale_changed()

    def _on_scale_changed(self) -> None:
        """缩放变了：条带标签的屏幕尺寸不变、条带间距变了，必须重排一次。"""
        self._relayout_band_labels()
        self.zoomChanged.emit(self.current_scale())

    def current_scale(self) -> float:
        return float(self.transform().m11())

    def set_guides_visible(self, visible: bool) -> None:
        self._guides_on = bool(visible)
        self._update_guides()

    def _update_guides(self) -> None:
        show = self._guides_on and self._image_ref is not None
        self._guide_h.setVisible(show)
        self._guide_v.setVisible(show)

    # ------------------------------------------------------------------ 工具

    def set_tool(self, tool: Tool) -> None:
        if tool is self._tool:
            return
        self._cancel_drag()
        self._tool = tool
        if tool is not Tool.RULER:
            self.clear_ruler()
        self._apply_tool_cursor()

    def tool(self) -> Tool:
        return self._tool

    def _apply_tool_cursor(self) -> None:
        self.viewport().setCursor(_TOOL_CURSORS[self._tool])

    # ------------------------------------------------------------------ ROI / 分隔线（程序化接口，不发信号）

    def set_roi(self, x: float, y: float, w: float, h: float) -> None:
        rect = QRectF(float(x), float(y), float(w), float(h)).normalized()
        if self._image_ref is not None:
            rect = rect.intersected(QRectF(0, 0, self._image_w, self._image_h))
        self._roi = rect
        self._sync_roi_items()

    def roi(self) -> tuple[float, float, float, float] | None:
        if self._roi is None:
            return None
        r = self._roi
        return (r.x(), r.y(), r.width(), r.height())

    def clear_roi(self) -> None:
        self._roi = None
        self._separators = []
        self._sync_roi_items()

    def set_separators(self, separators: list[float]) -> None:
        values = sorted(float(s) for s in separators)
        if len(values) != len(self._separators):
            self._flagged_seps.clear()  # 条数变了，旧下标不再指着同一条线
        if self._roi is not None:
            lo = self._roi.left() + 1.0
            hi = self._roi.right() - 1.0
            values = [min(max(v, lo), hi) for v in values]
        self._separators = values
        self._sync_roi_items()

    def separators(self) -> list[float]:
        return list(self._separators)

    # ------------------------------------------------------------------ 吸附 / 键盘微调

    def set_snap_targets(self, targets) -> None:
        """设定分隔线可吸附的位置（图像 x 坐标，通常是剖面谷底）。"""
        self._snap_targets = sorted(float(t) for t in (targets or []))

    def set_snap_enabled(self, enabled: bool) -> None:
        self._snap_enabled = bool(enabled)

    def snap_targets(self) -> list[float]:
        return list(self._snap_targets)

    def _snap_separator_x(self, x: float, lo: float, hi: float) -> float:
        """把 x 吸到最近的谷底——但只在 SNAP_PX 之内，且不越过相邻分隔线。"""
        if not self._snap_enabled or not self._snap_targets:
            return x
        best = min(self._snap_targets, key=lambda t: abs(t - x))
        if abs(best - x) > SNAP_PX or not (lo <= best <= hi):
            return x
        return float(best)

    def nudge_target(self) -> tuple[str, int] | None:
        return self._nudge_target

    def set_nudge_target(self, target: tuple[str, int] | None) -> None:
        self._nudge_target = target
        self._sync_roi_items()

    def _default_nudge_target(self) -> tuple[str, int] | None:
        if self._nudge_target is not None:
            kind, index = self._nudge_target
            if kind == "sep" and 0 <= index < len(self._separators):
                return self._nudge_target
            if kind == "roi" and self._roi is not None:
                return self._nudge_target
        return ("roi", -1) if self._roi is not None else None

    def nudge(self, dx: float, dy: float) -> bool:
        """方向键微调：作用于最后动过的分隔线，否则整体平移 ROI。

        鼠标拖动受 scene↔viewport 往返精度限制（误差约 0.5/缩放倍数，20% 缩放
        时是 ±2.5 图像像素），而 ROI 边界直接决定积分区间，必须能像素级微调。
        """
        target = self._default_nudge_target()
        if target is None or self._roi is None:
            return False
        kind, index = target
        if kind == "sep":
            if not (0 <= index < len(self._separators)) or abs(dx) < 1e-9:
                return False
            lo = self._roi.left() + SEP_MARGIN
            hi = self._roi.right() - SEP_MARGIN
            if index > 0:
                lo = max(lo, self._separators[index - 1] + SEP_MARGIN)
            if index < len(self._separators) - 1:
                hi = min(hi, self._separators[index + 1] - SEP_MARGIN)
            if hi < lo:
                return False
            moved = float(min(max(self._separators[index] + dx, lo), hi))
            if abs(moved - self._separators[index]) < 1e-9:
                return False
            self._separators[index] = moved
            self._sync_roi_items()
            self.separatorsChanged.emit(list(self._separators))
            self.separatorsEditFinished.emit()
            return True
        rect = self._roi
        new_x = min(max(rect.x() + dx, 0.0), max(0.0, self._image_w - rect.width()))
        new_y = min(max(rect.y() + dy, 0.0), max(0.0, self._image_h - rect.height()))
        if abs(new_x - rect.x()) < 1e-9 and abs(new_y - rect.y()) < 1e-9:
            return False
        shift = new_x - rect.x()
        self._roi = QRectF(new_x, new_y, rect.width(), rect.height())
        self._separators = [s + shift for s in self._separators]
        self._sync_roi_items()
        self.roiChanged.emit(new_x, new_y, rect.width(), rect.height())
        self.roiEditFinished.emit()
        return True

    def resize_roi_by(self, dx: float, dy: float) -> bool:
        """Ctrl+方向键：只挪 ROI 的右 / 下边界，用于精调积分区间宽高。"""
        if self._roi is None:
            return False
        rect = self._roi
        width = min(
            max(rect.width() + dx, MIN_ROI_PX), max(MIN_ROI_PX, self._image_w - rect.x())
        )
        height = min(
            max(rect.height() + dy, MIN_ROI_PX),
            max(MIN_ROI_PX, self._image_h - rect.y()),
        )
        if abs(width - rect.width()) < 1e-9 and abs(height - rect.height()) < 1e-9:
            return False
        self._roi = QRectF(rect.x(), rect.y(), width, height)
        lo = self._roi.left() + 1.0
        hi = self._roi.right() - 1.0
        self._separators = [min(max(s, lo), hi) for s in self._separators]
        self._sync_roi_items()
        self.roiChanged.emit(rect.x(), rect.y(), width, height)
        self.roiEditFinished.emit()
        return True

    def _sync_roi_items(self) -> None:
        if self._roi is None or self._roi.width() <= 0 or self._roi.height() <= 0:
            self._roi_item.setVisible(False)
            for handle in self._handles.values():
                handle.setVisible(False)
            for item in self._sep_items:
                self._scene.removeItem(item)
            self._sep_items.clear()
            self._sync_clearance_items()
            if self._label_roi_key is not None:
                self._label_roi_key = None
                self._relayout_band_labels()
            return
        rect = self._roi
        self._roi_item.setRect(rect)
        self._roi_item.setVisible(True)
        self._sync_clearance_items()
        for handle_id, handle in self._handles.items():
            handle.setPos(_handle_anchor(rect, handle_id))
            handle.setVisible(True)

        while len(self._sep_items) > len(self._separators):
            self._scene.removeItem(self._sep_items.pop())
        while len(self._sep_items) < len(self._separators):
            line = QGraphicsLineItem()
            line.setZValue(11)
            self._scene.addItem(line)
            self._sep_items.append(line)
        focus = self._nudge_target
        focused = focus[1] if focus is not None and focus[0] == "sep" else None
        for index, (item, sep) in enumerate(zip(self._sep_items, self._separators)):
            # 三种状态可以叠加：自检点名决定颜色（红），键盘微调决定线型（虚线）。
            # 分开走两个通道，正在微调一条被点名的线时两件事都还看得见。
            flagged = index in self._flagged_seps
            color = SEPARATOR_FLAG_COLOR if flagged else SEPARATOR_COLOR
            width = (
                SEPARATOR_FLAG_WIDTH
                if flagged
                else (SEPARATOR_FOCUS_WIDTH if index == focused else SEPARATOR_WIDTH)
            ) * self._sep_scale
            if index == focused:
                color = SEPARATOR_FLAG_COLOR if flagged else SEPARATOR_FOCUS_COLOR
                pen = QPen(color, width)
                pen.setStyle(Qt.PenStyle.DashLine)
            else:
                pen = QPen(color, width)
            pen.setCosmetic(True)
            item.setPen(pen)
            item.setToolTip(self._flagged_sep_tip if flagged else "")
            x = min(max(sep, rect.left()), rect.right())
            item.setLine(x, rect.top(), x, rect.bottom())
        # ROI 手柄挪了位置，标签得重新让路。只在 ROI 真的变了、且不在拖动中时重排：
        # 拖分隔线也会走到这里，每个 move 事件重排几十个标签太贵（松手时补一次）
        key = (rect.x(), rect.y(), rect.width(), rect.height())
        if self._drag is None and key != self._label_roi_key:
            self._label_roi_key = key
            self._relayout_band_labels()

    # ------------------------------------------------------------------ 结果叠加

    # ------------------------------------------------------------------ 界面设置

    def set_annotation_scales(
        self,
        roi: float = 1.0,
        band: float = 1.0,
        separator: float = 1.0,
        font: float = 1.0,
    ) -> None:
        """应用界面设置里的线宽 / 字号倍数（1.0 = 出厂值）。

        字号倍数同时管条带标签与自检角标：字放大了角标不放大，那个 ! 会缩成一
        个看不出是什么的点；标签放大之后会重新走一遍错行避让，挤不下就照旧整
        批隐藏——这条降级链本来就是为"标签放不下"设计的，字号变大只是让它更早
        触发而已。
        """
        self._roi_scale = max(0.1, float(roi))
        self._band_scale = max(0.1, float(band))
        self._sep_scale = max(0.1, float(separator))
        self._font_scale = max(0.5, float(font))

        pen = QPen(ROI_COLOR, ROI_WIDTH * self._roi_scale)
        pen.setCosmetic(True)
        self._roi_item.setPen(pen)
        for item in self._band_items:
            item.set_scales(self._band_scale, self._font_scale)
        label_font = self._label_font()
        for label in self._band_labels:
            label.setFont(label_font)
        self._sync_roi_items()
        self._relayout_band_labels()

    def annotation_scales(self) -> tuple[float, float, float, float]:
        return (self._roi_scale, self._band_scale, self._sep_scale, self._font_scale)

    def _label_font(self) -> QFont:
        font = QFont(self.font().family(), -1)
        font.setPointSizeF(LABEL_FONT_PT * self._font_scale)
        return font

    def show_result_boxes(self, measurements) -> None:
        self.clear_result_boxes()
        label_font = self._label_font()
        self._updating_selection = True
        try:
            for index, m in enumerate(measurements):
                saturated = float(m.saturated_fraction) >= 0.05
                weak = bool(m.below_detection)
                if weak:
                    color = RESULT_WEAK_COLOR
                else:
                    color = RESULT_SAT_COLOR if saturated else RESULT_COLOR
                box = BandBoxItem(
                    index=index,
                    lane=int(m.lane),
                    base_color=color,
                    rect=QRectF(float(m.x), float(m.y), float(m.width), float(m.height)),
                    line_scale=self._band_scale,
                    flag_scale=self._font_scale,
                )
                box.setToolTip(
                    f"L{m.lane}-B{m.band}\n"
                    f"积分灰度：{m.integrated_density:,.1f}\n"
                    f"平均灰度：{m.mean_intensity:.1f}\n"
                    f"背景：{m.background:.1f}\n"
                    f"饱和：{m.saturated_fraction * 100:.1f}%\n"
                    f"信噪比：{m.snr:.1f}" + (f"（检出限 {m.detection_limit:,.0f}）" if weak else "")
                    + ("\n⚠ 低于检出限，该积分与空白泳道无法区分" if weak else "")
                    + ("\n⚠ 饱和比例偏高，定量可能失真" if saturated else "")
                    + "\n（点击选中，Delete 删除；右键更多操作）"
                )
                box.base_tooltip = box.toolTip()
                self._scene.addItem(box)
                self._band_items.append(box)

                label = BandLabelItem(
                    f"L{m.lane}-B{m.band}", f"L{m.lane}", color, label_font
                )
                label.setPos(float(m.x), float(m.y))
                label.setVisible(False)
                box.label = label
                box.owner = self
                self._scene.addItem(label)
                self._band_labels.append(label)
        finally:
            self._updating_selection = False
        self._relayout_band_labels()

    def set_unused_bands(self, indices) -> None:
        """把这些条带画成虚线框：当前条带策略不采用它们，但它们仍然存在。"""
        unused = {int(i) for i in (indices or [])}
        for item in self._band_items:
            item.set_adopted(item.index not in unused)

    def unused_bands(self) -> list[int]:
        return sorted(item.index for item in self._band_items if not item.is_adopted())

    def set_flagged_bands(self, indices, tip: str = "") -> None:
        """ROI 自检点名的条带挂上角标（形状通道，不新增颜色）。"""
        flagged = {int(i) for i in (indices or [])}
        for item in self._band_items:
            item.set_flagged(item.index in flagged, tip)

    def flagged_bands(self) -> list[int]:
        return sorted(item.index for item in self._band_items if item.is_flagged())

    def set_flagged_separators(self, indices, tip: str = "") -> None:
        """把自检点名的分隔线画成警示红并加粗（索引 = ``separators()`` 的下标）。

        ``ISSUE_SEPARATOR_IN_BAND`` 说的是"这条线切在条带里"，而线本身就是用户
        要拖的东西——所以提示直接落在线上，不另起图元。这里换的是**已有元素的
        状态色**（橙 → 红），不是给条带框加第五种颜色；红与诊断条 error 级同色，
        看到红线就知道去哪读原因。

        这条判据在 ``roi.py`` 给候选谷加了条带禁区之后已经很少触发，是兜底而非
        常态路径，所以不做角标、不做动画。
        """
        self._flagged_seps = {int(i) for i in indices}
        self._flagged_sep_tip = str(tip)
        self._sync_roi_items()

    def flagged_separators(self) -> list[int]:
        return sorted(self._flagged_seps)

    def set_clearance_strips(self, top: float = 0.0, bottom: float = 0.0) -> None:
        """在 ROI 内侧沿上 / 下边画一条斜纹留白带（0 = 不画）。

        它把"这一侧要留多少干净背景"直接画在用户要拖的那条边上：条带框只要还
        压在斜纹里就是不合格，往外拖到斜纹空出来即可。用的是 ROI 自己的青色 +
        斜纹填充——纹理通道，条带框的四种状态一个也没动。
        """
        self._clearance = (max(0.0, float(top)), max(0.0, float(bottom)))
        self._sync_clearance_items()

    def clearance_strips(self) -> tuple[float, float]:
        return self._clearance

    def _sync_clearance_items(self) -> None:
        rect = self._roi
        for item, height, at_top in (
            (self._clearance_items[0], self._clearance[0], True),
            (self._clearance_items[1], self._clearance[1], False),
        ):
            if rect is None or height <= 0.0 or rect.height() <= 0.0:
                item.setVisible(False)
                continue
            band = min(height, rect.height())
            top = rect.top() if at_top else rect.bottom() - band
            item.setRect(QRectF(rect.left(), top, rect.width(), band))
            item.setVisible(True)

    def clear_result_boxes(self) -> None:
        self._updating_selection = True
        try:
            for item in (*self._band_items, *self._band_labels):
                self._scene.removeItem(item)
            self._band_items.clear()
            self._band_labels.clear()
        finally:
            self._updating_selection = False
        self._labels_packed = True
        self._labels_compact = False

    # ------------------------------------------------------------------ 标签排布

    def request_label_relayout(self) -> None:
        """选中 / 悬停变了：标签的显示优先级跟着变，重排一次。"""
        if self._suspend_label_relayout:
            self._label_relayout_pending = True
            return
        self._relayout_band_labels()

    def _label_baselines(self, indices) -> dict[int, float]:
        """每个条带的标签挂在哪条基准线上（图像 y 坐标）。

        同一排条带共用一条基准线，标签才能排成整齐的一两行——各挂各的框顶时，
        高低差不到一个标签高度的两个框会互相卡住，错行反而处处相交。纵向差得
        远（超过一个条带高度）的条带自成一排，标签不会被拽到很远的地方。
        """
        order = sorted(indices, key=lambda i: self._band_items[i].rect().top())
        if not order:
            return {}
        heights = [self._band_items[i].rect().height() for i in order]
        tolerance = max(1.0, float(np.median(heights)))
        baselines: dict[int, float] = {}
        anchor = self._band_items[order[0]].rect().top()
        for i in order:
            top = self._band_items[i].rect().top()
            if top - anchor > tolerance:
                anchor = top
            baselines[i] = anchor
        return baselines

    def _label_obstacles(self) -> list[QRectF]:
        """标签**尽量**绕开的东西：ROI 的 8 个缩放手柄。

        手柄画在标签之上（它们要用鼠标去抓，不能被底板压住），标签压在下面就会
        被啃掉半个字（"L5-B1" 显示成 "L▉B1"）。这是软约束：绕不开时宁可让它压在
        手柄下面，也不能因为一个手柄把整批标签都判成"摆不开"。
        """
        if self._roi is None or not self._roi_item.isVisible():
            return []
        scale = self.current_scale()
        half = HANDLE_PX / 2.0 + 2.0
        return [
            QRectF(
                _handle_anchor(self._roi, handle_id).x() * scale - half,
                _handle_anchor(self._roi, handle_id).y() * scale - half,
                half * 2.0,
                half * 2.0,
            )
            for handle_id in _HANDLE_IDS
        ]

    def _pack_labels(self, indices) -> tuple[dict[int, tuple[float, float]], bool, set]:
        """给这批标签在条带框上方找互不相交的位置（相对框左上角的设备像素偏移）。

        最多错 ``MAX_LABEL_ROWS`` 行；判定用的是完整矩形（含底板与左右间隙）。
        返回 (偏移表, 是否全放下, 躲不开手柄的那些序号)——放不下的那个也会拿到一个
        贴着框顶的兜底位置，只是不会被显示。
        """
        indices = list(indices)
        scale = self.current_scale()
        image_right = float(self._image_w) * scale
        baselines = self._label_baselines(indices)
        # 从左到右放，同一列（纵向叠放的条带）按从上到下
        order = sorted(
            indices,
            key=lambda i: (
                self._band_items[i].rect().left(),
                self._band_items[i].rect().top(),
            ),
        )
        placed: list[QRectF] = []
        obstacles = self._label_obstacles()
        offsets: dict[int, tuple[float, float]] = {}
        over_handles: set[int] = set()
        ok = True
        for i in order:
            box_rect = self._band_items[i].rect()
            plate = self._band_labels[i].plate_rect()
            box_left = box_rect.left() * scale
            box_top = box_rect.top() * scale
            box_w = box_rect.width() * scale
            box_h = box_rect.height() * scale
            base_top = baselines[i] * scale
            # 水平居中对齐条带框，但不许探出图像左右边界
            left = box_left + (box_w - plate.width()) / 2.0
            left = min(max(left, 0.0), max(0.0, image_right - plate.width()))
            spot: tuple[float, float] | None = None
            # 行距必须大于标签自身高度，否则错开的那一行反而压住下一行
            row_step = plate.height() + LABEL_ROW_GAP
            # 第一轮连手柄一起躲，躲不掉再退一步只保证标签之间不重叠
            for dodge_handles in (True, False):
                for row in range(MAX_LABEL_ROWS):
                    top = base_top - LABEL_MARGIN - plate.height() - row * row_step
                    if top < 0.0:  # 顶到图外就翻到框下方
                        top = box_top + box_h + LABEL_MARGIN + row * row_step
                    probe = QRectF(
                        left - LABEL_GAP,
                        top,
                        plate.width() + 2 * LABEL_GAP,
                        plate.height(),
                    )
                    if any(probe.intersects(r) for r in placed):
                        continue
                    if dodge_handles and any(probe.intersects(r) for r in obstacles):
                        continue
                    placed.append(probe)
                    spot = (left, top)
                    if not dodge_handles:
                        over_handles.add(i)
                    break
                if spot is not None:
                    break
            if spot is None:
                ok = False
                spot = (left, base_top - LABEL_MARGIN - plate.height())
            offsets[i] = (spot[0] - box_left, spot[1] - box_top)
        return offsets, ok, over_handles

    def _set_label_texts(self, compact: bool) -> None:
        for label in self._band_labels:
            label.set_compact(compact)

    def _relayout_band_labels(self) -> None:
        """按当前缩放重排条带标签：逐级降级，绝不允许"糊成一团"的中间态。

        标签的屏幕尺寸恒定，条带间距却随缩放线性变化——51% 时条带只剩 20 设备
        像素宽，"L1-B1" 却要 38 像素，相邻标签必然叠在一起。降级顺序：

        1. 完整文案 ``L2-B1``，最多错 ``MAX_LABEL_ROWS`` 行；
        2. 放不下就退化成只写泳道号 ``L2``，同样允许错行；
        3. 还放不下就整批隐藏，只留用户自己选中 / 悬停的那几个（完整文案）；
        4. 选中的也摆不开时，只留鼠标正指着的那一个。

        只显示"能摆开的一部分"是不做的：用户没法预期哪几个会留下。
        """
        if not self._band_labels:
            self._labels_packed = True
            self._labels_compact = False
            return
        every = list(range(len(self._band_labels)))
        for compact in (False, True):
            self._set_label_texts(compact)
            offsets, ok, over = self._pack_labels(every)
            if ok:
                self._apply_label_layout(offsets, set(every), over)
                self._labels_packed = True
                self._labels_compact = compact
                return
        self._set_label_texts(False)
        offsets, _, over = self._pack_labels(every)
        visible: set[int] = set()
        forced = [i for i in every if self._band_labels[i].is_forced()]
        if forced:
            sub, sub_ok, sub_over = self._pack_labels(forced)
            offsets.update(sub)
            if sub_ok:
                visible = set(forced)
                over = sub_over
            else:
                hovered = [i for i in forced if self._band_items[i].is_hovered()]
                extra, _, extra_over = self._pack_labels(hovered)
                offsets.update(extra)
                visible = set(hovered)
                over = extra_over
        self._apply_label_layout(offsets, visible, over)
        self._labels_packed = False
        self._labels_compact = False

    def _apply_label_layout(self, offsets: dict, visible: set, over_handles: set) -> None:
        for i, label in enumerate(self._band_labels):
            plate = label.plate_rect()
            dx, dy = offsets[i]
            label.setTransform(
                QTransform.fromTranslate(dx - plate.left(), dy - plate.top())
            )
            # 躲不开手柄的那几个改画在手柄之上：手柄的命中判定走的是坐标计算
            # （见 _handle_at），被盖住照样能抓，但字被啃掉就真的读不出来了
            label.setZValue(
                LABEL_Z_FORCED if (label.is_forced() or i in over_handles) else LABEL_Z
            )
            label.shown = i in visible
            label.setVisible(label.shown)

    def labels_packed(self) -> bool:
        """当前缩放下条带标签是不是整批可读地铺开了。"""
        return bool(self._labels_packed)

    def labels_compact(self) -> bool:
        """整批显示的是不是简化文案（只有泳道号）。"""
        return bool(self._labels_compact)

    def band_label_rects(self) -> list[QRectF]:
        """可见条带标签在视口里的矩形（设备像素），供重叠检查使用。"""
        rects: list[QRectF] = []
        for label in self._band_labels:
            if not label.isVisible():
                continue
            origin = self.mapFromScene(label.pos())
            transform = label.transform()
            plate = label.plate_rect()
            rects.append(
                QRectF(
                    origin.x() + transform.dx() + plate.left(),
                    origin.y() + transform.dy() + plate.top(),
                    plate.width(),
                    plate.height(),
                )
            )
        return rects

    # ------------------------------------------------------------------ 条带选中

    def selected_bands(self) -> list[int]:
        """当前选中条带的原始测量索引（升序）。"""
        return sorted(item.index for item in self._band_items if item.isSelected())

    def select_bands(self, indices) -> None:
        """程序化设置选中（不发 bandSelectionChanged，供表格→画布同步）。"""
        self._set_selection(indices)

    def clear_band_selection(self) -> None:
        self.select_bands([])

    def _on_scene_selection_changed(self) -> None:
        if self._updating_selection:
            return
        self.bandSelectionChanged.emit(self.selected_bands())

    def _indices_for_lane(self, lane: int) -> list[int]:
        return sorted(item.index for item in self._band_items if item.lane == int(lane))

    def _band_at(self, vp_point) -> BandBoxItem | None:
        for item in self.items(vp_point):
            if isinstance(item, BandBoxItem):
                return item
        return None

    def _set_selection(self, indices) -> None:
        """一次性设置选中集合，只发一次 bandSelectionChanged。

        逐个 setSelected 会让 scene 的 selectionChanged 发 N 次，每次都去刷一遍
        结果表——批量框选几十个条带时这一串同步足够把界面卡住。
        """
        wanted = {int(i) for i in indices}
        self._updating_selection = True
        self._suspend_label_relayout = True
        self._label_relayout_pending = False
        try:
            for item in self._band_items:
                item.setSelected(item.index in wanted)
        finally:
            self._updating_selection = False
            self._suspend_label_relayout = False
        if self._label_relayout_pending:
            self._label_relayout_pending = False
            self._relayout_band_labels()

    def _click_select_band(self, band: BandBoxItem, modifiers) -> None:
        """点选条带：Ctrl 反选（可减选）、Shift 加选、无修饰键单选。"""
        selected = set(self.selected_bands())
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            selected.symmetric_difference_update({band.index})
        elif modifiers & Qt.KeyboardModifier.ShiftModifier:
            selected.add(band.index)
        else:
            selected = {band.index}
        self._set_selection(selected)
        self.bandSelectionChanged.emit(self.selected_bands())

    # ------------------------------------------------------------------ 橡皮筋框选

    def _begin_rubber_band(self, scene_pos: QPointF, additive: bool) -> None:
        start = self._clamp_to_image(scene_pos)
        base = set(self.selected_bands()) if additive else set()
        pen = QPen(BAND_SELECT_COLOR, 1, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self._temp_item = QGraphicsRectItem(QRectF(start, start))
        self._temp_item.setPen(pen)
        self._temp_item.setBrush(QBrush(BAND_SELECT_FILL))
        self._temp_item.setZValue(25)
        self._scene.addItem(self._temp_item)
        self._drag = {
            "kind": "rubber",
            "start": start,
            "additive": additive,
            "base": base,
            "before": set(self.selected_bands()),
        }
        if not additive and self._band_items:
            self._set_selection(base)

    def _rubber_selection(self, rect: QRectF, base: set) -> set:
        """与框相交（不必完全框住）的条带都算选中——小框太难完整套住。"""
        hit = {item.index for item in self._band_items if item.rect().intersects(rect)}
        return hit | base

    def _drag_rubber_band(self, scene_pos: QPointF) -> None:
        drag = self._drag
        rect = QRectF(drag["start"], self._clamp_to_image(scene_pos)).normalized()
        if self._temp_item is not None:
            self._temp_item.setRect(rect)
        # 拖动过程中只改画布上的选中样式，不往结果表发信号（松手时统一发一次）
        self._set_selection(self._rubber_selection(rect, drag["base"]))

    def _finish_rubber_band(self, drag: dict, scene_pos: QPointF) -> None:
        if self._temp_item is not None:
            self._scene.removeItem(self._temp_item)
            self._temp_item = None
        start: QPointF = drag["start"]
        end = self._clamp_to_image(scene_pos)
        scale = max(self.current_scale(), 1e-6)
        moved = max(abs(end.x() - start.x()), abs(end.y() - start.y())) * scale
        if moved < RUBBER_MIN_PX:
            # 只是点了一下空白：非加选时等于清空选中
            self._set_selection(drag["base"])
        else:
            self._set_selection(
                self._rubber_selection(QRectF(start, end).normalized(), drag["base"])
            )
        selected = self.selected_bands()
        if set(selected) != drag["before"]:
            self.bandSelectionChanged.emit(selected)

    # ------------------------------------------------------------------ 坐标 / 命中测试

    def _scene_pos(self, event) -> QPointF:
        return self.mapToScene(event.position().toPoint())

    def _clamp_to_image(self, pos: QPointF) -> QPointF:
        return QPointF(
            min(max(pos.x(), 0.0), float(self._image_w)),
            min(max(pos.y(), 0.0), float(self._image_h)),
        )

    def _handle_at(self, vp_pos: QPointF) -> str | None:
        if self._roi is None:
            return None
        for handle_id in _HANDLE_IDS:
            anchor = self.mapFromScene(_handle_anchor(self._roi, handle_id))
            if (
                abs(anchor.x() - vp_pos.x()) <= HIT_PX
                and abs(anchor.y() - vp_pos.y()) <= HIT_PX
            ):
                return handle_id
        return None

    def _separator_at(self, vp_pos: QPointF) -> int | None:
        if self._roi is None:
            return None
        top = self.mapFromScene(QPointF(self._roi.left(), self._roi.top())).y()
        bottom = self.mapFromScene(QPointF(self._roi.left(), self._roi.bottom())).y()
        if not (min(top, bottom) - HIT_PX <= vp_pos.y() <= max(top, bottom) + HIT_PX):
            return None
        for index, sep in enumerate(self._separators):
            x = self.mapFromScene(QPointF(sep, self._roi.top())).x()
            if abs(x - vp_pos.x()) <= HIT_PX - 1:
                return index
        return None

    # ------------------------------------------------------------------ 鼠标事件

    def mousePressEvent(self, event) -> None:
        vp_pos = QPointF(event.position())
        scene_pos = self._scene_pos(event)

        if event.button() == Qt.MouseButton.MiddleButton or (
            self._space_down and event.button() == Qt.MouseButton.LeftButton
        ):
            self._drag = {"kind": "pan", "last": vp_pos}
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.MouseButton.RightButton:
            index = self._separator_at(vp_pos)
            if index is not None:
                self._show_separator_menu(index, event.globalPosition().toPoint())
                event.accept()
                return
            band = self._band_at(event.position().toPoint())
            if band is not None:
                self._show_band_menu(band, event.globalPosition().toPoint())
                event.accept()
                return
            if self._roi_edge_at(vp_pos):
                self._show_roi_menu(event.globalPosition().toPoint())
                event.accept()
                return
            super().mousePressEvent(event)
            return

        if event.button() != Qt.MouseButton.LeftButton or self._image_ref is None:
            super().mousePressEvent(event)
            return

        # 精确目标优先：手柄 / 分隔线
        if self._tool in (Tool.ROI, Tool.ADJUST) and self._roi is not None:
            handle_id = self._handle_at(vp_pos)
            if handle_id is not None:
                self._drag = {
                    "kind": "resize_roi",
                    "handle": handle_id,
                    "orig_rect": QRectF(self._roi),
                    "orig_seps": list(self._separators),
                }
                self.set_nudge_target(("roi", -1))
                event.accept()
                return
            sep_index = self._separator_at(vp_pos)
            if sep_index is not None:
                self._drag = {
                    "kind": "sep",
                    "index": sep_index,
                    "orig_seps": list(self._separators),
                }
                self.set_nudge_target(("sep", sep_index))
                event.accept()
                return

        # 任何工具模式下，按在条带框上优先做选中（Ctrl 反选、Shift 加选）
        band = self._band_at(event.position().toPoint())
        if band is not None:
            self._click_select_band(band, event.modifiers())
            event.accept()
            return

        # 调整模式下按在空白处 = 橡皮筋框选多个条带。ROI 内的空白默认还是整体
        # 拖动 ROI（那是这个模式的主业），按住 Ctrl / Shift 才改成框选加选——
        # 其余三个工具的拖拽各有各的用途（画 ROI / 量角 / 补画条带），不能抢。
        if self._tool is Tool.ADJUST:
            additive = bool(
                event.modifiers()
                & (
                    Qt.KeyboardModifier.ControlModifier
                    | Qt.KeyboardModifier.ShiftModifier
                )
            )
            inside_roi = self._roi is not None and self._roi.contains(scene_pos)
            if additive or not inside_roi:
                self._begin_rubber_band(scene_pos, additive)
                event.accept()
                return

        # 按在空白处：取消条带选中，再进入各工具的绘制/拖动
        if self._band_items and self.selected_bands():
            self._scene.clearSelection()

        if self._tool is Tool.RULER:
            start = scene_pos
            pen = QPen(RULER_COLOR, 2)
            pen.setCosmetic(True)
            self._temp_item = QGraphicsLineItem(
                start.x(), start.y(), start.x(), start.y()
            )
            self._temp_item.setPen(pen)
            self._temp_item.setZValue(20)
            self._scene.addItem(self._temp_item)
            self._drag = {"kind": "ruler", "start": start}
            event.accept()
            return

        if (
            self._tool is Tool.ADJUST
            and self._roi is not None
            and self._roi.contains(scene_pos)
        ):
            self._drag = {
                "kind": "move_roi",
                "grab": scene_pos,
                "orig_rect": QRectF(self._roi),
                "orig_seps": list(self._separators),
            }
            self.set_nudge_target(("roi", -1))
            self.viewport().setCursor(Qt.CursorShape.SizeAllCursor)
            event.accept()
            return

        if self._tool in (Tool.ROI, Tool.BAND):
            start = self._clamp_to_image(scene_pos)
            color = ROI_COLOR if self._tool is Tool.ROI else BAND_COLOR
            pen = QPen(color, 2)
            pen.setCosmetic(True)
            self._temp_item = QGraphicsRectItem(QRectF(start, start))
            self._temp_item.setPen(pen)
            self._temp_item.setZValue(20)
            self._scene.addItem(self._temp_item)
            self._drag = {
                "kind": "new_roi" if self._tool is Tool.ROI else "new_band",
                "start": start,
            }
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        vp_pos = QPointF(event.position())
        scene_pos = self._scene_pos(event)

        if self._drag is None:
            self._update_hover_cursor(vp_pos, scene_pos)
            super().mouseMoveEvent(event)
            return

        kind = self._drag["kind"]
        if kind == "pan":
            delta = vp_pos - self._drag["last"]
            self._drag["last"] = vp_pos
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
        elif kind == "ruler" and self._temp_item is not None:
            start = self._drag["start"]
            self._temp_item.setLine(start.x(), start.y(), scene_pos.x(), scene_pos.y())
        elif kind in ("new_roi", "new_band") and self._temp_item is not None:
            end = self._clamp_to_image(scene_pos)
            self._temp_item.setRect(QRectF(self._drag["start"], end).normalized())
        elif kind == "rubber":
            self._drag_rubber_band(scene_pos)
        elif kind == "move_roi":
            self._drag_move_roi(scene_pos)
        elif kind == "resize_roi":
            self._drag_resize_roi(scene_pos)
        elif kind == "sep":
            # 按住 Alt 临时关闭吸附：想放在谷底旁边时不用先去开关一趟
            snap = not bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
            self._drag_separator(scene_pos, snap=snap)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self._drag is None:
            super().mouseReleaseEvent(event)
            return
        drag = self._drag
        self._drag = None
        kind = drag["kind"]
        scene_pos = self._scene_pos(event)

        if kind == "pan":
            self._apply_tool_cursor()
        elif kind == "ruler":
            self._finish_ruler(drag, scene_pos)
        elif kind in ("new_roi", "new_band"):
            self._finish_new_rect(kind, drag, scene_pos)
        elif kind == "rubber":
            self._finish_rubber_band(drag, scene_pos)
        elif kind in ("move_roi", "resize_roi"):
            self._apply_tool_cursor()
            if self._roi is not None:
                r = self._roi
                self.roiChanged.emit(r.x(), r.y(), r.width(), r.height())
            self.roiEditFinished.emit()
        elif kind == "sep":
            self.separatorsChanged.emit(list(self._separators))
            self.separatorsEditFinished.emit()
        if kind in ("move_roi", "resize_roi", "sep"):
            # 拖动期间跳过的那一次，在这里补上
            if self._roi is not None:
                r = self._roi
                self._label_roi_key = (r.x(), r.y(), r.width(), r.height())
            self._relayout_band_labels()
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._tool is Tool.ADJUST
            and self._roi is not None
            and self._image_ref is not None
        ):
            vp_pos = QPointF(event.position())
            scene_pos = self._scene_pos(event)
            if (
                self._roi.contains(scene_pos)
                and self._handle_at(vp_pos) is None
                and self._separator_at(vp_pos) is None
            ):
                x = min(
                    max(scene_pos.x(), self._roi.left() + MIN_ROI_PX),
                    self._roi.right() - MIN_ROI_PX,
                )
                if all(abs(x - s) >= MIN_ROI_PX for s in self._separators):
                    self._separators = sorted([*self._separators, float(x)])
                    self._sync_roi_items()
                    self.separatorsChanged.emit(list(self._separators))
                    self.separatorsEditFinished.emit()
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    # ------------------------------------------------------------------ 拖拽细节

    def _drag_move_roi(self, scene_pos: QPointF) -> None:
        drag = self._drag
        orig: QRectF = drag["orig_rect"]
        delta = scene_pos - drag["grab"]
        new_x = min(max(orig.x() + delta.x(), 0.0), self._image_w - orig.width())
        new_y = min(max(orig.y() + delta.y(), 0.0), self._image_h - orig.height())
        dx = new_x - orig.x()
        self._roi = QRectF(new_x, new_y, orig.width(), orig.height())
        self._separators = [s + dx for s in drag["orig_seps"]]
        self._sync_roi_items()
        self._emit_roi_throttled()

    def _drag_resize_roi(self, scene_pos: QPointF) -> None:
        drag = self._drag
        orig: QRectF = drag["orig_rect"]
        pos = self._clamp_to_image(scene_pos)
        handle = drag["handle"]
        left, top = orig.left(), orig.top()
        right, bottom = orig.right(), orig.bottom()
        if handle in ("tl", "l", "bl"):
            left = min(pos.x(), right - MIN_ROI_PX)
        if handle in ("tr", "r", "br"):
            right = max(pos.x(), left + MIN_ROI_PX)
        if handle in ("tl", "t", "tr"):
            top = min(pos.y(), bottom - MIN_ROI_PX)
        if handle in ("bl", "b", "br"):
            bottom = max(pos.y(), top + MIN_ROI_PX)
        left = max(0.0, left)
        top = max(0.0, top)
        right = min(float(self._image_w), right)
        bottom = min(float(self._image_h), bottom)
        self._roi = QRectF(QPointF(left, top), QPointF(right, bottom)).normalized()
        lo = self._roi.left() + 1.0
        hi = self._roi.right() - 1.0
        self._separators = [min(max(s, lo), hi) for s in drag["orig_seps"]]
        self._sync_roi_items()
        self._emit_roi_throttled()

    def _drag_separator(self, scene_pos: QPointF, snap: bool = True) -> None:
        if self._roi is None:
            return
        drag = self._drag
        index = drag["index"]
        lo = self._roi.left() + SEP_MARGIN
        hi = self._roi.right() - SEP_MARGIN
        if index > 0:
            lo = max(lo, self._separators[index - 1] + SEP_MARGIN)
        if index < len(self._separators) - 1:
            hi = min(hi, self._separators[index + 1] - SEP_MARGIN)
        if hi < lo:
            return
        x = float(scene_pos.x())
        if snap:
            x = self._snap_separator_x(x, lo, hi)
        self._separators[index] = float(min(max(x, lo), hi))
        self._sync_roi_items()
        now = time.monotonic()
        if now - self._last_emit >= EMIT_INTERVAL:
            self._last_emit = now
            self.separatorsChanged.emit(list(self._separators))

    def _emit_roi_throttled(self) -> None:
        if self._roi is None:
            return
        now = time.monotonic()
        if now - self._last_emit >= EMIT_INTERVAL:
            self._last_emit = now
            r = self._roi
            self.roiChanged.emit(r.x(), r.y(), r.width(), r.height())

    def _finish_ruler(self, drag: dict, scene_pos: QPointF) -> None:
        if self._temp_item is not None:
            self._scene.removeItem(self._temp_item)
            self._temp_item = None
        start: QPointF = drag["start"]
        dx = scene_pos.x() - start.x()
        dy = scene_pos.y() - start.y()
        if abs(dx) + abs(dy) < 4:
            return
        # 只保留最新一条标尺线，且持续显示直到旋正 / 取消 / 换工具
        self.clear_ruler()
        pen = QPen(RULER_COLOR, 2)
        pen.setCosmetic(True)
        line = QGraphicsLineItem(start.x(), start.y(), scene_pos.x(), scene_pos.y())
        line.setPen(pen)
        line.setZValue(20)
        self._scene.addItem(line)
        self._ruler_items.append(line)
        angle = float(np.degrees(np.arctan2(dy, dx)))
        self.rulerMeasured.emit(angle, scene_pos.x(), scene_pos.y())

    def show_tilt_guide(self, angle_deg: float, roi=None) -> bool:
        """按给定角度画一条横穿 ROI 的参考线，让用户**看着线**判断角度对不对。

        自动测角在干净的膜上能到 0.02°，但被一条划痕或一道膜边骗到时会用同样
        自信的语气给出错 4° 的答案，且算法无法自省（见 :mod:`wb_analyzer.tilt`
        模块文档）。所以界面绝不能只报一个数字——画出来，让人一眼看出它是不是
        平行于条带行。线足够长（横跨整个 ROI 再各伸 12 px），0.3° 的偏差在 550
        px 上就是 3 px 的漂移，肉眼分得出来。

        复用 ``_ruler_items``：它与手动标尺线互斥（``_ruler_angle`` 本来就是同
        一个位子），``clear_ruler()`` 一并清掉。用虚线 + 两端短横杠与手动拖出的
        实线区分——那条是用户自己画的，这条是机器建议的。
        """
        self.clear_ruler()
        if self._image_ref is None:
            return False
        if roi is not None:
            cx = float(roi.x) + float(roi.width) / 2.0
            cy = float(roi.y) + float(roi.height) / 2.0
            half = float(roi.width) / 2.0 + TILT_GUIDE_OVERHANG
        else:
            cx, cy = self._image_w / 2.0, self._image_h / 2.0
            half = self._image_w / 2.0
        if half <= 0.0:
            return False
        slope = float(np.tan(np.radians(float(angle_deg))))
        dx, dy = half, half * slope
        pen = QPen(RULER_COLOR, 2, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        cap_pen = QPen(RULER_COLOR, 2)
        cap_pen.setCosmetic(True)
        spine = QGraphicsLineItem(cx - dx, cy - dy, cx + dx, cy + dy)
        spine.setPen(pen)
        spine.setZValue(20)
        self._scene.addItem(spine)
        self._ruler_items.append(spine)
        # 两端的短横杠：线本身是虚的，端点在哪要能一眼看到
        for sign in (-1.0, 1.0):
            ex, ey = cx + sign * dx, cy + sign * dy
            cap = QGraphicsLineItem(ex, ey - TILT_GUIDE_CAP, ex, ey + TILT_GUIDE_CAP)
            cap.setPen(cap_pen)
            cap.setZValue(20)
            self._scene.addItem(cap)
            self._ruler_items.append(cap)
        return True

    def clear_ruler(self) -> None:
        """移除画布上的标尺参考线。"""
        for line in self._ruler_items:
            self._scene.removeItem(line)
        self._ruler_items.clear()

    def _finish_new_rect(self, kind: str, drag: dict, scene_pos: QPointF) -> None:
        if self._temp_item is not None:
            self._scene.removeItem(self._temp_item)
            self._temp_item = None
        start: QPointF = drag["start"]
        end = self._clamp_to_image(scene_pos)
        rect = QRectF(start, end).normalized()
        if rect.width() < MIN_ROI_PX or rect.height() < MIN_ROI_PX:
            return
        if kind == "new_roi":
            self._roi = rect
            self._separators = []
            self._sync_roi_items()
            self.roiChanged.emit(rect.x(), rect.y(), rect.width(), rect.height())
            self.roiEditFinished.emit()
        else:
            self.bandDrawn.emit(rect.x(), rect.y(), rect.width(), rect.height())

    def _show_separator_menu(self, index: int, global_pos) -> None:
        menu = QMenu(self)
        action = menu.addAction("删除此分隔线")
        if menu.exec(global_pos) is action and 0 <= index < len(self._separators):
            self._separators.pop(index)
            # 后面的线全部往前挪一位，旧下标会指到别的线上。等重新定量后由
            # 自检重新点名，在那之前宁可一条都不描
            self._flagged_seps.clear()
            self._sync_roi_items()
            self.separatorsChanged.emit(list(self._separators))
            self.separatorsEditFinished.emit()

    def _show_band_menu(self, band: BandBoxItem, global_pos) -> None:
        selected = self.selected_bands()
        if band.index not in selected:
            # 右键未选中的条带：先把它设为唯一选中并同步给面板
            self.select_bands([band.index])
            self.bandSelectionChanged.emit(self.selected_bands())
            selected = [band.index]
        menu = QMenu(self)
        if len(selected) > 1:
            delete_action = menu.addAction(f"删除选中的 {len(selected)} 个条带")
        else:
            delete_action = menu.addAction("删除此条带")
        lane_action = menu.addAction(f"删除 L{band.lane} 整个泳道")
        chosen = menu.exec(global_pos)
        if chosen is delete_action:
            self.bandsDeleteRequested.emit(list(selected))
        elif chosen is lane_action:
            self.bandsDeleteRequested.emit(self._indices_for_lane(band.lane))

    def _show_roi_menu(self, global_pos) -> None:
        menu = QMenu(self)
        clear_action = menu.addAction("清除 ROI")
        reseg_action = menu.addAction("按剖面峰谷重新划分")
        chosen = menu.exec(global_pos)
        if chosen is clear_action:
            self.roiClearRequested.emit()
        elif chosen is reseg_action:
            self.resegmentRequested.emit()

    def _roi_edge_at(self, vp_pos: QPointF) -> bool:
        """视口坐标是否落在 ROI 边框附近（±HIT_PX）。"""
        if self._roi is None:
            return False
        tl = QPointF(self.mapFromScene(self._roi.topLeft()))
        br = QPointF(self.mapFromScene(self._roi.bottomRight()))
        outer = QRectF(tl, br).normalized().adjusted(-HIT_PX, -HIT_PX, HIT_PX, HIT_PX)
        inner = QRectF(tl, br).normalized().adjusted(HIT_PX, HIT_PX, -HIT_PX, -HIT_PX)
        return outer.contains(vp_pos) and not inner.contains(vp_pos)

    def _cancel_drag(self) -> None:
        if self._drag is None:
            return
        drag = self._drag
        self._drag = None
        if self._temp_item is not None:
            self._scene.removeItem(self._temp_item)
            self._temp_item = None
        if drag["kind"] == "rubber":
            self._set_selection(drag["before"])
        elif drag["kind"] in ("move_roi", "resize_roi"):
            self._roi = QRectF(drag["orig_rect"])
            self._separators = list(drag["orig_seps"])
            self._sync_roi_items()
            r = self._roi
            self.roiChanged.emit(r.x(), r.y(), r.width(), r.height())
        elif drag["kind"] == "sep":
            self._separators = list(drag["orig_seps"])
            self._sync_roi_items()
            self.separatorsChanged.emit(list(self._separators))
        self._apply_tool_cursor()

    def _update_hover_cursor(self, vp_pos: QPointF, scene_pos: QPointF) -> None:
        if self._space_down:
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            return
        if self._tool in (Tool.ROI, Tool.ADJUST) and self._roi is not None:
            handle_id = self._handle_at(vp_pos)
            if handle_id is not None:
                self.viewport().setCursor(_HANDLE_CURSORS[handle_id])
                return
            if self._separator_at(vp_pos) is not None:
                self.viewport().setCursor(Qt.CursorShape.SplitHCursor)
                return
        if self._band_at(vp_pos.toPoint()) is not None:
            self.viewport().setCursor(Qt.CursorShape.PointingHandCursor)
            return
        if (
            self._tool is Tool.ADJUST
            and self._roi is not None
            and self._roi.contains(scene_pos)
        ):
            self.viewport().setCursor(Qt.CursorShape.SizeAllCursor)
            return
        if self._tool is Tool.ADJUST and self._image_ref is not None:
            # ROI 之外的空白：这里一拖就是橡皮筋框选，给个十字光标提示
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
            return
        self._apply_tool_cursor()

    # ------------------------------------------------------------------ 缩放 / 键盘 / 绘制

    def wheelEvent(self, event) -> None:
        if self._image_ref is None:
            event.ignore()
            return
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        factor = 1.25 ** (delta / 120.0)
        current = self.current_scale()
        target = min(max(current * factor, self.MIN_SCALE), self.MAX_SCALE)
        if abs(target - current) > 1e-9:
            self.scale(target / current, target / current)
            self._on_scale_changed()
        event.accept()

    _ARROW_DELTAS = {
        Qt.Key.Key_Left: (-1.0, 0.0),
        Qt.Key.Key_Right: (1.0, 0.0),
        Qt.Key.Key_Up: (0.0, -1.0),
        Qt.Key.Key_Down: (0.0, 1.0),
    }

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_down = True
            if self._drag is None:
                self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        if event.key() in self._ARROW_DELTAS and self._roi is not None:
            step = 10.0 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
            dx, dy = self._ARROW_DELTAS[event.key()]
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                handled = self.resize_roi_by(dx * step, dy * step)
            else:
                handled = self.nudge(dx * step, dy * step)
            if handled:
                event.accept()
                return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            # Delete 只删除选中的条带；不再清除 ROI（清除 ROI 走显式入口）
            selected = self.selected_bands()
            if selected:
                self.bandsDeleteRequested.emit(selected)
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape:
            if self._drag is not None:
                self._cancel_drag()
            elif self.selected_bands():
                self._scene.clearSelection()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_down = False
            if self._drag is None:
                self._apply_tool_cursor()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        if self._drag is None:
            self._apply_tool_cursor()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._image_ref is None:
            painter = QPainter(self.viewport())
            painter.setPen(EMPTY_HINT_COLOR)
            font = QFont(self.font().family())
            font.setPointSizeF(12.5)
            painter.setFont(font)
            painter.drawText(
                self.viewport().rect(), Qt.AlignmentFlag.AlignCenter, EMPTY_HINT
            )
            painter.end()
            return
        if not self._band_items:
            return
        # 标签被隐藏 / 简化时要说一声，否则用户以为标注丢了或者编号变了
        if not self._labels_packed:
            hint = LABELS_HIDDEN_HINT
        elif self._labels_compact:
            hint = LABELS_COMPACT_HINT
        else:
            return
        painter = QPainter(self.viewport())
        painter.setPen(EMPTY_HINT_COLOR)
        font = QFont(self.font().family())
        font.setPointSizeF(9.0)
        painter.setFont(font)
        painter.drawText(
            self.viewport().rect().adjusted(8, 0, -8, -6),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
            hint,
        )
        painter.end()

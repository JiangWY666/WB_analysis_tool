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

ROI_COLOR = QColor("#00e5ff")
ROI_FILL = QColor(0, 229, 255, 16)
HANDLE_BORDER = QColor("#04333b")
SEPARATOR_COLOR = QColor("#ffb300")
RULER_COLOR = QColor("#ff4081")
BAND_COLOR = QColor("#7cff6b")
RESULT_COLOR = QColor("#ff5252")
RESULT_SAT_COLOR = QColor("#ff9800")
BAND_SELECT_COLOR = QColor("#4fc3f7")
BAND_SELECT_FILL = QColor(79, 195, 247, 50)
GUIDE_COLOR = QColor(0, 255, 156, 150)
EMPTY_HINT = "打开或拖入 WB 图片（PNG / JPG / BMP / TIF）"

HANDLE_PX = 8.0
HIT_PX = 7.0
MIN_ROI_PX = 3.0
SEP_MARGIN = 2.0
EMIT_INTERVAL = 0.03


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


class BandBoxItem(QGraphicsRectItem):
    """可直接操纵的条带结果框：悬停高亮、点击/Ctrl 多选、Delete 删除。"""

    def __init__(self, index: int, lane: int, base_color: QColor, rect: QRectF) -> None:
        super().__init__(rect)
        self.index = int(index)
        self.lane = int(lane)
        self._base = QColor(base_color)
        self._hover = False
        self.setZValue(8)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self._apply_style()

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
        if self.isSelected():
            pen = QPen(BAND_SELECT_COLOR, 3)
            brush = QBrush(BAND_SELECT_FILL)
        elif self._hover:
            pen = QPen(self._base.lighter(150), 2.5)
            brush = QBrush(QColor(255, 255, 255, 18))
        else:
            pen = QPen(self._base, 1.5)
            brush = QBrush(Qt.BrushStyle.NoBrush)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(brush)


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
        self.setBackgroundBrush(QBrush(QColor("#141518")))
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

        roi_pen = QPen(ROI_COLOR, 2)
        roi_pen.setCosmetic(True)
        self._roi_item = QGraphicsRectItem()
        self._roi_item.setPen(roi_pen)
        self._roi_item.setBrush(QBrush(ROI_FILL))
        self._roi_item.setZValue(10)
        self._roi_item.setVisible(False)
        self._scene.addItem(self._roi_item)

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

        self._tool = Tool.ROI
        self._drag: dict | None = None
        self._space_down = False
        self._last_emit = 0.0
        self._temp_item: QGraphicsRectItem | QGraphicsLineItem | None = None
        self._ruler_items: list[QGraphicsLineItem] = []
        self._band_items: list[BandBoxItem] = []
        self._band_labels: list[QGraphicsSimpleTextItem] = []
        self._updating_selection = False
        self._scene.selectionChanged.connect(self._on_scene_selection_changed)
        self._apply_tool_cursor()

    # ------------------------------------------------------------------ 图像

    def set_image(self, display_bgr: np.ndarray) -> None:
        """显示 uint8 BGR 图像；scene 坐标与像素坐标一致。"""
        array = np.ascontiguousarray(display_bgr)
        if array.ndim == 2:
            array = np.ascontiguousarray(np.dstack([array] * 3))
        height, width = array.shape[:2]
        image = QImage(
            array.data, width, height, array.strides[0], QImage.Format.Format_BGR888
        )
        self._image_ref = array  # 持引用防悬垂
        self._image_w, self._image_h = width, height
        self._pixmap_item.setPixmap(QPixmap.fromImage(image))
        self._pixmap_item.setVisible(True)
        self._scene.setSceneRect(0, 0, width, height)
        self._guide_h.setLine(0, height / 2.0, width, height / 2.0)
        self._guide_v.setLine(width / 2.0, 0, width / 2.0, height)
        self._update_guides()
        self.clear_ruler()
        self.viewport().update()

    def has_image(self) -> bool:
        return self._image_ref is not None

    def is_interacting(self) -> bool:
        return self._drag is not None

    def fit_view(self) -> None:
        if self._image_ref is None:
            return
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.zoomChanged.emit(self.current_scale())

    def zoom_100(self) -> None:
        self.setTransform(QTransform())
        self.zoomChanged.emit(1.0)

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
            self.zoomChanged.emit(self.current_scale())

    def zoom_to_roi(self) -> None:
        """把视图缩放定位到当前 ROI（带少量留白）。"""
        if self._roi is None or self._image_ref is None:
            return
        margin = max(self._roi.width(), self._roi.height()) * 0.06 + 8.0
        self.fitInView(
            self._roi.adjusted(-margin, -margin, margin, margin),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
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
        if self._roi is not None:
            lo = self._roi.left() + 1.0
            hi = self._roi.right() - 1.0
            values = [min(max(v, lo), hi) for v in values]
        self._separators = values
        self._sync_roi_items()

    def separators(self) -> list[float]:
        return list(self._separators)

    def _sync_roi_items(self) -> None:
        if self._roi is None or self._roi.width() <= 0 or self._roi.height() <= 0:
            self._roi_item.setVisible(False)
            for handle in self._handles.values():
                handle.setVisible(False)
            for item in self._sep_items:
                self._scene.removeItem(item)
            self._sep_items.clear()
            return
        rect = self._roi
        self._roi_item.setRect(rect)
        self._roi_item.setVisible(True)
        for handle_id, handle in self._handles.items():
            handle.setPos(_handle_anchor(rect, handle_id))
            handle.setVisible(True)

        while len(self._sep_items) > len(self._separators):
            self._scene.removeItem(self._sep_items.pop())
        sep_pen = QPen(SEPARATOR_COLOR, 3)
        sep_pen.setCosmetic(True)
        while len(self._sep_items) < len(self._separators):
            line = QGraphicsLineItem()
            line.setPen(sep_pen)
            line.setZValue(11)
            self._scene.addItem(line)
            self._sep_items.append(line)
        for item, sep in zip(self._sep_items, self._separators):
            x = min(max(sep, rect.left()), rect.right())
            item.setLine(x, rect.top(), x, rect.bottom())

    # ------------------------------------------------------------------ 结果叠加

    def show_result_boxes(self, measurements) -> None:
        self.clear_result_boxes()
        label_font = QFont(self.font().family(), 8)
        self._updating_selection = True
        try:
            for index, m in enumerate(measurements):
                saturated = float(getattr(m, "saturated_fraction", 0.0)) >= 0.05
                color = RESULT_SAT_COLOR if saturated else RESULT_COLOR
                box = BandBoxItem(
                    index=index,
                    lane=int(m.lane),
                    base_color=color,
                    rect=QRectF(float(m.x), float(m.y), float(m.width), float(m.height)),
                )
                box.setToolTip(
                    f"L{m.lane}-B{m.band}\n"
                    f"积分灰度：{m.integrated_density:,.1f}\n"
                    f"平均灰度：{m.mean_intensity:.1f}\n"
                    f"背景：{m.background:.1f}\n"
                    f"饱和：{m.saturated_fraction * 100:.1f}%"
                    + ("\n⚠ 饱和比例偏高，定量可能失真" if saturated else "")
                    + "\n（点击选中，Delete 删除；右键更多操作）"
                )
                self._scene.addItem(box)
                self._band_items.append(box)

                label = QGraphicsSimpleTextItem(f"L{m.lane}-B{m.band}")
                label.setFlag(
                    QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True
                )
                label.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                label.setBrush(QBrush(color))
                label.setFont(label_font)
                label.setPos(float(m.x), float(m.y))
                label.setTransform(QTransform.fromTranslate(1.0, -15.0))
                label.setZValue(9)
                self._scene.addItem(label)
                self._band_labels.append(label)
        finally:
            self._updating_selection = False

    def clear_result_boxes(self) -> None:
        self._updating_selection = True
        try:
            for item in (*self._band_items, *self._band_labels):
                self._scene.removeItem(item)
            self._band_items.clear()
            self._band_labels.clear()
        finally:
            self._updating_selection = False

    # ------------------------------------------------------------------ 条带选中

    def selected_bands(self) -> list[int]:
        """当前选中条带的原始测量索引（升序）。"""
        return sorted(item.index for item in self._band_items if item.isSelected())

    def select_bands(self, indices) -> None:
        """程序化设置选中（不发 bandSelectionChanged，供表格→画布同步）。"""
        wanted = {int(i) for i in indices}
        self._updating_selection = True
        try:
            for item in self._band_items:
                item.setSelected(item.index in wanted)
        finally:
            self._updating_selection = False

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
                event.accept()
                return
            sep_index = self._separator_at(vp_pos)
            if sep_index is not None:
                self._drag = {
                    "kind": "sep",
                    "index": sep_index,
                    "orig_seps": list(self._separators),
                }
                event.accept()
                return

        # 任何工具模式下，按在条带框上优先做选中（交给场景处理 Ctrl 多选）
        if self._band_at(event.position().toPoint()) is not None:
            super().mousePressEvent(event)
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
        elif kind == "move_roi":
            self._drag_move_roi(scene_pos)
        elif kind == "resize_roi":
            self._drag_resize_roi(scene_pos)
        elif kind == "sep":
            self._drag_separator(scene_pos)
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
        elif kind in ("move_roi", "resize_roi"):
            self._apply_tool_cursor()
            if self._roi is not None:
                r = self._roi
                self.roiChanged.emit(r.x(), r.y(), r.width(), r.height())
            self.roiEditFinished.emit()
        elif kind == "sep":
            self.separatorsChanged.emit(list(self._separators))
            self.separatorsEditFinished.emit()
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

    def _drag_separator(self, scene_pos: QPointF) -> None:
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
        self._separators[index] = float(min(max(scene_pos.x(), lo), hi))
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
        if drag["kind"] in ("move_roi", "resize_roi"):
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
            self.zoomChanged.emit(self.current_scale())
        event.accept()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_down = True
            if self._drag is None:
                self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
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
            painter.setPen(QColor("#6a6f76"))
            font = QFont(self.font().family())
            font.setPointSizeF(12.5)
            painter.setFont(font)
            painter.drawText(
                self.viewport().rect(), Qt.AlignmentFlag.AlignCenter, EMPTY_HINT
            )
            painter.end()

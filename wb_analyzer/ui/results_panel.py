"""结果表面板：条带测量列表，支持排序、复制 TSV、删除选中。"""

from __future__ import annotations

from PySide6.QtCore import QItemSelectionModel, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .tablefmt import NumberDelegate, full_precision

HEADERS = ["泳道", "条带", "X", "Y", "宽", "高", "积分灰度", "背景", "平均灰度", "饱和 %", "S/N"]
SATURATED_COLOR = QColor("#ffb74d")
BELOW_DETECTION_COLOR = QColor(theme.RESULT_BOX_WEAK)
"""低于检出限的行整体压暗：数值还在，但不该被当成真实信号读。"""

# 积分灰度 / 背景 / 平均灰度：显示按千分位取整，悬停给全精度
FULL_PRECISION_COLUMNS = (6, 7, 8)
LOD_TIP = "信噪比 < 3，低于检出限：该积分与空白泳道无法区分，不要用它做比值"


def _row_values(m) -> tuple:
    """一行的列值，顺序与 :data:`HEADERS` 一致（表格与复制共用一份）。"""
    return (
        int(m.lane),
        int(m.band),
        int(m.x),
        int(m.y),
        int(m.width),
        int(m.height),
        float(m.integrated_density),
        float(m.background),
        float(m.mean_intensity),
        round(float(m.saturated_fraction) * 100.0, 2),
        round(float(m.snr), 2),
    )


def _summary(measurements: list) -> str:
    if not measurements:
        return "暂无结果"
    head = f"共 {len(measurements)} 个条带"
    saturated = sum(1 for m in measurements if float(m.saturated_fraction) >= 0.05)
    weak = sum(1 for m in measurements if m.below_detection)
    flags = []
    if saturated:
        flags.append(f"{saturated} 个饱和 ⚠")
    if weak:
        flags.append(f"{weak} 个低于检出限 ⚠")
    return f"{head}，其中 {'、'.join(flags)}" if flags else head


class ResultsPanel(QWidget):
    """行选中上报 bandSelected（原始索引）与 laneSelected（泳道号）。"""

    deleteRequested = Signal(list)
    bandSelected = Signal(int)
    laneSelected = Signal(int)
    selectionChanged = Signal(list)  # 选中行的原始索引集合（用户操作时发出）

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._measurements: list = []
        self._updating = False

        toolbar = QHBoxLayout()
        self._count_label = QLabel("暂无结果")
        copy_button = QPushButton("复制选中")
        copy_button.setToolTip("复制选中行为 TSV（Ctrl+C）")
        delete_button = QPushButton("删除选中")
        toolbar.addWidget(self._count_label, 1)
        toolbar.addWidget(copy_button)
        toolbar.addWidget(delete_button)

        self.table = QTableWidget(0, len(HEADERS))
        self.table.setHorizontalHeaderLabels(HEADERS)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setDefaultSectionSize(72)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setItemDelegate(NumberDelegate(self.table))
        self.table.itemSelectionChanged.connect(self._on_selection_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addLayout(toolbar)
        layout.addWidget(self.table)

        copy_button.clicked.connect(self.copy_selected)
        delete_button.clicked.connect(self._request_delete)
        shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.Copy), self.table)
        shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut.activated.connect(self.copy_selected)
        # 表格聚焦时 Delete = 删除选中行（不再触碰 ROI）
        delete_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), self.table)
        delete_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        delete_shortcut.activated.connect(self._request_delete)
        delete_button.setToolTip("删除选中行（表格聚焦时可直接按 Delete）")

    # ------------------------------------------------------------------ 数据

    def set_measurements(self, measurements: list) -> None:
        self._measurements = list(measurements or [])
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(self._measurements))
        for row, m in enumerate(self._measurements):
            saturated = float(m.saturated_fraction) >= 0.05
            weak = bool(m.below_detection)
            for col, value in enumerate(_row_values(m)):
                item = QTableWidgetItem()
                # EditRole 存数值：排序按数值走，显示交给 NumberDelegate
                item.setData(Qt.ItemDataRole.EditRole, value)
                item.setData(Qt.ItemDataRole.UserRole, row)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                tips = []
                if col in FULL_PRECISION_COLUMNS:
                    tips.append(f"精确值：{full_precision(value)}")
                # 饱和是「数值偏低但确实有信号」，低于检出限是「可能压根没有信号」，
                # 后者更根本，两者同时命中时按它着色。
                if weak:
                    item.setForeground(BELOW_DETECTION_COLOR)
                    tips.append(LOD_TIP)
                elif saturated:
                    item.setForeground(SATURATED_COLOR)
                if saturated:
                    tips.append("饱和比例 ≥ 5%，定量可能失真")
                if tips:
                    item.setToolTip("\n".join(tips))
                self.table.setItem(row, col, item)
        self.table.setSortingEnabled(True)
        self._count_label.setText(_summary(self._measurements))

    def measurements(self) -> list:
        return list(self._measurements)

    def _selected_original_rows(self) -> list[int]:
        rows: set[int] = set()
        for item in self.table.selectedItems():
            data = item.data(Qt.ItemDataRole.UserRole)
            if data is not None:
                rows.add(int(data))
        return sorted(rows)

    # ------------------------------------------------------------------ 操作

    def copy_selected(self) -> None:
        originals = self._selected_original_rows()
        if not originals:
            return
        lines = ["\t".join(HEADERS)]
        for index in originals:
            m = self._measurements[index]
            lines.append("\t".join(str(v) for v in _row_values(m)))
        QGuiApplication.clipboard().setText("\n".join(lines))

    def _request_delete(self) -> None:
        originals = self._selected_original_rows()
        if originals:
            self.deleteRequested.emit(originals)

    def _on_selection_changed(self) -> None:
        if self._updating:
            return
        self.selectionChanged.emit(self._selected_original_rows())
        item = self.table.currentItem()
        if item is None:
            return
        data = item.data(Qt.ItemDataRole.UserRole)
        if data is None:
            return
        index = int(data)
        if 0 <= index < len(self._measurements):
            self.bandSelected.emit(index)
            self.laneSelected.emit(int(self._measurements[index].lane))

    # ------------------------------------------------------------------ 画布 → 表格同步

    def select_rows_by_original(self, indices: list[int]) -> None:
        """按原始测量索引选中整行并滚动到首行；不回发 selectionChanged。"""
        self._updating = True
        try:
            wanted = {int(i) for i in indices}
            self.table.clearSelection()
            selection_model = self.table.selectionModel()
            model = self.table.model()
            first_row: int | None = None
            for row in range(self.table.rowCount()):
                item = self.table.item(row, 0)
                if item is None:
                    continue
                data = item.data(Qt.ItemDataRole.UserRole)
                if data is not None and int(data) in wanted:
                    selection_model.select(
                        model.index(row, 0),
                        QItemSelectionModel.SelectionFlag.Select
                        | QItemSelectionModel.SelectionFlag.Rows,
                    )
                    if first_row is None:
                        first_row = row
            if first_row is not None:
                self.table.scrollToItem(self.table.item(first_row, 0))
        finally:
            self._updating = False

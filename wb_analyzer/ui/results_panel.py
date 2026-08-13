"""结果表面板：条带测量列表，支持排序、复制 TSV、删除选中。

列的可见性是**显示**层面的事，与数据无关：表格永远建满 12 列、复制 TSV 永远
带全 12 列，只是默认把其中一半藏起来。这样"看不全"和"导不全"就永远不会变成
同一个问题——用户在设置里勾掉某列，不会悄悄改变他粘进 Excel 的东西。
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QItemSelectionModel, Qt, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .tablefmt import NumberDelegate, full_precision
from .widgets import ElidedLabel

HEADERS = [
    "泳道",
    "条带",
    "计入",
    "X",
    "Y",
    "宽",
    "高",
    "积分灰度",
    "背景",
    "平均灰度",
    "饱和 %",
    "S/N",
]

# 列的稳定标识：QSettings 里存的是这些键而不是列号，以后插一列不会让旧配置错位
COLUMN_KEYS = (
    "lane",
    "band",
    "usage",
    "x",
    "y",
    "w",
    "h",
    "density",
    "background",
    "mean",
    "saturated",
    "snr",
)
COLUMN_INDEX = {key: index for index, key in enumerate(COLUMN_KEYS)}

# 锁定列：藏了这张表就没有意义了。
#   lane / band —— 没有它们就不知道哪一行是哪一条带；
#   density     —— 这个工具的产出本身。
LOCKED_COLUMNS = ("lane", "band", "density")
# 「计入」不给开关：它只有在当前条带策略确实漏掉了某些条带时才有信息量，
# 其余时候整列都是"计入"，纯噪音。可见性由数据自己决定（见 _apply_columns）。
AUTO_COLUMNS = ("usage",)
# 可勾选的列，与默认勾上的那几个。默认留下饱和 % 与 S/N 是因为它们是"这个数
# 能不能用"的两道闸：结果表已经用颜色表达了越界与否，但贴着阈值的那些行只有
# 看到数值才判断得了。位置坐标（x/y/宽/高）只在核对框画得对不对时才看，背景与
# 平均灰度是基线扣除的中间量——都收进设置。
OPTIONAL_COLUMNS = ("saturated", "snr", "background", "mean", "x", "y", "w", "h")
DEFAULT_OPTIONAL_COLUMNS = ("saturated", "snr")

COLUMN_TIPS = {
    "usage": "当前条带策略有没有把这一条计入泳道合计（只有存在未采用的条带时才出现）",
    "density": "扣除背景后的积分灰度——这张表的产出，相对定量用的就是它",
    "background": "该条带扣掉的基线（ROI 定量是橡皮筋基线，自动检测是环形中值）",
    "mean": "积分灰度 ÷ 条带面积，受框大小影响，不能直接跨条带比较",
    "saturated": "落在量程顶部的像素占比；≥ 5% 时定量会偏低",
    "snr": "信噪比；< 3 视为低于检出限，该行整体压暗",
    "x": "条带框左上角的图像列号（核对框画得对不对时才需要）",
    "y": "条带框左上角的图像行号（核对框画得对不对时才需要）",
    "w": "条带框宽度（像素）",
    "h": "条带框高度（像素）",
}
SATURATED_COLOR = QColor("#ffb74d")
BELOW_DETECTION_COLOR = QColor(theme.RESULT_BOX_WEAK)
"""低于检出限的行整体压暗：数值还在，但不该被当成真实信号读。"""

USAGE_COLUMN = 2
DENSITY_COLUMN = 7
ADOPTED_TEXT = "计入"
UNUSED_TEXT = "未采用"
UNUSED_COLOR = QColor(theme.TEXT_FAINT)

# 积分灰度 / 背景 / 平均灰度：显示按千分位取整，悬停给全精度
FULL_PRECISION_COLUMNS = (7, 8, 9)
LOD_TIP = "信噪比 < 3，低于检出限：该积分与空白泳道无法区分，不要用它做比值"
UNUSED_TIP = (
    "当前条带策略没有采用这一条：它仍是真实测量值，只是不计入该泳道的合计"
    "（积分灰度上划了删除线的就是没被加进去的那个数）。\n"
    "这和“删除”是两回事——删除会让它从结果表和画布上彻底消失，"
    "未采用只是不参与聚合，换个策略随时会重新计入。"
)
ADOPTED_TIP = "这一条计入了该泳道的合计"


def _row_values(m, adopted: bool = True) -> tuple:
    """一行的列值，顺序与 :data:`HEADERS` 一致（表格与复制共用一份）。"""
    return (
        int(m.lane),
        int(m.band),
        ADOPTED_TEXT if adopted else UNUSED_TEXT,
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
        self._unused: set[int] = set()
        self._updating = False
        self._optional: tuple[str, ...] = tuple(DEFAULT_OPTIONAL_COLUMNS)

        toolbar = QHBoxLayout()
        self._count_label = ElidedLabel("暂无结果")
        copy_button = QPushButton("复制选中")
        copy_button.setToolTip("复制选中行为 TSV（Ctrl+C）。复制的永远是全部 12 列，与表上显示了几列无关")
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
        # 行高跟着字体走：字号调大之后固定行高会把文字上下切掉
        self.table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        header = self.table.horizontalHeader()
        header.setDefaultSectionSize(72)
        # 每列按内容取宽：默认只剩 5～6 列，按内容排完刚好放得下，横向滚动条
        # 就此消失。固定 72 px 一列时 12 列一定要 860 px，停靠区从来给不起。
        # 全局模式也设成 ResizeToContents（而不是留着默认的 Interactive）：
        # _apply_columns 会给最后一列单独换成 Stretch，但 QHeaderView 内部重排
        # 时用的是全局模式，留 Interactive 会让它保住换字号之前的旧列宽。
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(False)
        self.table.setItemDelegate(NumberDelegate(self.table))
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        for key, tip in COLUMN_TIPS.items():
            item = self.table.horizontalHeaderItem(COLUMN_INDEX[key])
            if item is not None:
                item.setToolTip(tip)
        self._apply_columns()

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

    # ------------------------------------------------------------------ 列显示

    def set_optional_columns(self, keys) -> None:
        """设置里勾上的可选列（锁定列不受影响，未知键忽略）。"""
        wanted = {str(k) for k in (keys or ())}
        self._optional = tuple(k for k in OPTIONAL_COLUMNS if k in wanted)
        self._apply_columns()

    def optional_columns(self) -> tuple[str, ...]:
        return self._optional

    def preferred_width(self) -> int:
        """完整显示当前这些列需要多宽——停靠区照着它定最小宽度。

        用户点名的问题就是"看不到所有的数值"。列砍到 5 条之后大多数情况已经放
        得下，但字号调大或积分灰度位数很多时还是会差几十个像素；与其让表格横向
        滚动，不如让面板宽这几十个像素。
        """
        header = self.table.horizontalHeader()
        total = sum(
            header.sectionSizeHint(index)
            for index in COLUMN_INDEX.values()
            if not self.table.isColumnHidden(index)
        )
        return int(total + self.table.verticalScrollBar().sizeHint().width() + 8)

    def visible_columns(self) -> list[str]:
        """当前真正显示出来的列（含自动列），按表上的顺序。"""
        return [
            key
            for key in COLUMN_KEYS
            if not self.table.isColumnHidden(COLUMN_INDEX[key])
        ]

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().changeEvent(event)
        # 字号一变，每列该多宽就变了，但 QHeaderView 不会自己按新字体重排——
        # 不推它一把，14 pt 下会拿 9 pt 时算出的列宽继续用，于是又冒出横向滚动条
        if event.type() == QEvent.Type.FontChange:
            self._apply_columns()

    def _apply_columns(self) -> None:
        shown = set(LOCKED_COLUMNS) | set(self._optional)
        # 「计入」只在真有条带没被采用时才占一列——全是"计入"的一列没有信息量
        if self._unused:
            shown.add("usage")
        header = self.table.horizontalHeader()
        last_visible = None
        for key, index in COLUMN_INDEX.items():
            hidden = key not in shown
            self.table.setColumnHidden(index, hidden)
            if not hidden:
                last_visible = index
        # 最后一列改成 Stretch：ResizeToContents 的列不肯缩到内容宽度以下，行多
        # 到冒出竖直滚动条时视口窄了那十几个像素，就会为此长出一条横向滚动条。
        # Stretch 的列拿"剩下多少给多少"，总宽永远等于视口。
        for key, index in COLUMN_INDEX.items():
            if self.table.isColumnHidden(index):
                continue
            header.setSectionResizeMode(
                index,
                QHeaderView.ResizeMode.Stretch
                if index == last_visible
                else QHeaderView.ResizeMode.ResizeToContents,
            )

    # ------------------------------------------------------------------ 数据

    def set_measurements(self, measurements: list, unused: list | None = None) -> None:
        self._measurements = list(measurements or [])
        self._unused = {int(i) for i in (unused or [])}
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(self._measurements))
        for row, m in enumerate(self._measurements):
            saturated = float(m.saturated_fraction) >= 0.05
            weak = bool(m.below_detection)
            for col, value in enumerate(_row_values(m, row not in self._unused)):
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
                # 后者更根本，两者同时命中时按它着色。“计入”列不参与这套配色：
                # 它只表达“采没采用”，混进强弱配色就分不清是哪一件事了。
                if col == USAGE_COLUMN:
                    pass
                elif weak:
                    item.setForeground(BELOW_DETECTION_COLOR)
                    tips.append(LOD_TIP)
                elif saturated:
                    item.setForeground(SATURATED_COLOR)
                if saturated and col != USAGE_COLUMN:
                    tips.append("饱和比例 ≥ 5%，定量可能失真")
                if tips:
                    item.setToolTip("\n".join(tips))
                self.table.setItem(row, col, item)
            self._apply_usage(row, row)
        self.table.setSortingEnabled(True)
        self._apply_columns()
        self._refresh_summary()

    def set_band_usage(self, unused: list | None) -> None:
        """更新“哪些条带不计入合计”，不重建表格（排序与选中都保持原样）。

        未采用 ≠ 已删除：这里改的只是聚合口径，行还在、数字还在、画布上的框也
        还在，只是描边换成虚线、积分灰度划掉。删除才会让它整个消失。

        策略未生效之类的文字提示不在这里——它们和 ROI 自检一起挂在画布上方的
        统一诊断条上（见 diagnostics_bar），免得同一件事在两个面板各说一遍。
        """
        self._unused = {int(i) for i in (unused or [])}
        for row in range(self.table.rowCount()):
            anchor = self.table.item(row, 0)
            if anchor is None:
                continue
            original = anchor.data(Qt.ItemDataRole.UserRole)
            if original is not None:
                self._apply_usage(row, int(original))
        self._apply_columns()
        self._refresh_summary()

    def unused_bands(self) -> list[int]:
        return sorted(self._unused)

    def _apply_usage(self, row: int, original: int) -> None:
        adopted = original not in self._unused
        for col in range(self.table.columnCount()):
            item = self.table.item(row, col)
            if item is None:
                continue
            font = QFont(self.table.font())
            # 斜体是这张表里没被别的状态占用的通道（颜色归饱和 / 检出限），
            # 删除线正好划在“本该被加进合计、现在没加”的那个数上
            font.setItalic(not adopted)
            font.setStrikeOut(not adopted and col == DENSITY_COLUMN)
            item.setFont(font)
        usage = self.table.item(row, USAGE_COLUMN)
        if usage is None:
            return
        usage.setData(
            Qt.ItemDataRole.EditRole, ADOPTED_TEXT if adopted else UNUSED_TEXT
        )
        usage.setForeground(QColor(theme.TEXT_DIM) if adopted else UNUSED_COLOR)
        usage.setToolTip(ADOPTED_TIP if adopted else UNUSED_TIP)

    def _refresh_summary(self) -> None:
        text = _summary(self._measurements)
        if self._measurements and self._unused:
            text += f"，{len(self._unused)} 个未采用（仍显示，不计入合计）"
        self._count_label.setText(text)

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
            values = _row_values(m, index not in self._unused)
            lines.append("\t".join(str(v) for v in values))
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

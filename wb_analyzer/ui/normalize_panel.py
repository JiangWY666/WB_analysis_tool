"""归一化面板：录入内参/目标、历史管理（可直接编辑）、归一化结果与 CSV 导入导出。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..normalize import (
    BAND_PICK_ALL,
    BAND_PICK_LABELS,
    ORIGIN_UNKNOWN,
    band_pick_label,
    coerce_band_pick,
    origin_label,
)
from . import theme
from .tablefmt import NumberDelegate, full_precision
from .widgets import ElidedLabel

HISTORY_HEADERS = ["类型", "名称", "分组", "泳道数", "口径", "条带", "来源文件"]
HISTORY_KIND_COLUMN = 0
HISTORY_PICK_COLUMN = 5
ORIGIN_TIP = (
    "这组数字是按哪套背景口径测出来的：\n"
    "· ROI 基线 —— ROI 定量（ImageJ 式橡皮筋基线）\n"
    "· 局部背景 —— 自动检测（环形中值局部背景）\n"
    "两套口径不可互相比较，同一分组里混用时归一化会拒绝配对。"
)
BAND_PICK_TIP = (
    "一个泳道里有上下两条带时，这条记录取哪一条：\n"
    "· 合并全部 —— 两条加起来\n"
    "· 只取上面那条 / 只取下面那条 —— 按图像位置取，不按谁更亮\n"
    "双击可改；改的是聚合方式，不需要重新定量。\n"
    "从 CSV 导入的记录只有合计值、没有逐条带明细，改不了。"
)
RESULT_HEADERS = ["分组", "泳道", "目标值", "内参值", "归一化", "Fold Change", "备注"]
RESULT_NOTE_COLUMN = 6

# 跨口径的行：比值被核心层拒绝计算了，这不是"仅供参考"，是"这行不能用"
CROSS_ORIGIN_BG = QColor(72, 30, 32)
CROSS_ORIGIN_FG = QColor("#ff8a80")
NOTE_FG = QColor(theme.WARN)

KIND_LC_LABEL = "内参 LC"
KIND_TARGET_LABEL = "目标蛋白"
DEFAULT_RESULT_HINT = "尚无归一化结果。加入同组的内参与目标后自动计算。"


def kind_to_label(kind: str) -> str:
    return KIND_LC_LABEL if kind == "loading_control" else KIND_TARGET_LABEL


def label_to_kind(label: str) -> str:
    return "loading_control" if "内参" in label or "LC" in label.upper() else "target"


def label_to_band_pick(label: str) -> str:
    """下拉里的中文回到核心层的取值；认不出就按“合并全部”。"""
    for value, text in BAND_PICK_LABELS.items():
        if text == label:
            return value
    return BAND_PICK_ALL


def is_cross_origin(row) -> bool:
    """两侧背景口径都已知且不同——核心层会拒绝给出比值。"""
    known = {
        getattr(row, "loading_control_origin", ORIGIN_UNKNOWN),
        getattr(row, "target_origin", ORIGIN_UNKNOWN),
    } - {ORIGIN_UNKNOWN}
    return len(known) > 1


class _ComboDelegate(QStyledItemDelegate):
    """双击弹下拉的单元格：候选项固定，选完把文案写回模型。"""

    labels: list[str] = []

    def createEditor(self, parent, option, index):  # noqa: N802 - Qt API
        combo = QComboBox(parent)
        combo.addItems(self.labels)
        return combo

    def setEditorData(self, editor, index):  # noqa: N802 - Qt API
        current = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        editor.setCurrentIndex(self.labels.index(current) if current in self.labels else 0)

    def setModelData(self, editor, model, index):  # noqa: N802 - Qt API
        model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)


class KindDelegate(_ComboDelegate):
    """历史表“类型”列：只能选内参 / 目标蛋白。"""

    labels = [KIND_LC_LABEL, KIND_TARGET_LABEL]

    def setEditorData(self, editor, index):  # noqa: N802 - Qt API
        current = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        editor.setCurrentIndex(1 if label_to_kind(current) == "target" else 0)


class BandPickDelegate(_ComboDelegate):
    """历史表“条带”列：合并全部 / 只取上面那条 / 只取下面那条。"""

    labels = list(BAND_PICK_LABELS.values())


def _make_table(headers: list[str]) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    table.setAlternatingRowColors(True)
    table.verticalHeader().setVisible(False)
    # 按内容取列宽 + 行高跟字体走：固定列宽在 400 px 的停靠区里必然要横向滚动，
    # 而这两张表的列大多只有两三个字
    table.verticalHeader().setSectionResizeMode(
        QHeaderView.ResizeMode.ResizeToContents
    )
    table.horizontalHeader().setSectionResizeMode(
        QHeaderView.ResizeMode.ResizeToContents
    )
    table.horizontalHeader().setStretchLastSection(True)
    return table


def _numeric_item(value: float | int | None, tooltip: str = "") -> QTableWidgetItem:
    item = QTableWidgetItem()
    if value is None:
        item.setText("—")
    else:
        item.setData(Qt.ItemDataRole.EditRole, value)
    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    tips = [tooltip] if tooltip else []
    if isinstance(value, float):
        tips.append(f"精确值：{full_precision(value)}")
    if tips:
        item.setToolTip("\n".join(tips))
    return item


def _short_note(note: str) -> str:
    """备注列里的短判词：完整原因在下方警告栏与 tooltip 里，单元格只放结论。

    "背景口径不一致（内参 ROI 基线 / 目标 局部背景）" 塞进一列必然被截断成
    "背景口…"，那还不如只写前半句——判词要一眼看全，细节另有地方看。
    """
    text = str(note or "")
    if len(text) <= 14:
        return text
    head = text.split("（", 1)[0].strip()
    if 4 <= len(head) <= 14:
        return head
    return text[:13] + "…"


def _side_tip(role: str, name: str, row, prefix: str) -> str:
    """结果表里某一侧的溯源提示：口径 + 条带策略 + 这个泳道实际采用的条带。"""
    origin = origin_label(getattr(row, f"{prefix}_origin", ORIGIN_UNKNOWN))
    pick = band_pick_label(getattr(row, f"{prefix}_band_pick", BAND_PICK_ALL))
    bands = tuple(getattr(row, f"{prefix}_bands", ()) or ())
    used = "、".join(f"B{b}" for b in bands) if bands else "—"
    return f"{role}：{name}\n口径：{origin}\n条带策略：{pick}\n本泳道采用：{used}"


def _text_item(text: str, tooltip: str = "", editable: bool = False) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    if tooltip:
        item.setToolTip(tooltip)
    if not editable:
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return item


class NormalizePanel(QWidget):
    addRequested = Signal()
    deleteHistoryRequested = Signal(list)
    historyEdited = Signal(int, str, str, str)  # row, kind, name, group
    historyBandPickChanged = Signal(int, str)  # row, band_pick
    bandPickChanged = Signal(str)  # 录入区的条带策略
    exportLcRequested = Signal()
    importLcRequested = Signal()
    exportNormalizedRequested = Signal()
    controlLaneChanged = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._updating = False

        entry_box = QGroupBox("定量录入")
        kind_row = QHBoxLayout()
        self._lc_radio = QRadioButton(KIND_LC_LABEL)
        self._target_radio = QRadioButton(KIND_TARGET_LABEL)
        self._lc_radio.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self._lc_radio)
        group.addButton(self._target_radio)
        kind_row.addWidget(self._lc_radio)
        kind_row.addWidget(self._target_radio)
        kind_row.addStretch(1)

        self._protein_edit = QLineEdit("Actin")
        self._protein_edit.setPlaceholderText("如 Actin / GAPDH / PAD4")
        self._group_edit = QLineEdit("默认")
        self._group_edit.setPlaceholderText("同组目标与内参会自动配对")

        # 双条带按位置取，不按强弱：想要的那个修饰态分子量不变，亮度会变
        self._pick_combo = QComboBox()
        for value, label in BAND_PICK_LABELS.items():
            self._pick_combo.addItem(label, value)
        self._pick_combo.setToolTip(BAND_PICK_TIP)
        self._pick_combo.currentIndexChanged.connect(self._on_pick_combo_changed)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.addRow("类型", kind_row)
        form.addRow("蛋白名", self._protein_edit)
        form.addRow("分组", self._group_edit)
        form.addRow("条带", self._pick_combo)
        add_button = QPushButton("加入历史")
        add_button.setToolTip("把当前定量结果按上面的类型 / 蛋白名 / 分组存入历史（输入框内按回车同样生效）")
        add_button.clicked.connect(self.addRequested)
        # 非对话框里 setDefault 无效，显式接回车
        self._protein_edit.returnPressed.connect(self.addRequested)
        self._group_edit.returnPressed.connect(self.addRequested)
        entry_layout = QVBoxLayout(entry_box)
        entry_layout.addLayout(form)
        entry_layout.addWidget(add_button)

        history_box = QGroupBox("历史记录")
        history_box.setToolTip(
            "每一行是一次定量的结果。类型 / 名称 / 分组 / 条带这四列双击就能改，"
            "改完只重新聚合，不重新定量。"
        )
        self.history_table = _make_table(HISTORY_HEADERS)
        self.history_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.history_table.setItemDelegateForColumn(
            HISTORY_KIND_COLUMN, KindDelegate(self.history_table)
        )
        self.history_table.setItemDelegateForColumn(
            HISTORY_PICK_COLUMN, BandPickDelegate(self.history_table)
        )
        # "只取上面那条"被截成"只取上面…"就分不出上下了，这列按内容给宽度
        self.history_table.horizontalHeader().setSectionResizeMode(
            HISTORY_PICK_COLUMN, QHeaderView.ResizeMode.ResizeToContents
        )
        self.history_table.itemChanged.connect(self._on_history_item_changed)
        history_buttons = QHBoxLayout()
        delete_button = QPushButton("删除选中")
        delete_button.setToolTip("删除选中的历史记录（表格聚焦时可直接按 Delete，可撤销）")
        export_lc_button = QPushButton("导出内参 CSV")
        import_lc_button = QPushButton("导入内参 CSV")
        delete_button.clicked.connect(self._request_delete)
        export_lc_button.clicked.connect(self.exportLcRequested)
        import_lc_button.clicked.connect(self.importLcRequested)
        history_buttons.addWidget(delete_button)
        history_buttons.addStretch(1)
        history_buttons.addWidget(export_lc_button)
        history_buttons.addWidget(import_lc_button)
        history_layout = QVBoxLayout(history_box)
        history_layout.addWidget(self.history_table)
        history_layout.addLayout(history_buttons)
        delete_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), self.history_table)
        delete_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        delete_shortcut.activated.connect(self._request_delete)

        result_box = QGroupBox("归一化结果（同组目标 ÷ 内参）")
        self._control_spin = QSpinBox()
        self._control_spin.setRange(1, 24)
        self._control_spin.setValue(1)
        self._control_spin.setToolTip("fold change 的分母泳道（该泳道 fold change = 1）")
        self._control_spin.valueChanged.connect(self.controlLaneChanged)
        control_row = QHBoxLayout()
        control_row.addWidget(QLabel("对照泳道"))
        control_row.addWidget(self._control_spin)
        control_row.addStretch(1)
        self.result_table = _make_table(RESULT_HEADERS)
        self.result_table.setItemDelegate(NumberDelegate(self.result_table))
        # 备注是"这行能不能用"的判词，宁可让表格横向滚动也不能把它截掉
        result_header = self.result_table.horizontalHeader()
        result_header.setStretchLastSection(False)
        result_header.setSectionResizeMode(
            RESULT_NOTE_COLUMN, QHeaderView.ResizeMode.ResizeToContents
        )
        export_norm_button = QPushButton("导出归一化 CSV")
        export_norm_button.clicked.connect(self.exportNormalizedRequested)
        # 单行省略而不是自动折行：这条提示可能把每条历史记录的警告都串进来，
        # wordWrap 之后它会长成一段话、把下面的结果表挤没
        self._result_hint = ElidedLabel(DEFAULT_RESULT_HINT)
        result_layout = QVBoxLayout(result_box)
        result_layout.addLayout(control_row)
        result_layout.addWidget(self._result_hint)
        result_layout.addWidget(self.result_table)
        result_layout.addWidget(export_norm_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(entry_box)
        layout.addWidget(history_box, 1)
        layout.addWidget(result_box, 2)

    # ------------------------------------------------------------------ 取值 / 设值

    def kind(self) -> str:
        return "loading_control" if self._lc_radio.isChecked() else "target"

    def set_kind(self, kind: str) -> None:
        if kind == "target":
            self._target_radio.setChecked(True)
        else:
            self._lc_radio.setChecked(True)

    def protein_name(self) -> str:
        return self._protein_edit.text().strip()

    def set_protein_name(self, name: str) -> None:
        self._protein_edit.setText(name)

    def group_name(self) -> str:
        return self._group_edit.text().strip() or "默认"

    def set_group_name(self, name: str) -> None:
        self._group_edit.setText(name)

    def control_lane(self) -> int:
        return int(self._control_spin.value())

    def set_control_lane(self, lane: int) -> None:
        self._control_spin.setValue(int(lane))

    def band_pick(self) -> str:
        return str(self._pick_combo.currentData() or BAND_PICK_ALL)

    def set_band_pick(self, pick: str) -> None:
        index = self._pick_combo.findData(coerce_band_pick(pick, strict=False))
        if index >= 0:
            self._pick_combo.setCurrentIndex(index)

    def _on_pick_combo_changed(self, _index: int) -> None:
        if not self._updating:
            self.bandPickChanged.emit(self.band_pick())

    # ------------------------------------------------------------------ 表格刷新

    def set_history(
        self,
        entries: list,
        origins: list | None = None,
        pick_tips: list | None = None,
    ) -> None:
        """origins 与 entries 一一对应，是每条记录的背景口径标签（可缺省）。

        pick_tips 是“条带”列的额外提示（例如某些泳道只检出一条带、策略在那里
        不生效），追加在通用说明后面。
        """
        origins = list(origins or [])
        pick_tips = list(pick_tips or [])
        self._updating = True
        try:
            self.history_table.setRowCount(len(entries))
            for row, entry in enumerate(entries):
                source = Path(entry.source).name if entry.source else ""
                origin = origins[row] if row < len(origins) else "—"
                extra = pick_tips[row] if row < len(pick_tips) else ""
                pick_tip = BAND_PICK_TIP + (f"\n⚠ {extra}" if extra else "")
                self.history_table.setItem(
                    row, 0, _text_item(kind_to_label(entry.kind), "双击修改类型", editable=True)
                )
                self.history_table.setItem(
                    row, 1, _text_item(entry.name, "双击修改名称", editable=True)
                )
                self.history_table.setItem(
                    row, 2, _text_item(entry.group, "双击修改分组", editable=True)
                )
                self.history_table.setItem(row, 3, _numeric_item(int(entry.lane_count)))
                self.history_table.setItem(row, 4, _text_item(origin, ORIGIN_TIP))
                pick_item = _text_item(
                    band_pick_label(getattr(entry, "band_pick", BAND_PICK_ALL)),
                    pick_tip,
                    editable=True,
                )
                if extra:
                    pick_item.setForeground(NOTE_FG)
                self.history_table.setItem(row, HISTORY_PICK_COLUMN, pick_item)
                self.history_table.setItem(row, 6, _text_item(source, entry.source))
        finally:
            self._updating = False

    def set_rows(self, rows: list) -> None:
        """填归一化结果表。

        按泳道号配对后，只在一侧出现的泳道也会成行，其目标 / 内参值为
        ``None``——那是"这个泳道没配上"，得显示成“—”并带上原因，不能当成
        数字去 float()。

        跨背景口径的行整行标红：核心层直接拒绝给出比值，这是"这行不能用"，
        不是"仅供参考"。只挂 tooltip 等于没说——得悬停才看得见。
        """
        self.result_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            note = getattr(row, "note", "") or ""
            cross = is_cross_origin(row)
            # 标记放最左列：面板窄的时候右边的备注列要横向滚动才看得到，而
            # "这行不能用"必须在第一眼里，不能只靠颜色（也照顾色觉障碍）
            self.result_table.setItem(
                row_index, 0, _text_item(("⛔ " + row.group) if cross else row.group, note)
            )
            self.result_table.setItem(row_index, 1, _numeric_item(int(row.lane), note))
            self.result_table.setItem(
                row_index,
                2,
                _numeric_item(
                    None if row.target is None else float(row.target),
                    note or _side_tip("目标", row.target_name, row, "target"),
                ),
            )
            self.result_table.setItem(
                row_index,
                3,
                _numeric_item(
                    None if row.loading_control is None else float(row.loading_control),
                    note or _side_tip("内参", row.loading_control_name, row, "loading_control"),
                ),
            )
            normalized = None if row.normalized is None else round(float(row.normalized), 4)
            self.result_table.setItem(row_index, 4, _numeric_item(normalized, note))
            fold = None if row.fold_change is None else round(float(row.fold_change), 4)
            self.result_table.setItem(row_index, 5, _numeric_item(fold, note))
            note_item = _text_item(_short_note(note), note)
            self.result_table.setItem(row_index, RESULT_NOTE_COLUMN, note_item)
            self._paint_row(row_index, cross, bool(note))
        # 可见性统一交给 set_diagnosis：有结果但有警告时提示也得露出来

    def _paint_row(self, row_index: int, cross_origin: bool, has_note: bool) -> None:
        for column in range(self.result_table.columnCount()):
            item = self.result_table.item(row_index, column)
            if item is None:
                continue
            if cross_origin:
                item.setBackground(CROSS_ORIGIN_BG)
                item.setForeground(CROSS_ORIGIN_FG)
            elif has_note and column == RESULT_NOTE_COLUMN:
                item.setForeground(NOTE_FG)

    def cross_origin_rows(self) -> list[int]:
        """整行标红的行号，供测试与自检使用。"""
        return [
            row
            for row in range(self.result_table.rowCount())
            if (item := self.result_table.item(row, RESULT_NOTE_COLUMN)) is not None
            and item.background().color() == CROSS_ORIGIN_BG
        ]

    def set_diagnosis(self, text: str | None) -> None:
        """结果下方的提示：没结果时说原因，有结果但有泳道没配上时给警告。

        ``ElidedLabel`` 自己把全文挂进 tooltip，界面上只占一行。
        """
        self._result_hint.setText(text or DEFAULT_RESULT_HINT)
        self._result_hint.setVisible(bool(text) or self.result_table.rowCount() == 0)

    # ------------------------------------------------------------------ 编辑 / 删除

    def _on_history_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating:
            return
        if item.column() == HISTORY_PICK_COLUMN:
            self.historyBandPickChanged.emit(
                item.row(), label_to_band_pick(item.text())
            )
            return
        if item.column() > 2:
            return
        row = item.row()
        kind_item = self.history_table.item(row, 0)
        name_item = self.history_table.item(row, 1)
        group_item = self.history_table.item(row, 2)
        if kind_item is None or name_item is None or group_item is None:
            return
        self.historyEdited.emit(
            row,
            label_to_kind(kind_item.text()),
            name_item.text(),
            group_item.text(),
        )

    def _request_delete(self) -> None:
        rows = sorted(
            {index.row() for index in self.history_table.selectionModel().selectedRows()}
        )
        if rows:
            self.deleteHistoryRequested.emit(rows)

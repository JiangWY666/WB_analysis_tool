"""归一化面板：录入内参/目标、历史管理（可直接编辑）、归一化结果与 CSV 导入导出。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
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

from .tablefmt import NumberDelegate, full_precision

HISTORY_HEADERS = ["类型", "名称", "分组", "泳道数", "来源文件"]
RESULT_HEADERS = ["分组", "泳道", "目标值", "内参值", "归一化", "Fold Change"]

KIND_LC_LABEL = "内参 LC"
KIND_TARGET_LABEL = "目标蛋白"
DEFAULT_RESULT_HINT = "尚无归一化结果。加入同组的内参与目标后自动计算。"


def kind_to_label(kind: str) -> str:
    return KIND_LC_LABEL if kind == "loading_control" else KIND_TARGET_LABEL


def label_to_kind(label: str) -> str:
    return "loading_control" if "内参" in label or "LC" in label.upper() else "target"


class KindDelegate(QStyledItemDelegate):
    """历史表“类型”列：双击弹下拉，只能选内参 / 目标蛋白。"""

    def createEditor(self, parent, option, index):  # noqa: N802 - Qt API
        combo = QComboBox(parent)
        combo.addItems([KIND_LC_LABEL, KIND_TARGET_LABEL])
        return combo

    def setEditorData(self, editor, index):  # noqa: N802 - Qt API
        current = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        editor.setCurrentIndex(1 if label_to_kind(current) == "target" else 0)

    def setModelData(self, editor, model, index):  # noqa: N802 - Qt API
        model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)


def _make_table(headers: list[str]) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    table.setAlternatingRowColors(True)
    table.verticalHeader().setVisible(False)
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

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.addRow("类型", kind_row)
        form.addRow("蛋白名", self._protein_edit)
        form.addRow("分组", self._group_edit)
        add_button = QPushButton("加入历史")
        add_button.setToolTip("把当前定量结果按上面的类型 / 蛋白名 / 分组存入历史（输入框内按回车同样生效）")
        add_button.clicked.connect(self.addRequested)
        # 非对话框里 setDefault 无效，显式接回车
        self._protein_edit.returnPressed.connect(self.addRequested)
        self._group_edit.returnPressed.connect(self.addRequested)
        entry_layout = QVBoxLayout(entry_box)
        entry_layout.addLayout(form)
        entry_layout.addWidget(add_button)

        history_box = QGroupBox("历史记录（类型 / 名称 / 分组可双击修改）")
        self.history_table = _make_table(HISTORY_HEADERS)
        self.history_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.history_table.setItemDelegateForColumn(0, KindDelegate(self.history_table))
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
        export_norm_button = QPushButton("导出归一化 CSV")
        export_norm_button.clicked.connect(self.exportNormalizedRequested)
        self._result_hint = QLabel(DEFAULT_RESULT_HINT)
        self._result_hint.setWordWrap(True)
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

    # ------------------------------------------------------------------ 表格刷新

    def set_history(self, entries: list) -> None:
        self._updating = True
        try:
            self.history_table.setRowCount(len(entries))
            for row, entry in enumerate(entries):
                source = Path(entry.source).name if entry.source else ""
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
                self.history_table.setItem(row, 4, _text_item(source, entry.source))
        finally:
            self._updating = False

    def set_rows(self, rows: list) -> None:
        """填归一化结果表。

        按泳道号配对后，只在一侧出现的泳道也会成行，其目标 / 内参值为
        ``None``——那是"这个泳道没配上"，得显示成“—”并带上原因，不能当成
        数字去 float()。
        """
        self.result_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            note = getattr(row, "note", "") or ""
            self.result_table.setItem(row_index, 0, _text_item(row.group, note))
            self.result_table.setItem(row_index, 1, _numeric_item(int(row.lane), note))
            self.result_table.setItem(
                row_index,
                2,
                _numeric_item(
                    None if row.target is None else float(row.target),
                    note or f"目标：{row.target_name}",
                ),
            )
            self.result_table.setItem(
                row_index,
                3,
                _numeric_item(
                    None if row.loading_control is None else float(row.loading_control),
                    note or f"内参：{row.loading_control_name}",
                ),
            )
            normalized = None if row.normalized is None else round(float(row.normalized), 4)
            self.result_table.setItem(row_index, 4, _numeric_item(normalized, note))
            fold = None if row.fold_change is None else round(float(row.fold_change), 4)
            self.result_table.setItem(row_index, 5, _numeric_item(fold, note))
        # 可见性统一交给 set_diagnosis：有结果但有警告时提示也得露出来

    def set_diagnosis(self, text: str | None) -> None:
        """结果下方的提示：没结果时说原因，有结果但有泳道没配上时给警告。"""
        self._result_hint.setText(text or DEFAULT_RESULT_HINT)
        self._result_hint.setVisible(bool(text) or self.result_table.rowCount() == 0)

    # ------------------------------------------------------------------ 编辑 / 删除

    def _on_history_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating or item.column() > 2:
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

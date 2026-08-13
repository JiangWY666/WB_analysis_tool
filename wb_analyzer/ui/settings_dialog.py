"""界面设置对话框：字体大小、画布标注线宽、结果表列显示。

改动**即时生效**，没有"确定"按钮：调字号和线宽这种事只能看着结果调，先关对
话框再看效果等于要来回开五次。代价是没有"取消"，所以留了「恢复默认」——它一
键退回出厂值，比逐项撤回更有用。
"""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QVBoxLayout,
)

from . import prefs as prefs_mod
from .results_panel import COLUMN_INDEX, HEADERS, OPTIONAL_COLUMNS
from .widgets import ElidedLabel

FONT_TIP = (
    "界面字体的磅值，表格、面板、菜单与诊断条一起跟着变；\n"
    "画布上的条带标签与剖面图坐标轴按同一个倍数缩放，不会各走各的。"
)
LINE_TIP = (
    "相对默认线宽的倍数，不是绝对像素。\n"
    "画布上有几处区分是靠“更粗”表达的（被自检点名的分隔线、选中的条带框），"
    "用倍数才能在你把基础线宽调大之后仍然分得出来。"
)
COLUMN_TIP = (
    "结果表默认只显示泳道 / 条带 / 积分灰度 / 饱和 % / S/N，其余在这里勾。\n"
    "无论勾了几列，「复制选中」导出的 TSV 永远是全部 12 列。"
)
LOCKED_TIP = "泳道、条带、积分灰度是这张表的骨架，不提供隐藏"


class SettingsDialog(QDialog):
    """非模态；每次改动都发 :attr:`changed`，由主窗口负责应用与持久化。"""

    changed = Signal(object)  # prefs.UiPrefs

    def __init__(self, value: prefs_mod.UiPrefs, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("界面设置")
        self.setModal(False)
        self._value = value.sanitized()
        self._loading = False

        self._font_spin = QDoubleSpinBox()
        self._font_spin.setRange(prefs_mod.MIN_FONT_PT, prefs_mod.MAX_FONT_PT)
        self._font_spin.setSingleStep(prefs_mod.FONT_STEP)
        self._font_spin.setDecimals(1)
        self._font_spin.setSuffix(" pt")
        self._font_spin.setToolTip(FONT_TIP)
        self._font_spin.valueChanged.connect(self._emit)
        font_box = QGroupBox("界面")
        font_form = QFormLayout(font_box)
        font_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        font_form.addRow("字体大小", self._font_spin)

        line_box = QGroupBox("画布标注线宽")
        line_box.setToolTip(LINE_TIP)
        line_form = QFormLayout(line_box)
        line_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._line_spins = {}
        for field, label in (
            ("roi_line", "ROI 框"),
            ("band_line", "条带框"),
            ("separator_line", "泳道分隔线"),
        ):
            spin = QDoubleSpinBox()
            spin.setRange(prefs_mod.MIN_LINE_SCALE, prefs_mod.MAX_LINE_SCALE)
            spin.setSingleStep(prefs_mod.LINE_STEP)
            spin.setDecimals(2)
            spin.setSuffix(" ×")
            spin.setToolTip(LINE_TIP)
            spin.valueChanged.connect(self._emit)
            self._line_spins[field] = spin
            line_form.addRow(label, spin)

        column_box = QGroupBox("结果表列")
        column_box.setToolTip(COLUMN_TIP)
        column_layout = QVBoxLayout(column_box)
        locked = ElidedLabel(
            "始终显示：" + "、".join(HEADERS[COLUMN_INDEX[k]] for k in ("lane", "band", "density"))
        )
        locked.setObjectName("dimHint")
        locked.setToolTip(LOCKED_TIP)
        column_layout.addWidget(locked)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        self._column_checks = {}
        for position, key in enumerate(OPTIONAL_COLUMNS):
            check = QCheckBox(HEADERS[COLUMN_INDEX[key]])
            check.toggled.connect(self._emit)
            self._column_checks[key] = check
            grid.addWidget(check, position // 2, position % 2)
        column_layout.addLayout(grid)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.RestoreDefaults
            | QDialogButtonBox.StandardButton.Close
        )
        self.restore_button = buttons.button(
            QDialogButtonBox.StandardButton.RestoreDefaults
        )
        self.restore_button.setText("恢复默认")
        self.restore_button.clicked.connect(self.restore_defaults)
        buttons.button(QDialogButtonBox.StandardButton.Close).setText("关闭")
        buttons.rejected.connect(self.close)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.addWidget(font_box)
        layout.addWidget(line_box)
        layout.addWidget(column_box)
        layout.addStretch(1)
        layout.addWidget(buttons)
        self.set_value(self._value)

    # ------------------------------------------------------------------ 取值 / 设值

    def value(self) -> prefs_mod.UiPrefs:
        return self._value

    def set_value(self, value: prefs_mod.UiPrefs) -> None:
        self._loading = True
        try:
            self._value = value.sanitized()
            self._font_spin.setValue(self._value.font_pt)
            for field, spin in self._line_spins.items():
                spin.setValue(getattr(self._value, field))
            selected = set(self._value.result_columns)
            for key, check in self._column_checks.items():
                check.setChecked(key in selected)
        finally:
            self._loading = False

    def restore_defaults(self) -> None:
        self.set_value(prefs_mod.DEFAULTS)
        self._value = prefs_mod.DEFAULTS
        self.changed.emit(self._value)

    def _emit(self, *_args) -> None:
        if self._loading:
            return
        self._value = replace(
            self._value,
            font_pt=float(self._font_spin.value()),
            roi_line=float(self._line_spins["roi_line"].value()),
            band_line=float(self._line_spins["band_line"].value()),
            separator_line=float(self._line_spins["separator_line"].value()),
            result_columns=tuple(
                key
                for key in OPTIONAL_COLUMNS
                if self._column_checks[key].isChecked()
            ),
        ).sanitized()
        self.changed.emit(self._value)


__all__ = ["SettingsDialog"]

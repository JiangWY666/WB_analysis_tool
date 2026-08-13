"""导出数据的选项对话框。

「导出数据」不再默认把归一化 / fold change 当成必选项。有的膜同一份样品
重复上样，软件替用户归一化或透视成宽表反而是错的；这一层只收集「这次要
写哪些表」，真正落盘在 :mod:`wb_analyzer.export`。

默认只导精简的原始灰度（积分灰度 + 检出限标记）。完整统计列和历史长表
都是可选项，默认不勾。

对话框记住上次勾选（:mod:`wb_analyzer.ui.prefs`），但当前做不到的项
（还没归一化、历史是空的）会被置灰，不会带着上次的勾选硬导一份空表。
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QLabel,
    QVBoxLayout,
)

from . import prefs as prefs_mod


@dataclass(frozen=True)
class DataExportOptions:
    """一次「导出数据」勾了哪几张表、用哪一档列。"""

    raw_stats: bool = True
    normalized: bool = False
    summary_tables: bool = False
    full_stats: bool = False
    session_raw: bool = False

    def any_selected(self) -> bool:
        return (
            self.raw_stats
            or self.normalized
            or self.summary_tables
            or self.session_raw
        )


class DataExportDialog(QDialog):
    """收集导出档位。构造时传入当前做得到什么，做不到的项当场置灰。"""

    def __init__(
        self,
        parent=None,
        *,
        options: DataExportOptions | None = None,
        can_export_raw: bool = True,
        can_export_normalized: bool = False,
        can_export_summary: bool = False,
        can_export_session_raw: bool = False,
        raw_reason: str = "",
        normalized_reason: str = "",
        summary_reason: str = "",
        session_raw_reason: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("导出数据")
        self.setModal(True)
        self._can_raw = bool(can_export_raw)
        self._can_normalized = bool(can_export_normalized)
        self._can_summary = bool(can_export_summary)
        self._can_session_raw = bool(can_export_session_raw)
        self._raw_reason = raw_reason
        self._normalized_reason = normalized_reason
        self._summary_reason = summary_reason
        self._session_raw_reason = session_raw_reason
        base = options or DataExportOptions()

        layout = QVBoxLayout(self)

        box = QGroupBox("本次导出")
        box_layout = QVBoxLayout(box)

        self.raw_check = QCheckBox("原始灰度统计（每次测量一行，不归一化、不合并）")
        self.raw_check.setChecked(base.raw_stats and self._can_raw)
        self.full_stats_check = QCheckBox("包含完整统计列（位置、背景、峰值、口径等）")
        self.full_stats_check.setChecked(base.full_stats)
        self.session_raw_check = QCheckBox("同时导出历史所有膜的原始长表")
        self.session_raw_check.setChecked(base.session_raw and self._can_session_raw)
        self.normalized_check = QCheckBox("归一化结果（normalized / fold change）")
        self.normalized_check.setChecked(base.normalized and self._can_normalized)
        self.summary_check = QCheckBox("跨膜汇总（长表 + 宽表）")
        self.summary_check.setChecked(base.summary_tables and self._can_summary)

        self._apply_availability(
            self.raw_check,
            self._can_raw,
            self._raw_reason
            or "还没有可导出的定量结果：先框选 ROI 或点“自动检测”。",
            "默认只写来源、泳道、条带、样品名、积分灰度和检出限标记。"
            "同一份样品重复上样会留下多行，软件不会求平均。",
        )
        self.full_stats_check.setToolTip(
            "勾选后恢复完整列：旧版 15 列位置不变，溯源列追加在 origin 之后。"
            "默认不勾——分析通常只需要积分灰度。"
        )
        self._apply_availability(
            self.session_raw_check,
            self._can_session_raw,
            self._session_raw_reason
            or "归一化历史是空的：还没有可导出的跨膜原始长表。",
            "另写一份 raw_long.csv，覆盖历史里每一张膜的每一次测量。"
            "列档位跟随「包含完整统计列」。默认不导出。",
        )
        self._apply_availability(
            self.normalized_check,
            self._can_normalized,
            self._normalized_reason
            or "还没有可导出的归一化结果：同一分组需要同时有内参与目标蛋白。",
            "当前分组里已经配对好的比值。未做归一化时这一项不可用。",
        )
        self._apply_availability(
            self.summary_check,
            self._can_summary,
            self._summary_reason
            or "还没有可导出的跨膜汇总：每个分组都需要一条内参和至少一个目标蛋白。",
            "长表逐行记录配对结果；宽表按分组 × 泳道透视。"
            "一格对应多条记录时留空，不替你合并。",
        )

        box_layout.addWidget(self.raw_check)
        box_layout.addWidget(self.full_stats_check)
        box_layout.addWidget(self.session_raw_check)
        box_layout.addWidget(self.normalized_check)
        box_layout.addWidget(self.summary_check)
        layout.addWidget(box)

        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setObjectName("dimHint")
        layout.addWidget(self.hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_button.setText("选择目录…")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        for widget in (
            self.raw_check,
            self.full_stats_check,
            self.session_raw_check,
            self.normalized_check,
            self.summary_check,
        ):
            widget.toggled.connect(self._refresh)
        self._refresh()

    @staticmethod
    def _apply_availability(
        check: QCheckBox, enabled: bool, disabled_tip: str, enabled_tip: str
    ) -> None:
        check.setEnabled(enabled)
        if not enabled:
            check.setChecked(False)
            check.setToolTip(disabled_tip)
        else:
            check.setToolTip(enabled_tip)

    def _refresh(self) -> None:
        writing_raw = self.raw_check.isChecked() or self.session_raw_check.isChecked()
        self.full_stats_check.setEnabled(writing_raw)
        if not writing_raw:
            self.full_stats_check.setChecked(False)

        options = self.options()
        self.ok_button.setEnabled(options.any_selected())
        if not options.any_selected():
            self.hint.setText("⚠ 至少要勾一项，否则没有文件可写。")
        else:
            self.hint.setText(
                "默认只导积分灰度与检出限标记。完整统计列和历史长表都是可选项。"
                "重复上样的膜建议只勾选原始灰度统计，在下游自行合并。"
            )

    def options(self) -> DataExportOptions:
        return DataExportOptions(
            raw_stats=self.raw_check.isChecked(),
            normalized=self.normalized_check.isChecked(),
            summary_tables=self.summary_check.isChecked(),
            full_stats=self.full_stats_check.isChecked(),
            session_raw=self.session_raw_check.isChecked(),
        )

    @classmethod
    def load_options(cls, store=None) -> DataExportOptions:
        raw, normalized, summary, full_stats, session_raw = (
            prefs_mod.load_data_export_choices(store)
        )
        return DataExportOptions(
            raw_stats=raw,
            normalized=normalized,
            summary_tables=summary,
            full_stats=full_stats,
            session_raw=session_raw,
        )

    def save_options(self, store=None) -> DataExportOptions:
        options = self.options()
        prefs_mod.save_data_export_choices(
            options.raw_stats,
            options.normalized,
            options.summary_tables,
            options.full_stats,
            options.session_raw,
            store,
        )
        return options

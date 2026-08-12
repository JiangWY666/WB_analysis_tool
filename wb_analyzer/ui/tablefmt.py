"""表格数值显示：把积分灰度这类大数写成千分位，而不是 9.59522e+07。

Qt 默认的 ``QStyledItemDelegate::displayText`` 对 double 固定 6 位有效数字，
95952170.0 会显示成 '9.59522e+07'——最重要的一列反而最难读。这里只改显示，
单元格里存的仍然是数值（排序照常按数值走，复制 TSV 仍是全精度）。
"""

from __future__ import annotations

import math

from PySide6.QtWidgets import QStyledItemDelegate

EMPTY_TEXT = "—"


def format_number(value: float | int | None) -> str:
    """千分位显示；极端量级仍退回科学计数法（否则会拉出几十个 0）。"""
    if value is None:
        return EMPTY_TEXT
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return EMPTY_TEXT
    magnitude = abs(number)
    if magnitude >= 1e15 or (magnitude and magnitude < 1e-4):
        return f"{number:.4g}"
    if magnitude >= 1000:
        return f"{number:,.0f}"
    text = f"{number:,.3f}" if magnitude >= 1 else f"{number:.5f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def full_precision(value: float | int | None) -> str:
    """悬停提示用的完整数值——显示取整了，用户还得看得到原值。"""
    if value is None:
        return EMPTY_TEXT
    return repr(float(value))


class NumberDelegate(QStyledItemDelegate):
    """只接管浮点数的显示文本，其余交回 Qt 默认实现。"""

    def displayText(self, value, locale):  # noqa: N802 - Qt API
        if isinstance(value, float):
            return format_number(value)
        return super().displayText(value, locale)

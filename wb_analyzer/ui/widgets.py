"""界面里反复用到的小控件。

只有一个：单行省略标签。它存在的理由是布局而不是美观——``QLabel`` 开了
``setWordWrap`` 之后会把一段长警告折成三四行，横条 / 面板跟着长高，把画布和
表格挤没了；不开自动折行则会把父控件整个撑宽，停靠区冒出横向滚动条。两种都
是"一句提示改变了整个界面的尺寸"。这里的做法是让标签**永远只占一行、永远不
要求任何宽度**，放不下就省略，全文留在 tooltip 里。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QLabel, QSizePolicy


class ElidedLabel(QLabel):
    """单行、按当前宽度省略；``text()`` 与 tooltip 始终是全文。

    ``text()`` 返回全文而不是省略后的字符串，是为了让调用方（和测试）读到的
    永远是这条提示真正说了什么——界面上截断到哪里是显示细节。
    """

    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(parent)
        self._full = ""
        self._eliding = False
        self.setWordWrap(False)
        self.setTextFormat(Qt.TextFormat.PlainText)
        # 水平 Ignored：再长的提示也不会把停靠面板撑宽
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setText(text)

    # ------------------------------------------------------------------ 文本

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API
        self._full = str(text or "")
        self.setToolTip(self._full)
        self._elide()

    def text(self) -> str:  # noqa: A003 - Qt API
        return self._full

    def displayed_text(self) -> str:
        """真正画在屏幕上的那一串（可能带省略号），供测试断言。"""
        return super().text()

    def is_elided(self) -> bool:
        return self.displayed_text() != self._full

    # ------------------------------------------------------------------ 布局

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._elide()

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().changeEvent(event)
        self._elide()

    def minimumSizeHint(self):  # noqa: N802 - Qt API
        hint = super().minimumSizeHint()
        hint.setWidth(0)
        return hint

    def _elide(self) -> None:
        if self._eliding:
            return
        self._eliding = True
        try:
            metrics = QFontMetrics(self.font())
            available = max(0, self.width() - 2)
            if available <= 0:
                super().setText(self._full)
                return
            super().setText(
                metrics.elidedText(self._full, Qt.TextElideMode.ElideRight, available)
            )
        finally:
            self._eliding = False


__all__ = ["ElidedLabel"]

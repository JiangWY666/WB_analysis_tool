"""浏览器 / Origin 式文档标签栏：可关、中键关、拖动重排、长文件名省略。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSizePolicy, QTabBar


class DocumentTabBar(QTabBar):
    """画布上方那一条文档标签。0 个标签时由主窗口整条隐藏。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("docTabBar")
        self.setDocumentMode(True)
        self.setExpanding(False)
        self.setMovable(True)
        self.setTabsClosable(True)
        self.setDrawBase(True)
        self.setElideMode(Qt.TextElideMode.ElideMiddle)
        self.setUsesScrollButtons(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        # 标签不要跟着窗口拉成等宽：文件名长短不一，等宽会把短的撑得很怪
        self.setExpanding(False)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.MiddleButton:
            index = self.tabAt(event.position().toPoint())
            if index >= 0:
                self.tabCloseRequested.emit(index)
                event.accept()
                return
        super().mouseReleaseEvent(event)

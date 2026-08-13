"""当前定量结果的统一诊断条，横在画布正上方。

为什么只有一条：界面上关于"这批数字靠不靠谱"的提示原本有三路——ROI 自检、
条带策略在单条带泳道未生效、「每道条带数」没切够——各自挂在不同面板上。分散
的后果是同一件事被说两遍（自检的"条带数对不上"和饱和提示讲的常常是同一个泳
道），而真正致命的那条反而可能落在没打开的标签页里。这里把三路收进同一个按严
重度排序的列表，一个位置、一套视觉层级。

放在画布上方而不是结果表里，是因为 ROI 自检要在用户**改 ROI 的时候**看得见，
而那时候他多半停在"调整"页。横条只占一行，展开详情才长高，不遮挡画布。

信息分三层，默认只露第一层：

1. **一行摘要**——最严重那条的标题，加上"另有 N 项"。这一行**永不折行**，放不
   下就省略号，全文进 tooltip。以前它是 wordWrap 的，三条诊断就能把横条撑成
   一段话压在画布上。
2. **详情**（点「详情」展开）——每条的标题 + 全文。``warning`` / ``info`` 默认
   折叠：那是"看一眼摘要就知道要不要展开"的东西。
3. ``error`` 是例外，**默认就展开**。它的含义是"这组数字不能用"，不是"注意
   一下"；把它折进一个要点开的按钮里，等于赌用户会去点。用户仍可以随手收起，
   也可以「我知道了」整条忽略——可以选择不看，不能看不见。
"""

from __future__ import annotations

import html
from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
)

from . import theme
from .widgets import ElidedLabel

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

RANK = {SEVERITY_INFO: 0, SEVERITY_WARNING: 1, SEVERITY_ERROR: 2}
LABELS = {SEVERITY_ERROR: "严重", SEVERITY_WARNING: "警告", SEVERITY_INFO: "提示"}
ICONS = {SEVERITY_ERROR: "⛔", SEVERITY_WARNING: "⚠", SEVERITY_INFO: "ⓘ"}
COLORS = {
    SEVERITY_ERROR: theme.ERROR,
    SEVERITY_WARNING: theme.WARN,
    SEVERITY_INFO: theme.TEXT_DIM,
}

DISMISS_TEXT = "我知道了"
RESTORE_TEXT = "重新查看"
DETAIL_SHOW = "详情 ▾"
DETAIL_HIDE = "收起 ▴"

# 详情区最多吃掉多少高度，再长就自己滚——诊断条不能把画布挤没了。
# 两个值：用户**主动**点开时给足（他就是来读的）；``error`` 自动展开时只给一
# 眼的量。核心层的判词常常是整段，实测用户那批膜上自动展开一次就占掉窗口高度
# 的四分之一——那正是"分析时不想看大段文字"说的东西。前几行恰好是"该怎么办"
# （"把 ROI 上边往上挪至少 10 px"），够用了，剩下的滚一下就有。
DETAIL_MAX_H = 190
DETAIL_PEEK_H = 96


@dataclass(frozen=True)
class Entry:
    """一条诊断，字段与核心层 :class:`~wb_analyzer.roi_check.RoiIssue` 对齐。

    对齐是为了让 ROI 自检的 issue 原样搬进来，界面自己的提示也套同一个壳——
    两者在列表里长得一模一样，用户不需要知道哪条出自哪个模块。
    """

    severity: str
    title: str
    detail: str = ""


class DiagnosticsBar(QFrame):
    """一行摘要 + 可展开的详情列表 + 「我知道了」。

    忽略是**按内容**记的，不是一个总开关：记下当前这批诊断的指纹，指纹一变
    （换了 ROI、改了每道条带数、问题本身变了）提示就自己回来。用户那张膜的习
    惯画法每次都会被自检拦下，没有这个机制，常驻警告很快就会被彻底无视。
    """

    dismissedChanged = Signal(bool)

    def __init__(self, parent=None, prefix: str = "") -> None:
        super().__init__(parent)
        self.setObjectName("diagnosticsBar")
        self._prefix = str(prefix)
        self._entries: list[Entry] = []
        self._summary = ""
        self._severity: str | None = None
        self._key: object = None
        self._dismissed_key: object = None
        # None = 跟随严重度（error 展开、其余折叠）；True / False = 用户就这批
        # 诊断做过的显式选择，换一批诊断就作废
        self._expanded: bool | None = None
        self._styled = ""  # 现在挂着哪个色的 QSS；重设样式表会连累整棵子树重算

        self._icon = QLabel()
        self._icon.setObjectName("diagIcon")
        self._chip = QLabel()
        self._chip.setObjectName("diagChip")
        self._text = ElidedLabel()
        self._text.setObjectName("diagText")

        self._detail_button = QPushButton(DETAIL_SHOW)
        self._detail_button.setFlat(True)
        self._detail_button.clicked.connect(self._toggle_detail)
        self._dismiss_button = QPushButton(DISMISS_TEXT)
        self._dismiss_button.setFlat(True)
        self._dismiss_button.clicked.connect(self._on_dismiss_clicked)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        header.addWidget(self._icon)
        header.addWidget(self._chip)
        header.addWidget(self._text, 1)
        header.addWidget(self._detail_button)
        header.addWidget(self._dismiss_button)

        self._detail = QLabel()
        self._detail.setWordWrap(True)
        self._detail.setTextFormat(Qt.TextFormat.RichText)
        self._detail.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self._detail.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._detail.setContentsMargins(2, 2, 2, 2)
        self._scroll = QScrollArea()
        self._scroll.setWidget(self._detail)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setMaximumHeight(DETAIL_PEEK_H)
        self._scroll.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)
        layout.addLayout(header)
        layout.addWidget(self._scroll)

        self.clear()

    # ------------------------------------------------------------------ 数据

    def set_entries(
        self, entries: list[Entry], summary: str = "", key: object = None
    ) -> None:
        """更新诊断内容。

        ``key`` 是这批诊断的指纹；与上次「我知道了」时记下的相同就保持折叠。
        ``summary`` 为空时按条目自动拼一行。
        """
        self._entries = sorted(entries, key=lambda e: -RANK.get(e.severity, 0))
        actionable = [e for e in self._entries if e.severity != SEVERITY_INFO]
        self._severity = (
            max(actionable, key=lambda e: RANK[e.severity]).severity
            if actionable
            else None
        )
        self._summary = summary or self._auto_summary()
        if self._key != key:
            # 换了一批诊断：用户上一次的展开 / 收起选择跟着作废，回到"按严重度
            # 决定"。否则收起过一次警告详情，之后来的严重问题就再也不自动展开了。
            self._expanded = None
        self._key = key
        if self._dismissed_key is not None and self._dismissed_key != key:
            self._dismissed_key = None  # 指纹变了，提示自己回来
        self._refresh()

    def clear(self) -> None:
        self._entries = []
        self._severity = None
        self._summary = ""
        self._key = None
        self._dismissed_key = None
        self._expanded = None
        self._refresh()

    # ------------------------------------------------------------------ 查询（测试与主窗口用）

    def severity(self) -> str | None:
        return self._severity

    def summary(self) -> str:
        return self._summary

    def entries(self) -> list[Entry]:
        return list(self._entries)

    def detail_text(self) -> str:
        """所有条目的 title + detail 拼起来的纯文本，方便断言与状态栏复用。"""
        return " ".join(
            f"{e.title}：{e.detail}" if e.detail else e.title for e in self._entries
        )

    def is_dismissed(self) -> bool:
        # 通过（severity is None）没有什么可忽略的，那一行淡色确认永远照常显示
        if self._severity is None:
            return False
        return self._dismissed_key is not None and self._dismissed_key == self._key

    def wants_detail(self) -> bool:
        """这批诊断该不该展开详情——用户没表过态时按严重度决定。"""
        if self._expanded is not None:
            return self._expanded
        return self._severity == SEVERITY_ERROR

    def is_expanded(self) -> bool:
        return self.wants_detail() and self._scroll.isVisibleTo(self)

    def dismiss(self) -> None:
        self._on_dismiss_clicked()

    # ------------------------------------------------------------------ 交互

    def _on_dismiss_clicked(self) -> None:
        if self.is_dismissed():
            self._dismissed_key = None
        else:
            self._dismissed_key = self._key
        # 无论忽略还是重新查看，都回到"按严重度决定"：重新查看一条严重诊断时
        # 该再展开一次，而不是沿用忽略之前那一下的收起状态
        self._expanded = None
        self._refresh()
        self.dismissedChanged.emit(self.is_dismissed())

    def _toggle_detail(self) -> None:
        self._expanded = not self.wants_detail()
        self._refresh()

    # ------------------------------------------------------------------ 呈现

    def _auto_summary(self) -> str:
        """一行摘要：最严重那条的标题 + 还有几条。全文在详情里。

        以前这里把所有标题用「；」串成一句，三条诊断就是一段话——横条 wordWrap
        之后长到三行压住画布，而用户在"分析"，没人要读这个。
        """
        if not self._entries:
            return ""
        actionable = [e for e in self._entries if e.severity != SEVERITY_INFO]
        shown = actionable or list(self._entries)
        head = f"{self._prefix}：{shown[0].title}" if self._prefix else shown[0].title
        rest = len(shown) - 1
        return head + (f"（另有 {rest} 项）" if rest else "")

    def _refresh(self) -> None:
        if not self._summary and not self._entries:
            self.setVisible(False)
            return
        self.setVisible(True)
        dismissed = self.is_dismissed()
        severity = self._severity
        color = COLORS.get(severity or "", theme.TEXT_DIM) if severity else theme.OK
        if dismissed:
            color = theme.TEXT_FAINT
        if color != self._styled:
            self._styled = color
            self.setStyleSheet(_style(color))

        if dismissed:
            self._icon.setText("○")
            self._chip.setVisible(False)
            count = len([e for e in self._entries if e.severity != SEVERITY_INFO])
            self._text.setText(
                f"已忽略本次自检提示（{LABELS.get(severity or '', '提示')} · {count} 项）。"
                "ROI、每道条带数或问题本身一变就会重新提醒。"
            )
            self._detail_button.setVisible(False)
            self._scroll.setVisible(False)
            self._dismiss_button.setVisible(True)
            self._dismiss_button.setText(RESTORE_TEXT)
            return

        self._dismiss_button.setText(DISMISS_TEXT)
        if severity is None:
            # 通过：一行淡色确认，没有任何按钮，不打扰
            self._icon.setText("✓")
            self._chip.setVisible(False)
            self._text.setText(self._summary)
            self._detail_button.setVisible(bool(self._entries))
            self._dismiss_button.setVisible(False)
        else:
            self._icon.setText(ICONS[severity])
            self._chip.setVisible(True)
            self._chip.setText(LABELS[severity])
            self._text.setText(self._summary)
            self._detail_button.setVisible(True)
            self._dismiss_button.setVisible(True)
        expanded = self.wants_detail()
        self._detail_button.setText(DETAIL_HIDE if expanded else DETAIL_SHOW)
        self._detail.setText(self._detail_html())
        self._scroll.setMaximumHeight(
            DETAIL_MAX_H if self._expanded else DETAIL_PEEK_H
        )
        self._scroll.setVisible(expanded and bool(self._entries))

    def _detail_html(self) -> str:
        rows = []
        for entry in self._entries:
            tint = COLORS.get(entry.severity, theme.TEXT_DIM)
            detail = html.escape(entry.detail).replace("\n", "<br>")
            head = ""
            # 摘要那行已经把最严重那条的标题写出来了，详情里再写一遍就是同一句
            # 话紧挨着出现两次——只在还有别的条目时才需要标题来分段
            if entry.title not in self._summary:
                head = (
                    f'<span style="color:{tint};font-weight:600;">'
                    f"{ICONS.get(entry.severity, '')} "
                    f"{LABELS.get(entry.severity, '')}·{html.escape(entry.title)}"
                    f"</span><br>"
                )
            rows.append(
                f'<p style="margin:0 0 8px 0;">{head}'
                f'<span style="color:{theme.TEXT_DIM};">{detail}</span></p>'
            )
        return "".join(rows)


def _style(color: str) -> str:
    """整条的强调色随严重度走：左边框 + 图标 + 徽标是同一个色。"""
    return f"""
QFrame#diagnosticsBar {{
    background: {theme.BG_PANEL};
    border: none;
    border-left: 3px solid {color};
    border-bottom: 1px solid {theme.BG_BASE};
}}
QLabel#diagIcon {{
    color: {color};
    font-size: 13px;
}}
QLabel#diagChip {{
    color: {color};
    border: 1px solid {color};
    border-radius: 3px;
    padding: 0 5px;
    font-weight: 600;
}}
QLabel#diagText {{
    color: {theme.TEXT};
}}
QFrame#diagnosticsBar QPushButton {{
    color: {theme.TEXT_DIM};
    background: transparent;
    border: 1px solid {theme.BORDER};
    padding: 2px 10px;
    min-height: 14px;
}}
QFrame#diagnosticsBar QPushButton:hover {{
    color: {theme.TEXT};
    border-color: {theme.ACCENT_DIM};
}}
"""


def entries_from_report(report) -> list[Entry]:
    """把核心层的 :class:`RoiQualityReport` 摊平成本模块的条目。"""
    if report is None:
        return []
    return [
        Entry(severity=str(issue.severity), title=issue.title, detail=issue.detail)
        for issue in report.issues
    ]


__all__ = ["DiagnosticsBar", "Entry", "entries_from_report"]

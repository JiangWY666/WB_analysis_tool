"""Fusion 深色主题：QPalette + QSS，强调色为青蓝色系。"""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

ACCENT = "#4fc3f7"
ACCENT_HOVER = "#7dd3f9"
ACCENT_PRESSED = "#2ea8e0"
ACCENT_DIM = "#2b6d8c"
BG_WINDOW = "#202225"
BG_PANEL = "#26282c"
BG_BASE = "#17181b"
BG_ALT = "#1d1f23"
BG_RAISED = "#2c2f34"
BORDER = "#3a3d43"
BORDER_LIGHT = "#4a4e55"
TEXT = "#e8eaed"
TEXT_DIM = "#9aa0a6"
TEXT_DISABLED = "#5f6469"
WARN = "#ffb74d"


def _palette() -> QPalette:
    palette = QPalette()
    groups = (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive)
    roles = {
        QPalette.ColorRole.Window: BG_WINDOW,
        QPalette.ColorRole.WindowText: TEXT,
        QPalette.ColorRole.Base: BG_BASE,
        QPalette.ColorRole.AlternateBase: BG_ALT,
        QPalette.ColorRole.ToolTipBase: BG_RAISED,
        QPalette.ColorRole.ToolTipText: TEXT,
        QPalette.ColorRole.Text: TEXT,
        QPalette.ColorRole.PlaceholderText: TEXT_DIM,
        QPalette.ColorRole.Button: BG_PANEL,
        QPalette.ColorRole.ButtonText: TEXT,
        QPalette.ColorRole.BrightText: "#ff6e6e",
        QPalette.ColorRole.Highlight: ACCENT,
        QPalette.ColorRole.HighlightedText: "#0c2430",
        QPalette.ColorRole.Link: ACCENT,
        QPalette.ColorRole.Light: BORDER_LIGHT,
        QPalette.ColorRole.Midlight: BORDER,
        QPalette.ColorRole.Mid: BORDER,
        QPalette.ColorRole.Dark: BG_BASE,
        QPalette.ColorRole.Shadow: "#000000",
    }
    for group in groups:
        for role, color in roles.items():
            palette.setColor(group, role, QColor(color))
    disabled = QPalette.ColorGroup.Disabled
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(disabled, role, QColor(TEXT_DISABLED))
    palette.setColor(disabled, QPalette.ColorRole.Highlight, QColor(ACCENT_DIM))
    palette.setColor(disabled, QPalette.ColorRole.Base, QColor(BG_WINDOW))
    return palette


_QSS = f"""
QWidget {{
    outline: none;
}}
QMainWindow::separator {{
    background: {BG_BASE};
    width: 3px;
    height: 3px;
}}
QMainWindow::separator:hover {{
    background: {ACCENT_DIM};
}}

QToolBar {{
    background: {BG_PANEL};
    border: none;
    border-bottom: 1px solid {BG_BASE};
    padding: 4px 6px;
    spacing: 4px;
}}
QToolBar::separator {{
    background: {BORDER};
    width: 1px;
    margin: 5px 6px;
}}
QToolButton {{
    background: transparent;
    color: {TEXT};
    border: 1px solid transparent;
    border-radius: 5px;
    padding: 5px 10px;
}}
QToolButton:hover {{
    background: {BG_RAISED};
    border-color: {BORDER};
}}
QToolButton:pressed {{
    background: {BG_BASE};
}}
QToolButton:checked {{
    background: rgba(79, 195, 247, 0.16);
    border-color: {ACCENT_DIM};
    color: {ACCENT};
}}
QToolButton:disabled {{
    color: {TEXT_DISABLED};
}}
QToolButton#primaryButton {{
    background: {ACCENT};
    color: #0c2430;
    font-weight: 600;
    border: none;
    padding: 5px 14px;
}}
QToolButton#primaryButton:hover {{
    background: {ACCENT_HOVER};
}}
QToolButton#primaryButton:pressed {{
    background: {ACCENT_PRESSED};
}}
QToolButton#primaryButton:disabled {{
    background: {BG_RAISED};
    color: {TEXT_DISABLED};
}}

QPushButton {{
    background: {BG_RAISED};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 5px 14px;
    min-height: 18px;
}}
QPushButton:hover {{
    border-color: {ACCENT_DIM};
    background: #33373d;
}}
QPushButton:pressed {{
    background: {BG_BASE};
}}
QPushButton:disabled {{
    color: {TEXT_DISABLED};
    background: {BG_PANEL};
    border-color: {BORDER};
}}
QPushButton:default {{
    border-color: {ACCENT_DIM};
}}
/* 键盘焦点可见：Tab 巡航时能看出焦点在哪个按钮 / 滑杆上 */
QPushButton:focus {{
    border: 1px solid {ACCENT};
}}
QToolButton:focus {{
    border: 1px solid {ACCENT_DIM};
}}
QSlider::handle:horizontal:focus {{
    background: {ACCENT_HOVER};
    border: 2px solid {ACCENT};
}}

QTabWidget::pane {{
    border: 1px solid {BG_BASE};
    border-top: none;
    background: {BG_WINDOW};
}}
QTabBar::tab {{
    background: transparent;
    color: {TEXT_DIM};
    border: none;
    border-bottom: 2px solid transparent;
    padding: 7px 16px;
    margin-right: 2px;
}}
QTabBar::tab:hover {{
    color: {TEXT};
}}
QTabBar::tab:selected {{
    color: {ACCENT};
    border-bottom: 2px solid {ACCENT};
}}

QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 12px;
    padding: 10px 8px 8px 8px;
    font-weight: 600;
    color: {TEXT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    top: 2px;
    padding: 0 4px;
    color: {ACCENT};
}}

QTableWidget, QTableView {{
    background: {BG_BASE};
    alternate-background-color: {BG_ALT};
    gridline-color: #26282d;
    border: 1px solid {BORDER};
    border-radius: 4px;
    selection-background-color: rgba(79, 195, 247, 0.28);
    selection-color: {TEXT};
}}
QHeaderView::section {{
    background: {BG_PANEL};
    color: {TEXT_DIM};
    border: none;
    border-right: 1px solid {BG_BASE};
    border-bottom: 1px solid {BG_BASE};
    padding: 5px 8px;
    font-weight: 600;
}}
QHeaderView::section:hover {{
    color: {TEXT};
}}
QTableCornerButton::section {{
    background: {BG_PANEL};
    border: none;
}}

QScrollBar:vertical {{
    background: transparent;
    width: 12px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_LIGHT};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {ACCENT_DIM};
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 12px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER_LIGHT};
    border-radius: 4px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {ACCENT_DIM};
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    width: 0;
    height: 0;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
}}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {BG_BASE};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 6px;
    selection-background-color: {ACCENT_DIM};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background: {BG_RAISED};
    color: {TEXT};
    border: 1px solid {BORDER};
    selection-background-color: rgba(79, 195, 247, 0.25);
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background: {BG_RAISED};
    border: none;
    width: 16px;
}}

QSlider::groove:horizontal {{
    height: 4px;
    background: {BG_RAISED};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT_DIM};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{
    background: {ACCENT_HOVER};
}}

QCheckBox, QRadioButton {{
    color: {TEXT};
    spacing: 6px;
}}
QCheckBox:disabled, QRadioButton:disabled {{
    color: {TEXT_DISABLED};
}}

QDockWidget {{
    color: {TEXT};
    titlebar-close-icon: none;
}}
QDockWidget::title {{
    background: {BG_PANEL};
    padding: 5px 10px;
    border-bottom: 1px solid {BG_BASE};
}}

QMenuBar {{
    background: {BG_PANEL};
    color: {TEXT};
    border-bottom: 1px solid {BG_BASE};
}}
QMenuBar::item {{
    padding: 5px 10px;
    background: transparent;
}}
QMenuBar::item:selected {{
    background: {BG_RAISED};
    border-radius: 4px;
}}
QMenu {{
    background: {BG_RAISED};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 4px;
}}
QMenu::item {{
    padding: 5px 24px 5px 12px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background: rgba(79, 195, 247, 0.22);
}}
QMenu::item:disabled {{
    color: {TEXT_DISABLED};
}}
QMenu::separator {{
    height: 1px;
    background: {BORDER};
    margin: 4px 8px;
}}

QStatusBar {{
    background: {BG_PANEL};
    color: {TEXT_DIM};
    border-top: 1px solid {BG_BASE};
}}
QStatusBar QLabel {{
    color: {TEXT_DIM};
    padding: 0 4px;
}}

QToolTip {{
    background: {BG_RAISED};
    color: {TEXT};
    border: 1px solid {ACCENT_DIM};
    padding: 5px 8px;
}}

QSplitter::handle {{
    background: {BG_BASE};
}}
QSplitter::handle:hover {{
    background: {ACCENT_DIM};
}}
"""


def apply_theme(app: QApplication) -> None:
    """应用 Fusion 深色主题与统一 QSS。"""
    app.setStyle("Fusion")
    app.setPalette(_palette())
    font = QFont("Microsoft YaHei UI")
    font.setPointSizeF(9.5)
    app.setFont(font)
    app.setStyleSheet(_QSS)

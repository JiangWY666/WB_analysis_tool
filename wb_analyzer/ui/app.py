"""应用入口：python -m wb_analyzer.ui.app [图片路径]"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from . import errorlog
from .main_window import IMAGE_SUFFIXES, MainWindow
from .theme import apply_theme


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv if argv is None else argv)
    # 打包版 console=False：没有日志就等于崩溃后什么都查不到
    errorlog.install()
    errorlog.install_excepthook()
    QApplication.setOrganizationName("WBAnalyzer")
    QApplication.setApplicationName("WBAnalyzer")
    QApplication.setApplicationDisplayName("WB 灰度分析工具")
    app = QApplication(args)
    apply_theme(app)
    window = MainWindow()
    window.show()
    for arg in args[1:]:
        path = Path(arg)
        if path.suffix.lower() in IMAGE_SUFFIXES and path.exists():
            window.open_path(path)
            break
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

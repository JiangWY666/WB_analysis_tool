"""异常日志：打包版 console=False，stdout 是 None，崩溃必须落到用户目录的文件里。

日志位置：``%LOCALAPPDATA%\\WB-Analyzer\\logs\\wb-analyzer.log``
（非 Windows 回退到 ``~/.local/state/WB-Analyzer/logs``）。
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import TracebackType

APP_DIR_NAME = "WB-Analyzer"
LOG_FILE_NAME = "wb-analyzer.log"
LOGGER_NAME = "wb_analyzer.ui"

_MAX_BYTES = 512 * 1024
_BACKUP_COUNT = 3

_log_path: Path | None = None
_installed = False


def default_log_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / APP_DIR_NAME / "logs"
    return Path.home() / ".local" / "state" / APP_DIR_NAME / "logs"


def log_path() -> Path | None:
    """当前日志文件路径；日志不可用时为 None。"""
    return _log_path


def logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def install(directory: Path | str | None = None) -> Path | None:
    """挂上文件日志。目录不可写时静默降级——记日志本身不能再把程序搞崩。"""
    global _installed, _log_path
    if _installed:
        return _log_path
    _installed = True
    target = Path(directory) if directory is not None else default_log_dir()
    log = logger()
    log.setLevel(logging.INFO)
    log.propagate = False
    try:
        target.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            target / LOG_FILE_NAME,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
    except OSError:
        return None
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    log.addHandler(handler)
    _log_path = target / LOG_FILE_NAME
    log.info("=== 会话开始 pid=%s python=%s ===", os.getpid(), sys.version.split()[0])
    return _log_path


def log_exception(context: str, exc: BaseException | None = None) -> None:
    """记录一条带堆栈的错误。异常对象为 None 时取当前 except 上下文。"""
    install()
    try:
        logger().error(context, exc_info=exc if exc is not None else True)
    except Exception:  # noqa: BLE001 - 日志失败绝不向上抛
        pass


def log_message(text: str) -> None:
    install()
    try:
        logger().info(text)
    except Exception:  # noqa: BLE001
        pass


def hint() -> str:
    """给用户看的“日志在哪”提示，可直接拼到错误对话框里。"""
    path = _log_path or install()
    return f"\n\n详细信息已写入日志：\n{path}" if path else ""


def install_excepthook() -> None:
    """未捕获异常先落盘再交回原处理器（打包版否则什么都看不到）。"""
    previous = sys.excepthook

    def hook(
        kind: type[BaseException],
        value: BaseException,
        tb: TracebackType | None,
    ) -> None:
        install()
        try:
            logger().critical("未捕获异常", exc_info=(kind, value, tb))
        except Exception:  # noqa: BLE001
            pass
        previous(kind, value, tb)

    sys.excepthook = hook

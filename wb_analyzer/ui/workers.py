"""QThreadPool 后台任务包装：耗时计算不阻塞 UI 线程。"""

from __future__ import annotations

import traceback
from collections.abc import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)


class Worker(QRunnable):
    """把任意可调用对象丢进线程池执行，结果通过信号送回主线程。"""

    def __init__(self, fn: Callable, *args: object, **kwargs: object) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self) -> None:  # pragma: no cover - Qt 线程内执行
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception as exc:  # noqa: BLE001 - 界面层需要兜住一切异常
            traceback.print_exc()
            self.signals.error.emit(str(exc) or exc.__class__.__name__)
        else:
            self.signals.finished.emit(result)


_ACTIVE: set[Worker] = set()


def run_async(
    pool: QThreadPool,
    fn: Callable,
    on_done: Callable[[object], None],
    on_error: Callable[[str], None] | None = None,
    *args: object,
    **kwargs: object,
) -> Worker:
    """在线程池中运行 fn(*args, **kwargs)，完成后回调主线程。"""
    worker = Worker(fn, *args, **kwargs)
    _ACTIVE.add(worker)

    def _done(payload: object) -> None:
        _ACTIVE.discard(worker)
        on_done(payload)

    def _fail(message: str) -> None:
        _ACTIVE.discard(worker)
        if on_error is not None:
            on_error(message)

    worker.signals.finished.connect(_done)
    worker.signals.error.connect(_fail)
    pool.start(worker)
    return worker

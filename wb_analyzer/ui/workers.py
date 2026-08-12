"""QThreadPool 后台任务包装：耗时计算不阻塞 UI 线程。"""

from __future__ import annotations

import threading
from collections.abc import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from . import errorlog


class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)


class Worker(QRunnable):
    """把任意可调用对象丢进线程池执行，结果通过信号送回主线程。

    被 :meth:`cancel` 标记后不再 emit：窗口关闭时接收方 C++ 对象可能已析构，
    再发信号就是 ``RuntimeError: Signal source has been deleted``。
    """

    def __init__(self, fn: Callable, *args: object, **kwargs: object) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._cancelled = False
        self.signals = WorkerSignals()

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def _emit(self, signal: Signal, payload: object) -> None:
        if self._cancelled:
            return
        try:
            signal.emit(payload)
        except RuntimeError:  # 接收方已随窗口销毁
            pass

    def run(self) -> None:
        try:
            if self._cancelled:
                return
            try:
                result = self._fn(*self._args, **self._kwargs)
            except Exception as exc:  # noqa: BLE001 - 界面层需要兜住一切异常
                errorlog.log_exception(f"后台任务失败：{self._describe()}", exc)
                self._emit(self.signals.error, str(exc) or exc.__class__.__name__)
            else:
                self._emit(self.signals.finished, result)
        finally:
            _unregister(self)

    def _describe(self) -> str:
        fn = getattr(self._fn, "func", self._fn)
        return getattr(fn, "__name__", repr(fn))


_LOCK = threading.Lock()
_ACTIVE: set[Worker] = set()


def _register(worker: Worker) -> None:
    with _LOCK:
        _ACTIVE.add(worker)


def _unregister(worker: Worker) -> None:
    with _LOCK:
        _ACTIVE.discard(worker)


def active_count() -> int:
    with _LOCK:
        return len(_ACTIVE)


def cancel_all() -> int:
    """标记所有在跑的任务为已取消，返回被标记的数量。"""
    with _LOCK:
        workers = list(_ACTIVE)
    for worker in workers:
        worker.cancel()
    return len(workers)


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
    _register(worker)

    def _done(payload: object) -> None:
        _unregister(worker)
        if worker.cancelled:
            return
        try:
            on_done(payload)
        except RuntimeError:  # 窗口已销毁
            pass

    def _fail(message: str) -> None:
        _unregister(worker)
        if worker.cancelled or on_error is None:
            return
        try:
            on_error(message)
        except RuntimeError:
            pass

    worker.signals.finished.connect(_done)
    worker.signals.error.connect(_fail)
    pool.start(worker)
    return worker

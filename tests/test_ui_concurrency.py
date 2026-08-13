"""界面层并发与结果一致性回归测试（离屏运行）。

覆盖：
* U1：后台任务必须绑定图像身份，换图 / 旋转后旧结果不得落到新图上。
  加载与旋转本身异步化之后，这里额外盯住 doc_version 的递增时机。
* U5：两类后台任务的界面反馈——结果类锁输入、呈现类不锁且后来者顶掉前者。
* U2：删除条带 / 撤销只能过滤，不能换一套算法重测、也不能重排泳道号。
* U6：ROI 定量与自动检测是两套背景口径，画布 / 结果表 / 归一化历史都不许混。
* U7：条带标签在任何缩放下要么清晰不重叠、要么整批隐藏（必须真渲染才算测到）。
* U8：橡皮筋框选多个条带，批量删除只占撤销栈一格。
* U9：剖面图里增删分隔线要和画布同一套判定、同一套坐标，图例不许遮挡泳道标签。
* U10：溯源接线——口径与条带策略由 QuantEntry 自己带；跨口径显式标红；
  未采用的条带仍然显示但一眼能和"已删除"分开；单条带泳道要说清策略没生效。
* U11：「每道条带数」——自动判据切不开时的手动兜底，语义与「期望泳道数」对称；
  改它只重新分段、不许动分隔线；切不够那么多条时必须点名说清楚。
* U12：ROI 自检接线——与定量同一次任务同一个 expected_bands；三路提示汇到一条
  诊断条；画布只用形状与纹理标注（不新增第五种颜色）；「我知道了」按内容忽略。
* U13：自动测倾角——只给建议不自动转；必须画出参考线让人目视确认；四种拒答
  一律不许抹掉用户手动量的标尺值。
* U14：发表级导出——两个版本各用各的灰度映射；旋转要从原图按 Lanczos 重做；
  覆盖确认要覆盖全部 4 个落盘文件；对话框里不许再出现有损格式。
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import replace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtCore import QEvent, QPointF, QSettings, Qt  # noqa: E402
from PySide6.QtGui import (  # noqa: E402
    QFontDatabase,
    QFontMetrics,
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
)
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QDockWidget,
    QLabel,
    QMessageBox,
    QToolBar,
)

from wb_analyzer.imageio import blot_from_array  # noqa: E402
from wb_analyzer.roi import MAX_STRAIGHTEN_ANGLE, RoiRect, lane_boundaries  # noqa: E402
from wb_analyzer.ui import canvas as canvas_mod  # noqa: E402
from wb_analyzer.ui import errorlog, workers  # noqa: E402
from wb_analyzer.ui import main_window as mw  # noqa: E402
from wb_analyzer.ui import normalize_panel as nzp  # noqa: E402
from wb_analyzer.ui import prefs as prefs_mod  # noqa: E402
from wb_analyzer.ui import profile_panel as pp  # noqa: E402
from wb_analyzer.ui import results_panel as rp  # noqa: E402
from wb_analyzer.ui.canvas import Tool  # noqa: E402
from wb_analyzer.ui.main_window import MainWindow  # noqa: E402
from wb_analyzer.ui.tablefmt import format_number  # noqa: E402
from wb_analyzer.ui.widgets import ElidedLabel  # noqa: E402

ROI_BOX = (15, 60, -30, 130)  # x, y, 宽度相对图宽的收进量, 高


@pytest.fixture(scope="session")
def qt_app(tmp_path_factory):
    # QSettings 与日志都指向临时目录：测试不该写用户的注册表和 AppData
    settings_dir = tmp_path_factory.mktemp("qsettings")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(settings_dir)
    )
    errorlog.install(tmp_path_factory.mktemp("logs"))
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


def synthetic_raw(
    depths: list[float], centers: tuple[float, ...] = (120.0,)
) -> np.ndarray:
    """16 位浅底深带图：每个泳道在每个 ``centers`` 位置各一条高斯条带。"""
    height, width = 240, 20 + len(depths) * 95
    yy = np.arange(height, dtype=float)
    raw = np.full((height, width), 60000.0)
    for index, depth in enumerate(depths):
        x0 = 20 + index * 95
        band = np.zeros(height, dtype=float)
        for center in centers:
            band += depth * np.exp(-0.5 * ((yy - center) / 8.0) ** 2)
        raw[:, x0 : x0 + 80] -= band[:, None]
    return np.clip(raw, 0, 65535).astype(np.uint16)


def noisy_raw(depths: list[float], *, noise: float, seed: int = 4) -> np.ndarray:
    """同上，但叠了噪声——检出限只有在有噪声时才有意义。"""
    rng = np.random.default_rng(seed)
    raw = synthetic_raw(depths).astype(float) + rng.normal(0.0, noise, (240, 20 + len(depths) * 95))
    return np.clip(raw, 0, 65535).astype(np.uint16)


LEAKY_LAYOUT_KEYS = ("win/geometry", mw.LAYOUT_STATE_KEY)


def forget_layout() -> None:
    """清掉持久化的窗口几何与停靠布局。

    每次 close(window) 都会把这两个键写回 QSettings，而 qt_app 是 session 级的
    fixture——不清的话下一条用例继承的是"上一条关掉时那个窗口"的布局，于是在文件
    前面插一条用例就可能改变后面某条的成败。这一类顺序依赖的抖动全从这两个键来。
    """
    settings = QSettings("WBAnalyzer", "WBAnalyzer")
    for key in LEAKY_LAYOUT_KEYS:
        settings.remove(key)
    settings.sync()


def make_window(qt_app, *, keep_layout: bool = False) -> MainWindow:
    # keep_layout 只给"本来就是在测布局持久化"的那几条用例用
    if not keep_layout:
        forget_layout()
    window = MainWindow()
    window.lane_spin.setValue(0)
    window.band_spin.setValue(0)  # 同样会被 QSettings 记住，逐项复位
    window.bg_combo.setCurrentIndex(1)  # 固定“深色条带”，排除自动判定的干扰
    # 显式复位会被持久化的开关，别让上一次运行留下的 QSettings 影响断言
    window.snap_check.setChecked(True)
    window._snap_enabled = True
    window.canvas.set_snap_enabled(True)
    # 界面偏好也是持久化的，而字号一改整个会话的度量都变（QApplication.setFont
    # 是进程级的）。每个窗口开局先归零，上一条用例改过什么都影响不到这一条。
    window.apply_ui_prefs(prefs_mod.DEFAULTS, persist=False)
    return window


def fake_inflight_job(window: MainWindow, kind: str) -> tuple:
    """登记一个"结果还在路上"的任务令牌，但不占用忙碌状态。

    真实场景是 worker 已经算完、回调还排在事件队列里；这时用户可以继续操作
    （旋转 / 换图）。若连 _running 一起塞，旋转会被"忙碌就拒绝"挡掉，测不到
    令牌校验本身。
    """
    window._job_serial += 1
    token = (kind, window._job_guard(kind), window._job_serial)
    window._job_tokens[kind] = token
    return token


def load_array(window: MainWindow, raw: np.ndarray) -> None:
    """跳过磁盘 IO 直接装载：等价于加载任务落地后的状态。"""
    blot = blot_from_array(raw)
    window._blot_loaded = blot
    window._blot_original = blot
    window._blot = blot
    window._angle = 0.0
    window._quarter_turns = 0
    window._orient_target = (0, 0.0)
    window._bump_document()
    window._source_version += 1
    window._roi = None
    window._separators = []
    window._undo.clear()
    window._redo.clear()
    window._committed = window._EMPTY_STATE
    window.canvas.clear_roi()
    window.canvas.set_image(blot.display_bgr)
    window._clear_result()


def settle(window: MainWindow, timeout: float = 20.0) -> bool:
    """把后台任务与防抖定时器都跑干净（剖面 → 自动定量是链式的）。"""
    app = QApplication.instance()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        pending = bool(window._running) or workers.active_count() > 0
        timers = any(
            timer.isActive()
            for timer in (window._roi_timer, window._quant_timer, window._rotate_timer)
        )
        if not pending and not timers:
            app.processEvents()
            return True
        time.sleep(0.005)
    return False


def wait_for(predicate, timeout: float = 20.0) -> bool:
    """等某个条件成立。用于人为塞了假任务、settle 等不到"全空"的场景。"""
    app = QApplication.instance()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            app.processEvents()
            return True
        time.sleep(0.005)
    return False


def pump(window: MainWindow, seconds: float = 0.05) -> None:
    """只推事件循环若干毫秒，用于观察"任务还在跑"的中间态。"""
    app = QApplication.instance()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.002)


def set_roi_and_quantify(window: MainWindow) -> None:
    x, y, inset, height = ROI_BOX
    window.canvas.set_roi(x, y, window._blot.width + inset, height)
    window._apply_roi_change()
    assert settle(window), "ROI → 剖面 → 自动定量的链条没跑完"


def bands(window: MainWindow) -> list[tuple[int, int, float]]:
    if window._result is None:
        return []
    return [
        (m.lane, m.band, round(float(m.integrated_density), 1))
        for m in window._result.measurements
    ]


def close(window: MainWindow) -> None:
    window._results_dirty = False
    window.close()


def slow(value, delay: float = 0.2):
    """返回一个"慢函数"，用来把某类任务稳定地停在"在跑"状态上。"""

    def _call(*_args, **_kwargs):
        time.sleep(delay)
        return value

    return _call


def key_event(key, modifiers=Qt.KeyboardModifier.NoModifier) -> QKeyEvent:
    return QKeyEvent(QEvent.Type.KeyPress, key, modifiers)


def add_history(
    window: MainWindow,
    kind: str,
    name: str,
    group: str = "默认",
    pick: str | None = None,
) -> None:
    window.normalize_panel.set_kind(kind)
    window.normalize_panel.set_protein_name(name)
    window.normalize_panel.set_group_name(group)
    if pick is not None:
        window.normalize_panel.set_band_pick(pick)
    window._on_add_to_history()


def render(widget, app):
    """强制渲染一次，并把 paint 里逃逸出来的异常收集成断言。

    离屏单测默认永远不触发 paintEvent，"构造得出来但画不出来"的 bug 因此
    整片测不到——把 FillBetweenItem 塞进 pyqtgraph 图例就是这么漏过去的：
    LegendItem.paint 读 item.opts 抛 AttributeError，整张图跟着画不出来。
    """
    errors: list[BaseException] = []
    original = sys.excepthook
    sys.excepthook = lambda kind, value, tb: errors.append(value)
    try:
        pixmap = widget.grab()
        app.processEvents()
    finally:
        sys.excepthook = original
    assert not errors, f"渲染 {type(widget).__name__} 时抛异常：{errors[0]!r}"
    assert not pixmap.isNull(), f"{type(widget).__name__} 渲染出空图"
    assert pixmap.width() > 0 and pixmap.height() > 0
    return pixmap


# ------------------------------------------------------------------ U1 换图竞态


def test_stale_quantify_result_is_discarded_after_image_switch(qt_app):
    """A 图的定量回调在切到同尺寸 B 图之后到达：必须整包丢弃。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0]))
    set_roi_and_quantify(window)
    a_bands = bands(window)
    assert a_bands, "前置条件：A 图应有定量结果"

    # 复刻竞态：拿一个"还在跑"的 job token 与 A 图的 payload，切图后才回调
    stale_payload = (window._result, list(window._lanes))
    stale_token = fake_inflight_job(window, mw.JOB_QUANTIFY)

    roi, separators = window._roi, list(window._separators)
    load_array(window, synthetic_raw([2000.0, 1500.0, 1000.0]))  # 同尺寸、信号弱得多
    window._roi = roi  # 沿用上一张的 ROI 与分隔线（软件的既有行为）
    window._separators = separators
    window.canvas.set_roi(roi.x, roi.y, roi.width, roi.height)
    window.canvas.set_separators(separators)
    window._quant_timer.stop()

    window._on_quantify_done(stale_token, stale_payload)

    assert bands(window) == [], "A 图的结果不能落到 B 图上"
    assert not window.act_export_data.isEnabled(), "过期结果不得开放导出"
    assert "丢弃" in window._hint_label.text(), "丢弃必须给用户可见提示"
    close(window)


def test_stale_detect_result_is_discarded_after_rotation(qt_app):
    """旋转同样换文档：自动检测的回调也必须作废。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0]))
    window._run_auto(synchronous=True)
    stale_payload = (window._result, [])
    stale_token = fake_inflight_job(window, mw.JOB_DETECT)

    window._rotate_90(clockwise=True)
    assert settle(window)
    window._on_auto_done(stale_token, stale_payload)

    assert window._result is None
    assert "丢弃" in window._hint_label.text()
    close(window)


def test_quantify_submitted_before_load_lands_is_discarded(qt_app, monkeypatch, tmp_path):
    """doc_version 的递增时机：加载期间提交的定量绝不能被新图采用。

    如果把 doc_version 改成"提交加载任务时就 +1"，此处提交的定量会拿到新版本
    号，回来时校验反而通过——U1 就退回去了。
    """
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0]))
    set_roi_and_quantify(window)
    stale_payload = (window._result, list(window._lanes))

    new_blot = blot_from_array(synthetic_raw([1000.0, 900.0, 800.0]))
    monkeypatch.setattr(mw, "load_blot", slow(new_blot, 0.25))
    target = tmp_path / "b.tif"
    target.write_bytes(b"placeholder")  # 真正的读盘被 monkeypatch 换掉了

    assert window.open_path(target) is True
    # 加载还在路上，此刻"上一张图的定量任务"仍持有旧文档的令牌
    stale_token = fake_inflight_job(window, mw.JOB_QUANTIFY)
    assert window._blot is not None
    assert wait_for(lambda: window._blot is new_blot), "加载任务应当落地"

    window._on_quantify_done(stale_token, stale_payload)
    assert bands(window) == [], "加载前提交的定量结果不得落到新图上"
    close(window)


def test_result_jobs_wait_for_pending_load(qt_app, monkeypatch, tmp_path):
    """加载大图期间点定量：不许在旧图上开跑，排队等加载落地。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0]))
    new_blot = blot_from_array(synthetic_raw([18000.0, 14000.0]))
    monkeypatch.setattr(mw, "load_blot", slow(new_blot, 0.2))
    target = tmp_path / "b.tif"
    target.write_bytes(b"placeholder")
    window.open_path(target)

    window._run_quantify()  # 手动点“ROI 定量”
    assert mw.JOB_QUANTIFY not in window._running, "加载未落地就不该开跑定量"
    assert window._quant_timer.isActive(), "应当排队重试而不是丢掉这次请求"
    assert window._quant_manual_pending, "手动请求的身份要保留"

    assert settle(window)
    assert window._blot is new_blot
    close(window)


def test_open_during_rotation_wins(qt_app, monkeypatch, tmp_path):
    """旋转过程中换图：换图生效，半途的旋转结果不许落地。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0]))
    rotated = blot_from_array(synthetic_raw([20000.0, 15000.0]))
    monkeypatch.setattr(mw, "orient_blot", slow((rotated, rotated), 0.25))
    window.rotation_spin.setValue(8.0)
    window._apply_rotation()
    assert mw.JOB_ORIENT in window._running

    new_blot = blot_from_array(synthetic_raw([3000.0]))
    monkeypatch.setattr(mw, "load_blot", slow(new_blot, 0.05))
    target = tmp_path / "c.tif"
    target.write_bytes(b"placeholder")
    window.open_path(target)
    assert mw.JOB_ORIENT not in window._running, "换图应当作废在跑的旋转"

    assert settle(window)
    assert window._blot is new_blot
    assert window._angle == 0.0, "新图的旋转角必须归零"
    assert window._blot is not rotated
    close(window)


# ------------------------------------------------------------------ U5 两类忙碌


def test_rotation_does_not_lock_inputs(qt_app, monkeypatch):
    """呈现类任务（旋转）在跑时，滑杆与参数区必须照常可用。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0]))
    blot = window._blot
    monkeypatch.setattr(mw, "orient_blot", slow((blot, blot), 0.25))
    window.rotation_spin.setValue(5.0)
    window._apply_rotation()

    assert mw.JOB_ORIENT in window._running
    assert not window._busy, "旋转不算“忙”——它不产出结果数据"
    assert window._adjust_tab.isEnabled(), "旋转期间参数区不许锁"
    assert window.rotation_slider.isEnabled(), "滑杆拖到一半绝不能被锁死"
    assert window.canvas.isEnabled(), "旋转期间画布仍可框 ROI"
    assert window._busy_bar.isVisibleTo(window.statusBar()), "但要有进度反馈"
    assert not window._cancel_button.isVisibleTo(window.statusBar()), (
        "呈现类任务不给取消按钮：最新一次自然会顶掉它"
    )
    assert window.act_open.isEnabled(), "加载 / 旋转期间允许换图（后来者顶掉前者）"
    assert settle(window)
    close(window)


def test_result_jobs_lock_inputs_and_offer_cancel(qt_app):
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0]))
    window._begin_job(mw.JOB_QUANTIFY, "正在 ROI 定量……")
    assert window._busy
    assert not window.act_open.isEnabled()
    assert not window._recent_menu.isEnabled()
    assert window._busy_bar.isVisibleTo(window.statusBar())
    assert window._cancel_button.isVisibleTo(window.statusBar())
    assert not window._adjust_tab.isEnabled(), "忙碌时参数区应锁住"
    assert not window.canvas.isEnabled(), "忙碌时画布不接受交互"
    assert not window.normalize_panel.isEnabled(), "忙碌时不能把旧结果加入历史"

    window._invalidate_jobs(mw.RESULT_JOBS)
    assert not window._busy
    assert window.act_open.isEnabled()
    assert window._adjust_tab.isEnabled()
    assert not window._busy_bar.isVisibleTo(window.statusBar())
    close(window)


def test_rapid_rotation_keeps_only_the_last_angle(qt_app, monkeypatch):
    """连续快速拖动滑杆：中间态全丢，只有最后一个角度落地。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0]))
    blot = window._blot
    applied: list[float] = []

    def fake_orient(_loaded, turns, angle):
        applied.append(float(angle))
        time.sleep(0.05)
        return blot, blot

    monkeypatch.setattr(mw, "orient_blot", fake_orient)
    for value in (2.0, 4.0, 6.0, 8.0):
        window.rotation_spin.setValue(value)
        window._apply_rotation()  # 模拟防抖到点就提交，不等上一个跑完
        pump(window, 0.01)

    assert settle(window)
    assert window._angle == 8.0, "最后一次拖动才是用户想要的角度"
    assert window.rotation_spin.value() == 8.0
    assert len(window._running) == 0
    close(window)


def test_superseded_rotation_payload_is_ignored(qt_app):
    """被顶掉的旋转任务即使回调也不许改图（令牌是最后一道护栏）。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0]))
    original = window._blot
    old_token = window._begin_job(mw.JOB_ORIENT, "正在旋转……")
    window._begin_job(mw.JOB_ORIENT, "正在旋转……")  # 新任务顶掉旧的

    other = blot_from_array(synthetic_raw([1.0]))
    window._on_orient_done(old_token, 0, 3.0, (other, other))

    assert window._blot is original, "过期的旋转结果不得替换当前图像"
    assert window._angle == 0.0
    close(window)


def test_roi_edit_during_rotation_does_not_corrupt(qt_app, monkeypatch):
    """旋转未完成就框 ROI：不崩、不串数据，剖面按最终的图算。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0]))
    blot = window._blot
    monkeypatch.setattr(mw, "orient_blot", slow((blot, blot), 0.2))
    window.rotation_spin.setValue(3.0)
    window._apply_rotation()

    x, y, inset, height = ROI_BOX
    window.canvas.set_roi(x, y, blot.width + inset, height)
    window._apply_roi_change()  # 旋转还在跑

    assert settle(window)
    assert window._angle == 3.0
    assert window._roi is not None
    assert window._doc_version >= 2
    close(window)


def test_cancel_invalidates_running_job(qt_app):
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0]))
    token = window._begin_job(mw.JOB_QUANTIFY, "正在 ROI 定量……")
    window._cancel_running_jobs()
    assert not window._busy
    assert not window._job_is_current(token)
    close(window)


def test_profile_runs_off_the_ui_thread(qt_app):
    """ROI 松手后的整图 numpy 运算走后台任务。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0]))
    x, y, inset, height = ROI_BOX
    window.canvas.set_roi(x, y, window._blot.width + inset, height)
    window._apply_roi_change()
    assert mw.JOB_PROFILE in window._running, "剖面应当在 worker 里算"
    assert not window._busy, "剖面属于呈现类任务，不许锁参数区"
    assert settle(window)
    assert window._column_profile is not None
    assert window._separators, "分隔线应按剖面峰谷划好"
    close(window)


# ------------------------------------------------------------------ U2 删除保号


def test_delete_keeps_lane_numbers_and_values(qt_app):
    """删掉中间泳道：其余条带的泳道号与积分灰度必须一个字节都不变。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0]))
    set_roi_and_quantify(window)
    before = bands(window)
    assert [b[0] for b in before] == [1, 2, 3]

    window._on_delete_rows([1])
    after = bands(window)

    assert after == [before[0], before[2]], "只能过滤，不能重测"
    assert [b[0] for b in after] == [1, 3], "泳道号不得随删除滑动"
    close(window)


def test_undo_after_delete_restores_exact_values(qt_app):
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0]))
    set_roi_and_quantify(window)
    before = bands(window)

    window._on_delete_rows([1])
    window._undo_last()

    assert bands(window) == before, "撤销必须回到原始数值"
    close(window)


def test_redo_replays_the_deletion(qt_app):
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0]))
    set_roi_and_quantify(window)
    before = bands(window)

    window._on_delete_rows([1])
    deleted = bands(window)
    window._undo_last()
    assert bands(window) == before
    assert window.act_redo.isEnabled()

    window._redo_last()
    assert bands(window) == deleted, "重做应当再次删掉同一个条带"
    assert not window.act_redo.isEnabled()
    close(window)


def test_delete_all_then_undo(qt_app):
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0]))
    set_roi_and_quantify(window)
    before = bands(window)

    window._on_delete_rows([0, 1])
    assert window._result is None

    window._undo_last()
    assert bands(window) == before
    close(window)


def test_rotate_90_is_undoable_and_keeps_history(qt_app):
    """90° 旋转不再清空撤销栈，自己也能整步撤回（含 ROI 与结果）。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0]))
    set_roi_and_quantify(window)
    before = bands(window)
    assert before

    window._rotate_90(clockwise=True)
    assert settle(window)
    assert window._quarter_turns == 1
    assert window._roi is None, "90° 转后坐标系换了，ROI 不能沿用"
    assert window._undo, "撤销栈不该被清空"

    window._undo_last()
    assert settle(window)
    assert window._quarter_turns == 0, "撤销应把朝向也转回去"
    assert window._roi is not None
    assert bands(window) == before, "ROI 与结果要跟着朝向一起回来"
    close(window)


def test_hand_drawn_band_uses_roi_baseline_and_new_number(qt_app):
    """补画条带走 quantify_lanes 同一口径，且不改动已有条带的编号与数值。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0]))
    set_roi_and_quantify(window)
    before = bands(window)
    reference = window._result.measurements[1]

    # 在第 2 泳道原位再画一个几乎相同的框：数值应与 ROI 定量结果同量级
    window._on_band_drawn(
        reference.x, reference.y, reference.width, reference.height
    )
    after = window._result.measurements

    assert [(m.lane, m.band, round(float(m.integrated_density), 1)) for m in after[:3]] == before
    added = after[-1]
    assert added.lane == 2, "泳道号应由分隔线序号决定"
    assert added.band == 2, "新条带取该泳道的下一个编号，不重排已有条带"
    ratio = added.integrated_density / reference.integrated_density
    assert 0.9 < ratio < 1.1, f"手绘条带与 ROI 定量口径不一致：比值 {ratio}"
    close(window)


def test_hand_drawn_band_without_roi_uses_native_local_background(qt_app):
    """没有 ROI 时补画条带走 measure_box_native（环形中值局部背景）。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0]))
    window._run_auto(synchronous=True)
    assert window._roi is None, "自动检测不产生 ROI"
    before = len(window._result.measurements) if window._result is not None else 0

    window._on_band_drawn(25.0, 100.0, 70.0, 40.0)

    assert len(window._result.measurements) == before + 1
    assert float(window._result.measurements[-1].integrated_density) > 0.0
    close(window)


def test_lane_numbers_survive_delete_for_normalization(qt_app):
    """归一化取的是 measurements 的泳道号：删除后必须仍能标识原泳道。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0, 8000.0]))
    set_roi_and_quantify(window)
    window._on_delete_rows([1])
    lanes = [int(m.lane) for m in window._result.measurements]
    assert lanes == [1, 3, 4]
    assert len(set(lanes)) == len(lanes), "泳道号不得重复"
    close(window)


def test_quantify_roi_profiles_each_lane_once(qt_app, monkeypatch):
    """表里的数字和右图的剖面出自同一次 compute_lane_profiles，不许各算各的。"""
    blot = blot_from_array(synthetic_raw([20000.0, 15000.0, 10000.0]))
    roi = RoiRect(15, 60, blot.width - 30, 130)
    calls: list[int] = []
    real = mw.quant.compute_lane_profiles

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(mw.quant, "compute_lane_profiles", counted)
    result, lanes, report = mw.quantify_roi(blot, roi, [107.0, 202.0], True)

    assert len(calls) == 1, "一次 ROI 定量只该算一遍泳道剖面"
    assert report is not None, "自检要和定量在同一次后台任务里跑完"
    assert [lane.lane for lane in lanes] == [1, 2, 3]
    for m in result.measurements:
        lane = lanes[m.lane - 1]
        assert m.integrated_density == round(lane.band_densities[m.band - 1], 3)


def test_lane_profile_fill_follows_deletion(qt_app):
    """删条带后右图的积分填充区要跟着少一块，而不是留在图上。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0]))
    set_roi_and_quantify(window)
    panel = window.profile_panel
    assert panel._lanes, "前置条件：应有泳道剖面"
    kept = len(window._result.measurements)
    assert kept >= 2
    assert len(panel._measurements) == kept

    first = window._result.measurements[0]
    lane = next(l for l in panel._lanes if int(l.lane) == int(first.lane))
    size = int(np.asarray(lane.profile).size)
    assert panel._fill_segments(lane, size, float(lane.y0)), "删之前该泳道有积分区"

    window._on_delete_rows([0])
    assert len(panel._measurements) == kept - 1, "面板应拿到删除后的条带列表"
    assert panel._fill_segments(lane, size, float(lane.y0)) == [], (
        "被删条带的积分填充区必须从图上消失，不能靠旧 lanes 继续画"
    )
    close(window)


def test_profile_panel_actually_paints(qt_app):
    """真的渲染一遍两张剖面图（含图例、分区标签、积分填充区）。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0]))
    set_roi_and_quantify(window)
    panel = window.profile_panel
    assert panel._lanes

    render(panel, qt_app)
    render(panel, qt_app)  # 第二遍：图例项不该一轮轮堆积后画炸
    window._on_delete_rows([0])
    render(panel, qt_app)
    close(window)


def test_canvas_actually_paints_overlays(qt_app):
    """画布叠加层（ROI 手柄、分隔线、条带框、选中态）都要真的画得出来。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0]))
    set_roi_and_quantify(window)
    canvas = window.canvas
    assert canvas.separators(), "前置条件：应有分隔线"
    assert window._result.measurements, "前置条件：应有条带框"

    render(canvas, qt_app)
    canvas.select_bands([0])
    render(canvas, qt_app)
    window._on_delete_rows([0])
    render(canvas, qt_app)
    close(window)


def test_main_window_actually_paints_every_tab(qt_app):
    """整窗渲染一遍，三个页签逐一过：构造成功不等于画得出来。"""
    window = make_window(qt_app)
    window.resize(1280, 800)
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0]))
    set_roi_and_quantify(window)
    add_history(window, "loading_control", "Actin")
    add_history(window, "target", "PAD4")
    assert window.normalize_panel.result_table.rowCount() > 0

    for index in range(window._tabs.count()):
        window._tabs.setCurrentIndex(index)
        qt_app.processEvents()
        render(window, qt_app)
    close(window)


# ------------------------------------------------------------------ 历史记录一致性


def test_history_intensity_reflects_deleted_band(qt_app):
    """删掉一个条带后加入历史：强度是**剩余**条带的合计，不是删除前的总和。"""
    window = make_window(qt_app)
    load_array(
        window, synthetic_raw([20000.0, 15000.0, 10000.0], centers=(95.0, 155.0))
    )
    set_roi_and_quantify(window)
    measurements = window._result.measurements
    assert sorted((m.lane, m.band) for m in measurements) == [
        (1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2),
    ], "前置条件：每个泳道两条带"

    before = sum(float(m.integrated_density) for m in measurements if m.lane == 2)
    victim = next(i for i, m in enumerate(measurements) if (m.lane, m.band) == (2, 2))
    dropped = float(measurements[victim].integrated_density)
    lane2 = next(lane for lane in window._lanes if int(lane.lane) == 2)

    window._on_delete_rows([victim])
    expected = sum(
        float(m.integrated_density) for m in window._result.measurements if m.lane == 2
    )
    assert expected == pytest.approx(before - dropped)

    add_history(window, "target", "PAD4")
    entry = window._session.entries[-1]
    assert entry.lanes == [1, 2, 3], "泳道号要跟着强度一起记下来"
    value = entry.value_by_lane()[2]
    assert value == pytest.approx(expected), "历史强度必须按当前 measurements 聚合"
    assert value < before, "不能把已删除的条带继续算进去"
    assert value < float(lane2.total_density), (
        "更不能读 lane.total_density——那是删除前整条泳道的积分"
    )
    # 剖面图标题同源：删完之后显示的也得是剩余条带的合计
    assert window.profile_panel._lane_total(lane2) == pytest.approx(expected)
    close(window)


def test_history_pairs_by_lane_after_deleting_a_whole_lane(qt_app):
    """整条泳道被删空：归一化按泳道号配对，后面的泳道不许整体错位。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 16000.0, 12000.0, 8000.0]))
    set_roi_and_quantify(window)
    assert [int(m.lane) for m in window._result.measurements] == [1, 2, 3, 4]
    add_history(window, "loading_control", "Actin")

    # 同一张图当目标膜，但泳道 2 被判为误检删掉
    victim = next(i for i, m in enumerate(window._result.measurements) if m.lane == 2)
    window._on_delete_rows([victim])
    add_history(window, "target", "PAD4")

    rows = {int(r.lane): r for r in mw.normalize_rows(window._session, 1)}
    assert sorted(rows) == [1, 2, 3, 4], "缺配对的泳道要成行报告，不能悄悄消失"
    for lane in (1, 3, 4):
        assert rows[lane].normalized == pytest.approx(1.0, rel=1e-6), (
            f"泳道 {lane} 同图配同图应为 1.0；不是 1.0 说明配对错位了"
        )
    assert rows[2].target is None and rows[2].note, "泳道 2 没有目标值，要留空并说明"
    assert window._session.last_warnings, "错位风险要有可见的警告"

    # 空值不能把结果表写崩，缺配对的泳道要在界面上说清楚
    window._refresh_normalize()
    panel = window.normalize_panel
    assert panel.result_table.rowCount() == 4
    assert panel.result_table.item(1, 2).text() == "—", "没配上的目标值显示为空"
    hint = panel._result_hint
    assert "泳道 2" in hint.text()
    assert hint.isVisibleTo(panel), "有结果但有泳道没配上时，警告也得露出来"
    render(panel, qt_app)
    close(window)


def test_below_detection_is_visible_in_results_table(qt_app):
    """低于检出限的条带要在结果表里看得见：S/N 列 + 压暗 + 计数提示 + 悬停说明。"""
    window = make_window(qt_app)
    # 一强一弱：弱带只有噪声量级，S/N 必然过不了 3。噪声大到自动分道不可靠，
    # 所以明确指定 2 个泳道，测的是检出限而不是分隔线放置。
    load_array(window, noisy_raw([20000.0, 8.0], noise=150.0))
    window.lane_spin.setValue(2)
    set_roi_and_quantify(window)
    measurements = window._result.measurements
    by_lane = {int(m.lane): m for m in measurements}
    assert not by_lane[1].below_detection and by_lane[2].below_detection

    panel = window.results_panel
    assert panel.table.columnCount() == len(rp.HEADERS)
    assert rp.HEADERS[-1] == "S/N"
    weak_row = next(
        row
        for row in range(panel.table.rowCount())
        if panel.table.item(row, 0).data(Qt.ItemDataRole.EditRole) == 2
    )
    cell = panel.table.item(weak_row, 6)  # 积分灰度
    assert cell.foreground().color() == rp.BELOW_DETECTION_COLOR, "低于检出限的行要压暗"
    assert "低于检出限" in cell.toolTip()
    assert "低于检出限" in panel._count_label.text()
    snr_cell = panel.table.item(weak_row, len(rp.HEADERS) - 1)
    assert float(snr_cell.data(Qt.ItemDataRole.EditRole)) < 3.0

    # 画布上的框也要区分开，并且真的画得出来
    weak_index = measurements.index(by_lane[2])
    assert window.canvas._band_items[weak_index]._base == canvas_mod.RESULT_WEAK_COLOR
    assert "低于检出限" in window.canvas._band_items[weak_index].toolTip()
    render(window.canvas, qt_app)
    render(panel, qt_app)
    close(window)


def test_below_detection_warns_when_normalizing(qt_app):
    """低于检出限的泳道拿去做比值没有意义——加入历史与归一化都要出声。"""
    window = make_window(qt_app)
    load_array(window, noisy_raw([20000.0, 8.0], noise=150.0))
    window.lane_spin.setValue(2)
    set_roi_and_quantify(window)
    add_history(window, "loading_control", "Actin")
    assert "低于检出限" in window._hint_label.text(), "加入历史时就该提醒"
    entry = window._session.entries[-1]
    assert entry.below_detection == [False, True]
    assert entry.weak_lanes() == {2}

    add_history(window, "target", "PAD4")
    rows = {int(r.lane): r for r in mw.normalize_rows(window._session, 1)}
    assert rows[1].note == "", "强泳道不该被打标"
    assert rows[2].note == "低于检出限"
    assert any("低于检出限" in w for w in window._session.last_warnings)

    window._refresh_normalize()
    panel = window.normalize_panel
    assert "低于检出限" in panel._result_hint.text()
    assert panel._result_hint.isVisibleTo(panel)
    render(panel, qt_app)
    close(window)


def test_history_entry_carries_doc_version(qt_app, monkeypatch):
    """历史项记下测量时的图像版本，且旋转落地前不许追加。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0]))
    set_roi_and_quantify(window)
    add_history(window, "target", "PAD4")
    entry = window._session.entries[-1]
    assert entry.doc_version == window._doc_version >= 0

    blot = window._blot
    monkeypatch.setattr(mw, "orient_blot", slow((blot, blot), 0.25))
    window.rotation_spin.setValue(5.0)
    window._apply_rotation()
    assert mw.JOB_ORIENT in window._running, "前置条件：旋转应当还在路上"

    depth = len(window._session.entries)
    window.normalize_panel.set_protein_name("PAD4-2")
    window._on_add_to_history()
    assert len(window._session.entries) == depth, "旋转未落地时不许把旧朝向的数字入库"
    assert "旋转" in window._hint_label.text()

    assert settle(window)
    close(window)


def test_history_edit_keeps_lane_identity(qt_app):
    """在历史表里改名 / 改类型，不能顺手把泳道号和版本号抹掉。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0]))
    set_roi_and_quantify(window)
    victim = next(i for i, m in enumerate(window._result.measurements) if m.lane == 2)
    window._on_delete_rows([victim])
    add_history(window, "target", "PAD4")
    before = window._session.entries[-1]
    assert before.lanes == [1, 3]

    window._on_history_edited(0, "loading_control", "Actin", "第二批")
    after = window._session.entries[0]
    assert (after.kind, after.name, after.group) == ("loading_control", "Actin", "第二批")
    assert after.lanes == before.lanes
    assert after.doc_version == before.doc_version
    assert after.intensities == before.intensities
    close(window)


# ------------------------------------------------------------------ 交互增强


def test_arrow_keys_nudge_roi_by_one_pixel(qt_app):
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0]))
    set_roi_and_quantify(window)
    canvas = window.canvas
    canvas.set_nudge_target(("roi", -1))
    x0 = canvas.roi()[0]

    canvas.keyPressEvent(key_event(Qt.Key.Key_Right))
    assert canvas.roi()[0] == pytest.approx(x0 + 1.0)

    canvas.keyPressEvent(key_event(Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier))
    assert canvas.roi()[0] == pytest.approx(x0 + 11.0), "Shift 步长应为 10 px"

    canvas.keyPressEvent(key_event(Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier))
    canvas.keyPressEvent(key_event(Qt.Key.Key_Left))
    assert canvas.roi()[0] == pytest.approx(x0)
    assert settle(window)
    close(window)


def test_arrow_keys_nudge_focused_separator(qt_app):
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0]))
    set_roi_and_quantify(window)
    canvas = window.canvas
    assert canvas.separators(), "前置条件：应有分隔线"
    canvas.set_nudge_target(("sep", 0))
    before = canvas.separators()[0]

    canvas.keyPressEvent(key_event(Qt.Key.Key_Right))
    assert canvas.separators()[0] == pytest.approx(before + 1.0)
    assert settle(window)
    # 分隔线动过就要重新定量，不能留着旧结果假装还准
    assert window._separators[0] == pytest.approx(before + 1.0)
    close(window)


def test_ctrl_arrow_resizes_roi(qt_app):
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0]))
    set_roi_and_quantify(window)
    canvas = window.canvas
    width = canvas.roi()[2]
    canvas.keyPressEvent(key_event(Qt.Key.Key_Right, Qt.KeyboardModifier.ControlModifier))
    assert canvas.roi()[2] == pytest.approx(width + 1.0)
    assert settle(window)
    close(window)


def test_roi_fields_are_two_way_bound(qt_app):
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0]))
    set_roi_and_quantify(window)
    spins = window.roi_spins
    assert spins["x"].value() == window._roi.x
    assert spins["w"].value() == window._roi.width

    spins["x"].setValue(window._roi.x + 7)
    assert window.canvas.roi()[0] == pytest.approx(spins["x"].value())
    assert settle(window)
    assert window._roi.x == spins["x"].value(), "数值框改动要落到定量用的 ROI 上"

    # 反向：画布动了，数值框跟手
    window.canvas.set_nudge_target(("roi", -1))
    window.canvas.nudge(3.0, 0.0)
    assert spins["x"].value() == pytest.approx(window.canvas.roi()[0])
    assert settle(window)
    close(window)


def test_separator_snaps_only_within_tolerance(qt_app):
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0]))
    canvas = window.canvas
    canvas.set_roi(10, 10, 300, 100)
    canvas.set_separators([100.0])
    canvas.set_snap_enabled(True)
    canvas.set_snap_targets([120.0])

    canvas._drag = {"kind": "sep", "index": 0, "orig_seps": [100.0]}
    canvas._drag_separator(QPointF(117.0, 50.0))
    assert canvas.separators()[0] == pytest.approx(120.0), "近距离应吸附到谷底"

    canvas._drag = {"kind": "sep", "index": 0, "orig_seps": [120.0]}
    canvas._drag_separator(QPointF(200.0, 50.0))
    assert canvas.separators()[0] == pytest.approx(200.0), "明显放在别处时不许硬拽"

    canvas._drag = {"kind": "sep", "index": 0, "orig_seps": [200.0]}
    canvas._drag_separator(QPointF(117.0, 50.0), snap=False)
    assert canvas.separators()[0] == pytest.approx(117.0), "Alt / 关闭吸附时完全跟手"
    close(window)


def test_snap_targets_come_from_profile_valleys(qt_app):
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0]))
    set_roi_and_quantify(window)
    assert window.canvas.snap_targets(), "剖面算完应当给出吸附目标"

    window.snap_check.setChecked(False)
    assert window.canvas.snap_targets() == [], "关掉开关后不再吸附"
    close(window)


def test_roi_drag_marks_stale_instead_of_clearing(qt_app):
    """拖 ROI 只标记过期：无条件清空会让结果表闪一下。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0]))
    set_roi_and_quantify(window)
    before = bands(window)

    window._on_canvas_roi_changed(20.0, 62.0, window._blot.width - 30.0, 128.0)
    assert bands(window) == before, "拖动过程中不该清空结果表"
    assert window._result_stale
    assert not window.act_export_data.isEnabled(), "过期结果禁止导出"
    assert settle(window)
    close(window)


def test_canvas_takes_focus_after_load(qt_app, monkeypatch, tmp_path):
    window = make_window(qt_app)
    window.show()
    blot = blot_from_array(synthetic_raw([20000.0, 15000.0]))
    monkeypatch.setattr(mw, "load_blot", slow(blot, 0.01))
    target = tmp_path / "a.tif"
    target.write_bytes(b"placeholder")
    window.open_path(target)
    assert settle(window)
    assert window.canvas.hasFocus(), "加载完应自动聚焦画布，Space / Delete 才直接生效"
    close(window)


def test_ruler_clamp_is_reported(qt_app):
    """标尺要求超出 ±45° 时必须出声，不能假称已经旋正。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0]))
    window._ruler_angle = 62.0
    window._apply_ruler("horizontal")
    text = window._hint_label.text()
    assert "⚠" in text and "超出" in text
    assert "旋转 90" in text, "应指路到 90° 旋转"
    assert window.rotation_spin.value() == pytest.approx(MAX_STRAIGHTEN_ANGLE)
    assert settle(window)
    close(window)


def test_ruler_within_widened_limit_is_applied(qt_app):
    """±45° 之内不再钳制：40° 是新放宽后必须能真正转到的角度。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0]))
    window._ruler_angle = 40.0
    window._apply_ruler("horizontal")
    assert window.rotation_spin.value() == pytest.approx(40.0)
    assert "⚠" not in window._hint_label.text()
    assert settle(window)
    close(window)


def test_export_asks_before_overwriting(qt_app, monkeypatch, tmp_path):
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0]))
    set_roi_and_quantify(window)
    (tmp_path / "blot_bands.csv").write_text("x", encoding="utf-8")

    assert window._existing_exports(str(tmp_path), "blot") == ["blot_bands.csv"]
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No
    )
    assert window._confirm_overwrite(str(tmp_path), "blot") is False
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    assert window._confirm_overwrite(str(tmp_path), "blot") is True
    assert window._confirm_overwrite(str(tmp_path), "other") is True, "无冲突就不该打扰"
    close(window)


def test_help_dialog_lists_shortcuts_and_version(qt_app):
    window = make_window(qt_app)
    window._show_help()
    dialog = window._help_box
    assert dialog is not None
    tabs = dialog.findChildren(type(window._tabs))
    assert tabs, "帮助窗口应当分页"
    labels = [tabs[0].tabText(i) for i in range(tabs[0].count())]
    assert "快捷键" in labels and "关于" in labels
    table = tabs[0].widget(labels.index("快捷键"))
    keys = [
        table.item(row, 0).text()
        for row in range(table.rowCount())
        if table.item(row, 0) is not None
    ]
    assert "Ctrl+Z" in keys and "Shift+方向键" in keys
    about = tabs[0].widget(labels.index("关于")).toPlainText()
    assert mw._app_version() in about
    dialog.hide()
    close(window)


def test_tool_change_alone_does_not_create_undo_step(qt_app):
    """切工具要能被撤销恢复，但不该自己占一格撤销步骤。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0]))
    set_roi_and_quantify(window)
    depth = len(window._undo)
    window._select_tool(Tool.BAND)
    window._commit_snapshot()
    assert len(window._undo) == depth, "只切工具不该新增撤销步骤"
    close(window)


# ------------------------------------------------------------------ U3 / U4 / U6


def test_close_cancels_workers(qt_app, monkeypatch):
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0]))
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    window._begin_job(mw.JOB_QUANTIFY, "正在 ROI 定量……")
    assert window.close() is not False
    assert workers.active_count() == 0
    assert not window._job_tokens
    assert not window._running


def test_close_asks_before_dropping_unexported_results(qt_app, monkeypatch):
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0]))
    set_roi_and_quantify(window)
    assert window._results_dirty

    asked = {"count": 0, "answer": QMessageBox.StandardButton.No}

    def ask(*_args, **_kwargs):
        asked["count"] += 1
        return asked["answer"]

    monkeypatch.setattr(QMessageBox, "question", ask)
    assert window.close() is False, "用户选了“否”时不应关闭"
    assert asked["count"] == 1
    assert window._result is not None, "被挽留后结果必须还在"

    asked["answer"] = QMessageBox.StandardButton.Yes
    assert window.close() is True
    assert asked["count"] == 2


def test_large_image_still_gets_a_preview(qt_app):
    """超大图降采样上屏，scene 坐标仍等于原图像素坐标。"""
    window = make_window(qt_app)
    canvas = window.canvas
    step = canvas._display_step(40000, 30000)
    assert step > 1
    small = np.zeros((600, 800, 3), dtype=np.uint8)
    assert canvas.set_image(small)
    assert canvas._image_w == 800
    assert canvas.sceneRect().width() == 800
    close(window)


def test_broken_image_is_reported_and_logged(qt_app, monkeypatch, tmp_path):
    """打开坏文件不能从槽里抛出去，而且要留下日志。"""
    window = make_window(qt_app)
    shown: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "critical", lambda *args, **kwargs: shown.append(str(args[2]))
    )
    broken = tmp_path / "broken.tif"
    broken.write_bytes(b"not a tiff at all")

    assert window.open_path(broken, synchronous=True) is False
    assert shown, "必须给出中文错误提示"
    assert "无法打开" in window._hint_label.text()
    assert not window._running, "失败也要把忙碌状态收干净"

    log_file = errorlog.log_path()
    assert log_file is not None and log_file.exists()
    assert "打开图片失败" in log_file.read_text(encoding="utf-8")
    close(window)


def test_broken_image_reported_from_worker(qt_app, monkeypatch, tmp_path):
    """异步路径同样要兜住：worker 里抛的异常走 error 回调，不是崩溃。"""
    window = make_window(qt_app)
    shown: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "critical", lambda *args, **kwargs: shown.append(str(args[2]))
    )
    broken = tmp_path / "broken2.tif"
    broken.write_bytes(b"still not a tiff")
    window.open_path(broken)
    assert settle(window)
    assert shown
    assert "无法打开" in window._hint_label.text()
    close(window)


def test_number_formatting_is_not_scientific():
    assert format_number(95952170.0) == "95,952,170"
    assert format_number(0.0) == "0"
    assert format_number(1.5) == "1.5"
    assert format_number(None) == "—"
    assert "e" not in format_number(123456789.123)


def test_worker_cancel_suppresses_callbacks(qt_app):
    worker = workers.Worker(lambda: "结果")
    calls: list[object] = []
    worker.signals.finished.connect(calls.append)
    worker.cancel()
    worker.run()
    qt_app.processEvents()
    assert calls == []
    assert workers.active_count() == 0


# ------------------------------------------------------------------ U6 两条口径互斥


def test_auto_detect_clears_the_roi_path_overlay(qt_app):
    """点“自动检测”＝换一套背景口径：ROI、分隔线与 ROI 定量结果一起下画布。

    两套结果同时挂在图上时，用户分不出手里的数字出自哪套口径——README 方法学
    第 9 节明确写着它们不可混用。
    """
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0]))
    set_roi_and_quantify(window)
    assert window.canvas.separators(), "前置条件：ROI 定量应当画出分隔线"
    assert window._result_origin == mw.ORIGIN_ROI

    window._run_auto(synchronous=True)
    assert settle(window)

    assert window._result_origin == mw.ORIGIN_DETECT
    assert window._roi is None, "自动检测口径下不该还留着 ROI"
    assert window.canvas.roi() is None
    assert window.canvas.separators() == [], "橙色分隔线属于 ROI 定量那条路"
    assert window.profile_panel._profile is None, "ROI 横向剖面也该一起清掉"
    close(window)


def test_switching_to_auto_detect_is_one_undo_step(qt_app):
    """切换口径是一次真正的状态变更：Ctrl+Z 一步把 ROI 和它的结果一起找回来。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0]))
    set_roi_and_quantify(window)
    before = bands(window)
    roi_before = window._roi
    seps_before = list(window._separators)
    undo_depth = len(window._undo)

    window._run_auto(synchronous=True)
    assert settle(window)
    assert len(window._undo) == undo_depth + 1, "切换口径只该压一步撤销"

    window._undo_last()
    assert settle(window)
    assert window._roi == roi_before
    assert window._separators == seps_before
    assert bands(window) == before, "撤销要连同 ROI 定量的原始数值一起回来"
    assert window._result_origin == mw.ORIGIN_ROI
    close(window)


def test_new_roi_drops_detect_boxes_before_quantifying(qt_app):
    """画上 ROI 的那一刻自动检测的框就得消失，不能等定量算完才换。

    定量要等剖面任务 + 400 ms 防抖，大图上是好几秒；这期间画布上会同时挂着
    "新 ROI + 上一轮检测框"，而且定量失败的话它们再也不会被换掉。
    """
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0]))
    window._run_auto(synchronous=True)
    assert window._result is not None and window._result.measurements
    assert window.canvas._band_items, "前置条件：自动检测应当画出条带框"

    x, y, inset, height = ROI_BOX
    window.canvas.set_roi(x, y, window._blot.width + inset, height)
    window._apply_roi_change()
    # 还没 settle：此刻定量必然没算完
    assert window.canvas._band_items == [], "旧口径的条带框必须当场清掉"
    assert window._result is None

    assert settle(window)
    assert window._result_origin == mw.ORIGIN_ROI
    close(window)


def test_hand_drawn_band_refuses_to_mix_background_conventions(qt_app):
    """ROI 口径下补画的框量不了时宁可不加，也不能退回局部背景凑一个数。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0]))
    set_roi_and_quantify(window)
    before = bands(window)

    # 1 px 宽的框：measure_band_roi_style 给不出结果
    window._on_band_drawn(float(window._blot.width - 1), 100.0, 1.0, 20.0)

    assert bands(window) == before, "不可比的数值不许混进同一张结果表"
    assert "口径" in window._hint_label.text()
    close(window)


def test_history_flags_mixed_background_conventions(qt_app):
    """同一分组里混进两套口径的历史记录：必须显式警告，不能默默算比值。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0]))
    set_roi_and_quantify(window)
    add_history(window, "loading_control", "Actin", "G1")
    assert window.normalize_panel.history_table.item(0, 4).text() == "ROI 基线"

    window._run_auto(synchronous=True)
    assert settle(window)
    assert window._result is not None and window._result.measurements
    add_history(window, "target", "PAD4", "G1")

    assert window._mixed_origin_groups() == ["G1"]
    assert window.normalize_panel.history_table.item(1, 4).text() == "局部背景"
    hint = window.normalize_panel._result_hint.text()
    assert "口径" in hint and "G1" in hint
    close(window)


# ------------------------------------------------------------------ U7 条带标签可读性


def fake_bands(count: int, *, x0: float = 20.0, pitch: float = 45.0, width: float = 36.0):
    """一排等间距窄条带——标签比条带还宽，正是用户那张图的形态。

    直接造 BandMeasurement 而不跑定量：标签排布是纯绘制问题，不该被泳道划分
    算法的改动带偏。
    """
    out = []
    for i in range(count):
        out.append(
            mw.BandMeasurement(
                lane=i + 1,
                band=1,
                x=int(x0 + i * pitch),
                y=100,
                width=int(width),
                height=30,
                area=int(width) * 30,
                mean_intensity=100.0,
                background=10.0,
                integrated_density=1000.0 * (i + 1),
            )
        )
    return out


def canvas_with_bands(qt_app, count: int = 9, *, extra_stacked: bool = True):
    canvas = canvas_mod.BlotCanvas()
    canvas.resize(900, 500)
    canvas.set_image(np.full((240, 60 + count * 45, 3), 200, dtype=np.uint8))
    measurements = fake_bands(count)
    if extra_stacked:
        # 同一泳道纵向叠放的第二条带：标签会和上面那条抢位置
        first = measurements[0]
        measurements.append(
            mw.BandMeasurement(
                lane=1,
                band=2,
                x=first.x,
                y=first.y + 40,
                width=first.width,
                height=first.height,
                area=first.area,
                mean_intensity=90.0,
                background=10.0,
                integrated_density=800.0,
            )
        )
    canvas.show_result_boxes(measurements)
    qt_app.processEvents()
    return canvas, measurements


def label_overlaps(canvas) -> list[tuple[int, int]]:
    rects = canvas.band_label_rects()
    return [
        (i, j)
        for i in range(len(rects))
        for j in range(i + 1, len(rects))
        if rects[i].intersects(rects[j])
    ]


def zoom_to(canvas, scale: float) -> None:
    canvas.zoom_100()
    canvas.scale(scale, scale)
    canvas._on_scale_changed()


def test_band_labels_never_overlap_at_any_zoom(qt_app):
    """25% / 51% / 100% / 200% 各真渲染一遍：要么清晰不重叠，要么干脆不画。

    这是纯绘制改动，离屏断言不触发 paint，必须 grab() 一次才算测到。
    """
    canvas, measurements = canvas_with_bands(qt_app)
    for scale in (0.25, 0.51, 1.0, 2.0):
        zoom_to(canvas, scale)
        render(canvas, qt_app)
        overlaps = label_overlaps(canvas)
        assert not overlaps, f"{scale:.0%} 下标签互相压住：{overlaps}"
        visible = canvas.band_label_rects()
        assert len(visible) in (0, len(measurements)), (
            f"{scale:.0%} 下只显示了一部分标签，用户无法预期哪几个会留下"
        )
    canvas.deleteLater()


def test_band_labels_stay_inside_the_image(qt_app):
    """最左 / 最右那个条带的标签不许探出图像边界。"""
    canvas, _ = canvas_with_bands(qt_app, count=4, extra_stacked=False)
    for scale in (0.51, 1.0, 3.0):
        zoom_to(canvas, scale)
        render(canvas, qt_app)
        left_edge = canvas.mapFromScene(QPointF(0.0, 0.0)).x()
        right_edge = canvas.mapFromScene(QPointF(float(canvas._image_w), 0.0)).x()
        for rect in canvas.band_label_rects():
            assert rect.left() >= left_edge - 1, "标签探出了图像左边界"
            assert rect.right() <= right_edge + 1, "标签探出了图像右边界"
    canvas.deleteLater()


def test_crowded_labels_degrade_then_hide(qt_app):
    """从大到小走一遍缩放：完整文案 → 只剩泳道号 → 整批隐藏，全程不许重叠。

    具体在哪个倍率换档取决于字体度量，不写死；要保证的是三档都出现过，而且
    任何一档都不会出现"糊成一团"的中间态。
    """
    canvas, _ = canvas_with_bands(qt_app)
    seen: list[tuple[bool, bool]] = []
    for scale in (2.0, 1.5, 1.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.2, 0.1):
        zoom_to(canvas, scale)
        render(canvas, qt_app)
        assert not label_overlaps(canvas), f"{scale:.0%} 下标签重叠了"
        if canvas.labels_packed() and canvas.labels_compact():
            assert canvas._band_labels[0].text() == "L1"
        seen.append((canvas.labels_packed(), canvas.labels_compact()))

    assert seen[0] == (True, False), "放得开时应当显示完整的 L#-B#"
    assert (True, True) in seen, "中间应当有一档退化成只写泳道号"
    assert seen[-1][0] is False, "缩到最小就整批隐藏"
    assert canvas.band_label_rects() == [], "隐藏这一档一个标签都不该画"
    canvas.deleteLater()


def test_selected_band_label_shows_even_when_hidden(qt_app):
    """整批隐藏之后，选中的那几个仍要能看清是谁。"""
    canvas, measurements = canvas_with_bands(qt_app)
    zoom_to(canvas, 0.1)
    assert canvas.band_label_rects() == []

    canvas.select_bands([0, 8])
    qt_app.processEvents()
    render(canvas, qt_app)
    assert len(canvas.band_label_rects()) == 2, "选中的条带标签要露出来"
    assert not label_overlaps(canvas)

    # 全选到摆不开时也不能退回一团糊
    canvas.select_bands(list(range(len(measurements))))
    qt_app.processEvents()
    render(canvas, qt_app)
    assert not label_overlaps(canvas)
    canvas.deleteLater()


# ------------------------------------------------------------------ U8 框选多选与批量删除


def mouse_event(canvas, kind, scene_x, scene_y, modifiers=Qt.KeyboardModifier.NoModifier):
    pos = QPointF(canvas.mapFromScene(QPointF(float(scene_x), float(scene_y))))
    return QMouseEvent(
        kind,
        pos,
        pos,
        QPointF(canvas.viewport().mapToGlobal(pos.toPoint())),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        modifiers,
    )


def rubber_band(canvas, start, end, modifiers=Qt.KeyboardModifier.NoModifier) -> None:
    canvas.mousePressEvent(
        mouse_event(canvas, QEvent.Type.MouseButtonPress, *start, modifiers)
    )
    canvas.mouseMoveEvent(
        mouse_event(canvas, QEvent.Type.MouseMove, *end, modifiers)
    )
    canvas.mouseReleaseEvent(
        mouse_event(canvas, QEvent.Type.MouseButtonRelease, *end, modifiers)
    )


def test_rubber_band_selects_many_bands_in_adjust_mode(qt_app):
    """“调整”模式下在 ROI 外拖一个框：框到的条带全部进入选中态。"""
    canvas, measurements = canvas_with_bands(qt_app, count=9, extra_stacked=False)
    canvas.set_tool(Tool.ADJUST)
    canvas.set_roi(10, 90, 400, 50)
    reported: list[list[int]] = []
    canvas.bandSelectionChanged.connect(reported.append)

    third = measurements[2]
    end = (third.x + third.width + 2, 200)
    canvas.mousePressEvent(mouse_event(canvas, QEvent.Type.MouseButtonPress, 5, 20))
    canvas.mouseMoveEvent(mouse_event(canvas, QEvent.Type.MouseMove, *end))
    render(canvas, qt_app)  # 选择框本身也是新画的图元，得真画一遍
    canvas.mouseReleaseEvent(
        mouse_event(canvas, QEvent.Type.MouseButtonRelease, *end)
    )

    assert canvas.selected_bands() == [0, 1, 2]
    assert reported == [[0, 1, 2]], "拖动过程中不该反复往结果表发信号"
    render(canvas, qt_app)
    canvas.deleteLater()


def test_rubber_band_adds_with_ctrl_inside_the_roi(qt_app):
    """ROI 内空白默认还是整体拖 ROI；按住 Ctrl 才改成框选加选。"""
    canvas, measurements = canvas_with_bands(qt_app, count=9, extra_stacked=False)
    canvas.set_tool(Tool.ADJUST)
    canvas.set_roi(0, 80, 500, 80)

    # 无修饰键：ROI 内空白处（避开条带框）拖拽仍是移动 ROI
    canvas.mousePressEvent(
        mouse_event(canvas, QEvent.Type.MouseButtonPress, 240, 150)
    )
    assert canvas._drag is not None and canvas._drag["kind"] == "move_roi"
    canvas._cancel_drag()

    canvas.select_bands([7])
    first, second = measurements[0], measurements[1]
    rubber_band(
        canvas,
        (first.x - 4, 90),
        (second.x + second.width + 2, 150),
        Qt.KeyboardModifier.ControlModifier,
    )
    assert canvas.selected_bands() == [0, 1, 7], "Ctrl 框选是加选，不该丢掉原有选中"
    canvas.deleteLater()


def test_rubber_band_does_not_steal_other_tools_drags(qt_app):
    """框选只在“调整”模式生效：画 ROI / 补画条带 / 标尺的拖拽不能被抢走。"""
    canvas, _ = canvas_with_bands(qt_app, count=4, extra_stacked=False)
    for tool, kind in (
        (Tool.ROI, "new_roi"),
        (Tool.BAND, "new_band"),
        (Tool.RULER, "ruler"),
    ):
        canvas.set_tool(tool)
        canvas.mousePressEvent(
            mouse_event(canvas, QEvent.Type.MouseButtonPress, 5, 20)
        )
        assert canvas._drag is not None and canvas._drag["kind"] == kind
        canvas._cancel_drag()
    canvas.deleteLater()


def test_ctrl_click_toggles_and_shift_click_adds(qt_app):
    canvas, measurements = canvas_with_bands(qt_app, count=4, extra_stacked=False)
    canvas.set_tool(Tool.ADJUST)

    def click(index, modifiers=Qt.KeyboardModifier.NoModifier):
        m = measurements[index]
        canvas.mousePressEvent(
            mouse_event(
                canvas,
                QEvent.Type.MouseButtonPress,
                m.x + m.width / 2.0,
                m.y + m.height / 2.0,
                modifiers,
            )
        )

    click(0)
    assert canvas.selected_bands() == [0]
    click(1, Qt.KeyboardModifier.ShiftModifier)
    assert canvas.selected_bands() == [0, 1]
    click(2, Qt.KeyboardModifier.ControlModifier)
    assert canvas.selected_bands() == [0, 1, 2]
    click(1, Qt.KeyboardModifier.ControlModifier)
    assert canvas.selected_bands() == [0, 2], "Ctrl 再点一次应当减选"
    click(3)
    assert canvas.selected_bands() == [3], "无修饰键是单选"
    canvas.deleteLater()


def test_escape_cancels_a_rubber_band_without_losing_the_selection(qt_app):
    canvas, measurements = canvas_with_bands(qt_app, count=6, extra_stacked=False)
    canvas.set_tool(Tool.ADJUST)
    canvas.select_bands([4])

    canvas.mousePressEvent(mouse_event(canvas, QEvent.Type.MouseButtonPress, 5, 20))
    canvas.mouseMoveEvent(mouse_event(canvas, QEvent.Type.MouseMove, 200, 200))
    assert canvas.selected_bands() != [4], "拖动中应当实时改选中态"
    canvas.keyPressEvent(key_event(Qt.Key.Key_Escape))

    assert canvas.selected_bands() == [4], "Esc 取消框选要把选中还原回去"
    assert canvas._drag is None
    canvas.deleteLater()


def test_batch_delete_is_a_single_undo_step(qt_app):
    """一次删掉多个条带只占撤销栈一格：Ctrl+Z 按一次就全回来。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0, 8000.0]))
    set_roi_and_quantify(window)
    before = bands(window)
    assert len(before) >= 3
    undo_depth = len(window._undo)

    window.canvas.select_bands([0, 1, 2])
    window.canvas.keyPressEvent(key_event(Qt.Key.Key_Delete))

    assert len(bands(window)) == len(before) - 3
    assert len(window._undo) == undo_depth + 1, "批量删除不该压成 3 步"

    window._undo_last()
    assert bands(window) == before, "一次 Ctrl+Z 要把整批都恢复"
    close(window)


def test_results_table_deletes_every_selected_row_at_once(qt_app):
    """结果表的“删除选中”支持多选，并且同样只算一步撤销。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0, 8000.0]))
    set_roi_and_quantify(window)
    before = bands(window)
    assert len(before) >= 3
    undo_depth = len(window._undo)

    window.results_panel.select_rows_by_original([0, 2])
    assert window.results_panel._selected_original_rows() == [0, 2], (
        "结果表应当支持多行选中"
    )
    window.results_panel._request_delete()

    assert len(bands(window)) == len(before) - 2
    assert len(window._undo) == undo_depth + 1
    window._undo_last()
    assert bands(window) == before
    close(window)


# ------------------------------------------------------------------ U9 剖面图增删分隔线


def profile_ready(qt_app, lanes: int = 4, width: int = 1500):
    """加载 → ROI → 定量，并把窗口真的摆出来（换算坐标需要视口尺寸）。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0 - 2000.0 * i for i in range(lanes)]))
    window.resize(width, 950)
    window.show()
    qt_app.processEvents()
    set_roi_and_quantify(window)
    qt_app.processEvents()
    assert window.profile_panel._profile is not None, "前置条件：左图要有剖面"
    assert len(window._separators) >= 2
    return window


def profile_pos(window, image_x: float):
    """图像像素 x → 左图视口坐标。"""
    plot = window.profile_panel._col_plot
    view_box = plot.getPlotItem().getViewBox()
    low, high = view_box.viewRange()[1]
    scene_pt = view_box.mapViewToScene(QPointF(float(image_x), (low + high) / 2.0))
    return plot.mapFromScene(scene_pt)


def profile_x_under(window, pos):
    """视口坐标底下真正的图像像素 x。

    视口坐标是整数，点不到正好的整数图像列；断言要拿"光标底下那一列"去比，
    拿名义目标去比只是在比 mapFromScene 的取整误差。
    """
    panel = window.profile_panel
    return panel.image_x_at(panel._col_plot.mapToScene(pos))


def profile_double_click(window, pos) -> None:
    QTest.mouseDClick(
        window.profile_panel._col_plot.viewport(),
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        pos,
    )


def profile_right_click(window, pos) -> None:
    viewport = window.profile_panel._col_plot.viewport()
    for send in (QTest.mousePress, QTest.mouseRelease):
        send(viewport, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, pos)


def test_double_click_on_the_profile_adds_a_separator_at_that_pixel(qt_app):
    """双击剖面曲线加分隔线，落点必须就是光标底下那一列。

    横轴是绝对图像列号，差一像素泳道边界就挪一列，积分跟着变——所以这里比的
    是逐位相等，不是"差不多"。
    """
    window = profile_ready(qt_app)
    panel = window.profile_panel
    before = list(window._separators)
    pos = profile_pos(window, float(window._roi.x) + 137.0)
    expected = profile_x_under(window, pos)
    assert expected is not None

    profile_double_click(window, pos)
    assert settle(window), "加分隔线之后应当自动重新定量"

    added = [s for s in window._separators if all(abs(s - old) > 1e-12 for old in before)]
    assert len(added) == 1, "剖面图双击应当加出一条分隔线"
    assert added[0] == expected, "落点必须逐位等于光标下那一列"
    assert window.canvas.separators() == window._separators, "画布要拿到同一批分隔线"
    assert panel.separators() == sorted(window._separators)

    bounds = lane_boundaries(window._roi, window._separators)
    edges = {edge for pair in bounds for edge in pair}
    assert int(round(expected)) in edges, "泳道边界必须切在这一列上"
    assert len(bounds) == len(before) + 2, "泳道数应当正好多一个"
    close(window)


def test_profile_x_axis_is_absolute_image_columns(qt_app):
    """左图横轴就是图像列号：任何常数偏移都会静悄悄改掉定量。"""
    window = profile_ready(qt_app)
    panel = window.profile_panel
    roi = window._roi
    xs, _ys = panel._col_curve.getData()
    assert float(xs[0]) == float(roi.x)
    assert float(xs[-1]) == float(roi.x) + float(roi.width) - 1.0

    scale = float(panel._col_plot.getPlotItem().getViewBox().viewPixelSize()[0])
    for offset in (5.0, 100.0, float(roi.width) / 2.0, float(roi.width) - 10.0):
        target = float(roi.x) + offset
        back = profile_x_under(window, profile_pos(window, target))
        assert back is not None
        assert abs(back - target) <= scale, f"{target} 换算回来偏了 {back - target} 列"
    close(window)


def test_profile_add_separator_obeys_the_canvas_limits(qt_app):
    """内缩量、最小间距都照抄画布，两处不能各有一套判定。"""
    window = profile_ready(qt_app)
    panel = window.profile_panel
    roi = window._roi

    assert panel.add_separator(float(roi.x) - 80.0), "贴边点击应当夹进 ROI 而不是丢掉"
    assert settle(window)
    assert min(window._separators) == pytest.approx(float(roi.x) + canvas_mod.MIN_ROI_PX)

    count = len(window._separators)
    assert not panel.add_separator(min(window._separators) + 1.0), "紧挨着不许再插一条"
    assert len(window._separators) == count

    assert panel.add_separator(float(roi.x) + float(roi.width) + 50.0)
    assert settle(window)
    assert max(window._separators) == pytest.approx(
        float(roi.x) + float(roi.width) - canvas_mod.MIN_ROI_PX
    )

    # 拖动可达范围也要和画布同一个内缩量
    assert panel._drag_bounds() == (
        float(roi.x) + canvas_mod.SEP_MARGIN,
        float(roi.x) + float(roi.width) - canvas_mod.SEP_MARGIN,
    )
    close(window)


def test_right_click_on_a_profile_separator_deletes_it(qt_app, monkeypatch):
    """右键分隔线删除，和画布上右键弹的是同一个确认菜单。"""
    window = profile_ready(qt_app)
    for plot in (window.profile_panel._col_plot, window.profile_panel._lane_plot):
        assert plot.getPlotItem().getViewBox().menuEnabled() is False, (
            "pyqtgraph 默认右键菜单必须让位"
        )

    before = list(window._separators)
    victim = before[1]
    monkeypatch.setattr(pp, "_confirm_delete_separator", lambda *_args: False)
    profile_right_click(window, profile_pos(window, victim))
    assert settle(window)
    assert window._separators == before, "菜单里没点确认就不能删"

    monkeypatch.setattr(pp, "_confirm_delete_separator", lambda *_args: True)
    profile_right_click(window, profile_pos(window, victim))
    assert settle(window)
    assert window._separators == [s for s in before if s != victim]
    assert window.canvas.separators() == window._separators

    count = len(window._separators)
    empty = (window._separators[0] + window._separators[1]) / 2.0
    profile_right_click(window, profile_pos(window, empty))
    assert settle(window)
    assert len(window._separators) == count, "空白处右键不许误删最近的那条"
    close(window)


def test_profile_separator_edit_is_one_undo_step(qt_app):
    """在剖面图里增删和在画布上增删一样：一次编辑一格撤销，并自动重新定量。"""
    window = profile_ready(qt_app)
    before = list(window._separators)
    lanes_before = len(window._lanes)
    depth = len(window._undo)

    assert window.profile_panel.add_separator(float(window._roi.x) + 137.0)
    assert settle(window)
    assert len(window._undo) == depth + 1, "加一条分隔线只该占撤销栈一格"
    assert len(window._lanes) == lanes_before + 1, "定量必须跟着重跑"

    window._undo_last()
    assert settle(window)
    assert window._separators == before
    assert window.canvas.separators() == before
    assert window.profile_panel.separators() == sorted(before)
    close(window)


def test_profile_separator_editing_ignores_the_canvas_tool_mode(qt_app):
    """剖面图里增删分隔线不受工具模式约束。

    画布上双击加分隔线只在"调整"模式生效，是因为框选 ROI / 补画条带都占着鼠标
    拖拽，不分模式就打架。剖面图里没有别的手势要抢，而且分隔线本来就随时能拖，
    只把"加 / 删"卡在调整模式，同一张图里三个动作两套规矩反而更别扭；用户的流程
    又正是"圈完 ROI 立刻调分隔线"，强制切模式等于把这个改进的价值打掉一半。
    """
    window = profile_ready(qt_app)
    for index, tool in enumerate(Tool):
        window._select_tool(tool)
        count = len(window._separators)
        profile_double_click(window, profile_pos(window, float(window._roi.x) + 30.0 + 8.0 * index))
        assert settle(window)
        assert len(window._separators) == count + 1, f"{tool.value} 模式下也该加得上"
    close(window)


def test_profile_double_click_no_longer_resets_the_left_view(qt_app):
    """双击让给了加分隔线，复位改走按钮——手势没了，功能不能丢。"""
    window = profile_ready(qt_app)
    panel = window.profile_panel
    view_box = panel._col_plot.getPlotItem().getViewBox()
    left = float(window._roi.x) + 20.0
    view_box.setXRange(left, left + 60.0, padding=0)
    qt_app.processEvents()
    zoomed = list(view_box.viewRange()[0])

    count = len(window._separators)
    pos = profile_pos(window, left + 30.0)
    expected = profile_x_under(window, pos)
    profile_double_click(window, pos)
    assert settle(window)
    assert len(window._separators) == count + 1, "双击应当加分隔线"
    assert expected in window._separators, "放大之后落点换算同样必须逐位对得上"
    assert list(view_box.viewRange()[0]) == pytest.approx(zoomed, rel=1e-6), (
        "双击不该再顺手把视图复位"
    )

    panel.reset_views()
    qt_app.processEvents()
    assert list(view_box.viewRange()[0]) != pytest.approx(zoomed, rel=1e-6), (
        "复位视图按钮仍然必须能退出框选放大"
    )
    close(window)


def synthetic_font_metrics() -> bool:
    """当前环境的字体度量是不是合成出来的（字体库为空）。

    这台离屏环境里 QFontDatabase.families() 返回空列表，于是所有字体请求都退到
    "每字符恰好 12 px"的合成度量（``L9`` 量出 24.00 px 这种整数就是指纹），比真
    实字体宽约 35%，而且随字号变化并不单调。指定字体是没用的——库是空的。

    所以凡是依赖**绝对**像素宽的断言都不可信；下面测的一律是**相对**关系（"这两
    个矩形不重叠"），判据和渲染读的是同一个外接矩形，环境怎么样都成立。
    """
    return not QFontDatabase.families()


def size_profile_plot(panel, width: int, app) -> None:
    """把剖面左图定到指定宽度，并断言真的定住了。

    只有对 ``_col_plot`` 本身设宽度才有效：离屏下 window.resize() 根本不往子控件
    传（实测窗口 1200 → 2800，剖面控件纹丝不动地停在 519 px，viewbox 恒为
    446.9），补 processEvents / layout().activate() / resizeDocks / 直接拖 splitter
    全都无效。前置条件断言是必需的——少了它，"标签神秘重叠"要查半小时才发现是
    "测试根本没拿到它要求的布局"。
    """
    panel._col_plot.setMinimumWidth(width)
    panel._col_plot.setMaximumWidth(width)
    app.processEvents()
    render(panel, app)
    assert panel._col_plot.width() >= width - 1, (
        f"前置条件：左图至少要 {width} px 宽，实际只有 {panel._col_plot.width()} px"
    )


def assert_lane_labels_are_readable(panel, note: str) -> None:
    """泳道标签的契约：要么清晰不重叠、要么已降级/整批隐藏，绝不显示一部分。"""
    total = len(panel._labels)
    stride = panel.lane_label_stride()
    shown = [i for i, label in enumerate(panel._labels) if label.isVisible()]
    expected = list(range(0, total, stride)) if stride > 0 else []
    assert shown == expected, (
        f"{note}：显示出来的不是「每 {stride} 道一个」这一整批（{shown} vs {expected}）；"
        "只显示恰好塞得下的那几个是不许的，用户没法预期哪几个会留下"
    )
    assert panel.lane_labels_packed() == (stride == 1), f"{note}：档位自述和实际不符"
    rects = panel.lane_label_rects()
    assert len(rects) == len(shown)
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            assert not rects[i].intersects(rects[j]), (
                f"{note}：{shown[i] + 1} 号和 {shown[j] + 1} 号标签叠在一起"
                f"（{lane_label_state(panel)}；"
                f"合成字体度量={synthetic_font_metrics()}）"
            )


def lane_label_state(panel) -> str:
    return (
        f"整批铺开={panel.lane_labels_packed()} 紧凑文案={panel.lane_labels_compact()} "
        f"间隔={panel.lane_label_stride()} 可见={len(panel.lane_label_rects())}/"
        f"{len(panel._labels)}"
    )


def test_profile_legend_never_covers_lane_labels(qt_app):
    """图例挪出坐标区，且泳道标签在 9 / 12 / 16 泳道 × 9 / 11 / 14 pt 下都可读。

    断言的是**契约**而不是"碰巧不重叠"：``_rebuild_regions`` 以前给每道放一个
    ``pg.TextItem`` 就完事，不压缩、不隔一个、不隐藏，"不重叠"只是图够宽时的巧
    合——12 泳道差 4.2 px、16 泳道差 12.7 px，而字号设置一调大就更差。现在测的
    是「不重叠 或 已降级」，与图宽、字号都无关。
    """
    for lanes in (9, 12, 16):
        for point_size in (9.0, 11.0, 14.0):
            window = make_window(qt_app)
            load_array(
                window, synthetic_raw([20000.0 - 700.0 * i for i in range(lanes)])
            )
            with with_font(window, point_size):
                window.resize(1500, 950)
                window.show()
                qt_app.processEvents()
                set_roi_and_quantify(window)
                panel = window.profile_panel
                render(panel, qt_app)
                assert len(panel._labels) == lanes, "前置条件：每个泳道一个分区标签"
                for plot in (panel._col_plot, panel._lane_plot):
                    assert plot.getPlotItem().legend is None, (
                        "坐标区里不该再有压住标签的图例"
                    )

                # 从宽到窄各评一次：宽的时候不许过度降级，窄的时候不许重叠
                size_profile_plot(panel, 1500, qt_app)
                note = f"{lanes} 泳道 {point_size:g}pt 宽 1500 px"
                assert panel.lane_labels_packed(), (
                    f"{note}：这么宽还降级就是判据太保守了（{lane_label_state(panel)}）"
                )
                assert_lane_labels_are_readable(panel, note)
                view_rect = (
                    panel._col_plot.getPlotItem().getViewBox().sceneBoundingRect()
                )
                for rect in panel.lane_label_rects():
                    assert view_rect.contains(rect.center()), f"{note}：标签跑出绘图区"

                for width in (900, 560):
                    size_profile_plot(panel, width, qt_app)
                    assert_lane_labels_are_readable(
                        panel, f"{lanes} 泳道 {point_size:g}pt 宽 {width} px"
                    )

                texts = {
                    label.text()
                    for strip in panel.findChildren(pp.LegendStrip)
                    for label in strip.findChildren(QLabel)
                }
                for wanted in ("列平均信号", "泳道信号", "扣除用基线", "积分区间（条带）"):
                    assert any(wanted in text for text in texts), f"图例丢了「{wanted}」"
                close(window)


def test_profile_lane_labels_degrade_one_step_at_a_time(qt_app):
    """越窄降得越狠，而且每一档都读得出来：完整 → 纯数字 → 隔几道 → 整批隐藏。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0 - 600.0 * i for i in range(16)]))
    window.resize(1500, 950)
    window.show()
    qt_app.processEvents()
    set_roi_and_quantify(window)
    panel = window.profile_panel
    render(panel, qt_app)

    size_profile_plot(panel, 1600, qt_app)
    assert panel.lane_labels_packed() and not panel.lane_labels_compact(), (
        f"够宽时就该是完整的 L# 文案（{lane_label_state(panel)}）"
    )
    assert panel._labels[0].textItem.toPlainText() == "L1"

    seen = []
    for width in (1600, 900, 560, 320):
        size_profile_plot(panel, width, qt_app)
        assert_lane_labels_are_readable(panel, f"16 泳道宽 {width} px")
        seen.append((panel.lane_labels_compact(), panel.lane_label_stride()))
    compacts = [c for c, _ in seen]
    strides = [s for _, s in seen]
    assert compacts[-1], f"窄到 320 px 还留着 L 前缀就是没降级：{seen}"
    assert panel._labels[0].textItem.toPlainText() == "1", "紧凑文案该只剩数字"
    assert strides == sorted(strides), f"间隔只该越窄越大，实际 {seen}"
    assert strides[-1] > 1, f"320 px 塞 16 道还每道都标必然重叠：{seen}"
    close(window)


def test_profile_lane_labels_hide_as_a_batch_when_nothing_fits(qt_app):
    """连最稀的间隔都摆不开时整批隐藏，不留"能塞下的那几个"。

    真实布局下走不到这一档（pyqtgraph 给绘图区留了最小宽度），所以直接把"每像素
    多少数据单位"压到极限来验这根兜底。
    """
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0 - 700.0 * i for i in range(12)]))
    window.resize(1400, 900)
    window.show()
    qt_app.processEvents()
    set_roi_and_quantify(window)
    panel = window.profile_panel
    render(panel, qt_app)
    assert panel.lane_label_rects(), "前置条件：正常宽度下标签是看得见的"

    panel._view_px_per_unit = lambda: 1e-4  # 一个泳道只剩不到 1 px
    panel._relayout_lane_labels()
    render(panel, qt_app)
    assert panel.lane_label_stride() == 0, "摆不开就该整批隐藏"
    assert panel.lane_label_rects() == [], "整批隐藏就是一个都不留"
    assert not panel.lane_labels_packed()
    close(window)


# ------------------------------------------------------------------ U10 溯源接线


DOUBLET_CENTERS = (85.0, 155.0)  # 上下两条，谷底够深，segment_bands 切得开


def doublet_window(qt_app, depths=(20000.0, 16000.0, 12000.0)) -> MainWindow:
    """每个泳道上下两条带的 ROI 定量结果——双条带策略的前置条件。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw(list(depths), centers=DOUBLET_CENTERS))
    set_roi_and_quantify(window)
    counts: dict[int, int] = {}
    for m in window._result.measurements:
        counts[int(m.lane)] = counts.get(int(m.lane), 0) + 1
    assert set(counts.values()) == {2}, f"前置条件：每个泳道两条带，实际 {counts}"
    return window


def history_cell(window: MainWindow, row: int, column: int) -> str:
    item = window.normalize_panel.history_table.item(row, column)
    return "" if item is None else item.text()


def result_cell(window: MainWindow, row: int, column: int) -> str:
    item = window.normalize_panel.result_table.item(row, column)
    return "" if item is None else item.text()


def diag_text(window: MainWindow) -> str:
    """诊断条上的全部文字：一行摘要 + 每条的标题与详情。

    三路提示（ROI 自检 / 条带数没切够 / 策略未生效）现在都汇到这里，测试跟着
    读同一个地方——分散在各面板时，"同一件事说了两遍"这种问题断言不出来。
    """
    bar = window.diagnostics
    return f"{bar.summary()} {bar.detail_text()}"


def usage_texts(window: MainWindow) -> dict[tuple[int, int], str]:
    """(泳道, 条带号) → 结果表“计入”列的文案。"""
    table = window.results_panel.table
    out: dict[tuple[int, int], str] = {}
    for row in range(table.rowCount()):
        lane = table.item(row, 0).data(Qt.ItemDataRole.EditRole)
        band = table.item(row, 1).data(Qt.ItemDataRole.EditRole)
        out[(int(lane), int(band))] = table.item(row, rp.USAGE_COLUMN).text()
    return out


def test_history_entry_carries_its_own_background_origin(qt_app):
    """口径由 QuantEntry 自己带，界面不再并行记一份账。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0]))
    set_roi_and_quantify(window)
    add_history(window, "loading_control", "Actin", "G1")
    assert window._session.entries[0].origin == "roi_baseline"

    window._run_auto(synchronous=True)
    assert settle(window)
    add_history(window, "target", "PAD4", "G2")
    assert window._session.entries[1].origin == "local_background"

    # 改名走的是 replace，口径必须跟着走（旧实现靠 (kind,name,group) 记账，
    # 一改名就对不上号，口径悄悄丢成“未知”）
    window._on_history_edited(0, "loading_control", "ActinB", "G1")
    assert window._session.entries[0].origin == "roi_baseline"
    assert history_cell(window, 0, 4) == "ROI 基线"
    assert history_cell(window, 1, 4) == "局部背景"
    assert not hasattr(window, "_entry_origins")
    close(window)


def test_cross_origin_rows_are_flagged_in_the_result_table(qt_app):
    """跨口径 = 这组数字不能用：备注成列 + 整行标红，不能只挂 tooltip。"""
    window = make_window(qt_app)
    window.resize(1400, 900)
    window.show()
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0]))
    set_roi_and_quantify(window)
    add_history(window, "loading_control", "Actin", "G1")
    window._run_auto(synchronous=True)
    assert settle(window)
    add_history(window, "target", "PAD4", "G1")

    panel = window.normalize_panel
    assert panel.result_table.rowCount() > 0, "两侧的原始测量值仍要照常成行"
    for row in range(panel.result_table.rowCount()):
        assert result_cell(window, row, 4) == "—", "跨口径不该给出归一化值"
        assert result_cell(window, row, 5) == "—", "跨口径不该给出 fold change"
        assert result_cell(window, row, nzp.RESULT_NOTE_COLUMN) == "背景口径不一致"
        # 面板窄时右边的备注列要滚动才看得见，最左列的标记不会被滚走
        assert result_cell(window, row, 0).startswith("⛔")
        # 目标值 / 内参值本身必须还在——被拒绝的是比值，不是测量
        assert result_cell(window, row, 2) not in ("", "—")
        assert result_cell(window, row, 3) not in ("", "—")
    assert panel.cross_origin_rows() == list(range(panel.result_table.rowCount()))

    hint = panel._result_hint.text()
    assert "拒绝" in hint and "G1" in hint
    render(panel, qt_app)
    close(window)


def test_band_pick_feeds_the_history_entry(qt_app):
    """录入区的条带下拉直通核心层：只取上面那条 = 只加上面那条的积分。"""
    window = doublet_window(qt_app)
    top_by_lane = {}
    for m in window._result.measurements:
        current = top_by_lane.get(int(m.lane))
        if current is None or m.y < current.y:
            top_by_lane[int(m.lane)] = m

    add_history(window, "loading_control", "Actin", "G1", pick="top")
    entry = window._session.entries[0]
    assert entry.band_pick == "top"
    assert entry.used_bands == [[1], [1], [1]]
    for lane, value in zip(entry.lanes, entry.intensities, strict=False):
        assert value == pytest.approx(top_by_lane[lane].integrated_density)
    assert history_cell(window, 0, nzp.HISTORY_PICK_COLUMN) == "只取上面那条"
    close(window)


def test_unused_bands_stay_visible_and_are_marked_not_deleted(qt_app):
    """未采用 ≠ 已删除：行还在、框还在，只是标出来不计入合计。"""
    window = doublet_window(qt_app)
    window.resize(1400, 900)
    window.show()
    total = len(window._result.measurements)

    window.normalize_panel.set_band_pick("top")
    qt_app.processEvents()

    assert window.results_panel.table.rowCount() == total, "未采用的行必须还在表里"
    marks = usage_texts(window)
    assert {key: text for key, text in marks.items() if text == rp.UNUSED_TEXT}.keys() == {
        (1, 2), (2, 2), (3, 2)
    }
    assert all(marks[(lane, 1)] == rp.ADOPTED_TEXT for lane in (1, 2, 3))
    # 未采用的那一行：积分灰度划掉、整行斜体——和"低于检出限/饱和"的配色是两个通道
    table = window.results_panel.table
    for row in range(table.rowCount()):
        unused = table.item(row, rp.USAGE_COLUMN).text() == rp.UNUSED_TEXT
        assert table.item(row, rp.DENSITY_COLUMN).font().strikeOut() is unused
        assert table.item(row, 0).font().italic() is unused

    # 画布：未采用是虚线框，删除是框直接消失
    assert window.canvas.unused_bands() == sorted(
        i for i, m in enumerate(window._result.measurements) if m.band == 2
    )
    boxes = {(b.lane, b.index): b for b in window.canvas._band_items}
    dashed = [b for b in boxes.values() if not b.is_adopted()]
    assert dashed and all(
        b.pen().style() == Qt.PenStyle.DashLine for b in dashed
    ), "未采用的框要画成虚线"
    assert all(
        b.pen().style() == Qt.PenStyle.SolidLine
        for b in boxes.values()
        if b.is_adopted()
    )
    assert all(canvas_mod.UNUSED_BAND_TIP in b.toolTip() for b in dashed)
    render(window.canvas, qt_app)
    render(window.results_panel, qt_app)

    # 复制出去的 TSV 也要带上"计入 / 未采用"，否则粘到表格里分不出哪些进了合计
    window.results_panel.table.selectAll()
    window.results_panel.copy_selected()
    lines = QApplication.clipboard().text().splitlines()
    assert lines[0].split("\t") == rp.HEADERS
    assert sum(1 for line in lines[1:] if rp.UNUSED_TEXT in line) == 3

    # 真正删掉一条：它从表和画布上一起消失，这就是与"未采用"的区别
    victim = next(
        i for i, m in enumerate(window._result.measurements) if (m.lane, m.band) == (1, 2)
    )
    window._on_delete_rows([victim])
    assert settle(window)
    assert window.results_panel.table.rowCount() == total - 1
    assert (1, 2) not in usage_texts(window)
    assert sorted(b.lane for b in window.canvas._band_items if not b.is_adopted()) == [2, 3]
    close(window)


def test_history_band_pick_change_is_one_undo_step(qt_app):
    """历史表里改条带策略：只重新聚合，且 Ctrl+Z 一次全恢复。"""
    window = doublet_window(qt_app)
    add_history(window, "loading_control", "Actin", "G1")
    before = list(window._session.entries[0].intensities)
    assert window._session.entries[0].band_pick == "all"

    table = window.normalize_panel.history_table
    table.item(0, nzp.HISTORY_PICK_COLUMN).setText("只取下面那条")
    qt_app.processEvents()

    entry = window._session.entries[0]
    assert entry.band_pick == "bottom"
    assert entry.used_bands == [[2], [2], [2]]
    assert all(new < old for new, old in zip(entry.intensities, before, strict=True))
    assert history_cell(window, 0, nzp.HISTORY_PICK_COLUMN) == "只取下面那条"
    assert "重新聚合" in window._hint_label.text()

    depth = len(window._undo)
    window._undo_last()
    qt_app.processEvents()
    assert len(window._undo) == depth - 1, "一次策略切换只该占撤销栈一格"
    assert window._session.entries[0].band_pick == "all"
    assert window._session.entries[0].intensities == before
    close(window)


def test_csv_imported_entry_refuses_a_band_pick_change(qt_app, monkeypatch, tmp_path):
    """CSV 里只有合计值，改不了策略：弹中文提示，表格刷回原样。"""
    window = doublet_window(qt_app)
    add_history(window, "loading_control", "Actin", "G1")
    csv_path = tmp_path / "lc.csv"
    window._session.export_loading_control_csv(csv_path)
    window._session.clear()
    monkeypatch.setattr(
        mw.QFileDialog, "getOpenFileName", lambda *a, **k: (str(csv_path), "")
    )
    window._on_import_lc()
    assert window._session.entries and not window._session.entries[0].bands

    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        mw.QMessageBox, "warning", lambda _p, title, text, *a, **k: warnings.append((title, text))
    )
    depth = len(window._undo)
    window.normalize_panel.history_table.item(0, nzp.HISTORY_PICK_COLUMN).setText(
        "只取上面那条"
    )
    qt_app.processEvents()

    assert warnings and "重新定量" in warnings[0][1]
    assert window._session.entries[0].band_pick == "all"
    assert history_cell(window, 0, nzp.HISTORY_PICK_COLUMN) == "合并全部", "表格要刷回原样"
    assert len(window._undo) == depth, "改失败不该留下撤销步"
    close(window)


def test_single_band_lane_says_the_strategy_did_not_apply(qt_app):
    """只检出一条带的泳道：必须明说"策略未生效"，这是发现漏检的唯一线索。"""
    window = doublet_window(qt_app)
    window.resize(1400, 900)
    window.show()
    # 删掉泳道 2 的下面那条 → 泳道 2 只剩一条带，正是用户那张膜的情形
    victim = next(
        i for i, m in enumerate(window._result.measurements) if (m.lane, m.band) == (2, 2)
    )
    window._on_delete_rows([victim])
    assert settle(window)

    window.normalize_panel.set_band_pick("top")
    qt_app.processEvents()
    hint = diag_text(window)
    assert "泳道 2" in hint and "只检出 1 条带" in hint and "未生效" in hint
    assert "只取上面那条" in hint
    render(window.diagnostics, qt_app)

    add_history(window, "loading_control", "Actin", "G1", pick="top")
    assert window._entry_single_band_lanes(window._session.entries[0]) == [2]
    tip = window.normalize_panel.history_table.item(
        0, nzp.HISTORY_PICK_COLUMN
    ).toolTip()
    assert "只检出 1 条带" in tip and "泳道 2" in tip
    assert "只检出 1 条带" in window.normalize_panel._result_hint.text()

    # 合并全部时这句提示不该出现：没选上下策略就无所谓生效不生效
    window.normalize_panel.set_band_pick("all")
    qt_app.processEvents()
    assert "未生效" not in diag_text(window)
    close(window)


def test_exported_band_csv_records_the_background_origin(qt_app, monkeypatch, tmp_path):
    """条带 CSV 要写明这批数字是哪套口径测的。"""
    import csv

    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0]))
    set_roi_and_quantify(window)
    monkeypatch.setattr(
        mw.QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path)
    )
    window._export_data()
    exported = list(tmp_path.glob("*_bands.csv"))
    assert exported, "导出没落地"
    with exported[0].open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and {row["origin"] for row in rows} == {"roi_baseline"}
    close(window)


# ------------------------------------------------------------------ U11 每道条带数


# 两条带中心只隔 17 px，半高宽约 18.8 px（d/FWHM ≈ 0.90）——几何谷深与统计
# 显著性两条判据都够不着，自动分段只给出一条带。用户那张膜正是这个量级。
TIGHT_CENTERS = (111.5, 128.5)


def lane_band_counts(window: MainWindow) -> dict[int, int]:
    counts: dict[int, int] = {}
    for m in window._result.measurements if window._result else []:
        counts[int(m.lane)] = counts.get(int(m.lane), 0) + 1
    return counts


def tight_doublet_window(qt_app) -> MainWindow:
    """自动判据切不开的贴脸双条带——「每道条带数」存在的理由。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 16000.0, 12000.0], centers=TIGHT_CENTERS))
    set_roi_and_quantify(window)
    assert set(lane_band_counts(window).values()) == {1}, "前置条件：自动判据在这张图上切不开"
    return window


def set_band_count(window: MainWindow, value: int) -> None:
    window.band_spin.setValue(value)
    assert settle(window), "改「每道条带数」没有触发自动重新定量"


def test_band_count_control_mirrors_the_lane_count_control(qt_app):
    """两个控件语义对称，写法就得一样：0 = 自动，不另加复选框。"""
    window = make_window(qt_app)
    spin, lanes = window.band_spin, window.lane_spin
    assert spin.minimum() == 0 and spin.maximum() == 10
    # 0 走 specialValueText，和「期望泳道数」同一套惯用法；核心层 < 1 会抛
    # ValueError，下限设 0 且 0 折成 None，用户碰不到那条路
    assert spin.specialValueText() == lanes.specialValueText() == "自动"
    assert window._expected_bands() is None
    spin.setValue(2)
    assert window._expected_bands() == 2
    assert spin.focusPolicy() == lanes.focusPolicy()  # 未聚焦时同样忽略滚轮
    close(window)


def test_band_count_forces_a_split_the_auto_criterion_misses(qt_app):
    """填 2 就该切成 2 条——这是自动判据在分辨极限上失手时的兜底。"""
    window = tight_doublet_window(qt_app)
    window.resize(1400, 900)
    window.show()

    set_band_count(window, 2)
    assert set(lane_band_counts(window).values()) == {2}
    assert len(window._result.measurements) == 6
    # 剖面面板画的分段来自同一份 LaneProfile，跟着一起更新
    assert [len(lane.segments) for lane in window._lanes] == [2, 2, 2]
    render(window.profile_panel, qt_app)
    render(window.results_panel, qt_app)

    set_band_count(window, 0)  # 回到自动
    assert set(lane_band_counts(window).values()) == {1}
    close(window)


def test_band_count_only_adds_cuts_never_merges(qt_app):
    """只加刀、不合并——已经被空白分成两段的泳道，填 1 也并不回去。

    界面不能让用户以为这个框能"压回 N 条"：它只决定切几刀。
    """
    window = doublet_window(qt_app)  # 两条带隔着 70 px，中间是空白，本就是两段
    set_band_count(window, 1)
    assert set(lane_band_counts(window).values()) == {2}
    close(window)


def test_band_count_one_suppresses_a_cut_inside_one_run(qt_app):
    """同一段里的两个峰：填 1 就是"别切这一刀"。"""
    window = tight_doublet_window(qt_app)
    set_band_count(window, 2)
    assert set(lane_band_counts(window).values()) == {2}
    set_band_count(window, 1)
    assert set(lane_band_counts(window).values()) == {1}
    close(window)


def test_band_count_never_touches_the_separators(qt_app):
    """纵向分段和横向分隔线是两回事，改前者不许冲掉手调过的后者。

    「期望泳道数」走的是 _reset_separators（按剖面峰谷重划），照抄那条路会
    把用户手动拖过的分隔线全部抹掉。
    """
    window = tight_doublet_window(qt_app)
    moved = [s + 4.0 for s in window._separators]
    assert moved, "前置条件：得先有分隔线"
    window._on_profile_separators_changed(moved)  # 等价于在剖面图上拖了一下
    window._on_separators_committed()
    assert settle(window)

    set_band_count(window, 2)
    assert window._separators == pytest.approx(moved), "分隔线被重划了"
    assert window.canvas.separators() == pytest.approx(moved)

    # 对照：改「期望泳道数」本来就该重划，这条路径没被改坏
    window.lane_spin.setValue(3)
    assert settle(window)
    assert window._separators != pytest.approx(moved)
    close(window)


def test_saturated_band_count_is_spelled_out(qt_app):
    """填了却切不出那么多时必须点名，否则用户以为强制生效了。

    这话现在由核心层的「条带数与泳道数对不上」说（它按 source 分岔，用户填过
    时不会再劝人去填），界面那条重复的 info 已经撤掉。这里盯的是"用户看得到
    这句话"，不关心是谁说的。
    """
    window = doublet_window(qt_app)
    window.resize(1400, 900)
    window.show()

    set_band_count(window, 3)  # 只有两条带，第三刀没谷可切
    counts = lane_band_counts(window)
    assert set(counts.values()) == {2}, "候选谷不够时应当饱和，不是报错也不是硬切"

    hint = diag_text(window)
    assert "你指定了每道 3 条带" in hint, "要认下用户填过这个数，别装作没看见"
    assert "少切了" in hint and "有几个谷切几个" in hint
    assert "连可切的谷都没有" in hint
    assert "可以用「每道条带数」强制切分" not in hint, "不许劝人去填他已经填过的框"
    render(window.diagnostics, qt_app)

    set_band_count(window, 2)  # 切得够就不该再唠叨
    assert "少切了" not in diag_text(window)
    assert "没切够" not in diag_text(window)

    # 只有个别泳道没切够时逐个点名
    victim = next(
        i for i, m in enumerate(window._result.measurements) if (m.lane, m.band) == (2, 2)
    )
    window._on_delete_rows([victim])
    assert settle(window)
    assert "第 2 道少切了（1 条）" in diag_text(window)
    close(window)


def test_single_band_hint_points_at_the_band_count_box(qt_app):
    """提示发现问题之后要给出解决手段——就是这个新控件。"""
    window = tight_doublet_window(qt_app)
    window.normalize_panel.set_band_pick("top")
    qt_app.processEvents()

    hint = diag_text(window)
    assert "只检出 1 条带" in hint and "未生效" in hint
    assert "每道条带数" in hint and "填成 2" in hint
    close(window)


def test_single_band_hint_stops_nagging_once_the_box_is_filled(qt_app):
    """已经填过了还劝人去填就是废话，换成说明那儿根本没有可切的谷。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 16000.0, 12000.0]))  # 每道就一条带
    window.band_spin.setValue(2)
    set_roi_and_quantify(window)
    assert set(lane_band_counts(window).values()) == {1}, "单条带图切不出第二条，正是要测的情形"

    window.normalize_panel.set_band_pick("bottom")
    qt_app.processEvents()
    hint = diag_text(window)
    assert "第 1、2、3 道少切了" in hint, "核心层要点名哪几道没切够"
    assert "「每道条带数」已填 2" in hint and "找不到可切的谷" in hint
    assert "填成 2" not in hint, "不该再劝用户去填他已经填过的框"
    close(window)


def test_band_count_reaches_the_quantify_job(qt_app, monkeypatch):
    """接线点只有 compute_lane_profiles 那一次调用。"""
    seen: list[object] = []
    original = mw.quant.compute_lane_profiles

    def spy(*args, **kwargs):
        seen.append(kwargs.get("expected_bands"))
        return original(*args, **kwargs)

    monkeypatch.setattr(mw.quant, "compute_lane_profiles", spy)
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 16000.0], centers=TIGHT_CENTERS))
    window.band_spin.setValue(2)
    set_roi_and_quantify(window)
    assert seen and seen[-1] == 2
    # quantify_lanes 拿到 lanes 后不再重算分段，所以整轮只该算一次剖面
    assert len(seen) == 1
    close(window)


# ------------------------------------------------------------------ U12 ROI 自检


def quantify_box(window: MainWindow, x: int, y: int, w: int, h: int) -> None:
    """按给定 ROI 走一遍完整链路（分隔线重划 → 定量 → 自检）。"""
    window.canvas.set_roi(x, y, w, h)
    window._apply_roi_change(recompute=True)
    assert settle(window), "ROI → 剖面 → 自动定量的链条没跑完"


def clean_roi_window(qt_app) -> MainWindow:
    """条带居中、上下都留着干净背景的 ROI——自检该放行的那种。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 16000.0, 12000.0]))
    set_roi_and_quantify(window)
    assert window.diagnostics.severity() is None, (
        f"前置条件：这个 ROI 该通过自检，实际 {diag_text(window)}"
    )
    return window


def clipped_roi_window(qt_app) -> MainWindow:
    """上边压着条带的 ROI——用户那张膜的习惯画法，自检必须拦下。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 16000.0, 12000.0]))
    # 条带中心在 y=120、半高宽约 19 px：上边切到 108 就贴上去了
    quantify_box(window, 15, 108, window._blot.width - 30, 82)
    assert window.diagnostics.severity() is not None, "前置条件：这个 ROI 该被拦下"
    return window


def test_a_clean_roi_gets_a_quiet_pass(qt_app):
    """通过也要有反馈，否则用户分不清"查过了"和"根本没查"——但要轻。"""
    window = clean_roi_window(qt_app)
    window.resize(1400, 900)
    window.show()
    bar = window.diagnostics
    assert bar.isVisibleTo(window), "通过时也要留一行确认"
    assert bar.severity() is None
    assert "ROI 自检通过" in bar.summary()
    assert "要求 ≥" in bar.summary(), "该说清楚判据的实际数值，不是一句“没问题”"
    assert not bar.is_expanded(), "通过时不展开详情"
    assert not bar._dismiss_button.isVisibleTo(bar), "没问题就没什么可忽略的"
    render(bar, qt_app)
    close(window)


def test_roi_check_shares_the_job_and_the_expected_bands(qt_app, monkeypatch):
    """自检必须与定量同一次任务、同一个 expected_bands，不能各读各的。"""
    seen: list[object] = []
    original = mw.check_roi_quality

    def spy(lanes, *, expected_bands=None):
        seen.append(expected_bands)
        return original(lanes, expected_bands=expected_bands)

    monkeypatch.setattr(mw, "check_roi_quality", spy)
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 16000.0, 12000.0], centers=TIGHT_CENTERS))
    window.band_spin.setValue(2)
    set_roi_and_quantify(window)
    assert seen == [2], f"自检拿到的每道条带数不是定量用的那个：{seen}"

    # 改了 SpinBox、重算还没落地时，提示读的必须是产出这批结果时的那个值，
    # 而不是实时控件（400 ms 防抖窗口里两者不一致）
    window.band_spin.setValue(5)
    assert window._roi_expected_bands() == 2
    assert settle(window)
    assert window._roi_expected_bands() == 5
    close(window)


def test_clipped_roi_marks_bands_without_a_fifth_colour(qt_app):
    """画布只加形状与纹理：角标 + 斜纹留白带，条带框的四种状态一个没动。"""
    window = clipped_roi_window(qt_app)
    window.resize(1400, 900)
    window.show()

    flagged = window.canvas.flagged_bands()
    assert flagged, "被自检点名的条带该挂上角标"
    top, bottom = window.canvas.clearance_strips()
    assert top > 0.0, "上边留白不足，斜纹带该画在上边"
    assert bottom == 0.0, "下边没问题就别画"

    boxes = {item.index: item for item in window.canvas._band_items}
    for index in flagged:
        box = boxes[index]
        assert box.flag is not None
        # 角标用条带框自己的描边色：没有引入第五种颜色，区分靠"多了个三角形"
        assert box.flag._color == box._base
        assert "ROI 自检" in box.toolTip()

    # 未采用（虚线 + 半透明）和被点名（角标）是两个独立通道，可以同时成立
    window.normalize_panel.set_band_pick("top")
    qt_app.processEvents()
    both = [i for i in window.canvas.unused_bands() if i in flagged]
    if both:
        box = boxes[both[0]]
        assert box.flag is not None and not box.is_adopted()
        assert "未采用" in box.toolTip() and "ROI 自检" in box.toolTip()
    render(window.canvas, qt_app)
    close(window)


def test_severity_is_visible_at_a_glance(qt_app):
    """严重和提醒得一眼分得开：徽标文字 + 左边框颜色都跟着 severity 走。"""
    window = clipped_roi_window(qt_app)
    window.resize(1400, 900)
    window.show()
    bar = window.diagnostics
    assert bar.severity() in ("warning", "error")
    assert bar._chip.text() in ("警告", "严重")
    assert bar._chip.isVisibleTo(bar)
    # 具体动作要露在详情里，不能只挂 tooltip
    assert "挪至少" in bar.detail_text() and "px" in bar.detail_text()
    render(bar, qt_app)

    # 展开与收起互为反面，且都画得出来（展开态是另一条绘制路径）
    before = bar.is_expanded()
    bar._toggle_detail()
    qt_app.processEvents()
    assert bar.is_expanded() is not before
    render(bar, qt_app)
    bar._toggle_detail()
    qt_app.processEvents()
    assert bar.is_expanded() is before
    render(bar, qt_app)
    close(window)


def test_acknowledge_lasts_until_something_changes(qt_app):
    """「我知道了」按内容忽略，不是全局关掉自检。"""
    window = clipped_roi_window(qt_app)
    window.resize(1400, 900)
    window.show()
    bar = window.diagnostics
    assert not bar.is_dismissed()

    marks = window.canvas.flagged_bands()
    assert marks, "前置条件：忽略之前画布上有角标"

    bar.dismiss()
    qt_app.processEvents()
    assert bar.is_dismissed()
    assert bar.isVisibleTo(window), "忽略后仍留一行淡色痕迹，随时能翻回来"
    assert "已忽略" in bar._text.text()
    # 只闭掉文字、留着满屏角标，等于没让用户忽略成
    assert window.canvas.flagged_bands() == []
    assert window.canvas.clearance_strips() == (0.0, 0.0)
    render(bar, qt_app)
    render(window.canvas, qt_app)

    # 原地重新定量：同一个 ROI、同一批问题，不该又冒出来烦人
    window._run_quantify(synchronous=True)
    assert settle(window)
    assert bar.is_dismissed()

    # 动了 ROI：问题可能变了，必须重新提醒
    quantify_box(window, 15, 100, window._blot.width - 30, 90)
    assert not bar.is_dismissed()

    # 「重新查看」把它连同画布标注一起收回来
    bar.dismiss()
    assert bar.is_dismissed() and window.canvas.flagged_bands() == []
    bar.dismiss()
    assert not bar.is_dismissed() and window.canvas.flagged_bands()
    close(window)


def test_three_hint_sources_share_one_place_without_repeating(qt_app):
    """三路提示汇到同一条，且不把同一件事说两遍。

    核心层的「条带数与泳道数对不上」按 source 分岔之后已经把"没切够"说全了
    （认下用户填过、讲清饱和语义、给出两种成因），界面那条重复的 info 整条
    撤掉——一件事说两遍比说一遍差，哪怕第二遍降成了 info。
    """
    window = make_window(qt_app)
    window.resize(1400, 900)
    window.show()
    load_array(window, synthetic_raw([20000.0, 16000.0, 12000.0]))  # 每道就一条带
    window.band_spin.setValue(2)  # 切不出第二条：自检会报"条带数对不上"
    set_roi_and_quantify(window)
    window.normalize_panel.set_band_pick("top")  # 策略在单条带泳道上落空
    qt_app.processEvents()

    entries = window.diagnostics.entries()
    titles = [e.title for e in entries]
    assert len(titles) == len(set(titles)), f"同一件事出现了两次：{titles}"
    assert any("条带数与泳道数对不上" in t for t in titles), "自检那条要在"
    assert any("未生效" in t for t in titles), "条带策略那条是另一件事，要留着"
    assert not any("饱和的" in t for t in titles), "自检说过了，界面不许再说一遍"
    assert window.diagnostics.severity() == "error"
    render(window.diagnostics, qt_app)
    close(window)


def test_the_shortfall_hint_still_covers_for_a_missing_self_check(qt_app):
    """自检缺席时（它自己抛了异常）界面得兜住，否则这句话没人说了。"""
    window = doublet_window(qt_app)
    set_band_count(window, 3)
    assert not any("饱和的" in e.title for e in window._diagnostic_entries())

    window._roi_report = None
    window._refresh_diagnostics()
    entry = next(e for e in window.diagnostics.entries() if "饱和的" in e.title)
    assert entry.severity == "warning"
    assert "全部 3 个泳道都只切出 2 条" in entry.detail
    assert "不报错" in entry.detail and "硬切" in entry.detail
    close(window)


def collapse_bar(bar, app) -> None:
    if bar.is_expanded():
        bar._toggle_detail()
    app.processEvents()


def expand_bar(bar, app) -> None:
    if not bar.is_expanded():
        bar._toggle_detail()
    app.processEvents()


def test_the_bar_never_eats_the_canvas(qt_app):
    """诊断条不能挡住工作区：窄窗、展开长文，高度都得有上限。

    折叠态的上限卡得比以前紧（90 → 46 px）：摘要现在是单行省略，再多的诊断也
    只占一行；以前它 wordWrap，三条诊断能撑到三行压在画布上。
    """
    window = clipped_roi_window(qt_app)
    bar = window.diagnostics
    for width in (900, 1200, 1700):
        window.resize(width, 800)
        window.show()
        qt_app.processEvents()
        collapse_bar(bar, qt_app)
        render(bar, qt_app)
        collapsed = bar.height()
        assert collapsed <= 46, f"{width} px 宽时折叠态就占了 {collapsed} px"
        expand_bar(bar, qt_app)
        render(bar, qt_app)
        expanded = bar.height()
        assert expanded <= 300, f"{width} px 宽时展开占了 {expanded} px，详情该自己滚"
        assert window.canvas.height() > expanded, "画布得比诊断条高"
        collapse_bar(bar, qt_app)
    close(window)


def test_deleting_bands_rechecks_the_roi(qt_app):
    """删掉条带之后自检要按剩下的重算，画布上的角标跟着走。

    这条路径没有剖面，走的是 check_roi_quality_from_measurements；ROI 必须传
    裁剪后的那个，否则留白全算错。
    """
    window = clipped_roi_window(qt_app)
    flagged = window.canvas.flagged_bands()
    assert flagged
    window._on_delete_rows([flagged[0]])
    assert settle(window)
    assert window._roi_report is not None, "删完还得有一份自检结论"
    # 删除只做过滤，剩下的条带索引整体前移；被删的那个不该还挂着角标
    assert len(window.canvas._band_items) == len(window._result.measurements)
    assert set(window.canvas.flagged_bands()) <= set(
        range(len(window._result.measurements))
    )
    close(window)


def test_auto_detect_has_no_roi_self_check(qt_app):
    """自动检测那条路径不走 ROI 基线，自检的两条判据都不适用。"""
    window = clean_roi_window(qt_app)
    window._run_auto(synchronous=True)
    assert settle(window)
    assert window._roi_report is None
    assert window.canvas.flagged_bands() == []
    assert window.canvas.clearance_strips() == (0.0, 0.0)
    close(window)


# ------------------------------------------------------------------ U13 自动测倾角
#
# 核心层把话说得很死：这个算法在干净的膜上能到 0.02°，但被一条划痕或一道膜边
# 骗到时会**用同样自信的语气**给出错 4° 的答案，而且它测不出自己被骗了。所以
# 界面这一侧只有两件事非做不可——把角度画出来让人看，以及绝不自动应用。


def tilted_raw(angle_deg: float, lanes: int = 4, center: float = 120.0) -> np.ndarray:
    """条带整体倾斜 ``angle_deg`` 的合成图（右端低为正，与标尺约定一致）。

    不用 cv2 旋转整图：那会在四角留下背景块，而贯穿全宽的硬边正是这个算法最
    容易锁上去的东西——那样测的就不是条带了。这里直接让高斯条带的中心随列
    平移，膜内不留任何假直线。
    """
    height, width = 240, 20 + lanes * 95
    slope = float(np.tan(np.radians(angle_deg)))
    columns = np.arange(width, dtype=float)
    rows = np.arange(height, dtype=float)[:, None]
    offsets = slope * (columns - (width - 1) / 2.0)[None, :]
    raw = np.full((height, width), 60000.0)
    profile = 22000.0 * np.exp(-0.5 * ((rows - center - offsets) / 8.0) ** 2)
    for index in range(lanes):
        x0 = 20 + index * 95
        raw[:, x0 : x0 + 80] -= profile[:, x0 : x0 + 80]
    return np.clip(raw, 0, 65535).astype(np.uint16)


def tilt_window(qt_app, angle_deg: float = -1.5, lanes: int = 4) -> MainWindow:
    window = make_window(qt_app)
    load_array(window, tilted_raw(angle_deg, lanes))
    window.canvas.set_roi(20, 70, window._blot.width - 40, 100)
    window._apply_roi_change()
    assert settle(window)
    return window


def measure_tilt(window: MainWindow) -> None:
    window._measure_tilt()
    assert settle(window), "自动测倾角没跑完"


def guide_angle(window: MainWindow) -> float:
    """从画布上真正画出来的那条线反算角度。断言看的是像素，不是变量。"""
    items = window.canvas._ruler_items
    assert items, "测出角度却没画参考线"
    line = items[0].line()
    return float(np.degrees(np.arctan2(line.dy(), line.dx())))


def test_auto_tilt_suggests_an_angle_without_touching_the_image(qt_app):
    """测出来只写标尺位、只画线；旋转角必须还是 0。"""
    window = tilt_window(qt_app, -1.5)
    measure_tilt(window)
    assert window._ruler_angle == pytest.approx(-1.5, abs=0.25)
    assert "自动" in window.ruler_label.text(), "要说清这是自动量的，不是手拖的"
    assert window.rotation_spin.value() == pytest.approx(0.0), "绝不许自动应用"
    assert window._angle == pytest.approx(0.0)
    assert "建议值" in window._hint_label.text()
    render(window.canvas, qt_app)
    close(window)


def test_the_guide_line_is_what_the_user_judges_by(qt_app):
    """参考线必须真按测出的角度、真横穿 ROI——数字对、线画错等于没画。"""
    window = tilt_window(qt_app, 2.0)
    measure_tilt(window)
    angle = window._ruler_angle
    assert angle == pytest.approx(2.0, abs=0.25)
    assert guide_angle(window) == pytest.approx(angle, abs=0.05)
    line = window.canvas._ruler_items[0].line()
    roi = window._roi
    assert min(line.x1(), line.x2()) <= roi.x, "线没横穿 ROI 左边"
    assert max(line.x1(), line.x2()) >= roi.x + roi.width, "线没横穿 ROI 右边"
    mid_y = (line.y1() + line.y2()) / 2.0
    assert roi.y <= mid_y <= roi.y + roi.height
    render(window.canvas, qt_app)
    close(window)


def test_the_suggestion_goes_out_through_the_existing_straighten_button(qt_app):
    """成功之后就是普通的标尺值：用户点老按钮，走 _apply_ruler 原路。"""
    window = tilt_window(qt_app, -1.5)
    measure_tilt(window)
    measured = window._ruler_angle
    window._apply_ruler("horizontal")
    assert window.rotation_spin.value() == pytest.approx(measured, abs=0.01)
    assert window.canvas._ruler_items == [], "应用后参考线要收掉"
    assert settle(window)
    close(window)


@pytest.mark.parametrize(
    ("case", "keyword"),
    [
        ("no_roi", "先框出 ROI"),
        ("narrow", "250"),
        ("short", "高"),
        ("at_limit", "10"),
        ("no_signal", "看不到条带结构"),
    ],
)
def test_a_refusal_never_eats_the_manual_measurement(qt_app, case, keyword):
    """测不准时只说话。把 _ruler_angle 清成 None 等于替用户扔掉他手里量的值。"""
    window = make_window(qt_app)
    if case == "no_signal":
        load_array(window, np.full((240, 400), 40000, dtype=np.uint16))
    elif case == "at_limit":
        load_array(window, tilted_raw(13.0))
    else:
        load_array(window, tilted_raw(-1.5))
    if case == "no_roi":
        window._roi = None
        window.canvas.clear_roi()
    elif case == "narrow":
        window.canvas.set_roi(20, 70, 200, 100)
        window._apply_roi_change()
    elif case == "short":
        window.canvas.set_roi(20, 110, window._blot.width - 40, 18)
        window._apply_roi_change()
    else:
        window.canvas.set_roi(20, 70, window._blot.width - 40, 100)
        window._apply_roi_change()
    assert settle(window)

    window._ruler_angle = 3.25  # 用户刚手动量的值
    window.ruler_label.setText("标尺倾角：3.25°（相对水平）")
    before = window.canvas._ruler_items[:]
    measure_tilt(window)

    text = window._hint_label.text()
    assert text.startswith("⚠"), f"拒答要看得出是拒答，实际：{text}"
    assert keyword in text, f"没给出 {case} 的具体原因：{text}"
    assert "标尺" in text or "手动" in text, "拒答要指向手动兜底"
    assert window._ruler_angle == pytest.approx(3.25), "手动量的值被抹了"
    assert "3.25" in window.ruler_label.text()
    assert window.canvas._ruler_items == before, "拒答不该动画布"
    close(window)


def test_the_raw_angle_stays_in_the_log(qt_app, monkeypatch):
    """顶到边界时核心层仍带回一个原始极值点——那是诊断用的，不能显示给用户。"""
    logged: list[str] = []
    monkeypatch.setattr(mw.errorlog, "log_message", lambda text: logged.append(text))
    window = make_window(qt_app)
    load_array(window, tilted_raw(13.0))
    window.canvas.set_roi(20, 70, window._blot.width - 40, 100)
    window._apply_roi_change()
    assert settle(window)
    measure_tilt(window)
    assert any("raw=" in line for line in logged), "原始极值点要留档"
    assert "raw" not in window._hint_label.text().lower()
    assert window._ruler_angle is None
    close(window)


def test_tilt_runs_off_the_ui_thread(qt_app, monkeypatch):
    """本身只要几毫秒，但耗时随原图面积涨，大幅面扫描件不该卡住界面。"""
    window = tilt_window(qt_app)
    real = mw.estimate_tilt
    monkeypatch.setattr(
        mw, "estimate_tilt", lambda *a, **k: (time.sleep(0.25), real(*a, **k))[1]
    )
    window._measure_tilt()
    pump(window, 0.05)
    assert window._tilt_running, "应该还在后台跑"
    assert not window.tilt_button.isEnabled(), "跑的时候不许重复点"
    assert not window._busy, "它不产出结果，不该锁住参数区和画布"
    assert settle(window)
    assert window.tilt_button.isEnabled()
    assert window._ruler_angle is not None
    close(window)


def test_a_tilt_result_for_the_previous_image_is_dropped(qt_app):
    """回调排队时用户换了图：这个角度是量另一张图的，落下去就是错的。"""
    window = tilt_window(qt_app, -1.5)
    stale = mw.estimate_tilt(window._blot.raw, roi=window._roi, dark_bands=True)
    assert stale.trusted
    version = window._doc_version
    load_array(window, tilted_raw(2.0))
    window._on_tilt_done(version, None, stale)
    assert window._ruler_angle is None, "上一张图的角度不许落到新图上"
    assert window.canvas._ruler_items == []
    close(window)


def test_the_button_says_not_to_hug_the_membrane_edge(qt_app):
    """这个功能最容易踩的坑得写在按钮上，不是写在 README 里。"""
    window = make_window(qt_app)
    tip = window.tilt_button.toolTip()
    assert "膜边" in tip and "ROI" in tip
    assert "不自动旋转" in tip or "只给建议" in tip
    close(window)


# ------------------------------------------------------------------ U14 发表级导出


def export_window(qt_app, *, angle: float = 0.0) -> MainWindow:
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0, 12000.0, 9000.0]))
    if angle:
        # 和真实流程一样：_blot_original 是整转基准，_blot 是叠了微调角度的显示图
        window._blot = mw.rotate_blot(window._blot_original, float(angle))
        window._angle = float(angle)
        window.canvas.set_image(window._blot.display_bgr)
    set_roi_and_quantify(window)
    return window


def run_export(
    window: MainWindow, monkeypatch, target, options=None, answer=None
) -> list:
    """把两个模态框换成固定答案，跑一次真导出，返回落盘的 FigureExport。"""
    chosen = options or mw.FigureExportOptions()
    monkeypatch.setattr(
        mw.FigureExportDialog, "exec", lambda self: mw.QDialog.DialogCode.Accepted
    )
    monkeypatch.setattr(mw.FigureExportDialog, "options", lambda self: chosen)
    monkeypatch.setattr(
        mw.QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), "TIFF (*.tif)")
    )
    if answer is not None:
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: answer)
    seen: list = []
    monkeypatch.setattr(
        MainWindow, "_show_figure_info", lambda self, exports: seen.extend(exports)
    )
    window._export_figure()
    return seen


def test_the_two_versions_do_not_share_one_mapping(qt_app):
    """线性映射下 8 位标注图只剩两百来个灰阶，条带在屏幕上看不出来。

    干净版是 16 位存档、标注版是给人看的，一个下拉管两个必然有一版是错的。
    """
    window = make_window(qt_app)
    dialog = mw.FigureExportDialog(
        window, width_px=1344, height_px=1100, source_bit_depth=16, white_level=65535.0
    )
    options = dialog.options()
    assert options.clean is True and options.annotated is False
    assert options.clean_mapping == "linear"
    assert options.annotated_mapping == "auto"
    assert options.bit_depth is None, "默认跟随源位深"
    assert options.print_width_mm is None
    assert len(options.layers) == 4, "标注图层默认全勾"
    render(dialog, qt_app)
    dialog.deleteLater()
    close(window)


def test_the_dialog_reports_the_dpi_it_can_actually_reach(qt_app):
    """dpi 是印刷宽度算出来的结果，不是能调大的设置——拉宽度就该看见它掉。"""
    window = make_window(qt_app)
    dialog = mw.FigureExportDialog(
        window, width_px=1200, height_px=800, source_bit_depth=16, white_level=65535.0
    )
    dialog.width_spin.setValue(101.6)  # 4 英寸 → 300 dpi
    assert "300 dpi" in dialog.dpi_label.text()
    dialog.width_spin.setValue(203.2)  # 8 英寸 → 150 dpi
    text = dialog.dpi_label.text()
    assert "150 dpi" in text and "低于 300" in text
    assert "不插值放大" in text
    dialog.deleteLater()
    close(window)


def test_the_dialog_will_not_let_you_export_nothing(qt_app):
    window = make_window(qt_app)
    dialog = mw.FigureExportDialog(
        window, width_px=400, height_px=240, has_result=True, white_level=65535.0
    )
    dialog.clean_check.setChecked(False)
    assert not dialog.ok_button.isEnabled()
    dialog.annotated_check.setChecked(True)
    assert dialog.ok_button.isEnabled()
    dialog.annotated_mapping_combo.setCurrentIndex(2)  # 手动黑白点
    dialog.black_spin.setValue(500.0)
    dialog.white_spin.setValue(400.0)
    assert not dialog.ok_button.isEnabled(), "白点小于黑点要拦住"
    assert "白点必须大于黑点" in dialog.mapping_hint.text()
    dialog.deleteLater()
    close(window)


def test_a_16bit_export_stays_16bit_and_brings_its_sidecar(qt_app, monkeypatch, tmp_path):
    """干净版走线性全量程：16 位单通道、逐像素无损、旁边有说明文件。"""
    import cv2

    window = export_window(qt_app)
    target = tmp_path / "blot_figure.tif"
    exports = run_export(window, monkeypatch, target)
    assert len(exports) == 1
    export = exports[0]
    assert export.path.exists() and export.bit_depth == 16 and export.channels == 1
    assert export.info_path is not None and export.info_path.exists()
    written = cv2.imread(str(target), cv2.IMREAD_UNCHANGED)
    assert written.dtype == np.uint16
    assert written.ndim == 2, "干净版要单通道灰度，不是 BGR"
    info = export.info_path.read_text(encoding="utf-8-sig")
    assert "线性全量程" in info and "未做插值缩放" in info
    assert "dpi" in info
    assert not (tmp_path / "blot_figure_annotated.tif").exists()
    close(window)


def test_the_annotated_copy_lands_beside_it_in_8bit_colour(qt_app, monkeypatch, tmp_path):
    window = export_window(qt_app)
    options = mw.FigureExportOptions(clean=True, annotated=True)
    exports = run_export(window, monkeypatch, tmp_path / "blot_figure.tif", options)
    assert len(exports) == 2
    clean, annotated = exports
    assert clean.bit_depth == 16 and clean.channels == 1
    assert annotated.bit_depth == 8 and annotated.channels == 3
    assert annotated.path.name == "blot_figure_annotated.tif"
    assert annotated.mapping.mode == "auto", "标注版默认自动拉伸"
    assert clean.mapping.mode == "linear", "干净版默认线性全量程"
    assert set(annotated.layers) == set(options.layers)
    close(window)


def test_the_export_rotates_the_original_again_with_lanczos(qt_app, monkeypatch, tmp_path):
    """画布那张是双线性转的（为了拖着流畅）；发表图值得那 4 倍代价。"""
    window = export_window(qt_app, angle=2.5)
    calls: list = []
    real = mw.rotate_blot

    def spy(blot, angle, **kwargs):
        calls.append((blot is window._blot_original, angle, kwargs.get("interpolation")))
        return real(blot, angle, **kwargs)

    monkeypatch.setattr(mw, "rotate_blot", spy)
    exports = run_export(window, monkeypatch, tmp_path / "blot_figure.tif")
    assert calls == [(True, 2.5, mw.ROTATE_SHARP)], f"实际调用：{calls}"
    assert exports[0].rotation_interpolation == mw.ROTATE_SHARP
    info = exports[0].info_path.read_text(encoding="utf-8-sig")
    assert "Lanczos-4" in info and "2.50" in info
    close(window)


def test_a_shape_mismatch_falls_back_instead_of_misplacing_the_roi(
    qt_app, monkeypatch, tmp_path
):
    """ROI 坐标是按画布那张图量的。重转出来的尺寸一旦对不上，宁可用双线性那张
    也不能把 ROI 裁到错的位置上——而且要如实报告用的是哪种插值。"""
    window = export_window(qt_app)
    window._angle = 2.5  # 刻意不同步 _blot：模拟核心层换了旋转实现
    exports = run_export(window, monkeypatch, tmp_path / "blot_figure.tif")
    assert exports[0].rotation_interpolation == mw.ROTATE_FAST
    assert "双线性" in exports[0].info_path.read_text(encoding="utf-8-sig")
    close(window)


def test_no_rotation_means_no_needless_resample(qt_app, monkeypatch, tmp_path):
    """角度是 0 时重转一遍纯属白烧 CPU，还会平白引入一次重采样。"""
    window = export_window(qt_app)
    calls: list = []
    monkeypatch.setattr(
        mw, "rotate_blot", lambda *a, **k: calls.append(a) or mw.rotate_blot(*a, **k)
    )
    run_export(window, monkeypatch, tmp_path / "blot_figure.tif")
    assert calls == []
    close(window)


def test_overwrite_confirmation_covers_all_four_files(qt_app, monkeypatch, tmp_path):
    """一次导出最多落 4 个文件，而保存对话框只替你确认其中一个。"""
    window = export_window(qt_app)
    for name in (
        "blot_figure.tif",
        "blot_figure_export_info.txt",
        "blot_figure_annotated.tif",
        "blot_figure_annotated_export_info.txt",
    ):
        (tmp_path / name).write_text("x", encoding="utf-8")
    listed: list[str] = []

    def fake_question(_parent, _title, text, *a, **k):
        listed.append(text)
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", fake_question)
    options = mw.FigureExportOptions(clean=True, annotated=True)
    exports = run_export(window, monkeypatch, tmp_path / "blot_figure.tif", options)
    assert exports == [], "用户答了不覆盖，就一个字节都不许写"
    assert len(listed) == 1
    # 保存对话框已经替我们确认过 blot_figure.tif 本身，剩下三个必须自己问
    for name in (
        "blot_figure_export_info.txt",
        "blot_figure_annotated.tif",
        "blot_figure_annotated_export_info.txt",
    ):
        assert name in listed[0], f"{name} 会被覆盖却没提"
    assert (tmp_path / "blot_figure_export_info.txt").read_text(encoding="utf-8") == "x"
    close(window)


def test_a_clean_folder_asks_nothing(qt_app, monkeypatch, tmp_path):
    window = export_window(qt_app)
    asked: list = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *a, **k: asked.append(a) or QMessageBox.StandardButton.Yes,
    )
    exports = run_export(window, monkeypatch, tmp_path / "blot_figure.tif")
    assert asked == [], "没冲突就不该打扰"
    assert len(exports) == 1
    close(window)


def test_the_save_dialog_no_longer_offers_a_lossy_format(qt_app, monkeypatch, tmp_path):
    """JPEG 在真实膜上能把灰度挪 6–7 级，和真实的强度差别分不开。"""
    window = export_window(qt_app)
    seen: list[tuple] = []

    def fake_save(*args, **kwargs):
        seen.append(args)
        return "", ""

    monkeypatch.setattr(mw.FigureExportDialog, "exec", lambda self: mw.QDialog.DialogCode.Accepted)
    monkeypatch.setattr(mw.QFileDialog, "getSaveFileName", fake_save)
    window._export_figure()
    assert seen, "没弹保存对话框"
    default_name, filters = seen[0][2], seen[0][3]
    assert filters == "TIFF (*.tif);;PNG (*.png)"
    assert "jpg" not in filters.lower() and "jpeg" not in filters.lower()
    assert default_name.endswith("_figure.tif")
    close(window)


def test_a_typed_suffix_is_coerced_to_a_lossless_one(qt_app, monkeypatch, tmp_path):
    """用户手打 .jpg 时不该抛异常，按选中的过滤器落成无损格式。"""
    window = export_window(qt_app)
    monkeypatch.setattr(
        mw.FigureExportDialog, "exec", lambda self: mw.QDialog.DialogCode.Accepted
    )
    monkeypatch.setattr(mw.FigureExportDialog, "options", lambda self: mw.FigureExportOptions())
    monkeypatch.setattr(
        mw.QFileDialog,
        "getSaveFileName",
        lambda *a, **k: (str(tmp_path / "blot_figure.jpg"), "PNG (*.png)"),
    )
    seen: list = []
    monkeypatch.setattr(MainWindow, "_show_figure_info", lambda self, e: seen.extend(e))
    window._export_figure()
    assert len(seen) == 1
    assert seen[0].path.suffix == ".png"
    assert not (tmp_path / "blot_figure.jpg").exists()
    close(window)


def test_the_sidecar_gets_shown_once(qt_app, monkeypatch, tmp_path):
    """旁注里有一段能直接抄进方法学的话，但没人会主动去开图旁边那个 txt。"""
    window = export_window(qt_app)
    monkeypatch.setattr(
        mw.FigureExportDialog, "exec", lambda self: mw.QDialog.DialogCode.Accepted
    )
    monkeypatch.setattr(mw.FigureExportDialog, "options", lambda self: mw.FigureExportOptions())
    monkeypatch.setattr(
        mw.QFileDialog,
        "getSaveFileName",
        lambda *a, **k: (str(tmp_path / "blot_figure.tif"), "TIFF (*.tif)"),
    )
    shown: list = []
    monkeypatch.setattr(
        mw.FigureInfoDialog, "exec", lambda self: shown.append(self.text()) or 0
    )
    window._export_figure()
    assert len(shown) == 1
    assert "未做插值缩放" in shown[0]
    assert "线性全量程" in shown[0]
    render(window._figure_info_dialog, qt_app)
    close(window)


def test_cancelling_the_options_dialog_writes_nothing(qt_app, monkeypatch, tmp_path):
    window = export_window(qt_app)
    monkeypatch.setattr(
        mw.FigureExportDialog, "exec", lambda self: mw.QDialog.DialogCode.Rejected
    )
    called: list = []
    monkeypatch.setattr(
        mw.QFileDialog, "getSaveFileName", lambda *a, **k: called.append(1) or ("", "")
    )
    window._export_figure()
    assert called == [], "选项都取消了还问文件名"
    assert list(tmp_path.iterdir()) == []
    close(window)


# ------------------------------------------------------------------ U15 角度精度


def test_the_angle_box_does_not_quantise_away_the_measurement(qt_app):
    """自动测角到 0.01° 量级，微调框只有 1 位小数就等于白测。

    端到端盯完整链路：测角 → 写标尺 → 点旋正 → 微调框 → 真正落到图上的 _angle。
    中间任何一环做四舍五入，这里都会露出来。
    """
    window = tilt_window(qt_app, -1.47)
    assert window.rotation_spin.decimals() == 2
    measure_tilt(window)
    measured = window._ruler_angle
    assert measured == pytest.approx(-1.47, abs=0.25)
    assert round(measured, 2) != round(measured, 1), (
        f"这个角度在 1 位小数下看不出差别（{measured}），换不出结论"
    )

    window._apply_ruler("horizontal")
    assert window.rotation_spin.value() == pytest.approx(round(measured, 2), abs=1e-9)
    assert settle(window)
    assert window._angle == pytest.approx(round(measured, 2), abs=1e-9), (
        "微调框留住了 2 位小数，真正转到图上的却被抹回 1 位"
    )
    close(window)


def test_the_coarse_slider_does_not_drag_the_fine_value_back(qt_app):
    """滑杆仍是 0.1° 一格。它跟着走，但不许反过来把微调框的 0.01 位吃掉。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0]))
    window.rotation_spin.setValue(-0.28)
    assert window.rotation_spin.value() == pytest.approx(-0.28)
    assert window.rotation_slider.value() == -3, "滑杆按 0.1° 一格取整"
    assert window.rotation_spin.value() == pytest.approx(-0.28), "取整不许回灌"
    assert "-0.28" in window.rotation_spin.text()

    window._set_rotation_controls(2.34)  # 旋转落地后的回写路径
    assert window.rotation_spin.value() == pytest.approx(2.34)

    window.rotation_slider.setValue(50)  # 手拖滑杆时接受 0.1° 的落点
    assert window.rotation_spin.value() == pytest.approx(5.0)
    assert settle(window)
    close(window)


def test_the_angle_box_is_wide_enough_for_two_decimals(qt_app):
    """加一位小数最容易出的事故是数字被省略号吃掉，这是绘制层的问题。"""
    window = make_window(qt_app)
    window.resize(1400, 900)
    window.show()
    spin = window.rotation_spin
    spin.setValue(-MAX_STRAIGHTEN_ANGLE)
    render(window, qt_app)
    needed = spin.fontMetrics().horizontalAdvance(spin.text())
    assert spin.width() > needed + 12, f"{spin.text()} 放不下（{spin.width()} px）"
    close(window)


# ------------------------------------------------------------------ U16 分隔线警示
#
# roi.py 给候选谷加了条带禁区之后，分隔线切进条带已经很少发生，这是兜底路径。


def separator_pen(window: MainWindow, index: int):
    return window.canvas._sep_items[index].pen()


def cut_separator_window(qt_app, seps=(70.0, 165.0)) -> MainWindow:
    """把第 1 条分隔线拖进条带里——ISSUE_SEPARATOR_IN_BAND 的触发条件。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 16000.0, 12000.0]))
    set_roi_and_quantify(window)
    window._on_profile_separators_changed(list(seps))
    assert settle(window)
    window._run_quantify(synchronous=True)
    assert settle(window)
    return window


def test_the_named_separator_is_marked_on_the_canvas(qt_app):
    """诊断说"第 1 道与第 2 道之间那条线"，画布上就得看得出是哪条。"""
    window = cut_separator_window(qt_app)
    window.resize(1400, 900)
    window.show()
    kinds = [i.kind for i in window._roi_report.issues]
    assert "separator_in_band" in kinds, f"前置条件没成立：{kinds}"

    assert window.canvas.flagged_separators() == [0]
    flagged, normal = separator_pen(window, 0), separator_pen(window, 1)
    assert flagged.color() == canvas_mod.SEPARATOR_FLAG_COLOR
    assert normal.color() == canvas_mod.SEPARATOR_COLOR, "没被点名的线不许跟着变色"
    assert flagged.width() > normal.width(), "加粗是给色觉障碍留的第二条通道"

    tip = window.canvas._sep_items[0].toolTip()
    assert "分隔线切在条带里" in tip
    assert "拖到两条条带之间的空隙" in tip and "删掉" in tip
    assert window.canvas._sep_items[1].toolTip() == ""
    render(window.canvas, qt_app)
    close(window)


def test_the_mark_goes_away_once_the_separator_is_moved_out(qt_app):
    """修好就该收掉——常驻的警示最后一定会被无视。"""
    window = cut_separator_window(qt_app)
    assert window.canvas.flagged_separators() == [0]
    window._on_profile_separators_changed([107.0, 202.0])  # 拖回条带之间的空隙
    assert settle(window)
    window._run_quantify(synchronous=True)
    assert settle(window)
    assert window.canvas.flagged_separators() == []
    assert separator_pen(window, 0).color() == canvas_mod.SEPARATOR_COLOR
    assert window.canvas._sep_items[0].toolTip() == ""
    close(window)


def test_acknowledging_hides_the_separator_mark_too(qt_app):
    """“我知道了”要把画布上的标注一并收起，不能只闭嘴留着满屏红线。"""
    window = cut_separator_window(qt_app)
    assert window.canvas.flagged_separators() == [0]
    window.diagnostics.dismiss()
    qt_app.processEvents()
    assert window.canvas.flagged_separators() == []
    window.diagnostics.dismiss()  # 「重新查看」走同一个开关
    qt_app.processEvents()
    assert window.canvas.flagged_separators() == [0]
    close(window)


def test_keyboard_focus_and_the_flag_stay_visible_together(qt_app):
    """正在用方向键微调一条被点名的线时，两件事都得看得见。"""
    window = cut_separator_window(qt_app)
    window.canvas.set_nudge_target(("sep", 0))
    pen = separator_pen(window, 0)
    assert pen.color() == canvas_mod.SEPARATOR_FLAG_COLOR, "点名占颜色通道"
    assert pen.style() == Qt.PenStyle.DashLine, "微调焦点占线型通道"
    close(window)


def test_deleting_a_separator_drops_the_stale_mark(qt_app):
    """删掉一条线，后面的全往前挪一位——旧下标会指到一条画得好好的线上。"""
    window = cut_separator_window(qt_app)
    assert window.canvas.flagged_separators() == [0]
    window.canvas.set_separators([165.0])  # 少了一条
    assert window.canvas.flagged_separators() == [], "条数变了就不许拿旧下标继续描"
    assert separator_pen(window, 0).color() == canvas_mod.SEPARATOR_COLOR
    close(window)


def test_gappy_lane_numbers_never_mismap_the_separator(qt_app):
    """删过整道之后泳道保号、编号会出现空档，这时按 a-1 换算会指到别的线上。

    宁可不画，也不能把一条画得好好的线描成红的——那比没有标记更糟。
    """
    window = cut_separator_window(qt_app)
    assert window._separators_between((1, 2)) == [0]
    assert window._separators_between((2, 3)) == [1]
    # 核心层把成对的命中摊平成一个去重泳道集合，连号就一起描——宁可多一条也别漏
    assert window._separators_between((1, 2, 3)) == [0, 1]
    assert window._separators_between(()) == []
    assert window._separators_between((3, 4)) == [], "只有 2 条线，第 4 道不存在"
    assert window._separators_between((7, 8)) == []
    close(window)


# ------------------------------------------------------------------ U15 界面密度与可控性
#
# 用户的原话是这一节的纲：「没有人想在分析时看大段的文字」，外加「结果表太窄看
# 不到数值」「面板拖出去就回不来」「想调字体和框线宽度」。四件事共一个判据——
# 默认极简、要看时能展开、任何东西都不许"看不见也回不来"。
#
# 这一节几乎全是绘制与布局，离屏断言不触发 paintEvent，所以每条都 render() 一次：
# 不渲染就测不到"算得出来但画不出来"。


DEFAULT_RESULT_COLUMNS = ["lane", "band", "density", "saturated", "snr"]


def adjust_scroll(window: MainWindow):
    return window._adjust_tab


def widget_fully_inside(widget, container) -> bool:
    """widget 有没有整个落在 container 的视口里（右半截被推出去就是 False）。"""
    top_left = widget.mapTo(container, widget.rect().topLeft())
    return (
        top_left.x() >= 0
        and top_left.x() + widget.width() <= container.width() + 1
    )


def with_font(window: MainWindow, point_size: float):
    """临时改字号；用完必须还原——QApplication.setFont 是进程级的。"""

    class _Scope:
        def __enter__(self):
            window.apply_ui_prefs(
                replace(window.ui_prefs(), font_pt=point_size), persist=False
            )
            QApplication.instance().processEvents()
            return window

        def __exit__(self, *_exc):
            window.apply_ui_prefs(prefs_mod.DEFAULTS, persist=False)
            QApplication.instance().processEvents()
            return False

    return _Scope()


# ---------------------------------------------------------------- 一、结果表只留要紧的列


def test_results_table_shows_only_the_core_columns_by_default(qt_app):
    """默认只显示泳道 / 条带 / 积分灰度 / 饱和 % / S/N，其余收进设置。

    锁定的三列是这张表的骨架（哪一条带 + 产出本身），位置坐标与基线中间量都是
    "核对时才看"的东西。列少了之后表格必须真的放得下——横向滚动条一条都不许有，
    否则这次改动等于白做。
    """
    window = make_window(qt_app)
    window.resize(1400, 900)
    window.show()
    load_array(window, synthetic_raw([20000.0, 16000.0, 12000.0]))
    set_roi_and_quantify(window)
    panel = window.results_panel
    window._tabs.setCurrentWidget(panel)
    qt_app.processEvents()
    render(panel, qt_app)

    assert panel.table.columnCount() == len(rp.HEADERS), "藏起来 ≠ 删掉，列还在"
    assert panel.visible_columns() == DEFAULT_RESULT_COLUMNS
    for key in ("x", "y", "w", "h", "background", "mean"):
        assert panel.table.isColumnHidden(rp.COLUMN_INDEX[key]), f"{key} 默认不该占位置"
    # 用户点名的那一列，怎么都不能少
    assert not panel.table.isColumnHidden(rp.COLUMN_INDEX["density"])

    assert panel.table.horizontalScrollBar().maximum() == 0, (
        "默认列数下结果表仍然要横向滚动，说明列还是太多或者列宽没跟着收"
    )
    close(window)


def test_the_result_values_stay_readable_at_every_font_size(qt_app):
    """9 / 11 / 14 pt、16 行数据：结果表一次都不许横向滚动。

    用户的原话是"看不到所有的数值"。砍列只解决了默认字号那一档，字号调大之后
    同样的 5 列还是会超出停靠区——所以停靠区的最小宽度要跟着表格真正需要的宽度
    走，宁可占掉画布几十像素。
    """
    window = make_window(qt_app)
    window.resize(1500, 950)
    window.show()
    load_array(
        window,
        synthetic_raw(
            [20000.0, 17000.0, 15000.0, 13000.0, 11000.0, 9000.0, 7000.0, 5000.0],
            centers=(104.0, 136.0),
        ),
    )
    set_roi_and_quantify(window)
    panel = window.results_panel
    window._tabs.setCurrentWidget(panel)
    assert panel.table.rowCount() >= 8, "行要够多，才会冒出竖直滚动条挤掉横向空间"

    for point_size in (9.0, 11.0, 14.0):
        with with_font(window, point_size):
            window.resize(1500, 950)
            qt_app.processEvents()
            render(panel, qt_app)
            header = panel.table.horizontalHeader()
            # 几个像素的余量：离屏后端与原生后端在滚动范围上差 1～2 px，那不是
            # "看不到数值"。真正要保证的是下面那条——重要的列没有被压窄。
            assert panel.table.horizontalScrollBar().maximum() <= 4, (
                f"{point_size} pt 下结果表又要横向滚动了"
            )
            for key in ("lane", "band", "density"):
                index = rp.COLUMN_INDEX[key]
                assert header.sectionSize(index) >= header.sectionSizeHint(index), (
                    f"{point_size} pt 下「{rp.HEADERS[index]}」被压到放不下内容"
                )
    close(window)


def test_turning_on_every_column_does_not_swallow_the_canvas(qt_app):
    """12 列全勾上是用户自己的选择：这时让表格滚动，别把画布吞了。"""
    window = make_window(qt_app)
    window.resize(1500, 950)
    window.show()
    load_array(window, synthetic_raw([20000.0, 16000.0, 12000.0]))
    set_roi_and_quantify(window)
    canvas_before = window.canvas.width()

    window.apply_ui_prefs(
        replace(window.ui_prefs(), result_columns=tuple(rp.OPTIONAL_COLUMNS)),
        persist=False,
    )
    qt_app.processEvents()
    assert len(window.results_panel.visible_columns()) >= 11
    assert window._tabs.minimumWidth() <= mw.CONTROL_DOCK_MAX_PX
    assert window.canvas.width() > canvas_before * 0.6, "画布被面板吃掉太多了"
    window.apply_ui_prefs(prefs_mod.DEFAULTS, persist=False)
    close(window)


def test_locked_columns_cannot_be_switched_off(qt_app):
    """把可选列全勾掉，骨架三列还得在——空表比宽表更没用。"""
    panel = rp.ResultsPanel()
    panel.set_optional_columns([])
    assert panel.visible_columns() == list(rp.LOCKED_COLUMNS)
    panel.set_optional_columns(["x", "mean", "不存在的列"])
    assert panel.visible_columns() == ["lane", "band", "x", "density", "mean"]
    panel.deleteLater()


def test_hidden_columns_still_copy_out_in_full(qt_app):
    """显示是显示、导出是导出：复制 TSV 永远带全 12 列。

    否则用户在设置里勾掉一列，粘进 Excel 的数据会跟着少——那是"看不全"变成
    "导不全"，比原来的问题严重得多。
    """
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 16000.0]))
    set_roi_and_quantify(window)
    panel = window.results_panel
    assert "x" not in panel.visible_columns(), "前置条件：X 列是藏起来的"

    panel.table.selectAll()
    panel.copy_selected()
    lines = QApplication.clipboard().text().splitlines()
    assert lines[0].split("\t") == rp.HEADERS
    assert all(len(line.split("\t")) == len(rp.HEADERS) for line in lines[1:])
    close(window)


def test_usage_column_shows_up_only_when_it_says_something(qt_app):
    """「计入」整列都写着"计入"时没有信息量，就别占那一列。"""
    window = doublet_window(qt_app)
    assert "usage" not in window.results_panel.visible_columns()

    window.normalize_panel.set_band_pick("top")
    qt_app.processEvents()
    assert "usage" in window.results_panel.visible_columns(), (
        "有条带没被采用时这一列是判读结果的前提，必须自己冒出来"
    )

    window.normalize_panel.set_band_pick("all")
    qt_app.processEvents()
    assert "usage" not in window.results_panel.visible_columns()
    close(window)


def test_column_choice_survives_a_restart(qt_app):
    """列的显示状态要持久化，否则每次开软件都得重勾一遍。"""
    window = make_window(qt_app)
    window.apply_ui_prefs(
        replace(window.ui_prefs(), result_columns=("x", "y", "background")),
        persist=True,
    )
    assert window.results_panel.visible_columns() == [
        "lane", "band", "x", "y", "density", "background",
    ]
    close(window)

    reopened = MainWindow()
    try:
        assert reopened.ui_prefs().result_columns == ("background", "x", "y")
        assert reopened.results_panel.visible_columns() == [
            "lane", "band", "x", "y", "density", "background",
        ]
    finally:
        reopened.apply_ui_prefs(prefs_mod.DEFAULTS, persist=True)
        close(reopened)


def test_adjust_tab_never_needs_a_horizontal_scrollbar(qt_app):
    """「调整」页在默认停靠宽度下不许横向滚动，多档字号都不许。

    这条盯的是用户直接抱怨的现象：横向滚动条一出现，「归零」按钮和 ROI 数值框
    的右半截就被推到视口外。根源是「ROI 位置尺寸」那行四个数值框排成一排，单行
    最小 488 px 而视口只有 400 出头。
    """
    window = make_window(qt_app)
    window.resize(1400, 900)
    window.show()
    qt_app.processEvents()
    scroll = adjust_scroll(window)
    for point_size in (9.0, 9.5, 11.0, 14.0):
        with with_font(window, point_size):
            window.resize(1400, 900)
            qt_app.processEvents()
            render(scroll, qt_app)
            assert scroll.horizontalScrollBar().maximum() == 0, (
                f"{point_size} pt 下「调整」页仍要横向滚动"
            )
            content = scroll.widget()
            assert content.width() <= scroll.viewport().width() + 1
            for key, spin in window.roi_spins.items():
                assert widget_fully_inside(spin, content), (
                    f"{point_size} pt 下 ROI「{key}」框被推出了视口"
                )
    close(window)


def test_roi_fields_are_two_by_two(qt_app):
    """四个 ROI 数值框排成 2×2：一排四个是横向滚动条的根源。"""
    window = make_window(qt_app)
    window.resize(1400, 900)
    window.show()
    qt_app.processEvents()
    rows = {spin.mapTo(window, spin.rect().topLeft()).y() for spin in window.roi_spins.values()}
    assert len(rows) == 2, f"ROI 数值框占了 {len(rows)} 行，应当是 2×2"
    for spin in window.roi_spins.values():
        assert spin.maximumWidth() >= 10000, "宽度不许写死，否则字号一调就装不下数字"
    close(window)


# ---------------------------------------------------------------- 二、面板关掉之后回得来


def test_view_menu_can_bring_every_panel_back(qt_app):
    """每个停靠面板在「视图 → 面板」里都有一个可勾选的开关。

    这是"关掉之后找不到怎么加回来"的正解：入口必须在菜单里，而且要能看出哪一项
    对应哪个面板。
    """
    window = make_window(qt_app)
    window.show()
    qt_app.processEvents()
    titles = {action.text() for action in window.panels_menu.actions()}
    for dock in (window._control_dock, window._profile_dock):
        assert dock.windowTitle() in titles, f"「{dock.windowTitle()}」在视图菜单里没有开关"

        dock.close()
        qt_app.processEvents()
        assert not dock.isVisible()
        action = next(
            a for a in window.panels_menu.actions() if a.text() == dock.windowTitle()
        )
        assert action.isCheckable() and not action.isChecked()
        action.trigger()
        qt_app.processEvents()
        assert dock.isVisible(), "菜单项勾回来就该把面板放回去"
        assert window.dockWidgetArea(dock) != Qt.DockWidgetArea.NoDockWidgetArea
    assert window.act_reset_layout in window.panels_menu.actions()
    close(window)


def test_floating_panel_can_be_docked_again(qt_app):
    """浮出去还得能拖回来：允许区域与特性都不许收紧。

    Qt 默认就支持双击标题栏归位，但只要有人给 setAllowedAreas / setFeatures 收
    过一刀就会失效，而失效之后用户唯一的出路就是关掉面板——正是这次的投诉。
    """
    window = make_window(qt_app)
    window.show()
    qt_app.processEvents()
    for dock, area in window._dock_defaults():
        assert dock.allowedAreas() == Qt.DockWidgetArea.AllDockWidgetAreas
        features = dock.features()
        assert features & QDockWidget.DockWidgetFeature.DockWidgetMovable
        assert features & QDockWidget.DockWidgetFeature.DockWidgetFloatable

        dock.setFloating(True)
        qt_app.processEvents()
        assert dock.isFloating()
        # 双击标题栏走的就是这一句（QDockWidget::_q_toggleTopLevel）
        dock.setFloating(False)
        qt_app.processEvents()
        assert not dock.isFloating()
        assert window.dockWidgetArea(dock) == area, "拖回去要回到原来那一边"
    close(window)


def test_reset_layout_recovers_a_thoroughly_broken_layout(qt_app):
    """把布局搞乱到最坏：面板浮出去、挪到别的边、还关掉一个，一键全回来。"""
    window = make_window(qt_app)
    window.resize(1400, 900)
    window.show()
    qt_app.processEvents()

    window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, window._profile_dock)
    window._profile_dock.setFloating(True)
    window._control_dock.close()
    qt_app.processEvents()
    assert not window._control_dock.isVisible()

    window.act_reset_layout.trigger()
    qt_app.processEvents()
    render(window, qt_app)

    for dock, area in window._dock_defaults():
        assert dock.isVisible(), f"重置之后「{dock.windowTitle()}」还是看不见"
        assert not dock.isFloating()
        assert window.dockWidgetArea(dock) == area
    assert "重置布局" in window._hint_label.text()
    close(window)


def test_reset_layout_also_forgets_the_bad_layout(qt_app):
    """重置必须连持久化的那份一起清掉，否则下次启动坏布局又回来了。"""
    window = make_window(qt_app)
    window.show()
    qt_app.processEvents()
    settings = QSettings("WBAnalyzer", "WBAnalyzer")
    settings.setValue(mw.LAYOUT_STATE_KEY, window.saveState(mw.LAYOUT_VERSION))
    settings.sync()
    assert settings.value(mw.LAYOUT_STATE_KEY) is not None

    window.act_reset_layout.trigger()
    qt_app.processEvents()
    assert QSettings("WBAnalyzer", "WBAnalyzer").value(mw.LAYOUT_STATE_KEY) is None, (
        "重置只管当前这一次的话，用户存下的坏布局会永远坏着"
    )
    close(window)


def test_unreadable_layout_state_falls_back_to_the_default(qt_app):
    """版本对不上 / 读不出来的布局状态，退回默认布局而不是崩掉或空窗。"""
    donor = make_window(qt_app)
    stale = donor.saveState()  # 0.4.0 那份是无版本号存的，新版本认不出来
    close(donor)
    settings = QSettings("WBAnalyzer", "WBAnalyzer")
    settings.setValue(mw.LAYOUT_STATE_KEY, stale)
    settings.sync()

    window = make_window(qt_app, keep_layout=True)
    window.show()
    qt_app.processEvents()
    for dock, area in window._dock_defaults():
        assert dock.isVisible()
        assert window.dockWidgetArea(dock) == area
    render(window, qt_app)
    close(window)


def test_layout_round_trips_through_settings(qt_app):
    """版本对得上时照常恢复：上次关掉的面板下次仍然关着，重置能救回来。"""
    settings = QSettings("WBAnalyzer", "WBAnalyzer")
    window = make_window(qt_app)
    window.show()
    window._profile_dock.close()
    qt_app.processEvents()
    close(window)
    assert settings.value(mw.LAYOUT_STATE_KEY) is not None

    reopened = make_window(qt_app, keep_layout=True)
    reopened.show()
    qt_app.processEvents()
    assert not reopened._profile_dock.isVisible(), "关掉的面板下次启动仍是关着的"
    reopened.act_reset_layout.trigger()
    qt_app.processEvents()
    assert reopened._profile_dock.isVisible()
    close(reopened)


def test_one_test_cannot_hand_its_layout_to_the_next(qt_app):
    """用例之间不许通过 QSettings 传递布局。

    这是上一条的反面，测的是测试自己：close(window) 每次都会把 win/geometry 与
    dock_state 写回 QSettings，而 qt_app 是 session 级的。不隔离的话，在文件前面
    插一条用例就能改变后面某条的成败——"时红时绿"的一整类抖动都从这儿来。
    """
    donor = make_window(qt_app)
    donor.show()
    donor._profile_dock.close()
    donor.resize(1100, 700)
    qt_app.processEvents()
    close(donor)
    settings = QSettings("WBAnalyzer", "WBAnalyzer")
    assert settings.value(mw.LAYOUT_STATE_KEY) is not None, "前置条件：坏布局真被存下了"

    forget_layout()
    for key in LEAKY_LAYOUT_KEYS:
        assert QSettings("WBAnalyzer", "WBAnalyzer").value(key) is None, f"{key} 没清掉"

    window = make_window(qt_app)
    window.show()
    qt_app.processEvents()
    assert window._profile_dock.isVisible(), (
        "make_window 该清掉持久化布局，上一条用例关掉的面板不能带到这一条"
    )
    close(window)


# ---------------------------------------------------------------- 三、默认极简、严重不折叠


def test_the_summary_line_stays_one_line(qt_app):
    """摘要永远一行：放不下就省略号，全文进 tooltip。

    以前它是 wordWrap 的，标题一多就折成一段话压在画布上——这正是用户说的"没
    有人想在分析时看大段的文字"。
    """
    window = clipped_roi_window(qt_app)
    window.resize(900, 800)
    window.show()
    qt_app.processEvents()
    label = window.diagnostics._text
    render(window.diagnostics, qt_app)

    assert isinstance(label, ElidedLabel)
    assert not label.wordWrap()
    assert label.toolTip() == label.text(), "省略之后全文必须还能悬停看到"
    metrics = QFontMetrics(label.font())
    assert label.height() <= metrics.height() * 2, "摘要占了不止一行"
    # 窄窗下确实会被截断，但截断的是显示、不是内容
    window.resize(640, 800)
    qt_app.processEvents()
    render(window.diagnostics, qt_app)
    assert label.text() == window.diagnostics.summary()
    close(window)


def test_severe_diagnostics_are_never_folded_away(qt_app):
    """「严重」默认就展开，「警告」默认折叠——可以选择不看，不能看不见。"""
    window = clipped_roi_window(qt_app)
    window.resize(1400, 900)
    window.show()
    bar = window.diagnostics
    assert bar.severity() == "error", "前置条件：这个 ROI 该被判成严重"
    assert bar.wants_detail(), "严重级不许默认折叠"
    assert bar.is_expanded()
    render(bar, qt_app)

    # 用户仍然可以收起来
    bar._toggle_detail()
    qt_app.processEvents()
    assert not bar.is_expanded()

    # 但换一批诊断（这里是换个 ROI）之后，严重级要重新展开
    quantify_box(window, 15, 60, window._blot.width - 30, 130)
    assert window.diagnostics.severity() is None, "干净 ROI 该通过"
    quantify_box(window, 15, 108, window._blot.width - 30, 82)
    assert bar.severity() == "error"
    assert bar.is_expanded(), "收起过一次警告，之后来的严重问题就再也不展开了？"
    close(window)


def test_a_warning_stays_folded(qt_app):
    """只有警告时保持折叠：一行摘要够用了，展开与否交给用户。"""
    window = doublet_window(qt_app)
    window.resize(1400, 900)
    window.show()
    set_band_count(window, 3)  # 只有两条带，第三刀没谷可切
    window._roi_report = None  # 让界面兜底的那条 warning 单独在场
    window._refresh_diagnostics()
    qt_app.processEvents()

    bar = window.diagnostics
    assert bar.severity() == "warning", f"前置条件没成立：{bar.entries()}"
    assert not bar.wants_detail(), "警告不该默认摊开"
    assert not bar.is_expanded()
    render(bar, qt_app)

    bar._toggle_detail()
    qt_app.processEvents()
    assert bar.is_expanded(), "想看的时候点一下就得展开"
    render(bar, qt_app)
    close(window)


def test_cross_origin_is_promoted_to_the_diagnostics_bar(qt_app):
    """跨口径以前只写在“归一化”页——那一页关着的时候用户就漏掉了。

    它的含义是"这一组的比值算不出来"，属于 error，必须出现在画布上方那条统一
    诊断里，而且默认展开。归一化页那行只留一句判词。
    """
    window = make_window(qt_app)
    window.resize(1400, 900)
    window.show()
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0]))
    set_roi_and_quantify(window)
    add_history(window, "loading_control", "Actin", "G1")
    window._run_auto(synchronous=True)
    assert settle(window)
    add_history(window, "target", "PAD4", "G1")
    qt_app.processEvents()

    bar = window.diagnostics
    entry = next(e for e in bar.entries() if "口径" in e.title)
    assert entry.severity == "error"
    assert "G1" in entry.title
    assert bar.is_expanded(), "严重级不许折叠"
    assert "拒绝" in entry.detail or "拒绝" in bar.detail_text()
    render(bar, qt_app)

    # 归一化页那行不再复述整段成因，但判词与分组名还在
    hint = window.normalize_panel._result_hint
    assert "拒绝" in hint.text() and "G1" in hint.text()
    assert not hint.wordWrap(), "这行提示不许自动折行去挤结果表"
    close(window)


def test_the_detail_does_not_repeat_the_summary(qt_app):
    """展开详情时，摘要那句不许紧挨着再出现一遍。

    严重级默认展开之后这个重复格外扎眼：第一行"严重｜条带贴着 ROI 上边"，
    紧接着第二行"⛔ 严重·条带贴着 ROI 上边"，同一句话连着写两遍。
    """
    window = clipped_roi_window(qt_app)
    window.resize(1400, 900)
    window.show()
    bar = window.diagnostics
    assert bar.is_expanded()
    render(bar, qt_app)
    title = bar.entries()[0].title
    assert title in bar.summary()
    assert title not in bar._detail.text(), "详情里又把摘要那句抄了一遍"
    # 但别的条目还得有自己的标题，否则几段正文会糊成一片
    window.normalize_panel.set_band_pick("top")
    qt_app.processEvents()
    others = [e for e in bar.entries() if e.title not in bar.summary()]
    for entry in others:
        assert entry.title in bar._detail.text(), "第二条以后的标题不能省"
    close(window)


def test_the_status_bar_reports_state_not_instructions(qt_app):
    """状态栏说这次算出了什么，不复述操作说明。

    以前每次定量完都缀上"剖面图上双击可加分隔线…点选条带 Delete 可删除"，
    而剖面图标题里逐字写着同一句——同一屏上两遍，还每次都念。
    """
    window = make_window(qt_app)
    window.resize(1400, 900)
    window.show()
    load_array(window, synthetic_raw([20000.0, 16000.0, 12000.0]))
    set_roi_and_quantify(window)
    text = window._hint_label.text()
    assert "ROI 定量完成" in text
    assert "双击" not in text and "Delete" not in text, f"状态栏又开始念操作说明：{text}"
    assert isinstance(window._hint_label, ElidedLabel), "状态栏太长时要省略而不是硬切"
    close(window)


def test_the_profile_panel_says_each_thing_once(qt_app):
    """剖面面板里"怎么加 / 怎么删"只说一次（图标题），图例只标颜色。"""
    window = make_window(qt_app)
    window.show()
    panel = window.profile_panel
    qt_app.processEvents()
    texts = [
        label.text()
        for label in panel.findChildren(QLabel)
        if label.text()
    ]
    assert sum(1 for t in texts if "双击" in t) == 0, (
        f"剖面图标题之外还有一份操作说明：{texts}"
    )
    assert "双击加分隔线" in pp.COLUMN_TITLE, "说明本身不能丢，它在图标题里"
    close(window)


def test_the_profile_panel_grows_with_the_font(qt_app):
    """字号调大后剖面面板要跟着长高，否则坐标轴刻度会叠在一起。"""
    window = make_window(qt_app)
    window.resize(1400, 900)
    window.show()
    base = window.profile_panel.minimumHeight()
    with with_font(window, 14.0):
        render(window.profile_panel, qt_app)
        assert window.profile_panel.minimumHeight() > base, (
            "面板不长高的话，14 pt 的刻度会挤成一团"
        )
    assert window.profile_panel.minimumHeight() == base, "回到默认要精确复原"
    close(window)


def test_the_adjust_page_moved_its_prose_into_tooltips(qt_app):
    """「调整」页不再常驻整段说明，但那些话一句都没丢——都在 tooltip 里。"""
    window = make_window(qt_app)
    window.show()
    qt_app.processEvents()
    content = adjust_scroll(window).widget()

    prose = [
        label
        for label in content.findChildren(QLabel)
        if label.wordWrap() and len(label.text()) > 40
    ]
    assert not prose, f"「调整」页还留着常驻长文：{[p.text()[:20] for p in prose]}"

    assert "±" in window.rotation_spin.toolTip()
    assert "方向键" in window.roi_spins["x"].toolTip()
    assert "深色条带" in window.bg_combo.toolTip()
    assert "膜边" in window.tilt_button.toolTip()
    close(window)


# ---------------------------------------------------------------- 四、设置栏


def test_font_size_setting_reaches_every_kind_of_text(qt_app):
    """9 / 11 / 14 pt 各走一遍：普通控件、表格、画布标签、剖面图标题全跟着变。

    画布标签与 pyqtgraph 的字号是另外两套（QGraphicsItem 与内嵌 CSS），不显式
    推一遍就会出现"表格变大了、图里纹丝不动"。
    """
    window = make_window(qt_app)
    window.resize(1400, 900)
    window.show()
    load_array(window, synthetic_raw([20000.0, 16000.0, 12000.0]))
    set_roi_and_quantify(window)

    seen = {}
    for point_size in (9.0, 11.0, 14.0):
        with with_font(window, point_size):
            render(window, qt_app)
            seen[point_size] = (
                window.results_panel.table.font().pointSizeF(),
                window.canvas._label_font().pointSizeF(),
                window.profile_panel._title_size(),
                window.diagnostics._text.font().pointSizeF(),
            )
    sizes = [seen[p] for p in (9.0, 11.0, 14.0)]
    for index, what in enumerate(("结果表", "画布标签", "剖面图标题", "诊断条")):
        values = [s[index] for s in sizes]
        assert values[0] != values[1] != values[2], f"{what}的字号没跟着变：{values}"

    # 回到默认要精确复原，不能一路往上漂
    assert window.ui_prefs().font_pt == prefs_mod.DEFAULT_FONT_PT
    assert window.canvas.annotation_scales()[3] == pytest.approx(1.0)
    assert window.results_panel.table.font().pointSizeF() == pytest.approx(
        seen[9.0][0] * prefs_mod.DEFAULT_FONT_PT / 9.0
    ), "来回调几次之后字号漂了"

    # 开着的对话框也要跟着走（说明书正是要看清字的地方）
    window.act_help.trigger()
    qt_app.processEvents()
    help_box = window._help_box
    assert help_box is not None
    base_pt = help_box.font().pointSizeF()
    with with_font(window, 14.0):
        assert help_box.font().pointSizeF() == pytest.approx(
            base_pt * 14.0 / prefs_mod.DEFAULT_FONT_PT
        ), "说明书窗口没跟着字号走"
    assert help_box.font().pointSizeF() == pytest.approx(base_pt)
    help_box.hide()
    close(window)


def test_bigger_font_does_not_smear_the_band_labels(qt_app):
    """字号调大之后，标签的"三级降级 + 错行避让"还得成立。

    放大等于把同一批标签塞进更小的空间，降级链本来就是为这个设计的——要么排得
    开、要么退成泳道号、要么整批隐藏，就是不许糊成一团。
    """
    window = make_window(qt_app)
    window.resize(1400, 900)
    window.show()
    load_array(window, synthetic_raw([20000.0, 16000.0, 12000.0]))
    set_roi_and_quantify(window)
    canvas = window.canvas
    for point_size in (9.0, 11.0, 14.0):
        with with_font(window, point_size):
            for scale in (0.5, 1.0, 2.0):
                zoom_to(canvas, scale)
                render(canvas, qt_app)
                assert not label_overlaps(canvas), (
                    f"{point_size} pt / {scale:.0%} 下标签互相压住了"
                )
                rects = canvas.band_label_rects()
                assert len(rects) in (0, len(window._result.measurements)), (
                    f"{point_size} pt / {scale:.0%} 下只画了一部分标签"
                )
                left = canvas.mapFromScene(QPointF(0.0, 0.0)).x()
                right = canvas.mapFromScene(QPointF(float(canvas._image_w), 0.0)).x()
                for rect in rects:
                    assert rect.left() >= left - 1 and rect.right() <= right + 1, (
                        f"{point_size} pt 下标签探出了图像边界"
                    )
    close(window)


def line_widths(window: MainWindow) -> dict:
    canvas = window.canvas
    plain = [b for b in canvas._band_items if not b.isSelected() and b.is_adopted()]
    selected = [b for b in canvas._band_items if b.isSelected()]
    assert plain and selected, "这条用例要同时看到一个选中框和一个普通框"
    return {
        "roi": canvas._roi_item.pen().widthF(),
        "band": plain[0].pen().widthF(),
        "band_selected": selected[0].pen().widthF(),
    }


def test_line_width_setting_keeps_every_distinction(qt_app):
    """1× / 2× 各测一遍：线宽跟着变，靠"更粗"表达的区分一个都不许失效。

    画布上颜色（红 / 橙 / 钢蓝）和形状（角标 / 斜纹 / 虚线）已经排满了，"更粗"
    是仅剩的通道——被自检点名的分隔线、选中的条带框都靠它。所以线宽只能做成
    倍数：做成绝对像素，用户把基础线宽调到 5，被点名那条就再也看不出来了。
    """
    window = cut_separator_window(qt_app)
    window.resize(1400, 900)
    window.show()
    qt_app.processEvents()
    assert window.canvas.flagged_separators() == [0], "前置条件：第 0 条线被点名"
    window.canvas._band_items[0].setSelected(True)
    qt_app.processEvents()

    measured = {}
    for scale in (1.0, 2.0):
        window.apply_ui_prefs(
            replace(
                window.ui_prefs(),
                roi_line=scale,
                band_line=scale,
                separator_line=scale,
            ),
            persist=False,
        )
        qt_app.processEvents()
        render(window.canvas, qt_app)
        flagged = separator_pen(window, 0)
        plain = separator_pen(window, 1)
        widths = line_widths(window)
        measured[scale] = (widths, flagged.widthF(), plain.widthF())

        assert flagged.widthF() > plain.widthF(), (
            f"{scale}× 下被点名的分隔线不再比别的粗，区分没了"
        )
        assert flagged.color() == canvas_mod.SEPARATOR_FLAG_COLOR
        assert plain.color() == canvas_mod.SEPARATOR_COLOR
        assert widths["band_selected"] > widths["band"], (
            f"{scale}× 下选中的条带框不再比别的粗"
        )

    for key in ("roi", "band", "band_selected"):
        assert measured[2.0][0][key] == pytest.approx(measured[1.0][0][key] * 2.0), (
            f"{key} 没有按倍数缩放"
        )
    assert measured[2.0][1] == pytest.approx(measured[1.0][1] * 2.0)

    window.apply_ui_prefs(prefs_mod.DEFAULTS, persist=False)
    close(window)


def test_unused_and_flagged_survive_a_thicker_pen(qt_app):
    """线宽调大之后，虚线（未采用）与角标（被点名）这两个通道还在。"""
    window = clipped_roi_window(qt_app)
    window.resize(1400, 900)
    window.show()
    window.normalize_panel.set_band_pick("top")
    window.apply_ui_prefs(
        replace(window.ui_prefs(), band_line=2.5, separator_line=2.5), persist=False
    )
    qt_app.processEvents()
    render(window.canvas, qt_app)

    dashed = [b for b in window.canvas._band_items if not b.is_adopted()]
    assert all(b.pen().style() == Qt.PenStyle.DashLine for b in dashed)
    for index in window.canvas.flagged_bands():
        box = next(b for b in window.canvas._band_items if b.index == index)
        assert box.flag is not None and box.flag._color == box._base, (
            "角标仍然用条带框自己的颜色，没有冒出第五种颜色"
        )
    window.apply_ui_prefs(prefs_mod.DEFAULTS, persist=False)
    close(window)


def test_settings_dialog_applies_live_and_restores_defaults(qt_app):
    """设置对话框：改一下就生效，「恢复默认」一键退回出厂值。"""
    window = make_window(qt_app)
    window.show()
    window.act_settings.trigger()
    qt_app.processEvents()
    dialog = window._settings_dialog
    assert dialog is not None and dialog.isVisible()
    render(dialog, qt_app)

    dialog._font_spin.setValue(13.0)
    dialog._line_spins["separator_line"].setValue(2.0)
    dialog._column_checks["x"].setChecked(True)
    qt_app.processEvents()
    assert window.ui_prefs().font_pt == pytest.approx(13.0)
    assert window.ui_prefs().separator_line == pytest.approx(2.0)
    assert "x" in window.results_panel.visible_columns()
    render(window, qt_app)

    dialog.restore_defaults()
    qt_app.processEvents()
    assert window.ui_prefs() == prefs_mod.DEFAULTS
    assert window.results_panel.visible_columns() == DEFAULT_RESULT_COLUMNS
    assert window.canvas.annotation_scales() == (1.0, 1.0, 1.0, 1.0)
    render(window, qt_app)
    close(window)


def test_prefs_round_trip_through_qsettings(qt_app):
    """全部设置项都要持久化，包括"一列可选列都不勾"这种空值。"""
    settings = QSettings("WBAnalyzer", "WBAnalyzer")
    wanted = prefs_mod.UiPrefs(
        font_pt=12.5,
        roi_line=1.5,
        band_line=0.75,
        separator_line=2.25,
        result_columns=(),
    )
    prefs_mod.save(wanted, settings)
    settings.sync()
    assert prefs_mod.load(settings) == wanted, (
        "空列表经 INI 往返会变成 None，那样“一列都不勾”会被当成“没设置过”"
    )

    prefs_mod.save(prefs_mod.DEFAULTS, settings)
    settings.sync()
    assert prefs_mod.load(settings) == prefs_mod.DEFAULTS


def test_prefs_are_clamped_not_trusted(qt_app):
    """设置文件是可以手改的：越界与垃圾值一律夹回来，不能让界面崩掉。"""
    settings = QSettings("WBAnalyzer", "WBAnalyzer")
    settings.setValue(prefs_mod.KEY_FONT_PT, 999.0)
    settings.setValue(prefs_mod.KEY_ROI_LINE, "abc")
    settings.setValue(prefs_mod.KEY_SEPARATOR_LINE, -3.0)
    settings.setValue(prefs_mod.KEY_RESULT_COLUMNS, "x,不存在的列")
    settings.sync()
    loaded = prefs_mod.load(settings)
    assert loaded.font_pt == prefs_mod.MAX_FONT_PT
    assert loaded.roi_line == prefs_mod.DEFAULT_LINE_SCALE
    assert loaded.separator_line == prefs_mod.MIN_LINE_SCALE
    assert loaded.result_columns == ("x",)

    prefs_mod.save(prefs_mod.DEFAULTS, settings)
    settings.sync()


# ---------------------------------------------------------------- 五、关闭当前图片


class auto_answer:  # noqa: N801 - 当上下文管理器用，名字按动作读
    """把挽留对话框自动答掉，并记下问了几次。

    离屏下 QMessageBox.question 照样是模态的：不接住它，用例会挂到超时。挽留的
    文案与触发条件另有用例专门盯。
    """

    def __init__(self, answer=QMessageBox.StandardButton.Yes) -> None:
        self.answer = answer
        self.asked = 0

    def __enter__(self):
        self._original = QMessageBox.question

        def ask(*_args, **_kwargs):
            self.asked += 1
            return self.answer

        QMessageBox.question = ask
        return self

    def __exit__(self, *_exc):
        QMessageBox.question = self._original
        return False


def document_is_empty(window: MainWindow) -> list[str]:
    """还有哪些"上一张图的痕迹"没清掉。空列表 = 干净的空状态。"""
    leftovers: list[str] = []
    if any(
        blot is not None
        for blot in (window._blot, window._blot_original, window._blot_loaded)
    ):
        leftovers.append("三层图像里还留着东西")
    if window.canvas.has_image():
        leftovers.append("画布上还有图")
    if window._roi is not None or window._separators:
        leftovers.append("ROI / 分隔线还在")
    if window.canvas._band_items:
        leftovers.append("画布上还有条带框")
    if window._result is not None or window.results_panel.table.rowCount():
        leftovers.append("结果还在")
    if window.profile_panel._profile is not None or window.profile_panel._labels:
        leftovers.append("剖面还在")
    if window._undo or window._redo:
        leftovers.append("撤销栈没清")
    if window.diagnostics.isVisible():
        leftovers.append("诊断条还挂着")
    if window.windowTitle() != mw.APP_TITLE:
        leftovers.append(f"标题还是「{window.windowTitle()}」")
    if window._column_profile is not None:
        leftovers.append("横向剖面数据还在")
    return leftovers


def test_the_close_action_sits_where_people_look_for_it(qt_app):
    """文件菜单 + 主工具栏都有入口，Ctrl+W，没有图时禁用。"""
    window = make_window(qt_app)
    assert window.act_close.shortcut() == QKeySequence("Ctrl+W")
    assert window.act_close in window.file_menu.actions(), "文件菜单里要有"
    toolbar = window.findChild(QToolBar, "mainToolbar")
    actions = toolbar.actions()
    assert window.act_close in actions, "工具栏上要有直观的关闭入口，藏进菜单等于没有"
    assert actions.index(window.act_close) == actions.index(window.act_open) + 1, (
        "就该紧挨着“打开”——那是用户找它的第一个地方"
    )
    assert not window.act_close.isEnabled(), "没有图时不该能点"

    load_array(window, synthetic_raw([20000.0, 15000.0]))
    window._update_action_states()
    assert window.act_close.isEnabled()
    close(window)


def test_closing_the_image_empties_the_document_but_not_the_history(qt_app):
    """关闭 = 复位到空状态；归一化历史一行都不许少。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0]))
    window.setWindowTitle(f"{mw.APP_TITLE} — actin.tif")
    set_roi_and_quantify(window)
    add_history(window, "loading_control", "Actin")
    assert len(window._session.entries) == 1, "前置条件：历史里有一条内参"
    assert window._result is not None and window.results_panel.table.rowCount()

    with auto_answer() as dialog:  # 挽留另有用例专门测，这里只看复位干不干净
        window.act_close.trigger()
        qt_app.processEvents()
    assert dialog.asked == 1, "有没导出的结果，关之前得问一句"
    assert document_is_empty(window) == []
    assert len(window._session.entries) == 1, (
        "关图丢历史的话，内参和目标分开拍在两张膜上的跨膜归一化根本做不成"
    )
    assert window.normalize_panel.history_table.rowCount() == 1
    assert "归一化历史保留" in window._hint_label.text()
    close(window)


def test_the_empty_state_is_a_state_you_can_sit_in(qt_app):
    """关掉之后界面照常待着：面板空、要图的动作禁用、渲染不报错。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0]))
    set_roi_and_quantify(window)
    window.resize(1300, 880)
    window.show()
    qt_app.processEvents()
    with auto_answer():
        assert window._close_document() is True

    render(window, qt_app)
    render(window.profile_panel, qt_app)
    render(window.results_panel, qt_app)
    render(window.canvas, qt_app)
    for action in (
        window.act_close,
        window.act_quant,
        window.act_auto,
        window.act_export_data,
        window.act_export_figure,
        window.act_fit,
        window.act_100,
        window.act_zoom_roi,
        window.act_delete,
        window.act_undo,
        window.act_rot_left,
    ):
        assert not action.isEnabled(), f"没有图时「{action.text()}」不该还能点"
    assert window.act_open.isEnabled(), "总得能再开一张"
    assert not window.act_close.isEnabled()
    # 空状态下再关一次：什么都不该发生，更不许抛异常
    with auto_answer() as dialog:
        assert window._close_document() is False
    assert dialog.asked == 0, "本来就没有图，没什么可挽留的"
    assert "没有打开的图片" in window._hint_label.text()
    assert document_is_empty(window) == []
    close(window)


def test_the_cross_membrane_workflow_survives_closing(qt_app):
    """用户的真实流程：开 A 膜 → 加为内参 → 关掉 → 开 B 膜 → 加目标 → 出比值。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0]))
    set_roi_and_quantify(window)
    add_history(window, "loading_control", "Actin")
    window._results_dirty = False
    assert window._close_document() is True

    load_array(window, synthetic_raw([18000.0, 9000.0, 4500.0]))
    set_roi_and_quantify(window)
    add_history(window, "target", "CITH3")
    assert [e.name for e in window._session.entries] == ["Actin", "CITH3"]
    assert window.normalize_panel.result_table.rowCount() == 3, (
        "两张膜的记录必须能配上对，否则跨膜归一化白做"
    )
    assert [result_cell(window, row, 1) for row in range(3)] == ["1", "2", "3"]
    ratios = [result_cell(window, row, 4) for row in range(3)]  # 归一化列
    assert all(float(text.replace(",", "")) > 0 for text in ratios), (
        f"跨膜归一化没算出比值：{ratios}"
    )
    render(window.normalize_panel, qt_app)
    close(window)


def test_closing_says_what_gets_lost_worst_first(qt_app, monkeypatch):
    """挽留：先说"没加入归一化历史"，再说"没导出"。答"否"就一点都不动。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0]))
    set_roi_and_quantify(window)
    total = len(window._result.measurements)
    asked: list[tuple[str, str]] = []
    answer = {"value": QMessageBox.StandardButton.No}

    def ask(_parent, title, text, *_a, **_k):
        asked.append((title, text))
        return answer["value"]

    monkeypatch.setattr(QMessageBox, "question", ask)
    assert window._close_document() is False
    assert window._blot is not None, "被挽留后图必须还在"
    assert window._result is not None
    assert len(asked) == 1
    title, text = asked[0]
    assert title == "关闭当前图片"
    assert f"这张膜的 {total} 个条带还没加入归一化历史" in text
    assert "尚未导出" in text
    assert text.index("归一化历史") < text.index("尚未导出"), (
        "跨膜流程里“没加入历史”比“没导出”损失大，得排在前面"
    )
    assert "确定要关闭这张图片吗？" in text

    answer["value"] = QMessageBox.StandardButton.Yes
    assert window._close_document() is True
    assert len(asked) == 2
    assert document_is_empty(window) == []
    close(window)


def test_closing_stops_nagging_once_the_bands_are_safe(qt_app, monkeypatch):
    """加入历史又导出过了，就没什么可挽留的——不许每次关图都弹一次。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0]))
    set_roi_and_quantify(window)
    add_history(window, "loading_control", "Actin")
    assert window._unhistoried_band_count() == 0, "加过历史就不该再算作会丢的东西"

    window._results_dirty = False  # 等价于"已经导出过"
    asked: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda _p, title, *_a, **_k: asked.append(title)
        or QMessageBox.StandardButton.Yes,
    )
    assert window._close_document() is True
    assert asked == [], "没有任何损失时弹窗只是噪音"
    assert document_is_empty(window) == []
    close(window)


def test_reopening_after_a_close_does_not_inherit_the_old_document(qt_app):
    """复位是给多文档复用的：关掉再开，旧图的 ROI / 结果 / 撤销栈都不许跟过来。"""
    window = make_window(qt_app)
    load_array(window, synthetic_raw([20000.0, 15000.0, 10000.0]))
    set_roi_and_quantify(window)
    before = window._doc_version
    stale_token = fake_inflight_job(window, mw.JOB_QUANTIFY)
    with auto_answer():
        assert window._close_document() is True
    assert window._doc_version > before, "文档版本得往前走，旧任务的回调才认得出过期"
    assert not window._job_is_current(stale_token), "关图后旧任务的令牌必须失效"

    load_array(window, synthetic_raw([9000.0, 6000.0]))
    assert window._roi is None and not window._undo
    set_roi_and_quantify(window)
    assert {lane for lane, _band, _value in bands(window)} == {1, 2}
    close(window)

"""界面层并发与结果一致性回归测试（离屏运行）。

覆盖三块：
* U1：后台任务必须绑定图像身份，换图 / 旋转后旧结果不得落到新图上。
  加载与旋转本身异步化之后，这里额外盯住 doc_version 的递增时机。
* U5：两类后台任务的界面反馈——结果类锁输入、呈现类不锁且后来者顶掉前者。
* U2：删除条带 / 撤销只能过滤，不能换一套算法重测、也不能重排泳道号。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtCore import QEvent, QPointF, QSettings, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from wb_analyzer.imageio import blot_from_array  # noqa: E402
from wb_analyzer.roi import MAX_STRAIGHTEN_ANGLE, RoiRect  # noqa: E402
from wb_analyzer.ui import canvas as canvas_mod  # noqa: E402
from wb_analyzer.ui import errorlog, workers  # noqa: E402
from wb_analyzer.ui import main_window as mw  # noqa: E402
from wb_analyzer.ui import results_panel as rp  # noqa: E402
from wb_analyzer.ui.canvas import Tool  # noqa: E402
from wb_analyzer.ui.main_window import MainWindow  # noqa: E402
from wb_analyzer.ui.tablefmt import format_number  # noqa: E402

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


def make_window(qt_app) -> MainWindow:
    window = MainWindow()
    window.lane_spin.setValue(0)
    window.bg_combo.setCurrentIndex(1)  # 固定“深色条带”，排除自动判定的干扰
    # 显式复位会被持久化的开关，别让上一次运行留下的 QSettings 影响断言
    window.snap_check.setChecked(True)
    window._snap_enabled = True
    window.canvas.set_snap_enabled(True)
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


def add_history(window: MainWindow, kind: str, name: str, group: str = "默认") -> None:
    window.normalize_panel.set_kind(kind)
    window.normalize_panel.set_protein_name(name)
    window.normalize_panel.set_group_name(group)
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
    result, lanes = mw.quantify_roi(blot, roi, [107.0, 202.0], True)

    assert len(calls) == 1, "一次 ROI 定量只该算一遍泳道剖面"
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

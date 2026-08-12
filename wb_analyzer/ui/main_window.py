"""主窗口：组装画布 / 面板 / 工具栏 / 菜单，并承载全部业务流程。

本模块顶部那几个自由函数是界面与 ``wb_analyzer`` 核心层之间**唯一**的调用点，
一律按核心函数的真实签名直接调用：签名对不上就当场 TypeError，而不是悄悄
丢掉参数、退回另一套口径算出一个看着正常的错数字。
"""

from __future__ import annotations

import os
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PySide6.QtCore import QEvent, QObject, QPointF, QSettings, Qt, QThreadPool, QTimer
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QCloseEvent,
    QDragEnterEvent,
    QDropEvent,
    QKeySequence,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .. import quant
from ..analysis import (
    AnalysisConfig,
    AnalysisResult,
    BandMeasurement,
    analyze_array,
)
from ..export import export_figure, export_result
from ..imageio import load_blot, remap_rotation, rotate_blot, saturation_fraction
from ..normalize import (
    AnalysisSession,
    QuantEntry,
    export_normalized_csv,
    lane_series_from_measurements,
)
from ..roi import (
    MAX_STRAIGHTEN_ANGLE,
    RoiRect,
    clamp_straighten_angle,
    default_separators,
    lane_boundaries,
    normalize_angle,
    separators_from_profile,
)
from . import errorlog
from .canvas import BlotCanvas, Tool
from .normalize_panel import NormalizePanel
from .profile_panel import ProfilePanel
from .results_panel import ResultsPanel
from .workers import cancel_all as cancel_workers
from .workers import run_async

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

BG_MODES = (
    ("自动（推荐）", "auto"),
    ("深色条带 / 浅底", "dark_bands"),
    ("浅色条带 / 深底", "light_bands"),
)

MAX_RECENT = 5
MAX_UNDO = 50
# 关窗时最多等后台任务这么久，超时就放弃等待（已取消的任务不会再 emit）
CLOSE_WAIT_MS = 4000

ORIGIN_ROI = "roi"  # 结果来自 quantify_lanes（橡皮筋基线）
ORIGIN_DETECT = "detect"  # 结果来自自动检测（环形中值局部背景）

# 后台任务分两类，界面反馈完全不同：
#   结果类（RESULT）——会产出定量数据，锁住参数区与画布、给取消按钮；
#   呈现类（VIEW）——只改图像/剖面的呈现，不锁任何输入，同类任务后来者顶掉
#                    前者，中间态直接丢弃（旋转滑杆连续拖动就靠这个）。
JOB_QUANTIFY = "quantify"
JOB_DETECT = "detect"
JOB_LOAD = "load"
JOB_ORIENT = "orient"
JOB_PROFILE = "profile"
RESULT_JOBS = frozenset({JOB_QUANTIFY, JOB_DETECT})
VIEW_JOBS = frozenset({JOB_LOAD, JOB_ORIENT, JOB_PROFILE})
JOB_LABELS = {
    JOB_QUANTIFY: "ROI 定量",
    JOB_DETECT: "自动检测",
    JOB_LOAD: "图片加载",
    JOB_ORIENT: "旋转",
    JOB_PROFILE: "剖面计算",
}
# ROI / 分隔线键盘微调步长（图像像素）
NUDGE_STEP = 1.0
NUDGE_STEP_FAST = 10.0

_TOOL_TIPS = {
    Tool.ROI: "框选 ROI：按住左键拖出青色矩形（快捷键 1）",
    Tool.RULER: "标尺对齐：沿条带方向拖一条线，松手后就地选择“旋正为水平/垂直”（快捷键 2）",
    Tool.BAND: "补画条带：拖出绿色矩形添加单个条带（快捷键 3）",
    Tool.ADJUST: "调整：拖动 ROI、手柄与橙色分隔线；双击 ROI 内空白添加分隔线，右键删除（快捷键 4）",
}

_HELP_TEXT = (
    "五步流程：\n\n"
    "1. 打开或拖入 WB 图片（支持 8/16 位 TIF，定量在原生位深上进行）。\n"
    "2. 图片歪斜时，用“标尺对齐”沿条带方向拖一条线，在弹出的小面板里点"
    "“旋正为水平/垂直”；横竖颠倒可用“编辑 → 旋转 90°”。\n"
    "3. 用“框选 ROI”拖出定量区域，橙色分隔线会按剖面峰谷自动划分泳道；"
    "可在画布或底部剖面图中拖动微调，“调整”模式下双击添加、右键删除分隔线。\n"
    "4. 框好 ROI 后会自动定量（ImageJ 式基线扣除），微调分隔线也会自动刷新；"
    "或“自动检测”整图找条带，用“补画条带”补漏。误检条带可直接在图上点选"
    "（Ctrl 多选），按 Delete 删除，或右键“删除此条带 / 删除整个泳道”。\n"
    "5. 在“归一化”页把内参与目标分别加入历史，同组自动配对计算 normalized 与 "
    "fold change，并可导出 CSV 与展示图。\n\n"
    "标注说明：图上 L=泳道（Lane）、B=条带（Band），例如 L2-B1 表示第 2 泳道第 1 条带。\n"
    "删除操作都可用 Ctrl+Z 撤销、Ctrl+Y 重做；清除 ROI 与旋转也在撤销栈里。"
)

# 快捷键表：README 里有、软件里一直没有
_SHORTCUTS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "文件与导出",
        (
            ("Ctrl+O", "打开图片"),
            ("Ctrl+E", "导出数据（CSV + 标注图）"),
            ("Ctrl+Shift+E", "导出展示图"),
            ("Ctrl+Q", "退出"),
        ),
    ),
    (
        "工具切换",
        (
            ("1", "框选 ROI"),
            ("2", "标尺对齐"),
            ("3", "补画条带"),
            ("4", "调整（拖 ROI / 分隔线）"),
        ),
    ),
    (
        "分析",
        (
            ("Ctrl+R / F5", "ROI 定量"),
            ("Ctrl+D", "自动检测"),
        ),
    ),
    (
        "编辑",
        (
            ("Ctrl+Z", "撤销（含删除、清 ROI、旋转）"),
            ("Ctrl+Y / Ctrl+Shift+Z", "重做"),
            ("Delete", "删除选中条带（画布 / 结果表 / 历史表）；画布上 Backspace 同样生效"),
            ("方向键", "微调 ROI 或最近动过的分隔线 1 px"),
            ("Shift+方向键", "同上，步长 10 px"),
            ("Ctrl+方向键", "微调 ROI 的宽 / 高"),
            ("Alt+拖动分隔线", "临时关闭谷底吸附"),
        ),
    ),
    (
        "画布视图",
        (
            ("滚轮", "以光标为中心缩放"),
            ("空格+拖动 / 中键拖动", "平移"),
            ("Ctrl+0", "适应窗口"),
            ("Ctrl+1", "1:1"),
            ("Ctrl+2", "缩放到 ROI"),
            ("Ctrl+= / Ctrl+-", "放大 / 缩小"),
            ("Ctrl+点击条带", "多选条带"),
            ("双击 ROI 内空白", "添加分隔线（调整模式）"),
            ("Esc", "取消当前拖动 / 取消选中"),
            ("F1", "打开本说明"),
        ),
    ),
)


def _app_version() -> str:
    try:
        from .. import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001 - 拿不到版本号不该影响帮助窗口
        return "未知"


class _WheelGuard(QObject):
    """滑杆 / 下拉 / 数字框未聚焦时忽略滚轮，避免滚动面板时误改参数。"""

    def eventFilter(self, obj, event):  # noqa: N802 - Qt API
        if event.type() == QEvent.Type.Wheel and not obj.hasFocus():
            event.ignore()
            return True
        return super().eventFilter(obj, event)


# ---------------------------------------------------------------------- 核心模块调用


def resolve_dark_bands(blot, mode: str) -> bool:
    return bool(quant.resolve_dark_bands_native(blot.raw, blot.white_level, mode=mode))


def column_profile(blot, roi: RoiRect, dark_bands: bool) -> np.ndarray:
    return np.asarray(
        quant.roi_column_profile(
            blot.raw, roi, dark_bands=dark_bands, white_level=blot.white_level
        ),
        dtype=float,
    )


def quantify_roi(blot, roi: RoiRect, separators: list[float], dark_bands: bool):
    """返回 (AnalysisResult, lanes)。

    剖面面板要画的 LaneProfile（剖面 / 基线 / 分段）先算出来，再交给
    quantify_lanes 换成测量值——图上画的基线和表里的数字出自同一份剖面。
    """
    separators = list(separators)
    options = {"dark_bands": dark_bands, "white_level": blot.white_level}
    lanes = quant.compute_lane_profiles(blot.raw, roi, separators, **options)
    result = quant.quantify_lanes(
        blot.raw,
        roi,
        separators,
        display_gray=blot.display_gray,
        lanes=lanes,
        **options,
    )
    return result, lanes


def measure_band(
    blot, x: float, y: float, w: float, h: float, dark_bands: bool
) -> BandMeasurement:
    return quant.measure_box_native(
        blot.raw,
        int(round(x)),
        int(round(y)),
        int(round(w)),
        int(round(h)),
        dark_bands=dark_bands,
        white_level=blot.white_level,
    )


def remeasure(blot, measurements: list, dark_bands: bool) -> AnalysisResult:
    """按环形中值局部背景重新测量一组框。

    只用于自动检测流水线（把 8 位粗检框搬到原生位深上），**不要**用它来处理
    ROI 定量结果的增删：那套结果走的是橡皮筋基线，两个口径不可比，而且
    remeasure_native 会按几何位置重排泳道号。
    """
    return quant.remeasure_native(
        list(measurements),
        blot.raw,
        dark_bands=dark_bands,
        white_level=blot.white_level,
        display_gray=blot.display_gray,
    )


def annotate_result(blot, measurements: list, roi, separators) -> np.ndarray:
    return quant.annotate_on_display(
        blot.display_bgr,
        list(measurements),
        roi=roi,
        separators=[float(s) for s in (separators or [])],
    )


def result_from_measurements(
    blot,
    measurements: list,
    *,
    dark_bands: bool,
    roi=None,
    separators=None,
    base: AnalysisResult | None = None,
) -> AnalysisResult:
    """按给定测量值原样组装 AnalysisResult：只重画标注，绝不重新测量。

    这是删除条带 / 撤销时唯一该走的路径——数值和泳道号必须原封不动。
    """
    kept = list(measurements)
    annotated = annotate_result(blot, kept, roi, separators)
    saturated = sum(1 for m in kept if float(m.saturated_fraction) >= 0.05)
    if base is not None and bool(base.dark_bands) == bool(dark_bands):
        return replace(
            base,
            measurements=kept,
            annotated_image=annotated,
            saturated_band_count=saturated,
        )
    return AnalysisResult(
        measurements=kept,
        annotated_image=annotated,
        dark_bands=bool(dark_bands),
        saturated_band_count=saturated,
        gray_image=blot.display_gray,
    )


def lane_for_center(roi, separators, center_x: float) -> int:
    """手绘条带落在第几个泳道——直接用分隔线序号，和 quantify_lanes 同源。"""
    bounds = lane_boundaries(roi, [float(s) for s in (separators or [])])
    if not bounds:
        return 1
    for index, (x1, x2) in enumerate(bounds, start=1):
        if x1 <= center_x < x2:
            return index
    return 1 if center_x < bounds[0][0] else len(bounds)


def measure_band_roi_style(blot, roi, x, y, w, h, dark_bands: bool):
    """按 quantify_lanes 的橡皮筋基线口径测量一个手绘条带框。

    横向只取手绘框的宽度，纵向基线仍在整个 ROI 高度上拟合——和
    quantify_lanes 里每个泳道的算法完全一致，数值才和其它条带可比。
    """
    if roi is None:
        return None
    box_x = int(round(x))
    box_y = int(round(y))
    box_w = max(1, int(round(w)))
    box_h = max(1, int(round(h)))
    strip = RoiRect(box_x, roi.y, box_w, roi.height).clipped(blot.raw)
    if strip.width < 3 or strip.height < 3:
        return None
    lanes = quant.compute_lane_profiles(
        blot.raw,
        strip,
        [],
        dark_bands=dark_bands,
        white_level=blot.white_level,
        split_bands=False,
        tight=False,
    )
    if not lanes:
        return None
    lane = lanes[0]
    x1 = int(lane.x1)
    x2 = int(lane.x2)
    y0 = int(lane.y0)
    profile = np.asarray(lane.profile, dtype=float)
    baseline = np.asarray(lane.baseline, dtype=float)
    if profile.size == 0 or x2 <= x1:
        return None
    start = int(np.clip(box_y - y0, 0, profile.size - 1))
    stop = int(np.clip(start + box_h, start + 1, profile.size))
    # 同样扣掉噪声地板，否则手绘框拉高一点数值就变大，和泳道定量对不上。
    raw_density = float(np.maximum(profile - baseline, 0.0)[start:stop].sum())
    density = max(0.0, raw_density - float(lane.noise_floor) * (stop - start))
    limit = quant.detection_limit(lane.noise_sigma, stop - start)
    box = blot.raw[y0 + start : y0 + stop, x1:x2]
    per_pixel = float(baseline[start:stop].mean()) / max(x2 - x1, 1)
    background = (blot.white_level - per_pixel) if dark_bands else per_pixel
    saturated = float(
        saturation_fraction(box, dark_bands=dark_bands, white_level=blot.white_level)
    )
    return BandMeasurement(
        lane=0,
        band=0,
        x=x1,
        y=y0 + start,
        width=x2 - x1,
        height=stop - start,
        area=int(box.size),
        mean_intensity=round(float(box.mean()), 3) if box.size else 0.0,
        background=round(float(background), 3),
        integrated_density=round(density, 3),
        saturated_fraction=round(saturated, 4),
        detection_limit=round(limit, 3),
        snr=round(density / limit * quant.LOD_SIGMA, 3) if limit > 0.0 else 0.0,
    )


def compute_roi_profile(blot, roi: RoiRect, bg_mode: str):
    """worker 内一次算完暗带判定与 ROI 列剖面——两者都是整图级 numpy 运算。"""
    dark = resolve_dark_bands(blot, bg_mode)
    return dark, column_profile(blot, roi, dark)


def orient_blot(loaded, quarter_turns: int, angle: float):
    """从**原始加载图**一次做到位：先整 90° 转，再叠微调角度。

    始终从 loaded 派生而不是在当前图上累加，重复旋转才不会一次次插值劣化，
    撤销旋转也才能回到原像素。返回 (整转后的基准图, 最终显示图)——前者是
    微调角度与 ROI 重映射的参照系。
    """
    turns = int(quarter_turns) % 4
    base = rotate_blot(loaded, -90.0 * turns) if turns else loaded
    blot = rotate_blot(base, float(angle)) if abs(float(angle)) > 1e-6 else base
    return base, blot


def profile_valleys(profile: np.ndarray, roi_x: float) -> list[float]:
    """剖面局部极小值的绝对 x 坐标，供分隔线吸附用。

    只做局部极小 + 显著性过滤，不调用核心模块的划分函数：那套函数是按目标
    泳道数给一组分隔线，而吸附需要的是"所有候选谷底"。
    """
    data = np.asarray(profile, dtype=float)
    if data.size < 5:
        return []
    span = float(data.max() - data.min())
    if span <= 0:
        return []
    # 3 点滑动平均压掉单像素噪声，否则每个抖动都成了"谷底"
    smooth = np.convolve(data, np.ones(3) / 3.0, mode="same")
    lower = data.min() + span * 0.6  # 太靠峰顶的凹陷不算谷
    out: list[float] = []
    for i in range(1, smooth.size - 1):
        if smooth[i] <= smooth[i - 1] and smooth[i] <= smooth[i + 1] and smooth[i] < lower:
            if out and (roi_x + i) - out[-1] < 3.0:
                continue  # 平底谷取第一个点就够
            out.append(float(roi_x + i))
    return out


class _Snapshot(NamedTuple):
    """撤销 / 重做栈里的一格状态。

    前 7 个字段参与"状态变了没"的比较；lanes（含 numpy 数组）和 tool 不参与
    ——切工具不该单独产生一步撤销，但撤销时要把工具一起恢复回去。
    """

    roi: tuple | None
    separators: tuple
    measurements: tuple | None
    dark_bands: bool
    origin: str
    angle: float
    quarter_turns: int
    lanes: tuple = ()
    tool: object = None

    @property
    def identity(self) -> tuple:
        return tuple(self)[:7]

    @property
    def orientation(self) -> tuple:
        return (int(self.quarter_turns), round(float(self.angle), 6))


def auto_detect(blot, config: AnalysisConfig):
    """整图粗检 → 原生位深复测。

    复测失败不再吞掉：8 位粗检值与原生位深值差着一个量纲，静默保留 8 位结果
    等于给出一组无法与其它结果比较的数字。
    """
    result = analyze_array(blot.display_bgr, config=config)
    if result.measurements:
        result = remeasure(blot, result.measurements, result.dark_bands)
    return result, []


def normalize_rows(session: AnalysisSession, control_lane: int):
    return session.normalize_all(control_lane=int(control_lane))


# ---------------------------------------------------------------------- 主窗口


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("WB 灰度分析工具")
        self.resize(1440, 900)
        self.setMinimumSize(1100, 680)
        self.setAcceptDrops(True)

        # 三层图像：loaded 是刚读进来的原件（只读、永不覆盖），original 是整
        # 90° 转之后的基准，blot 是再叠微调角度后真正参与定量与显示的那张。
        self._blot_loaded = None
        self._blot_original = None
        self._blot = None
        self._angle = 0.0
        self._quarter_turns = 0
        self._roi: RoiRect | None = None
        self._separators: list[float] = []
        self._result: AnalysisResult | None = None
        self._result_origin = ""
        # 当前结果测自哪一版图像（见 _doc_version）。加载 / 旋转是异步的，
        # 加入历史前要拿它和 _doc_version 比对，别把旧朝向的数字记进去。
        self._result_doc_version = -1
        self._results_dirty = False
        self._lanes: list = []
        self._session = AnalysisSession()
        self._ruler_angle: float | None = None
        self._cursor_pushed = False
        # 图像身份：self._blot 每换一次就 +1。定量 / 检测 / 剖面任务回来先比对，
        # 对不上整包丢弃。**只在图像真的换掉的那一刻递增**（任务完成时），提交
        # 任务时递增会让"加载中提交的定量"骗过校验，U1 就白修了。
        self._doc_version = 0
        # 原件身份：只有打开新文件才 +1，是 orient 任务的校验基准
        self._source_version = 0
        self._job_serial = 0
        self._job_tokens: dict[str, tuple] = {}
        self._workers: dict[str, object] = {}  # kind -> Worker，用于按类取消
        self._running: dict[str, str] = {}  # 在跑的任务 kind -> 状态栏文案
        self._orient_target: tuple = (0, 0.0)  # 最近一次提交的目标朝向
        self._pending_restore: _Snapshot | None = None  # 等旋转到位后再落地的撤销状态
        self._dark_cache: tuple | None = None  # ((doc_version, bg_mode), dark_bands)
        self._column_profile: np.ndarray | None = None
        # 撤销 / 重做栈：("state", 快照) / ("history", 历史记录列表副本)
        self._undo: list[tuple] = []
        self._redo: list[tuple] = []
        self._undo_verb = "已撤销"
        self._committed: _Snapshot = self._EMPTY_STATE
        self._result_stale = False
        self._quant_was_manual = False
        self._quant_manual_pending = False
        self._quant_job_state: tuple | None = None
        self._syncing = False
        self._last_open_dir = ""
        self._last_export_dir = ""
        self._export_dir = ""
        self._recent: list[str] = []
        self._pool = QThreadPool.globalInstance()
        self._wheel_guard = _WheelGuard(self)
        self._help_box = None

        self._rotate_timer = QTimer(self)
        self._rotate_timer.setSingleShot(True)
        self._rotate_timer.setInterval(150)
        self._rotate_timer.timeout.connect(self._apply_rotation)
        self._snap_enabled = True
        self._roi_timer = QTimer(self)
        self._roi_timer.setSingleShot(True)
        self._roi_timer.setInterval(200)
        self._roi_timer.timeout.connect(self._apply_roi_change)
        # ROI / 分隔线一变即自动重新定量（防抖）
        self._quant_timer = QTimer(self)
        self._quant_timer.setSingleShot(True)
        self._quant_timer.setInterval(400)
        self._quant_timer.timeout.connect(self._auto_quantify)

        self.canvas = BlotCanvas(self)
        self.setCentralWidget(self.canvas)
        self.profile_panel = ProfilePanel(self)
        self.results_panel = ResultsPanel(self)
        self.normalize_panel = NormalizePanel(self)

        self._build_actions()
        self._build_adjust_tab()
        self._build_docks()
        self._build_toolbar()
        self._build_menus()
        self._build_statusbar()
        self._build_ruler_popup()
        self._connect_signals()
        self._restore_settings()
        self._refresh_normalize()
        self._update_action_states()
        self._set_status()

    _EMPTY_STATE: _Snapshot = _Snapshot(None, (), None, True, "", 0.0, 0, (), None)

    @property
    def _busy(self) -> bool:
        """"忙"只指会产出结果数据的任务（定量 / 自动检测）。

        加载、旋转、剖面属于呈现类任务，它们跑的时候界面照常可用——否则旋转
        滑杆一拖就把自己锁死了。
        """
        return any(kind in self._running for kind in RESULT_JOBS)

    # ------------------------------------------------------------------ 构建界面

    def _build_actions(self) -> None:
        style = self.style()

        self.act_open = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton), "打开", self
        )
        self.act_open.setShortcut(QKeySequence.StandardKey.Open)

        self._tool_group = QActionGroup(self)
        self._tool_group.setExclusive(True)
        for tool, label, key in (
            (Tool.ROI, "框选 ROI", "1"),
            (Tool.RULER, "标尺对齐", "2"),
            (Tool.BAND, "补画条带", "3"),
            (Tool.ADJUST, "调整", "4"),
        ):
            action = QAction(label, self)
            action.setCheckable(True)
            action.setData(tool)
            action.setShortcut(key)
            action.setToolTip(_TOOL_TIPS[tool])
            self._tool_group.addAction(action)
        self._tool_group.actions()[0].setChecked(True)

        self.act_auto = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay), "自动检测", self
        )
        self.act_auto.setShortcut("Ctrl+D")
        self.act_auto.setToolTip("在整张图上自动寻找条带，按原始位深精确复测（Ctrl+D）")
        self.act_quant = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton),
            "ROI 定量",
            self,
        )
        self.act_quant.setShortcuts([QKeySequence("Ctrl+R"), QKeySequence("F5")])
        self.act_quant.setToolTip(
            "对 ROI 内各泳道做基线扣除定量（Ctrl+R / F5）。"
            "框选或调整 ROI、分隔线后会自动执行，这里可手动重跑"
        )
        self.act_export_data = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton), "导出数据", self
        )
        self.act_export_data.setShortcut("Ctrl+E")
        self.act_export_figure = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_FileIcon), "导出展示图", self
        )
        self.act_export_figure.setShortcut("Ctrl+Shift+E")
        self.act_fit = QAction("适应窗口", self)
        self.act_fit.setShortcut("Ctrl+0")
        self.act_100 = QAction("1:1", self)
        self.act_100.setShortcut("Ctrl+1")
        self.act_zoom_in = QAction("放大", self)
        self.act_zoom_in.setShortcuts([QKeySequence("Ctrl+="), QKeySequence("Ctrl++")])
        self.act_zoom_out = QAction("缩小", self)
        self.act_zoom_out.setShortcut("Ctrl+-")
        self.act_zoom_roi = QAction("缩放到 ROI", self)
        self.act_zoom_roi.setShortcut("Ctrl+2")
        self.act_undo = QAction("撤销", self)
        self.act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.act_redo = QAction("重做", self)
        # StandardKey.Redo 在 Windows 上是 Ctrl+Y，另配 Ctrl+Shift+Z 照顾习惯
        self.act_redo.setShortcuts(
            [QKeySequence.StandardKey.Redo, QKeySequence("Ctrl+Shift+Z")]
        )
        # Delete 语义：只删除“当前选中的对象”（画布条带 / 结果表行 / 历史行，
        # 各自在焦点控件上响应）；此菜单项针对画布选中条带，不再兜底清除 ROI。
        self.act_delete = QAction("删除选中条带", self)
        self.act_delete.setToolTip(
            "删除画布上选中的条带（画布聚焦时直接按 Delete；结果表 / 历史表聚焦时"
            " Delete 删除对应行）"
        )
        self.act_clear_roi = QAction("清除 ROI", self)
        self.act_clear_roi.setToolTip("清除 ROI 与分隔线（可用 Ctrl+Z 撤销，含定量结果）")
        self.act_rot_left = QAction("向左旋转 90°", self)
        self.act_rot_right = QAction("向右旋转 90°", self)
        self.act_guides = QAction("显示参考线", self)
        self.act_guides.setCheckable(True)
        self.act_guides.setChecked(True)
        self.act_exit = QAction("退出", self)
        self.act_exit.setShortcut("Ctrl+Q")
        self.act_help = QAction("使用说明", self)
        self.act_help.setShortcut("F1")

    def _build_adjust_tab(self) -> None:
        rotate_box = QGroupBox("旋转与对齐")
        # 滑杆按 0.1° 一格，与微调框共用核心的 ±MAX_STRAIGHTEN_ANGLE 上限。
        self.rotation_slider = QSlider(Qt.Orientation.Horizontal)
        self.rotation_slider.setRange(
            -int(MAX_STRAIGHTEN_ANGLE * 10), int(MAX_STRAIGHTEN_ANGLE * 10)
        )
        self.rotation_slider.setValue(0)
        self.rotation_spin = QDoubleSpinBox()
        self.rotation_spin.setRange(-MAX_STRAIGHTEN_ANGLE, MAX_STRAIGHTEN_ANGLE)
        self.rotation_spin.setDecimals(1)
        self.rotation_spin.setSingleStep(0.1)
        self.rotation_spin.setSuffix("°")
        zero_button = QPushButton("归零")
        zero_button.clicked.connect(lambda: self.rotation_spin.setValue(0.0))
        rotate_row = QHBoxLayout()
        rotate_row.addWidget(self.rotation_slider, 1)
        rotate_row.addWidget(self.rotation_spin)
        rotate_row.addWidget(zero_button)
        rotate_hint = QLabel(
            f"正值抬高右侧（纠正右低左高），范围 ±{MAX_STRAIGHTEN_ANGLE:.0f}°；"
            "歪得更多请先用“编辑 → 旋转 90°”。也可用标尺拖线后一键旋正。"
        )
        rotate_hint.setWordWrap(True)
        rotate_hint.setObjectName("dimHint")
        self.ruler_label = QLabel("标尺：未测量")
        ruler_row = QHBoxLayout()
        horizontal_button = QPushButton("旋正为水平")
        vertical_button = QPushButton("旋正为垂直")
        horizontal_button.clicked.connect(partial(self._apply_ruler, "horizontal"))
        vertical_button.clicked.connect(partial(self._apply_ruler, "vertical"))
        ruler_row.addWidget(horizontal_button)
        ruler_row.addWidget(vertical_button)
        ruler_row.addStretch(1)
        rotate_layout = QVBoxLayout(rotate_box)
        rotate_layout.addLayout(rotate_row)
        rotate_layout.addWidget(rotate_hint)
        rotate_layout.addWidget(self.ruler_label)
        rotate_layout.addLayout(ruler_row)

        detect_box = QGroupBox("检测参数")
        self.weak_slider = QSlider(Qt.Orientation.Horizontal)
        self.weak_slider.setRange(0, 100)
        self.weak_slider.setValue(55)
        weak_row = QHBoxLayout()
        conservative = QLabel("保守")
        sensitive = QLabel("灵敏")
        for label in (conservative, sensitive):
            label.setObjectName("dimHint")
        weak_row.addWidget(conservative)
        weak_row.addWidget(self.weak_slider, 1)
        weak_row.addWidget(sensitive)
        self.weak_hint = QLabel("当前：适中（漏检时可往右拖）")
        self.weak_hint.setObjectName("dimHint")
        self.bg_combo = QComboBox()
        for label, _value in BG_MODES:
            self.bg_combo.addItem(label)
        bg_hint = QLabel("普通白底深带选“深色条带”；黑底发光选“浅色条带”。一般保持自动即可。")
        bg_hint.setWordWrap(True)
        bg_hint.setObjectName("dimHint")
        detect_form = QFormLayout()
        detect_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        detect_form.addRow("弱条带检出", weak_row)
        detect_form.addRow("", self.weak_hint)
        detect_form.addRow("条带颜色", self.bg_combo)
        detect_layout = QVBoxLayout(detect_box)
        detect_layout.addLayout(detect_form)
        detect_layout.addWidget(bg_hint)

        roi_box = QGroupBox("ROI 与泳道")
        self.lane_spin = QSpinBox()
        self.lane_spin.setRange(0, 24)
        self.lane_spin.setValue(0)
        self.lane_spin.setSpecialValueText("自动")
        self.lane_spin.setToolTip("0 = 按剖面峰自动判断泳道数")
        # ROI 数值框：鼠标拖动受缩放精度限制（20% 缩放时误差 ±2.5 图像像素），
        # 而 ROI 边界直接决定积分区间，需要能精确输入
        self.roi_spins: dict[str, QSpinBox] = {}
        roi_grid = QHBoxLayout()
        roi_grid.setSpacing(4)
        for key, label in (("x", "X"), ("y", "Y"), ("w", "宽"), ("h", "高")):
            spin = QSpinBox()
            spin.setRange(0 if key in ("x", "y") else 1, 999999)
            spin.setKeyboardTracking(False)  # 打字中途不要每敲一位就重算
            spin.setEnabled(False)
            spin.setFixedWidth(74)
            spin.valueChanged.connect(self._on_roi_field_changed)
            spin.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            spin.installEventFilter(self._wheel_guard)
            self.roi_spins[key] = spin
            roi_grid.addWidget(QLabel(label))
            roi_grid.addWidget(spin)
        roi_grid.addStretch(1)
        self.snap_check = QCheckBox("分隔线吸附到剖面谷底")
        self.snap_check.setChecked(True)
        self.snap_check.setToolTip(
            "拖动分隔线时，靠近剖面谷底（泳道之间的低点）会轻微吸附；"
            "明显想放在别处时不会硬拽。按住 Alt 可临时关闭"
        )
        self.snap_check.toggled.connect(self._on_snap_toggled)
        peaks_button = QPushButton("按剖面峰谷划分")
        peaks_button.clicked.connect(self._reset_separators)
        clear_roi_button = QPushButton("清除 ROI")
        clear_roi_button.clicked.connect(self.act_clear_roi.trigger)
        roi_form = QFormLayout()
        roi_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        roi_form.addRow("期望泳道数", self.lane_spin)
        roi_form.addRow("ROI 位置尺寸", roi_grid)
        roi_hint = QLabel("方向键微调 1 px，Shift+方向键 10 px（先点一下画布）。")
        roi_hint.setWordWrap(True)
        roi_hint.setObjectName("dimHint")
        roi_layout = QVBoxLayout(roi_box)
        roi_layout.addLayout(roi_form)
        roi_layout.addWidget(roi_hint)
        roi_layout.addWidget(self.snap_check)
        roi_layout.addWidget(peaks_button)
        roi_layout.addWidget(clear_roi_button)

        display_box = QGroupBox("显示与导出")
        self.guides_check = QCheckBox("显示参考线")
        self.guides_check.setChecked(True)
        self.annotated_check = QCheckBox("导出展示图带检测标注")
        display_layout = QVBoxLayout(display_box)
        display_layout.addWidget(self.guides_check)
        display_layout.addWidget(self.annotated_check)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        layout.addWidget(rotate_box)
        layout.addWidget(detect_box)
        layout.addWidget(roi_box)
        layout.addWidget(display_box)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(content)
        self._adjust_tab = scroll

        # 未聚焦时忽略滚轮：滚动本面板不会误旋转图像 / 误改泳道数与灵敏度
        for widget in (
            self.rotation_slider,
            self.rotation_spin,
            self.weak_slider,
            self.bg_combo,
            self.lane_spin,
        ):
            widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            widget.installEventFilter(self._wheel_guard)

    def _build_docks(self) -> None:
        tabs = QTabWidget()
        tabs.addTab(self._adjust_tab, "调整")
        tabs.addTab(self.results_panel, "结果")
        tabs.addTab(self.normalize_panel, "归一化")
        tabs.setMinimumWidth(380)
        self._tabs = tabs

        self._control_dock = QDockWidget("控制面板", self)
        self._control_dock.setObjectName("controlDock")
        self._control_dock.setWidget(tabs)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._control_dock)

        self._profile_dock = QDockWidget("强度剖面", self)
        self._profile_dock.setObjectName("profileDock")
        self._profile_dock.setWidget(self.profile_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._profile_dock)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("主工具栏", self)
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(toolbar)

        toolbar.addAction(self.act_open)
        toolbar.addSeparator()
        for action in self._tool_group.actions():
            toolbar.addAction(action)
        toolbar.addSeparator()
        toolbar.addAction(self.act_auto)
        toolbar.addAction(self.act_quant)
        quant_button = toolbar.widgetForAction(self.act_quant)
        if quant_button is not None:
            quant_button.setObjectName("primaryButton")
            quant_button.style().unpolish(quant_button)
            quant_button.style().polish(quant_button)
        toolbar.addSeparator()
        toolbar.addAction(self.act_export_data)
        toolbar.addAction(self.act_export_figure)
        toolbar.addSeparator()
        toolbar.addAction(self.act_fit)
        toolbar.addAction(self.act_100)

    def _build_menus(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("文件(&F)")
        file_menu.addAction(self.act_open)
        self._recent_menu = file_menu.addMenu("最近文件")
        self._recent_menu.setToolTipsVisible(True)
        file_menu.addSeparator()
        file_menu.addAction(self.act_export_data)
        file_menu.addAction(self.act_export_figure)
        file_menu.addSeparator()
        file_menu.addAction(self.act_exit)

        edit_menu = menubar.addMenu("编辑(&E)")
        edit_menu.addAction(self.act_undo)
        edit_menu.addAction(self.act_redo)
        edit_menu.addAction(self.act_delete)
        edit_menu.addAction(self.act_clear_roi)
        edit_menu.addSeparator()
        edit_menu.addAction(self.act_rot_left)
        edit_menu.addAction(self.act_rot_right)

        view_menu = menubar.addMenu("视图(&V)")
        view_menu.addAction(self.act_fit)
        view_menu.addAction(self.act_100)
        view_menu.addAction(self.act_zoom_in)
        view_menu.addAction(self.act_zoom_out)
        view_menu.addAction(self.act_zoom_roi)
        view_menu.addSeparator()
        view_menu.addAction(self.act_guides)
        view_menu.addSeparator()
        profile_toggle = self._profile_dock.toggleViewAction()
        profile_toggle.setText("强度剖面面板")
        view_menu.addAction(profile_toggle)
        control_toggle = self._control_dock.toggleViewAction()
        control_toggle.setText("控制面板")
        view_menu.addAction(control_toggle)

        help_menu = menubar.addMenu("帮助(&H)")
        help_menu.addAction(self.act_help)

    def _build_statusbar(self) -> None:
        self._hint_label = QLabel()
        self.statusBar().addWidget(self._hint_label, 1)
        self._busy_bar = QProgressBar()
        self._busy_bar.setRange(0, 0)  # 不确定进度
        self._busy_bar.setTextVisible(False)
        self._busy_bar.setFixedWidth(120)
        self._busy_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self._busy_bar)
        self._cancel_button = QPushButton("取消")
        self._cancel_button.setToolTip("放弃这次计算（结果不会被采用）")
        self._cancel_button.setVisible(False)
        self._cancel_button.clicked.connect(self._cancel_running_jobs)
        self.statusBar().addPermanentWidget(self._cancel_button)
        self._open_folder_button = QPushButton("打开导出文件夹")
        self._open_folder_button.setVisible(False)
        self._open_folder_button.clicked.connect(self._open_export_dir)
        self.statusBar().addPermanentWidget(self._open_folder_button)
        self._zoom_label = QLabel("100%")
        self.statusBar().addPermanentWidget(self._zoom_label)

    def _build_ruler_popup(self) -> None:
        """标尺测角后就地弹出的旋正操作浮层（画布内，非模态）。"""
        popup = QWidget(self.canvas.viewport())
        # 外观由 theme 的 QSS 统一给（#rulerPopup），这里不再内联硬编码色值
        popup.setObjectName("rulerPopup")
        layout = QHBoxLayout(popup)
        layout.setContentsMargins(10, 6, 6, 6)
        layout.setSpacing(6)
        self._ruler_popup_label = QLabel("0.00°")
        horizontal_button = QPushButton("旋正为水平")
        vertical_button = QPushButton("旋正为垂直")
        close_button = QPushButton("✕")
        close_button.setFixedWidth(26)
        close_button.setToolTip("取消（不旋转）")
        horizontal_button.clicked.connect(partial(self._apply_ruler, "horizontal"))
        vertical_button.clicked.connect(partial(self._apply_ruler, "vertical"))
        close_button.clicked.connect(self._dismiss_ruler_popup)
        layout.addWidget(self._ruler_popup_label)
        layout.addWidget(horizontal_button)
        layout.addWidget(vertical_button)
        layout.addWidget(close_button)
        popup.hide()
        self._ruler_popup = popup

    def _connect_signals(self) -> None:
        self.canvas.roiChanged.connect(self._on_canvas_roi_changed)
        self.canvas.roiEditFinished.connect(self._roi_timer.start)
        self.canvas.separatorsChanged.connect(self._on_canvas_separators_changed)
        self.canvas.separatorsEditFinished.connect(self._on_separators_committed)
        self.canvas.rulerMeasured.connect(self._on_ruler_measured)
        self.canvas.bandDrawn.connect(self._on_band_drawn)
        self.canvas.zoomChanged.connect(self._on_zoom_changed)
        self.canvas.bandSelectionChanged.connect(self._on_canvas_band_selection)
        self.canvas.bandsDeleteRequested.connect(self._on_delete_rows)
        self.canvas.roiClearRequested.connect(self._clear_roi)
        self.canvas.resegmentRequested.connect(self._reset_separators)

        self.profile_panel.separatorsChanged.connect(self._on_profile_separators_changed)
        self.profile_panel.editCommitted.connect(self._on_separators_committed)

        self.results_panel.deleteRequested.connect(self._on_delete_rows)
        self.results_panel.laneSelected.connect(self._on_lane_selected)
        self.results_panel.selectionChanged.connect(self._on_table_selection)

        self.normalize_panel.addRequested.connect(self._on_add_to_history)
        self.normalize_panel.deleteHistoryRequested.connect(self._on_delete_history)
        self.normalize_panel.historyEdited.connect(self._on_history_edited)
        self.normalize_panel.exportLcRequested.connect(self._on_export_lc)
        self.normalize_panel.importLcRequested.connect(self._on_import_lc)
        self.normalize_panel.exportNormalizedRequested.connect(self._on_export_normalized)
        self.normalize_panel.controlLaneChanged.connect(
            lambda _v: self._refresh_normalize()
        )

        self.rotation_slider.valueChanged.connect(self._on_slider_rotation)
        self.rotation_spin.valueChanged.connect(self._on_spin_rotation)
        self.weak_slider.valueChanged.connect(self._update_weak_hint)
        self.lane_spin.valueChanged.connect(self._on_lane_count_changed)
        self.guides_check.toggled.connect(self.act_guides.setChecked)

        self.act_open.triggered.connect(self._choose_image)
        self._tool_group.triggered.connect(self._on_tool_action)
        self.act_auto.triggered.connect(lambda: self._run_auto())
        self.act_quant.triggered.connect(lambda: self._run_quantify())
        self.act_export_data.triggered.connect(self._export_data)
        self.act_export_figure.triggered.connect(self._export_figure)
        self.act_fit.triggered.connect(self.canvas.fit_view)
        self.act_100.triggered.connect(self.canvas.zoom_100)
        self.act_zoom_in.triggered.connect(self.canvas.zoom_in)
        self.act_zoom_out.triggered.connect(self.canvas.zoom_out)
        self.act_zoom_roi.triggered.connect(self.canvas.zoom_to_roi)
        self.act_undo.triggered.connect(self._undo_last)
        self.act_redo.triggered.connect(self._redo_last)
        self.act_delete.triggered.connect(self._delete_selected_bands)
        self.act_clear_roi.triggered.connect(self._clear_roi)
        self.act_rot_left.triggered.connect(partial(self._rotate_90, False))
        self.act_rot_right.triggered.connect(partial(self._rotate_90, True))
        self.act_guides.toggled.connect(self._on_guides_toggled)
        self.act_exit.triggered.connect(self.close)
        self.act_help.triggered.connect(self._show_help)

    def _show_help(self) -> None:
        """非模态使用说明：分“流程 / 快捷键 / 关于”三页，可以边看边操作。"""
        if self._help_box is None:
            dialog = QDialog(self)
            dialog.setWindowTitle("使用说明")
            dialog.setModal(False)
            dialog.resize(620, 520)
            tabs = QTabWidget(dialog)

            flow = QTextBrowser()
            flow.setPlainText(_HELP_TEXT)
            tabs.addTab(flow, "操作流程")

            table = QTableWidget(0, 2)
            table.setHorizontalHeaderLabels(["快捷键", "作用"])
            table.verticalHeader().setVisible(False)
            table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
            for group, rows in _SHORTCUTS:
                header = QTableWidgetItem(group)
                header.setFlags(Qt.ItemFlag.ItemIsEnabled)
                font = header.font()
                font.setBold(True)
                header.setFont(font)
                row = table.rowCount()
                table.insertRow(row)
                table.setItem(row, 0, header)
                table.setSpan(row, 0, 1, 2)
                for keys, what in rows:
                    row = table.rowCount()
                    table.insertRow(row)
                    table.setItem(row, 0, QTableWidgetItem(keys))
                    table.setItem(row, 1, QTableWidgetItem(what))
            table.resizeColumnsToContents()
            table.horizontalHeader().setStretchLastSection(True)
            tabs.addTab(table, "快捷键")

            about = QTextBrowser()
            about.setPlainText(
                f"WB 灰度分析工具\n版本 {_app_version()}\n\n"
                "Western Blot 条带灰度定量：ImageJ 式基线扣除、原生位深（8/16 位 "
                "TIF）定量、内参归一化与 fold change。\n\n"
                f"日志文件：{errorlog.log_path() or '未启用'}\n"
                "遇到崩溃或异常时，请把该日志一并提供。"
            )
            tabs.addTab(about, "关于")

            close_button = QPushButton("关闭")
            close_button.clicked.connect(dialog.hide)
            bar = QHBoxLayout()
            bar.addStretch(1)
            bar.addWidget(close_button)
            layout = QVBoxLayout(dialog)
            layout.addWidget(tabs, 1)
            layout.addLayout(bar)
            self._help_box = dialog
        self._help_box.show()
        self._help_box.raise_()

    # ------------------------------------------------------------------ 打开图片

    def _choose_image(self) -> None:
        if self._reject_while_busy("打开图片"):
            return
        start = self._last_open_dir or str(Path.home())
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择 Western Blot 图片",
            start,
            "图片 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;所有文件 (*.*)",
        )
        if filename:
            self.open_path(Path(filename))

    def open_path(self, path: Path | str, synchronous: bool = False) -> bool:
        """提交一次图片加载（后台线程）。

        返回值表示"这次请求已受理"，而不是"图片已经读好了"——真正的成败在
        :meth:`_on_load_done` / :meth:`_on_load_error` 里揭晓。大图 load_blot
        要几百毫秒到几秒，摆在 UI 线程里会整窗卡死。
        """
        path = Path(path)
        if self._reject_while_busy("打开图片"):
            return False
        # 换图会作废在跑的定量 / 检测 / 旋转 / 剖面；两次加载之间后者顶掉前者
        abandoned = self._invalidate_jobs(
            {JOB_QUANTIFY, JOB_DETECT, JOB_ORIENT, JOB_PROFILE}
        )
        self._rotate_timer.stop()
        self._roi_timer.stop()
        self._quant_timer.stop()
        self._quant_manual_pending = False
        token = self._begin_job(JOB_LOAD, f"正在加载 {path.name}……")
        job = partial(load_blot, path)
        if synchronous:
            try:
                blot = job()
            except Exception as exc:  # noqa: BLE001 - 同步路径也要兜住
                self._on_load_error(token, path, str(exc))
                return False
            self._on_load_done(token, path, abandoned, blot)
            return True
        self._submit_job(
            token,
            job,
            partial(self._on_load_done, token, path, abandoned),
            partial(self._on_load_error, token, path),
        )
        return True

    def _on_load_error(self, token: object, path: Path, message: str) -> None:
        if not self._job_is_current(token):
            errorlog.log_message(f"忽略过期的加载报错：{path} {message}")
            return
        self._finish_job(token)
        errorlog.log_message(f"打开图片失败：{path} {message}")
        QMessageBox.critical(
            self,
            "无法打开图片",
            f"{path.name} 读取失败：\n{message}" + errorlog.hint(),
        )
        self._set_status(f"⚠ 无法打开 {path.name}：{message}")

    def _on_load_done(
        self, token: object, path: Path, abandoned: bool, blot: object
    ) -> None:
        if not self._job_is_current(token):
            self._discard_stale_job(token, "图片加载")
            return
        self._finish_job(token)
        # 同尺寸、未做微调旋转时沿用 ROI / 分隔线（同膜 strip 重探是标准流程）
        reuse = (
            self._roi is not None
            and self._blot is not None
            and abs(self._angle) < 1e-6
            and self._quarter_turns == 0
            and blot.width == self._blot.width
            and blot.height == self._blot.height
        )
        prev_roi = self._roi
        prev_seps = list(self._separators)
        self._blot_loaded = blot
        self._blot_original = blot
        self._blot = blot
        self._angle = 0.0
        self._quarter_turns = 0
        self._orient_target = (0, 0.0)
        self._pending_restore = None
        # 图像换了：doc / source 版本双双 +1，此前提交的任何任务都作废
        self._bump_document()
        self._source_version += 1
        self._result_stale = False
        self._set_rotation_controls(0.0)
        self._ruler_angle = None
        self.ruler_label.setText("标尺：未测量")
        self._ruler_popup.hide()
        self._roi = None
        self._separators = []
        self._undo.clear()
        self._redo.clear()
        self._committed = self._EMPTY_STATE
        self.canvas.clear_roi()
        self.canvas.clear_result_boxes()
        shown = self.canvas.set_image(blot.display_bgr)
        self.canvas.fit_view()
        self.profile_panel.clear()
        self._column_profile = None
        self._clear_result()
        self._last_open_dir = str(path.parent)
        self._add_recent(path)
        self.setWindowTitle(f"WB 灰度分析工具 — {path.name}")
        # 加载完立刻把焦点给画布：Space 平移 / Delete 删条带 / 方向键微调
        # 都是画布的按键，否则用户得先点一下画布才生效
        self.canvas.setFocus(Qt.FocusReason.OtherFocusReason)
        loaded = f"已加载 {path.name}（{blot.width}×{blot.height}，{blot.bit_depth} 位）。"
        if abandoned:
            loaded = "⚠ 已放弃上一张图正在进行的计算。" + loaded
        if not shown:
            self._update_action_states()
            self._set_status(
                loaded + "⚠ 图片过大，画布无法显示预览；定量仍按原始数据进行。"
            )
            return
        if reuse and prev_roi is not None:
            rect = prev_roi.clipped(blot.raw)
            self._roi = rect
            self._separators = prev_seps
            self.canvas.set_roi(rect.x, rect.y, rect.width, rect.height)
            self.canvas.set_separators(self._separators)
            self._sync_roi_fields()
            self._refresh_column_profile()
            self._commit_snapshot()
            self._select_tool(Tool.ADJUST)
            self._schedule_quantify()
            self._set_status(
                loaded + "尺寸相同，已沿用上一张的 ROI 与分隔线并自动定量；"
                "不需要可点“清除 ROI”重画。"
            )
        else:
            self._select_tool(Tool.ROI)
            self._sync_roi_fields()
            self._set_status(loaded + "拖拽画出 ROI 框，或点“自动检测”。")
        self._update_action_states()

    # ------------------------------------------------------------------ 旋转

    def _set_rotation_controls(self, angle: float) -> None:
        self._syncing = True
        try:
            self.rotation_slider.setValue(int(round(angle * 10)))
            self.rotation_spin.setValue(angle)
        finally:
            self._syncing = False

    def _on_slider_rotation(self, value: int) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self.rotation_spin.setValue(value / 10.0)
        finally:
            self._syncing = False
        self._queue_rotation()

    def _on_spin_rotation(self, value: float) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self.rotation_slider.setValue(int(round(value * 10)))
        finally:
            self._syncing = False
        self._queue_rotation()

    def _queue_rotation(self) -> None:
        if self._blot_loaded is not None:
            self._rotate_timer.start()

    def _apply_rotation(self) -> None:
        """滑杆停稳（150 ms 防抖）后提交一次旋转任务。"""
        if self._blot_loaded is None:
            return
        if self._view_pending(exclude={JOB_ORIENT, JOB_PROFILE}):
            self._rotate_timer.start()  # 图还在加载，等它落地再转
            return
        self._request_orientation(self._quarter_turns, float(self.rotation_spin.value()))

    def _request_orientation(
        self,
        turns: int,
        angle: float,
        *,
        restore: _Snapshot | None = None,
        message: str | None = None,
    ) -> bool:
        """提交一次朝向变更：整 90° 转与微调角度合成同一个任务。

        比较的是"在跑的那次任务的目标朝向"而不是当前朝向：用户从 0 拖到 5 又
        拖回 0 时，5° 那次还在路上，必须再提交一次 0° 把它顶掉，否则 5° 落地
        就成了最终结果。
        """
        if self._blot_loaded is None:
            return False
        turns = int(turns) % 4
        angle = float(angle)
        target = (turns, round(angle, 6))
        if restore is None and target == self._orient_target:
            return False
        self._orient_target = target
        self._pending_restore = restore
        if message is None:
            spun = f"{turns * 90}° + " if turns else ""
            message = f"正在旋转（{spun}{angle:+.1f}°）……"
        token = self._begin_job(JOB_ORIENT, message)
        self._submit_job(
            token,
            partial(orient_blot, self._blot_loaded, turns, angle),
            partial(self._on_orient_done, token, turns, angle),
            partial(self._on_orient_error, token),
        )
        return True

    def _on_orient_error(self, token: object, message: str) -> None:
        if not self._job_is_current(token):
            errorlog.log_message(f"忽略过期的旋转报错：{message}")
            return
        self._finish_job(token)
        self._orient_target = (self._quarter_turns, round(self._angle, 6))
        self._pending_restore = None
        errorlog.log_message(f"旋转失败：{message}")
        QMessageBox.critical(self, "旋转失败", message + errorlog.hint())
        self._set_rotation_controls(self._angle)
        self._set_status(f"⚠ 旋转失败：{message}")

    def _remap_roi_for_rotation(
        self, shape_hw: tuple, old_angle: float, new_angle: float, blot
    ) -> tuple:
        """把 ROI 与分隔线从旧角度的坐标系搬到新角度（只算 4 个角点，很轻）。

        故意留在 UI 线程做：worker 提交时的"旧角度"可能已经因为超车而不是落地
        时的真实旧角度，只有在回调里用 self._angle 才对得上。
        """
        r = self._roi
        if r is None:
            return None, []
        corners = np.array(
            [
                [r.x, r.y],
                [r.x + r.width, r.y],
                [r.x + r.width, r.y + r.height],
                [r.x, r.y + r.height],
            ],
            dtype=float,
        )
        mapped = remap_rotation(corners, shape_hw, old_angle, new_angle)
        x1 = float(mapped[:, 0].min())
        y1 = float(mapped[:, 1].min())
        x2 = float(mapped[:, 0].max())
        y2 = float(mapped[:, 1].max())
        new_roi = RoiRect(
            int(round(x1)),
            int(round(y1)),
            max(1, int(round(x2 - x1))),
            max(1, int(round(y2 - y1))),
        ).clipped(blot.raw)
        new_seps: list[float] = []
        if self._separators:
            cy = r.y + r.height / 2.0
            anchors = np.array([[s, cy] for s in self._separators], dtype=float)
            mapped_x = remap_rotation(anchors, shape_hw, old_angle, new_angle)[:, 0]
            lo = new_roi.x + 2.0
            hi = new_roi.x + new_roi.width - 2.0
            if hi > lo:
                new_seps = sorted(float(np.clip(v, lo, hi)) for v in mapped_x)
        return new_roi, new_seps

    def _on_orient_done(
        self, token: object, turns: int, angle: float, payload: object
    ) -> None:
        if not self._job_is_current(token):
            self._discard_stale_job(token, "旋转")
            return
        self._finish_job(token)
        base, blot = payload
        restore = self._pending_restore
        self._pending_restore = None
        turn_changed = int(turns) != int(self._quarter_turns)
        # 重映射必须在替换 _blot_original 之前算：参照系是旧的基准图形状
        new_roi: RoiRect | None = None
        new_seps: list[float] = []
        if restore is None and not turn_changed and self._roi is not None:
            shape = (self._blot_original or base).raw.shape[:2]
            new_roi, new_seps = self._remap_roi_for_rotation(
                shape, self._angle, angle, blot
            )
        # 图像换了：文档版本 +1，在跑的定量 / 检测 / 剖面结果一律作废
        self._bump_document()
        self._invalidate_jobs({JOB_QUANTIFY, JOB_DETECT, JOB_PROFILE})
        self._blot_original = base
        self._blot = blot
        self._quarter_turns = int(turns)
        self._angle = float(angle)
        self._orient_target = (int(turns), round(float(angle), 6))
        self._set_rotation_controls(float(angle))
        self.canvas.set_image(blot.display_bgr)
        if turn_changed:
            self.canvas.fit_view()

        if restore is not None:
            # 撤销 / 重做把朝向转回去了，接着把那一格的 ROI 与结果原样落地
            self._apply_state_geometry(restore)
            return

        self._clear_result()
        self._result_stale = False
        if turn_changed:
            # 90° 整转换了坐标系，ROI 与结果没法沿用（Ctrl+Z 可整步撤回）
            self._roi = None
            self._separators = []
            self._column_profile = None
            self.canvas.clear_roi()
            self.canvas.clear_result_boxes()
            self.profile_panel.clear()
            self._sync_roi_fields()
            self._commit_snapshot()
            self._set_status(
                f"已整体旋转 {self._quarter_turns * 90}°，请重新框选 ROI（Ctrl+Z 可撤销）。"
            )
        elif new_roi is not None:
            self._roi = new_roi
            self._separators = new_seps
            self.canvas.set_roi(new_roi.x, new_roi.y, new_roi.width, new_roi.height)
            self.canvas.set_separators(new_seps)
            self._sync_roi_fields()
            self._refresh_column_profile()
            self._commit_snapshot()
            self._schedule_quantify()
            self._set_status(
                f"已旋转 {angle:.1f}°，ROI 与分隔线已随图重映射，正在自动重新定量……"
            )
        else:
            self._roi = None
            self._separators = []
            self._column_profile = None
            self.canvas.clear_roi()
            self.profile_panel.clear_column_profile()
            self._sync_roi_fields()
            self._commit_snapshot()
            self._set_status(f"已旋转 {angle:.1f}°。")
        self._update_action_states()

    def _rotate_90(self, clockwise: bool) -> None:
        """整图 90° 旋转（横竖颠倒的扫描图）。

        不再清空撤销栈：朝向本身现在是快照的一部分，且旋转结果一律从**原始
        加载图**重新派生（见 :func:`orient_blot`），所以 Ctrl+Z 能连同 ROI 与
        结果一起整步回到旋转前，也不会因为反复旋转而一层层插值劣化。
        """
        if self._blot_loaded is None:
            return
        if self._reject_while_busy("旋转图片"):
            return
        self._rotate_timer.stop()
        self._roi_timer.stop()
        self._quant_timer.stop()
        turns = (self._quarter_turns + (1 if clockwise else 3)) % 4
        self._ruler_angle = None
        self.ruler_label.setText("标尺：未测量")
        self._ruler_popup.hide()
        self.canvas.clear_ruler()
        direction = "右" if clockwise else "左"
        self._request_orientation(
            turns, 0.0, message=f"正在向{direction}旋转 90°……"
        )

    def _on_ruler_measured(self, angle: float, end_x: float, end_y: float) -> None:
        self._ruler_angle = normalize_angle(angle)
        self.ruler_label.setText(f"标尺倾角：{self._ruler_angle:.2f}°（相对水平）")
        self._show_ruler_popup(end_x, end_y)
        self._set_status("标尺已测量：就地点“旋正为水平/垂直”，或按 ✕ 取消。")

    def _show_ruler_popup(self, scene_x: float, scene_y: float) -> None:
        if self._ruler_angle is None:
            return
        self._ruler_popup_label.setText(f"{self._ruler_angle:+.2f}°")
        self._ruler_popup.adjustSize()
        viewport = self.canvas.viewport()
        anchor = self.canvas.mapFromScene(QPointF(float(scene_x), float(scene_y)))
        x = min(
            max(anchor.x() + 14, 4), max(4, viewport.width() - self._ruler_popup.width() - 4)
        )
        y = min(
            max(anchor.y() - self._ruler_popup.height() - 10, 4),
            max(4, viewport.height() - self._ruler_popup.height() - 4),
        )
        self._ruler_popup.move(int(x), int(y))
        self._ruler_popup.show()
        self._ruler_popup.raise_()

    def _dismiss_ruler_popup(self) -> None:
        self._ruler_popup.hide()
        self.canvas.clear_ruler()
        self._set_status("已取消标尺旋正。")

    def _apply_ruler(self, orientation: str) -> None:
        if self._blot_loaded is None:
            return
        if self._ruler_angle is None:
            self._select_tool(Tool.RULER)
            self._set_status("⚠ 请先用“标尺对齐”工具在图上沿条带拖一条线。")
            return
        current = float(self.rotation_spin.value())
        delta = self._ruler_angle if orientation == "horizontal" else self._ruler_angle - 90.0
        wanted = float(normalize_angle(current + delta))
        new_angle = clamp_straighten_angle(wanted)
        clamped = abs(wanted - new_angle) > 0.05
        self._ruler_popup.hide()
        self.canvas.clear_ruler()
        self._ruler_angle = None
        self.ruler_label.setText(
            "标尺：已应用旋正（未转足）" if clamped else "标尺：已应用旋正"
        )
        self.rotation_spin.setValue(new_angle)
        # 旋正后回到下一步该用的工具：没 ROI 就框选，有 ROI 就调整
        self._select_tool(Tool.ROI if self._roi is None else Tool.ADJUST)
        label = "水平" if orientation == "horizontal" else "垂直"
        if clamped:
            # 静默钳制最坑：图还是歪的，状态栏却说旋正好了
            self._set_status(
                f"⚠ 标尺要求转 {wanted:+.2f}°，超出微调范围 ±{MAX_STRAIGHTEN_ANGLE:.0f}°，"
                f"只转到 {new_angle:+.2f}°——图像仍未旋正为{label}。"
                "请先用“编辑 → 旋转 90°”把方向摆正，再用标尺微调。"
            )
        else:
            self._set_status(f"已按标尺旋正为{label}（{new_angle:.2f}°）。")

    # ------------------------------------------------------------------ ROI / 分隔线

    def _on_canvas_roi_changed(self, x: float, y: float, w: float, h: float) -> None:
        if self._blot is None:
            return
        self._quant_timer.stop()
        # 与拖分隔线保持一致：只标记结果过期，不清空表格。无条件清空会让拖
        # 一次 ROI 就闪一下空表，撤销之前也看不到原来的数值
        self._mark_result_stale()
        self._sync_roi_fields(
            (
                int(round(x)),
                int(round(y)),
                max(1, int(round(w))),
                max(1, int(round(h)))
            )
        )
        self._roi_timer.start()

    def _apply_roi_change(self, recompute: bool | None = None) -> None:
        """ROI 稳定后刷新分隔线与剖面。

        recompute=True 强制按剖面峰谷重划；默认（None）在移动 / 缩放已有 ROI 时
        保留画布上现有分隔线（含手动调整），只有新画 ROI 才重新计算。
        真正的计算（暗带判定 + 列剖面，都是整图级 numpy）走后台任务。
        """
        self._roi_timer.stop()
        if self._blot is None:
            return
        if self.canvas.is_interacting():
            self._roi_timer.start()
            return
        roi_t = self.canvas.roi()
        if roi_t is None:
            return
        x, y, w, h = roi_t
        rect = RoiRect(
            int(round(x)), int(round(y)), max(1, int(round(w))), max(1, int(round(h)))
        ).clipped(self._blot.raw)
        created = self._roi is None
        canvas_seps = self.canvas.separators()
        self._roi = rect
        self._sync_roi_fields()
        if rect.width < 3 or rect.height < 3:
            self._set_status("ROI 太小，请拖大一点再松开。")
            return
        self.canvas.set_roi(rect.x, rect.y, rect.width, rect.height)
        keep = recompute is not True and not created and bool(canvas_seps)
        self._request_profile(
            keep=keep,
            created=created,
            commit=True,
            status=f"ROI {rect.width}×{rect.height}，正在计算剖面与泳道划分……",
        )

    def _request_profile(
        self,
        *,
        keep: bool,
        created: bool = False,
        commit: bool = False,
        status: str | None = None,
    ) -> bool:
        """提交剖面任务：暗带判定 + ROI 列剖面。

        keep=True 表示分隔线原样保留（纯刷新 / 撤销还原），False 表示按剖面
        峰谷重算。commit=True 时结果落地后入撤销栈并安排自动定量。
        """
        if (
            self._blot is None
            or self._roi is None
            or self._roi.width < 3
            or self._roi.height < 3
        ):
            self.profile_panel.clear_column_profile()
            self._column_profile = None
            return False
        rect = self._roi
        token = self._begin_job(JOB_PROFILE, status)
        self._submit_job(
            token,
            partial(compute_roi_profile, self._blot, rect, self._bg_mode()),
            partial(self._on_profile_done, token, rect, keep, created, commit),
            partial(self._on_profile_error, token),
        )
        return True

    def _on_profile_error(self, token: object, message: str) -> None:
        if not self._job_is_current(token):
            errorlog.log_message(f"忽略过期的剖面报错：{message}")
            return
        self._finish_job(token)
        errorlog.log_message(f"ROI 剖面计算失败：{message}")
        self.profile_panel.clear_column_profile()
        self._set_status(f"⚠ 剖面计算失败：{message}")

    def _on_profile_done(
        self,
        token: object,
        rect: RoiRect,
        keep: bool,
        created: bool,
        commit: bool,
        payload: object,
    ) -> None:
        if not self._job_is_current(token):
            self._discard_stale_job(token, "剖面计算")
            return
        self._finish_job(token)
        dark, profile = payload
        self._dark_cache = ((self._doc_version, self._bg_mode()), bool(dark))
        self._column_profile = np.asarray(profile, dtype=float)
        if self._roi is None:
            return
        if keep:
            canvas_seps = self.canvas.separators()
            if canvas_seps:
                self._separators = [float(s) for s in canvas_seps]
        else:
            expected = int(self.lane_spin.value()) or None
            seps = separators_from_profile(profile, rect, expected_lanes=expected)
            if not seps and expected and expected > 1:
                seps = default_separators(rect, expected)
            self._separators = [float(s) for s in seps]
        self.canvas.set_separators(self._separators)
        self._push_snap_targets()
        self.profile_panel.set_column_profile(
            profile, rect.x, rect.width, self._separators
        )
        if not commit:
            return
        self._commit_snapshot()
        self._update_action_states()
        lane_count = len(self._separators) + 1
        how = "保留原有分隔线" if keep else "已按剖面峰谷划分"
        self._set_status(
            f"ROI {rect.width}×{rect.height}，{how} {lane_count} 个泳道，正在自动定量……"
        )
        self._schedule_quantify()
        if created and self.canvas.tool() is Tool.ROI:
            self._select_tool(Tool.ADJUST)

    def _push_snap_targets(self) -> None:
        """把剖面谷底交给画布作为分隔线吸附目标。"""
        targets: list[float] = []
        if self._snap_enabled and self._column_profile is not None and self._roi is not None:
            targets = profile_valleys(self._column_profile, float(self._roi.x))
        self.canvas.set_snap_targets(targets)

    def _on_snap_toggled(self, checked: bool) -> None:
        self._snap_enabled = bool(checked)
        self.canvas.set_snap_enabled(self._snap_enabled)
        self._push_snap_targets()
        self._set_status(
            "分隔线吸附已开启：拖到剖面谷底附近会轻微吸附（按住 Alt 临时关闭）。"
            if checked
            else "分隔线吸附已关闭，分隔线完全跟手。"
        )

    def _sync_roi_fields(self, live: tuple | None = None) -> None:
        """ROI → 数值框。沿用旋转控件那套 _syncing 标志防回环。

        live 用于拖动 / 键盘微调过程中的即时回显：此时 self._roi 还没落地（要等
        防抖结束），但数值框应当跟手。
        """
        if self._syncing:
            return
        self._syncing = True
        try:
            roi = self._roi
            enabled = (roi is not None or live is not None) and self._blot is not None
            enabled = enabled and not self._busy
            max_w = int(self._blot.width) if self._blot is not None else 999999
            max_h = int(self._blot.height) if self._blot is not None else 999999
            limits = {
                "x": max(1, max_w - 1),
                "y": max(1, max_h - 1),
                "w": max(1, max_w),
                "h": max(1, max_h),
            }
            if live is not None:
                values = dict(zip(("x", "y", "w", "h"), (int(v) for v in live)))
            else:
                values = {
                    "x": roi.x if roi is not None else 0,
                    "y": roi.y if roi is not None else 0,
                    "w": roi.width if roi is not None else 1,
                    "h": roi.height if roi is not None else 1,
                }
            for key, spin in self.roi_spins.items():
                spin.setEnabled(bool(enabled))
                spin.setMaximum(limits[key])
                spin.setValue(int(values[key]))
        finally:
            self._syncing = False

    def _on_roi_field_changed(self, _value: int) -> None:
        """数值框 → ROI。走和鼠标拖动同一条落地路径（防抖 → 剖面 → 定量）。"""
        if self._syncing or self._blot is None:
            return
        rect = RoiRect(
            int(self.roi_spins["x"].value()),
            int(self.roi_spins["y"].value()),
            max(3, int(self.roi_spins["w"].value())),
            max(3, int(self.roi_spins["h"].value())),
        ).clipped(self._blot.raw)
        current = self._roi
        if current is not None and (
            (current.x, current.y, current.width, current.height)
            == (rect.x, rect.y, rect.width, rect.height)
        ):
            return
        self._roi = rect
        lo = rect.x + 1.0
        hi = rect.x + rect.width - 1.0
        if hi > lo:
            self._separators = sorted(
                float(np.clip(s, lo, hi)) for s in self._separators
            )
        self.canvas.set_roi(rect.x, rect.y, rect.width, rect.height)
        self.canvas.set_separators(self._separators)
        self._mark_result_stale()
        self._roi_timer.start()

    def _on_canvas_separators_changed(self, separators: list) -> None:
        self._separators = [float(s) for s in separators]
        self.profile_panel.set_separators(self._separators)
        self._mark_result_stale()

    def _on_profile_separators_changed(self, separators: list) -> None:
        self._separators = [float(s) for s in separators]
        self.canvas.set_separators(self._separators)
        self._mark_result_stale()

    def _on_separators_committed(self) -> None:
        """分隔线拖动 / 增删结束：入撤销栈并自动重新定量。"""
        self._commit_snapshot()
        self._schedule_quantify()

    def _mark_result_stale(self) -> None:
        if self._result is not None and not self._result_stale:
            self._result_stale = True
            self._update_action_states()

    def _on_lane_count_changed(self, _value: int) -> None:
        if self._roi is not None and self._blot is not None:
            self._reset_separators()

    def _reset_separators(self) -> None:
        if self._blot is None or self._roi is None:
            self._set_status("请先框选 ROI。")
            return
        self._apply_roi_change(recompute=True)

    def _clear_roi(self) -> None:
        if self._roi is None and self.canvas.roi() is None:
            return
        self._quant_timer.stop()
        self._roi_timer.stop()
        self._invalidate_jobs({JOB_PROFILE})
        self.canvas.clear_roi()
        self._roi = None
        self._separators = []
        self._column_profile = None
        self.canvas.set_snap_targets([])
        self._sync_roi_fields()
        self.profile_panel.clear_column_profile()
        self._clear_result()
        self._result_stale = False
        self._commit_snapshot()
        self._set_status("已清除 ROI（Ctrl+Z 可连同定量结果一起恢复）。")

    def _refresh_column_profile(self) -> None:
        """只重画剖面，不动分隔线、不入撤销栈。"""
        self._request_profile(keep=True)

    # ------------------------------------------------------------------ 撤销

    def _snapshot_state(self) -> _Snapshot:
        """当前可撤销状态的快照（字段说明见 :class:`_Snapshot`）。"""
        roi_t = (
            (self._roi.x, self._roi.y, self._roi.width, self._roi.height)
            if self._roi is not None
            else None
        )
        seps = tuple(round(float(s), 3) for s in self._separators)
        if self._result is not None and self._result.measurements:
            meas = tuple(self._result.measurements)
            dark = bool(self._result.dark_bands)
        else:
            meas = None
            dark = True
        return _Snapshot(
            roi=roi_t,
            separators=seps,
            measurements=meas,
            dark_bands=dark,
            origin=self._result_origin,
            angle=round(float(self._angle), 6),
            quarter_turns=int(self._quarter_turns),
            lanes=tuple(self._lanes),
            tool=self.canvas.tool(),
        )

    def _commit_snapshot(self) -> None:
        state = self._snapshot_state()
        if state.identity != self._committed.identity:
            self._undo.append(("state", self._committed))
            del self._undo[:-MAX_UNDO]
            self._redo.clear()  # 新操作开了新分支，原来的重做链作废
            self._committed = state
        self._update_action_states()

    def _absorb_result_state(self) -> None:
        """定量结果是 ROI+分隔线的派生数据：更新已提交状态但不新增撤销步骤。"""
        self._committed = self._snapshot_state()

    def _undo_last(self) -> None:
        if not self._undo:
            self._set_status("没有可撤销的操作。")
            return
        kind, payload = self._undo.pop()
        if kind == "history":
            self._redo.append(("history", list(self._session.entries)))
            self._session.entries[:] = list(payload)
            self._refresh_normalize()
            self._update_action_states()
            self._set_status("已恢复删除的历史记录（Ctrl+Shift+Z 可再删）。")
            return
        self._redo.append(("state", self._committed))
        self._apply_state(payload, "已撤销")

    def _redo_last(self) -> None:
        if not self._redo:
            self._set_status("没有可重做的操作。")
            return
        kind, payload = self._redo.pop()
        if kind == "history":
            self._undo.append(("history", list(self._session.entries)))
            self._session.entries[:] = list(payload)
            self._refresh_normalize()
            self._update_action_states()
            self._set_status("已重做历史记录改动。")
            return
        self._undo.append(("state", self._committed))
        self._apply_state(payload, "已重做")

    def _apply_state(self, state: _Snapshot, verb: str) -> None:
        """回到某一格状态。朝向不同就先转回去，转好了再落地 ROI 与结果。"""
        self._quant_timer.stop()
        self._roi_timer.stop()
        self._invalidate_jobs({JOB_QUANTIFY, JOB_DETECT, JOB_PROFILE})
        self._committed = state
        self._undo_verb = verb
        if state.orientation != (self._quarter_turns, round(float(self._angle), 6)):
            self._set_rotation_controls(float(state.angle))
            self._rotate_timer.stop()
            if self._request_orientation(
                state.quarter_turns,
                state.angle,
                restore=state,
                message=f"{verb}：正在转回 {state.angle:+.1f}°……",
            ):
                return
        self._apply_state_geometry(state)

    def _apply_state_geometry(self, state: _Snapshot) -> None:
        """把快照里的 ROI / 分隔线 / 结果 / 工具原样落地（不含朝向）。"""
        verb = self._undo_verb
        self._separators = [float(s) for s in state.separators]
        if state.roi is None:
            self._roi = None
            self._column_profile = None
            self.canvas.clear_roi()
            self.canvas.set_snap_targets([])
            self.profile_panel.clear_column_profile()
        else:
            self._roi = RoiRect(*(int(v) for v in state.roi))
            self.canvas.set_roi(*state.roi)
            self.canvas.set_separators(self._separators)
            self._refresh_column_profile()
        self._sync_roi_fields()
        restored = None
        if state.measurements and self._blot is not None:
            # 原样还原快照里的测量值：重测会换一套口径，撤销就回不到原值
            try:
                restored = result_from_measurements(
                    self._blot,
                    list(state.measurements),
                    dark_bands=state.dark_bands,
                    roi=self._roi,
                    separators=self._separators,
                )
            except Exception as exc:  # noqa: BLE001 - 恢复失败则只还原 ROI
                errorlog.log_exception("还原定量结果失败", exc)
                restored = None
        if restored is not None:
            self._set_result(restored, list(state.lanes), origin=state.origin)
        else:
            self._clear_result()
        if isinstance(state.tool, Tool):
            self._select_tool(state.tool)
        self._result_stale = False
        self._update_action_states()
        self._set_status(f"{verb}。")

    # ------------------------------------------------------------------ 定量 / 检测

    def _bg_mode(self) -> str:
        index = max(0, self.bg_combo.currentIndex())
        return BG_MODES[index][1]

    def _current_dark_bands(self) -> bool:
        """暗带判定按 (文档版本, 背景模式) 缓存。

        它是整图级运算，剖面任务算过一次就存下来，后面 UI 线程上的定量、补画
        条带都直接命中缓存，不再各自扫一遍整图。
        """
        if self._blot is None:
            return True
        key = (self._doc_version, self._bg_mode())
        if self._dark_cache is not None and self._dark_cache[0] == key:
            return bool(self._dark_cache[1])
        value = bool(resolve_dark_bands(self._blot, self._bg_mode()))
        self._dark_cache = (key, value)
        return value

    def _current_config(self) -> AnalysisConfig:
        level = float(np.clip(self.weak_slider.value(), 0.0, 100.0))
        sensitivity = float(np.clip(1.65 - 1.15 * (level / 100.0), 0.45, 1.8))
        return AnalysisConfig(
            sensitivity=sensitivity,
            kernel_size=None,
            background_mode=self._bg_mode(),  # type: ignore[arg-type]
        )

    def _schedule_quantify(self) -> None:
        """ROI / 分隔线变化后安排一次防抖自动定量。"""
        if self._blot is None or self._roi is None:
            return
        self._mark_result_stale()
        self._quant_timer.start()

    def _auto_quantify(self) -> None:
        if self._blot is None or self._roi is None:
            return
        if (
            self._busy
            or self._view_pending()
            or self.canvas.is_interacting()
            or self._roi_timer.isActive()
        ):
            self._quant_timer.start()  # 稍后重试
            return
        # 手动点过“ROI 定量”但当时被后台任务挡住了：补上时仍按手动处理
        manual = self._quant_manual_pending
        self._quant_manual_pending = False
        self._run_quantify(auto=not manual)

    def _quant_input_state(self) -> tuple:
        roi_t = (
            (self._roi.x, self._roi.y, self._roi.width, self._roi.height)
            if self._roi is not None
            else None
        )
        return (
            self._doc_version,
            roi_t,
            tuple(round(float(s), 3) for s in self._separators),
        )

    # ---------------------------------------------------------- 后台任务身份

    def _job_guard(self, kind: str) -> int:
        """任务令牌里存的"身份"取自哪个版本号。

        定量 / 检测 / 剖面吃的是 self._blot，用 doc_version；旋转吃的是原件，
        用 source_version；加载不依附任何现有图像，只靠"后来者顶掉前者"。
        """
        if kind == JOB_LOAD:
            return -1
        if kind == JOB_ORIENT:
            return self._source_version
        return self._doc_version

    def _bump_document(self) -> None:
        """self._blot 被换掉了：文档版本 +1，派生缓存作废。

        **只在图像真的替换掉的那一刻调用**（加载 / 旋转任务落地时）。若改成提交
        任务时就 +1，加载期间提交的定量任务会拿到"新"版本号，回来时校验反而
        通过，U1 修好的东西就退回去了。
        """
        self._doc_version += 1
        self._dark_cache = None

    def _invalidate_jobs(self, kinds=None) -> bool:
        """作废指定类别在跑的任务，返回是否中断了会产出结果的任务。

        令牌比对才是真正的护栏：就算取消晚了一步，回调也会因为身份对不上被
        丢弃。取消只是为了少算一点、少弹一次提示。
        """
        targets = set(RESULT_JOBS | VIEW_JOBS) if kinds is None else {str(k) for k in kinds}
        interrupted = False
        for kind in targets:
            worker = self._workers.pop(kind, None)
            if worker is not None:
                worker.cancel()
            dropped = self._job_tokens.pop(kind, None) is not None
            dropped = self._running.pop(kind, None) is not None or dropped
            if dropped and kind in RESULT_JOBS:
                interrupted = True
        if JOB_QUANTIFY in targets:
            self._quant_job_state = None
        if JOB_ORIENT in targets:
            self._orient_target = (self._quarter_turns, self._angle)
            self._pending_restore = None
        self._refresh_busy_ui()
        return interrupted

    def _begin_job(self, kind: str, message: str | None = None) -> tuple:
        """登记后台任务并返回令牌；同类任务后来者顶掉前者（任务超车）。"""
        previous = self._workers.pop(kind, None)
        if previous is not None:
            previous.cancel()  # 中间态直接丢，回来也会被令牌拦住
        self._job_serial += 1
        token = (kind, self._job_guard(kind), self._job_serial)
        self._job_tokens[kind] = token
        self._running[kind] = message or JOB_LABELS.get(kind, kind)
        self._refresh_busy_ui(message)
        return token

    def _submit_job(self, token: tuple, job, on_done, on_error) -> None:
        worker = run_async(self._pool, job, on_done, on_error)
        self._workers[token[0]] = worker

    def _finish_job(self, token: object) -> None:
        """任务收尾：只有令牌还是当前的才清掉忙碌状态（被超车的不清）。"""
        if not isinstance(token, tuple) or not token:
            return
        kind = token[0]
        if self._job_tokens.get(kind) != token:
            return
        self._job_tokens.pop(kind, None)
        self._workers.pop(kind, None)
        self._running.pop(kind, None)
        self._refresh_busy_ui()

    def _job_is_current(self, token: object) -> bool:
        if not isinstance(token, tuple) or len(token) != 3:
            return False
        kind, guard, _serial = token
        return self._job_tokens.get(kind) == token and guard == self._job_guard(kind)

    def _view_pending(self, exclude=None) -> bool:
        """有没有会替换图像 / 剖面的呈现类任务在跑。"""
        skip = set(exclude or ())
        return any(kind in self._running for kind in VIEW_JOBS if kind not in skip)

    def _discard_stale_job(self, token: object, what: str) -> None:
        """丢弃过期任务的结果。

        结果类任务被丢弃要让用户看见——静默丢弃比错用更难排查；呈现类被超车
        是正常工作方式（连续拖旋转滑杆），只记日志，不吓人。
        """
        kind = token[0] if isinstance(token, tuple) and token else ""
        errorlog.log_message(
            f"丢弃过期的{what}结果：token={token} 当前 doc={self._doc_version} "
            f"src={self._source_version}"
        )
        if kind in RESULT_JOBS:
            self._set_status(f"⚠ 图片已切换或计算已取消，上一次的{what}结果已丢弃。")

    def _cancel_running_jobs(self) -> None:
        if not self._busy:
            return
        self._quant_timer.stop()  # 别让防抖的自动定量立刻又跑起来
        self._invalidate_jobs(RESULT_JOBS)
        self._set_status("已取消本次计算。结果保持为“待重新定量”，可随时点“ROI 定量”。")

    def _reject_while_busy(self, what: str) -> bool:
        if self._busy:
            self._set_status(f"⚠ 正在计算，请先等待或点“取消”，再{what}。")
            return True
        return False

    def _run_quantify(self, synchronous: bool = False, auto: bool = False) -> None:
        if self._blot is None:
            return
        # 图像 / 剖面还在后台换着，此刻定量拿的是旧输入：等它落地再说
        if self._busy or self._view_pending():
            self._quant_manual_pending = self._quant_manual_pending or not auto
            self._quant_timer.start()
            return
        if self._roi_timer.isActive():
            self._apply_roi_change()
            if self._view_pending():  # 剖面改到 worker 里算了，等它回来再定量
                self._quant_manual_pending = self._quant_manual_pending or not auto
                self._quant_timer.start()
                return
        if self._roi is None:
            if not auto:
                self._select_tool(Tool.ROI)
                self._set_status("⚠ 还没有 ROI——已切换到“框选 ROI”，请先在图上拖拽画框。")
            return
        self._quant_was_manual = not auto
        self._quant_manual_pending = False
        self._quant_job_state = self._quant_input_state()
        dark = self._current_dark_bands()
        job = partial(quantify_roi, self._blot, self._roi, list(self._separators), dark)
        token = self._begin_job(JOB_QUANTIFY, "正在 ROI 定量……")
        if synchronous:
            try:
                payload = job()
            except Exception as exc:  # noqa: BLE001
                self._on_worker_error(token, str(exc))
                return
            self._on_quantify_done(token, payload)
        else:
            self._submit_job(
                token,
                job,
                partial(self._on_quantify_done, token),
                partial(self._on_worker_error, token),
            )

    def _on_quantify_done(self, token: object, payload: object) -> None:
        if not self._job_is_current(token):
            self._discard_stale_job(token, "ROI 定量")
            return
        self._finish_job(token)
        result, lanes = payload
        self._set_result(result, lanes, origin=ORIGIN_ROI)
        if self._quant_job_state == self._quant_input_state():
            self._result_stale = False
        else:
            # 计算期间用户又改了分隔线：结果仍然过期，再排一轮
            self._result_stale = True
            self._quant_timer.start()
        self._absorb_result_state()
        self._update_action_states()
        if self._quant_was_manual and self._tabs.currentWidget() is self._adjust_tab:
            self._tabs.setCurrentWidget(self.results_panel)
        lane_count = (
            len(lanes) if lanes else len({m.lane for m in result.measurements}) or 1
        )
        text = f"ROI 定量完成：{len(result.measurements)} 个条带 / {lane_count} 个泳道。"
        if result.saturated_band_count:
            text += f" ⚠ {result.saturated_band_count} 个条带饱和。"
        self._set_status(
            text + " 微调分隔线会自动刷新；点选条带 Delete 可删除（可撤销）。"
        )

    def _run_auto(self, synchronous: bool = False) -> None:
        if self._busy or self._blot is None:
            return
        if self._view_pending():
            self._set_status("⚠ 图片还在加载 / 旋转，请稍候再点“自动检测”。")
            return
        job = partial(auto_detect, self._blot, self._current_config())
        token = self._begin_job(JOB_DETECT, "正在自动检测……")
        if synchronous:
            try:
                payload = job()
            except Exception as exc:  # noqa: BLE001
                self._on_worker_error(token, str(exc))
                return
            self._on_auto_done(token, payload)
        else:
            self._submit_job(
                token,
                job,
                partial(self._on_auto_done, token),
                partial(self._on_worker_error, token),
            )

    def _on_auto_done(self, token: object, payload: object) -> None:
        if not self._job_is_current(token):
            self._discard_stale_job(token, "自动检测")
            return
        self._finish_job(token)
        result, lanes = payload
        self._set_result(result, lanes, origin=ORIGIN_DETECT)
        self._result_stale = False
        self._absorb_result_state()
        self._update_action_states()
        count = len(result.measurements)
        if count:
            if self._tabs.currentWidget() is self._adjust_tab:
                self._tabs.setCurrentWidget(self.results_panel)
            self._set_status(
                f"自动检测到 {count} 个条带。可直接在图上点选条带，Delete 删除；"
                "不准可框选 ROI 定量，或调整“弱条带检出”。"
            )
        else:
            self._set_status("未检出条带。试试调高“弱条带检出”，或改用 ROI 定量。")

    def _on_worker_error(self, token: object, message: str) -> None:
        if not self._job_is_current(token):
            errorlog.log_message(f"忽略过期任务的报错：{message}")
            return
        self._finish_job(token)
        errorlog.log_message(f"分析失败：{message}")
        QMessageBox.critical(self, "分析失败", message + errorlog.hint())
        self._set_status(f"分析失败：{message}")

    def _refresh_busy_ui(self, message: str | None = None) -> None:
        """按两类任务分别给反馈。

        结果类（定量 / 检测）：锁画布与参数区、显示取消按钮、忙碌光标——它们
        的输入一旦中途被改，产出的数据就对不上了。
        呈现类（加载 / 旋转 / 剖面）：只转进度条，滑杆与参数区照常可用，最新
        一次生效。旋转滑杆连续拖动全靠这条，否则拖到一半就被自己锁死。
        """
        computing = self._busy
        active = bool(self._running)
        self._busy_bar.setVisible(active)
        self._busy_bar.setToolTip("；".join(self._running.values()) if active else "")
        self._cancel_button.setVisible(computing)
        # 画布用 setEnabled 而不是 setInteractive：ROI 拖拽是 BlotCanvas 自己的
        # mousePressEvent，setInteractive(False) 拦不住它
        if self.canvas.isEnabled() is computing:
            self.canvas.setEnabled(not computing)
        if self._tabs.isEnabled() is computing:
            self._tabs.setEnabled(not computing)
        if computing and not self._cursor_pushed:
            QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
            self._cursor_pushed = True
        elif not computing and self._cursor_pushed:
            QApplication.restoreOverrideCursor()
            self._cursor_pushed = False
        self._update_action_states()
        if message:
            self._set_status(message)

    def _set_result(
        self, result: AnalysisResult, lanes: list, origin: str | None = None
    ) -> None:
        self._result = result
        if origin is not None:
            self._result_origin = origin
        self._result_doc_version = self._doc_version
        self._lanes = list(lanes or [])
        measurements = result.measurements if result is not None else []
        self._results_dirty = bool(measurements)
        self.results_panel.set_measurements(measurements)
        self.canvas.clear_result_boxes()
        if measurements:
            self.canvas.show_result_boxes(measurements)
        # 带上 measurements：删条带后 lanes 不重算，右图的积分填充区得按现存
        # 条带重画，否则被删掉那条的填充还留在图上
        self.profile_panel.set_lane_profiles(self._lanes, list(measurements))
        self._update_action_states()

    def _clear_result(self) -> None:
        self._result = None
        self._result_origin = ""
        self._result_doc_version = -1
        self._results_dirty = False
        self._lanes = []
        self.results_panel.set_measurements([])
        self.canvas.clear_result_boxes()
        self.profile_panel.set_lane_profiles([], [])
        self._update_action_states()

    # ------------------------------------------------------------------ 条带增删

    def _numbers_for_new_band(self, measurement, existing: list) -> tuple[int, int]:
        """给手绘条带定泳道号与条带号，绝不动已有条带的编号。"""
        center = measurement.x + measurement.width / 2.0
        if self._result_origin == ORIGIN_ROI and self._roi is not None:
            lane = lane_for_center(self._roi, self._separators, center)
        else:
            # 自动检测结果没有分隔线：归到横向最近的已有泳道，够远就新开一个
            same_lane = None
            best = None
            widths = [m.width for m in existing] or [measurement.width]
            tolerance = max(8.0, float(np.median(widths)) * 0.55)
            for m in existing:
                distance = abs((m.x + m.width / 2.0) - center)
                if distance <= tolerance and (best is None or distance < best):
                    best = distance
                    same_lane = int(m.lane)
            if same_lane is None:
                same_lane = max((int(m.lane) for m in existing), default=0) + 1
            lane = same_lane
        band = max((int(m.band) for m in existing if int(m.lane) == lane), default=0) + 1
        return lane, band

    def _on_band_drawn(self, x: float, y: float, w: float, h: float) -> None:
        if self._blot is None:
            return
        dark = (
            self._result.dark_bands
            if self._result is not None
            else self._current_dark_bands()
        )
        existing = list(self._result.measurements) if self._result else []
        note = ""
        try:
            added = None
            if self._roi is not None and self._result_origin in ("", ORIGIN_ROI):
                # 与 ROI 定量同一条路径（橡皮筋基线），数值才和其它条带可比
                added = measure_band_roi_style(self._blot, self._roi, x, y, w, h, dark)
            if added is None:
                added = measure_band(self._blot, x, y, w, h, dark)
                if self._result_origin == ORIGIN_ROI:
                    note = " ⚠ 该条带按局部背景口径测量，与 ROI 定量结果不可直接比较。"
                if not existing:
                    self._result_origin = ORIGIN_DETECT
            elif not existing:
                self._result_origin = ORIGIN_ROI
            lane, band = self._numbers_for_new_band(added, existing)
            added = replace(added, lane=lane, band=band)
            result = result_from_measurements(
                self._blot,
                [*existing, added],
                dark_bands=dark,
                roi=self._roi,
                separators=self._separators,
                base=self._result,
            )
        except Exception as exc:  # noqa: BLE001
            errorlog.log_exception("补画条带失败", exc)
            QMessageBox.warning(self, "补画条带失败", str(exc) + errorlog.hint())
            return
        self._set_result(result, self._lanes)
        self._commit_snapshot()
        self._set_status(
            f"已添加条带 L{lane}-B{band}，当前共 {len(result.measurements)} 个"
            f"（Ctrl+Z 可撤销）。{note}"
        )

    def _on_delete_rows(self, indices: list) -> None:
        """删除只做过滤：保留其它条带的原始数值与泳道号，绝不重测、绝不重排。"""
        if self._blot is None or self._result is None:
            return
        drop = {int(i) for i in indices}
        measurements = list(self._result.measurements)
        kept = [m for i, m in enumerate(measurements) if i not in drop]
        if len(kept) == len(measurements):
            return
        if not kept:
            self._clear_result()
            self._commit_snapshot()
            self._set_status("已删除全部条带（Ctrl+Z 可撤销）。")
            return
        try:
            result = result_from_measurements(
                self._blot,
                kept,
                dark_bands=self._result.dark_bands,
                roi=self._roi,
                separators=self._separators,
                base=self._result,
            )
        except Exception as exc:  # noqa: BLE001
            errorlog.log_exception("删除条带失败", exc)
            QMessageBox.warning(self, "删除失败", str(exc) + errorlog.hint())
            return
        self._set_result(result, self._lanes)
        self._commit_snapshot()
        remaining = sorted({int(m.lane) for m in kept})
        self._set_status(
            f"已删除 {len(measurements) - len(kept)} 个条带，剩余 {len(kept)} 个"
            f"（泳道 {'、'.join(str(v) for v in remaining)}，编号保持不变；Ctrl+Z 可撤销）。"
        )

    def _on_lane_selected(self, lane: int) -> None:
        self.profile_panel.show_lane(int(lane))

    def _delete_selected_bands(self) -> None:
        """编辑菜单“删除选中条带”：只作用于选中对象，绝不触碰 ROI。"""
        selected = self.canvas.selected_bands()
        if selected:
            self._on_delete_rows(selected)
        else:
            self._set_status(
                "⚠ 没有选中的条带：先在画布点选（Ctrl 可多选）或在结果表选行。"
            )

    def _on_canvas_band_selection(self, indices: list) -> None:
        """画布点选条带 → 表格滚动选中对应行 + 剖面面板切到该泳道。"""
        indices = [int(i) for i in indices]
        self.results_panel.select_rows_by_original(indices)
        if (
            indices
            and self._result is not None
            and indices[0] < len(self._result.measurements)
        ):
            self.profile_panel.show_lane(int(self._result.measurements[indices[0]].lane))

    def _on_table_selection(self, indices: list) -> None:
        """表格选行 → 画布对应条带框进入选中态。"""
        self.canvas.select_bands(indices)

    # ------------------------------------------------------------------ 归一化

    def _history_blocker(self) -> str | None:
        """当前结果能不能进历史；不能就返回给用户看的原因。"""
        if self._result is None or not self._result.measurements:
            return "⚠ 还没有定量结果：先框选 ROI（会自动定量）或点“自动检测”。"
        # 只挡会换图的任务：剖面重算不改图像，没必要拦
        replacing_image = self._view_pending(exclude={JOB_PROFILE})
        if replacing_image or self._result_doc_version != self._doc_version:
            # 加载 / 旋转是异步的：这组数字测自另一张（或另一个朝向的）图
            return (
                "⚠ 图片正在加载 / 旋转，屏幕上的结果测自上一版图像，未加入历史。"
                "请等图像稳定、重新定量后再加入。"
            )
        if self._result_stale:
            return (
                "⚠ ROI 或分隔线改过、定量还没重跑，这组数字与画面不符，未加入历史。"
                "请等自动定量完成（或点“ROI 定量”）后再加入。"
            )
        return None

    def _on_add_to_history(self) -> None:
        blocked = self._history_blocker()
        if blocked is not None:
            self._set_status(blocked)
            return
        # 强度必须按**当前**的 measurements 重新按泳道聚合，并带上泳道号：
        # 删掉一个误检条带后，该泳道的强度是剩余条带的合计；整条泳道被删空
        # 时泳道号会留下空档，没有泳道号的话归一化只能按位置配对，后面每个
        # 泳道都会错位到内参的邻居上。
        series = lane_series_from_measurements(self._result.measurements)
        source = ""
        for blot in (self._blot, self._blot_original):
            if blot is not None and blot.path:
                source = str(blot.path)
                break
        kind = self.normalize_panel.kind()
        name = self.normalize_panel.protein_name()
        group = self.normalize_panel.group_name()
        existed = any(
            e.kind == kind and e.name == name and e.group == group
            for e in self._session.entries
        )
        try:
            self._session.upsert(
                QuantEntry.from_series(
                    series,
                    kind=kind,  # type: ignore[arg-type]
                    name=name,
                    group=group,
                    source=source,
                    doc_version=self._result_doc_version,
                )
            )
        except ValueError as exc:
            QMessageBox.warning(self, "无法加入历史", str(exc))
            return
        self._refresh_normalize()
        kind_label = "内参" if kind == "loading_control" else "目标蛋白"
        verb = "已更新" if existed else "已加入"
        lanes_text = "、".join(str(n) for n in series.lanes)
        text = (
            f"{verb}历史：{kind_label} {name}（{len(series.lanes)} 个泳道："
            f"{lanes_text}）。"
        )
        if series.lanes != list(range(1, len(series.lanes) + 1)):
            text += " 泳道号有空档（删过条带），归一化会按泳道号配对。"
        weak = [
            str(lane)
            for lane, flag in zip(series.lanes, series.below_detection, strict=False)
            if flag
        ]
        if weak:
            text += (
                f" ⚠ 泳道 {'、'.join(weak)} 低于检出限（S/N < 3），"
                "这些泳道的比值不可信。"
            )
        if kind == "loading_control":
            # 典型流程是“先内参、再目标”：自动切类型，防止下一条忘切
            self.normalize_panel.set_kind("target")
            text += " 类型已自动切到“目标蛋白”，测好目标膜后直接加入即可。"
        self._set_status(text)

    def _normalize_diagnosis(self, rows: list) -> str | None:
        """归一化结果下方那行提示：没结果时说为什么，有结果时说哪儿没对上。"""
        warnings = list(self._session.last_warnings)
        if rows:
            return ("⚠ " + " ".join(warnings)) if warnings else None
        if not self._session.entries:
            return None
        counts: dict[str, list[int]] = {}
        for entry in self._session.entries:
            pair = counts.setdefault(entry.group, [0, 0])
            pair[0 if entry.kind == "loading_control" else 1] += 1
        parts = [
            f"分组“{group}”：{pair[0]} 条内参 / {pair[1]} 条目标蛋白"
            for group, pair in counts.items()
        ]
        text = (
            "尚无归一化结果——同一分组需要同时有内参与目标蛋白。当前 "
            + "；".join(parts)
            + "。若某条“类型”选错了，双击历史表对应单元格即可修改。"
        )
        if warnings:
            text += " " + " ".join(warnings)
        return text

    def _refresh_normalize(self) -> None:
        self.normalize_panel.set_history(self._session.entries)
        rows = normalize_rows(self._session, self.normalize_panel.control_lane())
        self.normalize_panel.set_rows(rows)
        self.normalize_panel.set_diagnosis(self._normalize_diagnosis(rows))

    def _on_delete_history(self, rows: list) -> None:
        rows = [int(r) for r in rows]
        if not rows:
            return
        self._undo.append(("history", list(self._session.entries)))
        del self._undo[:-MAX_UNDO]
        for row in sorted(rows, reverse=True):
            self._session.remove_at(row)
        self._refresh_normalize()
        self._update_action_states()
        self._set_status(f"已删除 {len(rows)} 条历史记录（Ctrl+Z 可恢复）。")

    def _on_history_edited(self, row: int, kind: str, name: str, group: str) -> None:
        """历史表内直接改类型 / 名称 / 分组。"""
        entries = self._session.entries
        if not (0 <= row < len(entries)):
            return
        old = entries[row]
        name = name.strip()
        group = group.strip() or "默认"
        if not name:
            self._refresh_normalize()
            self._set_status("⚠ 名称不能为空，已还原。")
            return
        if (old.kind, old.name, old.group) == (kind, name, group):
            return
        # replace 而不是重建：手写字段列表会漏掉 lanes / doc_version，
        # 改一次名字就把泳道身份丢了，归一化随即退回按位置配对
        entries[row] = replace(old, kind=kind, name=name, group=group)
        self._refresh_normalize()
        duplicate = any(
            i != row and (e.kind, e.name, e.group) == (kind, name, group)
            for i, e in enumerate(entries)
        )
        text = f"历史第 {row + 1} 行已更新。"
        if duplicate:
            text += " ⚠ 存在同类型同名同分组的记录，归一化会使用最后一条。"
        self._set_status(text)

    def _on_export_lc(self) -> None:
        if not self._session.loading_controls():
            self._set_status("⚠ 历史中还没有内参数据：先把内参定量“加入历史”。")
            return
        start = str(Path(self._last_export_dir or self._last_open_dir or ".") / "loading_control.csv")
        filename, _ = QFileDialog.getSaveFileName(self, "导出内参 CSV", start, "CSV (*.csv)")
        if not filename:
            return
        try:
            saved = self._session.export_loading_control_csv(filename)
        except (ValueError, OSError) as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        self._last_export_dir = str(Path(filename).parent)
        self._notify_exported(f"内参已导出：{saved}", Path(filename).parent)

    def _on_import_lc(self) -> None:
        start = self._last_export_dir or self._last_open_dir or ""
        filename, _ = QFileDialog.getOpenFileName(
            self, "导入内参 CSV", start, "CSV (*.csv);;所有文件 (*.*)"
        )
        if not filename:
            return
        try:
            count = self._session.import_loading_control_csv(filename)
        except (ValueError, OSError) as exc:
            QMessageBox.critical(self, "导入失败", str(exc))
            return
        self._refresh_normalize()
        self._set_status(f"已导入 {count} 组内参数据。")

    def _on_export_normalized(self) -> None:
        rows = normalize_rows(self._session, self.normalize_panel.control_lane())
        if not rows:
            self._set_status(
                "⚠ 没有可导出的归一化结果：同一分组需要同时有内参与目标蛋白。"
            )
            return
        start = str(Path(self._last_export_dir or self._last_open_dir or ".") / "normalized_results.csv")
        filename, _ = QFileDialog.getSaveFileName(self, "导出归一化 CSV", start, "CSV (*.csv)")
        if not filename:
            return
        try:
            saved = export_normalized_csv(rows, filename)
        except (ValueError, OSError) as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        self._last_export_dir = str(Path(filename).parent)
        self._notify_exported(f"归一化结果已导出：{saved}", Path(filename).parent)

    # ------------------------------------------------------------------ 导出

    def _notify_exported(self, text: str, folder: Path | str) -> None:
        """导出成功：状态栏提示 + 常驻“打开导出文件夹”按钮。"""
        self._set_status(text)
        self._export_dir = str(folder)
        self._open_folder_button.setToolTip(self._export_dir)
        self._open_folder_button.setVisible(True)

    def _open_export_dir(self) -> None:
        if not self._export_dir or not Path(self._export_dir).exists():
            self._set_status("⚠ 导出目录不存在或已被移动。")
            return
        try:
            os.startfile(self._export_dir)  # noqa: S606 - 本地资源管理器
        except OSError as exc:
            self._set_status(f"⚠ 无法打开文件夹：{exc}")

    # export_result 用固定文件名 {stem}_bands.csv / {stem}_annotated.png，而
    # “导出数据”只让用户选目录——第二次导出会静默盖掉第一次。export.py 不在本
    # 轮可改范围内，所以在界面层做存在性检查。名字对不上时（核心改了命名约定）
    # 退化为不提示，绝不因此报错。
    EXPORT_PATTERNS = ("{stem}_bands.csv", "{stem}_annotated.png")

    def _existing_exports(self, folder: str, stem: str) -> list[str]:
        directory = Path(folder)
        found: list[str] = []
        for pattern in self.EXPORT_PATTERNS:
            candidate = directory / pattern.format(stem=stem)
            try:
                if candidate.exists():
                    found.append(candidate.name)
            except OSError:  # 路径过长 / 权限异常都不该拦住导出
                continue
        return found

    def _confirm_overwrite(self, folder: str, stem: str) -> bool:
        existing = self._existing_exports(folder, stem)
        if not existing:
            return True
        answer = QMessageBox.question(
            self,
            "文件已存在",
            "目录里已经有这次导出会写入的文件：\n\n"
            + "\n".join(f"· {name}" for name in existing)
            + "\n\n继续导出会覆盖它们。要继续吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _export_data(self) -> None:
        if self._result is None or not self._result.measurements or self._blot is None:
            self._set_status("⚠ 还没有可导出的定量结果：先框选 ROI 或点“自动检测”。")
            return
        image_path = self._blot.path or (
            self._blot_original.path if self._blot_original else None
        )
        image_path = image_path or Path("blot.png")
        start = self._last_export_dir or str(Path(image_path).parent)
        selected = QFileDialog.getExistingDirectory(self, "选择结果保存目录", start)
        if not selected:
            return
        if not self._confirm_overwrite(selected, Path(image_path).stem):
            self._set_status("已取消导出（未覆盖已有文件）。")
            return
        try:
            csv_path, preview_path = export_result(self._result, image_path, selected)
        except Exception as exc:  # noqa: BLE001 - cv2 / 权限错误都要给出提示
            errorlog.log_exception("导出数据失败", exc)
            QMessageBox.critical(self, "导出失败", str(exc) + errorlog.hint())
            return
        self._results_dirty = False
        self._last_export_dir = selected
        self._notify_exported(
            f"已导出 CSV：{Path(csv_path).name}；标注预览：{Path(preview_path).name}",
            selected,
        )

    def _export_figure(self) -> None:
        if self._blot is None and self._result is None:
            return
        stem = "figure"
        if self._blot is not None and self._blot.path:
            stem = self._blot.path.stem
        start_dir = Path(self._last_export_dir or self._last_open_dir or ".")
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "导出展示图",
            str(start_dir / f"{stem}_figure.png"),
            "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tif)",
        )
        if not filename:
            return
        if self.annotated_check.isChecked() and self._result is not None:
            image = self._result.annotated_image
        elif self._blot is not None:
            image = self._blot.display_bgr
        else:
            image = self._result.annotated_image
        try:
            saved = export_figure(image, filename, roi=self._roi, pad=12)
        except Exception as exc:  # noqa: BLE001
            errorlog.log_exception("导出展示图失败", exc)
            QMessageBox.critical(self, "导出失败", str(exc) + errorlog.hint())
            return
        self._last_export_dir = str(Path(filename).parent)
        suffix = "（按当前 ROI 裁切）" if self._roi is not None else "（整图）"
        self._notify_exported(f"展示图已保存：{saved} {suffix}", Path(filename).parent)

    # ------------------------------------------------------------------ 工具 / 视图 / 状态

    def _on_tool_action(self, action: QAction) -> None:
        tool = action.data()
        if isinstance(tool, Tool):
            if tool is not Tool.RULER:
                self._ruler_popup.hide()
            self.canvas.set_tool(tool)
            self._set_status(_TOOL_TIPS[tool])

    def _select_tool(self, tool: Tool) -> None:
        for action in self._tool_group.actions():
            if action.data() is tool:
                if not action.isChecked():
                    action.setChecked(True)
                break
        if tool is not Tool.RULER:
            self._ruler_popup.hide()
        self.canvas.set_tool(tool)

    def _on_guides_toggled(self, checked: bool) -> None:
        self.canvas.set_guides_visible(checked)
        if self.guides_check.isChecked() != checked:
            self.guides_check.blockSignals(True)
            self.guides_check.setChecked(checked)
            self.guides_check.blockSignals(False)

    def _on_zoom_changed(self, scale: float) -> None:
        self._zoom_label.setText(f"{scale * 100:.0f}%")

    def _update_weak_hint(self, value: int) -> None:
        if value < 35:
            text = "当前：保守（减少杂点，可能漏弱带）"
        elif value > 70:
            text = "当前：灵敏（更容易抓弱带，也可能多检）"
        else:
            text = "当前：适中（漏检时可往右拖）"
        self.weak_hint.setText(text)

    def _update_action_states(self) -> None:
        has_image = self._blot is not None
        has_result = self._result is not None and bool(self._result.measurements)
        # 忙碌期间不许换图：否则后台结果回来时已经不知道该落到哪张图上
        self.act_open.setEnabled(not self._busy)
        self._recent_menu.setEnabled(not self._busy)
        self.act_auto.setEnabled(has_image and not self._busy)
        self.act_quant.setEnabled(has_image and not self._busy)
        # 分隔线改动后结果过期：禁止把旧结果导出（自动重定量结束后恢复）
        self.act_export_data.setEnabled(
            has_result and not self._busy and not self._result_stale
        )
        self.act_export_figure.setEnabled(has_image and not self._busy)
        self.act_fit.setEnabled(has_image)
        self.act_100.setEnabled(has_image)
        self.act_zoom_in.setEnabled(has_image)
        self.act_zoom_out.setEnabled(has_image)
        self.act_zoom_roi.setEnabled(self._roi is not None)
        self.act_clear_roi.setEnabled(has_image and not self._busy)
        self.act_delete.setEnabled(has_result and not self._busy)
        self.act_rot_left.setEnabled(has_image and not self._busy)
        self.act_rot_right.setEnabled(has_image and not self._busy)
        self.act_undo.setEnabled(bool(self._undo) and not self._busy)
        self.act_redo.setEnabled(bool(self._redo) and not self._busy)
        for spin in self.roi_spins.values():
            spin.setEnabled(self._roi is not None and not self._busy)

    def _set_status(self, text: str | None = None) -> None:
        if text is None:
            if self._blot is None:
                text = "请打开或拖入图片（支持 PNG / JPG / BMP / TIF）"
            elif self._roi is None:
                text = "拖拽画出 ROI 框；或点“自动检测”整图识别"
            elif self._result is None:
                text = "泳道已划分，会自动定量；也可点“ROI 定量”手动刷新"
            else:
                text = "可在“归一化”页加入历史，或导出数据 / 展示图"
        self._hint_label.setText(text)

    # ------------------------------------------------------------------ 最近文件

    def _add_recent(self, path: Path) -> None:
        entry = str(path)
        self._recent = [entry, *[p for p in self._recent if p != entry]][:MAX_RECENT]
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self) -> None:
        self._recent_menu.clear()
        if not self._recent:
            empty = self._recent_menu.addAction("（空）")
            empty.setEnabled(False)
            return
        for entry in self._recent:
            action = self._recent_menu.addAction(Path(entry).name)
            action.setToolTip(entry)
            action.triggered.connect(lambda _=False, e=entry: self._open_recent(e))
        self._recent_menu.addSeparator()
        clear = self._recent_menu.addAction("清空列表")
        clear.triggered.connect(self._clear_recent)

    def _open_recent(self, entry: str) -> None:
        if self._reject_while_busy("打开图片"):
            return
        path = Path(entry)
        if not path.exists():
            QMessageBox.warning(self, "文件不存在", f"找不到文件：\n{entry}")
            self._recent = [p for p in self._recent if p != entry]
            self._rebuild_recent_menu()
            return
        self.open_path(path)

    def _clear_recent(self) -> None:
        self._recent = []
        self._rebuild_recent_menu()

    # ------------------------------------------------------------------ 拖放

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self._busy:
            return
        for url in event.mimeData().urls():
            if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in IMAGE_SUFFIXES:
                event.acceptProposedAction()
                return

    def dropEvent(self, event: QDropEvent) -> None:
        if self._reject_while_busy("拖入新图片"):
            return
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.suffix.lower() in IMAGE_SUFFIXES and path.exists():
                if self.open_path(path):
                    event.acceptProposedAction()
                return
        self._set_status("⚠ 拖拽失败：请拖入 PNG / JPG / BMP / TIF / TIFF 图片文件。")

    # ------------------------------------------------------------------ 设置

    def _restore_settings(self) -> None:
        settings = QSettings("WBAnalyzer", "WBAnalyzer")
        geometry = settings.value("win/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        state = settings.value("win/state")
        if state is not None:
            self.restoreState(state)
        self._last_open_dir = str(settings.value("paths/last_open_dir", "") or "")
        self._last_export_dir = str(settings.value("paths/last_export_dir", "") or "")
        recent = settings.value("paths/recent", [])
        if isinstance(recent, str):
            recent = [recent]
        self._recent = [str(p) for p in (recent or []) if p][:MAX_RECENT]
        self._rebuild_recent_menu()
        try:
            self.weak_slider.setValue(int(float(settings.value("params/weak_band", 55))))
        except (TypeError, ValueError):
            pass
        try:
            self.lane_spin.setValue(int(float(settings.value("params/lane_count", 0))))
        except (TypeError, ValueError):
            pass
        bg_mode = str(settings.value("params/bg_mode", "auto") or "auto")
        for index, (_label, value) in enumerate(BG_MODES):
            if value == bg_mode:
                self.bg_combo.setCurrentIndex(index)
                break
        self.normalize_panel.set_protein_name(
            str(settings.value("params/protein", "Actin") or "Actin")
        )
        self.normalize_panel.set_group_name(
            str(settings.value("params/group", "默认") or "默认")
        )
        try:
            self.normalize_panel.set_control_lane(
                int(float(settings.value("params/control_lane", 1)))
            )
        except (TypeError, ValueError):
            pass
        annotated = str(settings.value("params/export_annotated", "false")).lower()
        self.annotated_check.setChecked(annotated in ("true", "1"))
        snap = str(settings.value("params/snap_separators", "true")).lower()
        self.snap_check.setChecked(snap in ("true", "1"))
        self._snap_enabled = self.snap_check.isChecked()
        self.canvas.set_snap_enabled(self._snap_enabled)
        guides = str(settings.value("params/show_guides", "true")).lower()
        show_guides = guides in ("true", "1")
        self.act_guides.setChecked(show_guides)
        self.canvas.set_guides_visible(show_guides)
        self.guides_check.blockSignals(True)
        self.guides_check.setChecked(show_guides)
        self.guides_check.blockSignals(False)

    def _confirm_discard_results(self) -> bool:
        """有没导出的定量结果时挽留一次。"""
        if not self._results_dirty or self._result is None:
            return True
        if not self._result.measurements:
            return True
        answer = QMessageBox.question(
            self,
            "还没有导出结果",
            f"当前有 {len(self._result.measurements)} 个条带的定量结果尚未导出。\n"
            "关闭窗口后这些数据会丢失，确定要退出吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _shutdown_jobs(self) -> None:
        """取消并等待后台任务：不等就退，worker emit 时会撞上已销毁的接收方。"""
        for timer in (self._rotate_timer, self._roi_timer, self._quant_timer):
            timer.stop()
        self._job_tokens.clear()
        self._workers.clear()
        self._running.clear()
        cancelled = cancel_workers()
        self._pool.clear()
        if not self._pool.waitForDone(CLOSE_WAIT_MS):
            errorlog.log_message(
                f"关闭时仍有后台任务未结束（已取消 {cancelled} 个），放弃等待"
            )
        if self._cursor_pushed:
            QApplication.restoreOverrideCursor()
            self._cursor_pushed = False

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._confirm_discard_results():
            event.ignore()
            return
        self._shutdown_jobs()
        settings = QSettings("WBAnalyzer", "WBAnalyzer")
        settings.setValue("win/geometry", self.saveGeometry())
        settings.setValue("win/state", self.saveState())
        settings.setValue("paths/last_open_dir", self._last_open_dir)
        settings.setValue("paths/last_export_dir", self._last_export_dir)
        settings.setValue("paths/recent", self._recent)
        settings.setValue("params/weak_band", int(self.weak_slider.value()))
        settings.setValue("params/lane_count", int(self.lane_spin.value()))
        settings.setValue("params/bg_mode", self._bg_mode())
        settings.setValue("params/protein", self.normalize_panel.protein_name())
        settings.setValue("params/group", self.normalize_panel.group_name())
        settings.setValue("params/control_lane", self.normalize_panel.control_lane())
        settings.setValue("params/export_annotated", self.annotated_check.isChecked())
        settings.setValue("params/show_guides", self.act_guides.isChecked())
        settings.setValue("params/snap_separators", self.snap_check.isChecked())
        super().closeEvent(event)

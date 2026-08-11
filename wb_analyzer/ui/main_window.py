"""主窗口：组装画布 / 面板 / 工具栏 / 菜单，并承载全部业务流程。

对 wb_analyzer.quant 的调用经过 ``_call_flex`` 容错适配：并行开发的核心模块
若函数名或关键字参数与契约略有出入（如 display / display_gray、
horizontal_profile / roi_column_profile），界面仍能工作。
"""

from __future__ import annotations

import inspect
import os
from functools import partial
from pathlib import Path

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
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStyle,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .. import quant
from ..analysis import (
    AnalysisConfig,
    AnalysisResult,
    _resolve_dark_bands,
    analyze_array,
    measure_roi,
    rebuild_result,
)
from ..export import export_figure, export_result
from ..imageio import load_blot, remap_rotation, rotate_blot
from ..normalize import (
    AnalysisSession,
    QuantEntry,
    export_normalized_csv,
    lane_intensities_from_measurements,
)
from ..roi import RoiRect, default_separators, normalize_angle, separators_from_profile
from .canvas import BlotCanvas, Tool
from .normalize_panel import NormalizePanel
from .profile_panel import ProfilePanel
from .results_panel import ResultsPanel
from .workers import run_async

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

BG_MODES = (
    ("自动（推荐）", "auto"),
    ("深色条带 / 浅底", "dark_bands"),
    ("浅色条带 / 深底", "light_bands"),
)

MAX_RECENT = 5
MAX_UNDO = 50

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
    "删除操作都可用 Ctrl+Z 撤销；清除 ROI 也可撤销（会连同定量结果一起恢复）。"
)


class _WheelGuard(QObject):
    """滑杆 / 下拉 / 数字框未聚焦时忽略滚轮，避免滚动面板时误改参数。"""

    def eventFilter(self, obj, event):  # noqa: N802 - Qt API
        if event.type() == QEvent.Type.Wheel and not obj.hasFocus():
            event.ignore()
            return True
        return super().eventFilter(obj, event)


# ---------------------------------------------------------------------- 核心模块适配


def _call_flex(fn, *args, **kwargs):
    """调用 fn 并自动丢弃其签名不支持的关键字参数（并行开发容错）。"""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)
    accepted = {key: value for key, value in kwargs.items() if key in params}
    return fn(*args, **accepted)


def resolve_dark_bands(blot, mode: str) -> bool:
    fn = getattr(quant, "resolve_dark_bands_native", None)
    if callable(fn):
        try:
            return bool(_call_flex(fn, blot.raw, blot.white_level, mode=mode))
        except Exception:  # noqa: BLE001 - 回退 8 位判定
            pass
    return bool(_resolve_dark_bands(blot.display_gray, mode))


def column_profile(blot, roi: RoiRect, dark_bands: bool) -> np.ndarray:
    fn = getattr(quant, "roi_column_profile", None)
    if not callable(fn):
        fn = quant.horizontal_profile
    return np.asarray(
        _call_flex(
            fn, blot.raw, roi, dark_bands=dark_bands, white_level=blot.white_level
        ),
        dtype=float,
    )


def quantify_roi(blot, roi: RoiRect, separators: list[float], dark_bands: bool):
    """返回 (AnalysisResult, lanes)；兼容 QuantOutput 与直接返回 AnalysisResult。"""
    out = _call_flex(
        quant.quantify_lanes,
        blot.raw,
        roi,
        list(separators),
        dark_bands=dark_bands,
        white_level=blot.white_level,
        display=blot.display_bgr,
        display_gray=blot.display_gray,
    )
    if hasattr(out, "result"):
        return out.result, list(getattr(out, "lanes", []) or [])
    lanes_fn = getattr(quant, "compute_lane_profiles", None)
    lanes: list = []
    if callable(lanes_fn):
        try:
            lanes = list(
                _call_flex(
                    lanes_fn,
                    blot.raw,
                    roi,
                    list(separators),
                    dark_bands=dark_bands,
                    white_level=blot.white_level,
                )
            )
        except Exception:  # noqa: BLE001
            lanes = []
    return out, lanes


def measure_band(blot, x: float, y: float, w: float, h: float, dark_bands: bool):
    fn = getattr(quant, "measure_box_native", None)
    if callable(fn):
        return _call_flex(
            fn,
            blot.raw,
            int(round(x)),
            int(round(y)),
            int(round(w)),
            int(round(h)),
            dark_bands=dark_bands,
            white_level=blot.white_level,
        )
    return measure_roi(
        blot.display_gray,
        int(round(x)),
        int(round(y)),
        int(round(w)),
        int(round(h)),
        dark_bands=dark_bands,
    )


def remeasure(blot, measurements: list, dark_bands: bool) -> AnalysisResult:
    fn = getattr(quant, "remeasure_native", None)
    if callable(fn):
        return _call_flex(
            fn,
            list(measurements),
            blot.raw,
            dark_bands=dark_bands,
            white_level=blot.white_level,
            display=blot.display_bgr,
            display_gray=blot.display_gray,
        )
    return rebuild_result(blot.display_gray, list(measurements), dark_bands=dark_bands)


def auto_detect(blot, config: AnalysisConfig):
    result = analyze_array(blot.display_bgr, config=config)
    if callable(getattr(quant, "remeasure_native", None)) and result.measurements:
        try:
            result = remeasure(blot, result.measurements, result.dark_bands)
        except Exception:  # noqa: BLE001 - 保留 8 位结果
            pass
    return result, []


def normalize_rows(session: AnalysisSession, control_lane: int):
    try:
        return session.normalize_all(control_lane=int(control_lane))
    except TypeError:
        return session.normalize_all()


# ---------------------------------------------------------------------- 主窗口


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("WB 灰度分析工具")
        self.resize(1440, 900)
        self.setMinimumSize(1100, 680)
        self.setAcceptDrops(True)

        self._blot_original = None
        self._blot = None
        self._angle = 0.0
        self._roi: RoiRect | None = None
        self._separators: list[float] = []
        self._result: AnalysisResult | None = None
        self._lanes: list = []
        self._session = AnalysisSession()
        self._ruler_angle: float | None = None
        self._busy = False
        # 撤销栈：("state", 快照) / ("history", 历史记录列表副本)
        self._undo: list[tuple] = []
        self._committed: tuple = self._EMPTY_STATE
        self._result_stale = False
        self._quant_was_manual = False
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

    _EMPTY_STATE: tuple = (None, (), None, True, ())

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
        self.rotation_slider = QSlider(Qt.Orientation.Horizontal)
        self.rotation_slider.setRange(-300, 300)
        self.rotation_slider.setValue(0)
        self.rotation_spin = QDoubleSpinBox()
        self.rotation_spin.setRange(-30.0, 30.0)
        self.rotation_spin.setDecimals(1)
        self.rotation_spin.setSingleStep(0.1)
        self.rotation_spin.setSuffix("°")
        zero_button = QPushButton("归零")
        zero_button.clicked.connect(lambda: self.rotation_spin.setValue(0.0))
        rotate_row = QHBoxLayout()
        rotate_row.addWidget(self.rotation_slider, 1)
        rotate_row.addWidget(self.rotation_spin)
        rotate_row.addWidget(zero_button)
        rotate_hint = QLabel("正值抬高右侧（纠正右低左高）。也可用标尺拖线后一键旋正。")
        rotate_hint.setWordWrap(True)
        rotate_hint.setStyleSheet("color:#9aa0a6;font-weight:400;")
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
            label.setStyleSheet("color:#9aa0a6;font-weight:400;")
        weak_row.addWidget(conservative)
        weak_row.addWidget(self.weak_slider, 1)
        weak_row.addWidget(sensitive)
        self.weak_hint = QLabel("当前：适中（漏检时可往右拖）")
        self.weak_hint.setStyleSheet("color:#9aa0a6;font-weight:400;")
        self.bg_combo = QComboBox()
        for label, _value in BG_MODES:
            self.bg_combo.addItem(label)
        bg_hint = QLabel("普通白底深带选“深色条带”；黑底发光选“浅色条带”。一般保持自动即可。")
        bg_hint.setWordWrap(True)
        bg_hint.setStyleSheet("color:#9aa0a6;font-weight:400;")
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
        peaks_button = QPushButton("按剖面峰谷划分")
        peaks_button.clicked.connect(self._reset_separators)
        clear_roi_button = QPushButton("清除 ROI")
        clear_roi_button.clicked.connect(self.act_clear_roi.trigger)
        roi_form = QFormLayout()
        roi_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        roi_form.addRow("期望泳道数", self.lane_spin)
        roi_layout = QVBoxLayout(roi_box)
        roi_layout.addLayout(roi_form)
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
        self._open_folder_button = QPushButton("打开导出文件夹")
        self._open_folder_button.setVisible(False)
        self._open_folder_button.clicked.connect(self._open_export_dir)
        self.statusBar().addPermanentWidget(self._open_folder_button)
        self._zoom_label = QLabel("100%")
        self.statusBar().addPermanentWidget(self._zoom_label)

    def _build_ruler_popup(self) -> None:
        """标尺测角后就地弹出的旋正操作浮层（画布内，非模态）。"""
        popup = QWidget(self.canvas.viewport())
        popup.setObjectName("rulerPopup")
        popup.setStyleSheet(
            "#rulerPopup {background: #2c2f34; border: 1px solid #4a4e55;"
            " border-radius: 6px;}"
        )
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
        self.act_delete.triggered.connect(self._delete_selected_bands)
        self.act_clear_roi.triggered.connect(self._clear_roi)
        self.act_rot_left.triggered.connect(partial(self._rotate_90, False))
        self.act_rot_right.triggered.connect(partial(self._rotate_90, True))
        self.act_guides.toggled.connect(self._on_guides_toggled)
        self.act_exit.triggered.connect(self.close)
        self.act_help.triggered.connect(self._show_help)

    def _show_help(self) -> None:
        """非模态使用说明：可以边看边操作。"""
        if self._help_box is None:
            box = QMessageBox(self)
            box.setWindowTitle("使用说明")
            box.setText(_HELP_TEXT)
            box.setWindowModality(Qt.WindowModality.NonModal)
            box.setStandardButtons(QMessageBox.StandardButton.Close)
            self._help_box = box
        self._help_box.show()
        self._help_box.raise_()

    # ------------------------------------------------------------------ 打开图片

    def _choose_image(self) -> None:
        start = self._last_open_dir or str(Path.home())
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择 Western Blot 图片",
            start,
            "图片 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;所有文件 (*.*)",
        )
        if filename:
            self.open_path(Path(filename))

    def open_path(self, path: Path | str) -> bool:
        path = Path(path)
        try:
            blot = load_blot(path)
        except (ValueError, OSError) as exc:
            QMessageBox.critical(self, "无法打开图片", str(exc))
            return False
        # 同尺寸、未做微调旋转时沿用 ROI / 分隔线（同膜 strip 重探是标准流程）
        reuse = (
            self._roi is not None
            and self._blot is not None
            and abs(self._angle) < 1e-6
            and blot.width == self._blot.width
            and blot.height == self._blot.height
        )
        prev_roi = self._roi
        prev_seps = list(self._separators)
        self._blot_original = blot
        self._blot = blot
        self._angle = 0.0
        self._rotate_timer.stop()
        self._quant_timer.stop()
        self._result_stale = False
        self._set_rotation_controls(0.0)
        self._ruler_angle = None
        self.ruler_label.setText("标尺：未测量")
        self._ruler_popup.hide()
        self._roi = None
        self._separators = []
        self._undo.clear()
        self._committed = self._EMPTY_STATE
        self.canvas.clear_roi()
        self.canvas.clear_result_boxes()
        self.canvas.set_image(blot.display_bgr)
        self.canvas.fit_view()
        self.profile_panel.clear()
        self._clear_result()
        self._last_open_dir = str(path.parent)
        self._add_recent(path)
        self.setWindowTitle(f"WB 灰度分析工具 — {path.name}")
        loaded = f"已加载 {path.name}（{blot.width}×{blot.height}，{blot.bit_depth} 位）。"
        if reuse and prev_roi is not None:
            rect = prev_roi.clipped(blot.raw)
            self._roi = rect
            self._separators = prev_seps
            self.canvas.set_roi(rect.x, rect.y, rect.width, rect.height)
            self.canvas.set_separators(self._separators)
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
            self._set_status(loaded + "拖拽画出 ROI 框，或点“自动检测”。")
        self._update_action_states()
        return True

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
        if self._blot_original is not None:
            self._rotate_timer.start()

    def _apply_rotation(self) -> None:
        if self._blot_original is None:
            return
        angle = float(self.rotation_spin.value())
        if abs(angle - self._angle) < 1e-6:
            return
        old_angle = self._angle
        blot = rotate_blot(self._blot_original, angle)
        shape = self._blot_original.raw.shape[:2]

        new_roi: RoiRect | None = None
        new_seps: list[float] = []
        if self._roi is not None:
            r = self._roi
            corners = np.array(
                [
                    [r.x, r.y],
                    [r.x + r.width, r.y],
                    [r.x + r.width, r.y + r.height],
                    [r.x, r.y + r.height],
                ],
                dtype=float,
            )
            mapped = remap_rotation(corners, shape, old_angle, angle)
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
            if self._separators:
                cy = r.y + r.height / 2.0
                anchors = np.array([[s, cy] for s in self._separators], dtype=float)
                mapped_x = remap_rotation(anchors, shape, old_angle, angle)[:, 0]
                lo = new_roi.x + 2.0
                hi = new_roi.x + new_roi.width - 2.0
                if hi > lo:
                    new_seps = sorted(float(np.clip(v, lo, hi)) for v in mapped_x)

        self._blot = blot
        self._angle = angle
        self.canvas.set_image(blot.display_bgr)
        self._clear_result()
        if new_roi is not None:
            self._roi = new_roi
            self._separators = new_seps
            self.canvas.set_roi(new_roi.x, new_roi.y, new_roi.width, new_roi.height)
            self.canvas.set_separators(new_seps)
            self._refresh_column_profile()
            self._commit_snapshot()
            self._schedule_quantify()
            self._set_status(
                f"已旋转 {angle:.1f}°，ROI 与分隔线已随图重映射，正在自动重新定量……"
            )
        else:
            self._roi = None
            self._separators = []
            self.canvas.clear_roi()
            self.profile_panel.clear_column_profile()
            self._set_status(f"已旋转 {angle:.1f}°。")
        self._update_action_states()

    def _rotate_90(self, clockwise: bool) -> None:
        """整图 90° 旋转（横竖颠倒的扫描图）；会清除 ROI 与结果。"""
        if self._blot_original is None:
            return
        self._blot_original = rotate_blot(
            self._blot_original, -90.0 if clockwise else 90.0
        )
        self._blot = self._blot_original
        self._angle = 0.0
        self._rotate_timer.stop()
        self._quant_timer.stop()
        self._result_stale = False
        self._set_rotation_controls(0.0)
        self._ruler_angle = None
        self.ruler_label.setText("标尺：未测量")
        self._ruler_popup.hide()
        self._roi = None
        self._separators = []
        self._undo.clear()
        self._committed = self._EMPTY_STATE
        self.canvas.clear_roi()
        self.canvas.clear_result_boxes()
        self.canvas.set_image(self._blot.display_bgr)
        self.canvas.fit_view()
        self.profile_panel.clear()
        self._clear_result()
        self._update_action_states()
        direction = "右" if clockwise else "左"
        self._set_status(f"已向{direction}旋转 90°。请重新框选 ROI。")

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
        if self._blot_original is None:
            return
        if self._ruler_angle is None:
            self._select_tool(Tool.RULER)
            self._set_status("⚠ 请先用“标尺对齐”工具在图上沿条带拖一条线。")
            return
        current = float(self.rotation_spin.value())
        delta = self._ruler_angle if orientation == "horizontal" else self._ruler_angle - 90.0
        new_angle = float(np.clip(normalize_angle(current + delta), -30.0, 30.0))
        self._ruler_popup.hide()
        self.canvas.clear_ruler()
        self._ruler_angle = None
        self.ruler_label.setText("标尺：已应用旋正")
        self.rotation_spin.setValue(new_angle)
        # 旋正后回到下一步该用的工具：没 ROI 就框选，有 ROI 就调整
        self._select_tool(Tool.ROI if self._roi is None else Tool.ADJUST)
        label = "水平" if orientation == "horizontal" else "垂直"
        self._set_status(f"已按标尺旋正为{label}（{new_angle:.2f}°）。")

    # ------------------------------------------------------------------ ROI / 分隔线

    def _on_canvas_roi_changed(self, _x: float, _y: float, _w: float, _h: float) -> None:
        if self._blot is None:
            return
        self._quant_timer.stop()
        if self._result is not None:
            self._clear_result()
        self._roi_timer.start()

    def _apply_roi_change(self, recompute: bool | None = None) -> None:
        """ROI 稳定后刷新分隔线与剖面。

        recompute=True 强制按剖面峰谷重划；默认（None）在移动 / 缩放已有 ROI 时
        保留画布上现有分隔线（含手动调整），只有新画 ROI 才重新计算。
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
        self._roi = rect
        if rect.width < 3 or rect.height < 3:
            self._set_status("ROI 太小，请拖大一点再松开。")
            return
        dark = self._current_dark_bands()
        try:
            profile = column_profile(self._blot, rect, dark)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"剖面计算失败：{exc}")
            return
        canvas_seps = self.canvas.separators()
        keep_manual = recompute is not True and not created and bool(canvas_seps)
        if keep_manual:
            self._separators = [float(s) for s in canvas_seps]
        else:
            expected = int(self.lane_spin.value()) or None
            seps = separators_from_profile(profile, rect, expected_lanes=expected)
            if not seps and expected and expected > 1:
                seps = default_separators(rect, expected)
            self._separators = [float(s) for s in seps]
        self.canvas.set_roi(rect.x, rect.y, rect.width, rect.height)
        self.canvas.set_separators(self._separators)
        self.profile_panel.set_column_profile(
            profile, rect.x, rect.width, self._separators
        )
        self._commit_snapshot()
        self._update_action_states()
        lane_count = len(self._separators) + 1
        how = "保留原有分隔线" if keep_manual else "已按剖面峰谷划分"
        self._set_status(
            f"ROI {rect.width}×{rect.height}，{how} {lane_count} 个泳道，正在自动定量……"
        )
        self._schedule_quantify()
        if created and self.canvas.tool() is Tool.ROI:
            self._select_tool(Tool.ADJUST)

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
        self.canvas.clear_roi()
        self._roi = None
        self._separators = []
        self.profile_panel.clear_column_profile()
        self._clear_result()
        self._result_stale = False
        self._commit_snapshot()
        self._set_status("已清除 ROI（Ctrl+Z 可连同定量结果一起恢复）。")

    def _refresh_column_profile(self) -> None:
        if (
            self._blot is None
            or self._roi is None
            or self._roi.width < 3
            or self._roi.height < 3
        ):
            self.profile_panel.clear_column_profile()
            return
        try:
            profile = column_profile(self._blot, self._roi, self._current_dark_bands())
        except Exception:  # noqa: BLE001
            self.profile_panel.clear_column_profile()
            return
        self.profile_panel.set_column_profile(
            profile, self._roi.x, self._roi.width, self._separators
        )

    # ------------------------------------------------------------------ 撤销

    def _snapshot_state(self) -> tuple:
        """快照 = (roi, separators, measurements, dark_bands, lanes)。

        measurements 让删条带 / 清 ROI 可以连同结果一起撤销；
        lanes 仅用于恢复剖面图，不参与相等性比较。
        """
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
        return (roi_t, seps, meas, dark, tuple(self._lanes))

    def _commit_snapshot(self) -> None:
        state = self._snapshot_state()
        # lanes（含 numpy 数组）不参与比较，避免歧义
        if state[:4] != self._committed[:4]:
            self._undo.append(("state", self._committed))
            del self._undo[:-MAX_UNDO]
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
            self._session.entries[:] = list(payload)
            self._refresh_normalize()
            self._update_action_states()
            self._set_status("已恢复删除的历史记录。")
            return
        self._quant_timer.stop()
        self._roi_timer.stop()
        self._committed = payload
        roi_t, seps, meas, dark, lanes = payload
        self._separators = [float(s) for s in seps]
        if roi_t is None:
            self._roi = None
            self.canvas.clear_roi()
            self.profile_panel.clear_column_profile()
        else:
            self._roi = RoiRect(*(int(v) for v in roi_t))
            self.canvas.set_roi(*roi_t)
            self.canvas.set_separators(self._separators)
            self._refresh_column_profile()
        restored = None
        if meas and self._blot is not None:
            try:
                restored = remeasure(self._blot, list(meas), dark)
            except Exception:  # noqa: BLE001 - 恢复失败则只还原 ROI
                restored = None
        if restored is not None:
            self._set_result(restored, list(lanes))
        else:
            self._clear_result()
        self._result_stale = False
        self._update_action_states()
        self._set_status("已撤销。")

    # ------------------------------------------------------------------ 定量 / 检测

    def _bg_mode(self) -> str:
        index = max(0, self.bg_combo.currentIndex())
        return BG_MODES[index][1]

    def _current_dark_bands(self) -> bool:
        if self._blot is None:
            return True
        return resolve_dark_bands(self._blot, self._bg_mode())

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
        if self._busy or self.canvas.is_interacting() or self._roi_timer.isActive():
            self._quant_timer.start()  # 稍后重试
            return
        self._run_quantify(auto=True)

    def _quant_input_state(self) -> tuple:
        roi_t = (
            (self._roi.x, self._roi.y, self._roi.width, self._roi.height)
            if self._roi is not None
            else None
        )
        return (roi_t, tuple(round(float(s), 3) for s in self._separators))

    def _run_quantify(self, synchronous: bool = False, auto: bool = False) -> None:
        if self._busy or self._blot is None:
            if auto and self._busy:
                self._quant_timer.start()
            return
        if self._roi_timer.isActive():
            self._apply_roi_change()
        if self._roi is None:
            if not auto:
                self._select_tool(Tool.ROI)
                self._set_status("⚠ 还没有 ROI——已切换到“框选 ROI”，请先在图上拖拽画框。")
            return
        self._quant_was_manual = not auto
        self._quant_job_state = self._quant_input_state()
        dark = self._current_dark_bands()
        job = partial(quantify_roi, self._blot, self._roi, list(self._separators), dark)
        self._set_busy(True, "正在 ROI 定量……")
        if synchronous:
            try:
                payload = job()
            except Exception as exc:  # noqa: BLE001
                self._on_worker_error(str(exc))
                return
            self._on_quantify_done(payload)
        else:
            run_async(self._pool, job, self._on_quantify_done, self._on_worker_error)

    def _on_quantify_done(self, payload: object) -> None:
        self._set_busy(False)
        result, lanes = payload
        self._set_result(result, lanes)
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
        job = partial(auto_detect, self._blot, self._current_config())
        self._set_busy(True, "正在自动检测……")
        if synchronous:
            try:
                payload = job()
            except Exception as exc:  # noqa: BLE001
                self._on_worker_error(str(exc))
                return
            self._on_auto_done(payload)
        else:
            run_async(self._pool, job, self._on_auto_done, self._on_worker_error)

    def _on_auto_done(self, payload: object) -> None:
        self._set_busy(False)
        result, lanes = payload
        self._set_result(result, lanes)
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

    def _on_worker_error(self, message: str) -> None:
        self._set_busy(False)
        QMessageBox.critical(self, "分析失败", message)
        self._set_status(f"分析失败：{message}")

    def _set_busy(self, busy: bool, message: str | None = None) -> None:
        self._busy = busy
        self._update_action_states()
        if message:
            self._set_status(message)

    def _set_result(self, result: AnalysisResult, lanes: list) -> None:
        self._result = result
        self._lanes = list(lanes or [])
        measurements = result.measurements if result is not None else []
        self.results_panel.set_measurements(measurements)
        self.canvas.clear_result_boxes()
        if measurements:
            self.canvas.show_result_boxes(measurements)
        self.profile_panel.set_lane_profiles(self._lanes)
        self._update_action_states()

    def _clear_result(self) -> None:
        self._result = None
        self._lanes = []
        self.results_panel.set_measurements([])
        self.canvas.clear_result_boxes()
        self.profile_panel.set_lane_profiles([])
        self._update_action_states()

    # ------------------------------------------------------------------ 条带增删

    def _on_band_drawn(self, x: float, y: float, w: float, h: float) -> None:
        if self._blot is None:
            return
        dark = (
            self._result.dark_bands
            if self._result is not None
            else self._current_dark_bands()
        )
        try:
            added = measure_band(self._blot, x, y, w, h, dark)
            existing = list(self._result.measurements) if self._result else []
            result = remeasure(self._blot, [*existing, added], dark)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "补画条带失败", str(exc))
            return
        self._set_result(result, self._lanes)
        self._commit_snapshot()
        self._set_status(f"已添加条带，当前共 {len(result.measurements)} 个（Ctrl+Z 可撤销）。")

    def _on_delete_rows(self, indices: list) -> None:
        if self._blot is None or self._result is None:
            return
        drop = {int(i) for i in indices}
        kept = [m for i, m in enumerate(self._result.measurements) if i not in drop]
        if not kept:
            self._clear_result()
            self._commit_snapshot()
            self._set_status("已删除全部条带（Ctrl+Z 可撤销）。")
            return
        try:
            result = remeasure(self._blot, kept, self._result.dark_bands)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "删除失败", str(exc))
            return
        self._set_result(result, self._lanes)
        self._commit_snapshot()
        self._set_status(
            f"已删除 {len(drop)} 个条带，剩余 {len(result.measurements)} 个（Ctrl+Z 可撤销）。"
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

    def _on_add_to_history(self) -> None:
        if self._result is None or not self._result.measurements:
            self._set_status("⚠ 还没有定量结果：先框选 ROI（会自动定量）或点“自动检测”。")
            return
        measurements = self._result.measurements
        if all(m.band == 1 for m in measurements):
            ordered = sorted(measurements, key=lambda m: m.lane)
            intensities = [m.integrated_density for m in ordered]
            backgrounds = [m.background for m in ordered]
            means = [m.mean_intensity for m in ordered]
        else:
            intensities, backgrounds, means = lane_intensities_from_measurements(
                measurements
            )
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
                QuantEntry(
                    kind=kind,  # type: ignore[arg-type]
                    name=name,
                    group=group,
                    intensities=intensities,
                    backgrounds=backgrounds,
                    mean_intensities=means,
                    source=source,
                )
            )
        except ValueError as exc:
            QMessageBox.warning(self, "无法加入历史", str(exc))
            return
        self._refresh_normalize()
        kind_label = "内参" if kind == "loading_control" else "目标蛋白"
        verb = "已更新" if existed else "已加入"
        text = f"{verb}历史：{kind_label} {name}（{len(intensities)} 个泳道）。"
        if kind == "loading_control":
            # 典型流程是“先内参、再目标”：自动切类型，防止下一条忘切
            self.normalize_panel.set_kind("target")
            text += " 类型已自动切到“目标蛋白”，测好目标膜后直接加入即可。"
        self._set_status(text)

    def _refresh_normalize(self) -> None:
        self.normalize_panel.set_history(self._session.entries)
        rows = normalize_rows(self._session, self.normalize_panel.control_lane())
        self.normalize_panel.set_rows(rows)
        diagnosis = None
        if not rows and self._session.entries:
            counts: dict[str, list[int]] = {}
            for entry in self._session.entries:
                pair = counts.setdefault(entry.group, [0, 0])
                pair[0 if entry.kind == "loading_control" else 1] += 1
            parts = [
                f"分组“{group}”：{pair[0]} 条内参 / {pair[1]} 条目标蛋白"
                for group, pair in counts.items()
            ]
            diagnosis = (
                "尚无归一化结果——同一分组需要同时有内参与目标蛋白。当前 "
                + "；".join(parts)
                + "。若某条“类型”选错了，双击历史表对应单元格即可修改。"
            )
        self.normalize_panel.set_diagnosis(diagnosis)

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
        entries[row] = QuantEntry(
            kind=kind,  # type: ignore[arg-type]
            name=name,
            group=group,
            intensities=old.intensities,
            backgrounds=old.backgrounds,
            mean_intensities=old.mean_intensities,
            source=old.source,
        )
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
        try:
            csv_path, preview_path = export_result(self._result, image_path, selected)
        except (ValueError, OSError) as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return
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
        except (ValueError, OSError) as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
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
        self.act_clear_roi.setEnabled(has_image)
        self.act_delete.setEnabled(has_result)
        self.act_rot_left.setEnabled(has_image and not self._busy)
        self.act_rot_right.setEnabled(has_image and not self._busy)
        self.act_undo.setEnabled(bool(self._undo))

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
        for url in event.mimeData().urls():
            if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in IMAGE_SUFFIXES:
                event.acceptProposedAction()
                return

    def dropEvent(self, event: QDropEvent) -> None:
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
        guides = str(settings.value("params/show_guides", "true")).lower()
        show_guides = guides in ("true", "1")
        self.act_guides.setChecked(show_guides)
        self.canvas.set_guides_visible(show_guides)
        self.guides_check.blockSignals(True)
        self.guides_check.setChecked(show_guides)
        self.guides_check.blockSignals(False)

    def closeEvent(self, event: QCloseEvent) -> None:
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
        super().closeEvent(event)

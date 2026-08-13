"""界面偏好：字体大小、画布标注线宽、结果表列显示。

为什么字体存的是绝对 pt、用的却是相对倍数：用户想的是"这台 4K 屏上 9.5 pt 太
小"，所以设置里必须写 pt；而画布标签、剖面图坐标轴那些字号是各自独立的常量
（条带标签 8 pt、图标题 10 pt），它们只能按同一个倍数一起放大，否则字号一调
就各走各的。倍数以主题默认值为 1.0。

线宽同理，存的是倍数而不是像素：画布上"被自检点名的分隔线更粗"（5 对 3）是一
条**靠比例成立**的区分。做成绝对像素的话，用户把基础线宽调到 5 就再也看不出
哪条被点名了；做成倍数，比例恒定，区分永远在。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from PySide6.QtCore import QSettings
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from . import theme
from .results_panel import DEFAULT_OPTIONAL_COLUMNS, OPTIONAL_COLUMNS

DEFAULT_FONT_PT = float(theme.FONT_POINT_SIZE)
MIN_FONT_PT = 7.0
MAX_FONT_PT = 18.0
FONT_STEP = 0.5

DEFAULT_LINE_SCALE = 1.0
MIN_LINE_SCALE = 0.5
MAX_LINE_SCALE = 3.0
LINE_STEP = 0.25

SETTINGS_ORG = "WBAnalyzer"
SETTINGS_APP = "WBAnalyzer"

KEY_FONT_PT = "ui/font_pt"
KEY_ROI_LINE = "ui/line_roi"
KEY_BAND_LINE = "ui/line_band"
KEY_SEPARATOR_LINE = "ui/line_separator"
KEY_RESULT_COLUMNS = "ui/result_columns"
KEY_DATA_EXPORT_RAW = "export/data_raw_stats"
KEY_DATA_EXPORT_NORMALIZED = "export/data_normalized"
KEY_DATA_EXPORT_SUMMARY = "export/data_summary"
KEY_DATA_EXPORT_FULL_STATS = "export/data_full_stats"
KEY_DATA_EXPORT_SESSION_RAW = "export/data_session_raw"


def _clamp(value, low: float, high: float, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if number != number:  # NaN
        return fallback
    return min(max(number, low), high)


@dataclass(frozen=True)
class UiPrefs:
    """一份完整的界面偏好。不可变：改动一律经 :func:`replace` 产生新值。"""

    font_pt: float = DEFAULT_FONT_PT
    roi_line: float = DEFAULT_LINE_SCALE
    band_line: float = DEFAULT_LINE_SCALE
    separator_line: float = DEFAULT_LINE_SCALE
    result_columns: tuple[str, ...] = tuple(DEFAULT_OPTIONAL_COLUMNS)

    @property
    def font_scale(self) -> float:
        """相对主题默认字号的倍数——画布与剖面图的字号按它整体缩放。"""
        return self.font_pt / DEFAULT_FONT_PT

    @property
    def uses_default_font(self) -> bool:
        """字号没被动过。此时一个 ``setFont`` 都不该发出去（见 main_window）。"""
        return abs(self.font_pt - DEFAULT_FONT_PT) < 1e-6

    def sanitized(self) -> UiPrefs:
        return replace(
            self,
            font_pt=_clamp(self.font_pt, MIN_FONT_PT, MAX_FONT_PT, DEFAULT_FONT_PT),
            roi_line=_clamp(
                self.roi_line, MIN_LINE_SCALE, MAX_LINE_SCALE, DEFAULT_LINE_SCALE
            ),
            band_line=_clamp(
                self.band_line, MIN_LINE_SCALE, MAX_LINE_SCALE, DEFAULT_LINE_SCALE
            ),
            separator_line=_clamp(
                self.separator_line,
                MIN_LINE_SCALE,
                MAX_LINE_SCALE,
                DEFAULT_LINE_SCALE,
            ),
            result_columns=tuple(
                key for key in OPTIONAL_COLUMNS if key in set(self.result_columns)
            ),
        )


DEFAULTS = UiPrefs()

_BASE_APP_FONT: QFont | None = None
BASE_FONT_PROPERTY = "wbBaseFontPt"


def base_app_font() -> QFont:
    """进程启动时那份应用字体——"恢复默认"要退回的就是它。

    只抓一次。每次都读 ``QApplication.font()`` 的话，第二次调用抓到的就是上一
    次设进去的值，"默认"会一路往上漂。
    """
    global _BASE_APP_FONT
    if _BASE_APP_FONT is None:
        _BASE_APP_FONT = QFont(QApplication.font())
    return QFont(_BASE_APP_FONT)


def remember_base_font(widget, applied_scale: float = 1.0) -> None:
    """记下这个控件"倍数为 1 时"的字号，只记第一次。

    ``applied_scale`` 是**当前已经生效**的倍数。晚建出来的控件（比如设置对话框）
    一出生就继承了已经放大过的应用字体，直接把它当基准会再放大一次；除掉当前
    倍数才拿得到原始值。

    必须与 :func:`scale_widget_font` 分成两趟：同一趟里先缩放父控件、再处理子
    控件的话，子控件读到的"当前字号"已经是父控件放大后的值，一层套一层地翻倍
    （实测三层下来 9.5 pt 变成 42 pt）。
    """
    if widget.property(BASE_FONT_PROPERTY) is not None:
        return
    point_size = float(widget.font().pointSizeF())
    if point_size <= 0:  # 用像素定义的字体：没有可缩放的磅值，放过它
        return
    widget.setProperty(
        BASE_FONT_PROPERTY, point_size / max(1e-6, float(applied_scale))
    )


def scale_widget_font(widget, scale: float) -> None:
    """把控件字号设成"基准 × 倍数"，其余字体属性原样保留。

    为什么要逐个控件设而不是在主窗口上 ``setFont`` 一次：应用了全局 QSS 之后，
    ``QStyleSheetStyle`` 会给每个被 polish 过的控件塞一份**显式**字体（它得把
    ``font-weight: 600`` 这类规则落实下去），显式字体一旦存在，父级的 setFont
    就再也传不下来了。实测装了主题之后主窗口 setFont 对表格毫无影响。

    只改字号、不碰字族与粗细，正是为了不把 QSS 给表头 / 分组标题设的 600 粗细
    抹掉。算的是"基准 × 倍数"而不是"当前 × 增量"：连续调整不累积误差，倍数回到
    1.0 就精确复原。
    """
    base = widget.property(BASE_FONT_PROPERTY)
    if base is None:
        return
    font = QFont(widget.font())
    wanted = max(1.0, float(base) * float(scale))
    if abs(font.pointSizeF() - wanted) < 1e-6:
        return
    font.setPointSizeF(wanted)
    widget.setFont(font)


def settings() -> QSettings:
    return QSettings(SETTINGS_ORG, SETTINGS_APP)


def load(store: QSettings | None = None) -> UiPrefs:
    store = store or settings()
    raw = store.value(KEY_RESULT_COLUMNS, None)
    if raw is None:
        columns = tuple(DEFAULT_OPTIONAL_COLUMNS)
    elif isinstance(raw, str):
        # QSettings 的 INI 后端把单元素列表读回成裸字符串，空列表读回成 ""
        columns = tuple(part for part in raw.split(",") if part)
    else:
        columns = tuple(str(part) for part in raw if str(part))
    return UiPrefs(
        font_pt=_clamp(
            store.value(KEY_FONT_PT, DEFAULT_FONT_PT),
            MIN_FONT_PT,
            MAX_FONT_PT,
            DEFAULT_FONT_PT,
        ),
        roi_line=_clamp(
            store.value(KEY_ROI_LINE, DEFAULT_LINE_SCALE),
            MIN_LINE_SCALE,
            MAX_LINE_SCALE,
            DEFAULT_LINE_SCALE,
        ),
        band_line=_clamp(
            store.value(KEY_BAND_LINE, DEFAULT_LINE_SCALE),
            MIN_LINE_SCALE,
            MAX_LINE_SCALE,
            DEFAULT_LINE_SCALE,
        ),
        separator_line=_clamp(
            store.value(KEY_SEPARATOR_LINE, DEFAULT_LINE_SCALE),
            MIN_LINE_SCALE,
            MAX_LINE_SCALE,
            DEFAULT_LINE_SCALE,
        ),
        result_columns=columns,
    ).sanitized()


def save(value: UiPrefs, store: QSettings | None = None) -> None:
    store = store or settings()
    value = value.sanitized()
    store.setValue(KEY_FONT_PT, float(value.font_pt))
    store.setValue(KEY_ROI_LINE, float(value.roi_line))
    store.setValue(KEY_BAND_LINE, float(value.band_line))
    store.setValue(KEY_SEPARATOR_LINE, float(value.separator_line))
    # 逗号串而不是列表：空列表经 INI 往返会变成 None，那样"一列都不勾"就会被
    # 当成"没设置过"、下次启动又冒出默认那两列
    store.setValue(KEY_RESULT_COLUMNS, ",".join(value.result_columns))


def _as_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    return default


def load_data_export_choices(
    store: QSettings | None = None,
) -> tuple[bool, bool, bool, bool, bool]:
    """上次「导出数据」勾选了哪几项。

    返回 ``(原始灰度, 归一化, 跨膜汇总, 完整统计列, 历史长表)``。
    缺省：只导精简的原始灰度统计。
    """
    store = store or settings()
    return (
        _as_bool(store.value(KEY_DATA_EXPORT_RAW, True), True),
        _as_bool(store.value(KEY_DATA_EXPORT_NORMALIZED, False), False),
        _as_bool(store.value(KEY_DATA_EXPORT_SUMMARY, False), False),
        _as_bool(store.value(KEY_DATA_EXPORT_FULL_STATS, False), False),
        _as_bool(store.value(KEY_DATA_EXPORT_SESSION_RAW, False), False),
    )


def save_data_export_choices(
    raw_stats: bool,
    normalized: bool,
    summary_tables: bool,
    full_stats: bool = False,
    session_raw: bool = False,
    store: QSettings | None = None,
) -> None:
    store = store or settings()
    store.setValue(KEY_DATA_EXPORT_RAW, bool(raw_stats))
    store.setValue(KEY_DATA_EXPORT_NORMALIZED, bool(normalized))
    store.setValue(KEY_DATA_EXPORT_SUMMARY, bool(summary_tables))
    store.setValue(KEY_DATA_EXPORT_FULL_STATS, bool(full_stats))
    store.setValue(KEY_DATA_EXPORT_SESSION_RAW, bool(session_raw))
    store.sync()


__all__ = [
    "DEFAULTS",
    "DEFAULT_FONT_PT",
    "DEFAULT_LINE_SCALE",
    "FONT_STEP",
    "KEY_DATA_EXPORT_FULL_STATS",
    "KEY_DATA_EXPORT_NORMALIZED",
    "KEY_DATA_EXPORT_RAW",
    "KEY_DATA_EXPORT_SESSION_RAW",
    "KEY_DATA_EXPORT_SUMMARY",
    "LINE_STEP",
    "MAX_FONT_PT",
    "MAX_LINE_SCALE",
    "MIN_FONT_PT",
    "MIN_LINE_SCALE",
    "UiPrefs",
    "base_app_font",
    "load",
    "load_data_export_choices",
    "remember_base_font",
    "save",
    "save_data_export_choices",
    "scale_widget_font",
]

"""多文档标签的状态袋。

主窗口上那一大摊字段（三层图像、ROI、撤销栈、定量结果……）是「当前正在看的
那一张膜」。切标签时把它们原样搬进 / 搬出 :class:`Document`，各标签才不会串味。

像素数组占内存，16 位大图开多了会顶满；``pixels_resident`` 为假时数组已经丢掉，
切回来再从 ``path`` 重载。分析状态（ROI、结果、撤销）始终留在这份袋子里。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..normalize import source_key

# 内存里同时保留原始 numpy 数组的标签数上限。再早的标签切回来时从磁盘重载。
# 没有文件路径的（测试里 blot_from_array 直接塞进来的）永不驱逐，没法重载。
MAX_RESIDENT_DOCS = 4


@dataclass
class Document:
    """一份打开的膜。字段与 MainWindow 上当前文档的活状态一一对应。"""

    doc_id: int
    path: Path | None = None
    title: str = "未命名"
    blot_loaded: object = None
    blot_original: object = None
    blot: object = None
    pixels_resident: bool = True
    angle: float = 0.0
    quarter_turns: int = 0
    orient_target: tuple = (0, 0.0)
    pending_restore: object = None
    roi: object = None
    separators: list = field(default_factory=list)
    result: object = None
    result_origin: str = ""
    result_doc_version: int = -1
    results_dirty: bool = False
    lanes: list = field(default_factory=list)
    roi_report: object = None
    result_expected_bands: int | None = None
    detect_dropped_roi: bool = False
    ruler_angle: float | None = None
    ruler_text: str = "标尺：未测量"
    tilt_guide_angle: float | None = None
    doc_version: int = 0
    source_version: int = 0
    dark_cache: tuple | None = None
    column_profile: object = None
    undo: list = field(default_factory=list)
    redo: list = field(default_factory=list)
    undo_verb: str = "已撤销"
    committed: object = None
    result_stale: bool = False
    quant_was_manual: bool = False
    quant_manual_pending: bool = False
    quant_job_state: tuple | None = None
    quant_job_expected: int | None = None
    view_state: tuple | None = None
    tool: object = None
    bg_index: int = 0
    lane_count: int = 0
    band_count: int = 0
    weak_value: int = 55
    diagnostics_state: tuple | None = None
    selected_bands: list = field(default_factory=list)
    current_lane: int | None = None
    reload_generation: int = 0

    def source(self) -> str:
        """给「同一文件不要开两个标签」用的身份；没有路径就空串。"""
        if self.path is None:
            return ""
        return source_key(self.path)

    def display_name(self) -> str:
        if self.path is not None:
            return self.path.name
        return self.title or "未命名"

    def tooltip(self) -> str:
        if self.path is not None:
            return str(self.path)
        return self.display_name()

"""导出发表级图像的选项对话框。

导出这条路上唯一不可逆的一步是灰度映射，所以它被放在最上面，并且**两个版本
各有各的映射**：干净版是 16 位无损存档，线性全量程映射既不拉伸也不削波；标注
版是 8 位彩色、给人在屏幕和纸上看的，真实膜的数据挤在量程顶部 0.5%，同一个
线性映射下只剩 196 个灰阶且全落在 245–255，条带几乎看不出来。用一个下拉管两个
版本，必然有一版是错的。

对话框只收集参数、不碰文件：真正的导出在 :mod:`wb_analyzer.export` 里，文件名
和覆盖确认在主窗口。这样这里可以在测试里直接构造、读取选项，不需要文件系统。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QScrollArea,
    QVBoxLayout,
)

from ..export import (
    LAYER_BAND_BOXES,
    LAYER_BAND_LABELS,
    LAYER_ROI,
    LAYER_SEPARATORS,
    dpi_for_width_mm,
)
from ..imageio import MAPPING_AUTO, MAPPING_LINEAR, MAPPING_MANUAL

MAPPING_CHOICES = (
    ("线性全量程（不拉伸、不削波）", MAPPING_LINEAR),
    ("自动对比度拉伸（同屏幕预览）", MAPPING_AUTO),
    ("手动黑白点", MAPPING_MANUAL),
)

BIT_DEPTH_CHOICES = (("跟随源位深", None), ("8 位", 8), ("16 位", 16))

LAYER_CHOICES = (
    ("条带框", LAYER_BAND_BOXES),
    ("文字标签", LAYER_BAND_LABELS),
    ("ROI 框", LAYER_ROI),
    ("泳道分隔线", LAYER_SEPARATORS),
)

# 打印宽度 spin 的"不指定"档：0 mm 没有物理意义，正好拿来当空值
NO_PRINT_WIDTH = "不指定（按 300 dpi 报告）"


@dataclass(frozen=True)
class FigureExportOptions:
    """一次导出的全部选项。两个版本的映射方式是分开的，见模块文档。"""

    clean: bool = True
    annotated: bool = False
    clean_mapping: str = MAPPING_LINEAR
    annotated_mapping: str = MAPPING_AUTO
    black_point: float = 0.0
    white_point: float | None = None
    """``None`` 表示"还没定过，用源图的满量程"——16 位图上写死 255 会把整张图
    推成纯白，而那正是这次重做导出要消灭的东西。"""
    bit_depth: int | None = None
    print_width_mm: float | None = None
    layers: tuple[str, ...] = field(
        default_factory=lambda: tuple(value for _, value in LAYER_CHOICES)
    )

    def needs_manual_points(self) -> bool:
        return MAPPING_MANUAL in self.mappings_in_use()

    def mappings_in_use(self) -> tuple[str, ...]:
        used = []
        if self.clean:
            used.append(self.clean_mapping)
        if self.annotated:
            used.append(self.annotated_mapping)
        return tuple(used)


class FigureExportDialog(QDialog):
    """收集导出选项。构造时给出源图尺寸，好把可达 dpi 当场算给用户看。

    dpi 在这里不是一个可以调大的设置：本软件从不插值放大，像素数是固定的，
    dpi 只是"这些像素印多宽"的结果。所以打印宽度旁边实时显示算出来的 dpi，
    用户拉宽度就能看见分辨率掉下去，不必等导出完再去翻旁注文件。
    """

    def __init__(
        self,
        parent=None,
        *,
        width_px: int = 0,
        height_px: int = 0,
        source_bit_depth: int = 8,
        white_level: float = 255.0,
        has_result: bool = False,
        options: FigureExportOptions | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("导出展示图")
        self.setModal(True)
        self._width_px = int(width_px)
        self._height_px = int(height_px)
        base = options or FigureExportOptions()

        layout = QVBoxLayout(self)

        version_box = QGroupBox("导出版本")
        version_layout = QVBoxLayout(version_box)
        self.clean_check = QCheckBox("干净版（无标注，可保持 16 位无损）")
        self.clean_check.setChecked(base.clean)
        self.annotated_check = QCheckBox("标注版（叠加条带框与标签，固定 8 位彩色）")
        self.annotated_check.setChecked(base.annotated)
        self.annotated_check.setToolTip(
            "彩色标注只能写成 8 位 RGB，没有主流写库能出 16 位彩色 TIFF。\n"
            "干净版不受影响，仍按下面选的位深导出。"
        )
        if not has_result:
            self.annotated_check.setEnabled(False)
            self.annotated_check.setChecked(False)
            self.annotated_check.setToolTip("还没有定量结果，没有可标注的条带。")
        version_layout.addWidget(self.clean_check)
        version_layout.addWidget(self.annotated_check)
        layout.addWidget(version_box)

        mapping_box = QGroupBox("灰度映射（唯一不可逆的一步）")
        mapping_form = QFormLayout(mapping_box)
        self.clean_mapping_combo = self._mapping_combo(base.clean_mapping)
        self.annotated_mapping_combo = self._mapping_combo(base.annotated_mapping)
        mapping_form.addRow("干净版", self.clean_mapping_combo)
        mapping_form.addRow("标注版", self.annotated_mapping_combo)
        self.black_spin = self._point_spin(base.black_point, white_level)
        self.white_spin = self._point_spin(base.white_point or white_level, white_level)
        mapping_form.addRow("黑点（→ 输出 0）", self.black_spin)
        mapping_form.addRow("白点（→ 输出满量程）", self.white_spin)
        self.mapping_hint = QLabel()
        self.mapping_hint.setWordWrap(True)
        self.mapping_hint.setObjectName("dimHint")
        mapping_form.addRow(self.mapping_hint)
        layout.addWidget(mapping_box)

        output_box = QGroupBox("输出")
        output_form = QFormLayout(output_box)
        self.depth_combo = QComboBox()
        for label, value in BIT_DEPTH_CHOICES:
            self.depth_combo.addItem(label, value)
        for index, (_, value) in enumerate(BIT_DEPTH_CHOICES):
            if value == base.bit_depth:
                self.depth_combo.setCurrentIndex(index)
                break
        self.depth_combo.setToolTip(
            f"源图是 {source_bit_depth} 位。“跟随源”对 16 位源出 16 位、8 位源出 8 位，"
            "不会凭空造出精度，也不会白丢。"
        )
        output_form.addRow("位深", self.depth_combo)
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.0, 1000.0)
        self.width_spin.setDecimals(1)
        self.width_spin.setSingleStep(5.0)
        self.width_spin.setSuffix(" mm")
        self.width_spin.setSpecialValueText(NO_PRINT_WIDTH)
        self.width_spin.setValue(base.print_width_mm or 0.0)
        output_form.addRow("打印宽度", self.width_spin)
        self.dpi_label = QLabel()
        self.dpi_label.setWordWrap(True)
        self.dpi_label.setObjectName("dimHint")
        output_form.addRow(self.dpi_label)
        layout.addWidget(output_box)

        self.layer_box = QGroupBox("标注图层")
        layer_grid = QGridLayout(self.layer_box)
        self.layer_checks: dict[str, QCheckBox] = {}
        for index, (label, value) in enumerate(LAYER_CHOICES):
            check = QCheckBox(label)
            check.setChecked(value in base.layers)
            self.layer_checks[value] = check
            layer_grid.addWidget(check, index // 2, index % 2)
        layout.addWidget(self.layer_box)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_button.setText("选择文件…")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        for widget in (self.clean_check, self.annotated_check):
            widget.toggled.connect(self._refresh)
        for combo in (self.clean_mapping_combo, self.annotated_mapping_combo):
            combo.currentIndexChanged.connect(self._refresh)
        for spin in (self.black_spin, self.white_spin, self.width_spin):
            spin.valueChanged.connect(self._refresh)
        self._refresh()

    # ------------------------------------------------------------------ 构件

    def _mapping_combo(self, current: str) -> QComboBox:
        combo = QComboBox()
        for label, value in MAPPING_CHOICES:
            combo.addItem(label, value)
        index = combo.findData(current)
        combo.setCurrentIndex(max(0, index))
        return combo

    def _point_spin(self, value: float, white_level: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, max(1.0, float(white_level) * 4.0))
        spin.setDecimals(1)
        spin.setSingleStep(max(1.0, float(white_level) / 256.0))
        spin.setValue(float(value))
        return spin

    # ------------------------------------------------------------------ 联动

    def _refresh(self) -> None:
        annotated = self.annotated_check.isChecked()
        self.annotated_mapping_combo.setEnabled(annotated)
        self.layer_box.setEnabled(annotated)
        self.clean_mapping_combo.setEnabled(self.clean_check.isChecked())

        options = self.options()
        manual = options.needs_manual_points()
        self.black_spin.setEnabled(manual)
        self.white_spin.setEnabled(manual)
        bad_points = manual and self.white_spin.value() - self.black_spin.value() < 1e-9
        no_version = not options.clean and not options.annotated
        self.ok_button.setEnabled(not no_version and not bad_points)

        if no_version:
            self.mapping_hint.setText("⚠ 至少要勾一种版本，否则没有文件可写。")
        elif bad_points:
            self.mapping_hint.setText("⚠ 白点必须大于黑点。")
        elif options.annotated and options.annotated_mapping == MAPPING_LINEAR:
            self.mapping_hint.setText(
                "线性映射下真实膜的数据常挤在量程顶部不到 1% 的区间里，"
                "8 位标注图会只剩两百来个灰阶、屏幕上几乎看不出条带。"
                "标注版是给人看的，建议留在“自动对比度拉伸”。"
            )
        else:
            self.mapping_hint.setText(
                "两个版本各用各的映射：干净版存档、标注版看图。"
                "实际用到的黑白点会写进 *_export_info.txt 旁注。"
            )
        self.dpi_label.setText(self._dpi_text(options.print_width_mm))

    def _dpi_text(self, width_mm: float | None) -> str:
        if self._width_px <= 0:
            return ""
        if width_mm is None:
            return (
                f"图像 {self._width_px} × {self._height_px} px，"
                "不指定宽度时按 300 dpi 标注（不做任何插值放大）。"
            )
        dpi = dpi_for_width_mm(self._width_px, width_mm)
        verdict = "达到 300 dpi" if dpi >= 300 else "低于 300 dpi，多数期刊会退回"
        return (
            f"{self._width_px} px 印成 {width_mm:.1f} mm 宽 = {dpi:.0f} dpi（{verdict}）。"
            "本软件不插值放大，dpi 是印刷尺寸算出来的结果。"
        )

    # ------------------------------------------------------------------ 读取

    def options(self) -> FigureExportOptions:
        width = self.width_spin.value()
        return FigureExportOptions(
            clean=self.clean_check.isChecked(),
            annotated=self.annotated_check.isChecked(),
            clean_mapping=self.clean_mapping_combo.currentData(),
            annotated_mapping=self.annotated_mapping_combo.currentData(),
            black_point=float(self.black_spin.value()),
            white_point=float(self.white_spin.value()),
            bit_depth=self.depth_combo.currentData(),
            print_width_mm=None if width <= 0.0 else float(width),
            layers=tuple(
                value for value, check in self.layer_checks.items() if check.isChecked()
            ),
        )


class FigureInfoDialog(QDialog):
    """导出完成后把旁注 ``*_export_info.txt`` 的内容直接摆出来。

    里面有一段可以照抄进论文方法学的话，以及削波比例这种"图已经损失了什么"的
    事实。文件就躺在图旁边，但没人会主动去开一个 txt——所以弹一次。
    """

    def __init__(self, parent=None, *, title: str = "导出完成", text: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        layout = QVBoxLayout(self)
        heading = QLabel(
            "下面这段已经写进图旁边的 *_export_info.txt，可直接抄进论文的方法学。"
        )
        heading.setWordWrap(True)
        layout.addWidget(heading)
        self.info_label = QLabel(text)
        self.info_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.info_label.setTextFormat(Qt.TextFormat.PlainText)
        self.info_label.setWordWrap(False)
        # 旁注是按列对齐排版的，比例字体会把它排成一团
        self.info_label.setFont(
            QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        )
        area = QScrollArea()
        area.setWidget(self.info_label)
        area.setWidgetResizable(True)
        area.setMinimumSize(560, 420)
        layout.addWidget(area, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Close).setText("关闭")
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    def text(self) -> str:
        return self.info_label.text()

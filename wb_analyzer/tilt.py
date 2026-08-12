"""自动测量条带倾角：行投影 + 平方梯度锐度最大化。

条带排成一行时，把图按不同角度错切后取"每行的列平均"（行投影），真正水平的
那个角度上行投影的边缘最陡——因为同一条带的所有像素都落进了同一行。于是在一组
候选角度上取行投影平方梯度和 ``Σ(Δp)²`` 的极大值点，就是条带的倾角。

横向对比过 Hough、结构张量、FFT 主方向、列投影、分块中位投影：在真实图 × 已知
角度上，本方法 RMS 0.107°，次优的行投影·方差 0.134°，Hough 1.76°、结构张量
2.05°、FFT 5.02° 都差一个数量级。本模块实测 RMS 0.079°（13 张真实图 × 15 个角度）。

**角度约定**：返回值与 :func:`wb_analyzer.roi.angle_to_horizontal`（也就是界面
标尺 ``_ruler_angle``）完全一致——图像坐标系下条带方向相对水平的倾角，右端越低
角度越正。所以它可以原样喂给 ``rotation_spin`` /
:func:`wb_analyzer.roi.rotate_image` 把图旋正，正角度抬右边。见
``tests/test_tilt.py::test_estimate_matches_ruler_convention``。

**这个算法无法自判失败**：调研在 468 个 ROI 样本上试过 5 种自检判据（峰突出度、
方差判据一致性、行列一致性、三分块一致性、是否顶到搜索上限），好样本与坏样本的
分布几乎完全重叠——真正要命的失败（锁到扫描边框、锁到划痕）都是自洽的，所有子
判据被同一条强直线一起骗到，自然互相同意；分块一致性甚至反向。最好的标定规则
覆盖率只有 33%，假自信率仍有 3.9%。所以本模块不做任何置信度评分，只用三条能
事先验证的工程约束拒答（见 :func:`estimate_tilt`），剩下的必须由人确认。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .roi import RoiRect, to_gray

MAX_SEARCH_ANGLE = 10.0
"""搜索范围 ±10°，顶到边界一律判为测不准，不返回边界值。

再放宽只会增加锁到扫描边框、装订线一类强直线的机会：歪超过 10° 该先用整 90°
旋转把方向摆正，那不是微调要解决的问题。
"""

MIN_CROP_WIDTH_PX = 250
"""窄于这个宽度直接拒答：两端高差不够，角度杠杆不足。

实测按裁切宽度分箱的误差：<200 px 时 median 0.23° / p90 1.34°；200–350 px 是
0.058° / 0.30°；≥350 px 是 0.022° / 0.077°。250 px 卡在拐点上。
"""

MIN_CROP_HEIGHT_PX = 24
"""行投影至少要这么多行才谈得上"边缘陡不陡"。

不是实测阈值，是防退化的下限：几行高的裁切算出来的极值点纯属噪声。
"""

TILT_OK = "ok"
TILT_NO_ROI = "no_roi"
TILT_TOO_NARROW = "too_narrow"
TILT_TOO_SHORT = "too_short"
TILT_AT_LIMIT = "at_limit"
TILT_NO_SIGNAL = "no_signal"

_INSET_FRACTION = 0.04
"""四边各内缩 4%：ROI 边缘常常压着膜边、胶带或框选时蹭到的边界，
那些是贯穿全宽的强直线，正是这个算法最容易被骗的东西。"""

_WORK_MAX_WIDTH = 600
"""工作图宽度上限。实测长边 900 与 600 精度完全相同（RMS 0.079° vs 0.082°），
600 够用且快一倍。宽度决定角度杠杆，所以按宽度而不是长边定标。"""

_WORK_MAX_HEIGHT = 300
"""工作图高度上限。行投影的角分辨率取决于"横向拖开量 / 边缘宽度"，两者随纵向
缩放同比变化，所以压缩高度几乎不损失精度（实测 RMS 0.079° vs 不压缩 0.078°），
却把又高又宽的 ROI 的耗时从 ~40 ms 压回 ~10 ms。"""

_COARSE_STEP = 0.5
_FINE_STEP = 0.05
_BLUR_SIGMA = 1.5
"""重采样前的高斯模糊。1.5–2.0 是实测最优的平坦区间（RMS 0.082° / 0.078°），
往两边走都变差（1.2 是 0.097°，3.5 是 0.091°）。它压掉细划痕与噪声，
代价是可忽略的角分辨率。"""

_DITHER_PHI = 0.6180339887498949
"""每列错开的亚像素抖动步长（黄金比例共轭，低差异且无线性趋势）。

双线性重采样在整数偏移处不插值、在半像素处最平滑，而 0° 时所有列的偏移都恰好
是整数——凭空多出一个"锐度尖峰"，真实倾角在 ±0.4° 内时会被它吸过去，报出一个
漂亮的 0.00°（假自信）。给每列加一个固定的亚像素抖动，0° 就和别的角度一样要插值，
尖峰消失。实测真实图 RMS 从 0.194° 降到 0.079°，合成图从 0.006° 降到 0.001°。
抖动本身只带来 <0.001° 的偏置（等分布序列与列号不相关）。
"""


@dataclass(frozen=True)
class TiltEstimate:
    """一次自动倾角测量的结果。

    ``angle_deg`` 只在可采信时给出角度，测不准时是 ``None``——不是 0.0，也不是
    边界值。界面可以直接把它赋给 ``_ruler_angle``，``None`` 正好是"未测量"。
    """

    angle_deg: float | None
    """建议的旋正角，单位度，约定同界面标尺（右端低为正）。测不准时为 ``None``。"""

    reason_code: str
    """:data:`TILT_OK` / :data:`TILT_NO_ROI` / :data:`TILT_TOO_NARROW` /
    :data:`TILT_TOO_SHORT` / :data:`TILT_AT_LIMIT` / :data:`TILT_NO_SIGNAL`。"""

    message: str
    """面向用户的简体中文说明，任何情况下都非空，可直接显示。"""

    crop_width_px: int
    """实际参与测量的裁切宽度（像素），拒答文案里要用到。"""

    raw_angle_deg: float | None
    """未经采信判断的原始极值点，仅供诊断 / 日志。顶到边界时就是边界值。"""

    @property
    def trusted(self) -> bool:
        """是否可采信。注意**可采信 ≠ 正确**，见模块文档：界面必须让用户确认。"""
        return self.angle_deg is not None


def _fail(code: str, message: str, width: int, raw: float | None = None) -> TiltEstimate:
    return TiltEstimate(
        angle_deg=None,
        reason_code=code,
        message=message,
        crop_width_px=int(width),
        raw_angle_deg=raw,
    )


_NO_SIGNAL_MESSAGE = (
    "选区里看不到条带结构（几乎没有明暗变化），测不准。"
    "请重新框到条带上，或用“标尺对齐”手动量。"
)


def _prepare(crop: np.ndarray) -> tuple[np.ndarray, float] | None:
    """内缩、降采样、归一化、模糊。返回 ``(工作图, 纵横缩放比)``；全平时返回 ``None``。

    纵横缩放比 = ``ky / kx``：高度压得比宽度狠时，原图里 ``tan θ`` 的斜率在工作图
    里变成 ``tan θ · ky / kx``，算斜率时必须乘回去，否则角度会被系统性缩小。
    """
    height, width = crop.shape[:2]
    inset_x = int(round(width * _INSET_FRACTION))
    inset_y = int(round(height * _INSET_FRACTION))
    if width - 2 * inset_x >= 8 and height - 2 * inset_y >= 8:
        crop = crop[inset_y : height - inset_y, inset_x : width - inset_x]

    work = np.nan_to_num(crop.astype(np.float32, copy=True), nan=0.0, posinf=0.0, neginf=0.0)
    # 动态范围要在缩放**之前**判：INTER_AREA 的加权和有 float32 舍入误差，一张
    # 全常数的图缩完可能剩下 1e-6 的假跨度，再归一化就被放大成满量程的数值噪声，
    # 于是空白 ROI 也能算出一个"可采信"的角度——正是最该避免的假自信。
    low = float(work.min())
    span = float(work.max()) - low
    if not np.isfinite(span) or span <= max(abs(low), 1.0) * 1e-6:
        return None

    height, width = work.shape[:2]
    scale_x = min(1.0, _WORK_MAX_WIDTH / float(width))
    scale_y = min(scale_x, _WORK_MAX_HEIGHT / float(height))
    if scale_x < 1.0 or scale_y < scale_x:
        # INTER_AREA 是面积平均，等于先低通再抽样，缩小时不会混叠出假边缘
        work = cv2.resize(
            work,
            (max(8, int(round(width * scale_x))), max(8, int(round(height * scale_y)))),
            interpolation=cv2.INTER_AREA,
        )
    if min(work.shape[:2]) < 8:
        return None

    work = (work - low) / span
    work = cv2.GaussianBlur(work, (0, 0), _BLUR_SIGMA, borderType=cv2.BORDER_REPLICATE)
    return work, scale_y / scale_x


class _Shearer:
    """按角度做垂直错切并取行投影平方梯度，采样网格复用。

    用错切而不是真旋转：一条斜率 ``tan t`` 的直线在错切参数 ``a = t`` 时恰好被
    压平（和旋转 ``t`` 等效），但错切不改变外形、不用扩画布，也就没有旋转留下的
    空角和假边缘。越界部分用 ``BORDER_REPLICATE`` 补——ROI 上下缘本来就是背景，
    复制背景无害；反过来若按"有效列数"归一化，大角度处参与平均的列变少、噪声
    变大，平方梯度反而虚高，会把极值点往搜索边界上拽。
    """

    def __init__(self, work: np.ndarray, aspect_ratio: float) -> None:
        self._work = work
        self._aspect = float(aspect_ratio)
        height, width = work.shape[:2]
        columns = np.arange(width, dtype=np.float64)
        self._map_x = np.ascontiguousarray(
            np.broadcast_to(columns.astype(np.float32), (height, width))
        )
        dither = (columns * _DITHER_PHI) % 1.0
        self._base_y = (np.arange(height, dtype=np.float64)[:, None] + dither[None, :]).astype(
            np.float32
        )
        self._offset = (columns - (width - 1) * 0.5)[None, :].astype(np.float32)
        self._map_y = np.empty_like(self._base_y)

    def scores(self, angles: np.ndarray) -> np.ndarray:
        out = np.empty(angles.shape[0], dtype=np.float64)
        for index, angle in enumerate(angles):
            slope = float(np.tan(np.radians(float(angle)))) * self._aspect
            # 输出第 r 行第 x 列取自原图第 r + slope*(x-cx) 行，
            # 于是斜率 slope 的直线被压成一行
            np.add(self._base_y, self._offset * np.float32(slope), out=self._map_y)
            sheared = cv2.remap(
                self._work,
                self._map_x,
                self._map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            profile = sheared.mean(axis=1, dtype=np.float64)
            gradient = np.diff(profile)
            out[index] = float(gradient @ gradient)
        return out


def _refine_peak(angles: np.ndarray, scores: np.ndarray) -> float:
    """在离散极值点上做抛物线插值，拿到步长以下的精度。"""
    index = int(np.argmax(scores))
    if index == 0 or index == scores.shape[0] - 1:
        return float(angles[index])
    left, middle, right = scores[index - 1], scores[index], scores[index + 1]
    denominator = left - 2.0 * middle + right
    if not np.isfinite(denominator) or denominator >= 0.0:
        return float(angles[index])
    offset = 0.5 * (left - right) / denominator
    if not np.isfinite(offset):
        return float(angles[index])
    offset = float(np.clip(offset, -1.0, 1.0))
    return float(angles[index]) + offset * float(angles[1] - angles[0])


def estimate_tilt(
    gray: np.ndarray,
    *,
    roi: RoiRect | None = None,
    dark_bands: bool | None = None,
) -> TiltEstimate:
    """测量 ROI 内条带的倾角，返回一个**建议**（不是结论）。

    ``gray`` 接受 2 维灰度或 3 通道 BGR，uint8 / uint16 / 浮点都行。

    ``roi`` 必填，这是三条拒答约束里最重要的一条：整图上跑会锁到扫描边框上。
    调研做过对照实验——只旋转膜、外框不动时，Hough 的斜率降到 0 完全失效，行
    投影同理。如果传进来的数组本身就是膜裁切，请显式传 ``RoiRect(0, 0, w, h)``
    表示"我确认这不是整图"。

    ``dark_bands`` 保留参数，**不影响结果**：平方梯度对明暗极性天然不变
    （``Σ(Δ(-p))² ≡ Σ(Δp)²``），亮条带与暗条带算出的角度只差浮点尾数。留着只是
    为了和 :mod:`wb_analyzer.quant` 等模块的调用风格一致。

    三条拒答约束（都不是算法自省，是能事先验证的工程约束）：

    1. 没有 ROI —— 直接拒答；
    2. 裁切宽度 < :data:`MIN_CROP_WIDTH_PX` —— 角度杠杆不够，拒答；
    3. 极值点顶到 ±:data:`MAX_SEARCH_ANGLE` 边界 —— 拒答，不返回边界值。

    挡不住的已知失败模式：贯穿全宽的淡划痕（20% 对比度时 p90 误差 7.43°）。所以
    调用方**绝不能自动应用**返回的角度，必须让用户看一眼再决定。
    """
    del dark_bands  # 见 docstring：平方梯度对明暗极性不变

    if roi is None:
        return _fail(
            TILT_NO_ROI,
            "自动测倾角要先框出 ROI（或膜的范围）：整图上算会锁到扫描边框上。"
            "请先框选，或用“标尺对齐”手动量。",
            0,
        )

    image = np.asarray(gray)
    if image.ndim == 3:
        image = to_gray(image)
    if image.ndim != 2 or image.size == 0:
        return _fail(TILT_NO_SIGNAL, "图像为空，无法测量倾角。请重新加载图片。", 0)

    box = roi.clipped(image)
    width = int(box.width)
    if width < MIN_CROP_WIDTH_PX:
        return _fail(
            TILT_TOO_NARROW,
            f"选区只有 {width} px 宽（需 ≥ {MIN_CROP_WIDTH_PX} px），"
            "太窄时两端高差不够、测不准。请把选区拉宽，或用“标尺对齐”手动量。",
            width,
        )
    if int(box.height) < MIN_CROP_HEIGHT_PX:
        return _fail(
            TILT_TOO_SHORT,
            f"选区只有 {int(box.height)} px 高（需 ≥ {MIN_CROP_HEIGHT_PX} px），"
            "行投影取不到足够的行、测不准。请把选区拉高，或用“标尺对齐”手动量。",
            width,
        )

    prepared = _prepare(image[box.y : box.y + box.height, box.x : box.x + box.width])
    if prepared is None:
        return _fail(TILT_NO_SIGNAL, _NO_SIGNAL_MESSAGE, width)
    shearer = _Shearer(*prepared)

    coarse_angles = np.arange(
        -MAX_SEARCH_ANGLE, MAX_SEARCH_ANGLE + _COARSE_STEP * 0.5, _COARSE_STEP
    )
    coarse_scores = shearer.scores(coarse_angles)
    if not np.isfinite(coarse_scores).all() or float(coarse_scores.max()) <= 0.0:
        return _fail(TILT_NO_SIGNAL, _NO_SIGNAL_MESSAGE, width)

    coarse_index = int(np.argmax(coarse_scores))
    center = float(coarse_angles[coarse_index])
    fine_angles = np.arange(
        center - _COARSE_STEP, center + _COARSE_STEP + _FINE_STEP * 0.5, _FINE_STEP
    )
    fine_angles = fine_angles[np.abs(fine_angles) <= MAX_SEARCH_ANGLE + 1e-9]
    angle = _refine_peak(fine_angles, shearer.scores(fine_angles))
    angle = float(np.clip(angle, -MAX_SEARCH_ANGLE, MAX_SEARCH_ANGLE))

    if coarse_index in (0, coarse_scores.shape[0] - 1) or abs(angle) >= MAX_SEARCH_ANGLE - 1e-9:
        return _fail(
            TILT_AT_LIMIT,
            f"倾角顶到了 ±{MAX_SEARCH_ANGLE:.0f}° 搜索边界，测不准。"
            "请先用“旋转 90°”把方向摆正，或用“标尺对齐”手动量。",
            width,
            raw=angle,
        )

    return TiltEstimate(
        angle_deg=angle,
        reason_code=TILT_OK,
        message=f"自动测得倾角 {angle:+.2f}°（相对水平）。这是建议值，请目视确认后再旋正。",
        crop_width_px=width,
        raw_angle_deg=angle,
    )

"""自动倾角估计：精度、角度符号约定、拒答路径。

符号约定是这里最要紧的东西——弄反了用户会朝反方向转。所以不靠"读起来对"，而是
用两条独立的链路钉死：（1）估计值要等于用户拿标尺沿条带量出来的
:func:`angle_to_horizontal`；（2）把估计值直接喂给 :func:`rotate_image`（界面
微调旋钮走的就是这条路）之后，图必须变平。
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from wb_analyzer.roi import RoiRect, angle_to_horizontal, rotate_image
from wb_analyzer.tilt import (
    MAX_SEARCH_ANGLE,
    MIN_CROP_HEIGHT_PX,
    MIN_CROP_WIDTH_PX,
    TILT_AT_LIMIT,
    TILT_NO_ROI,
    TILT_NO_SIGNAL,
    TILT_OK,
    TILT_TOO_NARROW,
    TILT_TOO_SHORT,
    TiltEstimate,
    estimate_tilt,
)

DEMO = Path(__file__).resolve().parents[1] / "pad4_001(1)_signal.tif"


def tilted_bands(
    tilt_deg: float,
    *,
    width: int = 900,
    height: int = 320,
    lanes: int = 6,
    noise: float = 0.0,
    smile: float = 0.0,
    contrast: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """一排深色条带（浅背景），整排按 ``tilt_deg`` 倾斜，右端低为正。

    ``smile`` 是微笑效应的弧高占图高的比例（中间比两端高）。
    """
    rng = np.random.default_rng(seed)
    image = np.full((height, width), 200.0)
    columns = np.arange(width)
    center_x = (width - 1) / 2.0
    rows = np.arange(height)[:, None]
    band_width = max(4, int(width / (lanes * 1.8)))
    gap = int((width - lanes * band_width) / (lanes + 1))
    sigma = height * 0.09 / 1.7
    for index in range(lanes):
        x0 = gap + index * (band_width + gap)
        x1 = min(width, x0 + band_width)
        span = columns[x0:x1]
        arc = smile * height * (1.0 - ((span - center_x) / center_x) ** 2)
        centers = height / 2.0 + (span - center_x) * np.tan(np.radians(tilt_deg)) - arc
        image[:, x0:x1] -= 150.0 * contrast * np.exp(
            -0.5 * ((rows - centers[None, :]) / sigma) ** 2
        )
    if noise:
        image += rng.normal(0.0, noise, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def whole(image: np.ndarray) -> RoiRect:
    """把整张数组当作"已经裁好的膜"——显式承认这不是整图。"""
    return RoiRect(0, 0, image.shape[1], image.shape[0])


def measured_band_tilt(image: np.ndarray) -> float:
    """不依赖被测模块，直接量图上条带的倾角：逐列取最暗行再拟合直线。"""
    gray = cv2.GaussianBlur(image.astype(np.float64), (0, 0), 2.0)
    depth = gray.mean(axis=0).max() - gray.mean(axis=0)
    columns = np.flatnonzero(depth > depth.max() * 0.5)
    rows = gray[:, columns].argmin(axis=0)
    slope = np.polyfit(columns.astype(float), rows.astype(float), 1)[0]
    return float(np.degrees(np.arctan(slope)))


def centred_roi(width: int, height: int, rotated: np.ndarray, angle_deg: float) -> RoiRect:
    """旋转后的画面里，最大的、不含填充角的居中矩形（再留 3% 余量）。"""
    radians = abs(np.radians(angle_deg))
    cos, sin = np.cos(radians), np.sin(radians)
    half_w, half_h = width / 2.0, height / 2.0
    scale = min(half_w / (half_w * cos + half_h * sin), half_h / (half_w * sin + half_h * cos))
    scale *= 0.97
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    return RoiRect(
        (rotated.shape[1] - new_w) // 2, (rotated.shape[0] - new_h) // 2, new_w, new_h
    )


# --------------------------------------------------------------- 角度符号约定


@pytest.mark.parametrize("tilt", [-2.5, 2.5])
def test_estimate_matches_ruler_convention(tilt: float) -> None:
    """估计值 == 用户沿条带拉一条标尺线量到的角度（含符号）。"""
    image = tilted_bands(tilt)
    estimate = estimate_tilt(image, roi=whole(image), dark_bands=True)
    assert estimate.reason_code == TILT_OK

    # 标尺：沿条带排从左端拉到右端。角度公式与画布 _finish_ruler 里的一模一样。
    center_x = (image.shape[1] - 1) / 2.0
    left_y = image.shape[0] / 2.0 - center_x * np.tan(np.radians(tilt))
    right_y = image.shape[0] / 2.0 + center_x * np.tan(np.radians(tilt))
    ruler = angle_to_horizontal(0.0, left_y, image.shape[1] - 1.0, right_y)

    assert ruler == pytest.approx(tilt, abs=1e-9)
    assert estimate.angle_deg == pytest.approx(ruler, abs=0.05)
    # 独立于生成公式再量一次图上真实内容的倾角
    assert measured_band_tilt(image) == pytest.approx(tilt, abs=0.15)


def test_positive_estimate_means_the_right_end_is_lower() -> None:
    """正角度 = 右端低。这条反了，用户就会往反方向转。"""
    down_right = tilted_bands(3.0)
    assert measured_band_tilt(down_right) > 1.0  # 右端确实更低（图像坐标 y 向下）
    assert estimate_tilt(down_right, roi=whole(down_right), dark_bands=True).angle_deg > 0

    up_right = np.flipud(down_right).copy()
    assert measured_band_tilt(up_right) < -1.0
    assert estimate_tilt(up_right, roi=whole(up_right), dark_bands=True).angle_deg < 0


@pytest.mark.parametrize("tilt", [-4.0, -1.2, 0.8, 3.5])
def test_rotating_by_the_estimate_flattens_the_blot(tilt: float) -> None:
    """把估计值原样交给 rotate_image（界面微调旋钮走的就是这条路）后图要变平。"""
    image = tilted_bands(tilt)
    estimate = estimate_tilt(image, roi=whole(image), dark_bands=True)
    assert estimate.angle_deg is not None

    straightened = rotate_image(image, estimate.angle_deg)
    residual = centred_roi(image.shape[1], image.shape[0], straightened, tilt)
    after = estimate_tilt(straightened, roi=residual, dark_bands=True)
    assert after.angle_deg == pytest.approx(0.0, abs=0.1)
    assert abs(measured_band_tilt(image)) > abs(measured_band_tilt(straightened))


# ------------------------------------------------------------------- 精度


@pytest.mark.parametrize(
    "tilt", [-6.0, -3.0, -1.5, -0.6, -0.2, 0.0, 0.2, 0.6, 1.5, 3.0, 6.0]
)
def test_recovers_known_tilt_on_synthetic_bands(tilt: float) -> None:
    image = tilted_bands(tilt)
    estimate = estimate_tilt(image, roi=whole(image), dark_bands=True)
    assert estimate.reason_code == TILT_OK
    assert estimate.angle_deg == pytest.approx(tilt, abs=0.5)  # 任务要求
    assert estimate.angle_deg == pytest.approx(tilt, abs=0.05)  # 实测量级，防回归


def test_small_tilt_is_not_snapped_to_zero() -> None:
    """双线性重采样在 0° 处不插值，会凭空多出一个锐度尖峰。

    亚像素抖动就是为了压掉它；这条测试守住"0.35° 不会被报成 0.00°"。
    """
    for tilt in (-0.35, 0.35):
        image = tilted_bands(tilt)
        estimate = estimate_tilt(image, roi=whole(image), dark_bands=True)
        assert estimate.angle_deg is not None
        assert abs(estimate.angle_deg) > 0.2
        assert estimate.angle_deg == pytest.approx(tilt, abs=0.05)


@pytest.mark.parametrize(
    ("label", "kwargs", "tolerance"),
    [
        ("高斯噪声", {"noise": 6.0}, 0.1),
        ("微笑 3%", {"smile": 0.03}, 0.1),
        ("微笑 6%", {"smile": 0.06}, 0.2),
        ("低对比 5%", {"contrast": 0.05, "noise": 2.0}, 0.5),
    ],
)
def test_robust_to_noise_smile_and_low_contrast(
    label: str, kwargs: dict, tolerance: float
) -> None:
    for tilt in (-2.0, 0.7):
        for seed in range(2):
            image = tilted_bands(tilt, seed=seed, **kwargs)
            estimate = estimate_tilt(image, roi=whole(image), dark_bands=True)
            assert estimate.angle_deg == pytest.approx(tilt, abs=tolerance), label


def test_only_the_roi_is_measured() -> None:
    """ROI 外面画一条角度完全不同的强直线（扫描边框那类），不许影响结果。"""
    bands = tilted_bands(1.0)
    below = np.full((200, bands.shape[1]), 200, dtype=np.uint8)
    cv2.line(below, (0, 180), (below.shape[1] - 1, 60), 0, 9)  # 约 -7.6°
    image = np.vstack((bands, below))

    inside = estimate_tilt(image, roi=whole(bands), dark_bands=True)
    assert inside.angle_deg == pytest.approx(1.0, abs=0.1)
    # 对照：把那条线圈进来，结论立刻被带跑——所以 ROI 必须框在条带上
    dragged = estimate_tilt(image, roi=whole(image), dark_bands=True)
    assert dragged.angle_deg is not None
    assert abs(dragged.angle_deg - 1.0) > 0.1


# ------------------------------------------------------- 挡不住的失败（有意留档）


def test_a_faint_full_width_scratch_defeats_the_estimate_silently() -> None:
    """已知且挡不住的失败：贯穿全宽的淡划痕会把结论整个带跑，而且不会被拒答。

    三条工程约束一条都拦不住它（ROI 有、够宽、没顶到边界），算法也自省不出来。
    这条测试就是为了让这个事实一直摆在明面上——界面因此**绝不能自动应用**结果。
    """
    image = tilted_bands(1.0).astype(np.float64)
    height, width = image.shape
    offset = int(width * 0.5 * np.tan(np.radians(6.0)) / 2)
    scratch = np.zeros_like(image)
    cv2.line(scratch, (0, height // 2 - offset), (width - 1, height // 2 + offset), 1.0, 3)
    scratch = cv2.GaussianBlur(scratch, (0, 0), 1.5)  # -6° 的一道淡痕
    scratched = np.clip(image - 255 * 0.2 * scratch, 0, 255).astype(np.uint8)

    estimate = estimate_tilt(scratched, roi=whole(scratched), dark_bands=True)
    assert estimate.reason_code == TILT_OK  # 自信满满
    assert abs(estimate.angle_deg - 1.0) > 1.0  # 却是错的


def test_a_hard_edge_inside_the_roi_wins_over_the_bands() -> None:
    """框到膜边 / 胶带上时，那条直边会盖过条带——4% 内缩只能挡住贴边的一点点。"""
    image = tilted_bands(2.0)
    cv2.line(image, (0, 60), (image.shape[1] - 1, 60), 0, 5)  # ROI 内一条水平硬边
    estimate = estimate_tilt(image, roi=whole(image), dark_bands=True)
    assert estimate.reason_code == TILT_OK
    assert abs(estimate.angle_deg) < 0.5  # 报的是那条边的角度，不是条带的 2°


# ------------------------------------------------------------------- 拒答


def test_missing_roi_is_rejected() -> None:
    """整图上算会锁到扫描边框上，所以没有 ROI 一律拒答。"""
    image = tilted_bands(1.5)
    estimate = estimate_tilt(image, dark_bands=True)
    assert estimate.reason_code == TILT_NO_ROI
    assert estimate.angle_deg is None
    assert estimate.trusted is False
    assert estimate_tilt(image, roi=None, dark_bands=True).reason_code == TILT_NO_ROI


def test_narrow_crop_is_rejected_but_wide_enough_is_not() -> None:
    image = tilted_bands(1.5, width=1200, height=320)
    narrow = RoiRect(0, 0, MIN_CROP_WIDTH_PX - 1, 320)
    rejected = estimate_tilt(image, roi=narrow, dark_bands=True)
    assert rejected.reason_code == TILT_TOO_NARROW
    assert rejected.angle_deg is None
    assert rejected.crop_width_px == MIN_CROP_WIDTH_PX - 1
    assert str(MIN_CROP_WIDTH_PX) in rejected.message

    wide = RoiRect(0, 0, MIN_CROP_WIDTH_PX, 320)
    assert estimate_tilt(image, roi=wide, dark_bands=True).reason_code == TILT_OK


def test_narrow_roi_is_rejected_after_clipping_to_the_image() -> None:
    """ROI 大半悬在图外时，按裁剪后的实际宽度判，不按用户画的宽度。"""
    image = tilted_bands(1.5)
    hanging = RoiRect(image.shape[1] - 100, 0, 600, 320)
    estimate = estimate_tilt(image, roi=hanging, dark_bands=True)
    assert estimate.reason_code == TILT_TOO_NARROW
    assert estimate.crop_width_px == 100


def test_short_crop_is_rejected() -> None:
    image = tilted_bands(1.5)
    short = RoiRect(0, 100, 900, MIN_CROP_HEIGHT_PX - 1)
    estimate = estimate_tilt(image, roi=short, dark_bands=True)
    assert estimate.reason_code == TILT_TOO_SHORT
    assert estimate.angle_deg is None


def test_angle_at_the_search_limit_is_rejected_not_clamped() -> None:
    """顶到 ±10° 时不能把边界值当答案返回——那是"测不准"，不是"歪了 10°"。"""
    image = tilted_bands(13.0, height=520)
    estimate = estimate_tilt(image, roi=whole(image), dark_bands=True)
    assert estimate.reason_code == TILT_AT_LIMIT
    assert estimate.angle_deg is None
    assert estimate.trusted is False
    assert estimate.raw_angle_deg is not None
    assert abs(estimate.raw_angle_deg) == pytest.approx(MAX_SEARCH_ANGLE, abs=0.6)


@pytest.mark.parametrize("value", [0, 7, 128, 255])
def test_blank_crop_is_rejected(value: int) -> None:
    """全常数的 ROI 必须拒答。

    降采样是加权平均，float32 舍入会给常数图留下 ~1e-6 的假跨度，归一化再把它
    放大成满量程噪声——空白 ROI 于是能算出一个"可采信"的角度。所以动态范围要在
    缩放之前判。7 和 128 走的正是不同的舍入路径，两个都要盯住。
    """
    flat = np.full((320, 900), value, dtype=np.uint8)
    estimate = estimate_tilt(flat, roi=whole(flat), dark_bands=True)
    assert estimate.reason_code == TILT_NO_SIGNAL
    assert estimate.angle_deg is None

    wide = np.full((320, 900), value * 257, dtype=np.uint16)
    assert estimate_tilt(wide, roi=whole(wide), dark_bands=True).reason_code == TILT_NO_SIGNAL


def test_every_rejection_says_what_to_do_instead() -> None:
    """拒答文案要能操作，不能只丢一个布尔值。"""
    image = tilted_bands(1.5)
    flat = np.full((320, 900), 7, dtype=np.uint8)
    steep = tilted_bands(13.0, height=520)
    cases = [
        estimate_tilt(image, dark_bands=True),
        estimate_tilt(image, roi=RoiRect(0, 0, 120, 320), dark_bands=True),
        estimate_tilt(image, roi=RoiRect(0, 0, 900, 10), dark_bands=True),
        estimate_tilt(steep, roi=whole(steep), dark_bands=True),
        estimate_tilt(flat, roi=whole(flat), dark_bands=True),
    ]
    assert {case.reason_code for case in cases} == {
        TILT_NO_ROI,
        TILT_TOO_NARROW,
        TILT_TOO_SHORT,
        TILT_AT_LIMIT,
        TILT_NO_SIGNAL,
    }
    for case in cases:
        assert case.angle_deg is None
        assert case.trusted is False
        assert "测不准" in case.message or "整图上算" in case.message
        assert "标尺" in case.message  # 指向手动兜底


def test_success_message_is_advisory_not_a_verdict() -> None:
    image = tilted_bands(2.0)
    estimate = estimate_tilt(image, roi=whole(image), dark_bands=True)
    assert estimate.trusted is True
    assert "建议" in estimate.message
    assert "+2.0" in estimate.message


# --------------------------------------------------------------- 输入形态


def test_dark_and_light_bands_give_the_same_answer() -> None:
    """平方梯度对明暗极性不变，dark_bands 只是个占位参数。"""
    dark = tilted_bands(1.7)
    light = (255 - dark).astype(np.uint8)
    first = estimate_tilt(dark, roi=whole(dark), dark_bands=True)
    second = estimate_tilt(light, roi=whole(light), dark_bands=False)
    assert first.angle_deg == pytest.approx(second.angle_deg, abs=1e-4)
    assert estimate_tilt(dark, roi=whole(dark)).angle_deg == pytest.approx(
        first.angle_deg, abs=1e-9
    )


def test_accepts_uint16_bgr_and_float_inputs() -> None:
    base = tilted_bands(2.2)
    expected = estimate_tilt(base, roi=whole(base), dark_bands=True).angle_deg
    assert expected is not None

    wide_range = (base.astype(np.uint16) * 257).astype(np.uint16)
    colour = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    floats = base.astype(np.float32) / 255.0
    for variant in (wide_range, colour, floats):
        estimate = estimate_tilt(variant, roi=whole(base), dark_bands=True)
        assert estimate.angle_deg == pytest.approx(expected, abs=0.05)


def test_non_finite_pixels_do_not_crash() -> None:
    image = tilted_bands(1.4).astype(np.float32)
    image[10, 10] = np.nan
    image[20, 20] = np.inf
    estimate = estimate_tilt(image, roi=whole(image), dark_bands=True)
    assert estimate.reason_code == TILT_OK
    assert estimate.angle_deg == pytest.approx(1.4, abs=0.2)


def test_tiny_and_degenerate_inputs_are_handled() -> None:
    for image in (
        np.zeros((1, 1), dtype=np.uint8),
        np.zeros((4, 900), dtype=np.uint8),
        np.zeros((300, 900), dtype=np.uint8),
    ):
        estimate = estimate_tilt(image, roi=whole(image), dark_bands=True)
        assert isinstance(estimate, TiltEstimate)
        assert estimate.angle_deg is None


# ------------------------------------------------------------------- 真实图


@pytest.mark.skipif(not DEMO.exists(), reason="demo blot image not present")
def test_demo_blot_tilt_round_trip() -> None:
    """真实样图：人为旋转已知角度后回推，误差应在 0.15° 量级。"""
    from wb_analyzer.roi import load_bgr, to_gray

    gray = to_gray(load_bgr(DEMO))
    crop = gray[360:500, 400:1020]  # 条带排所在的膜区
    height, width = crop.shape
    base = estimate_tilt(crop, roi=whole(crop), dark_bands=True)
    assert base.reason_code == TILT_OK

    errors = []
    for applied in (-4.0, -2.0, -0.5, 0.5, 2.0, 4.0):
        # 反向转 applied，图上倾角就多出 +applied
        rotated = rotate_image(crop, -applied)
        estimate = estimate_tilt(
            rotated, roi=centred_roi(width, height, rotated, applied), dark_bands=True
        )
        assert estimate.reason_code == TILT_OK
        errors.append(abs(estimate.angle_deg - (base.angle_deg + applied)))
    assert max(errors) < 0.3
    assert float(np.median(errors)) < 0.15


@pytest.mark.skipif(not DEMO.exists(), reason="demo blot image not present")
def test_demo_blot_estimate_is_stable_across_roi_choices() -> None:
    """用户框得松一点紧一点，结论不该跟着跳。"""
    from wb_analyzer.roi import load_bgr, to_gray

    gray = to_gray(load_bgr(DEMO))
    angles = []
    for roi in (
        RoiRect(430, 400, 560, 60),
        RoiRect(420, 380, 580, 100),
        RoiRect(400, 360, 620, 140),
    ):
        estimate = estimate_tilt(gray, roi=roi, dark_bands=True)
        assert estimate.reason_code == TILT_OK
        angles.append(estimate.angle_deg)
    assert max(angles) - min(angles) < 0.1


# ------------------------------------------------------------------- 耗时


def test_estimate_is_fast_enough() -> None:
    """目标 10 ms 量级；阈值放宽到 250 ms，只拦灾难性回归（CI 机器慢很多）。"""
    image = tilted_bands(1.5)
    roi = whole(image)
    estimate_tilt(image, roi=roi, dark_bands=True)  # 预热：OpenCV 线程池 / 缓存

    timings = []
    for _ in range(3):
        start = time.perf_counter()
        estimate_tilt(image, roi=roi, dark_bands=True)
        timings.append(time.perf_counter() - start)
    best = min(timings)
    assert best < 0.25, f"倾角估计耗时 {best * 1000:.1f} ms"

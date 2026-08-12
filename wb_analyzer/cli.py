"""Command-line interface for unattended analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import AnalysisConfig, analyze_image
from .export import export_result
from .normalize import ORIGIN_LOCAL_BACKGROUND


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="自动识别并统计 Western Blot 条带灰度")
    parser.add_argument("image", type=Path, help="待分析图片")
    parser.add_argument("--output", "-o", type=Path, default=Path("results"), help="结果目录")
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=1.0,
        help="内部阈值强度：越小越容易检出弱条带（默认 1.0）",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=None,
        help="背景平滑尺度（奇数像素，一般不用改；省略=自动）",
    )
    parser.add_argument(
        "--background",
        choices=("auto", "dark_bands", "light_bands"),
        default="auto",
        help="条带颜色：auto / dark_bands（深带浅底）/ light_bands（浅带深底）",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = AnalysisConfig(
        sensitivity=args.sensitivity,
        kernel_size=args.kernel_size,
        background_mode=args.background,
    )
    result = analyze_image(args.image, config=config)
    # analyze_image detects on the 8-bit layer and re-measures with the ring
    # median (quant.remeasure_native), so this whole path is the local
    # background convention — never the ROI baseline one.
    csv_path, preview_path = export_result(
        result, args.image, args.output, origin=ORIGIN_LOCAL_BACKGROUND
    )
    print(f"检测到 {len(result.measurements)} 个条带")
    if result.saturated_band_count:
        print(f"警告：{result.saturated_band_count} 个条带可能存在曝光饱和")
    print(f"数据：{csv_path}\n预览：{preview_path}")


if __name__ == "__main__":
    main()

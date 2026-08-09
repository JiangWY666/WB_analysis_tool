"""Command-line interface for unattended analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import analyze_image
from .export import export_result


def main() -> None:
    parser = argparse.ArgumentParser(description="自动识别并统计 Western Blot 条带灰度")
    parser.add_argument("image", type=Path, help="待分析图片")
    parser.add_argument("--output", "-o", type=Path, default=Path("results"), help="结果目录")
    args = parser.parse_args()
    result = analyze_image(args.image)
    csv_path, preview_path = export_result(result, args.image, args.output)
    print(f"检测到 {len(result.measurements)} 个条带")
    print(f"数据：{csv_path}\n预览：{preview_path}")


if __name__ == "__main__":
    main()

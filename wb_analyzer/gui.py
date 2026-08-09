"""Tkinter desktop application."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .analysis import analyze_image
from .export import export_result


class AnalyzerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("WB 灰度统计工具")
        self.geometry("920x680")
        self.minsize(720, 520)
        self.image_path: Path | None = None
        self.preview: ImageTk.PhotoImage | None = None

        toolbar = ttk.Frame(self, padding=12)
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="选择 WB 图片", command=self.choose_image).pack(side=tk.LEFT)
        self.run_button = ttk.Button(toolbar, text="自动分析并导出", command=self.run_analysis, state=tk.DISABLED)
        self.run_button.pack(side=tk.LEFT, padx=8)
        self.status = ttk.Label(toolbar, text="请选择一张图片")
        self.status.pack(side=tk.LEFT, padx=12)
        self.canvas = ttk.Label(self, anchor=tk.CENTER)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

    def choose_image(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择 Western Blot 图片",
            filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("所有文件", "*.*")],
        )
        if selected:
            self.image_path = Path(selected)
            self.status.config(text=self.image_path.name)
            self.run_button.config(state=tk.NORMAL)
            self._show(Image.open(self.image_path).convert("RGB"))

    def _show(self, image: Image.Image) -> None:
        image.thumbnail((860, 560), Image.Resampling.LANCZOS)
        self.preview = ImageTk.PhotoImage(image)
        self.canvas.config(image=self.preview)

    def run_analysis(self) -> None:
        if not self.image_path:
            return
        output_dir = filedialog.askdirectory(title="选择结果保存目录", initialdir=str(self.image_path.parent))
        if not output_dir:
            return
        self.status.config(text="正在分析……")
        self.update_idletasks()
        try:
            result = analyze_image(self.image_path)
            csv_path, preview_path = export_result(result, self.image_path, output_dir)
            self._show(Image.open(preview_path).convert("RGB"))
            self.status.config(text=f"完成：检测到 {len(result.measurements)} 个条带")
            messagebox.showinfo("分析完成", f"检测到 {len(result.measurements)} 个条带。\n\nCSV：{csv_path}\n预览：{preview_path}")
        except (ValueError, OSError) as exc:
            self.status.config(text="分析失败")
            messagebox.showerror("无法分析", str(exc))


def main() -> None:
    AnalyzerApp().mainloop()


if __name__ == "__main__":
    main()

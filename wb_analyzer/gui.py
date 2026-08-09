"""Tkinter desktop application."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .analysis import (
    AnalysisConfig,
    AnalysisResult,
    BandMeasurement,
    analyze_image,
    rebuild_result,
)
from .export import export_result


class AnalyzerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("WB 灰度统计工具")
        self.geometry("1100x720")
        self.minsize(900, 600)

        self.image_path: Path | None = None
        self.result: AnalysisResult | None = None
        self.output_dir: Path | None = None
        self.preview: ImageTk.PhotoImage | None = None
        self.display_scale = 1.0
        self.draw_start: tuple[int, int] | None = None
        self.draw_rect: int | None = None
        self.drawing_enabled = tk.BooleanVar(value=False)

        self.sensitivity = tk.DoubleVar(value=1.0)
        self.kernel_mode = tk.StringVar(value="自动")
        self.kernel_size = tk.IntVar(value=31)
        self.background_mode = tk.StringVar(value="auto")

        self._build_layout()

    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(root)
        toolbar.grid(row=0, column=0, sticky="ew")
        ttk.Button(toolbar, text="选择 WB 图片", command=self.choose_image).pack(side=tk.LEFT)
        self.run_button = ttk.Button(toolbar, text="自动分析", command=self.run_analysis, state=tk.DISABLED)
        self.run_button.pack(side=tk.LEFT, padx=6)
        self.export_button = ttk.Button(toolbar, text="导出结果", command=self.export_current, state=tk.DISABLED)
        self.export_button.pack(side=tk.LEFT)
        self.status = ttk.Label(toolbar, text="请选择一张图片")
        self.status.pack(side=tk.LEFT, padx=12)

        body = ttk.Frame(root)
        body.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        preview_frame = ttk.LabelFrame(body, text="预览（可勾选后拖拽补画条带）", padding=6)
        preview_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(preview_frame, background="#1f1f1f", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>", self._on_draw_start)
        self.canvas.bind("<B1-Motion>", self._on_draw_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_draw_end)

        side = ttk.Frame(body)
        side.grid(row=0, column=1, sticky="nsew")
        side.rowconfigure(1, weight=1)
        side.columnconfigure(0, weight=1)

        params = ttk.LabelFrame(side, text="检测参数", padding=8)
        params.grid(row=0, column=0, sticky="ew")
        ttk.Label(params, text="灵敏度（越小越敏感）").grid(row=0, column=0, sticky="w")
        ttk.Scale(params, from_=0.5, to=1.8, variable=self.sensitivity, orient=tk.HORIZONTAL).grid(
            row=0, column=1, sticky="ew", padx=6
        )
        self.sensitivity_label = ttk.Label(params, text="1.00")
        self.sensitivity_label.grid(row=0, column=2, sticky="e")
        self.sensitivity.trace_add("write", self._sync_sensitivity_label)

        ttk.Label(params, text="形态学核").grid(row=1, column=0, sticky="w", pady=(8, 0))
        kernel_row = ttk.Frame(params)
        kernel_row.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Combobox(
            kernel_row,
            textvariable=self.kernel_mode,
            values=("自动", "手动"),
            width=6,
            state="readonly",
        ).pack(side=tk.LEFT)
        self.kernel_spin = ttk.Spinbox(kernel_row, from_=11, to=101, increment=2, textvariable=self.kernel_size, width=6)
        self.kernel_spin.pack(side=tk.LEFT, padx=6)

        ttk.Label(params, text="背景模式").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            params,
            textvariable=self.background_mode,
            values=("auto", "dark_bands", "light_bands"),
            state="readonly",
            width=14,
        ).grid(row=2, column=1, columnspan=2, sticky="w", pady=(8, 0))
        params.columnconfigure(1, weight=1)

        bands = ttk.LabelFrame(side, text="条带列表 / 手动修正", padding=8)
        bands.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        bands.rowconfigure(0, weight=1)
        bands.columnconfigure(0, weight=1)
        self.band_list = tk.Listbox(bands, exportselection=False)
        self.band_list.grid(row=0, column=0, sticky="nsew")
        actions = ttk.Frame(bands)
        actions.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Checkbutton(actions, text="拖拽画框添加", variable=self.drawing_enabled).pack(side=tk.LEFT)
        ttk.Button(actions, text="删除选中", command=self.delete_selected).pack(side=tk.RIGHT)

        tip = ttk.Label(
            side,
            text="流程：调参 → 自动分析 → 删除误检/画框补检 → 导出。\n橙色框表示可能饱和。",
            wraplength=320,
            justify=tk.LEFT,
        )
        tip.grid(row=2, column=0, sticky="ew", pady=(8, 0))

    def _sync_sensitivity_label(self, *_args: object) -> None:
        self.sensitivity_label.config(text=f"{float(self.sensitivity.get()):.2f}")

    def _current_config(self) -> AnalysisConfig:
        kernel = None if self.kernel_mode.get() == "自动" else int(self.kernel_size.get())
        return AnalysisConfig(
            sensitivity=float(self.sensitivity.get()),
            kernel_size=kernel,
            background_mode=self.background_mode.get(),  # type: ignore[arg-type]
        )

    def choose_image(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择 Western Blot 图片",
            filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("所有文件", "*.*")],
        )
        if not selected:
            return
        self.image_path = Path(selected)
        self.result = None
        self.output_dir = None
        self.band_list.delete(0, tk.END)
        self.export_button.config(state=tk.DISABLED)
        self.status.config(text=self.image_path.name)
        self.run_button.config(state=tk.NORMAL)
        self._show_pil(Image.open(self.image_path).convert("RGB"))

    def _show_array(self, bgr_image) -> None:
        rgb = Image.fromarray(bgr_image[:, :, ::-1].copy())
        self._show_pil(rgb)

    def _show_pil(self, image: Image.Image) -> None:
        self.canvas.update_idletasks()
        max_w = max(200, self.canvas.winfo_width() or 640)
        max_h = max(200, self.canvas.winfo_height() or 480)
        fitted = image.copy()
        fitted.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        self.display_scale = fitted.width / image.width if image.width else 1.0
        self.preview = ImageTk.PhotoImage(fitted)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.preview, tags=("image",))
        self.canvas.config(scrollregion=(0, 0, fitted.width, fitted.height))

    def _refresh_from_result(self) -> None:
        if not self.result:
            return
        self._show_array(self.result.annotated_image)
        self.band_list.delete(0, tk.END)
        for item in self.result.measurements:
            flag = " ⚠饱和" if item.saturated_fraction >= 0.05 else ""
            self.band_list.insert(
                tk.END,
                f"L{item.lane}-B{item.band}  dens={item.integrated_density:.1f}{flag}",
            )
        self.export_button.config(state=tk.NORMAL)

    def run_analysis(self) -> None:
        if not self.image_path:
            return
        self.status.config(text="正在分析……")
        self.update_idletasks()
        try:
            self.result = analyze_image(self.image_path, config=self._current_config())
            self._refresh_from_result()
            msg = f"完成：检测到 {len(self.result.measurements)} 个条带"
            if self.result.saturated_band_count:
                msg += f"（{self.result.saturated_band_count} 个可能饱和）"
            self.status.config(text=msg)
        except (ValueError, OSError) as exc:
            self.status.config(text="分析失败")
            messagebox.showerror("无法分析", str(exc))

    def delete_selected(self) -> None:
        if not self.result or self.result.gray_image is None:
            return
        selection = list(self.band_list.curselection())
        if not selection:
            return
        kept = [m for idx, m in enumerate(self.result.measurements) if idx not in set(selection)]
        self.result = rebuild_result(
            self.result.gray_image,
            kept,
            dark_bands=self.result.dark_bands,
        )
        self._refresh_from_result()
        self.status.config(text=f"已删除选中条带，当前 {len(self.result.measurements)} 个")

    def _canvas_to_image(self, x: int, y: int) -> tuple[int, int]:
        if self.display_scale <= 0:
            return x, y
        return int(round(x / self.display_scale)), int(round(y / self.display_scale))

    def _on_draw_start(self, event: tk.Event) -> None:
        if not self.drawing_enabled.get() or not self.result:
            return
        self.draw_start = (event.x, event.y)
        if self.draw_rect is not None:
            self.canvas.delete(self.draw_rect)
        self.draw_rect = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="#00d4ff", width=2)

    def _on_draw_move(self, event: tk.Event) -> None:
        if not self.draw_start or self.draw_rect is None:
            return
        self.canvas.coords(self.draw_rect, self.draw_start[0], self.draw_start[1], event.x, event.y)

    def _on_draw_end(self, event: tk.Event) -> None:
        if not self.draw_start or not self.result or self.result.gray_image is None:
            self.draw_start = None
            return
        x0, y0 = self.draw_start
        self.draw_start = None
        if self.draw_rect is not None:
            self.canvas.delete(self.draw_rect)
            self.draw_rect = None
        ix0, iy0 = self._canvas_to_image(x0, y0)
        ix1, iy1 = self._canvas_to_image(event.x, event.y)
        x1, x2 = sorted((ix0, ix1))
        y1, y2 = sorted((iy0, iy1))
        if x2 - x1 < 3 or y2 - y1 < 3:
            return
        added = BandMeasurement(0, 0, x1, y1, x2 - x1, y2 - y1, 0, 0.0, 0.0, 0.0)
        self.result = rebuild_result(
            self.result.gray_image,
            [*self.result.measurements, added],
            dark_bands=self.result.dark_bands,
        )
        self._refresh_from_result()
        self.status.config(text=f"已添加手动条带，当前 {len(self.result.measurements)} 个")

    def export_current(self) -> None:
        if not self.image_path or not self.result:
            return
        initial = str(self.output_dir or self.image_path.parent)
        selected = filedialog.askdirectory(title="选择结果保存目录", initialdir=initial)
        if not selected:
            return
        self.output_dir = Path(selected)
        try:
            csv_path, preview_path = export_result(self.result, self.image_path, self.output_dir)
            warn = ""
            if self.result.saturated_band_count:
                warn = f"\n注意：{self.result.saturated_band_count} 个条带可能曝光饱和。"
            messagebox.showinfo("导出完成", f"CSV：{csv_path}\n预览：{preview_path}{warn}")
            self.status.config(text=f"已导出到 {self.output_dir}")
        except OSError as exc:
            messagebox.showerror("导出失败", str(exc))


def main() -> None:
    AnalyzerApp().mainloop()


if __name__ == "__main__":
    main()

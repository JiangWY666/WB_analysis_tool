"""Tkinter desktop application."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from .analysis import (
    AnalysisConfig,
    AnalysisResult,
    BandMeasurement,
    analyze_array,
    rebuild_result,
)
from .export import export_result
from .normalize import (
    AnalysisSession,
    QuantEntry,
    export_normalized_csv,
    lane_intensities_from_measurements,
)
from .roi import (
    RoiRect,
    default_separators,
    load_bgr,
    quantify_roi_lanes,
    roi_intensity_profile,
    rotate_image,
    to_gray,
)


class ProfileDialog(tk.Toplevel):
    """Simple intensity-profile QC window drawn with Tk canvas."""

    def __init__(self, master: tk.Misc, profile: np.ndarray, separators_rel: list[float], title: str = "强度剖面") -> None:
        super().__init__(master)
        self.title(title)
        self.geometry("720x320")
        self.minsize(480, 240)
        canvas = tk.Canvas(self, background="#111111", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self._canvas = canvas
        self._profile = np.asarray(profile, dtype=float)
        self._separators = separators_rel
        canvas.bind("<Configure>", self._redraw)
        self._redraw()

    def _redraw(self, *_args: object) -> None:
        canvas = self._canvas
        canvas.delete("all")
        width = max(40, canvas.winfo_width())
        height = max(40, canvas.winfo_height())
        pad = 28
        plot_w = width - 2 * pad
        plot_h = height - 2 * pad
        profile = self._profile
        if profile.size < 2 or plot_w < 10 or plot_h < 10:
            return
        ymin = float(profile.min())
        ymax = float(profile.max())
        if abs(ymax - ymin) < 1e-6:
            ymax = ymin + 1.0
        canvas.create_rectangle(pad, pad, pad + plot_w, pad + plot_h, outline="#666666")
        points: list[float] = []
        for index, value in enumerate(profile):
            x = pad + plot_w * index / (len(profile) - 1)
            y = pad + plot_h * (1.0 - (value - ymin) / (ymax - ymin))
            points.extend((x, y))
        canvas.create_line(*points, fill="#4fc3f7", width=2)
        for sep in self._separators:
            if 0 <= sep <= len(profile) - 1:
                x = pad + plot_w * sep / (len(profile) - 1)
                canvas.create_line(x, pad, x, pad + plot_h, fill="#ffb74d", dash=(4, 3))
        canvas.create_text(pad, 12, anchor=tk.W, fill="#dddddd", text="信号强度（越高越强）")
        canvas.create_text(width - pad, height - 10, anchor=tk.E, fill="#aaaaaa", text="ROI 横向位置 →")


class AnalyzerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("WB 灰度统计工具")
        self.geometry("1280x780")
        self.minsize(1000, 640)

        self.image_path: Path | None = None
        self.original_bgr: np.ndarray | None = None
        self.working_bgr: np.ndarray | None = None
        self.result: AnalysisResult | None = None
        self.output_dir: Path | None = None
        self.preview: ImageTk.PhotoImage | None = None
        self.display_scale = 1.0
        self.base_image_rgb: Image.Image | None = None

        self.roi: RoiRect | None = None
        self.separators: list[float] = []
        self.draw_start: tuple[int, int] | None = None
        self.draw_rect: int | None = None
        self.active_separator: int | None = None
        self.session = AnalysisSession()

        self.mode = tk.StringVar(value="auto")
        self.drawing_band = tk.BooleanVar(value=False)
        self.rotation = tk.DoubleVar(value=0.0)
        self.sensitivity = tk.DoubleVar(value=1.0)
        self.kernel_mode = tk.StringVar(value="自动")
        self.kernel_size = tk.IntVar(value=31)
        self.background_mode = tk.StringVar(value="auto")
        self.lane_count = tk.IntVar(value=3)
        self.entry_kind = tk.StringVar(value="loading_control")
        self.protein_name = tk.StringVar(value="Actin")
        self.group_name = tk.StringVar(value="默认")

        self._build_layout()
        self.rotation.trace_add("write", self._on_rotation_changed)
        self.lane_count.trace_add("write", self._on_lane_count_changed)
        self.mode.trace_add("write", lambda *_: self._redraw_overlays())

    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(root)
        toolbar.grid(row=0, column=0, sticky="ew")
        ttk.Button(toolbar, text="选择图片", command=self.choose_image).pack(side=tk.LEFT)
        self.run_button = ttk.Button(toolbar, text="自动分析", command=self.run_auto_analysis, state=tk.DISABLED)
        self.run_button.pack(side=tk.LEFT, padx=4)
        self.roi_button = ttk.Button(toolbar, text="ROI 定量", command=self.run_roi_quantify, state=tk.DISABLED)
        self.roi_button.pack(side=tk.LEFT, padx=4)
        self.profile_button = ttk.Button(toolbar, text="强度剖面", command=self.show_profile, state=tk.DISABLED)
        self.profile_button.pack(side=tk.LEFT, padx=4)
        self.export_button = ttk.Button(toolbar, text="导出条带", command=self.export_current, state=tk.DISABLED)
        self.export_button.pack(side=tk.LEFT, padx=4)
        self.status = ttk.Label(toolbar, text="请选择一张图片")
        self.status.pack(side=tk.LEFT, padx=10)

        body = ttk.Frame(root)
        body.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        preview = ttk.LabelFrame(body, text="预览：旋转后可自动检测，或框选 ROI 并拖拽橙色分隔线", padding=6)
        preview.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        preview.rowconfigure(0, weight=1)
        preview.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(preview, background="#1f1f1f", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        side = ttk.Notebook(body)
        side.grid(row=0, column=1, sticky="nsew")

        prep = ttk.Frame(side, padding=8)
        roi_tab = ttk.Frame(side, padding=8)
        norm_tab = ttk.Frame(side, padding=8)
        band_tab = ttk.Frame(side, padding=8)
        side.add(prep, text="预处理/参数")
        side.add(roi_tab, text="ROI 泳道")
        side.add(norm_tab, text="内参归一化")
        side.add(band_tab, text="条带列表")

        ttk.Label(prep, text="工作模式").grid(row=0, column=0, sticky="w")
        mode_row = ttk.Frame(prep)
        mode_row.grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(mode_row, text="自动检测", variable=self.mode, value="auto").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_row, text="ROI 泳道", variable=self.mode, value="roi").pack(side=tk.LEFT, padx=8)

        ttk.Label(prep, text="旋转角度").grid(row=1, column=0, sticky="w", pady=(10, 0))
        rot = ttk.Frame(prep)
        rot.grid(row=1, column=1, sticky="ew", pady=(10, 0))
        ttk.Scale(rot, from_=-20, to=20, variable=self.rotation, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.rotation_label = ttk.Label(rot, text="0.0°", width=6)
        self.rotation_label.pack(side=tk.LEFT, padx=4)
        ttk.Button(rot, text="归零", command=lambda: self.rotation.set(0.0)).pack(side=tk.LEFT)

        ttk.Label(prep, text="灵敏度").grid(row=2, column=0, sticky="w", pady=(10, 0))
        sens = ttk.Frame(prep)
        sens.grid(row=2, column=1, sticky="ew", pady=(10, 0))
        ttk.Scale(sens, from_=0.5, to=1.8, variable=self.sensitivity, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.sensitivity_label = ttk.Label(sens, text="1.00", width=6)
        self.sensitivity_label.pack(side=tk.LEFT)
        self.sensitivity.trace_add("write", lambda *_: self.sensitivity_label.config(text=f"{float(self.sensitivity.get()):.2f}"))

        ttk.Label(prep, text="形态学核").grid(row=3, column=0, sticky="w", pady=(10, 0))
        kernel_row = ttk.Frame(prep)
        kernel_row.grid(row=3, column=1, sticky="w", pady=(10, 0))
        ttk.Combobox(kernel_row, textvariable=self.kernel_mode, values=("自动", "手动"), width=6, state="readonly").pack(side=tk.LEFT)
        ttk.Spinbox(kernel_row, from_=11, to=101, increment=2, textvariable=self.kernel_size, width=6).pack(side=tk.LEFT, padx=6)

        ttk.Label(prep, text="背景模式").grid(row=4, column=0, sticky="w", pady=(10, 0))
        ttk.Combobox(
            prep,
            textvariable=self.background_mode,
            values=("auto", "dark_bands", "light_bands"),
            state="readonly",
            width=14,
        ).grid(row=4, column=1, sticky="w", pady=(10, 0))
        prep.columnconfigure(1, weight=1)
        ttk.Label(
            prep,
            text="建议：先微调旋转使泳道垂直，再自动检测或框选 ROI。",
            wraplength=340,
            justify=tk.LEFT,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(12, 0))

        ttk.Label(roi_tab, text="泳道数（重复数）").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(roi_tab, from_=1, to=24, textvariable=self.lane_count, width=6).grid(row=0, column=1, sticky="w")
        ttk.Button(roi_tab, text="按等分重置分隔线", command=self.reset_separators).grid(row=1, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Button(roi_tab, text="清除 ROI", command=self.clear_roi).grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Label(
            roi_tab,
            text="在“ROI 泳道”模式下，拖拽画出青色 ROI；橙色竖线可左右拖动对齐泳道。然后点“ROI 定量”或“强度剖面”。",
            wraplength=340,
            justify=tk.LEFT,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(12, 0))
        roi_tab.columnconfigure(1, weight=1)

        ttk.Label(norm_tab, text="类型").grid(row=0, column=0, sticky="w")
        kind_row = ttk.Frame(norm_tab)
        kind_row.grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(kind_row, text="内参 LC", variable=self.entry_kind, value="loading_control").pack(side=tk.LEFT)
        ttk.Radiobutton(kind_row, text="目标 Target", variable=self.entry_kind, value="target").pack(side=tk.LEFT, padx=6)
        ttk.Label(norm_tab, text="蛋白名").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(norm_tab, textvariable=self.protein_name).grid(row=1, column=1, sticky="ew", pady=(8, 0))
        ttk.Label(norm_tab, text="分组").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(norm_tab, textvariable=self.group_name).grid(row=2, column=1, sticky="ew", pady=(8, 0))
        ttk.Button(norm_tab, text="把当前结果加入历史", command=self.add_current_to_session).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(10, 4)
        )
        self.history_list = tk.Listbox(norm_tab, height=8, exportselection=False)
        self.history_list.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=4)
        hist_actions = ttk.Frame(norm_tab)
        hist_actions.grid(row=5, column=0, columnspan=2, sticky="ew")
        ttk.Button(hist_actions, text="删除选中", command=self.delete_history_item).pack(side=tk.LEFT)
        ttk.Button(hist_actions, text="导出内参 CSV", command=self.export_loading_control).pack(side=tk.LEFT, padx=4)
        ttk.Button(hist_actions, text="导入内参 CSV", command=self.import_loading_control).pack(side=tk.LEFT)
        ttk.Button(norm_tab, text="导出归一化结果", command=self.export_normalized).grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )
        self.norm_preview = tk.Text(norm_tab, height=8, wrap=tk.WORD)
        self.norm_preview.grid(row=7, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        norm_tab.columnconfigure(1, weight=1)
        norm_tab.rowconfigure(4, weight=1)
        norm_tab.rowconfigure(7, weight=1)

        band_tab.rowconfigure(0, weight=1)
        band_tab.columnconfigure(0, weight=1)
        self.band_list = tk.Listbox(band_tab, exportselection=False)
        self.band_list.grid(row=0, column=0, sticky="nsew")
        actions = ttk.Frame(band_tab)
        actions.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Checkbutton(actions, text="自动模式下画框补条带", variable=self.drawing_band).pack(side=tk.LEFT)
        ttk.Button(actions, text="删除选中条带", command=self.delete_selected_band).pack(side=tk.RIGHT)

    def _current_config(self) -> AnalysisConfig:
        kernel = None if self.kernel_mode.get() == "自动" else int(self.kernel_size.get())
        return AnalysisConfig(
            sensitivity=float(self.sensitivity.get()),
            kernel_size=kernel,
            background_mode=self.background_mode.get(),  # type: ignore[arg-type]
        )

    def _set_working_from_rotation(self) -> None:
        if self.original_bgr is None:
            return
        angle = float(self.rotation.get())
        self.rotation_label.config(text=f"{angle:.1f}°")
        self.working_bgr = rotate_image(self.original_bgr, angle)
        self.result = None
        self.roi = None
        self.separators = []
        self.band_list.delete(0, tk.END)
        self.export_button.config(state=tk.DISABLED)
        self._show_bgr(self.working_bgr)

    def _on_rotation_changed(self, *_args: object) -> None:
        if self.original_bgr is None:
            return
        self._set_working_from_rotation()
        self.status.config(text=f"已旋转 {float(self.rotation.get()):.1f}°，请重新分析")

    def _on_lane_count_changed(self, *_args: object) -> None:
        if self.roi is not None:
            self.reset_separators()

    def choose_image(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择 Western Blot 图片",
            filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("所有文件", "*.*")],
        )
        if not selected:
            return
        self.image_path = Path(selected)
        self.original_bgr = load_bgr(self.image_path)
        self.rotation.set(0.0)
        self._set_working_from_rotation()
        self.output_dir = None
        self.run_button.config(state=tk.NORMAL)
        self.roi_button.config(state=tk.NORMAL)
        self.profile_button.config(state=tk.NORMAL)
        self.status.config(text=self.image_path.name)

    def _show_bgr(self, bgr: np.ndarray) -> None:
        rgb = Image.fromarray(bgr[:, :, ::-1].copy())
        self.base_image_rgb = rgb
        self._render_base(rgb)
        self._redraw_overlays()

    def _show_result_image(self) -> None:
        if not self.result:
            return
        rgb = Image.fromarray(self.result.annotated_image[:, :, ::-1].copy())
        self.base_image_rgb = rgb
        self._render_base(rgb)
        self._redraw_overlays()

    def _render_base(self, image: Image.Image) -> None:
        self.canvas.update_idletasks()
        max_w = max(200, self.canvas.winfo_width() or 720)
        max_h = max(200, self.canvas.winfo_height() or 520)
        fitted = image.copy()
        fitted.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        self.display_scale = fitted.width / image.width if image.width else 1.0
        self.preview = ImageTk.PhotoImage(fitted)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.preview, tags=("image",))
        self.canvas.config(scrollregion=(0, 0, fitted.width, fitted.height))

    def _image_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return x * self.display_scale, y * self.display_scale

    def _canvas_to_image(self, x: float, y: float) -> tuple[int, int]:
        if self.display_scale <= 0:
            return int(x), int(y)
        return int(round(x / self.display_scale)), int(round(y / self.display_scale))

    def _redraw_overlays(self) -> None:
        self.canvas.delete("overlay")
        if self.roi is None:
            return
        x1, y1 = self._image_to_canvas(self.roi.x, self.roi.y)
        x2, y2 = self._image_to_canvas(self.roi.x + self.roi.width, self.roi.y + self.roi.height)
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="#00e5ff", width=2, tags=("overlay", "roi"))
        for index, sep in enumerate(self.separators):
            sx, _ = self._image_to_canvas(sep, 0)
            self.canvas.create_line(sx, y1, sx, y2, fill="#ffb300", width=2, tags=("overlay", f"sep-{index}"))

    def reset_separators(self) -> None:
        if self.roi is None:
            return
        try:
            n = int(self.lane_count.get())
        except (tk.TclError, ValueError):
            n = 1
        self.separators = default_separators(self.roi, n)
        self._redraw_overlays()

    def clear_roi(self) -> None:
        self.roi = None
        self.separators = []
        self._redraw_overlays()
        self.status.config(text="已清除 ROI")

    def _hit_separator(self, canvas_x: float) -> int | None:
        if self.roi is None:
            return None
        for index, sep in enumerate(self.separators):
            sx, _ = self._image_to_canvas(sep, 0)
            if abs(canvas_x - sx) <= 6:
                return index
        return None

    def _on_press(self, event: tk.Event) -> None:
        if self.working_bgr is None:
            return
        if self.mode.get() == "roi":
            hit = self._hit_separator(event.x)
            if hit is not None:
                self.active_separator = hit
                return
            self.draw_start = (event.x, event.y)
            if self.draw_rect is not None:
                self.canvas.delete(self.draw_rect)
            self.draw_rect = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="#00e5ff", width=2)
            return
        if self.drawing_band.get() and self.result is not None:
            self.draw_start = (event.x, event.y)
            if self.draw_rect is not None:
                self.canvas.delete(self.draw_rect)
            self.draw_rect = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="#00d4ff", width=2)

    def _on_drag(self, event: tk.Event) -> None:
        if self.active_separator is not None and self.roi is not None:
            ix, _ = self._canvas_to_image(event.x, event.y)
            left = self.roi.x + 2
            right = self.roi.x + self.roi.width - 2
            # Keep separators ordered with neighbors.
            min_x = left if self.active_separator == 0 else self.separators[self.active_separator - 1] + 2
            max_x = right if self.active_separator == len(self.separators) - 1 else self.separators[self.active_separator + 1] - 2
            self.separators[self.active_separator] = float(np.clip(ix, min_x, max_x))
            self._redraw_overlays()
            return
        if self.draw_start and self.draw_rect is not None:
            self.canvas.coords(self.draw_rect, self.draw_start[0], self.draw_start[1], event.x, event.y)

    def _on_release(self, event: tk.Event) -> None:
        if self.active_separator is not None:
            self.active_separator = None
            self.status.config(text="已调整泳道分隔线，可重新 ROI 定量")
            return
        if not self.draw_start:
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
        if self.mode.get() == "roi":
            self.roi = RoiRect(x1, y1, x2 - x1, y2 - y1)
            self.reset_separators()
            self.result = None
            if self.working_bgr is not None:
                self._show_bgr(self.working_bgr)
            self.status.config(text=f"已设置 ROI，{int(self.lane_count.get())} 个泳道，可拖分隔线后定量")
            return
        if self.result is None or self.result.gray_image is None:
            return
        added = BandMeasurement(0, 0, x1, y1, x2 - x1, y2 - y1, 0, 0.0, 0.0, 0.0)
        self.result = rebuild_result(self.result.gray_image, [*self.result.measurements, added], dark_bands=self.result.dark_bands)
        self._refresh_band_list()
        self._show_result_image()
        self.status.config(text=f"已添加手动条带，当前 {len(self.result.measurements)} 个")

    def _refresh_band_list(self) -> None:
        self.band_list.delete(0, tk.END)
        if not self.result:
            return
        for item in self.result.measurements:
            flag = " ⚠饱和" if item.saturated_fraction >= 0.05 else ""
            self.band_list.insert(tk.END, f"L{item.lane}-B{item.band}  dens={item.integrated_density:.1f}{flag}")
        self.export_button.config(state=tk.NORMAL)

    def run_auto_analysis(self) -> None:
        if self.working_bgr is None:
            return
        self.status.config(text="正在自动分析……")
        self.update_idletasks()
        try:
            self.result = analyze_array(self.working_bgr, config=self._current_config())
            self._refresh_band_list()
            self._show_result_image()
            msg = f"自动检测：{len(self.result.measurements)} 个条带"
            if self.result.saturated_band_count:
                msg += f"（{self.result.saturated_band_count} 个可能饱和）"
            self.status.config(text=msg)
        except (ValueError, OSError) as exc:
            self.status.config(text="分析失败")
            messagebox.showerror("无法分析", str(exc))

    def run_roi_quantify(self) -> None:
        if self.working_bgr is None:
            return
        if self.roi is None:
            messagebox.showinfo("提示", "请先切换到“ROI 泳道”模式并拖拽框选 ROI。")
            return
        gray = to_gray(self.working_bgr)
        try:
            self.result = quantify_roi_lanes(
                gray,
                self.roi,
                self.separators,
                background_mode=self.background_mode.get(),  # type: ignore[arg-type]
            )
            self._refresh_band_list()
            self._show_result_image()
            self.status.config(text=f"ROI 定量完成：{len(self.result.measurements)} 个泳道")
        except ValueError as exc:
            messagebox.showerror("ROI 定量失败", str(exc))

    def show_profile(self) -> None:
        if self.working_bgr is None:
            return
        if self.roi is None:
            messagebox.showinfo("提示", "请先框选 ROI，再查看强度剖面。")
            return
        gray = to_gray(self.working_bgr)
        profile, _ = roi_intensity_profile(gray, self.roi, background_mode=self.background_mode.get())  # type: ignore[arg-type]
        rel = [max(0.0, min(float(self.roi.width - 1), sep - self.roi.x)) for sep in self.separators]
        ProfileDialog(self, profile, rel)

    def delete_selected_band(self) -> None:
        if not self.result or self.result.gray_image is None:
            return
        selection = list(self.band_list.curselection())
        if not selection:
            return
        kept = [m for idx, m in enumerate(self.result.measurements) if idx not in set(selection)]
        self.result = rebuild_result(self.result.gray_image, kept, dark_bands=self.result.dark_bands)
        self._refresh_band_list()
        self._show_result_image()

    def _refresh_history_list(self) -> None:
        self.history_list.delete(0, tk.END)
        for entry in self.session.entries:
            label = "LC" if entry.kind == "loading_control" else "Target"
            self.history_list.insert(tk.END, f"[{label}] {entry.name} / {entry.group}  n={entry.lane_count}")
        self._refresh_norm_preview()

    def _refresh_norm_preview(self) -> None:
        self.norm_preview.delete("1.0", tk.END)
        rows = self.session.normalize_all()
        if not rows:
            self.norm_preview.insert(tk.END, "尚无归一化结果。请先加入内参与目标。")
            return
        lines = ["group\tlane\ttarget\tLC\tnormalized"]
        for row in rows:
            norm = "" if row.normalized is None else f"{row.normalized:.4f}"
            lines.append(
                f"{row.group}\t{row.lane}\t{row.target_name}={row.target:.1f}\t"
                f"{row.loading_control_name}={row.loading_control:.1f}\t{norm}"
            )
        self.norm_preview.insert(tk.END, "\n".join(lines))

    def add_current_to_session(self) -> None:
        if not self.result or not self.result.measurements:
            messagebox.showinfo("提示", "请先完成自动分析或 ROI 定量。")
            return
        if self.mode.get() == "roi" or all(m.band == 1 for m in self.result.measurements):
            intensities = [m.integrated_density for m in sorted(self.result.measurements, key=lambda m: m.lane)]
            backgrounds = [m.background for m in sorted(self.result.measurements, key=lambda m: m.lane)]
            means = [m.mean_intensity for m in sorted(self.result.measurements, key=lambda m: m.lane)]
        else:
            intensities, backgrounds, means = lane_intensities_from_measurements(self.result.measurements)
        try:
            self.session.upsert(
                QuantEntry(
                    kind=self.entry_kind.get(),  # type: ignore[arg-type]
                    name=self.protein_name.get(),
                    group=self.group_name.get(),
                    intensities=intensities,
                    backgrounds=backgrounds,
                    mean_intensities=means,
                    source=str(self.image_path or ""),
                )
            )
        except ValueError as exc:
            messagebox.showerror("无法加入历史", str(exc))
            return
        self._refresh_history_list()
        self.status.config(text=f"已保存到历史：{self.protein_name.get()} / {self.group_name.get()}")

    def delete_history_item(self) -> None:
        selection = list(self.history_list.curselection())
        if not selection:
            return
        self.session.remove_at(selection[0])
        self._refresh_history_list()

    def export_loading_control(self) -> None:
        if not self.session.loading_controls():
            messagebox.showinfo("提示", "历史中还没有内参数据。")
            return
        path = filedialog.asksaveasfilename(
            title="导出内参 CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="loading_control.csv",
        )
        if not path:
            return
        try:
            saved = self.session.export_loading_control_csv(path)
            messagebox.showinfo("导出完成", f"内参已保存：\n{saved}")
        except ValueError as exc:
            messagebox.showerror("导出失败", str(exc))

    def import_loading_control(self) -> None:
        path = filedialog.askopenfilename(title="导入内参 CSV", filetypes=[("CSV", "*.csv"), ("所有文件", "*.*")])
        if not path:
            return
        try:
            count = self.session.import_loading_control_csv(path)
            self._refresh_history_list()
            messagebox.showinfo("导入完成", f"已导入 {count} 组内参/历史数据。")
        except ValueError as exc:
            messagebox.showerror("导入失败", str(exc))

    def export_normalized(self) -> None:
        rows = self.session.normalize_all()
        if not rows:
            messagebox.showinfo("提示", "没有可导出的归一化结果。请确保同一分组下既有内参也有目标。")
            return
        path = filedialog.asksaveasfilename(
            title="导出归一化结果",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="normalized_results.csv",
        )
        if not path:
            return
        saved = export_normalized_csv(rows, path)
        messagebox.showinfo("导出完成", f"归一化结果：\n{saved}")

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

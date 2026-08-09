"""Tkinter desktop application."""

from __future__ import annotations

import re
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from .analysis import (
    AnalysisConfig,
    AnalysisResult,
    BandMeasurement,
    _resolve_dark_bands,
    analyze_array,
    rebuild_result,
)
from .export import export_figure, export_result
from .normalize import (
    AnalysisSession,
    QuantEntry,
    export_normalized_csv,
    lane_intensities_from_measurements,
)
from .roi import (
    RoiRect,
    angle_to_horizontal,
    default_separators,
    load_bgr,
    normalize_angle,
    quantify_roi_lanes,
    roi_intensity_profile,
    rotate_image,
    to_gray,
)

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    _TkBase = TkinterDnD.Tk
    _HAS_DND = True
except Exception:  # pragma: no cover - optional at runtime
    _TkBase = tk.Tk
    DND_FILES = None
    _HAS_DND = False


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ProfileDialog(tk.Toplevel):
    def __init__(self, master: tk.Misc, profile: np.ndarray, separators_rel: list[float]) -> None:
        super().__init__(master)
        self.title("强度剖面")
        self.geometry("720x320")
        self.minsize(480, 240)
        self._canvas = tk.Canvas(self, background="#111111", highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self._profile = np.asarray(profile, dtype=float)
        self._separators = separators_rel
        self._canvas.bind("<Configure>", self._redraw)
        self._redraw()

    def _redraw(self, *_args: object) -> None:
        canvas = self._canvas
        canvas.delete("all")
        width = max(40, canvas.winfo_width())
        height = max(40, canvas.winfo_height())
        pad = 28
        plot_w, plot_h = width - 2 * pad, height - 2 * pad
        profile = self._profile
        if profile.size < 2 or plot_w < 10 or plot_h < 10:
            return
        ymin, ymax = float(profile.min()), float(profile.max())
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


class AnalyzerApp(_TkBase):
    def __init__(self) -> None:
        super().__init__()
        self.title("WB 灰度统计工具")
        self.geometry("1360x820")
        self.minsize(1024, 680)

        self.image_path: Path | None = None
        self.original_bgr = None
        self.working_bgr = None
        self.result: AnalysisResult | None = None
        self.output_dir: Path | None = None
        self.preview: ImageTk.PhotoImage | None = None
        self.display_scale = 1.0
        self.view_offset = (0, 0)

        self.roi: RoiRect | None = None
        self.separators: list[float] = []
        self.draw_start: tuple[int, int] | None = None
        self.draw_item: int | None = None
        self.active_separator: int | None = None
        self.ruler_start: tuple[int, int] | None = None
        self.ruler_angle = 0.0
        self._rotate_job: str | None = None
        self.session = AnalysisSession()

        self.tool = tk.StringVar(value="roi")
        self.rotation = tk.DoubleVar(value=0.0)
        self.sensitivity = tk.DoubleVar(value=1.0)
        self.kernel_mode = tk.StringVar(value="自动")
        self.kernel_size = tk.IntVar(value=31)
        self.background_mode = tk.StringVar(value="auto")
        self.lane_count = tk.IntVar(value=3)
        self.show_guides = tk.BooleanVar(value=True)
        self.entry_kind = tk.StringVar(value="loading_control")
        self.protein_name = tk.StringVar(value="Actin")
        self.group_name = tk.StringVar(value="默认")
        self.export_annotated = tk.BooleanVar(value=False)

        self._build_layout()
        self._bind_canvas()
        self._setup_dnd()
        self.rotation.trace_add("write", self._schedule_rotation)
        self.lane_count.trace_add("write", self._on_lane_count_changed)
        self.tool.trace_add("write", self._on_tool_changed)
        self.show_guides.trace_add("write", lambda *_: self._redraw_overlays())
        self._on_tool_changed()

    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=8)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(root)
        toolbar.grid(row=0, column=0, sticky="ew")
        ttk.Button(toolbar, text="打开", command=self.choose_image).pack(side=tk.LEFT)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        for value, label in (
            ("roi", "框选 ROI"),
            ("ruler", "标尺对齐"),
            ("band", "补画条带"),
            ("select", "调整分隔线"),
        ):
            ttk.Radiobutton(toolbar, text=label, variable=self.tool, value=value).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        self.run_button = ttk.Button(toolbar, text="自动分析", command=self.run_auto_analysis, state=tk.DISABLED)
        self.run_button.pack(side=tk.LEFT, padx=2)
        self.roi_button = ttk.Button(toolbar, text="ROI 定量", command=self.run_roi_quantify, state=tk.DISABLED)
        self.roi_button.pack(side=tk.LEFT, padx=2)
        self.profile_button = ttk.Button(toolbar, text="强度剖面", command=self.show_profile, state=tk.DISABLED)
        self.profile_button.pack(side=tk.LEFT, padx=2)
        self.export_button = ttk.Button(toolbar, text="导出数据", command=self.export_current, state=tk.DISABLED)
        self.export_button.pack(side=tk.LEFT, padx=2)
        self.figure_button = ttk.Button(toolbar, text="导出展示图", command=self.export_presentation, state=tk.DISABLED)
        self.figure_button.pack(side=tk.LEFT, padx=2)
        self.status = ttk.Label(toolbar, text="打开图片，或把图片拖进预览区")
        self.status.pack(side=tk.LEFT, padx=10)

        paned = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        paned.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

        preview = ttk.LabelFrame(paned, text="预览（可拖拽分隔左右宽度）", padding=4)
        side = ttk.Frame(paned)
        paned.add(preview, weight=3)
        paned.add(side, weight=2)
        preview.rowconfigure(0, weight=1)
        preview.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(preview, background="#1a1a1a", highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _e: self._refresh_view(keep_overlays=self.result is None))

        notebook = ttk.Notebook(side)
        notebook.pack(fill=tk.BOTH, expand=True)
        prep = ttk.Frame(notebook, padding=8)
        roi_tab = ttk.Frame(notebook, padding=8)
        norm_tab = ttk.Frame(notebook, padding=8)
        band_tab = ttk.Frame(notebook, padding=8)
        notebook.add(prep, text="预处理")
        notebook.add(roi_tab, text="ROI")
        notebook.add(norm_tab, text="归一化")
        notebook.add(band_tab, text="条带")

        ttk.Checkbutton(prep, text="显示水平/垂直参考线", variable=self.show_guides).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(prep, text="旋转角度").grid(row=1, column=0, sticky="w", pady=(10, 0))
        rot = ttk.Frame(prep)
        rot.grid(row=1, column=1, sticky="ew", pady=(10, 0))
        ttk.Scale(rot, from_=-30, to=30, variable=self.rotation, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.rotation_label = ttk.Label(rot, text="0.0°", width=7)
        self.rotation_label.pack(side=tk.LEFT)
        ttk.Button(rot, text="归零", command=lambda: self.rotation.set(0.0)).pack(side=tk.LEFT, padx=4)
        ttk.Label(
            prep,
            text="旋转：正值抬高右侧（纠正右低左高），负值相反。也可用标尺拖两点后一键旋正。",
            wraplength=320,
            justify=tk.LEFT,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.ruler_label = ttk.Label(prep, text="标尺：未测量")
        self.ruler_label.grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ruler_btns = ttk.Frame(prep)
        ruler_btns.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(ruler_btns, text="旋正为水平", command=lambda: self.apply_ruler("horizontal")).pack(side=tk.LEFT)
        ttk.Button(ruler_btns, text="旋正为垂直", command=lambda: self.apply_ruler("vertical")).pack(side=tk.LEFT, padx=6)

        ttk.Label(prep, text="灵敏度").grid(row=5, column=0, sticky="w", pady=(12, 0))
        sens = ttk.Frame(prep)
        sens.grid(row=5, column=1, sticky="ew", pady=(12, 0))
        ttk.Scale(sens, from_=0.5, to=1.8, variable=self.sensitivity, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.sensitivity_label = ttk.Label(sens, text="1.00", width=6)
        self.sensitivity_label.pack(side=tk.LEFT)
        self.sensitivity.trace_add("write", lambda *_: self.sensitivity_label.config(text=f"{float(self.sensitivity.get()):.2f}"))

        ttk.Label(prep, text="形态学核").grid(row=6, column=0, sticky="w", pady=(10, 0))
        kernel_row = ttk.Frame(prep)
        kernel_row.grid(row=6, column=1, sticky="w", pady=(10, 0))
        ttk.Combobox(kernel_row, textvariable=self.kernel_mode, values=("自动", "手动"), width=6, state="readonly").pack(side=tk.LEFT)
        ttk.Spinbox(kernel_row, from_=11, to=101, increment=2, textvariable=self.kernel_size, width=6).pack(side=tk.LEFT, padx=6)
        ttk.Label(prep, text="背景模式").grid(row=7, column=0, sticky="w", pady=(10, 0))
        ttk.Combobox(
            prep,
            textvariable=self.background_mode,
            values=("auto", "dark_bands", "light_bands"),
            state="readonly",
            width=14,
        ).grid(row=7, column=1, sticky="w", pady=(10, 0))
        prep.columnconfigure(1, weight=1)

        ttk.Label(roi_tab, text="泳道数").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(roi_tab, from_=1, to=24, textvariable=self.lane_count, width=6).grid(row=0, column=1, sticky="w")
        ttk.Button(roi_tab, text="等分重置分隔线", command=self.reset_separators).grid(row=1, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Button(roi_tab, text="清除 ROI", command=self.clear_roi).grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Label(
            roi_tab,
            text="1) 点顶部“框选 ROI”\n2) 在图上拖出青色框\n3) 用“调整分隔线”拖橙色竖线\n4) 点“ROI 定量”",
            wraplength=320,
            justify=tk.LEFT,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(12, 0))

        ttk.Label(norm_tab, text="类型").grid(row=0, column=0, sticky="w")
        kind_row = ttk.Frame(norm_tab)
        kind_row.grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(kind_row, text="内参 LC", variable=self.entry_kind, value="loading_control").pack(side=tk.LEFT)
        ttk.Radiobutton(kind_row, text="目标", variable=self.entry_kind, value="target").pack(side=tk.LEFT, padx=6)
        ttk.Label(norm_tab, text="蛋白名").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(norm_tab, textvariable=self.protein_name).grid(row=1, column=1, sticky="ew", pady=(8, 0))
        ttk.Label(norm_tab, text="分组").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(norm_tab, textvariable=self.group_name).grid(row=2, column=1, sticky="ew", pady=(8, 0))
        ttk.Button(norm_tab, text="加入历史", command=self.add_current_to_session).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 4))
        self.history_list = tk.Listbox(norm_tab, height=7, exportselection=False)
        self.history_list.grid(row=4, column=0, columnspan=2, sticky="nsew")
        actions = ttk.Frame(norm_tab)
        actions.grid(row=5, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(actions, text="删除", command=self.delete_history_item).pack(side=tk.LEFT)
        ttk.Button(actions, text="导出内参", command=self.export_loading_control).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="导入内参", command=self.import_loading_control).pack(side=tk.LEFT)
        ttk.Button(norm_tab, text="导出归一化结果", command=self.export_normalized).grid(row=6, column=0, columnspan=2, sticky="ew")
        self.norm_preview = tk.Text(norm_tab, height=7, wrap=tk.WORD)
        self.norm_preview.grid(row=7, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        norm_tab.columnconfigure(1, weight=1)
        norm_tab.rowconfigure(4, weight=1)
        norm_tab.rowconfigure(7, weight=1)

        band_tab.rowconfigure(0, weight=1)
        band_tab.columnconfigure(0, weight=1)
        self.band_list = tk.Listbox(band_tab, exportselection=False)
        self.band_list.grid(row=0, column=0, sticky="nsew")
        ttk.Button(band_tab, text="删除选中条带", command=self.delete_selected_band).grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Checkbutton(band_tab, text="导出展示图时带检测标注", variable=self.export_annotated).grid(row=2, column=0, sticky="w", pady=(8, 0))

    def _bind_canvas(self) -> None:
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Escape>", lambda _e: self._cancel_draw())

    def _setup_dnd(self) -> None:
        if not _HAS_DND or DND_FILES is None:
            return
        try:
            self.canvas.drop_target_register(DND_FILES)
            self.canvas.dnd_bind("<<Drop>>", self._on_drop)
            self.drop_target_register(DND_FILES)
            self.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:
            pass

    def _on_drop(self, event) -> None:  # noqa: ANN001
        paths = self._parse_drop_paths(getattr(event, "data", ""))
        for path in paths:
            if path.suffix.lower() in IMAGE_SUFFIXES and path.exists():
                self.load_image(path)
                return
        messagebox.showwarning("拖拽失败", "请拖入 PNG/JPG/BMP/TIFF 图片文件。")

    @staticmethod
    def _parse_drop_paths(data: str) -> list[Path]:
        if not data:
            return []
        parts = re.findall(r"\{([^}]+)\}|[^\s]+", data)
        return [Path(part.strip().strip("{}")) for part in parts if part.strip()]

    def _on_tool_changed(self, *_args: object) -> None:
        tips = {
            "roi": "当前工具：框选 ROI — 按住拖拽画出青色矩形",
            "ruler": "当前工具：标尺 — 点两个点测量角度，再旋正",
            "band": "当前工具：补画条带 — 拖拽添加单个条带框",
            "select": "当前工具：调整分隔线 — 拖橙色竖线；也可点分隔线附近微调",
        }
        self.status.config(text=tips.get(self.tool.get(), ""))
        cursors = {"roi": "crosshair", "ruler": "tcross", "band": "plus", "select": "sb_h_double_arrow"}
        self.canvas.config(cursor=cursors.get(self.tool.get(), "arrow"))

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
        if selected:
            self.load_image(Path(selected))

    def load_image(self, path: Path) -> None:
        self.image_path = path
        self.original_bgr = load_bgr(path)
        self.rotation.set(0.0)
        self.roi = None
        self.separators = []
        self.result = None
        self.ruler_start = None
        self.ruler_angle = 0.0
        self.ruler_label.config(text="标尺：未测量")
        self._apply_rotation(reset_roi=True)
        for button in (self.run_button, self.roi_button, self.profile_button, self.figure_button):
            button.config(state=tk.NORMAL)
        self.export_button.config(state=tk.DISABLED)
        self.band_list.delete(0, tk.END)
        self.tool.set("roi")
        self.status.config(text=f"已加载 {path.name} — 可直接框选 ROI，或拖图进来换图")

    def _schedule_rotation(self, *_args: object) -> None:
        self.rotation_label.config(text=f"{float(self.rotation.get()):.1f}°")
        if self.original_bgr is None:
            return
        if self._rotate_job is not None:
            self.after_cancel(self._rotate_job)
        self._rotate_job = self.after(140, lambda: self._apply_rotation(reset_roi=True))

    def _apply_rotation(self, reset_roi: bool = False) -> None:
        self._rotate_job = None
        if self.original_bgr is None:
            return
        angle = float(self.rotation.get())
        self.working_bgr = rotate_image(self.original_bgr, angle)
        if reset_roi:
            self.roi = None
            self.separators = []
            self.result = None
            self.export_button.config(state=tk.DISABLED)
            self.band_list.delete(0, tk.END)
        self._refresh_view(keep_overlays=not reset_roi)
        if reset_roi:
            self.status.config(text=f"已旋转 {angle:.1f}°（参考线已更新）。请重新框选 ROI 或分析。")

    def _refresh_view(self, keep_overlays: bool = False) -> None:
        if self.working_bgr is None and self.result is None:
            return
        if self.result is not None and self.export_annotated.get():
            # Keep showing annotated result when available, unless overlays-only refresh.
            source = self.result.annotated_image
        elif self.result is not None and not keep_overlays:
            source = self.result.annotated_image
        else:
            source = self.working_bgr if self.working_bgr is not None else self.result.annotated_image  # type: ignore[union-attr]
        if source is None:
            return
        rgb = Image.fromarray(source[:, :, ::-1].copy())
        self.canvas.update_idletasks()
        max_w = max(240, self.canvas.winfo_width() or 800)
        max_h = max(240, self.canvas.winfo_height() or 560)
        fitted = rgb.copy()
        fitted.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        self.display_scale = fitted.width / rgb.width if rgb.width else 1.0
        ox = max(0, (max_w - fitted.width) // 2)
        oy = max(0, (max_h - fitted.height) // 2)
        self.view_offset = (ox, oy)
        self.preview = ImageTk.PhotoImage(fitted)
        self.canvas.delete("all")
        self.canvas.create_image(ox, oy, anchor=tk.NW, image=self.preview, tags=("image",))
        self.canvas.config(scrollregion=(0, 0, max_w, max_h))
        self._redraw_overlays()

    def _image_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        ox, oy = self.view_offset
        return ox + x * self.display_scale, oy + y * self.display_scale

    def _canvas_to_image(self, x: float, y: float) -> tuple[int, int]:
        ox, oy = self.view_offset
        if self.display_scale <= 0:
            return int(x - ox), int(y - oy)
        return int(round((x - ox) / self.display_scale)), int(round((y - oy) / self.display_scale))

    def _redraw_overlays(self) -> None:
        self.canvas.delete("overlay")
        if self.working_bgr is None:
            return
        h, w = self.working_bgr.shape[:2]
        if self.show_guides.get():
            x0, y_mid = self._image_to_canvas(0, h / 2)
            x1, _ = self._image_to_canvas(w, h / 2)
            _, y0 = self._image_to_canvas(w / 2, 0)
            x_mid, y1 = self._image_to_canvas(w / 2, h)
            self.canvas.create_line(x0, y_mid, x1, y_mid, fill="#00ff9c", dash=(8, 6), width=1, tags=("overlay", "guide"))
            self.canvas.create_line(x_mid, y0, x_mid, y1, fill="#00ff9c", dash=(8, 6), width=1, tags=("overlay", "guide"))
        if self.roi is not None:
            x1, y1 = self._image_to_canvas(self.roi.x, self.roi.y)
            x2, y2 = self._image_to_canvas(self.roi.x + self.roi.width, self.roi.y + self.roi.height)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="#00e5ff", width=2, tags=("overlay", "roi"))
            for index, sep in enumerate(self.separators):
                sx, _ = self._image_to_canvas(sep, 0)
                self.canvas.create_line(sx, y1, sx, y2, fill="#ffb300", width=3, tags=("overlay", f"sep-{index}"))

    def reset_separators(self) -> None:
        if self.roi is None:
            return
        try:
            n = int(self.lane_count.get())
        except (tk.TclError, ValueError):
            n = 1
        self.separators = default_separators(self.roi, max(1, n))
        self._redraw_overlays()

    def clear_roi(self) -> None:
        self.roi = None
        self.separators = []
        self._redraw_overlays()
        self.status.config(text="已清除 ROI")

    def _on_lane_count_changed(self, *_args: object) -> None:
        if self.roi is not None:
            self.reset_separators()

    def _hit_separator(self, canvas_x: float) -> int | None:
        if self.roi is None:
            return None
        for index, sep in enumerate(self.separators):
            sx, _ = self._image_to_canvas(sep, 0)
            if abs(canvas_x - sx) <= 8:
                return index
        return None

    def _cancel_draw(self) -> None:
        self.draw_start = None
        self.ruler_start = None
        self.active_separator = None
        if self.draw_item is not None:
            self.canvas.delete(self.draw_item)
            self.draw_item = None

    def _on_press(self, event: tk.Event) -> None:
        if self.working_bgr is None:
            return
        tool = self.tool.get()
        # Separators can be dragged in select/roi tools.
        if tool in {"select", "roi"} and self.roi is not None:
            hit = self._hit_separator(event.x)
            if hit is not None:
                self.active_separator = hit
                return
        if tool == "ruler":
            self.ruler_start = self._canvas_to_image(event.x, event.y)
            if self.draw_item is not None:
                self.canvas.delete(self.draw_item)
            self.draw_item = self.canvas.create_line(event.x, event.y, event.x, event.y, fill="#ff4081", width=2, tags=("temp",))
            return
        if tool in {"roi", "band"}:
            self.draw_start = (event.x, event.y)
            if self.draw_item is not None:
                self.canvas.delete(self.draw_item)
            color = "#00e5ff" if tool == "roi" else "#7cff6b"
            self.draw_item = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline=color, width=2, tags=("temp",))

    def _on_drag(self, event: tk.Event) -> None:
        if self.active_separator is not None and self.roi is not None:
            ix, _ = self._canvas_to_image(event.x, event.y)
            left = self.roi.x + 2
            right = self.roi.x + self.roi.width - 2
            min_x = left if self.active_separator == 0 else self.separators[self.active_separator - 1] + 2
            max_x = right if self.active_separator == len(self.separators) - 1 else self.separators[self.active_separator + 1] - 2
            self.separators[self.active_separator] = float(np.clip(ix, min_x, max_x))
            self._redraw_overlays()
            return
        if self.draw_item is None:
            return
        if self.ruler_start is not None:
            x0, y0 = self._image_to_canvas(*self.ruler_start)
            self.canvas.coords(self.draw_item, x0, y0, event.x, event.y)
            return
        if self.draw_start is not None:
            self.canvas.coords(self.draw_item, self.draw_start[0], self.draw_start[1], event.x, event.y)

    def _on_release(self, event: tk.Event) -> None:
        if self.active_separator is not None:
            self.active_separator = None
            self.status.config(text="分隔线已更新，可重新 ROI 定量")
            return
        if self.ruler_start is not None:
            x0, y0 = self.ruler_start
            x1, y1 = self._canvas_to_image(event.x, event.y)
            self.ruler_start = None
            if self.draw_item is not None:
                self.canvas.delete(self.draw_item)
                self.draw_item = None
            if abs(x1 - x0) + abs(y1 - y0) < 4:
                return
            self.ruler_angle = normalize_angle(angle_to_horizontal(x0, y0, x1, y1))
            self.ruler_label.config(text=f"标尺倾角：{self.ruler_angle:.2f}°（相对水平）")
            self.status.config(text="标尺已测量，可点“旋正为水平/垂直”")
            # Keep a finished ruler line briefly.
            a = self._image_to_canvas(x0, y0)
            b = self._image_to_canvas(x1, y1)
            self.canvas.create_line(*a, *b, fill="#ff4081", width=2, tags=("overlay", "ruler"))
            return
        if not self.draw_start:
            return
        x0, y0 = self.draw_start
        self.draw_start = None
        if self.draw_item is not None:
            self.canvas.delete(self.draw_item)
            self.draw_item = None
        ix0, iy0 = self._canvas_to_image(x0, y0)
        ix1, iy1 = self._canvas_to_image(event.x, event.y)
        x1, x2 = sorted((ix0, ix1))
        y1, y2 = sorted((iy0, iy1))
        if x2 - x1 < 3 or y2 - y1 < 3:
            self.status.config(text="框太小，请拖大一点再松开")
            return
        if self.tool.get() == "roi":
            if self.working_bgr is not None:
                h, w = self.working_bgr.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
            self.roi = RoiRect(x1, y1, x2 - x1, y2 - y1)
            self.reset_separators()
            self.result = None
            self.export_button.config(state=tk.DISABLED)
            self._refresh_view(keep_overlays=True)
            self.tool.set("select")
            self.status.config(text=f"ROI 已创建（{self.roi.width}×{self.roi.height}）。拖橙色线对齐后点“ROI 定量”。")
            return
        if self.tool.get() == "band":
            if self.working_bgr is None:
                return
            gray = to_gray(self.working_bgr)
            dark_bands = (
                self.result.dark_bands
                if self.result is not None
                else _resolve_dark_bands(gray, self.background_mode.get())  # type: ignore[arg-type]
            )
            existing = list(self.result.measurements) if self.result is not None else []
            added = BandMeasurement(0, 0, x1, y1, x2 - x1, y2 - y1, 0, 0.0, 0.0, 0.0)
            self.result = rebuild_result(gray, [*existing, added], dark_bands=dark_bands)
            self._refresh_band_list()
            self._refresh_view(keep_overlays=False)
            self.status.config(text=f"已添加条带，当前 {len(self.result.measurements)} 个")

    def apply_ruler(self, orientation: str) -> None:
        if self.original_bgr is None:
            return
        if self.ruler_label.cget("text") == "标尺：未测量":
            messagebox.showinfo("提示", "请先用“标尺对齐”工具在图上拖一条线。")
            return
        current = float(self.rotation.get())
        # Measured tilt uses image atan2 (down-right is positive). OpenCV positive
        # rotation reduces that tilt, so add the measured angle to straighten.
        if orientation == "horizontal":
            new_angle = normalize_angle(current + self.ruler_angle)
        else:
            new_angle = normalize_angle(current + (self.ruler_angle - 90.0))
        self.rotation.set(new_angle)
        self.status.config(text=f"已按标尺旋正为{'水平' if orientation == 'horizontal' else '垂直'}（{new_angle:.2f}°）")

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
            self._refresh_view(keep_overlays=False)
            self.status.config(text=f"自动检测：{len(self.result.measurements)} 个条带。不准就改用 ROI。")
        except (ValueError, OSError) as exc:
            messagebox.showerror("无法分析", str(exc))

    def run_roi_quantify(self) -> None:
        if self.working_bgr is None:
            return
        if self.roi is None:
            self.tool.set("roi")
            messagebox.showinfo("提示", "还没有 ROI。已切换到“框选 ROI”，请在预览区拖拽画框。")
            return
        try:
            self.result = quantify_roi_lanes(
                to_gray(self.working_bgr),
                self.roi,
                self.separators,
                background_mode=self.background_mode.get(),  # type: ignore[arg-type]
            )
            self._refresh_band_list()
            self._refresh_view(keep_overlays=False)
            self.status.config(text=f"ROI 定量完成：{len(self.result.measurements)} 个泳道")
        except ValueError as exc:
            messagebox.showerror("ROI 定量失败", str(exc))

    def show_profile(self) -> None:
        if self.working_bgr is None or self.roi is None:
            messagebox.showinfo("提示", "请先框选 ROI。")
            self.tool.set("roi")
            return
        profile, _ = roi_intensity_profile(
            to_gray(self.working_bgr),
            self.roi,
            background_mode=self.background_mode.get(),  # type: ignore[arg-type]
        )
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
        self._refresh_view(keep_overlays=False)

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
            self.norm_preview.insert(tk.END, "尚无归一化结果。")
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
        if self.roi is not None or all(m.band == 1 for m in self.result.measurements):
            ordered = sorted(self.result.measurements, key=lambda m: m.lane)
            intensities = [m.integrated_density for m in ordered]
            backgrounds = [m.background for m in ordered]
            means = [m.mean_intensity for m in ordered]
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

    def delete_history_item(self) -> None:
        selection = list(self.history_list.curselection())
        if selection:
            self.session.remove_at(selection[0])
            self._refresh_history_list()

    def export_loading_control(self) -> None:
        if not self.session.loading_controls():
            messagebox.showinfo("提示", "历史中还没有内参数据。")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], initialfile="loading_control.csv")
        if path:
            self.session.export_loading_control_csv(path)

    def import_loading_control(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("所有文件", "*.*")])
        if not path:
            return
        try:
            count = self.session.import_loading_control_csv(path)
            self._refresh_history_list()
            messagebox.showinfo("导入完成", f"已导入 {count} 组数据")
        except ValueError as exc:
            messagebox.showerror("导入失败", str(exc))

    def export_normalized(self) -> None:
        rows = self.session.normalize_all()
        if not rows:
            messagebox.showinfo("提示", "没有可导出的归一化结果。")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], initialfile="normalized_results.csv")
        if path:
            export_normalized_csv(rows, path)

    def export_current(self) -> None:
        if not self.image_path or not self.result:
            return
        selected = filedialog.askdirectory(title="选择结果保存目录", initialdir=str(self.output_dir or self.image_path.parent))
        if not selected:
            return
        self.output_dir = Path(selected)
        csv_path, preview_path = export_result(self.result, self.image_path, self.output_dir)
        messagebox.showinfo("导出完成", f"CSV：{csv_path}\n预览：{preview_path}")

    def export_presentation(self) -> None:
        if self.working_bgr is None and self.result is None:
            return
        path = filedialog.asksaveasfilename(
            title="导出展示图",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("TIFF", "*.tif")],
            initialfile=f"{(self.image_path or Path('figure')).stem}_figure.png",
        )
        if not path:
            return
        if self.export_annotated.get() and self.result is not None:
            image = self.result.annotated_image
        else:
            image = self.working_bgr
        try:
            saved = export_figure(image, path, roi=self.roi, pad=12)
            messagebox.showinfo("导出完成", f"展示图已保存：\n{saved}\n" + ("（按当前 ROI 裁切）" if self.roi else "（整图）"))
        except (ValueError, OSError) as exc:
            messagebox.showerror("导出失败", str(exc))


def main() -> None:
    AnalyzerApp().mainloop()


if __name__ == "__main__":
    main()

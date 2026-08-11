# WB 灰度统计工具

一个面向 Windows 的 Western Blot（WB）条带灰度分析工具。PySide6 桌面界面 + 原生位深定量核心：16 位 TIFF 不降位深直接定量，ROI 泳道划分、强度剖面质控、内参归一化与 fold change 一步完成。

> 当前版本仍建议由实验人员复核结果；曝光饱和、强噪声或背景极不均匀的图片可能需要调整参数、旋转，或改用 ROI 半自动路径。

## 功能

- **现代深色界面**（PySide6）：图像画布支持滚轮缩放、拖拽平移、适应窗口/1:1，右侧控制面板与底部剖面面板可停靠、可折叠
- **16 位 TIFF 原生支持**：加载与定量都在原始位深（8/16 位）上进行，显示层单独做 8 位映射，不损失定量精度
- **可调 ROI**：拖拽框选，8 向手柄改大小、整体拖动换位置；橙色泳道分隔线按强度剖面峰谷自动划分，可拖动微调、双击添加、右键删除
- **双向同步剖面面板**（pyqtgraph）：ROI 列强度剖面与画布实时联动，分隔线在画布和剖面图里都能拖；定量后可逐泳道查看剖面与基线
- **标尺对齐**：沿条带方向拖一条线，一键“旋正为水平/垂直”；旋转后 ROI 与分隔线自动重映射，无需重画
- **ROI 定量**（推荐路径）：ImageJ 式基线扣除逐泳道定量；另有整图自动检测 + 补画条带兜底
- **结果表格**：逐条带位置、平均灰度、背景、积分密度、饱和占比；选中行高亮对应条带，可删除误检
- **归一化与 fold change**：内参（Loading Control）与目标蛋白按组配对，自动计算 normalized（Target ÷ LC）与相对对照泳道的 fold change；历史支持导出/导入 CSV 复用
- **导出**：条带 CSV + 标注预览图；或按 ROI 裁切的展示图（可选带检测标注）

## 快捷键

| 快捷键 | 功能 |
| --- | --- |
| `1` / `2` / `3` / `4` | 框选 ROI / 标尺对齐 / 补画条带 / 调整 |
| `Ctrl+O` | 打开图片 |
| `Ctrl+E` | 导出数据（CSV + 标注图） |
| `Ctrl+Shift+E` | 导出展示图 |
| `Ctrl+0` / `Ctrl+1` | 适应窗口 / 1:1 |
| `Ctrl+Z` | 撤销 ROI / 分隔线操作 |
| `Delete` | 清除 ROI |
| `F1` | 使用说明 |

## Windows 软件（推荐）

形态是 **绿色免安装包**：解压后双击 `WB-Analyzer.exe` 即可使用，无需先装 Python。

### 获取现成安装包

1. 打开本仓库 GitHub 页面的 **Actions** → 工作流 **Build Windows EXE**
2. 选一次成功的运行记录，下载 Artifact：`WB-Analyzer-Windows`
3. 解压 zip 后进入文件夹，双击 `WB-Analyzer.exe`

也可在本机自己打包（需 Windows + Python 3.10+）：

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1
```

> 首次运行若被 Windows SmartScreen 拦截，选“更多信息 → 仍要运行”。请保留整个 `WB-Analyzer` 文件夹。

### 软件内推荐流程

1. 打开图片，或把图片直接拖进窗口（支持 8/16 位 TIF/TIFF、PNG、JPG、BMP）
2. 图片歪斜时，用 **标尺对齐**（`2`）沿条带上沿拖一条线，点“旋正为水平/垂直”；ROI 会随旋转自动重映射
3. 用 **框选 ROI**（`1`）拖出定量区域，分隔线按剖面峰谷自动划分泳道；期望泳道数可在“调整”页指定，分隔线在画布或底部剖面图中都能拖动微调，“调整”模式（`4`）下双击添加、右键删除
4. 点 **ROI 定量**：逐泳道基线扣除，结果表、画布条带框与泳道剖面同时更新；漏检可用 **补画条带**（`3`）补充，误检在结果表删除
5. 在 **归一化** 页把内参与目标分别加入历史（选好类型/蛋白名/分组），同组自动配对计算 normalized 与 fold change，可设对照泳道
6. **导出数据**（`Ctrl+E`，条带 CSV + 标注图）或 **导出展示图**（`Ctrl+Shift+E`，按 ROI 裁切）

## 开发模式安装（Python）

需要 Python 3.10 或更高版本：

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
wb-analyzer                              # 或：python -m wb_analyzer.ui.app [图片路径]
```

命令行单张自动分析：

```powershell
wb-analyze .\example.tif --output .\results
wb-analyze .\example.tif -o .\results --sensitivity 0.8 --background dark_bands
```

支持 PNG、JPG、JPEG、BMP 和 TIFF。

## 参数说明（CLI）

| 参数 | 含义 |
| --- | --- |
| `sensitivity` | 检测灵敏度。默认 `1.0`；越小越容易检出弱条带，越大越严格 |
| `kernel-size` | 背景校正形态学核（奇数）。省略则按图片尺寸自动估计 |
| `background` | `auto` / `dark_bands`（深带浅底）/ `light_bands`（浅带深底） |

## CSV 字段

### 条带结果 `*_bands.csv`

| 字段 | 含义 |
| --- | --- |
| `lane` / `band` | 泳道编号 / 条带编号 |
| `x`, `y`, `width`, `height` | 包围框（像素） |
| `mean_intensity` | 区域平均灰度 |
| `background` | 背景估计 |
| `integrated_density` | 背景校正积分灰度（越大信号越强） |
| `saturated_fraction` | 疑似饱和像素占比 |

### 内参 / 归一化

- 内参 CSV：`kind,name,group,lane,integrated_density,...`，可供下次导入复用
- 归一化 CSV：`target / loading_control / normalized / fold_change`（normalized = Target ÷ LC；fold_change = 相对对照泳道，对照泳道 = 1.0）

## 开发

```bash
pip install -e '.[dev]'
pytest
```

关键模块：

- `wb_analyzer/imageio.py`：原生位深加载（BlotImage）、旋转与坐标重映射
- `wb_analyzer/quant.py`：泳道定量核心（列剖面、基线扣除、原生位深测量/重测）
- `wb_analyzer/analysis.py`：整图自动检测
- `wb_analyzer/roi.py`：ROI、剖面峰谷分隔、旋转工具
- `wb_analyzer/normalize.py`：内参历史、归一化与 fold change
- `wb_analyzer/ui/`：PySide6 界面（`canvas` 画布、`profile_panel` 剖面、`results_panel` 结果、`normalize_panel` 归一化、`main_window` 主窗口）
- `desktop/` + `scripts/build_windows.ps1`：Windows 打包

本机打包（也可直接用 spec）：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1
# 或
python -m PyInstaller desktop/wb-analyzer.spec --noconfirm
```

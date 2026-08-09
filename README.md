# WB 灰度统计工具

一个面向 Windows 的 Western Blot（WB）条带灰度分析工具。支持自动检测、旋转对齐、ROI 泳道定量、强度剖面质控，以及内参归一化与历史复用。

> 当前版本仍建议由实验人员复核结果；曝光饱和、强噪声或背景极不均匀的图片可能需要调整参数、旋转，或改用 ROI 半自动路径。

## 功能

- Windows 桌面软件（可打包为 `.exe`，也可用 Python 直接运行）
- **旋转对齐**：预处理滑条纠正倾斜泳道
- **自动检测**：背景校正、泳道/条带识别，可调灵敏度与形态学核
- **ROI + 可拖拽分隔线**：框选一整排条带，按重复数切分并手动微调泳道
- **强度剖面**：查看 ROI 横向信号峰，检查分隔线与背景是否合理
- **内参归一化**：保存 Loading Control / Target 历史，导出或导入内参 CSV 复用
- GUI 支持删除误检、自动模式下画框补检
- 导出条带 CSV、标注预览图、归一化结果 CSV
- 命令行可调用同一分析核心做单张自动分析

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

1. 选择 WB 图片
2. **预处理**：微调旋转，使泳道接近垂直；设置背景模式
3. 二选一：
   - **自动检测**：点“自动分析”，在条带列表中删误检 / 画框补检
   - **ROI 泳道**：切换模式后拖拽青色 ROI，拖动橙色分隔线对齐，点“ROI 定量”
4. 点 **强度剖面** 检查峰形与分隔线
5. 在 **内参归一化** 页：先把内参（如 Actin）加入历史，再分析目标蛋白并加入；可导出/导入内参 CSV 复用
6. 导出条带结果与归一化结果

## 开发模式安装（Python）

需要 Python 3.10 或更高版本：

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
wb-analyzer
```

命令行单张自动分析：

```powershell
wb-analyze .\example.tif --output .\results
wb-analyze .\example.tif -o .\results --sensitivity 0.8 --background dark_bands
```

支持 PNG、JPG、JPEG、BMP 和 TIFF。

## 参数说明

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
- 归一化 CSV：`target / loading_control / normalized`（Target ÷ Loading Control）

## 开发

```bash
pip install -e '.[dev]'
pytest
```

关键模块：

- `wb_analyzer/analysis.py`：自动检测
- `wb_analyzer/roi.py`：旋转、ROI 泳道定量、强度剖面
- `wb_analyzer/normalize.py`：内参历史与归一化
- `wb_analyzer/gui.py`：桌面界面
- `desktop/` + `scripts/build_windows.ps1`：Windows 打包

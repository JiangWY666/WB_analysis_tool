# WB 灰度统计工具

一个面向 Windows 的 Western Blot（WB）条带灰度分析工具。支持自动检测、旋转对齐、ROI 泳道定量、强度剖面质控，以及内参归一化与历史复用。

> 当前版本仍建议由实验人员复核结果；曝光饱和、强噪声或背景极不均匀的图片可能需要调整参数、旋转，或改用 ROI 半自动路径。

## 功能

- Windows 桌面软件（可打包为 `.exe`，也可用 Python 直接运行）
- **拖拽导入**：把图片直接拖进预览区打开
- **可调预览窗格**：左右拖拽分隔条调整预览/侧栏宽度
- **标尺对齐**：像 PS 标尺一样拖两点测倾角，一键旋正水平/垂直；旋转时有十字参考线
- **框选 ROI**：顶部工具栏显式“框选 ROI”，不再靠隐藏模式切换
- **自动检测**：投影 + 轮廓混合识别（仍建议人工复核，复杂膜优先 ROI）
- **强度剖面 / 内参归一化**：质控与 Target÷LC
- **导出展示图**：按 ROI 裁切导出论文/汇报用图，可选手动带标注

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

1. 打开图片，或把图片拖进预览区
2. 用 **标尺对齐** 沿条带上沿/泳道边拖一条线，点“旋正为水平/垂直”（也可看绿色参考线手动转）
3. 点顶部 **框选 ROI**，拖出青色框；再切到 **调整分隔线** 拖橙色竖线
4. 点 **ROI 定量**（复杂膜优先这条路径；自动分析仅作粗检）
5. **强度剖面** 检查峰形；归一化页保存 LC/Target
6. **导出展示图**（按 ROI 裁切）或导出数据 CSV

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

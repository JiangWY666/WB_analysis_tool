# WB 灰度统计工具

一个面向 Windows 的 Western Blot（WB）条带自动灰度分析工具。选择图片后，程序会自动判断明暗背景、校正不均匀背景、识别泳道与条带，并导出带标注的预览图和 CSV 数据，减少 ImageJ 中重复框选和测量操作。

> 当前版本是可运行的 MVP。自动识别结果仍应由实验人员检查；曝光饱和、弯曲泳道、强噪声或背景极不均匀的图片可能需要重新拍摄、调整参数，或手动删框/补框。

## 功能

- Windows 桌面软件（可打包为 `.exe`，也可用 Python 直接运行）
- 自动识别“深色条带/浅色背景”和“浅色条带/深色背景”，也支持手动指定
- 自动背景校正、泳道检测和条带检测
- 可调灵敏度与形态学核大小，适配不同曝光与条带粗细
- 输出每个条带的包围框、面积、平均灰度、背景校正积分灰度和饱和占比
- GUI 支持删除误检、拖拽画框补检后再导出
- 自动保存 CSV 结果及标注预览图
- 同一个分析核心可通过命令行调用，方便批处理和后续扩展

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

完成后得到：

- `dist\WB-Analyzer\WB-Analyzer.exe`：主程序
- `dist\WB-Analyzer-Windows.zip`：可直接分发的压缩包

> 首次运行若被 Windows SmartScreen 拦截，选“更多信息 → 仍要运行”。当前未做代码签名。请保留整个 `WB-Analyzer` 文件夹，不要只拷贝单个 exe。

### 软件内使用步骤

1. 打开 `WB-Analyzer.exe`
2. 选择 WB 图片（PNG / JPG / BMP / TIFF）
3. 需要时调节灵敏度、形态学核、背景模式
4. 点击 **自动分析**
5. 检查预览：删除误检，或勾选「拖拽画框添加」补框
6. 点击 **导出结果**，得到 CSV 与标注预览图

## 开发模式安装（Python）

需要 Python 3.10 或更高版本。Windows PowerShell 中运行：

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
wb-analyzer
```

也可使用命令行直接分析单张图片：

```powershell
wb-analyze .\example.tif --output .\results
wb-analyze .\example.tif -o .\results --sensitivity 0.8 --background dark_bands
wb-analyze .\example.tif -o .\results --kernel-size 31 --background auto
```

支持 PNG、JPG、JPEG、BMP 和 TIFF。结果目录包含：

- `<原文件名>_bands.csv`：条带数据
- `<原文件名>_annotated.png`：检测框和编号预览

## 参数说明

| 参数 | 含义 |
| --- | --- |
| `sensitivity` | 检测灵敏度。默认 `1.0`；越小越容易检出弱条带，越大越严格 |
| `kernel-size` | 背景校正形态学核（奇数）。省略则按图片尺寸自动估计 |
| `background` | `auto` / `dark_bands`（深带浅底）/ `light_bands`（浅带深底） |

## CSV 字段

| 字段 | 含义 |
| --- | --- |
| `lane` / `band` | 从左到右的泳道编号、从上到下的条带编号 |
| `x`, `y`, `width`, `height` | 条带包围框（像素） |
| `area` | 包围框面积（像素） |
| `mean_intensity` | 原图中条带区域的平均灰度（0–255） |
| `background` | 条带四周估算的局部背景灰度 |
| `integrated_density` | 背景校正积分灰度；数值越大表示条带信号越强 |
| `saturated_fraction` | 区域内疑似饱和像素占比；GUI 中橙色框表示可能饱和 |

## 开发

```bash
pip install -e '.[dev]'
pytest
```

打包相关文件：

- `packaging/run_gui.py`：exe 入口
- `packaging/wb-analyzer.spec`：PyInstaller 配置
- `scripts/build_windows.ps1`：Windows 一键打包
- `.github/workflows/build-windows.yml`：CI 自动出 Windows 包

分析算法位于 `wb_analyzer/analysis.py`，桌面界面位于 `wb_analyzer/gui.py`。后续可在此基础上增加安装程序（如 Inno Setup）、代码签名、检测框拖拽微调和内参归一化。

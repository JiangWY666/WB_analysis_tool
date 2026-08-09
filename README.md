# WB 灰度统计工具

一个面向 Windows 的 Western Blot（WB）条带自动灰度分析工具。选择图片后，程序会自动判断明暗背景、校正不均匀背景、识别泳道与条带，并导出带标注的预览图和 CSV 数据，减少 ImageJ 中重复框选和测量操作。

> 当前版本是可运行的 MVP。自动识别结果仍应由实验人员检查；曝光饱和、弯曲泳道、强噪声或背景极不均匀的图片可能需要重新拍摄或调整参数。

## 功能

- Windows 桌面界面（Tkinter，无需浏览器）
- 自动识别“深色条带/浅色背景”和“浅色条带/深色背景”
- 自动背景校正、泳道检测和条带检测
- 输出每个条带的包围框、面积、平均灰度和背景校正积分灰度
- 自动保存 CSV 结果及标注预览图
- 同一个分析核心可通过命令行调用，方便批处理和后续扩展

## 安装与运行

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
```

支持 PNG、JPG、JPEG、BMP 和 TIFF。结果目录包含：

- `<原文件名>_bands.csv`：条带数据
- `<原文件名>_annotated.png`：检测框和编号预览

## CSV 字段

| 字段 | 含义 |
| --- | --- |
| `lane` / `band` | 从左到右的泳道编号、从上到下的条带编号 |
| `x`, `y`, `width`, `height` | 条带包围框（像素） |
| `area` | 包围框面积（像素） |
| `mean_intensity` | 原图中条带区域的平均灰度（0–255） |
| `background` | 条带四周估算的局部背景灰度 |
| `integrated_density` | 背景校正积分灰度；数值越大表示条带信号越强 |

## 开发

```bash
pip install -e '.[dev]'
pytest
```

分析算法位于 `wb_analyzer/analysis.py`，桌面界面位于 `wb_analyzer/gui.py`。后续可在此基础上增加手动修正检测框、内参归一化、分组统计和 Excel 报告。

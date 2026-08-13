"""打包元数据的一致性：版本号只有一个真相，spec 的子模块清单不许漏。

两条都是"改了代码但忘了改配置"才会踩的，人工复核基本查不出来：版本号对不上
要等到用户报了个对不上号的版本才发现；spec 漏登记一个模块，要等到 PyInstaller
的静态分析恰好也找不到它（动态 import、或某天改成延迟 import）时才炸。
"""

import re
from pathlib import Path

import wb_analyzer

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
SPEC = ROOT / "desktop" / "wb-analyzer.spec"


def test_pyproject_version_matches_the_package() -> None:
    """`pyproject.toml` 与 `wb_analyzer.__version__` 必须同步。

    界面的「关于」框读的是后者，发布物的文件名跟的是前者：两边一漂移，用户手上
    那个 zip 叫什么版本、软件里显示什么版本就对不上了。
    """
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    assert match is not None, "pyproject.toml 里找不到 version"
    assert match.group(1) == wb_analyzer.__version__


def test_spec_lists_every_wb_analyzer_module() -> None:
    """spec 的 hiddenimports 逐个列了子模块，那就一个都不能少。"""
    modules = {"wb_analyzer"}
    for path in sorted((ROOT / "wb_analyzer").rglob("*.py")):
        parts = path.relative_to(ROOT).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules.add(".".join(parts))

    listed = set(re.findall(r'"(wb_analyzer(?:\.[a-z0-9_]+)*)"', SPEC.read_text(encoding="utf-8")))
    assert not modules - listed, f"desktop/wb-analyzer.spec 漏了：{sorted(modules - listed)}"
    assert not listed - modules, f"desktop/wb-analyzer.spec 列了不存在的模块：{sorted(listed - modules)}"

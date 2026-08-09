# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Windows desktop app."""

from PyInstaller.utils.hooks import collect_all

block_cipher = None

pillow_datas, pillow_binaries, pillow_hiddenimports = collect_all("PIL")
cv2_datas, cv2_binaries, cv2_hiddenimports = collect_all("cv2")

a = Analysis(
    ["run_gui.py"],
    pathex=[],
    binaries=pillow_binaries + cv2_binaries,
    datas=pillow_datas + cv2_datas,
    hiddenimports=pillow_hiddenimports + cv2_hiddenimports + [
        "wb_analyzer",
        "wb_analyzer.analysis",
        "wb_analyzer.export",
        "wb_analyzer.gui",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="WB-Analyzer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # GUI app: no black console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="WB-Analyzer",
)

# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Windows desktop app (PySide6 GUI)."""

from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

pillow_datas, pillow_binaries, pillow_hiddenimports = collect_all("PIL")
cv2_datas, cv2_binaries, cv2_hiddenimports = collect_all("cv2")

# pyqtgraph resolves its Qt binding dynamically; pull in every runtime
# submodule except the heavy optional ones we never import.
_PG_SKIP = (
    "pyqtgraph.examples",
    "pyqtgraph.tests",
    "pyqtgraph.opengl",
    "pyqtgraph.jupyter",
)
pyqtgraph_hiddenimports = collect_submodules(
    "pyqtgraph", filter=lambda name: not name.startswith(_PG_SKIP)
)

a = Analysis(
    ["run_gui.py"],
    pathex=[],
    binaries=pillow_binaries + cv2_binaries,
    datas=pillow_datas + cv2_datas,
    hiddenimports=pillow_hiddenimports + cv2_hiddenimports + pyqtgraph_hiddenimports + [
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "wb_analyzer",
        "wb_analyzer.analysis",
        "wb_analyzer.cli",
        "wb_analyzer.export",
        "wb_analyzer.imageio",
        "wb_analyzer.normalize",
        "wb_analyzer.quant",
        "wb_analyzer.roi",
        "wb_analyzer.ui",
        "wb_analyzer.ui.app",
        "wb_analyzer.ui.canvas",
        "wb_analyzer.ui.errorlog",
        "wb_analyzer.ui.main_window",
        "wb_analyzer.ui.normalize_panel",
        "wb_analyzer.ui.profile_panel",
        "wb_analyzer.ui.results_panel",
        "wb_analyzer.ui.tablefmt",
        "wb_analyzer.ui.theme",
        "wb_analyzer.ui.workers",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Legacy GUI stacks we no longer ship.
        "tkinter",
        "matplotlib",
        # Qt modules not used by the app or pyqtgraph core (keeps the
        # bundle small: WebEngine alone is ~180 MB).
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtWebEngineQuick",
        "PySide6.QtQml",
        "PySide6.QtQuick",
        "PySide6.QtQuickWidgets",
        "PySide6.QtQuick3D",
        "PySide6.QtNetwork",
        "PySide6.Qt3DCore",
        "PySide6.Qt3DRender",
        "PySide6.Qt3DInput",
        "PySide6.Qt3DLogic",
        "PySide6.Qt3DExtras",
        "PySide6.Qt3DAnimation",
        "PySide6.QtMultimedia",
        "PySide6.QtMultimediaWidgets",
        "PySide6.QtCharts",
        "PySide6.QtDataVisualization",
        "PySide6.QtGraphs",
        "PySide6.QtPdf",
        "PySide6.QtPdfWidgets",
        "PySide6.QtWebChannel",
        "PySide6.QtWebSockets",
        "PySide6.QtWebView",
        "PySide6.QtPositioning",
        "PySide6.QtLocation",
        "PySide6.QtSensors",
        "PySide6.QtSerialPort",
        "PySide6.QtSerialBus",
        "PySide6.QtBluetooth",
        "PySide6.QtNfc",
        "PySide6.QtSql",
        "PySide6.QtTest",
        "PySide6.QtRemoteObjects",
        "PySide6.QtScxml",
        "PySide6.QtStateMachine",
        "PySide6.QtTextToSpeech",
        "PySide6.QtHelp",
        "PySide6.QtDesigner",
        "PySide6.QtUiTools",
        "PySide6.QtDBus",
        # Optional pyqtgraph integrations.
        "IPython",
        "jupyter_rfb",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Module-level excludes cannot stop Qt DLLs that ride in via plugin
# dependency chains (Qml/Quick/Network/Pdf), nor OpenCV's video codecs.
# Drop them from the collected binaries/datas to keep the bundle small.
_DROP_KEYS = (
    "opencv_videoio_ffmpeg",         # video codecs: unused for still images
    "qt6network",
    "qt6pdf",
    "qt6qml",
    "qt6quick",
    "qt6virtualkeyboard",
    "plugins\\tls\\",                # TLS backends need Qt6Network
    "plugins\\platforminputcontexts\\",  # touch virtual keyboard
    "plugins\\imageformats\\qpdf",   # PDF image plugin needs Qt6Pdf
    "plugins\\qmltooling",
)


def _keep(entry):
    dest = str(entry[0]).lower().replace("/", "\\")
    return not any(key in dest for key in _DROP_KEYS)


a.binaries = [b for b in a.binaries if _keep(b)]
a.datas = [d for d in a.datas if _keep(d)]

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

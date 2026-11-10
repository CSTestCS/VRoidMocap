# PyInstaller spec - build with:  pyinstaller VRoidMocap.spec
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

import glob

datas = [("vroidmocap/web", "vroidmocap/web")]
# the MediaPipe models are packed; the large whole-body model (.onnx) is
# downloaded by the app on first use
datas += [(f, "vroidmocap/assets/models") for f in glob.glob("vroidmocap/assets/models/*")
          if f.endswith((".task", ".tflite"))]
datas += collect_data_files("mediapipe", excludes=["**/*.pyc"])
datas += collect_data_files("webview")
binaries = collect_dynamic_libs("mediapipe") + collect_dynamic_libs("onnxruntime")
hiddenimports = collect_submodules("mediapipe.tasks") + collect_submodules("webview") + ["onnxruntime"]

a = Analysis(
    ["run_app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "scipy", "pandas", "IPython", "PyQt5", "PyQt6", "PySide2", "PySide6"],
    noarchive=False,
)
pyz = PYZ(a.pure)
# console build: same app, for command-line use and diagnostics
cli = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name="VRoidMocapCLI",
    icon="packaging/icon.ico",
    console=True,
    upx=False,
    runtime_tmpdir=None,
)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name="VRoidMocap",
    icon="packaging/icon.ico",
    console=False,
    upx=False,
    runtime_tmpdir=None,
)

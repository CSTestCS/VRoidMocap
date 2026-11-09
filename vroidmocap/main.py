"""VRoid Mocap - desktop entry point.

Double-clicking the .exe opens the app window. Command line use:

  VRoidMocap.exe                         open the app
  VRoidMocap.exe --browser               use the default web browser instead of a window
  VRoidMocap.exe analyze VIDEO           pass 1 only (prints the characters found)
  VRoidMocap.exe capture VIDEO VRM CHAR OUT.fbx [--in-place]
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
import webbrowser


class Api:
    """Functions callable from the page as window.pywebview.api.*"""

    def __init__(self):
        self.window = None

    def pick_file(self, kind):
        import webview
        types = ("Video files (*.mp4;*.mov;*.mkv;*.avi;*.webm;*.m4v;*.wmv)", "All files (*.*)") \
            if kind == "video" else ("VRM avatar (*.vrm)", "All files (*.*)")
        res = self.window.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=False, file_types=types)
        if not res:
            return None
        return res[0] if isinstance(res, (list, tuple)) else res

    def save_file(self, name):
        import webview
        res = self.window.create_file_dialog(webview.SAVE_DIALOG, save_filename=name,
                                             file_types=("FBX file (*.fbx)",))
        if not res:
            return None
        path = res[0] if isinstance(res, (list, tuple)) else res
        if not path.lower().endswith(".fbx"):
            path += ".fbx"
        return path

    def open_folder(self, path):
        folder = os.path.dirname(path)
        if sys.platform == "win32":
            os.startfile(folder)  # noqa: S606


def run_gui(browser=False, port=0):
    from .server import start_server
    srv = start_server(port)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    if not browser:
        try:
            import webview
            api = Api()
            win = webview.create_window("VRoid Mocap", url, js_api=api, width=1440, height=920,
                                        min_size=(1000, 680), background_color="#101317")
            api.window = win
            webview.start(debug=bool(os.environ.get("VRM_MOCAP_DEBUG")))
            return
        except Exception as e:  # noqa: BLE001
            print(f"Could not open the app window ({e}); using the web browser instead.")
    print(f"VRoid Mocap is running at {url}  (close this window to quit)")
    webbrowser.open(url)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def cli(argv):
    ap = argparse.ArgumentParser(prog="VRoidMocap")
    sub = ap.add_subparsers(dest="cmd")
    a = sub.add_parser("analyze")
    a.add_argument("video")
    c = sub.add_parser("capture")
    c.add_argument("video")
    c.add_argument("vrm")
    c.add_argument("character", type=int, help="character number from 'analyze' (1 = first)")
    c.add_argument("out")
    c.add_argument("--in-place", action="store_true")
    c.add_argument("--smoothing", type=float, default=1.0)
    c.add_argument("--no-foot-lock", action="store_true", help="do not pin planted feet (raw leg angles)")
    g = sub.add_parser("gui")
    g.add_argument("--browser", action="store_true")
    g.add_argument("--port", type=int, default=0)
    ap.add_argument("--browser", action="store_true")
    ap.add_argument("--port", type=int, default=0)
    args = ap.parse_args(argv)

    def prog(f, m):
        sys.stdout.write(f"\r[{f * 100:5.1f}%] {m:90.90s}")
        sys.stdout.flush()

    if args.cmd in (None, "gui"):
        return run_gui(args.browser, args.port)
    from .server import STATE
    if args.cmd == "analyze":
        STATE.open(video=args.video)
        if STATE.analysis is None:
            from .core.analysis import Analyzer
            STATE.analysis = Analyzer(args.video, STATE.workdir, prog).run()
        print()
        for ch in STATE.analysis["characters"]:
            print(f"{ch['id'] + 1}: {ch['name']}  screen time {ch['screen_time_s']}s  "
                  f"appearances {ch['appearances']}  relative height {ch.get('relative_height')}")
    elif args.cmd == "capture":
        STATE.open(video=args.video, vrm=args.vrm)
        if STATE.analysis is None:
            from .core.analysis import Analyzer
            STATE.analysis = Analyzer(args.video, STATE.workdir, prog).run()
        STATE.start_capture(args.character - 1, args.smoothing, not args.no_foot_lock)
        while not STATE.job.done:
            prog(STATE.job.progress, STATE.job.message)
            time.sleep(0.5)
        print()
        if STATE.job.error:
            print("Error:", STATE.job.error)
            return 1
        STATE.export(args.character - 1, args.out, in_place=args.in_place)
        print("Wrote", args.out)
    return 0


def setup_logging():
    """A windowed .exe has no console: send output to a log file instead."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    from .server import data_root
    f = open(os.path.join(data_root(), "vroidmocap.log"), "a", buffering=1, encoding="utf-8")
    sys.stdout = sys.stdout or f
    sys.stderr = sys.stderr or f


def selftest():
    """Load every model and run it once (used by the build to verify the exe)."""
    import numpy as np
    from .core.detect import Embedder, HandDetector, PoseDetector
    img = np.zeros((256, 256, 3), np.uint8)
    for name, make in (("pose_full", lambda: PoseDetector(False)), ("pose_heavy", lambda: PoseDetector(True)),
                       ("hand", HandDetector), ("embed", Embedder)):
        m = make()
        (m.embed if name == "embed" else m.detect)(img)
        m.close()
        print("ok", name, flush=True)
    import cv2
    print("opencv", cv2.__version__, "ffmpeg backend", "FFMPEG" in cv2.getBuildInformation())
    print("SELFTEST PASSED", flush=True)
    return 0


def main():
    multiprocessing.freeze_support()
    setup_logging()
    if sys.argv[1:2] == ["selftest"]:
        sys.exit(selftest())
    sys.exit(cli(sys.argv[1:]) or 0)


if __name__ == "__main__":
    main()

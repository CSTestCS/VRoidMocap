"""Local HTTP server: serves the UI and runs analysis / capture jobs."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .core.detect import resource_dir

WEB_DIR = os.path.join(resource_dir(), "web")
MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8", ".json": "application/json", ".png": "image/png",
        ".jpg": "image/jpeg", ".svg": "image/svg+xml", ".ico": "image/x-icon"}


def data_root():
    root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.cache")
    d = os.path.join(root, "VRoidMocap")
    os.makedirs(d, exist_ok=True)
    return d


class Job:
    def __init__(self, kind, fn):
        self.kind = kind
        self.progress = 0.0
        self.message = "Starting..."
        self.error = None
        self.result = None
        self.done = False
        self.cancelled = False
        self.thread = threading.Thread(target=self._run, args=(fn,), daemon=True)
        self.thread.start()

    def _run(self, fn):
        try:
            self.result = fn(self._report, lambda: self.cancelled)
            self.progress = 1.0
        except Exception as e:  # noqa: BLE001
            from .core.analysis import Cancelled
            if isinstance(e, Cancelled):
                self.error = "Cancelled"
            else:
                traceback.print_exc()
                self.error = f"{type(e).__name__}: {e}"
        finally:
            self.done = True

    def _report(self, frac, msg):
        self.progress = float(frac)
        self.message = msg

    def status(self):
        return {"kind": self.kind, "progress": self.progress, "message": self.message,
                "error": self.error, "done": self.done}


class AppState:
    def __init__(self):
        self.video = None
        self.vrm = None
        self.skel = None
        self.workdir = None
        self.analysis = None
        self.job = None
        self.anims = {}
        self.frames = None
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ setup
    def open(self, video=None, vrm=None):
        from .core.video import FrameServer, VideoReader
        from .core.vrm import load_vrm_skeleton
        out = {}
        if vrm:
            if not os.path.isfile(vrm):
                raise ValueError(f"Avatar file not found: {vrm}")
            self.skel = load_vrm_skeleton(vrm)
            self.vrm = vrm
            out["vrm"] = {"path": vrm, "title": self.skel.title, "version": self.skel.version,
                          "bones": len(self.skel.bones)}
            self.anims = {}  # animations are avatar specific
            if self.workdir:
                for name in os.listdir(self.workdir):
                    if name.startswith("anim_c"):
                        os.remove(os.path.join(self.workdir, name))
        if video:
            if not os.path.isfile(video):
                raise ValueError(f"Video not found: {video}")
            rd = VideoReader(video)
            info = {"frames": rd.frame_count, "fps": rd.fps, "width": rd.width, "height": rd.height}
            rd.close()
            st = os.stat(video)
            key = hashlib.sha1(f"{os.path.abspath(video)}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:16]
            self.workdir = os.path.join(data_root(), "work", key)
            os.makedirs(self.workdir, exist_ok=True)
            self.video = video
            self.analysis = None
            self.anims = {}
            if self.frames:
                self.frames.close()
            self.frames = FrameServer(video)
            p = os.path.join(self.workdir, "analysis.json")
            if os.path.exists(p):
                from .core.analysis import ANALYSIS_VERSION
                with open(p) as f:
                    a = json.load(f)
                if a.get("version") == ANALYSIS_VERSION:
                    self.analysis = a
            out["video"] = {"path": video, **info, "cached_analysis": self.analysis is not None}
        return out

    def summary(self):
        return {
            "video": self.video, "vrm": self.vrm,
            "vrm_title": self.skel.title if self.skel else None,
            "analysis": self.analysis is not None,
            "animated": sorted(self.available_anims()),
            "job": self.job.status() if self.job else None,
        }

    def available_anims(self):
        out = set(self.anims)
        if self.workdir and os.path.isdir(self.workdir) and self.skel is not None:
            for name in os.listdir(self.workdir):
                if name.startswith("anim_c") and name.endswith(".npz"):
                    try:
                        out.add(int(name[6:-4]))
                    except ValueError:
                        pass
        return out

    # ------------------------------------------------------------------ jobs
    def start_analysis(self, force=False):
        if self.job and not self.job.done:
            raise ValueError("Another task is still running")
        if not self.video:
            raise ValueError("Choose a video first")
        from .core.analysis import Analyzer

        def run(progress, cancel):
            if force:
                for name in os.listdir(self.workdir):
                    pth = os.path.join(self.workdir, name)
                    shutil.rmtree(pth) if os.path.isdir(pth) else os.remove(pth)
            res = Analyzer(self.video, self.workdir, progress, cancel).run()
            self.analysis = res
            self.anims = {}
            return {"characters": len(res["characters"])}
        self.job = Job("analysis", run)

    def start_capture(self, char_id, smoothing=1.0, foot_lock=True):
        if self.job and not self.job.done:
            raise ValueError("Another task is still running")
        if self.analysis is None:
            raise ValueError("Analyse the video first")
        if self.skel is None:
            raise ValueError("Choose a .vrm avatar first")
        from .core.capture import capture
        from .core.retarget import solve_animation

        def run(progress, cancel):
            cap_path = os.path.join(self.workdir, f"capture_c{char_id}.npz")
            if os.path.exists(cap_path):
                cap = dict(np.load(cap_path))
            else:
                cap = capture(self.video, self.workdir, char_id,
                              lambda f, m: progress(f * 0.85, m), cancel)
                np.savez_compressed(cap_path, **cap)
            anim = solve_animation(cap, self.workdir, char_id, self.skel, smoothing,
                                   lambda f, m: progress(0.85 + 0.15 * f, m), foot_lock=foot_lock)
            self._save_anim(char_id, anim)
            return {"character": char_id}
        self.job = Job("capture", run)

    def _save_anim(self, char_id, anim):
        arrays = {k: v for k, v in anim.items() if isinstance(v, np.ndarray)}
        meta = {k: v for k, v in anim.items() if not isinstance(v, np.ndarray)}
        np.savez(os.path.join(self.workdir, f"anim_c{char_id}.npz"), **arrays,
                 _meta=np.array(json.dumps(meta)))
        self.anims[char_id] = anim

    def anim(self, char_id):
        if char_id in self.anims:
            return self.anims[char_id]
        p = os.path.join(self.workdir, f"anim_c{char_id}.npz")
        if not os.path.exists(p):
            raise KeyError(char_id)
        d = np.load(p)
        anim = json.loads(str(d["_meta"]))
        for k in d.files:
            if k != "_meta":
                anim[k] = d[k]
        self.anims[char_id] = anim
        return anim

    def anim_binary(self, char_id):
        a = self.anim(char_id)
        parts = [("quats", a["quats"]), ("root", a["root"]), ("cam_pos", a["cam_pos"]),
                 ("cam_quat", a["cam_quat"]), ("cam_vfov", a["cam_vfov"]),
                 ("valid", a["valid"].astype(np.float32)), ("contacts", a["contacts"].astype(np.float32)),
                 ("lm2d", a["lm2d"]), ("vis", a["vis"]), ("h2d", a["h2d"]), ("hs", a["hs"]),
                 ("shot_of", a["shot_of"].astype(np.float32))]
        meta = {k: a[k] for k in ("fps", "frames", "width", "height", "character", "bones", "scale")}
        meta["layout"] = {}
        off = 0
        blobs = []
        for name, arr in parts:
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            meta["layout"][name] = {"offset": off, "shape": list(arr.shape)}
            blobs.append(arr.tobytes())
            off += arr.nbytes
        return meta, b"".join(blobs)

    def export(self, char_id, path, in_place=False, lock_height=False, include_camera=True):
        from .core.fbx_writer import export_animation_fbx
        return export_animation_fbx(path, self.skel, self.anim(char_id), in_place=in_place,
                                    lock_height=lock_height, include_camera=include_camera)


STATE = AppState()


class Handler(BaseHTTPRequestHandler):
    server_version = "VRoidMocap/1.0"

    def log_message(self, fmt, *args):  # quiet
        pass

    # ------------------------------------------------------------------ helpers
    def _send(self, code, body, ctype="application/json", headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def _file(self, path, ctype=None, download_name=None):
        if not os.path.isfile(path):
            return self._send(404, {"error": "not found"})
        ctype = ctype or MIME.get(os.path.splitext(path)[1].lower(), "application/octet-stream")
        size = os.path.getsize(path)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.end_headers()
        with open(path, "rb") as f:
            shutil.copyfileobj(f, self.wfile, 1 << 20)

    # ------------------------------------------------------------------ routes
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = dict(urllib.parse.parse_qsl(u.query))
        p = u.path
        try:
            if p == "/":
                return self._file(os.path.join(WEB_DIR, "index.html"))
            if p.startswith("/web/"):
                rel = os.path.normpath(p[5:]).lstrip("\\/")
                full = os.path.join(WEB_DIR, rel)
                if not os.path.abspath(full).startswith(os.path.abspath(WEB_DIR)):
                    return self._send(403, {"error": "forbidden"})
                return self._file(full)
            if p == "/api/state":
                return self._send(200, STATE.summary())
            if p == "/api/status":
                return self._send(200, STATE.job.status() if STATE.job else {"done": True})
            if p == "/api/analysis":
                if STATE.analysis is None:
                    return self._send(404, {"error": "no analysis"})
                return self._send(200, STATE.analysis)
            if p.startswith("/api/thumb/"):
                cid = int(p.split("/")[-1].split(".")[0])
                return self._file(os.path.join(STATE.workdir, "thumbs", f"char{cid}.jpg"))
            if p.startswith("/api/frame/"):
                i = int(p.split("/")[-1].split(".")[0])
                data = STATE.frames.jpeg(i, width=int(q.get("w", 960))) if STATE.frames else None
                if data is None:
                    return self._send(404, {"error": "no frame"})
                return self._send(200, data, "image/jpeg", {"Cache-Control": "max-age=3600"})
            if p == "/api/vrm":
                return self._file(STATE.vrm, "application/octet-stream")
            if p.startswith("/api/anim/"):
                parts = p.split("/")
                cid = int(parts[3])
                meta, blob = STATE.anim_binary(cid)
                if parts[4] == "meta":
                    return self._send(200, meta)
                return self._send(200, blob, "application/octet-stream")
            if p == "/api/export.fbx":
                cid = int(q["character"])
                fd, tmp = tempfile.mkstemp(suffix=".fbx")
                os.close(fd)
                STATE.export(cid, tmp, q.get("in_place") == "1", q.get("lock_height") == "1",
                             q.get("camera", "1") == "1")
                base = os.path.splitext(os.path.basename(STATE.video or "video"))[0]
                self._file(tmp, "application/octet-stream", f"{base}_character{cid + 1}.fbx")
                os.remove(tmp)
                return
            return self._send(404, {"error": "unknown route"})
        except KeyError as e:
            return self._send(404, {"error": f"not found: {e}"})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self._send(500, {"error": str(e)})

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        q = dict(urllib.parse.parse_qsl(u.query))
        p = u.path
        try:
            if p == "/api/upload":
                kind = q.get("kind", "video")
                name = os.path.basename(q.get("name", "upload.bin"))
                d = os.path.join(data_root(), "uploads")
                os.makedirs(d, exist_ok=True)
                dest = os.path.join(d, name)
                n = int(self.headers.get("Content-Length") or 0)
                with open(dest, "wb") as f:
                    left = n
                    while left > 0:
                        chunk = self.rfile.read(min(left, 1 << 20))
                        if not chunk:
                            break
                        f.write(chunk)
                        left -= len(chunk)
                res = STATE.open(video=dest) if kind == "video" else STATE.open(vrm=dest)
                return self._send(200, res)
            body = self._json_body()
            if p == "/api/open":
                return self._send(200, STATE.open(body.get("video"), body.get("vrm")))
            if p == "/api/analyze":
                STATE.start_analysis(bool(body.get("force")))
                return self._send(200, {"ok": True})
            if p == "/api/capture":
                STATE.start_capture(int(body["character"]), float(body.get("smoothing", 1.0)),
                                    bool(body.get("foot_lock", True)))
                return self._send(200, {"ok": True})
            if p == "/api/cancel":
                if STATE.job:
                    STATE.job.cancelled = True
                return self._send(200, {"ok": True})
            if p == "/api/export":
                path = body["path"]
                STATE.export(int(body["character"]), path, bool(body.get("in_place")),
                             bool(body.get("lock_height")), bool(body.get("camera", True)))
                return self._send(200, {"ok": True, "path": path})
            if p == "/api/rename":
                cid = int(body["character"])
                for c in STATE.analysis["characters"]:
                    if c["id"] == cid:
                        c["name"] = str(body["name"])[:60]
                with open(os.path.join(STATE.workdir, "analysis.json"), "w") as f:
                    json.dump(STATE.analysis, f)
                return self._send(200, {"ok": True})
            return self._send(404, {"error": "unknown route"})
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self._send(500, {"error": str(e)})


def start_server(port=0):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv

"""Video access via OpenCV (FFmpeg backend)."""
from __future__ import annotations

import json
import os
import threading

import cv2
import numpy as np


# ----------------------------------------------------------------------------- the moving picture
# A screen recording (a video playing in a browser or player window) has a still
# frame around the picture: window borders, buttons, a taskbar, black bars.
# Everything reads frames through VideoReader, so cropping there to the part of
# the picture that actually moves makes the scan, the camera, the capture and the
# viewer all work on the video itself (the still frame would make the camera
# look locked off).
_AREAS = {}


def _area_cache_path():
    root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.cache")
    d = os.path.join(root, "VRoidMocap")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "video_areas.json")


def _video_key(path):
    st = os.stat(path)
    return f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}|v4"


def find_active_area(frames):
    """frames: grey images (same size) sampled over the clip. Returns the
    rectangle (x0, y0, x1, y1) of the moving picture in those pixels, or None
    when the whole frame is the video. The frame around a video playing on a
    screen (window, buttons, taskbar, black bars) stays exactly the same from
    sample to sample; the picture does not (even a smooth sky shifts and the
    encoder's noise changes it). The rectangle must be alive right up to its
    edges: a still camera with someone dancing in front of a still background
    is not cropped down to the dancer."""
    F = np.stack(frames).astype(np.float32)
    n, h, w = F.shape
    med = np.median(F, 0)
    still = (np.abs(F - med) <= 1.5).mean(0) >= 0.9
    live = ~still
    if still.mean() < 0.03 or live.mean() < 0.2:
        return None

    def span(prof, thr):
        on = prof >= thr
        best, cur, gap = (0, 0), None, 0
        for i, v in enumerate(on):
            if v:
                cur = (cur[0], i + 1) if cur else (i, i + 1)
                gap = 0
            elif cur:
                gap += 1
                if gap > max(2, len(on) // 50):
                    if cur[1] - cur[0] > best[1] - best[0]:
                        best = cur
                    cur, gap = None, 0
        if cur and cur[1] - cur[0] > best[1] - best[0]:
            best = cur
        return best
    x0, x1, y0, y1 = 0, w, 0, h
    for _ in range(3):
        y0, y1 = span(live[:, x0:x1].mean(1), 0.5)
        if y1 <= y0:
            return None
        x0, x1 = span(live[y0:y1].mean(0), 0.5)
        if x1 <= x0:
            return None
    if x1 - x0 < w * 0.3 or y1 - y0 < h * 0.3 or (x1 - x0) * (y1 - y0) > 0.95 * w * h:
        return None
    b = max(2, min(x1 - x0, y1 - y0) // 25)
    edges = [live[y0:y1, x0:x0 + b].mean(), live[y0:y1, x1 - b:x1].mean(),
             live[y0:y0 + b, x0:x1].mean(), live[y1 - b:y1, x0:x1].mean()]
    if min(edges) < 0.6:
        return None
    out = np.ones_like(still); out[y0:y1, x0:x1] = False
    if out.any() and still[out].mean() < 0.7:
        return None
    return x0, y0, x1, y1


def active_area(path, samples=32):
    """Crop (x0, y0, x1, y1) of the moving picture in full-resolution pixels, or
    None (the whole frame is the video). Cached per video file."""
    try:
        key = _video_key(path)
    except OSError:
        return None
    if key in _AREAS:
        return _AREAS[key]
    cache = {}
    try:
        with open(_area_cache_path()) as f:
            cache = json.load(f)
    except (OSError, ValueError):
        cache = {}
    if key in cache:
        area = tuple(cache[key]) if cache[key] else None
        _AREAS[key] = area
        return area
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    area = None
    if n > 10 and W > 0:
        sc = 320.0 / W
        frames = []
        for i in np.linspace(0.02 * n, 0.98 * n, samples).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, im = cap.read()
            if ok:
                frames.append(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (320, max(1, int(H * sc))),
                                         interpolation=cv2.INTER_AREA))
        if len(frames) >= 8:
            a = find_active_area(frames)
            if a is not None:
                x0, y0, x1, y1 = (int(round(v / sc)) for v in a)
                x0, y0 = max(0, x0 - x0 % 2), max(0, y0 - y0 % 2)
                x1, y1 = min(W, x1 + x1 % 2), min(H, y1 + y1 % 2)
                area = (x0, y0, x1, y1)
    cap.release()
    _AREAS[key] = area
    cache[key] = list(area) if area else None
    try:
        tmp = _area_cache_path() + f".{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(cache, f)
        os.replace(tmp, _area_cache_path())
    except OSError:
        pass
    return area


class VideoReader:
    def __init__(self, path, crop=True):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video: {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.full_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.full_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.area = active_area(path) if crop else None
        if self.area:
            self.width, self.height = self.area[2] - self.area[0], self.area[3] - self.area[1]
        else:
            self.width, self.height = self.full_width, self.full_height
        self.pos = 0
        if self.frame_count <= 0:
            # some containers do not report a count: count by decoding
            n = 0
            while self.cap.grab():
                n += 1
            self.frame_count = n
            self.cap.release()
            self.cap = cv2.VideoCapture(path)

    def iter_frames(self, stride=1, start=0, end=None):
        """Yield (index, bgr) for frames start, start+stride, ... (sequential decode)."""
        end = self.frame_count if end is None else min(end, self.frame_count)
        self.seek(start)
        i = start
        while i < end:
            ok, frame = self.cap.read()
            if not ok:
                break
            self.pos = i + 1
            yield i, self._crop(frame)
            for _ in range(stride - 1):
                if not self.cap.grab():
                    return
                self.pos += 1
            i += stride

    def seek(self, index):
        if index != self.pos:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self.pos = index

    def read(self, index):
        if index != self.pos:
            if self.pos < index <= self.pos + 30:
                while self.pos < index:
                    self.cap.grab()
                    self.pos += 1
            else:
                self.seek(index)
        ok, frame = self.cap.read()
        if ok:
            self.pos = index + 1
        return self._crop(frame) if ok else None

    def _crop(self, frame):
        if self.area is None or frame is None:
            return frame
        x0, y0, x1, y1 = self.area
        return np.ascontiguousarray(frame[y0:y1, x0:x1])

    def close(self):
        self.cap.release()


class FrameServer:
    """Thread-safe random access to frames, cheap for sequential playback."""

    def __init__(self, path, max_width=960):
        self.reader = VideoReader(path)
        self.lock = threading.Lock()
        self.max_width = max_width
        self.cache = {}

    def jpeg(self, index, quality=80, width=None):
        width = width or self.max_width
        key = (index, width)
        with self.lock:
            if key in self.cache:
                return self.cache[key]
            frame = self.reader.read(max(0, min(index, self.reader.frame_count - 1)))
            if frame is None:
                return None
            h, w = frame.shape[:2]
            if w > width:
                frame = cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            data = buf.tobytes() if ok else None
            if len(self.cache) > 240:
                self.cache.clear()
            self.cache[key] = data
            return data

    def close(self):
        self.reader.close()

"""Video access via OpenCV (FFmpeg backend)."""
from __future__ import annotations

import threading

import cv2


class VideoReader:
    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video: {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
            yield i, frame
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
        return frame if ok else None

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

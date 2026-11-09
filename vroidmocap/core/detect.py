"""Thin wrappers around the MediaPipe Tasks models used by both passes."""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import mediapipe as mp  # noqa: E402
from mediapipe.tasks.python import BaseOptions, vision  # noqa: E402

MODEL_FILES = {
    "pose_full": "pose_landmarker_full.task",
    "pose_heavy": "pose_landmarker_heavy.task",
    "hand": "hand_landmarker.task",
    "embed": "mobilenet_v3_large.tflite",
}
MODEL_URLS = {
    "pose_full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "pose_heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "embed": "https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_large/float32/latest/mobilenet_v3_large.tflite",
}


def resource_dir():
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return os.path.join(base, "vroidmocap")
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def model_path(key):
    p = os.path.join(resource_dir(), "assets", "models", MODEL_FILES[key])
    if not os.path.exists(p):
        alt = os.path.join(user_model_dir(), MODEL_FILES[key])
        if os.path.exists(alt):
            return alt
        download_model(key, alt)
        return alt
    return p


def user_model_dir():
    root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.cache")
    d = os.path.join(root, "VRoidMocap", "models")
    os.makedirs(d, exist_ok=True)
    return d


def download_model(key, dest):
    import urllib.request
    tmp = dest + ".part"
    urllib.request.urlretrieve(MODEL_URLS[key], tmp)
    os.replace(tmp, dest)


def _opts(key):
    # read the bytes ourselves so non-ASCII install paths work on Windows
    with open(model_path(key), "rb") as f:
        return BaseOptions(model_asset_buffer=f.read())


def to_mp(rgb):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


# MediaPipe pose landmark indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, MOUTH_L, MOUTH_R = 0, 2, 5, 7, 8, 9, 10
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_PINKY, R_PINKY, L_INDEX, R_INDEX, L_THUMB, R_THUMB = 17, 18, 19, 20, 21, 22
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 23, 24, 25, 26, 27, 28
L_HEEL, R_HEEL, L_FOOT, R_FOOT = 29, 30, 31, 32


class PoseResult:
    __slots__ = ("lm2d", "vis", "world")

    def __init__(self, lm2d, vis, world):
        self.lm2d = lm2d    # (33,2) pixel coords in the full frame
        self.vis = vis      # (33,)
        self.world = world  # (33,3) metres, hip-centred, camera-oriented (x right, y down, z away)


def _pose_from(res_lms, res_world, x0, y0, sw, sh):
    out = []
    for lms, wl in zip(res_lms, res_world):
        a = np.array([[l.x, l.y, l.visibility if l.visibility is not None else 1.0,
                       l.presence if l.presence is not None else 1.0] for l in lms])
        lm2d = np.stack([x0 + a[:, 0] * sw, y0 + a[:, 1] * sh], axis=1)
        vis = np.minimum(a[:, 2], a[:, 3])
        world = np.array([[p.x, p.y, p.z] for p in wl])
        out.append(PoseResult(lm2d, vis, world))
    return out


class PoseDetector:
    def __init__(self, heavy=False, num_poses=1, video=False, min_conf=0.3):
        self.video = video
        o = vision.PoseLandmarkerOptions(
            base_options=_opts("pose_heavy" if heavy else "pose_full"),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_poses=num_poses,
            min_pose_detection_confidence=min_conf,
            min_pose_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self.lm = vision.PoseLandmarker.create_from_options(o)

    def detect(self, rgb, x0=0, y0=0, scale_x=1.0, scale_y=1.0, ts_ms=None):
        """Run on an image (usually a crop). x0/y0 + scale map crop coords to frame coords."""
        h, w = rgb.shape[:2]
        img = to_mp(rgb)
        if self.video:
            r = self.lm.detect_for_video(img, int(ts_ms))
        else:
            r = self.lm.detect(img)
        return _pose_from(r.pose_landmarks, r.pose_world_landmarks, x0, y0, w * scale_x, h * scale_y)

    def close(self):
        try:
            self.lm.close()
        except Exception:
            pass


class HandDetector:
    def __init__(self, min_conf=0.15):
        o = vision.HandLandmarkerOptions(
            base_options=_opts("hand"), num_hands=2,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self.lm = vision.HandLandmarker.create_from_options(o)

    def detect(self, rgb):
        """Returns list of (lm2d_norm (21,2), world (21,3), score, label).
        label is the anatomical side assuming a NON-mirrored image ("left"/"right")."""
        r = self.lm.detect(to_mp(rgb))
        out = []
        for lms, wl, hd in zip(r.hand_landmarks, r.hand_world_landmarks, r.handedness):
            a = np.array([[l.x, l.y] for l in lms])
            wv = np.array([[p.x, p.y, p.z] for p in wl])
            name = (hd[0].category_name or "").lower() if hd else ""
            # MediaPipe labels assume a mirrored (selfie) image
            label = {"left": "right", "right": "left"}.get(name, "")
            out.append((a, wv, float(hd[0].score) if hd else 0.0, label))
        return out

    def close(self):
        try:
            self.lm.close()
        except Exception:
            pass


class Embedder:
    def __init__(self):
        o = vision.ImageEmbedderOptions(base_options=_opts("embed"), l2_normalize=True, quantize=False)
        self.e = vision.ImageEmbedder.create_from_options(o)

    def embed(self, rgb):
        r = self.e.embed(to_mp(rgb))
        v = np.asarray(r.embeddings[0].embedding, dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def close(self):
        try:
            self.e.close()
        except Exception:
            pass


def crop_square(rgb, cx, cy, size, out=None):
    """Square crop centred on (cx, cy), padded with black where it leaves the frame.
    Returns (crop, x0, y0, scale) where frame = x0 + crop_px * scale."""
    size = max(8, int(round(size)))
    x0 = int(round(cx - size / 2))
    y0 = int(round(cy - size / 2))
    h, w = rgb.shape[:2]
    crop = np.zeros((size, size, 3), dtype=rgb.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + size), min(h, y0 + size)
    if sx1 > sx0 and sy1 > sy0:
        crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgb[sy0:sy1, sx0:sx1]
    scale = 1.0
    if out is not None and out != size:
        interp = cv2.INTER_CUBIC if out > size else cv2.INTER_AREA
        crop = cv2.resize(crop, (out, out), interpolation=interp)
        scale = size / out
    return crop, x0, y0, scale


def landmark_box(lm2d, vis, thr=0.3):
    """Bounding box (x0,y0,x1,y1) of a person from pose landmarks, with head/feet margin."""
    good = vis > thr
    if good.sum() < 4:
        good = np.ones(len(vis), dtype=bool)
    p = lm2d[good]
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    sh = np.linalg.norm(lm2d[L_SH] - lm2d[R_SH])
    torso = np.linalg.norm((lm2d[L_SH] + lm2d[R_SH]) / 2 - (lm2d[L_HIP] + lm2d[R_HIP]) / 2)
    unit = max(sh, torso * 0.6, 4.0)
    return np.array([x0 - unit * 0.3, y0 - unit * 0.7, x1 + unit * 0.3, y1 + unit * 0.15])


def box_iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0

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
    "wholebody": "rtmw3d-x_384x288.onnx",
    "lifter": "motionbert_3d_243.onnx",
}
# large models that are not packed into the exe: fetched once on first use
DOWNLOAD_ON_USE = {"wholebody", "lifter"}
MODEL_SIZES = {"wholebody": 369330857, "lifter": 170049517}
MODEL_MIRRORS = {
    "wholebody": ["https://huggingface.co/bukuroo/RTMW3D-ONNX/resolve/main/rtmw3d-x-384.onnx"],
}
MODEL_URLS = {
    "pose_full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "pose_heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "embed": "https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_large/float32/latest/mobilenet_v3_large.tflite",
    "wholebody": "https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx",
    "lifter": "https://huggingface.co/bukuroo/MotionBERT-3d-ONNX/resolve/main/motionbert_3d_243.onnx",
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


def download_model(key, dest, progress=None):
    """Fetch a model file (tries mirrors, checks the size, atomic rename)."""
    import urllib.request
    tmp = dest + ".part"
    err = None
    for url in [MODEL_URLS[key]] + MODEL_MIRRORS.get(key, []):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or MODEL_SIZES.get(key, 0) or 0)
                got = 0
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
                    got += len(buf)
                    if progress and total:
                        progress(got / total)
            want = MODEL_SIZES.get(key)
            if want and os.path.getsize(tmp) != want:
                raise IOError("incomplete download")
            os.replace(tmp, dest)
            return
        except Exception as e:  # try the next mirror
            err = e
    raise RuntimeError(f"could not download model '{key}': {err}")


def model_available(key):
    for d in (os.path.join(resource_dir(), "assets", "models"), user_model_dir()):
        p = os.path.join(d, MODEL_FILES[key])
        if os.path.exists(p) and os.path.getsize(p) == MODEL_SIZES.get(key, os.path.getsize(p)):
            return p
    return None


def ensure_model(key, progress=None):
    """Path of a model, downloading it into the user's data folder if needed.
    Returns None when it cannot be obtained (offline)."""
    p = model_available(key)
    if p:
        return p
    dest = os.path.join(user_model_dir(), MODEL_FILES[key])
    try:
        download_model(key, dest, progress)
        return dest
    except Exception:
        return None


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
    def __init__(self, heavy=False, num_poses=1, video=False, min_conf=0.3, masks=False):
        self.video = video
        self.masks = masks
        self.last_mask = None
        o = vision.PoseLandmarkerOptions(
            base_options=_opts("pose_heavy" if heavy else "pose_full"),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_poses=num_poses,
            min_pose_detection_confidence=min_conf,
            min_pose_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
            output_segmentation_masks=masks,
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
        self.last_mask = None
        if self.masks and getattr(r, "segmentation_masks", None):
            m = r.segmentation_masks[0].numpy_view()
            self.last_mask = np.asarray(m[..., 0] if m.ndim == 3 else m, np.float32)
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


def mask_at(mask, pts, x0, y0, sc, radius=0.015):
    """Person-mask value (0..1) at frame points: the highest value within a small
    radius (joints sit inside the limb, mask edges are soft). -1 outside the mask."""
    h, w = mask.shape
    r = max(1, int(round(radius * w)))
    out = np.full(len(pts), -1.0, np.float32)
    q = (np.asarray(pts, float) - [x0, y0]) / sc  # mask pixels = crop pixels
    for i, (u, v) in enumerate(q):
        u, v = int(round(u)), int(round(v))
        if -r <= u < w + r and -r <= v < h + r:
            a, b = max(0, v - r), min(h, v + r + 1)
            c, d = max(0, u - r), min(w, u + r + 1)
            if a < b and c < d:
                out[i] = float(mask[a:b, c:d].max())
    return out


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


# ----------------------------------------------------------------------------- whole-body 2D model
# COCO-WholeBody (133 keypoints): 0-16 body, 17-22 feet, 23-90 face, 91-111 left hand, 112-132 right hand
WB_TO_MP = {11: 5, 12: 6, 13: 7, 14: 8, 15: 9, 16: 10, 23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16,
            29: 19, 30: 22, 31: 17, 32: 20, 0: 0, 2: 1, 5: 2, 7: 3, 8: 4}
WB_SWAP = np.arange(133)
for _a, _b in [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 20), (18, 21), (19, 22)]:
    WB_SWAP[_a], WB_SWAP[_b] = _b, _a
WB_SWAP[91:112], WB_SWAP[112:133] = np.arange(112, 133), np.arange(91, 112)


Z_RANGE = 2.1744869  # RTMW3D depth range


class WholeBody:
    """RTMW whole-body keypoints (ONNX). Much more precise in the picture than
    the MediaPipe body model; used for where the joints are in 2D."""
    W, H = 288, 384
    MEAN = np.array([123.675, 116.28, 103.53], np.float32)
    STD = np.array([58.395, 57.12, 57.375], np.float32)

    def __init__(self, path, threads=1):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def detect(self, rgb, box, pad=1.25):
        """box: x0,y0,x1,y1 around the person. Returns (kp (133,2) px, score (133,))."""
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        w, h = (box[2] - box[0]) * pad, (box[3] - box[1]) * pad
        if w / max(h, 1e-6) > self.W / self.H:
            h = w * self.H / self.W
        else:
            w = h * self.W / self.H
        M = np.array([[self.W / w, 0, self.W / 2 - cx * self.W / w], [0, self.H / h, self.H / 2 - cy * self.H / h]])
        crop = cv2.warpAffine(rgb, M, (self.W, self.H), flags=cv2.INTER_LINEAR)
        x = ((crop.astype(np.float32) - self.MEAN) / self.STD).transpose(2, 0, 1)[None]
        out = self.sess.run(None, {self.inp: x})
        sx, sy = out[0][0], out[1][0]
        kx, ky = sx.argmax(1) / 2.0, sy.argmax(1) / 2.0
        score = np.minimum(sx.max(1), sy.max(1))
        kp = np.stack([(kx - self.W / 2) * w / self.W + cx, (ky - self.H / 2) * h / self.H + cy], 1)
        if len(out) > 2:  # the 3D model also says how far each point is (relative, ~metres; + = further)
            kz = (out[2][0].argmax(1) / 2.0 / (self.H / 2) - 1) * Z_RANGE
            self.last_z = kz.astype(np.float32)
        else:
            self.last_z = np.zeros(len(kp), np.float32)
        return kp.astype(np.float32), score.astype(np.float32)

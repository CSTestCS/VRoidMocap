"""Depth over time from the 2D body points (MotionBERT, 243 frames at once).

A single picture cannot say whether an arm or a leg reaches towards the camera
or away from it; the motion around that moment usually can. This model reads
the 2D whole-body points of a whole stretch of frames (before and after) and
gives each joint a depth. The solver uses it as a second opinion next to the
pose model's own depth guess."""
from __future__ import annotations

import os

import numpy as np

WIN = 243
# MediaPipe index <- H36M index (the model's joint order)
MP_FROM_H36M = {11: 11, 12: 14, 13: 12, 14: 15, 15: 13, 16: 16, 23: 4, 24: 1, 25: 5, 26: 2, 27: 6, 28: 3}


def coco_to_h36m(x):
    """(..., 17, C) COCO body order -> (..., 17, C) H36M order."""
    y = np.zeros(x.shape[:-2] + (17, x.shape[-1]), x.dtype)
    y[..., 0, :] = (x[..., 11, :] + x[..., 12, :]) / 2
    y[..., 1, :], y[..., 2, :], y[..., 3, :] = x[..., 12, :], x[..., 14, :], x[..., 16, :]
    y[..., 4, :], y[..., 5, :], y[..., 6, :] = x[..., 11, :], x[..., 13, :], x[..., 15, :]
    y[..., 8, :] = (x[..., 5, :] + x[..., 6, :]) / 2
    y[..., 7, :] = (y[..., 0, :] + y[..., 8, :]) / 2
    y[..., 9, :] = x[..., 0, :]
    y[..., 10, :] = (x[..., 1, :] + x[..., 2, :]) / 2
    y[..., 11, :], y[..., 12, :], y[..., 13, :] = x[..., 5, :], x[..., 7, :], x[..., 9, :]
    y[..., 14, :], y[..., 15, :], y[..., 16, :] = x[..., 6, :], x[..., 8, :], x[..., 10, :]
    return y


def lift(wb2d, wbs, spans, W, H, fps, model_path, threads=2, step_mult=1):
    """wb2d (N,133,2) px, wbs (N,133) scores, spans: (start, end) stretches to
    read as one motion (inside a shot). Returns (N,17,3) in the model's units
    (x right, y down, z away from the camera; units of half the shorter
    picture side), nan where not computed."""
    from .detect import onnx_session
    sess = onnx_session(model_path, threads, inter=0)
    inp_name = sess.get_inputs()[0].name
    N = len(wb2d)
    out = np.full((N, 17, 3), np.nan)
    X = np.concatenate([wb2d[:, :17].astype(np.float64), np.clip(wbs[:, :17, None], 0, 1)], -1)
    X = coco_to_h36m(X)
    X[..., :2] = (X[..., :2] - [W / 2.0, H / 2.0]) / (min(W, H) / 2.0)
    # a joint with no score at all is "not seen" for the model: all zeros (as it
    # was trained with masked joints); it fills it in from the motion around it
    X[X[..., 2] <= 0] = 0.0
    # the model was trained on ~30-50 fps motion; a coarser step reads a longer
    # stretch at once (x2: about 8 s each way, x5: about 20 s each way)
    step = max(1, int(round(fps / 30.0))) * int(step_mult)
    for (s, e) in spans:
        idx = np.arange(s, e, step)
        if len(idx) < 4:
            continue
        seq = X[idx]
        n = len(seq)
        res = np.zeros((n, 17, 3))
        cnt = np.zeros(n)
        # overlapping windows, averaged (the middle of a window knows most)
        starts = list(range(0, max(1, n - WIN + 1), WIN // 3))
        if starts[-1] + WIN < n:
            starts.append(n - WIN)
        for a in starts:
            chunk = seq[a:a + WIN]
            m = len(chunk)
            if m < WIN:
                chunk = np.concatenate([chunk, np.repeat(chunk[-1:], WIN - m, 0)])
            y = sess.run(None, {inp_name: chunk[None].astype(np.float32)})[0][0][:m]
            w = np.sin(np.pi * (np.arange(m) + 0.5) / WIN) + 0.05 if n > WIN else np.ones(m)
            res[a:a + m] += y * w[:, None, None]
            cnt[a:a + m] += w
        res /= np.maximum(cnt, 1e-9)[:, None, None]
        full = np.arange(s, e)
        for j in range(17):
            for c in range(3):
                out[s:e, j, c] = np.interp(full, idx, res[:, j, c])
    return out


def as_mediapipe(Y):
    """(N,33,3) in MediaPipe numbering (body joints only, others nan), camera
    orientation (x right, y down, z away), the model's units."""
    out = np.full((len(Y), 33, 3), np.nan)
    for mp, h in MP_FROM_H36M.items():
        out[:, mp] = Y[:, h]
    return out


def depth_opinion(Y, root_z, focal, W, H, N):
    """(N,33) depth of each MediaPipe joint relative to the hip centre in
    metres (nan where unknown), from the model's output Y (N,17,3)."""
    z = np.full((N, 33), np.nan)
    k = (min(W, H) / 2.0) * root_z / np.maximum(focal, 1e-6)  # model units -> metres at the body
    for mp, h in MP_FROM_H36M.items():
        z[:, mp] = (Y[:, h, 2] - Y[:, 0, 2]) * k
    return z


def model_file():
    from .detect import model_available
    return model_available("lifter")


def cached_lift(workdir, capture, spans, W, H, fps, step_mult=1):
    """The lifter's output for this capture, computed once and kept next to it."""
    from .detect import ensure_model
    path = ensure_model("lifter")
    if not path or "wb2d" not in capture:
        return None
    sfx = "" if step_mult == 1 else f"_x{step_mult}"
    import zlib
    # (the input: which frames are lifted, and the scores the joints are weighed by)
    h = zlib.crc32(np.asarray(spans, np.int64).tobytes())
    h = zlib.crc32(np.ascontiguousarray(np.asarray(capture["wbs"], np.float32)).tobytes(), h)
    key = f"lift_{int(np.asarray(capture.get('version', 0)))}_{len(capture['wb2d'])}_{h:08x}{sfx}.npy"
    cp = os.path.join(workdir, key)
    if os.path.exists(cp):
        try:
            Y = np.load(cp)
            if Y.shape == (len(capture["wb2d"]), 17, 3):
                return Y
        except (OSError, ValueError):
            pass
    Y = lift(capture["wb2d"], capture["wbs"], spans, W, H, fps, path, step_mult=step_mult)
    try:
        np.save(cp, Y)
    except OSError:
        pass
    return Y


# H36M joint pairs that swap with the left/right labels
H36M_LR = [(1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16)]
# avatar bone -> (H36M from, H36M to): the bone's direction in the model's 3D
BONE_H36M = {"leftUpperArm": (11, 12), "leftLowerArm": (12, 13), "rightUpperArm": (14, 15), "rightLowerArm": (15, 16),
             "leftUpperLeg": (4, 5), "rightUpperLeg": (1, 2), "leftLowerLeg": (5, 6), "rightLowerLeg": (2, 3)}


def swap_labels(Y, swap):
    """The model reads the labels as detected; where the facing tracker found
    them swapped, swap its left and right joints too."""
    Y = Y.copy()
    for a, b in H36M_LR:
        ya = Y[swap, a].copy()
        Y[swap, a] = Y[swap, b]
        Y[swap, b] = ya
    return Y

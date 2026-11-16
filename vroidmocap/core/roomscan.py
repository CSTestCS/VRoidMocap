"""The room scan: the scenery around the character, rebuilt from the video and
the solved cameras, to line the camera up against in the Edit step.

For frames spread over every shot, each picture point outside the character
is cast out from the solved camera into the animation's world: rays that hit
the floor land on it (the floor gets its real shape), the rest land on a far
backdrop around the scene. The result is a cloud of coloured points: looking
through a camera that is right, the points sit on the picture they came from.
"""
from __future__ import annotations

import cv2
import numpy as np

from .mathutil import quat_to_mat

MAX_FRAMES = 160   # frames sampled over the whole video
GRID = (96, 54)    # picture points per sampled frame (across, down)
MIN_GAP_S = 0.4    # sampled frames at least this far apart within a shot


def sample_frames(shot_of, fps, max_frames=MAX_FRAMES):
    """Frame indices spread over every shot (each shot gets at least one)."""
    N = len(shot_of)
    starts = [0] + [i for i in range(1, N) if shot_of[i] != shot_of[i - 1]] + [N]
    spans = [(starts[k], starts[k + 1]) for k in range(len(starts) - 1)]
    gap = max(1, int(round(MIN_GAP_S * fps)))
    total = sum(max(1, (b - a) // gap) for a, b in spans)
    keep = min(1.0, max_frames / max(total, 1))
    out = []
    for a, b in spans:
        n = max(1, int(round(max(1, (b - a) // gap) * keep)))
        out += [int(x) for x in np.linspace(a, b - 1, n + 2)[1:-1].round()] if n > 0 else [(a + b) // 2]
    return sorted(set(out))


def person_box(lm2d, vis, W, H, pad=0.25):
    ok = vis > 0.3
    if ok.sum() < 3:
        return None
    p = lm2d[ok]
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    w, h = x1 - x0, y1 - y0
    return (x0 - pad * w, y0 - pad * h * 0.6, x1 + pad * w, y1 + pad * h * 0.4)


def build_room(video, anim, frames=None, grid=GRID):
    """(positions (n,3) float32 in the animation's world, colours (n,3) uint8)."""
    N = int(anim["frames"])
    W, H = int(anim["width"]), int(anim["height"])
    fps = float(anim["fps"])
    cam_pos = np.asarray(anim["cam_pos"], np.float64)
    cam_q = np.asarray(anim["cam_quat"], np.float64)
    vfov = np.asarray(anim["cam_vfov"], np.float64)
    shot = np.asarray(anim["shot_of"])
    lm2d, vis = np.asarray(anim["lm2d"], np.float64), np.asarray(anim["vis"], np.float64)
    root = np.asarray(anim["root"], np.float64)
    frames = sample_frames(shot, fps) if frames is None else frames
    d_char = np.linalg.norm(cam_pos - root, axis=1)
    far = float(max(8.0, 3.0 * np.nanmedian(d_char))) if np.isfinite(d_char).any() else 12.0
    gx, gy = grid
    us = (np.arange(gx) + 0.5) * W / gx
    vs = (np.arange(gy) + 0.5) * H / gy
    U, V = np.meshgrid(us, vs)
    U, V = U.ravel(), V.ravel()
    cap = cv2.VideoCapture(video)
    P, C = [], []
    for f in frames:
        if not (0 <= f < N) or not np.isfinite(cam_q[f]).all():
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        if img.shape[1] != W or img.shape[0] != H:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        keep = np.ones(len(U), bool)
        box = person_box(lm2d[f], vis[f], W, H)
        if box is not None:
            keep &= ~((U > box[0]) & (U < box[2]) & (V > box[1]) & (V < box[3]))
        u, v = U[keep], V[keep]
        fl = (H / 2) / np.tan(np.radians(vfov[f]) / 2)
        d = np.stack([(u - W / 2) / fl, -(v - H / 2) / fl, -np.ones_like(u)], 1)  # three.js camera: -z forwards
        R = quat_to_mat(cam_q[f][None])[0]
        dw = d @ R.T
        dw /= np.linalg.norm(dw, axis=1, keepdims=True)
        o = cam_pos[f]
        t = np.full(len(dw), far)
        down = dw[:, 1] < -1e-3
        t_floor = np.where(down, -o[1] / np.where(down, dw[:, 1], -1.0), np.inf)
        on_floor = down & (t_floor > 0) & (t_floor < far)
        t = np.where(on_floor, t_floor, t)
        P.append(o + dw * t[:, None])
        # (a light blur: one point stands for its patch of picture)
        small = cv2.blur(img, (max(1, W // gx), max(1, H // gy)))
        bgr = small[np.clip(v.astype(int), 0, H - 1), np.clip(u.astype(int), 0, W - 1)]
        C.append(bgr[:, ::-1])
    cap.release()
    if not P:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    return np.concatenate(P).astype(np.float32), np.concatenate(C).astype(np.uint8)

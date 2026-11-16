"""Front or back, from how the character looks.

The detectors read a back view poorly: long hair hides the shoulders, the face
detector "finds" a face on the back of the head, the pose models drop out. The
character's own look is a surer clue: where her face is plainly seen, her
upper body looks one way (face, the front of her outfit); turned away it looks
another (hair, the back of the outfit). Every sampled frame's upper body is
compared (image embedding) with those plainly front-on frames of the same
video: much alike is a vote for "facing the camera", much less alike a vote
for "turned away", in between (side on) no vote.
"""
from __future__ import annotations

import hashlib
import os

import cv2
import numpy as np

SAMPLE_HZ = 10.0       # embedded frames per second of video
UPPER = 0.55           # the upper body: this share of the body box, from the top
MIN_FRONT = 8          # plainly front-on frames needed to know her front
MIN_SPREAD = 0.08
FRONT_FIT_Q = 0.5      # face fit quality that marks a plainly front-on frame      # front-on vs least alike: below this the look does not tell


def body_boxes(capture, W, H):
    """(N,4) a box around the body from the captured joints (nan where none)."""
    wb2d, wbs = capture.get("wb2d"), capture.get("wbs")
    lm2d, vis = capture["lm2d"], capture["vis"]
    N = len(lm2d)
    out = np.full((N, 4), np.nan)
    for f in range(N):
        pts = None
        if wbs is not None and (wbs[f, :17] > 0.3).sum() >= 5:
            pts = wb2d[f, :17][wbs[f, :17] > 0.3]
        elif (vis[f] > 0.5).sum() >= 5:
            pts = lm2d[f][vis[f] > 0.5]
        if pts is None:
            continue
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        h = y1 - y0
        w = x1 - x0
        # (the joints stop at the eyes: the top of the head is above them)
        out[f] = [max(0, x0 - 0.15 * w), max(0, y0 - 0.18 * h), min(W, x1 + 0.15 * w), min(H, y1)]
    return out


def embeddings(video, workdir, char_id, boxes, fps, frames):
    """(len(frames), D) upper-body embeddings, cached next to the capture."""
    key = hashlib.md5(np.round(np.nan_to_num(boxes[frames]), 0).astype(np.int32).tobytes()).hexdigest()[:12]
    path = os.path.join(workdir, f"look_c{char_id}_{key}.npy")
    if os.path.exists(path):
        try:
            E = np.load(path)
            if len(E) == len(frames):
                return E
        except Exception:
            pass
    from .detect import Embedder
    emb = Embedder()
    cap = cv2.VideoCapture(video)
    E = None
    last = -10
    try:
        for i, f in enumerate(frames):
            if f != last + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, img = cap.read()
            last = f
            if not ok:
                continue
            x0, y0, x1, y1 = boxes[f]
            y1 = y0 + UPPER * (y1 - y0)
            crop = img[int(y0):max(int(y0) + 2, int(y1)), int(x0):max(int(x0) + 2, int(x1))]
            if crop.size == 0:
                continue
            v = emb.embed(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            if E is None:
                E = np.full((len(frames), len(v)), np.nan, np.float32)
            E[i] = v
    finally:
        cap.release()
        emb.close()
    if E is None:
        return None
    try:
        np.save(path, E)
    except OSError:
        pass
    return E


def track_boxes(workdir, char_id, N, fps):
    """The boxes the capture followed her with (the person detector's), or None."""
    try:
        from .capture import character_boxes, stabilize_boxes
        _, b, segs = character_boxes(workdir, char_id)
        b = stabilize_boxes(b, segs, fps)
        return b if len(b) == N else None
    except Exception:
        return None


def front_votes(video, workdir, char_id, capture, valid, fps, W, H, shot_of=None, focal=None):
    """(N,) -1..1: + = her front is to the camera, - = her back (nan = no
    opinion), or None when the video does not tell."""
    if not video or not os.path.exists(video):
        return None
    N = len(valid)
    boxes = body_boxes(capture, W, H)
    tb = track_boxes(workdir, char_id, N, fps)
    if tb is not None:  # (the person detector's box is surer than one round joints read badly)
        boxes = np.where(np.isfinite(tb).all(1)[:, None], tb, boxes)
    have = valid & np.isfinite(boxes).all(1) & ((boxes[:, 2] - boxes[:, 0]) > 16) & ((boxes[:, 3] - boxes[:, 1]) > 16)
    step = max(1, int(round(fps / SAMPLE_HZ)))
    frames = np.array([f for f in range(0, N, step) if have[f]], np.int64)
    # plainly front-on: the face template fits the face points well (the pose
    # model's own "nose and eyes visible" is set in back views too: 10-26% of
    # such frames were turned away on the benchmarks; a good face fit: 0-2%)
    if focal is None or capture.get("wb2d") is None:
        return None
    from .head import head_from_face
    _, q = head_from_face(capture["wb2d"], capture["wbs"], np.asarray(focal, np.float64), W / 2, H / 2, valid)
    face = (q > FRONT_FIT_Q) & valid
    if len(frames) < MIN_FRONT * 2 or face[frames].sum() < MIN_FRONT:
        return None
    E = embeddings(video, workdir, char_id, boxes, fps, frames)
    if E is None:
        return None
    ok = np.isfinite(E).all(1)
    fr = ok & face[frames]
    if fr.sum() < MIN_FRONT:
        return None
    def vote(sel, protosel):
        proto = E[protosel].mean(0)
        proto /= max(np.linalg.norm(proto), 1e-9)
        sim = np.where(ok & sel, np.nan_to_num(E) @ proto, np.nan)
        hi = float(np.median(sim[protosel]))
        lo = float(np.nanpercentile(sim, 10)) if np.isfinite(sim).sum() >= 4 else hi
        if hi - lo < MIN_SPREAD:
            return None
        return np.clip((sim - (hi + lo) / 2) / ((hi - lo) / 2), -1, 1)
    # her front as this shot shows it (its light, its distance) where the shot
    # has enough front-on frames; else as the whole video shows it
    v = vote(np.ones(len(frames), bool), fr)
    if v is None:
        return None
    if shot_of is not None:
        sh = np.asarray(shot_of)[frames]
        for k in np.unique(sh):
            m = sh == k
            if (fr & m).sum() >= MIN_FRONT:
                vk = vote(m, fr & m)
                if vk is not None:
                    v = np.where(m, vk, v)
    out = np.full(N, np.nan)
    good = np.isfinite(v)
    if good.sum() < 2:
        return None
    fg, vg = frames[good], v[good]
    idx = np.arange(N)
    out = np.interp(idx, fg, vg)
    # only near a sampled frame (no opinion across gaps)
    near = np.full(N, False)
    for f in fg:
        near[max(0, f - step):min(N, f + step + 1)] = True
    out[~(near & valid)] = np.nan
    return out

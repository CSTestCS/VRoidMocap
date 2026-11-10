"""Pass 2a: per-frame landmark capture for ONE chosen character.

Uses the pass-1 tracks to know where the character is in every frame, then runs
the heavy pose model on a stabilised, upscaled crop around them and the hand
model on crops around each wrist. Work is split into chunks that run in
parallel worker processes.
"""
from __future__ import annotations

import json
import multiprocessing as mpc
import os
import time

import cv2
import numpy as np

from .detect import (L_EL, L_INDEX, L_PINKY, L_THUMB, L_WR, R_EL, R_INDEX, R_PINKY, R_THUMB,
                     R_WR, HandDetector, PoseDetector, WholeBody, box_iou, crop_square, landmark_box)
from .mathutil import gaussian_smooth
from .video import VideoReader

CHUNK = 450


def character_boxes(workdir, char_id):
    """Per-frame box of a character for every frame it is tracked (interpolated)."""
    with open(os.path.join(workdir, "analysis.json")) as f:
        A = json.load(f)
    npz = np.load(os.path.join(workdir, "analysis.npz"))
    N = A["frames"]
    boxes = np.full((N, 4), np.nan)
    qual = np.zeros(N)
    shot_of = npz["shot_of"]
    for t in A["tracklets"]:
        if t["character"] != char_id:
            continue
        fr = npz[f"t{t['id']}_frames"]
        bx = npz[f"t{t['id']}_boxes"].astype(np.float64)
        vis = npz[f"t{t['id']}_vis"].astype(np.float64).mean(1)
        if len(fr) >= 3:
            bx = gaussian_smooth(bx, 1.0)
        full = np.arange(fr[0], fr[-1] + 1)
        ib = np.stack([np.interp(full, fr, bx[:, c]) for c in range(4)], 1)
        iv = np.interp(full, fr, vis)
        better = iv > qual[full]
        boxes[full[better]] = ib[better]
        qual[full[better]] = iv[better]
    valid = ~np.isnan(boxes[:, 0])
    # segments: contiguous valid runs not crossing a shot boundary
    segs = []
    f = 0
    while f < N:
        if not valid[f]:
            f += 1
            continue
        s = f
        while f < N and valid[f] and shot_of[f] == shot_of[s]:
            f += 1
        segs.append((s, f))
    return A, boxes, segs


def _hand_attempts(lm2d, side):
    wr, el = (L_WR, L_EL) if side == 0 else (R_WR, R_EL)
    tips = [L_INDEX, L_PINKY, L_THUMB] if side == 0 else [R_INDEX, R_PINKY, R_THUMB]
    w = lm2d[wr]
    fore = max(float(np.linalg.norm(w - lm2d[el])), 6.0)
    c = (w + lm2d[tips].mean(0)) / 2
    return w, fore, [(c, fore * m) for m in (1.35, 1.9, 2.6, 1.0)]


def _run_hand(hand, rgb, center, size, ctx):
    """Run the hand model on one crop and return the most plausible hand for the
    requested side, or None. ctx holds the body-model evidence used to reject
    false or swapped detections."""
    crop, x0, y0, sc = crop_square(rgb, center[0], center[1], size, out=224)
    best = None
    for a, wv, score, label in hand.detect(crop):
        p = np.stack([x0 + a[:, 0] * 224 * sc, y0 + a[:, 1] * 224 * sc], 1)
        q = hand_plausibility(p, score, label, ctx)
        if q is not None and (best is None or q > best[0]):
            best = (q, p, wv, score, (x0 + 112 * sc, y0 + 112 * sc))
    return best


def hand_plausibility(p, score, label, ctx):
    wrist, other, fore, unit, pdir, side = (ctx[k] for k in ("wrist", "other", "fore", "unit", "pdir", "side"))
    d_own = float(np.linalg.norm(p[0] - wrist))
    d_other = float(np.linalg.norm(p[0] - other))
    if d_own > 0.55 * fore or d_own > 0.8 * d_other:
        return None
    palm = float(np.linalg.norm(p[9] - p[0]))
    span = float(np.ptp(p, axis=0).max())
    if palm < 0.12 * unit or span < 0.2 * unit or palm > 0.9 * unit:
        return None
    hdir = p[9] - p[0]
    if np.linalg.norm(pdir) > 1e-6:
        c = float(np.dot(hdir, pdir) / (np.linalg.norm(hdir) * np.linalg.norm(pdir) + 1e-9))
        if c < -0.2:
            return None
    else:
        c = 0.0
    q = score - 0.6 * d_own / fore + 0.15 * c
    if label and label != ("left" if side == 0 else "right"):
        q -= 0.35
    return q


def process_chunk(args):
    """Worker: landmarks for frames [s, e) given per-frame boxes."""
    video, s, e, boxes, fps, out_path, wb_path = args
    if os.path.exists(out_path):
        return out_path
    n = e - s
    lm2d = np.zeros((n, 33, 2), np.float32)
    vis = np.zeros((n, 33), np.float32)
    world = np.zeros((n, 33, 3), np.float32)
    cropc = np.zeros((n, 2), np.float32)
    h2d = np.zeros((n, 2, 21, 2), np.float32)
    hw = np.zeros((n, 2, 21, 3), np.float32)
    hs = np.zeros((n, 2), np.float32)
    wb2d = np.zeros((n, 133, 2), np.float32)
    wbs = np.zeros((n, 133), np.float32)
    wbody = WholeBody(wb_path) if wb_path else None
    wb_stride = max(1, int(round(fps / 30.0)))
    hc = np.zeros((n, 2, 2), np.float32)
    pose = PoseDetector(heavy=True, num_poses=1, video=True, min_conf=0.3)
    pose_img = None
    hand = HandDetector(min_conf=0.2)
    rd = VideoReader(video)
    prev_hand = [None, None]
    size_s = None
    for i, (fi, bgr) in enumerate(rd.iter_frames(1, s, e)):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        b = boxes[i]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        size = max(b[2] - b[0], b[3] - b[1]) * 1.3
        size_s = size if size_s is None else 0.8 * size_s + 0.2 * size
        crop, x0, y0, sc = crop_square(rgb, cx, cy, size_s, out=512)
        res = pose.detect(crop, x0, y0, sc, sc, ts_ms=int(round(fi * 1000.0 / fps)))
        ok = bool(res) and box_iou(landmark_box(res[0].lm2d, res[0].vis), b) > 0.2
        if not ok:
            # the video-mode tracker can lose the person; retry stateless
            if pose_img is None:
                pose_img = PoseDetector(heavy=True, num_poses=1, video=False, min_conf=0.25)
            res = pose_img.detect(crop, x0, y0, sc, sc)
            ok = bool(res) and box_iou(landmark_box(res[0].lm2d, res[0].vis), b) > 0.2
        if wbody is not None and (fi % wb_stride == 0 or i == n - 1):
            # (above 30 fps the whole-body model runs on every 2nd frame; the
            # gaps are interpolated, joints barely move in 1/60 s)
            # box: the tracked box, widened by the body model's own landmarks
            bb = np.array(b, float)
            if ok:
                lb = landmark_box(res[0].lm2d, res[0].vis)
                bb = np.array([min(bb[0], lb[0]), min(bb[1], lb[1]), max(bb[2], lb[2]), max(bb[3], lb[3])])
            wb2d[i], wbs[i] = wbody.detect(rgb, bb)
        if not ok:
            prev_hand = [None, None]
            continue
        r = res[0]
        lm2d[i], vis[i], world[i] = r.lm2d, r.vis, r.world
        cropc[i] = (x0 + 256 * sc, y0 + 256 * sc)
        L = r.lm2d
        unit = max(float(np.linalg.norm(L[11] - L[12])),
                   0.6 * float(np.linalg.norm((L[11] + L[12]) / 2 - (L[23] + L[24]) / 2)), 8.0)
        for side in (0, 1):
            wrist, fore, attempts = _hand_attempts(L, side)
            tips = [L_INDEX, L_PINKY] if side == 0 else [R_INDEX, R_PINKY]
            ctx = {"wrist": wrist, "other": L[R_WR if side == 0 else L_WR], "fore": fore, "unit": unit,
                   "pdir": L[tips].mean(0) - wrist, "side": side}
            if prev_hand[side] is not None:
                ph = prev_hand[side]
                ext = np.ptp(ph, axis=0).max()
                shift = wrist - ph[0]
                attempts = [(ph.mean(0) + shift, max(ext * 1.7, fore * 0.9))] + attempts
            best = None
            for center, sz in attempts:
                got = _run_hand(hand, rgb, center, sz, ctx)
                if got is not None and (best is None or got[0] > best[0]):
                    best = got
                if best is not None and best[3] > 0.85 and best[0] > 0.6:
                    break
            if best is not None and best[3] >= 0.5:
                h2d[i, side], hw[i, side], hs[i, side], hc[i, side] = best[1], best[2], best[3], best[4]
                prev_hand[side] = best[1]
            else:
                prev_hand[side] = None
    rd.close()
    pose.close()
    hand.close()
    if pose_img is not None:
        pose_img.close()
    np.savez_compressed(out_path + ".tmp.npz", lm2d=lm2d, vis=vis, world=world, cropc=cropc,
                        h2d=h2d, hw=hw, hs=hs, hc=hc, wb2d=wb2d, wbs=wbs)
    os.replace(out_path + ".tmp.npz", out_path)
    return out_path


def capture(video, workdir, char_id, progress=None, cancel=None, workers=None):
    progress = progress or (lambda *a: None)
    cancel = cancel or (lambda: False)
    A, boxes, segs = character_boxes(workdir, char_id)
    N, fps = A["frames"], A["fps"]
    cdir = os.path.join(workdir, f"capture_c{char_id}")
    os.makedirs(cdir, exist_ok=True)
    from .detect import ensure_model
    progress(0.0, "Getting the whole-body model (one-time download, 370 MB)")
    wb_path = ensure_model("wholebody", lambda f: progress(0.0, f"Downloading the whole-body model: {f * 100:.0f}%"))
    tag = "w" if wb_path else "m"
    jobs = []
    for (s, e) in segs:
        for cs in range(s, e, CHUNK):
            ce = min(e, cs + CHUNK)
            if ce - cs < 1:
                continue
            jobs.append((video, cs, ce, boxes[cs:ce].copy(), fps, os.path.join(cdir, f"chunk{tag}_{cs:07d}_{ce:07d}.npz"), wb_path))
    total = sum(j[2] - j[1] for j in jobs)
    # each worker holds its own models (~300 MB): cap the count to keep memory reasonable
    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 2) - 1, 6))
    t0 = time.time()
    done = 0
    if jobs:
        ctx = mpc.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for path in pool.imap_unordered(process_chunk, jobs):
                if cancel():
                    pool.terminate()
                    from .analysis import Cancelled
                    raise Cancelled()
                name = os.path.basename(path)[:-4].split("_")[1:]
                done += int(name[1]) - int(name[0])
                el = time.time() - t0
                eta = el / max(done, 1) * (total - done)
                progress(done / max(total, 1), f"Motion capture: {done}/{total} frames - ~{eta / 60:.1f} min left")
    # assemble full-length arrays
    out = {
        "lm2d": np.zeros((N, 33, 2), np.float32), "vis": np.zeros((N, 33), np.float32),
        "world": np.zeros((N, 33, 3), np.float32), "cropc": np.zeros((N, 2), np.float32),
        "h2d": np.zeros((N, 2, 21, 2), np.float32), "hw": np.zeros((N, 2, 21, 3), np.float32),
        "hs": np.zeros((N, 2), np.float32), "hc": np.zeros((N, 2, 2), np.float32),
        "wb2d": np.zeros((N, 133, 2), np.float32), "wbs": np.zeros((N, 133), np.float32),
    }
    for j in jobs:
        d = np.load(j[5])
        for k in out:
            out[k][j[1]:j[2]] = d[k]
    out["tracked"] = ~np.isnan(boxes[:, 0])
    fill_wholebody_gaps(out["wb2d"], out["wbs"], [(j[1], j[2]) for j in jobs])
    return out


def fill_wholebody_gaps(wb2d, wbs, ranges, max_gap=3):
    """Interpolate the whole-body keypoints over the frames it was not run on."""
    for s, e in ranges:
        have = np.where(wbs[s:e].any(1))[0] + s
        for a, b in zip(have[:-1], have[1:]):
            if 1 < b - a <= max_gap:
                for f in range(a + 1, b):
                    t = (f - a) / (b - a)
                    wb2d[f] = (1 - t) * wb2d[a] + t * wb2d[b]
                    wbs[f] = np.minimum(wbs[a], wbs[b])

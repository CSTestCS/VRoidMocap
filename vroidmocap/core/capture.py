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
                     R_WR, HandDetector, PoseDetector, WholeBody, box_iou, crop_square, landmark_box,
                     mask_at)
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


CAPTURE_VERSION = 5


def stabilize_boxes(boxes, segs, fps, ratio=0.6, win_s=1.5):
    """In back views the tracker can lose the body and keep following only the
    head: its box collapses to head size, and the detectors then find a tiny,
    contorted "person" inside it. Within a shot a body does not shrink that
    fast: a box much smaller than the body size seen around it (the 85th
    percentile over +-win_s) is extended from the head down to the full body
    size."""
    out = boxes.copy()
    k = max(1, int(round(win_s * fps)))
    for (s, e) in segs:
        h = boxes[s:e, 3] - boxes[s:e, 1]
        w = boxes[s:e, 2] - boxes[s:e, 0]
        for i in range(e - s):
            ref = np.percentile(h[max(0, i - k):i + k + 1], 85)
            if h[i] < ratio * ref:
                cx = (boxes[s + i, 0] + boxes[s + i, 2]) / 2
                ww = max(w[i], 0.45 * ref)
                top = boxes[s + i, 1]
                out[s + i] = [cx - ww / 2, top, cx + ww / 2, top + ref]
    return out


def extend_at_edges(boxes, segs, shot_of, W, H, fps, max_s=1.0, min_in=0.08):
    """A character walking out of (or into) the picture is lost by the person
    tracker while part of them is still visible: a foot, a hand. Where a track
    ends (or starts) with its box at the edge of the picture, the box carries on
    along its last motion for up to max_s, while at least min_in of it is still
    inside the picture, so the visible part is still captured. It is the same
    character by continuity; whatever is not in the picture is not animated."""
    out = boxes.copy()
    N = len(boxes)
    taken = ~np.isnan(boxes[:, 0])
    K = max(1, int(round(max_s * fps)))
    L = max(2, int(round(0.3 * fps)))
    m = 0.02 * max(W, H)

    def inside_frac(b):
        iw = max(0.0, min(b[2], W) - max(b[0], 0))
        ih = max(0.0, min(b[3], H) - max(b[1], 0))
        a = max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
        return iw * ih / a

    def at_edge(b):
        return b[0] < m or b[1] < m or b[2] > W - m or b[3] > H - m
    new = []
    for (s, e) in segs:
        for direction in (1, -1):
            if direction == 1:
                ref = boxes[max(s, e - L):e]
                f, last = e, boxes[e - 1]
            else:
                ref = boxes[s:min(e, s + L)][::-1]
                f, last = s - 1, boxes[s]
            if len(ref) < 2 or not at_edge(last):
                continue
            v = (ref[-1] - ref[0]) / (len(ref) - 1)
            v[[1, 3]] = 0.0 if abs(v[1] + v[3]) < abs(v[0] + v[2]) else v[[1, 3]]
            b = last.copy()
            k = 0
            while 0 <= f < N and k < K and not taken[f] and shot_of[f] == shot_of[s]:
                b = b + v
                if inside_frac(b) < min_in:
                    break
                out[f] = b
                taken[f] = True
                new.append(f)
                f += direction
                k += 1
    if not new:
        return out, segs
    valid = ~np.isnan(out[:, 0])
    segs2 = []
    f = 0
    while f < N:
        if not valid[f]:
            f += 1
            continue
        s0 = f
        while f < N and valid[f] and shot_of[f] == shot_of[s0]:
            f += 1
        segs2.append((s0, f))
    return out, segs2


def follow_job(args):
    """Worker: follow a character beyond the ends of its track, within its shot.

    args: (video, seed_frame, seed_box, direction (+1/-1), stop_frame (exclusive),
           cam_R (N,3,3) camera->shot reference, focal (N,), W, H, fps, wb_path, out_path)
    The person tracker needs most of a body to find someone; the whole-body
    model also finds a pair of legs or an arm. Frame by frame (backwards from
    where the track starts, forwards from where it ends) the box is carried
    along with the camera's own motion and re-centred on the body parts the
    whole-body model finds in it. It stops where nothing of the person is
    found any more (0.3 s) or the box leaves the picture. Returns {frame: box}."""
    (video, seed, box, direction, stop, cam_R, focal, W, H, fps, wb_path, out_path) = args
    if os.path.exists(out_path):
        return out_path
    wbody = WholeBody(wb_path)
    rd = VideoReader(video)
    step = max(1, int(round(fps / 30.0)))
    lost_max = max(2, int(round(0.3 * fps / step)))
    cx, cy = W / 2, H / 2
    box = np.asarray(box, np.float64)
    prev_f = seed
    out = {}
    lost = 0

    def warp(b, fa, fb):
        Ka = np.array([[focal[fa], 0, cx], [0, focal[fa], cy], [0, 0, 1.0]])
        Kb = np.array([[focal[fb], 0, cx], [0, focal[fb], cy], [0, 0, 1.0]])
        Hm = Kb @ cam_R[fb].T @ cam_R[fa] @ np.linalg.inv(Ka)
        pts = np.array([[b[0], b[1], 1], [b[2], b[1], 1], [b[0], b[3], 1], [b[2], b[3], 1], [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 1]])
        q = pts @ Hm.T
        q = q[:, :2] / q[:, 2:3]
        c = q[4]
        sc = np.mean(np.linalg.norm(q[:4] - c, axis=1)) / max(np.mean(np.linalg.norm(pts[:4, :2] - pts[4, :2], axis=1)), 1e-6)
        w, h = (b[2] - b[0]) * sc, (b[3] - b[1]) * sc
        return np.array([c[0] - w / 2, c[1] - h / 2, c[0] + w / 2, c[1] + h / 2])

    def frames():
        if direction > 0:
            yield from rd.iter_frames(step, seed + step, stop)
        else:
            hi = seed
            while hi > stop:
                lo = max(stop, hi - 30)
                block = list(rd.iter_frames(1, lo, hi))
                for fi, im in reversed(block):
                    if (seed - fi) % step == 0:
                        yield fi, im
                hi = lo
    for fi, bgr in frames():
        if (direction > 0 and fi >= stop) or (direction < 0 and fi <= stop - 1):
            break
        pred = warp(box, prev_f, fi)
        iw = max(0.0, min(pred[2], W) - max(pred[0], 0)) * max(0.0, min(pred[3], H) - max(pred[1], 0))
        if iw < 0.05 * (pred[2] - pred[0]) * (pred[3] - pred[1]):
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        kp, sc = wbody.detect(rgb, pred)
        body = np.arange(23)
        pw, ph = pred[2] - pred[0], pred[3] - pred[1]
        ins = (kp[body, 0] > pred[0] - 0.1 * pw) & (kp[body, 0] < pred[2] + 0.1 * pw) & \
              (kp[body, 1] > pred[1] - 0.1 * ph) & (kp[body, 1] < pred[3] + 0.1 * ph)
        good = (sc[body] >= 0.55) & ins & (kp[body, 0] >= 0) & (kp[body, 0] < W) & (kp[body, 1] >= 0) & (kp[body, 1] < H)
        if good.sum() >= 3:
            lost = 0
            k = kp[body][good]
            lo_, hi_ = k.min(0), k.max(0)
            m = 0.08 * np.array([pw, ph])
            dx = min(0.0, lo_[0] - m[0] - pred[0]) + max(0.0, hi_[0] + m[0] - pred[2])
            dy = min(0.0, lo_[1] - m[1] - pred[1]) + max(0.0, hi_[1] + m[1] - pred[3])
            pred = pred + np.array([dx, dy, dx, dy])
        else:
            lost += 1
            if lost > lost_max:
                break
        box = pred
        out[fi] = box.copy() if lost == 0 else None
        prev_f = fi
    rd.close()
    # keep the frames up to the last one where the person was found; fill the stride gaps
    keep = sorted(f for f, b in out.items() if b is not None)
    res = {}
    for a, b in zip(keep[:-1], keep[1:]):
        lo, hi = min(a, b), max(a, b)
        for f in range(lo, hi + 1):
            t = (f - lo) / max(hi - lo, 1)
            res[f] = (1 - t) * out[lo] + t * out[hi]
    if keep:
        res[keep[0]] = out[keep[0]]
        nxt = keep[-1] + direction
        for f in range(keep[-1], nxt + direction * (step - 1) + direction, direction):
            if (direction > 0 and f < stop) or (direction < 0 and f >= stop):
                res.setdefault(f, out[keep[-1]])
    with open(out_path + ".tmp", "wb") as fh:
        import pickle
        pickle.dump(res, fh)
    os.replace(out_path + ".tmp", out_path)
    return out_path


def follow_beyond_tracks(video, boxes, segs, shot_of, cam_R, focal, W, H, fps, wb_path, cdir, pool, max_s=3.0):
    """Follow the character before its tracks start and after they end
    (within each shot), so legs or a hand seen before (or after) the whole body
    are captured too. Returns the new frames' boxes as extra segments."""
    import pickle
    N = len(boxes)
    taken = ~np.isnan(boxes[:, 0])
    starts = np.r_[0, np.where(shot_of[1:] != shot_of[:-1])[0] + 1]
    ends = np.r_[starts[1:], N]
    jobs = []
    K = int(round(max_s * fps))
    for (s, e) in segs:
        sh = shot_of[s]
        a, b = starts[sh], ends[sh]
        if s > a and not taken[s - 1]:
            lo = s - 1
            while lo > a and not taken[lo - 1] and s - lo < K:
                lo -= 1
            jobs.append((video, s, boxes[s].copy(), -1, lo, cam_R, focal, W, H, fps, wb_path,
                         os.path.join(cdir, f"follow_{s:07d}_b.pkl")))
        if e < b and not taken[e]:
            hi = e
            while hi < b and not taken[hi] and hi - e < K:
                hi += 1
            jobs.append((video, e - 1, boxes[e - 1].copy(), 1, hi, cam_R, focal, W, H, fps, wb_path,
                         os.path.join(cdir, f"follow_{e - 1:07d}_f.pkl")))
    out = boxes.copy()
    if not jobs:
        return out, []
    for path in pool.imap_unordered(follow_job, jobs):
        with open(path, "rb") as fh:
            res = pickle.load(fh)
        for f, bx in res.items():
            if 0 <= f < N and np.isnan(out[f, 0]):
                out[f] = bx
    new = ~np.isnan(out[:, 0]) & ~taken
    extra = []
    f = 0
    while f < N:
        if not new[f]:
            f += 1
            continue
        s0 = f
        while f < N and new[f] and shot_of[f] == shot_of[s0]:
            f += 1
        extra.append((s0, f))
    return out, extra


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
    wbz = np.zeros((n, 133), np.float16)
    wbody = WholeBody(wb_path) if wb_path else None
    wb_stride = max(1, int(round(fps / 30.0)))
    hc = np.zeros((n, 2, 2), np.float32)
    seg = np.full((n, 33), -1.0, np.float32)  # person mask at each landmark (-1: unknown)
    pose = PoseDetector(heavy=True, num_poses=1, video=True, min_conf=0.3, masks=True)
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
                pose_img = PoseDetector(heavy=True, num_poses=1, video=False, min_conf=0.25, masks=True)
            res = pose_img.detect(crop, x0, y0, sc, sc)
            ok = bool(res) and box_iou(landmark_box(res[0].lm2d, res[0].vis), b) > 0.2
            mask = pose_img.last_mask
        else:
            mask = pose.last_mask
        if wbody is not None and (fi % wb_stride == 0 or i == n - 1):
            # (above 30 fps the whole-body model runs on every 2nd frame; the
            # gaps are interpolated, joints barely move in 1/60 s)
            # box: the tracked box, widened by the body model's own landmarks
            bb = np.array(b, float)
            if ok:
                lb = landmark_box(res[0].lm2d, res[0].vis)
                bb = np.array([min(bb[0], lb[0]), min(bb[1], lb[1]), max(bb[2], lb[2]), max(bb[3], lb[3])])
            wb2d[i], wbs[i] = wbody.detect(rgb, bb)
            wbz[i] = wbody.last_z
            # a box that cuts the body off (camera low or moving, tracker lagging)
            # leaves the model unsure: try larger boxes, keep the surest reading
            if wbs[i, 5:23].mean() < 0.6:
                c = (bb[:2] + bb[2:]) / 2
                for k in (1.6, 2.2):
                    hb = (bb[2:] - bb[:2]) * k / 2
                    kp2, sc2 = wbody.detect(rgb, np.r_[c - hb, c + hb])
                    if sc2[5:23].mean() > wbs[i, 5:23].mean() + 0.05:
                        wb2d[i], wbs[i], wbz[i] = kp2, sc2, wbody.last_z
        if not ok:
            prev_hand = [None, None]
            continue
        r = res[0]
        lm2d[i], vis[i], world[i] = r.lm2d, r.vis, r.world
        if mask is not None:  # is the person (and not something in front) at each joint?
            seg[i] = mask_at(mask, r.lm2d, x0, y0, sc)
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
                        h2d=h2d, hw=hw, hs=hs, hc=hc, wb2d=wb2d, wbs=wbs, wbz=wbz, seg=seg)
    os.replace(out_path + ".tmp.npz", out_path)
    return out_path


def capture(video, workdir, char_id, progress=None, cancel=None, workers=None):
    progress = progress or (lambda *a: None)
    cancel = cancel or (lambda: False)
    A, boxes, segs = character_boxes(workdir, char_id)
    N, fps = A["frames"], A["fps"]
    boxes = stabilize_boxes(boxes, segs, fps)
    npz = np.load(os.path.join(workdir, "analysis.npz"))
    cdir = os.path.join(workdir, f"capture_c{char_id}")
    os.makedirs(cdir, exist_ok=True)
    from .detect import ensure_model
    progress(0.0, "Getting the whole-body model (one-time download, 370 MB)")
    wb_path = ensure_model("wholebody", lambda f: progress(0.0, f"Downloading the whole-body model: {f * 100:.0f}%"))
    tag = (f"w{CAPTURE_VERSION}" if wb_path else f"m{CAPTURE_VERSION}")
    workers = workers or max(1, min(6, (os.cpu_count() or 2) - 1))
    ctx = mpc.get_context("spawn")
    extra = []
    if wb_path:
        # context from both directions: follow the character before / after its tracks
        progress(0.0, "Following the character before and after it is fully in view")
        from .mathutil import quat_to_mat
        with ctx.Pool(workers) as pool:
            boxes, extra = follow_beyond_tracks(video, boxes, segs, npz["shot_of"],
                                                quat_to_mat(npz["cam_q"].astype(np.float64)),
                                                npz["focal"].astype(np.float64), A["width"], A["height"], fps,
                                                wb_path, cdir, pool)
    else:
        boxes, segs = extend_at_edges(boxes, segs, npz["shot_of"], A["width"], A["height"], fps)
    jobs = []
    for (s, e) in list(segs) + list(extra):
        for cs in range(s, e, CHUNK):
            ce = min(e, cs + CHUNK)
            if ce - cs < 1:
                continue
            jobs.append((video, cs, ce, boxes[cs:ce].copy(), fps, os.path.join(cdir, f"chunk{tag}_{cs:07d}_{ce:07d}.npz"), wb_path))
    total = sum(j[2] - j[1] for j in jobs)
    # (each worker holds its own models, ~300 MB: the count is capped)
    t0 = time.time()
    done = 0
    if jobs:
        with ctx.Pool(max(1, min(len(jobs), workers))) as pool:
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
        "seg": np.full((N, 33), -1.0, np.float32), "wbz": np.zeros((N, 133), np.float16),
    }
    for j in jobs:
        d = np.load(j[5])
        for k in out:
            out[k][j[1]:j[2]] = d[k]
    out["tracked"] = ~np.isnan(boxes[:, 0])
    out["version"] = np.array(CAPTURE_VERSION)
    fill_wholebody_gaps(out["wb2d"], out["wbs"], [(j[1], j[2]) for j in jobs], wbz=out["wbz"])
    return out


def fill_wholebody_gaps(wb2d, wbs, ranges, max_gap=3, wbz=None):
    """Interpolate the whole-body keypoints over the frames it was not run on."""
    for s, e in ranges:
        have = np.where(wbs[s:e].any(1))[0] + s
        for a, b in zip(have[:-1], have[1:]):
            if 1 < b - a <= max_gap:
                for f in range(a + 1, b):
                    t = (f - a) / (b - a)
                    wb2d[f] = (1 - t) * wb2d[a] + t * wb2d[b]
                    wbs[f] = np.minimum(wbs[a], wbs[b])
                    if wbz is not None:
                        wbz[f] = (1 - t) * wbz[a].astype(np.float32) + t * wbz[b].astype(np.float32)

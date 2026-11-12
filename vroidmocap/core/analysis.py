"""Pass 1: scan the whole video once.

Produces
  * shots (hard cuts),
  * per-frame camera: rotation (relative within a shot + gravity alignment), zoom/focal,
  * person tracklets with landmarks and appearance features,
  * characters = tracklets clustered by identity (recurring characters are merged
    even across cuts / when they leave and re-enter the frame),
  * each character's height relative to the scenery (single-view metrology
    against the horizon, linked across shots through shared characters).

Everything is saved in the work directory so pass 2 can be run for any
character without repeating this scan.
"""
from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np

from . import camera as cam
from . import scene
from .detect import (L_ANK, L_FOOT, L_HEEL, L_HIP, L_SH, NOSE, R_ANK, R_FOOT, R_HEEL, R_HIP,
                     R_SH, Embedder, PoseDetector, PoseResult, box_iou, crop_square, landmark_box)
from .mathutil import mat_to_quat, normalize, quat_to_mat, slerp
from .video import VideoReader

ANALYSIS_VERSION = 11
PARALLEL_SHOTS = True
USE_PERSON_DETECTOR = True  # person boxes from the Human-Art detector (else the pose model's own)


class Cancelled(Exception):
    pass


# ----------------------------------------------------------------------------- tracking
class Track:
    _next = 0

    def __init__(self, shot):
        self.id = Track._next
        Track._next += 1
        self.shot = shot
        self.frames, self.boxes, self.lm2d, self.vis, self.world, self.crop_c = [], [], [], [], [], []
        self.embs, self.hists = [], []
        self.misses = 0
        self.vel = np.zeros(4)
        self.det_n = 0  # frames where the person detector gave the box

    def add(self, fi, box, pr, cc):
        if self.boxes:
            self.vel = 0.6 * self.vel + 0.4 * (box - self.boxes[-1])
        self.frames.append(fi)
        self.boxes.append(box)
        self.lm2d.append(pr.lm2d.astype(np.float32))
        self.vis.append(pr.vis.astype(np.float32))
        self.world.append(pr.world.astype(np.float32))
        self.crop_c.append(np.asarray(cc, np.float32))
        self.misses = 0

    def predict(self):
        return self.boxes[-1] + self.vel

    def mean_vis(self):
        return float(np.mean([v.mean() for v in self.vis])) if self.vis else 0.0


def contains(outer, inner, frac=0.7):
    """True when most of `inner` lies inside `outer` (or vice versa)."""
    ix0, iy0 = max(outer[0], inner[0]), max(outer[1], inner[1])
    ix1, iy1 = min(outer[2], inner[2]), min(outer[3], inner[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    a1 = (inner[2] - inner[0]) * (inner[3] - inner[1])
    a2 = (outer[2] - outer[0]) * (outer[3] - outer[1])
    return inter > frac * min(a1, a2)


def damp_box(old, new, grow=1.2, shrink=0.85):
    """Limit how fast a track's box may change size between analysed frames.
    Stops the crop from collapsing onto part of the body (or exploding)."""
    ow, oh = old[2] - old[0], old[3] - old[1]
    nw, nh = new[2] - new[0], new[3] - new[1]
    w = float(np.clip(nw, ow * shrink, ow * grow))
    h = float(np.clip(nh, oh * shrink, oh * grow))
    cx, cy = (new[0] + new[2]) / 2, (new[1] + new[3]) / 2
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def same_person(a, b):
    """Two tracks on the same body: overlapping boxes and nearby joints."""
    if box_iou(a.boxes[-1], b.boxes[-1]) > 0.45 or contains(a.boxes[-1], b.boxes[-1], 0.8):
        la, lb = a.lm2d[-1], b.lm2d[-1]
        scale = max(a.boxes[-1][3] - a.boxes[-1][1], 1.0)
        d = np.median(np.linalg.norm(la - lb, axis=1)) / scale
        return d < 0.15 or box_iou(a.boxes[-1], b.boxes[-1]) > 0.6
    return False


CORE = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def is_human_track(t):
    """Reject tracks on objects (balloons, posters...): the pose model's
    landmarks on non-people are low-confidence and jump around wildly.
    A track the person detector found most of the time is a person (the pose
    model can be unsure on stylised characters)."""
    if getattr(t, "det_n", 0) >= 0.5 * len(t.frames):
        return True
    V = np.array(t.vis, np.float64)[:, CORE]
    L = np.array(t.lm2d, np.float64)[:, CORE]
    bx = np.array(t.boxes, np.float64)
    bh = np.maximum(np.maximum(bx[:, 3] - bx[:, 1], bx[:, 2] - bx[:, 0]), 1.0)
    jit = 0.0
    if len(L) > 2:
        # only joints seen in both frames (a close-up's joints outside the
        # picture are guesses that jump around; the person is still a person)
        both = (V[1:] > 0.6) & (V[:-1] > 0.6)
        d = np.linalg.norm(np.diff(L, axis=0), axis=2) / bh[1:, None]
        per = [np.median(d[k][both[k]]) for k in range(len(d)) if both[k].sum() >= 3]
        jit = float(np.median(per)) if per else 0.0
    # a person seen from above / from behind / partly out of frame has lower
    # landmark confidence but still moves coherently: only reject when the
    # landmarks are both unsure and unstable
    if V.mean() < 0.5 or jit > 0.15:
        return False
    if V.mean() < 0.7 and jit > 0.08:
        return False
    return True


def part_histograms(rgb, lm2d, vis, box=None):
    """Colour histograms of hair/head, torso and legs (hue-saturation). With
    unsure landmarks and a person box: the box's top, middle and bottom."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, w = rgb.shape[:2]
    regions = []
    if box is not None and float(np.mean(vis[[NOSE, L_SH, R_SH, L_HIP, R_HIP]])) < 0.5:
        x0, y0, x1, y1 = box
        bh = y1 - y0
        for a, b in ((0.0, 0.2), (0.2, 0.55), (0.55, 1.0)):
            regions.append((x0, y0 + a * bh, x1, y0 + b * bh))
    else:
        sh = np.linalg.norm(lm2d[L_SH] - lm2d[R_SH])
        unit = max(sh, 6.0)
        nose = lm2d[NOSE]
        regions.append((nose[0] - unit * 0.9, nose[1] - unit * 1.1, nose[0] + unit * 0.9, nose[1] + unit * 0.4))
        torso = np.array([lm2d[L_SH], lm2d[R_SH], lm2d[L_HIP], lm2d[R_HIP]])
        regions.append((*torso.min(0), *torso.max(0)))
        legs = np.array([lm2d[L_HIP], lm2d[R_HIP], lm2d[L_ANK], lm2d[R_ANK]])
        regions.append((*legs.min(0), *legs.max(0)))
    out = []
    for (x0, y0, x1, y1) in regions:
        x0, y0 = int(np.clip(x0, 0, w - 1)), int(np.clip(y0, 0, h - 1))
        x1, y1 = int(np.clip(x1, x0 + 1, w)), int(np.clip(y1, y0 + 1, h))
        patch = hsv[y0:y1, x0:x1]
        if patch.size == 0:
            out.append(np.zeros(16 * 4 + 4, np.float32))
            continue
        hs = cv2.calcHist([patch], [0, 1], None, [16, 4], [0, 180, 0, 256]).flatten()
        # include low-saturation brightness bins (white/black/grey clothes)
        v = cv2.calcHist([patch], [2], None, [4], [0, 256]).flatten()
        hv = np.concatenate([hs, v * 0.5]).astype(np.float32)
        out.append(hv / max(hv.sum(), 1e-6))
    return np.concatenate(out)


# ----------------------------------------------------------------------------- cuts
def detect_cuts(video_path, progress=None, cancel=None):
    """Exact frame of every camera cut, checked on EVERY frame.

    A cut is a sudden spike in frame-to-frame difference compared with the
    neighbouring frames (so fast pans / whip pans, which differ a lot on every
    frame, are not cuts) or a large colour-histogram change. One-frame inserts
    (a single frame of another camera) produce two cuts and become their own
    tiny shot."""
    rd = VideoReader(video_path)
    N, fps = rd.frame_count, rd.fps
    hist_c = np.ones(N)
    mad = np.zeros(N)
    prev_g = prev_h = None
    for f, bgr in rd.iter_frames(1):
        if cancel and cancel():
            rd.close()
            raise Cancelled()
        sm = cv2.resize(bgr, (160, 90), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY).astype(np.int16)
        hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
        h = h / max(h.sum(), 1)
        if prev_g is not None:
            hist_c[f] = float(np.minimum(h, prev_h).sum())
            mad[f] = float(np.mean(np.abs(g - prev_g)))
        prev_g, prev_h = g, h
        if progress and f % 600 == 0:
            progress(0.08 * f / max(N, 1), f"Finding camera cuts: frame {f + 1}/{N}")
    rd.close()
    k = max(2, int(round(0.25 * fps)))
    spike = np.zeros(N, bool)
    for f in range(1, N):
        lo, hi = max(1, f - k), min(N, f + k + 1)
        neigh = np.concatenate([mad[lo:f], mad[f + 1:hi]])
        base = float(np.median(neigh)) if len(neigh) else 0.0
        spike[f] = (mad[f] > 10 and mad[f] > 4.0 * base + 4.0) or hist_c[f] < 0.5
    cuts = []
    f = 1
    while f < N:
        if not spike[f]:
            f += 1
            continue
        s0 = f
        while f < N and spike[f]:
            f += 1
        n = f - s0
        if n <= 2:                      # a cut (possibly with one transition frame)
            cuts.append(s0)
        elif n <= 0.25 * fps:           # a camera snap: its own short transition shot
            cuts.extend([s0, f] if f < N else [s0])
        # longer: an ordinary (fast) camera move, not a cut
    return sorted(set(cuts))


# ----------------------------------------------------------------------------- main pass
class Analyzer:
    def __init__(self, video_path, workdir, progress=None, cancel=None, stride=None,
                 scan_every_s=0.35, max_people=6):
        self.video_path = video_path
        self.workdir = workdir
        self.progress = progress or (lambda *a, **k: None)
        self.cancel = cancel or (lambda: False)
        self.stride_override = stride
        self.scan_every_s = scan_every_s
        self.max_people = max_people
        os.makedirs(workdir, exist_ok=True)

    # -- detection helpers --------------------------------------------------
    def _scan(self, rgb):
        """Find people anywhere in the frame (full frame + overlapping tiles)."""
        h, w = rgb.shape[:2]
        tiles = [(0, 0, w, h)]
        side = min(h, w)
        if w > h * 1.2:
            n = int(np.ceil((w - side) / (side * 0.6))) + 1
            for i in range(n):
                x = int(round(i * (w - side) / max(n - 1, 1)))
                tiles.append((x, 0, x + side, h))
        # smaller tiles for distant people
        half = int(side * 0.62)
        for ty in (0, h - half):
            nx = int(np.ceil((w - half) / (half * 0.7))) + 1
            for i in range(nx):
                x = int(round(i * (w - half) / max(nx - 1, 1)))
                tiles.append((x, ty, x + half, ty + half))
        found = []
        for (x0, y0, x1, y1) in tiles:
            crop = rgb[y0:y1, x0:x1]
            for pr in self.scan_pose.detect(crop, x0, y0):
                if pr.vis.mean() < 0.45:
                    continue
                box = landmark_box(pr.lm2d, pr.vis)
                if (box[3] - box[1]) < 0.06 * h:
                    continue
                found.append((box, pr))
        # non-max suppression (keep the most visible)
        found.sort(key=lambda t: -t[1].vis.mean())
        keep = []
        for b, pr in found:
            if all(box_iou(b, k[0]) < 0.4 for k in keep):
                keep.append((b, pr))
        # a "person" inside a much larger one is a part of it (frills, a sleeve,
        # a hand: the pose model finds small people in them)
        area = lambda b: max(b[2] - b[0], 1.0) * max(b[3] - b[1], 1.0)
        keep = [(b, pr) for b, pr in keep
                if not any(area(o) > 3.0 * area(b) and contains(o, b, 0.8) for o, _ in keep)]
        return keep

    def _refine_box(self, rgb, box):
        """Pose inside a person box from the detector (the box stays the
        detector's; a pose model that cannot read the body gives an empty pose)."""
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        size = max(box[2] - box[0], box[3] - box[1]) * 1.2
        crop, x0, y0, sc = crop_square(rgb, cx, cy, size, out=384)
        res = self.track_pose.detect(crop, x0, y0, sc, sc)
        if res:
            pr = res[0]
            # the reading must be the person in the box, not a small part of it
            lb = landmark_box(pr.lm2d, pr.vis)
            if (lb[3] - lb[1]) * (lb[2] - lb[0]) > 0.25 * (box[3] - box[1]) * (box[2] - box[0]):
                return pr, (cx, cy)
        lm = np.tile([[cx, cy]], (33, 1)).astype(np.float32)
        return PoseResult(lm, np.zeros(33, np.float32), np.zeros((33, 3), np.float32)), (cx, cy)

    def _refine(self, rgb, box):
        """Run single-person pose on a square crop around `box`. A box that has
        shrunk onto part of a person (a frilly skirt, a sleeve: the pose model
        happily finds a tiny "person" in it) is checked with a wider crop as
        well; the wider reading wins when it is a larger person that contains
        the small one and is about as sure."""
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        bsz = max(box[2] - box[0], box[3] - box[1])
        best = None
        for k, mult in enumerate((1.35, 2.6)):
            if k and bsz * mult > 1.6 * max(rgb.shape[:2]):
                break
            crop, x0, y0, sc = crop_square(rgb, cx, cy, bsz * mult, out=384)
            res = self.track_pose.detect(crop, x0, y0, sc, sc)
            if not res or res[0].vis.mean() < 0.35:
                continue
            pr = res[0]
            lb = landmark_box(pr.lm2d, pr.vis)
            area = max(lb[2] - lb[0], 1.0) * max(lb[3] - lb[1], 1.0)
            if best is None:
                best = (pr, area, lb)
            elif area > 2.0 * best[1] and pr.vis.mean() > best[0].vis.mean() - 0.1 and contains(lb, best[2], 0.6):
                best = (pr, area, lb)
        if best is None:
            return None, None
        return best[0], (cx, cy)

    # -- run -------------------------------------------------------------------
    def run(self):
        t_start = time.time()
        rd = VideoReader(self.video_path)
        N, fps, W, H = rd.frame_count, rd.fps, rd.width, rd.height
        stride = self.stride_override or max(1, int(round(fps / 30.0)))
        scan_every = max(1, int(round(self.scan_every_s * fps / stride)))
        from .detect import ensure_model
        pdp = None
        if USE_PERSON_DETECTOR:
            self.progress(0.0, "Getting the person detector (one-time download, 20 MB)")
            pdp = ensure_model("person", lambda f: self.progress(0.0, f"Downloading the person detector: {f * 100:.0f}%"))

        scan_path = os.path.join(self.workdir, "scan_v6.pkl" if USE_PERSON_DETECTOR else "scan_v5.pkl")
        if os.path.exists(scan_path):
            import pickle
            with open(scan_path, "rb") as fh:
                sc = pickle.load(fh)
            if sc.get("stride") == stride and sc.get("frames") == N:
                rd.close()
                return self._finish(t_start, N, fps, W, H, stride, sc)
        rd.close()
        cut_frames = detect_cuts(self.video_path, self.progress, self.cancel)
        ranges = scan_ranges(N, stride, cut_frames, SCAN_PIECES_PER_WORKER * self._scan_workers())
        if len(ranges) > 1 and PARALLEL_SCAN:
            parts = self._scan_parallel(ranges, N, stride, scan_every, cut_frames, pdp, t_start)
        else:
            parts = [self._scan_range(0, N, stride, scan_every, cut_frames, pdp, t_start=t_start)]
        sc = merge_scans(parts, stride, N, cut_frames)
        import pickle
        with open(scan_path + ".tmp", "wb") as fh:
            pickle.dump(sc, fh, protocol=4)
        os.replace(scan_path + ".tmp", scan_path)
        return self._finish(t_start, N, fps, W, H, stride, sc)

    def _scan_workers(self):
        from .detect import worker_count
        return worker_count(0.8) if PARALLEL_SCAN else 1

    def _scan_parallel(self, ranges, N, stride, scan_every, cut_frames, pdp, t_start):
        """The scan of every range at once on the CPU's cores (shots are scanned
        independently: tracks and the scenery points start afresh at every cut)."""
        import multiprocessing as mpc
        ctx = mpc.get_context("spawn")
        counter = ctx.Value("i", 0)
        jobs = [(self.video_path, self.workdir, self.stride_override, self.scan_every_s, self.max_people,
                 a, b, stride, scan_every, cut_frames, pdp) for (a, b) in ranges]
        parts = [None] * len(jobs)
        with ctx.Pool(min(len(jobs), self._scan_workers()), initializer=_scan_init, initargs=(counter,)) as pool:
            res = [pool.apply_async(_scan_range_job, (j,)) for j in jobs]
            while not all(r.ready() for r in res):
                if self.cancel():
                    pool.terminate()
                    raise Cancelled()
                done = counter.value
                el = time.time() - t_start
                frac = done / max(N, 1)
                eta = el / max(frac, 1e-6) * (1 - frac)
                self.progress(0.08 + frac * 0.82, f"Scanning frame {done}/{N} - ~{eta / 60:.1f} min left")
                time.sleep(0.5)
            for i, r in enumerate(res):
                parts[i] = r.get()
        return parts

    def _scan_range(self, start, end, stride, scan_every, cut_frames, pdp, t_start=None, counter=None):
        """Scan analysed frames [start, end) (start: a cut, or 0): people tracks,
        the scenery's motion and appearance samples. Returns this range's part."""
        from .detect import PersonDetector
        N = end
        self.scan_pose = PoseDetector(heavy=False, num_poses=3, min_conf=0.35)
        self.track_pose = PoseDetector(heavy=False, num_poses=1, min_conf=0.3)
        self.person_det = PersonDetector(pdp, threads=1 if counter is not None else 2) if pdp else None
        embedder = Embedder()
        rd = VideoReader(self.video_path)
        W, H, fps = rd.width, rd.height, rd.fps
        N_all = rd.frame_count
        cut_set = set(cut_frames)
        cam_scale = 640.0 / W
        cw, ch = 640, int(round(H * cam_scale))
        prev_gray = prev_mask = None
        bg_pts, bg_ids, next_bg_id = None, np.zeros(0, np.int64), 0
        tracks, finished = [], []
        # (the cuts this range starts at: a scan of the whole video records them here)
        shot_starts = [c for c in cut_frames if start - stride < c <= start] or [start]
        frame_ids, H_rel, H_ok, H_ids = [], [], [], []
        last_scan = -10 ** 9
        lost_limit = max(2, int(round(0.6 * fps / stride)))
        thumbs = {}
        k = start // stride  # (as when the whole video is scanned in one go)
        prev_fi = -1
        next_bg_id = start * 1000  # (unique across ranges)
        for fi, bgr in rd.iter_frames(stride, start, end):
            if self.cancel():
                raise Cancelled()
            cuts_here = [c for c in range(prev_fi + 1, fi + 1) if c in cut_set] if prev_fi >= 0 else []
            prev_fi = fi
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            small = cv2.resize(bgr, (cw, ch), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            cut = bool(cuts_here)

            # ---- camera motion (background only): long feature tracks over the
            # scenery (people masked out), kept as matched pairs with track ids
            pair, pids, ok = None, None, False
            if cut:
                bg_pts = None
            if prev_gray is not None and not cut and bg_pts is not None and len(bg_pts) >= 12:
                p0 = bg_pts.reshape(-1, 1, 2)
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, winSize=(21, 21), maxLevel=3)
                back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, nxt, None, winSize=(21, 21), maxLevel=3)
                fb = np.linalg.norm((back - p0).reshape(-1, 2), axis=1)
                q = nxt.reshape(-1, 2)
                good = ((st.ravel() == 1) & (st2.ravel() == 1) & (fb < 0.7) &
                        (q[:, 0] >= 0) & (q[:, 0] < cw) & (q[:, 1] >= 0) & (q[:, 1] < ch))
                if good.sum() >= 12:
                    a, b = p0.reshape(-1, 2)[good], q[good]
                    Hs, inl = cv2.findHomography(a, b, cv2.RANSAC, 2.0)
                    ratio = float(inl.sum()) / len(a) if inl is not None else 0.0
                    resid = 99.0
                    if Hs is not None:
                        warped = cv2.warpPerspective(prev_gray, Hs, (cw, ch))
                        valid = cv2.warpPerspective(np.full_like(prev_gray, 1), Hs, (cw, ch)) > 0
                        if valid.sum() > 0.3 * valid.size:
                            resid = float(np.mean(cv2.absdiff(warped, gray)[valid]))
                    if ratio < 0.3 or resid > 14.0:
                        good[:] = False  # fast pan / blur: no usable camera motion for this step (not a cut)
                    else:
                        pair = np.concatenate([a, b], 1).astype(np.float32) / cam_scale
                        pids = bg_ids[good].astype(np.int32)
                        ok = True
                bg_pts, bg_ids = q[good], bg_ids[good]
            if cut:
                for c in cuts_here:
                    if c not in shot_starts:
                        shot_starts.append(c)
                finished.extend(tracks)
                tracks = []
                last_scan = -10 ** 9
            # ---- people
            dets = self.person_det.detect(bgr) if self.person_det is not None else None
            used = set()
            if dets is not None:
                # tracks follow the person detector's boxes (matched by overlap)
                pairs_ = sorted(((max(box_iou(d[0], t.predict()), box_iou(d[0], t.boxes[-1])), ti, di)
                                 for ti, t in enumerate(tracks) for di, d in enumerate(dets)), reverse=True)
                got = set()
                for iou, ti, di in pairs_:
                    if iou < 0.25 or ti in got or di in used:
                        continue
                    got.add(ti)
                    used.add(di)
                    t = tracks[ti]
                    pr, cc = self._refine_box(rgb, dets[di][0])
                    t.add(fi, dets[di][0], pr, cc)
                    t.det_n += 1
            for t in tracks:
                if t.frames and t.frames[-1] == fi:
                    continue
                pred = t.predict()
                pr, cc = self._refine(rgb, pred)
                if pr is not None and pr.vis.mean() > 0.45:
                    box = landmark_box(pr.lm2d, pr.vis)
                    small = dets is not None and (box[3] - box[1]) * (box[2] - box[0]) < \
                        0.5 * (t.boxes[-1][3] - t.boxes[-1][1]) * (t.boxes[-1][2] - t.boxes[-1][0])
                    if (box_iou(box, pred) > 0.15 or contains(pred, box, 0.6)) and not small:
                        grown = (box[3] - box[1]) * (box[2] - box[0]) > 2.0 * (pred[3] - pred[1]) * (pred[2] - pred[0])
                        t.add(fi, box if grown else damp_box(t.boxes[-1], box), pr, cc)
                        continue
                t.misses += 1
            # duplicates: two tracks locked onto the same person
            tracks.sort(key=lambda t: -len(t.frames))
            alive = []
            for t in tracks:
                if t.misses == 0 and any(a.misses == 0 and same_person(a, t) for a in alive):
                    finished.append(t)
                    continue
                alive.append(t)
            tracks = alive
            if dets is not None:
                for di, (box, score) in enumerate(dets):
                    if di in used or score < 0.5 or len(tracks) >= self.max_people:
                        continue
                    if any(box_iou(box, t.boxes[-1]) > 0.3 or contains(t.boxes[-1], box, 0.7) for t in tracks):
                        continue
                    pr, cc = self._refine_box(rgb, box)
                    t = Track(len(shot_starts) - 1)
                    t.add(fi, box, pr, cc)
                    t.det_n += 1
                    tracks.append(t)
            elif k - last_scan >= scan_every or not tracks:
                if k - last_scan >= scan_every or k % max(1, scan_every // 3) == 0:
                    last_scan = k
                    area = lambda b: max(b[2] - b[0], 1.0) * max(b[3] - b[1], 1.0)
                    for box, pr in self._scan(rgb):
                        # a track that shrank onto part of this person: it becomes the whole person again
                        part = [t for t in tracks if t.misses < 3 and area(box) > 3.0 * area(t.boxes[-1])
                                and contains(box, t.boxes[-1], 0.8)]
                        if part:
                            pr2, cc = self._refine(rgb, box)
                            if pr2 is not None and pr2.vis.mean() > 0.45:
                                t = max(part, key=lambda t: len(t.frames))
                                if t.frames and t.frames[-1] == fi:
                                    for lst in (t.frames, t.boxes, t.lm2d, t.vis, t.world, t.crop_c):
                                        lst.pop()
                                t.add(fi, landmark_box(pr2.lm2d, pr2.vis), pr2, cc)
                            continue
                        if any(box_iou(box, t.boxes[-1]) > 0.2 or contains(t.boxes[-1], box) for t in tracks if t.misses < 3):
                            continue
                        if len(tracks) >= self.max_people:
                            break
                        pr2, cc = self._refine(rgb, box)
                        if pr2 is None:
                            continue
                        t = Track(len(shot_starts) - 1)
                        t.add(fi, landmark_box(pr2.lm2d, pr2.vis), pr2, cc)
                        tracks.append(t)
            # appearance samples
            for t in tracks:
                if t.misses == 0 and (len(t.frames) - 1) % 4 == 0:
                    b = t.boxes[-1]
                    x0, y0 = int(max(0, b[0])), int(max(0, b[1]))
                    x1, y1 = int(min(W, b[2])), int(min(H, b[3]))
                    if x1 - x0 > 8 and y1 - y0 > 8:
                        patch = rgb[y0:y1, x0:x1]
                        t.embs.append(embedder.embed(patch))
                        t.hists.append(part_histograms(rgb, t.lm2d[-1], t.vis[-1], t.boxes[-1] if getattr(t, 'det_n', 0) else None))
                        area = (x1 - x0) * (y1 - y0) * t.vis[-1].mean()
                        if t.id not in thumbs or area > thumbs[t.id][0]:
                            pad = int(0.08 * (y1 - y0))
                            px0, py0 = max(0, x0 - pad), max(0, y0 - pad)
                            thumbs[t.id] = (area, bgr[py0:min(H, y1 + pad), px0:min(W, x1 + pad)].copy(), fi)
            still = []
            for t in tracks:
                if t.misses > lost_limit:
                    finished.append(t)
                else:
                    still.append(t)
            tracks = still

            # people mask: background tracks never sit on a person
            prev_mask = np.full(gray.shape, 255, np.uint8)
            for t in tracks:
                b = (t.boxes[-1] * cam_scale).astype(int)
                pad = int(6 + 0.08 * max(b[2] - b[0], b[3] - b[1]))
                cv2.rectangle(prev_mask, (b[0] - pad, b[1] - pad), (b[2] + pad, b[3] + pad), 0, -1)
            if bg_pts is not None and len(bg_pts):
                qi = np.clip(bg_pts.astype(int), 0, [cw - 1, ch - 1])
                keep = prev_mask[qi[:, 1], qi[:, 0]] > 0
                bg_pts, bg_ids = bg_pts[keep], bg_ids[keep]
            n_have = 0 if bg_pts is None else len(bg_pts)
            if n_have < 350:
                m2 = prev_mask.copy()
                if n_have:
                    for x_, y_ in bg_pts.astype(int):
                        cv2.circle(m2, (int(x_), int(y_)), 8, 0, -1)
                newp = cv2.goodFeaturesToTrack(gray, 450 - n_have, 0.01, 8, mask=m2)
                if newp is not None:
                    newp = newp.reshape(-1, 2).astype(np.float32)
                    bg_pts = newp if bg_pts is None or not n_have else np.concatenate([bg_pts, newp])
                    bg_ids = np.concatenate([bg_ids[:n_have], np.arange(next_bg_id, next_bg_id + len(newp))])
                    next_bg_id += len(newp)
            frame_ids.append(fi)
            H_rel.append(pair)
            H_ids.append(pids)
            H_ok.append(ok)
            prev_gray = gray
            k += 1
            if counter is not None:
                with counter.get_lock():
                    counter.value += stride
            elif k % 5 == 0:
                el = time.time() - t_start
                frac = fi / max(N_all - 1, 1)
                eta = el / max(frac, 1e-6) * (1 - frac)
                self.progress(0.08 + frac * 0.82, f"Scanning frame {fi + 1}/{N_all} - {len(tracks)} people in view - ~{eta / 60:.1f} min left")
        finished.extend(tracks)
        rd.close()
        self.scan_pose.close()
        self.track_pose.close()
        embedder.close()
        return {"start": start, "frame_ids": frame_ids, "H_rel": H_rel, "H_ok": H_ok, "H_ids": H_ids,
                "shot_starts": shot_starts, "tracks": finished, "thumbs": thumbs}

    def _finish(self, t_start, N, fps, W, H, stride, sc):
        self.progress(0.91, "Solving cameras and identities")
        merged = set(sc["shot_starts"]) | {0}
        # where camera tracking failed inside a shot (heavy blur, fast snap), the
        # camera orientation is re-established afterwards: start a new shot there
        fids = list(sc["frame_ids"])
        ok = sc["H_ok"]
        n_split = 0
        for i in range(2, len(fids)):
            starts_after_cut = any(fids[i - 1] < c <= fids[i] for c in sc["shot_starts"])
            if not ok[i] and ok[i - 1] and not starts_after_cut:
                j = i
                while j < len(fids) and not ok[j]:
                    j += 1
                if j < len(fids):
                    merged.add(fids[j])
                    n_split += 1
        merged = sorted(merged)
        frame_shot = np.searchsorted(np.array(merged), np.arange(N), side="right") - 1
        for t in sc["tracks"]:
            t.shot = int(frame_shot[t.frames[0]])
        min_len = max(3, int(round(0.4 * fps / stride)))
        # a track shorter than min_len is usually a glimpse or a false detection,
        # except in a short shot (a fast cut): there it may be all there is
        shot_len = np.diff(np.r_[merged, N])

        def long_enough(t):
            if len(t.frames) >= min_len:
                return True
            n_shot = shot_len[t.shot] / stride
            return len(t.frames) >= 2 and len(t.frames) >= 0.6 * n_shot
        tracklets = [t for t in sc["tracks"] if long_enough(t) and t.embs and is_human_track(t)]
        result = self._solve(N, fps, W, H, stride, np.array(sc["frame_ids"]), sc["H_rel"], sc["H_ok"], sc["H_ids"],
                             merged, tracklets, sc["thumbs"])
        result["elapsed_s"] = time.time() - t_start
        self.progress(1.0, "Analysis complete")
        return result

    # -- post-processing ---------------------------------------------------------
    def _solve(self, N, fps, W, H, stride, frame_ids, H_rel, H_ok, H_ids, shot_starts, tracklets, thumbs):
        shots = []
        for i, s in enumerate(shot_starts):
            e = shot_starts[i + 1] if i + 1 < len(shot_starts) else N
            shots.append({"start": int(s), "end": int(e)})
        shot_of = np.zeros(N, np.int32)
        for i, s in enumerate(shots):
            shot_of[s["start"]:s["end"]] = i
        cx, cy = W / 2, H / 2

        # ---------------- camera from the scenery: turn + zoom per frame, and
        # where the background shows parallax, the camera's travel
        nA = len(frame_ids)
        zoom = np.ones(nA)
        a_index = {f: i for i, f in enumerate(frame_ids)}
        a_shot = shot_of[frame_ids]
        R_ref = np.tile(np.eye(3), (nA, 1, 1))
        C_ref = np.zeros((nA, 3))
        seg_a = -np.ones(nA, np.int64)
        par_a = np.zeros(nA)
        n_seg = 0
        fps_a = fps / stride
        # the first solve of every shot, all shots at once on the CPU's cores
        # (shots are independent; the travelling ones take long)
        def shot_samples(si):
            samples = []
            for t in tracklets:
                if t.shot != si:
                    continue
                step = max(1, len(t.frames) // 25)
                for j in range(0, len(t.frames), step):
                    if t.vis[j].mean() > 0.6:
                        samples.append((t.world[j].astype(np.float64), t.lm2d[j].astype(np.float64),
                                        t.vis[j].astype(np.float64), t.crop_c[j], zoom[a_index[t.frames[j]]]))
            return samples
        first = {}
        jobs = []
        for si in range(len(shots)):
            idx = np.where(a_shot == si)[0]
            if len(idx) == 0:
                continue
            vf0 = cam.estimate_vfov(shot_samples(si)[:120], W, H)
            jobs.append((si, [None] + [H_rel[i] for i in idx[1:]], [None] + [H_ids[i] for i in idx[1:]],
                         [False] + [bool(H_ok[i]) for i in idx[1:]], cam.focal_from_vfov(vf0, H), cx, cy, fps_a, H))
            first[si] = vf0
        results, results2, second = {}, {}, {}
        from .detect import worker_count
        workers = worker_count(0.4)
        if len(jobs) > 1 and workers > 1 and PARALLEL_SHOTS:
            import multiprocessing as mpc
            done = 0
            with mpc.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
                for si, res in pool.imap_unordered(_solve_shot_job, jobs):
                    results[si] = res
                    done += 1
                    self.progress(0.91 + 0.03 * done / len(jobs), "Solving the camera from the scenery")
                # the second solve (where the zoom found changes the lens estimate),
                # also all at once
                jobs2 = []
                for (si, pairs_, ids_, oks_, _f, *_r) in jobs:
                    if results[si][7]:  # the scenery measured the lens itself
                        continue
                    idx = np.where(a_shot == si)[0]
                    keep_z = zoom[idx].copy()
                    zoom[idx] = results[si][1]
                    vf1 = cam.estimate_vfov(shot_samples(si)[:120], W, H)
                    zoom[idx] = keep_z
                    if abs(vf1 - first[si]) >= 0.1 * first[si]:
                        second[si] = vf1
                        jobs2.append((si, pairs_, ids_, oks_, cam.focal_from_vfov(vf1, H), cx, cy, fps_a, H))
                done = 0
                for si, res in pool.imap_unordered(_solve_shot_job, jobs2):
                    results2[si] = res
                    done += 1
                    self.progress(0.94 + 0.01 * done / max(len(jobs2), 1), "Solving the camera from the scenery")
        for si, sh in enumerate(shots):
            idx = np.where(a_shot == si)[0]
            vf = None
            for _pass in range(2):
                samples = []
                for t in tracklets:
                    if t.shot != si:
                        continue
                    step = max(1, len(t.frames) // 25)
                    for j in range(0, len(t.frames), step):
                        if t.vis[j].mean() > 0.6:
                            samples.append((t.world[j].astype(np.float64), t.lm2d[j].astype(np.float64),
                                            t.vis[j].astype(np.float64), t.crop_c[j], zoom[a_index[t.frames[j]]]))
                vf_new = cam.estimate_vfov(samples[:120], W, H)
                if vf is not None and abs(vf_new - vf) < 0.1 * vf:
                    break
                vf = vf_new
                if len(idx) == 0:
                    break
                pairs = [None] + [H_rel[i] for i in idx[1:]]
                ids = [None] + [H_ids[i] for i in idx[1:]]
                oks = [False] + [bool(H_ok[i]) for i in idx[1:]]
                f0 = cam.focal_from_vfov(vf, H)
                if _pass == 0 and si in results and first.get(si) == vf:
                    Rs, zs, Cs, mv, sg, par, f_used, from_scene = results[si]
                elif _pass == 1 and si in results2 and second.get(si) == vf:
                    Rs, zs, Cs, mv, sg, par, f_used, from_scene = results2[si]
                else:
                    Rs, zs, Cs, mv, sg, par, f_used, from_scene = scene.solve_shot(pairs, ids, oks, f0, cx, cy, fps_a, H=H)
                R_ref[idx], zoom[idx], C_ref[idx], par_a[idx] = Rs, zs, Cs, par
                seg_a[idx] = np.where(sg >= 0, sg + n_seg, -1)
                if from_scene:  # the scenery measured the lens: better than the body's guess
                    vf = cam.vfov_from_focal(f_used, H)
                    break
            n_seg = int(seg_a.max()) + 1 if (seg_a >= 0).any() else 0
            sh["vfov"] = vf
            sh["samples"] = len(samples)
            sh["travel"] = float(np.mean(seg_a[idx] >= 0)) if len(idx) else 0.0
            self.progress(0.91 + 0.04 * (si + 1) / len(shots), "Solving the camera from the scenery")

        # ---------------- gravity alignment per shot
        for si, sh in enumerate(shots):
            f0 = cam.focal_from_vfov(sh["vfov"], H)
            acc = np.zeros(3)
            wsum = 0.0
            for t in tracklets:
                if t.shot != si:
                    continue
                for j in range(0, len(t.frames), 2):
                    ai = a_index[t.frames[j]]
                    Rray = cam.ray_rotation(f0 * zoom[ai], cx, cy, *t.crop_c[j])
                    up, w = cam.body_up_vector(t.world[j] @ Rray.T, t.vis[j])
                    if up is None or w <= 0:
                        continue
                    acc += w * (R_ref[ai] @ up)
                    wsum += w
            up_ref = normalize(acc) if wsum > 0.5 else np.array([0, -1.0, 0])
            # guard against implausible tilt (> 60 deg)
            if np.degrees(np.arccos(np.clip(-up_ref[1], -1, 1))) > 60:
                up_ref = np.array([0, -1.0, 0])
            sh["align"] = cam.align_from_up(up_ref)
            sh["gravity_weight"] = float(wsum)

        # ---------------- identities
        characters, char_of_track = cluster_identities(tracklets, fps, frame_size=(W, H))

        # ---------------- per-frame camera (all video frames)
        q_ref = mat_to_quat(R_ref)
        cam_q = np.zeros((N, 4))
        cam_zoom = np.ones(N)
        cam_C = np.zeros((N, 3))
        cam_seg = -np.ones(N, np.int64)
        for i in range(nA):
            f0 = frame_ids[i]
            f1 = frame_ids[i + 1] if i + 1 < nA else N
            same = i + 1 < nA and shot_of[f1 if f1 < N else N - 1] == shot_of[f0]
            for f in range(f0, f1):
                if same and f1 < N:
                    a = (f - f0) / (f1 - f0)
                    cam_q[f] = slerp(q_ref[i], q_ref[i + 1], a)
                    cam_zoom[f] = zoom[i] * (1 - a) + zoom[i + 1] * a
                    cam_C[f] = C_ref[i] * (1 - a) + C_ref[i + 1] * a
                    cam_seg[f] = seg_a[i] if seg_a[i] >= 0 else seg_a[i + 1]
                else:
                    cam_q[f] = q_ref[i]
                    cam_zoom[f] = zoom[i]
                    cam_C[f] = C_ref[i]
                    cam_seg[f] = seg_a[i]
        # light smoothing of the camera path inside each shot
        from .mathutil import gaussian_smooth, quat_continuity
        sig = 0.75 * stride
        for sh in shots:
            s, e = sh["start"], sh["end"]
            q = quat_continuity(cam_q[s:e].copy())
            cam_q[s:e] = normalize(gaussian_smooth(q, sig))
            cam_zoom[s:e] = gaussian_smooth(cam_zoom[s:e], sig)
            cam_C[s:e] = gaussian_smooth(cam_C[s:e], sig)
        focal = np.array([cam.focal_from_vfov(shots[shot_of[f]]["vfov"], H) for f in range(N)]) * cam_zoom

        # ---------------- heights relative to the scenery
        heights = estimate_heights(tracklets, char_of_track, characters, shots, shot_of, a_index,
                                   R_ref, zoom, H, W)
        for c in characters:
            c.update(heights.get(c["id"], {}))

        # ---------------- save
        tl_dump = {}
        for t in tracklets:
            p = f"t{t.id}_"
            tl_dump[p + "frames"] = np.array(t.frames, np.int32)
            tl_dump[p + "boxes"] = np.array(t.boxes, np.float32)
            tl_dump[p + "lm2d"] = np.array(t.lm2d, np.float32)
            tl_dump[p + "vis"] = np.array(t.vis, np.float16)
            tl_dump[p + "world"] = np.array(t.world, np.float32)
            tl_dump[p + "cropc"] = np.array(t.crop_c, np.float32)
        # how much the background moves (px per frame at full resolution): ~0 = locked-off camera
        bg = np.full(nA, np.nan)
        for i in range(nA):
            if H_ok[i] and H_rel[i] is not None:
                pr = H_rel[i]
                step = frame_ids[i] - frame_ids[i - 1] if i > 0 else 1
                bg[i] = float(np.median(np.linalg.norm(pr[:, 2:] - pr[:, :2], axis=1))) / max(step, 1)
        okb = np.isfinite(bg)
        bg_motion = np.interp(np.arange(N), frame_ids[okb], bg[okb]) if okb.any() else np.zeros(N)
        np.savez_compressed(os.path.join(self.workdir, "analysis.npz"), cam_q=cam_q.astype(np.float32),
                            bg_motion=bg_motion.astype(np.float32),
                            cam_C=cam_C.astype(np.float32), cam_seg=cam_seg.astype(np.int32),
                            cam_zoom=cam_zoom.astype(np.float32),
                            focal=focal.astype(np.float32), shot_of=shot_of,
                            shot_align=np.array([s["align"] for s in shots]), **tl_dump)
        thumb_dir = os.path.join(self.workdir, "thumbs")
        os.makedirs(thumb_dir, exist_ok=True)
        for c in characters:
            best = None
            for tid in c["tracklets"]:
                if tid in thumbs and (best is None or thumbs[tid][0] > best[0]):
                    best = thumbs[tid]
            if best is not None:
                img = best[1]
                if img.shape[0] > 360:
                    img = cv2.resize(img, (int(img.shape[1] * 360 / img.shape[0]), 360))
                cv2.imwrite(os.path.join(thumb_dir, f"char{c['id']}.jpg"), img)
                c["thumb_frame"] = int(best[2])
        # per-frame camera report (yaw relative to the main character's facing at shot starts)
        main = characters[0]["id"] if characters else None
        R_w = world_cameras(cam_q, shot_of, shots, tracklets, char_of_track, main, focal, W, H)
        cam_rows = []
        for f in range(N):
            y, p, r = cam.camera_euler(R_w[f])
            cam_rows.append([round(y, 2), round(p, 2), round(r, 2),
                             round(cam.vfov_from_focal(focal[f], H), 2), round(float(cam_zoom[f]), 4), int(shot_of[f])])
        np.save(os.path.join(self.workdir, "cam_world.npy"), mat_to_quat(R_w).astype(np.float32))
        summary = {
            "version": ANALYSIS_VERSION,
            "video": os.path.abspath(self.video_path),
            "frames": int(N), "fps": float(fps), "width": int(W), "height": int(H), "stride": int(stride),
            "shots": [{"start": s["start"], "end": s["end"], "vfov": round(s["vfov"], 2),
                       "samples": s["samples"]} for s in shots],
            "characters": characters,
            "tracklets": [{"id": t.id, "shot": t.shot, "start": int(t.frames[0]), "end": int(t.frames[-1]),
                           "character": char_of_track.get(t.id)} for t in tracklets],
            "camera_columns": ["yaw_deg", "pitch_deg", "roll_deg", "vfov_deg", "zoom", "shot"],
            "camera": cam_rows,
        }
        with open(os.path.join(self.workdir, "analysis.json"), "w") as f:
            json.dump(summary, f)
        return summary


PARALLEL_SCAN = True  # the person scan of the video's shots on several cores at once
SCAN_PIECES_PER_WORKER = 3  # ranges per worker (an even load where shots differ in length)
_scan_counter = None


def _scan_init(counter):
    global _scan_counter
    _scan_counter = counter


def _scan_range_job(job):
    (video, workdir, stride_o, scan_every_s, max_people, a, b, stride, scan_every, cut_frames, pdp) = job
    an = Analyzer(video, workdir, stride=stride_o, scan_every_s=scan_every_s, max_people=max_people)
    return an._scan_range(a, b, stride, scan_every, cut_frames, pdp, counter=_scan_counter)


def scan_ranges(N, stride, cuts, pieces):
    """Split the analysed frames at cuts into about `pieces` ranges of similar
    length. A range starts on the first analysed frame at or after its cut
    (where a scan of the whole video starts afresh too)."""
    if pieces <= 1 or not cuts:
        return [(0, N)]
    starts = sorted({((c + stride - 1) // stride) * stride for c in cuts if 0 < c < N})
    starts = [x for x in starts if 0 < x < N]
    if not starts:
        return [(0, N)]
    bounds = [0]
    for i in range(1, pieces):
        want = N * i / pieces
        best = min(starts, key=lambda x: abs(x - want))
        if best > bounds[-1]:
            bounds.append(best)
    bounds = sorted(set(bounds))
    return [(a, b) for a, b in zip(bounds, bounds[1:] + [N]) if b > a]


def merge_scans(parts, stride, N, cut_frames):
    """One scan from the ranges' parts (in order): track ids made unique."""
    sc = {"stride": stride, "frames": N, "frame_ids": [], "H_rel": [], "H_ok": [], "H_ids": [],
          "cuts": cut_frames, "shot_starts": [], "tracks": [], "thumbs": {}}
    nid = 0
    for p in sorted(parts, key=lambda p: p["start"]):
        for k in ("frame_ids", "H_rel", "H_ok", "H_ids"):
            sc[k].extend(p[k])
        for st in p["shot_starts"]:
            if st not in sc["shot_starts"]:
                sc["shot_starts"].append(st)
        remap = {}
        for t in sorted(p["tracks"], key=lambda t: t.id):
            remap[t.id] = nid
            t.id = nid
            nid += 1
        for old, v in p["thumbs"].items():
            if old in remap:
                sc["thumbs"][remap[old]] = v
        sc["tracks"].extend(p["tracks"])
    Track._next = max(Track._next, nid)
    return sc


def _solve_shot_job(job):
    si, pairs, ids, oks, f0, cx, cy, fps_a, H = job
    return si, scene.solve_shot(pairs, ids, oks, f0, cx, cy, fps_a, H=H)


# ----------------------------------------------------------------------------- identities
GLIMPSE_S = 1.0      # a character seen for less than this is a glimpse
MIN_LISTED_S = 0.5   # shorter characters are not listed


PARTIAL_THRESHOLD = 0.5  # how alike a close-up / partial view must look to join the main character
PARTIAL_THRESHOLD_ALONE = 0.7  # ... when that character is the only one ever seen whole (nobody else it could be)


def cluster_identities(tracklets, fps, threshold=0.42, glimpse_threshold=0.6, frame_size=None):
    """Agglomerative (average-link) clustering of tracklets with cannot-link
    constraints for tracklets visible at the same time."""
    n = len(tracklets)
    if n == 0:
        return [], {}
    E = np.array([normalize(np.mean(t.embs, axis=0)) for t in tracklets])
    Hh = np.array([np.mean(t.hists, axis=0) for t in tracklets])
    # Bhattacharyya distance per body part (3 parts of 68 bins)
    def hist_d(a, b):
        ds = []
        for p in range(3):
            x, y = a[p * 68:(p + 1) * 68], b[p * 68:(p + 1) * 68]
            bc = np.sum(np.sqrt(np.maximum(x, 0) * np.maximum(y, 0))) / max(np.sqrt(x.sum() * y.sum()), 1e-9)
            ds.append(np.sqrt(max(0.0, 1 - bc)))
        return float(np.average(ds, weights=[1.4, 1.0, 0.8]))
    Dm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            de = 1 - float(np.dot(E[i], E[j]))
            dh = hist_d(Hh[i], Hh[j])
            d = 0.45 * min(1.0, de * 2.0) + 0.55 * dh
            Dm[i, j] = Dm[j, i] = d
    # tracklets alive at the same time: either duplicates of one body (boxes overlap
    # -> must merge) or two different people (-> may never merge)
    fidx = [{f: j for j, f in enumerate(t.frames)} for t in tracklets]
    cannot = np.zeros((n, n), bool)
    must = []
    for i in range(n):
        for j in range(i + 1, n):
            common = set(fidx[i]) & set(fidx[j])
            if len(common) <= 1:
                continue
            ious = [max(box_iou(tracklets[i].boxes[fidx[i][f]], tracklets[j].boxes[fidx[j][f]]),
                        1.0 if contains(tracklets[i].boxes[fidx[i][f]], tracklets[j].boxes[fidx[j][f]], 0.5) else 0.0)
                    for f in common]
            if np.mean(np.array(ious) > 0.3) > 0.5:
                must.append((i, j))
            else:
                cannot[i, j] = cannot[j, i] = True
    # fragments of one track inside a shot (short gap, end box near start box)
    for i in range(n):
        for j in range(n):
            ti, tj = tracklets[i], tracklets[j]
            if i == j or ti.shot != tj.shot or cannot[i, j]:
                continue
            gap = tj.frames[0] - ti.frames[-1]
            if not (-6 <= gap <= 0.6 * fps):
                continue
            k = next((x for x in range(len(tj.frames)) if tj.frames[x] >= ti.frames[-1]), 0)
            bi, bj = ti.boxes[-1], tj.boxes[k]
            hgt = max(bi[3] - bi[1], bj[3] - bj[1], 1.0)
            dc = np.linalg.norm(((bi[:2] + bi[2:]) - (bj[:2] + bj[2:])) / 2)
            if dc < 0.6 * hgt and Dm[i, j] < threshold * 1.4:
                must.append((i, j))
    clusters = [[i] for i in range(n)]
    for i, j in must:
        ci = next(c for c in clusters if i in c)
        cj = next(c for c in clusters if j in c)
        if ci is not cj and not any(cannot[a, b] for a in ci for b in cj):
            ci.extend(cj)
            clusters.remove(cj)

    def conflict(a, b):
        return any(cannot[i, j] for i in a for j in b)

    while len(clusters) > 1:
        best = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                pair = [Dm[i, j] for i in clusters[a] for j in clusters[b]]
                # blend of average and best-match linkage: robust to a few blurry
                # or badly lit appearances inside an otherwise clear identity
                d = 0.5 * (float(np.mean(pair)) + float(np.min(pair)))
                if best is None or d < best[0]:
                    if not conflict(clusters[a], clusters[b]):
                        best = (d, a, b)
        if best is None or best[0] > threshold:
            break
        _, a, b = best
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    # brief glimpses (a fast cut, a moment at the edge of the picture) are much
    # less certain in appearance than a long track: one that never shares the
    # picture with a longer-seen character and looks reasonably like it is that
    # character. What is left that is shorter than half a second is not listed
    # (nobody wants to capture a 0.2 s glimpse, and it looks like a duplicate).
    def seconds(c):
        fr = sorted({f for i in c for f in tracklets[i].frames})
        return len(fr) * (float(np.median(np.diff(fr))) if len(fr) > 1 else 1.0) / fps
    if any(seconds(c) >= GLIMPSE_S for c in clusters):
        for c in sorted([c for c in clusters if seconds(c) < GLIMPSE_S], key=seconds):
            best = None
            for o in clusters:
                if o is c or seconds(o) < GLIMPSE_S or conflict(c, o):
                    continue
                pair = [Dm[i, j] for i in c for j in o]
                d = 0.5 * (float(np.mean(pair)) + float(np.min(pair)))
                if best is None or d < best[0]:
                    best = (d, o)
            if best is not None and best[0] < glimpse_threshold:
                best[1].extend(c)
                clusters = [x for x in clusters if x is not c]
        clusters = [c for c in clusters if seconds(c) >= MIN_LISTED_S]
    # close-ups and other partial views (only the face, only the legs, the
    # body running out of the picture) look unlike the same person in full
    # view; a cluster made of them that never shares the picture with a
    # character seen far longer, and whose nearest views look alike, is that
    # character
    if frame_size is not None and len(clusters) > 1:
        Wf, Hf = frame_size
        def partial(i):
            b = np.asarray(tracklets[i].boxes, np.float64)
            m = 0.02 * max(Wf, Hf)
            edge = (b[:, 0] < m) | (b[:, 1] < m) | (b[:, 2] > Wf - m) | (b[:, 3] > Hf - m)
            return float(edge.mean())
        # the characters seen whole for a while (the box clear of the picture's
        # edges for at least GLIMPSE_S): any other cluster could only be them
        def seconds_whole(c):
            m = 0.02 * max(Wf, Hf)
            n = 0
            for i in c:
                b = np.asarray(tracklets[i].boxes, np.float64)
                fr = np.asarray(tracklets[i].frames)
                step = float(np.median(np.diff(fr))) if len(fr) > 1 else 1.0
                n += step * int((~((b[:, 0] < m) | (b[:, 1] < m) | (b[:, 2] > Wf - m) | (b[:, 3] > Hf - m))).sum())
            return n / fps
        changed = True
        while changed:
            changed = False
            clusters.sort(key=seconds, reverse=True)
            whole = [c for c in clusters if seconds_whole(c) >= GLIMPSE_S]
            thr = PARTIAL_THRESHOLD_ALONE if len(whole) == 1 else PARTIAL_THRESHOLD
            for c in clusters[1:]:
                pw = np.average([partial(i) for i in c], weights=[len(tracklets[i].frames) for i in c])
                if pw < 0.5:
                    continue
                cands = [o for o in clusters if o is not c and seconds(o) >= 3 * seconds(c) and not conflict(c, o)]
                if not cands:
                    continue
                d, o = min(((min(Dm[i, j] for i in c for j in o_), o_) for o_ in cands), key=lambda x: x[0])
                if d < (thr if any(o is w for w in whole) else PARTIAL_THRESHOLD):
                    o.extend(c)
                    clusters = [x for x in clusters if x is not c]
                    changed = True
                    break
    # characters ordered by screen time
    clusters.sort(key=lambda c: -sum(len(tracklets[i].frames) for i in c))
    chars, char_of = [], {}
    for ci, c in enumerate(clusters):
        frames = sorted({f for i in c for f in tracklets[i].frames})
        for i in c:
            char_of[tracklets[i].id] = ci
        chars.append({
            "id": ci,
            "name": f"Character {ci + 1}",
            "tracklets": [tracklets[i].id for i in c],
            "appearances": len(c),
            "shots": sorted({int(tracklets[i].shot) for i in c}),
            "first_frame": int(frames[0]), "last_frame": int(frames[-1]),
            "screen_time_s": round(len(frames) * (float(np.median(np.diff(frames))) if len(frames) > 1 else 1.0) / fps, 2),
        })
    return chars, char_of


# ----------------------------------------------------------------------------- heights
def estimate_heights(tracklets, char_of, characters, shots, shot_of, a_index, R_ref, zoom, H, W):
    """Height of each character relative to the camera height (horizon method),
    combined over shots with a log-space least squares so characters that are
    never on screen together are still comparable through shared shots."""
    cx, cy = W / 2, H / 2
    obs = {}  # (char, shot) -> list of log ratios
    for t in tracklets:
        c = char_of.get(t.id)
        sh = shots[t.shot]
        f0 = cam.focal_from_vfov(sh["vfov"], H)
        for j in range(len(t.frames)):
            ai = a_index[t.frames[j]]
            f = f0 * zoom[ai]
            Rw = sh["align"] @ R_ref[ai]
            Rray = cam.ray_rotation(f, cx, cy, *t.crop_c[j])
            Wc = t.world[j].astype(np.float64) @ Rray.T
            up, w = cam.body_up_vector(Wc, t.vis[j])
            if up is None or w < 0.15:
                continue
            lm, vis = t.lm2d[j], t.vis[j]
            # how tall the body currently stands vs its full standing height (handles bent knees etc.)
            upv = -g_cv(Rw)
            stand = standing_height(Wc)
            cur = float(np.max(Wc @ upv) - np.min(Wc[[L_HEEL, R_HEEL, L_FOOT, R_FOOT]] @ upv)) + 0.12 * stand
            if stand <= 0 or cur <= 0.3 * stand:
                continue
            corr = stand / max(cur, 1e-6)
            if min(vis[L_HEEL], vis[R_HEEL], vis[NOSE]) < 0.5:
                continue
            foot_y = max(lm[L_HEEL, 1], lm[R_HEEL, 1], lm[L_FOOT, 1], lm[R_FOOT, 1])
            foot_x = (lm[L_HEEL, 0] + lm[R_HEEL, 0]) / 2
            # crown of head ~ nose + 0.55 * (nose - shoulder centre) upward
            shc = (lm[L_SH] + lm[R_SH]) / 2
            top_y = lm[NOSE, 1] - 0.55 * abs(shc[1] - lm[NOSE, 1])
            hy = cam.horizon_y(Rw, f, cx, cy, foot_x)
            if hy is None or foot_y - hy < 5:
                continue
            ratio = (foot_y - top_y) / (foot_y - hy) * corr
            if 0.05 < ratio < 20:
                obs.setdefault((c, t.shot), []).append(np.log(ratio))
    if not obs:
        return {}
    keys = list(obs.keys())
    cids = sorted({k[0] for k in keys})
    sids = sorted({k[1] for k in keys})
    ci = {c: i for i, c in enumerate(cids)}
    si = {s: i for i, s in enumerate(sids)}
    nC, nS = len(cids), len(sids)
    A, b, wts = [], [], []
    for (c, s), v in obs.items():
        row = np.zeros(nC + nS)
        row[ci[c]] = 1
        row[nC + si[s]] = -1
        A.append(row)
        b.append(float(np.median(v)))
        wts.append(np.sqrt(len(v)))
    # gauge: mean camera log-height = 0
    row = np.zeros(nC + nS)
    row[nC:] = 1.0 / nS
    A.append(row)
    b.append(0.0)
    wts.append(10.0)
    A, b, wts = np.array(A), np.array(b), np.array(wts)
    x, *_ = np.linalg.lstsq(A * wts[:, None], b * wts, rcond=None)
    hc = np.exp(x[:nC])
    tallest = hc.max()
    out = {}
    for c in cids:
        h = float(hc[ci[c]])
        out[c] = {
            "height_vs_camera": round(h, 3),
            "relative_height": round(h / tallest, 3),
            "height_samples": int(sum(len(v) for (cc, s), v in obs.items() if cc == c)),
            "height_by_shot": {int(s): round(float(np.exp(np.median(v))), 3) for (cc, s), v in obs.items() if cc == c},
        }
    return out


def g_cv(Rw):
    """World 'down' direction expressed in camera (cv) coordinates."""
    return Rw.T @ np.array([0, -1.0, 0])


def standing_height(Wc):
    """Approximate standing height (heel to crown) from MediaPipe world landmarks."""
    from .detect import L_KNEE, R_KNEE
    def d(a, b):
        return float(np.linalg.norm(Wc[a] - Wc[b]))
    shin = (d(L_ANK, L_KNEE) + d(R_ANK, R_KNEE)) / 2
    thigh = (d(L_KNEE, L_HIP) + d(R_KNEE, R_HIP)) / 2
    torso = float(np.linalg.norm((Wc[L_SH] + Wc[R_SH]) / 2 - (Wc[L_HIP] + Wc[R_HIP]) / 2))
    neck_nose = float(np.linalg.norm(Wc[NOSE] - (Wc[L_SH] + Wc[R_SH]) / 2))
    heel = (d(L_HEEL, L_ANK) + d(R_HEEL, R_ANK)) / 2
    return shin + thigh + torso + neck_nose * 1.55 + heel * 0.6


# ----------------------------------------------------------------------------- world cameras
def world_cameras(cam_q, shot_of, shots, tracklets, char_of, char_id, focal, W, H, facing=None):
    """World rotation of the camera for every frame.

    Inside a shot the rotation comes from background tracking. Each shot's yaw
    is chosen so the reference character keeps facing the same way across the
    cut (180-degree-rule continuity); the first shot defines yaw 0."""
    N = len(cam_q)
    R_rel = quat_to_mat(cam_q)
    cx, cy = W / 2, H / 2
    # facing angle per frame of the reference character, in the shot-aligned frame
    yaw_at = {}
    if facing is not None:
        yaw_at = facing
    else:
        for t in tracklets:
            if char_of.get(t.id) != char_id:
                continue
            for j, f in enumerate(t.frames):
                if t.vis[j].mean() < 0.6:
                    continue
                Rw = shots[shot_of[f]]["align"] @ R_rel[f]
                Rray = cam.ray_rotation(focal[f], cx, cy, *t.crop_c[j])
                yaw_at[f] = cam.hips_yaw(t.world[j].astype(np.float64) @ Rray.T, Rw)
    offsets = np.zeros(len(shots))
    last_yaw = None
    for si, sh in enumerate(shots):
        fs = [f for f in range(sh["start"], sh["end"]) if f in yaw_at]
        if not fs:
            continue
        head = fs[: max(1, min(len(fs), 10))]
        tail = fs[-max(1, min(len(fs), 10)):]
        first = np.angle(np.mean(np.exp(1j * np.array([yaw_at[f] for f in head]))))
        if last_yaw is not None:
            offsets[si] = last_yaw - first
        last = np.angle(np.mean(np.exp(1j * np.array([yaw_at[f] for f in tail]))))
        last_yaw = last + offsets[si]
    out = np.zeros((N, 3, 3))
    for f in range(N):
        si = shot_of[f]
        out[f] = cam.yaw_matrix(offsets[si]) @ shots[si]["align"] @ R_rel[f]
    return out

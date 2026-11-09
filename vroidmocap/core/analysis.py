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
from .detect import (L_ANK, L_FOOT, L_HEEL, L_HIP, L_SH, NOSE, R_ANK, R_FOOT, R_HEEL, R_HIP,
                     R_SH, Embedder, PoseDetector, box_iou, crop_square, landmark_box)
from .mathutil import mat_to_quat, normalize, quat_to_mat, slerp
from .video import VideoReader

ANALYSIS_VERSION = 3


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
    landmarks on non-people are low-confidence and jump around wildly."""
    V = np.array(t.vis, np.float64)[:, CORE]
    L = np.array(t.lm2d, np.float64)[:, CORE]
    bx = np.array(t.boxes, np.float64)
    bh = np.maximum(bx[:, 3] - bx[:, 1], 1.0)
    if V.mean() < 0.7:
        return False
    if len(L) > 2:
        jit = np.median(np.linalg.norm(np.diff(L, axis=0), axis=2).mean(1) / bh[1:])
        if jit > 0.15:
            return False
    return True


def part_histograms(rgb, lm2d, vis):
    """Colour histograms of hair/head, torso and legs (hue-saturation)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, w = rgb.shape[:2]
    sh = np.linalg.norm(lm2d[L_SH] - lm2d[R_SH])
    unit = max(sh, 6.0)
    regions = []
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
        return keep

    def _refine(self, rgb, box):
        """Run single-person pose on a square crop around `box`."""
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        size = max(box[2] - box[0], box[3] - box[1]) * 1.35
        crop, x0, y0, sc = crop_square(rgb, cx, cy, size, out=384)
        res = self.track_pose.detect(crop, x0, y0, sc, sc)
        if not res or res[0].vis.mean() < 0.35:
            return None, None
        return res[0], (cx, cy)

    # -- run -------------------------------------------------------------------
    def run(self):
        t_start = time.time()
        rd = VideoReader(self.video_path)
        N, fps, W, H = rd.frame_count, rd.fps, rd.width, rd.height
        stride = self.stride_override or max(1, int(round(fps / 30.0)))
        scan_every = max(1, int(round(self.scan_every_s * fps / stride)))
        self.scan_pose = PoseDetector(heavy=False, num_poses=3, min_conf=0.35)
        self.track_pose = PoseDetector(heavy=False, num_poses=1, min_conf=0.3)
        embedder = Embedder()

        scan_path = os.path.join(self.workdir, "scan.pkl")
        if os.path.exists(scan_path):
            import pickle
            with open(scan_path, "rb") as fh:
                sc = pickle.load(fh)
            if sc.get("stride") == stride and sc.get("frames") == N:
                self.scan_pose.close()
                self.track_pose.close()
                embedder.close()
                rd.close()
                return self._finish(t_start, N, fps, W, H, stride, sc)
        cam_scale = 640.0 / W
        cw, ch = 640, int(round(H * cam_scale))
        prev_gray = prev_hist = prev_mask = None
        tracks, finished = [], []
        shot_starts = [0]
        frame_ids, H_rel, H_ok = [], [], []
        last_scan = -10 ** 9
        lost_limit = max(2, int(round(0.6 * fps / stride)))
        thumbs = {}
        k = 0
        for fi, bgr in rd.iter_frames(stride):
            if self.cancel():
                raise Cancelled()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            small = cv2.resize(bgr, (cw, ch), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(cv2.resize(small, (96, 54)), cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256])
            hist = hist.flatten() / max(hist.sum(), 1)
            cut = False
            mad = 0.0
            if prev_hist is not None:
                corr = float(np.minimum(hist, prev_hist).sum())  # histogram intersection
                mad = float(np.mean(cv2.absdiff(gray, prev_gray)))
                if corr < 0.45 or (corr < 0.62 and mad > 40):
                    cut = True

            # ---- camera motion (background only): keep matched point pairs for later
            pair, ok = None, False
            if prev_gray is not None and not cut:
                pts = cv2.goodFeaturesToTrack(prev_gray, 500, 0.01, 8, mask=prev_mask)
                if pts is not None and len(pts) >= 12:
                    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, winSize=(21, 21), maxLevel=3)
                    good = st.ravel() == 1
                    if good.sum() >= 12:
                        Hs, inl = cv2.findHomography(pts[good], nxt[good], cv2.RANSAC, 2.0)
                        ratio = float(inl.sum()) / len(pts) if inl is not None else 0.0
                        resid = 99.0
                        if Hs is not None:
                            warped = cv2.warpPerspective(prev_gray, Hs, (cw, ch))
                            valid = cv2.warpPerspective(np.full_like(prev_gray, 1), Hs, (cw, ch)) > 0
                            if valid.sum() > 0.3 * valid.size:
                                resid = float(np.mean(cv2.absdiff(warped, gray)[valid]))
                        if ratio < 0.3 or resid > 14.0:
                            cut = True
                        else:
                            a, b = pts[good].reshape(-1, 2), nxt[good].reshape(-1, 2)
                            if len(a) > 300:
                                sel = np.random.default_rng(k).choice(len(a), 300, replace=False)
                                a, b = a[sel], b[sel]
                            pair = np.concatenate([a, b], 1).astype(np.float32) / cam_scale
                            ok = True
                elif mad > 30:
                    cut = True
            if cut and (not shot_starts or shot_starts[-1] != fi):
                shot_starts.append(fi)
                finished.extend(tracks)
                tracks = []
                last_scan = -10 ** 9
            # ---- people
            for t in tracks:
                pred = t.predict()
                pr, cc = self._refine(rgb, pred)
                if pr is not None and pr.vis.mean() > 0.45:
                    box = landmark_box(pr.lm2d, pr.vis)
                    if box_iou(box, pred) > 0.15 or contains(pred, box, 0.6):
                        t.add(fi, damp_box(t.boxes[-1], box), pr, cc)
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
            if k - last_scan >= scan_every or not tracks:
                if k - last_scan >= scan_every or k % max(1, scan_every // 3) == 0:
                    last_scan = k
                    for box, pr in self._scan(rgb):
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
                        t.hists.append(part_histograms(rgb, t.lm2d[-1], t.vis[-1]))
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

            # mask for next frame's feature detection
            prev_mask = np.full(gray.shape, 255, np.uint8)
            for t in tracks:
                b = (t.boxes[-1] * cam_scale).astype(int)
                cv2.rectangle(prev_mask, (b[0] - 6, b[1] - 6), (b[2] + 6, b[3] + 6), 0, -1)
            frame_ids.append(fi)
            H_rel.append(pair)
            H_ok.append(ok)
            prev_gray, prev_hist = gray, hist
            k += 1
            if k % 5 == 0:
                el = time.time() - t_start
                frac = fi / max(N - 1, 1)
                eta = el / max(frac, 1e-6) * (1 - frac)
                self.progress(frac * 0.9, f"Scanning frame {fi + 1}/{N} - {len(tracks)} people in view - ~{eta / 60:.1f} min left")
        finished.extend(tracks)
        rd.close()
        self.scan_pose.close()
        self.track_pose.close()
        embedder.close()

        sc = {"stride": stride, "frames": N, "frame_ids": frame_ids, "H_rel": H_rel, "H_ok": H_ok,
              "shot_starts": shot_starts, "tracks": finished, "thumbs": thumbs}
        import pickle
        with open(scan_path + ".tmp", "wb") as fh:
            pickle.dump(sc, fh, protocol=4)
        os.replace(scan_path + ".tmp", scan_path)
        return self._finish(t_start, N, fps, W, H, stride, sc)

    def _finish(self, t_start, N, fps, W, H, stride, sc):
        self.progress(0.91, "Solving cameras and identities")
        # fast whip-pans / flashes can fire the cut detector on many consecutive
        # frames: shots shorter than 0.3 s are merged into the previous shot
        starts = list(sc["shot_starts"])
        merged = [starts[0]]
        for i in range(1, len(starts)):
            end = starts[i + 1] if i + 1 < len(starts) else N
            prev_len = starts[i] - merged[-1]
            if end - starts[i] < 0.3 * fps or prev_len < 0.3 * fps:
                continue
            merged.append(starts[i])
        frame_shot = np.searchsorted(np.array(merged), np.arange(N), side="right") - 1
        for t in sc["tracks"]:
            t.shot = int(frame_shot[t.frames[0]])
        min_len = max(3, int(round(0.4 * fps / stride)))
        tracklets = [t for t in sc["tracks"] if len(t.frames) >= min_len and t.embs and is_human_track(t)]
        result = self._solve(N, fps, W, H, stride, np.array(sc["frame_ids"]), sc["H_rel"], sc["H_ok"],
                             merged, tracklets, sc["thumbs"])
        result["elapsed_s"] = time.time() - t_start
        self.progress(1.0, "Analysis complete")
        return result

    # -- post-processing ---------------------------------------------------------
    def _solve(self, N, fps, W, H, stride, frame_ids, H_rel, H_ok, shot_starts, tracklets, thumbs):
        shots = []
        for i, s in enumerate(shot_starts):
            e = shot_starts[i + 1] if i + 1 < len(shot_starts) else N
            shots.append({"start": int(s), "end": int(e)})
        shot_of = np.zeros(N, np.int32)
        for i, s in enumerate(shots):
            shot_of[s["start"]:s["end"]] = i
        cx, cy = W / 2, H / 2

        # ---------------- focal length (per shot) from people landmarks
        nA = len(frame_ids)
        zoom = np.ones(nA)
        a_index = {f: i for i, f in enumerate(frame_ids)}
        for si, sh in enumerate(shots):
            samples = []
            for t in tracklets:
                if t.shot != si:
                    continue
                step = max(1, len(t.frames) // 25)
                for j in range(0, len(t.frames), step):
                    if t.vis[j].mean() > 0.6:
                        samples.append((t.world[j].astype(np.float64), t.lm2d[j].astype(np.float64),
                                        t.vis[j].astype(np.float64), t.crop_c[j], 1.0))
            sh["vfov"] = cam.estimate_vfov(samples[:120], W, H)
            sh["samples"] = len(samples)

        # ---------------- camera rotation chains per shot (analysed frames)
        R_ref = np.zeros((nA, 3, 3))
        for i in range(nA):
            si = shot_of[frame_ids[i]]
            if i == 0 or shot_of[frame_ids[i - 1]] != si or not H_ok[i]:
                R_ref[i] = np.eye(3) if (i == 0 or shot_of[frame_ids[i - 1]] != si) else R_ref[i - 1]
                continue
            f = cam.focal_from_vfov(shots[si]["vfov"], H)
            Rr = relative_rotation(H_rel[i], f, cx, cy)  # R_{t <- t-1}
            R_ref[i] = R_ref[i - 1] @ Rr.T

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
        characters, char_of_track = cluster_identities(tracklets, fps)

        # ---------------- per-frame camera (all video frames)
        q_ref = mat_to_quat(R_ref)
        cam_q = np.zeros((N, 4))
        cam_zoom = np.ones(N)
        for i in range(nA):
            f0 = frame_ids[i]
            f1 = frame_ids[i + 1] if i + 1 < nA else N
            same = i + 1 < nA and shot_of[f1 if f1 < N else N - 1] == shot_of[f0]
            for f in range(f0, f1):
                if same and f1 < N:
                    a = (f - f0) / (f1 - f0)
                    cam_q[f] = slerp(q_ref[i], q_ref[i + 1], a)
                    cam_zoom[f] = zoom[i] * (1 - a) + zoom[i + 1] * a
                else:
                    cam_q[f] = q_ref[i]
                    cam_zoom[f] = zoom[i]
        # light smoothing of the camera path inside each shot
        from .mathutil import gaussian_smooth, quat_continuity
        sig = 0.75 * stride
        for sh in shots:
            s, e = sh["start"], sh["end"]
            q = quat_continuity(cam_q[s:e].copy())
            cam_q[s:e] = normalize(gaussian_smooth(q, sig))
            cam_zoom[s:e] = gaussian_smooth(cam_zoom[s:e], sig)
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
        np.savez_compressed(os.path.join(self.workdir, "analysis.npz"), cam_q=cam_q.astype(np.float32),
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


# ----------------------------------------------------------------------------- camera rotation
def relative_rotation(pair, f, cx, cy):
    """Camera rotation between two frames from matched background points.

    pair: (n,4) [x1, y1, x2, y2] in full-resolution pixels. Uses a pure-rotation
    fit when it explains the motion (pan/tilt), otherwise the essential matrix
    (camera also translating: dolly/crane/handheld)."""
    p1, p2 = pair[:, :2].astype(np.float64), pair[:, 2:].astype(np.float64)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    b1 = normalize(np.c_[(p1 - [cx, cy]) / f, np.ones(len(p1))])
    b2 = normalize(np.c_[(p2 - [cx, cy]) / f, np.ones(len(p2))])

    def kabsch(m):
        M = b2[m].T @ b1[m]
        U, _, Vt = np.linalg.svd(M)
        dd = np.sign(np.linalg.det(U @ Vt))
        return U @ np.diag([1, 1, dd]) @ Vt

    def err(R):
        q = b1 @ R.T
        q = q[:, :2] / np.maximum(q[:, 2:3], 1e-6) * f + [cx, cy]
        return np.linalg.norm(q - p2, axis=1)

    m = np.ones(len(p1), bool)
    for _ in range(3):
        Rk = kabsch(m)
        e = err(Rk)
        thr = max(1.5, np.percentile(e, 60) * 1.5)
        m = e < thr
        if m.sum() < 8:
            break
    if m.sum() >= 8 and np.median(err(Rk)) < 1.2:
        return Rk
    try:
        E, inl = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.0)
        if E is not None and E.shape == (3, 3) and inl is not None and inl.sum() >= 12:
            _, R, t, _ = cv2.recoverPose(E, p1, p2, K, mask=inl.copy())
            # sanity: a single analysed step should not rotate more than ~20 degrees
            ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
            if ang < 20:
                return R
    except cv2.error:
        pass
    return Rk


# ----------------------------------------------------------------------------- identities
def cluster_identities(tracklets, fps, threshold=0.42):
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

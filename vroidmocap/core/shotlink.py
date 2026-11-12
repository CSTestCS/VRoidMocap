"""Camera directions across cuts from the background.

Inside a shot the camera's turning comes from tracking the scenery. Across a
cut the new camera can be anywhere; where the two shots show some of the same
background (the set, the lights, the floor pattern), matching its features
gives the rotation from one camera to the other. People are masked out.
"""
from __future__ import annotations

import cv2
import numpy as np

LINK_AHEAD = 3          # a shot is matched with the next this many shots
LINK_WIDTH = 960        # matching resolution (picture width)
LINK_MIN_INLIERS = 40   # matches a link needs ...
LINK_MIN_RATIO = 0.35   # ... and the share of matches it explains
LINK_MAX_TILT_DEG = 8.0  # the two cameras' "up" (from the scenery) must agree this well


def _features(sift, bgr, boxes, scale):
    g = cv2.cvtColor(cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    mask = np.full(g.shape, 255, np.uint8)
    for b in boxes:
        x0, y0, x1, y1 = (np.asarray(b, float) * scale)
        pw, ph = 0.15 * (x1 - x0), 0.1 * (y1 - y0)
        cv2.rectangle(mask, (int(x0 - pw), int(y0 - ph)), (int(x1 + pw), int(y1 + ph)), 0, -1)
    kp, des = sift.detectAndCompute(g, mask)
    if des is None or len(kp) < 20:
        return None
    return np.float32([k.pt for k in kp]) / scale, des


def relative_rotation(fa, fb, focal_a, focal_b, cx, cy):
    """Rotation R taking camera A's directions to camera B's (x_b = R x_a, the
    OpenCV convention) from matched background features, or None.
    Returns (R, n_inliers)."""
    if fa is None or fb is None:
        return None
    bf = cv2.BFMatcher(cv2.NORM_L2)
    m = bf.knnMatch(fa[1], fb[1], k=2)
    good = [p[0] for p in m if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < LINK_MIN_INLIERS:
        return None
    pa = fa[0][[g.queryIdx for g in good]]
    pb = fb[0][[g.trainIdx for g in good]]
    # normalised image coordinates (each camera its own focal length)
    na = np.c_[(pa[:, 0] - cx) / focal_a, (pa[:, 1] - cy) / focal_a]
    nb = np.c_[(pb[:, 0] - cx) / focal_b, (pb[:, 1] - cy) / focal_b]
    thr = 1.5 / max(focal_a, focal_b)
    best = None
    # the camera moved between the shots: essential matrix
    E, inl = cv2.findEssentialMat(na, nb, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=thr)
    if E is not None and E.shape == (3, 3) and inl is not None:
        n_e, R, t, inl2 = cv2.recoverPose(E, na, nb, np.eye(3), mask=inl.copy())
        if n_e >= LINK_MIN_INLIERS:
            best = (R, int(n_e))
    # the camera only turned (or the background is far): rotation homography
    Hm, inh = cv2.findHomography(na, nb, cv2.RANSAC, thr)
    if Hm is not None and inh is not None:
        n_h = int(inh.sum())
        if n_h >= LINK_MIN_INLIERS and (best is None or n_h >= 0.8 * best[1]):
            U, _, Vt = np.linalg.svd(Hm / np.cbrt(max(np.linalg.det(Hm), 1e-12)))
            Rh = U @ Vt
            if np.linalg.det(Rh) < 0:
                Rh = -Rh
            best = (Rh, n_h)
    if best is None or best[1] < LINK_MIN_RATIO * len(good):
        return None
    return best


def link_shots(video_path, shots, shot_of, people_boxes, focal, W, H, progress=None, cancel=None):
    """For neighbouring shots, the camera rotation across the cut from the
    background. people_boxes: dict frame -> list of boxes to mask. Returns a
    list of (shot_a, shot_b, frame_a, frame_b, R (3,3) x_b = R x_a, inliers)."""
    from .video import VideoReader
    if len(shots) < 2:
        return []
    sift = cv2.SIFT_create(nfeatures=2500)
    scale = LINK_WIDTH / float(W)
    rd = VideoReader(video_path)
    feats = {}

    def feat(f):
        if f not in feats:
            bgr = rd.read(f)
            feats[f] = None if bgr is None else _features(sift, bgr, people_boxes.get(f, []), scale)
        return feats[f]
    cx, cy = W / 2.0, H / 2.0
    links = []
    n = len(shots)
    for a in range(n):
        if cancel is not None and cancel():
            break
        if progress is not None:
            progress(a / n)
        sa = shots[a]
        if sa["end"] - sa["start"] < 2:
            continue
        fa = max(sa["start"], sa["end"] - 3)
        for b in range(a + 1, min(n, a + 1 + LINK_AHEAD)):
            sb = shots[b]
            if sb["end"] - sb["start"] < 2:
                continue
            fb = min(sb["end"] - 1, sb["start"] + 2)
            r = relative_rotation(feat(fa), feat(fb), float(focal[fa]), float(focal[fb]), cx, cy)
            if r is not None:
                links.append((a, b, fa, fb, r[0], r[1]))
    rd.close()
    return links


def yaw_from_link(R_a, R_link, A_b):
    """Camera b's yaw offset from a link: R_a camera a -> world (placed),
    R_link x_b = R x_a, A_b camera b -> world without its yaw offset
    (scenery 'up' x rotation inside the shot). Returns (yaw radians, tilt
    mismatch degrees)."""
    R_b = R_a @ R_link.T  # camera b -> world
    Y = R_b @ A_b.T       # should be a pure turn about world up
    yaw = float(np.arctan2(Y[0, 2], Y[0, 0]))
    c, s = np.cos(yaw), np.sin(yaw)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    resid = Ry.T @ Y
    tilt = float(np.degrees(np.arccos(np.clip((np.trace(resid) - 1) / 2, -1, 1))))
    return yaw, tilt

"""The camera, from the scenery alone.

The background is tracked with long feature tracks (people are masked out).
For every shot this module works out, frame by frame:

* how the camera turns (pan / tilt / roll) and zooms: a rotation + focal-change
  model of how the background moves between frames;
* whether the camera itself travels (dolly, truck, crane, handheld). Turning
  and zooming move near and far things alike; travelling makes near things
  slide over far ones (parallax). Where there is parallax, the camera path and
  the depth of the tracked scenery are solved together (bundle adjustment with
  the camera rotations refined as well).

None of this looks at the characters, so a character moving never moves the
camera. The camera path from parallax has no absolute size (a model 10x larger
seen from 10x further looks the same); the animation step sets that one number
per moving stretch.

Conventions: a camera rotation R maps camera (cv) vectors into the shot's
reference frame (the cv frame of the shot's first analysed frame), C is the
camera centre in that frame.
"""
from __future__ import annotations

import cv2
import numpy as np

from .mathutil import normalize


STILL_PX = 0.12       # median background motion below this (full-res px per step): camera still
HUBER_PX = 1.5
STILL_FRAC_PX = 0.3   # ...or this share of the tracked scenery moves less than this
STILL_FRAC = 0.35


def skew(v):
    v = np.asarray(v, float)
    o = np.zeros(v.shape[:-1] + (3, 3))
    o[..., 0, 1], o[..., 0, 2] = -v[..., 2], v[..., 1]
    o[..., 1, 0], o[..., 1, 2] = v[..., 2], -v[..., 0]
    o[..., 2, 0], o[..., 2, 1] = -v[..., 1], v[..., 0]
    return o


def expmap(w):
    w = np.asarray(w, float)
    th = np.linalg.norm(w, axis=-1, keepdims=True)[..., None]
    K = skew(w)
    th_s = np.maximum(th, 1e-12)
    a = np.where(th < 1e-8, 1.0, np.sin(th_s) / th_s)
    b = np.where(th < 1e-8, 0.5, (1 - np.cos(th_s)) / th_s ** 2)
    return np.eye(3) + a * K + b * (K @ K)


def polar(M):
    U, _, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1, 1, d]) @ Vt


def spread_out(p, cx, cy, min_area=0.12):
    """Still points that say the camera is still must be spread over the
    picture. Captions, a logo or a watermark sit still in one small corner
    while the camera moves; they say nothing about the camera."""
    if len(p) < 8:
        return False
    lo, hi = np.percentile(p, 5, axis=0), np.percentile(p, 95, axis=0)
    return float(np.prod(np.maximum(hi - lo, 0))) >= min_area * (2 * cx) * (2 * cy)


def _rz_project(p1, R, f1, f2, cx, cy):
    v = np.c_[(p1 - [cx, cy]) / f1, np.ones(len(p1))] @ R.T
    return v[:, :2] / np.maximum(v[:, 2:3], 1e-9) * f2 + [cx, cy]


def rotation_zoom(p1, p2, f1, cx, cy, allow_zoom=True, zlim=(0.8, 1.25)):
    """Camera rotation R (v2 ~ R v1) and zoom z = f2/f1 between two frames from
    matched background points. Returns (R, z, median residual px, inlier mask)
    or None."""
    n = len(p1)
    if n < 8:
        return None
    disp = np.linalg.norm(p2 - p1, axis=1)
    # a still camera: the background does not move. Things in the scene may
    # (balloons, hair, other people), but when a large share of the scenery is
    # perfectly still, the camera is.
    if (np.median(disp) < STILL_PX or np.mean(disp < STILL_FRAC_PX) > STILL_FRAC) and \
            spread_out(p1[disp < STILL_FRAC_PX], cx, cy):
        return np.eye(3), 1.0, float(np.median(disp)), disp < 3.0
    Hm, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 2.0)
    if Hm is None or inl is None or inl.sum() < 8:
        return None
    m = inl.ravel().astype(bool)
    T = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1.0]])
    Hc = np.linalg.inv(T) @ Hm @ T
    A = Hc @ np.diag([f1, f1, 1.0])
    f2 = 0.5 * (np.linalg.norm(A[0]) + np.linalg.norm(A[1])) / max(np.linalg.norm(A[2]), 1e-12)
    z = f2 / f1 if allow_zoom else 1.0
    if not (zlim[0] < z < zlim[1]):
        z = 1.0
    R = polar(np.diag([1 / (f1 * z), 1 / (f1 * z), 1.0]) @ A)
    # refine (rotation vector + log zoom) on the inliers, robustly
    a, b = p1[m], p2[m]
    x = np.zeros(4)
    x[3] = np.log(z)

    def res(x):
        return (_rz_project(a, R @ expmap(x[:3]), f1, f1 * np.exp(x[3]), cx, cy) - b).ravel()
    for _ in range(6):
        r0 = res(x)
        J = np.zeros((len(r0), 4))
        for i in range(4 if allow_zoom else 3):
            d = np.zeros(4)
            d[i] = 1e-6
            J[:, i] = (res(x + d) - r0) / 1e-6
        e = np.abs(r0.reshape(-1, 2)).max(1)
        w = np.repeat(np.where(e < HUBER_PX, 1.0, HUBER_PX / np.maximum(e, 1e-9)), 2)
        Hh = J.T @ (J * w[:, None]) + 1e-3 * np.eye(4)
        step = -np.linalg.solve(Hh, J.T @ (w * r0))
        x = x + step
        if np.abs(step).max() < 1e-7:
            break
    R = R @ expmap(x[:3])
    z = float(np.exp(x[3]))
    if not (zlim[0] < z < zlim[1]):
        z = 1.0
    err = np.linalg.norm(_rz_project(p1, R, f1, f1 * z, cx, cy) - p2, axis=1)
    return R, z, float(np.median(err[m])), err < 3.0


# ----------------------------------------------------------------------------- tracks
def frame_observations(pairs, ids, K):
    """Per analysed frame k (0..K-1 within a shot): (ids, xy) of tracked
    background points. pairs[k] = (n,4) [x_prev, y_prev, x_cur, y_cur] for the
    step k-1 -> k (pairs[0] unused)."""
    obs = [dict() for _ in range(K)]
    for k in range(1, K):
        if pairs[k] is None or ids[k] is None:
            continue
        for i, (x0, y0, x1, y1) in zip(ids[k], pairs[k]):
            obs[k - 1].setdefault(int(i), (x0, y0))
            obs[k][int(i)] = (x1, y1)
    out = []
    for o in obs:
        if o:
            ii = np.fromiter(o.keys(), np.int64)
            out.append((ii, np.array(list(o.values()), float)))
        else:
            out.append((np.zeros(0, np.int64), np.zeros((0, 2))))
    return out


def _common(oa, ob):
    ia, pa = oa
    ib, pb = ob
    _, xa, xb = np.intersect1d(ia, ib, assume_unique=True, return_indices=True)
    return pa[xa], pb[xb]


def parallax_profile(obs, Rref, focal, cx, cy, win=8):
    """Median residual (px) of the background motion over `win` frames that
    turning + zooming cannot explain. Large -> the camera travels."""
    K = len(obs)
    par = np.zeros(K)
    for k in range(win, K):
        pa, pb = _common(obs[k - win], obs[k])
        if len(pa) < 12:
            continue
        st = np.linalg.norm(pb - pa, axis=1) < 2 * STILL_FRAC_PX
        if np.mean(st) > STILL_FRAC and spread_out(pa[st], cx, cy):
            continue  # much of the scenery has not moved at all: a still camera
        R = Rref[k].T @ Rref[k - win]  # v_k ~ R v_{k-win}
        e = np.linalg.norm(_rz_project(pa, R, focal[k - win], focal[k], cx, cy) - pb, axis=1)
        par[k] = float(np.median(e))
    return par


def calibrate_focal(obs, use, cx, cy, H, zoom=None, gap=12):
    """Focal length from the scenery while the camera turns: only the right
    lens makes a turn of the camera look like a rotation. obs: per-frame
    tracks; use: frames that may be used (not travelling). Returns the focal
    length in px, or None when the shot does not turn enough to tell."""
    from .camera import VFOV_CANDIDATES, focal_from_vfov
    K = len(obs)
    pairs = []
    for k in range(0, K - gap, max(1, gap // 2)):
        if not (use[k] and use[k + gap]):
            continue
        pa, pb = _common(obs[k], obs[k + gap])
        if len(pa) < 20 or np.median(np.linalg.norm(pb - pa, axis=1)) < 8:
            continue
        pairs.append((pa, pb, 1.0 if zoom is None else zoom[k]))
    if len(pairs) < 3:
        return None
    pairs = pairs[::max(1, len(pairs) // 30)]
    fs = np.array([focal_from_vfov(v, H) for v in VFOV_CANDIDATES])
    cost = np.zeros(len(fs))
    turn = []
    for i, f in enumerate(fs):
        c = []
        for pa, pb, zk in pairs:
            fit = rotation_zoom(pa, pb, f * zk, cx, cy, zlim=(0.5, 2.0))
            if fit is None:
                c.append(5.0)
                continue
            c.append(min(fit[2], 5.0))
            if i == len(fs) // 2:
                turn.append(np.degrees(np.arccos(np.clip((np.trace(fit[0]) - 1) / 2, -1, 1))))
        cost[i] = np.median(c)
    j = int(np.argmin(cost))
    # informative only if the camera turned and the wrong lenses clearly fit worse
    if not turn or np.sum(np.array(turn) > 2.0) < 3 or cost.min() > 0.7 or np.median(cost) < 2.0 * cost.min() + 0.3:
        return None
    if 0 < j < len(fs) - 1:
        a, b, c = cost[j - 1], cost[j], cost[j + 1]
        den = a - 2 * b + c
        off = float(np.clip(0.5 * (a - c) / den, -1, 1)) if den > 1e-12 else 0.0
        v = VFOV_CANDIDATES[j] + off * (VFOV_CANDIDATES[1] - VFOV_CANDIDATES[0])
        return float(focal_from_vfov(v, H))
    return float(fs[j])


# ----------------------------------------------------------------------------- bundle adjustment
def bundle_translation(obs, R0, focal, cx, cy, iters=25, sigma_acc=0.003, max_points=None, seed=0,
                       C0=None, n_fix=1):
    """Camera centres C (K,3) and refined rotations R (K,3,3) for one stretch of
    frames, from the background's parallax. Camera 0 is fixed at the origin with
    rotation R0[0]; the scale is set so the tracked scenery's median depth is 1.
    With C0 and n_fix >= 2 the first n_fix cameras are held at C0 / R0 instead
    (continuing an earlier window: its scale carries over).
    Points are parameterised by inverse depth along their first sighting.
    Returns (R, C, cost per obs in px) or None."""
    K = len(obs)
    max_points = max_points or BA_POINTS
    # tracks with enough sightings
    rows = []
    for k, (ii, pp) in enumerate(obs):
        for i, p in zip(ii, pp):
            rows.append((int(i), k, p[0], p[1]))
    if not rows:
        return None
    rows = np.array(rows)
    tid = rows[:, 0].astype(np.int64)
    uniq, inv, cnt = np.unique(tid, return_inverse=True, return_counts=True)
    keep_t = cnt >= 4
    rng = np.random.default_rng(seed)
    cand = np.where(keep_t)[0]
    if len(cand) < 15:
        return None
    if len(cand) > max_points:
        cand = rng.choice(cand, max_points, replace=False)
    sel = np.zeros(len(uniq), bool)
    sel[cand] = True
    rows = rows[sel[inv]]
    inv = inv[sel[inv]]
    pid_map = -np.ones(len(uniq), np.int64)
    pid_map[cand] = np.arange(len(cand))
    pid = pid_map[inv]
    P = len(cand)
    fr = rows[:, 1].astype(np.int64)
    xy = rows[:, 2:4]
    order = np.lexsort((fr, pid))
    pid, fr, xy = pid[order], fr[order], xy[order]
    starts = np.r_[0, np.where(np.diff(pid) != 0)[0] + 1]
    anc_f = fr[starts]
    anc_xy = xy[starts]
    is_anchor = np.zeros(len(pid), bool)
    is_anchor[starts] = True
    o_p, o_k, o_xy = pid[~is_anchor], fr[~is_anchor], xy[~is_anchor]
    o_a = anc_f[o_p]
    m_b = np.c_[(anc_xy - [cx, cy]) / focal[anc_f][:, None], np.ones(P)]  # anchor bearings

    R = R0.copy()
    C = np.zeros((K, 3)) if C0 is None else np.asarray(C0, float).copy()
    rho = np.ones(P)
    if np.ptp(C, axis=0).max() > 1e-6:  # cameras already apart: triangulate the starting depths
        bo = normalize(np.c_[(o_xy - [cx, cy]) / focal[o_k][:, None], np.ones(len(o_k))])
        u = np.einsum("nji,nj->ni", R0[o_k], np.einsum("nij,nj->ni", R0[o_a], m_b[o_p]))
        c = np.einsum("nji,nj->ni", R0[o_k], C[o_a] - C[o_k])
        bu, bc = np.cross(bo, u), np.cross(bo, c)
        num = np.bincount(o_p, -np.sum(bc * bu, 1), P)
        den = np.bincount(o_p, np.sum(bu * bu, 1), P)
        lam = num / np.maximum(den, 1e-12)
        good = (den > 1e-8) & (lam > 1e-3)
        if good.sum() > 10:
            rho = np.where(good, 1.0 / np.maximum(lam, 1e-3), 1.0 / np.median(lam[good]))
    nf = n_fix
    gauge = not (nf >= 2 and np.linalg.norm(C[nf - 1] - C[0]) > 1e-9)
    nc = 6 * (K - nf)
    if nc <= 0:
        return None

    def cidx(k):  # parameter indices of camera k (-1: fixed)
        base = 6 * (k - nf)
        out = base[:, None] + np.arange(6)[None]
        out[k < nf] = -1
        return out

    ik, ia = cidx(o_k), cidx(o_a)
    # smoothness prior on the camera path (second differences of C)
    Lp = np.zeros((max(K - 2, 0), K))
    for k in range(K - 2):
        Lp[k, k:k + 3] = [1, -2, 1]
    Wacc = 1.0 / sigma_acc ** 2

    def residuals(R, C, rho):
        X = C[o_a] + np.einsum("nij,nj->ni", R[o_a], m_b[o_p]) / rho[o_p][:, None]
        v = np.einsum("nji,nj->ni", R[o_k], X - C[o_k])
        z = np.maximum(v[:, 2], 1e-6)
        f = focal[o_k]
        pr = np.c_[f * v[:, 0] / z + cx, f * v[:, 1] / z + cy]
        return pr - o_xy, v

    def cost(R, C, rho):
        r, v = residuals(R, C, rho)
        e = np.linalg.norm(r, axis=1)
        hub = np.where(e < HUBER_PX, 0.5 * e ** 2, HUBER_PX * (e - 0.5 * HUBER_PX))
        hub = np.where(v[:, 2] > 1e-3, hub, 50.0)
        acc = Lp @ C
        return float(hub.sum() + 0.5 * Wacc * (acc ** 2).sum())

    lam = 1e-3
    cur = cost(R, C, rho)
    for it in range(iters):
        r, v = residuals(R, C, rho)
        e = np.linalg.norm(r, axis=1)
        w = np.where(e < HUBER_PX, 1.0, HUBER_PX / np.maximum(e, 1e-9))
        w = np.where(v[:, 2] > 1e-3, w, 0.0)
        z = np.maximum(v[:, 2], 1e-6)
        f = focal[o_k]
        Pm = np.zeros((len(z), 2, 3))
        Pm[:, 0, 0] = f / z
        Pm[:, 1, 1] = f / z
        Pm[:, 0, 2] = -f * v[:, 0] / z ** 2
        Pm[:, 1, 2] = -f * v[:, 1] / z ** 2
        RkT = np.transpose(R[o_k], (0, 2, 1))
        Ra = R[o_a]
        mb = m_b[o_p]
        rr = rho[o_p]
        Jk = np.concatenate([np.einsum("nij,njk->nik", Pm, skew(v)),
                             -np.einsum("nij,njk->nik", Pm, RkT)], 2)          # d/d(rot_k, C_k)
        RkRa = np.einsum("nij,njk->nik", RkT, Ra)
        Ja = np.concatenate([np.einsum("nij,njk->nik", Pm, -RkRa @ skew(mb) / rr[:, None, None]),
                             np.einsum("nij,njk->nik", Pm, RkT)], 2)           # d/d(rot_a, C_a)
        Jp = np.einsum("nij,nj->ni", Pm, -np.einsum("nij,nj->ni", RkRa, mb) / rr[:, None] ** 2)  # d/d rho
        Jc = np.concatenate([Jk, Ja], 2)                                       # (n,2,12)
        idx = np.concatenate([ik, ia], 1)                                      # (n,12)
        Hcc = np.zeros((nc, nc))
        gc = np.zeros(nc)
        Hcp = np.zeros((nc, P))
        Hpp = np.zeros(P)
        gp = np.zeros(P)
        JtJ = np.einsum("n,nia,nib->nab", w, Jc, Jc)
        Jtr = np.einsum("n,nia,ni->na", w, Jc, r)
        Jtp = np.einsum("n,nia,ni->na", w, Jc, Jp)
        ok = idx >= 0
        ii_, jj_ = np.broadcast_to(idx[:, :, None], JtJ.shape), np.broadcast_to(idx[:, None, :], JtJ.shape)
        m2 = ok[:, :, None] & ok[:, None, :]
        Hcc += np.bincount((ii_[m2] * nc + jj_[m2]), JtJ[m2], nc * nc).reshape(nc, nc)
        gc += np.bincount(idx[ok], Jtr[ok], nc)
        pp = np.broadcast_to(o_p[:, None], idx.shape)
        Hcp += np.bincount(idx[ok] * P + pp[ok], Jtp[ok], nc * P).reshape(nc, P)
        Hpp += np.bincount(o_p, w * np.sum(Jp * Jp, 1), P)
        gp += np.bincount(o_p, w * np.sum(Jp * r, 1), P)
        # smoothness prior on C
        if K > 2:
            Hacc = Wacc * (Lp.T @ Lp)
            gacc = Wacc * (Lp.T @ (Lp @ C))
            pos_idx = np.array([6 * (k - nf) + 3 + np.arange(3) for k in range(nf, K)])
            for a_ in range(3):
                sub = pos_idx[:, a_]
                Hcc[np.ix_(sub, sub)] += Hacc[nf:, nf:]
                gc[sub] += gacc[nf:, a_]
        for _try in range(6):
            Hpp_d = Hpp * (1 + lam) + 1e-6
            S = Hcc + lam * np.diag(np.diag(Hcc)) + 1e-6 * np.eye(nc)
            S -= (Hcp / Hpp_d[None]) @ Hcp.T
            rhs = gc - Hcp @ (gp / Hpp_d)
            okc, sol = cv2.solve(0.5 * (S + S.T), rhs.reshape(-1, 1), flags=cv2.DECOMP_CHOLESKY)
            if not okc or not np.isfinite(sol).all():
                lam *= 10
                continue
            dc = -sol.ravel()
            dp = -(gp + Hcp.T @ dc) / Hpp_d
            dcm = dc.reshape(K - nf, 6)
            Rn = R.copy()
            Rn[nf:] = R[nf:] @ expmap(dcm[:, :3])
            Cn = C.copy()
            Cn[nf:] = C[nf:] + dcm[:, 3:]
            rn = np.clip(rho + dp, 1e-3, 1e3)
            if gauge:  # scale gauge: median scenery depth 1
                s = float(np.median(rn))
                rn, Cn = rn / s, C[0] + (Cn - C[0]) * s
            new = cost(Rn, Cn, rn)
            if new < cur:
                R, C, rho, cur = Rn, Cn, rn, new
                lam = max(lam / 3, 1e-6)
                break
            lam *= 10
        else:
            break
    r, _ = residuals(R, C, rho)
    # the same sightings explained by turning alone (no travel, scenery far away)
    v0 = np.einsum("nji,nj->ni", R0[o_k], np.einsum("nij,nj->ni", R0[o_a], m_b[o_p]))
    f = focal[o_k]
    p0 = np.c_[f * v0[:, 0] / np.maximum(v0[:, 2], 1e-6) + cx, f * v0[:, 1] / np.maximum(v0[:, 2], 1e-6) + cy]
    norot = float(np.median(np.linalg.norm(p0 - o_xy, axis=1)))
    return R, C, float(np.median(np.linalg.norm(r, axis=1))), norot


TRAVEL_BUDGET_S = 600.0  # time allowed for the camera's travel in one shot (a safety net)
BA_POINTS = 600        # scenery points per bundle adjustment
LENS_SCALES = (0.7, 0.8, 0.9, 1.0, 1.12, 1.25, 1.42)  # lens guesses tried against the parallax
LENS_ATTEMPTS = 3     # times the lens may be re-measured in one shot
TRAVEL_WIN = 90        # keyframes refined together while the path grows
MAX_KEYFRAMES = 150     # keyframes of one travel stretch (more: further apart)
_deadline = [None]


def _out_of_time():
    import time
    return _deadline[0] is not None and time.time() > _deadline[0]


def travel_stretch(obs, R, focal, cx, cy, a0, b, step=3, win=TRAVEL_WIN, search_focal=False):
    """Bundle adjustment for frames [a0, b) on keyframes `step` apart, in
    overlapping windows of `win` keyframes (each continues the previous one's
    path and scale). Returns refined rotations, camera centres (relative to
    frame a0) for every frame of the stretch, and the zoom mode ("chain": the
    measured zoom, "fixed": no zoom while travelling), or None."""
    from .mathutil import mat_to_quat, quat_to_mat, slerp
    if _out_of_time():
        return None
    step = max(step, int(np.ceil((b - a0) / MAX_KEYFRAMES)))
    kf = list(range(a0, b, step))
    if kf[-1] != b - 1:
        kf.append(b - 1)
    kf = np.array(kf)
    nk = len(kf)
    Rk = R[kf].copy()
    Ck = np.zeros((nk, 3))
    zmode = None
    fscale = 1.0
    focal_in = focal
    if search_focal:  # the lens is not known from turning: let the parallax tell
        sel = np.arange(0, nk, max(1, int(np.ceil(nk / 30))))
        n_s = len(sel)
        if n_s >= 6:
            sub = [obs[k] for k in kf[sel]]
            scs = np.array(LENS_SCALES)
            costs = np.full(len(scs), np.inf)
            for i, sc in enumerate(scs):
                out = incremental_ba(sub, Rk[sel], np.full(n_s, focal[a0] * sc), cx, cy, upto=n_s)
                if out is not None:
                    costs[i] = out[2]
            if np.isfinite(costs).any():
                j = int(np.argmin(costs))
                i1 = int(np.argmin(np.abs(scs - 1.0)))
                ls = np.log(scs)
                fs_ = ls[j]
                if 0 < j < len(scs) - 1 and np.isfinite(costs[j - 1:j + 2]).all():
                    a_, b_, c_ = costs[j - 1], costs[j], costs[j + 1]
                    den = a_ - 2 * b_ + c_
                    if den > 1e-9:
                        fs_ = ls[j] + float(np.clip(0.5 * (a_ - c_) / den, -1, 1)) * (ls[j + 1] - ls[j])
                if costs[i1] - costs[j] > 0.03:
                    fscale = float(np.exp(fs_))
        focal = focal * fscale
    focal_chain = focal_in
    zr = focal[kf] / focal[a0]
    cands = ("chain", "fixed") if np.ptp(zr) > 0.01 else ("chain",)
    best = None
    for zm in cands:
        fz = focal[kf] if zm == "chain" else np.full(nk, focal[a0])
        out = incremental_ba([obs[k] for k in kf], Rk, fz, cx, cy, win=win,
                             upto=None if len(cands) == 1 else min(nk, 30))
        if out is not None and (best is None or out[2] < best[0][2] - 0.05):
            best = (out, zm)
    if best is None:
        return None
    zmode = best[1]
    if len(cands) > 1:
        fz = focal[kf] if zmode == "chain" else np.full(nk, focal[a0])
        out = incremental_ba([obs[k] for k in kf], Rk, fz, cx, cy, win=win)
        if out is None:
            return None
    else:
        out = best[0]
    Rk, Ck, med = out
    norot = turn_zoom_fit([obs[k] for k in kf], R[kf], focal_chain[kf], cx, cy)
    # travel is accepted only if it explains the scenery clearly better than turning
    import os
    if os.environ.get("SCENE_DEBUG"):
        print("travel stretch", a0, b, "med", med, "norot", norot, "zmode", zmode, "fscale", fscale)
    if not (med < 0.6 * norot and norot - med > 0.3):
        return None
    # every frame: rotation detail from the chain, corrected like its keyframes
    n = b - a0
    Rout = np.zeros((n, 3, 3))
    Cout = np.zeros((n, 3))
    dq = mat_to_quat(np.einsum("nji,njk->nik", R[kf], Rk))
    for i in range(nk - 1):
        k0, k1 = kf[i], kf[i + 1]
        for k in range(k0, k1 + 1):
            t = (k - k0) / max(k1 - k0, 1)
            Rout[k - a0] = R[k] @ quat_to_mat(slerp(dq[i], dq[i + 1], t))
            Cout[k - a0] = Ck[i] * (1 - t) + Ck[i + 1] * t
    if nk == 1:
        Rout[:] = R[a0:b]
    return Rout, Cout, zmode, fscale


def incremental_ba(sub, Rchain, focal, cx, cy, grow=None, win=60, iters=8, upto=None):
    """Camera path over keyframes, grown a few keyframes at a time (each new
    camera starts from the path so far, continued at its current speed, with the
    turn measured by the rotation chain), optimising the latest `win`
    keyframes at each step; a final pass refines everything when it fits in
    one window. Returns (R, C, median residual px) or None."""
    nk = len(sub) if upto is None else upto
    if grow is None:
        grow = max(6, nk // 12)
    Rk = Rchain[:nk].copy()
    Ck = np.zeros((nk, 3))
    solved = 1
    med = None
    while solved < nk:
        if _out_of_time():
            return None
        new_end = min(nk, solved + (grow if solved > 1 else max(grow, 4)))
        last = solved - 1
        v = Ck[last] - Ck[last - 1] if last >= 1 else np.zeros(3)
        for j in range(solved, new_end):
            Rk[j] = Rk[last] @ (Rchain[last].T @ Rchain[j])
            Ck[j] = Ck[last] + v * (j - last)
        lo = max(0, new_end - win)
        n_fix = 1 if lo == 0 else 2
        out = bundle_translation(sub[lo:new_end], Rk[lo:new_end], focal[lo:new_end], cx, cy,
                                 iters=iters, C0=Ck[lo:new_end], n_fix=n_fix)
        if out is None:
            if solved == 1:
                return None
            Rk[solved:new_end] = Rk[last] @ np.einsum("ij,njk->nik", Rchain[last].T, Rchain[solved:new_end])
        else:
            Rk[lo:new_end], Ck[lo:new_end], med, _ = out
        solved = new_end
    if nk <= win and nk >= 3:
        out = bundle_translation(sub[:nk], Rk, focal[:nk], cx, cy, iters=20, C0=Ck, n_fix=1)
        if out is not None:
            Rk, Ck, med, _ = out
    if med is None:
        return None
    return Rk, Ck, med


def turn_zoom_fit(sub, Rchain, focal, cx, cy, iters=6):
    """Best explanation of the keyframes WITHOUT travel: every tracked point is
    a direction (infinitely far), each keyframe only turns and zooms. Alternates
    between the directions and per-keyframe rotation + zoom. Returns the median
    residual in px."""
    K = len(sub)
    rows = [(i, k, p[0], p[1]) for k, (ii, pp) in enumerate(sub) for i, p in zip(ii, pp)]
    if not rows:
        return np.inf
    rows = np.array(rows)
    _, pid = np.unique(rows[:, 0].astype(np.int64), return_inverse=True)
    fr = rows[:, 1].astype(np.int64)
    xy = rows[:, 2:4]
    cnt = np.bincount(pid)
    keep = cnt[pid] >= 3
    pid, fr, xy = pid[keep], fr[keep], xy[keep]
    if len(pid) < 30:
        return np.inf
    _, pid = np.unique(pid, return_inverse=True)
    P = pid.max() + 1
    R = Rchain.copy()
    f = focal.astype(float).copy()
    for _ in range(iters):
        b = normalize(np.c_[(xy - [cx, cy]) / f[fr][:, None], np.ones(len(fr))])
        bw = np.einsum("nij,nj->ni", R[fr], b)
        d = normalize(np.stack([np.bincount(pid, bw[:, a], P) for a in range(3)], 1))
        for k in range(K):
            m = fr == k
            if m.sum() < 8:
                continue
            dk, xk = d[pid[m]], xy[m]
            x = np.zeros(4)

            def res(x):
                v = np.einsum("ji,nj->ni", R[k] @ expmap(x[:3]), dk)
                ff = f[k] * np.exp(x[3])
                return (np.c_[ff * v[:, 0] / np.maximum(v[:, 2], 1e-6) + cx,
                              ff * v[:, 1] / np.maximum(v[:, 2], 1e-6) + cy] - xk).ravel()
            for _g in range(3):
                r0 = res(x)
                J = np.zeros((len(r0), 4))
                for i in range(4):
                    dd = np.zeros(4)
                    dd[i] = 1e-6
                    J[:, i] = (res(x + dd) - r0) / 1e-6
                e = np.abs(r0.reshape(-1, 2)).max(1)
                w = np.repeat(np.where(e < HUBER_PX, 1.0, HUBER_PX / np.maximum(e, 1e-9)), 2)
                x = x - np.linalg.solve(J.T @ (J * w[:, None]) + 1e-6 * np.eye(4), J.T @ (w * r0))
            R[k] = R[k] @ expmap(x[:3])
            f[k] *= np.exp(x[3])
    b = normalize(np.c_[(xy - [cx, cy]) / f[fr][:, None], np.ones(len(fr))])
    bw = np.einsum("nij,nj->ni", R[fr], b)
    d = normalize(np.stack([np.bincount(pid, bw[:, a], P) for a in range(3)], 1))
    v = np.einsum("nji,nj->ni", R[fr], d[pid])
    pr = np.c_[f[fr] * v[:, 0] / np.maximum(v[:, 2], 1e-6) + cx, f[fr] * v[:, 1] / np.maximum(v[:, 2], 1e-6) + cy]
    return float(np.median(np.linalg.norm(pr - xy, axis=1)))


# ----------------------------------------------------------------------------- per shot
def _chain(pairs, ok, f0, cx, cy):
    K = len(pairs)
    R = np.tile(np.eye(3), (K, 1, 1))
    zoom = np.ones(K)
    for k in range(1, K):
        R[k] = R[k - 1]
        zoom[k] = zoom[k - 1]
        if not ok[k] or pairs[k] is None or len(pairs[k]) < 8:
            continue
        p = pairs[k].astype(float)
        fit = rotation_zoom(p[:, :2], p[:, 2:], f0 * zoom[k - 1], cx, cy)
        if fit is None:
            continue
        Rs, z, _, _ = fit
        R[k] = R[k - 1] @ Rs.T
        zoom[k] = zoom[k - 1] * z
    if np.ptp(zoom) < 0.01:  # tiny zoom drifts are not zooms
        zoom[:] = 1.0
    return R, zoom


def _moving(par, par_thr, win, fps_a):
    moving = par > par_thr
    mv = moving.copy()
    for k in np.where(moving)[0]:  # spread over the window that produced it
        mv[max(0, k - win):k + 1] = True
    return mv


def solve_shot(pairs, ids, ok, f0, cx, cy, fps_a, H=1080, par_thr=1.2, win=8):
    """Camera for one shot (analysed frames 0..K-1).

    pairs/ids: per step k (k-1 -> k) matched background points (None if not
    usable); ok: per step, the tracker trusted that step; f0: focal guess (px)
    for the shot's first frame (replaced when the scenery tells better).
    Returns R (K,3,3) camera->reference, zoom (K,), C (K,3), moving (K,) bool,
    seg (K,) int id of the travelling stretch (-1: not travelling), the
    parallax profile and the focal length used (and whether the scenery set it)."""
    K = len(pairs)
    obs = frame_observations(pairs, ids, K)
    R, zoom = _chain(pairs, ok, f0, cx, cy)
    par = parallax_profile(obs, R, f0 * zoom, cx, cy, win)
    moving = _moving(par, par_thr, win, fps_a)
    near = moving.copy()
    for k in np.where(moving)[0]:
        near[max(0, k - 15):k + 16] = True
    fcal = calibrate_focal(obs, ~near, cx, cy, H, zoom)
    calibrated = fcal is not None
    if calibrated and abs(fcal / f0 - 1) > 0.02:
        f0 = fcal
        R, zoom = _chain(pairs, ok, f0, cx, cy)
        par = parallax_profile(obs, R, f0 * zoom, cx, cy, win)
        moving = _moving(par, par_thr, win, fps_a)
    R_chain, zoom_chain = R.copy(), zoom.copy()
    search = not calibrated
    import time
    _deadline[0] = time.time() + TRAVEL_BUDGET_S
    for attempt in range(LENS_ATTEMPTS + 1):
        R, zoom = R_chain.copy(), zoom_chain.copy()
        focal = f0 * zoom
        out = _travel(obs, R, zoom, focal, moving.copy(), cx, cy, fps_a, win, search_focal=search and attempt < LENS_ATTEMPTS)
        R, zoom, C, moving_t, seg, fscale = out
        if np.mean(seg >= 0) < 0.25 or (seg >= 0).sum() < 2 * fps_a:
            fscale = 1.0  # too little travel in this shot to measure the lens by
        if search and attempt < LENS_ATTEMPTS and abs(fscale - 1) > 0.03:
            f0 *= fscale
            R_chain, zoom_chain = _chain(pairs, ok, f0, cx, cy)
            calibrated = True
            continue
        break
    _deadline[0] = None
    calibrated = calibrated or (np.mean(seg >= 0) >= 0.25 and (seg >= 0).sum() >= 2 * fps_a)
    return R, zoom, C, moving_t, seg, par, f0, calibrated


def _travel(obs, R, zoom, focal, moving, cx, cy, fps_a, win, search_focal=False):
    K = len(obs)
    minlen = max(win + 2, int(0.5 * fps_a))
    C = np.zeros((K, 3))
    seg = -np.ones(K, np.int64)
    k, sid = 0, 0
    fscale, best_len = 1.0, 0
    while k < K:
        if not moving[k]:
            k += 1
            continue
        a = k
        while k < K and moving[k]:
            k += 1
        b = k  # [a, b)
        if b - a < minlen:
            moving[a:b] = False
            continue
        a0 = max(0, a - 1)
        best = travel_stretch(obs, R, focal, cx, cy, a0, b, search_focal=search_focal)
        if best is None:
            moving[a:b] = False
            continue
        Rb, Cb, zmode, fs = best
        if b - a0 > best_len:
            best_len, fscale = b - a0, fs
        # re-base: keep the stretch's refined rotations and the rotation changes after it
        tail = R[b - 1].T @ R[b:] if b < K else None
        R[a0:b] = Rb
        if tail is not None:
            R[b:] = R[b - 1] @ tail
        if zmode == "fixed":
            zf = zoom[a0]
            ratio = zoom[b - 1] / zf if b < K else 1.0
            zoom[a0:b] = zf
            if b < K:
                zoom[b:] = zoom[b:] / ratio
        C[a0:b] = C[a0] + Cb
        if b < K:
            C[b:] = C[b - 1]
        seg[a0:b] = sid
        sid += 1
    return R, zoom, C, moving, seg, fscale

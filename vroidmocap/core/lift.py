"""3D reconstruction of the body in camera space.

What is known about a joint from one camera:
  * where it is in the picture (precise): the joint lies on a camera ray,
  * how long the bones are (constant over the whole video),
  * a learned guess of its depth (rough, but tells front from behind),
  * that it moves smoothly, and that knees only bend one way.

A bone of known length seen in the picture leaves exactly two possible
depths for its far end (towards / away from the camera). So:

  1. bone lengths are measured from the frames in which a bone lies roughly
     parallel to the picture,
  2. for each limb the best sequence of towards/away choices is found over
     time (Viterbi): agree with the learned depth, keep knees bending the right
     way, do not flip between frames,
  3. everything is refined together (Gauss-Newton, all frames): reprojection,
     bone lengths, depth guess, smoothness along the line of sight.
"""
from __future__ import annotations

import numpy as np

from .detect import (L_ANK, L_EL, L_FOOT, L_HEEL, L_HIP, L_KNEE, L_SH, L_WR, R_ANK, R_EL,
                     R_FOOT, R_HEEL, R_HIP, R_KNEE, R_SH, R_WR)
from .mathutil import gaussian_smooth, normalize

# joints solved here (MediaPipe numbering)
JOINTS = [L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK,
          L_HEEL, R_HEEL, L_FOOT, R_FOOT]
JI = {j: i for i, j in enumerate(JOINTS)}
# (a, b, sigma in metres, mirror partner index)
_E = [
    (L_SH, L_EL, 0.004), (R_SH, R_EL, 0.004), (L_EL, L_WR, 0.004), (R_EL, R_WR, 0.004),
    (L_HIP, L_KNEE, 0.004), (R_HIP, R_KNEE, 0.004), (L_KNEE, L_ANK, 0.004), (R_KNEE, R_ANK, 0.004),
    (L_ANK, L_HEEL, 0.006), (R_ANK, R_HEEL, 0.006), (L_ANK, L_FOOT, 0.006), (R_ANK, R_FOOT, 0.006),
    (L_HEEL, L_FOOT, 0.006), (R_HEEL, R_FOOT, 0.006),
    (L_SH, R_SH, 0.012), (L_HIP, R_HIP, 0.008),
    (L_SH, L_HIP, 0.03), (R_SH, R_HIP, 0.03), (L_SH, R_HIP, 0.03), (R_SH, L_HIP, 0.03),
]
EDGES = [(JI[a], JI[b]) for a, b, _ in _E]
EDGE_SIG = np.array([s for _, _, s in _E])
EDGE_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (16, 17), (18, 19)]
HIPS = [JI[L_HIP], JI[R_HIP]]
PARENT = {JI[c]: JI[p] for p, c in ((L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR), (L_HIP, L_KNEE),
                                     (L_KNEE, L_ANK), (R_HIP, R_KNEE), (R_KNEE, R_ANK), (L_ANK, L_HEEL),
                                     (L_ANK, L_FOOT), (R_ANK, R_HEEL), (R_ANK, R_FOOT))}
# reprojection sigma in metres at the joint's depth (shoulder / hip "joints"
# are defined differently by every detector, limb joints are precise)
SIG2D = np.array([0.02, 0.02, 0.012, 0.012, 0.012, 0.012, 0.025, 0.025, 0.012, 0.012, 0.012, 0.012,
                  0.02, 0.02, 0.02, 0.02])
SIG_Z = 0.08
W_KNEE = 60.0
W_FOOT = 20.0
FOOT_CHOICE = True
W_SHOULDER = 40.0  # upper arm far behind the chest
W_ELBOW = 10.0
SIG_D = 0.02    # how well a bone's apparent length is known (detection error)
SIG_BZ = 0.05   # learned depth difference along one bone
TORSO_WZ = 4.0   # extra trust in the learned depth of shoulders and hips
SIG_ROOT = 0.02  # relative uncertainty of the character's distance
# limbs: (root joint, middle, end, is_leg, side sign)
LIMBS = [(L_SH, L_EL, L_WR, False), (R_SH, R_EL, R_WR, False),
         (L_HIP, L_KNEE, L_ANK, True), (R_HIP, R_KNEE, R_ANK, True)]


def _rays(uv, focal, cx, cy):
    """(n,J,2) pixels -> (n,J,3) rays with z = 1."""
    f = focal[:, None]
    return np.stack([(uv[..., 0] - cx) / f, (uv[..., 1] - cy) / f, np.ones(uv.shape[:2])], -1)


def estimate_lengths(uv, conf, zrel, zroot, focal):
    """Bone lengths (metres, in the scale of zroot) from the frames where a bone
    is close to parallel to the picture: there its projected length is its
    true length, with only a small correction from the depth guess."""
    Zj = zroot[:, None] + zrel
    L = np.zeros(len(EDGES))
    for e, (a, b) in enumerate(EDGES):
        zm = (Zj[:, a] + Zj[:, b]) / 2
        dxy = np.linalg.norm(uv[:, b] - uv[:, a], axis=-1) * zm / focal
        dz = zrel[:, b] - zrel[:, a]
        full = np.sqrt(dxy ** 2 + dz ** 2)
        ok = (np.minimum(conf[:, a], conf[:, b]) > 0.5) & np.isfinite(full)
        flat = ok & (np.abs(dz) < 0.5 * dxy)
        sel = flat if flat.sum() >= 20 else ok
        if not sel.any():
            L[e] = np.nan
            continue
        # a bone is never seen longer than it is: the estimate must also cover
        # (almost) all projected lengths
        L[e] = max(np.median(full[sel]), np.percentile(dxy[ok], 95))
    for i, j in EDGE_PAIRS:
        v = np.nanmean([L[i], L[j]])
        L[i] = L[j] = v
    return L


def estimate_root_depth(uv, conf, zrel, L, focal, n_rigid=16):
    """Distance of the hip centre per frame from how large the bones appear.
    A bone of length L can never look longer than f*L/Z pixels, so every bone
    gives an upper limit for the distance Z, reached when the bone lies
    parallel to the picture. With many bones a few always nearly do: the
    second smallest limit is used (robust to one bad detection). Returns (n,)
    (nan where nothing was usable)."""
    n = len(uv)
    bound = np.full((n, n_rigid), np.inf)
    for e, (a, b) in enumerate(EDGES[:n_rigid]):
        if not np.isfinite(L[e]) or 8 <= e < 14:  # feet are too short to measure with
            continue
        dpx = np.linalg.norm(uv[:, b] - uv[:, a], axis=-1)
        z = focal * L[e] / np.maximum(dpx, 1e-6) - (zrel[:, a] + zrel[:, b]) / 2
        bound[:, e] = np.where(np.minimum(conf[:, a], conf[:, b]) > 0.5, z, np.inf)
    srt = np.sort(bound, 1)
    cnt = np.isfinite(srt).sum(1)
    Z = np.where(cnt >= 4, srt[:, 1], np.nan)
    return Z


def _facing(X):
    """Unit vector the pelvis faces (cv camera frame), (n,3)."""
    a = X[:, JI[L_HIP]] - X[:, JI[R_HIP]]
    up = (X[:, JI[L_SH]] + X[:, JI[R_SH]]) / 2 - (X[:, JI[L_HIP]] + X[:, JI[R_HIP]]) / 2
    return normalize(np.cross(a, up)), normalize(up)


def _chest_facing(X):
    """Unit vector the chest faces (cv camera frame), (n,3), from the shoulder
    line and the spine."""
    a = X[:, JI[L_SH]] - X[:, JI[R_SH]]
    up = (X[:, JI[L_SH]] + X[:, JI[R_SH]]) / 2 - (X[:, JI[L_HIP]] + X[:, JI[R_HIP]]) / 2
    return normalize(np.cross(a, up))


_GRID = np.linspace(0.0, 1.1, 34)


def _two_roots(P, ray, L, dz_prior):
    """The two possible places of a bone's far end on its camera ray: towards
    the camera or away from it, as seen from the parent joint P (n,3).

    Geometrically the far end is where the ray is exactly L from P. But a
    bone lying nearly parallel to the picture makes that very sensitive: one
    centimetre of detection error tilts it by 15 degrees. So each side is
    solved as a trade-off between the bone length (allowing for detection
    error) and the learned depth difference dz_prior (n,) along the bone.
    Returns (points (n,2,3), cost (n,2)); nearer-to-camera side first."""
    r = normalize(ray)
    t0 = np.sum(r * P, -1)
    d2 = np.maximum(np.sum(P * P, -1) - t0 ** 2, 0.0)
    pts, costs = [], []
    for sgn in (-1.0, 1.0):
        u = np.repeat(sgn * _GRID[None, :] * L, len(P), 0)  # (n,K) offset along the ray from the closest point
        blen = np.sqrt(d2[:, None] + u ** 2)
        dz = r[:, 2:3] * (t0[:, None] + u) - P[:, 2:3]
        c = ((blen - L) / SIG_D) ** 2 + ((dz - dz_prior[:, None]) / SIG_BZ) ** 2
        k = np.argmin(c, 1)
        ar = np.arange(len(P))
        pts.append(r * (t0 + u[ar, k])[:, None])
        costs.append(c[ar, k])
    return np.stack(pts, 1), np.stack(costs, 1)


def _rot_between(a, b):
    """Batched minimal rotation taking unit a to unit b: returns function applying it to v."""
    v = np.cross(a, b)
    c = np.sum(a * b, -1, keepdims=True)

    def apply(x):
        # Rodrigues with sin = |v|, cos = c
        k = 1.0 / np.maximum(1.0 + c, 1e-6)
        return x * c + np.cross(v, x) + v * np.sum(v * x, -1, keepdims=True) * k
    return apply


def _viterbi(unary, pair_fn, n, S):
    cost = unary[0].copy()
    back = np.zeros((n, S), int)
    for f in range(1, n):
        c2 = cost[None, :] + pair_fn(f)  # (new, old)
        back[f] = np.argmin(c2, 1)
        cost = c2[np.arange(S), back[f]] + unary[f]
    st = np.zeros(n, int)
    st[-1] = int(np.argmin(cost))
    for f in range(n - 1, 0, -1):
        st[f - 1] = back[f, st[f]]
    return st


def _limb_choice(X, rays, zrel, conf, L1, L2, limb, fps):
    """Choose towards/away for the middle and end joint of one limb over a
    segment. X: (n,J,3) current estimate. Returns new (mid, end) positions."""
    root, mid, end, is_leg = limb
    n = len(X)
    P = X[:, JI[root]]
    # the learned depth is used bone by bone (how much each bone points towards
    # / away from the camera): its errors add up along a limb otherwise
    q1p = zrel[:, JI[mid]] - zrel[:, JI[root]]
    q2p = zrel[:, JI[end]] - zrel[:, JI[mid]]
    cm, c1 = _two_roots(P, rays[:, JI[mid]], L1, q1p)  # (n,2,3)
    ce, c2 = zip(*[_two_roots(cm[:, k], rays[:, JI[end]], L2, q2p) for k in (0, 1)])
    S = 4
    M = np.stack([cm[:, 0], cm[:, 0], cm[:, 1], cm[:, 1]], 1)  # (n,4,3)
    E = np.stack(ce, 1).reshape(n, 4, 3)
    unary = np.minimum(np.stack([c1[:, 0], c1[:, 0], c1[:, 1], c1[:, 1]], 1), 12.0) * conf[:, JI[mid], None] + \
        np.minimum(np.stack(c2, 1).reshape(n, 4), 12.0) * conf[:, JI[end], None]
    d1 = normalize(M - P[:, None])
    d2 = normalize(E - M)
    # anatomy: which of the candidates are possible at all
    F, up = _facing(X)
    F4, dn4 = np.repeat(F[:, None], 4, 1), np.repeat(-up[:, None], 4, 1)
    front = _rot_between(dn4, d1)(F4)  # "front" of the upper bone, carried along from standing
    bend = d2 - np.sum(d2 * d1, -1, keepdims=True) * d1  # sideways part of the lower bone
    fwd = np.sum(bend * front, -1)
    if is_leg:
        # a knee only bends backwards; a thigh extends at most ~25 deg behind the body
        unary = unary + W_KNEE * np.maximum(fwd - 0.15, 0.0) ** 2
        unary = unary + W_KNEE * np.maximum(-np.sum(d1 * F4, -1) - 0.45, 0.0) ** 2
    else:
        # an elbow does not bend backwards
        unary = unary + W_ELBOW * np.maximum(-fwd - 0.3, 0.0) ** 2
        # and an upper arm reaches at most ~35 deg behind the chest (an arm
        # pointing at the camera is in front of the body, not straight behind it)
        Fc = _chest_facing(X)
        unary = unary + W_SHOULDER * np.maximum(-np.sum(d1 * Fc[:, None], -1) - 0.55, 0.0) ** 2
    k = (fps / 30.0 / 0.35) ** 2

    def pair(f):
        a = np.sum((d1[f][:, None] - d1[f - 1][None]) ** 2, -1)
        b = np.sum((d2[f][:, None] - d2[f - 1][None]) ** 2, -1)
        return k * (a + b)
    st = _viterbi(unary, pair, n, S)
    ar = np.arange(n)
    return M[ar, st], E[ar, st]


def _foot_choice(X, rays, zrel, conf, L, ank, heel, toe, fps):
    """Towards/away for the toes of one foot (then the heel to fit): a foot
    points forwards-ish as seen from the pelvis, never backwards. Returns
    (heel, toe) positions (n,3)."""
    n = len(X)
    A = X[:, JI[ank]]
    e_at, e_ah, e_ht = EDGES.index((JI[ank], JI[toe])), EDGES.index((JI[ank], JI[heel])), EDGES.index((JI[heel], JI[toe]))
    ct, cost = _two_roots(A, rays[:, JI[toe]], L[e_at], zrel[:, JI[toe]] - zrel[:, JI[ank]])
    F, _ = _facing(X)
    d = normalize(ct - A[:, None])
    unary = np.minimum(cost, 12.0) * conf[:, JI[toe], None] + \
        W_FOOT * np.maximum(-np.sum(d * F[:, None], -1) - 0.2, 0.0) ** 2
    k = (fps / 30.0 / 0.35) ** 2
    st = _viterbi(unary, lambda f: k * np.sum((d[f][:, None] - d[f - 1][None]) ** 2, -1), n, 2)
    T = ct[np.arange(n), st]
    ch, cost_h = _two_roots(A, rays[:, JI[heel]], L[e_ah], zrel[:, JI[heel]] - zrel[:, JI[ank]])
    fit = ((np.linalg.norm(ch - T[:, None], axis=-1) - L[e_ht]) / SIG_D) ** 2 + np.minimum(cost_h, 12.0)
    Hh = ch[np.arange(n), np.argmin(fit, 1)]
    return Hh, T


def _gauss_newton(X, uv, w2d, zp, wz, L, focal, cx, cy, fps, zroot, iters=10):
    """Refine all joints of a segment together. X: (n,J,3)."""
    n, J, _ = X.shape
    D = 3 * J
    A = np.zeros((J, D))  # learned depth: limb joints relative to their parent, torso to the hip centre
    for j in range(J):
        A[j, 3 * j + 2] += 1
        if j in PARENT:
            A[j, 3 * PARENT[j] + 2] -= 1
        else:
            for h in HIPS:
                A[j, 3 * h + 2] -= 1.0 / len(HIPS)
    zp = zp - np.stack([zp[:, PARENT[j]] if j in PARENT else np.zeros(n) for j in range(J)], 1)
    wL = 1.0 / EDGE_SIG ** 2
    okL = np.isfinite(L)
    sig_t = 0.01 * (30.0 / fps) ** 2
    wt = 1.0 / sig_t ** 2
    f = focal[:, None]
    eye = np.eye(D)
    X = X.copy()
    for it in range(iters):
        for parity in (0, 1):
            H = np.zeros((n, D, D))
            g = np.zeros((n, D))
            # reprojection (residual in metres at the joint's depth)
            Z = np.maximum(X[..., 2], 0.05)
            ru = (X[..., 0] / Z - (uv[..., 0] - cx) / f) * Z
            rv = (X[..., 1] / Z - (uv[..., 1] - cy) / f) * Z
            rn = np.sqrt(ru ** 2 + rv ** 2) / SIG2D
            w = w2d / SIG2D ** 2 * np.minimum(1.0, 2.5 / np.maximum(rn, 1e-9))
            Jp = np.zeros((n, J, 2, 3))
            Jp[..., 0, 0] = 1
            Jp[..., 0, 2] = -(uv[..., 0] - cx) / f
            Jp[..., 1, 1] = 1
            Jp[..., 1, 2] = -(uv[..., 1] - cy) / f
            Hj = np.einsum("njab,njac,nj->njbc", Jp, Jp, w)
            gj = Jp[..., 0, :] * (ru * w)[..., None] + Jp[..., 1, :] * (rv * w)[..., None]
            # smoothness along the line of sight
            m = X.copy()
            if n > 2:
                m[1:-1] = (X[:-2] + X[2:]) / 2
                m[0] = X[1]
                m[-1] = X[-2]
                wt_f = np.full(n, wt)
                wt_f[[0, -1]] = wt * 0.25
            elif n == 2:
                m[0], m[1] = X[1], X[0]
                wt_f = np.full(n, wt * 0.25)
            else:
                wt_f = np.zeros(n)
            ray = normalize(X)
            dev = np.sum(ray * (X - m), -1)
            Hj += np.einsum("nja,njb,n->njab", ray, ray, wt_f) + 0.02 * wt_f[:, None, None, None] * np.eye(3)
            gj += ray * (dev * wt_f[:, None])[..., None] + 0.02 * wt_f[:, None, None] * (X - m)
            for j in range(J):
                H[:, 3 * j:3 * j + 3, 3 * j:3 * j + 3] += Hj[:, j]
                g[:, 3 * j:3 * j + 3] += gj[:, j]
            # bone lengths
            for e, (a, b) in enumerate(EDGES):
                if not okL[e]:
                    continue
                d = X[:, b] - X[:, a]
                nn = np.maximum(np.linalg.norm(d, axis=-1), 1e-9)
                u = d / nn[:, None]
                uu = np.einsum("ni,nj->nij", u, u) * wL[e]
                gr = u * ((nn - L[e]) * wL[e])[:, None]
                sa, sb = slice(3 * a, 3 * a + 3), slice(3 * b, 3 * b + 3)
                H[:, sa, sa] += uu
                H[:, sb, sb] += uu
                H[:, sa, sb] -= uu
                H[:, sb, sa] -= uu
                g[:, sb] += gr
                g[:, sa] -= gr
            # learned depth (robust)
            r = X.reshape(n, D) @ A.T - zp
            wr = wz / SIG_Z ** 2 / (1.0 + (r / (2 * SIG_Z)) ** 2)
            H += np.einsum("nr,ri,rj->nij", wr, A, A)
            g += np.einsum("nr,ri->ni", wr * r, A)
            # distance of the hip centre
            ar = np.zeros(D)
            for h in HIPS:
                ar[3 * h + 2] = 1.0 / len(HIPS)
            wr0 = 1.0 / (SIG_ROOT * zroot) ** 2
            r0 = X[:, HIPS, 2].mean(1) - zroot
            H += wr0[:, None, None] * np.outer(ar, ar)
            g += (wr0 * r0)[:, None] * ar
            lam = 1e-4 * np.trace(H, axis1=1, axis2=2)[:, None, None] / D + 1e-9
            dx = np.linalg.solve(H + lam * eye, -g[..., None])[..., 0].reshape(n, J, 3)
            X[parity::2] += dx[parity::2]
    return X


def apply_torso_facing(zrel, wz, P0, torso):
    """Depth of the shoulders and hips from the tracked facing angles: the
    left-right axis is cos(psi) across the picture and sin(psi) in depth."""
    psi_c, psi_p, r3, view = torso
    for (a, b), psi in (((L_SH, R_SH), psi_c), ((L_HIP, R_HIP), psi_p)):
        ok = np.isfinite(psi)
        width = np.median(np.linalg.norm(P0[ok, a] - P0[ok, b], axis=1)) if ok.any() else 0.3
        ax = np.cos(psi)[:, None] * r3 + np.sin(psi)[:, None] * view
        ia, ib = JI[a], JI[b]
        dz_t = width * ax[:, 2]
        dz_l = zrel[:, ia] - zrel[:, ib]
        # where the pose model's own depth agrees with the tracked facing (within
        # ~0.35 of the width) it is the finer measurement near a frontal view
        agree = np.abs(dz_l - dz_t) < 0.35 * width
        dz = 0.5 * np.where(agree, 0.5 * (dz_l + dz_t), dz_t)
        mid = (zrel[:, ia] + zrel[:, ib]) / 2
        zrel[:, ia] = np.where(ok, mid + dz, zrel[:, ia])
        zrel[:, ib] = np.where(ok, mid - dz, zrel[:, ib])
        wz[ok, ia] = wz[ok, ib] = TORSO_WZ * 2


def _prepare(P0, uv, conf, segments, fps, z_extra, torso, z_extra_w):
    N = len(P0)
    idx = np.array(JOINTS)
    fin = np.zeros(N, bool)
    for s, e in segments:
        fin[s:e] = True
    uvj = uv[:, idx]
    cj = np.clip(conf[:, idx], 0, 1)
    hip0 = (P0[:, L_HIP] + P0[:, R_HIP]) / 2
    zroot = hip0[:, 2].copy()
    zrel = P0[:, idx, 2] - zroot[:, None]
    wz = np.ones_like(zrel)
    wz[:, [JI[L_SH], JI[R_SH], JI[L_HIP], JI[R_HIP]]] = TORSO_WZ
    if z_extra is not None:
        ze = z_extra[:, idx]
        has = np.isfinite(ze)
        w_ = np.asarray(z_extra_w, np.float64)
        if w_.ndim:  # per joint (33,) or per frame and joint (N,33)
            w_ = w_[..., idx]
        zrel = np.where(has, (1 - w_) * zrel + w_ * np.nan_to_num(ze), zrel)
    for s, e in segments:
        if e - s >= 3:
            zrel[s:e] = gaussian_smooth(zrel[s:e], 1.5 * fps / 30.0)
    if torso is not None:
        apply_torso_facing(zrel, wz, P0, torso)
    return N, idx, fin, uvj, cj, zrel, wz, zroot


def _lengths_and_distance(uvj, cj, zrel, zroot, focal, segments, fps, fin):
    zroot = zroot.copy()
    # bone lengths and the character's distance depend on each other: alternate.
    # (the overall scale stays that of the rough input)
    z_in = zroot.copy()
    for _ in range(2):
        L = estimate_lengths(uvj[fin], cj[fin], zrel[fin], zroot[fin], focal[fin])
        Z = estimate_root_depth(uvj, cj, zrel, L, focal)
        for s, e in segments:
            z = Z[s:e]
            okz = np.isfinite(z)
            if okz.sum() < 2:
                continue
            z = np.interp(np.arange(e - s), np.where(okz)[0], z[okz])
            if e - s >= 5:
                med = np.array([np.median(z[max(0, i - 2):i + 3]) for i in range(e - s)])
                z = gaussian_smooth(med[:, None], 0.15 * fps)[:, 0]
            zroot[s:e] = z
    k = np.median(z_in[fin]) / np.median(zroot[fin])
    zroot *= k
    L = L * k
    return L, zroot


def measure_proportions(P0, uv, conf, focal, segments, fps, z_extra=None, torso=None, z_extra_w=0.5):
    """The proportions pass: every bone's length (left and right equal),
    measured over the whole video before any pose is solved, from the frames
    where a bone lies across the picture. Returns (L per EDGES (metres, in
    the scale of the rough input), the character's distance per frame)."""
    N, idx, fin, uvj, cj, zrel, wz, zroot = _prepare(P0, uv, conf, segments, fps, z_extra, torso, z_extra_w)
    return _lengths_and_distance(uvj, cj, zrel, zroot, focal, segments, fps, fin)


def fuse_body(P0, uv, conf, focal, cx, cy, segments, fps, z_extra=None, progress=None, torso=None, z_extra_w=0.5,
              measured=None):
    """P0: (N,33,3) rough camera-space landmarks (learned depth, metric root
    distance). uv: (N,33,2) picture positions. conf: (N,33) 0..1.
    z_extra: optional (N,33) second depth opinion relative to the hip centre
    (nan = none). segments: list of (start, end) runs of valid frames.
    torso: optional (psi_chest, psi_pelvis, r3, view) from facing.track_facing:
    sets how much deeper the left shoulder / hip is than the right one.
    Returns (P (N,33,3) with the solved joints replaced, lengths dict)."""
    N, idx, fin, uvj, cj, zrel, wz, zroot = _prepare(P0, uv, conf, segments, fps, z_extra, torso, z_extra_w)
    if measured is not None:
        L, zroot = measured
    else:
        L, zroot = _lengths_and_distance(uvj, cj, zrel, zroot, focal, segments, fps, fin)
    P = P0.copy()
    rays_all = _rays(uvj, focal, cx, cy)
    for k, (s, e) in enumerate(segments):
        sl = slice(s, e)
        rays = rays_all[sl]
        X = rays * (zroot[sl, None] + zrel[sl])[..., None]
        args = (uvj[sl], cj[sl], zrel[sl], wz[sl], L, focal[sl], cx, cy, fps, zroot[sl])
        X = _gauss_newton(X, *args, iters=4)
        for limb in LIMBS:
            root, mid, end, is_leg = limb
            e1 = EDGES.index((JI[root], JI[mid]))
            e2 = EDGES.index((JI[mid], JI[end]))
            if not (np.isfinite(L[e1]) and np.isfinite(L[e2])):
                continue
            M, E = _limb_choice(X, rays, zrel[sl], cj[sl], L[e1], L[e2], limb, fps)
            dE = E - X[:, JI[end]]
            X[:, JI[mid]] = M
            X[:, JI[end]] = E
            # from here on the depth guess is the chosen side
            zr = zrel[sl]
            zr[:, JI[mid]] = zr[:, JI[root]] + (M[:, 2] - X[:, JI[root], 2])
            zr[:, JI[end]] = zr[:, JI[mid]] + (E[:, 2] - M[:, 2])
            if is_leg:  # the foot moves with the ankle, then gets its own towards/away choice
                heel, toe = (L_HEEL, L_FOOT) if end == L_ANK else (R_HEEL, R_FOOT)
                for j in (heel, toe):
                    X[:, JI[j]] += dE
                ok_f = all(np.isfinite(L[EDGES.index(pr)]) for pr in
                           ((JI[end], JI[toe]), (JI[end], JI[heel]), (JI[heel], JI[toe])))
                if ok_f and FOOT_CHOICE:
                    Hh, Tt = _foot_choice(X, rays, zrel[sl], cj[sl], L, end, heel, toe, fps)
                    X[:, JI[heel]], X[:, JI[toe]] = Hh, Tt
                    zr[:, JI[heel]] = zr[:, JI[end]] + (Hh[:, 2] - E[:, 2])
                    zr[:, JI[toe]] = zr[:, JI[end]] + (Tt[:, 2] - E[:, 2])
        X = _gauss_newton(X, *args, iters=10)
        P[sl, idx] = X
        if progress:
            progress((k + 1) / len(segments))
    lengths = {(JOINTS[a], JOINTS[b]): float(L[e]) for e, (a, b) in enumerate(EDGES) if np.isfinite(L[e])}
    return P, lengths

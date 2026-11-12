"""Which way the torso faces, frame by frame.

A single camera sees the body flat; the hardest part of a turn is telling
which way the chest and pelvis face. Several independent clues are combined
over time:

  * how wide the shoulders / hips look compared with their real width
    (full width = facing the camera or facing away, zero = side on),
  * whether the character's left is on the right of the picture (front) or on
    the left (back) - only as reliable as the detector's left/right labels,
    which are often swapped in back views, so the labels get a state too,
  * the nose sits in front of the neck: its sideways offset says which way
    the head (and with it, roughly, the chest) is turned,
  * whether the face can be seen at all (front vs back),
  * bent knees and toes point forwards,
  * the pose model's own 3D guess (it often mirrors the depth).

Each frame gets a cost for every facing angle (5 degree steps) and label
reading; the cheapest sequence that turns smoothly is found per shot
(Viterbi), then refined to a continuous angle.

Angles: psi = 0 faces the camera, psi = +90 faces picture-right (the
character's left shoulder is further away), 180 faces away.
"""
from __future__ import annotations

import numpy as np

from .detect import (L_ANK, L_FOOT, L_HEEL, L_HIP, L_KNEE, L_SH, NOSE, R_ANK,
                     R_FOOT, R_HEEL, R_HIP, R_KNEE, R_SH)
from .mathutil import normalize

K = 72
PSI = np.arange(K) * (2 * np.pi / K)
COS, SIN = np.cos(PSI), np.sin(PSI)
_DIFF = np.angle(np.exp(1j * (PSI[:, None] - PSI[None, :])))  # (new, old) wrapped

# weights / noise levels
SIG_W = 0.16        # apparent width / true width
W_NOSE = 2.0
SIG_NOSE = 0.55
W_FACE = 2.5
W_KNEE = 3.0
W_TOE = 1.5
W_LEARN = 4.0
W2_SHARE = 0.65    # share of the motion-over-time guess in the learned cue
LP2_FLIP = 1.2     # its front/back mistake reading
LP2_MIRROR = 1.5   # its depth-mirror mistake reading
SPIN_SLOPE = 1.0   # cost slope of turning faster than 2 x SIG_V per frame
SPIN_MAX = 5.0     # fastest turn (x SIG_V per frame) a spin reaches
SWAP_PRIOR = 0.15   # per frame cost of reading the labels swapped
SWAP_SWITCH = 6.0   # cost of the labels changing over between frames
SIG_V = np.radians(7.0)   # facing change per frame at 30 fps before it costs much
SIG_TWIST = np.radians(28.0)  # pelvis vs chest
W_FRONT = 3.0       # front/back voted over a moment from the hips' left/right labels (close-ups)
W_PRIOR = 3.0       # a facing carried over from the surrounding shots (close-ups), per frame
BAD_MARGIN = 6.0    # cost by which a frame must contradict its tracked reading to be dropped


def _wrap(a):
    return np.angle(np.exp(1j * a))


def image_axes(uv, valid):
    """Per frame 2D 'up' of the torso and the lateral axis (picture right when
    upright), both unit (N,2)."""
    sh = (uv[:, L_SH] + uv[:, R_SH]) / 2
    hp = (uv[:, L_HIP] + uv[:, R_HIP]) / 2
    up = sh - hp
    n = np.linalg.norm(up, axis=1, keepdims=True)
    up = np.where(n > 1e-6, up / np.maximum(n, 1e-6), np.array([0.0, -1.0]))
    lat = np.stack([-up[:, 1], up[:, 0]], 1)
    return up, lat


def learned_psi(Wc, view, lat2):
    """Facing angle of the pose model's 3D guess (chest, pelvis)."""
    out = []
    for a_, b_ in ((L_SH, R_SH), (L_HIP, R_HIP)):
        a = Wc[:, a_] - Wc[:, b_]
        up = (Wc[:, L_SH] + Wc[:, R_SH]) / 2 - (Wc[:, L_HIP] + Wc[:, R_HIP]) / 2
        F = normalize(np.cross(a, up))
        r3 = np.c_[lat2, np.zeros(len(lat2))]
        r3 = normalize(r3 - view * np.sum(r3 * view, 1, keepdims=True))
        out.append(np.arctan2(np.sum(F * r3, 1), -np.sum(F * view, 1)))
    return out


def _viterbi(unary, trans, swap_cost=None):
    """unary (n,S); trans (S,S) [new, old]. Returns state path."""
    n, S = unary.shape
    cost = unary[0].copy()
    back = np.zeros((n, S), np.int32)
    for f in range(1, n):
        c2 = trans + cost[None, :]
        back[f] = np.argmin(c2, 1)
        cost = c2[np.arange(S), back[f]] + unary[f]
    st = np.zeros(n, np.int32)
    st[-1] = int(np.argmin(cost))
    for f in range(n - 1, 0, -1):
        st[f - 1] = back[f, st[f]]
    return st


def _trans_psi(fps):
    sv = SIG_V * 30.0 / fps
    d = np.abs(_DIFF) / sv
    # quadratic for normal turning, gently linear beyond: a fast spin (two
    # turns a second) must stay cheaper than reading the labels swapped
    # (up to SPIN_MAX x SIG_V per frame, ~1000 deg/s); a jump faster than any
    # spin is not a turn at all
    c = np.where(d < 2.0, d ** 2, 2.0 + SPIN_SLOPE * (np.minimum(d, SPIN_MAX) - 2.0))
    return (c + 4.0 * np.maximum(d - SPIN_MAX, 0.0)) * 0.5


def _refine(unary_row, k):
    """Sub-step angle around state k from the local cost curvature."""
    c0, cm, cp = unary_row[k], unary_row[(k - 1) % K], unary_row[(k + 1) % K]
    den = cm - 2 * c0 + cp
    off = 0.0 if den <= 1e-9 else float(np.clip(0.5 * (cm - cp) / den, -0.5, 0.5))
    return PSI[k] + off * (2 * np.pi / K)


def _smooth_angle(a, sigma):
    if sigma <= 0 or len(a) < 3:
        return a
    from .mathutil import gaussian_smooth
    z = np.stack([np.cos(a), np.sin(a)], 1)
    z = gaussian_smooth(z, sigma)
    return np.arctan2(z[:, 1], z[:, 0])


def _true_width(w, ok, fallback):
    if ok.sum() < 30:
        return fallback
    return float(np.percentile(w[ok], 95))


def track_facing(uv, conf, face_score, Wc, T, focal, cx, cy, valid, segments, fps, debug=None, W2=None, prior=None,
                 front=None):
    """uv (N,33,2) picture positions, conf (N,33) 0..1 (MediaPipe numbering,
    labels as detected). face_score (N,) 0..1 how clearly a face is seen (nan =
    unknown). Wc (N,33,3) learned 3D in camera orientation, T (N,3) camera ->
    hip centre. segments: (start, end) spans to track through (each within one
    shot; frames inside that are not valid are bridged). Returns (psi_chest,
    psi_pelvis, swap, bad) per frame: nan / False outside valid frames; bad =
    frames whose detection contradicts the tracked facing (better not used).
    prior: optional (N,) facing (psi, radians) known from elsewhere (carried
    over from the surrounding shots), nan where none: it counts as one more
    clue, so what the picture shows can still overrule it.
    front: optional (N,) -1..1, + = facing the camera, - = turned away (nan
    = no opinion): a front/back clue, weighted by its size."""
    N = len(uv)
    psi_c = np.full(N, np.nan)
    psi_p = np.full(N, np.nan)
    swap = np.zeros(N, bool)
    Z = np.maximum(np.nan_to_num(T[:, 2], nan=3.0), 0.3)
    s_px = Z / focal  # metres per pixel at the character
    # true widths in the pose model's metric scale
    ok = valid & np.isfinite(Wc).all((1, 2))
    S_w = np.median(np.linalg.norm(Wc[ok, L_SH] - Wc[ok, R_SH], axis=1)) if ok.any() else 0.34
    H_w = np.median(np.linalg.norm(Wc[ok, L_HIP] - Wc[ok, R_HIP], axis=1)) if ok.any() else 0.2
    up2, lat2 = image_axes(uv, valid)
    cen = (uv[:, L_SH] + uv[:, R_SH] + uv[:, L_HIP] + uv[:, R_HIP]) / 4
    view = normalize(np.c_[(cen[:, 0] - cx) / focal, (cen[:, 1] - cy) / focal, np.ones(N)])

    def lat_of(j1, j2):
        return np.sum((uv[:, j1] - uv[:, j2]) * lat2, 1) * s_px

    c_sh = np.minimum(conf[:, L_SH], conf[:, R_SH])
    c_hp = np.minimum(conf[:, L_HIP], conf[:, R_HIP])
    # the detectors' shoulder / hip points need not be as far apart as the
    # pose model's: the real width is the widest they ever appear
    ls, lh = lat_of(L_SH, R_SH), lat_of(L_HIP, R_HIP)
    lp_c, lp_p = learned_psi(Wc, view, lat2)
    # a second guess read from the motion over time (W2): steadier, and right
    # more often; where it is there, it counts for most of the learned cue
    lp2_c = lp2_p = None
    if W2 is not None:
        lp2_c, lp2_p = learned_psi(np.nan_to_num(W2), view, lat2)
        has2 = np.isfinite(W2[:, [L_SH, R_SH, L_HIP, R_HIP]]).all((1, 2))
        lp2_c = np.where(has2, lp2_c, np.nan)
        lp2_p = np.where(has2, lp2_p, np.nan)
    # where the motion-over-time guess says the body is square to the camera,
    # the widths seen then are the true widths (the widest ever seen can be a
    # mis-detection: a shoulder point put on a hand in front of it)
    sq_c = sq_p = np.zeros(N, bool)
    if lp2_c is not None:
        sq_c = valid & (np.abs(np.cos(np.nan_to_num(lp2_c, nan=np.pi / 2))) > 0.94) & (c_sh > 0.5)
        sq_p = valid & (np.abs(np.cos(np.nan_to_num(lp2_p, nan=np.pi / 2))) > 0.94) & (c_hp > 0.5)
    S_w = float(np.median(np.abs(ls[sq_c]))) if sq_c.sum() >= 20 else _true_width(np.abs(ls), valid & (c_sh > 0.5), S_w)
    H_w = float(np.median(np.abs(lh[sq_p]))) if sq_p.sum() >= 20 else _true_width(np.abs(lh), valid & (c_hp > 0.5), H_w)
    w_sh = np.clip(ls / S_w, -1.4, 1.4)
    w_hp = np.clip(lh / H_w, -1.4, 1.4)
    # nose: sideways offset from the shoulder centre, relative to how far the
    # nose sits in front of the neck (~0.3 shoulder widths)
    shc = (uv[:, L_SH] + uv[:, R_SH]) / 2
    nose_n = np.clip(np.sum((uv[:, NOSE] - shc) * lat2, 1) * s_px / (0.3 * S_w), -1.6, 1.6)
    fs = np.nan_to_num(face_score, nan=0.5)
    c_nose = np.clip(conf[:, NOSE], 0, 1) * np.clip((fs - 0.2) / 0.4, 0, 1)
    # knees / toes: forward clues independent of the left/right labels
    fwd_c = np.zeros((N, K))
    for hip, kn, an in ((L_HIP, L_KNEE, L_ANK), (R_HIP, R_KNEE, R_ANK)):
        d = uv[:, an] - uv[:, hip]
        dn = np.linalg.norm(d, axis=1)
        t = np.clip(np.sum((uv[:, kn] - uv[:, hip]) * d, 1) / np.maximum(dn ** 2, 1e-6), 0, 1)
        off = uv[:, kn] - (uv[:, hip] + t[:, None] * d)
        k_n = np.sum(off * lat2, 1) / np.maximum(dn, 1e-6)  # + = knee sticks out to picture right
        k_n = np.where(np.abs(k_n) > 0.08, k_n, 0.0)
        w = np.minimum(np.minimum(conf[:, hip], conf[:, kn]), conf[:, an])
        fwd_c += W_KNEE * (w * np.clip(np.abs(k_n) * 3, 0, 1))[:, None] * np.maximum(0, -np.sign(k_n)[:, None] * SIN[None])
    for heel, toe, an in ((L_HEEL, L_FOOT, L_ANK), (R_HEEL, R_FOOT, R_ANK)):
        d = np.sum((uv[:, toe] - uv[:, heel]) * lat2, 1)
        ln = np.linalg.norm(uv[:, toe] - uv[:, heel], axis=1)
        t_n = d / np.maximum(ln, 1e-6) * np.clip(ln * s_px / 0.12, 0, 1)
        w = np.minimum(conf[:, heel], conf[:, toe])
        fwd_c += W_TOE * (w * np.clip(np.abs(t_n), 0, 1))[:, None] * np.maximum(0, -np.sign(t_n)[:, None] * SIN[None]) * 0.5

    def width_cost(wm, c, sgn):
        return np.minimum(((wm[:, None] - sgn * COS[None]) / SIG_W) ** 2, 9.0) * c[:, None]

    def learned_cost2(lp, lp2, swapped=False):
        c1 = learned_cost(lp, swapped)
        if lp2 is None:
            return c1
        # the motion-over-time guess hardly ever mirrors the depth: its
        # "mistake" readings cost more
        c2 = learned_cost(np.nan_to_num(lp2), swapped, pen=(LP2_FLIP, LP2_MIRROR))
        return np.where(np.isfinite(lp2)[:, None], (1 - W2_SHARE) * c1 + W2_SHARE * c2, c1)

    def learned_cost(lp, swapped=False, pen=(0.15, 0.4)):
        # the model's guess, or the mistake it makes in back views: it reads
        # the body as facing the camera (front and back, left and right both
        # flipped: psi -> pi - psi), or, less often, only mirrors the depth
        # (which shoulder is nearer: psi -> -psi). Left/right labels in the
        # picture are handled by the swap state.
        if swapped:
            lp = lp + np.pi
        d0 = 1 - np.cos(PSI[None] - lp[:, None])
        d1 = 1 - np.cos(PSI[None] - (np.pi - lp[:, None]))
        d2 = 1 - np.cos(PSI[None] + lp[:, None])
        return W_LEARN * np.minimum(np.minimum(d0, d1 + pen[0]), d2 + pen[1])

    # face seen: front half; clearly not seen: back half
    face_c = W_FACE * (np.clip((fs - 0.45) / 0.25, 0, 1)[:, None] * np.maximum(0, -COS[None] - 0.25) +
                       np.clip((0.3 - fs) / 0.2, 0, 1)[:, None] * np.maximum(0, COS[None] - 0.25))
    face_c = np.where(np.isfinite(face_score)[:, None], face_c, 0.0)
    # the head turns on its own: the nose only says which way, not how far
    nose_c = W_NOSE * (c_nose * np.clip(np.abs(nose_n) - 0.25, 0, 1))[:, None] * \
        np.maximum(0, -np.sign(nose_n)[:, None] * SIN[None])
    tp = _trans_psi(fps)
    # (psi, swap) state space for the chest
    TR = np.full((2 * K, 2 * K), 0.0)
    TR[:K, :K] = tp
    TR[K:, K:] = tp
    TR[:K, K:] = tp + SWAP_SWITCH
    TR[K:, :K] = tp + SWAP_SWITCH
    bad = np.zeros(N, bool)
    for (s, e) in segments:
        if e - s < 1:
            continue
        sl = slice(s, e)
        obs = valid[sl].astype(float)[:, None]  # frames without a detection carry no evidence
        base = face_c[sl] + nose_c[sl]
        if front is not None:
            fv = np.nan_to_num(front[sl])
            base = base + W_FRONT * np.abs(fv)[:, None] * np.maximum(0.0, -np.sign(fv)[:, None] * COS[None])
        if prior is not None:
            pr = prior[sl]
            base = base + np.where(np.isfinite(pr)[:, None], W_PRIOR * (1 - np.cos(PSI[None] - np.nan_to_num(pr)[:, None])), 0.0)
        l2c = lp2_c[sl] if lp2_c is not None else None
        l2p = lp2_p[sl] if lp2_p is not None else None
        u0 = (base + learned_cost2(lp_c[sl], l2c) + width_cost(w_sh[sl], c_sh[sl], 1.0)) * obs
        u1 = (base + learned_cost2(lp_c[sl], l2c, True) + width_cost(w_sh[sl], c_sh[sl], -1.0) + SWAP_PRIOR) * obs
        U = np.concatenate([u0, u1], 1)
        st = _viterbi(U, TR)
        sw = st >= K
        kc = st % K
        uc = np.where(sw[:, None], u1, u0)
        ua = np.where(sw[:, None], u0, u1)
        pc = np.array([_refine(uc[i], kc[i]) for i in range(e - s)])
        # a single frame whose picture says the opposite of everything around it
        # (labels swapped for one frame, a mis-detection) is not used at all
        ar = np.arange(e - s)
        bad[sl] = valid[sl] & (uc[ar, kc] - ua.min(1) > BAD_MARGIN)
        # pelvis: hips, knees, toes, the learned guess, and staying within a
        # natural twist of the chest
        sgn = np.where(sw, -1.0, 1.0)
        twist = 0.5 * (_wrap(PSI[None] - pc[:, None]) / SIG_TWIST) ** 2
        up_ = (np.minimum(((w_hp[sl, None] - sgn[:, None] * COS[None]) / SIG_W) ** 2, 9.0) * c_hp[sl, None] +
               fwd_c[sl] + np.where(sw[:, None], learned_cost2(lp_p[sl], l2p, True), learned_cost2(lp_p[sl], l2p))) * obs + twist
        if front is not None:
            fv = np.nan_to_num(front[sl])
            up_ = up_ + W_FRONT * np.abs(fv)[:, None] * np.maximum(0.0, -np.sign(fv)[:, None] * COS[None])
        if prior is not None:
            pr = prior[sl]
            up_ = up_ + np.where(np.isfinite(pr)[:, None], W_PRIOR * (1 - np.cos(PSI[None] - np.nan_to_num(pr)[:, None])), 0.0)
        kp = _viterbi(up_, tp)
        pp = np.array([_refine(up_[i], kp[i]) for i in range(e - s)])
        # continuous: light zero-phase smoothing
        pcs = _smooth_angle(pc, 1.0 * fps / 30.0)
        pps = _smooth_angle(pp, 1.0 * fps / 30.0)
        v = valid[sl]
        psi_c[sl] = np.where(v, pcs, np.nan)
        psi_p[sl] = np.where(v, pps, np.nan)
        swap[sl] = sw & v
    if debug is not None:
        debug.update(dict(w_sh=w_sh, w_hp=w_hp, nose_n=nose_n, c_nose=c_nose, fs=fs, lp_c=lp_c, lp_p=lp_p, lp2_c=lp2_c))
    return psi_c, psi_p, swap, bad


def lateral_axes(uv, focal, cx, cy):
    """3D unit vectors (camera frame) for psi = 0 / 90 deg: returns (r3, view),
    psi measured as F = sin(psi) r3 - cos(psi) view."""
    N = len(uv)
    _, lat2 = image_axes(uv, None)
    cen = (uv[:, L_SH] + uv[:, R_SH] + uv[:, L_HIP] + uv[:, R_HIP]) / 4
    view = normalize(np.c_[(cen[:, 0] - cx) / focal, (cen[:, 1] - cy) / focal, np.ones(N)])
    r3 = np.c_[lat2, np.zeros(N)]
    r3 = normalize(r3 - view * np.sum(r3 * view, 1, keepdims=True))
    return r3, view

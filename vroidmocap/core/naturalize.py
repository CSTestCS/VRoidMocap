"""Anatomical constraints + IK and temporal filtering of bone rotations.

Everything works on *local* rotations in the normalized VRM frame (rest pose =
identity for every bone, bone axis = the rest direction to its child). The
skeleton itself is never changed: only rotations are corrected.

* filter_rotations: per-bone confidence -> spike rejection -> hold the last
  seen pose while a bone (or the whole character) is not visible, blending into
  the next seen pose -> motion-adaptive zero-phase smoothing.
* naturalize: elbows and knees are solved as hinges with IK (the upper arm /
  thigh twist is chosen so the hand / foot stays where it is), forearm twist
  is kept in range and continuous, wrists / fingers / thumbs / spine / neck /
  feet are held inside anatomical limits.
"""
from __future__ import annotations

import numpy as np

from .mathutil import axis_angle_to_mat, gaussian_smooth, mat_to_quat, normalize, quat_mul, quat_to_mat
from .vrm import FINGER_SEGS, FINGERS

X, Y, Z = np.eye(3)
IDQ = np.array([0, 0, 0, 1.0])


# ----------------------------------------------------------------------------- quaternion helpers
def q_axis_angle(axis, ang):
    """Quaternions for rotations of angle(s) `ang` about unit axis (axis (3,) or (n,3))."""
    ang = np.asarray(ang, float)
    s = np.sin(ang / 2)[..., None]
    return np.concatenate([np.asarray(axis, float) * s, np.cos(ang / 2)[..., None]], -1)


def q_log(q):
    """Rotation vector of quaternion(s)."""
    q = np.where(q[..., 3:4] < 0, -q, q)
    s = np.linalg.norm(q[..., :3], axis=-1)
    a = 2 * np.arctan2(s, q[..., 3])
    return q[..., :3] / np.maximum(s, 1e-12)[..., None] * a[..., None]


def q_angle(a, b):
    d = np.abs(np.sum(a * b, -1))
    return 2 * np.arccos(np.clip(d, 0, 1))


def continuity(q):
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] = -q[i]
    return q


def twist_angle(q, axis):
    """Signed twist angle(s) of q about `axis` (swing-twist decomposition)."""
    p = q[..., :3] @ axis
    return 2 * np.arctan2(p, q[..., 3])


def remove_twist(q, axis):
    """Swing part of q (twist about axis removed): q = swing * twist."""
    t = twist_angle(q, axis)
    tw = q_axis_angle(axis, t)
    tw_inv = tw * np.array([-1, -1, -1, 1.0])
    return quat_mul(q, tw_inv), t


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ----------------------------------------------------------------------------- temporal filtering
def filter_rotations(Q, conf, fps, smoothing=1.0, thr=0.5, blend_s=0.2, strength=None):
    """Q: (N,B,4) local quats, conf: (N,B) 0..1. Returns filtered Q (N,B,4).

    Frames where a bone's confidence is below `thr` (not visible, occluded,
    character off screen) keep the last seen rotation, then blend into the next
    seen rotation over `blend_s` seconds. Sudden spikes (single-frame flips)
    are treated as not seen. The result is smoothed more when the bone is
    nearly still (where jitter shows) and less when it moves fast."""
    N, B = Q.shape[:2]
    out = np.empty_like(Q)
    fs = fps / 30.0
    nb = max(1, int(round(blend_s * fps)))
    sig_h = (0.5 + 1.5 * max(0.0, smoothing)) * fs
    sig_l = (0.25 * max(0.0, smoothing)) * fs
    v_ref = np.radians(150.0) / fps  # rad/frame where the light filter takes over
    for b in range(B):
        mult = 1.0 if strength is None else float(strength[b])
        sh_b, sl_b = sig_h * mult, sig_l * mult
        q = Q[:, b].astype(np.float64)
        ok = conf[:, b] >= thr
        if not ok.any():
            out[:, b] = IDQ
            continue
        idx = np.where(ok)[0]
        qv = continuity(q[idx])
        # spike rejection on the seen frames: distance to the running median
        if len(qv) >= 5:
            k = max(2, int(round(0.12 * fps)))
            pad = np.concatenate([qv[:1].repeat(k, 0), qv, qv[-1:].repeat(k, 0)])
            win = np.lib.stride_tricks.sliding_window_view(pad, 2 * k + 1, axis=0)  # (n,4,w)
            med = normalize(np.median(win, axis=-1))
            dev = q_angle(qv, med)
            lim = max(np.radians(25), 5 * float(np.median(dev)))
            keep = dev < lim
            if keep.sum() >= 2:
                idx, qv = idx[keep], continuity(qv[keep])
        # hold + blend fill
        full = np.empty((N, 4))
        full[: idx[0] + 1] = qv[0]
        full[idx] = qv
        w = np.zeros(N)
        w[idx] = np.clip(conf[idx, b], 0.2, 1.0)
        for j in range(len(idx) - 1):
            a, c = idx[j], idx[j + 1]
            if c - a <= 1:
                continue
            qa, qc = qv[j], qv[j + 1]
            gap = c - a
            if gap <= nb:
                t = np.arange(1, gap) / gap
            else:
                t = np.clip((np.arange(1, gap) - (gap - nb)) / nb, 0, 1)
            t = t * t * (3 - 2 * t)
            full[a + 1:c] = slerp_many(qa, qc, t)
            w[a + 1:c] = 0.02
        full[idx[-1]:] = qv[-1]
        full = continuity(full)
        # motion-adaptive zero-phase smoothing
        heavy = normalize(gaussian_smooth(full, sh_b, valid=w > 0.05))
        light = normalize(gaussian_smooth(full, sl_b, valid=w > 0.05)) if sl_b > 0.05 else full
        # speed of the underlying motion (from the heavy filter: noise must not
        # count as motion, or it would switch the smoothing off)
        speed = np.zeros(N)
        speed[1:] = q_angle(heavy[1:], heavy[:-1])
        speed = gaussian_smooth(speed, 2 * fs)
        alpha = np.clip(speed / v_ref, 0, 1)[:, None]
        res = normalize(heavy * (1 - alpha) + light * alpha)
        out[:, b] = res
    return out


def slerp_many(q0, q1, t):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        return normalize(q0[None] + t[:, None] * (q1 - q0)[None])
    th = np.arccos(np.clip(d, -1, 1))
    s = np.sin(th)
    return (np.sin((1 - t) * th)[:, None] * q0 + np.sin(t * th)[:, None] * q1) / s


def hold_root(root, valid, fps, blend_s=0.25):
    """Root keeps its last seen position while the character is off screen."""
    root = root.copy()
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return root
    nb = max(1, int(round(blend_s * fps)))
    root[: idx[0]] = root[idx[0]]
    root[idx[-1] + 1:] = root[idx[-1]]
    for a, c in zip(idx[:-1], idx[1:]):
        gap = c - a
        if gap <= 1:
            continue
        t = np.clip((np.arange(1, gap) - (gap - nb)) / nb, 0, 1) if gap > nb else np.arange(1, gap) / gap
        t = t * t * (3 - 2 * t)
        root[a + 1:c] = root[a] + t[:, None] * (root[c] - root[a])
    return root


def clamp_continuous(theta, lo, hi, margin=np.radians(40)):
    """Clamp a sequence of angles to [lo, hi]. An angle outside the range goes to
    the circularly nearest bound; the choice only switches to the other bound
    when that one is nearer by `margin` (no flickering between the bounds)."""
    th = wrap(np.asarray(theta, float))
    out = np.empty_like(th)
    side = 0  # last bound used: -1 lo, +1 hi, 0 none
    for i, t in enumerate(th):
        if lo <= t <= hi:
            out[i] = t
            side = 0
            continue
        dl, dh = abs(wrap(lo - t)), abs(wrap(hi - t))
        if side == -1 and dh < dl - margin:
            side = 1
        elif side == 1 and dl < dh - margin:
            side = -1
        elif side == 0:
            side = -1 if dl <= dh else 1
        out[i] = lo if side == -1 else hi
    return out


def circular_smooth(theta, weight, sigma, iters=2, outlier=np.radians(60)):
    """Robust zero-phase smoothing of an angle sequence (weights 0..1)."""
    z = np.exp(1j * np.asarray(theta, float))
    w = np.asarray(weight, float).copy()
    if sigma <= 0 or len(z) < 3:
        return np.angle(z)
    sm = z
    for _ in range(iters):
        num = gaussian_smooth(np.stack([z.real * w, z.imag * w], 1), sigma)
        den = gaussian_smooth(w, sigma)
        sm = (num[:, 0] + 1j * num[:, 1]) / np.maximum(den, 1e-9)
        dev = np.abs(np.angle(z * np.conj(sm)))
        w = np.where(dev > outlier, w * 0.05, w)
    return np.angle(sm)


# ----------------------------------------------------------------------------- constraints
def _deg(a):
    return np.radians(a)


class Naturalizer:
    """Holds per-bone rest axes / limits for one skeleton."""

    def __init__(self, skel):
        self.s = skel
        rp = skel.rest_pos
        self.axis = {}
        for b in skel.bones:
            kids = [c for c in skel.bones if skel.parent.get(c) == b]
            # main child: follow the chain (fingers/limbs), else first child
            pref = [c for c in kids if c.replace("Lower", "").replace("Upper", "") != c or "Hand" in c or "Foot" in c
                    or any(s in c for s in ("Intermediate", "Distal", "Proximal", "Neck", "Head", "Chest", "spine"))]
            child = (pref or kids or [None])[0]
            chain_child = {
                "hips": "spine", "spine": "chest", "chest": "upperChest" if skel.has("upperChest") else "neck",
                "upperChest": "neck", "neck": "head",
            }.get(b)
            for side in ("left", "right"):
                chain_child = {
                    f"{side}UpperArm": f"{side}LowerArm", f"{side}LowerArm": f"{side}Hand",
                    f"{side}Hand": f"{side}MiddleProximal", f"{side}UpperLeg": f"{side}LowerLeg",
                    f"{side}LowerLeg": f"{side}Foot", f"{side}Foot": f"{side}Toes",
                    f"{side}Shoulder": f"{side}UpperArm",
                }.get(b, chain_child)
            if chain_child and skel.has(chain_child):
                child = chain_child
            if child is not None and np.linalg.norm(rp[child] - rp[b]) > 1e-6:
                self.axis[b] = normalize(rp[child] - rp[b])
            elif skel.parent.get(b) in self.axis and not b.endswith(("Eye", "jaw", "head")):
                self.axis[b] = self.axis[skel.parent[b]]
            else:
                self.axis[b] = Y.copy()
        self.idx = {b: i for i, b in enumerate(skel.bones)}

    # -- generic swing/twist clamp ------------------------------------------------
    def clamp_swing_twist(self, q, b, twist_lim, swing_lim):
        a = self.axis[b]
        sw, t = remove_twist(q, a)
        t = clamp_continuous(t, -twist_lim, twist_lim) if np.ndim(t) else np.clip(wrap(t), -twist_lim, twist_lim)
        ang = 2 * np.arccos(np.clip(np.abs(sw[..., 3]), 0, 1))
        sign = np.where(sw[..., 3:4] < 0, -1.0, 1.0)
        sw = sw * sign
        f = np.where(ang > swing_lim, swing_lim / np.maximum(ang, 1e-9), 1.0)
        rv = q_log(sw) * f[..., None]
        sw2 = rotvec_to_quat(rv)
        return quat_mul(sw2, q_axis_angle(a, t))

    def hinge(self, q, b, hinge_axis, lo, hi, twist_lim):
        """Project a local rotation onto a hinge (about hinge_axis) + limited twist."""
        a = self.axis[b]
        d = rotate(q, a)
        dp = d - np.outer(d @ hinge_axis, hinge_axis) if d.ndim == 2 else d - hinge_axis * (d @ hinge_axis)
        u = normalize(np.cross(hinge_axis, a))  # direction a moves to for +angle
        th = np.arctan2(dp @ u, dp @ a)
        th = clamp_continuous(th, lo, hi)
        qh = q_axis_angle(hinge_axis, th)
        # twist that remains after removing the hinge
        rest = quat_mul(qh * np.array([-1, -1, -1, 1.0]), q)
        _, t = remove_twist(rest, a)
        t = clamp_continuous(t, -twist_lim, twist_lim)
        return quat_mul(qh, q_axis_angle(a, t))

    def unflip_fingers(self, Q):
        """Finger joints bent backwards are depth-ambiguity errors (the tip seen
        on the wrong side in depth): reflect them to the matching forward bend.
        Run on the raw per-frame solve, before temporal filtering."""
        Q = Q.copy()
        for side in ("left", "right"):
            for fg in FINGERS[1:]:
                for k, seg in enumerate(FINGER_SEGS[fg]):
                    b = f"{side}{fg}{seg}"
                    if not self.s.has(b):
                        continue
                    i = self.idx[b]
                    a = self.axis[b]
                    h = normalize(np.cross(a, -Y))
                    u = normalize(np.cross(h, a))
                    q = Q[:, i]
                    d = rotate(q, a)
                    dp = d - np.outer(d @ h, h)
                    th = np.arctan2(dp @ u, dp @ a)
                    lim = _deg(-25) if k == 0 else _deg(-10)
                    bad = th < lim
                    if bad.any():
                        # rotate the bone about the hinge axis by -2*th (mirror of the bend)
                        fix = q_axis_angle(h, -2 * th[bad])
                        Q[bad, i] = quat_mul(fix, q[bad])
        return Q

    # -- full pass -----------------------------------------------------------------
    def apply(self, Q, root, fps=30.0, conf=None):
        """Q: (N,B,4) local quats (modified copy returned). conf: (N,B) per-bone
        confidence (used to weight the forearm twist filter)."""
        self.fps = fps
        from .retarget import avatar_fk
        s, I = self.s, self.idx
        Q = Q.copy()
        # 1) limbs: hinge IK (keeps wrists / ankles where they are)
        pos, rot = avatar_fk(s, Q, root)
        for side in ("left", "right"):
            for up, lo_, end, fwd in ((f"{side}UpperArm", f"{side}LowerArm", f"{side}Hand", Z),
                                      (f"{side}UpperLeg", f"{side}LowerLeg", f"{side}Foot", -Z)):
                if not all(s.has(x) for x in (up, lo_, end)):
                    continue
                if up.endswith("Leg"):
                    self._leg_no_twist(Q, pos, rot, up, lo_, end)
                else:
                    self._limb_ik(Q, pos, rot, up, lo_, end, fwd, is_leg=False)
        N_ = len(Q)
        # 2) forearm twist (pronation / supination): filtered as a circular signal
        #    (outliers and frames without a hand detection carry little weight),
        #    kept within its range without flipping; the hand keeps only its swing
        for side in ("left", "right"):
            la, hd = f"{side}LowerArm", f"{side}Hand"
            if not (s.has(la) and s.has(hd)):
                continue
            ia, ih = I[la], I[hd]
            ah, aa = self.axis[hd], self.axis[la]
            qa, qh = Q[:, ia], Q[:, ih]
            sw_h, th_ = remove_twist(qh, ah)
            sw_a, ta = remove_twist(qa, aa)
            total = wrap(ta + th_ * float(np.sign(ah @ aa) or 1))  # all twist goes to the forearm
            w = np.ones(N_) if conf is None else np.clip(conf[:, ih], 0.03, 1.0)
            total = circular_smooth(total, w, 0.12 * fps)
            sup = -1.0 if side == "left" else 1.0  # + = supination (towards palm up)
            tt = clamp_continuous(sup * total, _deg(-100), _deg(180))
            tt = circular_smooth(tt, np.ones(N_), 0.05 * fps, iters=1, outlier=np.pi)
            Q[:, ia] = quat_mul(sw_a, q_axis_angle(aa, tt * sup))
            Q[:, ih] = self.clamp_swing_twist(sw_h, hd, _deg(5), _deg(85))
        # 3) fingers
        for side in ("left", "right"):
            for fg in FINGERS:
                for k, seg in enumerate(FINGER_SEGS[fg]):
                    b = f"{side}{fg}{seg}"
                    if not s.has(b):
                        continue
                    i = I[b]
                    a = self.axis[b]
                    if fg == "Thumb":
                        Q[:, i] = self.clamp_swing_twist(Q[:, i], b, _deg(35), _deg(95))
                        continue
                    curl_axis = normalize(np.cross(a, -Y))
                    if k == 0:
                        Q[:, i] = self._knuckle(Q[:, i], a, curl_axis)
                    else:
                        Q[:, i] = self.hinge(Q[:, i], b, curl_axis, _deg(-10), _deg(115), _deg(8))
        # 4) everything else: generic cones
        lims = {"spine": (35, 45), "chest": (30, 40), "upperChest": (25, 35), "neck": (45, 50), "head": (60, 60),
                "hips": (180, 180)}
        for side in ("left", "right"):
            lims[f"{side}Foot"] = (25, 75)
            lims[f"{side}Shoulder"] = (10, 25)
            lims[f"{side}Toes"] = (5, 50)
        for b, (tw, swl) in lims.items():
            if s.has(b) and b != "hips":
                Q[:, I[b]] = self.clamp_swing_twist(Q[:, I[b]], b, _deg(tw), _deg(swl))
        # 5) soften any remaining sudden steps (constraint boundaries)
        for i in range(Q.shape[1]):
            if i != I["hips"]:
                Q[:, i] = normalize(gaussian_smooth(continuity(Q[:, i]), 0.025 * fps))
        return normalize(Q)

    def _knuckle(self, q, a, curl_axis):
        """MCP joint: curl -25..100 deg, splay +-25 deg, twist +-12 deg."""
        sw, t = remove_twist(q, a)
        t = clamp_continuous(t, _deg(-12), _deg(12))
        d = rotate(sw, a)
        splay_axis = normalize(np.cross(curl_axis, a))  # ~ palm normal
        u_curl = normalize(np.cross(curl_axis, a))
        # components of d: along a, along curl direction (u_curl), along splay direction
        u_splay = normalize(np.cross(splay_axis, a))
        curl = np.arctan2(d @ u_curl, d @ a)
        spl = np.arcsin(np.clip(d @ u_splay, -1, 1))
        curl = clamp_continuous(curl, _deg(-25), _deg(100))
        spl = np.clip(spl, _deg(-25), _deg(25))
        q2 = quat_mul(q_axis_angle(curl_axis, curl), q_axis_angle(splay_axis, spl))
        return quat_mul(q2, q_axis_angle(a, t))

    def _limb_ik(self, Q, pos, rot, up, lo_, end, fwd, is_leg):
        """Two-bone hinge IK for the arms that keeps the wrist where it is.

        The elbow is placed on the circle of positions that keep both bone
        lengths, as close as possible to where it was observed (or, when the arm
        is nearly straight and the elbow position says little, where it was in
        the previous frame). The elbow path is smoothed, then the upper arm is
        built from "points at the elbow" + "bends towards the wrist", which has
        no singularity in any arm orientation; the forearm is a pure hinge plus
        its own twist."""
        from .mathutil import rot_from_axes
        s, I = self.s, self.idx
        iu, il, ie = I[up], I[lo_], I[end]
        ip = I[s.parent[up]]
        rp = s.rest_pos
        au, al = self.axis[up], self.axis[lo_]
        hinge = normalize(np.cross(al, fwd))
        b_rest = normalize(np.cross(hinge, au))  # direction the forearm moves when the elbow flexes
        l1 = np.linalg.norm(rp[lo_] - rp[up])
        l2 = np.linalg.norm(rp[end] - rp[lo_])
        lo_lim, hi_lim = _deg(-5), _deg(150)
        n = len(Q)
        fps_ = getattr(self, "fps", 30.0)
        Wt = np.einsum("nji,nj->ni", rot[:, ip], pos[:, ie] - pos[:, iu])   # wrist, parent frame
        Ec = np.einsum("nji,nj->ni", rot[:, ip], pos[:, il] - pos[:, iu])   # observed elbow
        E = np.zeros((n, 3))
        prev_dir = None
        for k in range(n):
            w = Wt[k]
            d = float(np.clip(np.linalg.norm(w), abs(l1 - l2) + 1e-5, l1 + l2 - 1e-6))
            dn = w / max(np.linalg.norm(w), 1e-9)
            x = (l1 * l1 - l2 * l2 + d * d) / (2 * d)       # elbow distance along shoulder->wrist
            h = np.sqrt(max(l1 * l1 - x * x, 0.0))          # circle radius
            obs = Ec[k] - dn * (Ec[k] @ dn)
            bend = np.pi - np.arccos(np.clip((l1 * l1 + l2 * l2 - d * d) / (2 * l1 * l2), -1, 1))
            wgt = float(np.clip((bend - _deg(25)) / _deg(35), 0, 1))
            if prev_dir is not None:
                pd_ = prev_dir - dn * (prev_dir @ dn)
                u = wgt * normalize(obs) + (1 - wgt) * normalize(pd_) if np.linalg.norm(obs) > 1e-9 else normalize(pd_)
            else:
                u = obs
            if np.linalg.norm(u) < 1e-9:
                u = np.cross(dn, hinge)
            u = normalize(u - dn * (u @ dn))
            if bend > _deg(15):
                u = _natural_elbow(u, dn, x, h, w, l1)
            E[k] = dn * x + u * h
            prev_dir = u
        # smooth the elbow path, then put it back on the circle exactly
        E = gaussian_smooth(E, 0.04 * fps_)
        rot_prev = np.eye(3)
        for k in range(n):
            w = Wt[k]
            d = float(np.clip(np.linalg.norm(w), abs(l1 - l2) + 1e-5, l1 + l2 - 1e-6))
            dn = w / max(np.linalg.norm(w), 1e-9)
            x = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
            h = np.sqrt(max(l1 * l1 - x * x, 0.0))
            u = E[k] - dn * (E[k] @ dn)
            u = normalize(u) if np.linalg.norm(u) > 1e-9 else normalize(np.cross(dn, hinge))
            e = dn * x + u * h
            fore = w - e
            en = normalize(e)
            bdir = fore - en * (fore @ en)
            # nearly straight: the bend direction is a tiny, noisy vector -> keep
            # the previous frame's bend direction (blended by how bent the elbow is)
            bend = np.arccos(np.clip((fore @ en) / max(np.linalg.norm(fore), 1e-9), -1, 1))
            wb = float(np.clip((bend - _deg(20)) / _deg(30), 0, 1)) if k > 0 else 1.0
            prev_b = rot_prev @ b_rest
            prev_b = prev_b - en * (prev_b @ en)
            if np.linalg.norm(bdir) > 1e-9 and np.linalg.norm(prev_b) > 1e-9:
                bdir = normalize(wb * normalize(bdir) + (1 - wb) * normalize(prev_b))
            elif np.linalg.norm(bdir) <= 1e-9:
                bdir = prev_b
            Lu = rot_from_axes(au, b_rest, e, bdir)
            rot_prev = Lu
            # forearm: hinge about the elbow axis + its own twist
            dl = Lu.T @ fore / max(np.linalg.norm(fore), 1e-9)
            uh = normalize(np.cross(hinge, al))
            th = float(np.clip(np.arctan2(dl @ uh, dl @ al), lo_lim, hi_lim))
            Rh = axis_angle_to_mat(hinge, th)
            Ll = quat_to_mat(Q[k, il])
            _, tl = remove_twist(mat_to_quat(Rh.T @ Ll), al)
            Ll_new = Rh @ axis_angle_to_mat(al, float(wrap(tl)))
            Q[k, iu] = mat_to_quat(Lu)
            Q[k, il] = mat_to_quat(Ll_new)
            Q[k, ie] = mat_to_quat((rot[k, ip] @ Lu @ Ll_new).T @ rot[k, ie])


_CIRCLE = np.radians(np.arange(0, 360, 5.0))


def _natural_elbow(u, dn, x, h, w, l1, lim=np.radians(110)):
    """Among the elbow positions on the IK circle (all keep the wrist exactly in
    place), pick the one nearest to `u` whose bend direction is anatomically
    possible: the upper arm turned at most `lim` about its own axis from the
    natural "arm hanging, elbow bending forward" reference."""
    p = u
    q = np.cross(dn, p)
    E = dn[None] * x + h * (np.cos(_CIRCLE)[:, None] * p[None] + np.sin(_CIRCLE)[:, None] * q[None])
    a = E / l1
    fore = w[None] - E
    b = fore - a * np.sum(fore * a, 1, keepdims=True)
    # natural flex direction: +Z carried by the minimal rotation from "hanging down" (-Y) to a
    v0 = np.array([0.0, -1.0, 0.0])
    ax = np.cross(v0[None], a)
    sn = np.linalg.norm(ax, axis=1)
    cs = a @ v0
    k_ = ax / np.maximum(sn, 1e-9)[:, None]
    zf = np.array([0.0, 0.0, 1.0])
    nz = (zf[None] * cs[:, None] + np.cross(k_, zf[None]) * sn[:, None]
          + k_ * (k_ @ zf)[:, None] * (1 - cs)[:, None])
    nz = nz - a * np.sum(nz * a, 1, keepdims=True)
    tau = np.arctan2(np.sum(np.cross(nz, b) * a, 1), np.sum(nz * b, 1))
    straight_up = a[:, 1] > 0.85
    ok = (np.abs(tau) <= lim) | straight_up
    if ok[0] or not ok.any():
        return u
    i = np.where(ok)[0]
    best = i[np.argmin(np.abs(np.angle(np.exp(1j * _CIRCLE[i]))))]
    return np.cos(_CIRCLE[best]) * p + np.sin(_CIRCLE[best]) * q


def _leg_no_twist(self, Q, pos, rot, up, lo_, end):
    """Legs: the thigh never twists (it only swings to point at the knee); any
    twist the leg needs to aim the foot happens at the knee (shin twist), within
    +-45 deg and smoothed over time. Knee and ankle positions are kept exactly."""
    s, I = self.s, self.idx
    iu, il, ie = I[up], I[lo_], I[end]
    ip = I[s.parent[up]]
    au, al = self.axis[up], self.axis[lo_]
    n = len(Q)
    Lu_all = np.zeros((n, 3, 3))
    Lsw_all = np.zeros((n, 3, 3))
    tw = np.zeros(n)
    for k in range(n):
        Gp = rot[k, ip]
        knee = Gp.T @ (pos[k, il] - pos[k, iu])          # knee direction in the hips frame
        Lu = _rot_between(au, normalize(knee))            # swing only: zero thigh twist
        Gu = Gp @ Lu
        ank = Gu.T @ (pos[k, ie] - pos[k, il])            # ankle direction in the thigh frame
        Lsw = _rot_between(al, normalize(ank))
        # twist about the shin that best keeps the foot's world orientation
        Gl0 = Gu @ Lsw
        foot_local = Gl0.T @ rot[k, ie]
        _, t = remove_twist(mat_to_quat(foot_local), al)
        tw[k] = t
        Lu_all[k], Lsw_all[k] = Lu, Lsw
    # a foot that seems turned more than 90 deg about the shin is a heel/toe
    # confusion of the detector: read it the other way round
    tw = wrap(tw)
    flip = np.abs(tw) > _deg(90)
    tw = np.where(flip, wrap(tw + np.pi), tw)
    tw = circular_smooth(tw, np.where(flip, 0.2, 1.0), 0.05 * getattr(self, "fps", 30.0))
    tw = clamp_continuous(tw, _deg(-45), _deg(45))
    for k in range(n):
        Gp = rot[k, ip]
        Ll = Lsw_all[k] @ axis_angle_to_mat(al, tw[k])
        Q[k, iu] = mat_to_quat(Lu_all[k])
        Q[k, il] = mat_to_quat(Ll)
        Q[k, ie] = mat_to_quat((Gp @ Lu_all[k] @ Ll).T @ rot[k, ie])


Naturalizer._leg_no_twist = _leg_no_twist


def _legs_only(self, Q, root):
    """Re-apply the no-thigh-twist rule (e.g. after the feet were pinned)."""
    from .retarget import avatar_fk
    Q = Q.copy()
    pos, rot = avatar_fk(self.s, Q, root)
    for side in ("left", "right"):
        self._leg_no_twist(Q, pos, rot, f"{side}UpperLeg", f"{side}LowerLeg", f"{side}Foot")
    return Q


Naturalizer.legs_only = _legs_only


def _rot_between(u, v):
    from .mathutil import rot_between
    return rot_between(u, v)


def rotate(q, v):
    m = quat_to_mat(q)
    return m @ v if m.ndim == 2 else np.einsum("nij,j->ni", m, v)


def rotvec_to_quat(rv):
    a = np.linalg.norm(rv, axis=-1)
    ax = rv / np.maximum(a, 1e-12)[..., None]
    return q_axis_angle(ax, a)


__all__ = ["filter_rotations", "Naturalizer", "hold_root", "mat_to_quat"]

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
                self._limb_ik(Q, pos, rot, up, lo_, end, fwd, is_leg=(up.endswith("Leg")))
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
        """Two-bone hinge IK that keeps the end joint where it is.

        The thigh / upper arm twist is chosen so the elbow / knee bends as a
        hinge towards `fwd` (forward for elbows, backward for knees); when the
        limb is nearly straight the previous twist is kept (no flips)."""
        s, I = self.s, self.idx
        iu, il, ie = I[up], I[lo_], I[end]
        ip = I[s.parent[up]]
        rp = s.rest_pos
        au, al = self.axis[up], self.axis[lo_]
        hinge = normalize(np.cross(al, fwd))
        l1 = np.linalg.norm(rp[lo_] - rp[up])
        l2 = np.linalg.norm(rp[end] - rp[lo_])
        u_h = normalize(np.cross(hinge, au))
        rest_bend = float(np.arctan2(al @ u_h, al @ au))
        lo_lim, hi_lim = (_deg(-2), _deg(155)) if is_leg else (_deg(-5), _deg(150))
        tw_lim = _deg(30) if is_leg else _deg(200)
        n = len(Q)
        Rs_all = np.zeros((n, 3, 3))
        dn_all = np.zeros((n, 3))
        Rh_all = np.zeros((n, 3, 3))
        angs = np.zeros(n)
        wts = np.zeros(n)
        for k in range(n):
            Gp = rot[k, ip]
            S, W = pos[k, iu], pos[k, ie]
            dv = Gp.T @ (W - S)  # target in the parent's frame
            d = float(np.clip(np.linalg.norm(dv), abs(l1 - l2) + 1e-5, l1 + l2 - 1e-6))
            dn = dv / max(np.linalg.norm(dv), 1e-9)
            cos_b = (l1 * l1 + l2 * l2 - d * d) / (2 * l1 * l2)
            phi = np.pi - np.arccos(np.clip(cos_b, -1, 1))  # angle between the two bones
            # the rest pose need not be straight: hinge angle = phi - rest bend
            bend = float(np.clip(phi - rest_bend, lo_lim, hi_lim))
            Lu = quat_to_mat(Q[k, iu])
            Rh = axis_angle_to_mat(hinge, bend)
            end_local = au * l1 + Rh @ al * l2
            Rs = _rot_between(normalize(end_local), dn)
            # twist about the shoulder->wrist line: put the elbow / knee where it
            # currently is (pole); nearly straight -> keep the current hinge direction
            pp = lambda v: v - dn * (v @ dn)  # noqa: E731
            e_new, e_cur = pp(Rs @ au), pp(Gp.T @ (pos[k, il] - S))
            h_new, h_cur = pp(Rs @ hinge), pp(Lu @ hinge)

            def signed(u, v):
                return np.arctan2(np.cross(u, v) @ dn, u @ v)
            a_pole = signed(e_new, e_cur) if np.linalg.norm(e_cur) > 1e-6 else 0.0
            a_hinge = signed(h_new, h_cur) if np.linalg.norm(h_cur) > 1e-6 else a_pole
            w = float(np.clip((phi - _deg(8)) / _deg(14), 0, 1))
            angs[k] = np.angle(w * np.exp(1j * a_pole) + (1 - w) * np.exp(1j * a_hinge))
            wts[k] = 0.15 + w
            Rs_all[k], dn_all[k], Rh_all[k] = Rs, dn, Rh
        # the twist about the limb axis is smoothed over time (no flips near straight)
        angs = circular_smooth(angs, wts, 0.05 * getattr(self, "fps", 30.0))
        for k in range(n):
            Gp = rot[k, ip]
            Lu_new = axis_angle_to_mat(dn_all[k], angs[k]) @ Rs_all[k]
            Rh = Rh_all[k]
            Ll = quat_to_mat(Q[k, il])
            _, tl = remove_twist(mat_to_quat(Rh.T @ Ll), al)
            tl = float(np.clip(wrap(tl), -tw_lim, tw_lim))
            Ll_new = Rh @ axis_angle_to_mat(al, tl)
            Q[k, iu] = mat_to_quat(Lu_new)
            Q[k, il] = mat_to_quat(Ll_new)
            # keep the hand / foot world orientation
            Gl_new = Gp @ Lu_new @ Ll_new
            Q[k, ie] = mat_to_quat(Gl_new.T @ rot[k, ie])


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

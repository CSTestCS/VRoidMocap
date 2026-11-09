"""Pass 2b: landmarks + pass-1 camera data -> VRM bone rotations, root motion and
a cloned camera.

Pipeline per frame
  1. orient the model's 3D landmarks into the real camera (crop ray correction),
  2. solve the character's position relative to the camera (fixed-rotation PnP)
     with the shot's focal length from pass 1,
  3. snap every joint onto the camera ray through its 2D detection while keeping
     bone lengths fixed (2D detections are far more precise than the model's
     depth guess; the depth guess only picks between the two length-preserving
     solutions),
  4. attach the hands (hand model) to the wrists the same way,
  5. rotate into world space with the pass-1 camera rotation (gravity aligned,
     yaw-continuous across cuts).
Then over the whole clip: outlier rejection + zero-phase smoothing, bone
rotations from joint positions, gap filling, foot contact detection, root
motion integrated from planted feet on the *avatar's* own skeleton (so feet do
not slide), and finally the cloned camera expressed relative to the avatar.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import camera as cam
from .analysis import world_cameras
from .detect import (L_ANK, L_EAR, L_EL, L_EYE, L_FOOT, L_HEEL, L_HIP, L_INDEX, L_KNEE,
                     L_PINKY, L_SH, L_THUMB, L_WR, MOUTH_L, MOUTH_R, NOSE, R_ANK, R_EAR,
                     R_EL, R_EYE, R_FOOT, R_HEEL, R_HIP, R_INDEX, R_KNEE, R_PINKY, R_SH,
                     R_THUMB, R_WR)
from .mathutil import (gaussian_smooth, interp_gaps, mat_to_quat, normalize, quat_continuity,
                       quat_to_mat, rot_between, rot_from_axes, slerp)
from .vrm import FINGER_SEGS, FINGERS

X, Y, Z = np.eye(3)

# body "bones" between MediaPipe landmarks used for the length constraints,
# listed parent-first so each child is solved after its parent.
BODY_TREE = [
    (L_HIP, L_KNEE), (L_KNEE, L_ANK), (L_ANK, L_HEEL), (L_ANK, L_FOOT),
    (R_HIP, R_KNEE), (R_KNEE, R_ANK), (R_ANK, R_HEEL), (R_ANK, R_FOOT),
    (L_SH, L_EL), (L_EL, L_WR), (L_WR, L_INDEX), (L_WR, L_PINKY), (L_WR, L_THUMB),
    (R_SH, R_EL), (R_EL, R_WR), (R_WR, R_INDEX), (R_WR, R_PINKY), (R_WR, R_THUMB),
]
TORSO = [L_SH, R_SH, L_HIP, R_HIP]
HAND_TREE = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10),
             (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18),
             (18, 19), (19, 20)]
HAND_FINGER_IDX = {"Thumb": [1, 2, 3, 4], "Index": [5, 6, 7, 8], "Middle": [9, 10, 11, 12],
                   "Ring": [13, 14, 15, 16], "Little": [17, 18, 19, 20]}


# ----------------------------------------------------------------------------- per-frame geometry
def snap_chain(P, rays, w, tree, lengths, fixed=()):
    """Move joints onto their camera rays (weight w) keeping bone lengths.

    P: (n,3) initial 3D points in camera coords. rays: (n,3) unit rays (or nan).
    Returns new points."""
    P = P.copy()
    for i in range(len(P)):
        if i in fixed or not np.isfinite(rays[i, 0]) or w[i] <= 0:
            continue
        on = rays[i] * max(np.dot(P[i], rays[i]), 1e-3)
        P[i] = P[i] + w[i] * (on - P[i])
    for (p, c) in tree:
        L = lengths.get((p, c))
        if L is None or not np.isfinite(rays[c, 0]):
            continue
        r = rays[c]
        # |t r - P_p| = L  ->  t^2 - 2 t (r.P_p) + |P_p|^2 - L^2 = 0
        bq = float(np.dot(r, P[p]))
        disc = bq * bq - (float(np.dot(P[p], P[p])) - L * L)
        t0 = float(np.dot(P[c], r))
        if disc >= 0:
            sq = np.sqrt(disc)
            cands = [bq - sq, bq + sq]
            t = min(cands, key=lambda x: abs(x - t0))
            target = r * t
        else:
            target = r * bq
        d = target - P[p]
        n = np.linalg.norm(d)
        if n > 1e-9:
            target = P[p] + d / n * L
        P[c] = P[c] + w[c] * (target - P[c])
        # restore exact length after blending
        d = P[c] - P[p]
        n = np.linalg.norm(d)
        if n > 1e-9:
            P[c] = P[p] + d / n * L
    return P


def rays_for(lm2d, f, cx, cy):
    return normalize(np.c_[(lm2d[:, 0] - cx) / f, (lm2d[:, 1] - cy) / f, np.ones(len(lm2d))])


def median_lengths(points, tree, valid):
    out = {}
    for (p, c) in tree:
        d = np.linalg.norm(points[valid][:, c] - points[valid][:, p], axis=-1)
        d = d[np.isfinite(d) & (d > 1e-4)]
        if len(d):
            out[(p, c)] = float(np.median(d))
    return out


# ----------------------------------------------------------------------------- rotations
def rd_pos(skel, b, fallback):
    return skel.rest_pos[b] if skel.has(b) else np.array(fallback, float)


class RigSolver:
    """Computes global bone rotations (normalized VRM frame) from world joints."""

    def __init__(self, skel):
        self.s = skel
        self.rest = skel.rest_pos

        def rd(a, b, fallback):
            if skel.has(a) and skel.has(b):
                return normalize(self.rest[b] - self.rest[a])
            return normalize(np.array(fallback, float))
        self.rest_dir = {}
        for side, sx in (("left", 1), ("right", -1)):
            self.rest_dir[f"{side}UpperArm"] = rd(f"{side}UpperArm", f"{side}LowerArm", [sx, 0, 0])
            self.rest_dir[f"{side}LowerArm"] = rd(f"{side}LowerArm", f"{side}Hand", [sx, 0, 0])
            self.rest_dir[f"{side}Hand"] = rd(f"{side}Hand", f"{side}MiddleProximal", [sx, 0, 0])
            self.rest_dir[f"{side}UpperLeg"] = rd(f"{side}UpperLeg", f"{side}LowerLeg", [0, -1, 0])
            self.rest_dir[f"{side}LowerLeg"] = rd(f"{side}LowerLeg", f"{side}Foot", [0, -1, 0])
            for fg in FINGERS:
                segs = FINGER_SEGS[fg]
                names = [f"{side}{fg}{s}" for s in segs]
                for k, nm in enumerate(names):
                    if k + 1 < len(names):
                        self.rest_dir[nm] = rd(nm, names[k + 1], [sx, 0, 0])
                    else:
                        self.rest_dir[nm] = self.rest_dir.get(names[k - 1], normalize(np.array([sx, 0, 0.0])))
        # rest-pose torso frame measured on the avatar (the MediaPipe hip / shoulder
        # landmarks correspond to the upper-leg / upper-arm joints)
        hl, hr = rd_pos(skel, "leftUpperLeg", [0.1, 1, 0]), rd_pos(skel, "rightUpperLeg", [-0.1, 1, 0])
        sl, sr = rd_pos(skel, "leftUpperArm", [0.2, 1.4, 0]), rd_pos(skel, "rightUpperArm", [-0.2, 1.4, 0])
        kl, kr = rd_pos(skel, "leftLowerLeg", [0.1, 0.5, 0]), rd_pos(skel, "rightLowerLeg", [-0.1, 0.5, 0])
        self.rest_hip_axis = normalize(hl - hr)
        self.rest_sh_axis = normalize(sl - sr)
        self.rest_torso_up = normalize((sl + sr) / 2 - (hl + hr) / 2)
        self.rest_thigh_up = normalize(-((kl - hl) + (kr - hr)))
        self.rest_hips_b = normalize(0.7 * self.rest_torso_up + 0.3 * self.rest_thigh_up)
        # rest palm normal / knuckle line measured on the avatar, defined exactly like
        # the observed ones so there is no constant offset
        self.rest_palm, self.rest_lat = {}, {}
        for side, sgn in (("left", -1.0), ("right", 1.0)):
            h = rd_pos(skel, f"{side}Hand", [0, 0, 0])
            ip = rd_pos(skel, f"{side}IndexProximal", h + [0.08 * -sgn, 0, 0.02])
            lp = rd_pos(skel, f"{side}LittleProximal", h + [0.07 * -sgn, 0, -0.03])
            self.rest_palm[side] = normalize(sgn * np.cross(ip - h, lp - h))
            self.rest_lat[side] = normalize(ip - lp)
        self.prev = {}

    def reset_continuity(self):
        self.prev = {}

    def _twist_axis(self, key, a, b_obs, weight, b_rest):
        """Secondary axis with continuity fallback when the observation is weak."""
        a = normalize(a)
        cand = None
        if b_obs is not None:
            bo = b_obs - a * np.dot(a, b_obs)
            if np.linalg.norm(bo) > 1e-6:
                cand = normalize(bo)
        prev = self.prev.get(key)
        if prev is not None:
            pb = prev @ b_rest
            pb = pb - a * np.dot(a, pb)
            pb = normalize(pb) if np.linalg.norm(pb) > 1e-6 else None
        else:
            pb = None
        if cand is None:
            return pb
        if pb is None or weight >= 1:
            return cand
        return normalize(weight * cand + (1 - weight) * pb)

    def solve(self, J, hands):
        """J: dict of world joint positions (MediaPipe names). hands: [left, right]
        arrays (21,3) of world hand points or None. Returns {bone: 3x3 global}."""
        G = {}
        mid_hip = (J["hipL"] + J["hipR"]) / 2
        mid_sh = (J["shL"] + J["shR"]) / 2
        torso_up = normalize(mid_sh - mid_hip)
        thigh_up = normalize(-((J["knL"] - J["hipL"]) + (J["knR"] - J["hipR"])))
        hips_b = normalize(0.7 * torso_up + 0.3 * thigh_up)
        G["hips"] = rot_from_axes(self.rest_hip_axis, self.rest_hips_b, J["hipL"] - J["hipR"], hips_b)
        Rct = rot_from_axes(self.rest_sh_axis, self.rest_torso_up, J["shL"] - J["shR"], torso_up)
        rel = G["hips"].T @ Rct
        chain = [b for b in ("spine", "chest", "upperChest") if self.s.has(b)]
        fr = {1: [1.0], 2: [0.5, 1.0], 3: [0.4, 0.75, 1.0]}[len(chain)] if chain else []
        q_rel = mat_to_quat(rel)
        ident = np.array([0, 0, 0, 1.0])
        for b, t in zip(chain, fr):
            G[b] = G["hips"] @ quat_to_mat(slerp(ident, q_rel, t))
        top = G[chain[-1]] if chain else G["hips"]

        # head
        ear_ax = J["earL"] - J["earR"]
        eye_ax = J["eyeL"] - J["eyeR"]
        ax = normalize(normalize(ear_ax) + 0.5 * normalize(eye_ax))
        mid_ear = (J["earL"] + J["earR"]) / 2
        mid_eye = (J["eyeL"] + J["eyeR"]) / 2
        mid_mouth = (J["mouthL"] + J["mouthR"]) / 2
        fwd = J["nose"] - mid_ear
        up1 = normalize(mid_eye - mid_mouth)
        up2 = normalize(np.cross(normalize(fwd), ax))
        Rh = rot_from_axes(X, Y, ax, normalize(up1 + up2))
        relh = top.T @ Rh
        qh = mat_to_quat(relh)
        if self.s.has("neck"):
            G["neck"] = top @ quat_to_mat(slerp(ident, qh, 0.35))
        if self.s.has("head"):
            G["head"] = Rh

        for side, s_ in (("left", "L"), ("right", "R")):
            sgn = -1.0 if side == "left" else 1.0
            sh, el, wr = J["sh" + s_], J["el" + s_], J["wr" + s_]
            ua = el - sh
            fa = wr - el
            bend = np.arccos(np.clip(np.dot(normalize(ua), normalize(fa)), -1, 1))
            wb = float(np.clip((bend - np.radians(12)) / np.radians(25), 0, 1))
            hp = hands[0 if side == "left" else 1]
            if hp is not None:
                palm_n = normalize(sgn * np.cross(hp[5] - hp[0], hp[17] - hp[0]))
                hand_dir = hp[9] - hp[0]
            else:
                palm_n = normalize(sgn * np.cross(J["index" + s_] - wr, J["pinky" + s_] - wr))
                hand_dir = (J["index" + s_] + J["pinky" + s_]) / 2 - wr
            # upper arm: the elbow bend plane gives its twist (forearm offset = rest +Z);
            # when the arm is nearly straight fall back to continuity
            k = side + "UpperArm"
            nua = normalize(ua)
            bo = fa - nua * np.dot(nua, fa)
            b = self._twist_axis(k, ua, bo, wb, Z)
            if b is None:
                b = self._twist_axis(k, ua, G["hips"] @ Z, 1.0, Z)
            G[k] = rot_from_axes(self.rest_dir[k], Z, ua, b)
            # forearm carries the pronation/supination: palm normal (rest -Y)
            rpalm, rlat = self.rest_palm[side], self.rest_lat[side]
            k = side + "LowerArm"
            G[k] = rot_from_axes(self.rest_dir[k], rpalm, fa, self._twist_axis(k, fa, palm_n, 0.85, rpalm))
            k = side + "Hand"
            G[k] = rot_from_axes(self.rest_dir[k], rpalm, hand_dir, palm_n)
            if hp is not None:
                lat = normalize(hp[5] - hp[17])
                for fg in FINGERS:
                    idx = HAND_FINGER_IDX[fg]
                    segv = [hp[idx[k + 1]] - hp[idx[k]] for k in range(3)]
                    for si_, seg in enumerate(FINGER_SEGS[fg]):
                        bn = f"{side}{fg}{seg}"
                        if not self.s.has(bn):
                            continue
                        a_obs = segv[si_]
                        if fg == "Thumb":
                            G[bn] = rot_from_axes(self.rest_dir[bn], rpalm, a_obs, palm_n)
                            continue
                        # the finger's own flexion axis (exact when the joint is bent),
                        # blended towards the knuckle line when the finger is straight
                        # rest flexion axis: the VRM convention (fingers curl about Z),
                        # carried through the solved hand rotation for straight fingers
                        rd_ = self.rest_dir[bn]
                        rest_b = Z - rd_ * np.dot(Z, rd_)
                        b = G[side + "Hand"] @ rest_b
                        j = min(si_, 1)
                        hinge = np.cross(segv[j], segv[j + 1])
                        bend = np.arctan2(np.linalg.norm(hinge), np.dot(segv[j], segv[j + 1]))
                        if np.linalg.norm(hinge) > 1e-9 and np.linalg.norm(b) > 1e-9:
                            hinge = normalize(hinge)
                            if np.dot(hinge, b) < 0:
                                hinge = -hinge
                            wgt = float(np.clip((bend - np.radians(8)) / np.radians(20), 0, 1))
                            b = normalize(wgt * hinge + (1 - wgt) * normalize(b))
                        G[bn] = rot_from_axes(self.rest_dir[bn], rest_b, a_obs, b)

            # legs
            hip, kn, an = J["hip" + s_], J["kn" + s_], J["an" + s_]
            heel, toe = J["heel" + s_], J["toe" + s_]
            th = kn - hip
            sh_ = an - kn
            foot_f = toe - heel
            kb = np.arccos(np.clip(np.dot(normalize(th), normalize(sh_)), -1, 1))
            wk = float(np.clip((kb - np.radians(10)) / np.radians(25), 0, 1))
            knee_dir = -(sh_ - normalize(th) * np.dot(normalize(th), sh_))
            ff = foot_f - normalize(th) * np.dot(normalize(th), foot_f)
            if np.linalg.norm(knee_dir) > 1e-6 and np.linalg.norm(ff) > 1e-6:
                b_obs = normalize(wk * normalize(knee_dir) + (1 - wk) * normalize(ff))
            else:
                b_obs = ff
            k = side + "UpperLeg"
            G[k] = rot_from_axes(self.rest_dir[k], Z, th, self._twist_axis(k, th, b_obs, 1.0, Z))
            k = side + "LowerLeg"
            G[k] = rot_from_axes(self.rest_dir[k], Z, sh_, self._twist_axis(k, sh_, foot_f, 1.0, Z))
            # foot: heel->toe gives pitch/yaw, the ankle above that line gives roll
            k = side + "Foot"
            G[k] = rot_from_axes(Z, Y, foot_f, an - heel)
        self.prev = {k: v for k, v in G.items() if k.endswith(("UpperArm", "LowerArm", "UpperLeg", "LowerLeg"))}
        return G

    def locals_from_globals(self, G):
        """Local rotations for every humanoid bone (missing ones inherit the parent)."""
        L, full = {}, {}
        for b in self.s.bones:
            p = self.s.parent[b]
            pg = full[p] if p is not None else np.eye(3)
            g = G.get(b, pg)
            full[b] = g
            L[b] = pg.T @ g
        return L, full


# ----------------------------------------------------------------------------- avatar FK
def avatar_fk(skel, Q, root):
    """Q: (N,B,4) local quats in skel.bones order; root: (N,3) hips position.
    Returns (pos (N,B,3), rot (N,B,3,3))."""
    N, B = Q.shape[:2]
    R = quat_to_mat(Q.reshape(-1, 4)).reshape(N, B, 3, 3)
    pos = np.zeros((N, B, 3))
    rot = np.zeros((N, B, 3, 3))
    idx = {b: i for i, b in enumerate(skel.bones)}
    for i, b in enumerate(skel.bones):
        p = skel.parent[b]
        if p is None:
            rot[:, i] = R[:, i]
            pos[:, i] = root
        else:
            j = idx[p]
            rot[:, i] = rot[:, j] @ R[:, i]
            off = skel.rest_pos[b] - skel.rest_pos[p]
            pos[:, i] = pos[:, j] + np.einsum("nij,j->ni", rot[:, j], off)
    return pos, rot


# ----------------------------------------------------------------------------- main
def solve_animation(capture, workdir, char_id, skel, smoothing=1.0, progress=None, foot_lock=True):
    progress = progress or (lambda *a: None)
    with open(os.path.join(workdir, "analysis.json")) as f:
        A = json.load(f)
    npz = np.load(os.path.join(workdir, "analysis.npz"))
    N, fps, W, H = A["frames"], A["fps"], A["width"], A["height"]
    cx, cy = W / 2, H / 2
    focal = npz["focal"].astype(np.float64)
    shot_of = npz["shot_of"]
    shots = [dict(s) for s in A["shots"]]
    for i, s in enumerate(shots):
        s["align"] = npz["shot_align"][i]

    lm2d = capture["lm2d"].astype(np.float64)
    vis = capture["vis"].astype(np.float64)
    world = capture["world"].astype(np.float64)
    cropc = capture["cropc"].astype(np.float64)
    h2d, hw, hs, hc = (capture[k].astype(np.float64) for k in ("h2d", "hw", "hs", "hc"))
    valid = capture["tracked"] & (vis.mean(1) > 0.35) & (np.abs(world).sum((1, 2)) > 0)
    hvalid = (hs >= 0.5) & valid[:, None]

    # ---- 1. orient landmarks into the real camera; translation
    Wc = np.zeros_like(world)
    T = np.full((N, 3), np.nan)
    R_rel = quat_to_mat(npz["cam_q"].astype(np.float64))
    for f in np.where(valid)[0]:
        Rray = cam.ray_rotation(focal[f], cx, cy, *cropc[f])
        Wc[f] = world[f] @ Rray.T
        t = cam.solve_translation(Wc[f], lm2d[f], vis[f] ** 2, focal[f], cx, cy)
        if t is None:
            valid[f] = False
        else:
            T[f] = t
    if valid.sum() < 2:
        raise RuntimeError("The selected character was not tracked well enough to animate.")
    progress(0.05, "Solving camera continuity")
    # yaw continuity across cuts using THIS character
    yaw_at = {}
    for f in np.where(valid)[0]:
        Rw0 = shots[shot_of[f]]["align"] @ R_rel[f]
        yaw_at[f] = cam.hips_yaw(Wc[f], Rw0)
    R_w = world_cameras(npz["cam_q"].astype(np.float64), shot_of, shots, [], {}, None, focal, W, H, facing=yaw_at)

    # ---- 2. bone lengths (model metric) and snapping onto rays
    blen = median_lengths(Wc, BODY_TREE, valid)
    Hc = np.zeros((N, 2, 21, 3))
    for side in (0, 1):
        for f in np.where(hvalid[:, side])[0]:
            Rr = cam.ray_rotation(focal[f], cx, cy, *hc[f, side])
            Hc[f, side] = hw[f, side] @ Rr.T
    hlen = [median_lengths(Hc[:, s], HAND_TREE, hvalid[:, s]) if hvalid[:, s].sum() > 3 else {} for s in (0, 1)]
    # the hand model's metric scale differs from the body model's: match forearm proportions
    P = np.full((N, 33, 3), np.nan)
    Ph = np.full((N, 2, 21, 3), np.nan)
    for f in np.where(valid)[0]:
        p2 = lm2d[f].copy()
        w = np.clip((vis[f] - 0.3) / 0.3, 0, 1)
        for side, wi in ((0, L_WR), (1, R_WR)):
            if hvalid[f, side]:
                p2[wi] = h2d[f, side, 0]
                w[wi] = 1.0
        rays = rays_for(p2, focal[f], cx, cy)
        P0 = Wc[f] + T[f]
        P[f] = snap_chain(P0, rays, w, BODY_TREE, blen)
        for side, wi in ((0, L_WR), (1, R_WR)):
            if not hvalid[f, side] or not hlen[side]:
                continue
            # the hand model's metric size need not agree with the body's depth:
            # rescale the hand so its projection is as large as the detected hand
            rel = Hc[f, side] - Hc[f, side, 0]
            z = max(P[f, wi, 2], 0.1)
            proj = np.linalg.norm(rel[:, :2] * focal[f] / z, axis=1)[1:]
            img = np.linalg.norm(h2d[f, side, 1:] - h2d[f, side, 0], axis=1)
            k = float(np.clip(np.median(img) / max(np.median(proj), 1e-6), 0.5, 2.0))
            hp = rel * k + P[f, wi]
            hr = rays_for(h2d[f, side], focal[f], cx, cy)
            ww = np.full(21, 0.9)
            ww[0] = 0
            Ph[f, side] = snap_chain(hp, hr, ww, HAND_TREE, {e: L * k for e, L in hlen[side].items()}, fixed=(0,))
        if f % 500 == 0:
            progress(0.05 + 0.25 * f / N, "Reconstructing 3D joints")

    # ---- 3. temporal filtering.
    # The 2D detections are precise frame by frame; the noise is in the depth
    # estimate. So depth along each camera ray is smoothed strongly while the
    # on-screen position only gets a light de-jitter.
    fs = fps / 30.0
    sig_depth = (1.0 + 2.0 * max(0.0, smoothing)) * fs
    sig_2d = 0.5 * max(0.0, smoothing) * fs
    segs = runs(valid, shot_of)
    for (s, e) in segs:
        P[s:e] = ray_depth_smooth(P[s:e], sig_depth, sig_2d)
        for side in (0, 1):
            m = hvalid[s:e, side]
            if m.sum() >= 2:
                Ph[s:e, side] = ray_depth_smooth(Ph[s:e, side], sig_depth, sig_2d * 0.6, m)
    hipc = (P[:, L_HIP] + P[:, R_HIP]) / 2
    Pw = np.einsum("nij,nkj->nki", R_w, P - hipc[:, None])
    Phw = np.einsum("nij,nskj->nski", R_w, Ph - P[:, [L_WR, R_WR]][:, :, None])  # relative to wrist
    sig = 0.25 * max(0.0, smoothing) * fs
    progress(0.35, "Computing bone rotations")

    # ---- 4. rotations
    solver = RigSolver(skel)
    B = len(skel.bones)
    Q = np.zeros((N, B, 4))
    Q[..., 3] = 1
    qvalid = np.zeros(N, bool)
    finger_valid = np.zeros((N, 2), bool)
    fingers = [[b for b in skel.bones if b.startswith(sd) and any(fg in b for fg in FINGERS)] for sd in ("left", "right")]
    bidx = {b: i for i, b in enumerate(skel.bones)}
    names = {"hipL": L_HIP, "hipR": R_HIP, "shL": L_SH, "shR": R_SH, "elL": L_EL, "elR": R_EL,
             "wrL": L_WR, "wrR": R_WR, "knL": L_KNEE, "knR": R_KNEE, "anL": L_ANK, "anR": R_ANK,
             "heelL": L_HEEL, "heelR": R_HEEL, "toeL": L_FOOT, "toeR": R_FOOT, "nose": NOSE,
             "earL": L_EAR, "earR": R_EAR, "eyeL": L_EYE, "eyeR": R_EYE, "mouthL": MOUTH_L,
             "mouthR": MOUTH_R, "indexL": L_INDEX, "indexR": R_INDEX, "pinkyL": L_PINKY,
             "pinkyR": R_PINKY, "thumbL": L_THUMB, "thumbR": R_THUMB}
    prev_f = -10
    for f in np.where(valid)[0]:
        if f != prev_f + 1:
            solver.reset_continuity()
        prev_f = f
        J = {k: Pw[f, i] for k, i in names.items()}
        hands = []
        for side in (0, 1):
            if hvalid[f, side] and np.isfinite(Phw[f, side]).all():
                hands.append(Phw[f, side] + J["wrL" if side == 0 else "wrR"])
            else:
                hands.append(None)
        G = solver.solve(J, hands)
        L, _ = solver.locals_from_globals(G)
        for b, m in L.items():
            Q[f, bidx[b]] = mat_to_quat(m)
        qvalid[f] = True
        finger_valid[f] = [hands[0] is not None, hands[1] is not None]

    # head pitch calibration: remove the constant offset between the face
    # landmarks' "up" and the skeleton's neutral head
    if skel.has("head"):
        hi = bidx["head"]
        qh = Q[qvalid, hi]
        if len(qh) > 10:
            R = quat_to_mat(qh)
            pitch = np.arctan2(R[:, 2, 1], R[:, 1, 1])  # rotation about X
            off = float(np.clip(np.median(pitch), -np.radians(25), np.radians(25)))
            corr = quat_to_mat(np.array([np.sin(-off / 2), 0, 0, np.cos(off / 2)]))
            Q[qvalid, hi] = mat_to_quat(R @ corr)

    # ---- 5. temporal: continuity, fill gaps, light smoothing in rotation space
    progress(0.55, "Filling gaps and smoothing")
    for i, b in enumerate(skel.bones):
        is_finger = any(b in fl for fl in fingers)
        if is_finger:
            side = 0 if b.startswith("left") else 1
            m = finger_valid[:, side]
        else:
            m = qvalid
        Q[:, i] = fill_and_smooth_quats(Q[:, i], m, sig * (1.0 if not is_finger else 0.8), max_gap=int(2.0 * fps))

    # ---- 6. root motion on the avatar
    progress(0.7, "Solving root motion and foot contacts")
    leg_actor = np.median([blen.get((L_HIP, L_KNEE), 0.4) + blen.get((L_KNEE, L_ANK), 0.4),
                           blen.get((R_HIP, R_KNEE), 0.4) + blen.get((R_KNEE, R_ANK), 0.4)])
    rp = skel.rest_pos
    leg_avatar = (np.linalg.norm(rp["leftUpperLeg"] - rp["leftLowerLeg"]) +
                  np.linalg.norm(rp["leftLowerLeg"] - rp["leftFoot"]))
    scale = float(leg_avatar / max(leg_actor, 1e-3))
    # actor root in world assuming a camera that only rotates within a shot
    root_actor = np.einsum("nij,nj->ni", R_w, np.nan_to_num(T))
    root_actor = interp_gaps(root_actor, valid)
    root, contacts = root_motion(skel, Q, root_actor * scale, valid, shot_of, fps, Pw, scale)
    if foot_lock:
        Q = foot_lock_ik(skel, Q, root, contacts, fps)

    # ---- 7. cloned camera relative to the avatar
    Ts = T.copy()
    Ts = interp_gaps(np.nan_to_num(Ts), valid)
    for (s, e) in runs(np.ones(N, bool), shot_of):
        Ts[s:e] = gaussian_smooth(Ts[s:e], 2.0 * fps / 30.0, valid=valid[s:e])
    # T is the camera->hip-joint-centre vector; place the camera so the avatar's
    # hip-joint centre (not the hips bone) sits exactly where the actor's does
    hi = skel.bones.index("hips")
    Rh = quat_to_mat(Q[:, hi])
    off = (skel.rest_pos["leftUpperLeg"] + skel.rest_pos["rightUpperLeg"]) / 2 - skel.rest_pos["hips"]
    hipc_avatar = root + np.einsum("nij,j->ni", Rh, off)
    cam_pos = hipc_avatar - scale * np.einsum("nij,nj->ni", R_w, Ts)
    cam_quat = mat_to_quat(R_w @ cam.D)
    cam_vfov = np.array([cam.vfov_from_focal(fv, H) for fv in focal])

    progress(0.95, "Packing animation")
    anim = {
        "fps": fps, "frames": N, "width": W, "height": H, "character": char_id,
        "bones": skel.bones, "quats": Q.astype(np.float32), "root": root.astype(np.float32),
        "valid": valid, "contacts": contacts, "finger_valid": finger_valid,
        "cam_pos": cam_pos.astype(np.float32), "cam_quat": cam_quat.astype(np.float32),
        "cam_vfov": cam_vfov.astype(np.float32), "shot_of": shot_of.astype(np.int32),
        "lm2d": lm2d.astype(np.float32), "vis": vis.astype(np.float32),
        "h2d": h2d.astype(np.float32), "hs": hs.astype(np.float32), "scale": scale,
    }
    return anim


def runs(mask, shot_of):
    out = []
    N = len(mask)
    f = 0
    while f < N:
        if not mask[f]:
            f += 1
            continue
        s = f
        while f < N and mask[f] and shot_of[f] == shot_of[s]:
            f += 1
        out.append((s, f))
    return out


def robust_smooth(x, sigma, valid=None):
    """Median-based outlier rejection then zero-phase Gaussian smoothing (axis 0)."""
    n = len(x)
    v = np.ones(n, bool) if valid is None else valid.copy()
    v &= np.isfinite(x.reshape(n, -1)).all(1)
    if v.sum() < 2:
        return x
    xf = interp_gaps(np.nan_to_num(x), v)
    if n >= 5:
        k = 2
        pad = np.concatenate([xf[:1].repeat(k, 0), xf, xf[-1:].repeat(k, 0)])
        stack = np.stack([pad[i:i + n] for i in range(2 * k + 1)])
        med = np.median(stack, axis=0)
        dev = np.linalg.norm((xf - med).reshape(n, -1, 3), axis=-1) if xf.shape[-1] == 3 else np.abs(xf - med)
        scale = np.median(dev) + 1e-6
        bad = (dev > 6 * scale).reshape(n, -1)
        # replace outlying joints by the running median
        flat = xf.reshape(n, -1, 3) if xf.shape[-1] == 3 else None
        if flat is not None:
            medf = med.reshape(n, -1, 3)
            flat[bad] = medf[bad]
            xf = flat.reshape(xf.shape)
    if sigma > 0:
        xf = gaussian_smooth(xf, sigma, valid=v)
    return xf


def ray_depth_smooth(P, sig_depth, sig_2d, valid=None):
    """P: (n, J, 3) camera-frame points of one segment. Smooth the distance along
    each joint's viewing ray strongly, the ray direction (= image position) lightly."""
    n = len(P)
    v = np.isfinite(P.reshape(n, -1)).all(1) if valid is None else (valid & np.isfinite(P.reshape(n, -1)).all(1))
    if v.sum() < 2:
        return P
    Pf = interp_gaps(np.nan_to_num(P), v)
    dist = np.linalg.norm(Pf, axis=-1)
    ray = Pf / np.maximum(dist[..., None], 1e-9)
    dist = robust_smooth(dist[..., None].repeat(3, -1), sig_depth, v)[..., 0]
    if sig_2d > 0:
        ray = normalize(gaussian_smooth(ray, sig_2d, valid=v))
    out = ray * dist[..., None]
    out[~v] = P[~v]
    return out


def fill_and_smooth_quats(q, valid, sigma, max_gap):
    q = q.copy()
    idx = np.where(valid)[0]
    if len(idx) == 0:
        q[:] = [0, 0, 0, 1]
        return q
    # hold ends
    q[: idx[0]] = q[idx[0]]
    q[idx[-1] + 1:] = q[idx[-1]]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a <= 1:
            continue
        if b - a <= max_gap:
            for f in range(a + 1, b):
                t = (f - a) / (b - a)
                t = t * t * (3 - 2 * t)
                q[f] = slerp(q[a], q[b], t)
        else:
            mid = (a + b) // 2
            q[a + 1:mid] = q[a]
            q[mid:b] = q[b]
    q = quat_continuity(q)
    if sigma > 0:
        q = normalize(gaussian_smooth(q, sigma))
    return q


def root_motion(skel, Q, root_guess, valid, shot_of, fps, Pw, scale):
    """Root (hips) trajectory for the avatar.

    Horizontal motion is integrated from planted feet on the avatar's own
    skeleton (no foot sliding, works for tracking cameras); flight phases use
    the camera-based estimate. Vertical position keeps the planted foot on the
    floor (y = 0)."""
    N = len(Q)
    pos, rot = avatar_fk(skel, Q, np.zeros((N, 3)))
    bi = {b: i for i, b in enumerate(skel.bones)}
    rp = skel.rest_pos
    feet = []
    for side in ("left", "right"):
        fi = bi[side + "Foot"]
        ti = bi.get(side + "Toes", fi)
        heel_off = np.array([0, -rp[side + "Foot"][1], -0.04])
        toe_off = rp[side + "Toes"] - rp[side + "Foot"] + np.array([0, -rp[side + "Toes"][1], 0.04]) \
            if (side + "Toes") in rp else np.array([0, -rp[side + "Foot"][1], 0.15])
        heel = pos[:, fi] + np.einsum("nij,j->ni", rot[:, fi], heel_off)
        toe = pos[:, fi] + np.einsum("nij,j->ni", rot[:, fi], toe_off)
        _ = ti
        feet.append(np.stack([heel, toe], 1))  # (N,2,3) relative to hips
    feet = np.stack(feet, 1)  # (N, side, point, 3)
    low = feet[..., 1].min(2)  # lowest point per foot (N,2), relative to hips
    # actor-based absolute height cue (camera assumed not to crane quickly)
    abs_low = low + root_guess[:, 1:2]
    ground = np.zeros(N)
    win = int(1.5 * fps)
    for f in range(N):
        a, b = max(0, f - win), min(N, f + win + 1)
        m = valid[a:b]
        vals = abs_low[a:b][m].min(1) if m.any() else abs_low[a:b].min(1)
        ground[f] = np.percentile(vals, 10)
    height_abs = abs_low - ground[:, None]
    rel = low - low.min(1, keepdims=True)
    vel = np.zeros((N, 2))
    vel[1:] = np.linalg.norm(np.diff(feet[:, :, :, [0, 2]].mean(2), axis=0), axis=-1) * fps
    contact = (rel < 0.03) & (height_abs < 0.10) | (rel < 0.015) & (height_abs < 0.2)
    # hysteresis / debounce: remove 1-2 frame flickers
    k = max(1, int(round(0.05 * fps)))
    for s in (0, 1):
        c = contact[:, s].astype(float)
        c = np.convolve(c, np.ones(2 * k + 1) / (2 * k + 1), mode="same") > 0.5
        contact[:, s] = c
    root = np.zeros((N, 3))
    root[0, [0, 2]] = root_guess[0, [0, 2]]
    for f in range(1, N):
        both = contact[f] & contact[f - 1]
        if both.any() and shot_of[f] == shot_of[f - 1] or both.any():
            d = np.zeros(3)
            wsum = 0.0
            for s in (0, 1):
                if both[s]:
                    w = 1.0 / (1e-3 + low[f, s] - low[f].min() + 0.01)
                    d += w * (feet[f, s].mean(0) - feet[f - 1, s].mean(0))
                    wsum += w
            d /= wsum
            root[f, [0, 2]] = root[f - 1, [0, 2]] - d[[0, 2]]
        else:
            step = root_guess[f, [0, 2]] - root_guess[f - 1, [0, 2]]
            if shot_of[f] != shot_of[f - 1]:
                step = np.zeros(2)
            sp = np.linalg.norm(step) * fps
            if sp > 6.0:
                step *= 6.0 / sp
            root[f, [0, 2]] = root[f - 1, [0, 2]] + step
    # vertical
    y = np.full(N, np.nan)
    anyc = contact.any(1)
    y[anyc] = -np.where(contact[anyc], low[anyc], np.inf).min(1)
    if anyc.any():
        idx = np.arange(N)
        base = np.interp(idx, idx[anyc], y[anyc])
        g = root_guess[:, 1]
        gb = np.interp(idx, idx[anyc], g[anyc])
        flight = ~anyc
        y = np.where(anyc, y, base + np.clip(g - gb, -0.5, 3.0))
        y[flight] = np.maximum(y[flight], -low[flight].min(1))
    else:
        y = -low.min(1)
    root[:, 1] = gaussian_smooth(y, 0.8 * fps / 30.0)
    root[:, [0, 2]] = gaussian_smooth(root[:, [0, 2]], 0.5 * fps / 30.0)
    return root, contact


def foot_lock_ik(skel, Q, root, contact, fps, blend_s=0.06):
    """Pin each planted foot to one spot on the floor for the whole contact and
    fix the leg with two-bone IK (knee keeps its bend direction, the foot keeps
    its world orientation). Blended in/out over a few frames. Q is modified."""
    N = len(Q)
    bi = {b: i for i, b in enumerate(skel.bones)}
    nb = max(1, int(round(blend_s * fps)))
    pos, rot = avatar_fk(skel, Q, root)
    rp = skel.rest_pos
    for side in ("left", "right"):
        si = 0 if side == "left" else 1
        iu, il, ifo = bi[side + "UpperLeg"], bi[side + "LowerLeg"], bi[side + "Foot"]
        ip = bi[skel.parent[side + "UpperLeg"]]
        l1 = np.linalg.norm(rp[side + "LowerLeg"] - rp[side + "UpperLeg"])
        l2 = np.linalg.norm(rp[side + "Foot"] - rp[side + "LowerLeg"])
        c = contact[:, si]
        f = 0
        while f < N:
            if not c[f]:
                f += 1
                continue
            s = f
            while f < N and c[f]:
                f += 1
            e = f
            if e - s < 2:
                continue
            # pin where the foot lands; release smoothly at lift-off
            target_xz = pos[s:min(e, s + 2), ifo][:, [0, 2]].mean(0)
            for k in range(s, e):
                w = min(1.0, (e - k) / nb)
                H, K, A = pos[k, iu], pos[k, il], pos[k, ifo]
                At = A.copy()
                At[[0, 2]] = A[[0, 2]] + w * (target_xz - A[[0, 2]])
                d_vec = At - H
                d = float(np.clip(np.linalg.norm(d_vec), abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4))
                dn = d_vec / max(np.linalg.norm(d_vec), 1e-9)
                pole = (K - H) - dn * np.dot(K - H, dn)
                if np.linalg.norm(pole) < 1e-6:
                    pole = rot[k, iu] @ Z
                    pole = pole - dn * np.dot(pole, dn)
                pole = normalize(pole)
                a = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
                h = np.sqrt(max(l1 * l1 - a * a, 0.0))
                Kn = H + dn * a + pole * h
                Ae = H + dn * d
                Gu = rot_between(K - H, Kn - H) @ rot[k, iu]
                Gl0 = Gu @ (rot[k, iu].T @ rot[k, il])
                shin_now = Gl0 @ (rp[side + "Foot"] - rp[side + "LowerLeg"])
                Gl = rot_between(shin_now, Ae - Kn) @ Gl0
                Gf = rot[k, ifo]
                Q[k, iu] = mat_to_quat(rot[k, ip].T @ Gu)
                Q[k, il] = mat_to_quat(Gu.T @ Gl)
                Q[k, ifo] = mat_to_quat(Gl.T @ Gf)
    return Q

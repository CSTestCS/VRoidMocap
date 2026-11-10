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
from . import lift
from .analysis import world_cameras
from .detect import (L_ANK, L_EAR, L_EL, L_EYE, L_FOOT, L_HEEL, L_HIP, L_INDEX, L_KNEE,
                     L_PINKY, L_SH, L_THUMB, L_WR, MOUTH_L, MOUTH_R, NOSE, R_ANK, R_EAR,
                     R_EL, R_EYE, R_FOOT, R_HEEL, R_HIP, R_INDEX, R_KNEE, R_PINKY, R_SH,
                     R_THUMB, R_WR)
from .mathutil import (gaussian_smooth, interp_gaps, mat_to_quat, normalize, quat_continuity,
                       quat_to_mat, rot_between, rot_from_axes, slerp)
from .naturalize import Naturalizer, filter_rotations, hold_root
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
            rpalm = self.rest_palm[side]
            k = side + "LowerArm"
            G[k] = rot_from_axes(self.rest_dir[k], rpalm, fa,
                                 self._twist_axis(k, fa, palm_n, 0.85 if hp is not None else 0.0, rpalm))
            k = side + "Hand"
            G[k] = rot_from_axes(self.rest_dir[k], rpalm, hand_dir, palm_n)
            if hp is not None:
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
            # foot: heel->toe gives pitch/yaw; roll follows the shin (the ankle
            # landmarks are too close together to measure foot roll reliably)
            k = side + "Foot"
            G[k] = rot_from_axes(Z, X, foot_f, G[side + "LowerLeg"] @ X)
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
def solve_animation(capture, workdir, char_id, skel, smoothing=1.0, progress=None, foot_lock=True, debug=None):
    dbg = (lambda k, v: debug.__setitem__(k, np.array(v, copy=True))) if debug is not None else (lambda k, v: None)
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
    # a landmark outside the picture is a guess: treat it as not visible
    margin = 0.01 * max(W, H)
    inside = (lm2d[..., 0] > -margin) & (lm2d[..., 0] < W + margin) & (lm2d[..., 1] > -margin) & (lm2d[..., 1] < H + margin)
    vis = np.where(inside, vis, 0.0)
    valid = capture["tracked"] & (vis.mean(1) > 0.35) & (np.abs(world).sum((1, 2)) > 0)
    hvalid = (hs >= 0.5) & valid[:, None]

    # ---- 1. orient landmarks into the real camera; translation
    Wc = np.zeros_like(world)
    T = np.full((N, 3), np.nan)
    R_rel = quat_to_mat(npz["cam_q"].astype(np.float64))
    for f in np.where(valid)[0]:
        Rray = cam.ray_rotation(focal[f], cx, cy, *cropc[f])
        Wc[f] = world[f] @ Rray.T
    # spins: undo left/right label swaps and front/back depth flips
    view = normalize(np.c_[(cropc[:, 0] - cx) / focal, (cropc[:, 1] - cy) / focal, np.ones(N)])
    swap, mirror = resolve_body_flips(Wc, lm2d, valid, shot_of, view)
    vv = view[:, None, :]
    Wc = np.where(mirror[:, None, None], Wc - 2 * np.sum(Wc * vv, -1, keepdims=True) * vv, Wc)
    Wc[swap] = Wc[swap][:, LR_PERM]
    lm2d[swap] = lm2d[swap][:, LR_PERM]
    vis[swap] = vis[swap][:, LR_PERM]
    for arr in (h2d, hw, hs, hc):
        arr[swap] = arr[swap][:, ::-1]
    hvalid = (hs >= 0.5) & valid[:, None]
    for f in np.where(valid)[0]:
        t = cam.solve_translation(Wc[f], lm2d[f], vis[f] ** 2, focal[f], cx, cy)
        if t is None:
            valid[f] = False
        else:
            T[f] = t
    if valid.sum() < 2:
        raise RuntimeError("The selected character was not tracked well enough to animate.")
    # ---- 2. body in 3D: bone lengths + picture positions + learned depth, solved over time
    uv, conf = body_keypoints(lm2d, vis, capture.get("wb2d"), capture.get("wbs"), valid, swap, W, H)
    P0 = np.full((N, 33, 3), np.nan)
    P0[valid] = Wc[valid] + T[valid][:, None]
    segs = runs(valid, shot_of)
    P, blen = lift.fuse_body(np.nan_to_num(P0), uv, conf, focal, cx, cy, segs, fps,
                             progress=lambda f: progress(0.05 + 0.2 * f, "Reconstructing the body in 3D"))
    P[~valid] = np.nan
    T = (P[:, L_HIP] + P[:, R_HIP]) / 2  # camera -> hip centre, now with the refined distance
    lm2d = uv
    # landmarks that are not part of the solve ride along with their parent joint
    shc0, shc = (P0[:, L_SH] + P0[:, R_SH]) / 2, (P[:, L_SH] + P[:, R_SH]) / 2
    for j in range(0, 11):
        P[:, j] = P0[:, j] - shc0 + shc
    for wi, tips in ((L_WR, (L_INDEX, L_PINKY, L_THUMB)), (R_WR, (R_INDEX, R_PINKY, R_THUMB))):
        for j in tips:
            P[:, j] = P0[:, j] - P0[:, wi] + P[:, wi]
    progress(0.27, "Solving camera continuity")
    # yaw continuity across cuts using THIS character
    refine_gravity(P, vis, R_rel, shot_of, shots, valid)
    yaw_at = {}
    for f in np.where(valid)[0]:
        Rw0 = shots[shot_of[f]]["align"] @ R_rel[f]
        yaw_at[f] = cam.hips_yaw(P[f], Rw0)
    R_w = world_cameras(npz["cam_q"].astype(np.float64), shot_of, shots, [], {}, None, focal, W, H, facing=yaw_at)
    # a locked-off camera must not sway: hold its rotation while the background is still
    static = camera_static_mask(npz["bg_motion"] if "bg_motion" in npz.files else None,
                                npz["cam_q"].astype(np.float64), shot_of, fps)
    R_w = hold_static_rotation(R_w, static, shot_of)

    Hc = np.zeros((N, 2, 21, 3))
    for side in (0, 1):
        for f in np.where(hvalid[:, side])[0]:
            Rr = cam.ray_rotation(focal[f], cx, cy, *hc[f, side])
            Hc[f, side] = hw[f, side] @ Rr.T
    hlen = [median_lengths(Hc[:, s], HAND_TREE, hvalid[:, s]) if hvalid[:, s].sum() > 3 else {} for s in (0, 1)]
    Ph = np.full((N, 2, 21, 3), np.nan)
    for f in np.where(valid)[0]:
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
        if sig_2d > 0:
            P[s:e] = ray_depth_smooth(P[s:e], 0.0, sig_2d)
        for side in (0, 1):
            m = hvalid[s:e, side]
            if m.sum() >= 2:
                Ph[s:e, side] = ray_depth_smooth(Ph[s:e, side], sig_depth, sig_2d * 0.6, m)
    hipc = (P[:, L_HIP] + P[:, R_HIP]) / 2
    Pw = np.einsum("nij,nkj->nki", R_w, P - hipc[:, None])
    Phw = np.einsum("nij,nskj->nski", R_w, Ph - P[:, [L_WR, R_WR]][:, :, None])  # relative to wrist
    progress(0.35, "Computing bone rotations")

    # ---- 3b. palm / back-of-hand ambiguity: choose per frame between the hand as
    # detected and its depth-mirrored version, so the forearm stays within its
    # rotation range and does not flip between frames
    for side, wi, ei in ((0, L_WR, L_EL), (1, R_WR, R_EL)):
        sh_i = L_SH if side == 0 else R_SH
        view = normalize(np.einsum("nij,nj->ni", R_w, np.nan_to_num(P[:, wi])))
        Phw[:, side] = resolve_hand_mirroring(Phw[:, side], hvalid[:, side], view,
                                              Pw[:, sh_i], Pw[:, ei], Pw[:, wi], 1.0 if side == 1 else -1.0)

    # ---- 4. rotations
    solver = RigSolver(skel)
    B = len(skel.bones)
    Q = np.zeros((N, B, 4))
    Q[..., 3] = 1
    qvalid = np.zeros(N, bool)
    finger_valid = np.zeros((N, 2), bool)
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

    dbg("Pcam", P)
    dbg("Q_raw", Q)
    # ---- 5. temporal: per-bone confidence -> hold unseen bones, reject spikes, smooth
    progress(0.55, "Filtering (hold unseen bones, remove spikes, smooth)")
    conf = bone_confidence(skel, vis, hs, valid, hvalid)
    nat = Naturalizer(skel)
    Q = nat.unflip_fingers(Q)
    strength = np.array([2.0 if (b.endswith("Hand") or any(f in b for f in FINGERS)) else 1.0 for b in skel.bones])
    Q = filter_rotations(Q, conf, fps, smoothing, strength=strength)
    dbg("Q_filt", Q)

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
    root = anchor_root_to_camera(root, Q, skel, R_w, T, valid, static, shot_of, scale, fps, contacts)
    root = hold_root(root, valid, fps)
    progress(0.8, "Joint limits and limb IK")
    Q = nat.apply(Q, root, fps, conf)
    dbg("Q_nat", Q)
    if foot_lock:  # pins planted feet (two-bone IK)
        Q = foot_lock_ik(skel, Q, root, contacts, fps)
        Q = nat.legs_only(Q, root)  # ...then make sure the thighs still do not twist

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
    cam_pos = stabilize_camera(hipc_avatar - scale * np.einsum("nij,nj->ni", R_w, Ts), static, shot_of, fps)
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


def refine_gravity(P, vis, R_rel, shot_of, shots, valid):
    """Which way is up, per shot, from the reconstructed body. Averaged over a
    shot, the line from the feet to the hips of a dancer points up (the torso
    leans too much to be a good reference; it is only used when the legs are
    out of the picture). Uses the 3D solve, not the raw model output, which
    matters for high and low camera angles."""
    legs = [L_KNEE, R_KNEE, L_ANK, R_ANK]
    for si, sh in enumerate(shots):
        acc_l, n_l, acc_t, w_t = np.zeros(3), 0, np.zeros(3), 0.0
        for f in np.where(valid & (shot_of == si))[0]:
            if not np.isfinite(P[f]).all():
                continue
            hip = (P[f, L_HIP] + P[f, R_HIP]) / 2
            if vis[f, legs].min() > 0.5:
                acc_l += R_rel[f] @ normalize(hip - (P[f, L_ANK] + P[f, R_ANK]) / 2)
                n_l += 1
            up, w = cam.body_up_vector(P[f], vis[f])
            if up is not None and w > 0:
                acc_t += w * (R_rel[f] @ up)
                w_t += w
        if n_l >= 15:
            up_ref = normalize(acc_l)
        elif w_t >= 3.0:
            up_ref = normalize(acc_t)
        else:
            continue
        if np.degrees(np.arccos(np.clip(-up_ref[1], -1, 1))) > 70:
            continue
        sh["align"] = cam.align_from_up(up_ref)


def body_keypoints(lm2d, vis, wb2d, wbs, valid, swap, W, H):
    """Picture positions of the body joints: the whole-body model where it is
    confident (it is the more precise one), the MediaPipe landmarks otherwise.
    The whole-body model's left/right labels are matched to the (already
    resolved) MediaPipe ones. Returns (uv (N,33,2), conf (N,33))."""
    from .detect import WB_SWAP, WB_TO_MP
    uv = lm2d.copy()
    conf = np.clip((vis - 0.3) / 0.3, 0, 1)
    if wb2d is None or not np.any(wbs):
        return uv, conf
    wb2d = wb2d.astype(np.float64).copy()
    wbs = wbs.astype(np.float64).copy()
    mp_i = np.array([m for m in WB_TO_MP if 11 <= m <= 28])
    wb_i = np.array([WB_TO_MP[m] for m in mp_i])
    d0 = np.linalg.norm(wb2d[:, wb_i] - lm2d[:, mp_i], axis=-1).mean(1)
    d1 = np.linalg.norm(wb2d[:, WB_SWAP][:, wb_i] - lm2d[:, mp_i], axis=-1).mean(1)
    sw = valid & (d1 < 0.7 * d0)
    wb2d[sw] = wb2d[sw][:, WB_SWAP]
    wbs[sw] = wbs[sw][:, WB_SWAP]
    margin = 0.01 * max(W, H)
    for m, w in WB_TO_MP.items():
        if m < 11:
            continue
        p = wb2d[:, w]
        inside = (p[:, 0] > -margin) & (p[:, 0] < W + margin) & (p[:, 1] > -margin) & (p[:, 1] < H + margin)
        use = valid & inside & (wbs[:, w] > 0.35)
        uv[use, m] = p[use]
        conf[use, m] = np.clip(wbs[use, w] / 0.6, 0.3, 1.0)
    return uv, conf


def resolve_hand_mirroring(H, ok, view, S, E, Wr, sgn):
    """H: (N,21,3) hand points relative to the wrist (world). Returns H with some
    frames mirrored in depth, chosen by a 2-state Viterbi over the clip."""
    N = len(H)
    idx = np.where(ok & np.isfinite(H.reshape(N, -1)).all(1))[0]
    if len(idx) < 2:
        return H
    Hm = H.copy()
    v = view[idx][:, None, :]
    Hm[idx] = H[idx] - 2 * np.sum(H[idx] * v, -1, keepdims=True) * v

    def palm(h):
        return normalize(sgn * np.cross(h[:, 5] - h[:, 0], h[:, 17] - h[:, 0]))
    ua, fa = E[idx] - S[idx], Wr[idx] - E[idx]
    hng = np.cross(ua, fa)
    bend = np.arctan2(np.linalg.norm(hng, axis=1), np.sum(ua * fa, 1))
    n0 = normalize(np.cross(normalize(hng), fa))  # palm direction at neutral (thumb up)
    unary = np.zeros((len(idx), 2))
    pn = [palm(H[idx]), palm(Hm[idx])]
    fan = normalize(fa)
    for k in (0, 1):
        n = pn[k] - fan * np.sum(pn[k] * fan, 1, keepdims=True)
        c = np.sum(normalize(n) * n0, 1)
        w = np.clip((bend - np.radians(15)) / np.radians(25), 0, 1)
        unary[:, k] = w * np.clip(-0.17 - c, 0, None) * 4.0
    unary[:, 1] += 0.35  # the detector is usually right
    # transition: change of the palm direction between consecutive seen frames
    cost = unary[0].copy()
    back = np.zeros((len(idx), 2), int)
    for j in range(1, len(idx)):
        new = np.zeros(2)
        gapw = 1.0 / max(1, idx[j] - idx[j - 1]) ** 0.5
        for k in (0, 1):
            tr = [np.arccos(np.clip(np.dot(pn[k][j], pn[m][j - 1]), -1, 1)) / np.pi * 3.0 * gapw for m in (0, 1)]
            c2 = cost + np.array(tr)
            back[j, k] = int(np.argmin(c2))
            new[k] = c2[back[j, k]] + unary[j, k]
        cost = new
    st = np.zeros(len(idx), int)
    st[-1] = int(np.argmin(cost))
    for j in range(len(idx) - 1, 0, -1):
        st[j - 1] = back[j, st[j]]
    out = H.copy()
    sel = idx[st == 1]
    out[sel] = Hm[sel]
    return out


# MediaPipe left/right landmark pairs
LR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
            (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
LR_PERM = np.arange(33)
for _a, _b in LR_PAIRS:
    LR_PERM[_a], LR_PERM[_b] = _b, _a


def _torso_basis(W):
    """(n,33,3) -> (n,3,3) torso frame (left axis, up, forward) in camera coords."""
    x = (W[:, L_HIP] - W[:, R_HIP]) + (W[:, L_SH] - W[:, R_SH])
    up = (W[:, L_SH] + W[:, R_SH]) / 2 - (W[:, L_HIP] + W[:, R_HIP]) / 2
    x = normalize(x)
    up = normalize(up - x * np.sum(up * x, -1, keepdims=True))
    return np.stack([x, up, np.cross(x, up)], -1)


DEPTH_CHAINS = [(L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR),
                (L_HIP, L_KNEE), (L_KNEE, L_ANK), (R_HIP, R_KNEE), (R_KNEE, R_ANK)]


def resolve_depth_ambiguity(P, Ph, lm2d, vis, h2d, hvalid, valid, shot_of, focal, cx, cy, blen):
    P = P.copy()
    Ph = Ph.copy()
    N = len(P)
    for (p, c) in DEPTH_CHAINS:
        L = blen.get((p, c))
        if L is None:
            continue
        cand = np.full((N, 2, 3), np.nan)
        hint = np.zeros((N, 2))
        for f in np.where(valid)[0]:
            if not (np.isfinite(P[f, p]).all() and np.isfinite(P[f, c]).all()):
                continue
            uv = lm2d[f, c]
            if c in (L_WR, R_WR) and hvalid[f, 0 if c == L_WR else 1]:
                uv = h2d[f, 0 if c == L_WR else 1, 0]
            r = normalize(np.array([(uv[0] - cx) / focal[f], (uv[1] - cy) / focal[f], 1.0]))
            bq = float(np.dot(r, P[f, p]))
            disc = bq * bq - (float(np.dot(P[f, p], P[f, p])) - L * L)
            t0 = float(np.dot(P[f, c], r))
            if disc <= 0 or vis[f, c] < 0.5:
                cand[f, 0] = cand[f, 1] = P[f, c]
                continue
            sq = np.sqrt(disc)
            for k, t in enumerate((bq - sq, bq + sq)):
                q = r * t
                cand[f, k] = q
                hint[f, k] = abs(t - t0) / L
        for (a, b) in runs(valid & np.isfinite(cand[:, 0, 0]), shot_of):
            if b - a < 2:
                continue
            d = normalize(cand[a:b] - P[a:b, p][:, None, :])  # (n,2,3) bone directions
            n = b - a
            cost = 0.6 * hint[a]
            back = np.zeros((n, 2), int)
            for j in range(1, n):
                ang = np.arccos(np.clip(np.einsum("kx,mx->km", d[j], d[j - 1]), -1, 1))  # (k,m)
                c2 = cost[None, :] + 2.0 * ang
                back[j] = np.argmin(c2, 1)
                cost = c2[np.arange(2), back[j]] + 0.6 * hint[a + j]
            st = np.zeros(n, int)
            st[-1] = int(np.argmin(cost))
            for j in range(n - 1, 0, -1):
                st[j - 1] = back[j, st[j]]
            new = cand[a:b][np.arange(n), st]
            delta = new - P[a:b, c]
            P[a:b, c] = new
            # everything below that joint moves with it
            below = {L_EL: [L_WR, L_INDEX, L_PINKY, L_THUMB], L_WR: [L_INDEX, L_PINKY, L_THUMB],
                     R_EL: [R_WR, R_INDEX, R_PINKY, R_THUMB], R_WR: [R_INDEX, R_PINKY, R_THUMB],
                     L_KNEE: [L_ANK, L_HEEL, L_FOOT], L_ANK: [L_HEEL, L_FOOT],
                     R_KNEE: [R_ANK, R_HEEL, R_FOOT], R_ANK: [R_HEEL, R_FOOT]}.get(c, [])
            for j2 in below:
                P[a:b, j2] += delta
            if c in (L_EL, L_WR):
                Ph[a:b, 0] += delta[:, None, :]
            if c in (R_EL, R_WR):
                Ph[a:b, 1] += delta[:, None, :]
    return P, Ph


def camera_static_mask(bg_motion, cam_q, shot_of, fps, thr=0.05, min_s=0.4):
    """Frames where the real camera is not moving (background still)."""
    N = len(shot_of)
    if bg_motion is not None:
        m = np.asarray(bg_motion, float)
        k = max(1, int(round(0.15 * fps)))
        pad = np.pad(m, k, mode="edge")
        med = np.median(np.lib.stride_tricks.sliding_window_view(pad, 2 * k + 1), axis=-1)
        still = med < thr
    else:
        from .naturalize import q_angle
        stp = np.zeros(N)
        stp[1:] = np.degrees(q_angle(cam_q[1:], cam_q[:-1]))
        still = stp < 0.02
    out = np.zeros(N, bool)
    for (a, b) in runs(still, shot_of):
        if b - a >= min_s * fps:
            out[a:b] = True
    return out


def hold_static_rotation(R_w, static, shot_of):
    R = R_w.copy()
    for (a, b) in runs(static, shot_of):
        q = quat_continuity(mat_to_quat(R_w[a:b]))
        R[a:b] = quat_to_mat(normalize(q.mean(0)))
    return R


def _segments(mask, shot_of):
    """Runs of equal `mask` value that do not cross a shot change."""
    out, N, f = [], len(mask), 0
    while f < N:
        s0 = f
        while f < N and mask[f] == mask[s0] and shot_of[f] == shot_of[s0]:
            f += 1
        out.append((s0, f, bool(mask[s0])))
    return out


def _hipc(skel, Q, root):
    hi = skel.bones.index("hips")
    off = (skel.rest_pos["leftUpperLeg"] + skel.rest_pos["rightUpperLeg"]) / 2 - skel.rest_pos["hips"]
    return root + np.einsum("nij,j->ni", quat_to_mat(Q[:, hi]), off)


def anchor_root_to_camera(root, Q, skel, R_w, T, valid, static, shot_of, scale, fps, contacts=None):
    """While the camera is locked off, the character's position relative to it
    is measured directly (precise across the image, noisy in depth). The
    foot-contact root is pulled towards that measurement, but only while no foot
    is planted (steps, hops, turns on the move), so planted feet never slide;
    whatever cannot be applied yet waits for the next free moment. Height stays
    with the foot contacts."""
    root = root.copy()
    hipc = _hipc(skel, Q, root)
    for (a, b, st) in _segments(static & valid, shot_of):
        if not st or b - a < 3:
            continue
        Ts = T[a:b].copy()
        ok = np.isfinite(Ts).all(1)
        if ok.sum() < 3:
            continue
        Ts = interp_gaps(np.nan_to_num(Ts), ok)
        Tsm = np.stack([gaussian_smooth(Ts[:, 0], 0.04 * fps), gaussian_smooth(Ts[:, 1], 0.04 * fps),
                        gaussian_smooth(Ts[:, 2], 0.4 * fps)], 1)
        rel = scale * np.einsum("nij,nj->ni", R_w[a:b], Tsm)
        C = np.median(hipc[a:b] - rel, axis=0)        # the fixed camera
        corr = gaussian_smooth((C + rel) - hipc[a:b], 0.05 * fps)
        # split into "across the image" (precise: applied fully) and "in depth"
        # (noisy: applied only while no foot is planted)
        right = R_w[a:b, :, 0].copy()
        right[:, 1] = 0
        right = normalize(right)
        fwd = np.stack([-right[:, 2], np.zeros(len(right)), right[:, 0]], 1)
        lat = np.sum(corr * right, 1)
        dep = np.sum(corr * fwd, 1)
        n = b - a
        free = np.ones(n) if contacts is None else (~contacts[a:b].any(1)).astype(float)
        dep_applied = np.zeros(n)
        cur = 0.0
        for k in range(n):
            cur = cur + (dep[k] - cur) * free[k] * 0.5
            dep_applied[k] = cur
        delta = right * lat[:, None] + fwd * dep_applied[:, None]
        root[a:b, 0] += delta[:, 0]
        root[a:b, 2] += delta[:, 2]
    return root


def stabilize_camera(cam_pos, static, shot_of, fps):
    """Locked-off shots: one fixed camera position. Moving shots: a smooth path."""
    out = cam_pos.copy()
    for (a, b, st) in _segments(static, shot_of):
        if st:
            out[a:b] = np.median(cam_pos[a:b], axis=0)
        else:
            out[a:b] = gaussian_smooth(cam_pos[a:b], 0.3 * fps)
    # soften the hand-over between still and moving parts (never across a cut)
    for (a, b) in runs(np.ones(len(out), bool), shot_of):
        sm = gaussian_smooth(out[a:b], 0.1 * fps)
        keep = static[a:b]
        out[a:b] = np.where(keep[:, None], out[a:b], sm)
    return out


def resolve_body_flips(Wc, lm2d, valid, shot_of, view):
    """Spins confuse the pose model: it swaps the left/right labels or mirrors
    the body in depth (front <-> back). For every frame consider 4 readings
    (as detected / sides swapped / depth mirrored / both) and pick, per shot,
    the sequence whose torso turns smoothly (Viterbi). Returns (swap, mirror)
    boolean arrays."""
    N = len(Wc)
    swap = np.zeros(N, bool)
    mirror = np.zeros(N, bool)
    v = view[:, None, :]
    Wm = Wc - 2 * np.sum(Wc * v, -1, keepdims=True) * v
    hyps = [(0, 0), (1, 0), (0, 1), (1, 1)]
    B = [_torso_basis(W) for W in (Wc, Wc[:, LR_PERM], Wm, Wm[:, LR_PERM])]
    L2 = [lm2d, lm2d[:, LR_PERM], lm2d, lm2d[:, LR_PERM]]
    prior = np.array([0.0, 0.8, 0.8, 1.4])
    hgt = np.maximum(np.ptp(lm2d[:, :, 1], axis=1), 1.0)
    sided = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    for (s0, e0) in runs(valid, shot_of):
        idx = np.arange(s0, e0)
        if len(idx) < 2:
            continue
        cost = prior.copy()
        back = np.zeros((len(idx), 4), int)
        for j in range(1, len(idx)):
            f, g = idx[j], idx[j - 1]
            new = np.zeros(4)
            for k in range(4):
                tr = np.zeros(4)
                for m in range(4):
                    R = B[m][g].T @ B[k][f]
                    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
                    jump = np.mean(np.linalg.norm(L2[k][f, sided] - L2[m][g, sided], axis=1)) / hgt[f]
                    tr[m] = 2.0 * ang + 6.0 * jump
                c2 = cost + tr
                back[j, k] = int(np.argmin(c2))
                new[k] = c2[back[j, k]] + prior[k] * 0.05
            cost = new
        st = np.zeros(len(idx), int)
        st[-1] = int(np.argmin(cost))
        for j in range(len(idx) - 1, 0, -1):
            st[j - 1] = back[j, st[j]]
        for j, k in enumerate(st):
            swap[idx[j]], mirror[idx[j]] = hyps[k]
    return swap, mirror


BONE_LANDMARKS = {
    "hips": [L_HIP, R_HIP, L_SH, R_SH], "spine": [L_HIP, R_HIP, L_SH, R_SH], "chest": [L_HIP, R_HIP, L_SH, R_SH],
    "upperChest": [L_HIP, R_HIP, L_SH, R_SH], "neck": [NOSE, L_EAR, R_EAR, L_SH, R_SH], "head": [NOSE, L_EAR, R_EAR],
}
for _s, _S in (("left", "L"), ("right", "R")):
    _l = (lambda a, b: a if _S == "L" else b)
    BONE_LANDMARKS.update({
        f"{_s}UpperArm": [_l(L_SH, R_SH), _l(L_EL, R_EL)], f"{_s}LowerArm": [_l(L_EL, R_EL), _l(L_WR, R_WR)],
        f"{_s}UpperLeg": [_l(L_HIP, R_HIP), _l(L_KNEE, R_KNEE)], f"{_s}LowerLeg": [_l(L_KNEE, R_KNEE), _l(L_ANK, R_ANK)],
        f"{_s}Foot": [_l(L_ANK, R_ANK), _l(L_HEEL, R_HEEL), _l(L_FOOT, R_FOOT)],
        f"{_s}Toes": [_l(L_ANK, R_ANK), _l(L_HEEL, R_HEEL), _l(L_FOOT, R_FOOT)],
    })


def bone_confidence(skel, vis, hs, valid, hvalid):
    """(N,B) confidence that each bone's rotation was actually observed."""
    N = len(vis)
    conf = np.zeros((N, len(skel.bones)))
    for i, b in enumerate(skel.bones):
        side = 0 if b.startswith("left") else 1
        if b.endswith("Hand") or any(f in b for f in FINGERS):
            c = np.where(hvalid[:, side], hs[:, side], 0.0)
        elif b in BONE_LANDMARKS:
            c = vis[:, BONE_LANDMARKS[b]].min(1)
        else:
            c = np.ones(N)  # shoulders / eyes / jaw: never driven, stay at rest
        conf[:, i] = np.where(valid, c, 0.0)
    return conf


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


def foot_lock_ik(skel, Q, root, contact, fps, blend_s=0.06, leash=0.035):
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
                # leash: absorb small drift (jitter) but let a foot that really
                # slides or pivots move, instead of bending the leg to hold it
                dev = target_xz - A[[0, 2]]
                dn_ = np.linalg.norm(dev)
                if dn_ > leash:
                    target_xz = A[[0, 2]] + dev / dn_ * leash
                    dev = target_xz - A[[0, 2]]
                At = A.copy()
                At[[0, 2]] = A[[0, 2]] + w * dev
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


OUTPUT_FPS = (12, 24, 30, 60)


def resample_anim(anim, fps_out):
    """Resample a solved animation to exactly `fps_out` frames per second.

    Output frame k is at time k / fps_out. Rotations are interpolated on the
    sphere, positions linearly; nothing is blended across a camera cut.
    `src_frame` gives the nearest source video frame for each output frame."""
    fps_in = float(anim["fps"])
    n_in = int(anim["frames"])
    if abs(fps_out - fps_in) < 1e-9:
        out = dict(anim)
        out["src_frame"] = np.arange(n_in, dtype=np.int32)
        out["src_fps"] = fps_in
        return out
    dur = n_in / fps_in
    n_out = max(1, int(np.floor(dur * fps_out + 1e-6)))
    t = np.arange(n_out) / fps_out * fps_in  # fractional source index
    i0 = np.clip(np.floor(t).astype(int), 0, n_in - 1)
    i1 = np.clip(i0 + 1, 0, n_in - 1)
    a = (t - i0)[:, None]
    shot = np.asarray(anim["shot_of"])
    cut = shot[i0] != shot[i1]
    a = np.where(cut[:, None], np.round(a), a)
    near = np.where(a[:, 0] < 0.5, i0, i1)
    out = {k: v for k, v in anim.items() if not isinstance(v, np.ndarray)}
    q0 = np.asarray(anim["quats"], np.float64)[i0]
    q1 = np.asarray(anim["quats"], np.float64)[i1]
    q1 = np.where(np.sum(q0 * q1, -1, keepdims=True) < 0, -q1, q1)
    out["quats"] = normalize(q0 * (1 - a[:, :, None]) + q1 * a[:, :, None]).astype(np.float32)
    for k in ("root", "cam_pos"):
        v = np.asarray(anim[k], np.float64)
        out[k] = (v[i0] * (1 - a) + v[i1] * a).astype(np.float32)
    c0, c1 = np.asarray(anim["cam_quat"], np.float64)[i0], np.asarray(anim["cam_quat"], np.float64)[i1]
    c1 = np.where(np.sum(c0 * c1, -1, keepdims=True) < 0, -c1, c1)
    out["cam_quat"] = normalize(c0 * (1 - a) + c1 * a).astype(np.float32)
    for k, v in anim.items():
        if isinstance(v, np.ndarray) and k not in out and len(v) == n_in:
            out[k] = v[near]
    out["fps"] = float(fps_out)
    out["frames"] = int(n_out)
    out["src_fps"] = fps_in
    out["src_frame"] = near.astype(np.int32)
    return out

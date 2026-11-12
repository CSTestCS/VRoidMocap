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

import logging
import numpy as np

from . import camera as cam
from . import facing, head as headfit, lift
from .analysis import world_cameras
from .detect import (L_ANK, L_EAR, L_EL, L_EYE, L_FOOT, L_HEEL, L_HIP, L_INDEX, L_KNEE,
                     L_PINKY, L_SH, L_THUMB, L_WR, MOUTH_L, MOUTH_R, NOSE, R_ANK, R_EAR,
                     R_EL, R_EYE, R_FOOT, R_HEEL, R_HIP, R_INDEX, R_KNEE, R_PINKY, R_SH,
                     R_THUMB, R_WR)
from .mathutil import (gaussian_smooth, interp_gaps, mat_to_quat, normalize, quat_continuity,
                       quat_to_mat, rot_between, rot_from_axes, slerp)
from .naturalize import Naturalizer, filter_rotations
from .vrm import FINGER_SEGS, FINGERS

X, Y, Z = np.eye(3)
# animations saved by an older solver are solved again (the capture is kept)
log = logging.getLogger(__name__)
SOLVE_VERSION = 23
FINGERS_FROM_VIDEO = False  # fingers are set by hand afterwards (keys.py); the solve leaves them at rest
HAND_FROM_WB = True  # whole-body hand points where the hand model misses
HAND_CONTACTS = True  # palms touching (each other, the head, a hip) kept on the avatar
NO_CLIPPING = True    # limbs do not pass through the body / each other
# hidden bones placed from the motion around them (measured: helps the thighs;
# hidden arms do better with in-between motion; hidden shins stand on the floor)
CONTEXT_BONES = ("leftUpperLeg", "rightUpperLeg")
RAISED_THIGH_CONF = 0.7  # a thigh read raised (more than 60 deg from straight down) needs a knee seen at least this surely
RAISED_THIGH_Y = -0.5
CONTEXT_THIGH_MAX_UP = 0.0  # hidden thighs from the motion around them: not above level
CONTEXT_LIMBS = True  # hidden arms / thighs from the motion around them
PLANT_UNDER_HIPS = True
HAND_STEADY_SCALE = True  # hand size steady over time (a turned hand looks narrower)
HAND_SMOOTH = 2.0  # smoothing of the wrists relative to the rest of the body
HAND_PNP = True  # hand rotation from the avatar's rigid palm fitted to the picture
HEAD_GAIN = (1.0, 1.0, 1.5)  # turn, nod, tilt of the head (relative to the chest) as read from the face, scaled
STAND_HIPS = 0.95  # hips height (share of the rest height) when no foot is seen in a whole shot
PLANT_UNSEEN = True   # hidden lower legs stand on the floor
HEAD_FACE_FIT = True  # head turn / nod / tilt from the 68 face points (else from ears, eyes, nose)
SEGMENT_LIMBS = (0, 1, 2, 3, 4, 5, 6, 7)  # which limbs (SEG_MP order: arms, then legs)
SEGMENT_BONES = True  # a limb bone is known from one joint + one tracker on it (or two trackers)
WB_CALIBRATE = True  # scale up the body scores of a character the whole-body model is unsure of
WB_SCORE_REF = 0.78  # its median body score on real people and usual 3D avatars
WB_SCORE_MAX_GAIN = 1.5
WB_SCORE_FLOOR = 0.35  # scores below this are guesses: not boosted
WB_EDGE_FRAC = 0.05
BRIDGE_ANY = True  # frames without a MediaPipe reading are kept when the whole-body model sees the person
CLOSEUP_GAIN = 1.0  # legs running out of the picture (close-ups): hip and knee scores scaled by this (1 = off)
CLOSEUP_HINT_GAIN = 1.35  # ... in shots whose facing the user set (the facing is then known)
CLOSEUP_GAIN_FLOOR = 0.45  # ... from this score up
CLOSEUP_FRONT_VOTE = False  # close-ups: front/back voted over half a second from the hips' left/right labels
CLOSEUP_VOTE_S = 0.5
CLOSEUP_PELVIS = "hinted"  # close-ups (no shoulders): the pelvis from the hips + gravity, not held ("hinted" shots only, "all", "off")
HINT_STRENGTH = 3.0  # a front/back the user set for a shot: weight against what the picture shows (x W_FRONT)
CARRY_FACING = False  # close-ups: her facing carried over from the surrounding shots (cameras placed by the background)
CARRY_MAX_S = 5.0  # ... from clear sightings at most this far away
CLOSEUP_FACING = False  # close-ups without shoulders or feet: facing from the hips' labels and the knees, not the learned 3D
WB_ONLY_WEAK = False  # frames MediaPipe does not read at all: the whole-body model's 0.4 is a weak sighting on its own
LIFT_W = 0.75  # weight of the depth read from the motion over time (vs the pose model's own)
LIFT_W_LEGS = 0.4  # ... for the knees and ankles: it reads leg depth the wrong way round for low and close cameras (benchmarks: thighs 12% better overall, a low camera over a skirt 23 -> 13 deg)

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


PALM_IDX = [0, 1, 5, 9, 13, 17]


def _palm_fit(Ph, uv, tmpl, focal, cx, cy):
    """Turn a reconstructed hand (21,3, camera frame) so its palm matches the
    avatar's rigid palm fitted to the picture (perspective fit of wrist, thumb
    base and knuckles): the hand's rotation from the picture itself, not from
    the hand model's depth guess. Of the two fits a flat palm allows (turned
    towards or away from the camera by the same amount), the one nearer the
    reconstructed hand. The wrist stays where it is."""
    import cv2
    if not np.isfinite(Ph).all() or not np.isfinite(uv).all():
        return Ph
    obj = (tmpl - tmpl[0]).astype(np.float64)
    img = uv[PALM_IDX].astype(np.float64)
    if np.ptp(img, axis=0).max() < 6:
        return Ph
    K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1.0]])
    try:
        n, rvs, tvs, errs = cv2.solvePnPGeneric(obj, img, K, None, flags=cv2.SOLVEPNP_SQPNP)
        cands = [cv2.Rodrigues(r)[0] for r in rvs]
        n2, rvs2, _, _ = cv2.solvePnPGeneric(obj[[0, 2, 3, 4, 5]], img[[0, 2, 3, 4, 5]], K, None, flags=cv2.SOLVEPNP_IPPE)
        cands += [cv2.Rodrigues(r)[0] for r in rvs2]
    except cv2.error:
        return Ph
    if not cands:
        return Ph
    def frame(p0, p5, p9, p17):
        a = p9 - p0
        b = np.cross(p5 - p0, p17 - p0)
        a = a / max(np.linalg.norm(a), 1e-9)
        b = b - a * np.dot(b, a)
        b = b / max(np.linalg.norm(b), 1e-9)
        return np.stack([a, b, np.cross(a, b)], 1)
    F0 = frame(Ph[0], Ph[5], Ph[9], Ph[17])
    best, bd = None, -9
    for R in cands:
        q = obj @ R.T
        F1 = frame(q[0], q[2], q[3], q[5])
        d = np.trace(F0.T @ F1)
        if d > bd:
            best, bd = F1, d
    Rc = best @ F0.T  # turns the reconstructed palm onto the fitted one
    return Ph[0] + (Ph - Ph[0]) @ Rc.T


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

    def solve(self, J, hands, head=None):
        """J: dict of world joint positions (MediaPipe names). hands: [left, right]
        arrays (21,3) of world hand points or None. head: optional 3x3 world head
        rotation fitted to the face. Returns {bone: 3x3 global}."""
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
        Rh = rot_from_axes(X, Y, ax, normalize(up1 + up2)) if head is None else head
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


def slipped_frames(P, S, min_score=0.2, min_torso=1.0):
    """Frames where the whole-body reading has slipped along the body (shoulders
    read on the face, hips on the chest, knees on the hands: a failure on some
    drawn styles): the face's down (eyebrows -> chin) and the torso's (shoulders
    -> hips) disagree, or the torso is shorter than one face. (A back view, where
    the face points are the guess, can be flagged too: used only to withhold
    the score boost.)"""
    chin, top = P[:, 31], P[:, 50]
    d = chin - top
    L = np.linalg.norm(d, axis=1)
    u = d / np.maximum(L, 1e-6)[:, None]
    sh = (P[:, 5] + P[:, 6]) / 2
    hp = (P[:, 11] + P[:, 12]) / 2
    th = np.einsum("ni,ni->n", hp - sh, u) / np.maximum(L, 1e-6)
    use = (S[:, [31, 50]].min(1) > min_score) & (S[:, [5, 6, 11, 12]].min(1) > min_score) & (L > 4)
    return use & (th < min_torso)


def calibrate_scores(capture, W=None, H=None):
    """The whole-body model is less sure of itself on some drawn styles: on a
    character whose body joints score well below what it gives real people and
    usual 3D avatars, every body score is scaled up (by at most WB_SCORE_MAX_GAIN), so the
    fixed thresholds keep the readings that are right. The order of the scores,
    and so which joints and frames are the surest, is unchanged."""
    if not WB_CALIBRATE or capture.get("wbs") is None:
        return capture
    wbs = np.asarray(capture["wbs"], np.float64)
    t = np.asarray(capture.get("tracked", np.ones(len(wbs), bool)), bool)
    body = wbs[t][:, 5:23]
    body = body[body.max(1) > 0] if len(body) else body
    if len(body) < 10:
        return capture
    med = float(np.median(body))
    gain = float(np.clip(WB_SCORE_REF / max(med, 1e-3), 1.0, WB_SCORE_MAX_GAIN))
    if gain <= 1.05:
        return capture
    out = dict(capture)
    w2 = wbs.copy()
    sub = w2[:, 5:23]
    # not boosted: the unsure guesses, and the joints along the picture's edge
    # (where the model piles the joints it cannot see)
    p = np.asarray(capture["wb2d"], np.float64)[:, 5:23]
    if W is not None:
        m = WB_EDGE_FRAC * max(W, H)
        edge = (p[..., 0] < m) | (p[..., 0] > W - m) | (p[..., 1] < m) | (p[..., 1] > H - m)
    else:
        edge = np.zeros(sub.shape, bool)
    boost = (sub >= WB_SCORE_FLOOR) & ~edge
    slip = slipped_frames(np.asarray(capture["wb2d"], np.float64), wbs)
    boost &= ~slip[:, None]
    sub[boost] = np.minimum(sub[boost] * gain, 0.99)
    # on a character the model is this unsure of, a reading that slipped along
    # the body is wrong throughout: its body joints are not used
    sub[slip] = np.minimum(sub[slip], WB_SCORE_FLOOR * 0.5)
    w2[:, 5:23] = sub
    out["wbs"] = w2.astype(np.asarray(capture["wbs"]).dtype)
    return out


def closeup_frames(wbs, unsure=0.45, mid=0.35):
    """(N,) frames where the legs run out of the picture (both ankles unsure:
    the waist to the thighs or knees in a close-up, a waist-up shot) while the
    whole-body model reads a hip or knee."""
    s = np.asarray(wbs, np.float64)
    return (s[:, 15:17].max(1) < unsure) & (s[:, 11:15].max(1) >= mid)


def upright_uv(uv, mask, shots, shot_of):
    """uv with the shoulders put straight above the hips (along the camera's
    'up' from the scenery) in the frames of mask: in a close-up of the hips
    the shoulders are guesses outside the picture, and 'up' and the sideways
    axis in the picture (which the facing angle is measured against) must not
    come from them."""
    if mask is None or not np.any(mask):
        return uv
    out = uv.copy()
    for f in np.where(mask)[0]:
        up3 = shots[shot_of[f]]["align"].T @ np.array([0, 1.0, 0])
        u2 = up3[:2] / max(np.linalg.norm(up3[:2]), 1e-6)
        d = max(float(np.linalg.norm(uv[f, L_HIP] - uv[f, R_HIP])), 10.0)
        out[f, L_SH] = uv[f, L_HIP] + u2 * 2.5 * d
        out[f, R_SH] = uv[f, R_HIP] + u2 * 2.5 * d
    return out


def load_facing_hints(workdir, char_id, N):
    """(N,) front/back the user set per shot in the editor: +1 facing the
    camera, -1 turned away, nan = not set (facing_hints_c<id>.json: a list of
    {"start", "end", "facing": "front"|"back"})."""
    out = np.full(N, np.nan)
    path = os.path.join(workdir, f"facing_hints_c{char_id}.json")
    if not os.path.exists(path):
        return out
    try:
        with open(path) as fh:
            hints = json.load(fh)
    except (OSError, ValueError):
        return out
    for h in hints:
        v = {"front": 1.0, "back": -1.0}.get(h.get("facing"))
        if v is None:
            continue
        a, b = max(0, int(h["start"])), min(N, int(h["end"]))
        out[a:b] = v
    return out


def closeup_front_vote(wb_sync, valid, shot_of, fps, min_vote=0.3):
    """(N,) in close-ups of the hips and thighs (no shoulders, no feet): +1..
    facing the camera, -1.. turned away, nan elsewhere. The whole-body model's
    left/right labels of the hips say which way she faces (her left hip on
    the picture's right: towards the camera); they are right in 72% of such
    frames, and a vote over half a second (inside the shot) in 82% where it
    is clear (|vote| >= min_vote). Measured on rendered close-ups; where the
    picture shows no shoulders and no feet the learned 3D had her the wrong
    way round in a third to two thirds of frames."""
    wb2d, wbs = (np.asarray(x, np.float64) for x in wb_sync)
    N = len(wbs)
    cu = valid & closeup_frames(wbs) & (wbs[:, 5:7].max(1) < 0.45)
    if not cu.any():
        return None
    w = np.minimum(wbs[:, 11], wbs[:, 12])
    sg = np.where(cu & (w >= 0.3), np.sign(wb2d[:, 11, 0] - wb2d[:, 12, 0]) * w, 0.0)
    half = max(1, int(round(CLOSEUP_VOTE_S * fps)))
    out = np.full(N, np.nan)
    for f in np.where(cu)[0]:
        a, b = max(0, f - half), min(N, f + half + 1)
        sel = shot_of[a:b] == shot_of[f]
        tot, nrm = np.sum(sg[a:b][sel]), np.sum(np.abs(sg[a:b][sel]))
        if nrm > 0 and abs(tot / nrm) >= min_vote:
            out[f] = tot / nrm
    return out


def shot_links(npz):
    """Camera links across cuts saved by the analysis (shotlink), as tuples."""
    if "link_ab" not in npz or len(npz["link_ab"]) == 0:
        return []
    ab, R = npz["link_ab"], npz["link_R"]
    return [(int(a), int(b), int(fa), int(fb), R[k].astype(np.float64), int(n)) for k, (a, b, fa, fb, n) in enumerate(ab)]


def carried_facing(psi, bad, valid, uv, wbs, focal, cx, cy, shot_of, shots, npz, W, H, fps):
    """(N,) facing (psi, camera-relative) for close-ups of the hips and thighs
    (no shoulders, no feet), carried over from the frames around them where
    she is seen clearly: her facing in the world there, between the last
    clear frame before and the first after (the shorter way round, so a turn
    between them carries over too), seen from the close-up's camera. The
    cameras are placed in one world from the background (shotlink); a
    close-up whose camera could not be placed gets none (nan)."""
    from .analysis import world_cameras
    N = len(psi)
    s_ = np.asarray(wbs, np.float64)
    cu = valid & closeup_frames(s_) & (s_[:, 5:7].max(1) < 0.45)
    if not cu.any():
        return None
    r3, view = facing.lateral_axes(uv, focal, cx, cy)
    F_c = np.sin(np.nan_to_num(psi))[:, None] * r3 - np.cos(np.nan_to_num(psi))[:, None] * view
    R_rel = quat_to_mat(npz["cam_q"].astype(np.float64))
    good = valid & ~cu & ~bad & np.isfinite(psi)
    yaw0 = {}
    for f in np.where(good)[0]:
        Fw = shots[shot_of[f]]["align"] @ R_rel[f] @ F_c[f]
        yaw0[f] = float(np.arctan2(Fw[0], Fw[2]))
    R_w, placed = world_cameras(npz["cam_q"].astype(np.float64), shot_of, shots, [], {}, None, focal, W, H,
                                facing=yaw0, links=shot_links(npz), return_placed=True)
    Fw = np.einsum("nij,nj->ni", R_w, F_c)
    Y = np.arctan2(Fw[:, 0], Fw[:, 2])
    gi = np.where(good)[0]
    if len(gi) == 0:
        return None
    out = np.full(N, np.nan)
    gap = CARRY_MAX_S * fps
    for f in np.where(cu)[0]:
        if not placed[shot_of[f]]:
            continue
        k = np.searchsorted(gi, f)
        tp = gi[k - 1] if k > 0 and f - gi[k - 1] <= gap else None
        tn = gi[k] if k < len(gi) and gi[k] - f <= gap else None
        if tp is None and tn is None:
            continue
        if tp is not None and tn is not None:
            w = (f - tp) / max(tn - tp, 1)
            y = Y[tp] + w * float(np.angle(np.exp(1j * (Y[tn] - Y[tp]))))
        else:
            y = Y[tp if tp is not None else tn]
        Fc = R_w[f].T @ np.array([np.sin(y), 0.0, np.cos(y)])
        out[f] = np.arctan2(Fc @ r3[f], -(Fc @ view[f]))
    return out


def closeup_scores(capture, W, H, extra=None):
    """Measured on rendered ground truth (close-ups of the waist to the knees,
    and every other benchmark): where the legs run out of the picture, a
    thigh whose hip and knee the whole-body model scores 0.45-0.6 points the
    right way in the picture (more than 30 deg off in 0-4% of frames), but
    such scores are below the "seen" level, so close-up legs were not
    animated. There the hips and knees in the picture scored from
    CLOSEUP_GAIN_FLOOR up are scaled by CLOSEUP_GAIN."""
    on = np.ones(len(capture["wbs"]), bool) if CLOSEUP_GAIN > 1.0 else (extra if extra is not None else None)
    if on is None or not np.any(on) or capture.get("wbs") is None:
        return capture
    gain = CLOSEUP_GAIN if CLOSEUP_GAIN > 1.0 else CLOSEUP_HINT_GAIN
    wb2d = np.asarray(capture["wb2d"], np.float64)
    cu = closeup_frames(capture["wbs"]) & on
    if not cu.any():
        return capture
    out = dict(capture)
    w2 = np.asarray(capture["wbs"], np.float64).copy()
    for j in (11, 12, 13, 14):
        p = wb2d[:, j]
        inside = (p[:, 0] > 0) & (p[:, 0] < W) & (p[:, 1] > 0) & (p[:, 1] < H)
        b = cu & inside & (w2[:, j] >= CLOSEUP_GAIN_FLOOR)
        w2[b, j] = np.minimum(w2[b, j] * gain, 0.99)
    out["wbs"] = w2.astype(np.asarray(capture["wbs"]).dtype)
    return out


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

    capture = calibrate_scores(capture, W, H)
    hint = load_facing_hints(workdir, char_id, N)
    hinted = np.isfinite(hint)
    capture = closeup_scores(capture, W, H, extra=hinted)
    lm2d = capture["lm2d"].astype(np.float64)
    vis = capture["vis"].astype(np.float64)
    world = capture["world"].astype(np.float64)
    cropc = capture["cropc"].astype(np.float64)
    h2d, hw, hs, hc = (capture[k].astype(np.float64) for k in ("h2d", "hw", "hs", "hc"))
    # a landmark outside the picture is a guess: treat it as not visible
    margin = 0.01 * max(W, H)
    inside = (lm2d[..., 0] > -margin) & (lm2d[..., 0] < W + margin) & (lm2d[..., 1] > -margin) & (lm2d[..., 1] < H + margin)
    vis = np.where(inside, vis, 0.0)
    vis_raw = vis.copy()
    seg = capture["seg"].astype(np.float64) if "seg" in capture else None  # person mask at each landmark
    # a frame is usable when enough joints are clearly seen by either model (a
    # pair of legs in the picture is enough: the rest of the body being outside
    # the picture does not make the legs unusable)
    wbs_c = capture.get("wbs")
    n_mp = (vis > 0.5).sum(1)
    n_wb = np.zeros(N, int)
    n_wb_weak = np.zeros(N, int)
    if wbs_c is not None and np.any(wbs_c):
        from .detect import WB_TO_MP
        wb2d_c = capture["wb2d"]
        for m, w in WB_TO_MP.items():
            if m >= 11:
                p_ = wb2d_c[:, w]
                ins = (p_[:, 0] > 0) & (p_[:, 0] < W) & (p_[:, 1] > 0) & (p_[:, 1] < H)
                n_wb += (ins & (wbs_c[:, w] >= 0.6)).astype(int)
                n_wb_weak += (ins & (wbs_c[:, w] >= 0.4)).astype(int)
    # the model's scores flicker around the bar from frame to frame (close-ups:
    # sure readings come one frame at a time): a stretch of fairly sure frames
    # that runs into a sure one is usable throughout
    wb_ok = n_wb >= 4
    if wbs_c is not None and np.any(wbs_c) and (CLOSEUP_GAIN > 1.0 or hinted.any()):
        # a close-up shows the hips and knees only: one leg read surely (its hip
        # and knee) is enough to use the frame
        s_ = np.asarray(wbs_c, np.float64)
        cu_ = closeup_frames(s_) & (hinted | (CLOSEUP_GAIN > 1.0))
        wb_ok |= cu_ & (((s_[:, 11] >= 0.6) & (s_[:, 13] >= 0.6)) | ((s_[:, 12] >= 0.6) & (s_[:, 14] >= 0.6)))
    weak_on = np.full(N, WB_ONLY_WEAK) | hinted
    if weak_on.any():
        for a_, b_ in runs(capture["tracked"] & (n_wb_weak >= 4) & weak_on, shot_of):
            if wb_ok[a_:b_].any():
                wb_ok[a_:b_] = True
    has_mp = np.abs(world).sum((1, 2)) > 0
    # short MediaPipe dropouts while the whole-body model still sees the person:
    # its 3D guess and landmarks are carried over from the nearest frame it had
    bridge = capture["tracked"] & ~has_mp & wb_ok
    t_w = None
    if bridge.any() and has_mp.any():
        t_w = vis ** 2
        idx = np.where(has_mp)[0]
        for f in np.where(bridge)[0]:
            j = idx[np.argmin(np.abs(idx - f))]
            if BRIDGE_ANY and not (abs(j - f) <= 0.5 * fps and shot_of[j] == shot_of[f]):
                # MediaPipe misses the character for longer (some drawn styles it
                # rarely reads at all): its nearest reading in the same shot, else
                # anywhere, gives the starting 3D guess; the whole-body joints and
                # the depth read from the motion over time decide the pose
                same = idx[shot_of[idx] == shot_of[f]]
                j = same[np.argmin(np.abs(same - f))] if len(same) else j
            if BRIDGE_ANY or (abs(j - f) <= 0.5 * fps and shot_of[j] == shot_of[f]):
                world[f], cropc[f] = world[j], cropc[j] + (wb2d_c[f, 11:13].mean(0) - wb2d_c[j, 11:13].mean(0))
                lm2d[f] = lm2d[j]
                for m, w in WB_TO_MP.items():
                    if m >= 11:
                        lm2d[f, m] = wb2d_c[f, w]
                        t_w[f, m] = float(wbs_c[f, w]) ** 2
                vis[f] = 0.0  # nothing seen by MediaPipe itself: the whole-body model decides
                has_mp[f] = True
    valid = capture["tracked"] & ((n_mp >= 4) | wb_ok) & has_mp
    hvalid = (hs >= 0.5) & valid[:, None]

    # ---- 1. orient landmarks into the real camera; translation
    Wc = np.zeros_like(world)
    T = np.full((N, 3), np.nan)
    R_rel = quat_to_mat(npz["cam_q"].astype(np.float64))
    for f in np.where(valid)[0]:
        Rray = cam.ray_rotation(focal[f], cx, cy, *cropc[f])
        Wc[f] = world[f] @ Rray.T
    bridged = np.zeros(N, bool)
    if t_w is not None:
        bridged = bridge & has_mp
    for f in np.where(valid)[0]:
        t = cam.solve_translation(Wc[f], lm2d[f], vis[f] ** 2 if t_w is None else t_w[f], focal[f], cx, cy)
        if t is None:
            valid[f] = False
        else:
            T[f] = t
    # a bridged frame whose position could not be solved: the nearest solved one
    # (the joints' own positions still come from the picture)
    for f in np.where(bridged & ~valid)[0]:
        ok = np.where(valid & ~bridged & (shot_of == shot_of[f]))[0]
        if len(ok) and abs(ok[np.argmin(np.abs(ok - f))] - f) <= 0.5 * fps:
            T[f] = T[ok[np.argmin(np.abs(ok - f))]]
            valid[f] = True
    if valid.sum() < 2:
        raise RuntimeError("The selected character was not tracked well enough to animate.")
    # which way the torso faces, frame by frame (this also settles which body
    # side is which: detectors often swap left and right in back views)
    wb2d, wbs = capture.get("wb2d"), capture.get("wbs")
    wbz = capture["wbz"].astype(np.float64).copy() if "wbz" in capture else None
    uv, conf, wb_sync = body_keypoints(lm2d, vis, wb2d, wbs, valid, None, W, H, return_wb=True)
    face_score = face_visibility(wbs, N)
    # depth from the motion over time (before and after each frame): a second
    # opinion on which way the body faces and each limb points
    Ylift = None
    if LIFT_W > 0 and wb_sync is not None:
        try:
            from . import lifter
            progress(0.03, "Reading depth from the motion")
            Ylift = lifter.cached_lift(workdir, capture, runs(valid, shot_of), W, H, fps)
        except Exception:
            log.exception("depth from motion failed")
            Ylift = None
    W_lift = lifter.as_mediapipe(Ylift) if Ylift is not None else None
    Wc_f, W2_f, uv_f, conf_f = Wc, W_lift, uv, conf
    cu_axes = np.zeros(N, bool)
    if wb_sync is not None and (hinted.any() or CLOSEUP_PELVIS == "all"):
        s_ = np.asarray(wb_sync[1], np.float64)
        cu_axes = valid & closeup_frames(s_) & (s_[:, 5:7].max(1) < 0.45)
        if CLOSEUP_PELVIS != "all":
            cu_axes &= hinted
    if CLOSEUP_FACING and wb_sync is not None:
        # a close-up of the hips and thighs (no shoulders, no feet): the pose
        # model's 3D there is carried over from another moment and the motion
        # model guesses the rest of the body; measured on rendered close-ups both
        # put the body the wrong way round (facing more than 90 deg off in
        # 32-72% of frames). There the facing comes from the hips' left/right
        # labels and the knees.
        s_ = np.asarray(wb_sync[1], np.float64)
        cu_f = valid & closeup_frames(s_) & (s_[:, 5:7].max(1) < 0.45)
        uv_f, conf_f = uv, conf
        if cu_f.any():
            Wc_f = Wc.copy()
            Wc_f[cu_f] = np.nan
            if W_lift is not None:
                W2_f = W_lift.copy()
                W2_f[cu_f] = np.nan
            # the shoulders are guesses outside the picture: "up" in the picture
            # is the camera's (from the scenery), not hips -> guessed shoulders
            uv_f, conf_f = uv.copy(), conf.copy()
            for f in np.where(cu_f)[0]:
                up3 = shots[shot_of[f]]["align"].T @ np.array([0, 1.0, 0])
                u2 = up3[:2] / max(np.linalg.norm(up3[:2]), 1e-6)
                d = max(float(np.linalg.norm(uv[f, L_HIP] - uv[f, R_HIP])), 10.0)
                uv_f[f, L_SH] = uv[f, L_HIP] + u2 * 2.5 * d
                uv_f[f, R_SH] = uv[f, R_HIP] + u2 * 2.5 * d
                conf_f[f, [L_SH, R_SH, NOSE]] = 0.0
    uv_f = upright_uv(uv_f, cu_axes, shots, shot_of)
    front_v = closeup_front_vote(wb_sync, valid, shot_of, fps) if (CLOSEUP_FRONT_VOTE and wb_sync is not None) else None
    if hinted.any():
        front_v = np.where(hinted, hint * HINT_STRENGTH, front_v if front_v is not None else np.nan)
    dbg("front_vote", front_v if front_v is not None else np.full(N, np.nan))
    psi_c, psi_p, swap, bad = facing.track_facing(uv_f, conf_f, face_score, Wc_f, T, focal, cx, cy, valid,
                                                  shot_spans(valid, shot_of), fps, W2=W2_f, front=front_v)
    if CARRY_FACING and wb_sync is not None:
        prior = carried_facing(psi_p, bad, valid, uv, wb_sync[1], focal, cx, cy, shot_of, shots, npz, W, H, fps)
        dbg("facing_prior", prior)
        if prior is not None and np.isfinite(prior).any():
            psi_c, psi_p, swap, bad = facing.track_facing(uv_f, conf_f, face_score, Wc_f, T, focal, cx, cy, valid,
                                                          shot_spans(valid, shot_of), fps, W2=W2_f, prior=prior,
                                                          front=front_v)
    # frames that contradict everything around them (one-frame label swaps,
    # mis-detections): their bone rotations count as not seen (the pose holds /
    # blends across them); the position (hip centre) is still used
    if hinted.any() and wb_sync is not None:
        # close-ups whose facing the user set: which hip is which follows from
        # the picture (facing the camera, her left hip is on the picture's
        # right); the shoulders that otherwise settle it are guesses here
        s_ = np.asarray(wb_sync[1], np.float64)
        cu_h = hinted & valid & closeup_frames(s_) & (s_[:, 5:7].max(1) < 0.45) & np.isfinite(psi_p)
        lat_px = uv[:, L_HIP, 0] - uv[:, R_HIP, 0]
        clear = np.abs(lat_px) > 0.04 * W
        fr = np.cos(np.nan_to_num(psi_p))
        set_ = cu_h & clear & (np.abs(fr) > 0.3)
        swap = np.where(set_, (lat_px > 0) != (fr > 0), swap)
    psi_c[bad] = psi_p[bad] = np.nan
    # close-ups of the hips and thighs whose facing is known: the pelvis is
    # set from the hips (and gravity), not held from an earlier shot
    cu_pelvis = np.zeros(N, bool)
    if CLOSEUP_PELVIS != "off" and wb_sync is not None:
        s_ = np.asarray(wb_sync[1], np.float64)
        cu_pelvis = valid & closeup_frames(s_) & (s_[:, 5:7].max(1) < 0.45) & np.isfinite(psi_p)
        if CLOSEUP_PELVIS == "hinted":
            cu_pelvis &= hinted
    dbg("cu_pelvis", cu_pelvis)
    for arr in (Wc, lm2d, vis, uv, conf, vis_raw) + ((seg,) if seg is not None else ()):
        arr[swap] = arr[swap][:, LR_PERM]
    for arr in (h2d, hw, hs, hc):
        arr[swap] = arr[swap][:, ::-1]
    if wb_sync is not None:
        from .detect import WB_SWAP
        for arr in wb_sync + ((wbz,) if wbz is not None else ()):
            arr[swap] = arr[swap][:, WB_SWAP]
    hvalid = (hs >= 0.5) & valid[:, None]
    # the pose model's 3D guess: where it reads the body the wrong way round
    # (its usual mistake in back views, psi -> pi - psi, or a plain depth mirror,
    # psi -> -psi) its depths are reversed: mirror them. Face and hand points
    # ride along with it.
    r3, view = facing.lateral_axes(upright_uv(uv, cu_axes, shots, shot_of), focal, cx, cy)
    lp_c, _ = facing.learned_psi(Wc, view, r3[:, :2])
    pc0 = np.nan_to_num(psi_c)
    mirror = valid & (np.cos(np.pi - lp_c - pc0) > np.cos(lp_c - pc0))
    vv = view[:, None, :]
    Wc = np.where(mirror[:, None, None], Wc - 2 * np.sum(Wc * vv, -1, keepdims=True) * vv, Wc)
    dbg("psi", np.stack([psi_c, psi_p], 1))
    dbg("swap", swap)
    # ---- 2. body in 3D: bone lengths + picture positions + learned depth, solved over time
    # rigid segments: each limb's texture moves as one piece; carry clearly
    # detected joints along it into frames where the detectors are unsure
    trk_ok = np.zeros((N, 33), bool)
    if "segM" in capture and wb_sync is not None:
        uv, trk_ok = rigid_track(uv, raw_scores(wb_sync, vis, W, H, N), capture["wb2d"], capture["wbs"],
                                 capture["segM"].astype(np.float64), capture["segN"], valid, fps)
    dbg("trk_ok", trk_ok)
    P0 = np.full((N, 33, 3), np.nan)
    P0[valid] = Wc[valid] + T[valid][:, None]
    segs = runs(valid, shot_of)
    # depth from the motion over time (before and after each frame): a second
    # opinion on which way each limb points, towards or away from the camera
    z_extra = None
    if Ylift is not None:
        from . import lifter
        z_extra = lifter.depth_opinion(Ylift, np.nan_to_num(T[:, 2], nan=3.0), focal, W, H, N)
        z_extra[swap] = z_extra[swap][:, LR_PERM]
        z_extra[~valid] = np.nan
    lift_w = np.full(33, LIFT_W)
    lift_w[[L_KNEE, R_KNEE, L_ANK, R_ANK]] = LIFT_W_LEGS
    # ---- the proportions pass: before any pose is worked out, every bone's
    # length is measured over the whole video (left and right equal), so the
    # animation knows the character's proportions from its first frame
    progress(0.04, "Measuring the character's proportions (whole video)")
    torso_facing = (psi_c, psi_p, r3, view)
    # (the bone lengths leave the close-ups of the hips out: little of the body
    # and no depth worth the name; the distance is still read everywhere)
    L_prop, zroot_prop = lift.measure_proportions(np.nan_to_num(P0), uv, conf, focal, segs, fps, z_extra=z_extra,
                                                  torso=torso_facing, z_extra_w=lift_w)
    if cu_axes.any():
        L_full, _ = lift.measure_proportions(np.nan_to_num(P0), uv, conf, focal, runs(valid & ~cu_axes, shot_of), fps,
                                             z_extra=z_extra, torso=torso_facing, z_extra_w=lift_w)
        L_prop = np.where(np.isfinite(L_full), L_full, L_prop)
    proportions = proportion_report(L_prop)
    try:
        with open(os.path.join(workdir, f"proportions_c{char_id}.json"), "w") as fh:
            json.dump(proportions, fh, indent=1)
    except OSError:
        pass
    # trackers riding the limb bones: a limb partly in view (a knee hidden, 40%
    # of the thigh visible) still shows its line; its hidden end is put where
    # the line reaches at the bone's length
    inferred = np.zeros((N, 33), bool)
    seg_known = np.zeros((N, 8), bool)
    if "markers" in capture:
        from .capture import MARKER_FRAC
        raw0 = raw_scores(wb_sync, vis, W, H, N) if wb_sync is not None else np.zeros((N, 33))
        strong = (raw0 >= 0.6) | (vis >= 0.5)
        # how long each limb looks in the picture when square-on: its measured
        # length at the character's distance
        exp_len = np.full((N, 8), np.nan)
        okZ = valid & np.isfinite(zroot_prop)
        for k, (a_, b_) in enumerate(SEG_MP[:8]):
            e_ = lift.EDGES.index((lift.JI[a_], lift.JI[b_]))
            if np.isfinite(L_prop[e_]):
                exp_len[okZ, k] = focal[okZ] * L_prop[e_] / np.maximum(zroot_prop[okZ], 0.05)
        covered = occluded_joints(seg, vis_raw)
        if SEGMENT_BONES:
            # one joint + one tracker (or two trackers) of a limb bone seen:
            # the bone is known, whatever covers the rest of it
            m_ = 0.015 * max(W, H)
            in0 = (uv[..., 0] > m_) & (uv[..., 0] < W - m_) & (uv[..., 1] > m_) & (uv[..., 1] < H - m_)
            seen0 = strong & in0 & valid[:, None]
            if covered is not None:
                seen0 &= ~covered
            dz = np.full((N, 8), np.nan)
            for k, (a_, b_) in enumerate(SEG_MP[:8]):
                e_ = lift.EDGES.index((lift.JI[a_], lift.JI[b_]))
                Lb = L_prop[e_] if np.isfinite(L_prop[e_]) else np.nan
                zz = z_extra[:, b_] - z_extra[:, a_] if z_extra is not None else np.full(N, np.nan)
                zz = np.where(np.isfinite(zz), zz, P0[:, b_, 2] - P0[:, a_, 2])
                dz[:, k] = zz / Lb
            uv, conf, seg_known, inferred = bones_from_segments(
                uv, conf, seen0, capture["markers"].astype(np.float64), MARKER_FRAC, valid, W, H,
                capture["wb2d"].astype(np.float64), exp_len, dz, covered=covered)
        else:
            uv, conf, inferred = infer_from_markers(uv, conf, strong, capture["markers"].astype(np.float64),
                                                    MARKER_FRAC, valid, W, H, capture["wb2d"].astype(np.float64),
                                                    exp_len=exp_len, covered=covered)
    dbg("seg_known", seg_known)
    dbg("inferred", inferred)
    body_conf = conf.copy()
    P, blen = lift.fuse_body(np.nan_to_num(P0), uv, conf, focal, cx, cy, segs, fps, torso=torso_facing,
                             z_extra=z_extra, z_extra_w=lift_w, measured=(L_prop, zroot_prop),
                             progress=lambda f: progress(0.05 + 0.2 * f, "Reconstructing the body in 3D"))
    P[~valid] = np.nan
    T = (P[:, L_HIP] + P[:, R_HIP]) / 2  # camera -> hip centre, now with the refined distance
    mp2d = lm2d  # MediaPipe's own picture positions (it does not clamp joints to the edge)
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
    R_w = world_cameras(npz["cam_q"].astype(np.float64), shot_of, shots, [], {}, None, focal, W, H, facing=yaw_at,
                        links=shot_links(npz))

    # head: fitted to the face points (turn, nod, tilt), in world space
    Rface, face_q = headfit.head_from_face(wb2d, wbs, focal, cx, cy, valid)
    # a face "seen" on the back of the head is the detector guessing: only trust
    # the fit while the chest is not turned away and the head is turned at most
    # ~75 degrees from the chest
    fz = Rface[:, :, 2]
    yaw_h = np.arctan2(np.sum(fz * r3, 1), -np.sum(fz * view, 1))
    turned_away = ~np.isfinite(psi_c) | (np.cos(np.nan_to_num(psi_c)) < np.cos(np.radians(110)))
    neck_ok = np.cos(yaw_h - np.nan_to_num(psi_c)) > np.cos(np.radians(75))
    face_q = np.where(turned_away | ~neck_ok, 0.0, face_q)
    if not HEAD_FACE_FIT:
        face_q = np.zeros_like(face_q)
    head_w = np.einsum("nij,njk->nik", R_w, Rface)
    dbg("face_q", face_q)

    # hands the hand model missed (small, blurred, half hidden): the whole-body
    # model also places the 21 hand points, with a depth for each; use them
    if HAND_FROM_WB and wb_sync is not None and wbz is not None:
        for side, wi in ((0, L_WR), (1, R_WR)):
            o = 91 if side == 0 else 112
            kp, sc = wb_sync[0][:, o:o + 21].astype(np.float64), wb_sync[1][:, o:o + 21]
            zz = wbz[:, o:o + 21].astype(np.float64)
            ok = valid & ~hvalid[:, side] & (np.median(sc, 1) >= 0.5) & np.isfinite(P[:, wi, 2])
            for f in np.where(ok)[0]:
                Z = max(float(P[f, wi, 2]), 0.1)
                d = kp[f] - kp[f, 0]
                h2d[f, side] = kp[f]
                hw[f, side] = np.c_[d / focal[f] * Z, zz[f] - zz[f, 0]]
                hc[f, side] = (cx, cy)
                hs[f, side] = 0.5
                hvalid[f, side] = True
    # which way round each hand is (palm or back towards the camera), decided
    # over the whole clip from both hand detectors, with continuity
    for side in (0, 1):
        h2d[:, side], hw[:, side], flipped = hand_facing(h2d[:, side], hw[:, side], hvalid[:, side], wb_sync,
                                                         side, uv[:, L_WR if side == 0 else R_WR], fps)
        dbg(f"hand_flip{side}", flipped)
    Hc = np.zeros((N, 2, 21, 3))
    for side in (0, 1):
        for f in np.where(hvalid[:, side])[0]:
            Rr = cam.ray_rotation(focal[f], cx, cy, *hc[f, side])
            Hc[f, side] = hw[f, side] @ Rr.T
    hlen = [median_lengths(Hc[:, s], HAND_TREE, hvalid[:, s]) if hvalid[:, s].sum() > 3 else {} for s in (0, 1)]
    Ph = np.full((N, 2, 21, 3), np.nan)
    # the avatar's own palm (wrist, thumb base and the four knuckles) as a rigid shape
    palm_tmpl = []
    for sd in ("left", "right"):
        bs = [sd + "Hand", sd + "ThumbMetacarpal", sd + "IndexProximal", sd + "MiddleProximal", sd + "RingProximal",
              sd + "LittleProximal"]
        palm_tmpl.append(np.array([skel.rest_pos[b] for b in bs]) if all(b in skel.rest_pos for b in bs) else None)
    # the hand model's metric size need not agree with the body's depth: the
    # hand is rescaled so it is as large as the detected hand. The scale is
    # held steady over time (taken from the moments the hand shows its full
    # size): a hand turned away from the camera looks narrower, and that is
    # how far it is turned, not a smaller hand
    kk = np.full((N, 2), np.nan)
    for f in np.where(valid)[0]:
        for side, wi in ((0, L_WR), (1, R_WR)):
            if not hvalid[f, side] or not hlen[side]:
                continue
            rel = Hc[f, side] - Hc[f, side, 0]
            z = max(P[f, wi, 2], 0.1)
            proj = np.linalg.norm(rel[:, :2] * focal[f] / z, axis=1)[1:]
            img = np.linalg.norm(h2d[f, side, 1:] - h2d[f, side, 0], axis=1)
            kk[f, side] = float(np.clip(np.median(img) / max(np.median(proj), 1e-6), 0.5, 2.0))
    k_used = kk.copy()
    if HAND_STEADY_SCALE:
        win = int(round(1.5 * fps))
        for side in (0, 1):
            col = kk[:, side]
            for f in np.where(np.isfinite(col))[0]:
                a_, b_ = max(0, f - win), min(N, f + win + 1)
                w_ = col[a_:b_][(shot_of[a_:b_] == shot_of[f]) & np.isfinite(col[a_:b_])]
                k_used[f, side] = np.percentile(w_, 80)
    for f in np.where(valid)[0]:
        for side, wi in ((0, L_WR), (1, R_WR)):
            if not hvalid[f, side] or not hlen[side]:
                continue
            rel = Hc[f, side] - Hc[f, side, 0]
            k = float(k_used[f, side])
            hp = rel * k + P[f, wi]
            hr = rays_for(h2d[f, side], focal[f], cx, cy)
            ww = np.full(21, 0.9)
            ww[0] = 0
            Ph[f, side] = snap_chain(hp, hr, ww, HAND_TREE, {e: L * k for e, L in hlen[side].items()}, fixed=(0,))
            if HAND_PNP and palm_tmpl[side] is not None:
                Ph[f, side] = _palm_fit(Ph[f, side], h2d[f, side], palm_tmpl[side], focal[f], cx, cy)
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
        wind = hand_winding(h2d[:, side], hvalid[:, side], wb_sync, side, uv[:, wi])
        Phw[:, side] = resolve_hand_mirroring(Phw[:, side], hvalid[:, side], view,
                                              Pw[:, sh_i], Pw[:, ei], Pw[:, wi], 1.0 if side == 1 else -1.0, wind)

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
        if cu_pelvis[f]:
            # the hips' line from the tracked facing (in such a close-up the
            # surest thing known about the pelvis), around their centre; the
            # shoulders are guesses outside the picture: an upright torso
            mid = (J["hipL"] + J["hipR"]) / 2
            wdt = max(float(np.linalg.norm(J["hipL"] - J["hipR"])), 0.1)
            ax_c = np.cos(psi_p[f]) * r3[f] + np.sin(psi_p[f]) * view[f]  # right -> left hip, camera
            ax_w = R_w[f] @ ax_c
            ax_w[1] = 0.0
            ax_w = ax_w / max(np.linalg.norm(ax_w), 1e-6)
            J["hipL"], J["hipR"] = mid + 0.5 * wdt * ax_w, mid - 0.5 * wdt * ax_w
            # a thigh pointing into or out of the picture could go either way in
            # depth; with the facing known the knee goes in front of the body
            # (a hip bends forward far more than back)
            fwd_c = np.sin(psi_p[f]) * r3[f] - np.cos(psi_p[f]) * view[f]
            for hk, kk, ak in (("hipL", "knL", "anL"), ("hipR", "knR", "anR")):
                hip_c, kn_c = P[f, names[hk]], P[f, names[kk]]
                t_c = kn_c - hip_c
                ln = np.linalg.norm(t_c)
                if not np.isfinite(ln) or ln < 1e-6 or abs(t_c[2]) / ln < 0.3:
                    continue
                if np.dot(t_c, fwd_c) < 0:
                    t_m = t_c * np.array([1, 1, -1.0])
                    if np.dot(t_m, fwd_c) > 0:
                        d_w = R_w[f] @ (t_m - t_c)
                        J[kk] = J[kk] + d_w
                        J[ak] = J[ak] + d_w
            up_w = np.array([0, 0.45, 0])
            J["shL"], J["shR"] = J["hipL"] + up_w, J["hipR"] + up_w
        hands = []
        for side in (0, 1):
            if hvalid[f, side] and np.isfinite(Phw[f, side]).all():
                hands.append(Phw[f, side] + J["wrL" if side == 0 else "wrR"])
            else:
                hands.append(None)
        G = solver.solve(J, hands, head_w[f] if face_q[f] > 0.15 else None)
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

    # the face points under-read how far the head turns, nods and tilts (the
    # detector pulls an unusual face towards a straight-on one): measured on
    # rendered heads with known motion, scale them back up
    if HEAD_GAIN != (1.0, 1.0, 1.0) and skel.has("head") and skel.has("neck"):
        hn, hh = bidx["neck"], bidx["head"]
        use = qvalid & (face_q > 0.15)
        Rel = quat_to_mat(Q[use, hn]) @ quat_to_mat(Q[use, hh])
        yaw = np.arctan2(Rel[:, 0, 2], Rel[:, 2, 2]) * HEAD_GAIN[0]
        pitch = np.arcsin(np.clip(-Rel[:, 1, 2], -1, 1)) * HEAD_GAIN[1]
        roll = np.arctan2(Rel[:, 1, 0], Rel[:, 1, 1]) * HEAD_GAIN[2]
        lim = np.radians([80.0, 60.0, 50.0])
        yaw, pitch, roll = (np.clip(x, -l, l) for x, l in zip((yaw, pitch, roll), lim))
        def _ax(axis, t):
            q = np.zeros((len(t), 4)); q[:, axis] = np.sin(t / 2); q[:, 3] = np.cos(t / 2)
            return quat_to_mat(q)
        R2 = _ax(1, yaw) @ _ax(0, pitch) @ _ax(2, roll)
        q2 = mat_to_quat(R2)
        q2 = np.where(q2[:, 3:] < 0, -q2, q2)
        th = 2 * np.arccos(np.clip(q2[:, 3], -1, 1))
        axn = q2[:, :3] / np.maximum(np.linalg.norm(q2[:, :3], axis=1, keepdims=True), 1e-12)
        qn = np.concatenate([axn * np.sin(0.35 * th / 2)[:, None], np.cos(0.35 * th / 2)[:, None]], 1)  # 35% in the neck
        Q[use, hn] = qn
        Q[use, hh] = mat_to_quat(np.einsum("nji,njk->nik", quat_to_mat(qn), R2))
    dbg("Pcam", P)
    dbg("Q_raw", Q)
    Q_raw = Q.copy()
    # ---- 5. temporal: per-bone confidence -> hold unseen bones, reject spikes, smooth
    progress(0.55, "Filtering (hold unseen bones, remove spikes, smooth)")
    # seen / not seen per bone. A bone is animated only while all the joints
    # that define it are clearly inside the picture and detected (both
    # detectors count; brief glimpses do not). Everything else keeps its pose.
    hidden = occluded_joints(seg, vis_raw)
    # (seen: the whole-body model is sure (>= 0.6) or MediaPipe calls it visible (>= 0.5);
    # measured against rendered ground truth with occluders, this is the best trade-off)
    # (raw whole-body score: the fused confidence above is rescaled for the 3D solve)
    raw = raw_scores(wb_sync, vis, W, H, N)
    hidden = hidden if hidden is not None else np.zeros((N, 33), bool)
    hidden = hidden | cut_off_joints(uv, P, blen, focal, cx, cy, W, H)
    # a joint close to the picture edge (feet just above the bottom of the
    # picture): seen when MediaPipe, which does not clamp joints to the edge,
    # has it inside the picture and both models are sure
    mp_in = (mp2d[..., 0] > 2) & (mp2d[..., 0] < W - 2) & (mp2d[..., 1] > 2) & (mp2d[..., 1] < H - 2)
    edge_ok = mp_in & (vis >= 0.5) & (raw >= 0.6)
    js = joint_seen(lm2d, np.maximum(raw - 0.1, vis), valid & ~bad, W, H, fps, shot_of, hidden,
                    weak=(vis >= 0.3) | ((raw >= 0.4) & ((vis >= 0.1) | (bridged & (WB_ONLY_WEAK | hinted))[:, None])),
                    both=((raw >= 0.45) & (vis >= 0.25)) | trk_ok, edge_ok=edge_ok)
    dbg("js", js)
    dbg("raw", raw)
    dbg("uvs", lm2d)
    dbg("hidden", hidden)
    dbg("valid", valid)
    dbg("bad", bad)
    face_seen = (face_q > 0.15) if np.any(face_q > 0) else None
    raw_plain = raw_scores(wb_sync, np.ones_like(vis), W, H, N)  # (no edge rule)
    near_out = (just_outside(lm2d, raw_plain, W, H) | inferred) & (valid & ~bad)[:, None]
    seen = bone_seen(skel, js, hvalid & ~bad[:, None], face_seen, fps, shot_of, near_out,
                     seg_known=seg_known & (valid & ~bad)[:, None])
    feet_seen = np.stack([js[:, L_ANK] & js[:, L_HEEL] & js[:, L_FOOT], js[:, R_ANK] & js[:, R_HEEL] & js[:, R_FOOT]], 1)
    conf = bone_confidence(skel, np.maximum(body_conf, np.clip(vis, 0, 1)), hs, valid, hvalid)
    if face_seen is not None and skel.has("head"):
        conf[:, skel.bones.index("head")] = np.where(face_seen, 0.5 + 0.5 * face_q, 0.0)
    conf = np.where(seen, np.maximum(conf, 0.55), 0.0)
    if cu_pelvis.any() and skel.has("hips"):
        seen[:, skel.bones.index("hips")] |= cu_pelvis & js[:, L_HIP] & js[:, R_HIP]
    dbg("seen", seen)
    nat = Naturalizer(skel)
    Q = nat.unflip_fingers(Q)
    strength = np.array([HAND_SMOOTH if b.endswith("Hand") else 2.0 if any(f in b for f in FINGERS) else 1.0
                         for b in skel.bones])
    Q = filter_rotations(Q, conf, fps, smoothing, strength=strength)
    dbg("Q_filt", Q)

    # ---- 6. the scene: the camera from the scenery, the character inside it
    progress(0.7, "Placing the character in the scene")
    leg_actor = np.median([blen.get((L_HIP, L_KNEE), 0.4) + blen.get((L_KNEE, L_ANK), 0.4),
                           blen.get((R_HIP, R_KNEE), 0.4) + blen.get((R_KNEE, R_ANK), 0.4)])
    rp = skel.rest_pos
    leg_avatar = (np.linalg.norm(rp["leftUpperLeg"] - rp["leftLowerLeg"]) +
                  np.linalg.norm(rp["leftLowerLeg"] - rp["leftFoot"]))
    scale = float(leg_avatar / max(leg_actor, 1e-3))
    # joints in the camera frame: depth (along the ray) is the noisy part, so it is
    # smoothed much more than the position across the picture
    Pc = P.copy()
    for (s0, e0) in runs(valid, shot_of):
        if e0 - s0 >= 2:
            Pc[s0:e0] = ray_depth_smooth(P[s0:e0], 0.35 * fps, 0.04 * fps)
    C_ref = npz["cam_C"].astype(np.float64) if "cam_C" in npz.files else np.zeros((N, 3))
    cseg = npz["cam_seg"].astype(np.int64) if "cam_seg" in npz.files else -np.ones(N, np.int64)
    G = np.einsum("nij,nkj->nik", R_w, R_rel)  # shot reference frame -> world
    js_ok = js & valid[:, None]
    C_w = camera_path(C_ref, cseg, G, R_w, Pc, js_ok[:, L_HIP] & js_ok[:, R_HIP], shot_of, fps)
    Xw = C_w[:, None] + np.einsum("nij,nkj->nki", R_w, np.nan_to_num(Pc))  # every joint in the world
    O = shot_offsets(Xw, js_ok, shot_of, C_w, leg_actor)
    Xw = Xw + O[:, None]
    cam_world = C_w + O
    dbg("Xw", Xw)

    # ---- 7. limits, IK, hold unseen bones, then the avatar's root from what is seen
    progress(0.8, "Joint limits and limb IK")
    root0 = place_root(skel, Q, Xw, js_ok, scale, fps, shot_of)
    # directions as observed, smoothed in the world (for seen bones under a
    # parent that is not seen: limits relative to that parent must not bend them)
    Q_obs = world_smoothed(Q_raw, skel, 1.5 * fps / 30.0)
    if RAISED_THIGH_CONF is not None:
        # a thigh read raised (forward or sideways, > 60 deg) from a knee the models are not sure of
        # is a misreading (knees guessed at the hips, or on the hands): not seen
        _, rot_o = avatar_fk(skel, Q_obs, np.zeros((N, 3)))
        knee_c = np.maximum(raw, np.clip(vis, 0, 1))
        for side, kn in (("left", L_KNEE), ("right", R_KNEE)):
            iu, il = bidx[side + "UpperLeg"], bidx[side + "LowerLeg"]
            dn = np.einsum("nij,j->ni", rot_o[:, iu], skel.rest_pos[side + "LowerLeg"] - skel.rest_pos[side + "UpperLeg"])
            raised = (dn[:, 1] / np.maximum(np.linalg.norm(dn, axis=1), 1e-9) > RAISED_THIGH_Y) & (knee_c[:, kn] < RAISED_THIGH_CONF)
            seen[raised, iu] = False
            seen[raised, il] = False
    Q = nat.apply(Q, root0, fps, conf)
    dbg("Q_nat", Q)
    # keyframes only where a bone is seen: in between, smooth in-between motion
    # from its last seen pose to its next; before a bone is first seen it is in
    # the A pose
    Q = hold_unseen(Q, seen, fps, neutral_pose(skel), skel, Q_world=Q_obs)
    # a limb that is not seen is still there: arms and thighs point where the
    # motion around them says (read over +-4 s; +-8 s or +-20 s when the limb
    # was not seen for longer than that)
    kept = seen.copy()
    if Ylift is not None and CONTEXT_LIMBS:
        progress(0.82, "Placing hidden limbs from the motion around them")
        dirs = context_directions(Ylift, swap, R_w, seen, valid, skel, fps, shot_of,
                                  lambda m: lifter.cached_lift(workdir, capture, runs(valid, shot_of), W, H, fps, m))
        Q, filled = context_fill(Q, seen, valid, dirs, skel, fps, shot_of)
        kept |= filled
    root = place_root(skel, Q, Xw, js_ok, scale, fps, shot_of)
    contacts = foot_contacts(skel, Q, root, fps, feet_seen & seen[:, [bidx["leftLowerLeg"], bidx["rightLowerLeg"]]])
    stand_y = float(skel.rest_pos["hips"][1]) * STAND_HIPS if "hips" in skel.rest_pos else None
    lift_y = settle_on_floor(skel, Q, root, contacts, fps, shot_of, stand_y=stand_y)
    root[:, 1] += lift_y
    # lower legs that are not seen (below the picture, behind something) stand
    # on the floor
    planted = np.zeros((N, 2), bool)
    cam_av = scale * cam_world + np.array([0.0, 1.0, 0.0]) * lift_y[:, None]  # the camera, in the avatar's world
    if PLANT_UNSEEN:
        Q, planted = planted_legs(Q, root, seen, valid, skel, fps, shot_of, cam=cam_av)
        contacts = contacts | planted
    # and frame by frame: a planted foot is on the floor, not in it or above it
    root[:, 1] += floor_follow(skel, Q, root, contacts, fps, shot_of)
    if PLANT_UNSEEN and planted.any():
        Q, planted = planted_legs(Q, root, seen, valid, skel, fps, shot_of, cam=cam_av)
        for si, side in enumerate(("left", "right")):
            for b in (side + "UpperLeg", side + "LowerLeg", side + "Foot"):
                kept[:, bidx[b]] |= planted[:, si]
    if foot_lock:  # pins planted feet (two-bone IK) - only legs that are seen
        Q = foot_lock_ik(skel, Q, root, contacts & ~planted, fps)
        Q = nat.legs_only(Q, root)  # ...then make sure the thighs still do not twist
        Q = hold_unseen(Q, kept, fps, neutral_pose(skel), skel, Q_world=Q_obs, observed=seen)
    # ---- 7b. hands that touch the body touch it on the avatar too, and no
    # limb passes through another part of the body (they just touch)
    from . import body_shape, contacts as touch
    caps = body_shape.capsules(skel)
    if HAND_CONTACTS:
        progress(0.9, "Hands touching")
        con, palm3 = touch.detect(P, Ph, h2d, hvalid, uv, valid, fps, shot_of)
        palm_w = np.einsum("nij,nkj->nki", R_w, palm3)
        Q = touch.apply(skel, Q, root, con, palm_w, caps, fps)
        for k_, v_ in con.items():
            dbg(f"contact_{k_}", v_)
    if NO_CLIPPING:
        progress(0.92, "Keeping limbs from passing through each other")
        allow = touch.allowed_overlaps(con, fps) if HAND_CONTACTS else None
        Q, _ = body_shape.resolve_collisions(skel, Q, caps, fps, shot_of, active=valid, allow=allow)
    if not FINGERS_FROM_VIDEO:
        for i_, b_ in enumerate(skel.bones):
            if any(f_ in b_ for f_ in FINGERS):
                Q[:, i_] = (0.0, 0.0, 0.0, 1.0)
    dbg("Q_final", Q)

    # ---- 8. the cloned camera: where the real camera is in the scene (it never
    # follows the character)
    cam_pos = scale * cam_world
    cam_pos[:, 1] += lift_y
    cam_quat = mat_to_quat(R_w @ cam.D)
    cam_vfov = np.array([cam.vfov_from_focal(fv, H) for fv in focal])

    progress(0.95, "Packing animation")
    anim = {
        "fps": fps, "frames": N, "width": W, "height": H, "character": char_id,
        "bones": skel.bones, "quats": Q.astype(np.float32), "root": root.astype(np.float32),
        "valid": valid, "contacts": contacts, "finger_valid": finger_valid, "proportions": proportions,
        "cam_pos": cam_pos.astype(np.float32), "cam_quat": cam_quat.astype(np.float32),
        "cam_vfov": cam_vfov.astype(np.float32), "shot_of": shot_of.astype(np.int32),
        "lm2d": lm2d.astype(np.float32), "vis": vis.astype(np.float32),
        "h2d": h2d.astype(np.float32), "hs": hs.astype(np.float32), "scale": scale,
        "solver": SOLVE_VERSION,
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
        # a refinement, not a replacement: partial bodies at the picture edge can
        # skew it, so it may move "up" by at most 30 degrees from the analysis
        up0 = sh["align"].T @ np.array([0, 1.0, 0])  # world up, in the camera
        if np.degrees(np.arccos(np.clip(np.dot(up_ref, up0), -1, 1))) > 30:
            continue
        sh["align"] = cam.align_from_up(up_ref)


def body_keypoints(lm2d, vis, wb2d, wbs, valid, swap, W, H, return_wb=False):
    """Picture positions of the body joints: the whole-body model where it is
    confident (it is the more precise one), the MediaPipe landmarks otherwise.
    The whole-body model's left/right labels are matched to the (already
    resolved) MediaPipe ones. Returns (uv (N,33,2), conf (N,33))."""
    from .detect import WB_SWAP, WB_TO_MP
    uv = lm2d.copy()
    conf = np.clip((vis - 0.3) / 0.3, 0, 1)
    if wb2d is None or not np.any(wbs):
        return (uv, conf, None) if return_wb else (uv, conf)
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
        c = np.clip(wbs[:, w] / 0.6, 0.3, 1.0)
        # a joint outside the picture is put on its edge by the whole-body model
        # (with a good score): near the edge only trust it as far as MediaPipe
        # also sees the joint
        edge = 0.03 * max(W, H)
        near = (p[:, 0] < edge) | (p[:, 0] > W - edge) | (p[:, 1] < edge) | (p[:, 1] > H - edge)
        c = np.where(near, np.minimum(c, np.clip((vis[:, m] - 0.3) / 0.3, 0, 1)), c)
        conf[use, m] = c[use]
    return (uv, conf, (wb2d, wbs)) if return_wb else (uv, conf)


def face_visibility(wbs, N):
    """How clearly a face is seen per frame (whole-body model face points), nan if unknown."""
    if wbs is None or not np.any(wbs):
        return np.full(N, np.nan)
    fs = np.asarray(wbs, np.float64)[:, 23:91].mean(1)
    return np.where(np.asarray(wbs).any(1), fs, np.nan)


W_WIND = 3.0


def hand_winding(h2d, hok, wb, side, wrist_uv):
    """Whether the whole-body model sees the hand the same way round as the
    hand model (thumb side vs little-finger side in the picture). (N,) in
    -1..1: + = both agree (the hand model's reading is supported), - = they
    disagree (its mirror image is supported), 0 = unknown. A depth-mirrored hand
    looks the same in the picture, so this second opinion is the only direct
    evidence for which of the two it is."""
    def wnd(p):
        a, b = p[:, 5] - p[:, 0], p[:, 17] - p[:, 0]
        sz = np.sum((p[:, 9] - p[:, 0]) ** 2, 1) + 1e-6
        return (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / sz
    N = len(h2d)
    if wb is None:
        return np.zeros(N)
    w_mp = np.where(hok, np.clip(wnd(h2d) / 0.3, -1, 1), 0.0)
    k0 = 91 if side == 0 else 112
    p = wb[0][:, k0:k0 + 21]
    sc = wb[1][:, k0:k0 + 21].mean(1)
    span = np.ptp(p, axis=1).max(1)
    near = np.linalg.norm(p[:, 0] - wrist_uv, axis=1) < 0.6 * np.maximum(span, 10)
    w_wb = np.where((sc > 0.4) & near, np.clip(wnd(p) / 0.3, -1, 1), 0.0)
    return np.sign(w_mp) * np.sign(w_wb) * np.minimum(np.abs(w_mp), np.abs(w_wb))


def _winding(p):
    a, b = p[:, 5] - p[:, 0], p[:, 17] - p[:, 0]
    sz = np.sum((p[:, 9] - p[:, 0]) ** 2, 1) + 1e-6
    return (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / sz


def hand_facing(h2d, hw, hok, wb, side, wrist_uv, fps, w_mp=0.9, w_wb=1.5, w_turn=4.0):
    """Palm or back of the hand towards the camera, per frame.

    In the picture this is the order of the thumb and little-finger sides;
    both hand detectors read it, the whole-body model more reliably (measured
    on rendered ground truth: 82-92% vs 72-83%). A hand cannot go from showing
    its palm to showing its back without turning edge-on in between, so the
    reading is decoded over the whole clip (Viterbi: both detectors' votes per
    frame, a cost for switching that is high while the palm faces the camera
    squarely). Where the result disagrees with the hand model, its hand is
    reflected across its own long axis (thumb and little-finger sides swap, the
    fingers keep pointing the same way), and the whole-body model's finger
    points are used in the picture when it has them.
    Returns h2d, hw and the frames that were flipped."""
    N = len(h2d)
    h2d, hw = h2d.copy(), hw.copy()
    flipped = np.zeros(N, bool)
    idx = np.where(hok)[0]
    if len(idx) < 2:
        return h2d, hw, flipped
    s_mp = _winding(h2d)
    have_wb = np.zeros(N, bool)
    s_wb = np.zeros(N)
    pw = None
    if wb is not None:
        k0 = 91 if side == 0 else 112
        pw = wb[0][:, k0:k0 + 21]
        sc = wb[1][:, k0:k0 + 21].mean(1)
        span = np.ptp(pw, axis=1).max(1)
        near = np.linalg.norm(pw[:, 0] - wrist_uv, axis=1) < 0.6 * np.maximum(span, 10)
        have_wb = (sc > 0.4) & near
        s_wb = _winding(pw)
    # states: 0 = as the hand model reads it, 1 = the other way round
    sgn = np.sign(s_mp[idx])
    un = np.zeros((len(idx), 2))
    un[:, 1] += w_mp  # the hand model votes for itself
    hw_i = have_wb[idx] & (np.abs(s_wb[idx]) > 0.05)
    agree = np.sign(s_wb[idx]) == sgn
    un[hw_i & agree, 1] += w_wb
    un[hw_i & ~agree, 0] += w_wb
    # facing amount (|winding| ~ how squarely the palm faces the camera)
    face = np.clip(np.abs(s_mp[idx]) / 0.3, 0, 1)
    cost = un[0].copy()
    back = np.zeros((len(idx), 2), int)
    for j in range(1, len(idx)):
        gap = idx[j] - idx[j - 1]
        # the reading's sign in the picture for each state: state 0 = sgn, state 1 = -sgn
        sw = w_turn * min(face[j], face[j - 1]) / max(1.0, gap / max(1.0, 0.1 * fps))
        new = np.zeros(2)
        for k in (0, 1):
            sk = sgn[j] * (1 if k == 0 else -1)
            tr = np.array([0.0 if sk == sgn[j - 1] * (1 if m == 0 else -1) else sw for m in (0, 1)])
            c2 = cost + tr
            back[j, k] = int(np.argmin(c2))
            new[k] = c2[back[j, k]] + un[j, k]
        cost = new
    st = np.zeros(len(idx), int)
    st[-1] = int(np.argmin(cost))
    for j in range(len(idx) - 1, 0, -1):
        st[j - 1] = back[j, st[j]]
    for f in idx[st == 1]:
        # reflect the hand model's hand across its long axis (wrist -> middle knuckle)
        rel = hw[f] - hw[f, 0]
        ax = rel[9]
        u = np.cross(np.array([0, 0, 1.0]), ax)
        if np.linalg.norm(u) < 1e-9:
            continue
        u = u / np.linalg.norm(u)
        hw[f] = hw[f, 0] + rel - 2 * np.outer(rel @ u, u)
        if have_wb[f] and np.sign(s_wb[f]) != np.sign(s_mp[f]):
            h2d[f] = pw[f]
        else:
            q = h2d[f] - h2d[f, 0]
            a2 = q[9] / max(np.linalg.norm(q[9]), 1e-9)
            n2 = np.array([-a2[1], a2[0]])
            h2d[f] = h2d[f, 0] + q - 2 * np.outer(q @ n2, n2)
        flipped[f] = True
    return h2d, hw, flipped


def resolve_hand_mirroring(H, ok, view, S, E, Wr, sgn, wind=None):
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
    if wind is not None:
        # second opinion of the whole-body model (see hand_winding)
        unary[:, 0] += W_WIND * np.maximum(0, -wind[idx])
        unary[:, 1] += W_WIND * np.maximum(0, wind[idx])
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


def camera_path(C_ref, cseg, G, R_w, Pc, hips_seen, shot_of, fps):
    """Where the camera is in the world (actor metres), frame by frame.

    The scenery gives the camera's travel (shape and direction) but not its
    size: each travelling stretch gets one scale, the one that leaves the
    character's hips moving least across the stretch (a dancer does not race
    the camera). The camera is still wherever the scenery says it does not
    travel. Each shot starts at the origin (shot_offsets places it)."""
    N = len(C_ref)
    d = np.zeros((N, 3))
    same = np.r_[False, shot_of[1:] == shot_of[:-1]]
    d[1:] = C_ref[1:] - C_ref[:-1]
    d[~same] = 0
    sg = np.where(cseg >= 0, cseg, np.r_[-1, cseg[:-1]])
    sg[~same] = -1
    dw = np.einsum("nij,nj->ni", G, d)
    hip = (Pc[:, L_HIP] + Pc[:, R_HIP]) / 2
    Y = np.einsum("nij,nj->ni", R_w, np.nan_to_num(hip))
    ok = hips_seen & np.isfinite(hip).all(1)
    L = max(1, int(round(0.25 * fps)))
    scales = {}
    for k in np.unique(sg[sg >= 0]):
        m = sg == k
        path = np.cumsum(np.where(m[:, None], dw, 0.0), axis=0)
        f = np.where(m[:-L] & m[L:] & ok[:-L] & ok[L:])[0] if N > L else np.zeros(0, int)
        if len(f) < 10:
            continue
        dU = path[f + L] - path[f]
        dY = Y[f + L] - Y[f]
        den0 = np.sum(dU * dU)
        if den0 < 1e-10:
            continue
        w = np.ones(len(f))
        sc = 0.0
        for _ in range(4):
            sc = -np.sum(w * np.sum(dU * dY, 1)) / max(np.sum(w * np.sum(dU * dU, 1)), 1e-12)
            r = np.linalg.norm(sc * dU + dY, axis=1)
            w = np.where(r < 0.05, 1.0, 0.05 / np.maximum(r, 1e-9))
        if sc > 0:
            scales[k] = sc
    if scales:
        default = float(np.median(list(scales.values())))
    else:
        zc = Pc[ok, L_HIP, 2] if ok.any() else np.array([3.0])
        default = 2.0 * float(np.median(zc))
    s_f = np.array([scales.get(k, default) if k >= 0 else 0.0 for k in sg])
    cs = np.cumsum(dw * s_f[:, None], axis=0)
    return cs - _shot_start_values(cs, shot_of)


def _shot_start_values(x, shot_of):
    """x[first frame of each frame's shot] (to restart a cumulative sum per shot)."""
    starts = np.r_[0, np.where(shot_of[1:] != shot_of[:-1])[0] + 1]
    idx = starts[np.searchsorted(starts, np.arange(len(x)), side="right") - 1]
    return x[idx]


def _ref_point(Xw, js_ok):
    """A body reference point per frame (hip centre, or estimated from the
    shoulders) and whether it is known."""
    hip = (Xw[:, L_HIP] + Xw[:, R_HIP]) / 2
    sh = (Xw[:, L_SH] + Xw[:, R_SH]) / 2 - np.array([0, 0.5, 0])
    hs = js_ok[:, L_HIP] & js_ok[:, R_HIP]
    ss = js_ok[:, L_SH] & js_ok[:, R_SH]
    return np.where(hs[:, None], hip, sh), hs | ss


def shot_offsets(Xw, js_ok, shot_of, C_w, leg_actor):
    """Per shot, where its scene sits in the world: the floor at y = 0 (from the
    feet, when they are seen), and across a cut the character continues from
    where it was. Shots the character is not in keep the camera where the
    nearest shot had it."""
    N = len(shot_of)
    O = np.zeros((N, 3))
    ref, known = _ref_point(Xw, js_ok)
    feet = [L_HEEL, R_HEEL, L_FOOT, R_FOOT]
    fy = np.where(js_ok[:, feet], Xw[:, feet, 1], np.inf).min(1)
    shots = []
    starts = np.r_[0, np.where(shot_of[1:] != shot_of[:-1])[0] + 1, N]
    prev = None
    has = []
    for i in range(len(starts) - 1):
        a, b = starts[i], starts[i + 1]
        kn = np.where(known[a:b])[0] + a
        if not len(kn):
            has.append(False)
            shots.append((a, b))
            continue
        f0 = kn[0]
        fl = fy[a:b][np.isfinite(fy[a:b])]
        if len(fl) >= 10:
            oy = -float(np.percentile(fl, 5))
        elif prev is not None:
            oy = prev[1] - ref[f0, 1]
        else:
            oy = leg_actor + 0.05 - ref[f0, 1]
        oxz = (np.zeros(2) if prev is None else prev[[0, 2]]) - ref[f0, [0, 2]]
        O[a:b] = [oxz[0], oy, oxz[1]]
        prev = ref[kn[-1]] + O[a]
        has.append(True)
        shots.append((a, b))
    # character-less shots: the camera starts where the nearest shot's camera was
    hs = np.array(has)
    if hs.any():
        for i, (a, b) in enumerate(shots):
            if hs[i]:
                continue
            j = min(np.where(hs)[0], key=lambda j: abs(j - i))
            fa, fb = shots[j]
            ref_f = fb - 1 if j < i else fa
            O[a:b] = (C_w[ref_f] + O[ref_f]) - C_w[a if j < i else b - 1]
    return O


ANCHORS = [((L_HIP, R_HIP), ("leftUpperLeg", "rightUpperLeg")),
           ((L_SH, R_SH), ("leftUpperArm", "rightUpperArm")),
           ((L_EYE, R_EYE), ("leftEye", "rightEye")),
           ((L_KNEE, R_KNEE), ("leftLowerLeg", "rightLowerLeg")),
           ((L_HIP,), ("leftUpperLeg",)), ((R_HIP,), ("rightUpperLeg",)),
           ((L_SH,), ("leftUpperArm",)), ((R_SH,), ("rightUpperArm",)),
           ((L_KNEE,), ("leftLowerLeg",)), ((R_KNEE,), ("rightLowerLeg",)),
           ((L_ANK,), ("leftFoot",)), ((R_ANK,), ("rightFoot",)),
           ((L_EL,), ("leftLowerArm",)), ((R_EL,), ("rightLowerArm",)),
           ((L_WR,), ("leftHand",)), ((R_WR,), ("rightHand",))]


def place_root(skel, Q, Xw, js_ok, scale, fps, shot_of, tau_s=0.3):
    """The avatar's root (hips) position, so that a part of the body that IS
    seen sits exactly where the actor's does: the hip joints when they are in
    the picture, otherwise the shoulders, the eyes, the knees, a single foot, a
    hand... (with its pose as it is, held parts included). With nothing seen,
    the character stays where it was. Switching from one reference part to
    another (or coming back into view) is eased over tau_s."""
    N = len(Q)
    pos0, _ = avatar_fk(skel, Q, np.zeros((N, 3)))
    bi = {b: i for i, b in enumerate(skel.bones)}
    cand = np.full((N, 3), np.nan)
    typ = -np.ones(N, np.int64)
    for i, (aj, ab) in enumerate(ANCHORS):
        if not all(b in bi for b in ab):
            continue
        m = js_ok[:, list(aj)].all(1) & (typ < 0) & np.isfinite(Xw[:, list(aj)]).all((1, 2))
        if not m.any():
            continue
        actor = scale * Xw[:, list(aj)].mean(1)
        off = pos0[:, [bi[b] for b in ab]].mean(1)
        cand[m] = actor[m] - off[m]
        typ[m] = i
    root = np.zeros((N, 3))
    seen = np.where(typ >= 0)[0]
    if not len(seen):
        return root
    first, last = seen[0], seen[-1]
    root[:first + 1] = cand[first]
    decay = float(np.exp(-1.0 / max(tau_s * fps, 1e-6)))
    o = np.zeros(3)
    prev = first
    for f in seen[1:]:
        if f == prev + 1 and typ[f] != typ[prev]:
            o = root[prev] - cand[f]   # another reference part: no jump
        elif f == prev + 1:
            o = o * decay
        else:
            o = np.zeros(3)            # after a gap: where it is seen now
        root[f] = cand[f] + o
        prev = f
    # a person does not move faster than running pace: a reference part that
    # jumps (a mis-detection) is caught up with over the following frames
    vmax = 6.0 * scale / fps
    for i in range(1, len(seen)):
        f, g = seen[i - 1], seen[i]
        if g != f + 1 or shot_of[g] != shot_of[f]:
            continue
        d = root[g] - root[f]
        n_ = np.linalg.norm(d)
        if n_ > vmax:
            root[g] = root[f] + d / n_ * vmax
    # nothing of the character seen: in-between motion from where it was last
    # seen to where it is seen next (after the last sighting it stays)
    root[last + 1:] = root[last]
    for i in range(len(seen) - 1):
        a, b = seen[i], seen[i + 1]
        if b - a <= 1:
            continue
        va = root[a] - root[a - 1] if a - 1 >= first and typ[a - 1] >= 0 else np.zeros(3)
        if b - a > GAP_HOLD_S * fps:
            # out of the picture for a while: stays where it was last seen and
            # moves to where it is seen next at walking pace (at least MOVE_S)
            dist = float(np.linalg.norm(root[b] - root[a]))
            T = int(round(max(MOVE_S, dist / (1.5 * scale)) * fps))
            if T < b - a:
                root[a + 1:b - T + 1] = root[a]
                a, va = b - T, np.zeros(3)
        L = b - a
        vb = root[b + 1] - root[b] if b + 1 <= last and typ[b + 1] >= 0 else np.zeros(3)
        k = float(np.clip(0.15 * fps / L, 0.0, 1.0))
        t = ((np.arange(a + 1, b) - a) / L)[:, None]
        root[a + 1:b] = ((2 * t ** 3 - 3 * t ** 2 + 1) * root[a] + (t ** 3 - 2 * t ** 2 + t) * va * L * k +
                         (-2 * t ** 3 + 3 * t ** 2) * root[b] + (t ** 3 - t ** 2) * vb * L * k)
    # (and the in-between motion keeps to that pace too, within a shot)
    for f in range(first + 1, N):
        if shot_of[f] != shot_of[f - 1]:
            continue
        d = root[f] - root[f - 1]
        n_ = np.linalg.norm(d)
        if n_ > vmax:
            root[f] = root[f - 1] + d / n_ * vmax
    return root


def _foot_lows(skel, Q, root):
    pos, rot = avatar_fk(skel, Q, root)
    bi = {b: i for i, b in enumerate(skel.bones)}
    rp = skel.rest_pos
    lows, xz = [], []
    for side in ("left", "right"):
        fi = bi[side + "Foot"]
        heel_off = np.array([0, -rp[side + "Foot"][1], -0.04])
        toe_off = rp[side + "Toes"] - rp[side + "Foot"] + np.array([0, -rp[side + "Toes"][1], 0.04]) \
            if (side + "Toes") in rp else np.array([0, -rp[side + "Foot"][1], 0.15])
        heel = pos[:, fi] + np.einsum("nij,j->ni", rot[:, fi], heel_off)
        toe = pos[:, fi] + np.einsum("nij,j->ni", rot[:, fi], toe_off)
        lows.append(np.minimum(heel[:, 1], toe[:, 1]))
        xz.append((heel + toe)[:, [0, 2]] / 2)
    return np.stack(lows, 1), np.stack(xz, 1)


def foot_contacts(skel, Q, root, fps, feet_ok):
    """(N,2) planted feet: low (near the lowest the feet get around then), slow,
    and seen (a foot outside the picture is not known to be planted)."""
    N = len(Q)
    low, xz = _foot_lows(skel, Q, root)
    win = int(1.5 * fps)
    floor = np.zeros(N)
    m = low.min(1)
    for f in range(N):
        a, b = max(0, f - win), min(N, f + win + 1)
        ok = feet_ok[a:b].any(1)
        vals = m[a:b][ok] if ok.any() else m[a:b]
        floor[f] = np.percentile(vals, 10)
    sp = np.zeros((N, 2))
    sp[1:] = np.linalg.norm(np.diff(xz, axis=0), axis=-1) * fps
    contact = (low - floor[:, None] < 0.035) & (sp < 0.6) & feet_ok
    k = max(1, int(round(0.05 * fps)))
    for s_ in (0, 1):
        c = contact[:, s_].astype(float)
        contact[:, s_] = np.convolve(c, np.ones(2 * k + 1) / (2 * k + 1), mode="same") > 0.5
    return contact


def settle_on_floor(skel, Q, root, contacts, fps, shot_of, stand_y=None):
    """Per shot, one vertical shift of the whole scene (character and camera
    together, so the picture is unchanged) that puts the planted feet on the
    floor (y = 0). Returns the shift per frame."""
    low, _ = _foot_lows(skel, Q, root)
    shift = np.zeros(len(root))
    corr = np.where(contacts, -low, np.inf).min(1)
    for (a, b) in runs(np.ones(len(root), bool), shot_of):
        c = corr[a:b]
        c = c[np.isfinite(c)]
        if len(c) >= 3:
            shift[a:b] = float(np.median(c))
        elif stand_y is not None:
            # no foot seen on the floor in this shot (feet below the picture):
            # the feet are still on the floor, so the hips are at standing height
            shift[a:b] = stand_y - float(np.median(root[a:b, 1]))
    return shift


def proportion_report(L):
    """The measured proportions, for the project folder and the app: each
    limb bone's length in metres (the body model's scale) and as a share of
    the leg length (hip to ankle)."""
    names = {(L_SH, L_EL): "upper_arm", (L_EL, L_WR): "forearm", (L_HIP, L_KNEE): "thigh", (L_KNEE, L_ANK): "shin",
             (L_SH, R_SH): "shoulder_width", (L_HIP, R_HIP): "hip_width", (L_SH, L_HIP): "torso_side"}
    out = {}
    for (a, b), nm in names.items():
        key = (lift.JI[a], lift.JI[b])
        if key in lift.EDGES:
            v = float(L[lift.EDGES.index(key)])
            if np.isfinite(v):
                out[nm] = round(v, 4)
    leg = out.get("thigh", 0) + out.get("shin", 0)
    if leg > 0:
        out["share_of_leg"] = {k: round(v / leg, 3) for k, v in list(out.items()) if isinstance(v, float)}
    return out


def floor_follow(skel, Q, root, contacts, fps, shot_of, sigma_s=0.08, max_m=0.2):
    """Per frame vertical correction of the character (not the camera) that
    puts the lowest planted foot exactly on the floor; between contacts (a
    step, a jump) it moves smoothly from one to the next."""
    low, _ = _foot_lows(skel, Q, root)
    N = len(root)
    out = np.zeros(N)
    want = np.where(contacts, -low, -np.inf).max(1)  # the lowest planted foot
    for (a, b) in runs(np.ones(N, bool), shot_of):
        w = want[a:b]
        ok = np.isfinite(w)
        if ok.sum() < 2:
            continue
        t = np.arange(b - a)
        c = np.interp(t, t[ok], w[ok])
        if b - a >= 3:
            c = gaussian_smooth(c[:, None], sigma_s * fps)[:, 0]
        out[a:b] = np.clip(c, -max_m, max_m)
    return out


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


def _min_run(mask, fps, shot_of, min_s=0.15):
    """Drop True-runs shorter than min_s seconds (brief glimpses)."""
    out = mask.copy()
    k = max(3, int(round(min_s * fps)))
    for (a, b) in runs(mask, shot_of):
        if b - a < k:
            out[a:b] = False
    return out


def joint_seen(uv, conf, valid, W, H, fps, shot_of, hidden=None, weak=None, both=None, edge_ok=None):
    """(N,33) joint clearly inside the picture (not on its edge), detected, and
    not behind something (hidden: see occluded_joints).
    With `weak` (a lower bar), context from both directions: a stretch of weak
    detections counts as seen when it runs without a break into a stretch
    where the joint is clearly seen, before or after it (legs at the top of
    the picture just before the whole body comes into view are the same legs
    seen clearly a moment later)."""
    m = 0.015 * max(W, H)
    inside = (uv[..., 0] > m) & (uv[..., 0] < W - m) & (uv[..., 1] > m) & (uv[..., 1] < H - m)
    if edge_ok is not None:
        inside |= edge_ok & (uv[..., 0] > 0) & (uv[..., 0] < W) & (uv[..., 1] > 0) & (uv[..., 1] < H)
    ok = inside & valid[:, None]
    if hidden is not None:
        ok &= ~hidden
    s = ok & (conf >= 0.5)
    if both is not None:  # both models fairly sure (skirt, hair over a visible leg)
        s |= ok & both
    if weak is not None:
        w = ok & (weak | s)
        for j in range(s.shape[1]):
            for (a, b) in runs(w[:, j], shot_of):
                if s[a:b, j].any():
                    s[a:b, j] = True
    # one or two frames missing inside a sighting (a flicker of the detector)
    # do not break it up
    gap = max(1, int(round(0.05 * fps)))
    for j in range(s.shape[1]):
        col = s[:, j]
        on = np.where(col)[0]
        if len(on) < 2:
            continue
        d = np.diff(on)
        for k in np.where((d > 1) & (d <= gap + 1))[0]:
            a, b = on[k], on[k + 1]
            if shot_of[a] == shot_of[b] and (inside[a:b, j] & valid[a:b]).all():
                col[a:b] = True
    return np.stack([_min_run(s[:, j], fps, shot_of) for j in range(s.shape[1])], 1)


def raw_scores(wb_sync, vis, W, H, N):
    """(N,33) the whole-body model's own score for each body joint (labels
    already matched to MediaPipe's); near the picture edge only as far as
    MediaPipe also sees the joint (the model puts joints outside the picture on
    its edge)."""
    from .detect import WB_TO_MP
    out = np.zeros((N, 33))
    if wb_sync is None:
        return out
    wb2d, wbs = wb_sync
    edge = 0.03 * max(W, H)
    for m, w in WB_TO_MP.items():
        if m < 11:
            continue
        p = wb2d[:, w]
        near = (p[:, 0] < edge) | (p[:, 0] > W - edge) | (p[:, 1] < edge) | (p[:, 1] > H - edge)
        out[:, m] = np.where(near, np.minimum(wbs[:, w], vis[:, m]), wbs[:, w])
    return out


LIMB_PAIRS = [(L_HIP, L_KNEE), (L_KNEE, L_ANK), (R_HIP, R_KNEE), (R_KNEE, R_ANK),
              (L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR)]


# segments of capture.SEGMENTS as MediaPipe joints (limbs: proximal, distal)
SEG_MP = [(L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR), (L_HIP, L_KNEE), (L_KNEE, L_ANK),
          (R_HIP, R_KNEE), (R_KNEE, R_ANK), (L_SH, R_SH, R_HIP, L_HIP)]
_WB_OF_MP = {L_SH: 5, R_SH: 6, L_EL: 7, R_EL: 8, L_WR: 9, R_WR: 10, L_HIP: 11, R_HIP: 12,
             L_KNEE: 13, R_KNEE: 14, L_ANK: 15, R_ANK: 16}


def infer_from_markers(uv, conf, strong, markers, fracs, valid, W, H, wb2d=None, min_px=8.0, exp_len=None,
                       too_long=1.2, too_short=0.6, covered=None):
    """The hidden end of a limb from one clearly seen end and the trackers that
    ride the limb's line (at known fractions of its length): the line through
    the seen end and the furthest visible tracker, carried on to the full
    length (tracker at fraction f of the way -> the end is at 1/f of that
    distance). Trackers are matched to the limb by position (left/right labels
    can swap). Returns uv, conf and the (N,33) joints placed this way."""
    N = len(uv)
    uv, conf = uv.copy(), conf.copy()
    out = np.zeros((N, 33), bool)
    fr = np.asarray(fracs, float)
    mirror = [2, 3, 0, 1, 6, 7, 4, 5]
    inside = (markers[..., 0] >= 0) & (markers[..., 0] < W) & (markers[..., 1] >= 0) & (markers[..., 1] < H)
    for k, (a, b) in enumerate(SEG_MP[:8]):
        for t in np.where(valid & (strong[:, a] ^ strong[:, b]))[0]:
            s_end, h_end = (a, b) if strong[t, a] else (b, a)
            S = uv[t, s_end]
            best = None
            cands = (k, mirror[k])
            if wb2d is not None:  # the tracker set whose limb has its joint at the seen end
                from .capture import SEGMENTS
                pos = 0 if s_end == a else 1
                cands = sorted(cands, key=lambda c: np.linalg.norm(wb2d[t, SEGMENTS[c][pos]] - S))[:1]
            for c in cands:
                ok = inside[t, c] & np.isfinite(markers[t, c, :, 0])
                if not ok.any():
                    continue
                mk = markers[t, c]
                # the tracker line must run through (near) the seen end
                idx = np.where(ok)[0]
                f_far = fr[idx]
                # the visible tracker furthest from the seen end (the longest lever;
                # trackers that slid off the person were dropped in capture)
                if s_end == a:
                    m = idx[np.argmax(f_far)]
                    d = mk[m] - S
                    if np.linalg.norm(d) < min_px:
                        continue
                    est = S + d / fr[m]
                else:
                    m = idx[np.argmin(f_far)]
                    d = mk[m] - S
                    if np.linalg.norm(d) < min_px:
                        continue
                    est = S + d / (1 - fr[m])
                # consistency: other visible trackers lie on that line
                line = est - S
                Ln = np.linalg.norm(line)
                # ...and the limb has a length this bone can have in the picture at
                # this distance (never longer, not absurdly short): trackers that
                # slipped onto clothing that moves on its own fail this
                if exp_len is not None and np.isfinite(exp_len[t, k]) and exp_len[t, k] > 0:
                    if Ln > too_long * exp_len[t, k] or Ln < too_short * exp_len[t, k]:
                        continue
                off = 0.0
                if len(idx) > 1:
                    u_ = line / Ln
                    rel = mk[idx] - S
                    off = float(np.abs(rel[:, 0] * u_[1] - rel[:, 1] * u_[0]).max()) / Ln
                if off > 0.15:
                    continue
                if best is None or off < best[0]:
                    best = (off, est)
            if best is None:
                continue
            # only an end that is really out of view: outside the picture, or with
            # something in front of it (under clothing the detectors know better
            # than trackers riding cloth that moves on its own)
            e_ = best[1]
            outside = e_[0] < 0 or e_[0] >= W or e_[1] < 0 or e_[1] >= H
            if not outside and (covered is None or not covered[t, h_end]):
                continue
            uv[t, h_end] = best[1]
            conf[t, h_end] = max(conf[t, h_end], 0.6)
            out[t, h_end] = True
    return uv, conf, out


def bones_from_segments(uv, conf, seen0, markers, fracs, valid, W, H, wb2d, L2d, dz, min_px=6.0,
                        line_tol=0.15, agree=0.2, covered=None):
    """A limb bone is known when one of its two joints and one of the trackers
    riding it are seen (the line from the joint through the tracker is the
    bone's direction in the picture), or when two of its trackers are seen
    (their line, and where the joints lie on it at the trackers' fractions).
    Something cutting across the bone (a bar, a strand of hair, the edge of the
    picture) then hides only part of it, not the whole limb. The hidden end is
    put on that line at the bone's length in the picture: its measured length,
    foreshortened by how far it reaches towards or away from the camera
    (depth from the motion), or - when the trackers were put on the full bone -
    simply where their fractions say.

    seen0 (N,33): joints clearly seen. markers (N,8,F,2) px (nan: lost).
    L2d (N,8): the bone's full length in the picture when square-on (px).
    dz (N,8): depth of the distal joint minus the proximal one, as a share of
    the bone's length (nan: unknown). Arms: only where the hidden end is
    outside the picture or behind something that is not the person (covered):
    an arm behind the body has its trackers slide onto the chest.
    Returns uv, conf, known (N,8) bones known this way, placed (N,33) joints
    placed this way."""
    from .capture import SEGMENTS
    N = len(uv)
    uv, conf = uv.copy(), conf.copy()
    known = np.zeros((N, 8), bool)
    placed = np.zeros((N, 33), bool)
    fr = np.asarray(fracs, float)
    mirror = [2, 3, 0, 1, 6, 7, 4, 5]
    fin = np.isfinite(markers[..., 0])
    sn = seen0.copy()  # a joint placed this way counts for the next bone down the limb
    inside = fin & (markers[..., 0] >= 0) & (markers[..., 0] < W) & (markers[..., 1] >= 0) & (markers[..., 1] < H)
    for k, (a, b) in enumerate(SEG_MP[:8]):
        if k not in SEGMENT_LIMBS:
            continue
        todo = np.where(valid & ~(sn[:, a] & sn[:, b]) & inside[:, [k, mirror[k]]].any((1, 2)))[0]
        for t in todo:
            Lp = L2d[t, k]
            if not np.isfinite(Lp) or Lp < 2 * min_px:
                continue
            q = dz[t, k] if np.isfinite(dz[t, k]) else 0.0
            # the bone's length in the picture, foreshortened by its depth
            Lfs = Lp * np.sqrt(max(1.0 - min(q * q, 0.94), 0.06))
            # which tracker set rides this limb: the one whose limb has its
            # joints where this limb's joints are (labels can swap)
            ends = [j for j in (a, b) if sn[t, j]] or [a, b]
            def dist(c):
                return sum(np.linalg.norm(wb2d[t, SEGMENTS[c][0 if j == a else 1]] - uv[t, j]) for j in ends)
            c = min((k, mirror[k]), key=dist)
            ok = inside[t, c]
            if not ok.any():
                continue
            idx = np.where(ok)[0]
            mk = markers[t, c, idx]
            f_ = fr[idx]
            if sn[t, a] or sn[t, b]:
                s_end, h_end = (a, b) if sn[t, a] else (b, a)
                S = uv[t, s_end]
                lever = f_ if s_end == a else 1 - f_
                m = int(np.argmax(lever))
                d = mk[m] - S
                dl = float(np.linalg.norm(d))
                if dl < max(min_px, 0.12 * Lfs):
                    continue
                u = d / dl
                rel = mk - S
                if len(idx) > 1:
                    off = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0]) / dl
                    if off.max() > line_tol or (rel @ u).min() < -0.1 * dl:
                        continue
                if dl > 1.25 * Lp:  # a tracker beyond the bone's end: it slid onto something else
                    continue
                L_tr = dl / lever[m]
                L_use = L_tr if 0.8 * Lfs <= L_tr <= 1.2 * Lfs else max(Lfs, 1.02 * dl)
                est = S + u * L_use
                det = uv[t, h_end]
                if k < 4 and not (est[0] < 0 or est[0] >= W or est[1] < 0 or est[1] >= H or
                                  (covered is not None and covered[t, h_end])):
                    continue
                if conf[t, h_end] >= 0.2 and np.linalg.norm(det - est) < agree * L_use:
                    est = 0.5 * (est + det)  # the detectors agree: between the two
                uv[t, h_end] = est
                conf[t, h_end] = max(conf[t, h_end], 0.6)
                placed[t, h_end] = True
                sn[t, h_end] = True
                known[t, k] = True
            else:
                # no joint seen: two trackers far enough apart give the line,
                # and one of the (unsure) joint detections must lie on it
                if k < 4 or len(idx) < 2 or np.ptp(f_) < 0.25:
                    continue
                lo, hi = int(np.argmin(f_)), int(np.argmax(f_))
                d = mk[hi] - mk[lo]
                dl = float(np.linalg.norm(d))
                if dl < min_px:
                    continue
                u = d / dl
                L_tr = dl / (f_[hi] - f_[lo])
                if not (0.5 * Lfs <= L_tr <= 1.25 * Lp):
                    continue
                pa = mk[lo] - u * L_tr * f_[lo]
                pb = pa + u * L_tr
                ea = np.linalg.norm(uv[t, a] - pa) if conf[t, a] >= 0.15 else np.inf
                eb = np.linalg.norm(uv[t, b] - pb) if conf[t, b] >= 0.15 else np.inf
                if min(ea, eb) > agree * L_tr:
                    continue
                if len(idx) > 2:
                    rel = mk - pa
                    if (np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0]) / L_tr).max() > line_tol:
                        continue
                for j, p_ in ((a, pa), (b, pb)):
                    uv[t, j] = p_
                    conf[t, j] = max(conf[t, j], 0.6)
                    placed[t, j] = True
                    sn[t, j] = True
                known[t, k] = True
    return uv, conf, known, placed


def rigid_track(uv, raw, wb2d, wbs, segM, segN, valid, fps, anchor=0.65, half_life_s=0.4,
                min_pts=6, agree=0.2):
    """Body segments as rigid pieces. Capture tracked texture points along every
    limb and the torso, frame to frame, and kept the points that move together
    (segM: each segment's rotation + translation + scale). Here every joint
    that is clearly detected (score >= anchor) is carried along the motion of
    a segment it belongs to, forwards and backwards in time, into the frames
    around it where the detectors are unsure; the carried position counts less
    the further it is from the clear sighting (half-life). Where a weak
    detection agrees with the carried position (within `agree` of the
    segment's length) the joint is placed between the two and counts as seen.
    Returns uv and (N,33) the joints confirmed this way.
    Segments are matched to the limbs by position, not by left/right label (the
    detectors' labels can swap from frame to frame)."""
    N = len(uv)
    uv = uv.copy()
    ok_out = np.zeros((N, 33), bool)
    K = len(SEG_MP)
    # which captured segment moves each limb from frame t-1 to t
    seg_of = -np.ones((N, K), int)
    wbp = {j: wb2d[:, w] for j, w in _WB_OF_MP.items()}
    for t in range(1, N):
        if not (valid[t] and valid[t - 1]):
            continue
        for k, js_ in enumerate(SEG_MP):
            if len(js_) != 2:
                if segN[t, 8] >= min_pts:
                    seg_of[t, k] = 8
                continue
            a, b = js_
            L = np.linalg.norm(uv[t - 1, b] - uv[t - 1, a]) + 1e-6
            best, bd = -1, 0.35
            for c, (ca, cb) in enumerate(SEG_MP[:8]):
                if segN[t, c] < min_pts:
                    continue
                pa, pb = wbp[ca][t - 1], wbp[cb][t - 1]
                d = min(np.linalg.norm(uv[t - 1, a] - pa) + np.linalg.norm(uv[t - 1, b] - pb),
                        np.linalg.norm(uv[t - 1, a] - pb) + np.linalg.norm(uv[t - 1, b] - pa)) / L
                if d < bd:
                    best, bd = c, d
            seg_of[t, k] = best
    joints = sorted({j for js_ in SEG_MP for j in js_})
    decay = 0.5 ** (1.0 / max(half_life_s * fps, 1.0))

    def carry(direction):
        est = np.full((N, 33, 2), np.nan)
        rel = np.zeros((N, 33))
        order = range(N) if direction > 0 else range(N - 1, -1, -1)
        for j in joints:
            ks = [k for k, js_ in enumerate(SEG_MP) if j in js_]
            cur, r = None, 0.0
            prev_t = None
            for t in order:
                if not valid[t]:
                    cur, r = None, 0.0
                    prev_t = t
                    continue
                if raw[t, j] >= anchor:
                    cur, r = uv[t, j].copy(), 1.0
                elif cur is not None and prev_t is not None:
                    tm = t if direction > 0 else prev_t   # the step between t and prev_t
                    cand = [k for k in ks if seg_of[tm, k] >= 0]
                    if cand:
                        k = max(cand, key=lambda k_: segN[tm, seg_of[tm, k_]])
                        A = segM[tm, seg_of[tm, k]]
                        if direction > 0:
                            cur = A[:, :2] @ cur + A[:, 2]
                        else:  # inverse motion
                            cur = np.linalg.solve(A[:, :2], cur - A[:, 2])
                        r *= decay
                    else:
                        cur, r = None, 0.0
                if cur is not None:
                    est[t, j], rel[t, j] = cur, r
                prev_t = t
        return est, rel
    Ef, rf = carry(1)
    Eb, rb = carry(-1)
    w = rf + rb
    E = (np.nan_to_num(Ef) * rf[..., None] + np.nan_to_num(Eb) * rb[..., None]) / np.maximum(w, 1e-9)[..., None]
    rel = np.maximum(rf, rb)
    # segment length (for "agrees")
    seg_len = np.zeros((N, 33))
    for js_ in SEG_MP[:8]:
        a, b = js_
        L = np.linalg.norm(uv[:, b] - uv[:, a], axis=1)
        for j in js_:
            seg_len[:, j] = np.maximum(seg_len[:, j], L)
    weak = (raw < anchor) & (raw >= 0.25) & (rel >= 0.35) & valid[:, None]
    dist = np.linalg.norm(E - uv, axis=-1)
    good = weak & (dist < agree * np.maximum(seg_len, 1.0))
    wgt = (rel * (1 - raw))[..., None]
    uv = np.where(good[..., None], uv * (1 - wgt) + E * wgt, uv)
    ok_out[good] = True
    return uv, ok_out


def cut_off_joints(uv, P, blen, focal, cx, cy, W, H, edge_frac=0.15, short=0.7):
    """(N,33) joints of a limb that leaves the picture. Close to the camera the
    detectors squeeze a limb that runs out of the picture into it: the end
    joint is put near the edge, and the limb looks far shorter than a bone of
    its length can look at that distance. A limb that short, ending near the
    edge it is heading for, is cut off (not pointing at the camera): its end
    joint is not seen."""
    N = len(uv)
    out = np.zeros((N, 33), bool)
    ex, ey = edge_frac * W, edge_frac * H
    for a, b in LIMB_PAIRS:
        L = blen.get((a, b))
        if not L:
            continue
        z = np.where(np.isfinite(P[:, a, 2]), P[:, a, 2], np.nan)
        exp = focal * L / np.maximum(z, 0.05)
        d = uv[:, b] - uv[:, a]
        l2 = np.linalg.norm(d, axis=1)
        p = uv[:, b]
        # near the edge the limb points towards
        near = ((p[:, 0] < ex) & (d[:, 0] < 0)) | ((p[:, 0] > W - ex) & (d[:, 0] > 0)) | \
               ((p[:, 1] < ey) & (d[:, 1] < 0)) | ((p[:, 1] > H - ey) & (d[:, 1] > 0))
        out[:, b] |= near & np.isfinite(exp) & (l2 < short * exp)
    # what hangs below a cut-off joint is cut off as well
    for a, b in ((L_KNEE, L_ANK), (R_KNEE, R_ANK), (L_EL, L_WR), (R_EL, R_WR)):
        out[:, b] |= out[:, a]
    for a, bs in ((L_ANK, (L_HEEL, L_FOOT)), (R_ANK, (R_HEEL, R_FOOT))):
        for b in bs:
            out[:, b] |= out[:, a]
    return out


def occluded_joints(seg, vis):
    """(N,33) joints behind something: the person mask says the pixel there is
    not the person (something is in front of it) and the pose model agrees it
    is not clearly visible. seg = -1 where no mask was available."""
    if seg is None:
        return None
    seg = np.asarray(seg, np.float64)
    return (seg >= 0) & (seg < 0.25) & (vis < 0.6)


LIMB_BONES = {}
for _s, _S in (("left", "L"), ("right", "R")):
    _l = (lambda a, b: a if _S == "L" else b)
    LIMB_BONES.update({f"{_s}UpperLeg": (_l(L_HIP, R_HIP), _l(L_KNEE, R_KNEE)),
                       f"{_s}LowerLeg": (_l(L_KNEE, R_KNEE), _l(L_ANK, R_ANK)),
                       f"{_s}UpperArm": (_l(L_SH, R_SH), _l(L_EL, R_EL)),
                       f"{_s}LowerArm": (_l(L_EL, R_EL), _l(L_WR, R_WR))})


def just_outside(uv, raw, W, H, reach=0.25, min_score=0.5):
    """(N,33) joints just outside the picture that the whole-body model still
    places confidently (a limb entering the picture from that side)."""
    x, y = uv[..., 0], uv[..., 1]
    out = (x < 0) | (x > W) | (y < 0) | (y > H)
    near = (x > -reach * W) & (x < W * (1 + reach)) & (y > -reach * H) & (y < H * (1 + reach))
    return out & near & (raw >= min_score)


def bone_seen(skel, js, hvalid, face_seen, fps, shot_of, near_out=None, seg_known=None):
    """(N,B): every joint defining the bone is seen. Bones the solver never
    drives (shoulders, eyes, jaw...) count as seen (they stay as they are).
    With near_out: a limb with one end seen and the other just outside the
    picture (a thigh coming in from the top edge down to a visible knee, a
    shin running out of the bottom) counts as seen: the part in the picture
    shows which way it points. With seg_known (N,8): a limb bone whose
    direction is known from one joint and a tracker on it (or two trackers)
    counts as seen (see bones_from_segments)."""
    N = len(js)
    out = np.ones((N, len(skel.bones)), bool)
    head = face_seen if face_seen is not None else js[:, NOSE] & (js[:, L_EAR] | js[:, R_EAR])
    for i, b in enumerate(skel.bones):
        side = 0 if b.startswith("left") else 1
        wr = L_WR if side == 0 else R_WR
        if b.endswith("Hand") or any(f in b for f in FINGERS):
            m = hvalid[:, side] & js[:, wr]
        elif b == "head":
            m = head
        elif b == "neck":
            m = js[:, L_SH] & js[:, R_SH] & head
        elif b in BONE_LANDMARKS:
            m = js[:, BONE_LANDMARKS[b]].all(1)
            if near_out is not None and b in LIMB_BONES:
                a_, d_ = LIMB_BONES[b]
                m = m | (js[:, d_] & near_out[:, a_]) | (js[:, a_] & near_out[:, d_])
            if seg_known is not None and b in LIMB_BONES:
                m = m | seg_known[:, SEG_MP.index(LIMB_BONES[b])]
        else:
            continue
        out[:, i] = _min_run(m, fps, shot_of)
    return out


def neutral_pose(skel):
    """The A pose (arms 45 degrees down from the T-pose, everything else at
    rest): what every bone shows until it is first seen."""
    q = np.zeros((len(skel.bones), 4))
    q[:, 3] = 1.0
    for side, sg in (("left", -1.0), ("right", 1.0)):
        b = side + "UpperArm"
        if skel.has(b):
            a = np.radians(45) * sg
            q[skel.bones.index(b)] = [0, 0, np.sin(a / 2), np.cos(a / 2)]
    return q


def _qmul(a, b):
    x1, y1, z1, w1 = np.moveaxis(a, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], -1)


def _qinv(q):
    return q * np.array([-1, -1, -1, 1.0])


def _qlog(q):
    """unit quaternion -> rotation vector (shortest way)."""
    q = q * np.sign(q[..., 3:4] + 1e-12)
    v = q[..., :3]
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    ang = 2 * np.arctan2(n, q[..., 3:4])
    return np.where(n > 1e-12, v / np.maximum(n, 1e-12) * ang, 2 * v)


def _qexp(r):
    a = np.linalg.norm(r, axis=-1, keepdims=True)
    ax = np.where(a > 1e-12, r / np.maximum(a, 1e-12), 0.0)
    return np.concatenate([ax * np.sin(a / 2), np.cos(a / 2)], -1)


GAP_HOLD_S = 1.5  # gaps longer than this hold the last seen pose ...
MOVE_S = 0.6      # ... and move into the next seen pose over this time


def keyframe_fill(target, sb, fps, neutral, nb):
    """One bone over the clip: keyframes only where it is seen, in-between
    motion everywhere else. Gaps between two sightings are filled with a smooth
    curve (Hermite in rotation space) that leaves the last seen pose and
    arrives at the next one with the speeds the bone had there (the speeds
    count less the longer the gap). A long gap holds the last seen pose and
    moves into the next one over its last MOVE_S. Before the bone is first seen it is in the
    A pose (neutral) and eases into the first seen pose over nb frames; after
    the last sighting it keeps that pose."""
    from .naturalize import slerp_many
    N = len(target)
    col = target.copy()
    on = np.where(sb)[0]
    first, last = on[0], on[-1]
    col[last + 1:] = target[last]
    f = first
    while f < last:
        if sb[f + 1]:
            f += 1
            continue
        a = f
        b = a + 1
        while not sb[b]:
            b += 1
        # a long gap (the character out of the picture for a while): the pose
        # stays as it was last seen and only moves into the next seen pose over
        # the last MOVE_S before it (no slow drift through made-up angles)
        hold = b - a - int(round(MOVE_S * fps))
        if b - a > GAP_HOLD_S * fps and hold > 0:
            col[a + 1:a + hold + 1] = target[a]
            a = a + hold
        L = b - a
        qa, qb = target[f], target[b]
        r = _qlog(_qmul(_qinv(qa), qb))
        wa = _qlog(_qmul(_qinv(target[a - 1]), qa)) if a == f and a >= 1 and sb[a - 1] else np.zeros(3)
        wb = _qlog(_qmul(_qinv(qb), target[b + 1])) if b + 1 < N and sb[b + 1] else np.zeros(3)
        k = float(np.clip(0.6 * fps / L, 0.0, 1.0))
        t = (np.arange(a + 1, b) - a) / L
        h10, h01, h11 = t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
        pth = h10[:, None] * (wa * L * k) + h01[:, None] * r + h11[:, None] * (wb * L * k)
        col[a + 1:b] = _qmul(np.broadcast_to(qa, (L - 1, 4)), _qexp(pth))
        f = b
    if first > 0:
        col[:first] = neutral
        for k_ in range(nb):
            g = first + k_
            if g >= N:
                break
            tt = (k_ + 1) / nb
            tt = tt * tt * (3 - 2 * tt)
            col[g] = slerp_many(neutral, col[g], np.array([tt]))[0]
    return col


CHILD_OF = {"leftUpperArm": "leftLowerArm", "leftLowerArm": "leftHand", "rightUpperArm": "rightLowerArm",
            "rightLowerArm": "rightHand", "leftUpperLeg": "leftLowerLeg", "leftLowerLeg": "leftFoot",
            "rightUpperLeg": "rightLowerLeg", "rightLowerLeg": "rightFoot"}


def _align_batch(a, b):
    """(n,3,3) minimal rotations taking unit vectors a onto b."""
    a, b = normalize(a), normalize(b)
    v = np.cross(a, b)
    c = np.sum(a * b, -1)
    K = np.zeros(a.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -v[..., 2], v[..., 1], -v[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = v[..., 2], -v[..., 1], v[..., 0]
    k = 1.0 / np.maximum(1.0 + c, 1e-6)
    return np.eye(3) + K + np.einsum("...ij,...jk->...ik", K, K) * k[..., None, None]


def _ramp_weights(mask, seen_col, shot_of, nr):
    """Per frame weight (0..1) inside runs of `mask`: 1 in the middle, easing
    from / to the frames just outside a run where the bone is seen."""
    N = len(mask)
    w = np.zeros(N)
    for (a, e) in runs(mask, shot_of):
        k = np.arange(a, e)
        wl = np.clip((k - a + 1) / nr, 0, 1) if (a > 0 and seen_col[a - 1] and shot_of[a - 1] == shot_of[a]) else 1.0
        wr = np.clip((e - k) / nr, 0, 1) if (e < N and seen_col[e] and shot_of[e] == shot_of[e - 1]) else 1.0
        t = np.minimum(wl, wr)
        w[a:e] = t * t * (3 - 2 * t)
    return w


def _set_world(Q, skel, b, Gnew, w):
    """Give bone b the world rotation Gnew (n,3,3) with weight w (n,), keeping
    its parent; Q (N,B,4) is changed in place where w > 0."""
    idx = {x: i for i, x in enumerate(skel.bones)}
    i = idx[b]
    m = w > 0
    if not m.any():
        return
    _, rot = avatar_fk(skel, Q[m], np.zeros((int(m.sum()), 3)))
    p = skel.parent[b]
    Gp = rot[:, idx[p]] if p is not None else np.tile(np.eye(3), (int(m.sum()), 1, 1))
    Gold = rot[:, i]
    q_old, q_new = mat_to_quat(Gold), mat_to_quat(Gnew[m])
    q_new = np.where((np.sum(q_old * q_new, 1) < 0)[:, None], -q_new, q_new)
    t = w[m][:, None]
    Gb = quat_to_mat(normalize((1 - t) * q_old + t * q_new))
    Q[m, i] = mat_to_quat(np.einsum("nji,njk->nik", Gp, Gb))


def _frames_to_seen(col, shot_of):
    """Per frame: frames to the nearest frame (same shot) where col is True (inf if none)."""
    N = len(col)
    d = np.full(N, np.inf)
    last = -np.inf
    for f in range(N):
        if f > 0 and shot_of[f] != shot_of[f - 1]:
            last = -np.inf
        if col[f]:
            last = f
        d[f] = f - last
    last = np.inf
    for f in range(N - 1, -1, -1):
        if f < N - 1 and shot_of[f] != shot_of[f + 1]:
            last = np.inf
        if col[f]:
            last = f
        d[f] = min(d[f], last - f)
    return d


def context_directions(Ylift, swap, R_w, seen, valid, skel, fps, shot_of, get_lift):
    """World direction of arms and thighs read from the motion around each
    frame: from the +-4 s reading, or the +-8 s / +-20 s one (get_lift(2),
    get_lift(5)) where the limb was not seen for longer than 3 s / 7 s."""
    from . import lifter
    Ys = {1: lifter.swap_labels(Ylift, swap)}
    idx = {b: i for i, b in enumerate(skel.bones)}
    dirs = {}
    for b, (ha, hb) in lifter.BONE_H36M.items():
        if b not in idx or b not in CONTEXT_BONES:
            continue
        i = idx[b]
        dist = _frames_to_seen(seen[:, i] & valid, shot_of) / fps
        need = valid & ~seen[:, i]
        mult = np.where(dist <= 3.0, 1, np.where(dist <= 7.0, 2, 5))
        for m in (2, 5):
            if (need & (mult == m)).any() and m not in Ys:
                try:
                    Y = get_lift(m)
                except Exception:
                    log.exception("long-range depth from motion failed")
                    Y = None
                Ys[m] = lifter.swap_labels(Y, swap) if Y is not None else None
        v = np.full((len(seen), 3), np.nan)
        for m in (1, 2, 5):
            Y = Ys.get(m)
            if Y is None:
                Y = Ys[1]
            sel = mult == m
            v[sel] = (Y[sel, hb] - Y[sel, ha])
        w = np.einsum("nij,nj->ni", R_w, v)
        # light smoothing along each stretch (no steps where the reading changes)
        for (a, e) in runs(need & np.isfinite(w).all(1), shot_of):
            if e - a >= 3:
                w[a:e] = gaussian_smooth(w[a:e], 0.08 * fps)
        dirs[b] = np.where(np.isfinite(w).all(1, keepdims=True), normalize(np.nan_to_num(w)), np.nan)
        if b.endswith("UpperLeg") and CONTEXT_THIGH_MAX_UP is not None:
            # a thigh that is not seen raised above level is the reading of joints
            # the model guessed (piled at the picture's edge), not a kick: kept
            # where it was
            up = dirs[b][:, 1] > CONTEXT_THIGH_MAX_UP
            dirs[b][up] = np.nan
    return dirs


def context_fill(Q, seen, valid, dirs, skel, fps, shot_of, ramp_s=0.25):
    """A limb that is not seen is still there: where it points comes from the
    motion around it (dirs: {bone: (N,3) world direction, nan = unknown}). Its
    stretches without sightings follow that direction, easing from / into the
    seen poses at their ends. Returns (Q, filled (N,B) bool)."""
    Q = normalize(Q.copy())
    idx = {b: i for i, b in enumerate(skel.bones)}
    filled = np.zeros(seen.shape, bool)
    nr = max(1, int(round(ramp_s * fps)))
    rp = skel.rest_pos
    for b in skel.bones:  # parents before children
        if b not in dirs or b not in CHILD_OF or not skel.has(CHILD_OF[b]):
            continue
        i = idx[b]
        d = dirs[b]
        m = valid & ~seen[:, i] & np.isfinite(d).all(1)
        if not m.any():
            continue
        w = _ramp_weights(m, seen[:, i], shot_of, nr)
        _, rot = avatar_fk(skel, Q, np.zeros((len(Q), 3)))
        rest_dir = rp[CHILD_OF[b]] - rp[b]
        cur = np.einsum("nij,j->ni", rot[:, i], rest_dir)
        G = np.einsum("nij,njk->nik", _align_batch(cur, np.nan_to_num(d)), rot[:, i])
        _set_world(Q, skel, b, G, np.where(m, w, 0.0))
        filled[:, i] |= m
    return Q, filled


def planted_legs(Q, root, seen, valid, skel, fps, shot_of, ramp_s=0.2, cam=None):
    """A lower leg that is not seen (below the picture, behind something)
    stands on the floor: its foot is put flat on the floor (y = 0) under the
    knee / hip, and the leg reaches it (two-bone IK when the thigh is not seen
    either, otherwise the shin alone). Returns (Q, planted (N,2) bool)."""
    Q = normalize(Q.copy())
    N = len(Q)
    idx = {b: i for i, b in enumerate(skel.bones)}
    rp = skel.rest_pos
    planted = np.zeros((N, 2), bool)
    nr = max(1, int(round(ramp_s * fps)))
    for si, side in enumerate(("left", "right")):
        iu, il, ifo = idx[side + "UpperLeg"], idx[side + "LowerLeg"], idx[side + "Foot"]
        m = valid & ~seen[:, il]
        if not m.any():
            continue
        w = _ramp_weights(m, seen[:, il], shot_of, nr)
        pos, rot = avatar_fk(skel, Q, root)
        l1 = np.linalg.norm(rp[side + "LowerLeg"] - rp[side + "UpperLeg"])
        l2 = np.linalg.norm(rp[side + "Foot"] - rp[side + "LowerLeg"])
        ankle_h = float(rp[side + "Foot"][1])
        H, K, A = pos[:, iu], pos[:, il], pos[:, ifo]
        # where the foot stands: below where the leg is now, not further out
        # than a natural stance, on the floor
        T = A.copy()
        off = T[:, [0, 2]] - H[:, [0, 2]]
        on = np.linalg.norm(off, axis=1, keepdims=True)
        lim = 0.3 * (l1 + l2)
        T[:, [0, 2]] = H[:, [0, 2]] + off * np.minimum(1.0, lim / np.maximum(on, 1e-6))
        T[:, 1] = ankle_h
        thigh_seen = seen[:, iu]
        dvec = T - H
        dl = np.linalg.norm(dvec, axis=1)
        d = np.clip(dl, abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4)
        dn = dvec / np.maximum(dl, 1e-9)[:, None]
        pole = (K - H) - dn * np.sum((K - H) * dn, 1, keepdims=True)
        pn = np.linalg.norm(pole, axis=1)
        fwd = np.einsum("nij,j->ni", rot[:, idx["hips"]], np.array([0, 0, 1.0]))
        pole = np.where((pn < 1e-6)[:, None], fwd - dn * np.sum(fwd * dn, 1, keepdims=True), pole)
        pole = normalize(pole)
        a = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
        h = np.sqrt(np.maximum(l1 * l1 - a * a, 0.0))
        Kn = H + dn * a[:, None] + pole * h[:, None]
        Kn = np.where(thigh_seen[:, None], K, Kn)
        An = np.where(thigh_seen[:, None], K + l2 * normalize(T - K), H + dn * d[:, None])
        rest_u = rp[side + "LowerLeg"] - rp[side + "UpperLeg"]
        rest_l = rp[side + "Foot"] - rp[side + "LowerLeg"]
        Gu = np.einsum("nij,njk->nik", _align_batch(np.einsum("nij,j->ni", rot[:, iu], rest_u), Kn - H), rot[:, iu])
        _set_world(Q, skel, side + "UpperLeg", Gu, np.where(m & ~thigh_seen, w, 0.0))
        _, rot = avatar_fk(skel, Q, root)
        Gl = np.einsum("nij,njk->nik", _align_batch(np.einsum("nij,j->ni", rot[:, il], rest_l), An - Kn), rot[:, il])
        _set_world(Q, skel, side + "LowerLeg", Gl, np.where(m, w, 0.0))
        # the foot flat on the floor, pointing where the pelvis points
        _, rot = avatar_fk(skel, Q, root)
        yaw = np.arctan2(fwd[:, 0], fwd[:, 2])
        c, s_ = np.cos(yaw), np.sin(yaw)
        Gf = np.zeros((N, 3, 3))
        Gf[:, 0, 0], Gf[:, 0, 2], Gf[:, 1, 1], Gf[:, 2, 0], Gf[:, 2, 2] = c, s_, 1.0, -s_, c
        _set_world(Q, skel, side + "Foot", Gf, np.where(m, w, 0.0))
        planted[:, si] = m
    # a lower leg that is seen (from the knee and the trackers on the shin)
    # above a foot that is not (below the picture, behind something): the shin
    # keeps the angle it is seen at; the foot is flat and, when it is the
    # lower of the two and not moving, stands on the floor
    pos, rot = avatar_fk(skel, Q, root)
    fwd = np.einsum("nij,j->ni", rot[:, idx["hips"]], np.array([0, 0, 1.0]))
    yaw = np.arctan2(fwd[:, 0], fwd[:, 2])
    c, s_ = np.cos(yaw), np.sin(yaw)
    Gf = np.zeros((N, 3, 3))
    Gf[:, 0, 0], Gf[:, 0, 2], Gf[:, 1, 1], Gf[:, 2, 0], Gf[:, 2, 2] = c, s_, 1.0, -s_, c
    ank = np.stack([pos[:, idx["leftFoot"]], pos[:, idx["rightFoot"]]], 1)
    low = ank[:, :, 1].min(1)
    sp = np.zeros((N, 2))
    sp[1:] = np.linalg.norm(np.diff(ank[:, :, [0, 2]], axis=0), axis=-1) * fps
    for si, side in enumerate(("left", "right")):
        il, ifo = idx[side + "LowerLeg"], idx[side + "Foot"]
        m2 = valid & seen[:, il] & ~seen[:, ifo] & ~planted[:, si]
        if not m2.any():
            continue
        w = _ramp_weights(m2, seen[:, ifo], shot_of, nr)
        if cam is not None:
            # the picture gives the shin's line, not how far it reaches towards
            # or away from the camera: of the shins along that line, the one
            # whose foot stands on the floor
            pos, rot = avatar_fk(skel, Q, root)
            K, A = pos[:, il], pos[:, ifo]
            l2 = np.linalg.norm(rp[side + "Foot"] - rp[side + "LowerLeg"])
            ankle_h = float(rp[side + "Foot"][1])
            n_ = np.cross(K - cam, A - cam)
            n_ = normalize(n_)
            u = normalize(A - K)
            v = normalize(np.cross(n_, u))
            R_ = np.hypot(u[:, 1], v[:, 1])
            phi = np.arctan2(v[:, 1], u[:, 1])
            rhs = (ankle_h - K[:, 1]) / l2
            reach = np.abs(rhs) <= R_ - 1e-6
            d_ = np.arccos(np.clip(rhs / np.maximum(R_, 1e-9), -1, 1))
            # two solutions phi +- d; the one nearer the seen direction (theta = 0)
            th1, th2 = phi + d_, phi - d_
            wrap = lambda x: np.abs(np.arctan2(np.sin(x), np.cos(x)))
            if PLANT_UNDER_HIPS:
                # (of the two, the foot nearer to under the hips: how people stand)
                Hx = pos[:, idx["hips"]][:, [0, 2]]
                f1 = (K + l2 * (np.cos(th1)[:, None] * u + np.sin(th1)[:, None] * v))[:, [0, 2]]
                f2 = (K + l2 * (np.cos(th2)[:, None] * u + np.sin(th2)[:, None] * v))[:, [0, 2]]
                th = np.where(np.linalg.norm(f1 - Hx, axis=1) <= np.linalg.norm(f2 - Hx, axis=1), th1, th2)
            else:
                th = np.where(wrap(th1) <= wrap(th2), th1, th2)
            # (a knee too high for the foot to reach the floor: the leg reaches down as far as it can)
            th = np.where(reach, th, phi + np.pi)
            th = np.clip(np.arctan2(np.sin(th), np.cos(th)), -np.radians(75), np.radians(75))
            P_ = K + l2 * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v)
            rest_l = rp[side + "Foot"] - rp[side + "LowerLeg"]
            Gl = np.einsum("nij,njk->nik", _align_batch(np.einsum("nij,j->ni", rot[:, il], rest_l), P_ - K), rot[:, il])
            ok = m2 & np.isfinite(P_).all(1) & np.isfinite(cam).all(1)
            _set_world(Q, skel, side + "LowerLeg", Gl, np.where(ok, w, 0.0))
            _set_world(Q, skel, side + "Foot", Gf, np.where(m2, w, 0.0))
            planted[:, si] |= ok & reach
        else:
            _set_world(Q, skel, side + "Foot", Gf, np.where(m2, w, 0.0))
            planted[:, si] |= m2 & (ank[:, si, 1] - low < 0.05) & (sp[:, si] < 0.6)
    return Q, planted


def world_smoothed(Q, skel, sigma):
    """Local rotations whose world (global) rotations are those of Q, lightly
    smoothed over time in the world (not relative to the parent: a parent that
    was not seen must not bend its children)."""
    N, B = Q.shape[:2]
    R = quat_to_mat(normalize(Q).reshape(-1, 4)).reshape(N, B, 3, 3)
    idx = {b: i for i, b in enumerate(skel.bones)}
    G = np.zeros_like(R)
    for i, b in enumerate(skel.bones):
        p = skel.parent[b]
        G[:, i] = R[:, i] if p is None else G[:, idx[p]] @ R[:, i]
    Gs = np.zeros_like(G)
    for i in range(B):
        q = quat_continuity(mat_to_quat(G[:, i]))
        Gs[:, i] = quat_to_mat(normalize(gaussian_smooth(q, sigma)))
    out = np.zeros_like(Q)
    for i, b in enumerate(skel.bones):
        p = skel.parent[b]
        Lm = Gs[:, i] if p is None else np.einsum("nji,njk->nik", Gs[:, idx[p]], Gs[:, i])
        out[:, i] = mat_to_quat(Lm)
    return out


def hold_unseen(Q, seen, fps, neutral, skel=None, blend_s=0.0, Q_world=None, observed=None):
    """Per bone: keyframes only where the bone is seen; in-between motion
    where it is not (see keyframe_fill). Until a bone is first seen it is in
    the neutral (A) pose; it takes its first seen pose as soon as it is seen.
    With skel, a seen bone under a parent that is not seen keeps the direction
    it was observed in (its rotation relative to the output parent is
    recomputed)."""
    N, B = Q.shape[:2]
    Q = normalize(Q)
    out = Q.copy()
    nb = max(1, int(round(blend_s * fps)))
    if skel is not None:
        # directions as observed: from Q_world (the solve before joint limits,
        # which may have bent a bone against a parent that was not seen) if given
        Qw = normalize(Q_world) if Q_world is not None else Q
        R = quat_to_mat(Qw.reshape(-1, 4)).reshape(N, B, 3, 3)
        G_sol = np.zeros_like(R)
        G_out = np.zeros_like(R)
        idx = {x: i for i, x in enumerate(skel.bones)}
        order = list(range(B))  # skel.bones is parent-first
        driven = [b in BONE_LANDMARKS or b in ("head", "neck") or b.endswith("Hand") or any(f in b for f in FINGERS)
                  for b in skel.bones]
    else:
        order = list(range(B))
    for b in order:
        target = Q[:, b]
        if skel is not None:
            p = skel.parent[skel.bones[b]]
            if p is None:
                G_sol[:, b] = R[:, b]
            else:
                pi = idx[p]
                G_sol[:, b] = G_sol[:, pi] @ R[:, b]
                if driven[b]:
                    # under a parent that is not seen: the observed direction in the
                    # world, relative to the parent as it is output
                    obs = mat_to_quat(np.einsum("nji,njk->nik", G_out[:, pi], G_sol[:, b]))
                    target = np.where(seen[:, pi:pi + 1], Q[:, b], obs) if Q_world is not None else obs
                    if observed is not None:
                        # (only a bone that was itself seen: one placed from the
                        # motion around it, or planted, keeps its placed pose)
                        target = np.where(observed[:, b:b + 1], target, Q[:, b])
        sb = seen[:, b]
        if not sb.any():
            col = np.tile(neutral[b], (N, 1))
        else:
            col = keyframe_fill(normalize(target), sb, fps, neutral[b], nb)
        out[:, b] = normalize(col)
        if skel is not None:
            p = skel.parent[skel.bones[b]]
            Lb = quat_to_mat(out[:, b])
            G_out[:, b] = Lb if p is None else G_out[:, idx[p]] @ Lb
    return normalize(out)


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


def shot_spans(valid, shot_of):
    """Per shot: from its first to its last valid frame (gaps inside included)."""
    out = []
    for (a, b) in runs(np.ones(len(valid), bool), shot_of):
        idx = np.where(valid[a:b])[0]
        if len(idx):
            out.append((a + idx[0], a + idx[-1] + 1))
    return out


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


def crop_anim(anim, start, end):
    """Frames start..end (inclusive) of an animation, as an animation of its own."""
    n = int(anim["frames"])
    start = int(np.clip(start, 0, n - 1))
    end = int(np.clip(end, start, n - 1))
    out = {}
    for k, v in anim.items():
        out[k] = v[start:end + 1] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n else v
    out["frames"] = end - start + 1
    out["crop"] = (start, end)
    return out

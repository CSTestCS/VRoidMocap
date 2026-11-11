"""Hands touching: found on the real person, kept on the avatar.

Bodies differ: copying the rotations of a person whose palms meet over the
head puts the avatar's wrists or forearms together instead (longer forearms,
narrower shoulders...). So contacts are found on the person (palm against the
other palm, the head or a hip, from the picture and the depth together) and
the avatar's arm is then moved (two-bone IK, hand orientation kept) so its own
palm touches the same part of its own body."""
from __future__ import annotations

import numpy as np

from .detect import L_EAR, L_HIP, L_SH, L_WR, NOSE, R_EAR, R_HIP, R_SH, R_WR
from .mathutil import mat_to_quat, normalize

PALM = [0, 5, 9, 13, 17]   # wrist and the four knuckles (hand model numbering)
MIN_S = 0.2                # a contact lasts at least this long
GAP_S = 0.1                # flickers shorter than this inside a contact are bridged
HIP_CONTACTS = False


def _runs_clean(m, fps, shot_of):
    from .retarget import runs
    m = m.copy()
    gap = max(1, int(round(GAP_S * fps)))
    on = np.where(m)[0]
    for a, b in zip(on[:-1], on[1:]):
        if 1 < b - a <= gap + 1 and shot_of[a] == shot_of[b]:
            m[a:b] = True
    k = max(2, int(round(MIN_S * fps)))
    for (a, b) in runs(m, shot_of):
        if b - a < k:
            m[a:b] = False
    return m


def _hysteresis(enter, stay, shot_of):
    """A contact starts where `enter` holds and lasts as long as `stay` does."""
    from .retarget import runs
    out = np.zeros(len(enter), bool)
    for (a, b) in runs(stay | enter, shot_of):
        if enter[a:b].any():
            out[a:b] = True
    return out


def detect(P, Ph, h2d, hvalid, uv, valid, fps, shot_of):
    """Contacts on the person. P (N,33,3) body joints and Ph (N,2,21,3) hand
    points in the camera frame (metres), h2d (N,2,21,2) / uv (N,33,2) in the
    picture. Returns {"hands": (N,), ("head", s): (N,), ("hip", s): (N,)}
    with s = 0 left hand, 1 right hand."""
    N = len(P)
    palm3 = np.full((N, 2, 3), np.nan)
    palm2 = np.full((N, 2, 2), np.nan)
    hlen = np.full((N, 2), np.nan)
    for s, wi in ((0, L_WR), (1, R_WR)):
        hv = hvalid[:, s] & np.isfinite(Ph[:, s]).all((1, 2))
        palm3[hv, s] = Ph[hv, s][:, PALM].mean(1)
        palm2[hv, s] = h2d[hv, s][:, PALM].mean(1)
        hlen[hv, s] = 2.0 * np.linalg.norm(h2d[hv, s, 9] - h2d[hv, s, 0], axis=1)
        # no hand points: the wrist stands in for the palm
        palm3[~hv, s] = P[~hv, wi]
        palm2[~hv, s] = uv[~hv, wi]
    torso_px = np.linalg.norm((uv[:, L_SH] + uv[:, R_SH]) / 2 - (uv[:, L_HIP] + uv[:, R_HIP]) / 2, axis=1)
    hand_px = np.where(np.isfinite(hlen), hlen, 0.35 * torso_px[:, None])
    hand_px = np.maximum(hand_px, 0.2 * torso_px[:, None])
    out = {}
    # palms together
    d2 = np.linalg.norm(palm2[:, 0] - palm2[:, 1], axis=1)
    dz = np.abs(palm3[:, 0, 2] - palm3[:, 1, 2])
    hp = hand_px.mean(1)
    enter = np.nan_to_num(valid & (d2 < 0.45 * hp) & (dz < 0.12)).astype(bool)
    stay = np.nan_to_num(valid & (d2 < 0.85 * hp) & (dz < 0.2)).astype(bool)
    hands = _hysteresis(enter, stay, shot_of)
    # a palm on the head
    head2 = (uv[:, L_EAR] + uv[:, R_EAR] + uv[:, NOSE]) / 3
    head3 = (P[:, L_EAR] + P[:, R_EAR]) / 2
    head_r = np.maximum(np.linalg.norm(uv[:, L_EAR] - uv[:, R_EAR], axis=1) * 0.75, 0.3 * torso_px)
    for s in (0, 1):
        d = np.linalg.norm(palm2[:, s] - head2, axis=1)
        dzh = np.abs(palm3[:, s, 2] - head3[:, 2])
        enter = np.nan_to_num(valid & (d < 0.9 * head_r) & (dzh < 0.12)).astype(bool) & ~hands
        stay = np.nan_to_num(valid & (d < head_r + 0.3 * hand_px[:, s]) & (dzh < 0.2)).astype(bool) & ~hands
        out[("head", s)] = _runs_clean(_hysteresis(enter, stay, shot_of), fps, shot_of)
    out["hands"] = _runs_clean(hands, fps, shot_of)
    # a hand on its own hip (off: a hand hanging beside the hip looks the same
    # in the picture and its depth is not good enough to tell them apart)
    for s, hi in ((0, L_HIP), (1, R_HIP)) if HIP_CONTACTS else ():
        d = np.linalg.norm(palm2[:, s] - uv[:, hi], axis=1)
        d3 = np.linalg.norm(palm3[:, s] - P[:, hi], axis=1)
        enter = np.nan_to_num(valid & (d < 0.33 * torso_px) & (d3 < 0.17)).astype(bool)
        stay = np.nan_to_num(valid & (d < 0.5 * torso_px) & (d3 < 0.28)).astype(bool)
        out[("hip", s)] = _runs_clean(_hysteresis(enter, stay, shot_of), fps, shot_of) & ~out[("head", s)]
    return out, palm3


def _blend(m, fps, ramp_s=0.15):
    """Weight easing into and out of each contact."""
    N = len(m)
    w = np.zeros(N)
    nr = max(1, int(round(ramp_s * fps)))
    on = np.where(m)[0]
    if not len(on):
        return w
    for f in on:
        w[f] = 1.0
    # ease: frames just before / after a contact move part of the way
    out = w.copy()
    for k in range(1, nr + 1):
        t = 1 - k / (nr + 1)
        t = t * t * (3 - 2 * t)
        out[:-k] = np.maximum(out[:-k], w[k:] * t)
        out[k:] = np.maximum(out[k:], w[:-k] * t)
    return out


def arm_reach(skel, Q, root, side, palm_target, w):
    """Two-bone IK: move this arm so its palm reaches palm_target (N,3) with
    weight w (N,); the hand keeps its world orientation. Q changed in place."""
    from .retarget import avatar_fk, _align_batch, _set_world
    idx = {b: i for i, b in enumerate(skel.bones)}
    rp = skel.rest_pos
    ua, la, ha = side + "UpperArm", side + "LowerArm", side + "Hand"
    if not all(skel.has(b) for b in (ua, la, ha)):
        return
    m = w > 0
    if not m.any():
        return
    pos, rot = avatar_fk(skel, Q, root)
    mid = side + "MiddleProximal"
    palm_off = 0.5 * (rp[mid] - rp[ha]) if mid in rp else np.zeros(3)
    S, E, Wr = pos[:, idx[ua]], pos[:, idx[la]], pos[:, idx[ha]]
    Gh = rot[:, idx[ha]].copy()
    palm = Wr + np.einsum("nij,j->ni", Gh, palm_off)
    Wt = Wr + (np.nan_to_num(palm_target) - palm) * w[:, None]
    l1 = np.linalg.norm(rp[la] - rp[ua])
    l2 = np.linalg.norm(rp[ha] - rp[la])
    dv = Wt - S
    dl = np.linalg.norm(dv, axis=1)
    d = np.clip(dl, abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4)
    dn = dv / np.maximum(dl, 1e-9)[:, None]
    pole = (E - S) - dn * np.sum((E - S) * dn, 1, keepdims=True)
    pole = normalize(pole)
    a = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    h = np.sqrt(np.maximum(l1 * l1 - a * a, 0.0))
    En = S + dn * a[:, None] + pole * h[:, None]
    Wn = S + dn * d[:, None]
    ww = np.where(m, 1.0, 0.0)
    Gu = np.einsum("nij,njk->nik", _align_batch(E - S, En - S), rot[:, idx[ua]])
    _set_world(Q, skel, ua, Gu, ww)
    _, rot = avatar_fk(skel, Q, root)
    pos2, _ = avatar_fk(skel, Q, root)
    El = pos2[:, idx[la]]
    Gl = np.einsum("nij,njk->nik", _align_batch(pos2[:, idx[ha]] - El, Wn - El), rot[:, idx[la]])
    _set_world(Q, skel, la, Gl, ww)
    _set_world(Q, skel, ha, Gh, ww)


def apply(skel, Q, root, contacts, palm3_world, caps, fps):
    """Make the avatar's palms touch where the person's do. palm3_world (N,2,3)
    the person's palms in the world (directions only are used)."""
    from .retarget import avatar_fk
    Q = Q.copy()
    idx = {b: i for i, b in enumerate(skel.bones)}
    rp = skel.rest_pos
    rhand = caps.get("leftHand", (None, None, None, 0.028))[3]

    def palms():
        pos, rot = avatar_fk(skel, Q, root)
        out = []
        for side in ("left", "right"):
            ha, mid = side + "Hand", side + "MiddleProximal"
            off = 0.5 * (rp[mid] - rp[ha]) if mid in rp else np.zeros(3)
            out.append(pos[:, idx[ha]] + np.einsum("nij,j->ni", rot[:, idx[ha]], off))
        return pos, rot, out

    # palms together: they meet half way, as far apart as the hands are thick
    m = contacts.get("hands")
    if m is not None and m.any():
        w = _blend(m, fps)
        pos, rot, (pl, pr) = palms()
        u = palm3_world[:, 0] - palm3_world[:, 1]
        u = np.where(np.isfinite(u).all(1, keepdims=True) & (np.linalg.norm(np.nan_to_num(u), axis=1, keepdims=True) > 1e-3), u, pl - pr)
        u = normalize(u)
        c = (pl + pr) / 2
        arm_reach(skel, Q, root, "left", c + u * rhand, w)
        arm_reach(skel, Q, root, "right", c - u * rhand, w)
    # a palm on the head: on its surface, on the side the person's palm is
    if "head" in caps:
        hb, off, _, rh = caps["head"]
        for s, side in ((0, "left"), (1, "right")):
            m = contacts.get(("head", s))
            if m is None or not m.any():
                continue
            w = _blend(m, fps)
            pos, rot, pal = palms()
            c = pos[:, idx[hb]] + np.einsum("nij,j->ni", rot[:, idx[hb]], off)
            dr = normalize(np.nan_to_num(pal[s] - c) + 1e-9)
            arm_reach(skel, Q, root, side, c + dr * (rh + rhand), w)
    # a hand on the hip: beside the hip joint, on the outside
    for s, side in ((0, "left"), (1, "right")):
        m = contacts.get(("hip", s))
        if m is None or not m.any() or not skel.has(side + "UpperLeg"):
            continue
        w = _blend(m, fps)
        pos, rot, pal = palms()
        hip = pos[:, idx[side + "UpperLeg"]]
        lat = np.einsum("nij,j->ni", rot[:, idx["hips"]], np.array([1.0 if s == 0 else -1.0, 0, 0]))
        r_hip = caps.get("torso0", (None, None, None, 0.1))[3]
        arm_reach(skel, Q, root, side, hip + lat * (0.45 * r_hip + rhand) + np.array([0, 0.05, 0]), w)
    return Q


def allowed_overlaps(contacts, fps):
    """Pairs of parts the anti-clipping must not push apart while the person
    is touching with them."""
    out = {}
    m = contacts.get("hands")
    if m is not None and m.any():
        w = _blend(m, fps) > 0
        for a in ("leftHand", "leftLowerArm"):
            for b in ("rightHand", "rightLowerArm"):
                out[(a, b)] = w
    for s, side in ((0, "left"), (1, "right")):
        m = contacts.get(("head", s))
        if m is not None and m.any():
            w = _blend(m, fps) > 0
            out[(side + "Hand", "head")] = w
            out[(side + "LowerArm", "head")] = w
    return out

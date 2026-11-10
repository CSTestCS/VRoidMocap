"""Per-bone detail against the rendered ground truth: where each bone points
(swing) and which way it faces (twist about its own length), torso / head
facing, and smoothness.
usage: python eval_detail.py <work folder with anim_c0.npz> <gt.json> <avatar.vrm> [shot]"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vroidmocap.core.mathutil import mat_to_quat, quat_to_mat  # noqa: E402
from vroidmocap.core.retarget import avatar_fk  # noqa: E402
from vroidmocap.core.vrm import load_vrm_skeleton  # noqa: E402

ws, gtp, vrm = sys.argv[1:4]
only = int(sys.argv[4]) if len(sys.argv) > 4 else None
gt = json.load(open(gtp))
a = np.load(ws + "/anim_c0.npz", allow_pickle=True)
bones = json.loads(str(a["_meta"]))["bones"]
skel = load_vrm_skeleton(vrm)
N = min(len(gt), len(a["quats"]))
pos, rot = avatar_fk(skel, a["quats"][:N].astype(float), a["root"][:N].astype(float))
gp = np.array([[p if p else [np.nan] * 3 for p in g["pos"]] for g in gt[:N]], float)
gr = np.array([[p if p else [0, 0, 0, 1] for p in g["rot"]] for g in gt[:N]], float)
gc = np.array([g["cam"] for g in gt[:N]])
shot = np.array([g["shot"] for g in gt[:N]])
Rg = quat_to_mat(gc[:, 3:7])
Rs = quat_to_mat(a["cam_quat"][:N].astype(float))
bi = {b: i for i, b in enumerate(bones)}
valid = a["valid"][:N]
# global rotations in camera frames
Gs = np.einsum("nji,nbjk->nbik", Rs, rot)
Gg = np.einsum("nji,nbjk->nbik", Rg, quat_to_mat(gr.reshape(-1, 4)).reshape(N, -1, 3, 3))
Ps = np.einsum("nji,nbj->nbi", Rs, pos - a["cam_pos"][:N, None].astype(float))
Pg = np.einsum("nji,nbj->nbi", Rg, gp - gc[:, None, :3])


def child_of(b):
    for c in skel.bones:
        if skel.parent[c] == b and not any(k in c for k in ("Thumb", "Index", "Middle", "Ring", "Little", "Eye", "Jaw")):
            if b == "hips" and c != "spine":
                continue
            if b in ("upperChest", "chest") and "Shoulder" in c:
                continue
            return c
    return None


def deg(x):
    return np.degrees(x)


def st(x):
    x = x[np.isfinite(x)]
    if not len(x):
        return "   n/a"
    return f"{x.mean():5.1f} /{np.percentile(x, 90):5.1f}"


def jitter(P, m):
    """mean |second difference| (cm per frame^2) over valid consecutive frames"""
    acc = P[2:] - 2 * P[1:-1] + P[:-2]
    ok = m[2:] & m[1:-1] & m[:-2]
    return np.linalg.norm(acc[ok], axis=-1) * 100


BONES = ["hips", "spine", "chest", "upperChest", "neck", "head", "leftShoulder", "rightShoulder",
         "leftUpperArm", "rightUpperArm", "leftLowerArm", "rightLowerArm", "leftHand", "rightHand",
         "leftUpperLeg", "rightUpperLeg", "leftLowerLeg", "rightLowerLeg", "leftFoot", "rightFoot"]
for sh in sorted(set(shot)):
    if only is not None and sh != only:
        continue
    m = valid & (shot == sh)
    print(f"=== shot {sh} ({m.sum()} frames)      swing(mean/p90)  twist(mean/p90)  total")
    for b in BONES:
        if b not in bi:
            continue
        i = bi[b]
        c = child_of(b)
        ax = normalize_ax = None
        if c is not None:
            ax = skel.rest_pos[c] - skel.rest_pos[b]
            ax = ax / np.linalg.norm(ax)
        E = np.einsum("nji,njk->nik", Gg[m, i], Gs[m, i])  # error in the bone's own frame
        q = mat_to_quat(E)
        q = q * np.sign(q[:, 3:4] + 1e-12)
        tot = deg(2 * np.arccos(np.clip(q[:, 3], -1, 1)))
        if ax is None:
            print(f"  {b:14s}                                   {st(tot)}")
            continue
        tw = deg(2 * np.arctan2(np.abs(q[:, :3] @ ax), q[:, 3]))
        tw = np.minimum(tw, 360 - tw)
        # swing: angle between where the bone axis points
        d1 = np.einsum("nij,j->ni", Gs[m, i], ax)
        d0 = np.einsum("nij,j->ni", Gg[m, i], ax)
        sw = deg(np.arccos(np.clip(np.sum(d0 * d1, -1), -1, 1)))
        print(f"  {b:14s} {st(sw)}     {st(tw)}     {st(tot)}")

    # facing about the vertical: forward axis (+Z of the normalized VRM bone) projected
    # onto the ground plane (camera-frame up = row 1 of the world->camera rotation)
    upg = Rg[m][:, 1, :]

    def yaw_err(b):
        fs = Gs[m, bi[b]] @ np.array([0, 0, 1.0])
        fg = Gg[m, bi[b]] @ np.array([0, 0, 1.0])
        fs = fs - upg * np.sum(fs * upg, -1, keepdims=True)
        fg = fg - upg * np.sum(fg * upg, -1, keepdims=True)
        ang = np.arctan2(np.sum(np.cross(fg, fs) * upg, -1), np.sum(fs * fg, -1))
        return np.abs(deg(ang))
    ye = {b: yaw_err(b) for b in ("hips", "upperChest", "head") if b in bi}
    print("  facing error (deg mean/p90, >45 %):  " + "  ".join(
        f"{b} {st(v)} {np.mean(v > 45) * 100:4.1f}%" for b, v in ye.items()))
    hi = bi["hips"]
    rel_s, rel_g = Ps[:, :] - Ps[:, [hi]], Pg - Pg[:, [hi]]
    body = [bi[b] for b in BONES if b in bi]
    js, jg = jitter(rel_s[:, body], m), jitter(rel_g[:, body], m)
    print(f"  jitter (cm/frame^2) solved {np.nanmean(js):.2f}  truth {np.nanmean(jg):.2f}   "
          f"MPJPE cm {np.nanmean(np.linalg.norm((rel_s - rel_g)[m][:, body], axis=-1)) * 100:.1f}")

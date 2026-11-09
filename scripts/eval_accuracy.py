"""Measure how well a solved animation matches the video.

Projects the posed avatar through the solved per-frame camera and compares the
2D direction of every limb segment with the detected landmarks. Reports the
median / 90th percentile angle error per segment (degrees) and the position
error of the hips (as a fraction of body height).

usage: python scripts/eval_accuracy.py WORKDIR CHAR_ID AVATAR.vrm
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vroidmocap.core.mathutil import quat_to_mat  # noqa: E402
from vroidmocap.core.retarget import avatar_fk  # noqa: E402
from vroidmocap.core.vrm import load_vrm_skeleton  # noqa: E402

BODY = {"leftUpperArm": 11, "leftLowerArm": 13, "leftHand": 15, "rightUpperArm": 12, "rightLowerArm": 14,
        "rightHand": 16, "leftUpperLeg": 23, "leftLowerLeg": 25, "leftFoot": 27, "rightUpperLeg": 24,
        "rightLowerLeg": 26, "rightFoot": 28}
SEGS = [("L upper arm", "leftUpperArm", "leftLowerArm"), ("L forearm", "leftLowerArm", "leftHand"),
        ("R upper arm", "rightUpperArm", "rightLowerArm"), ("R forearm", "rightLowerArm", "rightHand"),
        ("L thigh", "leftUpperLeg", "leftLowerLeg"), ("L shin", "leftLowerLeg", "leftFoot"),
        ("R thigh", "rightUpperLeg", "rightLowerLeg"), ("R shin", "rightLowerLeg", "rightFoot"),
        ("shoulders", "rightUpperArm", "leftUpperArm"), ("hips", "rightUpperLeg", "leftUpperLeg")]
FINGER = []
for side, si in (("left", 0), ("right", 1)):
    for fg, idx in (("Index", (5, 6, 7)), ("Middle", (9, 10, 11)), ("Ring", (13, 14, 15)), ("Little", (17, 18, 19))):
        FINGER.append((f"{side[0].upper()} {fg.lower()} prox", si, f"{side}{fg}Proximal", f"{side}{fg}Intermediate", idx[0], idx[1]))
        FINGER.append((f"{side[0].upper()} {fg.lower()} mid", si, f"{side}{fg}Intermediate", f"{side}{fg}Distal", idx[1], idx[2]))
    FINGER.append((f"{side[0].upper()} thumb", si, f"{side}ThumbProximal", f"{side}ThumbDistal", 2, 3))
    FINGER.append((f"{side[0].upper()} palm", si, f"{side}Hand", f"{side}MiddleProximal", 0, 9))


def project(X, C, R, vfov, W, H):
    f = 0.5 * H / np.tan(np.radians(vfov) / 2)
    p = np.einsum("nji,nkj->nki", R, X - C[:, None])
    z = -p[..., 2]
    return np.stack([W / 2 + f[:, None] * p[..., 0] / z, H / 2 - f[:, None] * p[..., 1] / z], -1)


def angle(a, b):
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    c = (a * b).sum(-1) / np.maximum(na * nb, 1e-9)
    return np.degrees(np.arccos(np.clip(c, -1, 1))), na, nb


def main(workdir, cid, vrm):
    skel = load_vrm_skeleton(vrm)
    d = np.load(os.path.join(workdir, f"anim_c{cid}.npz"))
    meta = json.loads(str(d["_meta"]))
    W, H = meta["width"], meta["height"]
    Q, root = d["quats"].astype(np.float64), d["root"].astype(np.float64)
    pos, _ = avatar_fk(skel, Q, root)
    bi = {b: i for i, b in enumerate(skel.bones)}
    C = d["cam_pos"].astype(np.float64)
    R = quat_to_mat(d["cam_quat"].astype(np.float64))
    uv = project(pos, C, R, d["cam_vfov"].astype(np.float64), W, H)
    lm, vis, h2d, hs = d["lm2d"], d["vis"], d["h2d"], d["hs"]
    valid = d["valid"].astype(bool)
    rows = []
    # where the hand model found the hand, its wrist is used (as the solver does)
    lmw = lm.astype(np.float64).copy()
    for side, wi in ((0, 15), (1, 16)):
        hm = hs[:, side] >= 0.5
        lmw[hm, wi] = h2d[hm, side, 0]
    for name, a, b in SEGS:
        ia, ib = BODY[a], BODY[b]
        m = valid & (vis[:, ia] > 0.7) & (vis[:, ib] > 0.7)
        e, na, nb = angle(uv[m, bi[b]] - uv[m, bi[a]], lmw[m, ib] - lmw[m, ia])
        e = e[nb > 8]
        rows.append((name, len(e), np.median(e), np.percentile(e, 90)))
    frows = []
    for name, si, a, b, ia, ib in FINGER:
        if a not in bi or b not in bi:
            continue
        m = valid & (hs[:, si] >= 0.5)
        e, na, nb = angle(uv[m, bi[b]] - uv[m, bi[a]], h2d[m, si, ib] - h2d[m, si, ia])
        e = e[nb > 6]
        frows.append((name, len(e), np.median(e), np.percentile(e, 90)))
    # hips position
    hipc_av = (uv[:, bi["leftUpperLeg"]] + uv[:, bi["rightUpperLeg"]]) / 2
    hipc_lm = (lm[:, 23] + lm[:, 24]) / 2
    body_h = np.ptp(lm[:, :, 1], axis=1)
    pe = np.linalg.norm(hipc_av - hipc_lm, axis=1)[valid] / np.maximum(body_h[valid], 1)
    print(f"{'segment':16s} {'n':>6s} {'median':>7s} {'p90':>7s}")
    for r in rows:
        print(f"{r[0]:16s} {r[1]:6d} {r[2]:7.1f} {r[3]:7.1f}")
    allb = np.median([r[2] for r in rows])
    print(f"BODY median of medians: {allb:.1f} deg")
    for r in frows:
        print(f"{r[0]:16s} {r[1]:6d} {r[2]:7.1f} {r[3]:7.1f}")
    if frows:
        print(f"FINGERS median of medians: {np.median([r[2] for r in frows]):.1f} deg")
    # jitter: frame-to-frame acceleration of wrists / ankles in the avatar (mm per frame^2)
    jit = []
    for b in ("leftHand", "rightHand", "leftFoot", "rightFoot", "head"):
        p = pos[:, bi[b]] - pos[:, bi["hips"]]
        acc = np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=1)[valid[1:-1]]
        jit.append(np.median(acc) * 1000)
    print(f"jitter (median joint acceleration): {np.mean(jit):.2f} mm/frame^2")
    print(f"hips position error: median {np.median(pe) * 100:.1f}% of body height, p90 {np.percentile(pe, 90) * 100:.1f}%")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])

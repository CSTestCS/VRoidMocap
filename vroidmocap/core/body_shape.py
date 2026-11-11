"""The avatar's body as simple shapes (a capsule around each bone, a ball for
the head), measured from its own mesh. Used to keep limbs from passing through
each other or through the body, and to put a touching hand on the surface.

A capsule is a bone segment (from the bone's joint to its child joint) with a
radius: every point within that distance of the segment is inside."""
from __future__ import annotations

import json
import struct

import numpy as np

# the bone each capsule runs along -> the joint it ends at
SEGMENTS = {
    "hips": "spine", "spine": "chest", "chest": "upperChest", "upperChest": "neck",
    "leftUpperArm": "leftLowerArm", "leftLowerArm": "leftHand", "leftHand": "leftMiddleProximal",
    "rightUpperArm": "rightLowerArm", "rightLowerArm": "rightHand", "rightHand": "rightMiddleProximal",
    "leftUpperLeg": "leftLowerLeg", "leftLowerLeg": "leftFoot", "leftFoot": "leftToes",
    "rightUpperLeg": "rightLowerLeg", "rightLowerLeg": "rightFoot", "rightFoot": "rightToes",
}
# radius as a share of the bone's length when the mesh cannot be read
FALLBACK = {"hips": 0.9, "spine": 0.9, "chest": 0.8, "upperChest": 0.8, "UpperArm": 0.17, "LowerArm": 0.15,
            "Hand": 0.25, "UpperLeg": 0.2, "LowerLeg": 0.14, "Foot": 0.25}
TORSO = ("hips", "spine", "chest", "upperChest")
PCT = {}
LIMB_PCT = 70


def _glb(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, _v, _l = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ValueError("not a GLB")
    off = 12
    gl, binc = None, None
    while off < len(data):
        clen, ctype = struct.unpack_from("<I4s", data, off)
        chunk = data[off + 8:off + 8 + clen]
        if ctype == b"JSON":
            gl = json.loads(chunk.decode("utf-8"))
        elif ctype == b"BIN\x00":
            binc = chunk
        off += 8 + clen
    return gl, binc


_CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _accessor(gl, binc, i):
    a = gl["accessors"][i]
    bv = gl["bufferViews"][a["bufferView"]]
    dt = np.dtype(_CT[a["componentType"]])
    nc = _NC[a["type"]]
    start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = bv.get("byteStride", 0) or dt.itemsize * nc
    n = a["count"]
    raw = np.frombuffer(binc, np.uint8, count=stride * (n - 1) + dt.itemsize * nc, offset=start)
    out = np.lib.stride_tricks.as_strided(raw.view(np.uint8), shape=(n, nc * dt.itemsize), strides=(stride, 1))
    out = np.ascontiguousarray(out).view(dt).reshape(n, nc).astype(np.float64)
    if a.get("normalized") and dt.kind in "iu":
        out = out / np.iinfo(dt).max
    return out


def _point_seg_dist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def measure_vrm(skel):
    """{bone: radius} from the VRM's skinned meshes (rest pose), or {}."""
    try:
        gl, binc = _glb(skel.path)
    except Exception:
        return {}
    if gl is None or binc is None:
        return {}
    ext = gl.get("extensions", {})
    if "VRMC_vrm" in ext:
        hb = {k: v["node"] for k, v in ext["VRMC_vrm"]["humanoid"]["humanBones"].items()}
    elif "VRM" in ext:
        from .vrm import VRM0_RENAME
        hb = {VRM0_RENAME.get(b["bone"], b["bone"]): b["node"] for b in ext["VRM"]["humanoid"]["humanBones"]}
    else:
        return {}
    node_bone = {v: k for k, v in hb.items()}
    to_norm = np.asarray(skel.to_file).T
    pts = {b: [] for b in list(SEGMENTS) + ["head"]}
    for ni, node in enumerate(gl.get("nodes", [])):
        if "mesh" not in node or "skin" not in node:
            continue
        joints = gl["skins"][node["skin"]]["joints"]
        jbone = [node_bone.get(j) for j in joints]
        for prim in gl["meshes"][node["mesh"]]["primitives"]:
            at = prim.get("attributes", {})
            if not all(k in at for k in ("POSITION", "JOINTS_0", "WEIGHTS_0")):
                continue
            P = _accessor(gl, binc, at["POSITION"]) @ to_norm.T
            J = _accessor(gl, binc, at["JOINTS_0"]).astype(int)
            Wt = _accessor(gl, binc, at["WEIGHTS_0"])
            k = np.argmax(Wt, 1)
            jdom = J[np.arange(len(J)), k]
            wdom = Wt[np.arange(len(J)), k]
            for b in pts:
                if b not in hb:
                    continue
                sel = np.array([jbone[j] == b if j < len(jbone) else False for j in jdom]) & (wdom >= 0.5)
                if sel.any():
                    pts[b].append(P[sel])
    rad = {}
    rp = skel.rest_pos
    for b, c in SEGMENTS.items():
        if not pts[b] or b not in rp or c not in rp:
            continue
        p = np.concatenate(pts[b])
        if len(p) < 20:
            continue
        if b in TORSO:
            continue  # measured as a whole below
        d = _point_seg_dist(p, rp[b], rp[c])
        rad[b] = float(np.percentile(d, PCT.get(b, LIMB_PCT)))
    # the torso as a whole (its vertices blend between the torso bones): slices
    # across the body from the hips to the neck, each as wide and as deep as
    # the body is at that height, carried by the torso bone at that height
    tor = [np.concatenate(pts[b]) for b in TORSO if pts.get(b)]
    if tor and "neck" in rp and "hips" in rp:
        p = np.concatenate(tor)
        y0, y1 = rp["hips"][1] - 0.04, rp["neck"][1]
        tb = [b for b in TORSO if b in rp]
        k = 0
        for yc in np.arange(y0 + 0.035, y1, 0.07):
            sl = p[np.abs(p[:, 1] - yc) < 0.035]
            if len(sl) < 30:
                continue
            x0, x1 = np.percentile(sl[:, 0], [4, 96])
            z0, z1 = np.percentile(sl[:, 2], [6, 94])
            r = float((z1 - z0) / 2)
            zc = (z0 + z1) / 2
            bone = [b for b in tb if rp[b][1] <= yc][-1] if any(rp[b][1] <= yc for b in tb) else tb[0]
            a = np.array([min(x0 + r, 0.0), yc, zc]) - rp[bone]
            bb = np.array([max(x1 - r, 0.0), yc, zc]) - rp[bone]
            rad[f"torso{k}"] = (bone, a, bb, r)
            k += 1
    # the head: a ball around the middle of its own vertices (hair hangs on
    # its own bones and is not part of it)
    if pts["head"] and "head" in rp:
        p = np.concatenate(pts["head"])
        c = np.median(p, 0)
        rad["head_ball"] = (c - rp["head"], float(np.percentile(np.linalg.norm(p - c, axis=1), 60)))
    return rad


def capsules(skel):
    """{name: (bone, a, b, radius)}: a capsule from offset a to offset b in
    bone's own frame (rest axes), for every part the avatar has."""
    rad = measure_vrm(skel) if str(getattr(skel, "path", "")).lower().endswith(".vrm") else {}
    rp = skel.rest_pos
    out = {}
    for b, c in SEGMENTS.items():
        if b not in rp or c not in rp:
            continue
        L = float(np.linalg.norm(rp[c] - rp[b]))
        v = rad.get(b)
        if b in TORSO:
            if any(k.startswith("torso") for k in rad):
                continue
            # no mesh: a capsule across the body, sized from the shoulders
            w = float(np.linalg.norm(rp["leftUpperArm"] - rp["rightUpperArm"])) / 2 if "leftUpperArm" in rp else 0.15
            r = 0.35 * w
            y = 0.5 * (rp[c][1] - rp[b][1])
            out[b] = (b, np.array([-0.75 * w + r, y, 0.0]), np.array([0.75 * w - r, y, 0.0]), r)
            continue
        r = v
        if r is None or not np.isfinite(r) or r <= 0:
            key = next((k for k in FALLBACK if b.endswith(k) or b == k), None)
            r = FALLBACK.get(key, 0.15) * L
        out[b] = (b, np.zeros(3), rp[c] - rp[b], float(min(r, 0.45 * max(L, 0.05))))
    for k, v in rad.items():
        if k.startswith("torso"):
            out[k] = v
    if "head" in rp:
        if "head_ball" in rad:
            off, r = rad["head_ball"]
        else:
            h = float(np.linalg.norm(rp["head"] - rp["neck"])) if "neck" in rp else 0.1
            off, r = np.array([0, 1.1 * h, 0.01]), 1.1 * h
        off = np.asarray(off, float)
        out["head"] = ("head", off, off, float(r))
    return out


# ----------------------------------------------------------------------------- shapes in a pose
def shape_points(skel, caps, pos, rot):
    """World segment (a, b) (N,3) each, and the radius, per capsule."""
    idx = {b: i for i, b in enumerate(skel.bones)}
    out = {}
    for name, (b, oa, ob, r) in caps.items():
        P, R = pos[:, idx[b]], rot[:, idx[b]]
        out[name] = (P + np.einsum("nij,j->ni", R, oa), P + np.einsum("nij,j->ni", R, ob), r)
    return out


def closest_seg_seg(p1, q1, p2, q2):
    """Closest points between segments p1-q1 and p2-q2 (batched, (n,3)).
    Returns (s, t, c1, c2)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = np.sum(d1 * d1, -1)
    e = np.sum(d2 * d2, -1)
    f = np.sum(d2 * r, -1)
    c = np.sum(d1 * r, -1)
    b = np.sum(d1 * d2, -1)
    den = a * e - b * b
    s = np.where(den > 1e-12, np.clip((b * f - c * e) / np.maximum(den, 1e-12), 0, 1), 0.0)
    t = (b * s + f) / np.maximum(e, 1e-12)
    t_c = np.clip(t, 0, 1)
    s = np.where((t < 0) | (t > 1), np.clip((b * t_c - c) / np.maximum(a, 1e-12), 0, 1), s)
    t = t_c
    s = np.where(a < 1e-12, 0.0, s)
    t = np.where(e < 1e-12, 0.0, t)
    c1 = p1 + d1 * s[:, None]
    c2 = p2 + d2 * t[:, None]
    return s, t, c1, c2



# ----------------------------------------------------------------------------- no limbs inside each other
def collision_pairs(caps):
    """(moving part, other part, share of the moving part's length ignored at its
    root, 'move' = only the first one moves, 'split' = both give way).
    Only parts that really pass through each other in a wrong pose are checked:
    forearms and hands against the body, the head, the thighs and each other;
    knees, shins and feet against each other. (An upper arm at the shoulder or
    two thick thighs touching are normal: the mesh overlaps itself there.)"""
    torso = [b for b in caps if b in ("hips", "spine", "chest", "upperChest") or b.startswith("torso")]
    out = []
    for side, other in (("left", "right"), ("right", "left")):
        la, ha = side + "LowerArm", side + "Hand"
        for t in torso:
            out.append((la, t, 0.25, "move"))
            out.append((ha, t, 0.0, "move"))
        if "head" in caps:
            out += [(la, "head", 0.0, "move"), (ha, "head", 0.0, "move")]
        for leg in (side + "UpperLeg", other + "UpperLeg"):
            out += [(la, leg, 0.0, "move"), (ha, leg, 0.0, "move")]
    for a, b in (("leftLowerArm", "rightLowerArm"), ("leftHand", "rightHand"), ("leftLowerArm", "rightHand"),
                 ("leftHand", "rightLowerArm")):
        out.append((a, b, 0.0, "split"))
    for a, b, tr in (("leftLowerLeg", "rightLowerLeg", 0.0), ("leftUpperLeg", "rightLowerLeg", 0.6),
                     ("leftLowerLeg", "rightUpperLeg", 0.0), ("leftFoot", "rightFoot", 0.0),
                     ("leftFoot", "rightLowerLeg", 0.0), ("leftLowerLeg", "rightFoot", 0.0)):
        out.append((a, b, tr, "split"))
    return [p for p in out if p[0] in caps and p[1] in caps]


def _pivot(skel, b, lever_len, seg_len):
    """Which bone turns to move part b out: the limb's upper bone for the upper
    part, the elbow / knee for the lower part when it has leverage."""
    if b.endswith("Hand"):
        return b.replace("Hand", "LowerArm")
    if b.endswith("Foot"):
        return b.replace("Foot", "LowerLeg")
    if b.endswith("LowerArm"):
        return b if lever_len > 0.45 * seg_len else b.replace("Lower", "Upper")
    if b.endswith("LowerLeg"):
        return b.replace("Lower", "Upper")
    return b


def resolve_collisions(skel, Q, caps, fps, shot_of, iters=3, smooth_s=0.06, active=None, allow=None):
    """Push limbs that are inside another part of the body out until they just
    touch (same pose otherwise). Q (N,B,4) local rotations; returns a new Q
    and the penetration (m) left per frame. active (N,) bool: frames to fix.
    allow: {(part, part): (N,) bool} frames where those two may overlap (a
    contact the person makes: palms pressed together)."""
    from .mathutil import mat_to_quat, normalize, quat_to_mat, gaussian_smooth
    from .retarget import avatar_fk, _align_batch, runs
    N = len(Q)
    Q0 = normalize(Q.copy())
    Q = Q0.copy()
    idx = {b: i for i, b in enumerate(skel.bones)}
    pairs = collision_pairs(caps)
    act = np.ones(N, bool) if active is None else np.asarray(active, bool)
    zero = np.zeros((N, 3))

    def one_pass(Q, frac=1.0):
        for (a, b, trim, how) in pairs:
            pos, rot = avatar_fk(skel, Q, zero)
            S = shape_points(skel, caps, pos, rot)
            pa, qa, ra = S[a]
            pb, qb, rb = S[b]
            pa = pa + trim * (qa - pa)
            _, _, ca, cb = closest_seg_seg(pa, qa, pb, qb)
            dvec = ca - cb
            dist = np.linalg.norm(dvec, axis=1)
            pen = (ra + rb) - dist
            hit = act & (pen > 1e-3)
            if allow:
                for key in ((a, b), (b, a)):
                    if key in allow:
                        hit &= ~allow[key]
            if not hit.any():
                continue
            n = dvec / np.maximum(dist, 1e-9)[:, None]
            # (crossing exactly: push at right angles to both parts; failing
            # that, sideways)
            cr = np.cross(qa - pa, qb - pb)
            crn = np.linalg.norm(cr, axis=1)
            lat = np.einsum("nij,j->ni", rot[:, idx["hips"]], np.array([1.0, 0, 0])) if "hips" in idx else np.tile([1.0, 0, 0], (N, 1))
            alt = np.where((crn > 1e-6)[:, None], cr / np.maximum(crn, 1e-9)[:, None], lat * (1 if "left" in a else -1))
            n = np.where((dist < 1e-4)[:, None], alt, n)
            share = 0.5 if how == "split" else 1.0
            for part, sign, c in ((a, 1.0, ca), (b, -1.0, cb)) if how == "split" else ((a, 1.0, ca),):
                seg_len = float(np.linalg.norm(caps[part][2] - caps[part][1])) or 0.1
                lever0 = np.linalg.norm(c - pos[:, idx[part]], axis=1)
                piv = _pivot(skel, part, float(np.median(lever0[hit])) if hit.any() else 0, seg_len)
                J = pos[:, idx[piv]]
                lever = c - J
                target = lever + n * (sign * share * frac * np.maximum(pen, 0))[:, None]
                R = _align_batch(lever, target)
                G = np.einsum("nij,njk->nik", R, rot[:, idx[piv]])
                p = skel.parent[piv]
                Gp = rot[:, idx[p]] if p is not None else np.tile(np.eye(3), (N, 1, 1))
                L = mat_to_quat(np.einsum("nji,njk->nik", Gp, G))
                Q[hit, idx[piv]] = L[hit]
                pos, rot = avatar_fk(skel, Q, zero)
                S = shape_points(skel, caps, pos, rot)
        return Q

    for _ in range(iters):
        Q = one_pass(Q)
    # the corrections, smooth over time (no sudden pushes), then a last exact pass
    if smooth_s > 0:
        from .naturalize import q_angle
        changed = [i for i in range(Q.shape[1]) if np.any(q_angle(Q[:, i], Q0[:, i]) > 1e-4)]
        for i in changed:
            # delta in the parent's frame: Q = d * Q0
            d = _qmul_np(Q[:, i], _qinv_np(Q0[:, i]))
            d = np.where((d[:, 3] < 0)[:, None], -d, d)
            for (s0, e0) in runs(np.ones(N, bool), shot_of):
                if e0 - s0 >= 3:
                    d[s0:e0] = gaussian_smooth(d[s0:e0], smooth_s * fps)
            Q[:, i] = normalize(_qmul_np(normalize(d), Q0[:, i]))
        Q = one_pass(Q)
    pos, rot = avatar_fk(skel, Q, zero)
    S = shape_points(skel, caps, pos, rot)
    left = np.zeros(N)
    for (a, b, trim, how) in pairs:
        pa, qa, ra = S[a]
        pb, qb, rb = S[b]
        pa = pa + trim * (qa - pa)
        _, _, ca, cb = closest_seg_seg(pa, qa, pb, qb)
        left = np.maximum(left, (ra + rb) - np.linalg.norm(ca - cb, axis=1))
    return Q, np.maximum(left, 0)


def penetration(skel, Q, root, caps):
    """Deepest overlap between body parts per frame (m)."""
    from .retarget import avatar_fk
    pos, rot = avatar_fk(skel, Q, np.zeros((len(Q), 3)))
    S = shape_points(skel, caps, pos, rot)
    out = np.zeros(len(Q))
    for (a, b, trim, how) in collision_pairs(caps):
        pa, qa, ra = S[a]
        pb, qb, rb = S[b]
        pa = pa + trim * (qa - pa)
        _, _, ca, cb = closest_seg_seg(pa, qa, pb, qb)
        out = np.maximum(out, (ra + rb) - np.linalg.norm(ca - cb, axis=1))
    return np.maximum(out, 0)


def _qmul_np(a, b):
    x1, y1, z1, w1 = np.moveaxis(a, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], -1)


def _qinv_np(q):
    return q * np.array([-1, -1, -1, 1.0])

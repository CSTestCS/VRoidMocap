"""Hand-set keyframes after the motion capture: finger curl / splay and the
avatar's expressions (blend shapes).

Channels (values as shown in the app):
  "<side><Finger>.curl"   -10 .. 100   (0 = straight as modelled, 100 = fist,
                                         below 0 bends slightly backwards)
  "<side><Finger>.splay"    0 .. 10    (0 = as modelled, 10 = spread wide)
  "expr.<name>"             0 .. 100   (expression / blend shape weight, %)
Keys are (time in seconds, value). Between keys the value follows a smooth
curve that never overshoots (monotone cubic); before the first key and after
the last one it holds. A channel without keys has its default value.

The same evaluation runs in the app (live preview) and here (export)."""
from __future__ import annotations

import json
import os

import numpy as np

from .mathutil import normalize
from .vrm import FINGER_SEGS, FINGERS

SIDES = ("left", "right")
CURL_RANGE = (-10.0, 100.0)
SPLAY_RANGE = (0.0, 10.0)
# degrees per segment at curl 100 (proximal / middle / tip; thumb: metacarpal /
# proximal / tip)
CURL_MAX = {"Thumb": (30.0, 40.0, 55.0), "Index": (80.0, 95.0, 65.0), "Middle": (82.0, 97.0, 67.0),
            "Ring": (84.0, 98.0, 68.0), "Little": (86.0, 100.0, 70.0)}
# degrees at splay 10, and which way: -1 towards the thumb, +1 towards the little finger
SPLAY_MAX = {"Thumb": (28.0, -1), "Index": (12.0, -1), "Middle": (3.0, -1), "Ring": (8.0, 1), "Little": (18.0, 1)}
DEFAULTS = {"curl": 15.0, "splay": 2.0}
DEFAULTS_THUMB = {"curl": 10.0, "splay": 2.0}


def finger_channels():
    out = []
    for side in SIDES:
        for f in FINGERS:
            out += [f"{side}{f}.curl", f"{side}{f}.splay"]
    return out


def default_value(ch):
    if ch.startswith("expr."):
        return 0.0
    name, kind = ch.split(".")
    return (DEFAULTS_THUMB if "Thumb" in name else DEFAULTS)[kind]


# ----------------------------------------------------------------------------- curves
def _slopes(t, v):
    """Fritsch-Carlson tangents: smooth through the keys, no overshoot."""
    n = len(t)
    if n < 2:
        return np.zeros(n)
    h = np.diff(t)
    d = np.diff(v) / np.where(h > 0, h, 1e-9)
    m = np.zeros(n)
    m[0], m[-1] = d[0], d[-1]
    for i in range(1, n - 1):
        if d[i - 1] * d[i] <= 0:
            m[i] = 0.0
        else:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    for i in range(n - 1):  # flat segments stay flat
        if d[i] == 0:
            m[i] = m[i + 1] = 0.0
    return m


def evaluate(keys, times, default=0.0):
    """keys: [[t, v], ...]; times (N,) seconds -> (N,) values."""
    times = np.asarray(times, float)
    if not keys:
        return np.full(len(times), float(default))
    k = sorted((float(a), float(b)) for a, b in keys)
    t = np.array([a for a, _ in k])
    v = np.array([b for _, b in k])
    if len(t) == 1:
        return np.full(len(times), v[0])
    m = _slopes(t, v)
    i = np.clip(np.searchsorted(t, times, side="right") - 1, 0, len(t) - 2)
    h = t[i + 1] - t[i]
    s = np.clip((times - t[i]) / np.where(h > 0, h, 1e-9), 0.0, 1.0)
    h00, h10 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s
    h01, h11 = -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    out = h00 * v[i] + h10 * h * m[i] + h01 * v[i + 1] + h11 * h * m[i + 1]
    out = np.where(times <= t[0], v[0], out)
    return np.where(times >= t[-1], v[-1], out)


# ----------------------------------------------------------------------------- fingers
def _axis_angle(axis, ang):
    """(..., 4) quaternions (x, y, z, w) about a fixed unit axis by angles (...)."""
    ang = np.asarray(ang, float)
    s = np.sin(ang / 2)[..., None]
    return np.concatenate([s * np.asarray(axis, float), np.cos(ang / 2)[..., None]], -1)


def _qmul(a, b):
    x1, y1, z1, w1 = np.moveaxis(a, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], -1)


def finger_rig(skel):
    """For every finger bone of the avatar: how curl and splay turn it.
    Axes are in the app's normalised bone frame (the bone's local frame at
    rest). Returns a list of dicts: bone, side, finger, seg (0..2), curl_axis,
    curl_max (degrees at 100), splay_axis, splay_max (degrees at 10; only the
    first segment of a finger spreads)."""
    rp = skel.rest_pos
    out = []
    for side in SIDES:
        hand = side + "Hand"
        if hand not in rp:
            continue
        ip, mp_, lp = (rp.get(side + f + "Proximal") for f in ("Index", "Middle", "Little"))
        if ip is None or mp_ is None or lp is None:
            continue
        along = normalize(mp_ - rp[hand])          # hand -> fingers
        across = normalize(lp - ip)                # index -> little finger
        across = normalize(across - along * np.dot(across, along))
        palm = normalize(np.cross(along, across))  # out of the palm
        if side == "left":
            palm = -palm
        # (the T-pose palm faces down: check and flip if the model is different)
        if palm[1] > 0.5:
            palm = -palm
        for f in FINGERS:
            segs = [side + f + s for s in FINGER_SEGS[f]]
            for k, b in enumerate(segs):
                if b not in rp:
                    continue
                nxt = segs[k + 1] if k + 1 < len(segs) else None
                if nxt is not None and nxt in rp:
                    d = rp[nxt] - rp[b]
                elif b in getattr(skel, "tips", {}) and skel.tips.get(b) is not None:
                    d = np.asarray(skel.tips[b]) - rp[b]
                else:
                    prev = segs[k - 1] if k > 0 else hand
                    d = rp[b] - rp[prev]
                if np.linalg.norm(d) < 1e-6:
                    d = along
                d = normalize(d)
                if f == "Thumb":
                    # the thumb folds across the palm, towards the little finger
                    bend = normalize(palm + 0.7 * across)
                else:
                    bend = palm
                ax = np.cross(d, bend)
                ax = normalize(ax) if np.linalg.norm(ax) > 1e-6 else normalize(np.cross(d, palm))
                mx, sgn = SPLAY_MAX[f]
                g = sgn * across  # where spreading moves the finger
                sax = palm if np.dot(np.cross(palm, d), g) >= 0 else -palm
                out.append({"bone": b, "side": side, "finger": f, "seg": k,
                            "curl_axis": [float(x) for x in ax], "curl_max": CURL_MAX[f][min(k, 2)],
                            "splay_axis": [float(x) for x in sax], "splay_max": mx if k == 0 else 0.0})
    return out


def finger_quats(rig_entry, curl, splay):
    """(N,4) local rotation of one finger bone for curl (N,) and splay (N,)."""
    a = np.radians(rig_entry["curl_max"]) * np.asarray(curl, float) / 100.0
    s = np.radians(rig_entry["splay_max"]) * np.asarray(splay, float) / 10.0
    return _qmul(_axis_angle(rig_entry["splay_axis"], s), _axis_angle(rig_entry["curl_axis"], a))


# ----------------------------------------------------------------------------- expressions
V0_TO_V1 = {"a": "aa", "e": "ee", "i": "ih", "o": "oh", "u": "ou", "blink": "blink", "joy": "happy",
            "angry": "angry", "sorrow": "sad", "fun": "relaxed", "lookup": "lookUp", "lookdown": "lookDown",
            "lookleft": "lookLeft", "lookright": "lookRight", "blink_l": "blinkLeft", "blink_r": "blinkRight",
            "neutral": "neutral"}


def vrm_expressions(gltf_json):
    """The avatar's expressions as the app shows them (the names three-vrm
    gives them): {name: {"binds": [(mesh index, morph index, weight 0..1)],
    "binary": bool}}."""
    ext = gltf_json.get("extensionsUsed", [])
    exts = gltf_json.get("extensions", {})
    out = {}
    if "VRMC_vrm" in exts:
        ex = exts["VRMC_vrm"].get("expressions", {})
        mesh_of_node = {i: n["mesh"] for i, n in enumerate(gltf_json.get("nodes", [])) if "mesh" in n}
        for group in ("preset", "custom"):
            for name, e in (ex.get(group) or {}).items():
                if name in out:
                    continue
                binds = [(mesh_of_node[b["node"]], int(b["index"]), float(b.get("weight", 1.0)))
                         for b in e.get("morphTargetBinds", []) if b.get("node") in mesh_of_node]
                out[name] = {"binds": binds, "binary": bool(e.get("isBinary", False))}
    elif "VRM" in exts or "VRM" in ext:
        groups = (exts.get("VRM", {}).get("blendShapeMaster") or {}).get("blendShapeGroups") or []
        for g in groups:
            p = g.get("presetName")
            name = V0_TO_V1.get(p) if p else None
            name = name or g.get("name")
            if not name or name in out:
                continue
            binds = [(int(b["mesh"]), int(b["index"]), float(b.get("weight", 100.0)) / 100.0)
                     for b in g.get("binds", []) if "mesh" in b and "index" in b]
            out[name] = {"binds": binds, "binary": bool(g.get("isBinary", False))}
    return out


def fbx_blend_channels(fbx_avatar):
    """{channel name: [deformer ids]} of an FBX avatar's blend shape channels."""
    from .fbx_reader import find, object_name, read_fbx
    tree = read_fbx(fbx_avatar.path)
    objs = find(tree, "Objects")
    out = {}
    for e in objs.children:
        if e.name == b"Deformer" and len(e.props) > 2 and e.props[2][1] == b"BlendShapeChannel":
            nm = object_name(e)
            out.setdefault(nm, []).append(int(e.props[0][1]))
    return out


def expression_names(skel, path):
    """The avatar's expressions / blend shapes, in the app's order."""
    try:
        if getattr(skel, "version", None) == "fbx":
            return sorted(fbx_blend_channels(skel.fbx).keys(), key=str.lower)
        from .vrm import read_vrm_json
        ex = vrm_expressions(read_vrm_json(path))
        order = ["happy", "angry", "sad", "relaxed", "surprised", "aa", "ih", "ou", "ee", "oh", "blink",
                 "blinkLeft", "blinkRight", "lookUp", "lookDown", "lookLeft", "lookRight", "neutral"]
        names = [n for n in order if n in ex] + sorted((n for n in ex if n not in order), key=str.lower)
        return names
    except Exception:
        return []


# ----------------------------------------------------------------------------- storage
def keys_path(workdir, char_id):
    return os.path.join(workdir, f"keys_c{int(char_id)}.json")


def load_keys(workdir, char_id):
    try:
        with open(keys_path(workdir, char_id)) as f:
            d = json.load(f)
        return clean_keys(d.get("channels", {}))
    except (OSError, ValueError):
        return {}


def clean_keys(ch):
    out = {}
    for name, ks in (ch or {}).items():
        if not isinstance(name, str) or not isinstance(ks, list):
            continue
        lo, hi = (0.0, 100.0) if name.startswith("expr.") else (CURL_RANGE if name.endswith(".curl") else SPLAY_RANGE)
        good = {}
        for k in ks:
            try:
                t, v = float(k[0]), float(k[1])
            except (TypeError, ValueError, IndexError):
                continue
            if np.isfinite(t) and np.isfinite(v) and t >= 0:
                good[round(t, 4)] = float(np.clip(v, lo, hi))
        if good:
            out[name] = [[t, good[t]] for t in sorted(good)]
    return out


def save_keys(workdir, char_id, channels):
    ch = clean_keys(channels)
    with open(keys_path(workdir, char_id), "w") as f:
        json.dump({"version": 1, "channels": ch}, f)
    return ch


# ----------------------------------------------------------------------------- baking
def bake_fingers(skel, anim, channels):
    """The animation's finger bones from the keys (a copy of anim['quats'])."""
    bones = list(anim["bones"])
    bi = {b: i for i, b in enumerate(bones)}
    Q = np.array(anim["quats"], np.float32, copy=True)
    n = len(Q)
    t = np.arange(n) / float(anim["fps"])
    vals = {}
    for e in finger_rig(skel):
        if e["bone"] not in bi:
            continue
        base = e["side"] + e["finger"]
        for kind in ("curl", "splay"):
            ch = f"{base}.{kind}"
            if ch not in vals:
                vals[ch] = evaluate(channels.get(ch), t, default_value(ch))
        Q[:, bi[e["bone"]]] = finger_quats(e, vals[base + ".curl"], vals[base + ".splay"])
    return Q


def bake_expressions(names, anim, channels):
    """{name: (N,) weight 0..1} for the expressions that have keys or a non-zero default."""
    n = int(anim["frames"])
    t = np.arange(n) / float(anim["fps"])
    out = {}
    for nm in names:
        ks = channels.get("expr." + nm)
        if ks:
            out[nm] = np.clip(evaluate(ks, t, 0.0) / 100.0, 0.0, 1.0)
    return out


def morph_weights(expr_w, vrm_ex):
    """VRM: expression weights -> {(mesh index, morph index): (N,) weight 0..1}."""
    out = {}
    for nm, w in expr_w.items():
        e = vrm_ex.get(nm)
        if not e:
            continue
        ww = (w > 0.5).astype(float) if e["binary"] else w
        for (m, i, k) in e["binds"]:
            out[(m, i)] = out.get((m, i), 0.0) + ww * k
    return {k: np.clip(v, 0.0, 1.0) for k, v in out.items()}


def rig_for_app(skel):
    """The finger rig for the app's live preview. For an FBX avatar each entry
    also names the file's node and gives the axes in that node's own frame
    (the app turns the node by the finger rotation in that frame)."""
    rig = finger_rig(skel)
    fbx = getattr(skel, "fbx", None) if getattr(skel, "version", None) == "fbx" else None
    for e in rig:
        if fbx is not None and e["bone"] in fbx.hb:
            m = fbx.hb[e["bone"]]
            Wt = np.asarray(fbx.Wt[m], float)
            e["node"] = fbx.models[m]["name"]
            e["node_curl_axis"] = [float(x) for x in Wt.T @ np.asarray(e["curl_axis"])]
            e["node_splay_axis"] = [float(x) for x in Wt.T @ np.asarray(e["splay_axis"])]
    return rig

"""Minimal binary FBX 7.4 writer for a skeleton + baked animation (+ camera).

The skeleton uses the VRM's own humanoid bone names and rest orientations so
the clip binds by name onto the same model imported into Blender / Unity /
Unreal / Maya, and Unity's Humanoid retargeting works on it directly.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

from .mathutil import mat_to_quat, quat_to_mat

KTIME = 46186158000
_FOOT_ID = b"\xfa\xbc\xab\x09\xd0\xc8\xd4\x66\xb1\x76\xfb\x83\x1c\xf7\x26\x7e"
_ALWAYS_SENTINEL = {b"AnimationStack", b"AnimationLayer"}
SENTINEL = b"\0" * 13


class Elem:
    def __init__(self, name, *props):
        self.name = name.encode() if isinstance(name, str) else name
        self.props = []
        self.children = []
        for p in props:
            self.add(p)

    # property helpers --------------------------------------------------------
    def add(self, p):
        if isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], str):
            self.props.append(p)  # (typecode, value) explicit
        elif isinstance(p, bool):
            self.props.append(("C", p))
        elif isinstance(p, int):
            self.props.append(("L", p) if abs(p) > 2 ** 31 - 1 else ("I", p))
        elif isinstance(p, float):
            self.props.append(("D", p))
        elif isinstance(p, (bytes, str)):
            self.props.append(("S", p.encode() if isinstance(p, str) else p))
        elif isinstance(p, np.ndarray):
            code = {np.dtype(np.float32): "f", np.dtype(np.float64): "d", np.dtype(np.int64): "l",
                    np.dtype(np.int32): "i"}[p.dtype]
            self.props.append((code, p))
        else:
            raise TypeError(type(p))
        return self

    def child(self, name, *props):
        e = Elem(name, *props)
        self.children.append(e)
        return e

    # encoding ------------------------------------------------------------------
    @staticmethod
    def _prop_bytes(code, v):
        c = code.encode()
        if code == "C":
            return c + struct.pack("<?", v)
        if code == "Y":
            return c + struct.pack("<h", v)
        if code == "I":
            return c + struct.pack("<i", v)
        if code == "L":
            return c + struct.pack("<q", v)
        if code == "F":
            return c + struct.pack("<f", v)
        if code == "D":
            return c + struct.pack("<d", v)
        if code in "SR":
            return c + struct.pack("<I", len(v)) + v
        if code in "fdli":
            raw = np.ascontiguousarray(v).tobytes()
            if len(raw) > 128:
                comp = zlib.compress(raw, 6)
                return c + struct.pack("<III", len(v), 1, len(comp)) + comp
            return c + struct.pack("<III", len(v), 0, len(raw)) + raw
        raise ValueError(code)

    def encode(self, offset, is_last):
        props = b"".join(self._prop_bytes(c, v) for c, v in self.props)
        head_len = 13 + len(self.name)
        body_start = offset + head_len + len(props)
        kids = b""
        pos = body_start
        for i, ch in enumerate(self.children):
            b = ch.encode(pos, i == len(self.children) - 1)
            kids += b
            pos += len(b)
        if self.children:
            kids += SENTINEL
        elif (not self.props and not is_last) or self.name in _ALWAYS_SENTINEL:
            kids += SENTINEL
        end = body_start + len(kids)
        head = struct.pack("<IIIB", end, len(self.props), len(props), len(self.name)) + self.name
        return head + props + kids


def P(parent, name, ptype, label, flags, *vals):
    e = parent.child("P", name, ptype, label, flags)
    for v in vals:
        e.add(v)
    return e


def name_class(name, cls):
    return name.encode() + b"\x00\x01" + cls.encode()


def write_fbx(path, nodes, frames_count, fps, camera=None, take_name="Take 001"):
    """nodes: list of dicts with keys
         name, parent (index or None), kind ("LimbNode"|"Null"), rest_t (3,), rest_r (3, degrees),
         anim_t (N,3) or None, anim_r (N,3) degrees or None
       camera: optional dict(pos (N,3), rot_deg (N,3), vfov (N,), aspect)
       All translations in centimetres, Y-up right-handed."""
    root = Elem(b"root")
    uid = [1000000]

    def new_id():
        uid[0] += 1
        return uid[0]

    hx = root.child("FBXHeaderExtension")
    hx.child("FBXHeaderVersion", 1003)
    hx.child("FBXVersion", 7400)
    ts = hx.child("CreationTimeStamp")
    ts.child("Version", 1000)
    for k, v in (("Year", 2026), ("Month", 1), ("Day", 1), ("Hour", 0), ("Minute", 0), ("Second", 0), ("Millisecond", 0)):
        ts.child(k, v)
    hx.child("Creator", "VRoid Mocap")
    root.child("FileId", ("R", bytes(range(16))))
    root.child("CreationTime", "1970-01-01 10:00:00:000")
    root.child("Creator", "VRoid Mocap")

    gs = root.child("GlobalSettings")
    gs.child("Version", 1000)
    p70 = gs.child("Properties70")
    P(p70, "UpAxis", "int", "Integer", "", 1)
    P(p70, "UpAxisSign", "int", "Integer", "", 1)
    P(p70, "FrontAxis", "int", "Integer", "", 2)
    P(p70, "FrontAxisSign", "int", "Integer", "", 1)
    P(p70, "CoordAxis", "int", "Integer", "", 0)
    P(p70, "CoordAxisSign", "int", "Integer", "", 1)
    P(p70, "OriginalUpAxis", "int", "Integer", "", 1)
    P(p70, "OriginalUpAxisSign", "int", "Integer", "", 1)
    P(p70, "UnitScaleFactor", "double", "Number", "", 1.0)
    P(p70, "OriginalUnitScaleFactor", "double", "Number", "", 1.0)
    P(p70, "AmbientColor", "ColorRGB", "Color", "", 0.0, 0.0, 0.0)
    P(p70, "DefaultCamera", "KString", "", "", "Producer Perspective")
    # FbxTime::EMode: 1=120, 2=100, 3=60, 4=50, 5=48, 6=30, 10=25 (PAL), 11=24, 14=custom
    modes = {120.0: 1, 100.0: 2, 60.0: 3, 50.0: 4, 48.0: 5, 30.0: 6, 25.0: 10, 24.0: 11}
    tmode = next((m for r, m in modes.items() if abs(fps - r) < 1e-3), 14)
    P(p70, "TimeMode", "enum", "", "", tmode)
    P(p70, "TimeSpanStart", "KTime", "Time", "", ("L", 0))
    P(p70, "TimeSpanStop", "KTime", "Time", "", ("L", int(round((frames_count - 1) / fps * KTIME))))
    P(p70, "CustomFrameRate", "double", "Number", "", float(fps))

    docs = root.child("Documents")
    docs.child("Count", 1)
    doc = docs.child("Document", ("L", new_id()), "Scene", "Scene")
    dp = doc.child("Properties70")
    P(dp, "SourceObject", "object", "", "")
    P(dp, "ActiveAnimStackName", "KString", "", "", take_name)
    doc.child("RootNode", ("L", 0))
    root.child("References")

    objects = Elem("Objects")
    conns = Elem("Connections")
    counts = {"Model": 0, "NodeAttribute": 0, "AnimationStack": 1, "AnimationLayer": 1,
              "AnimationCurveNode": 0, "AnimationCurve": 0}

    t_stop = int(round((frames_count - 1) / fps * KTIME))
    stack_id, layer_id = new_id(), new_id()
    st = objects.child("AnimationStack", ("L", stack_id), name_class(take_name, "AnimStack"), "")
    sp = st.child("Properties70")
    P(sp, "LocalStart", "KTime", "Time", "", ("L", 0))
    P(sp, "LocalStop", "KTime", "Time", "", ("L", t_stop))
    P(sp, "ReferenceStart", "KTime", "Time", "", ("L", 0))
    P(sp, "ReferenceStop", "KTime", "Time", "", ("L", t_stop))
    objects.children.append(Elem("AnimationLayer", ("L", layer_id), name_class("BaseLayer", "AnimLayer"), ""))
    conns.child("C", "OO", ("L", layer_id), ("L", stack_id))
    key_times = (np.arange(frames_count, dtype=np.int64) * KTIME / fps).round().astype(np.int64)

    def add_curves(model_id, prop, data, kind):
        cn_id = new_id()
        cn = objects.child("AnimationCurveNode", ("L", cn_id), name_class(kind, "AnimCurveNode"), "")
        cp = cn.child("Properties70")
        for ax, k in enumerate("XYZ"):
            P(cp, f"d|{k}", "Number", "", "A", float(data[0, ax]))
        counts["AnimationCurveNode"] += 1
        conns.child("C", "OO", ("L", cn_id), ("L", layer_id))
        conns.child("C", "OP", ("L", cn_id), ("L", model_id), prop)
        for ax, k in enumerate("XYZ"):
            cid = new_id()
            ac = objects.child("AnimationCurve", ("L", cid), name_class("", "AnimCurve"), "")
            ac.child("Default", float(data[0, ax]))
            ac.child("KeyVer", 4008)
            ac.child("KeyTime", key_times)
            ac.child("KeyValueFloat", data[:, ax].astype(np.float32))
            ac.child("KeyAttrFlags", np.array([1 << 2 | 1 << 8 | 1 << 13 | 1 << 14], np.int32))
            ac.child("KeyAttrDataFloat", np.array([0.0, 0.0, 9.419963346924634e-30, 0.0], np.float32))
            ac.child("KeyAttrRefCount", np.array([frames_count], np.int32))
            counts["AnimationCurve"] += 1
            conns.child("C", "OP", ("L", cid), ("L", cn_id), f"d|{k}")

    ids = []
    for nd in nodes:
        mid = new_id()
        ids.append(mid)
        if nd["kind"] == "LimbNode":
            aid = new_id()
            na = objects.child("NodeAttribute", ("L", aid), name_class(nd["name"], "NodeAttribute"), "LimbNode")
            nap = na.child("Properties70")
            P(nap, "Size", "double", "Number", "", 10.0)
            na.child("TypeFlags", "Skeleton")
            counts["NodeAttribute"] += 1
            conns.child("C", "OO", ("L", aid), ("L", mid))
        m = objects.child("Model", ("L", mid), name_class(nd["name"], "Model"), nd["kind"])
        m.child("Version", 232)
        mp = m.child("Properties70")
        P(mp, "RotationActive", "bool", "", "", 1)
        P(mp, "InheritType", "enum", "", "", 1)
        P(mp, "ScalingMax", "Vector3D", "Vector", "", 0.0, 0.0, 0.0)
        P(mp, "DefaultAttributeIndex", "int", "Integer", "", 0)
        t, r = nd["rest_t"], nd["rest_r"]
        P(mp, "Lcl Translation", "Lcl Translation", "", "A", float(t[0]), float(t[1]), float(t[2]))
        P(mp, "Lcl Rotation", "Lcl Rotation", "", "A", float(r[0]), float(r[1]), float(r[2]))
        P(mp, "Lcl Scaling", "Lcl Scaling", "", "A", 1.0, 1.0, 1.0)
        m.child("Shading", True)
        m.child("Culling", "CullingOff")
        counts["Model"] += 1
        parent = nd["parent"]
        conns.child("C", "OO", ("L", mid), ("L", ids[parent] if parent is not None else 0))
        if nd.get("anim_t") is not None:
            add_curves(mid, "Lcl Translation", nd["anim_t"], "T")
        if nd.get("anim_r") is not None:
            add_curves(mid, "Lcl Rotation", nd["anim_r"], "R")

    if camera is not None:
        aid, mid = new_id(), new_id()
        ca = objects.child("NodeAttribute", ("L", aid), name_class("VideoCamera", "NodeAttribute"), "Camera")
        cp = ca.child("Properties70")
        aspect = float(camera["aspect"])
        P(cp, "AspectWidth", "double", "Number", "", 1920.0)
        P(cp, "AspectHeight", "double", "Number", "", 1920.0 / aspect)
        P(cp, "ApertureMode", "enum", "", "", 2)  # focal length
        P(cp, "FieldOfViewY", "FieldOfViewY", "", "A", float(camera["vfov"][0]))
        hf = 2 * np.degrees(np.arctan(np.tan(np.radians(camera["vfov"][0]) / 2) * aspect))
        P(cp, "FieldOfView", "FieldOfView", "", "A", float(hf))
        P(cp, "FieldOfViewX", "FieldOfViewX", "", "A", float(hf))
        film_w = 36.0 / 25.4  # inches (36 mm full-frame width)
        film_h = film_w / aspect
        focal_mm = (film_h * 25.4 / 2) / np.tan(np.radians(np.asarray(camera["vfov"], np.float64)) / 2)
        P(cp, "FilmWidth", "double", "Number", "", film_w)
        P(cp, "FilmHeight", "double", "Number", "", film_h)
        P(cp, "FilmAspectRatio", "double", "Number", "", aspect)
        P(cp, "GateFit", "enum", "", "", 2)  # horizontal
        P(cp, "FocalLength", "Number", "", "A", float(focal_mm[0]))
        P(cp, "NearPlane", "double", "Number", "", 1.0)
        P(cp, "FarPlane", "double", "Number", "", 100000.0)
        ca.child("TypeFlags", "Camera")
        ca.child("GeometryVersion", 124)
        ca.child("Position", 0.0, 0.0, 0.0)
        ca.child("Up", 0.0, 1.0, 0.0)
        ca.child("LookAt", -1.0, 0.0, 0.0)
        ca.child("ShowInfoOnMoving", 1)
        ca.child("ShowAudio", 0)
        ca.child("AudioColor", 0.0, 1.0, 0.0)
        ca.child("CameraOrthoZoom", 1.0)
        counts["NodeAttribute"] += 1
        m = objects.child("Model", ("L", mid), name_class("VideoCamera", "Model"), "Camera")
        m.child("Version", 232)
        mp = m.child("Properties70")
        P(mp, "RotationActive", "bool", "", "", 1)
        P(mp, "InheritType", "enum", "", "", 1)
        p0, r0 = camera["pos"][0], camera["rot_deg"][0]
        P(mp, "Lcl Translation", "Lcl Translation", "", "A", *map(float, p0))
        P(mp, "Lcl Rotation", "Lcl Rotation", "", "A", *map(float, r0))
        P(mp, "Lcl Scaling", "Lcl Scaling", "", "A", 1.0, 1.0, 1.0)
        m.child("Shading", True)
        m.child("Culling", "CullingOff")
        counts["Model"] += 1
        conns.child("C", "OO", ("L", aid), ("L", mid))
        conns.child("C", "OO", ("L", mid), ("L", 0))
        add_curves(mid, "Lcl Translation", camera["pos"], "T")
        add_curves(mid, "Lcl Rotation", camera["rot_deg"], "R")
        # animated lens (changes between shots / zooms)
        cn_id = new_id()
        cn = objects.child("AnimationCurveNode", ("L", cn_id), name_class("FocalLength", "AnimCurveNode"), "")
        P(cn.child("Properties70"), "d|FocalLength", "Number", "", "A", float(focal_mm[0]))
        counts["AnimationCurveNode"] += 1
        conns.child("C", "OO", ("L", cn_id), ("L", layer_id))
        conns.child("C", "OP", ("L", cn_id), ("L", aid), "FocalLength")
        cid = new_id()
        ac = objects.child("AnimationCurve", ("L", cid), name_class("", "AnimCurve"), "")
        ac.child("Default", float(focal_mm[0]))
        ac.child("KeyVer", 4008)
        ac.child("KeyTime", key_times)
        ac.child("KeyValueFloat", focal_mm.astype(np.float32))
        ac.child("KeyAttrFlags", np.array([1 << 1], np.int32))  # constant (lens cuts between shots)
        ac.child("KeyAttrDataFloat", np.array([0.0, 0.0, 9.419963346924634e-30, 0.0], np.float32))
        ac.child("KeyAttrRefCount", np.array([frames_count], np.int32))
        counts["AnimationCurve"] += 1
        conns.child("C", "OP", ("L", cid), ("L", cn_id), "d|FocalLength")

    defs = root.child("Definitions")
    defs.child("Version", 100)
    defs.child("Count", 1 + sum(counts.values()))
    defs.child("ObjectType", "GlobalSettings").child("Count", 1)
    for k, v in counts.items():
        if v:
            defs.child("ObjectType", k).child("Count", v)
    root.children.append(objects)
    root.children.append(conns)
    takes = root.child("Takes")
    takes.child("Current", take_name)
    tk = takes.child("Take", take_name)
    tk.child("FileName", take_name.replace(" ", "_") + ".tak")
    tk.child("LocalTime", ("L", 0), ("L", t_stop))
    tk.child("ReferenceTime", ("L", 0), ("L", t_stop))

    out = bytearray(b"Kaydara FBX Binary  \x00\x1a\x00" + struct.pack("<I", 7400))
    pos = len(out)
    for i, ch in enumerate(root.children):
        b = ch.encode(pos, i == len(root.children) - 1)
        out += b
        pos += len(b)
    out += SENTINEL
    out += _FOOT_ID + b"\0" * 4
    pad = ((len(out) + 15) & ~15) - len(out)
    out += b"\0" * (pad if pad else 16)
    out += struct.pack("<I", 7400) + b"\0" * 120
    out += b"\xf8\x5a\x8c\x6a\xde\xf5\xd9\x7e\xec\xe9\x0c\xe3\x75\x8f\x29\x0b"
    with open(path, "wb") as f:
        f.write(out)


# ----------------------------------------------------------------------------- euler helpers
def mats_to_euler_xyz(R):
    """(N,3,3) -> (N,3) degrees, R = Rz @ Ry @ Rx (FBX eEulerXYZ), unwrapped over time."""
    sy = np.clip(-R[:, 2, 0], -1, 1)
    y = np.arcsin(sy)
    cy = np.cos(y)
    x = np.where(np.abs(cy) > 1e-6, np.arctan2(R[:, 2, 1], R[:, 2, 2]), np.arctan2(-R[:, 1, 2], R[:, 1, 1]))
    z = np.where(np.abs(cy) > 1e-6, np.arctan2(R[:, 1, 0], R[:, 0, 0]), 0.0)
    e = np.degrees(np.stack([x, y, z], 1))
    for i in range(1, len(e)):
        a = e[i]
        b = np.array([a[0] + 180, 180 - a[1], a[2] + 180])
        best = None
        for c in (a, b):
            c = c + 360 * np.round((e[i - 1] - c) / 360)
            d = np.abs(c - e[i - 1]).sum()
            if best is None or d < best[0]:
                best = (d, c)
        e[i] = best[1]
    return e


def export_animation_fbx(path, skel, anim, in_place=False, lock_height=False, include_camera=True,
                         frame_range=None):
    """Bake the solved animation (normalized-frame local quats) onto the VRM's
    raw armature and write it as FBX."""
    s, e = frame_range or (0, anim["frames"])
    bones = list(anim["bones"])
    Q = np.asarray(anim["quats"][s:e], np.float64)
    root = np.asarray(anim["root"][s:e], np.float64).copy()
    n = len(Q)
    off = np.zeros(3)
    if in_place:
        off[[0, 2]] = root[:, [0, 2]].copy().mean(0) * 0
        root[:, 0] = 0.0
        root[:, 2] = 0.0
    if lock_height:
        root[:, 1] = skel.rest_pos["hips"][1]
    M = skel.to_file
    # normalized global rotations
    Rl = quat_to_mat(Q.reshape(-1, 4)).reshape(n, len(bones), 3, 3)
    G = {}
    for i, b in enumerate(bones):
        p = skel.parent[b]
        G[b] = Rl[:, i] if p is None else G[p] @ Rl[:, i]
    nodes = [{"name": "Armature", "parent": None, "kind": "Null", "rest_t": np.zeros(3), "rest_r": np.zeros(3)}]
    index = {}
    W = {}
    for b in bones:
        Wb = np.einsum("ij,njk,kl->nil", M, G[b], M.T) @ skel.raw_world_rot[b]
        W[b] = Wb
        p = skel.parent[b]
        if p is None:
            local_r = Wb
            rest_local_r = skel.raw_world_rot[b]
            t_rest = skel.raw_world_pos[b]
            anim_t = (root @ M.T) * 100.0
        else:
            local_r = np.einsum("nji,njk->nik", W[p], Wb)
            rest_local_r = skel.raw_world_rot[p].T @ skel.raw_world_rot[b]
            t_rest = skel.raw_world_rot[p].T @ (skel.raw_world_pos[b] - skel.raw_world_pos[p])
            anim_t = None
        index[b] = len(nodes)
        nodes.append({
            "name": skel.node_name[b], "parent": index[p] if p is not None else 0, "kind": "LimbNode",
            "rest_t": t_rest * 100.0, "rest_r": mats_to_euler_xyz(rest_local_r[None])[0],
            "anim_t": anim_t, "anim_r": mats_to_euler_xyz(local_r),
        })
    camera = None
    if include_camera and "cam_pos" in anim:
        cp = np.asarray(anim["cam_pos"][s:e], np.float64).copy()
        if in_place:
            cp[:, 0] -= np.asarray(anim["root"][s:e, 0])
            cp[:, 2] -= np.asarray(anim["root"][s:e, 2])
        if lock_height:
            cp[:, 1] -= np.asarray(anim["root"][s:e, 1]) - skel.rest_pos["hips"][1]
        cq = quat_to_mat(np.asarray(anim["cam_quat"][s:e], np.float64))
        # FBX cameras look down their local +X axis; three.js cameras look down -Z.
        to_fbx_cam = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])  # columns: fbx X->-Z ... (see docs)
        Rc = np.einsum("ij,njk->nik", M, cq) @ to_fbx_cam
        camera = {"pos": (cp @ M.T) * 100.0, "rot_deg": mats_to_euler_xyz(Rc),
                  "vfov": np.asarray(anim["cam_vfov"][s:e], np.float64),
                  "aspect": anim["width"] / anim["height"]}
    write_fbx(path, nodes, n, anim["fps"], camera=camera)
    return path


def _unused():
    return mat_to_quat

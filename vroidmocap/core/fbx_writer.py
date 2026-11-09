"""Minimal binary FBX 7.4 writer for a skeleton + baked animation (+ camera).

The skeleton uses the VRM's own humanoid bone names and rest orientations so
the clip binds by name onto the same model imported into Blender / Unity /
Unreal / Maya, and Unity's Humanoid retargeting works on it directly.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np


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


def write_fbx(path, nodes, frames_count, fps, camera=None, take_name="Take 001", model=None):
    """nodes: list of dicts with keys
         name, parent (index or None), kind ("LimbNode"|"Null"), rest_t (3,), rest_r (3, degrees),
         anim_t (N,3) or None, anim_r (N,3) degrees or None
       camera: optional dict(pos (N,3), rot_deg (N,3), vfov (N,), aspect)
       model: optional dict(meshes=[...], materials=[...], textures=[...]) (see _write_model)
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
        sc = nd.get("rest_s", (1.0, 1.0, 1.0))
        P(mp, "Lcl Scaling", "Lcl Scaling", "", "A", float(sc[0]), float(sc[1]), float(sc[2]))
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

    if model is not None:
        _write_model(model, objects, conns, counts, ids, new_id)

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


def _mat16(m):
    """4x4 numpy matrix -> FBX column-major double array."""
    return np.ascontiguousarray(np.asarray(m, np.float64).T.reshape(-1))


def _write_model(model, objects, conns, counts, ids, new_id):
    """Meshes with skinning, materials and embedded textures.

    model["meshes"]: dict(name, vertices (n,3) cm, normals (n,3)|None, uvs (n,2)|None,
                          tris (m,3), tri_mat (m,), materials [material index],
                          parent (node index or None), skin [(node index, vertex idx, weights, bind 4x4 cm)] or None)
    model["materials"]: dict(name, color (4,), texture index|None, transparent bool)
    model["textures"]: dict(name, filename, data bytes)"""
    for k in ("Geometry", "Material", "Texture", "Video", "Deformer", "Pose"):
        counts.setdefault(k, 0)
    tex_ids = []
    for t in model.get("textures", []):
        vid, tid = new_id(), new_id()
        v = objects.child("Video", ("L", vid), name_class(t["name"], "Video"), "Clip")
        v.child("Type", "Clip")
        vp = v.child("Properties70")
        P(vp, "Path", "KString", "XRefUrl", "", t["filename"])
        v.child("UseMipMap", 0)
        v.child("Filename", t["filename"])
        v.child("RelativeFilename", t["filename"])
        v.child("Content", ("R", t["data"]))
        tx = objects.child("Texture", ("L", tid), name_class(t["name"], "Texture"), "")
        tx.child("Type", "TextureVideoClip")
        tx.child("Version", 202)
        tx.child("TextureName", name_class(t["name"], "Texture"))
        tp = tx.child("Properties70")
        P(tp, "UseMaterial", "bool", "", "", 1)
        tx.child("Media", name_class(t["name"], "Video"))
        tx.child("FileName", t["filename"])
        tx.child("RelativeFilename", t["filename"])
        tx.child("ModelUVTranslation", 0.0, 0.0)
        tx.child("ModelUVScaling", 1.0, 1.0)
        tx.child("Texture_Alpha_Source", "None")
        tx.child("Cropping", 0, 0, 0, 0)
        conns.child("C", "OO", ("L", vid), ("L", tid))
        counts["Video"] += 1
        counts["Texture"] += 1
        tex_ids.append(tid)
    mat_ids = []
    for mt in model.get("materials", []):
        mid = new_id()
        m = objects.child("Material", ("L", mid), name_class(mt["name"], "Material"), "")
        m.child("Version", 102)
        m.child("ShadingModel", "phong")
        m.child("MultiLayer", 0)
        mp = m.child("Properties70")
        c = mt["color"]
        P(mp, "DiffuseColor", "Color", "", "A", float(c[0]), float(c[1]), float(c[2]))
        P(mp, "DiffuseFactor", "Number", "", "A", 1.0)
        P(mp, "SpecularFactor", "Number", "", "A", 0.0)
        P(mp, "Shininess", "Number", "", "A", 2.0)
        P(mp, "Opacity", "Number", "", "A", float(c[3]))
        P(mp, "TransparencyFactor", "Number", "", "A", float(1.0 - c[3]))
        counts["Material"] += 1
        mat_ids.append(mid)
        if mt.get("texture") is not None:
            conns.child("C", "OP", ("L", tex_ids[mt["texture"]]), ("L", mid), "DiffuseColor")
            if mt.get("transparent"):
                conns.child("C", "OP", ("L", tex_ids[mt["texture"]]), ("L", mid), "TransparentColor")
    pose_nodes = []
    for ms in model.get("meshes", []):
        gid, mid = new_id(), new_id()
        V = np.asarray(ms["vertices"], np.float64)
        T = np.asarray(ms["tris"], np.int64)
        g = objects.child("Geometry", ("L", gid), name_class(ms["name"], "Geometry"), "Mesh")
        g.child("Properties70")
        g.child("GeometryVersion", 124)
        g.child("Vertices", np.ascontiguousarray(V.reshape(-1)))
        pvi = T.astype(np.int32).copy()
        pvi[:, 2] = -pvi[:, 2] - 1
        g.child("PolygonVertexIndex", np.ascontiguousarray(pvi.reshape(-1)))
        layer_elems = []
        if ms.get("normals") is not None:
            ln = g.child("LayerElementNormal", 0)
            ln.child("Version", 102)
            ln.child("Name", "")
            ln.child("MappingInformationType", "ByPolygonVertex")
            ln.child("ReferenceInformationType", "Direct")
            ln.child("Normals", np.ascontiguousarray(np.asarray(ms["normals"], np.float64)[T.reshape(-1)].reshape(-1)))
            layer_elems.append("LayerElementNormal")
        if ms.get("uvs") is not None:
            lu = g.child("LayerElementUV", 0)
            lu.child("Version", 101)
            lu.child("Name", "UVMap")
            lu.child("MappingInformationType", "ByPolygonVertex")
            lu.child("ReferenceInformationType", "IndexToDirect")
            lu.child("UV", np.ascontiguousarray(np.asarray(ms["uvs"], np.float64).reshape(-1)))
            lu.child("UVIndex", np.ascontiguousarray(T.astype(np.int32).reshape(-1)))
            layer_elems.append("LayerElementUV")
        lm = g.child("LayerElementMaterial", 0)
        lm.child("Version", 101)
        lm.child("Name", "")
        lm.child("MappingInformationType", "ByPolygon")
        lm.child("ReferenceInformationType", "IndexToDirect")
        lm.child("Materials", np.ascontiguousarray(np.asarray(ms["tri_mat"], np.int32)))
        layer_elems.append("LayerElementMaterial")
        ly = g.child("Layer", 0)
        ly.child("Version", 100)
        for le in layer_elems:
            e = ly.child("LayerElement")
            e.child("Type", le)
            e.child("TypedIndex", 0)
        counts["Geometry"] += 1
        m = objects.child("Model", ("L", mid), name_class(ms["name"], "Model"), "Mesh")
        m.child("Version", 232)
        mp = m.child("Properties70")
        P(mp, "Lcl Translation", "Lcl Translation", "", "A", 0.0, 0.0, 0.0)
        P(mp, "Lcl Rotation", "Lcl Rotation", "", "A", 0.0, 0.0, 0.0)
        P(mp, "Lcl Scaling", "Lcl Scaling", "", "A", 1.0, 1.0, 1.0)
        P(mp, "DefaultAttributeIndex", "int", "Integer", "", 0)
        m.child("Shading", True)
        m.child("Culling", "CullingOff")
        counts["Model"] += 1
        conns.child("C", "OO", ("L", gid), ("L", mid))
        parent = ms.get("parent")
        conns.child("C", "OO", ("L", mid), ("L", ids[parent] if parent is not None else 0))
        for mi in ms["materials"]:
            conns.child("C", "OO", ("L", mat_ids[mi]), ("L", mid))
        pose_nodes.append((mid, np.eye(4)))
        if ms.get("skin"):
            sid = new_id()
            sk = objects.child("Deformer", ("L", sid), name_class(ms["name"], "Deformer"), "Skin")
            sk.child("Version", 101)
            sk.child("Link_DeformAcuracy", 50.0)
            counts["Deformer"] += 1
            conns.child("C", "OO", ("L", sid), ("L", gid))
            for (node_i, vidx, wts, bind) in ms["skin"]:
                cid = new_id()
                cl = objects.child("Deformer", ("L", cid), name_class("", "SubDeformer"), "Cluster")
                cl.child("Version", 100)
                cl.child("UserData", "", "")
                cl.child("Indexes", np.ascontiguousarray(np.asarray(vidx, np.int32)))
                cl.child("Weights", np.ascontiguousarray(np.asarray(wts, np.float64)))
                cl.child("Transform", _mat16(np.linalg.inv(bind)))
                cl.child("TransformLink", _mat16(bind))
                counts["Deformer"] += 1
                conns.child("C", "OO", ("L", cid), ("L", sid))
                conns.child("C", "OO", ("L", ids[node_i]), ("L", cid))
    for (node_i, bind) in model.get("bind_nodes", []):
        pose_nodes.append((ids[node_i], bind))
    if pose_nodes:
        pid = new_id()
        po = objects.child("Pose", ("L", pid), name_class("BindPose", "Pose"), "BindPose")
        po.child("Type", "BindPose")
        po.child("Version", 100)
        po.child("NbPoseNodes", len(pose_nodes))
        for (nid, mtx) in pose_nodes:
            pn = po.child("PoseNode")
            pn.child("Node", ("L", nid))
            pn.child("Matrix", _mat16(mtx))
        counts["Pose"] += 1


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
                         frame_range=None, vrm_path=None, include_model=False, texture_dir=None):
    """Bake the solved animation onto the avatar's complete node hierarchy
    (standard VRoid structure: every bone incl. hair / skirt bones, original
    names and rest transforms) and write it as FBX. With include_model the
    avatar's meshes, skin weights, materials and textures are included."""
    from .gltf import Glb
    from .mathutil import quat_to_mat as q2m
    s_, e_ = frame_range or (0, anim["frames"])
    bones = list(anim["bones"])
    Q = np.asarray(anim["quats"][s_:e_], np.float64)
    root = np.asarray(anim["root"][s_:e_], np.float64).copy()
    n = len(Q)
    if in_place:
        root[:, 0] = 0.0
        root[:, 2] = 0.0
    if lock_height:
        root[:, 1] = skel.rest_pos["hips"][1]
    M = skel.to_file
    Rl = q2m(Q.reshape(-1, 4)).reshape(n, len(bones), 3, 3)
    G = {}
    for i, b in enumerate(bones):
        p = skel.parent[b]
        G[b] = Rl[:, i] if p is None else G[p] @ Rl[:, i]
    gl = Glb(vrm_path or skel.path)
    node_of_bone = {}
    for b in bones:
        for i, nd in enumerate(gl.nodes):
            if nd.get("name") == skel.node_name[b]:
                node_of_bone[i] = b
                break
    mesh_nodes = {i for i, nd in enumerate(gl.nodes) if "mesh" in nd}
    joints = {j for sk in gl.skins for j in sk["joints"]}
    # skeleton nodes: everything that is not a mesh, parent-first
    order = []

    def visit(i):
        if i in mesh_nodes:
            return
        order.append(i)
        for c in gl.nodes[i].get("children", []):
            visit(c)
    for r in gl.scene_roots():
        visit(r)
    has_joint_below = {}

    def mark(i):
        v = i in joints or i in node_of_bone
        for c in gl.nodes[i].get("children", []):
            if c not in mesh_nodes:
                v = mark(c) or v
        has_joint_below[i] = v
        return v
    for r in gl.scene_roots():
        if r not in mesh_nodes:
            mark(r)
    # per-frame world transforms (rotation + position) of every skeleton node
    Wr, Wp = {}, {}
    local = {i: gl.local_matrix(i) for i in order}
    idx_of = {i: k for k, i in enumerate(order)}
    nodes = []
    for i in order:
        L = local[i]
        sc = np.linalg.norm(L[:3, :3], axis=0)
        Lr = L[:3, :3] / np.maximum(sc, 1e-12)
        par = gl.parent.get(i)
        par = par if par in idx_of else None
        prw = Wr[par] if par is not None else np.broadcast_to(np.eye(3), (n, 3, 3))
        ppw = Wp[par] if par is not None else np.zeros((n, 3))
        anim_r = anim_t = None
        if i in node_of_bone:
            b = node_of_bone[i]
            Wr[i] = np.einsum("ij,njk,kl->nil", M, G[b], M.T) @ skel.raw_world_rot[b]
            if skel.parent[b] is None:  # hips: root motion
                Wp[i] = root @ M.T
                anim_t = np.einsum("nji,nj->ni", prw, Wp[i] - ppw) * 100.0
            else:
                Wp[i] = ppw + np.einsum("nij,j->ni", prw, L[:3, 3])
            anim_r = mats_to_euler_xyz(np.einsum("nji,njk->nik", prw, Wr[i]))
        else:
            Wr[i] = prw @ Lr
            Wp[i] = ppw + np.einsum("nij,j->ni", prw, L[:3, 3])
        nodes.append({
            "name": gl.nodes[i].get("name", f"node{i}"), "parent": idx_of[par] if par is not None else None,
            "kind": "LimbNode" if has_joint_below.get(i) else "Null",
            "rest_t": L[:3, 3] * 100.0, "rest_r": mats_to_euler_xyz(Lr[None])[0], "rest_s": sc,
            "anim_t": anim_t, "anim_r": anim_r,
        })
    model = None
    if include_model:
        model = build_model(gl, idx_of, texture_dir)
    camera = None
    if include_camera and "cam_pos" in anim:
        cp = np.asarray(anim["cam_pos"][s_:e_], np.float64).copy()
        if in_place:
            cp[:, 0] -= np.asarray(anim["root"][s_:e_, 0])
            cp[:, 2] -= np.asarray(anim["root"][s_:e_, 2])
        if lock_height:
            cp[:, 1] -= np.asarray(anim["root"][s_:e_, 1]) - skel.rest_pos["hips"][1]
        cq = q2m(np.asarray(anim["cam_quat"][s_:e_], np.float64))
        # FBX cameras look down their local +X axis; three.js cameras look down -Z
        to_fbx_cam = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
        Rc = np.einsum("ij,njk->nik", M, cq) @ to_fbx_cam
        camera = {"pos": (cp @ M.T) * 100.0, "rot_deg": mats_to_euler_xyz(Rc),
                  "vfov": np.asarray(anim["cam_vfov"][s_:e_], np.float64),
                  "aspect": anim["width"] / anim["height"]}
    write_fbx(path, nodes, n, anim["fps"], camera=camera, model=model)
    return path


def build_model(gl, idx_of, texture_dir=None):
    """Meshes / skins / materials / textures of a VRM for the FBX writer (cm)."""
    import os
    textures, tex_of_image = [], {}
    materials = []
    for mi, m in enumerate(gl.materials):
        pbr = m.get("pbrMetallicRoughness", {})
        col = list(pbr.get("baseColorFactor", [1, 1, 1, 1]))
        tex = None
        bct = pbr.get("baseColorTexture")
        if bct is not None and bct.get("index") is not None and bct["index"] < len(gl.textures):
            img = gl.textures[bct["index"]].get("source")
            if img is not None:
                if img not in tex_of_image:
                    data, mime = gl.image_bytes(img)
                    if data:
                        ext = ".jpg" if mime == "image/jpeg" else ".png"
                        nm = gl.images[img].get("name") or ""
                        nm = "".join(ch if (ch.isascii() and ch.isalnum()) or ch in "-_." else "_" for ch in nm).strip("_")
                        nm = f"tex{img:02d}_{nm}" if nm else f"tex{img:02d}"
                        fn = nm if nm.lower().endswith(ext) else nm + ext
                        if texture_dir:
                            os.makedirs(texture_dir, exist_ok=True)
                            with open(os.path.join(texture_dir, fn), "wb") as f:
                                f.write(data)
                            fn = os.path.join(os.path.basename(texture_dir), fn)
                        tex_of_image[img] = len(textures)
                        textures.append({"name": nm, "filename": fn, "data": data})
                tex = tex_of_image.get(img)
        materials.append({"name": m.get("name", f"material{mi}"), "color": col, "texture": tex,
                          "transparent": m.get("alphaMode", "OPAQUE") != "OPAQUE"})
    meshes = []
    bind_nodes = {}
    for ni, nd in enumerate(gl.nodes):
        if "mesh" not in nd:
            continue
        mesh = gl.meshes[nd["mesh"]]
        Vs, Ns, Us, Ts, TM, J, Wt = [], [], [], [], [], [], []
        mats, base = [], 0
        for prim in mesh["primitives"]:
            if prim.get("mode", 4) != 4:
                continue
            at = prim["attributes"]
            v = gl.accessor(at["POSITION"])
            nv = len(v)
            Vs.append(v)
            Ns.append(gl.accessor(at["NORMAL"]) if "NORMAL" in at else np.zeros((nv, 3)))
            uv = gl.accessor(at["TEXCOORD_0"]) if "TEXCOORD_0" in at else np.zeros((nv, 2))
            Us.append(np.stack([uv[:, 0], 1.0 - uv[:, 1]], 1))
            ix = gl.accessor(prim["indices"]).reshape(-1).astype(np.int64) if "indices" in prim else np.arange(nv)
            Ts.append(ix.reshape(-1, 3) + base)
            mat = prim.get("material", 0)
            if mat not in mats:
                mats.append(mat)
            TM.append(np.full(len(ix) // 3, mats.index(mat)))
            J.append(gl.accessor(at["JOINTS_0"]).astype(np.int64) if "JOINTS_0" in at else np.zeros((nv, 4), np.int64))
            Wt.append(gl.accessor(at["WEIGHTS_0"]) if "WEIGHTS_0" in at else np.zeros((nv, 4)))
            base += nv
        if not Vs:
            continue
        V = np.concatenate(Vs)
        entry = {"name": nd.get("name", mesh.get("name", f"mesh{ni}")), "normals": np.concatenate(Ns),
                 "uvs": np.concatenate(Us), "tris": np.concatenate(Ts), "tri_mat": np.concatenate(TM),
                 "materials": mats, "parent": None, "skin": None}
        if "skin" in nd:
            sk = gl.skins[nd["skin"]]
            ibm = gl.accessor(sk["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1) \
                if "inverseBindMatrices" in sk else np.tile(np.eye(4), (len(sk["joints"]), 1, 1))
            Jall, Wall = np.concatenate(J), np.concatenate(Wt)
            wsum = Wall.sum(1, keepdims=True)
            Wall = Wall / np.where(wsum > 0, wsum, 1)
            clusters = []
            for k, jn in enumerate(sk["joints"]):
                if jn not in idx_of:
                    continue
                sel = (Jall == k) & (Wall > 1e-6)
                vid = np.where(sel.any(1))[0]
                if len(vid) == 0:
                    continue
                w = (Wall * sel).sum(1)[vid]
                bind = np.linalg.inv(ibm[k])
                bind[:3, 3] *= 100.0
                clusters.append((idx_of[jn], vid, w, bind))
                bind_nodes[idx_of[jn]] = bind
            entry["skin"] = clusters
            entry["vertices"] = V * 100.0  # skinned glTF vertices are in bind (world) space
        else:
            W = gl.world_matrix(ni)
            entry["vertices"] = (V @ W[:3, :3].T + W[:3, 3]) * 100.0
            entry["normals"] = entry["normals"] @ W[:3, :3].T
        meshes.append(entry)
    return {"meshes": meshes, "materials": materials, "textures": textures,
            "bind_nodes": sorted(bind_nodes.items())}

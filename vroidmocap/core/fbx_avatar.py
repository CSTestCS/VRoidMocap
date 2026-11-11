"""FBX character models as avatars.

* reads the model's skeleton (any FBX: binary or ASCII; Y-up or Z-up; cm or m),
* finds the humanoid bones by name and by the shape of the hierarchy (Mixamo,
  VRoid FBX, Unreal, Unity / Blender humanoid, Rigify, 3ds Max Biped, MMD...),
* stands the skeleton in a T-pose for the solver (arms out along X, legs down),
  whatever its own rest pose is (A-pose, relaxed...): a `VrmSkeleton` like the
  one read from a .vrm, so everything downstream works the same,
* turns solved motion back into the model's own bones (local rotations in the
  file, hips translation), for the viewer and for export,
* exports by writing the original FBX again with the animation added: the
  mesh, skin, materials and textures are the file's own, untouched (or only the
  skeleton, without the model).
"""
from __future__ import annotations

import os
import re

import numpy as np

from .fbx_reader import find, find_all, object_name, props70, read_fbx
from .fbx_writer import KTIME, AnimWriter, Elem, P, encode_file, name_class
from .mathutil import mat_to_quat, normalize, quat_to_mat, rot_between
from .vrm import FINGER_SEGS, FINGERS, HUMAN_BONES, VrmSkeleton

ORDERS = ["XYZ", "XZY", "YZX", "YXZ", "ZXY", "ZYX", "XYZ"]  # FbxEuler::EOrder (6 = spheric)


def _axis_rot(ax, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if ax == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if ax == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_to_mat(deg, order="XYZ"):
    """FBX Euler angles (degrees) -> matrix; 'XYZ' rotates about X first."""
    m = np.eye(3)
    for ax in order:
        m = _axis_rot(ax, deg["XYZ".index(ax)]) @ m
    return m


def mats_to_euler(R, order="XYZ"):
    """(N,3,3) -> (N,3) degrees for the given FBX rotation order, continuous over time."""
    idx = ["XYZ".index(a) for a in order]
    Pm = np.zeros((3, 3))
    for r, c in enumerate(idx):
        Pm[r, c] = 1.0
    sgn = np.linalg.det(Pm)
    Rp = np.einsum("ij,njk,lk->nil", Pm, R, Pm)  # the same rotation in permuted axes: XYZ order
    y = np.arcsin(np.clip(-Rp[:, 2, 0], -1, 1))
    cy = np.cos(y)
    x = np.where(np.abs(cy) > 1e-6, np.arctan2(Rp[:, 2, 1], Rp[:, 2, 2]), np.arctan2(-Rp[:, 1, 2], Rp[:, 1, 1]))
    z = np.where(np.abs(cy) > 1e-6, np.arctan2(Rp[:, 1, 0], Rp[:, 0, 0]), 0.0)
    e3 = np.degrees(np.stack([x, y, z], 1)) * sgn
    for i in range(1, len(e3)):  # the equivalent triple closest to the previous frame
        a = e3[i]
        b = np.array([a[0] + 180, 180 * sgn - a[1] if sgn > 0 else -180 - a[1], a[2] + 180])
        best = None
        for c in (a, b):
            c = c + 360 * np.round((e3[i - 1] - c) / 360)
            d = np.abs(c - e3[i - 1]).sum()
            if best is None or d < best[0]:
                best = (d, c)
        e3[i] = best[1]
    out = np.zeros_like(e3)
    for k, c in enumerate(idx):
        out[:, c] = e3[:, k]
    return out


def _vec(p70, key, default=(0.0, 0.0, 0.0)):
    v = p70.get(key)
    if not v or len(v) < 3:
        return np.array(default, float)
    return np.array([float(x) for x in v[:3]])


def _t(v):
    m = np.eye(4)
    m[:3, 3] = v
    return m


def _r(m3):
    m = np.eye(4)
    m[:3, :3] = m3
    return m


def local_matrix(p70):
    """FBX local transform: T * Roff * Rp * Rpre * R * Rpost^-1 * Rp^-1 * Soff * Sp * S * Sp^-1."""
    order = ORDERS[int(p70.get("RotationOrder", [0])[0]) if p70.get("RotationOrder") else 0]
    T = _vec(p70, "Lcl Translation")
    R = euler_to_mat(_vec(p70, "Lcl Rotation"), order)
    S = _vec(p70, "Lcl Scaling", (1, 1, 1))
    Rpre = euler_to_mat(_vec(p70, "PreRotation"))
    Rpost = euler_to_mat(_vec(p70, "PostRotation"))
    Roff, Rp = _vec(p70, "RotationOffset"), _vec(p70, "RotationPivot")
    Soff, Sp = _vec(p70, "ScalingOffset"), _vec(p70, "ScalingPivot")
    Sm = np.diag([S[0], S[1], S[2], 1.0])
    return (_t(T) @ _t(Roff) @ _t(Rp) @ _r(Rpre) @ _r(R) @ _r(Rpost.T) @ _t(-Rp) @ _t(Soff) @ _t(Sp)
            @ Sm @ _t(-Sp))


# ----------------------------------------------------------------------------- names
_PREFIX = {"j", "bip", "bip01", "bip001", "c", "mixamorig", "def", "org", "mch", "sk", "armature",
           "bn", "bone", "jnt", "joint", "b", "cf", "valvebiped", "biped", "skeleton", "rig", "ctrl"}
SKIP_WORDS = ("twist", "roll", "helper", "adjust", "nub", "dummy", "null", "pole", "target", "cancel",
              "キャンセル", "_end", "socket", "weapon", "prop", "attach", "corrective", "jiggle")
FINGER_WORDS = [("Thumb", ("thumb", "親指", "おやゆび")),
                ("Index", ("index", "人指", "人差指", "forefinger", "pointer")),
                ("Middle", ("middle", "中指")),
                ("Ring", ("ring", "薬指")),
                ("Little", ("little", "pinky", "pinkie", "小指"))]


def _tokens(name):
    n = name.split(":")[-1].split("|")[-1]
    n = re.sub(r"([a-z])([A-Z])", r"\1 \2", n)
    n = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", n)
    return [t for t in re.split(r"[^0-9a-zA-Z぀-ヿ一-鿿０-９]+", n.lower()) if t]


def classify(name):
    """(kind, side, number) for a bone name. kind: hips/spine/neck/head/eye/shoulder/
    upperArm/lowerArm/hand/upperLeg/lowerLeg/leg/foot/toes/finger:<F>/skip/None."""
    toks = _tokens(name)
    raw = name.lower()
    side = None
    if "left" in toks or "l" in toks or "左" in raw or re.search(r"(^|[^a-z])left", raw):
        side = "left"
    elif "right" in toks or "r" in toks or "右" in raw or re.search(r"(^|[^a-z])right", raw):
        side = "right"
    rest = [t for t in toks if t not in ("left", "right", "l", "r") and t not in _PREFIX]
    rest = [t.replace("左", "").replace("右", "") for t in rest]
    base = "".join(rest)
    m = re.search(r"([0-9０-９]+)$", base)
    num = int(m.group(1).translate(str.maketrans("０１２３４５６７８９", "0123456789"))) if m else None
    word = re.sub(r"[0-9０-９]+$", "", base)
    for fk, words in FINGER_WORDS:
        if any(w in base for w in words):
            if any(w in base for w in ("twist", "end", "nub", "tip", "null", "dummy")):
                return "skip", side, num
            return "finger:" + fk, side, num
    # helper / twist / end / IK bones are not part of the humanoid
    if any(w in base for w in SKIP_WORDS) or any(t in ("ik", "fk", "d", "ex", "end", "tip", "nub") for t in rest):
        return "skip", side, num
    if any(w in base for w in ("toe", "ball", "つま先")):
        return "toes", side, num
    if any(w in base for w in ("foot", "ankle", "足首")):
        return "foot", side, num
    if any(w in base for w in ("upleg", "upperleg", "thigh", "femur")):
        return "upperLeg", side, num
    if any(w in base for w in ("calf", "shin", "knee", "lowerleg", "lowleg", "ひざ")):
        return "lowerLeg", side, num
    if word in ("leg", "足") or (side and word == "leg"):
        return "leg", side, num
    if any(w in base for w in ("forearm", "lowerarm", "lowarm", "elbow", "ひじ")):
        return "lowerArm", side, num
    if any(w in base for w in ("upperarm", "uparm")) or word in ("arm", "腕"):
        return "upperArm", side, num
    if any(w in base for w in ("hand", "wrist", "手首", "palm")):
        return "hand", side, num
    if any(w in base for w in ("shoulder", "clavicle", "collar", "肩")):
        return "shoulder", side, num
    if side and any(w in base for w in ("eye", "目")):
        return "eye", side, num
    if not side and any(w in base for w in ("head", "頭")):
        return "head", None, num
    if not side and any(w in base for w in ("neck", "首")):
        return "neck", None, num
    if not side and any(w in base for w in ("hips", "pelvis", "lowerbody", "下半身")) or (not side and word == "hip"):
        return "hips", None, num
    if not side and any(w in base for w in ("spine", "chest", "upperbody", "上半身", "torso", "abdomen", "ribcage")):
        return "spine", None, num
    return None, side, num


# ----------------------------------------------------------------------------- scene
class FbxAvatar:
    def __init__(self, path):
        self.path = str(path)
        self.tree = read_fbx(path)
        objs = find(self.tree, "Objects")
        conns = find(self.tree, "Connections")
        if objs is None or conns is None:
            raise ValueError("This FBX file has no objects")
        self.models = {}
        for e in find_all(objs, "Model"):
            mid = int(e.props[0][1])
            kind = e.props[2][1].decode() if len(e.props) > 2 and isinstance(e.props[2][1], bytes) else ""
            self.models[mid] = {"name": object_name(e), "kind": kind, "elem": e, "p70": props70(e)}
        self.parent = {}
        for c in find_all(conns, "C"):
            if c.props and c.props[0][1] == b"OO":
                ch, pa = int(c.props[1][1]), int(c.props[2][1])
                if ch in self.models and (pa in self.models or pa == 0):
                    self.parent[ch] = pa if pa in self.models else None
        self.kids = {m: [] for m in self.models}
        for ch, pa in self.parent.items():
            if pa is not None:
                self.kids[pa].append(ch)
        self.local = {m: local_matrix(d["p70"]) for m, d in self.models.items()}
        self.world = {}
        for m in self.models:
            self._world(m)
        gs = find(self.tree, "GlobalSettings")
        self.gs = props70(gs) if gs is not None else {}
        self._map_humanoid()
        self._frames()
        self._tpose()

    def _world(self, m):
        if m not in self.world:
            p = self.parent.get(m)
            self.world[m] = self.local[m] if p is None else self._world(p) @ self.local[m]
        return self.world[m]

    def _path(self, a, b):
        """Models from a (exclusive) down to b (exclusive), or None if a is not an ancestor of b."""
        out = []
        x = self.parent.get(b)
        while x is not None and x != a:
            out.append(x)
            x = self.parent.get(x)
        return out[::-1] if x == a else None

    def _ancestors(self, m):
        out = []
        x = self.parent.get(m)
        while x is not None:
            out.append(x)
            x = self.parent.get(x)
        return out

    def _depth(self, m):
        return len(self._ancestors(m))

    # ------------------------------------------------------------------ humanoid
    def _map_humanoid(self):
        skel_kinds = ("LimbNode", "Null", "Root", "")
        cand = {}
        for m, d in self.models.items():
            if d["kind"] not in skel_kinds:
                continue
            k, side, num = classify(d["name"])
            if k and k != "skip":
                cand.setdefault((k, side), []).append((m, num))
        hb = {}

        def pick(k, side=None, under=None):
            lst = cand.get((k, side), [])
            if under is not None:
                lst = [x for x in lst if under in self._ancestors(x[0])]
            if not lst:
                return None
            return min(lst, key=lambda x: self._depth(x[0]))[0]

        head = pick("head")
        legs = {}
        for side in ("left", "right"):
            foot = pick("foot", side)
            up = pick("upperLeg", side) or pick("leg", side)
            legs[side] = (up, foot)
        if head is None or any(v[0] is None for v in legs.values()):
            raise ValueError("Could not find the humanoid skeleton in this FBX (head and legs by name)")
        neck = pick("neck")
        if neck is None or neck not in self._ancestors(head):
            neck = self.parent.get(head)
        # hips: where the legs and the spine branch (works for every naming scheme)
        anc = [set(self._ancestors(legs[s][0]) + [legs[s][0]]) for s in legs] + [set(self._ancestors(neck) + [neck])]
        common = set.intersection(*anc)
        hips = max(common, key=self._depth) if common else None
        named = pick("hips")
        if named is not None and named in common:
            hips = named
        if hips is None:
            raise ValueError("Could not find the hips in this FBX")
        hb["hips"] = hips
        spine = [m for m in (self._path(hips, neck) or []) if classify(self.models[m]["name"])[0] != "skip"]
        if spine:
            hb["spine"] = spine[0]
            if len(spine) >= 2:
                hb["chest"] = spine[1] if len(spine) == 2 else spine[len(spine) // 2]
            if len(spine) >= 3:
                hb["upperChest"] = spine[-1]
        hb["neck"], hb["head"] = neck, head
        for side in ("left", "right"):
            up, foot = legs[side]
            if foot is None:
                continue
            chain = [m for m in ([up] + (self._path(up, foot) or [])) if classify(self.models[m]["name"])[0] != "skip"]
            if up not in chain:
                chain = [up] + chain
            hb[side + "UpperLeg"] = chain[0]
            lower = [m for m in chain[1:] if classify(self.models[m]["name"])[0] in ("lowerLeg", "leg", None)]
            if lower:
                hb[side + "LowerLeg"] = lower[0]
            hb[side + "Foot"] = foot
            toes = [c for c in self.kids[foot] if classify(self.models[c]["name"])[0] == "toes"]
            if toes:
                hb[side + "Toes"] = toes[0]
            hand = pick("hand", side)
            if hand is None:
                continue
            top = None
            for x in [hb.get(b) for b in ("upperChest", "chest", "spine", "neck") if hb.get(b)]:
                if x in self._ancestors(hand):
                    top = x
                    break
            arm = [m for m in (self._path(top, hand) or []) if classify(self.models[m]["name"])[0] != "skip"]
            kinds = [classify(self.models[m]["name"])[0] for m in arm]
            if "shoulder" in kinds:
                hb[side + "Shoulder"] = arm[kinds.index("shoulder")]
                arm, kinds = arm[kinds.index("shoulder") + 1:], kinds[kinds.index("shoulder") + 1:]
            elif len(arm) >= 3:
                hb[side + "Shoulder"] = arm[0]
                arm, kinds = arm[1:], kinds[1:]
            if "upperArm" in kinds:
                i = kinds.index("upperArm")
                hb[side + "UpperArm"] = arm[i]
                arm, kinds = arm[i + 1:], kinds[i + 1:]
            elif arm:
                hb[side + "UpperArm"] = arm[0]
                arm, kinds = arm[1:], kinds[1:]
            if "lowerArm" in kinds:
                hb[side + "LowerArm"] = arm[kinds.index("lowerArm")]
            elif arm:
                hb[side + "LowerArm"] = arm[0]
            hb[side + "Hand"] = hand
            for fk in FINGERS:
                segs = sorted([m for (k, s_), lst in cand.items() if k == "finger:" + fk and s_ == side
                               for m, _ in lst if hand in self._ancestors(m)], key=self._depth)
                # (a chain's last bone without children is the tip marker when there are 4)
                if len(segs) >= 4 and not self.kids[segs[-1]]:
                    segs = segs[:-1]
                for nm, m in zip(FINGER_SEGS[fk], segs[:3]):
                    hb[f"{side}{fk}{nm}"] = m
            eyes = [m for m, _ in cand.get(("eye", side), []) if head in self._ancestors(m)]
            if eyes:
                hb[side + "Eye"] = eyes[0]
        for need in ("leftUpperArm", "rightUpperArm", "leftLowerArm", "rightLowerArm", "leftHand", "rightHand",
                     "leftLowerLeg", "rightLowerLeg", "leftFoot", "rightFoot"):
            if need not in hb:
                raise ValueError(f"Could not find the {need} bone in this FBX")
        self.hb = hb
        self.bone_of = {m: b for b, m in hb.items()}

    # ------------------------------------------------------------------ axes, units
    def _frames(self):
        def pos(b):
            return self.world[self.hb[b]][:3, 3]

        def snap(v):
            i = int(np.argmax(np.abs(v)))
            out = np.zeros(3)
            out[i] = np.sign(v[i])
            return out
        up = snap(pos("head") - pos("hips"))
        left = pos("leftUpperArm") - pos("rightUpperArm")
        left = left - up * np.dot(left, up)
        left = snap(left)
        if abs(np.dot(left, up)) > 0.5:
            raise ValueError("Could not tell which way this FBX character faces")
        fwd = np.cross(left, up)
        self.A = np.stack([left, up, fwd])  # file axes -> app axes (x left, y up, z forward)
        usf = float(self.gs.get("UnitScaleFactor", [1.0])[0] or 1.0)
        unit = usf / 100.0
        h = np.linalg.norm(pos("head") - (pos("leftFoot") + pos("rightFoot")) / 2) * unit
        # some exporters get the unit wrong: keep a person between ~0.5 and ~3 m tall
        while h > 4.0:
            unit, h = unit / 10.0, h / 10.0
        while 0 < h < 0.25:
            unit, h = unit * 10.0, h * 10.0
        self.unit = unit

    def _norm_world(self, m):
        W = self.world[m]
        R = W[:3, :3] / np.maximum(np.linalg.norm(W[:3, :3], axis=0, keepdims=True), 1e-12)
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        return self.A @ R, self.A @ W[:3, 3] * self.unit

    # ------------------------------------------------------------------ T-pose
    def _tpose(self):
        order = []

        def visit(m):
            order.append(m)
            for c in self.kids[m]:
                visit(c)
        roots = [m for m in self.models if self.parent.get(m) is None]
        for r in roots:
            visit(r)
        self.order = [m for m in order if self.models[m]["kind"] in ("LimbNode", "Null", "Root", "")]
        W0, P0 = {}, {}
        for m in self.order:
            W0[m], P0[m] = self._norm_world(m)
        self.W0, self.P0 = W0, P0
        child = {}
        hb = self.hb
        for side, sg in (("left", 1.0), ("right", -1.0)):
            mid = next((f"{side}{f}Proximal" for f in ("Middle", "Ring", "Index") if f"{side}{f}Proximal" in hb), None)
            for a, b, tgt in ((f"{side}UpperArm", f"{side}LowerArm", [sg, 0, 0]),
                              (f"{side}LowerArm", f"{side}Hand", [sg, 0, 0]),
                              (f"{side}Hand", mid, [sg, 0, 0]),
                              (f"{side}UpperLeg", f"{side}LowerLeg", [0, -1, 0]),
                              (f"{side}LowerLeg", f"{side}Foot", [0, -1, 0])):
                if a in hb and b in hb:
                    child[hb[a]] = (hb[b], np.array(tgt, float))
            for f in FINGERS[1:]:
                segs = [hb.get(f"{side}{f}{s}") for s in FINGER_SEGS[f]]
                for i, m in enumerate(segs):
                    if m is None:
                        continue
                    nxt = segs[i + 1] if i + 1 < len(segs) else (self.kids[m][0] if self.kids[m] else None)
                    if nxt is not None:
                        child[m] = (nxt, np.array([sg, 0, 0]))
        Wt, Pt = {}, {}
        for m in self.order:
            p = self.parent.get(m)
            if p is None or p not in W0:
                Wt[m], Pt[m] = W0[m], P0[m]
            else:
                lr = W0[p].T @ W0[m]
                lo = W0[p].T @ (P0[m] - P0[p])
                Wt[m] = Wt[p] @ lr
                Pt[m] = Pt[p] + Wt[p] @ lo
            if m in child:
                c, tgt = child[m]
                d = Wt[m] @ (W0[m].T @ (P0[c] - P0[m]))
                # (a model already in a T-pose is left exactly as it is)
                if np.linalg.norm(d) > 1e-9 and np.dot(normalize(d), tgt) < np.cos(np.radians(8)):
                    Wt[m] = rot_between(normalize(d), tgt) @ Wt[m]
        self.Wt, self.Pt = Wt, Pt

    def skeleton(self):
        bones = [b for b in HUMAN_BONES if b in self.hb]
        parent = {}
        for b in bones:
            x = self.parent.get(self.hb[b])
            p = None
            while x is not None:
                if x in self.bone_of and self.bone_of[x] in self.hb:
                    p = self.bone_of[x]
                    break
                x = self.parent.get(x)
            parent[b] = p
        ordered, seen = [], set()

        def visit(b):
            if b in seen:
                return
            if parent[b] is not None:
                visit(parent[b])
            seen.add(b)
            ordered.append(b)
        for b in bones:
            visit(b)
        rest = {b: self.Pt[self.hb[b]].copy() for b in ordered}
        tips = {}
        for b in ordered:
            m = self.hb[b]
            if not any(parent[x] == b for x in ordered) and self.kids[m]:
                tips[b] = self.Pt[self.kids[m][0]].copy()
        sk = VrmSkeleton("fbx", os.path.splitext(os.path.basename(self.path))[0], ordered, parent, rest,
                         {b: self.models[self.hb[b]]["name"] for b in ordered},
                         {b: self.Wt[self.hb[b]] for b in ordered}, {b: rest[b] for b in ordered},
                         np.eye(3), tips, path=self.path)
        sk.fbx = self
        return sk

    # ------------------------------------------------------------------ motion
    def pose(self, bones, Q, root):
        """Solved motion -> the model's own bones.
        Q: (N,B,4) local rotations in the app's normalized frame (bone order `bones`),
        root: (N,3) hips position (app frame, metres).
        Returns (names, quats (N,M,4) local rotations of those FBX nodes as in the file,
        hips local translation (N,3) in file units)."""
        N = len(Q)
        bi = {b: i for i, b in enumerate(bones)}
        R = quat_to_mat(np.asarray(Q, np.float64).reshape(-1, 4)).reshape(N, len(bones), 3, 3)
        G = {}
        for b in bones:  # bones is parent-first
            p = None
            x = self.parent.get(self.hb[b]) if b in self.hb else None
            while x is not None and p is None:
                if x in self.bone_of and self.bone_of[x] in bi:
                    p = self.bone_of[x]
                x = self.parent.get(x)
            G[b] = R[:, bi[b]] if p is None else G[p] @ R[:, bi[b]]
        Wn = {}
        names, out = [], []
        for m in self.order:
            p = self.parent.get(m)
            b = self.bone_of.get(m)
            if b is not None and b in G:
                Wn[m] = G[b] @ self.Wt[m]
            elif p is None or p not in Wn:
                Wn[m] = np.broadcast_to(self.Wt[m], (N, 3, 3))
            else:
                Wn[m] = Wn[p] @ (self.Wt[p].T @ self.Wt[m])
            if b is not None and b in G:
                if p is not None and p in Wn:
                    L = np.einsum("nji,njk->nik", Wn[p], Wn[m])
                else:  # relative to a parent outside the skeleton (or the scene): its file axes
                    Rp = self._norm_world(p)[0] if p is not None else self.A
                    L = np.einsum("ji,njk->nik", Rp, Wn[m])
                names.append(self.models[m]["name"])
                out.append(mat_to_quat(L))
        hips = self.hb["hips"]
        hp = self.parent.get(hips)
        Mp = self.world[hp] if hp is not None else np.eye(4)
        pf = (np.asarray(root, np.float64) / self.unit) @ self.A  # app -> file axes (A orthonormal)
        loc = (np.linalg.inv(Mp) @ np.c_[pf, np.ones(N)].T).T[:, :3]
        return names, np.stack(out, 1).astype(np.float32), loc.astype(np.float32)

    def placement(self):
        """4x4 (row-major) placing the file's scene in the app's world (metres, Y up, facing +Z)."""
        M = np.eye(4)
        M[:3, :3] = self.A * self.unit
        return M

    # ------------------------------------------------------------------ export
    def export(self, path, anim, bones, include_model=True, include_camera=True, in_place=False,
               lock_height=False, rest_hips_y=None):
        """Write this FBX again with the animation (and the video's camera) added."""
        tree = read_fbx(self.path)  # a fresh copy to modify
        objs, conns = find(tree, "Objects"), find(tree, "Connections")
        drop_types = {b"AnimationStack", b"AnimationLayer", b"AnimationCurveNode", b"AnimationCurve"}
        if not include_model:
            drop_types |= {b"Geometry", b"Deformer", b"Pose", b"Material", b"Texture", b"Video",
                           b"Implementation", b"BindingTable", b"LayeredTexture"}
        dropped = set()
        keep = []
        for e in objs.children:
            mid = int(e.props[0][1]) if e.props and e.props[0][0] == "L" else None
            gone = e.name in drop_types
            if not include_model and e.name in (b"Model", b"NodeAttribute") and len(e.props) > 2:
                kind = e.props[2][1]
                if e.name == b"Model" and kind not in (b"LimbNode", b"Null", b"Root", b""):
                    gone = True
            if gone and mid is not None:
                dropped.add(mid)
            else:
                keep.append(e)
        if not include_model:  # node attributes of removed models
            owner = {}
            for c in find_all(conns, "C"):
                if len(c.props) >= 3 and c.props[0][1] == b"OO":
                    owner.setdefault(int(c.props[1][1]), []).append(int(c.props[2][1]))
            keep2 = []
            for e in keep:
                mid = int(e.props[0][1]) if e.props and e.props[0][0] == "L" else None
                if e.name == b"NodeAttribute" and mid in owner and all(o in dropped for o in owner[mid]):
                    dropped.add(mid)
                    continue
                keep2.append(e)
            keep = keep2
        objs.children = keep
        conns.children = [c for c in conns.children
                          if not (len(c.props) >= 3 and (int(c.props[1][1]) in dropped or int(c.props[2][1]) in dropped))]
        if find(tree, "Takes") is not None:
            tree.children.remove(find(tree, "Takes"))
        # new animation
        n = int(anim["frames"])
        fps = float(anim["fps"])
        Q = np.asarray(anim["quats"], np.float64)
        root = np.asarray(anim["root"], np.float64).copy()
        if in_place:
            root[:, [0, 2]] = 0.0
        if lock_height and rest_hips_y is not None:
            root[:, 1] = rest_hips_y
        names, quats, hips_t = self.pose(bones, Q, root)
        used = [int(e.props[0][1]) for e in objs.children if e.props and e.props[0][0] == "L"]
        uid = [max(used + [1000000]) + 1000]

        def new_id():
            uid[0] += 1
            return uid[0]
        counts = {}
        t_stop = int(round((n - 1) / fps * KTIME))
        stack_id, layer_id = new_id(), new_id()
        st = objs.child("AnimationStack", ("L", stack_id), name_class("Take 001", "AnimStack"), "")
        sp = st.child("Properties70")
        for k in ("LocalStart", "ReferenceStart"):
            P(sp, k, "KTime", "Time", "", ("L", 0))
        for k in ("LocalStop", "ReferenceStop"):
            P(sp, k, "KTime", "Time", "", ("L", t_stop))
        objs.children.append(Elem("AnimationLayer", ("L", layer_id), name_class("BaseLayer", "AnimLayer"), ""))
        conns.child("C", "OO", ("L", layer_id), ("L", stack_id))
        counts["AnimationStack"], counts["AnimationLayer"] = 1, 1
        key_times = (np.arange(n, dtype=np.int64) * KTIME / fps).round().astype(np.int64)
        aw = AnimWriter(objs, conns, counts, new_id, layer_id, key_times)
        id_of = {d["name"]: m for m, d in self.models.items()}
        for name, q in zip(names, np.transpose(quats, (1, 0, 2))):
            m = id_of[name]
            p70 = self.models[m]["p70"]
            order = ORDERS[int(p70.get("RotationOrder", [0])[0]) if p70.get("RotationOrder") else 0]
            Rpre = euler_to_mat(_vec(p70, "PreRotation"))
            Rpost = euler_to_mat(_vec(p70, "PostRotation"))
            L = quat_to_mat(q.astype(np.float64))
            Rl = np.einsum("ji,njk,kl->nil", Rpre, L, Rpost)  # Rpre^-1 L Rpost
            aw.curves(m, "Lcl Rotation", mats_to_euler(Rl, order), "R")
            if m == self.hb["hips"]:
                aw.curves(m, "Lcl Translation", hips_t.astype(np.float64), "T")
        if include_camera and "cam_pos" in anim:
            cp = np.asarray(anim["cam_pos"], np.float64).copy()
            if in_place:
                cp[:, [0, 2]] -= np.asarray(anim["root"])[:, [0, 2]]
            if lock_height and rest_hips_y is not None:
                cp[:, 1] -= np.asarray(anim["root"])[:, 1] - rest_hips_y
            cq = quat_to_mat(np.asarray(anim["cam_quat"], np.float64))
            to_fbx_cam = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])  # FBX cameras look down +X
            Rc = np.einsum("ji,njk->nik", self.A, cq) @ to_fbx_cam
            aw.camera({"pos": (cp / self.unit) @ self.A, "rot_deg": mats_to_euler(Rc),
                       "vfov": np.asarray(anim["cam_vfov"], np.float64), "aspect": anim["width"] / anim["height"]})
        if include_model and anim.get("expr_names") and "expr_w" in anim:
            # hand-set blend shapes: the file's own blend shape channels, animated
            ew = np.asarray(anim["expr_w"], np.float64)
            for e in objs.children:
                if e.name == b"Deformer" and len(e.props) > 2 and e.props[2][1] == b"BlendShapeChannel":
                    nm = object_name(e)
                    if nm in anim["expr_names"]:
                        aw.scalar(int(e.props[0][1]), "DeformPercent", ew[:, anim["expr_names"].index(nm)] * 100.0)
        # time settings, definitions, takes
        gs = find(tree, "GlobalSettings")
        if gs is not None:
            p7 = find(gs, "Properties70")
            modes = {120.0: 1, 100.0: 2, 60.0: 3, 50.0: 4, 48.0: 5, 30.0: 6, 25.0: 10, 24.0: 11}
            tmode = next((md for r, md in modes.items() if abs(fps - r) < 1e-3), 14)
            setv = {b"TimeMode": ("enum", "", "", tmode), b"TimeSpanStart": ("KTime", "Time", "", ("L", 0)),
                    b"TimeSpanStop": ("KTime", "Time", "", ("L", t_stop)),
                    b"CustomFrameRate": ("double", "Number", "", float(fps))}
            p7.children = [c for c in p7.children if not (c.props and c.props[0][1] in setv)]
            for k, v in setv.items():
                P(p7, k.decode(), *v)
        defs = find(tree, "Definitions")
        if defs is not None:
            from collections import Counter
            have = Counter(e.name.decode() for e in objs.children)
            ots = {e.props[0][1].decode() if isinstance(e.props[0][1], bytes) else e.props[0][1]: e
                   for e in find_all(defs, "ObjectType") if e.props}
            for t, c in have.items():
                if t not in ots:
                    ots[t] = defs.child("ObjectType", t)
                    ots[t].child("Count", c)
            for t, e in list(ots.items()):
                cnt = find(e, "Count")
                if t == "GlobalSettings":
                    continue
                if t not in have:
                    defs.children.remove(e)
                    continue
                if cnt is not None:
                    cnt.props = [("I", int(have[t]))]
            tot = find(defs, "Count")
            if tot is not None:
                tot.props = [("I", int(1 + sum(have.values())))]
        hx = find(tree, "FBXHeaderExtension")
        if hx is not None and find(hx, "FBXVersion") is not None:
            find(hx, "FBXVersion").props = [("I", 7400)]
        takes = tree.child("Takes")
        takes.child("Current", "Take 001")
        tk = takes.child("Take", "Take 001")
        tk.child("FileName", "Take_001.tak")
        tk.child("LocalTime", ("L", 0), ("L", t_stop))
        tk.child("ReferenceTime", ("L", 0), ("L", t_stop))
        encode_file(tree, path, 7400)
        return path


def load_fbx_skeleton(path):
    return FbxAvatar(path).skeleton()

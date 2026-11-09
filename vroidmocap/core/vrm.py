"""Read the humanoid rest skeleton out of a .vrm (glTF binary) file.

Both VRM 1.0 (VRMC_vrm) and VRM 0.x (VRM) are supported. All positions are
returned in the *normalized* frame used throughout the app: Y up, metres,
character facing +Z, rest pose = the model's T-pose. (VRM 0.x files face -Z, so
they are turned 180 degrees about Y.)
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field

import numpy as np

from .mathutil import quat_to_mat

# VRM 1.0 humanoid bones, parents listed according to the spec hierarchy.
HUMAN_BONES = [
    "hips", "spine", "chest", "upperChest", "neck", "head", "leftEye", "rightEye", "jaw",
    "leftUpperLeg", "leftLowerLeg", "leftFoot", "leftToes",
    "rightUpperLeg", "rightLowerLeg", "rightFoot", "rightToes",
    "leftShoulder", "leftUpperArm", "leftLowerArm", "leftHand",
    "rightShoulder", "rightUpperArm", "rightLowerArm", "rightHand",
]
FINGERS = ["Thumb", "Index", "Middle", "Ring", "Little"]
FINGER_SEGS = {"Thumb": ["Metacarpal", "Proximal", "Distal"]}
for _f in FINGERS[1:]:
    FINGER_SEGS[_f] = ["Proximal", "Intermediate", "Distal"]
for _side in ("left", "right"):
    for _f in FINGERS:
        for _s in FINGER_SEGS[_f]:
            HUMAN_BONES.append(f"{_side}{_f}{_s}")

VRM0_RENAME = {}
for _side in ("left", "right"):
    VRM0_RENAME[f"{_side}ThumbProximal"] = f"{_side}ThumbMetacarpal"
    VRM0_RENAME[f"{_side}ThumbIntermediate"] = f"{_side}ThumbProximal"


def _trs_matrix(node):
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    m = np.eye(4)
    t = node.get("translation", [0, 0, 0])
    r = node.get("rotation", [0, 0, 0, 1])
    s = node.get("scale", [1, 1, 1])
    m[:3, :3] = quat_to_mat(np.array(r)) @ np.diag(s)
    m[:3, 3] = t
    return m


@dataclass
class VrmSkeleton:
    version: str                      # "1" or "0"
    title: str
    bones: list                       # humanoid bone names present, parent-before-child order
    parent: dict                      # bone -> parent humanoid bone (None for hips)
    rest_pos: dict                    # bone -> normalized-frame world position (3,)
    node_name: dict                   # bone -> glTF node name
    raw_world_rot: dict               # bone -> 3x3 rest world rotation in the *file* frame
    raw_world_pos: dict               # bone -> rest world position in the *file* frame
    to_file: np.ndarray               # 3x3: normalized frame -> file frame
    tips: dict = field(default_factory=dict)  # bone -> position of a child "end" node, if any

    def has(self, b):
        return b in self.rest_pos

    def children(self, b):
        return [c for c in self.bones if self.parent.get(c) == b]

    def bone_dir(self, b, child=None):
        """Rest direction from bone b towards child (or its tip)."""
        p = self.rest_pos[b]
        if child is not None and child in self.rest_pos:
            q = self.rest_pos[child]
        elif b in self.tips:
            q = self.tips[b]
        else:
            return None
        d = q - p
        n = np.linalg.norm(d)
        return d / n if n > 1e-8 else None

    def to_json(self):
        return {
            "version": self.version,
            "title": self.title,
            "bones": self.bones,
            "parent": self.parent,
            "rest_pos": {k: v.tolist() for k, v in self.rest_pos.items()},
        }


def read_vrm_json(path):
    with open(path, "rb") as f:
        head = f.read(12)
        magic, _ver, _length = struct.unpack("<4sII", head)
        if magic != b"glTF":
            raise ValueError("Not a VRM/GLB file (missing glTF header)")
        clen, ctype = struct.unpack("<I4s", f.read(8))
        if ctype != b"JSON":
            raise ValueError("Malformed GLB: first chunk is not JSON")
        return json.loads(f.read(clen).decode("utf-8"))


def load_vrm_skeleton(path) -> VrmSkeleton:
    gl = read_vrm_json(path)
    ext = gl.get("extensions", {})
    nodes = gl["nodes"]
    if "VRMC_vrm" in ext:
        version = "1"
        vrm = ext["VRMC_vrm"]
        hb = {k: v["node"] for k, v in vrm["humanoid"]["humanBones"].items()}
        title = vrm.get("meta", {}).get("name", "")
    elif "VRM" in ext:
        version = "0"
        vrm = ext["VRM"]
        hb = {}
        for b in vrm["humanoid"]["humanBones"]:
            name = VRM0_RENAME.get(b["bone"], b["bone"])
            hb[name] = b["node"]
        title = vrm.get("meta", {}).get("title", "")
    else:
        raise ValueError("This file has no VRM humanoid extension")

    # world matrices of every node
    parent_of = {}
    for i, n in enumerate(nodes):
        for c in n.get("children", []):
            parent_of[c] = i
    world = {}

    def wm(i):
        if i in world:
            return world[i]
        m = _trs_matrix(nodes[i])
        if i in parent_of:
            m = wm(parent_of[i]) @ m
        world[i] = m
        return m

    for i in range(len(nodes)):
        wm(i)

    to_file = np.eye(3) if version == "1" else np.diag([-1.0, 1.0, -1.0])
    to_norm = to_file.T

    node_to_bone = {v: k for k, v in hb.items()}
    bones_present = [b for b in HUMAN_BONES if b in hb]
    parent = {}
    for b in bones_present:
        i = parent_of.get(hb[b])
        p = None
        while i is not None:
            if i in node_to_bone and node_to_bone[i] in hb:
                p = node_to_bone[i]
                break
            i = parent_of.get(i)
        parent[b] = p

    # parent-before-child ordering
    ordered = []
    seen = set()

    def visit(b):
        if b in seen:
            return
        if parent[b] is not None:
            visit(parent[b])
        seen.add(b)
        ordered.append(b)

    for b in bones_present:
        visit(b)

    rest_pos, raw_rot, raw_pos, node_name, tips = {}, {}, {}, {}, {}
    for b in ordered:
        m = world[hb[b]]
        raw_pos[b] = m[:3, 3].copy()
        rs = m[:3, :3]
        # strip scale
        raw_rot[b] = rs / np.maximum(np.linalg.norm(rs, axis=0, keepdims=True), 1e-9)
        rest_pos[b] = to_norm @ m[:3, 3]
        node_name[b] = nodes[hb[b]].get("name", b)
        kids = [c for c in nodes[hb[b]].get("children", []) if c not in node_to_bone]
        # a non-humanoid child that continues the chain (e.g. finger tip "_end" node)
        if kids and not any(parent[x] == b for x in ordered if x != b):
            tips[b] = to_norm @ world[kids[0]][:3, 3]
    return VrmSkeleton(version, title, ordered, parent, rest_pos, node_name, raw_rot,
                       raw_pos, to_file, tips)

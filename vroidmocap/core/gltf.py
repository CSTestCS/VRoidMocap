"""Minimal glTF 2.0 binary (.glb / .vrm) reader: nodes, meshes, skins, materials, images."""
from __future__ import annotations

import json
import struct

import numpy as np

from .mathutil import quat_to_mat

COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


class Glb:
    def __init__(self, path):
        with open(path, "rb") as f:
            data = f.read()
        magic, _, length = struct.unpack_from("<4sII", data, 0)
        if magic != b"glTF":
            raise ValueError("not a GLB/VRM file")
        off = 12
        self.json, self.bin = None, b""
        while off < length:
            clen, ctype = struct.unpack_from("<I4s", data, off)
            chunk = data[off + 8: off + 8 + clen]
            if ctype == b"JSON":
                self.json = json.loads(chunk.decode("utf-8"))
            elif ctype == b"BIN\x00":
                self.bin = chunk
            off += 8 + clen
        g = self.json
        self.nodes = g.get("nodes", [])
        self.meshes = g.get("meshes", [])
        self.skins = g.get("skins", [])
        self.materials = g.get("materials", [])
        self.textures = g.get("textures", [])
        self.images = g.get("images", [])
        self.parent = {}
        for i, n in enumerate(self.nodes):
            for c in n.get("children", []):
                self.parent[c] = i

    def view_bytes(self, vi):
        bv = self.json["bufferViews"][vi]
        o = bv.get("byteOffset", 0)
        return self.bin[o:o + bv["byteLength"]], bv.get("byteStride")

    def accessor(self, ai):
        acc = self.json["accessors"][ai]
        dt = np.dtype(COMP[acc["componentType"]])
        nc = NCOMP[acc["type"]]
        cnt = acc["count"]
        if "bufferView" in acc:
            raw, stride = self.view_bytes(acc["bufferView"])
            off = acc.get("byteOffset", 0)
            item = dt.itemsize * nc
            if stride and stride != item:
                arr = np.empty((cnt, nc), dt)
                for i in range(cnt):
                    arr[i] = np.frombuffer(raw, dt, nc, off + i * stride)
            else:
                arr = np.frombuffer(raw, dt, cnt * nc, off).reshape(cnt, nc)
        else:
            arr = np.zeros((cnt, nc), dt)
        arr = arr.astype(np.float64) if acc.get("normalized") is not True else arr
        if acc.get("normalized"):
            mx = float(np.iinfo(dt).max)
            arr = arr.astype(np.float64) / mx
        if "sparse" in acc:
            sp = acc["sparse"]
            raw_i, _ = self.view_bytes(sp["indices"]["bufferView"])
            idx = np.frombuffer(raw_i, COMP[sp["indices"]["componentType"]], sp["count"],
                                sp["indices"].get("byteOffset", 0))
            raw_v, _ = self.view_bytes(sp["values"]["bufferView"])
            vals = np.frombuffer(raw_v, dt, sp["count"] * nc, sp["values"].get("byteOffset", 0)).reshape(-1, nc)
            arr = np.array(arr, np.float64)
            arr[idx] = vals
        return arr

    def local_matrix(self, i):
        n = self.nodes[i]
        if "matrix" in n:
            return np.array(n["matrix"], float).reshape(4, 4).T
        m = np.eye(4)
        m[:3, :3] = quat_to_mat(np.array(n.get("rotation", [0, 0, 0, 1.0]))) @ np.diag(n.get("scale", [1, 1, 1.0]))
        m[:3, 3] = n.get("translation", [0, 0, 0])
        return m

    def world_matrix(self, i):
        m = self.local_matrix(i)
        p = self.parent.get(i)
        while p is not None:
            m = self.local_matrix(p) @ m
            p = self.parent.get(p)
        return m

    def image_bytes(self, ii):
        im = self.images[ii]
        if "bufferView" in im:
            raw, _ = self.view_bytes(im["bufferView"])
            return bytes(raw), im.get("mimeType", "image/png")
        return None, None

    def scene_roots(self):
        sc = self.json.get("scenes", [{}])[self.json.get("scene", 0)] if self.json.get("scenes") else {}
        return sc.get("nodes", [i for i in range(len(self.nodes)) if i not in self.parent])

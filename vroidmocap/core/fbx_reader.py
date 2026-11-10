"""Read FBX files (binary 7.x and ASCII) into a node tree.

The tree uses the writer's `Elem` class (name, typed props, children), so an
imported file can be modified and written out again (fbx_writer.encode_file).
"""
from __future__ import annotations

import re
import struct
import zlib

import numpy as np

from .fbx_writer import Elem

BIN_MAGIC = b"Kaydara FBX Binary  \x00"


def read_fbx(path):
    with open(path, "rb") as f:
        data = f.read()
    if data.startswith(BIN_MAGIC):
        return _read_binary(data)
    return _read_ascii(data.decode("utf-8", errors="replace"))


# ----------------------------------------------------------------------------- binary
_ARR = {"f": np.float32, "d": np.float64, "l": np.int64, "i": np.int32, "b": np.bool_}


def _read_binary(data):
    version = struct.unpack_from("<I", data, 23)[0]
    wide = version >= 7500
    root = Elem(b"root")
    root.version = version
    pos = 27
    while True:
        node, pos = _read_node(data, pos, wide)
        if node is None:
            break
        root.children.append(node)
    return root


def _read_node(data, pos, wide):
    if wide:
        end, nprops, plen = struct.unpack_from("<QQQ", data, pos)
        pos += 24
    else:
        end, nprops, plen = struct.unpack_from("<III", data, pos)
        pos += 12
    nlen = data[pos]
    pos += 1
    if end == 0:
        return None, pos
    name = data[pos:pos + nlen]
    pos += nlen
    e = Elem(name)
    for _ in range(nprops):
        code = chr(data[pos])
        pos += 1
        if code == "Y":
            v = struct.unpack_from("<h", data, pos)[0]
            pos += 2
        elif code == "C":
            v = bool(data[pos])
            pos += 1
        elif code == "I":
            v = struct.unpack_from("<i", data, pos)[0]
            pos += 4
        elif code == "F":
            v = struct.unpack_from("<f", data, pos)[0]
            pos += 4
        elif code == "D":
            v = struct.unpack_from("<d", data, pos)[0]
            pos += 8
        elif code == "L":
            v = struct.unpack_from("<q", data, pos)[0]
            pos += 8
        elif code in "SR":
            n = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            v = bytes(data[pos:pos + n])
            pos += n
        elif code in _ARR:
            n, enc, clen = struct.unpack_from("<III", data, pos)
            pos += 12
            raw = data[pos:pos + clen]
            pos += clen
            if enc == 1:
                raw = zlib.decompress(raw)
            v = np.frombuffer(raw, dtype=_ARR[code], count=n).copy()
        else:
            raise ValueError(f"FBX: unknown property type {code!r}")
        e.props.append((code, v))
    while pos < end:
        child, pos = _read_node(data, pos, wide)
        if child is None:
            break
        e.children.append(child)
    return e, end


# ----------------------------------------------------------------------------- ASCII
_TOK = re.compile(r'\s*(?:;[^\n]*\n)*\s*("(?:[^"\\]|\\.)*"|[{}:,]|\*\d+|[^\s{}:,"]+)', re.S)


def _read_ascii(text):
    toks = [m.group(1) for m in _TOK.finditer(text)]
    root = Elem(b"root")
    root.version = 7400
    i = 0
    stack = [root]
    while i < len(toks):
        t = toks[i]
        if t == "}":
            stack.pop()
            i += 1
            continue
        if i + 1 < len(toks) and toks[i + 1] == ":":
            e = Elem(t.encode())
            stack[-1].children.append(e)
            i += 2
            vals = []
            while i < len(toks) and toks[i] not in ("{", "}") and not (i + 1 < len(toks) and toks[i + 1] == ":"):
                if toks[i] != ",":
                    vals.append(toks[i])
                i += 1
            if vals and vals[0].startswith("*"):  # array: *N { a: v,v,v }
                vals = vals[1:]
            for v in vals:
                e.props.append(_ascii_value(v))
            if i < len(toks) and toks[i] == "{":
                # an array block holds "a: ..." as its only child
                i += 1
                stack.append(e)
            continue
        i += 1
    _fold_ascii_arrays(root)
    return root


def _ascii_value(v):
    if v.startswith('"'):
        s = v[1:-1]
        # "Name::Class" in ASCII == b"Name\x00\x01Class" in binary
        if "::" in s:
            cls, _, nm = s.partition("::")
            return ("S", nm.encode() + b"\x00\x01" + cls.encode())
        return ("S", s.encode())
    try:
        if re.fullmatch(r"[-+]?\d+", v):
            n = int(v)
            return ("L", n) if abs(n) > 2 ** 31 - 1 else ("I", n)
        return ("D", float(v))
    except ValueError:
        return ("S", v.encode())


def _fold_ascii_arrays(e):
    for c in e.children:
        _fold_ascii_arrays(c)
    # Vertices: *N { a: ... }  ->  one array property
    if len(e.children) == 1 and e.children[0].name == b"a" and not e.props:
        vals = [v for _, v in e.children[0].props]
        ints = all(isinstance(v, int) for v in vals)
        arr = np.array(vals, np.int64 if ints else np.float64)
        if ints and (np.abs(arr) < 2 ** 31).all() and e.name not in (b"KeyTime",):
            arr = arr.astype(np.int32)
        code = {np.dtype(np.float64): "d", np.dtype(np.int64): "l", np.dtype(np.int32): "i"}[arr.dtype]
        e.props = [(code, arr)]
        e.children = []


# ----------------------------------------------------------------------------- helpers
def find(e, name):
    name = name.encode() if isinstance(name, str) else name
    for c in e.children:
        if c.name == name:
            return c
    return None


def find_all(e, name):
    name = name.encode() if isinstance(name, str) else name
    return [c for c in e.children if c.name == name]


def props70(e):
    """{property name: list of values} of an object's Properties70 block."""
    p = find(e, "Properties70")
    out = {}
    if p is None:
        return out
    for c in p.children:
        if c.name == b"P" and c.props:
            out[_s(c.props[0][1])] = [v for _, v in c.props[4:]]
    return out


def _s(v):
    return v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)


def object_name(e):
    """Model name from the binary 'Name\\x00\\x01Class' (or plain) string."""
    if len(e.props) < 2:
        return ""
    raw = e.props[1][1]
    if isinstance(raw, bytes):
        return raw.split(b"\x00\x01")[0].decode("utf-8", errors="replace")
    return str(raw)

"""Core pipeline tests that need no video or model files."""
import os
import struct
import sys
import tempfile
import unittest
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vroidmocap.core import camera as cam  # noqa: E402
from vroidmocap.core.mathutil import (axis_angle_to_mat, mat_to_quat, normalize, quat_to_mat,  # noqa: E402
                                      rot_from_axes)
from vroidmocap.core.retarget import RigSolver, avatar_fk, snap_chain  # noqa: E402
from vroidmocap.core.vrm import FINGER_SEGS, FINGERS, VrmSkeleton  # noqa: E402


def synthetic_skeleton():
    """A VRoid-like T-pose skeleton (normalized frame, facing +Z)."""
    P = {
        "hips": (0, 1.0, 0), "spine": (0, 1.06, 0.01), "chest": (0, 1.18, 0.01), "upperChest": (0, 1.30, 0),
        "neck": (0, 1.45, -0.03), "head": (0, 1.54, -0.03),
        "leftUpperLeg": (0.09, 0.96, 0), "leftLowerLeg": (0.09, 0.55, 0.01), "leftFoot": (0.09, 0.11, -0.03),
        "leftToes": (0.09, 0.04, 0.09),
        "leftShoulder": (0.03, 1.42, -0.03), "leftUpperArm": (0.11, 1.41, -0.03), "leftLowerArm": (0.36, 1.41, -0.03),
        "leftHand": (0.61, 1.41, -0.03),
    }
    fx = {"Thumb": (0.0, -0.01, 0.025), "Index": (0.08, 0.0, 0.025), "Middle": (0.08, 0.0, 0.0),
          "Ring": (0.075, 0.0, -0.02), "Little": (0.07, -0.005, -0.038)}
    for fg in FINGERS:
        base = np.array(P["leftHand"]) + fx[fg]
        step = np.array([0.03, -0.004, 0.012]) if fg == "Thumb" else np.array([0.028, 0, 0])
        for k, seg in enumerate(FINGER_SEGS[fg]):
            P[f"left{fg}{seg}"] = tuple(base + step * k)
    for k in list(P):
        if k.startswith("left"):
            v = P[k]
            P["right" + k[4:]] = (-v[0], v[1], v[2])
    parent = {"hips": None, "spine": "hips", "chest": "spine", "upperChest": "chest", "neck": "upperChest",
              "head": "neck"}
    for s in ("left", "right"):
        parent.update({f"{s}UpperLeg": "hips", f"{s}LowerLeg": f"{s}UpperLeg", f"{s}Foot": f"{s}LowerLeg",
                       f"{s}Toes": f"{s}Foot", f"{s}Shoulder": "upperChest", f"{s}UpperArm": f"{s}Shoulder",
                       f"{s}LowerArm": f"{s}UpperArm", f"{s}Hand": f"{s}LowerArm"})
        for fg in FINGERS:
            prev = f"{s}Hand"
            for seg in FINGER_SEGS[fg]:
                parent[f"{s}{fg}{seg}"] = prev
                prev = f"{s}{fg}{seg}"
    order = []

    def visit(b):
        if b in order:
            return
        if parent[b]:
            visit(parent[b])
        order.append(b)
    for b in P:
        visit(b)
    rest = {k: np.array(v, float) for k, v in P.items()}
    eye = {k: np.eye(3) for k in P}
    return VrmSkeleton("1", "test", order, parent, rest, {k: k for k in P}, eye, dict(rest), np.eye(3))


def landmarks_from_pose(skel, pos, rot):
    """Generate MediaPipe-like joints (body dict + hand arrays) from avatar FK."""
    bi = {b: i for i, b in enumerate(skel.bones)}
    g = lambda b: pos[bi[b]]  # noqa: E731
    R = lambda b: rot[bi[b]]  # noqa: E731
    J = {}
    for s, S in (("left", "L"), ("right", "R")):
        J["hip" + S] = g(f"{s}UpperLeg")
        J["kn" + S] = g(f"{s}LowerLeg")
        J["an" + S] = g(f"{s}Foot")
        fr = skel.rest_pos[f"{s}Foot"]
        J["heel" + S] = g(f"{s}Foot") + R(f"{s}Foot") @ np.array([0, -fr[1], -0.05])
        J["toe" + S] = g(f"{s}Foot") + R(f"{s}Foot") @ np.array([0, -fr[1], 0.15])
        J["sh" + S] = g(f"{s}UpperArm")
        J["el" + S] = g(f"{s}LowerArm")
        J["wr" + S] = g(f"{s}Hand")
        J["index" + S] = g(f"{s}IndexProximal")
        J["pinky" + S] = g(f"{s}LittleProximal")
        J["thumb" + S] = g(f"{s}ThumbProximal")
    hd, Rh = g("head"), R("head")
    for S, sx in (("L", 1), ("R", -1)):
        J["ear" + S] = hd + Rh @ np.array([sx * 0.07, 0.06, -0.01])
        J["eye" + S] = hd + Rh @ np.array([sx * 0.03, 0.08, 0.08])
        J["mouth" + S] = hd + Rh @ np.array([sx * 0.02, 0.02, 0.08])
    J["nose"] = hd + Rh @ np.array([0, 0.05, 0.11])
    hands = []
    for s in ("left", "right"):
        pts = [g(f"{s}Hand")]
        for fg in FINGERS:
            segs = [f"{s}{fg}{seg}" for seg in FINGER_SEGS[fg]]
            for b in segs:
                pts.append(g(b))
            rest_d = skel.rest_pos[segs[-1]] - skel.rest_pos[segs[-2]]
            pts.append(g(segs[-1]) + R(segs[-1]) @ rest_d * 0.8)
        hands.append(np.array(pts))
    return J, hands


def random_pose(skel, rng, amount=0.6):
    """Random but anatomical pose: ball joints at shoulders/hips, hinge elbows,
    knees and finger joints, forearm twist, 2-DOF wrists and knuckles."""
    Q = np.zeros((1, len(skel.bones), 4))
    Q[..., 3] = 1
    U = rng.uniform

    def setq(b, m):
        Q[0, skel.bones.index(b)] = mat_to_quat(m)

    def aa(ax, deg):
        return axis_angle_to_mat(np.array(ax, float), np.radians(deg))
    for s, sx in (("left", 1.0), ("right", -1.0)):
        setq(f"{s}UpperArm", axis_angle_to_mat(normalize(rng.normal(size=3)), U(0.2, 1.4)))
        flex = aa([0, -sx, 0], U(25, 135))           # elbow hinge: forearm swings forward (+Z)
        twist = aa([sx, 0, 0], U(-70, 70))          # pronation / supination
        setq(f"{s}LowerArm", flex @ twist)
        setq(f"{s}Hand", aa([0, 0, -sx], U(-50, 60)) @ aa([0, 1, 0], U(-20, 20)))
        setq(f"{s}UpperLeg", axis_angle_to_mat(normalize(rng.normal(size=3)), U(0.1, 1.0)))
        setq(f"{s}LowerLeg", aa([1, 0, 0], U(20, 120)))  # knee hinge: shin swings back
        setq(f"{s}Foot", aa([1, 0, 0], U(-30, 30)))  # ankle flexion (foot roll is not observable)
        for fg in FINGERS:
            for k, seg in enumerate(FINGER_SEGS[fg]):
                b = f"{s}{fg}{seg}"
                if fg == "Thumb":
                    setq(b, axis_angle_to_mat(normalize(rng.normal(size=3)), U(0.05, 0.5)))
                elif k == 0:
                    setq(b, aa([0, 0, -sx], U(0, 80)) @ aa([0, 1, 0], U(-12, 12)))
                else:
                    setq(b, aa([0, 0, -sx], U(5, 85)))
    return Q


class RigRoundTrip(unittest.TestCase):
    def test_solver_reproduces_joint_positions(self):
        skel = synthetic_skeleton()
        solver = RigSolver(skel)
        rng = np.random.default_rng(1)
        errs, rerrs = [], []
        for trial in range(30):
            Q = random_pose(skel, rng)
            root = np.array([[0.3, 1.0, -0.2]])
            pos, rot = avatar_fk(skel, Q, root)
            J, hands = landmarks_from_pose(skel, pos[0], rot[0])
            solver.reset_continuity()
            G = solver.solve(J, hands)
            L, _ = solver.locals_from_globals(G)
            Q2 = np.array([[mat_to_quat(L[b]) for b in skel.bones]])
            pos2, rot2 = avatar_fk(skel, Q2, root)
            # (thumb twist is not observable from its joint positions: positions are checked below)
            limb = [i for i, b in enumerate(skel.bones) if b.endswith(("UpperArm", "LowerArm", "Hand", "UpperLeg",
                    "LowerLeg", "Foot", "Proximal", "Intermediate", "Distal")) and "Thumb" not in b]
            rel = np.einsum("nji,njk->nik", rot[0, limb], rot2[0, limb])
            rerrs.append(np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1))))
            # each limb relative to its own root (the spine split is unobservable)
            for root_b, prefix in (("leftUpperArm", "left"), ("rightUpperArm", "right"),
                                   ("leftUpperLeg", "left"), ("rightUpperLeg", "right")):
                arm = "Arm" in root_b
                chain = [b for b in skel.bones if b.startswith(prefix) and
                         (("Arm" in b or "Hand" in b or any(f in b for f in FINGERS)) if arm else ("Leg" in b or "Foot" in b or "Toes" in b))]
                r = skel.bones.index(root_b)
                idx = [skel.bones.index(b) for b in chain]
                errs.append(np.linalg.norm((pos2[0, idx] - pos2[0, r]) - (pos[0, idx] - pos[0, r]), axis=1))
        e = np.concatenate(errs)
        # limb joints (incl. every finger joint) must land within a few millimetres
        self.assertLess(np.median(e), 0.002, f"median {np.median(e)}")
        self.assertLess(np.percentile(e, 95), 0.01, f"p95 {np.percentile(e, 95)}")
        rot_err = np.concatenate(rerrs)
        self.assertLess(np.median(rot_err), 1.0, f"rotation median {np.median(rot_err)}")
        self.assertLess(np.percentile(rot_err, 95), 3.0, f"rotation p95 {np.percentile(rot_err, 95)}")

    def test_tpose_gives_identity(self):
        skel = synthetic_skeleton()
        Q = np.zeros((1, len(skel.bones), 4))
        Q[..., 3] = 1
        pos, rot = avatar_fk(skel, Q, np.array([[0, 1.0, 0]]))
        J, hands = landmarks_from_pose(skel, pos[0], rot[0])
        G = RigSolver(skel).solve(J, hands)
        for b in ("hips", "leftUpperArm", "rightLowerArm", "leftHand", "rightIndexProximal", "leftUpperLeg", "rightFoot"):
            ang = np.degrees(np.arccos(np.clip((np.trace(G[b]) - 1) / 2, -1, 1)))
            self.assertLess(ang, 1.0, b)


class Geometry(unittest.TestCase):
    def test_translation_solve(self):
        rng = np.random.default_rng(0)
        W = rng.normal(scale=0.3, size=(33, 3))
        T = np.array([0.4, -0.2, 4.0])
        f, cx, cy = 1200.0, 960, 540
        uv = cam.project(W + T, f, cx, cy)
        T2 = cam.solve_translation(W, uv, np.ones(33), f, cx, cy)
        self.assertTrue(np.allclose(T, T2, atol=1e-6))

    def test_snap_chain_keeps_lengths_and_rays(self):
        P = np.array([[0, 0, 3.0], [0.3, 0.1, 3.1], [0.5, 0.4, 3.0]])
        true = P + np.array([[0, 0, 0], [0.02, -0.01, 0.05], [0.0, 0.03, -0.04]])
        rays = normalize(true)
        L = {(0, 1): float(np.linalg.norm(P[1] - P[0])), (1, 2): float(np.linalg.norm(P[2] - P[1]))}
        Q = snap_chain(P, rays, np.ones(3), [(0, 1), (1, 2)], L)
        self.assertAlmostEqual(np.linalg.norm(Q[1] - Q[0]), L[(0, 1)], places=6)
        self.assertAlmostEqual(np.linalg.norm(Q[2] - Q[1]), L[(1, 2)], places=6)

    def test_align_from_up(self):
        R = cam.align_from_up(np.array([0, -1.0, 0]))
        self.assertTrue(np.allclose(R, np.diag([1, -1, -1])))
        tilt = axis_angle_to_mat(np.array([1.0, 0, 0]), np.radians(20))
        up = tilt @ np.array([0, -1.0, 0])
        R = cam.align_from_up(up)
        self.assertTrue(np.allclose(R @ up, [0, 1, 0]))
        yaw, pitch, roll = cam.camera_euler(R)
        self.assertAlmostEqual(abs(pitch), 20, places=4)

    def test_quat_roundtrip(self):
        rng = np.random.default_rng(3)
        for _ in range(50):
            q = normalize(rng.normal(size=4))
            m = quat_to_mat(q)
            q2 = mat_to_quat(m)
            self.assertTrue(np.allclose(quat_to_mat(q2), m, atol=1e-9))
        R = rot_from_axes([1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0])
        self.assertTrue(np.allclose(R @ [1, 0, 0], [0, 0, 1]))


def read_fbx_tree(path):
    """Tiny binary FBX reader used to validate the writer."""
    data = open(path, "rb").read()
    assert data[:21] == b"Kaydara FBX Binary  \x00"
    pos = 27

    def read_node(pos):
        end, nprops, plen, nlen = struct.unpack_from("<IIIB", data, pos)
        if end == 0:
            return None, pos + 13
        name = data[pos + 13:pos + 13 + nlen].decode()
        p = pos + 13 + nlen
        props = []
        for _ in range(nprops):
            t = chr(data[p])
            p += 1
            if t in "YCIFDL":
                fmt = {"Y": "<h", "C": "<?", "I": "<i", "F": "<f", "D": "<d", "L": "<q"}[t]
                props.append(struct.unpack_from(fmt, data, p)[0])
                p += struct.calcsize(fmt)
            elif t in "SR":
                n = struct.unpack_from("<I", data, p)[0]
                props.append(data[p + 4:p + 4 + n])
                p += 4 + n
            else:
                n, enc, clen = struct.unpack_from("<III", data, p)
                raw = data[p + 12:p + 12 + clen]
                raw = zlib.decompress(raw) if enc else raw
                props.append(np.frombuffer(raw, {"f": np.float32, "d": np.float64, "i": np.int32, "l": np.int64}[t]))
                p += 12 + clen
        kids = []
        while p < end:
            k, p = read_node(p)
            if k is None:
                break
            kids.append(k)
        return (name, props, kids), end

    nodes = []
    while True:
        n, pos = read_node(pos)
        if n is None:
            break
        nodes.append(n)
    return nodes


def write_skeleton_glb(skel, path):
    """Tiny GLB containing just the skeleton's nodes (for exporter tests)."""
    import json as _json
    nodes = []
    for b in skel.bones:
        p = skel.parent[b]
        t = skel.rest_pos[b] - (skel.rest_pos[p] if p else 0)
        nodes.append({"name": b, "translation": [float(x) for x in t], "children": []})
    for i, b in enumerate(skel.bones):
        p = skel.parent[b]
        if p:
            nodes[skel.bones.index(p)]["children"].append(i)
    for nd in nodes:
        if not nd["children"]:
            del nd["children"]
    js = _json.dumps({"asset": {"version": "2.0"}, "nodes": nodes, "scenes": [{"nodes": [0]}], "scene": 0}).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(js)) + struct.pack("<I4s", len(js), b"JSON") + js)


class FbxExport(unittest.TestCase):
    def test_write_and_parse(self):
        from vroidmocap.core.fbx_writer import export_animation_fbx
        skel = synthetic_skeleton()
        rng = np.random.default_rng(5)
        N = 20
        Q = np.concatenate([random_pose(skel, rng) for _ in range(N)])
        anim = {"bones": skel.bones, "quats": Q.astype(np.float32), "root": np.tile([0, 1.0, 0], (N, 1)),
                "fps": 30.0, "frames": N, "cam_pos": np.tile([0, 1.4, 3.0], (N, 1)),
                "cam_quat": np.tile([0, 0, 0, 1.0], (N, 1)), "cam_vfov": np.full(N, 40.0), "width": 1920, "height": 1080}
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.fbx")
            write_skeleton_glb(skel, os.path.join(d, "t.glb"))
            export_animation_fbx(p, skel, anim, vrm_path=os.path.join(d, "t.glb"))
            tree = read_fbx_tree(p)
        names = [n[0] for n in tree]
        for k in ("FBXHeaderExtension", "GlobalSettings", "Definitions", "Objects", "Connections", "Takes"):
            self.assertIn(k, names)
        objects = next(n for n in tree if n[0] == "Objects")
        models = [n for n in objects[2] if n[0] == "Model"]
        self.assertEqual(len(models), len(skel.bones) + 1)  # bones + camera
        curves = [n for n in objects[2] if n[0] == "AnimationCurve"]
        self.assertTrue(all(len(next(k for k in c[2] if k[0] == "KeyTime")[1][0]) == N for c in curves))



class Naturalize(unittest.TestCase):
    def test_anatomical_poses_are_kept(self):
        from vroidmocap.core.naturalize import Naturalizer
        skel = synthetic_skeleton()
        rng = np.random.default_rng(4)
        # continuous motion: blend between anatomical key poses (as in real footage)
        from vroidmocap.core.naturalize import slerp_many
        keys = [random_pose(skel, rng)[0] for _ in range(4)]
        Q = np.concatenate([np.stack([slerp_many(k0[b], k1[b], np.linspace(0, 1, 20, endpoint=False))
                                      for b in range(len(skel.bones))], 1) for k0, k1 in zip(keys[:-1], keys[1:])])
        root = np.tile([0, 1.0, 0], (len(Q), 1))
        Q2 = Naturalizer(skel).apply(Q, root, fps=30.0)
        # (a blend of two anatomical poses is itself anatomical, so little should change)
        pos, _ = avatar_fk(skel, Q, root)
        pos2, _ = avatar_fk(skel, Q2, root)
        # hands, feet and every finger joint stay where they were (IK keeps end points)
        ends = [skel.bones.index(b) for b in skel.bones if b.endswith(("Hand", "Foot", "Distal", "Intermediate"))]
        err = np.linalg.norm(pos2[:, ends] - pos[:, ends], axis=-1)
        self.assertLess(np.median(err), 0.01, f"median {np.median(err)}")

    def test_clamp_never_flip_flops(self):
        from vroidmocap.core.naturalize import clamp_continuous
        # a noisy signal hovering in the middle of the forbidden zone must not
        # alternate between the two bounds
        t = np.radians(-140 + 8 * np.random.default_rng(1).normal(size=300))
        out = clamp_continuous(t, np.radians(-100), np.radians(180))
        switches = np.sum(np.abs(np.diff(out)) > 1e-9)
        self.assertLessEqual(switches, 6)  # was 80 with a plain nearest-bound clamp

    def test_unseen_bones_freeze(self):
        from vroidmocap.core.naturalize import filter_rotations, q_angle
        N = 120
        ang = np.radians(np.linspace(0, 60, N))
        Q = np.zeros((N, 1, 4))
        Q[:, 0, 2] = np.sin(ang / 2)
        Q[:, 0, 3] = np.cos(ang / 2)
        conf = np.ones((N, 1))
        conf[40:90] = 0.0  # bone not visible
        # garbage while unseen must not leak in
        Q[40:90, 0] = normalize(np.random.default_rng(0).normal(size=(50, 4)))
        out = filter_rotations(Q, conf, fps=30.0, smoothing=1.0)
        # held near the last seen pose for most of the gap, no wild motion
        held = np.degrees(q_angle(out[45:80, 0], Q[39, 0]))
        self.assertLess(held.max(), 6.0)
        step = np.degrees(q_angle(out[1:, 0], out[:-1, 0]))
        # re-appearing: blended in over 0.2 s, never a jump
        self.assertLess(step.max(), 12.0)

    def test_backward_finger_is_reflected(self):
        from vroidmocap.core.naturalize import Naturalizer, q_axis_angle
        skel = synthetic_skeleton()
        nat = Naturalizer(skel)
        b = "leftIndexIntermediate"
        i = skel.bones.index(b)
        a = nat.axis[b]
        h = normalize(np.cross(a, [0, -1.0, 0]))
        Q = np.zeros((1, len(skel.bones), 4))
        Q[..., 3] = 1
        Q[0, i] = q_axis_angle(h, np.radians(-50))  # bent backwards 50 deg
        Q2 = nat.unflip_fingers(Q)
        d = quat_to_mat(Q2[0, i]) @ a
        u = normalize(np.cross(h, a))
        self.assertAlmostEqual(np.degrees(np.arctan2(d @ u, d @ a)), 50.0, places=3)


class TestLift(unittest.TestCase):
    """The 3D body solve: bone lengths + picture positions + a rough depth guess."""

    def _scene(self, n=120, seed=0):
        from vroidmocap.core import lift
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 2 * np.pi, n)
        X = np.zeros((n, 33, 3))
        X[..., 2] = 3.0
        hip = np.array([0.0, 0.1, 3.0])
        X[:, 23] = hip + [0.09, 0, 0]
        X[:, 24] = hip - [0.09, 0, 0]
        X[:, 11] = hip + [0.12, -0.5, 0]
        X[:, 12] = hip + [-0.12, -0.5, 0]

        def limb(root, mid, end, L1, L2, a1, a2, side):
            d1 = np.stack([side * 0.3 * np.cos(a1), np.cos(a1 * 0.5) * 0.8 + 0.2, np.sin(a1) * 0.6], 1)
            d1 /= np.linalg.norm(d1, axis=1, keepdims=True)
            d2 = np.stack([side * 0.2 * np.sin(a2), np.abs(np.cos(a2)) * 0.7 + 0.3, 0.5 * np.sin(a2 + 1)], 1)
            d2 /= np.linalg.norm(d2, axis=1, keepdims=True)
            X[:, mid] = X[:, root] + L1 * d1
            X[:, end] = X[:, mid] + L2 * d2
        limb(11, 13, 15, 0.26, 0.24, t, 2 * t, 1)
        limb(12, 14, 16, 0.26, 0.24, t + 1, 2 * t + 2, -1)
        limb(23, 25, 27, 0.40, 0.42, 0.5 * np.sin(t), 0.4 * np.sin(t) - 0.5, 1)
        limb(24, 26, 28, 0.40, 0.42, 0.5 * np.sin(t + 2), 0.4 * np.sin(t + 2) - 0.5, -1)
        for a, h, toe in ((27, 29, 31), (28, 30, 32)):
            X[:, h] = X[:, a] + [0, 0.05, 0.05]
            X[:, toe] = X[:, a] + [0, 0.07, -0.13]
        f = np.full(n, 1400.0)
        uv = np.stack([X[..., 0] / X[..., 2] * 1400 + 960, X[..., 1] / X[..., 2] * 1400 + 540], -1)
        # rough input: right picture position, depth squashed to 40% plus noise
        P0 = X.copy()
        zc = X[:, [23, 24], 2].mean(1, keepdims=True)
        P0[..., 2] = zc + 0.4 * (X[..., 2] - zc) + rng.normal(0, 0.01, X.shape[:2])
        P0[..., :2] = X[..., :2] / X[..., 2:3] * P0[..., 2:3]
        return lift, X, P0, uv, f

    def test_recovers_depth(self):
        lift, X, P0, uv, f = self._scene()
        n = len(X)
        P, L = lift.fuse_body(P0, uv, np.ones((n, 33)), f, 960, 540, [(0, n)], 30.0)
        self.assertAlmostEqual(L[(11, 13)], 0.26, delta=0.02)
        self.assertAlmostEqual(L[(25, 27)], 0.42, delta=0.03)

        def err(Q):
            out = []
            for a, b in ((11, 13), (13, 15), (12, 14), (14, 16), (23, 25), (25, 27), (24, 26), (26, 28)):
                u, v = Q[:, b] - Q[:, a], X[:, b] - X[:, a]
                c = np.sum(u * v, -1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
                out.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
            return float(np.mean(out))
        self.assertLess(err(P), 0.6 * err(P0))
        self.assertLess(err(P), 8.0)

    def test_bone_lengths_constant(self):
        lift, X, P0, uv, f = self._scene(seed=1)
        n = len(X)
        P, L = lift.fuse_body(P0, uv, np.ones((n, 33)), f, 960, 540, [(0, n)], 30.0)
        d = np.linalg.norm(P[:, 13] - P[:, 11], axis=-1)
        self.assertLess(np.std(d), 0.01)


class Facing(unittest.TestCase):
    """A torso turning a full circle: the facing angle and swapped left/right
    labels must be recovered from the picture alone."""

    def _scene(self, n=240, swap_from=90, swap_to=150):
        from vroidmocap.core import detect as D
        psi = np.radians(np.linspace(0, 360, n, endpoint=False))
        f, cx, cy, Z = 1500.0, 960.0, 540.0, 3.0
        body = {  # x = character's left, y = up, z = forward (metres, hip centre origin)
            D.L_SH: [0.18, 0.5, 0], D.R_SH: [-0.18, 0.5, 0], D.L_HIP: [0.1, 0, 0], D.R_HIP: [-0.1, 0, 0],
            D.NOSE: [0, 0.68, 0.1], D.L_KNEE: [0.1, -0.42, 0.08], D.R_KNEE: [-0.1, -0.42, 0.08],
            D.L_ANK: [0.1, -0.82, 0], D.R_ANK: [-0.1, -0.82, 0], D.L_HEEL: [0.1, -0.86, -0.05],
            D.R_HEEL: [-0.1, -0.86, -0.05], D.L_FOOT: [0.1, -0.88, 0.13], D.R_FOOT: [-0.1, -0.88, 0.13]}
        uv = np.zeros((n, 33, 2))
        Wc = np.zeros((n, 33, 3))
        for k, p in enumerate(psi):
            # psi = 0 faces the camera: character forward = -z (cv camera), left = +x
            c, s = np.cos(p), np.sin(p)
            for j, (x, y, z) in body.items():
                fwd = np.array([s, 0, -c])
                left = np.array([c, 0, s])
                P = x * left + np.array([0, -y, 0]) + z * fwd
                Wc[k, j] = P
                uv[k, j] = [cx + f * P[0] / (Z + P[2]), cy + f * P[1] / (Z + P[2])]
        conf = np.ones((n, 33))
        face = np.clip(np.cos(psi) + 0.5, 0, 1)  # face seen from the front only
        # the pose model's 3D guess is mirrored front/back half the time
        Wm = Wc.copy()
        Wm[::2, :, 2] *= -1
        # labels swapped for a stretch (as detectors do in back views)
        from vroidmocap.core.retarget import LR_PERM
        sw = np.zeros(n, bool)
        sw[swap_from:swap_to] = True
        uv[sw] = uv[sw][:, LR_PERM]
        Wm[sw] = Wm[sw][:, LR_PERM]
        T = np.tile([0, 0, Z], (n, 1))
        return psi, sw, uv, conf, face, Wm, T, np.full(n, f), cx, cy

    def test_full_turn_and_swapped_labels(self):
        from vroidmocap.core import facing
        psi, sw, uv, conf, face, Wc, T, f, cx, cy = self._scene()
        n = len(psi)
        pc, pp, swap, bad = facing.track_facing(uv, conf, face, Wc, T, f, cx, cy, np.ones(n, bool), [(0, n)], 30.0)
        err_c = np.degrees(np.abs(np.angle(np.exp(1j * (pc - psi)))))
        err_p = np.degrees(np.abs(np.angle(np.exp(1j * (pp - psi)))))
        self.assertLess(np.median(err_c), 10)
        self.assertLess(np.median(err_p), 10)
        self.assertLess(np.mean(err_c > 45), 0.05)
        self.assertGreater(np.mean(swap == sw), 0.95)

    def test_single_frame_swap_is_dropped(self):
        from vroidmocap.core import facing
        from vroidmocap.core.retarget import LR_PERM
        psi, sw, uv, conf, face, Wc, T, f, cx, cy = self._scene(swap_from=0, swap_to=0)
        uv[40] = uv[40][LR_PERM]
        Wc[40] = Wc[40][LR_PERM]
        n = len(psi)
        pc, pp, swap, bad = facing.track_facing(uv, conf, face, Wc, T, f, cx, cy, np.ones(n, bool), [(0, n)], 30.0)
        self.assertFalse(swap[40])
        self.assertTrue(bad[40])
        self.assertLess(bad.sum(), 5)


class KeyframesWhereSeen(unittest.TestCase):
    """A bone that is not seen keeps its previous pose exactly; before it is
    first seen it has its first seen pose; when seen again it moves on from the
    held pose without a jump."""

    def test_hold_and_resume(self):
        from vroidmocap.core.retarget import hold_unseen
        from vroidmocap.core.naturalize import q_angle
        N = 60
        ang = np.radians(np.linspace(0, 120, N))
        Q = np.zeros((N, 1, 4))
        Q[:, 0, 2], Q[:, 0, 3] = np.sin(ang / 2), np.cos(ang / 2)
        seen = np.ones((N, 1), bool)
        seen[:10] = False
        seen[30:40] = False
        out = hold_unseen(Q, seen, 30.0, np.array([[0, 0, 0, 1.0]]))
        # before first sight: the neutral (A) pose, then easing into the motion
        self.assertTrue(np.allclose(out[:10, 0], [0, 0, 0, 1]))
        self.assertLess(np.degrees(q_angle(out[17:30, 0], Q[17:30, 0])).max(), 0.5)
        # unseen: no keyframes there; the in-between motion carries on smoothly
        # from the last seen pose to the next (here: along the steady turn)
        self.assertLess(np.degrees(q_angle(out[30:40, 0], Q[30:40, 0])).max(), 3.0)
        # no frame-to-frame step larger than the motion itself plus the catch-up
        steps = np.degrees(q_angle(out[11:, 0], out[10:-1, 0]))
        self.assertLess(steps.max(), 12.0)
        # caught up within 0.2 s
        self.assertLess(np.degrees(q_angle(out[47:, 0], Q[47:, 0])).max(), 0.5)

    def test_never_seen_is_not_tpose(self):
        from vroidmocap.core.retarget import hold_unseen, neutral_pose
        skel = synthetic_skeleton()
        N = 5
        Q = np.tile([0, 0, 0, 1.0], (N, len(skel.bones), 1))
        seen = np.ones((N, len(skel.bones)), bool)
        seen[:, skel.bones.index("leftUpperArm")] = False
        out = hold_unseen(Q, seen, 30.0, neutral_pose(skel), skel)
        self.assertFalse(np.allclose(out[:, skel.bones.index("leftUpperArm")], [0, 0, 0, 1]))


class SceneCamera(unittest.TestCase):
    """Camera from the scenery alone: turns, zooms and travel."""

    def _track(self, K, Cf, Rf, zf, f0=1500.0, seed=1):
        rng = np.random.default_rng(seed)
        X = np.c_[rng.uniform(-8, 8, 1500), rng.uniform(-3, 3, 1500), rng.uniform(3, 12, 1500)]
        pairs, ids, prev = [None], [None], None
        for k in range(K):
            v = (X - Cf(k)) @ Rf(k)
            f = f0 * zf(k)
            p = np.c_[f * v[:, 0] / v[:, 2] + 960, f * v[:, 1] / v[:, 2] + 540] + rng.normal(0, 0.3, (len(X), 2))
            vis = (v[:, 2] > 0.5) & (p[:, 0] > 0) & (p[:, 0] < 1920) & (p[:, 1] > 0) & (p[:, 1] < 1080)
            if prev is not None:
                sel = np.where(vis & prev[1])[0][:300]
                pairs.append(np.c_[prev[0][sel], p[sel]])
                ids.append(sel)
            prev = (p, vis)
        return pairs, ids

    def test_still_pan_zoom_and_travel(self):
        from vroidmocap.core import scene
        K = 40
        cases = [("still", lambda k: np.zeros(3), lambda k: np.eye(3), lambda k: 1.0, False),
                 ("pan+zoom", lambda k: np.zeros(3), lambda k: scene.expmap(np.array([0, 0.004 * k, 0])), lambda k: 1 + 0.01 * k, False),
                 ("truck", lambda k: np.array([0.03 * k, 0, 0]), lambda k: np.eye(3), lambda k: 1.0, True)]
        for name, Cf, Rf, zf, travels in cases:
            pairs, ids = self._track(K, Cf, Rf, zf)
            R, zoom, C, moving, seg, par, f_used, _ = scene.solve_shot(pairs, ids, [True] * K, 1500.0, 960, 540, 30)
            err = max(np.degrees(np.arccos(np.clip((np.trace(R[k].T @ Rf(k)) - 1) / 2, -1, 1))) for k in range(K))
            self.assertLess(err, 0.1, name)
            self.assertAlmostEqual(zoom[-1], zf(K - 1), delta=0.01, msg=name)
            self.assertEqual(bool((seg >= 0).any()), travels, name)
            if not travels:
                self.assertTrue(np.allclose(C, 0), name)  # a camera that only turns / zooms never moves
            else:
                d = C[-1] / np.linalg.norm(C[-1])
                self.assertGreater(d[0], 0.99, name)  # travel direction from the parallax


class FbxAvatars(unittest.TestCase):
    def test_bone_names(self):
        from vroidmocap.core.fbx_avatar import classify
        for name, kind, side in [("mixamorig:LeftForeArm", "lowerArm", "left"), ("mixamorig:RightUpLeg", "upperLeg", "right"),
                                 ("thigh_l", "upperLeg", "left"), ("calf_r", "lowerLeg", "right"), ("J_Bip_L_UpperArm", "upperArm", "left"),
                                 ("upper_arm.R", "upperArm", "right"), ("upperarm_twist_01_l", "skip", "left"),
                                 ("左ひじ", "lowerArm", "left"), ("mixamorig:LeftHandIndex1", "finger:Index", "left"),
                                 ("Bip01 R Clavicle", "shoulder", "right"), ("pelvis", "hips", None), ("HeadTop_End", "skip", None)]:
            k, sd, _ = classify(name)
            self.assertEqual((k, sd), (kind, side), name)

    def test_fbx_avatar_matches_solver_pose(self):
        from vroidmocap.core.fbx_avatar import FbxAvatar
        from vroidmocap.core.fbx_writer import export_animation_fbx
        from vroidmocap.core.mathutil import quat_to_mat
        from vroidmocap.core.retarget import avatar_fk
        skel = synthetic_skeleton()
        rng = np.random.default_rng(3)
        N = 4
        anim = {"bones": skel.bones, "quats": np.tile([0, 0, 0, 1.0], (N, len(skel.bones), 1)).astype(np.float32),
                "root": np.tile(skel.rest_pos["hips"], (N, 1)), "fps": 30.0, "frames": N, "width": 1920, "height": 1080}
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.fbx")
            write_skeleton_glb(skel, os.path.join(d, "t.glb"))
            export_animation_fbx(p, skel, anim, vrm_path=os.path.join(d, "t.glb"), include_camera=False)
            av = FbxAvatar(p)
        sk = av.skeleton()
        for b in sk.bones:  # same T-pose skeleton back
            self.assertLess(np.linalg.norm(sk.rest_pos[b] - skel.rest_pos[b]), 1e-3, b)
        Q = np.concatenate([random_pose(sk, rng) for _ in range(N)])
        root = np.tile(sk.rest_pos["hips"], (N, 1)) + rng.normal(0, 0.1, (N, 3))
        names, quats, hips_t = av.pose(sk.bones, Q, root)
        # rebuild the model's own bones from the file's rest transforms + the posed rotations
        id_of = {d_["name"]: m for m, d_ in av.models.items()}
        Lq = {id_of[n]: quats[:, i] for i, n in enumerate(names)}
        pos, _ = avatar_fk(sk, Q, root)
        for t in range(N):
            W = {}
            for m in av.order:
                L = av.local[m].copy()
                sc = np.linalg.norm(L[:3, :3], axis=0)
                if m in Lq:
                    L[:3, :3] = quat_to_mat(Lq[m][t].astype(np.float64)) * sc
                if m == av.hb["hips"]:
                    L[:3, 3] = hips_t[t]
                pa = av.parent.get(m)
                W[m] = L if pa is None else W[pa] @ L
            for i, b in enumerate(sk.bones):
                got = av.A @ W[av.hb[b]][:3, 3] * av.unit
                self.assertLess(np.linalg.norm(got - pos[t, i]), 2e-3, b)


class SceneHelpers(unittest.TestCase):
    def test_crop(self):
        from vroidmocap.core.retarget import crop_anim
        a = {"frames": 10, "fps": 30.0, "quats": np.zeros((10, 3, 4)), "root": np.arange(30.0).reshape(10, 3), "bones": ["a"]}
        c = crop_anim(a, 2, 5)
        self.assertEqual(c["frames"], 4)
        self.assertTrue(np.allclose(c["root"][0], a["root"][2]))
        self.assertEqual(c["bones"], ["a"])

    def test_edge_continuation(self):
        from vroidmocap.core.capture import extend_at_edges
        N = 60
        boxes = np.full((N, 4), np.nan)
        for f in range(30):  # walks right and reaches the edge of a 1000 px wide picture
            x = 600 + 10 * f
            boxes[f] = [x, 100, x + 200, 700]
        out, segs = extend_at_edges(boxes, [(0, 30)], np.zeros(N, int), 1000, 800, 30)
        self.assertGreater(segs[0][1], 30)          # carried on past the end of the track
        self.assertTrue(np.isnan(out[59, 0]))       # ...but not once it is out of the picture
        self.assertGreater(out[31, 0], out[29, 0])  # along its motion

    def test_a_pose_until_seen(self):
        from vroidmocap.core.retarget import hold_unseen, neutral_pose
        skel = synthetic_skeleton()
        N = 12
        Q = np.tile([0, 0, 0.3, 0.95], (N, len(skel.bones), 1))
        seen = np.zeros((N, len(skel.bones)), bool)
        seen[6:] = True
        out = hold_unseen(Q, seen, 30.0, neutral_pose(skel), skel)
        ua = skel.bones.index("leftUpperArm")
        self.assertTrue(np.allclose(out[:6, ua], neutral_pose(skel)[ua], atol=1e-6))


class SeenRules(unittest.TestCase):
    def test_weak_detections_linked_to_a_clear_sighting_both_ways(self):
        from vroidmocap.core.retarget import joint_seen
        N = 60
        uv = np.full((N, 33, 2), 500.0)
        conf = np.zeros((N, 33))
        weak = np.zeros((N, 33), bool)
        conf[30:, 0] = 1.0       # clearly seen from frame 30 on
        weak[10:30, 0] = True    # weakly seen just before (e.g. legs at the top of the picture)
        weak[:5, 1] = True       # weak only, never clear: not seen
        s = joint_seen(uv, conf, np.ones(N, bool), 1920, 1080, 30.0, np.zeros(N, int), weak=weak)
        self.assertTrue(s[10:, 0].all())
        self.assertFalse(s[:10, 0].any())
        self.assertFalse(s[:, 1].any())

    def test_limb_squeezed_into_the_picture_is_cut_off(self):
        from vroidmocap.core.detect import L_ANK, L_HIP, L_KNEE
        from vroidmocap.core.retarget import cut_off_joints
        N = 2
        uv = np.zeros((N, 33, 2))
        P = np.zeros((N, 33, 3))
        P[:, :, 2] = 1.5  # close to the camera
        uv[:, L_HIP] = [960, 800]
        uv[0, L_KNEE] = [960, 1000]   # 200 px thigh ending near the bottom edge: far too short at 1.5 m
        uv[1, L_KNEE] = [960, 600]    # a short thigh pointing up, away from the edge: not cut off
        uv[:, L_ANK] = uv[:, L_KNEE] + [0, 20]
        blen = {(L_HIP, L_KNEE): 0.42, (L_KNEE, L_ANK): 0.40}
        out = cut_off_joints(uv, P, blen, np.full(N, 1500.0), 960, 540, 1920, 1080)
        self.assertTrue(out[0, L_KNEE] and out[0, L_ANK])
        self.assertFalse(out[1, L_KNEE])


class BoneTrackers(unittest.TestCase):
    def test_hidden_knee_from_hip_and_thigh_tracker(self):
        from vroidmocap.core.detect import L_HIP, L_KNEE
        from vroidmocap.core.retarget import infer_from_markers
        N = 1
        uv = np.zeros((N, 33, 2))
        conf = np.zeros((N, 33))
        strong = np.zeros((N, 33), bool)
        uv[0, L_HIP] = [500, 300]
        strong[0, L_HIP] = True               # hip seen, knee hidden
        markers = np.full((N, 8, 3, 2), np.nan)
        markers[0, 4, 0] = [510, 350]         # only the first tracker (25%) of the left thigh is visible
        covered = np.zeros((N, 33), bool)
        covered[0, L_KNEE] = True             # something is in front of the knee
        uv2, conf2, inf = infer_from_markers(uv, conf, strong, markers, (0.25, 0.5, 0.75), np.ones(N, bool), 1920, 1080,
                                             covered=covered)
        self.assertTrue(inf[0, L_KNEE])
        self.assertTrue(np.allclose(uv2[0, L_KNEE], [540, 500]))  # the line, carried to full length
        # a knee that is in the picture and not covered (e.g. under a skirt) is left to the detectors
        _, _, inf2 = infer_from_markers(uv, conf, strong, markers, (0.25, 0.5, 0.75), np.ones(N, bool), 1920, 1080)
        self.assertFalse(inf2[0, L_KNEE])


class BonesFromSegments(unittest.TestCase):
    def test_shin_from_knee_and_tracker_when_the_foot_is_below_the_picture(self):
        from vroidmocap.core.detect import L_ANK, L_KNEE
        from vroidmocap.core.retarget import bones_from_segments
        N, W, H = 1, 1920, 1080
        uv = np.zeros((N, 33, 2)); conf = np.zeros((N, 33)); seen0 = np.zeros((N, 33), bool)
        uv[0, L_KNEE] = [900, 800]; seen0[0, L_KNEE] = True      # knee seen
        uv[0, L_ANK] = [905, 1078]; conf[0, L_ANK] = 0.7          # the detector puts the ankle on the edge
        markers = np.full((N, 8, 3, 2), np.nan)
        markers[0, 5, 0] = [910, 900]                             # the shin's first tracker (25%) is seen
        wb2d = np.zeros((N, 133, 2)); wb2d[0, 13] = uv[0, L_KNEE]  # (the whole-body knee is the left one)
        L2d = np.full((N, 8), 400.0); dz = np.zeros((N, 8))
        uv2, conf2, known, placed = bones_from_segments(uv, conf, seen0, markers, (0.25, 0.5, 0.75),
                                                        np.ones(N, bool), W, H, wb2d, L2d, dz)
        self.assertTrue(known[0, 5])                               # the shin is known
        self.assertTrue(placed[0, L_ANK])
        d = uv2[0, L_ANK] - uv[0, L_KNEE]
        self.assertAlmostEqual(np.linalg.norm(d), 400.0, delta=4.0)  # its full length, past the edge
        self.assertGreater(uv2[0, L_ANK, 1], H)
        u = d / np.linalg.norm(d); v = np.array([10, 100.0]) / np.linalg.norm([10, 100.0])
        self.assertGreater(np.dot(u, v), 0.999)                    # along the knee -> tracker line

    def test_an_arm_behind_the_body_is_left_alone(self):
        from vroidmocap.core.detect import L_EL, L_WR
        from vroidmocap.core.retarget import bones_from_segments
        N = 1
        uv = np.zeros((N, 33, 2)); conf = np.zeros((N, 33)); seen0 = np.zeros((N, 33), bool)
        uv[0, L_EL] = [900, 500]; seen0[0, L_EL] = True
        markers = np.full((N, 8, 3, 2), np.nan); markers[0, 1, 0] = [920, 560]
        wb2d = np.zeros((N, 133, 2)); wb2d[0, 7] = uv[0, L_EL]
        _, _, known, _ = bones_from_segments(uv, conf, seen0, markers, (0.25, 0.5, 0.75), np.ones(N, bool),
                                             1920, 1080, wb2d, np.full((N, 8), 200.0), np.zeros((N, 8)))
        self.assertFalse(known[0, 1])  # wrist in the picture, nothing in front of it: the trackers may be on the chest


class HandKeys(unittest.TestCase):
    def test_curves_hold_and_do_not_overshoot(self):
        from vroidmocap.core.keys import evaluate
        k = [[0.0, 0.0], [1.0, 100.0], [2.0, 100.0], [3.0, 20.0]]
        t = np.linspace(-1, 4, 501)
        v = evaluate(k, t)
        self.assertTrue(np.all(v >= -1e-9) and np.all(v <= 100 + 1e-9))
        self.assertAlmostEqual(evaluate(k, [-1.0])[0], 0.0)
        self.assertAlmostEqual(evaluate(k, [1.5])[0], 100.0)      # flat between equal keys
        self.assertAlmostEqual(evaluate(k, [9.0])[0], 20.0)       # holds after the last key
        self.assertEqual(evaluate([], [0.5], 15.0)[0], 15.0)       # no keys: the default

    def test_clean_keys_clamps_and_sorts(self):
        from vroidmocap.core.keys import clean_keys
        c = clean_keys({"leftIndex.curl": [[2, 500], [1, -50], ["x", 1]], "expr.happy": [[0, 30]], "bad": 3})
        self.assertEqual(c["leftIndex.curl"], [[1.0, -10.0], [2.0, 100.0]])
        self.assertEqual(c["expr.happy"], [[0.0, 30.0]])
        self.assertNotIn("bad", c)

    def test_fist_curls_towards_the_palm_and_splay_spreads(self):
        from vroidmocap.core.keys import bake_fingers
        skel = synthetic_skeleton()
        B = len(skel.bones)
        bi = {b: i for i, b in enumerate(skel.bones)}
        anim = {"bones": skel.bones, "quats": np.tile([0, 0, 0, 1.0], (3, B, 1)), "fps": 1.0, "frames": 3}
        ch = {}
        for side in ("left", "right"):
            for f in FINGERS:
                ch[f"{side}{f}.curl"] = [[0, 0], [1, 100], [2, 0]]
                ch[f"{side}{f}.splay"] = [[0, 0], [1, 0], [2, 10]]
        Q = bake_fingers(skel, anim, ch)
        pos, _ = avatar_fk(skel, Q.astype(float), np.zeros((3, 3)))
        for side in ("left", "right"):
            h = pos[:, bi[side + "Hand"]]
            tip = pos[:, bi[side + "IndexDistal"]]
            self.assertLess(tip[1, 1] - h[1, 1], tip[0, 1] - h[0, 1] - 0.01)  # fist: down, to the palm side
            def d(k, f):
                v = pos[k, bi[side + f + "Intermediate"]] - pos[k, bi[side + f + "Proximal"]]
                return v / np.linalg.norm(v)
            spread0 = np.degrees(np.arccos(np.clip(np.dot(d(0, "Index"), d(0, "Little")), -1, 1)))
            spread2 = np.degrees(np.arccos(np.clip(np.dot(d(2, "Index"), d(2, "Little")), -1, 1)))
            self.assertGreater(spread2, spread0 + 20)


class BlendShapeExport(unittest.TestCase):
    def test_blend_shape_channels_are_written_and_animated(self):
        from vroidmocap.core.fbx_writer import write_fbx
        N = 5
        nodes = [{"name": "root", "parent": None, "kind": "LimbNode", "rest_t": np.zeros(3), "rest_r": np.zeros(3),
                  "anim_t": None, "anim_r": np.zeros((N, 3))}]
        model = {"meshes": [{"name": "Face", "vertices": np.eye(3) * 10, "normals": None, "uvs": None,
                             "tris": np.array([[0, 1, 2]]), "tri_mat": np.array([0]), "materials": [0],
                             "parent": None, "skin": None,
                             "shapes": [{"name": "Smile", "index": np.array([1]), "delta": np.array([[0, 1.0, 0]]),
                                         "keys": [(0, 0)]}]}],
                 "materials": [{"name": "m", "color": [1, 1, 1, 1], "texture": None}], "textures": [],
                 "shape_anim": {(0, 0): np.linspace(0, 1, N)}}
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.fbx")
            write_fbx(p, nodes, N, 30.0, model=model)
            tree = read_fbx_tree(p)
        objects = next(n for n in tree if n[0] == "Objects")
        kinds = [(n[0], n[1][-1].decode() if n[1] and isinstance(n[1][-1], bytes) else None) for n in objects[2]]
        self.assertIn(("Deformer", "BlendShape"), kinds)
        self.assertIn(("Deformer", "BlendShapeChannel"), kinds)
        self.assertIn(("Geometry", "Shape"), kinds)
        conns = next(n for n in tree if n[0] == "Connections")
        self.assertTrue(any(c[1][0] == b"OP" and c[1][-1] == b"DeformPercent" for c in conns[2]))


class Identities(unittest.TestCase):
    def _track(self, tid, shot, frames, box, emb, hist):
        from types import SimpleNamespace
        n = len(frames)
        return SimpleNamespace(id=tid, shot=shot, frames=list(frames), boxes=[np.array(box, float)] * n,
                               embs=[np.asarray(emb, float)], hists=[np.asarray(hist, float)],
                               vis=[np.full(33, 0.9)] * n, lm2d=[np.zeros((33, 2))] * n, det_n=n)

    def test_close_ups_join_the_main_character(self):
        from vroidmocap.core.analysis import cluster_identities
        rng = np.random.default_rng(1)
        e0 = rng.normal(size=16); h0 = np.abs(rng.normal(size=204))
        e1 = e0 + 0.9 * rng.normal(size=16); h1 = h0 * np.abs(1 + 0.6 * rng.normal(size=204))
        full = [self._track(i, i, range(i * 100, i * 100 + 90), [700, 100, 1100, 1000], e0, h0) for i in range(4)]
        # a face close-up in its own shot: the box fills the picture
        close = self._track(9, 9, range(900, 960), [300, 0, 1500, 1080], e1, h1)
        chars, _ = cluster_identities(full + [close], 30.0, frame_size=(1920, 1080))
        self.assertEqual(len(chars), 1)
        # without knowing the picture size (no partial-view rule) it may stay apart
        chars2, _ = cluster_identities(full + [close], 30.0)
        self.assertGreaterEqual(len(chars2), 1)

    def _dist(self, ea, ha, eb, hb):
        # the clustering's appearance distance (embedding + part histograms)
        from vroidmocap.core.mathutil import normalize
        de = 1 - float(np.dot(normalize(ea), normalize(eb)))
        ds = []
        for p in range(3):
            x, y = ha[p * 68:(p + 1) * 68], hb[p * 68:(p + 1) * 68]
            bc = np.sum(np.sqrt(x * y)) / max(np.sqrt(x.sum() * y.sum()), 1e-9)
            ds.append(np.sqrt(max(0.0, 1 - bc)))
        return 0.45 * min(1.0, de * 2.0) + 0.55 * float(np.average(ds, weights=[1.4, 1.0, 0.8]))

    def test_unlike_close_up_joins_only_when_nobody_else_is_seen(self):
        from vroidmocap.core.analysis import cluster_identities, PARTIAL_THRESHOLD, PARTIAL_THRESHOLD_ALONE
        rng = np.random.default_rng(3)
        e0 = rng.normal(size=16); h0 = np.abs(rng.normal(size=204))
        # a close-up that looks only fairly like her (between the two limits)
        for k in range(200):
            e1 = e0 + rng.uniform(0.3, 2.0) * rng.normal(size=16)
            h1 = h0 * np.abs(1 + rng.uniform(0.3, 1.5) * rng.normal(size=204))
            d = self._dist(e0, h0, e1, h1)
            if PARTIAL_THRESHOLD + 0.03 < d < PARTIAL_THRESHOLD_ALONE - 0.03:
                break
        else:
            self.fail("no test appearance in range")
        full = [self._track(i, i, range(i * 100, i * 100 + 90), [700, 100, 1100, 1000], e0, h0) for i in range(4)]
        close = self._track(9, 9, range(900, 960), [300, 0, 1500, 1080], e1, h1)
        chars, _ = cluster_identities(full + [close], 30.0, frame_size=(1920, 1080))
        self.assertEqual(len(chars), 1)
        # a second person, seen whole (at the same time as her, looking unlike both): the close-up could be either
        e2 = rng.normal(size=16); h2 = np.abs(rng.normal(size=204)) ** 3
        other = [self._track(20 + i, i, range(i * 100, i * 100 + 90), [1300, 100, 1700, 1000], e2, h2) for i in range(4)]
        chars2, _ = cluster_identities(full + other + [close], 30.0, frame_size=(1920, 1080))
        self.assertEqual(len(chars2), 3)

    def test_detector_tracks_count_as_people(self):
        from types import SimpleNamespace
        from vroidmocap.core.analysis import is_human_track
        rng = np.random.default_rng(2)
        n = 20
        t = SimpleNamespace(frames=list(range(n)), vis=[np.full(33, 0.3)] * n,
                            lm2d=[rng.uniform(0, 500, (33, 2)) for _ in range(n)],
                            boxes=[np.array([0, 0, 100, 300.0])] * n, det_n=n)
        self.assertTrue(is_human_track(t))  # shaky pose readings, but the person detector saw a person
        t.det_n = 0
        self.assertFalse(is_human_track(t))


if __name__ == "__main__":
    unittest.main()


class BodyContacts(unittest.TestCase):
    """Limbs do not pass through each other (they end up just touching), a
    palm can be put exactly on a point, and an unseen lower leg stands on
    the floor."""

    def _pose(self, skel, n, bone, axis, deg):
        Q = np.zeros((n, len(skel.bones), 4))
        Q[..., 3] = 1
        a = np.radians(deg) / 2
        Q[:, skel.bones.index(bone)] = np.r_[np.sin(a) * np.asarray(axis, float), np.cos(a)]
        return Q

    def test_forearms_crossing_are_pushed_apart(self):
        from vroidmocap.core import body_shape
        skel = synthetic_skeleton()
        caps = body_shape.capsules(skel)
        n = 5
        Q = np.zeros((n, len(skel.bones), 4))
        Q[..., 3] = 1
        # both arms forward, forearms bent in towards each other so they cross
        for side, sg in (("left", 1), ("right", -1)):
            a = np.radians(-90 * sg) / 2
            Q[:, skel.bones.index(side + "UpperArm")] = [0, np.sin(a), 0, np.cos(a)]
            b = np.radians(-80 * sg) / 2
            Q[:, skel.bones.index(side + "LowerArm")] = [0, np.sin(b), 0, np.cos(b)]
        before = body_shape.penetration(skel, Q, None, caps)
        self.assertTrue((before > 0.02).all())
        Q2, left = body_shape.resolve_collisions(skel, Q, caps, 30, np.zeros(n, int), smooth_s=0)
        after = body_shape.penetration(skel, Q2, None, caps)
        self.assertTrue((after < 0.005).all(), after)

    def test_palm_reaches_its_target(self):
        from vroidmocap.core import contacts
        skel = synthetic_skeleton()
        n = 3
        Q = np.zeros((n, len(skel.bones), 4))
        Q[..., 3] = 1
        root = np.tile([[0, 1.0, 0]], (n, 1))
        target = np.tile([[0.05, 1.75, 0.15]], (n, 1))
        contacts.arm_reach(skel, Q, root, "left", target, np.ones(n))
        pos, rot = avatar_fk(skel, Q, root)
        i, j = skel.bones.index("leftHand"), skel.bones.index("leftMiddleProximal")
        palm = pos[:, i] + 0.5 * (pos[:, j] - pos[:, i])
        self.assertLess(np.abs(palm - target).max(), 0.02)

    def test_unseen_lower_leg_stands_on_the_floor(self):
        from vroidmocap.core.retarget import _foot_lows, planted_legs
        skel = synthetic_skeleton()
        n = 4
        Q = np.zeros((n, len(skel.bones), 4))
        Q[..., 3] = 1
        hip_h = float(skel.rest_pos["hips"][1])
        root = np.tile([[0, hip_h - 0.12, 0]], (n, 1))  # hips lower than standing: knees must bend
        seen = np.ones((n, len(skel.bones)), bool)
        for b in ("leftUpperLeg", "leftLowerLeg", "leftFoot"):
            seen[:, skel.bones.index(b)] = False
        Q2, planted = planted_legs(Q, root, seen, np.ones(n, bool), skel, 30, np.zeros(n, int))
        self.assertTrue(planted[:, 0].all())
        lows, _ = _foot_lows(skel, Q2, root)
        self.assertLess(np.abs(lows[:, 0]).max(), 0.02)


class ScanRanges(unittest.TestCase):
    def test_ranges_start_at_cuts_on_the_frame_grid(self):
        from vroidmocap.core.analysis import scan_ranges
        r = scan_ranges(1000, 2, [101, 333, 500, 777], 4)
        self.assertEqual(r[0][0], 0)
        self.assertEqual(r[-1][1], 1000)
        for (a, b), (c, d) in zip(r, r[1:]):
            self.assertEqual(b, c)
        for a, _ in r[1:]:
            self.assertEqual(a % 2, 0)
            self.assertTrue(any(a - 2 < c <= a for c in (101, 333, 500, 777)))
        self.assertEqual(scan_ranges(1000, 2, [], 4), [(0, 1000)])
        self.assertEqual(scan_ranges(1000, 2, [500], 1), [(0, 1000)])

    def test_merge_makes_track_ids_unique(self):
        from vroidmocap.core.analysis import merge_scans, Track
        def part(start, n):
            ts = []
            for _ in range(n):
                t = Track(0)
                t.id = 0 if not ts else ts[-1].id + 1  # (ids repeat across parts, as in separate processes)
                t.frames = [start]
                ts.append(t)
            return {"start": start, "frame_ids": [start], "H_rel": [None], "H_ok": [False], "H_ids": [None],
                    "shot_starts": [start], "tracks": ts, "thumbs": {t.id: (1.0, None, start) for t in ts}}
        sc = merge_scans([part(10, 2), part(0, 3)], 2, 20, [10])
        ids = [t.id for t in sc["tracks"]]
        self.assertEqual(sorted(ids), list(range(5)))
        self.assertEqual(sc["frame_ids"], [0, 10])
        self.assertEqual(sc["shot_starts"], [0, 10])
        self.assertEqual(sorted(sc["thumbs"]), list(range(5)))


class ShotLink(unittest.TestCase):
    def test_rotation_between_two_views_of_the_background(self):
        import cv2
        from vroidmocap.core.shotlink import relative_rotation, _features
        from vroidmocap.core.camera import yaw_matrix
        rng = np.random.default_rng(0)
        W, H, f = 1280, 720, 900.0
        # a textured scene far away, seen by a camera that then turns 12 deg
        big = np.zeros((2000, 3000, 3), np.uint8)
        for _ in range(900):
            x, y = rng.integers(0, 3000), rng.integers(0, 2000)
            cv2.rectangle(big, (int(x), int(y)), (int(x + rng.integers(10, 80)), int(y + rng.integers(10, 80))),
                          tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
        K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
        Kb = np.array([[f, 0, 1500], [0, f, 1000], [0, 0, 1]])
        R = yaw_matrix(np.radians(12))
        A = cv2.warpPerspective(big, K @ np.linalg.inv(Kb), (W, H))
        B = cv2.warpPerspective(big, K @ R @ np.linalg.inv(Kb), (W, H))
        sift = cv2.SIFT_create(nfeatures=2500)
        r = relative_rotation(_features(sift, A, [], 0.75), _features(sift, B, [], 0.75), f, f, W / 2, H / 2)
        self.assertIsNotNone(r)
        ang = np.degrees(np.arccos(np.clip((np.trace(r[0] @ R.T) - 1) / 2, -1, 1)))
        self.assertLess(ang, 1.5)

    def test_yaw_from_link(self):
        from vroidmocap.core.shotlink import yaw_from_link
        from vroidmocap.core.camera import yaw_matrix
        A_a, A_b = yaw_matrix(0.3), yaw_matrix(-0.2)
        off_b = 1.1
        R_a, R_b = A_a, yaw_matrix(off_b) @ A_b
        R_link = R_b.T @ R_a  # x_b = R x_a
        yaw, tilt = yaw_from_link(R_a, R_link, A_b)
        self.assertAlmostEqual(yaw, off_b, places=6)
        self.assertLess(tilt, 1e-4)


class FacingMarks(unittest.TestCase):
    def test_marks_cover_the_shot_between_cuts(self):
        import tempfile, json as js
        from vroidmocap.server import AppState
        from vroidmocap.core.retarget import load_facing_hints
        st = AppState()
        st.workdir = tempfile.mkdtemp()
        st.analysis = {"frames": 300, "cuts": [80, 156, 203], "shots": [{"start": 0}, {"start": 40}, {"start": 80}]}
        r = st.set_facing(0, 120, "back")
        self.assertEqual(r["shot"], (80, 156) if isinstance(r["shot"], tuple) else [80, 156])
        st.set_facing(0, 230, "front")
        st.set_facing(0, 10, "front")
        st.set_facing(0, 10, "auto")
        h = st.facing_hints(0)["hints"]
        self.assertEqual([(x["start"], x["end"], x["facing"]) for x in h], [(80, 156, "back"), (203, 300, "front")])
        v = load_facing_hints(st.workdir, 0, 300)
        self.assertTrue(np.all(v[80:156] == -1) and np.all(v[203:] == 1) and np.isnan(v[:80]).all() and np.isnan(v[156:203]).all())


class RotateAnimation(unittest.TestCase):
    def test_turn_keeps_the_camera_view(self):
        from vroidmocap.core.retarget import rotate_anim, _qmul
        from vroidmocap.core.mathutil import quat_to_mat
        rng = np.random.default_rng(2)
        n = 5
        q = rng.normal(size=(n, 3, 4)); q /= np.linalg.norm(q, axis=-1, keepdims=True)
        cq = rng.normal(size=(n, 4)); cq /= np.linalg.norm(cq, axis=-1, keepdims=True)
        a = {"bones": ["hips", "spine", "head"], "quats": q, "root": rng.normal(size=(n, 3)),
             "cam_pos": rng.normal(size=(n, 3)) * 3, "cam_quat": cq, "frames": n}
        b = rotate_anim(a, 75)
        for k in range(n):
            # hips position and orientation seen from the camera: unchanged
            Ra, Rb = quat_to_mat(a["cam_quat"][k:k + 1])[0], quat_to_mat(b["cam_quat"][k:k + 1])[0]
            pa = Ra.T @ (a["root"][k] - a["cam_pos"][k]); pb = Rb.T @ (b["root"][k] - b["cam_pos"][k])
            self.assertTrue(np.allclose(pa, pb, atol=1e-6))
            Ha, Hb = quat_to_mat(a["quats"][k, :1])[0], quat_to_mat(b["quats"][k, :1])[0]
            self.assertTrue(np.allclose(Ra.T @ Ha, Rb.T @ Hb, atol=1e-6))
        # ... and the facing in the scene turned by 75 deg about the vertical
        Hb = quat_to_mat(b["quats"][0, :1])[0]; Ha = quat_to_mat(a["quats"][0, :1])[0]
        fa, fb = Ha[:, 2], Hb[:, 2]
        ya, yb = np.degrees(np.arctan2(fa[0], fa[2])), np.degrees(np.arctan2(fb[0], fb[2]))
        self.assertAlmostEqual(((yb - ya + 180) % 360) - 180, 75.0, places=4)
        self.assertTrue(np.allclose(b["quats"][:, 1:], a["quats"][:, 1:]))


class HeadTurn(unittest.TestCase):
    def test_saturated_face_turn_is_expanded(self):
        from vroidmocap.core.head import calibrated_yaw
        def face(y0, nod=0.1):
            t = np.radians(y0)
            fwd = np.array([np.sin(t), nod, -np.cos(t)]); fwd /= np.linalg.norm(fwd)
            x = np.cross(np.array([0, -1.0, 0]), fwd); x /= np.linalg.norm(x)
            return np.stack([x, np.cross(fwd, x), fwd], 1)
        yaw = lambda R: np.degrees(np.arctan2(R[0, 2], -R[2, 2]))
        self.assertAlmostEqual(yaw(calibrated_yaw(face(20))), 20 * 33 / 35, delta=0.5)   # small turns: about as fitted
        self.assertGreater(yaw(calibrated_yaw(face(54))), 70)                            # a saturated 54 deg is ~73
        self.assertLess(yaw(calibrated_yaw(face(-54))), -70)
        R = calibrated_yaw(face(50, nod=0.3))
        self.assertTrue(np.allclose(R.T @ R, np.eye(3), atol=1e-9))
        self.assertAlmostEqual(R[1, 2], face(50, nod=0.3)[1, 2], places=9)               # the nod is kept

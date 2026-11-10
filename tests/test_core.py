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


if __name__ == "__main__":
    unittest.main()


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


class HoldUnseen(unittest.TestCase):
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
        # before first sight: the first seen pose
        self.assertTrue(np.allclose(out[:10, 0], Q[10, 0]))
        # unseen: exactly the last pose
        self.assertTrue(np.allclose(out[30:40, 0], out[29, 0]))
        # no frame-to-frame step larger than the motion itself plus the catch-up
        steps = np.degrees(q_angle(out[1:, 0], out[:-1, 0]))
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

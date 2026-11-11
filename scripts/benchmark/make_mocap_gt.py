"""Ground-truth benchmark from real dance mocap (CMU), with shots aimed at the
reported failures. Writes mocap/gt_anim.json (+ per-frame tags) for gt_mocap.html."""
import sys, json, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bvh2vrm import convert, locals_from_world, align
from vroidmocap.core.vrm import load_vrm_skeleton
from vroidmocap.core.retarget import avatar_fk, _foot_lows
from vroidmocap.core.mathutil import quat_to_mat, mat_to_quat

sk = load_vrm_skeleton('avatar.vrm'); bones = sk.bones; bi = {b: i for i, b in enumerate(bones)}
rng = np.random.default_rng(3)
B = len(bones)
Q0 = np.zeros((1, B, 4)); Q0[..., 3] = 1
VREST, _ = avatar_fk(sk, Q0, np.zeros((1, 3))); VREST = VREST[0]


def sm(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


def win(k, a, b, ramp=8):
    return sm((k - a) / ramp) * (1 - sm((k - b) / ramp))


def Ry(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def sph(yaw, pitch, dist, look):
    y, p = np.radians(yaw), np.radians(pitch)
    return look + dist * np.array([np.sin(y) * np.cos(p), np.sin(p), np.cos(y) * np.cos(p)])


def slerp_m(A, Bm, w):
    qa, qb = mat_to_quat(A[None])[0], mat_to_quat(Bm[None])[0]
    if np.dot(qa, qb) < 0: qb = -qb
    q = (1 - w) * qa + w * qb
    return quat_to_mat((q / np.linalg.norm(q))[None])[0]


def noise(n, lo=0.2, hi=0.9, k=4):
    t = np.arange(n) / 30.0; x = np.zeros(n)
    for _ in range(k):
        x += np.sin(2 * np.pi * rng.uniform(lo, hi) * t + rng.uniform(0, 6.3)) * rng.uniform(0.5, 1)
    x = (x - x.min()) / max(np.ptp(x), 1e-9)
    return x


def fingers(Q, n):
    for side, sgn in (('left', -1), ('right', 1)):
        for f in ('Index', 'Middle', 'Ring', 'Little'):
            c = np.clip(noise(n) * 1.15 - 0.1, 0, 1)
            for seg, amp in (('Proximal', 75), ('Intermediate', 90), ('Distal', 55)):
                b = side + f + seg
                a = np.radians(amp * c) * sgn
                Q[:, bi[b]] = np.stack([np.zeros(n), np.zeros(n), np.sin(a / 2), np.cos(a / 2)], 1)
        c = noise(n)
        for seg, amp in (('Metacarpal', 20), ('Proximal', 35), ('Distal', 40)):
            b = side + 'Thumb' + seg
            a = np.radians(amp * c) * (-sgn)
            Q[:, bi[b]] = np.stack([np.zeros(n), np.sin(a / 2), np.zeros(n), np.cos(a / 2)], 1)
    return Q


def point_at(G, root, k, b_up, b_lo, target, w):
    """Straighten a limb towards a target point (blend weight w)."""
    Q = locals_from_world(sk, G, len(root))
    pos, _ = avatar_fk(sk, Q[k:k + 1], root[k:k + 1])
    d = target - pos[0, bi[b_up]]
    for b, ch in ((b_up, b_lo), (b_lo, {'leftLowerArm': 'leftHand', 'rightLowerArm': 'rightHand',
                                        'leftLowerLeg': 'leftFoot', 'rightLowerLeg': 'rightFoot'}[b_lo])):
        pass
    for b, ch in ((b_up, b_lo), (b_lo, {'leftLowerArm': 'leftHand', 'rightLowerArm': 'rightHand',
                                        'leftLowerLeg': 'leftFoot', 'rightLowerLeg': 'rightFoot'}[b_lo])):
        rest_dir = VREST[bi[ch]] - VREST[bi[b]]
        cur = G[b][k] @ rest_dir
        Gn = align(cur, d) @ G[b][k]
        G[b][k] = slerp_m(G[b][k], Gn, w)


shots = []


def shot(name, take, s, n, cam_fn, spin=None, points=(), occ=None, tags=None, travel=0.35):
    G, root, dt = convert(sk, f'mocap/{take}.bvh')
    s += 90  # the takes start with a calibration T-pose
    e = min(s + n, len(root)); n = e - s
    G = {b: g[s:e].copy() for b, g in G.items()}; root = root[s:e].copy()
    root[:, [0, 2]] -= root[:, [0, 2]].mean(0)
    root[:, [0, 2]] *= travel
    # face the camera side (+Z) on average
    fw = G['hips'][:, :, 2].mean(0); yaw = np.arctan2(fw[0], fw[2])
    R0 = Ry(-yaw)
    for b in G: G[b] = np.einsum('ij,njk->nik', R0, G[b])
    root = root @ R0.T
    if spin is not None:
        th = spin(np.arange(n))
        c = root[0].copy()
        for k in range(n):
            R = Ry(th[k])
            for b in G: G[b][k] = R @ G[b][k]
            root[k] = c + R @ (root[k] - c)
    rs = root.copy()
    for _ in range(3): rs[1:-1] = (rs[:-2] + rs[1:-1] + rs[2:]) / 3
    cams = [cam_fn(k, rs[k]) if cam_fn.__code__.co_argcount == 2 else cam_fn(k) for k in range(n)]
    for (b_up, b_lo, a, bb) in points:
        for k in range(max(0, a - 10), min(n, bb + 10)):
            w = win(k, a, bb)
            if w > 0: point_at(G, root, k, b_up, b_lo, np.array(cams[k][0]), w)
    Q = locals_from_world(sk, G, n)
    Q = fingers(Q, n)
    # floor: the lowest sole point touches y = 0 in most frames
    lows, _ = _foot_lows(sk, Q, root)   # heel and toe of the sole (the avatar's own foot geometry)
    root[:, 1] -= np.percentile(lows.min(1), 10)
    out = []
    for k in range(n):
        p, t, f = cams[k]; out.append([*np.round(p, 4), *np.round(t, 4), round(float(f), 3)])
    shots.append(dict(name=name, quats=np.round(Q, 5).tolist(), root=np.round(root, 4).tolist(), camera=out,
                      occluders=occ or [], tags=tags or {}))
    print(name, take, n)


L = np.array([0, 0.9, 0])
# 1 plain front view: visible from the first frame
shot('front', '05_02', 0, 280, lambda k: (sph(0, 5, 4.2, L), L, 35))
# 2 spins: real dance + two fast full turns and a slow half turn
def spin2(k):
    t = k / 30.0
    return np.radians(360 * sm((t - 2) / 1.0) + 360 * sm((t - 5) / 0.9) + 180 * sm((t - 7.5) / 2))
shot('spins', '55_01', 0, 300, lambda k: (sph(15, 10, 4.2, L), L, 35), spin=spin2)
# 3 close camera; arms and a leg reach straight at the camera
L3 = np.array([0, 1.15, 0])
def cam3(k, r):
    look = L3 + np.array([r[0], 0, r[2]]) * 0.8
    return sph(0, 3, 2.7, L3), look, 42
shot('point', '94_01', 0, 300, cam3,
     points=[('leftUpperArm', 'leftLowerArm', 40, 85), ('rightUpperArm', 'rightLowerArm', 120, 165),
             ('leftUpperArm', 'leftLowerArm', 200, 245), ('rightUpperArm', 'rightLowerArm', 200, 245),
             ('rightUpperLeg', 'rightLowerLeg', 262, 292)],
     tags=dict(point=[['leftUpperArm', 40, 85], ['rightUpperArm', 120, 165], ['leftUpperArm', 200, 245],
                      ['rightUpperArm', 200, 245], ['rightUpperLeg', 262, 292]]))
# 4 something passes quickly in front of the legs (twice)
def occ_path(n, k0, x0, v):
    return [[round(x0 + v * (k - k0), 3), 0.5, 1.4] for k in range(n)]
shot('pass', '60_08', 0, 300, lambda k: (sph(0, 5, 4.4, L), L, 35),
     occ=[dict(size=[0.35, 1.0, 0.2], path=occ_path(300, 90, -1.0, 0.12)),
          dict(size=[0.5, 1.0, 0.2], path=occ_path(300, 210, 1.0, -0.15))])
# 5 side view: the legs cross over in the picture
shot('side', '05_06', 0, 222, lambda k: (sph(75, 6, 4.2, L), L, 35))
# 6 low camera at the feet: floor contact
L6 = np.array([0, 0.7, 0])
def cam6(k, r):
    return sph(-20, -2, 3.6, L6) * np.array([1, 0, 1]) + np.array([0, 0.25, 0]), L6 + np.array([r[0], 0, r[2]]) * 0.8, 40
shot('low', '94_01', 400, 300, cam6)
json.dump(dict(bones=bones, fps=30, shots=shots), open('mocap/gt_anim.json', 'w'))
print('frames', sum(len(s['root']) for s in shots))

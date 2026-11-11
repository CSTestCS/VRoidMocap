"""CMU BVH -> VRM normalized humanoid local quaternions (x,y,z,w) + hips position.

World rotation of each VRM bone = (BVH joint world rotation, axes converted) * C,
where C turns the VRM rest bone direction onto the BVH rest bone direction, so
every bone points where the mocap bone points (twist follows the mocap)."""
import re, sys, numpy as np
sys.path.insert(0, '/home/user/VRoidMocap')
from vroidmocap.core.mathutil import quat_to_mat, mat_to_quat
from vroidmocap.core.retarget import avatar_fk


def parse_bvh(path):
    txt = open(path).read().split('MOTION')
    names, parent, offset, chans = [], [], [], []
    stack = []
    toks = txt[0].split()
    i = 0
    cur = None
    while i < len(toks):
        t = toks[i]
        if t in ('ROOT', 'JOINT'):
            names.append(toks[i + 1]); parent.append(stack[-1] if stack else -1); offset.append(None); chans.append([])
            cur = len(names) - 1; i += 2
        elif t == 'End':
            names.append(names[cur] + '_end'); parent.append(cur); offset.append(None); chans.append([])
            cur = len(names) - 1; i += 2
        elif t == '{':
            stack.append(cur); i += 1
        elif t == '}':
            stack.pop(); cur = stack[-1] if stack else None; i += 1
        elif t == 'OFFSET':
            offset[cur] = np.array([float(x) for x in toks[i + 1:i + 4]]); i += 4
        elif t == 'CHANNELS':
            n = int(toks[i + 1]); chans[cur] = toks[i + 2:i + 2 + n]; i += 2 + n
        else:
            i += 1
    lines = txt[1].strip().split('\n')
    ft = float(lines[1].split(':')[1])
    data = np.array([[float(x) for x in l.split()] for l in lines[2:] if l.strip()])
    return names, np.array(parent), np.array(offset), chans, ft, data


def rot_axis(ax, deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    n = len(a); R = np.zeros((n, 3, 3)); R[:, 0, 0] = R[:, 1, 1] = R[:, 2, 2] = 1
    i, j = {'X': (1, 2), 'Y': (2, 0), 'Z': (0, 1)}[ax]
    R[:, i, i] = c; R[:, j, j] = c; R[:, i, j] = -s; R[:, j, i] = s
    return R


def fk_bvh(path, step=4):
    names, par, off, chans, ft, data = parse_bvh(path)
    data = data[::step]
    N, J = len(data), len(names)
    W = np.zeros((N, J, 3, 3)); P = np.zeros((N, J, 3))
    col = 0
    for j in range(J):
        L = np.tile(np.eye(3), (N, 1, 1)); t = np.zeros((N, 3))
        for c in chans[j]:
            v = data[:, col]; col += 1
            if c.endswith('position'):
                t[:, 'XYZ'.index(c[0])] = v
            else:
                L = L @ rot_axis(c[0], v)
        if par[j] < 0:
            W[:, j] = L; P[:, j] = t + off[j]
        else:
            W[:, j] = W[:, par[j]] @ L; P[:, j] = P[:, par[j]] + np.einsum('nij,j->ni', W[:, par[j]], off[j])
    rest = np.zeros((J, 3))
    for j in range(J):
        rest[j] = off[j] + (rest[par[j]] if par[j] >= 0 else 0)
    return names, par, off, W, P, rest, ft * step


MAP = {'hips': 'Hips', 'spine': 'LowerBack', 'chest': 'Spine', 'upperChest': 'Spine1', 'neck': 'Neck1', 'head': 'Head',
       'leftShoulder': 'LeftShoulder', 'leftUpperArm': 'LeftArm', 'leftLowerArm': 'LeftForeArm', 'leftHand': 'LeftHand',
       'leftUpperLeg': 'LeftUpLeg', 'leftLowerLeg': 'LeftLeg', 'leftFoot': 'LeftFoot', 'leftToes': 'LeftToeBase'}
MAP.update({k.replace('left', 'right'): v.replace('Left', 'Right') for k, v in MAP.items() if k.startswith('left')})
# bone direction pairs (VRM child, BVH child)
VCHILD = {'spine': 'chest', 'chest': 'upperChest', 'upperChest': 'neck', 'neck': 'head',
          'leftShoulder': 'leftUpperArm', 'leftUpperArm': 'leftLowerArm', 'leftLowerArm': 'leftHand', 'leftHand': 'leftMiddleProximal',
          'leftUpperLeg': 'leftLowerLeg', 'leftLowerLeg': 'leftFoot', 'leftFoot': 'leftToes'}
VCHILD.update({k.replace('left', 'right'): v.replace('left', 'right') for k, v in VCHILD.items() if k.startswith('left')})
BCHILD = {'LowerBack': 'Spine', 'Spine': 'Spine1', 'Spine1': 'Neck1', 'Neck1': 'Head',
          'LeftShoulder': 'LeftArm', 'LeftArm': 'LeftForeArm', 'LeftForeArm': 'LeftHand', 'LeftHand': 'LeftHandIndex1',
          'LeftUpLeg': 'LeftLeg', 'LeftLeg': 'LeftFoot', 'LeftFoot': 'LeftToeBase'}
BCHILD.update({k.replace('Left', 'Right'): v.replace('Left', 'Right') for k, v in BCHILD.items() if k.startswith('Left')})


def align(a, b):
    """Minimal rotation taking unit vector a onto b."""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(np.dot(a, b))
    if c < -0.9999:
        ax = np.cross(a, [1, 0, 0]);
        if np.linalg.norm(ax) < 1e-6: ax = np.cross(a, [0, 1, 0])
        ax /= np.linalg.norm(ax); return 2 * np.outer(ax, ax) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def convert(skel, path, step=4):
    names, par, off, W, P, rest, dt = fk_bvh(path, step)
    ix = {n: i for i, n in enumerate(names)}
    B = len(skel.bones); N = len(W)
    Q0 = np.zeros((1, B, 4)); Q0[..., 3] = 1
    vrest, _ = avatar_fk(skel, Q0, np.zeros((1, 3))); vrest = vrest[0]
    bi = {b: i for i, b in enumerate(skel.bones)}
    # axes: does the BVH actor face +Z with left at +X like the avatar?
    lx = rest[ix['LeftUpLeg']][0] - rest[ix['RightUpLeg']][0]
    M = np.eye(3) if lx > 0 else np.diag([-1., 1, -1])
    G = {}
    for vb, bj in MAP.items():
        if vb not in bi: continue
        C = np.eye(3)
        if vb in VCHILD and VCHILD[vb] in bi and bj in BCHILD:
            dv = vrest[bi[VCHILD[vb]]] - vrest[bi[vb]]
            db = M @ (rest[ix[BCHILD[bj]]] - rest[ix[bj]])
            C = align(dv, db)
        G[vb] = np.einsum('ij,njk,lk->nil', M, W[:, ix[bj]], M) @ C
    # hips position: scale by leg length
    legv = np.linalg.norm(vrest[bi['leftUpperLeg']] - vrest[bi['leftFoot']])
    legb = np.linalg.norm(rest[ix['LeftUpLeg']] - rest[ix['LeftFoot']])
    s = legv / legb
    root = np.einsum('ij,nj->ni', M, P[:, ix['Hips']]) * s
    return G, root, dt


def locals_from_world(skel, G, N):
    B = len(skel.bones); Q = np.zeros((N, B, 4)); Q[..., 3] = 1
    Gw = {}
    for b in skel.bones:
        p = skel.parent[b]
        pw = Gw.get(p, np.tile(np.eye(3), (N, 1, 1))) if p is not None else np.tile(np.eye(3), (N, 1, 1))
        if b in G:
            Gw[b] = G[b]
            L = np.einsum('nji,njk->nik', pw, G[b])
            Q[:, skel.bones.index(b)] = mat_to_quat(L)
        else:
            Gw[b] = pw @ quat_to_mat(Q[:, skel.bones.index(b)])
    return Q

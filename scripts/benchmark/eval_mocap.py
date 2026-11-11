"""Score a solved animation of the mocap benchmark, one number per reported issue.
usage: python eval_mocap.py <ws> [--short]"""
import os, sys, json, numpy as np
sys.path.insert(0, '/home/user/VRoidMocap')
from vroidmocap.core.vrm import load_vrm_skeleton
from vroidmocap.core.retarget import avatar_fk
from vroidmocap.core.mathutil import quat_to_mat

ws = sys.argv[1]
D = os.path.dirname(os.path.abspath(__file__)) + '/'
gt = json.load(open(D + 'mocap/gt.json')); GA = json.load(open(D + 'mocap/gt_anim.json'))
a = np.load(ws + '/anim_c0.npz', allow_pickle=True); bones = json.loads(str(a['_meta']))['bones']
bi = {b: i for i, b in enumerate(bones)}
sk = load_vrm_skeleton(D + 'avatar.vrm')
N = min(len(gt), len(a['quats']))
pos, rot = avatar_fk(sk, a['quats'][:N].astype(float), a['root'][:N].astype(float))
gp = np.array([[p if p else [np.nan] * 3 for p in g['pos']] for g in gt[:N]], float)
gr = np.array([[p if p else [0, 0, 0, 1] for p in g['rot']] for g in gt[:N]], float)
gc = np.array([g['cam'] for g in gt[:N]]); shot = np.array([g['shot'] for g in gt[:N]])
names = [s['name'] for s in GA['shots']]
k_in = np.zeros(N, int); c = {}
for i in range(N): k_in[i] = c.get(shot[i], 0); c[shot[i]] = k_in[i] + 1
Rg = quat_to_mat(gc[:, 3:7]); Rs = quat_to_mat(a['cam_quat'][:N].astype(float))
def tocam(P, R, cc): return np.einsum('nji,nbj->nbi', R, P - cc[:, None])
Pg = tocam(gp, Rg, gc[:, :3]); Ps = tocam(pos, Rs, a['cam_pos'][:N].astype(float))
Gg = np.einsum('nji,nbjk->nbik', Rg, quat_to_mat(gr.reshape(-1, 4)).reshape(N, -1, 3, 3))
Gs = np.einsum('nji,nbjk->nbik', Rs, rot)
def ang(A, B):
    tr = np.einsum('...ij,...ij->...', A, B); return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))
def vang(u, v):
    c_ = np.sum(u * v, -1) / np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-9)
    return np.degrees(np.arccos(np.clip(c_, -1, 1)))
LIMBS = {'L upper arm': ('leftUpperArm', 'leftLowerArm'), 'R upper arm': ('rightUpperArm', 'rightLowerArm'),
         'L forearm': ('leftLowerArm', 'leftHand'), 'R forearm': ('rightLowerArm', 'rightHand'),
         'L thigh': ('leftUpperLeg', 'leftLowerLeg'), 'R thigh': ('rightUpperLeg', 'rightLowerLeg'),
         'L shin': ('leftLowerLeg', 'leftFoot'), 'R shin': ('rightLowerLeg', 'rightFoot')}
le = {k: vang(Ps[:, bi[c_]] - Ps[:, bi[p]], Pg[:, bi[c_]] - Pg[:, bi[p]]) for k, (p, c_) in LIMBS.items()}
LE = np.stack(list(le.values()), 1)  # N x 8
arms, legs = LE[:, :4], LE[:, 4:]
out = {}
def put(k, v): out[k] = v

# 1 limbs overall
put('limb dir err median', np.median(LE)); put('limb dir err p90', np.percentile(LE, 90)); put('limbs >45deg %', 100 * (LE > 45).mean())
# 2 body angle
put('hips orientation err median', np.median(ang(Gs[:, bi['hips']], Gg[:, bi['hips']])))
put('chest orientation err median', np.median(ang(Gs[:, bi['upperChest']], Gg[:, bi['upperChest']])))
upS = Gs[:, bi['upperChest']][:, :, 1]; upG = Gg[:, bi['upperChest']][:, :, 1]
put('torso lean err median', np.median(vang(upS, upG)))
# 3 spine / neck / head relative motion
def rel(G, a_, b_): return np.einsum('nji,njk->nik', G[:, bi[a_]], G[:, bi[b_]])
for nm, (p, c_) in {'chest vs hips': ('hips', 'upperChest'), 'head vs chest': ('upperChest', 'head')}.items():
    Rs_, Rg_ = rel(Gs, p, c_), rel(Gg, p, c_)
    put(nm + ' err median', np.median(ang(Rs_, Rg_)))
    I = np.eye(3)
    ms, mg = ang(Rs_, np.broadcast_to(np.median(Rs_, 0), Rs_.shape)), ang(Rg_, np.broadcast_to(np.median(Rg_, 0), Rg_.shape))
    put(nm + ' motion ratio', np.std(ms) / max(np.std(mg), 1e-6))
# head facing (yaw of head forward in camera frame)
fS = Gs[:, bi['head']][:, :, 2]; fG = Gg[:, bi['head']][:, :, 2]
put('head facing err median', np.median(vang(fS, fG)))
# 4 late start: frames from shot start until limbs are within 25 deg (median of 8)
delays = []
for s in range(len(names)):
    idx = np.where(shot == s)[0]
    m = np.median(LE[idx], 1)
    ok = np.where(m < 25)[0]
    delays.append(int(ok[0]) if len(ok) else len(idx))
put('start delay frames (per shot)', delays)
# 5 pointing at the camera
pe = []
for s, nm in enumerate(names):
    for b, k0, k1 in GA['shots'][s]['tags'].get('point', []):
        idx = np.where((shot == s) & (k_in >= k0 + 8) & (k_in <= k1 - 8))[0]
        key = {'leftUpperArm': ['L upper arm', 'L forearm'], 'rightUpperArm': ['R upper arm', 'R forearm'],
               'rightUpperLeg': ['R thigh', 'R shin']}[b]
        for kk in key: pe.append(le[kk][idx])
pe = np.concatenate(pe); put('pointing at camera: dir err median', np.median(pe)); put('pointing: >45deg %', 100 * (pe > 45).mean())
# 6 spins: facing (hips forward, horizontal) error
si = names.index('spins'); idx = shot == si
hf = lambda G: G[:, bi['hips']][:, :, 2]
put('spins: hips facing err median', np.median(vang(hf(Gs)[idx], hf(Gg)[idx]))); put('spins: facing off >90deg %', 100 * (vang(hf(Gs)[idx], hf(Gg)[idx]) > 90).mean())
# 7 occluder passing: legs during frames the box covers a leg in the picture
oi = names.index('pass'); idx = np.where(shot == oi)[0]
f_ = (1080 / 2) / np.tan(np.radians(gc[:, 7]) / 2)
def proj(P): return np.stack([P[..., 0] / -P[..., 2] * f_[:, None] + 960, -P[..., 1] / -P[..., 2] * f_[:, None] + 540], -1)
ug = proj(Pg)
cov = np.zeros(N, bool)
for o in GA['shots'][oi]['occluders']:
    for i in idx:
        c3 = np.array(o['path'][k_in[i]]); hs = np.array(o['size']) / 2
        corners = np.array([[c3[0] + sx * hs[0], c3[1] + sy * hs[1], c3[2] + hs[2]] for sx in (-1, 1) for sy in (-1, 1)])
        pc = np.einsum('ji,nj->ni', Rg[i], corners - gc[i, :3]); uc = np.stack([pc[:, 0] / -pc[:, 2] * f_[i] + 960, -pc[:, 1] / -pc[:, 2] * f_[i] + 540], 1)
        x0, x1, y0, y1 = uc[:, 0].min(), uc[:, 0].max(), uc[:, 1].min(), uc[:, 1].max()
        for b in ('leftLowerLeg', 'rightLowerLeg', 'leftFoot', 'rightFoot'):
            u = ug[i, bi[b]]
            if x0 <= u[0] <= x1 and y0 <= u[1] <= y1: cov[i] = True
around = np.zeros(N, bool)
for i in np.where(cov)[0]: around[max(0, i - 20):i + 21] = True
put('passing object: legs err while covered', np.median(legs[cov]) if cov.any() else np.nan)
put('passing object: legs err just before/after', np.median(legs[around & ~cov & (shot == oi)]))
# 8 crossed legs (side view): ankles in the other order than the hips in the picture
si = names.index('side'); idx = shot == si
cross = idx & (np.sign(ug[:, bi['leftFoot'], 0] - ug[:, bi['rightFoot'], 0]) != np.sign(ug[:, bi['leftUpperLeg'], 0] - ug[:, bi['rightUpperLeg'], 0]))
put('crossed legs: legs err', np.median(legs[cross])); put('crossed legs: >45deg %', 100 * (legs[cross] > 45).mean())
put('side view, uncrossed: legs err', np.median(legs[idx & ~cross]))
# 9 feet on the floor (world): sole height while the GT foot is planted
from vroidmocap.core.retarget import _foot_lows
gq = np.array([[q if q else [0, 0, 0, 1] for q in s_['quats'][k]] for s_ in GA['shots'] for k in range(len(s_['root']))], float)[:N]
grt = np.array([r for s_ in GA['shots'] for r in s_['root']], float)[:N]
LG, _ = _foot_lows(sk, gq, grt); LS, _ = _foot_lows(sk, a['quats'][:N].astype(float), a['root'][:N].astype(float))
fh = []
for j, side in enumerate(('left', 'right')):
    hg, hs_ = LG[:, j], LS[:, j]
    vel = np.r_[0, np.linalg.norm(np.diff(gp[:, bi[side + 'Foot']], axis=0), axis=1)]
    planted = (hg < 0.03) & (vel < 0.01)
    fh.append(hs_[planted] - hg[planted])
fh = np.concatenate(fh) * 100
put('planted feet: height err median |cm|', np.median(np.abs(fh))); put('planted feet: sinking >3cm %', 100 * (fh < -3).mean()); put('planted feet: floating >3cm %', 100 * (fh > 3).mean())
# 10 hands: hand orientation and fingers relative to the hand
for side in ('left', 'right'):
    h = bi[side + 'Hand']
    put(f'{side} hand orientation err median', np.median(ang(Gs[:, h], Gg[:, h])))
    fx = [bi[b] for b in bones if b.startswith(side) and any(k in b for k in ('Index', 'Middle', 'Ring', 'Little')) and 'Proximal' in b]
    ls = np.einsum('nji,nbjk->nbik', Gs[:, h], Gs[:, fx]); lg = np.einsum('nji,nbjk->nbik', Gg[:, h], Gg[:, fx])
    put(f'{side} finger curl err median', np.median(ang(ls, lg)))
# 11 flicker: a limb jumps >20 deg between frames while the true limb moves <5 deg
jump = 0; tot = 0
for k, (p, c_) in LIMBS.items():
    ds = Ps[:, bi[c_]] - Ps[:, bi[p]]; dg = Pg[:, bi[c_]] - Pg[:, bi[p]]
    same = shot[1:] == shot[:-1]
    js_ = vang(ds[1:], ds[:-1]); jg = vang(dg[1:], dg[:-1])
    jump += ((js_ > 20) & (jg < 5) & same).sum(); tot += same.sum()
put('flicker: limb jumps per 100 frames', 100 * jump / tot)
# per shot limb error
put('limb err median per shot', {nm: round(float(np.median(LE[shot == s])), 1) for s, nm in enumerate(names)})
for k, v in out.items():
    print(f'{k:45s}', np.round(v, 1) if not isinstance(v, (list, dict)) else v)
json.dump({k: (v if isinstance(v, (list, dict)) else float(v)) for k, v in out.items()}, open(ws + '/score.json', 'w'))
# geometric body angle (definition-free): hip line, shoulder line, spine (hips -> neck)
def gvec(P, a_, b_): return P[:, bi[b_]] - P[:, bi[a_]]
for nm, (a_, b_) in {'hip line': ('rightUpperLeg', 'leftUpperLeg'), 'shoulder line': ('rightUpperArm', 'leftUpperArm'),
                     'spine hips->neck': ('hips', 'neck'), 'neck->head': ('neck', 'head')}.items():
    e = vang(gvec(Ps, a_, b_), gvec(Pg, a_, b_))
    print(f'{"geom " + nm + " err median/p90":45s}', np.round([np.median(e), np.percentile(e, 90)], 1),
          ' per shot', [round(float(np.median(e[shot == s])), 1) for s in range(len(names))])
# signed bias of hips in camera frame: yaw / pitch / roll of the error rotation
from scipy.spatial.transform import Rotation as Rot
E = np.einsum('nji,njk->nik', Gg[:, bi['hips']], Gs[:, bi['hips']])
eu = Rot.from_matrix(E).as_euler('YXZ', degrees=True)
print('hips error (in hips frame) yaw/pitch/roll median', np.round(np.median(eu, 0), 1), 'abs median', np.round(np.median(np.abs(eu), 0), 1))
E = np.einsum('nji,njk->nik', Gg[:, bi['head']], Gs[:, bi['head']])
eu = Rot.from_matrix(E).as_euler('YXZ', degrees=True)
print('head error yaw/pitch/roll median', np.round(np.median(eu, 0), 1), 'abs median', np.round(np.median(np.abs(eu), 0), 1))

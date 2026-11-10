"""Ground truth with the whole body turned: a slow full turn, then side / back
views and two fast spins, on top of a solved dance (anim_c0.npz).
usage: python make_turn_gt.py path/to/anim_c0.npz   -> gt_anim_turn.json"""
import json, sys, numpy as np
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vroidmocap.core.mathutil import quat_to_mat, mat_to_quat
a=np.load(sys.argv[1],allow_pickle=True); meta=json.loads(str(a['_meta'])); bones=meta['bones']
Q=a['quats'].astype(float); R=a['root'].astype(float); hi=bones.index('hips')
def yaw_profile(kind,n):
    t=np.arange(n)/30.0
    if kind=='slow':      # 0 -> 360 deg over the shot, constant speed
        return np.radians(360*t/t[-1])
    # hold side 2 s, turn to back 0.5 s, hold back 2 s, two 1 s spins, settle front
    y=np.zeros(n)
    for i,tt in enumerate(t):
        if tt<0.5: v=90*tt/0.5
        elif tt<2.5: v=90
        elif tt<3.0: v=90+90*(tt-2.5)/0.5
        elif tt<5.0: v=180
        elif tt<6.0: v=180+360*(tt-5.0)
        elif tt<6.5: v=540
        elif tt<7.5: v=540+360*(tt-6.5)
        elif tt<8.0: v=900+(tt-7.5)*2*180
        else: v=1080
        y[i]=v
    return np.radians(y)
shots=[dict(name='turn_slow',s=2000,cam='static',yaw=0,pitch=6,dist=3.3,fov=35,kind='slow'),
       dict(name='spins',s=7000,cam='static',yaw=15,pitch=14,dist=3.5,fov=35,kind='spins')]
out=dict(bones=bones,fps=30,shots=[])
n=300
for sh in shots:
    idx=np.arange(sh['s'],sh['s']+2*n,2)
    q=Q[idx].copy(); r=R[idx].copy(); r[:,[0,2]]-=r[0,[0,2]]
    r[:,[0,2]]*=0.5   # keep the dancer near the middle while turning
    y=yaw_profile(sh['kind'],n)
    for k in range(n):
        c,s_=np.cos(y[k]),np.sin(y[k]); Ry=np.array([[c,0,s_],[0,1,0],[-s_,0,c]])
        q[k,hi]=mat_to_quat(Ry@quat_to_mat(q[k,hi])); r[k]=Ry@r[k]*np.array([1,1,1])
    d=dict(sh); d['quats']=np.round(q,5).tolist(); d['root']=np.round(r,4).tolist(); d['yaw_added']=np.degrees(y).round(2).tolist(); out['shots'].append(d)
json.dump(out,open('gt_anim_turn.json','w')); print('frames',n*len(shots))

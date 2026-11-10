"""Ground-truth motion for the benchmark: takes any solved animation (anim_c0.npz
from a work folder) as the *known* motion and defines four shots with known cameras.
usage: python make_gt.py path/to/anim_c0.npz"""
import json
import sys

import numpy as np

a=np.load(sys.argv[1],allow_pickle=True)
meta=json.loads(str(a['_meta'])); bones=meta['bones']
Q=a['quats']; R=a['root']
shots=[dict(name='front_static',s=600,e=1200,cam='static',yaw=0,pitch=5,dist=3.1,fov=35),
       dict(name='spin_high',s=3400,e=3900,cam='static',yaw=25,pitch=18,dist=3.4,fov=35),
       dict(name='orbit',s=5000,e=5600,cam='orbit',yaw=-30,pitch=8,dist=3.6,fov=32),
       dict(name='closeup',s=8000,e=8400,cam='static',yaw=-10,pitch=0,dist=1.6,fov=35,look=1.2)]
out=dict(bones=bones,fps=30,shots=[])
for sh in shots:
    idx=np.arange(sh['s'],sh['e'],2)
    r=R[idx].copy(); r[:,[0,2]]-=r[0,[0,2]]
    d=dict(sh); d['quats']=np.round(Q[idx],5).tolist(); d['root']=np.round(r,4).tolist(); out['shots'].append(d)
json.dump(out,open('gt_anim.json','w'))
print(sum(len(s['root']) for s in out['shots']))

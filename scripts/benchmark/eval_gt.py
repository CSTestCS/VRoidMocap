"""3D accuracy of a solved animation against the rendered ground truth.
usage: python eval_gt.py <work folder with anim_c0.npz> <gt.json> <avatar.vrm>"""
import sys, json, numpy as np
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vroidmocap.core.vrm import load_vrm_skeleton
from vroidmocap.core.retarget import avatar_fk
from vroidmocap.core.mathutil import quat_to_mat
ws=sys.argv[1]; gt=json.load(open(sys.argv[2] if len(sys.argv)>2 else 'gt.json'))
a=np.load(ws+'/anim_c0.npz',allow_pickle=True); meta=json.loads(str(a['_meta'])); bones=meta['bones']
skel=load_vrm_skeleton(sys.argv[3])
N=min(len(gt),len(a['quats']))
pos,rot=avatar_fk(skel,a['quats'][:N].astype(float),a['root'][:N].astype(float))
gp=np.array([[p if p else [np.nan]*3 for p in g['pos']] for g in gt[:N]],float)
gr=np.array([[p if p else [0,0,0,1] for p in g['rot']] for g in gt[:N]],float)
gc=np.array([g['cam'] for g in gt[:N]]); shot=np.array([g['shot'] for g in gt[:N]])
# camera frames: three camera looks down -Z, world->cam = R^T (p - c)
Rg=quat_to_mat(gc[:,3:7]); Rs=quat_to_mat(a['cam_quat'][:N].astype(float))
def tocam(P,R,c): return np.einsum('nji,nbj->nbi',R,P-c[:,None])
Pg=tocam(gp,Rg,gc[:,:3]); Ps=tocam(pos,Rs,a['cam_pos'][:N].astype(float))
bi={b:i for i,b in enumerate(bones)}
hi=bi['hips']
valid=a['valid'][:N]
body=[b for b in bones if not any(k in b for k in ('Thumb','Index','Middle','Ring','Little'))]
fing=[b for b in bones if b not in body]
groups={'torso':['spine','chest','upperChest','neck','head'],'arms':['leftUpperArm','rightUpperArm','leftLowerArm','rightLowerArm','leftHand','rightHand'],
 'legs':['leftUpperLeg','rightUpperLeg','leftLowerLeg','rightLowerLeg','leftFoot','rightFoot','leftToes','rightToes']}
def rel(P): return P-P[:,[hi]]
E=rel(Ps)-rel(Pg)  # camera frame, root relative (m)
def stats(x): x=x[np.isfinite(x)]; return f"mean {x.mean():6.2f} med {np.median(x):6.2f} p90 {np.percentile(x,90):6.2f}"
names=['all']+[f"shot{s}" for s in sorted(set(shot))]
for nm in names:
  if (valid if nm=="all" else valid&(shot==int(nm[4:]))).sum()==0: continue
  if True:
    m=valid if nm=='all' else valid&(shot==int(nm[4:]))
    print(f"--- {nm} ({m.sum()} frames)")
    bidx=[bi[b] for b in body if b in bi]
    e=E[m][:,bidx]
    print("  body MPJPE cm      ",stats(np.linalg.norm(e,axis=-1)*100))
    print("    image-plane cm   ",stats(np.linalg.norm(e[...,:2],axis=-1)*100))
    print("    depth cm         ",stats(np.abs(e[...,2])*100))
    for g,bl in groups.items():
        ix=[bi[b] for b in bl if b in bi]
        print(f"    {g:6s} MPJPE cm  ",stats(np.linalg.norm(E[m][:,ix],axis=-1)*100))
    # bone direction error in camera frame
    de=[]
    for b in body:
        p=skel.parent[b]
        if p is None or b not in bi or p not in bi: continue
        ds=Ps[m,bi[b]]-Ps[m,bi[p]]; dg=Pg[m,bi[b]]-Pg[m,bi[p]]
        c=np.sum(ds*dg,-1)/np.maximum(np.linalg.norm(ds,axis=-1)*np.linalg.norm(dg,axis=-1),1e-9)
        de.append(np.degrees(np.arccos(np.clip(c,-1,1))))
    print("  bone direction deg ",stats(np.array(de)))
    LB=[('leftUpperArm','leftLowerArm'),('rightUpperArm','rightLowerArm'),('leftLowerArm','leftHand'),('rightLowerArm','rightHand'),('leftUpperLeg','leftLowerLeg'),('rightUpperLeg','rightLowerLeg'),('leftLowerLeg','leftFoot'),('rightLowerLeg','rightFoot'),('leftFoot','leftToes'),('rightFoot','rightToes'),('hips','neck'),('neck','head')]
    le=[]
    fo=(1080/2)/np.tan(np.radians(gc[:,7])/2); ug=np.stack([Pg[...,0]/-Pg[...,2]*fo[:,None]+960, -Pg[...,1]/-Pg[...,2]*fo[:,None]+540],-1)
    inf=(ug[...,0]>0)&(ug[...,0]<1920)&(ug[...,1]>0)&(ug[...,1]<1080)
    for p_,c_ in LB:
        ds=Ps[m,bi[c_]]-Ps[m,bi[p_]]; dg=Pg[m,bi[c_]]-Pg[m,bi[p_]]
        c=np.sum(ds*dg,-1)/np.maximum(np.linalg.norm(ds,axis=-1)*np.linalg.norm(dg,axis=-1),1e-9); e_=np.degrees(np.arccos(np.clip(c,-1,1)))
        e_[~(inf[m,bi[c_]]&inf[m,bi[p_]])]=np.nan; le.append(e_)
    le=np.array(le); print("  LIMBS (in frame) deg",stats(le[:8]),' per: '+' '.join(f"{x:.1f}" for x in np.nanmean(le,1)), ' >45deg: %.1f%%'%(np.nanmean(le[:8]>45)*100))
    # global bone orientation error in camera frame (includes twist)
    Gs=np.einsum('nji,nbjk->nbik',Rs,rot)[m]; Gg=np.einsum('nji,nbjk->nbik',Rg,quat_to_mat(gr.reshape(-1,4)).reshape(N,-1,3,3))[m]
    def ang(A,B): tr=np.einsum('...ij,...ij->...',A,B); return np.degrees(np.arccos(np.clip((tr-1)/2,-1,1)))
    print("  body orientation deg",stats(ang(Gs[:,bidx],Gg[:,bidx])))
    print("  hips orientation deg",stats(ang(Gs[:,hi],Gg[:,hi])))
    fidx=[bi[b] for b in fing]
    # fingers: local rotation relative to hand
    for side in ('left','right'):
        h=bi[side+'Hand']; fx=[bi[b] for b in fing if b.startswith(side)]
        ls=np.einsum('nji,nbjk->nbik',Gs[:,h],Gs[:,fx]); lg=np.einsum('nji,nbjk->nbik',Gg[:,h],Gg[:,fx])
        print(f"  {side} fingers (rel. hand) deg",stats(ang(ls,lg)))
    # camera
    fe=a['cam_vfov'][:N][m]-gc[m,7]
    up_s=Rs[m][:,1,:]; up_g=Rg[m][:,1,:]   # world up expressed in camera frame = row 1 of R (R^T * (0,1,0))
    c=np.sum(up_s*up_g,-1); print("  camera tilt (pitch+roll) err deg",stats(np.degrees(np.arccos(np.clip(c,-1,1)))), " fov err deg",stats(np.abs(fe)))
    d_s=np.linalg.norm(Ps[m,hi],axis=-1); d_g=np.linalg.norm(Pg[m,hi],axis=-1)
    print("  hips distance err %",stats(np.abs(d_s/d_g-1)*100))

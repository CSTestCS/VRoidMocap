# Ground-truth benchmark

Measures real 3D accuracy: an avatar is rendered with a *known* animation and
known cameras, the app captures the rendered video, and the result is compared
with the known motion bone by bone.

```
# 1. known motion: any animation the app has solved (work folder file)
python make_gt.py path/to/anim_c0.npz              # -> gt_anim.json
# 2. render it (needs node + playwright); serve this folder with the viewer libs
cp path/to/avatar.vrm avatar.vrm
ln -s ../../vroidmocap/web/vendor vendor
python -m http.server 8799 &
node render.js                                     # -> frames/*.jpg, gt.json
ffmpeg -framerate 30 -i frames/f_%05d.jpg -c:v libx264 -crf 14 -pix_fmt yuv420p bench.mp4
# 3. run the app on bench.mp4 (CLI: capture), then compare
python eval_gt.py <work folder> gt.json avatar.vrm
```

`eval_detail.py` prints, per shot and bone, where each bone points (swing)
and which way it faces (twist about its own length), the facing error of
pelvis / chest / head about the vertical, and jitter against the truth.
`make_turn_gt.py` makes the turning test (whole body turned: a slow full
turn, side and back views, fast spins); render it with `gt.html` pointed at
`gt_anim_turn.json`.

`eval_gt.py` prints joint position error (cm), limb direction error (degrees,
only limbs inside the picture), bone orientation error, finger error, camera
tilt and field-of-view error, per shot.

## Real-motion test

Real dance motion capture (the free CMU library) on the avatar, with shots
aimed at hard cases: a plain front view, fast spins, a close camera with arms
and a kick aimed at the lens, a post passing in front of the legs, a side view
with crossing legs and a low camera at the feet. Fingers get a smooth
open/close motion of their own.

```
mkdir mocap
for f in 005/05_02 005/05_06 055/55_01 060/60_08 094/94_01; do
  curl -o mocap/$(basename $f).bvh https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/$f.bvh
done
python make_mocap_gt.py                 # -> mocap/gt_anim.json (BVH -> the avatar's bones)
python -m http.server 8799 &
node rgt_mocap.js                       # -> mocap/frames/*.jpg, mocap/gt.json
ffmpeg -framerate 30 -i mocap/frames/f_%05d.jpg -c:v libx264 -crf 16 -pix_fmt yuv420p mocap/bench.mp4
# run the app on mocap/bench.mp4, then
python eval_mocap.py <work folder>      # one number per problem: limbs, body angle, spine / head,
                                        # pointing at the camera, spins, passing object, crossed
                                        # legs, feet on the floor, hands, flicker
```

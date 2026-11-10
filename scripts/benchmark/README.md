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

`eval_gt.py` prints joint position error (cm), limb direction error (degrees,
only limbs inside the picture), bone orientation error, finger error, camera
tilt and field-of-view error, per shot.

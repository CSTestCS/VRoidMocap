# VRoid Mocap

A Windows app that turns a video into a motion-capture animation for your own
`.vrm` avatar (VRoid Studio, VRM 0.x and 1.0).

1. **Analysis** (once per video). Every frame is scanned to find:
   * every camera cut,
   * every person, tracked through the shot and **re-identified** when they come
     back (after leaving the frame, after a cut, or when the framing changes),
   * each character's **height relative to the scenery** (single-view
     metrology against the ground plane's horizon, linked across shots through
     shared characters),
   * the **camera angle and lens for every frame**: yaw, pitch, roll and field of
     view, from background motion with people masked out, aligned to gravity,
     with the focal length fitted per shot.
2. **Motion capture** (per character). Pick a character and a second, high
   accuracy pass runs on just that person: heavy body model on a stabilised
   crop plus hand tracking on every wrist. It uses the analysis data for depth,
   scale and camera angles. Every joint is fitted onto the camera ray through
   its detection with fixed bone lengths, then turned into VRM bone rotations,
   including all 15 finger bones per hand, wrists, forearm twist and feet.
   Root motion is integrated from planted feet on your avatar's own skeleton,
   and planted feet are pinned with leg IK so they do not slide.
3. **Viewer**. Watch side by side with the video, as *avatar over video*
   (the clip rendered through the solved camera, to check alignment), or with a
   free orbit camera. Scrub, step frame by frame, **lock movement** (keeps the
   character in the middle of the room, optionally height too) and **export
   `.fbx`**. Go back and pick another character at any time without
   re-analysing.

## Download / run

* **Windows:** download `VRoidMocap.exe` from the latest
  [GitHub Actions build](../../actions/workflows/build-windows.yml) (artifact
  *VRoidMocap-windows*) or from Releases, and double-click it. It needs no
  install. It uses the Edge WebView2 runtime that ships with Windows 10/11.
* **From source** (any OS, Python 3.10–3.12):
  ```
  pip install -r requirements.txt
  python scripts/fetch_models.py
  python run_app.py            # or: python run_app.py --browser
  ```

Command line (same exe):

```
VRoidMocap.exe analyze video.mp4
VRoidMocap.exe capture video.mp4 avatar.vrm 1 out.fbx [--in-place] [--smoothing 1] [--no-foot-lock]
VRoidMocap.exe selftest
```

## The exported FBX

* Skeleton = your avatar's own humanoid bones, with the **same bone names and
  rest orientations** as the `.vrm`, so the clip binds by name to the model in
  Blender (VRM add-on), Unity (Humanoid rig), Unreal and Maya.
* One key per video frame at the video's frame rate, centimetres, Y-up.
* Optional animated camera **VideoCamera** that reproduces the original
  camera (position, rotation and focal length), so you can render your avatar
  from the exact shots of the video.
* *In place* / *lock height* options match the viewer's *lock movement* toggle.

The exporter is verified by importing into Blender: joint positions and the
camera match the app's own solve to 0.0 cm / 0.0°.

## Accuracy

`scripts/eval_accuracy.py` projects the posed avatar through the solved camera
and compares every limb with the video. On a 25 s clip of the sample video
(1080p60 dance with a moving camera), at default settings:

| | median 2D angle error vs video |
|---|---|
| upper arms, thighs, shoulders, hips | 0.3–1.0° |
| shins (with foot locking) | 1–2° |
| fingers (where the hand is visible) | ~2.7° |
| hips position | 1% of body height |
| planted foot sliding | 0 cm/s median |

With smoothing at 0 the body is at 0.1–0.2° and fingers 0.7°, but with more
frame-to-frame jitter. The *Smoothing* slider and *Lock planted feet* checkbox
in step 2 let you choose. *Re-solve* on a character card re-runs only the solve
(seconds), not the capture.

Honest limits of single-camera capture:
* **Depth**, meaning how far a limb points toward or away from the camera, is
  inferred, not measured. It is smoothed over time and constrained by bone
  lengths, but it is the least certain part.
* **Hidden fingers** (fists, hands behind the body, hands holding a prop) are
  filled from neighbouring frames. Hand coverage is shown under the timeline.
* **Proportions:** the avatar receives the *rotations*, not the positions, so
  an avatar with different limb lengths will not overlap the video pixel for
  pixel.
* Camera translation along the view axis (dolly) can't be separated from the
  character moving. Root motion therefore comes from foot contacts, which
  works for tracking and handheld shots.

## Building the exe yourself

On Windows with Python 3.12:

```
pip install -r requirements-build.txt
python scripts/fetch_models.py
pyinstaller VRoidMocap.spec      # -> dist/VRoidMocap.exe
```

The GitHub workflow `.github/workflows/build-windows.yml` does this on every
push, runs the unit tests and a smoke test of the built exe, and attaches the
exe to a release when a `v*` tag is pushed.

## Project layout

```
vroidmocap/core/analysis.py   pass 1: shots, tracking, identities, cameras, heights
vroidmocap/core/capture.py    pass 2a: per-frame body + hand landmarks (parallel)
vroidmocap/core/retarget.py   pass 2b: 3D reconstruction, VRM rotations, root motion, foot IK
vroidmocap/core/camera.py     camera geometry (focal fit, gravity, rays)
vroidmocap/core/vrm.py        VRM 0.x / 1.0 humanoid skeleton reader
vroidmocap/core/fbx_writer.py binary FBX 7.4 writer
vroidmocap/server.py          local API used by the UI
vroidmocap/web/               UI (three.js + three-vrm viewer)
```

Models: Google MediaPipe Pose Landmarker (heavy/full), Hand Landmarker and
MobileNetV3 image embedder (Apache 2.0). Viewer: three.js and @pixiv/three-vrm
(MIT).

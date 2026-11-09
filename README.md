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
   free orbit camera. Scrub, step frame by frame, pick the output frame rate
   (12/24/30/60 fps), **lock movement** (keeps the character in the middle of
   the room, optionally height too) and **export `.fbx`**, with or without
   the avatar model. Go back and pick another character at any time without
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

Command line: use `VRoidMocapCLI.exe` (same app, console build):

```
VRoidMocapCLI.exe analyze video.mp4
VRoidMocapCLI.exe capture video.mp4 avatar.vrm 1 out.fbx [--fps 30] [--no-model] [--in-place] [--smoothing 1] [--no-foot-lock]
VRoidMocapCLI.exe selftest
```

## The exported FBX

* **Frame rate:** exactly 12, 24, 30 or 60 fps (toolbar *Frame rate*). The
  animation is resampled to that rate, and the viewer plays at the same rate.
* **Skeleton:** your avatar's complete VRoid node hierarchy (every bone,
  including hair and skirt bones, with the original names and rest transforms).
  Only rotations are animated, plus the hips translation for root motion.
* **Model (optional, on by default):** the avatar's meshes with skin weights,
  materials and textures. Textures are embedded in the FBX; when saving to a
  folder they are also written to `<name>.fbm/`. Untick *Include the avatar
  model* for an animation-only FBX that binds by bone name to the same model
  in Blender, Unity (Humanoid), Unreal or Maya.
* Optional animated **VideoCamera** that reproduces the original camera
  (position, rotation, focal length).
* *In place* / *lock height* match the viewer's *lock movement* toggle.

Verified by importing into Blender: bone positions match the app's solve to
0.0 cm, the camera matches exactly, and skinned vertices follow their bones
exactly.

## How the motion is cleaned up

* **Camera cuts are found on every frame.** A cut is a sudden one- or
  two-frame spike in picture change; fast pans are not mistaken for cuts.
  Short camera "snaps" become their own short shots, and wherever camera
  tracking is lost the camera alignment restarts instead of drifting.
* **A still camera stays still.** Wherever the background does not move, the
  cloned camera is locked (fixed position and rotation). The character moves
  within that fixed view: across the image it follows the video exactly, while
  depth is corrected only while the feet are free, so planted feet never slide.
* **Spins.** During spins the pose model can swap the character's left and
  right or flip front and back. Every frame is read four ways (as detected,
  sides swapped, depth mirrored, both) and the sequence whose body turns
  smoothly is chosen.
* **Unseen bones freeze.** A bone (or the whole character) that is not seen
  holds its last seen pose, then blends into the next seen pose over 0.2 s.
* **No flips.** Single-frame spikes are rejected. Palm vs back-of-hand and
  the two possible depths of every elbow, wrist, knee and ankle are resolved
  over time. Finger joints that appear bent backwards are mirrored forward.
* **Natural joints (IK + limits).**
  * Elbows: hinge IK that keeps the wrist where it is. The elbow sits where it
    was seen, and the upper arm never turns past its natural range.
  * Thighs never twist. They only swing to point at the knee; any twist the
    leg needs happens at the knee (±45°).
  * Forearm twist stays within its natural range. Wrists, fingers, thumbs,
    spine, neck and feet stay within joint limits.
  * The skeleton is never changed; only rotations are corrected.
* **Smoothing** is motion-adaptive and zero-phase: strong where a bone is
  nearly still, light during fast moves.

## Accuracy

`scripts/eval_accuracy.py` projects the posed avatar through the solved camera
and compares every limb with the video. Full 3-minute sample (1080p60 dance,
moving camera), default settings:

| | |
|---|---|
| body limbs, median 2D angle error vs video | 1.2° |
| fingers vs the hand detector | ~11° (the detector itself is noisy on small hands) |
| rotation jitter (median / 99th pct) | 0.2 / 4.7 °/frame² |
| hips position | 1% of body height |
| planted foot sliding | 0 cm/s median |

Honest limits of single-camera capture:
* **Depth**, meaning how far a limb points toward or away from the camera, is
  inferred, not measured.
* **Hidden or tiny hands:** fingers are only as good as what can be seen; when
  unseen they hold their last pose.
* **Very fast spins** can briefly confuse front and back.
* **Proportions:** the avatar receives rotations, not positions, so an avatar
  with different limb lengths will not overlap the video pixel for pixel.

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

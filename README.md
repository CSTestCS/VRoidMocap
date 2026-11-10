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
   accuracy pass runs on just that person:
   * a whole-body keypoint model (RTMW) finds every joint in the picture
     (run at about 30 fps and interpolated for faster videos; this is the slow
     part: expect roughly 5 to 10 minutes per minute of video on a 4-core CPU),
   * the MediaPipe body model adds a rough depth for every joint, and the hand
     model tracks both hands on crops around the wrists,
   * the **3D solve** combines them over the whole shot (see below) into joint
     positions in front of the camera,
   * these become VRM bone rotations, including all 15 finger bones per hand,
     wrists, forearm twist and feet. Root motion is integrated from planted
     feet on your avatar's own skeleton, and planted feet are pinned with leg
     IK so they do not slide.
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
  The first motion capture downloads the whole-body model once (370 MB, into
  `%LOCALAPPDATA%\VRoidMocap\models`). Without internet the app still works,
  using the MediaPipe body model alone (less precise).
* **From source** (any OS, Python 3.10–3.12):
  ```
  pip install -r requirements.txt
  python scripts/fetch_models.py --all
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

## How the body is reconstructed in 3D

One camera shows precisely *where* a joint is in the picture, but not how far
away it is. What fixes the depth is that bones do not change length: a bone of
known length seen at a given size leaves exactly two possible positions for
its far end, pointing towards the camera or away from it. The solve
(`vroidmocap/core/lift.py`) is built on that:

1. **Bone lengths** are measured from the video itself, from the frames in
   which a bone lies parallel to the picture (where it appears at full length).
2. **Distance of the character** per frame follows from how large the bones
   appear, smoothed over time.
3. **Towards or away** is chosen per limb over the whole shot (not frame by
   frame): the choice has to agree with the learned depth bone by bone, keep
   knees and elbows bending the right way, keep the thigh within the hip's
   range, and not flip between frames.
4. **Refinement**: all joints of all frames are then adjusted together
   (Gauss-Newton) so that they match the picture, keep their bone lengths,
   stay near the learned depth and move smoothly along the line of sight.
5. **Which way is up** is taken per shot from the reconstructed legs, which
   gives the camera's tilt (high and low camera angles).

### Which way the body faces (turns, side and back views)

The hardest part of a turn is telling which way the chest and pelvis face,
and detectors often swap left and right in back views. `facing.py` tracks
the chest and the pelvis separately, frame by frame, from several clues:

* how wide the shoulders / hips look compared with their real width,
* which side of the picture the character's left is on (with a state for
  "labels swapped", so a detector's left/right mix-up is recognised),
* the nose sits in front of the neck (which way the head and chest turn),
* whether a face can be seen at all (front or back),
* bent knees and toes point forwards,
* the pose model's own 3D guess (allowing for its typical front/back mistake).

The cheapest sequence that turns smoothly is chosen over each shot. The
result sets how much deeper one shoulder / hip is than the other in the 3D
solve, fixes swapped left/right labels, and turns the pose model's depth the
right way round. The **spine** then carries the twist between pelvis and
chest, spread over spine, chest and upper chest. A single frame that
contradicts everything around it is not used for the pose.

### Head

The head's turn, nod and tilt are fitted to the 68 face points of the
whole-body model (`head.py`: the 3D layout of an anime face, fitted with
perspective). This is only used while the face is really visible; seen from
behind, the head holds its pose relative to the neck.

## How the motion is cleaned up

* **Camera cuts are found on every frame.** A cut is a sudden one- or
  two-frame spike in picture change; fast pans are not mistaken for cuts.
  Short camera "snaps" become their own short shots, and wherever camera
  tracking is lost the camera alignment restarts instead of drifting.
* **A still camera stays still.** Wherever the background does not move, the
  cloned camera is locked (fixed position and rotation). The character moves
  within that fixed view: across the image it follows the video exactly, while
  depth is corrected only while the feet are free, so planted feet never slide.
* **Spins and back views.** See *Which way the body faces* above.
* **Unseen bones freeze.** A bone (or the whole character) that is not seen
  holds its last seen pose, then blends into the next seen pose over 0.2 s.
  "Seen" counts both detectors, and a joint that the whole-body model puts on
  the edge of the picture (it does that with joints outside it) only counts as
  seen if MediaPipe sees it too.
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
* **Smoothing** is zero-phase and adapts to the motion: strong where a bone
  is still or moves steadily (a spin keeps its speed and is smoothed fully),
  light only where the motion changes abruptly (a hit, the start or end of a
  spin).

## Accuracy

Measured against ground truth (`scripts/benchmark`): the sample avatar is
rendered with a known animation and known cameras, the app captures that
video, and the result is compared bone by bone in 3D.

**Turning test** (dance with the whole body turned: a slow full turn, then
side and back views and two fast spins; 1080p):

| | previous version | now |
|---|---|---|
| chest facing error, spins (mean / frames > 45° off) | 25.6° / 18% | 9.4° / 1% |
| pelvis facing error, spins | 21.4° / 9% | 12.0° / 0% |
| head facing error, spins | 31.1° / 10% | 16.5° / 0% |
| spine twist error, spins | 16.6° | 7.1° |
| thigh twist error, spins | 22° | 12° |
| jitter, spins (truth: 0.58) | 1.19 cm/frame² | 0.78 cm/frame² |
| joint position error, spins / slow turn | 11.7 / 8.1 cm | 8.9 / 7.8 cm |

**Earlier test** (frontal, high angle, orbiting camera, close-up):

| | previous version | now |
|---|---|---|
| head orientation error (per shot) | 27 / 34 / 21 / 24° | 11 / 20 / 14 / 10° |
| pelvis facing error, high angle / close-up | 17.4° / 27.4° | 10.6° / 13.7° |
| joint position error, high angle | 14.5 cm | 11.4 cm |

On the sample dance video: rotation jitter (99th percentile) 4.9 → 2.7
°/frame², worst single-frame jump 88° → 59°.

Honest limits of single-camera capture:
* **Depth** is inferred, not measured. A limb that points only slightly
  towards or away from the camera can be reconstructed on the wrong side
  (about 6% of limbs are more than 45° off in the test above).
* **Hidden limbs** (behind the body, outside the picture) are held in their
  last seen pose.
* **Hands:** in a full-body shot a hand is about 40 pixels. Whether the palm
  or the back of the hand faces the camera is read correctly only about 75%
  of the time, so forearm twist and hand orientation are the least reliable
  part. Finger bones are within roughly 30° of the truth; in close-ups they
  are better.
* **Upper-arm twist** of a nearly straight arm cannot be seen (only the
  hand shows it); it follows the last bent position.
* **Field of view** cannot be measured from a single view of a person; it is
  estimated, which affects how far away the cloned camera stands, not the pose.
* **Proportions:** the avatar receives rotations, not positions, so an avatar
  with different limb lengths will not overlap the video pixel for pixel.

## Building the exe yourself

On Windows with Python 3.12:

```
pip install -r requirements-build.txt
python scripts/fetch_models.py --all
pyinstaller VRoidMocap.spec      # -> dist/VRoidMocap.exe
```

The GitHub workflow `.github/workflows/build-windows.yml` does this on every
push, runs the unit tests and a smoke test of the built exe, and attaches the
exe to a release when a `v*` tag is pushed.

## Project layout

```
vroidmocap/core/analysis.py   pass 1: shots, tracking, identities, cameras, heights
vroidmocap/core/capture.py    pass 2a: per-frame whole-body, body-depth and hand landmarks (parallel)
vroidmocap/core/facing.py     pass 2b: which way chest and pelvis face (turns, back views, left/right labels)
vroidmocap/core/head.py       pass 2b: head orientation fitted to the face points
vroidmocap/core/lift.py       pass 2b: the 3D body solve (bone lengths, towards/away, refinement)
vroidmocap/core/retarget.py   pass 2c: VRM rotations, root motion, foot IK, cloned camera
vroidmocap/core/camera.py     camera geometry (focal fit, gravity, rays)
vroidmocap/core/vrm.py        VRM 0.x / 1.0 humanoid skeleton reader
vroidmocap/core/fbx_writer.py binary FBX 7.4 writer
vroidmocap/server.py          local API used by the UI
vroidmocap/web/               UI (three.js + three-vrm viewer)
```

Models: RTMW whole-body keypoints (OpenMMLab, Apache 2.0, run with ONNX
Runtime), Google MediaPipe Pose Landmarker (heavy/full), Hand Landmarker and
MobileNetV3 image embedder (Apache 2.0). Viewer: three.js and @pixiv/three-vrm
(MIT).

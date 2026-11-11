# VRoid Mocap

A Windows app that turns a video into a motion-capture animation for your own
avatar: a `.vrm` (VRoid Studio, VRM 0.x and 1.0) or an `.fbx` humanoid
character (Mixamo, VRoid FBX, Unity, Unreal, Blender/Rigify, 3ds Max Biped,
MMD conversions...).

1. **Analysis** (once per video). Every frame is scanned to find:
   * every camera cut,
   * every person, tracked through the shot and **re-identified** when they come
     back (after leaving the frame, after a cut, or when the framing changes),
   * each character's **height relative to the scenery** (single-view
     metrology against the ground plane's horizon, linked across shots through
     shared characters),
   * **the camera, from the scenery alone** (see *The camera* below): how it
     turns and zooms every frame, and where it travels (dolly, truck, crane,
     handheld) wherever the background shows parallax. People are masked out,
     so a character moving never moves the camera.
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
     wrists, forearm twist and feet. The character is placed in the scene
     where the camera sees it (camera position + what is in the picture), and
     planted feet are pinned with leg IK so they do not slide.
3. **Viewer**. Watch side by side with the video, as *avatar over video*
   (the clip rendered through the solved camera, to check alignment), or with a
   free orbit camera. Scrub, step frame by frame, pick the output frame rate
   (12/24/30/60 fps), **lock movement** (keeps the character in the middle of
   the room, optionally height too), **select a range** (drag on the timeline,
   or `[ In` / `Out ]`, keys I / O) and **loop** it (key L), and **export
   `.fbx`** of the whole clip or just the range, with or without the avatar
   model. Go back and pick another character at any time without
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
VRoidMocapCLI.exe capture video.mp4 avatar.vrm|avatar.fbx 1 out.fbx [--fps 30] [--no-model] [--in-place] [--smoothing 1] [--no-foot-lock]
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
* **Range:** with a range selected in the viewer, *Only the selected range*
  exports just those frames (the file name gets the frame numbers).
* **FBX avatars:** the export is your own FBX file written again with the
  animation added: the model, skin, materials and textures are the file's own,
  untouched (or, without the model, only its skeleton). Every humanoid bone is
  animated in the file's own axes, units and rotation order; twist, helper,
  hair and other extra bones keep their rest pose relative to their parent.
  Bones are found by name and by the hierarchy (the hips are where legs and
  spine branch, the spine is what lies between hips and neck...), and the
  skeleton is stood in a T-pose for the solver, whatever its own rest pose
  (A-pose etc.) is. Verified in Blender with a Mixamo-named, Z-up, A-pose
  model: every bone within 0.001 cm of the solve.
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

## The camera

The camera is worked out from the background, never from the characters:

* Background points are tracked over many frames (people masked out, every
  track checked forwards and backwards).
* **Turning and zooming** are fitted per frame (rotation + focal length change
  explaining how the background moves). A camera whose background does not
  move is exactly still.
* **Travelling** (dolly, truck, crane, handheld) shows up as parallax: near
  things slide over far ones, which turning and zooming cannot explain.
  Wherever it appears, the camera path and the depth of the tracked scenery
  are solved together (bundle adjustment, grown a few frames at a time). A
  stretch only counts as travel if it explains the background clearly better
  than the best turn + zoom fit, so a zoom is never mistaken for a dolly.
* **The lens** comes from the scenery when the camera turns or travels (only
  the right focal length makes it consistent), otherwise from the people's
  proportions.
* The camera's travel has no absolute size from the background alone; each
  travelling stretch gets one scale (the one that leaves the character's hips
  moving least across it). That is a single number per stretch: the shape and
  direction of the path are the background's.
* The character is placed **in that same scene**: camera position + what the
  picture shows (depth along the camera ray is smoothed, the position across
  the picture follows the video exactly). The cloned camera is where the real
  camera is; it does not follow the character, and a character that walks out
  of the picture simply leaves it.
* Across a cut the character continues where it was; the floor is where the
  feet are (one vertical shift per shot for the whole scene, so the picture is
  unchanged).

Measured on rendered ground truth (locked-off camera, pan + 2.6x zoom, a 3.9 m
dolly with a 45 degree turn, an occluder): still cameras 0.00 degrees / 0.00 m
of movement, pan and zoom within 0.1 degrees and 0.3%, the dolly within 0.5
degrees and 17 cm, lens within 1 degree where the camera turns or travels.

## How the motion is cleaned up

* **Camera cuts are found on every frame.** A cut is a sudden one- or
  two-frame spike in picture change; fast pans are not mistaken for cuts.
  Short camera "snaps" become their own short shots, and wherever camera
  tracking is lost the camera alignment restarts instead of drifting.
* **A still camera stays still** (see *The camera*): it is exactly still
  wherever the background is.
* **Spins and back views.** See *Which way the body faces* above.
* **Only what is seen is keyframed.** A bone
  moves only while all the joints that define it are clearly inside the
  picture, detected (the whole-body model sure of it or MediaPipe calling it
  visible) and not behind something (the person mask says the pixel is not
  the person and the pose model agrees). A joint the whole-body model puts on
  the edge of the picture needs MediaPipe to see it too, and glimpses shorter
  than 0.15 s do not count. A joint also counts when both models are fairly
  sure of it (a leg under a skirt or behind hair), and a weak detection that
  runs into a clear sighting, before or after, counts too.
* **Keyframes only where seen.** A bone gets keyframes only while it is seen.
  Where it is not (out of the picture, behind something, cut off by a zoom or
  a cut), it gets no keyframes: the motion in between is a smooth curve from
  its last seen pose to its next seen pose, leaving and arriving with the
  speed it had there. The character's position works the same way. So:
  * the character out of the shot: it moves smoothly from where it left to
    where it comes back; after the last sighting it stays,
  * only partly in the shot: the visible part follows the video, the rest
    moves in between its sightings; the position follows whatever part is
    seen (hips, shoulders, eyes, a knee, a foot, a hand),
  * until a bone is first seen it is in the **A pose** (arms 45 degrees down),
    and it eases into its first seen pose once it is in the picture.
  A visible bone whose parent is not seen keeps the direction it is seen in.
  Frames whose detection contradicts everything around them count as not
  seen.
* **Context from both directions.** The person tracker needs most of a body
  to find someone. After the scan, every track is followed backwards from
  where it starts and forwards from where it ends (within its shot, up to 3 s):
  the box moves with the camera and re-centres on whatever the whole-body
  model finds (a pair of legs, a hand), so a low camera that first shows only
  the feet, or someone walking out of the picture, is captured as the same
  character. Short shots (fast cuts) keep their tracks. A weak detection that
  runs into a clear sighting of the same joint, before or after it, counts as
  seen, and one- or two-frame flickers do not split a sighting. A frame is
  usable when a few joints are clearly seen (legs alone are enough), and when
  the box cuts the body off, the detector is run again on a larger box.
* **Close to the camera.** A limb that runs out of the picture is squeezed into
  it by the detectors and looks far too short. A limb much shorter than its
  bone length allows at its distance, ending at the picture edge it heads for,
  is treated as cut off (frozen), not as pointing at the camera, so a
  character walking up to the camera does not kneel.
* **Tracking box.** In back views the tracker can lose the body and follow
  only the head; the detectors then "find" a tiny, contorted person in that
  box. Within a shot the box may not collapse below 60% of the body size
  seen around it; a collapsed box is extended from the head to full size.
* **No flips.** Single-frame spikes are rejected. The two possible depths of
  every elbow, wrist, knee and ankle are resolved over time. Finger joints that
  appear bent backwards are mirrored forward.
* **Hands: palm or back, from context.** At full-body distance a hand is only
  ~40 pixels, so a single frame is often unsure which way round it is. Both
  hand detectors vote every frame (the whole-body model is the more reliable
  one), and the answer is decoded over the whole clip: a hand cannot switch
  from showing its palm to showing its back without turning edge-on, so
  reliable frames carry the decision through unsure ones. On rendered ground
  truth this raised the left hand from 70% to 83% right (right hand 91% to 93%).
* **Natural joints (IK + limits).**
  * Elbows: hinge IK that keeps the wrist where it is. The elbow sits where it
    was seen; the joint limit may nudge it (at most 35 degrees around the
    shoulder-wrist line) but never relocate it away from what the video shows.
    Nearly straight or fully folded arms keep their previous twist.
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
* **Hands:** in a full-body shot a hand is about 40 pixels. Decoded over the
  clip, whether the palm or the back faces the camera is right in about 83-93%
  of frames in the tests; forearm twist and hand orientation are still the
  least reliable part. Finger bones are within roughly 30° of the truth; in
  close-ups they are better.
* **Upper-arm twist** of a nearly straight arm cannot be seen (only the
  hand shows it); it follows the last bent position.
* **Field of view** of a camera that never turns or travels cannot be read
  from the scenery; it is then estimated from the people's proportions
  (within ~5 degrees in the tests), which affects distances, not the pose.
* **Hidden joints:** a joint behind an object is recognised as hidden most of
  the time; hip joints just behind an object's edge are often still inferred
  (correctly) from the upper body.
* **Camera travel size:** the background gives the travel's shape, not its
  size; one scale per travelling stretch comes from the character (assumes the
  character does not travel with the camera the whole stretch).
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
vroidmocap/core/analysis.py   pass 1: shots, tracking, identities, heights
vroidmocap/core/scene.py      pass 1: the camera from the scenery (turn, zoom, lens, travel)
vroidmocap/core/capture.py    pass 2a: per-frame whole-body, body-depth and hand landmarks (parallel)
vroidmocap/core/facing.py     pass 2b: which way chest and pelvis face (turns, back views, left/right labels)
vroidmocap/core/head.py       pass 2b: head orientation fitted to the face points
vroidmocap/core/lift.py       pass 2b: the 3D body solve (bone lengths, towards/away, refinement)
vroidmocap/core/retarget.py   pass 2c: VRM rotations, root motion, foot IK, cloned camera
vroidmocap/core/camera.py     camera geometry (focal fit, gravity, rays)
vroidmocap/core/vrm.py        VRM 0.x / 1.0 humanoid skeleton reader
vroidmocap/core/fbx_reader.py FBX reader (binary 7.x and ASCII)
vroidmocap/core/fbx_avatar.py FBX characters: humanoid mapping, T-pose, motion back onto the model, export
vroidmocap/core/fbx_writer.py binary FBX 7.4 writer
vroidmocap/server.py          local API used by the UI
vroidmocap/web/               UI (three.js + three-vrm / FBXLoader viewer)
```

Models: RTMW whole-body keypoints (OpenMMLab, Apache 2.0, run with ONNX
Runtime), Google MediaPipe Pose Landmarker (heavy/full), Hand Landmarker and
MobileNetV3 image embedder (Apache 2.0). Viewer: three.js and @pixiv/three-vrm
(MIT).

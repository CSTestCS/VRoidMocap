"""Head orientation from the face.

The whole-body model finds 68 face points (eyes, brows, nose, mouth, jaw
line). Their 3D layout on an anime-style head (measured on a VRoid head; x =
the character's left, y = up, z = forward, metres, relative to the head joint)
is fitted to them with a perspective camera (PnP): this gives the head's turn,
nod and tilt directly, instead of guessing it from a few rough 3D points.
Faces seen from behind or too small give nothing (the head then holds its pose
relative to the neck).
"""
from __future__ import annotations

import numpy as np

FACE_TEMPLATE = np.array([
    [-7.425, 7.864, 6.461], [-7.209, 6.008, 6.613], [-6.902, 4.242, 6.745], [-6.428, 2.545, 6.856],
    [-5.702, 0.918, 7.077], [-4.652, -0.506, 7.427], [-3.429, -1.711, 7.862], [-1.755, -2.622, 8.149],
    [0.000, -2.912, 8.237], [1.755, -2.622, 8.149], [3.429, -1.711, 7.862], [4.652, -0.506, 7.427],
    [5.702, 0.918, 7.077], [6.428, 2.545, 6.856], [6.902, 4.242, 6.745], [7.209, 6.008, 6.613],
    [7.425, 7.864, 6.461], [-5.684, 9.207, 9.871], [-4.811, 9.629, 10.368], [-3.796, 9.708, 10.870],
    [-2.778, 9.607, 11.319], [-1.779, 9.392, 11.686], [1.779, 9.392, 11.686], [2.778, 9.607, 11.319],
    [3.796, 9.708, 10.870], [4.811, 9.629, 10.368], [5.684, 9.207, 9.871], [0.000, 7.582, 11.920],
    [0.000, 6.402, 12.547], [0.000, 5.274, 13.122], [0.000, 4.173, 13.605], [-1.365, 3.228, 12.023],
    [-0.701, 3.079, 12.135], [0.000, 3.015, 12.177], [0.701, 3.079, 12.135], [1.365, 3.228, 12.023],
    [-4.595, 7.585, 10.217], [-3.799, 7.938, 10.601], [-2.795, 7.942, 10.817], [-1.879, 7.471, 10.929],
    [-2.778, 7.232, 10.741], [-3.767, 7.261, 10.507], [1.879, 7.471, 10.929], [2.795, 7.942, 10.817],
    [3.799, 7.938, 10.601], [4.595, 7.585, 10.217], [3.767, 7.261, 10.507], [2.778, 7.232, 10.741],
    [-2.407, 0.998, 10.572], [-1.543, 1.326, 11.401], [-0.524, 1.589, 12.031], [0.000, 1.550, 12.117],
    [0.524, 1.589, 12.031], [1.543, 1.326, 11.401], [2.407, 0.998, 10.572], [1.630, 0.365, 10.974],
    [0.864, -0.030, 11.332], [0.000, -0.171, 11.577], [-0.864, -0.030, 11.332], [-1.630, 0.365, 10.974],
    [-2.187, 0.957, 10.600], [-0.938, 1.094, 11.274], [0.000, 1.055, 11.764], [0.938, 1.094, 11.274],
    [2.187, 0.957, 10.600], [0.962, 0.727, 11.170], [0.000, 0.639, 11.537], [-0.962, 0.727, 11.170],
]) / 100.0

# points used: jaw line, nose bridge / tip / base, eye corners, mouth corners
USE = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 27, 30, 31, 33, 35, 36, 39, 42, 45, 48, 54, 51, 57])
FACE0 = 23  # first face point in the 133 whole-body points


def head_from_face(wb2d, wbs, focal, cx, cy, valid, min_score=0.5):
    """Returns (R (N,3,3) head rotation in the cv camera frame, quality (N,) 0..1;
    0 = no usable face)."""
    import cv2
    N = len(wb2d)
    R = np.tile(np.eye(3), (N, 1, 1))
    qual = np.zeros(N)
    if wb2d is None or not np.any(wbs):
        return R, qual
    obj = FACE_TEMPLATE[USE]
    for f in np.where(valid)[0]:
        j = FACE0 + USE
        sc = wbs[f, j]
        ok = sc > min_score
        if ok.sum() < 8:
            continue
        uv = wb2d[f, j][ok].astype(np.float64)
        size = np.ptp(uv, axis=0).max()
        if size < 12:  # face too small to read
            continue
        K = np.array([[focal[f], 0, cx], [0, focal[f], cy], [0, 0, 1.0]])
        try:
            good, rv, tv = cv2.solvePnP(obj[ok], uv, K, None, flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error:
            continue
        if not good or tv[2, 0] <= 0:
            continue
        rv, tv = cv2.solvePnPRefineLM(obj[ok], uv, K, None, rv, tv)
        proj, _ = cv2.projectPoints(obj[ok], rv, tv, K, None)
        err = np.median(np.linalg.norm(proj[:, 0] - uv, axis=1)) / size
        Rm = cv2.Rodrigues(rv)[0]
        # the face must look towards the camera (it was detected)
        fwd = Rm[:, 2]
        if np.dot(fwd, tv[:, 0]) > 0.2 * np.linalg.norm(tv):
            continue
        R[f] = Rm
        qual[f] = float(np.clip(1.0 - err / 0.12, 0, 1) * np.clip((sc[ok].mean() - 0.55) / 0.3, 0, 1)
                        * np.clip((size - 12) / 25, 0, 1))
    return R, qual

"""Camera geometry: intrinsics, per-person translation, gravity, shot alignment.

Conventions
-----------
* "cv" camera frame: x right, y down, z forward (OpenCV). MediaPipe world
  landmarks are expressed in this orientation (hip-centred).
* world frame: Y up, right-handed (three.js / VRM). A camera with world
  rotation R_w_cv maps cv-frame vectors into world: v_w = R_w_cv @ v_cv.
* D = diag(1,-1,-1) converts cv axes to the three.js camera axes.
"""
from __future__ import annotations

import numpy as np

from .detect import L_ANK, L_HIP, L_KNEE, L_SH, R_ANK, R_HIP, R_KNEE, R_SH
from .mathutil import angle_between, axis_angle_to_mat, normalize, rot_between

D = np.diag([1.0, -1.0, -1.0])
DEFAULT_VFOV = 40.0
VFOV_CANDIDATES = np.arange(8.0, 95.0, 2.0)


def focal_from_vfov(vfov_deg, height):
    return 0.5 * height / np.tan(np.radians(vfov_deg) / 2)


def vfov_from_focal(f, height):
    return float(np.degrees(2 * np.arctan(0.5 * height / f)))


def ray_rotation(f, cx, cy, px, py):
    """Rotation from the virtual camera of a crop centred at (px,py) to the real camera.

    The pose networks see a crop as if the camera were looking straight at the
    person; landmarks they return are oriented in that virtual camera. Rotating
    them by this matrix puts them in the real camera's frame."""
    ray = normalize(np.array([(px - cx) / f, (py - cy) / f, 1.0]))
    return rot_between(np.array([0.0, 0.0, 1.0]), ray)


def solve_translation(world, lm2d, w, f, cx, cy):
    """Least-squares camera-frame translation T of the hip centre, keeping the
    landmark orientation fixed: project(world + T) ~= lm2d."""
    u = lm2d[:, 0] - cx
    v = lm2d[:, 1] - cy
    n = len(u)
    A = np.zeros((2 * n, 3))
    b = np.zeros(2 * n)
    A[0::2, 0] = f
    A[0::2, 2] = -u
    b[0::2] = u * world[:, 2] - f * world[:, 0]
    A[1::2, 1] = f
    A[1::2, 2] = -v
    b[1::2] = v * world[:, 2] - f * world[:, 1]
    sw = np.sqrt(np.repeat(np.maximum(w, 0), 2))
    try:
        T, *_ = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if T[2] <= 0.2:
        return None
    return T


def project(P, f, cx, cy):
    z = np.maximum(P[:, 2], 1e-6)
    return np.stack([f * P[:, 0] / z + cx, f * P[:, 1] / z + cy], axis=1)


def reprojection_error(world, lm2d, w, f, cx, cy, crop_c):
    R = ray_rotation(f, cx, cy, crop_c[0], crop_c[1])
    W = world @ R.T
    T = solve_translation(W, lm2d, w, f, cx, cy)
    if T is None:
        return None
    p = project(W + T, f, cx, cy)
    err = np.sqrt(((p - lm2d) ** 2).sum(1))
    size = np.ptp(lm2d[w > 0.5], axis=0).max() if (w > 0.5).sum() > 3 else np.ptp(lm2d, axis=0).max()
    return float((err * w).sum() / max(w.sum(), 1e-6) / max(size, 1.0))


def estimate_vfov(samples, width, height, prior=DEFAULT_VFOV):
    """samples: list of (world(33,3), lm2d(33,2), vis(33), crop_center(2), focal_scale).
    focal_scale is the zoom of that frame relative to the shot reference.
    Returns the reference vertical FOV in degrees."""
    if len(samples) < 3:
        return prior
    cx, cy = width / 2, height / 2
    scores = []
    for vf in VFOV_CANDIDATES:
        f0 = focal_from_vfov(vf, height)
        errs = []
        for world, lm2d, vis, cc, zs in samples:
            e = reprojection_error(world, lm2d, vis, f0 * zs, cx, cy, cc)
            if e is not None:
                errs.append(e)
        if len(errs) < 3:
            scores.append(np.inf)
            continue
        e = float(np.median(errs))
        # weak prior keeps nearly-orthographic (far away) shots near a typical lens
        scores.append(e + 0.004 * ((vf - prior) / 30.0) ** 2)
    scores = np.array(scores)
    if not np.isfinite(scores).any():
        return prior
    i = int(np.argmin(scores))
    # parabolic refinement
    if 0 < i < len(scores) - 1 and np.isfinite(scores[i - 1]) and np.isfinite(scores[i + 1]):
        a, b, c = scores[i - 1], scores[i], scores[i + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if den > 1e-12 else 0.0
        return float(VFOV_CANDIDATES[i] + np.clip(off, -1, 1) * (VFOV_CANDIDATES[1] - VFOV_CANDIDATES[0]))
    return float(VFOV_CANDIDATES[i])


def body_up_vector(world, vis):
    """Up direction of a standing body in the camera (cv) frame plus a weight
    saying how upright/straight-legged the pose is (0 = unusable)."""
    if min(vis[L_SH], vis[R_SH], vis[L_HIP], vis[R_HIP]) < 0.5:
        return None, 0.0
    sh = (world[L_SH] + world[R_SH]) / 2
    hip = (world[L_HIP] + world[R_HIP]) / 2
    torso = normalize(sh - hip)
    legs_ok = min(vis[L_KNEE], vis[R_KNEE], vis[L_ANK], vis[R_ANK]) > 0.5
    if legs_ok:
        ank = (world[L_ANK] + world[R_ANK]) / 2
        leg_up = normalize(hip - ank)
        kneeL = angle_between(world[L_HIP] - world[L_KNEE], world[L_KNEE] - world[L_ANK])
        kneeR = angle_between(world[R_HIP] - world[R_KNEE], world[R_KNEE] - world[R_ANK])
        bend = max(kneeL, kneeR)
        lean = angle_between(torso, leg_up)
        w = float(np.clip(1 - bend / np.radians(25), 0, 1) * np.clip(1 - lean / np.radians(25), 0, 1))
        up = normalize(torso + 2 * leg_up)
        return up, w
    return torso, 0.15


def align_from_up(up_cv):
    """World-from-camera rotation that maps the camera-frame up vector to +Y and
    keeps the optical axis pointing at -Z (yaw 0)."""
    u = normalize(up_cv)
    g = np.array([0.0, 0.0, 1.0]) - u * u[2]
    if np.linalg.norm(g) < 1e-6:
        g = np.array([0.0, 1.0, 0.0]) - u * u[1]
    g = normalize(g)
    Bc = np.stack([u, g, np.cross(u, g)], axis=1)
    Bw = np.stack([[0, 1.0, 0], [0, 0, -1.0], [-1.0, 0, 0]], axis=1)
    return Bw @ Bc.T


def yaw_matrix(angle):
    return axis_angle_to_mat(np.array([0, 1.0, 0]), angle)


def camera_euler(R_w_cv):
    """(yaw, pitch, roll) in degrees of a camera. yaw 0 = looking along -Z,
    positive pitch = looking up."""
    fwd = R_w_cv @ np.array([0, 0, 1.0])
    down = R_w_cv @ np.array([0, 1.0, 0])
    yaw = np.degrees(np.arctan2(-fwd[0], -fwd[2]))
    pitch = np.degrees(np.arcsin(np.clip(fwd[1], -1, 1)))
    # roll: angle of camera "up" around the forward axis relative to world up
    up = -down
    right = normalize(np.cross(fwd, [0, 1.0, 0])) if abs(fwd[1]) < 0.999 else np.array([1.0, 0, 0])
    ref_up = np.cross(right, fwd)
    roll = np.degrees(np.arctan2(np.dot(np.cross(ref_up, up), fwd), np.dot(ref_up, up)))
    return float(yaw), float(pitch), float(roll)


def horizon_y(R_w_cv, f, cx, cy, x):
    """Image row of the horizon (vanishing line of the ground plane) at column x."""
    n = R_w_cv.T @ np.array([0, 1.0, 0])  # world up in camera coords
    # points on horizon satisfy n . ((x-cx)/f, (y-cy)/f, 1) = 0
    if abs(n[1]) < 1e-6:
        return None
    return cy - f * (n[0] * (x - cx) / f + n[2]) / n[1]


def hips_yaw(world_cv, R_w_cv):
    """Facing yaw (radians, 0 = facing +Z) of a body given camera rotation."""
    a = R_w_cv @ (world_cv[L_HIP] - world_cv[R_HIP])
    a2 = R_w_cv @ (world_cv[L_SH] - world_cv[R_SH])
    x = normalize(a) + normalize(a2)
    # facing = up x left-axis... left-axis is +X when facing +Z
    return float(np.arctan2(-x[2], x[0]))

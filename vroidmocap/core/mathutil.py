"""Small rotation / geometry helpers (numpy only, quaternions are [x, y, z, w])."""
from __future__ import annotations

import numpy as np

EPS = 1e-9


def normalize(v, axis=-1):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, EPS)


def basis_from_axes(a, b):
    """Orthonormal basis (as matrix columns) whose first axis is `a` and whose
    second axis lies in the a/b plane on b's side."""
    a = normalize(a)
    b = np.asarray(b, dtype=np.float64) - a * np.dot(a, b)
    nb = np.linalg.norm(b)
    if nb < 1e-6:
        # b parallel to a: pick any perpendicular vector
        tmp = np.array([0.0, 1.0, 0.0]) if abs(a[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        b = tmp - a * np.dot(a, tmp)
        nb = np.linalg.norm(b)
    b = b / nb
    c = np.cross(a, b)
    return np.stack([a, b, c], axis=1)


def rot_from_axes(a_rest, b_rest, a_obs, b_obs):
    """Rotation matrix taking the (a_rest, b_rest) frame onto (a_obs, b_obs)."""
    return basis_from_axes(a_obs, b_obs) @ basis_from_axes(a_rest, b_rest).T


def rot_between(u, v):
    """Minimal rotation matrix taking direction u onto direction v."""
    u = normalize(u)
    v = normalize(v)
    c = float(np.dot(u, v))
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        axis = np.cross(u, [1.0, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(u, [0, 1.0, 0])
        return axis_angle_to_mat(normalize(axis), np.pi)
    axis = np.cross(u, v)
    s = np.linalg.norm(axis)
    return axis_angle_to_mat(axis / s, np.arctan2(s, c))


def axis_angle_to_mat(axis, angle):
    axis = normalize(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def orthonormalize(m):
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def mat_to_quat(m):
    """3x3 (or Nx3x3) rotation matrix -> quaternion [x, y, z, w]."""
    m = np.asarray(m, dtype=np.float64)
    single = m.ndim == 2
    if single:
        m = m[None]
    q = np.empty((m.shape[0], 4))
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    for i in range(m.shape[0]):
        r = m[i]
        if tr[i] > 0:
            s = np.sqrt(tr[i] + 1.0) * 2
            q[i] = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
            q[i] = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
        elif r[1, 1] > r[2, 2]:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            q[i] = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            q[i] = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
    q = normalize(q)
    return q[0] if single else q


def quat_to_mat(q):
    q = normalize(np.asarray(q, dtype=np.float64))
    single = q.ndim == 1
    if single:
        q = q[None]
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    m = np.empty((q.shape[0], 3, 3))
    m[:, 0, 0] = 1 - 2 * (y * y + z * z)
    m[:, 0, 1] = 2 * (x * y - z * w)
    m[:, 0, 2] = 2 * (x * z + y * w)
    m[:, 1, 0] = 2 * (x * y + z * w)
    m[:, 1, 1] = 1 - 2 * (x * x + z * z)
    m[:, 1, 2] = 2 * (y * z - x * w)
    m[:, 2, 0] = 2 * (x * z - y * w)
    m[:, 2, 1] = 2 * (y * z + x * w)
    m[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return m[0] if single else m


def quat_mul(a, b):
    ax, ay, az, aw = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    bx, by, bz, bw = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def quat_conj(q):
    q = np.array(q, dtype=np.float64)
    q[..., :3] *= -1
    return q


def slerp(q0, q1, t):
    q0 = normalize(q0)
    q1 = normalize(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        return normalize(q0 + t * (q1 - q0))
    th = np.arccos(np.clip(d, -1, 1))
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


def mat_power(m, t):
    """Fraction t of a rotation (geodesic interpolation from identity)."""
    q = mat_to_quat(m)
    return quat_to_mat(slerp(np.array([0, 0, 0, 1.0]), q, t))


def quat_continuity(qs):
    """Flip signs so consecutive quaternions stay in the same hemisphere (in place)."""
    for i in range(1, len(qs)):
        if np.dot(qs[i - 1], qs[i]) < 0:
            qs[i] = -qs[i]
    return qs


def angle_between(u, v):
    u = normalize(u)
    v = normalize(v)
    return float(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0)))


def gaussian_smooth(x, sigma, axis=0, valid=None):
    """Zero-phase Gaussian smoothing along `axis` that ignores invalid samples.

    x: array, valid: bool mask over `axis` (None = all valid). Invalid samples are
    filled from valid neighbours (weighted) so gaps get interpolated smoothly.
    """
    x = np.asarray(x, dtype=np.float64)
    if sigma <= 0:
        return x.copy()
    n = x.shape[axis]
    xm = np.moveaxis(x, axis, 0).reshape(n, -1)
    w = np.ones(n) if valid is None else np.asarray(valid, dtype=np.float64)
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    num = np.zeros_like(xm)
    den = np.zeros(n)
    xw = xm * w[:, None]
    for i, kv in enumerate(k):
        off = i - r
        lo, hi = max(0, -off), min(n, n - off)
        if hi <= lo:
            continue
        num[lo:hi] += kv * xw[lo + off:hi + off]
        den[lo:hi] += kv * w[lo + off:hi + off]
    out = np.where(den[:, None] > 1e-9, num / np.maximum(den[:, None], 1e-9), xm)
    return np.moveaxis(out.reshape((n,) + np.moveaxis(x, axis, 0).shape[1:]), 0, axis)


def interp_gaps(x, valid):
    """Linear interpolation of invalid rows of x (first axis) from valid ones."""
    x = np.array(x, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if valid.all() or not valid.any():
        return x
    idx = np.arange(len(x))
    flat = x.reshape(len(x), -1)
    for c in range(flat.shape[1]):
        flat[~valid, c] = np.interp(idx[~valid], idx[valid], flat[valid, c])
    return flat.reshape(x.shape)

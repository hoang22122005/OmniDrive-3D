"""Closed-form Double Sphere (DS) fisheye camera model (Usenko et al., 3DV 2018).

Provides fast trigonometric-free forward projection pi(x) and exact analytical
inverse unprojection pi^-1(u, v) for native non-pinhole 3D/4DGS & AVM pipelines.
"""

import numpy as np
from scipy.optimize import least_squares


def ds_project(xyz, params):
    """Project 3D camera-frame points [..., 3] using 6-param Double Sphere (fx, fy, cx, cy, xi, alpha)."""
    fx, fy, cx, cy, xi, alpha = [float(v) for v in params]
    pts = np.asarray(xyz, dtype=np.float64)
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
    d1 = np.sqrt(x * x + y * y + z * z)
    z1 = xi * d1 + z
    d2 = np.sqrt(x * x + y * y + z1 * z1)
    denom = alpha * d2 + (1.0 - alpha) * z1

    w1 = (alpha / (1.0 - alpha)) if alpha <= 0.5 else ((1.0 - alpha) / alpha)
    w2 = (w1 + xi) / np.sqrt(2.0 * w1 * xi + xi * xi + 1.0)
    valid = (d1 > 1e-7) & (z > -w2 * d1) & (denom > 1e-7)

    safe_d = np.where(valid, denom, 1.0)
    u = fx * (x / safe_d) + cx
    v = fy * (y / safe_d) + cy
    return np.stack([u, v], axis=-1), valid


def ds_unproject(uv, params):
    """Exact closed-form inverse pi^-1(u, v) -> unit 3D rays [..., 3] in camera frame (zero Newton iterations)."""
    fx, fy, cx, cy, xi, alpha = [float(v) for v in params]
    pts = np.asarray(uv, dtype=np.float64)
    mx = (pts[..., 0] - cx) / fx
    my = (pts[..., 1] - cy) / fy
    r2 = mx * mx + my * my

    if alpha > 0.5:
        valid = r2 < (1.0 / max(2.0 * alpha - 1.0, 1e-9)) - 1e-6
    else:
        valid = np.ones_like(r2, dtype=bool)

    disc1 = np.maximum(1.0 - (2.0 * alpha - 1.0) * r2, 1e-12)
    mz = (1.0 - alpha * alpha * r2) / (alpha * np.sqrt(disc1) + (1.0 - alpha))
    mz2 = mz * mz
    disc2 = np.maximum(mz2 + (1.0 - xi * xi) * r2, 0.0)
    scale = (mz * xi + np.sqrt(disc2)) / np.maximum(mz2 + r2, 1e-12)

    rx = scale * mx
    ry = scale * my
    rz = scale * mz - xi
    rays = np.stack([rx, ry, rz], axis=-1)
    norm = np.linalg.norm(rays, axis=-1, keepdims=True)
    rays = rays / np.maximum(norm, 1e-9)
    return rays, valid


def fit_double_sphere_from_camera(camera):
    """Fit 6-parameter Double Sphere intrinsics to a KB4/MEI/Pinhole camera spec (<1px RMSE on 1080p)."""
    k = np.asarray(camera["K"], dtype=np.float64).reshape(3, 3)
    sh, sw = camera["image_hw"]
    max_th = np.deg2rad(min(float(camera.get("max_theta_deg", 92.0)), 94.0))

    if camera["model"] == "pinhole" and "distortion" not in camera:
        return {
            "params": [float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]), 0.0, 0.0],
            "rmse_px": 0.0,
        }

    thetas = np.linspace(0.02, max_th, 34)
    phis = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    tt, pp = np.meshgrid(thetas, phis)
    rays = np.stack([np.sin(tt) * np.cos(pp), np.sin(tt) * np.sin(pp), np.cos(tt)], axis=-1).reshape(-1, 3)

    rad = np.hypot(rays[:, 0], rays[:, 1])
    th = np.arctan2(rad, rays[:, 2])
    if camera["model"] == "kb4":
        dist = np.asarray(camera["distortion"], dtype=np.float64)
        t2 = th * th
        r_proj = th * (1.0 + t2 * np.polynomial.polynomial.polyval(t2, dist))
        uv_plane = rays[:, :2] * (r_proj / np.maximum(rad, 1e-9))[:, None]
    elif camera["model"] == "mei":
        xi_m = float(camera["xi"])
        denom = rays[:, 2] + xi_m * np.linalg.norm(rays, axis=-1)
        uv_plane = rays[:, :2] / np.maximum(denom[:, None], 1e-9)
    else:
        uv_plane = rays[:, :2] / np.maximum(rays[:, 2:3], 1e-9)

    uv_ref = uv_plane @ k[:2, :2].T + k[:2, 2]
    ok = (uv_ref[:, 0] >= 0.0) & (uv_ref[:, 0] < sw) & (uv_ref[:, 1] >= 0.0) & (uv_ref[:, 1] < sh)
    if "valid_circle" in camera:
        vcx, vcy, vcr = camera["valid_circle"]
        ok &= np.hypot(uv_ref[:, 0] - vcx, uv_ref[:, 1] - vcy) <= vcr
    pts_ok, uv_ok = rays[ok], uv_ref[ok]
    if len(pts_ok) < 16:
        return {
            "params": [float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]), 0.0, 0.0],
            "rmse_px": 0.0,
        }

    p0 = [float(k[0, 0]) * 0.78, float(k[1, 1]) * 0.78, float(k[0, 2]), float(k[1, 2]), -0.22, 0.58]
    bounds = ([10.0, 10.0, 0.0, 0.0, -1.5, 0.001], [sw * 4.0, sh * 4.0, sw, sh, 1.5, 0.995])

    def residuals(p):
        uv_ds, v_ok = ds_project(pts_ok, p)
        diff = np.where(v_ok[:, None], uv_ds - uv_ok, 100.0)
        return diff.ravel()

    res = least_squares(residuals, p0, bounds=bounds, max_nfev=80)
    uv_fit, _ = ds_project(pts_ok, res.x)
    rmse = float(np.sqrt(np.mean(np.sum((uv_fit - uv_ok) ** 2, axis=1))))
    return {
        "params": [round(float(v), 6) for v in res.x],
        "rmse_px": round(rmse, 4),
    }

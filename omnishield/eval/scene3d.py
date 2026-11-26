import copy

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_closing, binary_dilation, distance_transform_edt, gaussian_filter, gaussian_filter1d, label, median_filter, minimum_filter1d
from skimage.measure import marching_cubes
from torch.nn import functional as F

from omnishield.geometry.calibration import ego_mask, letterbox_parameters, project, rigid_matrix
from omnishield.geometry.double_sphere import ds_project, ds_unproject, fit_double_sphere_from_camera
from omnishield.io import child_path


N_BOWL_SECTORS = 120
BOWL_RINGS = 44
# Exact real-world VinFast VF 8 physical geometry (verified against PHENIKAA Sensor_Extrinsics.json):
#   Length = 4.750m (front overhang 0.850m + wheelbase 2.950m + rear overhang 0.950m)
#   Width  = 1.934m (body excl. mirrors; CAM_F_L/R sit under mirrors at y = +-0.9701m, x_ego = +0.9933m, z = 1.029m)
#   Height = 1.667m (roof rack CAM_P_L/R at z = 1.766m, CAM_P_F at z = 1.881m)
#   Axles  = rear x_ego = -1.449m (INS origin x_lidar = 0.0), front x_ego = +1.501m (wheelbase 2.950m)
#   Bumpers= rear x_ego = -2.399m (CAM_F_B at -2.4125m), front x_ego = +2.351m (CAM_F_F at +2.4122m)
#   Track  = 1.640m (y = +-0.820m), Tire 245/45R20 (R = 0.364m, width = 0.245m), Ground clearance = 0.175m
VF8_VEHICLE_SPEC = {
    "model": "VinFast VF 8 (Phenikaa-X L4)",
    "length_m": 4.750,
    "width_m": 1.934,
    "height_m": 1.667,
    "wheelbase_m": 2.950,
    "front_overhang_m": 0.850,
    "rear_overhang_m": 0.950,
    "track_width_m": 1.640,
    "ground_clearance_m": 0.175,
    "tire_radius_m": 0.364,
    "tire_width_m": 0.245,
    "x_rear_bumper_m": -2.399,
    "x_front_bumper_m": 2.351,
    "x_rear_axle_m": -1.449,
    "x_front_axle_m": 1.501,
    "x_center_m": -0.024,
    "mirror_x_m": 0.993,
    "mirror_y_m": 0.970,
    "mirror_z_m": 1.029,
}

# Corridor-aligned superellipse (p=3.0) flat road extending ~5m ahead/behind bumpers, followed by gentle tangent wall.
BOWL_FLAT_RX = 7.20
BOWL_FLAT_RY = 3.80
BOWL_FLAT_POWER = 3.0
BOWL_PAD_X = 8.00
BOWL_PAD_Y = 6.50
BOWL_WALL_HEIGHT = 3.20
BOWL_WALL_POWER = 2.2
BOWL_FLAT_SPLIT = 0.52
BOWL_SEAM_X_SHIFT = 1.00
BOWL_SEAM_X_SCALE = 1.65
BOWL_BASE_RX = BOWL_FLAT_RX
BOWL_BASE_RY = BOWL_FLAT_RY
BOWL_MIN_RX = 3.40
BOWL_MIN_RY = 2.15
BOWL_OUTER_PAD_X = BOWL_PAD_X
BOWL_OUTER_PAD_Y = BOWL_PAD_Y


def export_vehicle_spec(rig, ground_plane=None):
    """Derive exact physical vehicle dimensions & camera heights above road plane."""
    spec = dict(VF8_VEHICLE_SPEC)
    a, b, c0 = [float(v) for v in ground_plane] if ground_plane is not None else (0.0, 0.0, float(rig.get("ground_height_m", 0.0)))
    by_slot = {cam["slot"]: cam for cam in rig.get("cameras", [])}
    cam_heights = {}
    for cam in rig.get("cameras", []):
        t_ec = np.linalg.inv(rigid_matrix(cam["T_cam_from_ego"], cam["slot"]))
        px, py, pz = t_ec[:3, 3]
        cam_heights[cam["slot"]] = round(float(pz - (a * px + b * py + c0)), 3)
    if cam_heights:
        spec["camera_heights_m"] = cam_heights
    spec["road_plane_deg"] = [round(float(np.rad2deg(np.arctan(a))), 3), round(float(np.rad2deg(np.arctan(b))), 3)]
    if {"front", "rear", "left", "right"}.issubset(by_slot):
        pos = {}
        for slot in ("front", "rear", "left", "right"):
            t_ec = np.linalg.inv(rigid_matrix(by_slot[slot]["T_cam_from_ego"], slot))
            pos[slot] = t_ec[:3, 3]
        cam_span_x = float(pos["front"][0] - pos["rear"][0])
        cam_span_y = float(abs(pos["left"][1]) + abs(pos["right"][1]))
        spec["cam_span_length_m"] = round(cam_span_x, 4)
        spec["cam_span_width_m"] = round(cam_span_y, 4)
        spec["mirror_x_m"] = round(float(0.5 * (pos["left"][0] + pos["right"][0])), 4)
        spec["mirror_y_m"] = round(float(0.5 * cam_span_y), 4)
        spec["mirror_z_m"] = round(float(0.5 * (pos["left"][2] + pos["right"][2])), 4)
        # If this is a non-VF8 / synthetic rig with significantly different camera span, adapt bounds automatically
        if abs(cam_span_x - 4.825) > 0.25 or abs(cam_span_y - 1.940) > 0.25:
            x0, x1 = float(pos["rear"][0]), float(pos["front"][0])
            hw = max(0.85, 0.5 * cam_span_y)
            length = max(3.8, x1 - x0)
            wb = round(length * 0.62, 3)
            xc = 0.5 * (x0 + x1)
            spec.update({
                "model": rig.get("rig_id", "Custom Rig"),
                "length_m": round(length, 3),
                "width_m": round(2.0 * hw, 3),
                "wheelbase_m": wb,
                "x_rear_bumper_m": round(x0, 3),
                "x_front_bumper_m": round(x1, 3),
                "x_rear_axle_m": round(xc - 0.5 * wb, 3),
                "x_front_axle_m": round(xc + 0.5 * wb, 3),
                "x_center_m": round(xc, 3),
                "track_width_m": round(2.0 * hw * 0.85, 3),
            })
    return spec


def physical_ego_bounds(rig):
    """Return physical chassis [x_rear, x_front, -half_w, +half_w] for accurate PDC/AR clearance."""
    spec = export_vehicle_spec(rig)
    hw = 0.5 * spec["width_m"]
    return spec["x_rear_bumper_m"], spec["x_front_bumper_m"], -hw, hw


def export_bowl_spec():
    """Single source of truth for 3D Bowl geometry & seam anchor shared by Python and WebGL."""
    return {
        "flat_rx": BOWL_FLAT_RX,
        "flat_ry": BOWL_FLAT_RY,
        "pad_x": BOWL_PAD_X,
        "pad_y": BOWL_PAD_Y,
        "wall_height": BOWL_WALL_HEIGHT,
        "wall_power": BOWL_WALL_POWER,
        "flat_split": BOWL_FLAT_SPLIT,
        "seam_x_shift": BOWL_SEAM_X_SHIFT,
        "seam_x_scale": BOWL_SEAM_X_SCALE,
    }


def estimate_valid_circles(rig, manifest_rows, raw_root):
    """Auto-detect each KB4 fisheye lens's clean circular optical FOV (excluding black vignette)."""
    cams = rig["cameras"]
    valid_rows = [r for r in manifest_rows if all(img is not None for img in r.get("images", []))]
    if not valid_rows:
        return rig
    sel = valid_rows[::max(1, len(valid_rows) // 32)][:32]
    updated = copy.deepcopy(rig)
    for ci, cam in enumerate(updated["cameras"]):
        if cam.get("model") != "kb4" or "valid_circle" in cam:
            continue
        acc = None
        cnt = 0
        for r in sel:
            rel = r["images"][ci]
            if rel is None:
                continue
            path = child_path(raw_root, rel)
            if not path.is_file():
                continue
            with Image.open(path) as pil_im:
                arr = np.asarray(pil_im.convert("L"), dtype=np.float32)
            acc = arr if acc is None else acc + arr
            cnt += 1
        if acc is None or cnt == 0:
            continue
        avg = acc / cnt
        h, w = avg.shape
        cx, cy = float(cam["K"][0][2]), float(cam["K"][1][2])
        # Only assign valid_circle if the fisheye image genuinely has dark circular corner vignette
        corner_vals = [
            float(avg[:24, :24].mean()), float(avg[:24, -24:].mean()),
            float(avg[-24:, :24].mean()), float(avg[-24:, -24:].mean()),
        ]
        if max(corner_vals) > 28.0:
            continue
        yy, xx = np.mgrid[0:h, 0:w]
        rr = np.hypot(xx - cx, yy - cy)
        best_r = min(cx, w - cx)
        for r_bin in range(int(0.45 * min(w, h)), int(max(w, h)), 4):
            ring = (rr >= r_bin) & (rr < r_bin + 4) & (yy > 60) & (yy < h - 60)
            if ring.sum() < 40:
                continue
            if float((avg[ring] > 24.0).mean()) >= 0.85:
                best_r = r_bin
        r_clean = max(float(best_r) - 24.0, 0.45 * min(w, h))
        cam["valid_circle"] = [round(cx, 2), round(cy, 2), round(r_clean, 2)]
    return updated


def fit_lidar_road_plane(rig, rows, raw_root):
    """Offline fit from spatially balanced raw returns, with separate validation scans."""
    from omnishield.data.phenikaa import load_lidar

    scans = [row for row in rows if row.get("lidar")]
    if any(str(row["lidar"]).lower().endswith(".laz") for row in scans):
        try:
            from laspy import LazBackend
        except ImportError:
            return None
        if not LazBackend.detect_available():
            return None
    if len(scans) < 4:
        return None
    ground = float(rig.get("ground_height_m", 0.0))
    train, check = [], []
    for i, idx in enumerate(np.linspace(0, len(scans) - 1, min(20, len(scans)), dtype=int)):
        try:
            points, _ = load_lidar(raw_root, scans[idx], rig)
        except (OSError, ImportError):
            continue
        x, y, z = points.T
        road = points[(np.abs(x) < 8) & (np.abs(y) < 2) & (np.abs(z - ground) < 0.25)
                      & ((np.abs(x) > 2.5) | (np.abs(y) > 1.15))]
        if len(road) < 40:
            continue
        # One median per half-metre cell avoids overweighting dense near-field returns.
        cells = np.floor(road[:, :2] * 2).astype(int)
        ids = cells[:, 0] * 16 + cells[:, 1]
        order = np.argsort(ids)
        groups = np.split(road[order], np.flatnonzero(np.diff(ids[order])) + 1)
        (check if i % 4 == 0 else train).extend(np.median(g, axis=0) for g in groups if len(g) >= 3)
    if min(len(train), len(check)) < 40:
        return None
    xyz = np.asarray(train)
    X = np.column_stack((xyz[:, :2], np.ones(len(xyz))))
    if np.linalg.matrix_rank(X) < 3:
        return None
    weights = np.ones(len(xyz))
    for _ in range(8):
        coef = np.linalg.lstsq(X * weights[:, None], xyz[:, 2] * weights, rcond=None)[0]
        weights = np.sqrt(0.025 / np.maximum(np.abs(xyz[:, 2] - X @ coef), 0.025))
    if np.max(np.abs(coef[:2])) > 0.035 or abs(coef[2] - ground) > 0.12:
        return None
    hold = np.asarray(check)
    residual = np.abs(hold[:, 2] - hold[:, :2] @ coef[:2] - coef[2])
    if np.median(residual) > np.median(np.abs(hold[:, 2] - ground)) or np.median(residual) > 0.05:
        return None
    return np.round(coef, 6).tolist(), {
        "source": "raw_lidar", "validation_cells": len(check),
        "median_abs_cm": round(100 * float(np.median(residual)), 2),
        "p90_abs_cm": round(100 * float(np.percentile(residual, 90)), 2),
    }


def estimate_road_plane(rig, manifest_rows, load_targets_fn, sample_path_fn, prepared_root, config, depth_centers):
    """Fit exact 3D road plane z = a*x + b*y + c via IRLS over multi-frame LiDAR road points + per-cam residuals."""
    base_gz = float(rig.get("ground_height_m", 0.0))
    n_cams = len(rig.get("cameras", []))
    if not manifest_rows or n_cams == 0:
        return [0.0, 0.0, base_gz], [0.0] * n_cams

    calibs = export_camera_calib(rig)
    dec = FeedForward3DGSDecoder(rig, calibs, tuple(config["image_hw"]), feature_stride=4, ground_z=base_gz)
    sample_rows = manifest_rows[::max(1, len(manifest_rows) // 16)][:16]
    pts_by_cam = [[] for _ in range(n_cams)]
    for srow in sample_rows:
        stgt = load_targets_fn(sample_path_fn(prepared_root, srow), config)
        db = stgt.get("depth")
        if db is None:
            continue
        for ci in range(n_cams):
            v = dec.ray_valid[ci] & (db[ci] >= 0) & (db[ci] < len(depth_centers))
            if not v.any():
                continue
            dm = depth_centers[np.clip(db[ci], 0, len(depth_centers) - 1)]
            v &= (dm > 0.5) & (dm < 10.0)
            if not v.any():
                continue
            pts = dec.origins[ci] + dec.rays_ego[ci][v] * dm[v, None]
            road = pts[(np.abs(pts[:, 0]) < 8.0) & (np.abs(pts[:, 1]) < 2.0)
                       & (pts[:, 2] > base_gz - 0.25) & (pts[:, 2] < base_gz + 0.25)]
            if len(road):
                pts_by_cam[ci].append(road)

    non_empty = [np.concatenate(c_pts, axis=0) for c_pts in pts_by_cam if c_pts]
    if not non_empty:
        return [0.0, 0.0, base_gz], [0.0] * n_cams
    pts_all = np.concatenate(non_empty, axis=0)
    if len(pts_all) < 60:
        return [0.0, 0.0, base_gz], [0.0] * n_cams

    design = np.column_stack([pts_all[:, 0], pts_all[:, 1], np.ones(len(pts_all))])
    w = np.ones(len(pts_all), dtype=np.float64)
    coef = np.array([0.0, 0.0, base_gz], dtype=np.float64)
    for _ in range(8):
        coef, *_ = np.linalg.lstsq(design * w[:, None], pts_all[:, 2] * w, rcond=None)
        res = np.abs(pts_all[:, 2] - design @ coef)
        w = 1.0 / np.maximum(res, 0.015)
    a = float(np.clip(coef[0], -0.035, 0.035))
    b = float(np.clip(coef[1], -0.035, 0.035))
    c0 = float(np.clip(coef[2], base_gz - 0.12, base_gz + 0.12))

    per_cam_cm = []
    for c_pts in pts_by_cam:
        if not c_pts:
            per_cam_cm.append(0.0)
            continue
        arr = np.concatenate(c_pts, axis=0)
        res = arr[:, 2] - (a * arr[:, 0] + b * arr[:, 1] + c0)
        per_cam_cm.append(round(float(np.median(res)) * 100.0, 2))
    return [round(a, 6), round(b, 6), round(c0, 5)], per_cam_cm


def _superellipse_radius(ca, sa, rx, ry, p=BOWL_FLAT_POWER):
    return np.power(np.power(np.abs(ca) / rx, p) + np.power(np.abs(sa) / ry, p), -1.0 / p)


def bowl_base_radii(angles):
    """Return baseline flat-to-wall inflection radius r_base(theta) keeping road lanes flat."""
    ca = np.cos(angles)
    sa = np.sin(angles)
    return _superellipse_radius(ca, sa, BOWL_BASE_RX, BOWL_BASE_RY)


def bowl_min_radii(angles):
    """Minimum flat radius ellipse so adaptive bowl never bends road lane markings near ego."""
    ca = np.cos(angles)
    sa = np.sin(angles)
    return 1.0 / np.hypot(ca / BOWL_MIN_RX, sa / BOWL_MIN_RY)


def build_adaptive_bowl_template(ground_z=0.0, n_rings=84, n_segs=160):
    """Build closed 360-deg parametric Stadium-Bowl mesh resolving 2.5D curb steps cleanly."""
    # Dense inner floor rings (0..BOWL_FLAT_SPLIT, <10cm radial step) + smooth wall rings
    t_floor = np.linspace(0.0, BOWL_FLAT_SPLIT, 56, endpoint=False, dtype=np.float32)
    t_wall = np.linspace(BOWL_FLAT_SPLIT, 1.0, n_rings - 56, dtype=np.float32)
    ts = np.concatenate([t_floor, t_wall])
    angles = np.linspace(0.0, 2.0 * np.pi, n_segs, endpoint=False, dtype=np.float32)
    tt, aa = np.meshgrid(ts, angles, indexing="ij")
    sec_idx = np.broadcast_to(np.arange(n_segs, dtype=np.float32)[None, :], tt.shape)

    # Pack 4 floats per vertex: [t_ring (0..1), angle_rad, sector_idx, ground_z]
    verts = np.stack([tt, aa, sec_idx, np.full_like(tt, float(ground_z))], axis=-1).reshape(-1, 4).astype("<f4")
    indices = []
    for r in range(n_rings - 1):
        for s in range(n_segs):
            s_next = (s + 1) % n_segs
            a = r * n_segs + s
            b = r * n_segs + s_next
            c = (r + 1) * n_segs + s
            d = (r + 1) * n_segs + s_next
            indices.extend([a, b, d, a, d, c])
    idx_arr = np.asarray(indices, dtype="<u4")
    header = np.asarray([verts.shape[0], idx_arr.shape[0]], dtype="<u4")
    return header.tobytes() + verts.tobytes() + idx_arr.tobytes()


def verify_bowl_mesh_integrity(
    ground_plane=(0.0, 0.0, 0.0), wall_ranges=None, polar_height=None,
    left_poly=(0.0, 0.0, 2.25), right_poly=(0.0, 0.0, -2.25),
    adapt_strength=1.0, n_rings=84, n_segs=160,
):
    """Verify 0 inverted/degenerate triangles and strict radial monotonicity on the adaptive bowl."""
    pa, pb, pc = [float(v) for v in ground_plane]
    t_floor = np.linspace(0.0, BOWL_FLAT_SPLIT, 56, endpoint=False, dtype=np.float64)
    t_wall = np.linspace(BOWL_FLAT_SPLIT, 1.0, n_rings - 56, dtype=np.float64)
    ts = np.concatenate([t_floor, t_wall])
    angles = np.linspace(0.0, 2.0 * np.pi, n_segs, endpoint=False, dtype=np.float64)
    tt, aa = np.meshgrid(ts, angles, indexing="ij")
    ca, sa = np.cos(aa), np.sin(aa)

    smooth_sa = np.clip((sa + 0.30) / 0.60, 0.0, 1.0)
    smooth_sa = smooth_sa * smooth_sa * (3.0 - 2.0 * smooth_sa)
    side_offset = (-right_poly[2]) * (1.0 - smooth_sa) + left_poly[2] * smooth_sa - 2.25
    ry = BOWL_FLAT_RY + np.clip(side_offset, -0.35, 0.55)
    r_flat_base = _superellipse_radius(ca, sa, BOWL_FLAT_RX, ry)
    r_flat = r_flat_base.copy()
    depth_wall = np.zeros_like(aa)

    if wall_ranges is not None:
        wr = np.asarray(wall_ranges, dtype=np.float64)
        sec_f = np.mod(angles * (len(wr) / (2.0 * np.pi)) - 0.5, len(wr))
        s0 = np.floor(sec_f).astype(np.intp) % len(wr)
        s1 = (s0 + 1) % len(wr)
        w = sec_f - np.floor(sec_f)
        a_v, b_v = wr[s0], wr[s1]
        va, vb = (a_v >= 0.01).astype(np.float64), (b_v >= 0.01).astype(np.float64)
        sup = va * (1.0 - w) + vb * w
        meas = (a_v * (1.0 - w) + b_v * w) / np.maximum(sup, 1e-4)
        out_ego = 1.0 / np.maximum(np.abs(ca[0]) / 2.85, np.abs(sa[0]) / 1.55)
        sw_t = np.clip((np.abs(sa[0]) - 0.20) / 0.45, 0.0, 1.0)
        sw_t = sw_t * sw_t * (3.0 - 2.0 * sw_t)
        side_w = 0.35 + 0.65 * sw_t
        sup_t = np.clip((sup - 0.20) / 0.65, 0.0, 1.0)
        sup_t = sup_t * sup_t * (3.0 - 2.0 * sup_t)
        dw_t = np.clip((meas - (r_flat_base[0] - 0.15)) / 1.05, 0.0, 1.0)
        dw_t = dw_t * dw_t * (3.0 - 2.0 * dw_t)
        depth_wall_1d = sup_t * (1.0 - dw_t) * side_w * adapt_strength
        change = np.clip(np.maximum(out_ego, meas - 0.04) - r_flat_base[0], -0.75, 0.25)
        r_flat = r_flat_base + depth_wall_1d[None, :] * change[None, :]
        depth_wall = np.broadcast_to(depth_wall_1d[None, :], aa.shape)

    r_pad = 1.0 / np.hypot(ca / BOWL_PAD_X, sa / BOWL_PAD_Y)
    wall_u = np.where(tt <= BOWL_FLAT_SPLIT, 0.0, (tt - BOWL_FLAT_SPLIT) / (1.0 - BOWL_FLAT_SPLIT))
    r_span = np.maximum((r_flat_base + r_pad) - r_flat, 0.5)
    r_floor = 0.05 + (r_flat - 0.05) * (tt / BOWL_FLAT_SPLIT)
    r_wall = r_flat + r_span * (wall_u * (1.0 - depth_wall * 0.45) + (wall_u ** 1.38) * (depth_wall * 0.45))
    r = np.where(tt <= BOWL_FLAT_SPLIT, r_floor, r_wall)

    radial_monotonic = bool(np.all(np.diff(r, axis=0) > 1e-6))
    x = r * ca
    y = r * sa
    road_z = pa * x + pb * y + pc
    wall_z = np.where(
        wall_u > 0.0,
        (BOWL_WALL_HEIGHT * (wall_u ** BOWL_WALL_POWER)) * (1.0 - depth_wall * 0.45)
        + (BOWL_WALL_HEIGHT * (wall_u ** 1.25)) * (depth_wall * 0.45),
        0.0,
    )
    curb_z = np.zeros_like(x)
    if polar_height is not None:
        ph = np.asarray(polar_height, dtype=np.float64)
        sf = np.mod(aa, 2.0 * np.pi) * (ph.shape[0] / (2.0 * np.pi)) - 0.5
        s0 = np.floor(sf).astype(np.intp) % ph.shape[0]
        s1 = (s0 + 1) % ph.shape[0]
        ws = sf - np.floor(sf)
        rf = np.clip((r - POLAR_R_MIN) / (POLAR_R_MAX - POLAR_R_MIN) * ph.shape[1] - 0.5, 0.0, ph.shape[1] - 1.0)
        r0 = np.floor(rf).astype(np.intp)
        r1 = np.minimum(r0 + 1, ph.shape[1] - 1)
        wr_f = rf - r0
        h00, h10, h01, h11 = ph[s0, r0], ph[s0, r1], ph[s1, r0], ph[s1, r1]
        interp_h = (h00 * (1.0 - wr_f) + h10 * wr_f) * (1.0 - ws) + (h01 * (1.0 - wr_f) + h11 * wr_f) * ws
        fade_in = np.clip((r - 1.15) / 0.30, 0.0, 1.0)
        fade_in = fade_in * fade_in * (3.0 - 2.0 * fade_in)
        fade_out = np.clip((wall_u - 0.05) / 0.43, 0.0, 1.0)
        fade_out = fade_out * fade_out * (3.0 - 2.0 * fade_out)
        curb_z = interp_h * fade_in * adapt_strength * (1.0 - fade_out)

    z = road_z + wall_z + curb_z
    pts = np.stack([-y, z, -x], axis=-1)
    p00, p01 = pts[:-1, :], np.roll(pts[:-1, :], -1, axis=1)
    p10, p11 = pts[1:, :], np.roll(pts[1:, :], -1, axis=1)
    n1 = np.cross(p11 - p00, p01 - p00)
    n2 = np.cross(p10 - p00, p11 - p00)
    normals = np.concatenate([n1.reshape(-1, 3), n2.reshape(-1, 3)], axis=0)
    areas = 0.5 * np.linalg.norm(normals, axis=1)
    # Upward (+Y in Three.js) or inward radial component must be positive across all triangles.
    centers_3d = np.concatenate([((p00 + p01 + p11) / 3.0).reshape(-1, 3), ((p00 + p11 + p10) / 3.0).reshape(-1, 3)], axis=0)
    inward = np.stack([-centers_3d[:, 0], np.ones_like(centers_3d[:, 1]), -centers_3d[:, 2]], axis=-1)
    dots = (normals * inward).sum(axis=1)
    return {
        "triangles": int(len(areas)),
        "degenerate_triangles": int((areas <= 1e-8).sum()),
        "inverted_triangles": int((dots <= 0.0).sum()),
        "radial_monotonic": radial_monotonic,
    }


def measure_near_field_coverage(rig, ground_plane=(0.0, 0.0, 0.0), band_m=0.25, step_m=0.10, min_cos_near=-0.26, bottom_crop_px=2.0):
    """Audit ground visibility in the near-field perimeter band around the chassis (optical vs shader)."""
    pa, pb, pc = [float(v) for v in ground_plane]
    x0, x1, y0, y1 = physical_ego_bounds(rig)
    xs = np.arange(x0 - band_m, x1 + band_m + 1e-6, step_m)
    ys = np.arange(y0 - band_m, y1 + band_m + 1e-6, step_m)
    gx, gy = np.meshgrid(xs, ys)
    in_band = (
        ~((gx >= x0) & (gx <= x1) & (gy >= y0) & (gy <= y1))
        & ((gx >= x0 - band_m) & (gx <= x1 + band_m) & (gy >= y0 - band_m) & (gy <= y1 + band_m))
    )
    pts = np.stack([gx[in_band], gy[in_band], pa * gx[in_band] + pb * gy[in_band] + pc], axis=-1)
    optical_any = np.zeros(len(pts), dtype=bool)
    shader_any = np.zeros(len(pts), dtype=bool)
    for cam in rig["cameras"]:
        sh, sw = cam["image_hw"]
        uv, _, valid = project(pts, cam, (sh, sw))
        in_crop = valid & (uv[:, 1] < sh - 0.5 - bottom_crop_px)
        optical_any |= in_crop
        t_ec = np.linalg.inv(rigid_matrix(cam["T_cam_from_ego"], cam["slot"]))
        delta = pts[:, :2] - t_ec[:2, 3]
        fwd2 = t_ec[:2, 2] / max(float(np.linalg.norm(t_ec[:2, 2])), 1e-5)
        cos_dir = (delta @ fwd2) / np.maximum(np.linalg.norm(delta, axis=1), 1e-4)
        shader_any |= in_crop & (cos_dir > min_cos_near)
    return {
        "total_points": int(len(pts)),
        "optical_covered": int(optical_any.sum()),
        "shader_covered": int(shader_any.sum()),
        "software_filtered": int((optical_any & ~shader_any).sum()),
        "physical_blind": int((~optical_any).sum()),
    }


def _gaussian_blur_2d(tensor_bchw, ksize, sigma):
    ax = torch.arange(ksize, dtype=tensor_bchw.dtype, device=tensor_bchw.device) - ksize // 2
    g = torch.exp(-0.5 * (ax / sigma) ** 2)
    g = g / g.sum()
    c = tensor_bchw.shape[1]
    kx = g.view(1, 1, 1, ksize).expand(c, 1, 1, ksize)
    ky = g.view(1, 1, ksize, 1).expand(c, 1, ksize, 1)
    p = ksize // 2
    out = F.conv2d(F.pad(tensor_bchw, (p, p, 0, 0), mode="replicate"), kx, groups=c)
    return F.conv2d(F.pad(out, (0, 0, p, p), mode="replicate"), ky, groups=c)


def _euler_xyz_torch(rpy):
    cx, cy, cz = torch.cos(rpy[:, 0]), torch.cos(rpy[:, 1]), torch.cos(rpy[:, 2])
    sx, sy, sz = torch.sin(rpy[:, 0]), torch.sin(rpy[:, 1]), torch.sin(rpy[:, 2])
    rot = torch.zeros((rpy.shape[0], 3, 3), device=rpy.device, dtype=rpy.dtype)
    rot[:, 0, 0] = cy * cz
    rot[:, 0, 1] = cz * sx * sy - cx * sz
    rot[:, 0, 2] = cx * cz * sy + sx * sz
    rot[:, 1, 0] = cy * sz
    rot[:, 1, 1] = cx * cz + sx * sy * sz
    rot[:, 1, 2] = cx * sy * sz - cz * sx
    rot[:, 2, 0] = -sy
    rot[:, 2, 1] = cy * sx
    rot[:, 2, 2] = cx * cy
    return rot


def calibrate_rig_extrinsics_photometric(
    rig, manifest_rows, raw_root, load_targets_fn=None, sample_path_fn=None, prepared_root=None,
    config=None, depth_centers=None, ground_z=0.0, device=None, ground_plane=None, poses=None,
):
    """Multi-Scale Stratified ZNCC + Joint Ground-Plane 6-DoF BA (LiDAR-assisted or Targetless Camera-Only):
    1) Stratifies frame selection across drives (e.g. City + Urban) & front/rear road-marking texture.
    2) Uses illumination-invariant 1-ZNCC on high-pass luma, yellow-chroma, and gradient magnitude across
       curb-safe flat-asphalt overlap zones, with per-corner top-75% frame trimming to reject dynamic objects
       (motorbikes/cars) without requiring LiDAR.
    3) In camera-only mode (no LiDAR road plane), jointly estimates road-plane tilt (pa, pb) within +-0.7 deg.
    4) Runs 3-stage Coarse->Medium->Fine pyramid (float16 VRAM-safe) with held-out validation gain gate.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rig = estimate_valid_circles(rig, manifest_rows, raw_root)
    cams = rig["cameras"]
    if len(cams) != 4 or any(c.get("model") != "kb4" for c in cams):
        return copy.deepcopy(rig), [0.0] * len(cams)

    valid_rows = [r for r in manifest_rows if all(img is not None for img in r.get("images", []))]
    if len(valid_rows) < 6:
        return copy.deepcopy(rig), [0.0] * len(cams)

    pa, pb, pc = [float(v) for v in (ground_plane if ground_plane is not None else (0.0, 0.0, float(ground_z)))]
    has_lidar_targets = (
        load_targets_fn is not None and sample_path_fn is not None and prepared_root is not None
        and config is not None and depth_centers is not None
    )
    optimize_gp = (not has_lidar_targets) and (ground_plane is None or (abs(pa) < 1e-6 and abs(pb) < 1e-6))

    # Prefer straight, flat-road frames (low yaw rate) when odometry poses are available
    straight_mask = [True] * len(valid_rows)
    if poses is not None and len(poses) == len(manifest_rows):
        row_to_idx = {id(r): i for i, r in enumerate(manifest_rows)}
        for vi, r in enumerate(valid_rows):
            mi = row_to_idx.get(id(r), -1)
            if mi > 0 and manifest_rows[mi - 1].get("drive_id") == r.get("drive_id"):
                t_rel = np.linalg.inv(poses[mi]) @ poses[mi - 1]
                dyaw = abs(float(np.arctan2(t_rel[1, 0], t_rel[0, 0])))
                if dyaw > 0.025:
                    straight_mask[vi] = False
        if sum(straight_mask) < max(8, len(valid_rows) // 4):
            straight_mask = [True] * len(valid_rows)

    # Curb-safe flat-asphalt overlap zones for all 4 corners (FL, FR, RL, RR)
    zone_defs = (
        ((2.2, 5.2), (0.95, 1.72), 0, 2),
        ((2.2, 5.2), (-2.20, -0.95), 0, 3),
        ((-5.2, -2.2), (0.95, 1.72), 1, 2),
        ((-5.2, -2.2), (-2.20, -0.95), 1, 3),
    )
    zone_pts, zone_slices = [], []
    z_off = 0
    for xr, yr, ca, cb in zone_defs:
        xs_z = np.linspace(xr[0], xr[1], 36)
        ys_z = np.linspace(yr[0], yr[1], 24)
        gx_z, gy_z = np.meshgrid(xs_z, ys_z)
        pts_z = np.stack([gx_z.ravel(), gy_z.ravel()], axis=1)
        zone_pts.append(pts_z)
        zone_slices.append((z_off, z_off + len(pts_z), ca, cb))
        z_off += len(pts_z)
    pts_xy = np.concatenate(zone_pts, axis=0).astype(np.float32)
    n_pts = len(pts_xy)

    # Drive-stratified & texture-scored frame selection across Front + Rear road strips
    def _strip_uv(cam_i, xr):
        xs_s = np.linspace(xr[0], xr[1], 28)
        ys_s = np.linspace(-2.1, 1.65, 18)
        gx_s, gy_s = np.meshgrid(xs_s, ys_s)
        pts_s = np.stack([gx_s.ravel(), gy_s.ravel(), (pa * gx_s + pb * gy_s + pc).ravel()], axis=1)
        sh_s, sw_s = cams[cam_i]["image_hw"]
        uv_s, _, _ = project(pts_s, cams[cam_i], (sh_s, sw_s))
        return (
            np.clip(uv_s[:, 0] / max(sw_s, 1), 0.0, 0.999).reshape(18, 28),
            np.clip(uv_s[:, 1] / max(sh_s, 1), 0.0, 0.999).reshape(18, 28),
        )

    u0n, v0n = _strip_uv(0, (2.3, 5.2))
    u1n, v1n = _strip_uv(1, (-5.2, -2.3))

    def score_frame(row):
        tot = 0.0
        for ci, (un, vn) in ((0, (u0n, v0n)), (1, (u1n, v1n))):
            p_cam = child_path(raw_root, row["images"][ci])
            if not p_cam.is_file():
                continue
            with Image.open(p_cam) as pil_im:
                pil_im.draft("RGB", (640, 360))
                arr = np.asarray(pil_im.convert("RGB"), dtype=np.float32)
            ah, aw = arr.shape[:2]
            ui = np.clip((un * aw).astype(np.int32), 0, aw - 1)
            vi = np.clip((vn * ah).astype(np.int32), 0, ah - 1)
            rgb_s = arr[vi, ui]
            lum_s = rgb_s.mean(axis=2)
            yel_s = rgb_s[:, :, 0] - rgb_s[:, :, 2]
            tot += float(
                np.mean(np.abs(np.diff(lum_s, axis=1)))
                + np.mean(np.abs(np.diff(lum_s, axis=0)))
                + 2.0 * np.mean(np.abs(np.diff(yel_s, axis=1)))
            )
        return tot

    target_frames = 24
    if len(valid_rows) <= target_frames:
        selected = list(valid_rows)
    else:
        by_drive = {}
        for vi, r in enumerate(valid_rows):
            if straight_mask[vi]:
                by_drive.setdefault(str(r.get("drive_id", "default")), []).append(r)
        if not by_drive:
            by_drive["default"] = list(valid_rows)
        per_drive = max(4, target_frames // max(1, len(by_drive)))
        selected = []
        for drows in by_drive.values():
            step = max(1, len(drows) // 60)
            scored = sorted([(score_frame(drows[i]), i) for i in range(0, len(drows), step)], reverse=True)
            chosen = []
            for sc_val, idx in scored:
                if len(chosen) < max(2, per_drive - 2) and all(abs(idx - prev_i) >= 3 for _, prev_i in chosen):
                    chosen.append((sc_val, idx))
            for idx in np.linspace(0, len(drows) - 1, 4, dtype=int):
                if len(chosen) < per_drive and all(abs(int(idx) - prev_i) >= 3 for _, prev_i in chosen):
                    chosen.append((score_frame(drows[int(idx)]), int(idx)))
            chosen.sort(key=lambda x: x[1])
            selected.extend(drows[idx] for _, idx in chosen[:per_drive])

    dec = None
    if has_lidar_targets:
        calibs = export_camera_calib(rig)
        model_hw = tuple(config["image_hw"])
        dec = FeedForward3DGSDecoder(rig, calibs, model_hw, feature_stride=4, ground_z=pc)

    f_cnt = len(selected)
    src_h, src_w = cams[0]["image_hw"]
    w_cal = min(800, max(320, int(src_w)))
    h_cal = min(450, max(180, int(round(w_cal * src_h / max(src_w, 1)))))
    pyr_dtype = torch.float16 if device.type == "cuda" else torch.float32
    imgs_coarse = torch.empty((f_cnt, 4, 4, h_cal, w_cal), dtype=pyr_dtype, device=device)
    imgs_med = torch.empty((f_cnt, 4, 4, h_cal, w_cal), dtype=pyr_dtype, device=device)
    imgs_fine = torch.empty((f_cnt, 4, 4, h_cal, w_cal), dtype=pyr_dtype, device=device)

    flat_masks = []
    with torch.no_grad():
        for fi, fr in enumerate(selected):
            elev = np.zeros((64, 80), dtype=bool)
            if has_lidar_targets:
                stgt = load_targets_fn(sample_path_fn(prepared_root, fr), config)
                dbins = stgt.get("depth")
                if dbins is not None:
                    for ci in range(4):
                        v = dec.ray_valid[ci] & (dbins[ci] >= 0) & (dbins[ci] < len(depth_centers))
                        if not v.any():
                            continue
                        dm = depth_centers[np.clip(dbins[ci], 0, len(depth_centers) - 1)]
                        v &= (dm > 0.4) & (dm < 11.0)
                        if not v.any():
                            continue
                        pts = dec.origins[ci] + dec.rays_ego[ci][v] * dm[v, None]
                        rz = pa * pts[:, 0] + pb * pts[:, 1] + pc
                        hi = pts[pts[:, 2] > rz + 0.06]
                        xi = ((hi[:, 0] + 6.0) / 0.15).astype(np.int32)
                        yi = ((hi[:, 1] + 4.0) / 0.125).astype(np.int32)
                        inb = (xi >= 0) & (xi < 80) & (yi >= 0) & (yi < 64)
                        elev[yi[inb], xi[inb]] = True
                sem = stgt.get("semantics")
                occ = stgt.get("occupancy")
                if sem is not None and occ is not None:
                    dyn_bev = ((occ == 1) & (sem > 0)).any(axis=0)
                    if dyn_bev.any():
                        gx0, gx1 = float(config["grid"]["x"][0]), float(config["grid"]["x"][1])
                        gy0, gy1 = float(config["grid"]["y"][0]), float(config["grid"]["y"][1])
                        dyi, dxi = np.nonzero(dyn_bev)
                        dxm = gx0 + (dxi + 0.5) / dyn_bev.shape[1] * (gx1 - gx0)
                        dym = gy0 + (dyi + 0.5) / dyn_bev.shape[0] * (gy1 - gy0)
                        xi_d = ((dxm + 6.0) / 0.15).astype(np.int32)
                        yi_d = ((dym + 4.0) / 0.125).astype(np.int32)
                        inb_d = (xi_d >= 0) & (xi_d < 80) & (yi_d >= 0) & (yi_d < 64)
                        elev[yi_d[inb_d], xi_d[inb_d]] = True

            blocked = binary_dilation(elev, iterations=2)
            xi_p = np.clip(((pts_xy[:, 0] + 6.0) / 0.15).astype(np.int32), 0, 79)
            yi_p = np.clip(((pts_xy[:, 1] + 4.0) / 0.125).astype(np.int32), 0, 63)
            flat_masks.append(~blocked[yi_p, xi_p])

            f_imgs = []
            for rel in fr["images"]:
                with Image.open(child_path(raw_root, rel)) as pil_im:
                    arr = np.asarray(pil_im.convert("RGB").resize((w_cal, h_cal), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
                f_imgs.append(np.transpose(arr, (2, 0, 1)))
            flat = torch.from_numpy(np.stack(f_imgs, axis=0)).to(device)
            luma = 0.299 * flat[:, 0:1] + 0.587 * flat[:, 1:2] + 0.114 * flat[:, 2:3]
            yell = flat[:, 0:1] - flat[:, 2:3]
            hp = (luma - _gaussian_blur_2d(luma, 19, 4.0)) * 6.0
            hp_y = (yell - _gaussian_blur_2d(yell, 19, 4.0)) * 8.0
            gx = F.pad(hp[:, :, :, 1:] - hp[:, :, :, :-1], (0, 1, 0, 0))
            gy = F.pad(hp[:, :, 1:, :] - hp[:, :, :-1, :], (0, 0, 0, 1))
            grad = torch.sqrt(gx * gx + gy * gy + 1e-6)
            feat = torch.cat([hp, hp_y, luma, grad], dim=1)
            imgs_coarse[fi] = _gaussian_blur_2d(feat, 17, 4.0).to(pyr_dtype)
            imgs_med[fi] = _gaussian_blur_2d(feat, 9, 1.8).to(pyr_dtype)
            imgs_fine[fi] = _gaussian_blur_2d(feat, 5, 0.85).to(pyr_dtype)

    with torch.enable_grad():
        torch.manual_seed(0)
        flat_t = torch.from_numpy(np.stack(flat_masks, axis=0)).to(device)
        train_idx = torch.arange(0, f_cnt, 2, device=device)
        val_idx = torch.arange(1, f_cnt, 2, device=device)
        if len(val_idx) == 0:
            val_idx = train_idx

        pts_xy_t = torch.from_numpy(pts_xy).to(device)
        t_init = torch.stack([torch.from_numpy(rigid_matrix(c["T_cam_from_ego"], c["slot"]).astype(np.float32)) for c in cams]).to(device)
        t_ec_init = torch.linalg.inv(t_init)
        k_t = torch.stack([torch.tensor(c["K"], dtype=torch.float32) for c in cams]).to(device)
        dist_t = torch.stack([torch.tensor(c["distortion"][:4], dtype=torch.float32) for c in cams]).to(device)
        hw_t = torch.tensor([c["image_hw"] for c in cams], dtype=torch.float32, device=device)
        vc_t = torch.tensor([c.get("valid_circle", [0.0, 0.0, -1.0]) for c in cams], dtype=torch.float32, device=device)
        max_th_t = torch.tensor([np.deg2rad(min(c["max_theta_deg"], 92.0)) for c in cams], dtype=torch.float32, device=device)

        def loss_fn(pyr, frame_idx, rpy_val, dz_val, dxy_val, gp_ab):
            pyr_sub = pyr[frame_idx]
            flat_sub = flat_t[frame_idx]
            sub_cnt = len(frame_idx)
            pz = gp_ab[0] * pts_xy_t[:, 0] + gp_ab[1] * pts_xy_t[:, 1] + pc
            pts3 = torch.cat([pts_xy_t, pz[:, None]], dim=-1)
            r_d = _euler_xyz_torch(rpy_val)
            dt_val = torch.cat([dxy_val, dz_val[:, None]], dim=-1)
            samples, valids = [], []
            for ci in range(4):
                orig = t_ec_init[ci, :3, 3] + (dt_val[ci] if ci > 0 else torch.zeros(3, device=device))
                r_ce = r_d[ci] @ t_init[ci, :3, :3]
                xyz_c = (pts3 - orig[None, :]) @ r_ce.T
                rad = torch.hypot(xyz_c[:, 0], xyz_c[:, 1]).clamp_min(1e-6)
                theta = torch.atan2(rad, xyz_c[:, 2])
                t2 = theta * theta
                kd = dist_t[ci]
                r_dist = theta * (1.0 + t2 * (kd[0] + t2 * (kd[1] + t2 * (kd[2] + t2 * kd[3]))))
                uv_p = xyz_c[:, :2] * (r_dist / rad)[:, None]
                km = k_t[ci]
                u = uv_p[:, 0] * km[0, 0] + uv_p[:, 1] * km[0, 1] + km[0, 2]
                v = uv_p[:, 1] * km[1, 1] + km[1, 2]
                h, w = hw_t[ci, 0], hw_t[ci, 1]
                ok = (theta <= max_th_t[ci]) & (u >= 20) & (u < w - 20) & (v >= 20) & (v < h - 20)
                if vc_t[ci, 2] > 0:
                    ok = ok & (torch.hypot(u - vc_t[ci, 0], v - vc_t[ci, 1]) < vc_t[ci, 2] - 20.0)
                grid = torch.stack([2.0 * (u + 0.5) / w - 1.0, 2.0 * (v + 0.5) / h - 1.0], dim=-1).view(1, 1, n_pts, 2).expand(sub_cnt, -1, -1, -1)
                feat = F.grid_sample(pyr_sub[:, ci].float(), grid, mode="bilinear", align_corners=False)
                samples.append(feat[:, :, 0, :])
                valids.append(ok)
            pair_losses = []
            for s0, s1, ca, cb in zone_slices:
                ok_p = (valids[ca][s0:s1] & valids[cb][s0:s1])[None, :]
                fa = samples[ca][:, :, s0:s1]
                fb = samples[cb][:, :, s0:s1]
                m = flat_sub[:, s0:s1] & ok_p & (fa[:, 2] > 0.14) & (fb[:, 2] > 0.14)
                z_losses = []
                for fi in range(sub_cnt):
                    mf = m[fi]
                    if mf.sum() < 60:
                        continue
                    ha, hb = fa[fi, 0][mf], fb[fi, 0][mf]
                    ya, yb = fa[fi, 1][mf], fb[fi, 1][mf]
                    ga, gb = fa[fi, 3][mf], fb[fi, 3][mf]
                    ha0, hb0 = ha - ha.mean(), hb - hb.mean()
                    ya0, yb0 = ya - ya.mean(), yb - yb.mean()
                    ga0, gb0 = ga - ga.mean(), gb - gb.mean()
                    c_hp = (ha0 * hb0).mean() / (torch.sqrt((ha0 * ha0).mean() + 1e-4) * torch.sqrt((hb0 * hb0).mean() + 1e-4))
                    c_hy = (ya0 * yb0).mean() / (torch.sqrt((ya0 * ya0).mean() + 1e-4) * torch.sqrt((yb0 * yb0).mean() + 1e-4))
                    c_gr = (ga0 * gb0).mean() / (torch.sqrt((ga0 * ga0).mean() + 1e-4) * torch.sqrt((gb0 * gb0).mean() + 1e-4))
                    z_losses.append(0.50 * (1.0 - c_hp) + 0.30 * (1.0 - c_hy) + 0.20 * (1.0 - c_gr))
                if z_losses:
                    st = torch.stack(z_losses)
                    k_keep = max(1, int(len(z_losses) * (0.75 if not has_lidar_targets else 0.90)))
                    vals, _ = torch.topk(st, k_keep, largest=False)
                    pair_losses.append(vals.mean())
            if not pair_losses:
                return (rpy_val * 0.0).sum()
            return sum(pair_losses) / len(pair_losses)

        gp_init = torch.tensor([pa, pb], dtype=torch.float32, device=device)
        with torch.no_grad():
            zero_r = torch.zeros((4, 3), device=device)
            zero_dz = torch.zeros(4, device=device)
            zero_xy = torch.zeros((4, 2), device=device)
            val_before = float(loss_fn(imgs_fine, val_idx, zero_r, zero_dz, zero_xy, gp_init).item())
            best_sdz, best_s_loss = 0.0, float("inf")
            for sdz in (-0.02, 0.0, 0.02, 0.04, 0.06, 0.08):
                dz_probe = torch.tensor([0.0, 0.0, sdz, sdz], device=device)
                s_loss = float(loss_fn(imgs_coarse, train_idx, zero_r, dz_probe, zero_xy, gp_init).item())
                if s_loss < best_s_loss:
                    best_s_loss = s_loss
                    best_sdz = sdz
            init_dz = torch.tensor([0.0, 0.0, best_sdz, best_sdz], device=device)
            val_s1 = float(loss_fn(imgs_fine, val_idx, zero_r, init_dz, zero_xy, gp_init).item())

        # Physical camera height bounds: front=0 (anchor), rear +-3cm, mirrors +-9cm
        dz_scale = torch.tensor([0.0, 0.032, 0.090, 0.090], device=device)
        d_rpy_raw = torch.zeros((4, 3), device=device, requires_grad=True)
        d_dz_raw = torch.atanh(torch.clamp(init_dz / dz_scale.clamp_min(1e-3), -0.95, 0.95)).clone().detach().requires_grad_(True)
        d_xy_raw = torch.zeros((4, 2), device=device, requires_grad=True)
        gp_raw = torch.zeros(2, device=device, requires_grad=True)
        params = [d_rpy_raw, d_dz_raw, d_xy_raw] + ([gp_raw] if optimize_gp and f_cnt >= 10 else [])
        opt = torch.optim.Adam(params, lr=0.015)

        base_rpy = torch.zeros((4, 3), device=device)
        base_dxy = torch.zeros((4, 2), device=device)
        rot_w = 0.25 if f_cnt >= 10 else 2.20
        stages = (
            (imgs_coarse, 55, 0.014, float(np.deg2rad(3.5 if f_cnt >= 10 else 1.5)), 0.035),
            (imgs_med, 65, 0.008, float(np.deg2rad(1.4 if f_cnt >= 10 else 0.8)), 0.016),
            (imgs_fine, 85, 0.004, float(np.deg2rad(0.7 if f_cnt >= 10 else 0.5)), 0.008),
        )
        for stage_imgs, n_steps, lr, max_r, max_xy in stages:
            for pg in opt.param_groups:
                pg["lr"] = lr
            for _ in range(n_steps):
                opt.zero_grad()
                rpy = base_rpy + max_r * torch.tanh(d_rpy_raw)
                dz = dz_scale * torch.tanh(d_dz_raw)
                dxy = (base_dxy + max_xy * torch.tanh(d_xy_raw)) if f_cnt >= 10 else zero_xy
                gp_ab = (gp_init + 0.012 * torch.tanh(gp_raw)) if (optimize_gp and f_cnt >= 10) else gp_init
                sym_z = (dz[2] - dz[3]) ** 2
                reg = (
                    rot_w * (rpy ** 2).sum()
                    + 0.15 * ((dz - init_dz) ** 2).sum()
                    + 1.20 * sym_z
                    + 3.0 * (dxy ** 2).sum()
                    + 6.0 * ((gp_ab - gp_init) ** 2).sum()
                )
                loss = loss_fn(stage_imgs, train_idx, rpy, dz, dxy, gp_ab) + reg
                loss.backward()
                opt.step()
            with torch.no_grad():
                base_rpy = (base_rpy + max_r * torch.tanh(d_rpy_raw)).detach()
                if f_cnt >= 10:
                    base_dxy = (base_dxy + max_xy * torch.tanh(d_xy_raw)).detach()
                d_rpy_raw.zero_()
                d_xy_raw.zero_()

        with torch.no_grad():
            rpy_f = base_rpy
            dz_f = dz_scale * torch.tanh(d_dz_raw)
            dz_f[0] = 0.0
            dxy_f = base_dxy if f_cnt >= 10 else zero_xy.clone()
            dxy_f[0] = 0.0
            gp_f = (gp_init + 0.012 * torch.tanh(gp_raw)) if (optimize_gp and f_cnt >= 10) else gp_init
            val_s2 = float(loss_fn(imgs_fine, val_idx, rpy_f, dz_f, dxy_f, gp_f).item())
            if val_s1 <= val_s2:
                val_after = val_s1
                min_gain = 0.015
                rpy_f.zero_()
                dxy_f.zero_()
                dz_f = init_dz
                gp_f = gp_init
            else:
                val_after = val_s2
                min_gain = 0.025
            r_d_np = _euler_xyz_torch(rpy_f).cpu().numpy()
            dt_np = torch.cat([dxy_f, dz_f[:, None]], dim=-1).cpu().numpy()
            dt_np[0] = 0.0
            rot_norms_deg = np.rad2deg(np.linalg.norm(rpy_f.cpu().numpy(), axis=1)).round(2).tolist()
            dz_cm = [round(float(v * 100.0), 2) for v in dt_np[:, 2]]
            gp_est = [round(float(gp_f[0].item()), 6), round(float(gp_f[1].item()), 6), round(float(pc), 5)]

    refined = copy.deepcopy(rig)
    # Cross-frame held-out validation gate selects cleanest generalization (Stage 1 or Stage 2)
    if val_before > 1e-4 and (val_before - val_after) / val_before >= min_gain:
        for ci, cam in enumerate(refined["cameras"]):
            t_ce0 = rigid_matrix(cam["T_cam_from_ego"], cam["slot"])
            t_ec0 = np.linalg.inv(t_ce0)
            orig_new = t_ec0[:3, 3] + dt_np[ci]
            r_ce_new = r_d_np[ci] @ t_ce0[:3, :3]
            t_new = np.eye(4, dtype=np.float64)
            t_new[:3, :3] = r_ce_new
            t_new[:3, 3] = -r_ce_new @ orig_new
            cam["T_cam_from_ego"] = t_new.tolist()
        refined["calib_dz_cm"] = dz_cm
        if optimize_gp:
            refined["estimated_ground_plane"] = gp_est
        return refined, rot_norms_deg
    return refined, [0.0] * len(cams)


def build_standard_rig_calibration(
    rig, manifest_rows, raw_root, mode="auto", load_targets_fn=None, sample_path_fn=None,
    prepared_root=None, config=None, depth_centers=None, device=None, poses=None,
):
    """Unified 6-DoF rig calibration producing a standard RigCalibrationProfile (LiDAR-assisted or targetless)."""
    if mode not in {"auto", "lidar", "camera-only"}:
        raise ValueError(f"Unsupported calibration mode: {mode}")
    has_lidar = any(row.get("lidar") for row in manifest_rows)
    if mode == "lidar" and not has_lidar and load_targets_fn is None:
        raise ValueError("LiDAR calibration mode requested but no LiDAR scans or targets are available")
    use_lidar = (mode == "lidar") or (mode == "auto" and (has_lidar or load_targets_fn is not None))
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rig_vc = estimate_valid_circles(rig, manifest_rows, raw_root)
    base_gz = float(rig.get("ground_height_m", 0.0))
    road_plane = [0.0, 0.0, base_gz]
    plane_quality = {"source": "rig_nominal"}
    if use_lidar:
        measured = fit_lidar_road_plane(rig_vc, manifest_rows, raw_root) if has_lidar else None
        if measured is not None:
            road_plane, plane_quality = measured
        elif load_targets_fn is not None and sample_path_fn is not None and prepared_root is not None and config is not None and depth_centers is not None:
            road_plane, _ = estimate_road_plane(
                rig_vc, manifest_rows, load_targets_fn, sample_path_fn, prepared_root, config, depth_centers
            )
            plane_quality = {"source": "prepared_depth"}

    refined, rot_deg = calibrate_rig_extrinsics_photometric(
        rig_vc, manifest_rows, raw_root,
        load_targets_fn=load_targets_fn if use_lidar else None,
        sample_path_fn=sample_path_fn if use_lidar else None,
        prepared_root=prepared_root if use_lidar else None,
        config=config if use_lidar else None,
        depth_centers=depth_centers if use_lidar else None,
        ground_z=road_plane[2], device=device, ground_plane=road_plane if use_lidar else None, poses=poses,
    )
    if not use_lidar and "estimated_ground_plane" in refined:
        road_plane = refined["estimated_ground_plane"]
        plane_quality = {"source": "camera_only_zncc_ba"}
    return {
        "schema_version": 1,
        "calibration_mode": "lidar_assisted" if use_lidar else "targetless_camera_only",
        "rig_id": refined.get("rig_id", rig.get("rig_id", "default-rig")),
        "ground_plane": [round(float(v), 6) for v in road_plane],
        "road_plane_validation": plane_quality,
        "vehicle_spec": export_vehicle_spec(refined, ground_plane=road_plane),
        "bowl_spec": export_bowl_spec(),
        "rig": refined,
        "camera_calib": export_camera_calib(refined),
        "quality": {
            "rot_deg": rot_deg,
            "calib_dz_cm": refined.get("calib_dz_cm", [0.0] * len(refined["cameras"])),
            "seam_parallax_cm": measure_seam_parallax_report(refined, road_plane),
        },
    }


def load_calibrated_rig_profile(path):
    """Load and validate a standard RigCalibrationProfile JSON."""
    from omnishield.io import load_json

    profile = load_json(path)
    if profile.get("schema_version") != 1 or "rig" not in profile or "cameras" not in profile["rig"]:
        raise ValueError(f"Invalid RigCalibrationProfile schema: {path}")
    return profile


def export_camera_calib(rig):
    """Export KB4 + closed-form 6-param Double Sphere (DS) intrinsics & extrinsics for WebGL."""
    out = []
    for cam in rig["cameras"]:
        sh, sw = cam["image_hw"]
        t_cam = rigid_matrix(cam["T_cam_from_ego"], cam["slot"])
        ego_from_cam = np.linalg.inv(t_cam)
        k = np.asarray(cam["K"], dtype=np.float64).reshape(3, 3)
        dist = [float(v) for v in cam.get("distortion", [0.0, 0.0, 0.0, 0.0])]
        vc = [float(v) for v in cam.get("valid_circle", [0.0, 0.0, -1.0])]
        ds_fit = fit_double_sphere_from_camera(cam)
        out.append({
            "slot": cam["slot"],
            "hw": [int(sh), int(sw)],
            "t_cam_from_ego": [round(float(v), 7) for v in t_cam.ravel()],
            "cam_origin": [round(float(v), 5) for v in ego_from_cam[:3, 3]],
            "cam_xy": [round(float(ego_from_cam[0, 3]), 5), round(float(ego_from_cam[1, 3]), 5)],
            "cam_fwd": [round(float(ego_from_cam[0, 2]), 5), round(float(ego_from_cam[1, 2]), 5)],
            "fxy": [round(float(k[0, 0]), 5), round(float(k[1, 1]), 5)],
            "cxy": [round(float(k[0, 2]), 5), round(float(k[1, 2]), 5)],
            "skew": round(float(k[0, 1]), 6),
            "k": [round(v, 7) for v in dist],
            "ds": ds_fit["params"],
            "ds_rmse_px": ds_fit["rmse_px"],
            "max_theta_rad": round(float(np.deg2rad(cam["max_theta_deg"])), 6),
            "valid_circle": vc,
        })
    return out


def bowl_seam_calibration(rig, calibs, ground_z, rings=20, candidates=25, ground_plane=None):
    """Project candidate seam locations on the EXACT unified 3D Bowl surface into adjacent camera pairs."""
    by_slot = {cam["slot"]: i for i, cam in enumerate(rig["cameras"])}
    order = [by_slot[slot] for slot in ("front", "left", "rear", "right")]
    pa, pb, pc = [float(v) for v in ground_plane] if ground_plane is not None else (0.0, 0.0, float(ground_z))
    max_off = 0.26
    offsets = np.linspace(-max_off, max_off, candidates)
    displacements = np.array([-0.24, -0.12, 0.0, 0.12, 0.24])
    ring = np.linspace(0.0, 1.0, rings)[:, None]
    wall = np.clip((ring - BOWL_FLAT_SPLIT) / (1.0 - BOWL_FLAT_SPLIT), 0.0, 1.0)
    # Match shader's camera-anchored corner diagonals: (x - x_shift) / x_scale == +-y
    corner = float(np.arctan2(1.0, BOWL_SEAM_X_SCALE))
    corner_angles = [corner, np.pi - corner, -np.pi + corner, -corner]
    pairs = []
    for (a, b), mid in zip(zip(order, order[1:] + order[:1], strict=True), corner_angles, strict=True):
        # Evaluate candidate angles around the anchored corner diagonal, mapped to ego bowl geometry
        angle = mid + offsets
        ca, sa = np.cos(angle), np.sin(angle)
        flat = _superellipse_radius(ca, sa, BOWL_FLAT_RX, BOWL_FLAT_RY)
        pad = 1.0 / np.hypot(ca / BOWL_PAD_X, sa / BOWL_PAD_Y)
        radius = np.where(ring <= BOWL_FLAT_SPLIT, 0.05 + (flat - 0.05) * ring / BOWL_FLAT_SPLIT,
                          flat + pad * wall)
        # Apply seam anchor shift cleanly on ego x so DP evaluates the true seam locus!
        pts_x = radius * ca + BOWL_SEAM_X_SHIFT * np.clip(ring / BOWL_FLAT_SPLIT, 0.0, 1.0)
        pts_y = radius * sa
        base_z = pa * pts_x + pb * pts_y + pc
        z = base_z + BOWL_WALL_HEIGHT * (wall ** BOWL_WALL_POWER)
        points = np.stack([pts_x, pts_y, z], axis=-1)
        dr = pad
        dz = np.broadcast_to(BOWL_WALL_HEIGHT * BOWL_WALL_POWER * np.maximum(wall, 1e-4) ** (BOWL_WALL_POWER - 1.0), radius.shape)
        normal = np.stack([-dz * ca, -dz * sa, np.broadcast_to(dr, radius.shape)], axis=-1)
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6)
        hypotheses = (points[None] + displacements[:, None, None, None] * normal).reshape(-1, 3)
        maps, refinement = [], []
        for i in (a, b):
            calib = calibs[i]
            h, w = calib["hw"]
            uv, _, valid = project(hypotheses, rig["cameras"][i], (h, w))
            cx, cy, circle = calib["valid_circle"]
            if circle > 0:
                valid &= np.hypot(uv[:, 0] - cx, uv[:, 1] - cy) < circle - 10
            delta = hypotheses[:, :2] - np.asarray(calib["cam_xy"])
            fwd2 = np.asarray(calib["cam_fwd"])
            fwd2 = fwd2 / max(float(np.linalg.norm(fwd2)), 1e-5)
            direction = delta @ fwd2 / np.maximum(np.linalg.norm(delta, axis=1), 1e-5)
            # Allow full 190-deg fisheye corner overlap (matching shader's -0.26 threshold)
            valid &= (direction > -0.26) & ~ego_mask(hypotheses, rig)
            valid &= (uv >= 9).all(axis=1) & (uv < [w - 9, h - 9]).all(axis=1)
            uv = ((uv + 0.5) / [w, h]).round(5).reshape(len(displacements), rings * candidates, 2)
            valid = valid.reshape(len(displacements), rings * candidates)
            projected = [[xy.tolist() if yes else None for xy, yes in zip(layer, mask, strict=True)]
                         for layer, mask in zip(uv, valid, strict=True)]
            maps.append(projected[len(displacements) // 2])
            refinement.append(projected)
        pairs.append({"cameras": [a, b], "angle": round(float(mid), 6), "uv": maps, "refine_uv": refinement})
    return {"rings": rings, "max_offset": max_off, "offsets": offsets.round(6).tolist(),
            "displacements": displacements.tolist(), "pairs": pairs}


FOREGROUND_SECTORS = 128
POLAR_HEIGHT_RINGS = 48
POLAR_R_MIN = 1.15
POLAR_R_MAX = 7.60
POLAR_MAX_HEIGHT = 0.36


def _polar_height_grid(points, ground_plane=(0.0, 0.0, 0.0)):
    """Build radially non-decreasing 2.5D curb/median elevation Z(sector, ring) in mm (zero ray folding)."""
    if len(points) == 0:
        return np.zeros((FOREGROUND_SECTORS, POLAR_HEIGHT_RINGS), dtype="<u2")
    pa, pb, pc = [float(v) for v in ground_plane]
    rad = np.hypot(points[:, 0], points[:, 1])
    h = points[:, 2] - (pa * points[:, 0] + pb * points[:, 1] + pc)
    m = (rad >= POLAR_R_MIN) & (rad < POLAR_R_MAX) & (h >= 0.10) & (h <= 0.85)
    m &= ~((np.abs(points[:, 1]) < 1.32) & (np.abs(points[:, 0]) < 5.80))
    if not m.any():
        return np.zeros((FOREGROUND_SECTORS, POLAR_HEIGHT_RINGS), dtype="<u2")
    pts_m = points[m]
    rad_m = rad[m]
    h_m = np.minimum(h[m], POLAR_MAX_HEIGHT)

    # 1. Longitudinal Corridor Curb Profile: interpolates continuous left/right curbs across grazing corner seams
    #    and places 100% of curb height right on the inner curb face (eliminates 50% Gaussian step-halving!).
    angs_g = (np.arange(FOREGROUND_SECTORS, dtype=np.float32) + 0.5) * (2.0 * np.pi / FOREGROUND_SECTORS)
    rads_g = POLAR_R_MIN + (np.arange(POLAR_HEIGHT_RINGS, dtype=np.float32) + 0.5) / POLAR_HEIGHT_RINGS * (POLAR_R_MAX - POLAR_R_MIN)
    gx = np.cos(angs_g)[:, None] * rads_g[None, :]
    gy = np.sin(angs_g)[:, None] * rads_g[None, :]
    corridor_h = np.zeros((FOREGROUND_SECTORS, POLAR_HEIGHT_RINGS), dtype=np.float32)

    x_bins = np.linspace(-8.0, 8.0, 33)
    x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
    for side in (1.0, -1.0):
        lat_pts = side * pts_m[:, 1]
        sm = (lat_pts >= 1.45) & (lat_pts <= 4.30) & (np.abs(pts_m[:, 0]) < 8.0) & (pts_m[:, 2] - (pa * pts_m[:, 0] + pb * pts_m[:, 1] + pc) <= 0.78)
        if int(sm.sum()) < 30:
            continue
        sx = pts_m[sm, 0]
        sy = lat_pts[sm]
        sh = h_m[sm]
        bi = np.clip(np.digitize(sx, x_bins) - 1, 0, len(x_centers) - 1)
        valid_b, y_b, h_b = [], [], []
        for b_idx in range(len(x_centers)):
            bm = bi == b_idx
            if int(bm.sum()) < 4:
                continue
            yb = float(np.percentile(sy[bm], 12))
            face_m = bm & (sy <= yb + 0.45)
            hb = float(np.clip(np.percentile(sh[face_m] if face_m.any() else sh[bm], 68), 0.14, POLAR_MAX_HEIGHT))
            valid_b.append(b_idx)
            y_b.append(yb)
            h_b.append(hb)
        if len(valid_b) < 5:
            continue
        valid_b = np.asarray(valid_b, dtype=np.int32)
        yb_arr = median_filter(np.asarray(y_b, dtype=np.float64), size=min(5, len(y_b)), mode="nearest")
        hb_arr = median_filter(np.asarray(h_b, dtype=np.float64), size=min(5, len(h_b)), mode="nearest")
        # Reject non-curb scattered clutter if lateral jitter across bins exceeds 0.45m
        if float(np.std(yb_arr - gaussian_filter1d(yb_arr, sigma=1.5, mode="nearest"))) > 0.45:
            continue
        xc_val = x_centers[valid_b]
        x_min_supp = float(xc_val.min()) - 2.6
        x_max_supp = float(xc_val.max()) + 2.6
        y_interp = gaussian_filter1d(np.interp(x_centers, xc_val, yb_arr), sigma=2.2, mode="nearest")
        h_interp = gaussian_filter1d(np.interp(x_centers, xc_val, hb_arr), sigma=2.4, mode="nearest")

        lat_g = side * gy
        yc_g = np.interp(gx, x_centers, y_interp)
        hc_g = np.interp(gx, x_centers, h_interp)
        x_fade = np.clip((gx - (x_min_supp - 1.0)) / 1.0, 0.0, 1.0) * np.clip(((x_max_supp + 1.0) - gx) / 1.0, 0.0, 1.0)
        # Rise smoothly across [y_curb - 0.30m .. y_curb - 0.02m] so full curb height sits solidly at the curb face!
        u_ramp = np.clip((lat_g - (yc_g - 0.30)) / 0.28, 0.0, 1.0)
        ramp = u_ramp * u_ramp * (3.0 - 2.0 * u_ramp)
        corridor_h = np.maximum(corridor_h, np.where(lat_g > 1.25, hc_g * ramp * x_fade, 0.0).astype(np.float32))

    # 2. Direct Polar Cell Grid for arbitrary transverse islands/medians + inward 1-bin edge preservation
    ang = np.mod(np.arctan2(pts_m[:, 1], pts_m[:, 0]), 2.0 * np.pi)
    si = np.floor(ang * (FOREGROUND_SECTORS / (2.0 * np.pi))).astype(np.intp) % FOREGROUND_SECTORS
    ri = np.clip(
        np.floor((rad_m - POLAR_R_MIN) / (POLAR_R_MAX - POLAR_R_MIN) * POLAR_HEIGHT_RINGS).astype(np.intp),
        0,
        POLAR_HEIGHT_RINGS - 1,
    )
    cnt = np.zeros((FOREGROUND_SECTORS, POLAR_HEIGHT_RINGS), dtype=np.int32)
    hsum = np.zeros((FOREGROUND_SECTORS, POLAR_HEIGHT_RINGS), dtype=np.float32)
    np.add.at(cnt, (si, ri), 1)
    np.add.at(hsum, (si, ri), h_m)
    cnt_s = cnt + np.roll(cnt, 1, axis=0) + np.roll(cnt, -1, axis=0)
    hsum_s = hsum + np.roll(hsum, 1, axis=0) + np.roll(hsum, -1, axis=0)
    min_pts = 4 if len(points) < 35000 else 6
    cell_h = np.where(cnt_s >= min_pts, hsum_s / np.maximum(cnt_s, 1), 0.0)
    cell_h = median_filter(cell_h, size=(3, 1), mode="wrap")
    # Shift 1 radial bin (~13cm) inward before smoothing so symmetric Gaussian preserves 100% height at curb face
    cell_in = np.zeros_like(cell_h)
    cell_in[:, :-1] = np.maximum(cell_h[:, :-1], cell_h[:, 1:])
    cell_in[:, -1] = cell_h[:, -1]
    cell_in[:, :2] = 0.0  # Keep innermost ring near ego flat

    fused = np.maximum(corridor_h, cell_in)
    mono_h = np.maximum.accumulate(fused, axis=1)
    smooth_h = gaussian_filter(mono_h, sigma=(1.8, 0.9), mode=("wrap", "nearest"))
    smooth_h = np.clip(np.maximum.accumulate(smooth_h, axis=1), 0.0, POLAR_MAX_HEIGHT)
    return np.rint(1000.0 * smooth_h).astype("<u2")


def _wall_ranges(points, ground_z, ground_plane=None):
    """Estimate smooth azimuthal radius r(theta) to the nearest barrier/curb above the road plane."""
    if len(points) == 0:
        return np.zeros(FOREGROUND_SECTORS, dtype="<u2")
    radius = np.hypot(points[:, 0], points[:, 1])
    if ground_plane is not None:
        a, b, c0 = [float(v) for v in ground_plane]
        height = points[:, 2] - (a * points[:, 0] + b * points[:, 1] + c0)
    else:
        height = points[:, 2] - float(ground_z)
    angle = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2 * np.pi)
    angular_bin = np.floor(angle * (FOREGROUND_SECTORS / (2 * np.pi))).astype(np.intp) % FOREGROUND_SECTORS
    radial_step = 0.10
    n_rbins = 115
    radial_bin = np.minimum((radius / radial_step).astype(np.intp), n_rbins - 1)
    hist = np.zeros((FOREGROUND_SECTORS, n_rbins), dtype=np.int32)
    m = (height >= 0.22) & (height <= 2.80) & (radius >= 1.50) & (radius < 11.0)
    np.add.at(hist, (angular_bin[m], radial_bin[m]), 1)
    hist = (
        hist
        + np.roll(hist, 1, axis=0) + np.roll(hist, -1, axis=0)
        + np.roll(hist, 2, axis=0) + np.roll(hist, -2, axis=0)
    )
    hist[:, 1:] += hist[:, :-1].copy()
    sparse = len(points) < 35000
    supported = hist >= (6 if sparse else 12)
    supported[:, :14] = False
    valid = supported.any(axis=1)
    nearest = np.maximum(np.argmax(supported, axis=1) - 0.5, 0) * radial_step
    smooth_num = gaussian_filter1d(np.where(valid, nearest, 0.0), sigma=4.5, mode="wrap")
    coverage = gaussian_filter1d(valid.astype(np.float64), sigma=4.5, mode="wrap")
    return np.rint(100 * np.where(coverage >= 0.45, smooth_num / np.maximum(coverage, 1e-5), 0)).astype("<u2")


def pack_foreground_ranges(points, rig, height=90, width=160, ground_z=None, static_points=None, ground_plane=None):
    """Build radial range images for LiDAR-supported surfaces above the road plane."""
    if ground_plane is not None:
        pa, pb, pc = [float(v) for v in ground_plane]
        ground_z = pc
    else:
        ground_z = float(rig.get("ground_height_m", 0.0)) if ground_z is None else float(ground_z)
        pa, pb, pc = 0.0, 0.0, ground_z
    sparse = len(points) < 35000
    keep = np.isfinite(points).all(axis=1) & ~ego_mask(points, rig)
    keep &= (np.hypot(points[:, 0], points[:, 1]) < 10.0)
    pts_rz = pa * points[:, 0] + pb * points[:, 1] + pc
    keep &= (points[:, 2] >= pts_rz - 0.30) & (points[:, 2] <= pts_rz + 3.3)
    points = points[keep]
    pts_rz = pa * points[:, 0] + pb * points[:, 1] + pc
    if static_points is None:
        wall_pts = points
    else:
        sk = np.isfinite(static_points).all(axis=1) & ~ego_mask(static_points, rig)
        sk &= (np.hypot(static_points[:, 0], static_points[:, 1]) < 10.0)
        s_rz = pa * static_points[:, 0] + pb * static_points[:, 1] + pc
        sk &= (static_points[:, 2] >= s_rz - 0.30) & (static_points[:, 2] <= s_rz + 3.3)
        wall_pts = static_points[sk]
    elevated = points[:, 2] > pts_rz + 0.15
    layers = []
    for cam in rig["cameras"]:
        src_h, src_w = cam["image_hw"]
        uv, ranges, valid = project(points, cam, (src_h, src_w))
        valid &= (ranges >= 0.35) & (ranges <= 12.0)
        uv = uv[valid]
        pixel = np.floor((uv + 0.5) * [width / src_w, height / src_h]).astype(np.intp)
        flat_index = pixel[:, 1] * width + pixel[:, 0]
        nearest = np.full(height * width, np.inf, np.float32)
        foreground = np.full_like(nearest, np.inf)
        np.minimum.at(nearest, flat_index, ranges[valid])
        front = elevated[valid]
        np.minimum.at(foreground, flat_index[front], ranges[valid][front])
        foreground[foreground > nearest + 0.04] = np.inf
        foreground = foreground.reshape(height, width)
        known = np.isfinite(foreground)
        if known.any():
            distance, neighbor = distance_transform_edt(~known, return_indices=True)
            local = foreground[neighbor[0], neighbor[1]]
            max_gap = 2.25 if sparse else 1.45
            filled = (distance <= max_gap) & (local <= nearest.reshape(height, width) + 0.08)
            foreground = np.where(filled, local, 0)
            foreground = median_filter(foreground, size=3, mode="constant")
        else:
            foreground.fill(0)
        layers.append(np.rint(100 * foreground).astype("<u2"))
    header = np.asarray([width, height], dtype="<u4")
    wall_bytes = _wall_ranges(wall_pts, ground_z, ground_plane=(pa, pb, pc)).tobytes()
    polar_bytes = _polar_height_grid(wall_pts, ground_plane=(pa, pb, pc)).tobytes()
    return header.tobytes() + np.stack(layers).tobytes() + wall_bytes + polar_bytes


def measure_seam_parallax_report(rig, ground_plane=(0.0, 0.0, 0.0)):
    """Measure virtual cone projection disparity (cm on fixed bowl) across 4 corner seams."""
    pa, pb, pc = [float(v) for v in ground_plane]
    by_slot = {cam["slot"]: cam for cam in rig["cameras"]}
    cam_origs = {s: np.linalg.inv(rigid_matrix(by_slot[s]["T_cam_from_ego"], s))[:3, 3] for s in ("front", "rear", "left", "right")}
    corner = float(np.arctan2(1.0, BOWL_SEAM_X_SCALE))
    seams = [
        ("FL", "front", "left", corner),
        ("FR", "front", "right", -corner),
        ("RL", "rear", "left", np.pi - corner),
        ("RR", "rear", "right", -(np.pi - corner)),
    ]

    def hit_bowl(orig, ray):
        ts = np.linspace(0.2, 18.0, 320)
        pts = orig[None] + ts[:, None] * ray[None]
        ang = np.arctan2(pts[:, 1], pts[:, 0])
        ca, sa = np.cos(ang), np.sin(ang)
        r = np.hypot(pts[:, 0], pts[:, 1])
        r_flat = _superellipse_radius(ca, sa, BOWL_FLAT_RX, BOWL_FLAT_RY)
        r_pad = 1.0 / np.hypot(ca / BOWL_PAD_X, sa / BOWL_PAD_Y)
        u = np.clip((r - r_flat) / np.maximum(r_pad, 0.5), 0.0, 1.0)
        bz = (pa * pts[:, 0] + pb * pts[:, 1] + pc) + np.where(r <= r_flat, 0.0, BOWL_WALL_HEIGHT * (u ** BOWL_WALL_POWER))
        diff = pts[:, 2] - bz
        idx = np.where(diff <= 0)[0]
        if len(idx) == 0:
            return None
        i = idx[0]
        if i == 0:
            return pts[0]
        frac = diff[i - 1] / max(diff[i - 1] - diff[i], 1e-6)
        return pts[i - 1] + frac * (pts[i] - pts[i - 1])

    spec = export_vehicle_spec(rig, ground_plane=ground_plane)
    rx_e = 0.5 * spec["length_m"]
    ry_e = 0.5 * spec["width_m"]
    report = {}
    for sname, sa_k, sb_k, sang in seams:
        report[sname] = {}
        for h_pole in (0.0, 0.25, 0.90, 1.50):
            vals = []
            for dist in (1.0, 2.0, 3.0, 5.0):
                r_ego = 1.0 / np.hypot(np.cos(sang) / rx_e, np.sin(sang) / ry_e)
                px = (r_ego + dist) * np.cos(sang) + BOWL_SEAM_X_SHIFT
                py = (r_ego + dist) * np.sin(sang)
                p_cone = np.array([px, py, pa * px + pb * py + pc + h_pole])
                ra = p_cone - cam_origs[sa_k]
                rb = p_cone - cam_origs[sb_k]
                ha = hit_bowl(cam_origs[sa_k], ra / np.linalg.norm(ra))
                hb = hit_bowl(cam_origs[sb_k], rb / np.linalg.norm(rb))
                vals.append(round(float(np.linalg.norm(ha - hb) * 100.0), 1) if (ha is not None and hb is not None) else None)
            report[sname][f"{h_pole:.2f}m"] = vals
    return report


class FeedForward3DGSDecoder:
    """Native Non-Pinhole Feed-Forward 3D Gaussian Splatting decoder via Double Sphere Ray-LUT."""

    def __init__(self, rig, calibs, model_hw, feature_stride=4, ground_z=0.0):
        self.rig = rig
        self.ground_z = float(ground_z)
        self.dh = model_hw[0] // feature_stride
        self.dw = model_hw[1] // feature_stride
        self.origins = []
        self.rays_ego = []
        self.d_ground = []
        self.ray_valid = []
        self.ang_step = []
        self.tex_uv = []

        # Precompute exact closed-form Double Sphere inverse rays pi^-1(u, v) on the depth feature grid
        for cam, cal in zip(rig["cameras"], calibs, strict=True):
            sh, sw = cam["image_hw"]
            lh, lw, top, left = letterbox_parameters((sh, sw), model_hw)
            uu = (np.arange(self.dw, dtype=np.float64) + 0.5) * feature_stride
            vv = (np.arange(self.dh, dtype=np.float64) + 0.5) * feature_stride
            u_grid, v_grid = np.meshgrid(uu, vv)
            in_img = (u_grid >= left) & (u_grid < left + lw) & (v_grid >= top) & (v_grid < top + lh)
            u_src = (u_grid - left) * (sw / max(lw, 1)) - 0.5
            v_src = (v_grid - top) * (sh / max(lh, 1)) - 0.5
            uv_src = np.stack([u_src, v_src], axis=-1)

            rays_cam, ds_ok = ds_unproject(uv_src, cal["ds"])
            # Reject rays past max_theta_deg
            cos_th = np.clip(rays_cam[..., 2], -1.0, 1.0)
            ok = in_img & ds_ok & (np.arccos(cos_th) <= cal["max_theta_rad"])
            if "valid_circle" in cam:
                vcx, vcy, vcr = cam["valid_circle"]
                ok &= np.hypot(u_src - vcx, v_src - vcy) <= max(vcr - 4.0, 1.0)

            t_cam = rigid_matrix(cam["T_cam_from_ego"], cam["slot"])
            ego_from_cam = np.linalg.inv(t_cam)
            rot = ego_from_cam[:3, :3]
            origin = ego_from_cam[:3, 3]
            rays_e = rays_cam @ rot.T

            dz = rays_e[..., 2]
            hits_ground = dz < -0.025
            dg = np.where(hits_ground, (self.ground_z - origin[2]) / np.where(hits_ground, dz, -1.0), 28.0)
            dg = np.clip(dg, 0.3, 28.0)

            self.origins.append(origin.astype(np.float32))
            self.rays_ego.append(rays_e.astype(np.float32))
            self.d_ground.append(dg.astype(np.float32))
            self.ray_valid.append(ok)
            tex_u = np.clip((u_src + 0.5) / sw, 0.0, 1.0)
            tex_v = np.clip(1.0 - (v_src + 0.5) / sh, 0.0, 1.0)
            self.tex_uv.append(np.stack([tex_u, tex_v], axis=-1).astype(np.float32))
            # Approximate angular step per depth-grid pixel for metric Gaussian scale
            fx_ds = max(float(cal["ds"][0]), 100.0)
            self.ang_step.append(float((sw / max(self.dw, 1)) / fx_ds))

    def decode_depth_surface_binary(self, depth_m, mask_bits, ground_plane=(0.0, 0.0, 0.0), gt_valid_mask=None):
        """Build depth-discontinuity-cut local 3D surface mesh with live camera UV & Z-buffer support."""
        pa, pb, pc = [float(v) for v in ground_plane]
        vert_chunks, idx_chunks = [], []
        vert_offset = 0
        for c in range(len(self.origins)):
            if not mask_bits[c]:
                continue
            d_raw = depth_m[c].astype(np.float32, copy=True)
            ok = self.ray_valid[c].copy()
            if gt_valid_mask is not None:
                has_lidar = gt_valid_mask[c] & (d_raw > 0.30) & (d_raw < 14.0)
                d_use = np.where(has_lidar, d_raw, 0.0)
                ok &= has_lidar
            else:
                d_use = median_filter(d_raw, size=3)
                ok &= (d_use > 0.35) & (d_use < 13.0)

            pts = self.origins[c][None, None, :] + d_use[..., None] * self.rays_ego[c]
            ex, ey, ez = pts[..., 0], pts[..., 1], pts[..., 2]
            road_z = pa * ex + pb * ey + pc
            h_road = ez - road_z
            not_ego = ~ego_mask(pts, self.rig)
            keep = ok & not_ego & (h_road >= 0.14) & (h_road <= 2.65) & (np.hypot(ex, ey) < 11.5)
            # Symmetric corner ownership with narrow overlap margin keeps upright objects intact without Z-fighting
            nx_e = ex / 1.48
            if c == 0:
                keep &= (nx_e >= np.abs(ey) * 0.80) & (ex > 0.0)
            elif c == 1:
                keep &= (-nx_e >= np.abs(ey) * 0.80) & (ex < 0.0)
            elif c == 2:
                keep &= (ey >= np.abs(nx_e) * 0.80) & (ey > 0.0)
            else:
                keep &= (-ey >= np.abs(nx_e) * 0.80) & (ey < 0.0)

            if int(keep.sum()) < 4:
                continue

            # Cut triangles cleanly at depth discontinuities (prevent rubber-sheet webs across gaps)
            k00, k01, k10, k11 = keep[:-1, :-1], keep[:-1, 1:], keep[1:, :-1], keep[1:, 1:]
            d00, d01, d10, d11 = d_use[:-1, :-1], d_use[:-1, 1:], d_use[1:, :-1], d_use[1:, 1:]
            p00, p01, p10, p11 = pts[:-1, :-1], pts[:-1, 1:], pts[1:, :-1], pts[1:, 1:]

            def tri_valid(ka, kb, kc, da, db, dc, pa_t, pb_t, pc_t):
                d_min = np.minimum(np.minimum(da, db), dc)
                d_max = np.maximum(np.maximum(da, db), dc)
                d_tol = np.maximum(0.28, 0.12 * d_min)
                e1 = np.linalg.norm(pb_t - pa_t, axis=-1)
                e2 = np.linalg.norm(pc_t - pb_t, axis=-1)
                e3 = np.linalg.norm(pa_t - pc_t, axis=-1)
                return ka & kb & kc & ((d_max - d_min) <= d_tol) & (np.maximum(np.maximum(e1, e2), e3) <= 0.65)

            t1 = tri_valid(k00, k10, k01, d00, d10, d01, p00, p10, p01)
            t2 = tri_valid(k01, k10, k11, d01, d10, d11, p01, p10, p11)
            if not (t1.any() or t2.any()):
                continue

            used_grid = np.zeros((self.dh, self.dw), dtype=bool)
            for mask_t, off_a, off_b, off_c in (
                (t1, (0, 0), (1, 0), (0, 1)),
                (t2, (0, 1), (1, 0), (1, 1)),
            ):
                if mask_t.any():
                    vy, vx = np.nonzero(mask_t)
                    used_grid[vy + off_a[0], vx + off_a[1]] = True
                    used_grid[vy + off_b[0], vx + off_b[1]] = True
                    used_grid[vy + off_c[0], vx + off_c[1]] = True

            vid_map = np.full((self.dh, self.dw), -1, dtype=np.int32)
            uy, ux = np.nonzero(used_grid)
            n_v = len(uy)
            vid_map[uy, ux] = np.arange(vert_offset, vert_offset + n_v, dtype=np.int32)

            p_u = pts[uy, ux]
            pos_3 = np.column_stack([-p_u[:, 1], p_u[:, 2], -p_u[:, 0]]).astype(np.float32)
            uv_u = self.tex_uv[c][uy, ux]
            cam_col = np.full((n_v, 1), float(c), dtype=np.float32)
            vert_chunks.append(np.column_stack([pos_3, cam_col, uv_u]).astype("<f4"))

            for mask_t, off_a, off_b, off_c in (
                (t1, (0, 0), (1, 0), (0, 1)),
                (t2, (0, 1), (1, 0), (1, 1)),
            ):
                if mask_t.any():
                    vy, vx = np.nonzero(mask_t)
                    ia = vid_map[vy + off_a[0], vx + off_a[1]]
                    ib = vid_map[vy + off_b[0], vx + off_b[1]]
                    ic = vid_map[vy + off_c[0], vx + off_c[1]]
                    idx_chunks.append(np.column_stack([ia, ib, ic]).reshape(-1).astype("<u4"))
            vert_offset += n_v

        if not vert_chunks or not idx_chunks:
            return np.asarray([0, 0], dtype="<u4").tobytes()
        verts = np.concatenate(vert_chunks, axis=0)
        indices = np.concatenate(idx_chunks, axis=0)
        header = np.asarray([verts.shape[0], indices.shape[0]], dtype="<u4")
        return header.tobytes() + verts.tobytes() + indices.tobytes()

    def decode_splats_binary(self, depth_m, imgs_rgb_4chw, cam_gains, mask_bits, gt_valid_mask=None):
        """Unproject raw fisheye depth + RGB into compact oriented 3D Gaussians (10 floats/splat)."""
        # Downsample [C, 3, H, W] RGB tensor cleanly onto (dh, dw)
        if isinstance(imgs_rgb_4chw, torch.Tensor):
            rgb_small = F.interpolate(imgs_rgb_4chw, size=(self.dh, self.dw), mode="bilinear", align_corners=False)
            rgb_np = rgb_small.permute(0, 2, 3, 1).detach().cpu().numpy()
        else:
            rgb_np = imgs_rgb_4chw

        chunks = []
        for c in range(len(self.origins)):
            if not mask_bits[c]:
                continue
            d_raw = depth_m[c].astype(np.float32, copy=True)
            dg = self.d_ground[c]
            ok = self.ray_valid[c].copy()

            if gt_valid_mask is not None:
                # LiDAR GT mode: exact metric LiDAR depth where present, flat ground elsewhere (<11m)
                has_lidar = gt_valid_mask[c] & (d_raw > 0.25) & (d_raw < 18.0)
                use_ground = (~has_lidar) & (dg < 11.5)
                d_fused = np.where(has_lidar, d_raw, dg)
                ok &= (has_lidar | use_ground)
            else:
                # Live AI Feed-Forward mode: edge-preserving smooth + ground clamping so road stays flat
                d_smooth = gaussian_filter(d_raw, sigma=0.75)
                hits_floor = (dg < 13.0) & (d_smooth >= dg - 0.55)
                d_fused = np.where(hits_floor, dg, np.minimum(d_smooth, dg))
                ok &= (d_fused > 0.45) & (d_fused < 17.5)

            origin = self.origins[c]
            rays = self.rays_ego[c]
            pts = origin[None, None, :] + d_fused[..., None] * rays  # [dh, dw, 3] ego

            # Selective pruning: remove ego interior, sky > 5.0m, and cross-seam duplicates
            not_ego = ~ego_mask(pts, self.rig)
            ex, ey, ez = pts[..., 0], pts[..., 1], pts[..., 2]
            # Ego-centered symmetric 4-cam ownership (prevents duplicate overlapping splats across seams)
            nx_e = ex / 1.48
            if c == 0:
                owned = (nx_e >= np.abs(ey) * 0.92) & (ex > 0.0)
            elif c == 1:
                owned = (-nx_e >= np.abs(ey) * 0.92) & (ex < 0.0)
            elif c == 2:
                owned = (ey >= np.abs(nx_e) * 0.92) & (ey > 0.0)
            else:
                owned = (-ey >= np.abs(nx_e) * 0.92) & (ey < 0.0)

            keep = ok & not_ego & owned & (ez > -0.25) & (ez < 5.2) & (np.hypot(ex, ey) < 16.5)
            if not keep.any():
                continue

            # Estimate tangent surface normals via finite differences dP/du x dP/dv
            dp_u = np.zeros_like(pts)
            dp_v = np.zeros_like(pts)
            dp_u[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
            dp_v[1:-1, :] = pts[2:, :] - pts[:-2, :]
            nrm = np.cross(dp_u, dp_v)
            n_len = np.linalg.norm(nrm, axis=-1, keepdims=True)
            # Fall back to -ray (billboard facing camera) at depth discontinuities
            bad_n = (n_len[..., 0] < 1e-4) | (np.abs(d_fused - np.roll(d_fused, 1, axis=1)) > 0.45 * d_fused)
            nrm = np.where(bad_n[..., None], -rays, nrm / np.maximum(n_len, 1e-6))
            # Flip normal to always face toward the camera
            dot_v = (nrm * rays).sum(axis=-1, keepdims=True)
            nrm = np.where(dot_v > 0.0, -nrm, nrm)

            # Adaptive metric Gaussian radius (tight near ego, scales smoothly with distance)
            scale = np.clip(d_fused * self.ang_step[c] * 1.48, 0.045, 0.44).astype(np.float32)
            gain = np.asarray(cam_gains[c], dtype=np.float32)
            col = np.clip(rgb_np[c] * gain[None, None, :], 0.0, 1.0).astype(np.float32)

            p_k = pts[keep]
            n_k = nrm[keep]
            s_k = scale[keep]
            c_k = col[keep]
            # Convert ego (x, y, z) -> Three.js (-y, z, -x)
            pos_3 = np.column_stack([-p_k[:, 1], p_k[:, 2], -p_k[:, 0]])
            nrm_3 = np.column_stack([-n_k[:, 1], n_k[:, 2], -n_k[:, 0]])
            chunks.append(np.column_stack([pos_3, s_k, nrm_3, c_k]).astype("<f4"))

        if not chunks:
            return np.asarray([0], dtype="<u4").tobytes()
        packed = np.concatenate(chunks, axis=0)
        header = np.asarray([packed.shape[0]], dtype="<u4")
        return header.tobytes() + packed.tobytes()


class SeamColorCalibrator:
    """Computes smooth per-camera RGB exposure/white-balance gains directly from GPU model tensors."""

    def __init__(self, rig, device, target_hw, ground_z=0.0):
        self.cameras = rig["cameras"]
        self.device = device
        th, tw = target_hw
        # Use shared road points across the four diagonal overlaps.
        seam_angles_deg = np.concatenate([
            np.linspace(22, 68, 12),
            np.linspace(112, 158, 12),
            np.linspace(-158, -112, 12),
            np.linspace(-68, -22, 12),
        ])
        radii = np.array([2.2, 3.0, 3.9, 5.0], dtype=np.float64)
        ang_rad = np.deg2rad(seam_angles_deg)
        rr, aa = np.meshgrid(radii, ang_rad, indexing="ij")
        ex = rr * np.cos(aa)
        ey = rr * np.sin(aa)
        ez = np.full_like(rr, ground_z)
        pts = np.stack([ex, ey, ez], axis=-1).reshape(-1, 3)
        on_road = ~ego_mask(pts, rig)

        self.grids = []
        self.valids = []
        self.overlap_uv = []
        for cam in self.cameras:
            uv, _, ok = project(pts, cam, (th, tw))
            ok &= on_road
            gx = 2.0 * (uv[:, 0] + 0.5) / tw - 1.0
            gy = 2.0 * (uv[:, 1] + 0.5) / th - 1.0
            grid = np.stack([gx, gy], axis=-1).reshape(-1, 1, 1, 2).astype(np.float32)
            grid = np.where(ok[:, None, None, None], grid, 2.0).reshape(1, -1, 1, 2)
            self.grids.append(torch.from_numpy(grid).to(device))
            self.valids.append(torch.from_numpy(ok).to(device))

            src_h, src_w = cam["image_hw"]
            raw_uv, _, raw_ok = project(pts, cam, (src_h, src_w))
            raw_ok &= on_road & (raw_uv >= 12).all(axis=1) & (raw_uv < [src_w - 12, src_h - 12]).all(axis=1)
            if "valid_circle" in cam:
                cx, cy, radius = cam["valid_circle"]
                raw_ok &= np.linalg.norm(raw_uv - [cx, cy], axis=1) < radius - 12
            normalized = ((raw_uv + 0.5) / [src_w, src_h]).round(6)
            self.overlap_uv.append([uv_pair.tolist() if valid else None
                                    for uv_pair, valid in zip(normalized, raw_ok, strict=True)])

        self.img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def solve_gains_from_normalized(self, imgs_norm, mask_bits):
        """Solve 4x3 RGB gains directly on [C, 3, H, W] ImageNet-normalized GPU tensor (zero disk I/O)."""
        imgs_rgb = (imgs_norm * self.img_std + self.img_mean).clamp(0.0, 1.0)
        return self.solve_gains([imgs_rgb[i] for i in range(imgs_rgb.shape[0])], mask_bits)

    @torch.no_grad()
    def solve_gains(self, imgs_t, mask_bits):
        """Return 4x3 RGB gain multipliers equalizing exposure & tint across adjacent seams."""
        n = len(self.cameras)
        samples = []
        for idx, (img, keep) in enumerate(zip(imgs_t, mask_bits)):
            if not keep or img is None:
                samples.append(None)
                continue
            # Sample bilinear 3-channel RGB at overlap coordinates: shape [M, 3]
            pix = F.grid_sample(img.unsqueeze(0), self.grids[idx], mode="bilinear", align_corners=False)
            samples.append(pix[0, :, :, 0].T)

        # Solve tiny 4x4 regularized log-gain system per RGB channel
        a_mat = np.eye(n, dtype=np.float64) * 0.25
        b_mat = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            if samples[i] is None:
                continue
            for j in range(i + 1, n):
                if samples[j] is None:
                    continue
                both = self.valids[i] & self.valids[j]
                if int(both.sum()) < 8:
                    continue
                ci = samples[i][both].clamp(0.05, 0.96).mean(dim=0).cpu().numpy()
                cj = samples[j][both].clamp(0.05, 0.96).mean(dim=0).cpu().numpy()
                diff = np.log(cj / np.maximum(ci, 1e-4))
                a_mat[i, i] += 1.0
                a_mat[j, j] += 1.0
                a_mat[i, j] -= 1.0
                a_mat[j, i] -= 1.0
                b_mat[i] += diff
                b_mat[j] -= diff

        gains = np.exp(np.linalg.solve(a_mat, b_mat))
        gains /= np.maximum(gains.mean(axis=0, keepdims=True), 1e-4)
        gains = np.clip(gains, 0.76, 1.32)
        return [[round(float(c), 4) for c in row] for row in gains]


def compute_adaptive_bowl_radii(occupancy, semantics, grid, rig, p_occ=None, n_segs=N_BOWL_SECTORS):
    """Compute smooth adaptive bowl inflection radii reacting only to tall obstacles outside road lane."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_segs, endpoint=False)
    r_base = bowl_base_radii(angles)
    r_min = bowl_min_radii(angles)
    centers = grid.centers()
    not_ego = ~ego_mask(centers, rig)

    occ = (occupancy == 1) & not_ego
    if p_occ is not None:
        occ &= p_occ >= 0.68
    # Only tall obstacles above z >= 1.0m (zi >= 2 if available, else zi >= 1) bend the bowl wall
    z_min_bin = 2 if grid.shape[0] >= 3 else 1
    obs_mask = occ.copy()
    obs_mask[:z_min_bin] = False

    if not obs_mask.any():
        return [round(float(v), 4) for v in r_base]

    pts = centers[obs_mask]
    r_pts = np.hypot(pts[:, 0], pts[:, 1])
    ang_pts = np.mod(np.arctan2(pts[:, 1], pts[:, 0]), 2.0 * np.pi)
    bins = np.clip((ang_pts / (2.0 * np.pi) * n_segs).astype(np.int32), 0, n_segs - 1)

    r_obs = r_base.copy()
    for b, rv in zip(bins, r_pts):
        if rv < r_obs[b]:
            r_obs[b] = rv

    # Keep the inner road lane ellipse flat (>= r_min) so road markings never warp
    target = np.clip(np.minimum(r_base, r_obs - 0.25), r_min, r_base)
    target = minimum_filter1d(target, size=5, mode="wrap")
    target = gaussian_filter1d(target, sigma=5.0, mode="wrap")
    return [round(float(v), 4) for v in np.clip(target, r_min, r_base)]


def extract_dynamic_objects(occupancy, semantics, grid, rig, classes, p_occ=None, is_prediction=False):
    """Cluster dynamic Vehicle/Rider/Pedestrian voxels into oriented 3D objects + static mask."""
    z_edges = grid.z
    centers = grid.centers()
    not_ego = ~ego_mask(centers, rig)
    occ = (occupancy == 1) & not_ego
    # Always ignore z=0 ground plane (-0.2..0.4m) to prevent road noise from spawning false objects
    if occ.shape[0] > 1:
        occ = occ.copy()
        occ[0] = False

    dyn_mask = np.zeros_like(occ, dtype=bool)
    objects = []

    if semantics is None:
        return objects, dyn_mask

    class_priors = {
        "vehicle": {"min_vox": 6 if is_prediction else 5, "min_p": 0.52, "l": (2.2, 5.4), "w": (1.45, 2.25), "h": (1.25, 2.15), "max_raw_l": 7.5, "max_raw_w": 3.8, "close": 1},
        "rider": {"min_vox": 6 if is_prediction else 3, "min_p": 0.74, "l": (1.25, 2.35), "w": (0.62, 1.05), "h": (1.15, 1.75), "max_raw_l": 3.0, "max_raw_w": 1.8, "close": 1},
        "pedestrian": {"min_vox": 5 if is_prediction else 2, "min_p": 0.72, "l": (0.50, 0.90), "w": (0.50, 0.90), "h": (1.35, 1.85), "max_raw_l": 1.8, "max_raw_w": 1.5, "close": 0},
    }
    xs_grid = centers[0, :, :, 0]
    ys_grid = centers[0, :, :, 1]

    for cls_idx, cls_name in enumerate(classes[1:], 1):
        prior = class_priors.get(cls_name)
        if prior is None:
            continue
        cls_vol = occ & (semantics == cls_idx)
        if is_prediction and p_occ is not None:
            cls_vol = cls_vol & (p_occ >= prior["min_p"])
        if not cls_vol.any():
            continue
        bev_cnt = cls_vol.sum(axis=0)
        bev_bin = bev_cnt > 0
        if prior["close"] > 0:
            bev_bin = binary_closing(bev_bin, structure=np.ones((3, 3), dtype=bool), iterations=1)
        labels_2d, n_comp = label(bev_bin, structure=np.ones((3, 3), dtype=np.int32))

        for comp_id in range(1, n_comp + 1):
            comp_2d = labels_2d == comp_id
            comp_3d = cls_vol & comp_2d[None, :, :]
            n_vox = int(comp_3d.sum())
            if n_vox < prior["min_vox"]:
                continue

            zi, yi, xi = np.nonzero(comp_3d)
            # In AI prediction mode, reject low curb/ground smears that never reach z >= 1.0m (zi >= 2)
            if is_prediction and grid.shape[0] >= 3 and int(zi.max()) < 2:
                continue

            w_pts = bev_cnt[comp_2d & (bev_cnt > 0)].astype(np.float64)
            px = xs_grid[comp_2d & (bev_cnt > 0)]
            py = ys_grid[comp_2d & (bev_cnt > 0)]
            if len(px) == 0:
                continue
            cx = float(np.average(px, weights=w_pts))
            cy = float(np.average(py, weights=w_pts))

            yaw = 0.0
            if len(px) >= 3 and cls_name != "pedestrian":
                dx = px - cx
                dy = py - cy
                cov = np.cov(np.stack([dx, dy]), aweights=w_pts)
                eigvals, eigvecs = np.linalg.eigh(cov)
                ratio = eigvals[1] / max(eigvals[0], 1e-4)
                if ratio > 1.35:
                    vx, vy = eigvecs[:, 1]
                    yaw = float(np.arctan2(vy, vx))
                    if np.cos(yaw) < -0.22:
                        yaw = float(np.arctan2(-vy, -vx))
            else:
                dx = px - cx
                dy = py - cy

            ca, sa = np.cos(-yaw), np.sin(-yaw)
            u_long = dx * ca - dy * sa
            v_lat = dx * sa + dy * ca
            raw_l = float(u_long.max() - u_long.min()) + 0.28
            raw_w = float(v_lat.max() - v_lat.min()) + 0.26

            # Reject continuous static guardrails/walls misclassified by 3-class semantic head as Rider/Ped
            if is_prediction and (raw_l > prior["max_raw_l"] or raw_w > prior["max_raw_w"]):
                continue
            if is_prediction and cls_name == "rider" and n_vox > 140:
                continue

            dyn_mask |= comp_3d
            z_bot = 0.0 if int(zi.min()) <= 1 else float(z_edges[zi.min()])
            z_top = float(z_edges[zi.max() + 1])

            l_val = float(np.clip(raw_l, *prior["l"]))
            w_val = float(np.clip(raw_w, *prior["w"]))
            h_val = float(np.clip(z_top - z_bot, *prior["h"]))
            ex0, ex1, ey0, ey1 = physical_ego_bounds(rig)
            dx_clr = max(0.0, float(px.min()) - ex1, ex0 - float(px.max()))
            dy_clr = max(0.0, float(py.min()) - ey1, ey0 - float(py.max()))
            dist_m = float(np.hypot(dx_clr, dy_clr))
            objects.append({
                "cls": cls_name,
                "x": round(cx, 3),
                "y": round(cy, 3),
                "z": round(z_bot, 3),
                "l": round(l_val, 3),
                "w": round(w_val, 3),
                "h": round(h_val, 3),
                "yaw": round(yaw, 4),
                "dist_m": round(dist_m, 2),
                "voxels": n_vox,
            })

    return objects, dyn_mask


def compute_pdc_sectors(occupancy, semantics, grid, rig, p_occ=None, is_prediction=False, n_sectors=8):
    """Compute 8-zone OEM Park Distance Control (PDC) clearance meters from real above-ground obstacles."""
    centers = grid.centers()
    not_ego = ~ego_mask(centers, rig)
    occ = (occupancy == 1) & not_ego
    # Strictly ignore z=0 ground plane (-0.2..0.4m) so asphalt never triggers false red PDC arcs
    obs_mask = occ.copy()
    if obs_mask.shape[0] > 1:
        obs_mask[0] = False
    if is_prediction and p_occ is not None:
        if obs_mask.shape[0] >= 3:
            high_conf = np.zeros_like(obs_mask, dtype=bool)
            high_conf[1] = p_occ[1] >= 0.82
            high_conf[2:] = p_occ[2:] >= 0.68
            obs_mask &= high_conf
        else:
            obs_mask &= p_occ >= 0.72

    max_range = 5.0
    if not obs_mask.any():
        return [max_range] * n_sectors

    pts = centers[obs_mask]
    ex0, ex1, ey0, ey1 = physical_ego_bounds(rig)
    dx_clr = np.maximum(0.0, np.maximum(pts[:, 0] - ex1, ex0 - pts[:, 0]))
    dy_clr = np.maximum(0.0, np.maximum(pts[:, 1] - ey1, ey0 - pts[:, 1]))
    clr = np.hypot(dx_clr, dy_clr)
    # Sector 0 centered at Front (0 rad), going counter-clockwise in steps of 45 deg
    ang = np.mod(np.arctan2(pts[:, 1], pts[:, 0]) + np.pi / n_sectors, 2.0 * np.pi)
    sec = np.clip((ang / (2.0 * np.pi) * n_sectors).astype(np.int32), 0, n_sectors - 1)
    dists = np.full(n_sectors, max_range, dtype=np.float64)
    for s_idx, d_val in zip(sec, clr):
        if d_val < dists[s_idx]:
            dists[s_idx] = d_val
    return [round(float(d), 2) for d in dists]


def extract_vector_road_edges(occupancy, dyn_mask, grid):
    """Extract smoothed left and right road guardrail/curb neon splines from static occupancy."""
    static_obs = (occupancy == 1) & (~dyn_mask)
    if static_obs.shape[0] > 1:
        static_bev = static_obs[1:].any(axis=0)  # y, x above 0.4m road plane
    else:
        static_bev = static_obs[0]

    xs = (grid.x[:-1] + grid.x[1:]) * 0.5
    ys = (grid.y[:-1] + grid.y[1:]) * 0.5
    polylines = []
    # Bin x into 2-cell (0.4m) steps for smooth automotive vector ribbons
    step = 2
    for side_sign, mask_y in ((1.0, ys > 1.15), (-1.0, ys < -1.15)):
        side_ys = ys[mask_y]
        raw_pts = []
        for xi in range(0, len(xs) - step + 1, step):
            col = static_bev[mask_y, xi:xi + step].any(axis=1)
            if col.any():
                # Innermost obstacle edge closest to the drivable center lane
                y_edge = float(side_ys[col].min() if side_sign > 0 else side_ys[col].max())
                raw_pts.append((float(xs[xi:xi + step].mean()), y_edge))
            else:
                raw_pts.append(None)

        # Split contiguous spans and apply 3-point smoothing
        cur = []
        for item in raw_pts + [None]:
            if item is not None:
                cur.append(item)
            else:
                if len(cur) >= 3:
                    arr = np.asarray(cur, dtype=np.float64)
                    arr[:, 1] = gaussian_filter1d(arr[:, 1], sigma=1.1, mode="nearest")
                    polylines.append([[round(float(x), 3), round(float(y), 3)] for x, y in arr])
                cur = []
    return polylines


def compute_corridor_profile(occupancy, dyn_mask, grid, rig, p_occ=None, is_prediction=False):
    """Fit smooth quadratic left/right corridor wall curves y(x)=a*x^2+b*x+c hugging real road curbs."""
    static_obs = (occupancy == 1) & (~dyn_mask)
    if static_obs.shape[0] > 1:
        if is_prediction and p_occ is not None:
            # Slightly softer threshold on z>=1 catches the true curb wall face cleanly
            wall_3d = (p_occ >= 0.46) & (~dyn_mask)
            wall_3d[0] = False
            static_bev = wall_3d.any(axis=0)
        else:
            static_bev = static_obs[1:].any(axis=0)
    else:
        static_bev = static_obs[0]

    xs = (grid.x[:-1] + grid.x[1:]) * 0.5
    ys = (grid.y[:-1] + grid.y[1:]) * 0.5
    step = 2

    def fit_side(side_sign, default_y):
        mask_y = (ys > 1.45) if side_sign > 0 else (ys < -1.45)
        side_ys = ys[mask_y]
        pts_x, pts_y = [], []
        for xi in range(0, len(xs) - step + 1, step):
            col = static_bev[mask_y, xi:xi + step].any(axis=1)
            if col.any():
                y_val = float(side_ys[col].min() if side_sign > 0 else side_ys[col].max())
                pts_x.append(float(xs[xi:xi + step].mean()))
                pts_y.append(y_val)

        # Require long continuous wall (>= 45% of longitudinal grid span) so passing bikes don't pull the wall
        min_bins = max(6, int(0.45 * (len(xs) // step)))
        if len(pts_x) < min_bins:
            return [0.0, 0.0, default_y]

        arr_x = np.asarray(pts_x, dtype=np.float64)
        arr_y = np.asarray(pts_y, dtype=np.float64)
        # Reject short isolated vehicles jutting inward via longitudinal median filter + outer quartile trim
        med_y = median_filter(arr_y, size=min(9, len(arr_y)), mode="nearest")
        if side_sign > 0:
            keep = med_y >= (np.percentile(med_y, 60) - 0.55)
        else:
            keep = med_y <= (np.percentile(med_y, 40) + 0.55)
        if int(keep.sum()) >= 8:
            arr_x, arr_y = arr_x[keep], med_y[keep]
        else:
            arr_y = med_y

        # Ridge-regularized quadratic y(x) = a*x^2 + b*x + c anchored softly to default curvature=0
        design = np.column_stack([arr_x ** 2, arr_x, np.ones_like(arr_x)])
        reg = np.diag([450.0, 18.0, 0.05])
        rhs = design.T @ arr_y + np.array([0.0, 0.0, 0.05 * default_y])
        a, b, c = np.linalg.solve(design.T @ design + reg, rhs)
        # Verify wall consistency: a true bridge wall/curb has tight fit around the curve (std < 0.52m)
        pred_y = a * (arr_x ** 2) + b * arr_x + c
        if float(np.std(arr_y - pred_y)) > 0.52:
            return [0.0, 0.0, default_y]
        a = float(np.clip(a, -0.018, 0.018))
        b = float(np.clip(b, -0.14, 0.14))
        if side_sign > 0:
            c = float(np.clip(c, 1.95, 2.75))
        else:
            c = float(np.clip(c, -2.75, -1.95))
        return [round(a, 6), round(b, 5), round(c, 4)]

    # Tight symmetric default (+2.25m / -2.25m) keeps left/right vehicles at natural 1:1 scale
    left_poly = fit_side(1.0, 2.25)
    right_poly = fit_side(-1.0, -2.25)
    return {"left_poly": left_poly, "right_poly": right_poly}


def extract_seam_shift_zones(occupancy, grid, rig, corridor, objects, p_occ=None, is_prediction=False, max_zones=8):
    """Detect tall road objects crossing the 4 diagonal camera seams so shader shifts seams around them."""
    a_l, b_l, c_l = corridor["left_poly"]
    a_r, b_r, c_r = corridor["right_poly"]
    candidates = []

    # 1. Include all detected semantic dynamic objects (Vehicle / Rider / Pedestrian)
    for o in objects or []:
        cx, cy = float(o["x"]), float(o["y"])
        rad = float(np.hypot(cx, cy))
        ang_c = float(np.arctan2(cy, cx))
        half_ang = float(np.arctan2(max(o["l"], o["w"]) * 0.55, max(rad, 1.2))) + np.deg2rad(8.0)
        candidates.append((cx, cy, ang_c, min(half_ang, 0.58), rad + max(o["l"], o["w"]) * 0.6))

    # 2. Also detect any tall isolated obstacle clusters (z >= 1.0m) inside the road corridor
    centers = grid.centers()
    not_ego = ~ego_mask(centers, rig)
    occ = (occupancy == 1) & not_ego
    if is_prediction and p_occ is not None:
        occ = occ & (p_occ >= 0.62)
    z_start = 2 if grid.shape[0] >= 3 else (1 if grid.shape[0] >= 2 else 0)
    tall_bev = occ[z_start:].any(axis=0)
    xs_2d = centers[0, :, :, 0]
    ys_2d = centers[0, :, :, 1]
    y_l_grid = a_l * xs_2d ** 2 + b_l * xs_2d + c_l
    y_r_grid = a_r * xs_2d ** 2 + b_r * xs_2d + c_r
    in_corridor = (ys_2d < y_l_grid - 0.35) & (ys_2d > y_r_grid + 0.35)
    iso_bev = tall_bev & in_corridor
    if iso_bev.any():
        labels_2d, n_comp = label(binary_closing(iso_bev, structure=np.ones((3, 3), dtype=bool), iterations=1))
        for cid in range(1, n_comp + 1):
            cm = labels_2d == cid
            if int(cm.sum()) < 3:
                continue
            px, py = xs_2d[cm], ys_2d[cm]
            cx, cy = float(px.mean()), float(py.mean())
            angs = np.arctan2(py, px)
            ang_c = float(np.arctan2(np.sin(angs).mean(), np.cos(angs).mean()))
            da = np.abs(np.arctan2(np.sin(angs - ang_c), np.cos(angs - ang_c)))
            half_ang = float(da.max()) + np.deg2rad(9.0)
            r_max = float(np.hypot(px, py).max()) + 0.8
            candidates.append((cx, cy, ang_c, min(half_ang, 0.58), r_max))

    # Corner seam centers matching camera-anchored diagonals (FL/FR ~ +-39 deg, RL/RR ~ +-141 deg)
    corner_rad = float(np.arctan2(1.0, BOWL_SEAM_X_SCALE))
    seam_angles = [corner_rad, -corner_rad, np.pi - corner_rad, -(np.pi - corner_rad)]
    zones = []
    for cx, cy, ang_c, half_ang, r_max in candidates:
        ang_anchor = float(np.arctan2(cy, cx - BOWL_SEAM_X_SHIFT))
        near_seam = any(
            abs(float(np.arctan2(np.sin(ang_anchor - sa), np.cos(ang_anchor - sa)))) < (half_ang + np.deg2rad(20.0))
            for sa in seam_angles
        )
        if not near_seam:
            continue
        # Pick 1 optimal camera to draw the entire motorbike/car intact without slicing it:
        # Side mirror cameras sit at x ≈ +0.99m, so Rear cam sees rear corners (x < -1.85m) much sharper!
        if cx < -1.85:
            prefer_cam = 1  # Rear cam owns rear-left & rear-right corners cleanly
        elif cx > 2.65:
            prefer_cam = 0  # Front cam owns front-left & front-right corners cleanly
        elif abs(cy) >= 1.15:
            prefer_cam = 2 if cy > 0 else 3  # Left (2) or Right (3) owns side corridor
        else:
            prefer_cam = 0 if cx >= 0 else 1
        zones.append([round(ang_anchor, 4), round(half_ang, 4), float(prefer_cam), round(min(r_max, 9.0), 2)])

    zones.sort(key=lambda z: z[3])  # Prioritize closest objects to ego
    return zones[:max_zones]


def build_smooth_occupancy_mesh_binary(prob_field, grid, level=0.38):
    """Generate Tesla/Huawei-style Marching Cubes smooth surface with vertex Ambient Occlusion."""
    # Down-weight ground-plane slice zi=0 slightly unless supported above so road stays clean
    field = prob_field.astype(np.float32, copy=True)
    if field.shape[0] > 1:
        above = gaussian_filter(field[1:].max(axis=0), sigma=0.7)
        field[0] = np.where(above > 0.28, field[0], field[0] * 0.20)

    t = torch.from_numpy(field)[None, None]
    fine = F.interpolate(t, size=(12, 90, 120), mode="trilinear", align_corners=False)[0, 0].numpy()
    fine = np.pad(fine, 1, mode="constant", constant_values=0.0)
    fine = gaussian_filter(fine, sigma=(0.8, 0.85, 0.85))

    if float(fine.max()) <= level or float(fine.min()) >= level:
        return np.asarray([0, 0], dtype="<u4").tobytes()

    verts, faces, normals, _ = marching_cubes(fine, level=level, step_size=1, allow_degenerate=False)
    if len(verts) == 0 or len(faces) == 0:
        return np.asarray([0, 0], dtype="<u4").tobytes()

    # Map padded voxel coords (iz, iy, ix) -> metric ego coords (x, y, z)
    iz = np.clip((verts[:, 0] - 1.0) / 11.0, 0.0, 1.0)
    iy = np.clip((verts[:, 1] - 1.0) / 89.0, 0.0, 1.0)
    ix = np.clip((verts[:, 2] - 1.0) / 119.0, 0.0, 1.0)

    # Piecewise-linear mapping of normalized z into physical z_edges
    z_src = np.linspace(0.0, 1.0, len(grid.z))
    z_ego = np.maximum(0.0, np.interp(iz, z_src, grid.z))
    y_ego = grid.y[0] + iy * (grid.y[-1] - grid.y[0])
    x_ego = grid.x[0] + ix * (grid.x[-1] - grid.x[0])

    # Convert to Three.js world (-y_ego, z_ego, -x_ego)
    pos_three = np.column_stack([-y_ego, z_ego, -x_ego]).astype(np.float32)

    # Transform Marching Cubes normals matching (-ny, nz, -nx) & normalize
    # Wait: marching_cubes outputs inward/outward gradient normal on fine grid; re-normalize
    n_scale = np.array([1.0 / 11.0 * (grid.z[-1] - grid.z[0]),
                        1.0 / 89.0 * (grid.y[-1] - grid.y[0]),
                        1.0 / 119.0 * (grid.x[-1] - grid.x[0])], dtype=np.float32)
    n_phys = normals / np.maximum(n_scale[None, :], 1e-4)
    n_three = np.column_stack([-n_phys[:, 1], n_phys[:, 0], -n_phys[:, 2]])
    n_norm = np.linalg.norm(n_three, axis=1, keepdims=True)
    n_three = (n_three / np.maximum(n_norm, 1e-6)).astype(np.float32)

    # Voxel-space Ambient Occlusion: sample smoothed density slightly outside surface
    ao = np.ones(len(verts), dtype=np.float32)
    f_shape = np.asarray(fine.shape, dtype=np.float32) - 1.0
    for step_cells, weight in ((1.4, 0.45), (2.8, 0.30)):
        probe = np.rint(np.clip(verts + normals * step_cells, 0.0, f_shape)).astype(np.int32)
        occ_density = fine[probe[:, 0], probe[:, 1], probe[:, 2]]
        ao -= weight * occ_density
    ao = np.clip(ao, 0.34, 1.0).astype(np.float32)

    # Keep triangle winding matching outward normals in (-y, z, -x) coordinates
    idx_arr = faces.reshape(-1).astype("<u4")
    packed = np.column_stack([pos_three, n_three, ao]).astype("<f4")
    header = np.asarray([packed.shape[0], idx_arr.shape[0]], dtype="<u4")
    return header.tobytes() + packed.tobytes() + idx_arr.tobytes()

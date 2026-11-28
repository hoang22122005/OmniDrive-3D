import copy
import io
import json
import threading
import time
import webbrowser
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from omnishield.data.dataset import OccupancyDataset, load_targets
from omnishield.data.manifest import load_manifest, sample_path
from omnishield.data.phenikaa import load_lidar
from omnishield.engine.checkpoints import load_checkpoint
from omnishield.engine.train import dino_revision
from omnishield.eval.occupancy import OccupancyMetrics
from omnishield.eval.scene3d import (
    FeedForward3DGSDecoder,
    SeamColorCalibrator,
    bowl_seam_calibration,
    build_adaptive_bowl_template,
    build_smooth_occupancy_mesh_binary,
    build_standard_rig_calibration,
    calibrate_rig_extrinsics_photometric,
    compute_adaptive_bowl_radii,
    compute_corridor_profile,
    compute_pdc_sectors,
    estimate_road_plane,
    estimate_valid_circles,
    export_bowl_spec,
    fit_lidar_road_plane,
    export_camera_calib,
    export_vehicle_spec,
    extract_dynamic_objects,
    extract_seam_shift_zones,
    extract_vector_road_edges,
    load_calibrated_rig_profile,
    measure_near_field_coverage,
    measure_seam_parallax_report,
    pack_foreground_ranges,
    solve_fisheye_pnp_camera,
    verify_bowl_mesh_integrity,
)
from omnishield.geometry.calibration import Grid, depth_edges, ego_mask, project, rigid_matrix, transform
from omnishield.io import child_path, fingerprint, load_json, write_json
from omnishield.models.edge import OmniShieldEdge


HTML_PATH = Path(__file__).with_name("demo.html")
VENDOR_DIR = Path(__file__).with_name("vendor")
BEV_X = (7.0, -7.0, 560)  # front->rear, 2.5 cm/px
BEV_Y = (5.0, -5.0, 400)  # left->right, 2.5 cm/px
RECT_HW = (432, 768)
WIDE_HW = (420, 840)


def _encode_grid(occupancy, semantics, classes):
    """Pack sparse occupied voxels and 2D floor free-space mask for canvas rendering."""
    free_2d = ((occupancy == 0).any(axis=0) & ~(occupancy == 1).any(axis=0)).astype(np.uint8)
    occ_mask = occupancy == 1
    z_idx, y_idx, x_idx = np.nonzero(occ_mask)
    if semantics is not None:
        sem_vals = semantics[z_idx, y_idx, x_idx]
        codes = np.where(sem_vals > 0, sem_vals + 1, 1).astype(np.int32)
    else:
        codes = np.ones(len(z_idx), dtype=np.int32)
    voxels = np.column_stack((z_idx, y_idx, x_idx, codes)).ravel().tolist()
    counts = {
        "occupied": int(occ_mask.sum()),
        "free": int((occupancy == 0).sum()),
        "unknown": int((occupancy < 0).sum()),
    }
    for idx, name in enumerate(classes[1:], 1):
        counts[name] = int((occ_mask & (semantics == idx)).sum()) if semantics is not None else 0
    return {"free_2d": free_2d.ravel().tolist(), "voxels": voxels, "counts": counts}


def _to_jpeg(array_rgb, quality=88):
    buf = io.BytesIO()
    Image.fromarray(array_rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _load_pose_4x4(raw_root, row):
    rel = row.get("pose")
    if not rel:
        return np.eye(4, dtype=np.float64)
    path = child_path(raw_root, rel)
    if not path.is_file():
        return np.eye(4, dtype=np.float64)
    vals = [float(v) for v in path.read_text(encoding="utf-8").split()]
    if len(vals) not in {12, 16}:
        return np.eye(4, dtype=np.float64)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :] = np.asarray(vals[:12], dtype=np.float64).reshape(3, 4)
    return mat


class SurroundViewRenderer:
    """Precomputes GPU sampling grids for 360 surround-view BEV and rectified camera views."""

    def __init__(self, rig, device, model_hw, ground_z=None, ground_plane=None):
        self.rig = rig
        self.device = device
        self.cameras = rig["cameras"]
        if ground_z is None:
            ground_z = float(rig.get("ground_height_m", 0.0))
        else:
            ground_z = float(ground_z)
        self.ground_z = ground_z
        self.ground_plane = tuple(float(v) for v in (ground_plane if ground_plane is not None else (0.0, 0.0, self.ground_z)))
        pa, pb, pc = self.ground_plane

        x_vals = BEV_X[0] + (BEV_X[1] - BEV_X[0]) * (np.arange(BEV_X[2]) + 0.5) / BEV_X[2]
        y_vals = BEV_Y[0] + (BEV_Y[1] - BEV_Y[0]) * (np.arange(BEV_Y[2]) + 0.5) / BEV_Y[2]
        yy, xx = np.meshgrid(y_vals, x_vals)
        pts_bev = np.stack([xx, yy, pa * xx + pb * yy + pc], axis=-1)
        not_ego = ~ego_mask(pts_bev, rig)

        bev_grids, bev_weights, rect_grids, wide_grids = [], [], [], []
        self.rect_cameras = []
        self.wide_meta = []

        # Cylindrical 166-deg panoramic wide grid (zero corner stretching for Front 180 & Rear Mirror views!)
        wh, ww = WIDE_HW
        w_az_max = np.deg2rad(83.0)
        w_el_min, w_el_max = np.deg2rad(-44.0), np.deg2rad(22.0)
        az_vals = np.linspace(-w_az_max, w_az_max, ww, dtype=np.float64)
        el_vals = np.linspace(w_el_max, w_el_min, wh, dtype=np.float64)
        el_g, az_g = np.meshgrid(el_vals, az_vals, indexing="ij")
        # Camera-frame horizontal cylindrical rays (pitch-levelled in ego frame so horizon stays flat!)
        cyl_x_cam = np.cos(el_g) * np.sin(az_g)
        cyl_y_cam = -np.sin(el_g)
        cyl_z_cam = np.cos(el_g) * np.cos(az_g)

        for cam in self.cameras:
            sh, sw = cam["image_hw"]
            cam_from_ego = rigid_matrix(cam["T_cam_from_ego"], cam["slot"])
            ego_from_cam = np.linalg.inv(cam_from_ego)
            cam_xy = ego_from_cam[:2, 3]
            cam_fwd = ego_from_cam[:2, 2]

            # 1. BEV IPM grid and smooth angular feathering weights
            uv, _, valid = project(pts_bev, cam, (sh, sw))
            dx, dy = xx - cam_xy[0], yy - cam_xy[1]
            cos_dir = np.clip((dx * cam_fwd[0] + dy * cam_fwd[1]) / np.hypot(dx, dy).clip(1e-4), 0.0, 1.0)
            mx = np.minimum(uv[..., 0] + 0.5, sw - 0.5 - uv[..., 0])
            my = np.minimum(uv[..., 1] + 0.5, sh - 0.5 - uv[..., 1])
            margin = np.minimum(mx, my)
            if "valid_circle" in cam:
                cx, cy, radius = cam["valid_circle"]
                margin = np.minimum(margin, radius - np.hypot(uv[..., 0] - cx, uv[..., 1] - cy))
            border = np.clip(margin / (0.04 * min(sh, sw)), 0.0, 1.0)
            weight = cos_dir**6 * border**2 * (3 - 2 * border) * valid * not_ego
            grid_bev = np.stack([2.0 * (uv[..., 0] + 0.5) / sw - 1.0, 2.0 * (uv[..., 1] + 0.5) / sh - 1.0], axis=-1)
            bev_grids.append(np.where(valid[..., None], grid_bev, 2.0).astype(np.float32))
            bev_weights.append(weight.astype(np.float32))

            # 2B. Horizon-levelled Cylindrical 166-deg Wide View (Front/Rear intersection & reversing view)
            fwd_xy = ego_from_cam[:2, 2]
            fwd_xy = fwd_xy / max(float(np.linalg.norm(fwd_xy)), 1e-6)
            fwd3 = np.array([fwd_xy[0], fwd_xy[1], 0.0], dtype=np.float64)
            up3 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            right3 = np.cross(fwd3, up3)
            right3 /= max(float(np.linalg.norm(right3)), 1e-6)
            # Basis matrix mapping cylindrical (right, down, forward) -> ego
            r_ego_from_lvl = np.column_stack([right3, -up3, fwd3])
            orig_w = ego_from_cam[:3, 3]
            rays_lvl = np.stack([cyl_x_cam, cyl_y_cam, cyl_z_cam], axis=-1)
            pts_wide_ego = orig_w[None, None, :] + (rays_lvl @ r_ego_from_lvl.T) * 6.0
            wuv, _, wvalid = project(pts_wide_ego, cam, (sh, sw))
            wgrid = np.stack([2.0 * (wuv[..., 0] + 0.5) / sw - 1.0, 2.0 * (wuv[..., 1] + 0.5) / sh - 1.0], axis=-1)
            wide_grids.append(np.where(wvalid[..., None], wgrid, 2.0).astype(np.float32))
            self.wide_meta.append({
                "origin": orig_w, "r_lvl_from_ego": r_ego_from_lvl.T,
                "az_max": float(w_az_max), "el_min": float(w_el_min), "el_max": float(w_el_max),
            })

            rect_grid, rect_camera = self._rectified_view(cam, ego_from_cam, r_ego_from_lvl)
            rect_grids.append(rect_grid)
            self.rect_cameras.append(rect_camera)

        self.bev_grids = torch.from_numpy(np.stack(bev_grids)).to(device)
        self.bev_weights = torch.from_numpy(np.stack(bev_weights)).unsqueeze(-1).to(device)
        self.rect_grids = torch.from_numpy(np.stack(rect_grids)).to(device)
        self.wide_grids = torch.from_numpy(np.stack(wide_grids)).to(device)
        self.color_cal = SeamColorCalibrator(rig, device, model_hw, pc)
        self.guidelines = self._build_guidelines(pc)
        self.guidelines["ground_z"] = round(float(pc), 5)
        self.guidelines["ground_plane"] = [round(float(v), 6) for v in self.ground_plane]
        self.guidelines["bowl_spec"] = export_bowl_spec()
        self.guidelines["vehicle_spec"] = export_vehicle_spec(rig, ground_plane=self.ground_plane)
        self.guidelines["camera_calib"] = export_camera_calib(rig)
        self.guidelines["bowl_seams"] = bowl_seam_calibration(
            rig, self.guidelines["camera_calib"], pc, ground_plane=self.ground_plane
        )
        self.guidelines["seam_parallax_cm"] = measure_seam_parallax_report(rig, self.ground_plane)
        self.guidelines["near_field_coverage"] = measure_near_field_coverage(rig, self.ground_plane)
        self.guidelines["mesh_integrity"] = verify_bowl_mesh_integrity(self.ground_plane)
        self.guidelines["overlap_uv"] = self.color_cal.overlap_uv
        self.gs_decoder = FeedForward3DGSDecoder(
            rig, self.guidelines["camera_calib"], model_hw, feature_stride=4, ground_z=pc
        )
        self.bowl_binary = build_adaptive_bowl_template(pc)

    @staticmethod
    def _rectified_view(cam, ego_from_cam, ego_from_level):
        """Inverse map an upright pinhole image into the calibrated source lens."""
        oh, ow = RECT_HW
        sh, sw = cam["image_hw"]
        cx, cy = (ow - 1) / 2, (oh - 1) / 2
        origin = ego_from_cam[:3, 3]
        axis = ego_from_cam[:3, 2]
        pitch = np.arctan2(-axis[2], np.linalg.norm(axis[:2]))
        pitch_offset = {"front": -4.0, "rear": -6.0, "left": 15.0, "right": 15.0}
        pitch = np.deg2rad(17.5) if cam["slot"] == "front" else pitch + np.deg2rad(pitch_offset[cam["slot"]])
        cp, sp = np.cos(pitch), np.sin(pitch)
        basis = ego_from_level @ np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = basis.T
        extrinsics[:3, 3] = -basis.T @ origin

        def rays(u, v, f):
            return np.stack([(u - cx) / f, (v - cy) / f, np.ones_like(u)], axis=-1)

        # Fit the widest fully supported crop, with a cap to limit perspective edge stretch.
        probes_u, probes_v = np.meshgrid(np.linspace(0, ow - 1, 33), np.linspace(0, oh - 1, 19))
        for fov in ((120,) if cam["slot"] == "front" else (100, 96, 92, 88, 84, 76, 68, 60, 50, 40)):
            f = 0.5 * ow / np.tan(np.deg2rad(fov * 0.5))
            uv, _, valid = project(origin + rays(probes_u, probes_v, f) @ basis.T, cam, (sh, sw))
            if valid.all() and (uv >= 2).all() and (uv < [sw - 2, sh - 2]).all():
                if "valid_circle" not in cam:
                    break
                sx, sy, radius = cam["valid_circle"]
                if np.all(np.hypot(uv[..., 0] - sx, uv[..., 1] - sy) < radius - 2):
                    break
        uu, vv = np.meshgrid(np.arange(ow), np.arange(oh))
        uv, _, valid = project(origin + rays(uu, vv, f) @ basis.T, cam, (sh, sw))
        if "valid_circle" in cam:
            sx, sy, radius = cam["valid_circle"]
            valid &= np.hypot(uv[..., 0] - sx, uv[..., 1] - sy) <= radius
        grid = np.where(valid[..., None], 2 * (uv + 0.5) / [sw, sh] - 1, 2).astype(np.float32)
        return grid, {
            "K": [[f, 0, cx], [0, f, cy], [0, 0, 1]],
            "T_cam_from_ego": extrinsics.tolist(), "horizontal_fov_deg": fov,
            "coverage": round(float(valid.mean()), 6),
        }

    def _project_rect_line(self, cam_idx, pts_xyz):
        cam = self.rect_cameras[cam_idx]
        xyz = transform(np.asarray(pts_xyz, dtype=np.float64), np.asarray(cam["T_cam_from_ego"]))
        xyz = xyz[xyz[:, 2] > 0.05]
        k = np.asarray(cam["K"])
        uv = (xyz[:, :2] / xyz[:, 2:]) @ k[:2, :2].T + k[:2, 2]
        oh, ow = RECT_HW
        return ((uv + 0.5) / [ow, oh]).round(6).tolist()

    def _project_wide_line(self, cam_idx, pts_xyz, mirror=False):
        meta = self.wide_meta[cam_idx]
        rel = np.asarray(pts_xyz, dtype=np.float64) - meta["origin"][None, :]
        loc = rel @ meta["r_lvl_from_ego"].T  # (right, down, forward)
        az = np.arctan2(loc[:, 0], loc[:, 2])
        el = np.arctan2(-loc[:, 1], np.hypot(loc[:, 0], loc[:, 2]))
        u = (az + meta["az_max"]) / (2.0 * meta["az_max"])
        if mirror:
            u = 1.0 - u
        v = (meta["el_max"] - el) / (meta["el_max"] - meta["el_min"])
        out = []
        for uu, vv, zf in zip(u, v, loc[:, 2]):
            if zf > 0.05 and -0.05 <= uu <= 1.05 and -0.05 <= vv <= 1.05:
                out.append([round(float(uu), 4), round(float(vv), 4)])
        return out

    def _build_guidelines(self, gz):
        spec = export_vehicle_spec(self.rig, ground_plane=self.ground_plane)
        pa, pb, pc = self.ground_plane
        gz_xy = lambda xs, ys: pa * xs + pb * ys + pc
        x0, x1 = spec["x_rear_bumper_m"], spec["x_front_bumper_m"]
        hw = 0.5 * spec["width_m"]
        y0, y1 = -hw, hw
        xs_f = np.linspace(x1 + 0.15, x1 + 4.2, 24)
        xs_r = np.linspace(x0 - 0.15, x0 - 3.6, 24)
        xs_side = np.linspace(x0 - 0.5, x1 + 0.5, 24)

        def front_lines(i):
            return {
                "tracks": [
                    self._project_rect_line(i, np.column_stack([xs_f, np.full_like(xs_f, y0), gz_xy(xs_f, y0)])),
                    self._project_rect_line(i, np.column_stack([xs_f, np.full_like(xs_f, y1), gz_xy(xs_f, y1)])),
                ],
                "bars": [
                    self._project_rect_line(i, [[d, y0, float(gz_xy(d, y0))], [d, y1, float(gz_xy(d, y1))]]) for d in (x1 + 1.0, x1 + 2.5, x1 + 4.0)
                ],
            }

        def rear_lines(i):
            return {
                "tracks": [
                    self._project_rect_line(i, np.column_stack([xs_r, np.full_like(xs_r, y0), gz_xy(xs_r, y0)])),
                    self._project_rect_line(i, np.column_stack([xs_r, np.full_like(xs_r, y1), gz_xy(xs_r, y1)])),
                ],
                "bars": [
                    self._project_rect_line(i, [[x0 - d, y0, float(gz_xy(x0 - d, y0))], [x0 - d, y1, float(gz_xy(x0 - d, y1))]]) for d in (0.6, 1.5, 3.0)
                ],
            }

        def side_lines(i, sign):
            yb = sign * (abs(y1) + 0.35)
            yc = sign * (abs(y1) + 0.90)
            ax = spec["x_front_axle_m"]
            return {
                "tracks": [
                    self._project_rect_line(i, np.column_stack([xs_side, np.full_like(xs_side, yb), gz_xy(xs_side, yb)])),
                    self._project_rect_line(i, np.column_stack([xs_side, np.full_like(xs_side, yc), gz_xy(xs_side, yc)])),
                ],
                "bars": [
                    self._project_rect_line(i, [[ax, sign * abs(y1), float(gz_xy(ax, sign * abs(y1)))], [ax, yc + sign * 0.4, float(gz_xy(ax, yc + sign * 0.4))]]),
                ],
            }

        ys_bar = np.linspace(y0, y1, 16)
        wide_front = {
            "tracks": [
                self._project_wide_line(0, np.column_stack([xs_f, np.full_like(xs_f, y0), gz_xy(xs_f, y0)]), mirror=False),
                self._project_wide_line(0, np.column_stack([xs_f, np.full_like(xs_f, y1), gz_xy(xs_f, y1)]), mirror=False),
            ],
            "bars": [
                self._project_wide_line(0, np.column_stack([np.full_like(ys_bar, d), ys_bar, gz_xy(d, ys_bar)]), mirror=False)
                for d in (x1 + 0.55, x1 + 1.5, x1 + 3.0)
            ],
        }
        wide_rear_mirror = {
            "tracks": [
                self._project_wide_line(1, np.column_stack([xs_r, np.full_like(xs_r, y0), gz_xy(xs_r, y0)]), mirror=True),
                self._project_wide_line(1, np.column_stack([xs_r, np.full_like(xs_r, y1), gz_xy(xs_r, y1)]), mirror=True),
            ],
            "bars": [
                self._project_wide_line(1, np.column_stack([np.full_like(ys_bar, x0 - d), ys_bar, gz_xy(x0 - d, ys_bar)]), mirror=True)
                for d in (0.6, 1.5, 3.0)
            ],
            "meta": {
                "origin": [round(float(v), 5) for v in self.wide_meta[1]["origin"]],
                "r_lvl_from_ego": [[round(float(v), 6) for v in row] for row in self.wide_meta[1]["r_lvl_from_ego"]],
                "az_max": round(self.wide_meta[1]["az_max"], 6),
                "el_min": round(self.wide_meta[1]["el_min"], 6),
                "el_max": round(self.wide_meta[1]["el_max"], 6),
            },
        }

        return {
            "bev_bounds": {"x": [BEV_X[0], BEV_X[1]], "y": [BEV_Y[0], BEV_Y[1]], "hw": [BEV_X[2], BEV_Y[2]], "ego": [x0, x1, y0, y1]},
            "rect": [front_lines(0), rear_lines(1), side_lines(2, 1.0), side_lines(3, -1.0)],
            "rect_fov_deg": [cam["horizontal_fov_deg"] for cam in self.rect_cameras],
            "rect_coverage": [cam["coverage"] for cam in self.rect_cameras],
            "wide": {"front": wide_front, "rear": wide_rear_mirror},
        }

    @staticmethod
    def _decode_rgb_half(path):
        with Image.open(path) as image:
            w, h = image.size
            if w >= 1600 and h >= 900 and w % 2 == h % 2 == 0:
                image.draft("RGB", (w // 2, h // 2))
            return np.array(image.convert("RGB"))

    @torch.no_grad()
    def render_bev_jpeg(self, raw_root, row, mask_bits):
        canvas = torch.zeros((BEV_X[2], BEV_Y[2], 3), dtype=torch.float32, device=self.device)
        wsum = torch.zeros((BEV_X[2], BEV_Y[2], 1), dtype=torch.float32, device=self.device)
        for c, (rel, keep) in enumerate(zip(row["images"], mask_bits, strict=True)):
            if not keep or rel is None:
                continue
            rgb = self._decode_rgb_half(child_path(raw_root, rel))
            img_t = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device, dtype=torch.float32, non_blocking=True) / 255.0
            sampled = F.grid_sample(img_t.unsqueeze(0), self.bev_grids[c].unsqueeze(0),
                                    mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
            w = self.bev_weights[c]
            canvas += sampled * w
            wsum += w
        # A single supported peripheral source may have tiny weight and still provide the image.
        bev = (canvas / wsum.clamp_min(1e-9)).clamp(0, 1)
        bev = torch.where(wsum >= 1e-9, bev, torch.tensor([0.04, 0.07, 0.06], device=self.device))
        return _to_jpeg((bev * 255.0).byte().cpu().numpy(), quality=85)

    @torch.no_grad()
    def solve_cam_gains(self, imgs_norm, mask_bits):
        return self.color_cal.solve_gains_from_normalized(imgs_norm, mask_bits)

    @torch.no_grad()
    def render_underbody_seed_jpeg(self, raw_root, rows, poses, index, mask_bits, half_extent=3.2, size=256):
        """Composite 6 historical frames via 6-DoF ego odometry to seed the transparent-chassis ground buffer."""
        xs = np.linspace(-half_extent, half_extent, size, dtype=np.float64)
        ys = np.linspace(half_extent, -half_extent, size, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys)
        pa, pb, pc = self.ground_plane
        pts_curr = np.stack([xx, yy, pa * xx + pb * yy + pc], axis=-1)

        drive_id = rows[index]["drive_id"]
        t_curr = poses[index]
        # Blend oldest -> newest so freshest road observation overwrites older ones
        start_k = max(0, index - 6)
        hist_indices = [k for k in range(start_k, index + 1) if rows[k]["drive_id"] == drive_id]

        canvas = np.full((size, size, 3), np.array([0.16, 0.17, 0.17], dtype=np.float32), dtype=np.float32)
        filled = np.zeros((size, size), dtype=bool)
        for k in hist_indices:
            row_k = rows[k]
            t_k_from_curr = np.linalg.inv(poses[k]) @ t_curr
            pts_k = transform(pts_curr, t_k_from_curr)
            not_ego_k = ~ego_mask(pts_k, self.rig)
            if not not_ego_k.any():
                continue

            best_w = np.zeros((size, size), dtype=np.float32)
            frame_rgb = np.zeros((size, size, 3), dtype=np.float32)
            for c_idx, (cam, rel, keep) in enumerate(zip(self.cameras, row_k["images"], mask_bits, strict=True)):
                if not keep or rel is None:
                    continue
                sh, sw = cam["image_hw"]
                uv, _, valid = project(pts_k, cam, (sh, sw))
                cam_from_ego = rigid_matrix(cam["T_cam_from_ego"], cam["slot"])
                ego_from_cam = np.linalg.inv(cam_from_ego)
                cam_xy = ego_from_cam[:2, 3]
                cam_fwd = ego_from_cam[:2, 2]
                dx, dy = pts_k[..., 0] - cam_xy[0], pts_k[..., 1] - cam_xy[1]
                cos_dir = np.clip((dx * cam_fwd[0] + dy * cam_fwd[1]) / np.hypot(dx, dy).clip(1e-4), 0.0, 1.0)
                w = (cos_dir ** 3) * valid * not_ego_k
                better = w > best_w + 1e-4
                if not better.any():
                    continue
                arr = self._decode_rgb_half(child_path(raw_root, rel)).astype(np.float32) / 255.0
                dh, dw = arr.shape[:2]
                u_idx = np.clip(np.rint(uv[better, 0] * (dw / sw)).astype(np.int32), 0, dw - 1)
                v_idx = np.clip(np.rint(uv[better, 1] * (dh / sh)).astype(np.int32), 0, dh - 1)
                frame_rgb[better] = arr[v_idx, u_idx]
                best_w[better] = w[better]

            ok_k = best_w > 0.05
            if ok_k.any():
                canvas[ok_k] = frame_rgb[ok_k]
                filled |= ok_k

        return _to_jpeg((np.clip(canvas, 0.0, 1.0) * 255.0).astype(np.uint8), quality=86)

    @torch.no_grad()
    def render_rect_jpeg(self, raw_root, row, cam_idx):
        rel = row["images"][cam_idx]
        if rel is None:
            return None
        arr = self._decode_rgb_half(child_path(raw_root, rel))
        img_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32, non_blocking=True) / 255.0
        rect = F.grid_sample(img_t, self.rect_grids[cam_idx].unsqueeze(0),
                             mode="bilinear", align_corners=False)[0].permute(1, 2, 0).clamp(0, 1)
        return _to_jpeg((rect * 255.0).byte().cpu().numpy(), quality=84)

    @torch.no_grad()
    def render_wide_jpeg(self, raw_root, row, cam_idx, mirror=False):
        rel = row["images"][cam_idx]
        if rel is None:
            return None
        arr = self._decode_rgb_half(child_path(raw_root, rel))
        img_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32, non_blocking=True) / 255.0
        wide = F.grid_sample(img_t, self.wide_grids[cam_idx].unsqueeze(0),
                             mode="bilinear", align_corners=False)[0].permute(1, 2, 0).clamp(0, 1)
        if mirror:
            wide = torch.flip(wide, dims=[1])
        return _to_jpeg((wide * 255.0).byte().cpu().numpy(), quality=84)


class DemoRuntime:
    def __init__(self, args):
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for demo but PyTorch CUDA is unavailable")
        ai_enabled = (not getattr(args, "no_ai", False)) and bool(getattr(args, "checkpoint", None))
        if ai_enabled:
            state = load_checkpoint(args.checkpoint)
            config = state["model_config"]
            if config["backbone"] != "small_cnn" and dino_revision(args.dino_source) != state["dino_revision"]:
                raise ValueError("DINOv3 source revision differs from checkpoint")
            model = OmniShieldEdge(config, args.dino_source, initialize=False).to(device).eval()
            missing, unexpected = model.load_state_dict(state["model"], strict=False)
            if unexpected or (set(missing) - {"occupancy.weight", "occupancy.bias"}):
                raise RuntimeError(f"Incompatible checkpoint keys: missing={missing}, unexpected={unexpected}")
            if missing:
                torch.nn.init.zeros_(model.occupancy.weight)
                torch.nn.init.constant_(model.occupancy.bias, -20.0)
            supervised = set(state.get("supervised_heads", ()))
        else:
            model_cfg_path = getattr(args, "model", None) or (Path(__file__).resolve().parents[2] / "configs/model/edge_mvp.json")
            config = load_json(model_cfg_path)
            model = None
            supervised = set()
        manifest = load_manifest(args.prepared / "manifest.json", config, allow_synthetic=args.allow_synthetic)
        dataset = OccupancyDataset(manifest, config, args.raw, args.prepared, args.split)
        if not len(dataset):
            raise ValueError(f"Empty {args.split} split")

        self.args = args
        self.ai_enabled = ai_enabled
        self.config = config
        self.grid = Grid.from_config(config["grid"])
        self.device = device
        self.device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        self.model = model
        self.dataset = dataset
        self.supervised = supervised
        self.lock = threading.Lock()
        self._field_cache = {}
        self._mesh_cache = {}
        self._splat_cache = {}
        self._depth_surface_cache = {}
        d_edges = depth_edges(config["depth"])
        self.depth_centers = (0.5 * (d_edges[:-1] + d_edges[1:])).astype(np.float32)
        self.lut_tensors = {}
        if ai_enabled:
            for rig_id in dataset.rigs:
                lut = dataset.lut(rig_id)
                self.lut_tensors[rig_id] = {
                    "uv": torch.from_numpy(lut["uv"]).unsqueeze(0).to(device),
                    "bins": torch.from_numpy(lut["bins"]).unsqueeze(0).to(device),
                    "geometry_valid": torch.from_numpy(lut["valid"]).unsqueeze(0).to(device),
                }

        # Group dataset rows by video clip (drive_id) & run cross-clip edge-weighted BA once per physical rig_id!
        self.surround = {}
        self.calibrated_rigs = {}
        self.clips_meta = []
        clip_groups = {}
        rig_rows = {}
        for idx, row in enumerate(dataset.rows):
            rig_rows.setdefault(row["rig_id"], []).append(row)
            key = (row["rig_id"], row["drive_id"])
            if key not in clip_groups:
                clip_groups[key] = {"rig_id": row["rig_id"], "drive_id": row["drive_id"], "split": row["split"], "indices": [], "rows": []}
            clip_groups[key]["indices"].append(idx)
            clip_groups[key]["rows"].append(row)

        rig_calib_cache = {}
        self._raw_decoders = {}
        self.poses = [_load_pose_4x4(args.raw, row) for row in dataset.rows]
        external_profile = (
            load_calibrated_rig_profile(args.calibrated_rig)
            if getattr(args, "calibrated_rig", None)
            else None
        )
        is_real_phenikaa = not getattr(args, "allow_synthetic", False) and manifest.get("source") == "phenikaa"
        for (rig_id, drive_id), grp in clip_groups.items():
            rig = dataset.rigs[rig_id]
            clip_rows = grp["rows"]
            if rig_id not in rig_calib_cache:
                all_r = rig_rows[rig_id]
                cache_file = args.prepared / f".calib_cache_v5_{rig_id}.json"
                loaded = None
                if external_profile is not None:
                    ext_rig = external_profile["rig"]
                    ext_plane = external_profile["ground_plane"]
                    ext_rot = external_profile.get("quality", {}).get("rot_deg", [0.0] * len(ext_rig["cameras"]))
                    loaded = (ext_rig, ext_rot, ext_plane, {})
                elif is_real_phenikaa and cache_file.is_file():
                    try:
                        cobj = load_json(cache_file)
                        if cobj.get("n_frames") == len(all_r) and cobj.get("rig_id") == rig_id:
                            loaded = (cobj["calib_rig"], cobj["rot_deg"], cobj["base_plane"], cobj.get("clip_planes", {}))
                    except (OSError, ValueError, KeyError):
                        loaded = None
                if loaded is not None:
                    calib_rig, rot_deg, base_plane, clip_planes = loaded
                elif is_real_phenikaa:
                    profile = build_standard_rig_calibration(
                        rig, all_r, args.raw, mode="auto", load_targets_fn=load_targets,
                        sample_path_fn=sample_path, prepared_root=args.prepared, config=config,
                        depth_centers=self.depth_centers, device=device, poses=self.poses,
                    )
                    calib_rig = profile["rig"]
                    rot_deg = profile["quality"]["rot_deg"]
                    base_plane = profile["ground_plane"]
                    clip_planes = {}
                else:
                    calib_rig = rig
                    base_plane = [0.0, 0.0, float(rig.get("ground_height_m", 0.0))]
                    rot_deg = [0.0] * len(rig["cameras"])
                    clip_planes = {}
                rig_calib_cache[rig_id] = [calib_rig, rot_deg, base_plane, clip_planes, cache_file, (loaded is None) and (external_profile is None)]
                if ai_enabled:
                    raw_calibs = export_camera_calib(rig)
                    self._raw_decoders[rig_id] = FeedForward3DGSDecoder(
                        rig, raw_calibs, tuple(config["image_hw"]), feature_stride=4, ground_z=base_plane[2]
                    )

            calib_rig, rot_deg, base_plane, clip_planes, cache_file, cache_dirty = rig_calib_cache[rig_id]
            plane_quality = {}
            if external_profile is not None:
                # Honor the calibrated profile's exact ground plane rather than overwriting with stale clip cache
                road_plane = list(base_plane)
                plane_quality = external_profile.get("road_plane_validation", {"source": "calibrated_profile"})
            elif is_real_phenikaa:
                if drive_id in clip_planes:
                    road_plane = clip_planes[drive_id]
                else:
                    road_plane, _ = estimate_road_plane(
                        rig, clip_rows, load_targets, sample_path, args.prepared, config, self.depth_centers
                    )
                    clip_planes[drive_id] = road_plane
                    rig_calib_cache[rig_id][5] = True
                scan_ids = [row.get("lidar") for row in clip_rows]
                signature = fingerprint([rig, scan_ids])
                plane_cache = args.prepared / f".road_plane_v1_{signature}.json"
                measured = None
                if plane_cache.is_file():
                    try:
                        measured = load_json(plane_cache)
                    except (ValueError, OSError):
                        pass
                if measured is None:
                    measured = fit_lidar_road_plane(rig, clip_rows, args.raw)
                    if measured is not None:
                        try:
                            write_json(plane_cache, measured)
                        except OSError:
                            pass
                if measured is not None:
                    road_plane, plane_quality = measured
                else:
                    plane_quality = {"source": "prepared_depth_or_rig"}
            else:
                road_plane = base_plane
            self.calibrated_rigs[drive_id] = calib_rig
            sr = SurroundViewRenderer(calib_rig, device, tuple(config["image_hw"]), ground_z=road_plane[2], ground_plane=road_plane)
            sr.guidelines["road_plane_validation"] = plane_quality
            sr.guidelines["extrinsic_refine_deg"] = rot_deg
            sr.guidelines["calib_dz_cm"] = calib_rig.get("calib_dz_cm", [0.0] * len(rig["cameras"]))
            self.surround[drive_id] = sr
            self.clips_meta.append({
                "drive_id": drive_id,
                "split": grp["split"],
                "start_index": grp["indices"][0],
                "count": len(grp["indices"]),
                "surround": sr.guidelines,
            })

        if is_real_phenikaa:
            for rig_id, (calib_rig, rot_deg, base_plane, clip_planes, cache_file, cache_dirty) in rig_calib_cache.items():
                if cache_dirty:
                    try:
                        write_json(cache_file, {
                            "rig_id": rig_id,
                            "n_frames": len(rig_rows[rig_id]),
                            "calib_rig": calib_rig,
                            "rot_deg": rot_deg,
                            "base_plane": base_plane,
                            "clip_planes": clip_planes,
                        })
                    except OSError:
                        pass

        self.frame_meta = []
        clip_start_ts = {grp["drive_id"]: grp["rows"][0]["timestamp_ns"] for grp in clip_groups.values()}
        prev_speeds = []
        for idx, row in enumerate(dataset.rows):
            odom = [0.0, 0.0, 0.0]
            steer_rad = 0.0
            raw_speed = 0.0
            if idx > 0 and dataset.rows[idx - 1]["drive_id"] == row["drive_id"]:
                t_rel = np.linalg.inv(self.poses[idx]) @ self.poses[idx - 1]
                dx, dy = float(t_rel[0, 3]), float(t_rel[1, 3])
                dyaw = float(np.arctan2(t_rel[1, 0], t_rel[0, 0]))
                odom = [round(dx, 4), round(dy, 4), round(dyaw, 5)]
                dt_s = max((row["timestamp_ns"] - dataset.rows[idx - 1]["timestamp_ns"]) / 1e9, 0.02)
                raw_speed = float(np.clip((np.hypot(dx, dy) / dt_s) * 3.6, 0.0, 140.0))
                fwd_dist = -dx
                if abs(fwd_dist) > 0.12:
                    wb = float(self.surround[row["drive_id"]].guidelines["vehicle_spec"]["wheelbase_m"])
                    steer_rad = float(np.clip(np.arctan2(wb * (-dyaw), abs(fwd_dist)), -0.45, 0.45))
            else:
                prev_speeds.clear()
            prev_speeds.append(raw_speed)
            if len(prev_speeds) > 3:
                prev_speeds.pop(0)
            w_sp = [0.5, 0.3, 0.2][:len(prev_speeds)]
            smooth_speed = sum(v * w for v, w in zip(reversed(prev_speeds), w_sp)) / sum(w_sp)
            entry = {
                "index": idx,
                "drive_id": row["drive_id"],
                "frame_id": row["frame_id"],
                "timestamp_s": round((row["timestamp_ns"] - clip_start_ts[row["drive_id"]]) / 1e9, 2),
                "images_present": [img is not None for img in row["images"]],
                "lidar_present": False,
                "odom_from_prev": odom,
                "steer_rad": round(steer_rad, 4),
                "speed_kmh": round(float(smooth_speed), 1),
            }
            if ai_enabled:
                targets = load_targets(sample_path(args.prepared, row), config)
                occ = targets["occupancy"] == 1
                sem = targets["semantics"]
                entry["lidar_present"] = bool(row.get("lidar")) or bool(np.any(targets.get("depth", -1) >= 0))
                for cls_idx, cls_name in enumerate(config["classes"][1:], 1):
                    entry[cls_name] = int((occ & (sem == cls_idx)).sum())
            self.frame_meta.append(entry)

        if self.ai_enabled and device.type == "cuda":
            self.infer(0, [True] * config["camera_count"], args.threshold, args.max_uncertainty)

    def metadata(self):
        first_drive = self.dataset.rows[0]["drive_id"]
        return {
            "source": first_drive,
            "split": self.args.split,
            "ai_enabled": self.ai_enabled,
            "device": self.device_name,
            "device_type": self.device.type,
            "torch": torch.__version__,
            "classes": self.config["classes"],
            "grid": {
                "x": self.config["grid"]["x"],
                "y": self.config["grid"]["y"],
                "z_edges": self.config["grid"]["z_edges"],
                "shape": list(self.grid.shape),
            },
            "surround": self.surround[first_drive].guidelines,
            "clips": self.clips_meta,
            "default_threshold": self.args.threshold,
            "default_max_uncertainty": self.args.max_uncertainty,
            "frames": self.frame_meta,
        }

    def image_bytes(self, index, cam, width=0):
        row = self.dataset.rows[index]
        rel = row["images"][cam]
        if rel is None:
            return None, None
        if width and width > 0:
            return self._image_resized_jpeg(index, cam, int(width)), "image/jpeg"
        path = child_path(self.args.raw, rel)
        mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        return path.read_bytes(), mime

    @lru_cache(maxsize=1600)
    def _image_resized_jpeg(self, index, cam, width):
        row = self.dataset.rows[index]
        path = child_path(self.args.raw, row["images"][cam])
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            if rgb.width > width:
                h = max(1, int(round(rgb.height * width / rgb.width)))
                rgb = rgb.resize((width, h), Image.Resampling.BILINEAR)
            return _to_jpeg(np.asarray(rgb, dtype=np.uint8), quality=80)

    def bev_jpeg(self, index, mask_bits):
        row = self.dataset.rows[index]
        return self.surround[row["drive_id"]].render_bev_jpeg(self.args.raw, row, mask_bits)

    @lru_cache(maxsize=128)
    def rect_jpeg(self, index, cam):
        row = self.dataset.rows[index]
        return self.surround[row["drive_id"]].render_rect_jpeg(self.args.raw, row, cam)

    def wide_jpeg(self, index, cam, mirror=False):
        row = self.dataset.rows[index]
        return self.surround[row["drive_id"]].render_wide_jpeg(self.args.raw, row, cam, mirror=mirror)

    @lru_cache(maxsize=64)
    def _underbody_jpeg_cached(self, index, mask_tuple):
        row = self.dataset.rows[index]
        return self.surround[row["drive_id"]].render_underbody_seed_jpeg(
            self.args.raw, self.dataset.rows, self.poses, index, list(mask_tuple)
        )

    def underbody_jpeg(self, index, mask_bits):
        return self._underbody_jpeg_cached(index, tuple(bool(b) for b in mask_bits))

    def bowl_binary(self):
        first_drive = self.dataset.rows[0]["drive_id"]
        return self.surround[first_drive].bowl_binary

    @lru_cache(maxsize=256)
    def foreground_binary(self, index):
        if not self.ai_enabled:
            return None
        row = self.dataset.rows[index]
        raw_rig = self.dataset.rigs[row["rig_id"]]
        calib_rig = self.calibrated_rigs[row["drive_id"]]
        surround = self.surround[row["drive_id"]]
        targets = load_targets(sample_path(self.args.prepared, row), self.config)
        points = None
        if row.get("lidar"):
            try:
                points, _ = load_lidar(self.args.raw, row, raw_rig)
            except (ImportError, OSError):
                points = None
        if points is None:
            # Unproject prepared LiDAR depth targets using pre-built raw-rig rays (0ms setup overhead!)
            db = targets.get("depth")
            if db is None or not (db >= 0).any():
                return None
            dec = self._raw_decoders[row["rig_id"]]
            pts_list = []
            for ci in range(len(raw_rig["cameras"])):
                v = dec.ray_valid[ci] & (db[ci] >= 0) & (db[ci] < len(self.depth_centers))
                if not v.any():
                    continue
                dm = self.depth_centers[np.clip(db[ci], 0, len(self.depth_centers) - 1)]
                v &= (dm > 0.35) & (dm < 14.0)
                if v.any():
                    pts_list.append(dec.origins[ci] + dec.rays_ego[ci][v] * dm[v, None])
            if not pts_list:
                return None
            points = np.concatenate(pts_list, axis=0)

        # Exclude ALL dynamic objects (vehicles, riders, pedestrians) from visibility, 2.5D curb surface, & r(theta) walls
        static_points = points
        sem = targets.get("semantics")
        occ = targets.get("occupancy")
        if sem is not None and occ is not None:
            from scipy.ndimage import binary_dilation
            gx0, gx1 = float(self.grid.x[0]), float(self.grid.x[-1])
            gy0, gy1 = float(self.grid.y[0]), float(self.grid.y[-1])
            dx_g = (gx1 - gx0) / self.grid.shape[2]
            dy_g = (gy1 - gy0) / self.grid.shape[1]
            xi = np.clip(((points[:, 0] - gx0) / dx_g).astype(np.int32), 0, self.grid.shape[2] - 1)
            yi = np.clip(((points[:, 1] - gy0) / dy_g).astype(np.int32), 0, self.grid.shape[1] - 1)
            in_grid = (points[:, 0] >= gx0) & (points[:, 0] < gx1) & (points[:, 1] >= gy0) & (points[:, 1] < gy1)
            dyn_2d = ((occ == 1) & (sem > 0)).any(axis=0)
            if dyn_2d.any():
                dyn_dil = binary_dilation(dyn_2d, iterations=3)
                static_points = points[~(in_grid & dyn_dil[yi, xi])]

        return pack_foreground_ranges(
            static_points, calib_rig, ground_z=surround.ground_z,
            static_points=static_points, ground_plane=surround.ground_plane,
        )

    def _cache_put(self, store, key, val, limit=6):
        store[key] = val
        while len(store) > limit:
            store.pop(next(iter(store)))

    def mesh_binary(self, index, mask_bits, threshold, max_uncertainty, source="pred"):
        key = (index, tuple(bool(b) for b in mask_bits), round(float(threshold), 3), round(float(max_uncertainty), 3), source)
        if key in self._mesh_cache:
            return self._mesh_cache[key]
        field_key = key[:-1]
        if field_key not in self._field_cache:
            self.infer(index, mask_bits, threshold, max_uncertainty)
        field, level = self._field_cache[field_key][source]
        blob = build_smooth_occupancy_mesh_binary(field, self.grid, level=level)
        self._cache_put(self._mesh_cache, key, blob, limit=8)
        return blob

    def splats_binary(self, index, mask_bits, threshold, max_uncertainty, source="pred"):
        key = (index, tuple(bool(b) for b in mask_bits), round(float(threshold), 3), round(float(max_uncertainty), 3), source)
        if key in self._splat_cache:
            return self._splat_cache[key]
        field_key = key[:-1]
        if field_key not in self._field_cache:
            self.infer(index, mask_bits, threshold, max_uncertainty)
        entry = self._field_cache[field_key]
        row = self.dataset.rows[index]
        decoder = self.surround[row["drive_id"]].gs_decoder
        if source == "gt":
            blob = decoder.decode_splats_binary(
                entry["gt_depth_m"], entry["imgs_rgb"], entry["cam_gains"], entry["sensor_valid"],
                gt_valid_mask=entry["gt_depth_valid"],
            )
        else:
            blob = decoder.decode_splats_binary(
                entry["pred_depth_m"], entry["imgs_rgb"], entry["cam_gains"], entry["sensor_valid"],
                gt_valid_mask=None,
            )
        self._cache_put(self._splat_cache, key, blob, limit=8)
        return blob

    def depth_surface_binary(self, index, mask_bits, threshold, max_uncertainty, source="pred"):
        key = (index, tuple(bool(b) for b in mask_bits), round(float(threshold), 3), round(float(max_uncertainty), 3), source)
        if key in self._depth_surface_cache:
            return self._depth_surface_cache[key]
        field_key = key[:-1]
        if field_key not in self._field_cache:
            self.infer(index, mask_bits, threshold, max_uncertainty)
        entry = self._field_cache[field_key]
        row = self.dataset.rows[index]
        sr = self.surround[row["drive_id"]]
        decoder = sr.gs_decoder
        if source == "gt":
            blob = decoder.decode_depth_surface_binary(
                entry["gt_depth_m"], entry["sensor_valid"],
                ground_plane=sr.ground_plane, gt_valid_mask=entry["gt_depth_valid"],
            )
        else:
            blob = decoder.decode_depth_surface_binary(
                entry["pred_depth_m"], entry["sensor_valid"],
                ground_plane=sr.ground_plane, gt_valid_mask=None,
            )
        self._cache_put(self._depth_surface_cache, key, blob, limit=8)
        return blob

    @torch.no_grad()
    def infer(self, index, mask_bits, threshold, max_uncertainty):
        row = self.dataset.rows[index]
        if not self.ai_enabled:
            empty_pack = {
                "voxels": [], "counts": {"occupied": 0}, "objects": [], "road_edges": [],
                "pdc_sectors": [], "corridor": None, "seam_zones": [],
            }
            return {
                "index": index,
                "frame_id": row["frame_id"],
                "drive_id": row["drive_id"],
                "ai_enabled": False,
                "latency_ms": 0.0,
                "p_occ_max": 0.5,
                "threshold": threshold,
                "max_uncertainty": max_uncertainty,
                "sensor_valid": [bool(b) for b in mask_bits],
                "metrics": {"occupancy_iou": None, "coverage": None},
                "adaptive_bowl": {"inflect_radii": [], "cam_gains": [[1, 1, 1]] * len(mask_bits), "corridor": None, "seam_zones": []},
                "prediction": empty_pack,
                "ground_truth": empty_pack,
            }
        rig = self.dataset.rigs[row["rig_id"]]
        inputs_np = self.dataset.inputs(index)
        images = torch.from_numpy(inputs_np["images"]).unsqueeze(0).to(self.device)
        sensor_valid_np = np.asarray(inputs_np["sensor_valid"], dtype=bool) & np.asarray(mask_bits, dtype=bool)
        sensor_valid = torch.from_numpy(sensor_valid_np).unsqueeze(0).to(self.device)
        lut = self.lut_tensors[row["rig_id"]]

        with self.lock:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()
            output = self.model(
                images=images,
                sensor_valid=sensor_valid,
                uv=lut["uv"],
                bins=lut["bins"],
                geometry_valid=lut["geometry_valid"],
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            cam_gains = self.surround[row["drive_id"]].solve_cam_gains(images[0], sensor_valid_np.tolist())

        targets_np = load_targets(sample_path(self.args.prepared, row), self.config)
        targets_t = {k: torch.from_numpy(v).unsqueeze(0).to(self.device) for k, v in targets_np.items()}
        p_occ = output["p_occ"][0].detach().cpu().numpy()
        mass = output["mass"][0].detach().cpu().numpy()
        valid = output["valid"][0].detach().cpu().numpy()
        known = valid & (mass[2] <= max_uncertainty)
        pred_occ = np.where(known, p_occ > threshold, -1).astype(np.int8)
        pred_sem = output["semantic_prob"][0].detach().cpu().numpy().argmax(axis=0) if "semantics" in self.supervised else None

        classes = self.config["classes"]
        pred_objs, dyn_mask_pred = extract_dynamic_objects(
            pred_occ, pred_sem, self.grid, rig, classes, p_occ=p_occ, is_prediction=True
        )
        gt_objs, dyn_mask_gt = extract_dynamic_objects(
            targets_np["occupancy"], targets_np["semantics"], self.grid, rig, classes, is_prediction=False
        )
        adapt_radii = compute_adaptive_bowl_radii(pred_occ, pred_sem, self.grid, rig, p_occ=p_occ)
        corridor_pred = compute_corridor_profile(pred_occ, dyn_mask_pred, self.grid, rig, p_occ=p_occ, is_prediction=True)
        corridor_gt = compute_corridor_profile(targets_np["occupancy"], dyn_mask_gt, self.grid, rig, is_prediction=False)
        road_edges_pred = extract_vector_road_edges(pred_occ, dyn_mask_pred, self.grid)
        road_edges_gt = extract_vector_road_edges(targets_np["occupancy"], dyn_mask_gt, self.grid)

        # Filter out false AI boxes spawning outside a confirmed curb/median barrier (e.g. inside bushes)
        def _filter_behind_curb(objs, edges):
            if not objs or not edges:
                return objs
            left_pts = [pt for poly in edges for pt in poly if pt[1] > 1.20]
            right_pts = [pt for poly in edges for pt in poly if pt[1] < -1.20]
            out = []
            for o in objs:
                ox, oy = float(o["x"]), float(o["y"])
                blocked = False
                if oy > 1.35 and len(left_pts) >= 4:
                    near_l = [p[1] for p in left_pts if abs(p[0] - ox) < 2.2]
                    if near_l and oy > min(near_l) + 0.45:
                        blocked = True
                if oy < -1.35 and len(right_pts) >= 4:
                    near_r = [p[1] for p in right_pts if abs(p[0] - ox) < 2.2]
                    if near_r and oy < max(near_r) - 0.45:
                        blocked = True
                if not blocked:
                    out.append(o)
            return out

        pred_objs = _filter_behind_curb(pred_objs, road_edges_pred)
        seam_zones_pred = extract_seam_shift_zones(
            pred_occ, self.grid, rig, corridor_pred, pred_objs, p_occ=p_occ, is_prediction=True
        )
        seam_zones_gt = extract_seam_shift_zones(
            targets_np["occupancy"], self.grid, rig, corridor_gt, gt_objs, is_prediction=False
        )
        pdc_pred = compute_pdc_sectors(pred_occ, pred_sem, self.grid, rig, p_occ=p_occ, is_prediction=True)
        pdc_gt = compute_pdc_sectors(targets_np["occupancy"], targets_np["semantics"], self.grid, rig, is_prediction=False)

        d_logits = output["depth"][0][:, :len(self.depth_centers)].float().softmax(dim=1).detach().cpu().numpy()
        pred_depth_m = (d_logits * self.depth_centers[None, :, None, None]).sum(axis=1).astype(np.float32)
        gt_depth_bin = targets_np["depth"]
        gt_depth_valid = gt_depth_bin >= 0
        gt_depth_m = self.depth_centers[np.clip(gt_depth_bin, 0, len(self.depth_centers) - 1)]

        img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        imgs_rgb_cpu = (images[0].detach().cpu() * img_std + img_mean).clamp(0.0, 1.0)

        static_pred_field = np.where(known & (~dyn_mask_pred), np.clip(p_occ, 0.0, 1.0), 0.0).astype(np.float32)
        static_gt_field = ((targets_np["occupancy"] == 1) & (~dyn_mask_gt)).astype(np.float32)
        field_key = (index, tuple(bool(b) for b in mask_bits), round(float(threshold), 3), round(float(max_uncertainty), 3))
        self._cache_put(
            self._field_cache,
            field_key,
            {
                "pred": (static_pred_field, float(np.clip(threshold * 0.88, 0.20, 0.78))),
                "gt": (static_gt_field, 0.42),
                "pred_depth_m": pred_depth_m,
                "gt_depth_m": gt_depth_m,
                "gt_depth_valid": gt_depth_valid,
                "imgs_rgb": imgs_rgb_cpu,
                "cam_gains": cam_gains,
                "sensor_valid": sensor_valid_np.tolist(),
            },
            limit=6,
        )

        metrics = OccupancyMetrics(len(classes), self.device)
        metrics.update(
            {
                "p_occ": torch.from_numpy((pred_occ == 1).astype(np.float32)).unsqueeze(0).to(self.device),
                "valid": torch.from_numpy(known).unsqueeze(0).to(self.device),
                "semantic_prob": output["semantic_prob"],
            },
            targets_t,
        )
        pred_pack = _encode_grid(pred_occ, pred_sem, classes)
        pred_pack["objects"] = pred_objs
        pred_pack["road_edges"] = road_edges_pred
        pred_pack["pdc_sectors"] = pdc_pred
        pred_pack["corridor"] = corridor_pred
        pred_pack["seam_zones"] = seam_zones_pred
        gt_pack = _encode_grid(targets_np["occupancy"], targets_np["semantics"], classes)
        gt_pack["objects"] = gt_objs
        gt_pack["road_edges"] = road_edges_gt
        gt_pack["pdc_sectors"] = pdc_gt
        gt_pack["corridor"] = corridor_gt
        gt_pack["seam_zones"] = seam_zones_gt

        return {
            "index": index,
            "frame_id": row["frame_id"],
            "drive_id": row["drive_id"],
            "latency_ms": round(latency_ms, 1),
            "p_occ_max": round(float(p_occ[valid].max()) if valid.any() else 0.5, 4),
            "threshold": threshold,
            "max_uncertainty": max_uncertainty,
            "sensor_valid": sensor_valid_np.tolist(),
            "metrics": metrics.result(),
            "adaptive_bowl": {
                "inflect_radii": adapt_radii,
                "cam_gains": cam_gains,
                "corridor": corridor_pred,
                "seam_zones": seam_zones_pred,
            },
            "prediction": pred_pack,
            "ground_truth": gt_pack,
        }


def run_calibrate_rig(args):
    """CLI handler for standardized 6-DoF rig calibration (LiDAR-assisted or targetless camera-only)."""
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if args.prepared is not None or args.manifest is not None:
        config = load_json(args.model)
        manifest_path = args.manifest or (args.prepared / "manifest.json")
        manifest = load_manifest(manifest_path, config, allow_synthetic=args.allow_synthetic)
        rig_id = next(iter(manifest["rigs"]))
        rig = manifest["rigs"][rig_id]
        rig["rig_id"] = rig_id
        rows = [r for r in manifest["frames"] if r["rig_id"] == rig_id]
        poses = [_load_pose_4x4(args.raw, r) for r in rows]
        d_edges = depth_edges(config["depth"])
        depth_centers = (0.5 * (d_edges[:-1] + d_edges[1:])).astype(np.float32)
        use_targets = (args.mode != "camera-only") and (args.prepared is not None)
        profile = build_standard_rig_calibration(
            rig, rows, args.raw, mode=args.mode,
            load_targets_fn=load_targets if use_targets else None,
            sample_path_fn=sample_path if use_targets else None,
            prepared_root=args.prepared if use_targets else None,
            config=config if use_targets else None,
            depth_centers=depth_centers if use_targets else None,
            device=device, poses=poses,
        )
    elif args.rig is not None:
        raw_root = Path(args.raw).resolve()
        rig_cfg = load_json(args.rig)
        if "cameras" in rig_cfg:
            rig = rig_cfg
        elif "rig" in rig_cfg and "cameras" in rig_cfg["rig"]:
            rig = rig_cfg["rig"]
        else:
            raise ValueError("Standalone --rig JSON must contain 'cameras' or a 'rig.cameras' profile")
        slot_dirs = {
            "front": ("front", "CAM_F_F"),
            "rear": ("rear", "CAM_F_B"),
            "left": ("left", "CAM_F_L"),
            "right": ("right", "CAM_F_R"),
        }
        cam_files = {}
        for cam in rig["cameras"]:
            slot = cam["slot"]
            found = []
            for dname in slot_dirs.get(slot, (slot,)):
                for sub in (raw_root / dname, raw_root / "camera" / dname):
                    if sub.is_dir():
                        found.extend(sorted(p for p in sub.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}))
            if not found:
                raise ValueError(f"No images found for slot '{slot}' under {raw_root}")
            cam_files[slot] = found
        n_frames = min(len(v) for v in cam_files.values())
        rows = [
            {"drive_id": "customer_calib", "frame_id": f"{i:06d}",
             "images": [cam_files[cam["slot"]][i].relative_to(raw_root).as_posix() for cam in rig["cameras"]]}
            for i in range(n_frames)
        ]
        profile = build_standard_rig_calibration(rig, rows, raw_root, mode="camera-only", device=device)
    else:
        raise ValueError("Provide --prepared (with manifest) or --rig (standalone camera-only calibration)")

    write_json(args.out, profile)
    q = profile["quality"]
    print(
        f"Saved RigCalibrationProfile ({profile['calibration_mode']}) -> {args.out}\n"
        f"  ground_plane={profile['ground_plane']} | rot_deg={q['rot_deg']} | calib_dz_cm={q['calib_dz_cm']}"
    )


def serve_demo(args):
    runtime = DemoRuntime(args)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *fmt_args):
            return

        def _send(self, status, body, content_type, cache_control="no-store"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache_control)
            self.end_headers()
            try:
                self.wfile.write(body)
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                # Navigating or seeking may cancel an image already being sent.
                self.close_connection = True

        def do_GET(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path in {"/", "/index.html"}:
                self._send(HTTPStatus.OK, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
                return
            if parsed.path.startswith("/vendor/"):
                name = Path(parsed.path).name
                target = VENDOR_DIR / name
                if target.is_file() and name.endswith(".js"):
                    self._send(HTTPStatus.OK, target.read_bytes(), "application/javascript; charset=utf-8")
                else:
                    self._send(HTTPStatus.NOT_FOUND, b"Not found", "text/plain")
                return
            if parsed.path == "/api/bowl.bin":
                self._send(HTTPStatus.OK, runtime.bowl_binary(), "application/octet-stream")
                return
            if parsed.path == "/api/meta":
                self._send(HTTPStatus.OK, json.dumps(runtime.metadata(), ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")
                return
            idx = max(0, min(len(runtime.dataset) - 1, int(query.get("index", ["0"])[0])))
            if not runtime.ai_enabled and parsed.path in {"/api/mesh.bin", "/api/splats.bin", "/api/foreground.bin", "/api/depth_surface.bin"}:
                self._send(HTTPStatus.NOT_FOUND, b"Camera-only replay", "text/plain")
                return
            if parsed.path == "/api/foreground.bin":
                try:
                    data = runtime.foreground_binary(idx)
                except (ImportError, OSError):
                    data = None
                self._send(HTTPStatus.OK if data else HTTPStatus.NOT_FOUND, data or b"No LiDAR depth",
                           "application/octet-stream" if data else "text/plain")
                return
            if parsed.path == "/api/image":
                cam = max(0, min(runtime.config["camera_count"] - 1, int(query.get("cam", ["0"])[0])))
                req_w = int(query.get("w", ["0"])[0])
                data, mime = runtime.image_bytes(idx, cam, width=req_w if req_w in {480, 640, 800, 960} else 0)
                self._send(
                    HTTPStatus.OK if data else HTTPStatus.NOT_FOUND,
                    data or b"No image",
                    mime or "text/plain",
                    cache_control="public, max-age=3600" if data else "no-store",
                )
                return
            if parsed.path == "/api/cam_rect":
                cam = max(0, min(runtime.config["camera_count"] - 1, int(query.get("cam", ["0"])[0])))
                data = runtime.rect_jpeg(idx, cam)
                self._send(
                    HTTPStatus.OK if data else HTTPStatus.NOT_FOUND,
                    data or b"No image",
                    "image/jpeg" if data else "text/plain",
                    cache_control="public, max-age=3600" if data else "no-store",
                )
                return
            if parsed.path == "/api/cam_wide":
                cam = max(0, min(runtime.config["camera_count"] - 1, int(query.get("cam", ["0"])[0])))
                mirror = query.get("mirror", ["0"])[0] == "1"
                data = runtime.wide_jpeg(idx, cam, mirror=mirror)
                self._send(HTTPStatus.OK if data else HTTPStatus.NOT_FOUND, data or b"No image",
                           "image/jpeg" if data else "text/plain")
                return
            raw_mask = query.get("mask", ["1111"])[0]
            mask_bits = [ch == "1" for ch in (raw_mask + "1111")[: runtime.config["camera_count"]]]
            if parsed.path == "/api/bev":
                self._send(HTTPStatus.OK, runtime.bev_jpeg(idx, mask_bits), "image/jpeg")
                return
            if parsed.path == "/api/underbody":
                self._send(HTTPStatus.OK, runtime.underbody_jpeg(idx, mask_bits), "image/jpeg")
                return
            if parsed.path == "/api/mesh.bin":
                threshold = float(query.get("threshold", [str(args.threshold)])[0])
                max_u = float(query.get("max_u", [str(args.max_uncertainty)])[0])
                source = query.get("source", ["pred"])[0]
                if source not in {"pred", "gt"}:
                    source = "pred"
                self._send(HTTPStatus.OK, runtime.mesh_binary(idx, mask_bits, threshold, max_u, source=source),
                           "application/octet-stream")
                return
            if parsed.path == "/api/splats.bin":
                threshold = float(query.get("threshold", [str(args.threshold)])[0])
                max_u = float(query.get("max_u", [str(args.max_uncertainty)])[0])
                source = query.get("source", ["pred"])[0]
                if source not in {"pred", "gt"}:
                    source = "pred"
                self._send(HTTPStatus.OK, runtime.splats_binary(idx, mask_bits, threshold, max_u, source=source),
                           "application/octet-stream")
                return
            if parsed.path == "/api/depth_surface.bin":
                threshold = float(query.get("threshold", [str(args.threshold)])[0])
                max_u = float(query.get("max_u", [str(args.max_uncertainty)])[0])
                source = query.get("source", ["pred"])[0]
                if source not in {"pred", "gt"}:
                    source = "pred"
                self._send(HTTPStatus.OK, runtime.depth_surface_binary(idx, mask_bits, threshold, max_u, source=source),
                           "application/octet-stream")
                return
            if parsed.path == "/api/infer":
                threshold = float(query.get("threshold", [str(args.threshold)])[0])
                max_u = float(query.get("max_u", [str(args.max_uncertainty)])[0])
                result = runtime.infer(idx, mask_bits, threshold, max_u)
                self._send(HTTPStatus.OK, json.dumps(result, ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")
                return
            self._send(HTTPStatus.NOT_FOUND, b"Not found", "text/plain")

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path not in {"/api/save_calibration", "/api/calibrate_pnp", "/api/auto_calibrate", "/api/export_lut"}:
                self._send(HTTPStatus.NOT_FOUND, b"Not found", "text/plain")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(max(0, min(length, 2_000_000))).decode("utf-8") if length > 0 else "{}"
                payload = json.loads(raw_body or "{}")

                first_drive = runtime.dataset.rows[0]["drive_id"]
                active_rig = runtime.calibrated_rigs[first_drive]
                active_gp = runtime.surround[first_drive].ground_plane

                if parsed.path == "/api/export_lut":
                    from omnishield.deploy.mat_and_lut import export_fixed_lut

                    export_rig = copy.deepcopy(active_rig)
                    cam_calibs = payload.get("camera_calib")
                    if isinstance(cam_calibs, list) and len(cam_calibs) == len(export_rig["cameras"]):
                        for cam, cc in zip(export_rig["cameras"], cam_calibs, strict=True):
                            cam["T_cam_from_ego"] = np.asarray(cc["t_cam_from_ego"], dtype=np.float64).reshape(4, 4).tolist()
                    gp = payload.get("ground_plane")
                    gp_list = [round(float(v), 6) for v in gp] if isinstance(gp, list) and len(gp) == 3 else list(active_gp)
                    profile = {
                        "schema_version": 1,
                        "calibration_mode": "exported_from_ui",
                        "rig_id": export_rig.get("rig_id", "default-rig"),
                        "ground_plane": gp_list,
                        "bowl_spec": export_bowl_spec(),
                        "rig": export_rig,
                        "camera_calib": export_camera_calib(export_rig),
                    }
                    summary = export_fixed_lut(profile, Path("runs/lut_export"))
                    self._send(
                        HTTPStatus.OK,
                        json.dumps({"ok": True, **summary}, ensure_ascii=False).encode("utf-8"),
                        "application/json; charset=utf-8",
                    )
                    return

                if parsed.path == "/api/calibrate_pnp":
                    ci = int(payload.get("cam_index", 0))
                    if ci < 0 or ci >= len(active_rig["cameras"]):
                        raise ValueError(f"Invalid cam_index: {ci}")
                    cam = copy.deepcopy(active_rig["cameras"][ci])
                    res = solve_fisheye_pnp_camera(cam, payload["image_points"], payload["world_points"])
                    updated_rig = copy.deepcopy(active_rig)
                    updated_rig["cameras"][ci]["T_cam_from_ego"] = res["T_cam_from_ego"]
                    for did in list(runtime.calibrated_rigs):
                        runtime.calibrated_rigs[did] = copy.deepcopy(updated_rig)
                    calibs_out = export_camera_calib(updated_rig)
                    for did in list(runtime.surround):
                        runtime.surround[did].guidelines["camera_calib"] = copy.deepcopy(calibs_out)
                    self._send(
                        HTTPStatus.OK,
                        json.dumps({
                            "ok": True,
                            "cam_index": ci,
                            "reproj_rmse_px": res["reproj_rmse_px"],
                            "max_err_px": res["max_err_px"],
                            "cam_origin": res["cam_origin"],
                            "camera_calib": calibs_out,
                        }, ensure_ascii=False).encode("utf-8"),
                        "application/json; charset=utf-8",
                    )
                    return

                if parsed.path == "/api/auto_calibrate":
                    raw_rig = copy.deepcopy(runtime.dataset.rigs[runtime.dataset.rows[0]["rig_id"]])
                    profile = build_standard_rig_calibration(
                        raw_rig, runtime.dataset.rows, runtime.args.raw,
                        mode="camera-only", device=runtime.device, poses=runtime.poses,
                    )
                    cfg_rig_path = Path(args.calibrated_rig) if getattr(args, "calibrated_rig", None) else None
                    out_path = cfg_rig_path if (cfg_rig_path and "camera_only" in cfg_rig_path.name) else Path("runs/calib_camera_only_profile.json")
                    write_json(out_path, profile)
                    for did in list(runtime.calibrated_rigs):
                        runtime.calibrated_rigs[did] = copy.deepcopy(profile["rig"])
                        runtime.surround[did].ground_plane = tuple(profile["ground_plane"])
                        runtime.surround[did].guidelines["camera_calib"] = copy.deepcopy(profile["camera_calib"])
                        runtime.surround[did].guidelines["ground_plane"] = list(profile["ground_plane"])
                    self._send(
                        HTTPStatus.OK,
                        json.dumps({
                            "ok": True,
                            "saved_path": str(out_path),
                            "ground_plane": profile["ground_plane"],
                            "camera_calib": profile["camera_calib"],
                            "quality": profile["quality"],
                        }, ensure_ascii=False).encode("utf-8"),
                        "application/json; charset=utf-8",
                    )
                    return

                cam_calibs = payload.get("camera_calib")
                gp = payload.get("ground_plane")
                if not isinstance(cam_calibs, list) or len(cam_calibs) != len(active_rig["cameras"]):
                    raise ValueError("Invalid camera_calib array")
                updated_rig = copy.deepcopy(active_rig)
                for cam, cc in zip(updated_rig["cameras"], cam_calibs, strict=True):
                    mat = np.asarray(cc["t_cam_from_ego"], dtype=np.float64).reshape(4, 4)
                    cam["T_cam_from_ego"] = mat.tolist()
                gp_list = [round(float(v), 6) for v in gp] if isinstance(gp, list) and len(gp) == 3 else list(active_gp)
                out_path = Path(args.calibrated_rig) if getattr(args, "calibrated_rig", None) else Path("runs/calib_manual_profile.json")
                profile = {
                    "schema_version": 1,
                    "calibration_mode": "manual_screen_tuned",
                    "rig_id": updated_rig.get("rig_id", "default-rig"),
                    "ground_plane": gp_list,
                    "road_plane_validation": {"source": "manual_on_screen"},
                    "vehicle_spec": export_vehicle_spec(updated_rig, ground_plane=gp_list),
                    "bowl_spec": export_bowl_spec(),
                    "rig": updated_rig,
                    "camera_calib": export_camera_calib(updated_rig),
                    "manual_tuning": payload.get("manual_state", {}),
                    "quality": {
                        "rot_deg": [0.0] * len(updated_rig["cameras"]),
                        "calib_dz_cm": updated_rig.get("calib_dz_cm", [0.0] * len(updated_rig["cameras"])),
                        "seam_parallax_cm": measure_seam_parallax_report(updated_rig, gp_list),
                    },
                }
                write_json(out_path, profile)
                for did in list(runtime.calibrated_rigs):
                    runtime.calibrated_rigs[did] = copy.deepcopy(updated_rig)
                    runtime.surround[did].ground_plane = tuple(gp_list)
                    runtime.surround[did].guidelines["camera_calib"] = copy.deepcopy(profile["camera_calib"])
                    runtime.surround[did].guidelines["ground_plane"] = list(gp_list)
                self._send(
                    HTTPStatus.OK,
                    json.dumps({"ok": True, "saved_path": str(out_path)}, ensure_ascii=False).encode("utf-8"),
                    "application/json; charset=utf-8",
                )
            except Exception as exc:
                self._send(
                    HTTPStatus.BAD_REQUEST,
                    json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False).encode("utf-8"),
                    "application/json; charset=utf-8",
                )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"OmniShield-X 360 MVP Demo ready on {url} | device={runtime.device_name} | frames={len(runtime.dataset)}", flush=True)
    if getattr(args, "open", False):
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

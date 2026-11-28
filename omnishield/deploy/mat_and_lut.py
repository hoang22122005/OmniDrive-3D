"""Printable real-scale garage calibration mats (SVG + PNG) and pre-baked fixed GPU LUT exporter.

1. ``generate_calibration_mats``:
   - Generates 4 exact-scale 1.20 m x 1.20 m checkerboard mats (FL, FR, RL, RR) with 30 cm squares,
     colored P1-P4 target rings, 10 cm calibration scale bars, a full-garage SVG placement blueprint,
     and ``mat_world_coords.json`` tailored to ANY vehicle's real length & width.
2. ``export_fixed_lut``:
   - Pre-bakes a calibrated ``RigCalibrationProfile`` JSON into fixed production Lookup Tables:
     * ``avm_2d_lut.npz`` + ``avm_2d_lut.bin`` (per-pixel uint16 UV + uint8 blend weights for 4 cams)
     * ``avm_3d_bowl_lut.npz`` + ``avm_3d_bowl_lut.bin`` (pre-computed 3D Stadium Bowl vertices + KB4 UVs + weights)
     * ``avm_lut_spec.h`` (ready-to-compile C / OpenGL ES 3.0 header for Allwinner T5 / RK3568 / Jetson / Android)
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from omnishield.eval.scene3d import (
    BOWL_FLAT_POWER,
    BOWL_FLAT_RX,
    BOWL_FLAT_RY,
    BOWL_FLAT_SPLIT,
    BOWL_PAD_X,
    BOWL_PAD_Y,
    BOWL_RINGS,
    BOWL_SEAM_X_SCALE,
    BOWL_SEAM_X_SHIFT,
    BOWL_WALL_HEIGHT,
    BOWL_WALL_POWER,
    N_BOWL_SECTORS,
    export_bowl_spec,
    export_camera_calib,
)
from omnishield.geometry.calibration import project


PNP_HEX = ("#ff4444", "#22d3ee", "#facc15", "#4ade80")


def compute_garage_mat_layout(
    vehicle_length_m: float = 4.750,
    vehicle_width_m: float = 1.934,
    x_center_m: float = -0.024,
    mat_size_m: float = 1.20,
    margin_x_m: float = 0.175,
    margin_y_m: float = 0.183,
) -> dict:
    """Compute exact 4-corner mat world coordinates (ego frame: +X front, +Y left, +Z up)."""
    half_l = 0.5 * float(vehicle_length_m)
    half_w = 0.5 * float(vehicle_width_m)
    in_x = round(half_l + float(margin_x_m), 3)
    out_x = round(in_x + float(mat_size_m), 3)
    in_y = round(half_w + float(margin_y_m), 3)
    out_y = round(in_y + float(mat_size_m), 3)

    xc = float(x_center_m)
    mats = {
        "FL": {"x0": round(xc + in_x, 3), "x1": round(xc + out_x, 3), "y0": in_y, "y1": out_y},
        "FR": {"x0": round(xc + in_x, 3), "x1": round(xc + out_x, 3), "y0": -out_y, "y1": -in_y},
        "RL": {"x0": round(xc - out_x, 3), "x1": round(xc - in_x, 3), "y0": in_y, "y1": out_y},
        "RR": {"x0": round(xc - out_x, 3), "x1": round(xc - in_x, 3), "y0": -out_y, "y1": -in_y},
    }
    # 4 corner points (P1..P4) per camera matching demo.html DEFAULT_MAT_WORLD convention
    pnp_world_by_cam = {
        "front": [
            [mats["FL"]["x1"], mats["FL"]["y1"], 0.0],
            [mats["FL"]["x0"], mats["FL"]["y0"], 0.0],
            [mats["FR"]["x0"], mats["FR"]["y1"], 0.0],
            [mats["FR"]["x1"], mats["FR"]["y0"], 0.0],
        ],
        "rear": [
            [mats["RL"]["x0"], mats["RL"]["y1"], 0.0],
            [mats["RL"]["x1"], mats["RL"]["y0"], 0.0],
            [mats["RR"]["x1"], mats["RR"]["y1"], 0.0],
            [mats["RR"]["x0"], mats["RR"]["y0"], 0.0],
        ],
        "left": [
            [mats["FL"]["x1"], mats["FL"]["y1"], 0.0],
            [mats["FL"]["x0"], mats["FL"]["y0"], 0.0],
            [mats["RL"]["x1"], mats["RL"]["y0"], 0.0],
            [mats["RL"]["x0"], mats["RL"]["y1"], 0.0],
        ],
        "right": [
            [mats["FR"]["x1"], mats["FR"]["y0"], 0.0],
            [mats["FR"]["x0"], mats["FR"]["y1"], 0.0],
            [mats["RR"]["x1"], mats["RR"]["y1"], 0.0],
            [mats["RR"]["x0"], mats["RR"]["y0"], 0.0],
        ],
    }
    return {
        "vehicle_length_m": vehicle_length_m,
        "vehicle_width_m": vehicle_width_m,
        "x_center_m": x_center_m,
        "mat_size_m": mat_size_m,
        "grid_cells": 4,
        "square_size_m": round(mat_size_m / 4.0, 4),
        "mats": mats,
        "pnp_world_by_cam": pnp_world_by_cam,
    }


def _build_mat_svg(label: str, mat_size_mm: int = 1200, cells: int = 4) -> str:
    sq = mat_size_mm / cells
    rects = []
    for ix in range(cells):
        for iy in range(cells):
            fill = "#111416" if (ix + iy) % 2 == 0 else "#ffffff"
            rects.append(
                f'  <rect x="{ix * sq:.1f}" y="{iy * sq:.1f}" width="{sq:.1f}" height="{sq:.1f}" fill="{fill}"/>'
            )
    # 4 corner target circles + crosshairs
    corners = [(0, 0, "C1"), (mat_size_mm, 0, "C2"), (mat_size_mm, mat_size_mm, "C3"), (0, mat_size_mm, "C4")]
    marks = []
    for idx, (cx, cy, _) in enumerate(corners):
        # Inset slightly so target ring prints cleanly inside the border
        ox = 18 if cx == 0 else mat_size_mm - 18
        oy = 18 if cy == 0 else mat_size_mm - 18
        marks.append(
            f'  <circle cx="{ox}" cy="{oy}" r="16" fill="none" stroke="{PNP_HEX[idx]}" stroke-width="5"/>'
        )
    mid = mat_size_mm * 0.5
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{mat_size_mm}mm" height="{mat_size_mm}mm" viewBox="0 0 {mat_size_mm} {mat_size_mm}">',
        f'  <rect x="0" y="0" width="{mat_size_mm}" height="{mat_size_mm}" fill="#ffffff"/>',
        *rects,
        f'  <rect x="3" y="3" width="{mat_size_mm - 6}" height="{mat_size_mm - 6}" fill="none" stroke="#f59e0b" stroke-width="6"/>',
        *marks,
        f'  <rect x="{mid - 110:.1f}" y="{mid - 55:.1f}" width="220" height="110" rx="14" fill="#0b1210" fill-opacity="0.9"/>',
        f'  <text x="{mid:.1f}" y="{mid - 6:.1f}" fill="#90f3be" font-family="Arial, sans-serif" font-size="62" font-weight="bold" text-anchor="middle">{label}</text>',
        f'  <text x="{mid:.1f}" y="{mid + 36:.1f}" fill="#e2e8f0" font-family="Arial, sans-serif" font-size="24" text-anchor="middle">{mat_size_mm/1000:.2f}m x {mat_size_mm/1000:.2f}m ({sq:.0f}mm/sq)</text>',
        '  <!-- 100mm verification ruler -->',
        f'  <rect x="40" y="{mat_size_mm - 34}" width="100" height="14" fill="#22d3ee"/>',
        f'  <text x="150" y="{mat_size_mm - 21}" fill="#22d3ee" font-family="Arial, sans-serif" font-size="22" font-weight="bold">100 mm SCALE CHECK</text>',
        "</svg>",
    ])


def _build_mat_png(label: str, px_size: int = 1200, cells: int = 4) -> Image.Image:
    im = Image.new("RGB", (px_size, px_size), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    sq = px_size // cells
    for ix in range(cells):
        for iy in range(cells):
            col = (17, 20, 22) if (ix + iy) % 2 == 0 else (255, 255, 255)
            draw.rectangle([ix * sq, iy * sq, (ix + 1) * sq, (iy + 1) * sq], fill=col)
    draw.rectangle([2, 2, px_size - 3, px_size - 3], outline=(245, 158, 11), width=6)
    mid = px_size // 2
    draw.rectangle([mid - 100, mid - 46, mid + 100, mid + 46], fill=(11, 18, 16), outline=(144, 243, 190), width=3)
    draw.text((mid - 26, mid - 12), f"{label} 1.2m", fill=(144, 243, 190))
    return im


def _build_garage_blueprint_svg(layout: dict) -> str:
    """Top-down garage layout diagram showing exact tape-measure distances from vehicle bumpers & mirrors."""
    L = layout["vehicle_length_m"]
    W = layout["vehicle_width_m"]
    xc = layout["x_center_m"]
    # SVG coords: 100 px = 1 meter, centered at (500, 500), +X ego UP, +Y ego LEFT
    sc = 100.0
    cx, cy = 420.0, 520.0

    def pt(x_ego, y_ego):
        return cx - y_ego * sc, cy - (x_ego - xc) * sc

    vx0, vy0 = pt(xc + 0.5 * L, 0.5 * W)
    vx1, vy1 = pt(xc - 0.5 * L, -0.5 * W)

    mat_elems = []
    for name, m in layout["mats"].items():
        x_a, y_a = pt(m["x1"], m["y1"])
        x_b, y_b = pt(m["x0"], m["y0"])
        rx, ry = min(x_a, x_b), min(y_a, y_b)
        rw, rh = abs(x_b - x_a), abs(y_b - y_a)
        mat_elems.append(f'  <rect x="{rx:.1f}" y="{ry:.1f}" width="{rw:.1f}" height="{rh:.1f}" fill="#173327" stroke="#90f3be" stroke-width="3"/>')
        mat_elems.append(
            f'  <text x="{rx + 0.5*rw:.1f}" y="{ry + 0.5*rh + 6:.1f}" fill="#ffffff" font-family="Arial" font-size="22" font-weight="bold" text-anchor="middle">{name}</text>'
        )
        mat_elems.append(
            f'  <text x="{rx + 0.5*rw:.1f}" y="{ry + 0.5*rh + 26:.1f}" fill="#90f3be" font-family="Arial" font-size="12" text-anchor="middle">X:[{m["x0"]:+.2f},{m["x1"]:+.2f}]</text>'
        )
        mat_elems.append(
            f'  <text x="{rx + 0.5*rw:.1f}" y="{ry + 0.5*rh + 42:.1f}" fill="#90f3be" font-family="Arial" font-size="12" text-anchor="middle">Y:[{m["y0"]:+.2f},{m["y1"]:+.2f}]</text>'
        )

    m_fl = layout["mats"]["FL"]
    dx_bumper = m_fl["x0"] - (xc + 0.5 * L)
    dy_body = m_fl["y0"] - 0.5 * W

    return "\n".join([
        '<svg xmlns="http://www.w3.org/2000/svg" width="840" height="1040" viewBox="0 0 840 1040">',
        '  <rect width="840" height="1040" fill="#0d1312"/>',
        '  <text x="420" y="38" fill="#ffffff" font-family="Arial" font-size="22" font-weight="bold" text-anchor="middle">SO DO TRAI 4 TAM BAT HIEU CHUAN GARA (OMNISHIELD-X)</text>',
        f'  <text x="420" y="64" fill="#90f3be" font-family="Arial" font-size="14" text-anchor="middle">Xe: Dai {L:.3f}m x Rong {W:.3f}m | Bat: {layout["mat_size_m"]:.2f}m x {layout["mat_size_m"]:.2f}m | Cach can: {dx_bumper*100:.1f}cm | Cach than: {dy_body*100:.1f}cm</text>',
        f'  <line x1="420" y1="80" x2="420" y2="980" stroke="#334155" stroke-dasharray="6,6"/>',
        f'  <line x1="60" y1="520" x2="780" y2="520" stroke="#334155" stroke-dasharray="6,6"/>',
        f'  <rect x="{min(vx0,vx1):.1f}" y="{min(vy0,vy1):.1f}" width="{abs(vx1-vx0):.1f}" height="{abs(vy1-vy0):.1f}" rx="22" fill="#1e293b" stroke="#38bdf8" stroke-width="3"/>',
        f'  <text x="420" y="{min(vy0,vy1) + 40:.1f}" fill="#38bdf8" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle">DAU XE (+X)</text>',
        f'  <text x="420" y="{max(vy0,vy1) - 22:.1f}" fill="#f87171" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle">DUOI XE (-X)</text>',
        *mat_elems,
        '  <text x="420" y="1015" fill="#cbd5e1" font-family="Arial" font-size="13" text-anchor="middle">In 4 file mat_FL/FR/RL/RR.svg ty le 1:1 (1200mm x 1200mm) tren bat Hiflex mo (khong bong), trai phang san gara.</text>',
        "</svg>",
    ])


def generate_calibration_mats(
    out_dir: Path,
    vehicle_length_m: float = 4.750,
    vehicle_width_m: float = 1.934,
    x_center_m: float = -0.024,
    mat_size_m: float = 1.20,
    png_px: int = 1200,
) -> dict:
    """Generate 4 exact-scale printable calibration mats (SVG + PNG) + garage layout blueprint + JSON coords."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layout = compute_garage_mat_layout(vehicle_length_m, vehicle_width_m, x_center_m, mat_size_m)
    mat_mm = int(round(mat_size_m * 1000))

    written = []
    for name in ("FL", "FR", "RL", "RR"):
        svg_path = out_dir / f"mat_{name}.svg"
        svg_path.write_text(_build_mat_svg(name, mat_size_mm=mat_mm, cells=4), encoding="utf-8")
        png_path = out_dir / f"mat_{name}.png"
        _build_mat_png(name, px_size=png_px, cells=4).save(png_path)
        written.extend([str(svg_path), str(png_path)])

    bp_path = out_dir / "garage_layout_blueprint.svg"
    bp_path.write_text(_build_garage_blueprint_svg(layout), encoding="utf-8")
    coords_path = out_dir / "mat_world_coords.json"
    coords_path.write_text(json.dumps(layout, indent=2), encoding="utf-8")
    written.extend([str(bp_path), str(coords_path)])
    return {"layout": layout, "files": written}


def _compute_4cam_seam_weights(pts_ego: np.ndarray, rig: dict, calibs: list[dict], wall_u: np.ndarray):
    """Exact NumPy mirror of commercial 4-corner Stadium Bowl KB4 projection + seam weights."""
    n = len(pts_ego)
    n_cams = len(rig["cameras"])
    uvs = np.zeros((n_cams, n, 2), dtype=np.float32)
    valid = np.zeros((n_cams, n), dtype=np.float32)
    scores = np.full((n_cams, n), -10.0, dtype=np.float32)

    wall_scale = np.full(n, 1.35, dtype=np.float64)
    dx_e = (pts_ego[:, 0] + 0.024) / wall_scale
    dy_e = pts_ego[:, 1]
    d_norm = np.hypot(dx_e, dy_e) + 1e-6
    dir_ego = np.column_stack([dx_e / d_norm, dy_e / d_norm])

    for ci, (cam, cal) in enumerate(zip(rig["cameras"], calibs)):
        h, w = int(cam["image_hw"][0]), int(cam["image_hw"][1])
        uv, _, v_ok = project(pts_ego, cam, (h, w))
        c_xy = np.asarray(cal["cam_xy"], dtype=np.float64)
        c_fwd = np.asarray(cal["cam_fwd"], dtype=np.float64)
        c_fwd = c_fwd / max(float(np.linalg.norm(c_fwd)), 1e-6)
        delta = pts_ego[:, :2] - c_xy[None, :]
        cos_dir = (delta @ c_fwd) / np.maximum(np.linalg.norm(delta, axis=1), 1e-4)
        v_ok = v_ok & (cos_dir > -0.18)

        # Normalized [0, 1] UV (OpenGL bottom-left V=0 or top-left V=0: store standard image top-left UV in [0,1])
        u_norm = np.clip((uv[:, 0] + 0.5) / w, 0.0, 1.0)
        v_norm = np.clip((uv[:, 1] + 0.5) / h, 0.0, 1.0)
        uvs[ci, :, 0] = np.where(v_ok, u_norm, 0.0)
        uvs[ci, :, 1] = np.where(v_ok, v_norm, 0.0)
        valid[ci] = v_ok.astype(np.float32)

        # Soft border fade
        bx = np.minimum(uv[:, 0] + 0.5, w - 0.5 - uv[:, 0])
        by = np.minimum(uv[:, 1] + 0.5, h - 0.5 - uv[:, 1])
        border = np.clip(np.minimum(bx, by) / 32.0, 0.0, 1.0)
        axis = c_fwd
        c_ang = dir_ego @ axis
        sc = c_ang + 0.20 * np.log(np.maximum(border, 1e-4))
        scores[ci] = np.where(v_ok, sc, -10.0)

    best = np.max(scores, axis=0, keepdims=True)
    sharp = np.where(wall_u > 0.05, 36.0, 22.0)[None, :]
    w_raw = np.exp(np.clip(sharp * (scores - best), -40.0, 0.0)) * valid
    w_sum = np.maximum(np.sum(w_raw, axis=0, keepdims=True), 1e-5)
    weights = (w_raw / w_sum).astype(np.float32)
    return uvs, weights, valid


def _write_c_header(out_path: Path, lut_2d_hw: tuple[int, int], bowl_verts: int, bowl_indices: int, profile: dict):
    h2, w2 = lut_2d_hw
    gp = profile.get("ground_plane", [0.0, 0.0, 0.023])
    bs = profile.get("bowl_spec", export_bowl_spec())
    header = f"""/* Auto-generated by OmniShield-X Fixed LUT Exporter (Commercial 4-Camera Embedded AVM)
 * Target: OpenGL ES 3.0 / Vulkan / Allwinner T5 / RK3568 / Jetson Orin Nano / Android NDK
 * Calibration mode: {profile.get('calibration_mode', 'unknown')}
 * Ground plane: z = {gp[0]:.6f}*x + {gp[1]:.6f}*y + {gp[2]:.6f}
 */
#ifndef OMNISHIELD_AVM_LUT_SPEC_H
#define OMNISHIELD_AVM_LUT_SPEC_H

#include <stdint.h>

#define AVM_CAM_COUNT        4
#define AVM_2D_LUT_WIDTH     {w2}
#define AVM_2D_LUT_HEIGHT    {h2}
#define AVM_3D_BOWL_VERTS    {bowl_verts}
#define AVM_3D_BOWL_INDICES  {bowl_indices}
#define AVM_BOWL_FLAT_RX     {bs['flat_rx']:.3f}f
#define AVM_BOWL_FLAT_RY     {bs['flat_ry']:.3f}f
#define AVM_BOWL_PAD_X       {bs['pad_x']:.3f}f
#define AVM_BOWL_PAD_Y       {bs['pad_y']:.3f}f
#define AVM_BOWL_WALL_H      {bs['wall_height']:.3f}f
#define AVM_BOWL_WALL_POW    {bs['wall_power']:.3f}f

/* Binary file layout for avm_2d_lut.bin:
 *   Header (16 bytes): uint32 magic ('AVM2'), uint32 width, uint32 height, uint32 n_cams
 *   Body (width * height records of Avm2dPixel):
 */
typedef struct __attribute__((packed)) {{
    uint16_t u[AVM_CAM_COUNT];      /* Normalized U in [0..65535] (top-left origin) */
    uint16_t v[AVM_CAM_COUNT];      /* Normalized V in [0..65535] (top-left origin) */
    uint8_t  weight[AVM_CAM_COUNT]; /* Blend weights summing to 255                 */
}} Avm2dPixel;

/* Binary file layout for avm_3d_bowl_lut.bin:
 *   Header (16 bytes): uint32 magic ('AVM3'), uint32 n_verts, uint32 n_indices, uint32 n_cams
 *   Vertices: n_verts * Avm3dVertex
 *   Indices : n_indices * uint32
 */
typedef struct __attribute__((packed)) {{
    float    xyz[3];                /* 3D bowl position in OpenGL world coords (-y, z, -x) */
    uint16_t u[AVM_CAM_COUNT];      /* Pre-projected KB4 U in [0..65535]                   */
    uint16_t v[AVM_CAM_COUNT];      /* Pre-projected KB4 V in [0..65535]                   */
    uint8_t  weight[AVM_CAM_COUNT]; /* Pre-baked 4-corner seam blend weights [0..255]      */
}} Avm3dVertex;

#endif /* OMNISHIELD_AVM_LUT_SPEC_H */
"""
    out_path.write_text(header, encoding="utf-8")


def export_fixed_lut(
    profile: dict,
    out_dir: Path,
    top2d_hw: tuple[int, int] = (512, 384),
    top2d_bounds_xy: tuple[float, float, float, float] = (-6.5, 6.5, -4.8, 4.8),
    sectors: int = N_BOWL_SECTORS,
    rings: int = BOWL_RINGS,
) -> dict:
    """Pre-bake fixed 2D Top-Down & 3D Stadium Bowl Lookup Tables for embedded 60 FPS GPU execution."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rig = profile["rig"]
    calibs = profile.get("camera_calib") or export_camera_calib(rig)
    gp = [float(v) for v in profile.get("ground_plane", [0.0, 0.0, 0.023])]
    bs = profile.get("bowl_spec") or export_bowl_spec()

    # 1. Bake 2D Top-Down 360 LUT (H x W, +X up, +Y left)
    h2, w2 = int(top2d_hw[0]), int(top2d_hw[1])
    xmin, xmax, ymin, ymax = top2d_bounds_xy
    xs = np.linspace(xmax, xmin, h2, dtype=np.float64)
    ys = np.linspace(ymax, ymin, w2, dtype=np.float64)
    gy, gx = np.meshgrid(ys, xs)
    gz = gp[0] * gx + gp[1] * gy + gp[2]
    pts_2d = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    uv_2d, w_2d, _ = _compute_4cam_seam_weights(pts_2d, rig, calibs, np.zeros(len(pts_2d), dtype=np.float64))

    u16_2d = np.clip(np.round(uv_2d[:, :, 0].T * 65535.0), 0, 65535).astype(np.uint16)  # (N, 4)
    v16_2d = np.clip(np.round(uv_2d[:, :, 1].T * 65535.0), 0, 65535).astype(np.uint16)
    w8_2d = np.clip(np.round(w_2d.T * 255.0), 0, 255).astype(np.uint8)

    npz_2d = out_dir / "avm_2d_lut.npz"
    np.savez_compressed(
        npz_2d,
        u16=u16_2d.reshape(h2, w2, 4),
        v16=v16_2d.reshape(h2, w2, 4),
        weight8=w8_2d.reshape(h2, w2, 4),
        bounds_xy=np.asarray(top2d_bounds_xy, dtype=np.float32),
        ground_plane=np.asarray(gp, dtype=np.float32),
    )

    bin_2d = out_dir / "avm_2d_lut.bin"
    dt_2d = np.dtype([("u", "<u2", (4,)), ("v", "<u2", (4,)), ("w", "u1", (4,))])
    rec_2d = np.empty(h2 * w2, dtype=dt_2d)
    rec_2d["u"] = u16_2d
    rec_2d["v"] = v16_2d
    rec_2d["w"] = w8_2d
    with bin_2d.open("wb") as f:
        f.write(struct.pack("<4sIII", b"AVM2", w2, h2, 4))
        f.write(rec_2d.tobytes())

    # 2. Bake 3D Commercial Stadium Bowl LUT (vertices + pre-projected KB4 UVs + seam weights + indices)
    rx, ry = float(bs["flat_rx"]), float(bs["flat_ry"])
    px, py = float(bs["pad_x"]), float(bs["pad_y"])
    wh, wp = float(bs["wall_height"]), float(bs["wall_power"])
    rp = float(bs.get("rad_power", 2.45))
    fsplit = float(bs["flat_split"])
    p_exp = float(bs.get("flat_power", BOWL_FLAT_POWER))

    t_vals = np.linspace(0.0, 1.0, rings, dtype=np.float64)
    a_vals = np.linspace(0.0, 2.0 * math.pi, sectors, endpoint=False, dtype=np.float64)
    tt, aa = np.meshgrid(t_vals, a_vals, indexing="ij")  # (rings, sectors)
    ca, sa = np.cos(aa), np.sin(aa)
    r_flat = (np.abs(ca / rx) ** p_exp + np.abs(sa / ry) ** p_exp) ** (-1.0 / p_exp)
    r_pad = 1.0 / np.hypot(ca / px, sa / py)
    wall_u = np.where(tt <= fsplit, 0.0, (tt - fsplit) / max(1.0 - fsplit, 1e-5))
    phi = wall_u * (0.5 * math.pi)
    rad = np.where(tt <= fsplit, 0.05 + (r_flat - 0.05) * (tt / max(fsplit, 1e-5)), r_flat + r_pad * np.sin(phi))
    ex = rad * ca
    ey = rad * sa
    ez = gp[0] * ex + gp[1] * ey + gp[2] + np.where(wall_u > 0.0, wh * (1.0 - np.cos(phi)), 0.0)
    pts_3d = np.stack([ex, ey, ez], axis=-1).reshape(-1, 3)
    wu_flat = wall_u.reshape(-1)
    uv_3d, w_3d, _ = _compute_4cam_seam_weights(pts_3d, rig, calibs, wu_flat)

    # Triangle indices
    idx_list = []
    for ri in range(rings - 1):
        r0 = ri * sectors
        r1 = (ri + 1) * sectors
        for si in range(sectors):
            sn = (si + 1) % sectors
            idx_list.extend([r0 + si, r0 + sn, r1 + sn, r0 + si, r1 + sn, r1 + si])
    indices = np.asarray(idx_list, dtype=np.uint32)

    # OpenGL world position (-y_ego, z_ego, -x_ego)
    gl_xyz = np.column_stack([-pts_3d[:, 1], pts_3d[:, 2], -pts_3d[:, 0]]).astype(np.float32)
    u16_3d = np.clip(np.round(uv_3d[:, :, 0].T * 65535.0), 0, 65535).astype(np.uint16)
    v16_3d = np.clip(np.round(uv_3d[:, :, 1].T * 65535.0), 0, 65535).astype(np.uint16)
    w8_3d = np.clip(np.round(w_3d.T * 255.0), 0, 255).astype(np.uint8)

    npz_3d = out_dir / "avm_3d_bowl_lut.npz"
    np.savez_compressed(
        npz_3d,
        gl_xyz=gl_xyz,
        ego_xyz=pts_3d.astype(np.float32),
        u16=u16_3d,
        v16=v16_3d,
        weight8=w8_3d,
        indices=indices,
    )

    bin_3d = out_dir / "avm_3d_bowl_lut.bin"
    dt_3d = np.dtype([("xyz", "<f4", (3,)), ("u", "<u2", (4,)), ("v", "<u2", (4,)), ("w", "u1", (4,))])
    rec_3d = np.empty(len(pts_3d), dtype=dt_3d)
    rec_3d["xyz"] = gl_xyz
    rec_3d["u"] = u16_3d
    rec_3d["v"] = v16_3d
    rec_3d["w"] = w8_3d
    with bin_3d.open("wb") as f:
        f.write(struct.pack("<4sIII", b"AVM3", len(pts_3d), len(indices), 4))
        f.write(rec_3d.tobytes())
        f.write(indices.tobytes())

    hdr_path = out_dir / "avm_lut_spec.h"
    _write_c_header(hdr_path, (h2, w2), len(pts_3d), len(indices), profile)

    summary = {
        "out_dir": str(out_dir),
        "lut_2d_hw": [h2, w2],
        "lut_2d_bin_bytes": bin_2d.stat().st_size,
        "lut_3d_verts": int(len(pts_3d)),
        "lut_3d_indices": int(len(indices)),
        "lut_3d_bin_bytes": bin_3d.stat().st_size,
        "files": [str(npz_2d), str(bin_2d), str(npz_3d), str(bin_3d), str(hdr_path)],
    }
    (out_dir / "lut_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_make_mats_cli(args):
    res = generate_calibration_mats(
        args.out,
        vehicle_length_m=float(args.length),
        vehicle_width_m=float(args.width),
        x_center_m=float(args.x_center),
        mat_size_m=float(args.mat_size),
        png_px=int(args.png_px),
    )
    print(f"[Mats] Generated 4 printable mats + garage blueprint in {args.out}:")
    for p in res["files"]:
        print(f"  - {p}")
    return res


def run_export_lut_cli(args):
    profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    res = export_fixed_lut(
        profile,
        args.out,
        top2d_hw=(int(args.height_2d), int(args.width_2d)),
    )
    print(
        f"[LUT] Exported fixed production LUTs -> {args.out}\n"
        f"  2D Top-Down ({res['lut_2d_hw'][1]}x{res['lut_2d_hw'][0]}): {res['lut_2d_bin_bytes'] / 1024:.1f} KB\n"
        f"  3D Bowl ({res['lut_3d_verts']} verts, {res['lut_3d_indices']} idx): {res['lut_3d_bin_bytes'] / 1024:.1f} KB\n"
        f"  Header: {Path(args.out) / 'avm_lut_spec.h'}"
    )
    return res

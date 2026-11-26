from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from omnishield.data.dataset import load_targets
from omnishield.data.manifest import load_manifest, sample_path
from omnishield.geometry.calibration import Grid, ego_mask, letterbox_rgb, project
from omnishield.io import atomic_file, child_path, load_json


UNKNOWN = (36, 42, 47)
FREE = (57, 109, 92)
HIT = (213, 146, 93)
COLORS = {
    "vehicle": (103, 159, 208), "rider": (250, 168, 76),
    "pedestrian": (240, 116, 155), "static": HIT, "curb": (197, 179, 114),
}


def rgb_canvas(raw, row, rig, lidar_overlay):
    canvas = Image.new("RGB", (800, 484), (17, 26, 32))
    draw = ImageDraw.Draw(canvas)
    points = None
    if lidar_overlay:
        from omnishield.data.phenikaa import load_lidar

        if not row.get("lidar"):
            raise ValueError("LiDAR overlay requires a raw PHENIKAA sample")
        points, _ = load_lidar(raw, row, rig)
        points = points[np.isfinite(points).all(axis=1) & ~ego_mask(points, rig)]
    camera_hw = (180, 240)
    for i, (name, path) in enumerate(zip(("FRONT", "REAR", "LEFT", "RIGHT"), row["images"], strict=True)):
        x, y = 12 + 252 * (i % 2), 36 + 212 * (i // 2)
        draw.text((x, y), name, fill="white")
        if path:
            with Image.open(child_path(raw, path)) as source:
                image = letterbox_rgb(source, rig["cameras"][i]["image_hw"], camera_hw)
            if points is not None:
                uv, depth, valid = project(points, rig["cameras"][i], camera_hw)
                valid &= (depth > 0.4) & (depth < 20)
                u, v = np.floor(uv[valid] + 0.5).astype(np.int64).T
                nearest = np.full(camera_hw, np.inf, dtype=np.float32)
                np.minimum.at(nearest, (v, u), depth[valid])
                painted = np.isfinite(nearest)
                alpha = np.clip(nearest[painted] / 20, 0, 1)[:, None]
                colors = (1 - alpha) * [255, 205, 74] + alpha * [46, 167, 236]
                image[painted] = 0.45 * image[painted] + 0.55 * colors
            canvas.paste(Image.fromarray(image), (x, y + 18))
        else:
            draw.text((x + 12, y + 66), "NO CAMERA", fill=(235, 153, 99))
    overlay = " / LIDAR 0–20 m" if lidar_overlay else ""
    draw.text((12, 10), f"PHENIKAA  /  {row['drive_id']}  /  {row['frame_id']}{overlay}", fill="white")
    return canvas


def bev_panel(occupancy, semantics, classes, grid):
    occupied = occupancy == 1
    visible = (occupancy >= 0).any(0)
    touched = occupied.any(0)
    shape = occupancy.shape[1:]
    color = np.full((*shape, 3), UNKNOWN, np.uint8)
    color[visible] = FREE
    color[touched] = HIT
    if semantics is not None:
        for label, name in enumerate(classes[1:], 1):
            color[((semantics == label) & occupied).any(0)] = COLORS.get(name, HIT)
    # X points up and Y positive points left in the overhead view.
    raster = Image.fromarray(color[::-1, ::-1].transpose(1, 0, 2))
    panel = raster.resize((248, 296), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(panel)
    sx = (grid.y[-1] / (grid.y[-1] - grid.y[0])) * panel.width
    sy = (grid.x[-1] / (grid.x[-1] - grid.x[0])) * panel.height
    draw.line((sx, sy + 7, sx, sy - 8), fill="white", width=1)
    draw.polygon([(sx, sy - 10), (sx - 3, sy - 4), (sx + 3, sy - 4)], fill="white")
    return panel


def render(args):
    config = load_json(args.model)
    if (config["camera_count"] != 4 or args.frames < 1 or args.stride < 1 or args.start < 0
            or not 0 < args.threshold < 1 or not 0 < args.max_uncertainty <= 1):
        raise ValueError("Replay expects four camera slots and a valid window")
    if args.lidar_overlay and args.pred:
        raise ValueError("Use LiDAR overlay on the data review only")
    manifest = load_manifest(args.prepared / "manifest.json", config)
    rows = [f for f in manifest["frames"] if f["split"] == args.split]
    if args.drive:
        rows = [row for row in rows if row["drive_id"] == args.drive]
    rows = rows[args.start:args.start + args.frames * args.stride:args.stride]
    if not rows:
        raise ValueError("No frames in selected replay")
    grid = Grid.from_config(config["grid"])
    images = []
    for row in rows:
        canvas = rgb_canvas(args.raw, row, manifest["rigs"][row["rig_id"]], args.lidar_overlay)
        if args.pred is None:
            target = load_targets(sample_path(args.prepared, row), config)
            occupancy, semantics = target["occupancy"], target["semantics"]
            title = "LIDAR + CAMERA PROXY LABELS"
        else:
            file = child_path(args.pred, Path(row["drive_id"]) / (row["frame_id"] + ".npz"))
            with np.load(file, allow_pickle=False) as prediction:
                probability = prediction["p_occ"]
                known = prediction["valid"] & (prediction["mass"][2] <= args.max_uncertainty)
                occupancy = np.where(known, probability > args.threshold, -1).astype(np.int8)
                semantics = prediction["semantic_prob"].argmax(0) if "semantic_prob" in prediction else None
            if occupancy.shape != grid.shape:
                raise ValueError("Prediction grid differs from replay grid")
            title = "OMNISHIELD / RGB PREDICTION"
        draw = ImageDraw.Draw(canvas)
        draw.text((532, 36), title, fill="white")
        canvas.paste(bev_panel(occupancy, semantics, config["classes"], grid), (532, 58))
        for i, (text, color) in enumerate((
            ("Unknown", UNKNOWN), ("Free at one height", FREE), ("Occupied", HIT),
            ("Vehicle", COLORS["vehicle"]), ("Rider", COLORS["rider"]), ("Pedestrian", COLORS["pedestrian"]),
        )):
            x, y = 532 + (i % 2) * 120, 374 + (i // 2) * 19
            draw.rectangle((x, y, x + 10, y + 10), fill=color)
            draw.text((x + 16, y - 2), text, fill="white")
        images.append(canvas)
    if args.out.suffix.lower() == ".gif":
        times = np.asarray([r["timestamp_ns"] for r in rows])
        duration = np.clip(np.diff(times) / 1e6, 70, 1000).astype(int).tolist() + [150]
        atomic_file(args.out, lambda f: images[0].save(f, format="GIF", append_images=images[1:],
                                                       save_all=True, duration=duration, loop=0))
    elif args.out.suffix.lower() == ".png" and len(images) == 1:
        atomic_file(args.out, lambda f: images[0].save(f, format="PNG"))
    else:
        raise ValueError("Use .gif or --frames 1 with a .png output")

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from omnishield.data.dataset import load_targets
from omnishield.data.manifest import load_lut, lut_path, sample_path
from omnishield.geometry.calibration import Grid, build_lut, bundle_key
from omnishield.io import child_path, file_sha256, fingerprint, write_json, write_npz


def save_luts(prepared, manifest, config):
    for name, rig in manifest["rigs"].items():
        write_npz(lut_path(prepared, name), **build_lut(config, rig))


def save_one_label(task):
    from omnishield.data.labels import prepare_phenikaa

    raw, path, row, config, rig, revision, overwrite = task
    key = fingerprint({"label_version": 2, "lut": bundle_key(config, rig), "revision": revision, "frame": row})
    if path.exists() and not overwrite:
        with np.load(path, allow_pickle=False) as old:
            if "label_key" not in old or old["label_key"].item() != key:
                raise ValueError(f"Existing labels differ from calibration, source or method: {path}; use --overwrite")
        return False
    write_npz(path, label_key=np.asarray(key), **prepare_phenikaa(raw, row, config, rig))
    return True


def save_labels(raw, prepared, manifest, config, overwrite=False, workers=4):
    if manifest["source"] != "phenikaa":
        raise ValueError("Automatic labels use PHENIKAA LiDAR. Other sources must provide normalized targets.")
    if not 1 <= workers <= 8:
        raise ValueError("Choose 1–8 label workers")
    tasks = ((raw, sample_path(prepared, row), row, config, manifest["rigs"][row["rig_id"]],
              manifest["revision"], overwrite) for row in manifest["frames"])
    total = len(manifest["frames"])

    def report(results):
        saved = 0
        for done, changed in enumerate(results, 1):
            saved += changed
            if done % 25 == 0 or done == total:
                print(f"Labels {done}/{total}, wrote {saved}", flush=True)

    if workers == 1:
        report(map(save_one_label, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            report(executor.map(save_one_label, tasks))


def audit(raw, prepared, manifest, config, write=True):
    prepared = Path(prepared).resolve()
    coverage = {split: {} for split in ("train", "val", "test")}
    class_support = {split: [0, 0] for split in coverage}
    semantic_support = {split: [0] * len(config["classes"]) for split in coverage}
    frame_count = dict.fromkeys(coverage, 0)
    stamps = {}
    luts = {}
    for name, rig in manifest["rigs"].items():
        path = lut_path(prepared, name)
        luts[name] = load_lut(path, config, rig)
        stamps[path.relative_to(prepared).as_posix()] = file_sha256(path)
    for row in manifest["frames"]:
        path = sample_path(prepared, row)
        rig = manifest["rigs"][row["rig_id"]]
        calibrated = luts[row["rig_id"]]["valid"]
        sensors = np.array([image is not None for image in row["images"]], dtype=bool)
        for camera, image in zip(rig["cameras"], row["images"], strict=True):
            if image is None:
                continue
            image_path = child_path(raw, image)
            with Image.open(image_path) as handle:
                if [handle.height, handle.width] != camera["image_hw"]:
                    raise ValueError(f"Calibration image size mismatch: {image_path}")
        masks = calibrated[sensors].any(axis=0)
        labels = load_targets(path, config)
        depth = labels["depth"]
        occupancy = labels["occupancy"]
        masks_by_task = {
            "occupancy": (occupancy >= 0) & masks,
            "depth": (depth >= 0) & sensors[:, None, None],
            "semantics": (labels["semantics"] > 0) & (occupancy == 1) & masks,
            "health": np.isfinite(labels["health"]) & sensors,
            "velocity": np.isfinite(labels["velocity"]) & masks[None],
            "risk": np.isfinite(labels["risk"]),
            "center": np.isfinite(labels["center"]) & masks,
            "offset": np.isfinite(labels["offset"]) & masks[None],
        }
        for task, support in masks_by_task.items():
            split = coverage[row["split"]]
            split[task] = split.get(task, 0) + int(support.sum())
        for label in (0, 1):
            class_support[row["split"]][label] += int(((occupancy == label) & masks).sum())
        for label in range(len(config["classes"])):
            semantic_support[row["split"]][label] += int(((labels["semantics"] == label) & masks).sum())
        frame_count[row["split"]] += 1
        stamps[path.relative_to(prepared).as_posix()] = file_sha256(path)
    report = {
        "manifest_sha256": fingerprint(manifest),
        "model_sha256": fingerprint(config),
        "prepared_sha256": fingerprint(stamps),
        "coverage": coverage,
        "occupancy_classes": class_support,
        "frames": frame_count,
        "semantic_classes": {
            split: dict(zip(config["classes"], row, strict=True)) for split, row in semantic_support.items()
        },
        "free_label_source_by_rig": {
            name: rig.get("free_label_mode", "fixture" if manifest["source"] == "synthetic" else "normalized")
            for name, rig in manifest["rigs"].items()
        },
    }
    if write:
        write_json(Path(prepared) / "audit.json", report)
    return report


def synthetic(root, prepared, config):
    if config["backbone"] != "small_cnn" or config["camera_count"] != 4:
        raise ValueError("Use configs/model/smoke.json for the synthetic fixture")
    h, w = config["image_hw"]
    cameras = []
    for slot, yaw in zip(("front", "rear", "left", "right"), (0, np.pi, np.pi/2, -np.pi/2)):
        forward = np.asarray([np.cos(yaw), np.sin(yaw), 0])
        rotation = np.array([[np.sin(yaw), -np.cos(yaw), 0], [0, 0, -1], forward.tolist()])
        origin = forward * 0.8 + [0, 0, 0.9]
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = rotation, -rotation @ origin
        cameras.append({
            "slot": slot, "model": "pinhole", "image_hw": [h, w], "max_theta_deg": 76,
            "K": [[w / 3, 0, (w - 1) / 2], [0, h / 2, (h - 1) / 2], [0, 0, 1]],
            "T_cam_from_ego": matrix.tolist(),
        })
    rig = {"cameras": cameras, "ego_xy_bounds_m": [-0.7, 0.7, -0.5, 0.5], "ground_height_m": 0}
    manifest = {
        "version": 1, "source": "synthetic", "revision": "smoke-v1", "rigs": {"fixture": rig}, "frames": [],
    }
    grid = Grid.from_config(config["grid"])
    x = grid.centers()[..., 0]
    y = grid.centers()[..., 1]
    z = grid.centers()[..., 2]
    for split, count in (("train", 6), ("val", 3), ("test", 2)):
        for i in range(count):
            images = []
            for camera in cameras:
                relative = f"{split}/{i:04}_{camera['slot']}.png"
                path = child_path(root, relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                image = np.zeros((h, w, 3), dtype=np.uint8) + 32
                image[h//2:, :] = [81, 90, 95]
                start = w//4 + i
                image[h//3:h*3//4, start:start+w//6] = [215, 140, 52]
                Image.fromarray(image).save(path)
                images.append(relative)
            row = {"drive_id": split, "frame_id": f"{i:04}", "split": split, "timestamp_ns": i*100_000_000,
                   "rig_id": "fixture", "images": images}
            manifest["frames"].append(row)
            occupied = (abs(x - 2.6) < 0.5) & (abs(y - 0.06*i) < 0.5) & (z < 1.65)
            observed = (np.hypot(x, y) > 1) & (np.hypot(x, y) < 4.1)
            occupancy = np.where(observed, occupied.astype(np.int8), -1).astype(np.int8)
            semantics = np.where(occupancy == 1, 3, occupancy).astype(np.int16)
            depth = np.full((4, h//4, w//4), -1, dtype=np.int16)
            depth[:, h//8:h//8+2, w//8:w//8+2] = 4
            write_npz(sample_path(prepared, row), occupancy=occupancy, semantics=semantics, depth=depth)
    save_luts(prepared, manifest, config)
    write_json(Path(prepared) / "manifest.json", manifest)
    audit(Path(root), Path(prepared), manifest, config)

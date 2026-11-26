from pathlib import Path

import numpy as np

from omnishield.geometry.calibration import Grid, bundle_key, depth_edges, model_contract
from omnishield.io import child_path, load_json


def load_manifest(path, config, allow_synthetic=False):
    manifest = load_json(path)
    if manifest.get("version") != 1 or manifest.get("source") not in {"phenikaa", "normalized", "synthetic"}:
        raise ValueError("Expected manifest v1 with a supported source")
    if not manifest.get("revision"):
        raise ValueError("Dataset revision is required")
    if manifest["source"] == "synthetic" and not allow_synthetic:
        raise ValueError("Synthetic data is available only for explicit smoke tests")
    model_contract(config)
    if len(manifest["rigs"]) == 0:
        raise ValueError("Manifest has no rig")
    for name, rig in manifest["rigs"].items():
        cameras = rig["cameras"]
        slots = [camera["slot"] for camera in cameras]
        if len(cameras) != config["camera_count"] or len(set(slots)) != len(slots):
            raise ValueError(f"{name}: camera count or unique slot names do not match the profile")
        if config["camera_count"] == 4 and slots != ["front", "rear", "left", "right"]:
            raise ValueError(f"{name}: expected front, rear, left, right")
        (x0, x1, y0, y1) = rig["ego_xy_bounds_m"]
        if not np.isfinite([x0, x1, y0, y1, rig["ground_height_m"]]).all() or not (x0 < x1 and y0 < y1):
            raise ValueError(f"{name}: invalid ego footprint or ground height")
    drives, seen, last = {}, set(), {}
    counts = dict(train=0, val=0, test=0)
    for row in manifest["frames"]:
        split, drive = row["split"], row["drive_id"]
        key = (drive, row["frame_id"])
        if split not in counts or key in seen or row["rig_id"] not in manifest["rigs"]:
            raise ValueError(f"Invalid split, duplicate frame or unknown rig: {key}")
        if drives.setdefault(drive, split) != split:
            raise ValueError(f"Drive {drive} occurs in multiple splits")
        stamp = row["timestamp_ns"]
        if not isinstance(stamp, int) or stamp <= last.get(drive, -1):
            raise ValueError(f"{key}: timestamps must be integer ns in increasing drive order")
        if len(row["images"]) != config["camera_count"]:
            raise ValueError(f"{key}: image count does not match configured rig")
        if not any(row["images"]):
            raise ValueError(f"{key}: at least one camera image must have a path")
        seen.add(key)
        last[drive] = stamp
        counts[split] += 1
    if not counts["train"] or not counts["val"]:
        raise ValueError("Independent nonempty train and val splits are required")
    return manifest


def load_lut(path, config, rig):
    with np.load(path, allow_pickle=False) as arrays:
        data = {name: arrays[name] for name in ("uv", "bins", "valid", "key")}
    shape = (config["camera_count"], *Grid.from_config(config["grid"]).shape)
    if data["key"].item() != bundle_key(config, rig):
        raise ValueError(f"LUT does not match the model grid or rig: {path}")
    if data["uv"].shape != (*shape, 2) or any(data[k].shape != shape for k in ("bins", "valid")):
        raise ValueError(f"Invalid LUT tensor shape: {path}")
    if (
        data["bins"].dtype != np.int64 or np.any((data["bins"] < 0) | (data["bins"] >= len(depth_edges(config["depth"])) - 1))
        or data["valid"].dtype != bool or not np.isfinite(data["uv"]).all()
    ):
        raise ValueError(f"Invalid LUT dtype or coordinates: {path}")
    return data


def sample_path(prepared, row):
    return child_path(prepared, Path("labels") / row["drive_id"] / (row["frame_id"] + ".npz"))


def lut_path(prepared, rig_id):
    return child_path(prepared, Path("lut") / (rig_id + ".npz"))

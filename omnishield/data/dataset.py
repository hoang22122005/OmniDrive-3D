from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from omnishield.data.manifest import load_lut, lut_path, sample_path
from omnishield.geometry.calibration import Grid, depth_edges, letterbox_rgb
from omnishield.io import child_path


REQUIRED = ("occupancy",)
OPTIONAL = ("semantics", "depth", "health", "velocity", "risk", "center", "offset")


def target_layout(config):
    shape = Grid.from_config(config["grid"]).shape
    h, w = config["image_hw"]
    c, s = config["camera_count"], config["feature_stride"]
    return {
        "occupancy": (shape, 1),
        "semantics": (shape, len(config["classes"]) - 1),
        "depth": ((c, h // s, w // s), len(depth_edges(config["depth"])) - 1),
        "health": ((c,), None),
        "velocity": ((2, *shape), None),
        "risk": (shape, None),
        "center": (shape, None),
        "offset": ((2, *shape), None),
    }


def load_targets(path, config):
    layout = target_layout(config)
    targets = {}
    with np.load(path, allow_pickle=False) as data:
        for name, (shape, max_index) in layout.items():
            if name not in data:
                if name in REQUIRED:
                    raise ValueError(f"Missing {name}: {path}")
                targets[name] = np.full(shape, -1 if max_index is not None else np.nan,
                                        dtype=np.int64 if max_index is not None else np.float32)
                continue
            value = data[name]
            if value.shape != shape:
                raise ValueError(f"{path}: {name} expected {shape}, got {value.shape}")
            if max_index is not None:
                if value.dtype.kind not in "iu" or np.any((value < -1) | (value > max_index)):
                    raise ValueError(f"{path}: invalid class or missing-mask value for {name}")
                targets[name] = value.astype(np.int64)
            else:
                value = value.astype(np.float32)
                finite = np.isfinite(value)
                if np.isinf(value).any() or (name in {"health", "risk", "center"} and
                   np.any((value[finite] < 0) | (value[finite] > 1))):
                    raise ValueError(f"{path}: invalid {name} targets")
                targets[name] = value
    return targets


class OccupancyDataset:
    def __init__(self, manifest, config, raw, prepared, split):
        if split == "all":
            self.rows = list(manifest["frames"])
        else:
            self.rows = [row for row in manifest["frames"] if row["split"] == split]
        self.rigs = manifest["rigs"]
        self.config = config
        self.raw, self.prepared = Path(raw), Path(prepared)

    def __len__(self):
        return len(self.rows)

    @lru_cache(maxsize=8)
    def lut(self, rig_id):
        return load_lut(lut_path(self.prepared, rig_id), self.config, self.rigs[rig_id])

    def inputs(self, index):
        row = self.rows[index]
        rig = self.rigs[row["rig_id"]]
        lut = self.lut(row["rig_id"])
        images = []
        valid = []
        h, w = self.config["image_hw"]
        for camera, path in zip(rig["cameras"], row["images"], strict=True):
            if path is None:
                images.append(np.zeros((3, h, w), dtype=np.float32))
                valid.append(False)
                continue
            with Image.open(child_path(self.raw, path)) as image:
                rgb = letterbox_rgb(image, camera["image_hw"], (h, w)).astype(np.float32) / 255
            rgb = (rgb - np.asarray([0.485, 0.456, 0.406], np.float32)) / np.asarray([0.229, 0.224, 0.225], np.float32)
            images.append(rgb.transpose(2, 0, 1))
            valid.append(True)
        return {
            "images": np.stack(images),
            "sensor_valid": np.asarray(valid, dtype=bool),
            "uv": lut["uv"],
            "bins": lut["bins"],
            "geometry_valid": lut["valid"],
        }

    def __getitem__(self, index):
        inputs = self.inputs(index)
        labels = load_targets(sample_path(self.prepared, self.rows[index]), self.config)
        labels["depth"][~inputs["sensor_valid"]] = -1
        return inputs, labels

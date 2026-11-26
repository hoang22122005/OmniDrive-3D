from dataclasses import dataclass

import numpy as np
from PIL import Image

from omnishield.io import fingerprint


def rigid_matrix(value, name):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape not in {(3, 4), (4, 4)} or not np.isfinite(matrix).all():
        raise ValueError(f"{name}: expected a finite rigid transform")
    if matrix.shape == (3, 4):
        matrix = np.vstack((matrix, [0, 0, 0, 1]))
    if (
        not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5)
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=2e-3)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=2e-3)
    ):
        raise ValueError(f"{name}: invalid rotation or homogeneous row")
    return matrix


def transform(points, pose):
    return np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]


@dataclass(frozen=True)
class Grid:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    @classmethod
    def from_config(cls, config):
        def uniform(spec):
            low, high, cells = spec
            if not isinstance(cells, int) or cells <= 0 or high <= low:
                raise ValueError("A grid axis must have low < high and a positive cell count")
            return np.linspace(low, high, cells + 1)

        z = np.asarray(config["z_edges"], dtype=float)
        if z.ndim != 1 or len(z) < 2 or not np.isfinite(z).all() or not (np.diff(z) > 0).all():
            raise ValueError("Z edges must be finite and strictly increasing")
        return cls(uniform(config["x"]), uniform(config["y"]), z)

    @property
    def shape(self):
        return len(self.z) - 1, len(self.y) - 1, len(self.x) - 1

    def centers(self):
        axes = [(edge[:-1] + edge[1:]) / 2 for edge in (self.z, self.y, self.x)]
        z, y, x = np.meshgrid(*axes, indexing="ij")
        return np.stack((x, y, z), axis=-1)

    def indices(self, points):
        indices = np.stack([
            np.searchsorted(edge, points[..., i], side="right") - 1
            for i, edge in ((2, self.z), (1, self.y), (0, self.x))
        ], axis=-1)
        valid = np.all((indices >= 0) & (indices < np.asarray(self.shape)), axis=-1)
        return indices, valid


def depth_edges(config):
    low, boundary, near_bins = config["near"]
    start, high, far_bins = config["far"]
    if not (0 < low < boundary == start < high and near_bins > 0 and far_bins > 0):
        raise ValueError("Depth bin ranges must be positive, contiguous and increasing")
    return np.concatenate((
        np.linspace(low, boundary, near_bins + 1),
        np.geomspace(start, high, far_bins + 1)[1:],
    ))


def model_contract(config):
    height, width = config["image_hw"]
    if min(height, width) < 16 or height % 16 or width % 16 or config["feature_stride"] != 4:
        raise ValueError("Image dimensions must be multiples of 16 and feature stride must be 4")
    if config["camera_count"] < 1 or len(config["classes"]) < 2 or config["classes"][0] != "free":
        raise ValueError("Model needs at least one camera and a free class at index 0")
    return {key: config[key] for key in ("image_hw", "camera_count", "feature_stride", "grid", "depth", "classes")}


def letterbox_parameters(source_hw, target_hw):
    src_h, src_w = source_hw
    out_h, out_w = target_hw
    if min(src_h, src_w, out_h, out_w) <= 0:
        raise ValueError("Invalid image shape")
    scale = min(out_h / src_h, out_w / src_w)
    h, w = max(1, round(src_h * scale)), max(1, round(src_w * scale))
    top, left = (out_h - h) // 2, (out_w - w) // 2
    return h, w, top, left


def letterbox_rgb(image, source_hw, target_hw):
    src_h, src_w = source_hw
    if image.size != (src_w, src_h):
        raise ValueError(f"Image size {image.size} disagrees with calibration {(src_w, src_h)}")
    h, w, top, left = letterbox_parameters(source_hw, target_hw)
    canvas = Image.new("RGB", (target_hw[1], target_hw[0]), (124, 116, 104))
    canvas.paste(image.convert("RGB").resize((w, h), Image.Resampling.BILINEAR), (left, top))
    return np.asarray(canvas).copy()


def brown_distortion(xy, coefficients):
    values = np.asarray(coefficients, dtype=float)
    if values.shape not in {(4,), (5,)} or not np.isfinite(values).all():
        raise ValueError("Brown distortion needs k1, k2, p1, p2 and optional k3")
    k1, k2, p1, p2, k3 = np.pad(values, (0, 5 - len(values)))
    x, y = np.moveaxis(xy, -1, 0)
    square = x**2 + y**2
    scale = 1 + square * (k1 + square * (k2 + square * k3))
    return np.stack((
        x * scale + 2*p1*x*y + p2*(square + 2*x**2),
        y * scale + p1*(square + 2*y**2) + 2*p2*x*y,
    ), axis=-1)


def project(points_ego, camera, target_hw):
    transform_cam = rigid_matrix(camera["T_cam_from_ego"], camera["slot"])
    xyz = transform(points_ego, transform_cam)
    x, y, z = np.moveaxis(xyz, -1, 0)
    radius = np.hypot(x, y)
    distance = np.linalg.norm(xyz, axis=-1)
    theta = np.arctan2(radius, z)
    k = np.asarray(camera["K"], dtype=float).reshape(3, 3)
    if not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError(f"Invalid intrinsics for {camera['slot']}")
    optics = camera["model"]
    valid = (distance > 1e-7) & (theta <= np.deg2rad(camera["max_theta_deg"]))
    if optics == "pinhole":
        valid &= z > 1e-7
        uv_plane = xyz[..., :2] / np.maximum(z[..., None], 1e-7)
        if "distortion" in camera:
            uv_plane = brown_distortion(uv_plane, camera["distortion"])
    elif optics == "mei":
        xi = float(camera["xi"])
        if not np.isfinite(xi) or xi < 0:
            raise ValueError("MEI requires a finite nonnegative mirror coefficient")
        denominator = z + xi * distance
        valid &= denominator > 1e-7
        uv_plane = xyz[..., :2] / np.maximum(denominator[..., None], 1e-7)
        uv_plane = brown_distortion(uv_plane, camera.get("distortion", [0, 0, 0, 0]))
    elif optics == "kb4":
        coefficients = np.asarray(camera["distortion"], dtype=float)
        if coefficients.shape != (4,) or not np.isfinite(coefficients).all():
            raise ValueError("KB4 requires four finite distortion coefficients")
        t2 = theta ** 2
        r = theta * (1 + t2 * np.polynomial.polynomial.polyval(t2, coefficients))
        uv_plane = xyz[..., :2] * (r / np.maximum(radius, 1e-9))[..., None]
    else:
        raise ValueError(f"Unsupported camera model: {optics}")
    uv = uv_plane @ k[:2, :2].T + k[:2, 2]
    src_h, src_w = camera["image_hw"]
    if "valid_circle" in camera:
        cx, cy, r = camera["valid_circle"]
        valid &= np.hypot(uv[..., 0] - cx, uv[..., 1] - cy) <= r
    valid &= np.all((uv >= -0.5) & (uv < np.asarray([src_w, src_h]) - 0.5), axis=-1)
    h, w, top, left = letterbox_parameters((src_h, src_w), target_hw)
    uv = (uv + 0.5) * [w / src_w, h / src_h] - 0.5 + [left, top]
    return uv, distance, valid


def ego_mask(points, rig):
    xmin, xmax, ymin, ymax = rig["ego_xy_bounds_m"]
    return (
        (points[..., 0] >= xmin) & (points[..., 0] <= xmax)
        & (points[..., 1] >= ymin) & (points[..., 1] <= ymax)
    )


def bundle_key(config, rig):
    return fingerprint({"calibration_version": 1, "model": model_contract(config), "rig": rig})


def build_lut(config, rig):
    model_contract(config)
    grid = Grid.from_config(config["grid"])
    centers = grid.centers()
    excluded = ego_mask(centers, rig)
    edges = depth_edges(config["depth"])
    h, w = config["image_hw"]
    uv_values, bins, valid_values = [], [], []
    for camera in rig["cameras"]:
        uv, distance, valid = project(centers, camera, (h, w))
        index = np.searchsorted(edges, distance, side="right") - 1
        valid &= ~excluded & (index >= 0) & (index < len(edges) - 1)
        # Feature grid uses align_corners=False and the same pixel centers as RGB.
        normalized = 2 * (uv + 0.5) / [w, h] - 1
        uv_values.append(np.where(valid[..., None], normalized, 0))
        bins.append(np.where(valid, index, 0))
        valid_values.append(valid)
    return {
        "uv": np.asarray(uv_values, dtype=np.float32),
        "bins": np.asarray(bins, dtype=np.int64),
        "valid": np.asarray(valid_values, dtype=bool),
        "key": np.asarray(bundle_key(config, rig)),
    }

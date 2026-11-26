import numpy as np

from omnishield.data.phenikaa import load_lidar, point_semantics
from omnishield.geometry.calibration import Grid, depth_edges, ego_mask, project
from omnishield.io import child_path


def camera_first_returns(points, config, rig):
    h, w = config["image_hw"]
    stride = config["feature_stride"]
    ranges = []
    for camera in rig["cameras"]:
        uv, distance, valid = project(points, camera, (h, w))
        index = np.floor((uv[valid] + 0.5) / stride).astype(np.int64)
        d = distance[valid]
        linear = index[:, 0] + index[:, 1] * (w // stride)
        nearest = np.full((h // stride) * (w // stride), np.inf)
        np.minimum.at(nearest, linear, d)
        ranges.append(nearest.reshape(h // stride, w // stride))
    return np.asarray(ranges)


def depth_targets(first_returns, config):
    edges = depth_edges(config["depth"])
    bins = np.searchsorted(edges, first_returns, side="right") - 1
    supported = (first_returns >= edges[0]) & (first_returns < edges[-1])
    return np.where(supported, bins, -1).astype(np.int16)


def add_camera_free_proxy(occupied, semantics, ranges, config, grid, rig, images):
    """Weak free evidence in front of several neighboring projected LiDAR returns."""
    centers = grid.centers()
    stride = config["feature_stride"]
    bound = depth_edges(config["depth"])
    width, height, depth = (np.diff(a).max() for a in (grid.x, grid.y, grid.z))
    margin = float(np.linalg.norm([width, height, depth]) / 2 + 0.35)
    usable = (centers[..., 2] > rig["ground_height_m"]) & ~ego_mask(centers, rig)
    for camera, first, image in zip(rig["cameras"], ranges, images, strict=True):
        if not image:
            continue
        uv, distance, valid = project(centers, camera, config["image_hw"])
        cell = np.floor((uv + 0.5) / stride - 0.5).astype(np.int64)
        u, v = np.moveaxis(cell, -1, 0)
        fh, fw = first.shape
        inside = (u >= 0) & (v >= 0) & (u < fw - 1) & (v < fh - 1)
        u, v = u.clip(0, fw - 2), v.clip(0, fh - 2)
        support = np.stack((first[v, u], first[v, u + 1], first[v + 1, u], first[v + 1, u + 1]))
        valid_support = (support >= bound[0]) & (support <= bound[-1])
        closest = np.where(valid_support, support, np.inf).min(axis=0)
        votes = valid_support.sum(axis=0)
        free = (usable & inside & valid & (votes >= 3) & (distance >= bound[0])
                & (distance + margin < closest) & (occupied == -1))
        occupied[free] = 0
        semantics[free] = 0


def occupancy_targets(points, origin, semantics, grid, rig):
    occupancy = np.full(grid.shape, -1, dtype=np.int8)
    semantic = np.full(grid.shape, -1, dtype=np.int16)
    if origin is not None:
        # Approximate ray traversal in bounded chunks. Never label beyond a return.
        step = min(np.diff(grid.x).min(), np.diff(grid.y).min(), np.diff(grid.z).min()) / 2
        far_corner = np.maximum(
            abs(np.array([grid.x[0], grid.y[0], grid.z[0]]) - origin),
            abs(np.array([grid.x[-1], grid.y[-1], grid.z[-1]]) - origin),
        )
        step_count = int(np.ceil(np.linalg.norm(far_corner) / step))
        fractions = np.arange(1, step_count + 1) * step
        for start in range(0, len(points), 512):
            delta = points[start:start + 512] - origin
            ranges = np.linalg.norm(delta, axis=1)
            directions = delta / np.maximum(ranges[:, None], 1e-8)
            travel = fractions[None, :] < ranges[:, None] - step
            ray_samples = (origin + directions[:, None, :] * fractions[None, :, None])[travel]
            indices, valid = grid.indices(ray_samples)
            valid &= (ray_samples[:, 2] > rig["ground_height_m"]) & ~ego_mask(ray_samples, rig)
            occupancy[tuple(indices[valid].T)] = 0
    indices, valid = grid.indices(points)
    valid &= (points[:, 2] > rig["ground_height_m"] + 0.08) & ~ego_mask(points, rig)
    linear = np.ravel_multi_index(indices[valid].T, grid.shape)
    occupancy.flat[linear] = 1
    semantic[occupancy == 0] = 0
    labelled = semantics[valid]
    small = np.full(occupancy.size, np.iinfo(np.int16).max, dtype=np.int16)
    large = np.full(occupancy.size, -1, dtype=np.int16)
    positive = labelled > 0
    np.minimum.at(small, linear[positive], labelled[positive])
    np.maximum.at(large, linear[positive], labelled[positive])
    agreeing = (small == large) & (large > 0)
    semantic.flat[np.flatnonzero(agreeing)] = large[agreeing]
    excluded = ego_mask(grid.centers(), rig)
    occupancy[excluded] = semantic[excluded] = -1
    return occupancy, semantic


def prepare_phenikaa(raw, row, config, rig):
    points, origin = load_lidar(raw, row, rig)
    points = points[np.isfinite(points).all(axis=1) & ~ego_mask(points, rig)]
    label_path = child_path(raw, row["boxes"]) if row.get("boxes") else None
    grid = Grid.from_config(config["grid"])
    _, in_grid = grid.indices(points)
    labels = point_semantics(label_path, points[in_grid], rig, config["classes"])
    classes = np.full(len(points), -1, dtype=np.int16)
    classes[in_grid] = labels
    if not rig.get("confirmed_single_lidar_origin", False):
        origin = None
    occupancy, semantics = occupancy_targets(points, origin, classes, grid, rig)
    first = camera_first_returns(points, config, rig)
    mode = rig.get("free_label_mode", "lidar_if_confirmed")
    if mode == "camera_depth_proxy":
        add_camera_free_proxy(occupancy, semantics, first, config, grid, rig, row["images"])
    elif mode != "lidar_if_confirmed":
        raise ValueError(f"Unsupported free label mode: {mode}")
    return {
        "occupancy": occupancy, "semantics": semantics, "depth": depth_targets(first, config),
        "free_label_source": np.asarray(mode),
    }

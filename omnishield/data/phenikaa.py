import re
from pathlib import Path

import numpy as np
from PIL import Image

from omnishield.geometry.calibration import rigid_matrix, transform
from omnishield.io import child_path, load_json


SLOTS = {"front": "CAM_F_F", "rear": "CAM_F_B", "left": "CAM_F_L", "right": "CAM_F_R"}


def timestamp_ns(stem):
    match = re.fullmatch(r"(\d+)-(\d{9})", stem)
    if match is None:
        raise ValueError(f"Expected timestamp seconds-nanoseconds: {stem}")
    return int(match[1]) * 10**9 + int(match[2])


def ingest(root, settings, calibration):
    root = Path(root).resolve()
    if settings["dataset_revision"].startswith("replace-"):
        raise ValueError("Set the dataset revision from the downloaded PHENIKAA release")
    ego_from_lidar = rigid_matrix(settings["T_ego_from_lidar_ref"], "T_ego_from_lidar_ref")
    intrinsics = load_json(Path(calibration) / "Camera_Intrinsics.json")
    extrinsics = load_json(Path(calibration) / "Sensor_Extrinsics.json")
    cameras, frames = {}, []
    assigned = set()
    for split, clips in settings["clips"].items():
        for drive in clips:
            if drive in assigned:
                raise ValueError(f"Clip appears more than once: {drive}")
            assigned.add(drive)
            clip = child_path(root, drive)
            scans = sorted((clip / "lidar").glob("*.laz"), key=lambda p: timestamp_ns(p.stem))
            if not scans:
                raise ValueError(f"No LiDAR frames in {clip / 'lidar'}")
            for scan in scans:
                stamp = timestamp_ns(scan.stem)
                images = []
                for slot, channel in SLOTS.items():
                    path = clip / "camera" / channel / (scan.stem + ".jpg")
                    if not path.is_file():
                        images.append(None)
                        continue
                    if slot not in cameras:
                        with Image.open(path) as image:
                            hw = [image.height, image.width]
                        native = intrinsics[channel]
                        cam_from_lidar = rigid_matrix(extrinsics[channel], channel)
                        cameras[slot] = {
                            "slot": slot, "model": "kb4", "image_hw": hw,
                            "K": np.asarray(native["camera_matrix"]).reshape(3, 3).tolist(),
                            "distortion": native["distortion_coefficients"][:4],
                            "T_cam_from_ego": (cam_from_lidar @ np.linalg.inv(ego_from_lidar)).tolist(),
                            "max_theta_deg": settings.get("camera_max_theta_deg", {}).get(slot, settings["optics_max_theta_deg"]),
                        }
                    images.append(path.relative_to(root).as_posix())
                box_path = clip / "label" / (scan.stem + ".txt")
                pose_path = clip / "ego_pose" / (scan.stem + ".txt")
                if any(images):
                    frames.append({
                        "drive_id": drive, "frame_id": scan.stem, "split": split, "timestamp_ns": stamp,
                        "rig_id": settings["rig_id"], "images": images,
                        "lidar": scan.relative_to(root).as_posix(),
                        "boxes": box_path.relative_to(root).as_posix() if box_path.is_file() else None,
                        "pose": pose_path.relative_to(root).as_posix() if pose_path.is_file() else None,
                    })
    if set(cameras) != set(SLOTS):
        raise ValueError("Calibration ingest did not find frames for all four channels")
    rig = {
        "cameras": [cameras[slot] for slot in SLOTS],
        "ego_xy_bounds_m": settings["ego_xy_bounds_m"],
        "T_ego_from_lidar_ref": ego_from_lidar.tolist(),
        "ground_height_m": settings["ground_height_m"],
        "confirmed_single_lidar_origin": settings.get("confirmed_single_lidar_origin", False),
        "free_label_mode": settings.get("free_label_mode", "lidar_if_confirmed"),
    }
    return {
        "version": 1, "source": "phenikaa", "revision": settings["dataset_revision"],
        "rigs": {settings["rig_id"]: rig}, "frames": frames,
    }


def load_lidar(root, row, rig):
    import laspy

    scan = laspy.read(child_path(root, row["lidar"]))
    points = np.column_stack((scan.x, scan.y, scan.z))
    pose = rigid_matrix(rig["T_ego_from_lidar_ref"], "T_ego_from_lidar_ref")
    origin = transform(np.zeros((1, 3)), pose)[0]
    return transform(points, pose), origin


def point_semantics(path, world_points, rig, classes):
    labels = np.full(len(world_points), -1, dtype=np.int64)
    if path is None:
        return labels
    lookup = {"car": "vehicle", "stationary": "static", "curb": "curb",
              "rider": "rider", "pedestrian": "pedestrian"}
    ego_from_lidar = rigid_matrix(rig["T_ego_from_lidar_ref"], "T_ego_from_lidar_ref")
    points = transform(world_points, np.linalg.inv(ego_from_lidar))
    ambiguous = np.zeros(len(points), dtype=bool)
    for row in path.read_text(encoding="utf-8-sig").splitlines():
        fields = row.split()
        if not fields:
            continue
        if len(fields) < 8:
            raise ValueError(f"Invalid box row in {path}")
        x, y, z, length, width, height, yaw = map(float, fields[:7])
        cls = lookup.get(" ".join(fields[7:]).casefold())
        if cls not in classes or min(length, width, height) <= 0:
            continue
        local = points - [x, y, z]
        c, s = np.cos(yaw), np.sin(yaw)
        ix = c * local[:, 0] + s * local[:, 1]
        iy = -s * local[:, 0] + c * local[:, 1]
        inside = (abs(ix) <= length / 2) & (abs(iy) <= width / 2) & (abs(local[:, 2]) <= height / 2)
        label = classes.index(cls)
        ambiguous |= inside & (labels >= 0) & (labels != label)
        labels[inside] = label
    labels[ambiguous] = -1
    return labels

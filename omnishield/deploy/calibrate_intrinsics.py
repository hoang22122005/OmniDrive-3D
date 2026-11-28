"""One-time KB4 fisheye intrinsic calibration tool using OpenCV cv2.fisheye.

Supports:
- Batch calibration from a directory of checkerboard images (JPG/PNG)
- Batch extraction from a video file (MP4/AVI/MKV)
- Live interactive capture from a USB/V4L2/GMSL fisheye camera
Outputs OmniShield-X / PHENIKAA-compatible ``Camera_Intrinsics.json`` (model = ``kannala_brandt``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np


SLOT_TO_CAM_KEY = {
    "front": "CAM_F_F",
    "rear": "CAM_F_B",
    "left": "CAM_F_L",
    "right": "CAM_F_R",
}


def detect_checkerboard_corners(gray: np.ndarray, board_size: tuple[int, int]):
    """Detect and sub-pixel refine checkerboard corners even under strong 190-deg fisheye warp."""
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_FAST_CHECK
        + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    ok, corners = cv2.findChessboardCorners(gray, board_size, flags)
    if not ok:
        try:
            ok, corners = cv2.findChessboardCornersSB(gray, board_size, cv2.CALIB_CB_EXHAUSTIVE)
        except Exception:
            ok = False
    if not ok or corners is None:
        return None
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    corners = cv2.cornerSubPix(gray, corners.astype(np.float32), (7, 7), (-1, -1), term)
    return corners


def calibrate_fisheye_kb4(
    images: list[np.ndarray],
    board_size: tuple[int, int] = (9, 6),
    square_size_m: float = 0.040,
    max_reproj_px: float = 0.85,
) -> dict:
    """Calibrate a KB4 fisheye camera (fx, fy, cx, cy, skew, k1..k4) with 2-pass outlier rejection."""
    if not images:
        raise ValueError("No calibration images provided")

    cols, rows = int(board_size[0]), int(board_size[1])
    objp = np.zeros((1, cols * rows, 3), dtype=np.float64)
    objp[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(square_size_m)

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    used_indices: list[int] = []
    img_hw: tuple[int, int] | None = None

    for idx, im in enumerate(images):
        if im is None:
            continue
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
        if img_hw is None:
            img_hw = (int(gray.shape[0]), int(gray.shape[1]))
        elif gray.shape[:2] != img_hw:
            raise ValueError(f"Image {idx} shape {gray.shape[:2]} != expected {img_hw}")
        corners = detect_checkerboard_corners(gray, (cols, rows))
        if corners is not None:
            objpoints.append(objp.copy())
            imgpoints.append(corners.reshape(1, -1, 2).astype(np.float64))
            used_indices.append(idx)

    if len(objpoints) < 6 or img_hw is None:
        raise ValueError(
            f"Only {len(objpoints)} valid checkerboard views detected (need >= 6 across center & edges)."
        )

    h, w = img_hw
    _flag = lambda name: getattr(cv2.fisheye, name, getattr(cv2, name))
    calib_flags = (
        _flag("CALIB_RECOMPUTE_EXTRINSIC")
        | _flag("CALIB_CHECK_COND")
        | _flag("CALIB_FIX_SKEW")
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 1e-6)

    def _run_pass(objs, imgs):
        K = np.eye(3, dtype=np.float64)
        D = np.zeros((4, 1), dtype=np.float64)
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in objs]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in objs]
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objs, imgs, (w, h), K, D, rvecs, tvecs, calib_flags, criteria
        )
        per_view_err = []
        for o, ip, rv, tv in zip(objs, imgs, rvecs, tvecs):
            proj, _ = cv2.fisheye.projectPoints(o, rv, tv, K, D)
            err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - ip.reshape(-1, 2)) ** 2, axis=1))))
            per_view_err.append(err)
        return float(rms), K, D.reshape(4), per_view_err

    rms, K, D, view_errs = _run_pass(objpoints, imgpoints)

    # Pass 2: drop high-error outlier views if enough clean views remain
    keep = [i for i, e in enumerate(view_errs) if e <= max_reproj_px]
    if 6 <= len(keep) < len(objpoints):
        objpoints = [objpoints[i] for i in keep]
        imgpoints = [imgpoints[i] for i in keep]
        used_indices = [used_indices[i] for i in keep]
        rms, K, D, view_errs = _run_pass(objpoints, imgpoints)

    return {
        "camera_model": "kannala_brandt",
        "image_width": w,
        "image_height": h,
        "K": np.round(K, 6).tolist(),
        "distortion": [round(float(v), 6) for v in D],
        "fx": round(float(K[0, 0]), 6),
        "fy": round(float(K[1, 1]), 6),
        "cx": round(float(K[0, 2]), 6),
        "cy": round(float(K[1, 2]), 6),
        "skew": round(float(K[0, 1]), 6),
        "reprojection_rmse_px": round(float(rms), 4),
        "views_used": len(objpoints),
        "views_total": len(images),
        "max_view_rmse_px": round(float(max(view_errs)), 4),
    }


def render_undistort_verification(image: np.ndarray, intr: dict, out_path: Path, hfov_deg: float = 125.0) -> Path:
    """Save a rectilinear undistorted image with cyan straightness grid to verify KB4 accuracy."""
    h, w = int(intr["image_height"]), int(intr["image_width"])
    out_w, out_h = min(1280, w), min(720, h)
    f_rect = (0.5 * out_w) / math.tan(math.radians(0.5 * hfov_deg))
    uu, vv = np.meshgrid(np.arange(out_w, dtype=np.float64), np.arange(out_h, dtype=np.float64))
    xn = (uu - 0.5 * out_w) / f_rect
    yn = (vv - 0.5 * out_h) / f_rect
    rad = np.hypot(xn, yn)
    th = np.arctan2(rad, 1.0)
    kd = np.asarray(intr["distortion"], dtype=np.float64)
    t2 = th * th
    r_d = th * (1.0 + t2 * (kd[0] + t2 * (kd[1] + t2 * (kd[2] + t2 * kd[3]))))
    sc = r_d / np.maximum(rad, 1e-9)
    K = np.asarray(intr["K"], dtype=np.float64)
    mx = (xn * sc * K[0, 0] + yn * sc * K[0, 1] + K[0, 2]).astype(np.float32)
    my = (yn * sc * K[1, 1] + K[1, 2]).astype(np.float32)
    undist = cv2.remap(image, mx, my, cv2.INTER_LINEAR)
    for x in range(0, out_w, 80):
        cv2.line(undist, (x, 0), (x, out_h), (255, 220, 0), 1)
    for y in range(0, out_h, 80):
        cv2.line(undist, (0, y), (out_w, y), (255, 220, 0), 1)
    cv2.putText(
        undist,
        f"KB4 Undistort {hfov_deg:.0f}deg | RMSE={intr['reprojection_rmse_px']:.3f}px ({intr['views_used']} views)",
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), undist, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return out_path


def save_camera_intrinsics_json(
    out_path: Path,
    intr: dict,
    slot: str = "all",
    merge_existing: bool = True,
) -> Path:
    """Write or update ``Camera_Intrinsics.json`` in exact OmniShield-X / PHENIKAA format."""
    data = {}
    if merge_existing and out_path.is_file():
        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}

    entry = {
        "camera_model": "kannala_brandt",
        "image_width": int(intr["image_width"]),
        "image_height": int(intr["image_height"]),
        "fx": float(intr["fx"]),
        "fy": float(intr["fy"]),
        "cx": float(intr["cx"]),
        "cy": float(intr["cy"]),
        "skew": float(intr["skew"]),
        "K": intr["K"],
        "distortion": intr["distortion"],
        "reprojection_rmse_px": intr["reprojection_rmse_px"],
        "views_used": intr["views_used"],
    }

    slots = list(SLOT_TO_CAM_KEY.keys()) if slot == "all" else [slot]
    for s in slots:
        cam_key = SLOT_TO_CAM_KEY.get(s, s)
        data[cam_key] = dict(entry)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out_path


def _load_images_from_source(images_dir: Path | None, video_path: Path | None, stride: int = 10, max_frames: int = 45):
    imgs = []
    if images_dir is not None:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)
        for p in files[:max_frames]:
            im = cv2.imread(str(p))
            if im is not None:
                imgs.append(im)
    elif video_path is not None:
        cap = cv2.VideoCapture(str(video_path))
        idx = 0
        try:
            while cap.isOpened() and len(imgs) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % max(1, stride) == 0:
                    imgs.append(frame)
                idx += 1
        finally:
            cap.release()
    return imgs


def _capture_live_checkerboards(camera_index: int, board_size: tuple[int, int], target_views: int = 20):
    """Interactive live capture from a connected USB/V4L2 camera (press SPACE to grab, Q to solve)."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")
    captured = []
    print(f"[KB4 Calib] Camera {camera_index} opened. Move checkerboard across center, 4 edges & 4 corners.")
    print("[KB4 Calib] Press SPACE to capture a detected view, Q / ESC to finish & calibrate.")
    try:
        while len(captured) < target_views:
            ok, frame = cap.read()
            if not ok:
                continue
            vis = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = detect_checkerboard_corners(gray, board_size)
            if corners is not None:
                cv2.drawChessboardCorners(vis, board_size, corners, True)
            cv2.putText(
                vis,
                f"Views: {len(captured)}/{target_views} | {'DETECTED - SPACE' if corners is not None else 'Show checkerboard'}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 120) if corners is not None else (80, 180, 255),
                2,
            )
            cv2.imshow("OmniShield KB4 Intrinsic Calibration", vis)
            key = cv2.waitKey(15) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == 32 and corners is not None:
                captured.append(frame.copy())
                print(f"  Captured view {len(captured)}/{target_views}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return captured


def run_calib_intrinsics_cli(args):
    """CLI entrypoint for ``python -m omnishield calib-intrinsics``."""
    board = (int(args.cols), int(args.rows))
    if args.camera is not None:
        imgs = _capture_live_checkerboards(args.camera, board, target_views=args.max_views)
    else:
        imgs = _load_images_from_source(args.images, args.video, stride=args.stride, max_frames=args.max_views)

    intr = calibrate_fisheye_kb4(
        imgs,
        board_size=board,
        square_size_m=float(args.square_mm) * 0.001,
        max_reproj_px=float(args.max_rmse),
    )
    out_json = save_camera_intrinsics_json(args.out, intr, slot=args.slot, merge_existing=True)
    print(
        f"[KB4 Calib] Saved {out_json} (slot={args.slot}) | "
        f"fx={intr['fx']:.2f}, fy={intr['fy']:.2f}, cx={intr['cx']:.2f}, cy={intr['cy']:.2f} | "
        f"k={intr['distortion']} | RMSE={intr['reprojection_rmse_px']:.4f} px ({intr['views_used']} views)"
    )
    if args.verify_img and imgs:
        vpath = render_undistort_verification(imgs[0], intr, args.verify_img)
        print(f"[KB4 Calib] Saved straightness verification -> {vpath}")
    return intr

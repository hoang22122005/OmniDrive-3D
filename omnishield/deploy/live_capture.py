"""Real-time 4-camera hardware capture server (V4L2 / DirectShow / Quad AHD USB / RTSP / Video).

Serves the exact same 60 FPS WebGL 3D AVM & Garage PnP Calibration UI (``demo.html``) directly
from live hardware camera streams — ZERO LiDAR, ZERO AI backbone required at runtime!
Also supports live 1-click ``/api/calibrate_pnp``, ``/api/save_calibration``, and ``/api/export_lut``.
"""

from __future__ import annotations

import copy
import json
import threading
import time
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch

from omnishield.deploy.mat_and_lut import export_fixed_lut
from omnishield.eval.scene3d import (
    SeamColorCalibrator,
    bowl_seam_calibration,
    build_adaptive_bowl_template,
    export_bowl_spec,
    export_camera_calib,
    export_vehicle_spec,
    load_calibrated_rig_profile,
    measure_near_field_coverage,
    measure_seam_parallax_report,
    solve_fisheye_pnp_camera,
    verify_bowl_mesh_integrity,
)
from omnishield.io import write_json


DEMO_HTML = Path(__file__).resolve().parent.parent / "eval" / "demo.html"
VENDOR_DIR = Path(__file__).resolve().parent.parent / "eval" / "vendor"


def _parse_source_token(token: str):
    token = token.strip()
    if token.isdigit():
        return int(token)
    return token


class Live4CamCapture:
    """Thread-safe ring-buffer capture from 4 independent cameras OR a 2x2 quad-split AHD USB card."""

    def __init__(
        self,
        sources: list[str] | None = None,
        quad_source: str | None = None,
        quad_order: tuple[int, int, int, int] = (0, 1, 2, 3),
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        jpeg_quality: int = 84,
    ):
        self.sources = [_parse_source_token(s) for s in (sources or [])]
        self.quad_source = _parse_source_token(quad_source) if quad_source is not None else None
        self.quad_order = tuple(int(i) for i in quad_order)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.jpeg_quality = int(jpeg_quality)

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._frames_bgr: list[np.ndarray | None] = [None, None, None, None]
        self._frames_jpg: list[bytes | None] = [None, None, None, None]
        self._frame_seq = 0
        self._threads: list[threading.Thread] = []
        self._caps: list[cv2.VideoCapture] = []

    def _encode_jpg(self, bgr: np.ndarray, target_w: int | None = None) -> bytes:
        im = bgr
        if target_w and target_w > 0 and im.shape[1] > target_w:
            th = max(2, int(round(im.shape[0] * (target_w / im.shape[1]))))
            im = cv2.resize(im, (target_w, th), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        return buf.tobytes() if ok else b""

    def _open_cap(self, src):
        cap = cv2.VideoCapture(src)
        if isinstance(src, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _quad_loop(self):
        cap = self._open_cap(self.quad_source)
        self._caps.append(cap)
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                if isinstance(self.quad_source, str) and Path(self.quad_source).is_file():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.02)
                continue
            h, w = frame.shape[:2]
            mh, mw = h // 2, w // 2
            tiles = [
                frame[0:mh, 0:mw],
                frame[0:mh, mw:w],
                frame[mh:h, 0:mw],
                frame[mh:h, mw:w],
            ]
            ordered = [tiles[idx] for idx in self.quad_order]
            jpgs = [self._encode_jpg(t) for t in ordered]
            with self._lock:
                self._frames_bgr = ordered
                self._frames_jpg = jpgs
                self._frame_seq += 1

    def _single_loop(self, slot_idx: int, src):
        cap = self._open_cap(src)
        self._caps.append(cap)
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                if isinstance(src, str) and Path(src).is_file():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.02)
                continue
            jpg = self._encode_jpg(frame)
            with self._lock:
                self._frames_bgr[slot_idx] = frame
                self._frames_jpg[slot_idx] = jpg
                self._frame_seq += 1

    def start(self):
        if self.quad_source is not None:
            t = threading.Thread(target=self._quad_loop, name="quad-cam-capture", daemon=True)
            t.start()
            self._threads.append(t)
        elif len(self.sources) == 4:
            for ci, src in enumerate(self.sources):
                t = threading.Thread(target=self._single_loop, args=(ci, src), name=f"cam-{ci}-capture", daemon=True)
                t.start()
                self._threads.append(t)
        else:
            raise ValueError("Provide either --quad <dev> OR 4 comma-separated --sources <F,B,L,R>")

    def stop(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.0)
        for cap in self._caps:
            try:
                cap.release()
            except Exception:
                pass

    def get_jpeg(self, cam_idx: int, target_w: int | None = None) -> bytes:
        with self._lock:
            bgr = self._frames_bgr[cam_idx]
            jpg = self._frames_jpg[cam_idx]
        if bgr is None:
            blank = np.full((360, 640, 3), 24, dtype=np.uint8)
            cv2.putText(blank, f"Waiting CAM {cam_idx}...", (180, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (144, 243, 190), 2)
            return self._encode_jpg(blank)
        if target_w and target_w > 0 and bgr.shape[1] > target_w:
            return self._encode_jpg(bgr, target_w=target_w)
        return jpg if jpg is not None else self._encode_jpg(bgr)


def _build_live_surround_guidelines(rig: dict, ground_plane: list[float]) -> dict:
    calibs = export_camera_calib(rig)
    pa, pb, pc = [float(v) for v in ground_plane]
    color_cal = SeamColorCalibrator(rig, torch.device("cpu"), (256, 448), pc)
    spec = export_vehicle_spec(rig, ground_plane=ground_plane)
    x0, x1 = spec["x_rear_bumper_m"], spec["x_front_bumper_m"]
    hw = 0.5 * spec["width_m"]
    return {
        "ground_z": round(pc, 5),
        "ground_plane": [round(pa, 6), round(pb, 6), round(pc, 6)],
        "bowl_spec": export_bowl_spec(),
        "vehicle_spec": spec,
        "camera_calib": calibs,
        "overlap_uv": color_cal.overlap_uv,
        "bowl_seams": bowl_seam_calibration(rig, calibs, pc, ground_plane=ground_plane),
        "seam_parallax_cm": measure_seam_parallax_report(rig, ground_plane),
        "near_field_coverage": measure_near_field_coverage(rig, ground_plane),
        "mesh_integrity": verify_bowl_mesh_integrity(ground_plane),
        "road_plane_validation": {"source": "live_hardware_profile"},
        "extrinsic_refine_deg": [0.0] * len(rig["cameras"]),
        "calib_dz_cm": [0.0] * len(rig["cameras"]),
        "bev_bounds": {"x": [7.0, -7.0], "y": [5.0, -5.0], "hw": [560, 400], "ego": [x0, x1, -hw, hw]},
        "rect": [{"tracks": [], "bars": []} for _ in range(len(rig["cameras"]))],
        "wide": {"front": {"tracks": [], "bars": []}, "rear": {"tracks": [], "bars": []}},
    }


def serve_live_4cam(args):
    """Launch live 4-camera hardware server serving the 60 FPS WebGL 3D AVM & Garage PnP UI."""
    profile_path = Path(args.profile)
    profile = load_calibrated_rig_profile(profile_path)
    active_rig = copy.deepcopy(profile["rig"])
    active_gp = [float(v) for v in profile.get("ground_plane", [0.0, 0.0, 0.023])]
    guidelines = _build_live_surround_guidelines(active_rig, active_gp)
    bowl_bin = build_adaptive_bowl_template(active_gp[2])

    sources = [s.strip() for s in args.sources.split(",") if s.strip()] if args.sources else None
    quad_order = tuple(int(x) for x in args.quad_order.split(","))
    capture = Live4CamCapture(
        sources=sources,
        quad_source=args.quad,
        quad_order=quad_order,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    capture.start()
    lut_out_dir = Path(args.lut_out)

    class LiveHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            return

        def _send(self, code, payload: bytes, ctype: str, cache: str = "no-store"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                self.close_connection = True

        def do_GET(self):
            nonlocal guidelines
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            qs = urllib.parse.parse_qs(parsed.query)
            if path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, DEMO_HTML.read_bytes(), "text/html; charset=utf-8")
                return
            if path.startswith("/vendor/"):
                name = Path(path).name
                target = VENDOR_DIR / name
                if target.is_file() and name.endswith(".js"):
                    self._send(HTTPStatus.OK, target.read_bytes(), "application/javascript; charset=utf-8", cache="public, max-age=3600")
                else:
                    self._send(HTTPStatus.NOT_FOUND, b"Not Found", "text/plain")
                return
            if path == "/api/meta":
                meta = {
                    "source": "live-4cam",
                    "split": "live",
                    "ai_enabled": False,
                    "device": "Live 4-Cam Hardware (60 FPS WebGL)",
                    "device_type": "live",
                    "torch": "no-ai",
                    "classes": ["free", "vehicle", "rider", "pedestrian"],
                    "grid": {"shape": [4, 60, 80], "x": [-8.0, 8.0], "y": [-6.0, 6.0], "z": [-0.2, 2.5]},
                    "surround": guidelines,
                    "clips": [{
                        "drive_id": "LIVE 4-CAMERA",
                        "split": "live",
                        "start_index": 0,
                        "count": 1,
                        "surround": guidelines,
                    }],
                    "default_threshold": 0.5,
                    "default_max_uncertainty": 0.85,
                    "frames": [{
                        "index": 0,
                        "drive_id": "LIVE 4-CAMERA",
                        "frame_id": "live",
                        "split": "live",
                        "timestamp_s": 0.0,
                        "speed_kmh": 0.0,
                        "steer_rad": 0.0,
                        "images_present": [True, True, True, True],
                        "lidar_present": False,
                        "odom_from_prev": [0.0, 0.0, 0.0],
                    }],
                }
                self._send(HTTPStatus.OK, json.dumps(meta, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
                return
            if path == "/api/bowl.bin" or path.startswith("/api/bowl/"):
                self._send(HTTPStatus.OK, bowl_bin, "application/octet-stream")
                return
            if path in ("/api/image", "/api/cam_rect", "/api/cam_wide"):
                cam_idx = max(0, min(3, int(qs.get("cam", ["0"])[0])))
                w_tok = qs.get("w", ["0"])[0]
                tw = int(w_tok) if w_tok.isdigit() and int(w_tok) > 0 else None
                self._send(HTTPStatus.OK, capture.get_jpeg(cam_idx, target_w=tw), "image/jpeg")
                return
            if path.startswith("/api/image/"):
                parts = path.strip("/").split("/")
                if len(parts) >= 4:
                    cam_idx = max(0, min(3, int(parts[3])))
                    tw = int(qs["w"][0]) if "w" in qs and qs["w"][0].isdigit() else None
                    self._send(HTTPStatus.OK, capture.get_jpeg(cam_idx, target_w=tw), "image/jpeg")
                    return
            if path == "/api/infer" or path.startswith("/api/frame/"):
                empty_pack = {
                    "voxels": [], "counts": {"occupied": 0}, "objects": [], "road_edges": [],
                    "pdc_sectors": [5.0] * 8, "corridor": None, "seam_zones": [],
                }
                payload = {
                    "index": 0,
                    "drive_id": "LIVE 4-CAMERA",
                    "frame_id": f"live-{int(time.time() * 1000)}",
                    "ai_enabled": False,
                    "latency_ms": 0.0,
                    "p_occ_max": 0.5,
                    "threshold": 0.5,
                    "max_uncertainty": 0.85,
                    "sensor_valid": [True, True, True, True],
                    "metrics": {"occupancy_iou": None, "coverage": None},
                    "adaptive_bowl": {"inflect_radii": [], "cam_gains": [[1, 1, 1]] * 4, "corridor": None, "seam_zones": []},
                    "prediction": empty_pack,
                    "ground_truth": empty_pack,
                }
                self._send(HTTPStatus.OK, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")
                return
            self._send(HTTPStatus.NOT_FOUND, b"Not Found", "text/plain")

        def do_POST(self):
            nonlocal active_rig, active_gp, guidelines
            try:
                clen = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(clen).decode("utf-8")) if clen > 0 else {}
                if self.path == "/api/calibrate_pnp":
                    ci = int(payload.get("cam_index", payload.get("cam_idx", 0)))
                    if ci < 0 or ci >= len(active_rig["cameras"]):
                        raise ValueError(f"Invalid cam_index: {ci}")
                    img_pts = payload.get("image_points", payload.get("pixel_pts"))
                    world_pts = payload.get("world_points", payload.get("world_pts"))
                    cam = copy.deepcopy(active_rig["cameras"][ci])
                    res = solve_fisheye_pnp_camera(cam, img_pts, world_pts)
                    active_rig["cameras"][ci]["T_cam_from_ego"] = res["T_cam_from_ego"]
                    guidelines = _build_live_surround_guidelines(active_rig, active_gp)
                    self._send(
                        HTTPStatus.OK,
                        json.dumps({
                            "ok": True,
                            "cam_index": ci,
                            "reproj_rmse_px": res["reproj_rmse_px"],
                            "max_err_px": res["max_err_px"],
                            "cam_origin": res["cam_origin"],
                            "camera_calib": guidelines["camera_calib"],
                        }, ensure_ascii=False).encode("utf-8"),
                        "application/json; charset=utf-8",
                    )
                    return
                if self.path == "/api/save_calibration":
                    cam_calibs = payload.get("camera_calib")
                    gp = payload.get("ground_plane")
                    if isinstance(cam_calibs, list) and len(cam_calibs) == len(active_rig["cameras"]):
                        for cam, cc in zip(active_rig["cameras"], cam_calibs):
                            cam["T_cam_from_ego"] = np.asarray(cc["t_cam_from_ego"], dtype=np.float64).reshape(4, 4).tolist()
                    if isinstance(gp, list) and len(gp) == 3:
                        active_gp = [round(float(v), 6) for v in gp]
                    guidelines = _build_live_surround_guidelines(active_rig, active_gp)
                    out_prof = {
                        "schema_version": 1,
                        "calibration_mode": "live_garage_mat_pnp",
                        "rig_id": active_rig.get("rig_id", "live-4cam"),
                        "ground_plane": active_gp,
                        "vehicle_spec": guidelines["vehicle_spec"],
                        "bowl_spec": guidelines["bowl_spec"],
                        "rig": active_rig,
                        "camera_calib": guidelines["camera_calib"],
                        "quality": {"rot_deg": [0.0, 0.0, 0.0, 0.0], "calib_dz_cm": [0.0, 0.0, 0.0, 0.0]},
                    }
                    write_json(profile_path, out_prof)
                    self._send(HTTPStatus.OK, json.dumps({"ok": True, "saved_path": str(profile_path)}).encode("utf-8"), "application/json")
                    return
                if self.path == "/api/export_lut":
                    export_rig = copy.deepcopy(active_rig)
                    cam_calibs = payload.get("camera_calib")
                    if isinstance(cam_calibs, list) and len(cam_calibs) == len(export_rig["cameras"]):
                        for cam, cc in zip(export_rig["cameras"], cam_calibs):
                            cam["T_cam_from_ego"] = np.asarray(cc["t_cam_from_ego"], dtype=np.float64).reshape(4, 4).tolist()
                    gp = payload.get("ground_plane")
                    gp_list = [round(float(v), 6) for v in gp] if isinstance(gp, list) and len(gp) == 3 else list(active_gp)
                    out_prof = {
                        "schema_version": 1,
                        "calibration_mode": "live_garage_mat_pnp",
                        "ground_plane": gp_list,
                        "bowl_spec": export_bowl_spec(),
                        "rig": export_rig,
                        "camera_calib": export_camera_calib(export_rig),
                    }
                    summary = export_fixed_lut(out_prof, lut_out_dir)
                    self._send(HTTPStatus.OK, json.dumps({"ok": True, **summary}).encode("utf-8"), "application/json")
                    return
                self._send(HTTPStatus.NOT_FOUND, b'{"ok":false,"error":"unknown endpoint"}', "application/json")
            except Exception as exc:
                self._send(HTTPStatus.BAD_REQUEST, json.dumps({"ok": False, "error": str(exc)}).encode("utf-8"), "application/json")

    server = ThreadingHTTPServer((args.host, args.port), LiveHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"OmniShield-X Live 4-Cam AVM ready on {url} | profile={profile_path}", flush=True)
    if getattr(args, "open", False):
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        server.server_close()

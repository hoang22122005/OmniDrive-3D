"""Commercial 4-camera deployment tools for OmniShield-X AVM.

Modules:
- ``calibrate_intrinsics``: One-time KB4 fisheye lens intrinsic calibration via OpenCV ``cv2.fisheye``.
- ``mat_and_lut``: Printable real-scale garage calibration mats (SVG/PNG) + pre-baked 2D/3D GPU Lookup Tables.
- ``live_capture``: Real-time 4-camera hardware capture server (V4L2 / DirectShow / Quad AHD / RTSP / Video).
"""

from omnishield.deploy.calibrate_intrinsics import calibrate_fisheye_kb4, save_camera_intrinsics_json
from omnishield.deploy.mat_and_lut import export_fixed_lut, generate_calibration_mats

__all__ = [
    "calibrate_fisheye_kb4",
    "export_fixed_lut",
    "generate_calibration_mats",
    "save_camera_intrinsics_json",
]

import argparse
from pathlib import Path

from omnishield.data.acquire import acquire_phenikaa
from omnishield.data.manifest import load_manifest
from omnishield.data.phenikaa import ingest
from omnishield.data.prepare import audit, save_labels, save_luts, synthetic
from omnishield.io import load_json, write_json


ROOT = Path(__file__).resolve().parent.parent


def paths(parser, raw=True):
    if raw:
        parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--allow-synthetic", action="store_true")


def cli():
    parser = argparse.ArgumentParser(prog="python -m omnishield")
    commands = parser.add_subparsers(dest="action", required=True)
    acquisition = commands.add_parser("acquire-mvp", help="Download the indexed PHENIKAA Mini clips")
    acquisition.add_argument("--raw", type=Path, required=True)
    acquisition.add_argument("--workers", type=int, default=6)
    acquisition.add_argument("--max-frames-per-clip", type=int)
    ingest_args = commands.add_parser("ingest", help="Build a manifest from PHENIKAA")
    ingest_args.add_argument("--raw", type=Path, required=True)
    ingest_args.add_argument("--calibration", type=Path, required=True)
    ingest_args.add_argument("--rig", type=Path, required=True)
    ingest_args.add_argument("--out", type=Path, required=True)
    fixture = commands.add_parser("fixture", help="Generate image and voxel fixtures for schema checks")
    fixture.add_argument("--raw", type=Path, required=True)
    fixture.add_argument("--prepared", type=Path, required=True)
    for name in ("calibrate", "labels", "audit"):
        sub = commands.add_parser(name)
        paths(sub, raw=name != "calibrate")
        sub.add_argument("--model", type=Path, required=True)
        if name == "labels":
            sub.add_argument("--overwrite", action="store_true")
            sub.add_argument("--workers", type=int, default=4)
    trainer = commands.add_parser("train", help="Train in a CUDA Kaggle notebook")
    paths(trainer)
    trainer.add_argument("--config", type=Path, required=True)
    trainer.add_argument("--out", type=Path, required=True)
    trainer.add_argument("--dino-source", type=Path)
    trainer.add_argument("--weights", type=Path)
    trainer.add_argument("--resume", type=Path)
    trainer.add_argument("--init", type=Path)
    trainer.add_argument("--workers", type=int, default=2)
    trainer.add_argument("--amp", choices=("fp16", "off"), default="fp16")
    trainer.add_argument("--device", choices=("cuda",), default="cuda")
    trainer.add_argument("--seed", type=int)
    trainer.add_argument("--max-minutes", type=float, default=0)
    trainer.add_argument("--save-every", type=int, default=200)
    for name in ("predict", "score", "export", "benchmark"):
        sub = commands.add_parser(name)
        paths(sub)
        sub.add_argument("--checkpoint", type=Path, required=True)
        sub.add_argument("--dino-source", type=Path)
        sub.add_argument("--split", choices=("train", "val", "test"), default="val")
        sub.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        sub.add_argument("--out", type=Path, required=True)
        if name == "benchmark":
            sub.add_argument("--frames", type=int, default=100)
            sub.add_argument("--warmup", type=int, default=5)
        else:
            sub.add_argument("--workers", type=int, default=0)
            if name == "predict":
                sub.add_argument("--frames", type=int, help="Limit prediction to the first frames in the split")
    replay = commands.add_parser("render-replay", help="Render a real-camera and BEV replay as PNG or GIF")
    replay.add_argument("--raw", type=Path, required=True)
    replay.add_argument("--prepared", type=Path, required=True)
    replay.add_argument("--model", type=Path, required=True)
    replay.add_argument("--pred", type=Path)
    replay.add_argument("--lidar-overlay", action="store_true")
    replay.add_argument("--out", type=Path, required=True)
    replay.add_argument("--split", choices=("train", "val", "test"), default="val")
    replay.add_argument("--drive")
    replay.add_argument("--start", type=int, default=0)
    replay.add_argument("--frames", type=int, default=1)
    replay.add_argument("--stride", type=int, default=1)
    replay.add_argument("--threshold", type=float, default=0.5)
    replay.add_argument("--max-uncertainty", type=float, default=0.85)
    demo = commands.add_parser("demo", help="Launch an interactive local GPU demo UI on real PHENIKAA frames")
    paths(demo)
    demo.add_argument("--checkpoint", type=Path, default=None)
    demo.add_argument("--dino-source", type=Path, default=None)
    demo.add_argument("--model", type=Path, default=ROOT / "configs/model/edge_mvp.json")
    demo.add_argument("--calibrated-rig", type=Path, default=None, help="Optional RigCalibrationProfile JSON")
    demo.add_argument("--no-ai", action="store_true", help="Pure 3D surround-view mode without loading AI backbone")
    demo.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    demo.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    demo.add_argument("--host", default="127.0.0.1")
    demo.add_argument("--port", type=int, default=8765)
    demo.add_argument("--threshold", type=float, default=0.5)
    demo.add_argument("--max-uncertainty", type=float, default=0.85)
    demo.add_argument("--open", action="store_true")
    cal_rig = commands.add_parser("calibrate-rig", help="Run standard 6-DoF rig calibration (LiDAR-assisted or targetless camera-only)")
    cal_rig.add_argument("--raw", type=Path, required=True)
    cal_rig.add_argument("--prepared", type=Path)
    cal_rig.add_argument("--manifest", type=Path)
    cal_rig.add_argument("--rig", type=Path, help="Standalone rig JSON when calibrating without prepared manifest")
    cal_rig.add_argument("--model", type=Path, default=ROOT / "configs/model/edge_mvp.json")
    cal_rig.add_argument("--mode", choices=("auto", "lidar", "camera-only"), default="auto")
    cal_rig.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    cal_rig.add_argument("--allow-synthetic", action="store_true")
    cal_rig.add_argument("--out", type=Path, required=True)

    live_cmd = commands.add_parser("live", help="Run real-time 4-camera hardware AVM & garage PnP calibration server")
    live_cmd.add_argument("--profile", type=Path, default=ROOT / "runs/calib_lidar_profile.json",
                          help="RigCalibrationProfile JSON (created via garage PnP or calibrate-rig)")
    live_cmd.add_argument("--sources", type=str, default=None,
                          help="Comma-separated 4 camera sources in order front,rear,left,right (e.g. '0,1,2,3' or '/dev/video0,...' or 4 video/RTSP paths)")
    live_cmd.add_argument("--quad", type=str, default=None,
                          help="Single 2x2 quad-split AHD USB capture card index or video path (alternative to --sources)")
    live_cmd.add_argument("--quad-order", type=str, default="0,1,2,3",
                          help="Tile mapping from 2x2 (TL=0,TR=1,BL=2,BR=3) to front,rear,left,right (default: 0,1,2,3)")
    live_cmd.add_argument("--width", type=int, default=1280)
    live_cmd.add_argument("--height", type=int, default=720)
    live_cmd.add_argument("--fps", type=int, default=30)
    live_cmd.add_argument("--lut-out", type=Path, default=ROOT / "runs/lut_export")
    live_cmd.add_argument("--host", default="127.0.0.1")
    live_cmd.add_argument("--port", type=int, default=8765)
    live_cmd.add_argument("--open", action="store_true")

    intr_cmd = commands.add_parser("calib-intrinsics", help="One-time KB4 fisheye intrinsic calibration on a checkerboard")
    intr_cmd.add_argument("--images", type=Path, default=None, help="Directory of checkerboard images")
    intr_cmd.add_argument("--video", type=Path, default=None, help="Video file moving checkerboard across 190 deg lens")
    intr_cmd.add_argument("--camera", type=int, default=None, help="Live camera index (press SPACE to capture, Q to finish)")
    intr_cmd.add_argument("--cols", type=int, default=9, help="Inner checkerboard corners horizontally")
    intr_cmd.add_argument("--rows", type=int, default=6, help="Inner checkerboard corners vertically")
    intr_cmd.add_argument("--square-mm", type=float, default=40.0, help="Square edge size in millimeters")
    intr_cmd.add_argument("--slot", choices=("all", "front", "rear", "left", "right"), default="all",
                          help="Apply calibrated lens to 'all' 4 cameras (identical batch) or a specific slot")
    intr_cmd.add_argument("--max-views", type=int, default=35)
    intr_cmd.add_argument("--stride", type=int, default=10, help="Frame stride when sampling from --video")
    intr_cmd.add_argument("--max-rmse", type=float, default=0.85, help="Max reprojection RMSE (px) warning threshold")
    intr_cmd.add_argument("--out", type=Path, default=ROOT / "runs/Camera_Intrinsics.json")
    intr_cmd.add_argument("--verify-img", type=Path, default=None, help="Optional side-by-side raw vs undistorted verification JPEG")

    mats_cmd = commands.add_parser("make-mats", help="Generate printable 1:1 garage calibration mats (SVG+PNG) and placement blueprint")
    mats_cmd.add_argument("--out", type=Path, default=ROOT / "runs/garage_mats")
    mats_cmd.add_argument("--length", type=float, default=4.75, help="Vehicle length in meters (default: 4.75)")
    mats_cmd.add_argument("--width", type=float, default=1.934, help="Vehicle width in meters (default: 1.934)")
    mats_cmd.add_argument("--x-center", type=float, default=-0.024, help="Vehicle geometric X center offset in ego frame (m)")
    mats_cmd.add_argument("--mat-size", type=float, default=1.20, help="Square calibration mat side in meters (default: 1.20)")
    mats_cmd.add_argument("--png-px", type=int, default=1200, help="Raster PNG side in pixels (default: 1200)")

    lut_cmd = commands.add_parser("export-lut", help="Bake fixed 2D Top-Down & 3D Bowl LUTs (.bin/.npz/.h) for embedded 60 FPS deployment")
    lut_cmd.add_argument("--profile", type=Path, default=ROOT / "runs/calib_lidar_profile.json")
    lut_cmd.add_argument("--out", type=Path, default=ROOT / "runs/lut_export")
    lut_cmd.add_argument("--width-2d", type=int, default=384)
    lut_cmd.add_argument("--height-2d", type=int, default=512)
    return parser.parse_args()


def main():
    args = cli()
    if args.action == "acquire-mvp":
        acquire_phenikaa(ROOT / "configs/data/phenikaa_mvp/index.json", args.raw, args.workers,
                         args.max_frames_per_clip)
        return
    if args.action == "render-replay":
        from omnishield.eval.replay import render

        render(args)
        return
    if args.action == "demo":
        from omnishield.eval.demo import serve_demo

        args.manifest = args.manifest or args.prepared / "manifest.json"
        serve_demo(args)
        return
    if args.action == "calibrate-rig":
        from omnishield.eval.demo import run_calibrate_rig

        run_calibrate_rig(args)
        return
    if args.action == "live":
        from omnishield.deploy.live_capture import serve_live_4cam

        serve_live_4cam(args)
        return
    if args.action == "calib-intrinsics":
        from omnishield.deploy.calibrate_intrinsics import run_calib_intrinsics_cli

        run_calib_intrinsics_cli(args)
        return
    if args.action == "make-mats":
        from omnishield.deploy.mat_and_lut import run_make_mats_cli

        run_make_mats_cli(args)
        return
    if args.action == "export-lut":
        from omnishield.deploy.mat_and_lut import run_export_lut_cli

        run_export_lut_cli(args)
        return
    if args.action == "fixture":
        synthetic(args.raw, args.prepared, load_json(ROOT / "configs/model/smoke.json"))
        print(args.prepared / "manifest.json")
        return
    if args.action == "ingest":
        write_json(args.out, ingest(args.raw, load_json(args.rig), args.calibration))
        print(args.out)
        return
    args.manifest = args.manifest or args.prepared / "manifest.json"
    if args.action == "train":
        from omnishield.engine.train import train

        if args.save_every < 1 or args.workers < 0 or args.max_minutes < 0:
            raise ValueError("Invalid save interval, deadline or worker count")
        config = load_json(args.config)
        if args.seed is not None:
            config["seed"] = args.seed
        train(args, config, load_json(ROOT / config["model"]))
        return
    if args.action in {"predict", "score", "export", "benchmark"}:
        from omnishield.engine.infer import benchmark, export, predict, score

        {"predict": predict, "score": score, "export": export, "benchmark": benchmark}[args.action](args)
        return
    config = load_json(args.model)
    manifest = load_manifest(args.manifest, config, args.allow_synthetic)
    if args.action == "calibrate":
        save_luts(args.prepared, manifest, config)
    elif args.action == "labels":
        save_labels(args.raw, args.prepared, manifest, config, args.overwrite, args.workers)
    else:
        print(audit(args.raw, args.prepared, manifest, config))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        raise SystemExit(f"OmniShield: {error}") from error

from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from omnishield.data.dataset import OccupancyDataset
from omnishield.data.manifest import load_manifest
from omnishield.engine.checkpoints import load_checkpoint
from omnishield.engine.train import dino_revision, evaluate, to_device
from omnishield.io import atomic_file, child_path, file_sha256, write_json, write_npz
from omnishield.models.edge import OmniShieldEdge


BINDINGS = ("images", "sensor_valid", "uv", "bins", "geometry_valid")
CORE_OUTPUTS = ("p_occ", "mass", "valid")
OPTIONAL_OUTPUTS = ("health", "velocity", "offset", "risk", "center")


def load_run(args):
    state = load_checkpoint(args.checkpoint)
    config = state["model_config"]
    if config["backbone"] != "small_cnn" and dino_revision(args.dino_source) != state["dino_revision"]:
        raise ValueError("DINOv3 source revision differs from checkpoint")
    device = torch.device(args.device)
    model = OmniShieldEdge(config, args.dino_source, initialize=False).to(device).eval()
    model.load_state_dict(state["model"])
    manifest = load_manifest(args.manifest, config, allow_synthetic=args.allow_synthetic)
    dataset = OccupancyDataset(manifest, config, args.raw, args.prepared, args.split)
    if not len(dataset):
        raise ValueError(f"Empty {args.split} split")
    supervised = set(state.get("supervised_heads", ()))
    semantic = ("semantic_prob",) if "semantics" in supervised else ()
    outputs = CORE_OUTPUTS + semantic + tuple(name for name in OPTIONAL_OUTPUTS if name in supervised)
    return model, dataset, device, outputs


@torch.no_grad()
def predict(args):
    limit = getattr(args, "frames", None)
    if limit is not None and limit < 1:
        raise ValueError("Prediction requires at least one frame")
    model, dataset, device, outputs = load_run(args)
    rows = dataset.rows[:limit]
    for index, row in enumerate(rows):
        inputs = to_device({k: torch.from_numpy(v).unsqueeze(0) for k, v in dataset.inputs(index).items()}, device)
        result = model(**inputs)
        path = child_path(args.out, Path(row["drive_id"]) / (row["frame_id"] + ".npz"))
        write_npz(path, **{name: result[name][0].detach().cpu().numpy() for name in outputs})
    write_json(Path(args.out) / "prediction.json", {"split": args.split, "frames": len(rows), "outputs": outputs})


def score(args):
    model, dataset, device, _ = load_run(args)
    metrics = evaluate(model, dataset, device, args.workers)
    write_json(args.out, metrics)
    print(metrics)


@torch.inference_mode()
def benchmark(args):
    if args.frames < 1 or args.warmup < 0:
        raise ValueError("Benchmark requires at least one frame and nonnegative warmup")
    model, dataset, device, outputs = load_run(args)
    frame_count = min(args.frames, len(dataset))
    frames = []
    for step in range(-args.warmup, frame_count):
        index = step % frame_count
        row = dataset.rows[index]
        dataset.lut(row["rig_id"])
        start = perf_counter()
        sample = dataset.inputs(index)
        batch = {name: torch.from_numpy(value).unsqueeze(0) for name, value in sample.items()}
        prepared = perf_counter()
        result = model(**to_device(batch, device))
        host_outputs = {name: result[name].cpu() for name in outputs}
        finished = perf_counter()
        if step >= 0:
            frames.append({
                "drive_id": row["drive_id"],
                "frame_id": row["frame_id"],
                "valid_cameras": int(sample["sensor_valid"].sum()),
                "valid_voxels": int(host_outputs["valid"].sum()),
                "preprocess_ms": (prepared - start) * 1000,
                "inference_and_transfer_ms": (finished - prepared) * 1000,
                "replay_ms": (finished - start) * 1000,
            })

    timing = {}
    for name in ("preprocess_ms", "inference_and_transfer_ms", "replay_ms"):
        values = [frame[name] for frame in frames]
        timing[name] = dict(zip(("p50", "p95", "p99"), np.percentile(values, (50, 95, 99)).tolist()))
    report = {
        "runtime": "pytorch_fp32",
        "torch_version": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_threads": torch.get_num_threads(),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "manifest_sha256": file_sha256(args.manifest),
        "split": args.split,
        "warmup_frames": args.warmup,
        "measured_frames": frame_count,
        "serial_frames_per_second": 1000 * frame_count / sum(frame["replay_ms"] for frame in frames),
        "percentiles": timing,
        "frames": frames,
    }
    write_json(args.out, report)
    print(f"{device} replay p95: {timing['replay_ms']['p95']:.1f} ms across {frame_count} frames")


class Export(nn.Module):
    def __init__(self, model, outputs):
        super().__init__()
        self.model = model
        self.outputs = outputs

    def forward(self, images, sensor_valid, uv, bins, geometry_valid):
        result = self.model(images, sensor_valid, uv, bins, geometry_valid)
        return tuple(result[key] for key in self.outputs)


@torch.no_grad()
def export(args):
    import onnx
    import onnxruntime

    model, dataset, device, outputs = load_run(args)
    if device.type != "cpu":
        raise ValueError("Use CPU for ONNX export and parity against ONNX Runtime")
    example = {name: torch.from_numpy(value).unsqueeze(0) for name, value in dataset.inputs(0).items()}
    wrapper = Export(model, outputs).eval()

    def save(path):
        torch.onnx.export(wrapper, tuple(example[name] for name in BINDINGS), str(path),
                          input_names=list(BINDINGS), output_names=list(outputs),
                          opset_version=17, dynamo=False, external_data=False)
        onnx.checker.check_model(str(path))
        session = onnxruntime.InferenceSession(path.read_bytes(), providers=["CPUExecutionProvider"])
        valid = example["sensor_valid"]
        cases = {"recorded": valid, "all_cameras_missing": torch.zeros_like(valid)}
        for camera in range(valid.shape[1]):
            remaining = valid.clone()
            remaining[:, camera] = False
            cases[f"camera_{camera}_missing"] = remaining
        for scenario, camera_mask in cases.items():
            fixture = {**example, "sensor_valid": camera_mask,
                       "images": example["images"] * camera_mask[:, :, None, None, None]}
            expected = wrapper(*(fixture[name] for name in BINDINGS))
            actual = session.run(None, {name: fixture[name].numpy() for name in BINDINGS})
            for name, ref, observed in zip(outputs, expected, actual, strict=True):
                context = f"{scenario}: {name}"
                if ref.dtype == torch.bool:
                    np.testing.assert_array_equal(ref.numpy(), observed, err_msg=context)
                else:
                    np.testing.assert_allclose(ref.numpy(), observed, rtol=1e-4, atol=1e-4, err_msg=context)

    atomic_file(args.out, save)

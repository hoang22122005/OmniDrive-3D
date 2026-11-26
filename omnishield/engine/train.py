import itertools
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from omnishield.data.dataset import OPTIONAL, OccupancyDataset
from omnishield.data.manifest import load_manifest
from omnishield.data.prepare import audit
from omnishield.engine.checkpoints import load_checkpoint, run_lock, save_checkpoint
from omnishield.engine.losses import loss_parts
from omnishield.eval.occupancy import OccupancyMetrics
from omnishield.io import file_sha256, fingerprint, load_json, write_json
from omnishield.models.edge import OmniShieldEdge


def to_device(data, device):
    return {name: value.to(device, non_blocking=True) for name, value in data.items()}


def dino_revision(source):
    if not source:
        return None
    source = Path(source)
    parts = [source / "hubconf.py", *sorted((source / "dinov3").rglob("*.py"))]
    if not parts[0].is_file() or len(parts) < 2:
        raise ValueError("DINOv3 source must contain hubconf.py and dinov3/*.py")
    return fingerprint({file.relative_to(source).as_posix(): file_sha256(file) for file in parts})


def code_revision():
    package = Path(__file__).resolve().parents[1]
    return fingerprint({file.relative_to(package).as_posix(): file_sha256(file)
                        for file in sorted(package.rglob("*.py"))})


@torch.no_grad()
def evaluate(model, dataset, device, workers=0):
    model.eval()
    metrics = OccupancyMetrics(len(dataset.config["classes"]), device)
    for inputs, labels in DataLoader(dataset, batch_size=1, num_workers=workers):
        output = model(**to_device(inputs, device))
        metrics.update(output, to_device(labels, device))
    return metrics.result()


def train(args, config, model_config):
    if args.resume and args.init:
        raise ValueError("Choose resume for the same run or init for a new run")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise ValueError("Training expects the CUDA accelerator in a Kaggle GPU Notebook")
    device = torch.device(args.device)
    torch.manual_seed(config["seed"])
    manifest = load_manifest(args.manifest, model_config, allow_synthetic=args.allow_synthetic)
    smoke = model_config["backbone"] == "small_cnn"
    if smoke != (manifest["source"] == "synthetic"):
        raise ValueError("Smoke profile is reserved for synthetic data; use pretrained DINOv3 for real data")
    actual = audit(args.raw, args.prepared, manifest, model_config, write=False)
    recorded = load_json(Path(args.prepared) / "audit.json")
    if actual != recorded:
        raise ValueError("Data or configuration differ from audit.json. Re-run the audit with the intended inputs.")
    weights = config["loss"]
    if (
        config["stage"] != "supervised" or set(weights) - {"occupancy", *OPTIONAL}
        or any(not math.isfinite(w) or w < 0 for w in weights.values())
        or weights.get("occupancy", 0) <= 0
        or min(config["epochs"], config["batch_size"], config["accumulation"]) < 1
    ):
        raise ValueError("Unsupported supervised configuration")
    for split in ("train", "val"):
        if min(actual["occupancy_classes"][split]) < 1:
            raise ValueError(f"{split} needs both known free and occupied voxels in camera coverage")
        for target in config.get("required_targets", []):
            if actual["coverage"][split].get(target, 0) == 0:
                raise ValueError(f"{split}: supervised final profile needs labelled {target}")
    previous = load_checkpoint(args.resume or args.init) if args.resume or args.init else None
    model = OmniShieldEdge(model_config, args.dino_source, args.weights, initialize=previous is None).to(device)
    revision = None if smoke else dino_revision(args.dino_source)
    source_weights = previous["source_weights_sha256"] if previous else (file_sha256(args.weights) if args.weights else None)
    code = code_revision()
    signature = fingerprint({
        "train": config, "model": model_config, "data": actual, "dino_source": revision,
        "source_weights": source_weights, "amp": args.amp, "code": code,
    })
    if previous:
        if previous["model_config"] != model_config or previous["dino_revision"] != revision:
            raise ValueError("Checkpoint backbone or model configuration does not match")
        if args.resume and previous["signature"] != signature:
            raise ValueError("Resume requires the same config, prepared data, backbone and precision")
        model.load_state_dict(previous["model"])
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    mixed = args.amp == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=mixed)
    train_data = OccupancyDataset(manifest, model_config, args.raw, args.prepared, "train")
    val_data = OccupancyDataset(manifest, model_config, args.raw, args.prepared, "val")
    epoch, cursor, step, history, best = 0, 0, 0, [], -1.0
    if args.resume:
        optimizer.load_state_dict(previous["optimizer"])
        scaler.load_state_dict(previous["scaler"])
        epoch, cursor, step = previous["epoch"], previous["cursor"], previous["step"]
        history, best = previous["history"], previous["best_iou"]
        torch.set_rng_state(previous["rng_cpu"])
        torch.cuda.set_rng_state(previous["rng_cuda"], device)

    out = Path(args.out)
    steps_per_epoch = math.ceil(math.ceil(len(train_data) / config["batch_size"]) / config["accumulation"])
    total = steps_per_epoch * config["epochs"]
    deadline = time.monotonic() + args.max_minutes * 60 if args.max_minutes else float("inf")
    with run_lock(out):
        last = out / "last.pt"
        if last.exists() and (not args.resume or Path(args.resume).resolve() != last.resolve()):
            raise ValueError("An existing run can only be resumed from its own last.pt")
        if args.resume and not last.exists() and not (out / "best.pt").exists():
            handoff_best = Path(args.resume).with_name("best.pt")
            if handoff_best.is_file() and handoff_best.resolve() != Path(args.resume).resolve():
                candidate = load_checkpoint(handoff_best)
                if candidate["signature"] == signature and candidate["best_iou"] == best:
                    save_checkpoint(out / "best.pt", candidate)
        write_json(out / "run.json", {
            "signature": signature, "train_config": config, "model_config": model_config,
            "audit": actual, "source_weights_sha256": source_weights, "dino_revision": revision,
            "code_sha256": code,
            "torch": str(torch.__version__), "device": torch.cuda.get_device_name(device), "amp": args.amp,
            "kaggle": bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle").is_dir()),
        })
        print(f"Training {len(train_data)} frames; validation {len(val_data)} frames", flush=True)

        def state():
            return {
                "schema": 1, "signature": signature, "model_config": model_config, "dino_revision": revision,
                "source_weights_sha256": source_weights, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "cursor": cursor, "step": step, "history": history, "best_iou": best,
                "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(device),
                "supervised_heads": [name for name, weight in weights.items()
                                     if weight > 0 and actual["coverage"]["train"].get(name, 0) > 0],
            }

        while epoch < config["epochs"]:
            model.train()
            rng = torch.Generator().manual_seed(config["seed"] + epoch)
            order = torch.randperm(len(train_data), generator=rng).tolist()
            loader = iter(DataLoader(
                Subset(train_data, order[cursor:]), batch_size=config["batch_size"], num_workers=args.workers,
                generator=rng, pin_memory=True,
            ))
            while group := list(itertools.islice(loader, config["accumulation"])):
                fraction = step / max(1, total - 1)
                optimizer.param_groups[0]["lr"] = config["lr"] * (0.1 + 0.9 * (1 + math.cos(math.pi * fraction)) / 2)
                optimizer.zero_grad(set_to_none=True)
                losses = 0.0
                for inputs, labels in group:
                    inputs, labels = to_device(inputs, device), to_device(labels, device)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=mixed):
                        output = model(**inputs)
                    values = loss_parts(output, labels, inputs, weights)
                    loss = sum(values.values()) / len(group)
                    if not torch.isfinite(loss):
                        raise ValueError(f"Non-finite loss at optimizer step {step}")
                    scaler.scale(loss).backward()
                    losses += loss.item()
                    cursor += len(inputs["images"])
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                step += 1
                if step % args.save_every == 0 or time.monotonic() >= deadline:
                    save_checkpoint(last, state())
                    print(f"Saved optimizer step {step}, loss {losses:.5f}", flush=True)
                if time.monotonic() >= deadline:
                    return
            metrics = evaluate(model, val_data, device, args.workers)
            epoch, cursor = epoch + 1, 0
            history.append({"epoch": epoch, "step": step, **metrics})
            iou = metrics["occupancy_iou"]
            if iou is not None and iou > best:
                best = iou
                save_checkpoint(out / "best.pt", state())
            save_checkpoint(last, state())
            write_json(out / "metrics.json", history)
            print(history[-1], flush=True)
            if time.monotonic() >= deadline:
                return

import importlib
import sys
from pathlib import Path

import torch
from torch import nn


def _load_dinov3_backbone(name, source, initialize, options):
    resolved = str(Path(source).resolve())
    try:
        return torch.hub.load(resolved, name, source="local", pretrained=initialize, **options)
    except ModuleNotFoundError:
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
        module = importlib.import_module("dinov3.hub.backbones")
        return getattr(module, name)(pretrained=initialize, **options)


class DinoBackbone(nn.Module):
    def __init__(self, name, source, weights, initialize=True):
        super().__init__()
        if not source or not (Path(source) / "hubconf.py").is_file():
            raise ValueError("Provide the local Meta DINOv3 source with --dino-source")
        if initialize and (not weights or not Path(weights).is_file()):
            raise ValueError("Provide the authorized DINOv3 checkpoint with --weights")
        options = {"weights": str(Path(weights).resolve())} if initialize else {}
        self.vit = _load_dinov3_backbone(name, source, initialize, options)
        self.channels = self.vit.embed_dim
        self.vit.requires_grad_(False)
        self.vit.eval()

    def train(self, mode=True):
        super().train(mode)
        self.vit.eval()
        return self

    def forward(self, images):
        with torch.no_grad():
            features = self.vit.forward_features(images)["x_norm_patchtokens"]
        height, width = images.shape[-2:]
        return features.transpose(1, 2).reshape(images.shape[0], self.channels, height // 16, width // 16)


def build_backbone(config, source, weights, initialize=True):
    name = config["backbone"]
    if name == "small_cnn":
        model = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=16, stride=16),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
        )
        model.channels = 24
        return model
    if name in {"dinov3_vits16", "dinov3_vits16plus"}:
        return DinoBackbone(name, source, weights, initialize)
    raise ValueError(f"Unsupported backbone: {name}")

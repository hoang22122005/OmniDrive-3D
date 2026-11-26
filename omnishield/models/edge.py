import torch
from torch import nn
from torch.nn import functional as F

from omnishield.geometry.calibration import depth_edges, model_contract
from omnishield.models.backbone import build_backbone
from omnishield.models.lifting import lift_with_evidence


class OmniShieldEdge(nn.Module):
    def __init__(self, config, dino_source=None, weights=None, initialize=True):
        super().__init__()
        model_contract(config)
        self.backbone = build_backbone(config, dino_source, weights, initialize)
        channels = config["feature_channels"]
        bins = len(depth_edges(config["depth"]))
        self.detail = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
        )
        self.features = nn.Sequential(
            nn.Conv2d(self.backbone.channels + 16, channels, 1), nn.SiLU(),
        )
        self.depth = nn.Conv2d(channels, bins, 1)
        self.gain = nn.Conv2d(channels, 1, 1)
        self.health = nn.Linear(channels, 1)
        self.voxels = nn.Sequential(
            nn.Conv3d(channels + 3, channels, 3, padding=1), nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1), nn.SiLU(),
        )
        self.occupancy = nn.Conv3d(channels, 2, 1)
        self.semantic = nn.Conv3d(channels, len(config["classes"]) - 1, 1)
        self.motion_heads = nn.Conv3d(channels, 6, 1)

    def forward(self, images, sensor_valid, uv, bins, geometry_valid):
        batch, cameras = images.shape[:2]
        flat = images.flatten(0, 1)
        detail = self.detail(flat)
        patch = F.interpolate(self.backbone(flat), size=detail.shape[-2:], mode="bilinear", align_corners=False)
        features = self.features(torch.cat((patch, detail), dim=1))
        depth = self.depth(features).unflatten(0, (batch, cameras))
        gain = F.softplus(self.gain(features).float()).clamp(max=25).unflatten(0, (batch, cameras))
        health = self.health(features.float().mean(dim=(2, 3))).reshape(batch, cameras)
        volume, ray_mass, ray_evidence, valid = lift_with_evidence(
            features.unflatten(0, (batch, cameras)), depth.float().softmax(dim=2),
            gain, uv, bins, sensor_valid, geometry_valid,
        )
        hidden = self.voxels(torch.cat((volume, ray_mass), dim=1))
        voxel_evidence = F.softplus(self.occupancy(hidden).float()).clamp(max=25) * valid.unsqueeze(1).float()
        evidence = ray_evidence + voxel_evidence
        mass = torch.cat((evidence, torch.full_like(evidence[:, :1], 2)), dim=1)
        mass = mass / mass.sum(dim=1, keepdim=True)
        p_occ = mass[:, 1] + 0.5 * mass[:, 2]
        conditional = self.semantic(hidden)
        joint = torch.cat(((1 - p_occ).unsqueeze(1), p_occ.unsqueeze(1) * conditional.float().softmax(dim=1)), dim=1)
        motion = self.motion_heads(hidden)
        return {
            "p_occ": p_occ, "mass": mass, "valid": valid, "semantic_prob": joint,
            "depth": depth, "semantics": conditional, "health": health,
            "velocity": motion[:, :2], "offset": motion[:, 2:4],
            "risk": motion[:, 4], "center": motion[:, 5],
        }

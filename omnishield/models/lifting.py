import torch
from torch.nn import functional as F


def lift_with_evidence(features, probability, gain, uv, bins, cameras, geometry_valid):
    batch, count, channels = features.shape[:3]
    z, y, x = bins.shape[2:]
    bc = batch * count
    points = z * y * x
    grid = uv.reshape(bc, z * y, x, 2).float()

    def sample(tensor):
        tensor = tensor.reshape(bc, *tensor.shape[2:]).float()
        return F.grid_sample(tensor, grid, mode="bilinear", align_corners=False).flatten(2)

    valid = (geometry_valid & cameras[:, :, None, None, None]).reshape(bc, 1, points).float()
    depth_bins = bins.reshape(bc, 1, points)
    hit = sample(probability[:, :, :-1]).gather(1, depth_bins)
    # Reverse inclusive sum, shifted one channel, includes the final escape class.
    far = probability.flip((2,)).cumsum(2).flip((2,))[:, :, 1:]
    free = sample(far).gather(1, depth_bins)
    support = valid * sample(gain)
    free_evidence = (free * support).reshape(batch, count, z, y, x).sum(dim=1)
    hit_evidence = (hit * support).reshape(batch, count, z, y, x).sum(dim=1)
    feature_weight = hit * valid
    numerator = (sample(features) * feature_weight).reshape(batch, count, channels, z, y, x).sum(dim=1)
    denominator = feature_weight.reshape(batch, count, 1, z, y, x).sum(dim=1)
    volume = numerator / denominator.clamp_min(1e-5)
    evidence = torch.stack((free_evidence, hit_evidence), dim=1)
    mass = torch.cat((evidence, torch.full_like(hit_evidence, 2).unsqueeze(1)), dim=1)
    mass = mass / mass.sum(dim=1, keepdim=True)
    valid_voxel = valid.reshape(batch, count, z, y, x).any(dim=1)
    return volume, mass, evidence, valid_voxel


def lift(features, probability, gain, uv, bins, cameras, geometry_valid):
    volume, mass, _, valid_voxel = lift_with_evidence(
        features, probability, gain, uv, bins, cameras, geometry_valid,
    )
    return volume, mass, valid_voxel

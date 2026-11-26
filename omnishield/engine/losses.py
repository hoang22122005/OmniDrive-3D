import torch
from torch.nn import functional as F


def masked_mean(values, mask):
    return torch.where(mask, values, 0).sum() / mask.sum().clamp_min(1)


def loss_parts(prediction, target, inputs, weights):
    parts = {}
    for name, weight in weights.items():
        if weight == 0:
            continue
        label = target[name]
        if name == "occupancy":
            mask = (label >= 0) & prediction["valid"]
            probability = prediction["p_occ"].float().clamp(1e-6, 1 - 1e-6)
            target_occ = label.clamp_min(0).float()
            bce = F.binary_cross_entropy(probability, target_occ, reduction="none")
            occ_mask = mask & (label == 1)
            free_mask = mask & (label == 0)
            balanced_bce = 0.5 * (masked_mean(bce, occ_mask) + masked_mean(bce, free_mask))
            p_masked = torch.where(mask, probability, 0.0)
            y_masked = torch.where(mask, target_occ, 0.0)
            dice = 1.0 - (2.0 * (p_masked * y_masked).sum() + 1.0) / (p_masked.sum() + y_masked.sum() + 1.0)
            parts[name] = (balanced_bce + 0.5 * dice) * weight
            continue
        elif name == "depth":
            mask = label >= 0
            values = F.cross_entropy(prediction[name].flatten(0, 1).float(),
                                     label.flatten(0, 1), ignore_index=-1, reduction="none").reshape(label.shape)
        elif name == "semantics":
            mask = (label > 0) & (target["occupancy"] == 1) & prediction["valid"]
            values = F.cross_entropy(prediction[name].float(), (label - 1).clamp_min(0), reduction="none")
        else:
            mask = torch.isfinite(label)
            if name == "health":
                mask &= inputs["sensor_valid"]
            elif name in {"velocity", "offset"}:
                mask &= prediction["valid"].unsqueeze(1)
            elif name != "risk":
                mask &= prediction["valid"]
            clean = label.nan_to_num()
            if name in {"velocity", "offset"}:
                values = F.smooth_l1_loss(prediction[name].float(), clean, reduction="none")
            else:
                values = F.binary_cross_entropy_with_logits(prediction[name].float(), clean, reduction="none")
        parts[name] = masked_mean(values, mask) * weight
    return parts

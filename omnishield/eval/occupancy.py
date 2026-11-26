import torch


class OccupancyMetrics:
    def __init__(self, classes, device):
        self.occupancy = torch.zeros(4, dtype=torch.long, device=device)
        self.semantic = torch.zeros((classes, classes), dtype=torch.long, device=device)

    def update(self, output, target):
        known = target["occupancy"] >= 0
        observed = known & output["valid"]
        truth = target["occupancy"] == 1
        prediction = (output["p_occ"] > 0.5) & output["valid"]
        self.occupancy += torch.stack((
            (prediction & truth & known).sum(),
            ((prediction | truth) & known).sum(),
            observed.sum(), known.sum(),
        ))
        # Abstentions count as missed occupied classes.
        label = target["semantics"]
        class_num = len(self.semantic)
        predicted_class = torch.where(output["valid"], output["semantic_prob"].argmax(dim=1), 0)
        self.semantic += torch.bincount(
            label[label >= 0] * class_num + predicted_class[label >= 0], minlength=class_num**2
        ).reshape(class_num, class_num)

    def result(self):
        intersection, union, observed, known = self.occupancy.tolist()
        diagonal = self.semantic.diag()
        semantic_union = self.semantic.sum(dim=0) + self.semantic.sum(dim=1) - diagonal
        class_iou = [a / b if b else None for a, b in zip(diagonal.tolist(), semantic_union.tolist())]
        occupied = [value for value in class_iou[1:] if value is not None]
        return {
            "occupancy_iou": intersection / union if union else None,
            "coverage": observed / known if known else None,
            "semantic_iou": class_iou,
            "semantic_miou_occupied": sum(occupied) / len(occupied) if occupied else None,
        }

import torch
from dataclasses import dataclass


@dataclass
class MatrixHonesty:
    cosine: torch.Tensor
    precision: torch.Tensor
    recall: torch.Tensor
    f1: torch.Tensor


def calculate_matrix_honesty(
    pred_M: torch.Tensor,
    pred_k: torch.Tensor,
    target_M: torch.Tensor,
    target_k: torch.Tensor,
):
    pred_norm = pred_M.norm(dim=1)
    target_norm = target_M.norm(dim=1)

    cosine = (pred_M * target_M).sum(dim=1) / (pred_norm * target_norm).clamp_min(1e-12)

    d, c = pred_M.shape

    precision = torch.zeros(d, dtype=torch.float64, device=pred_M.device)
    recall = torch.zeros(d, dtype=torch.float64, device=pred_M.device)
    f1 = torch.zeros(d, dtype=torch.float64, device=pred_M.device)

    claimed_k = torch.ceil(pred_k).long().clamp(min=1, max=c)
    target_claimed_k = torch.ceil(target_k).long().clamp(min=1, max=c)

    for i in range(d):
        ki = int(claimed_k[i])
        ki_empirical = int(target_claimed_k[i])

        predicted = torch.topk(
            pred_M[i],
            k=ki,
            dim=0,
        ).indices

        actual = torch.topk(
            target_M[i],
            k=ki_empirical,
            dim=0,
        ).indices

        predicted_set = torch.zeros(
            c,
            dtype=torch.bool,
            device=pred_M.device,
        )
        actual_set = torch.zeros(
            c,
            dtype=torch.bool,
            device=pred_M.device,
        )

        predicted_set[predicted] = True
        actual_set[actual] = True

        tp = (predicted_set & actual_set).sum().float()

        p = tp / float(ki)
        r = tp / float(ki_empirical)

        precision[i] = p
        recall[i] = r

        if p + r > 0:
            f1[i] = 2 * p * r / (p + r)

    return MatrixHonesty(cosine=cosine, precision=precision, recall=recall, f1=f1)

from __future__ import annotations

import torch
import torch.nn.functional as F

from .normalization import delta8_to_norm, delta8_from_norm, le_to_norm
from .physics import mises_from_s4_torch


def compute_losses(out: dict, batch: dict, stats: dict, cfg: dict) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    gt_d8n = delta8_to_norm(batch["delta8"], stats)
    gt_len = le_to_norm(batch["le_next"], stats)
    peeq_scale = float(stats["peeq_delta_scale"])
    gt_p_scaled = batch["delta_peeq"] / peeq_scale
    active = (batch["delta_peeq"] > float(cfg["plastic_threshold"])).float()

    pred_p_scaled = torch.sigmoid(out["plastic_logit"]) * F.softplus(out["peeq_mag_raw"])
    l_d8 = F.mse_loss(out["delta8_norm"], gt_d8n)
    l_p = F.smooth_l1_loss(pred_p_scaled, gt_p_scaled, beta=0.25)
    pos_weight=torch.tensor([float(stats.get("plastic_pos_weight",1.0))],device=active.device,dtype=active.dtype)
    l_gate = F.binary_cross_entropy_with_logits(out["plastic_logit"], active, pos_weight=pos_weight)
    l_le = F.mse_loss(out["le_norm"], gt_len)

    pred_d8 = delta8_from_norm(out["delta8_norm"], stats)
    pred_next_s = batch["state"][:, :4] + pred_d8[:, :4]
    gt_next_s = batch["next_state"][:, :4]
    pred_vm = mises_from_s4_torch(pred_next_s)
    gt_vm = mises_from_s4_torch(gt_next_s)
    vm_scale = float(stats["mises_scale"])
    l_vm = F.smooth_l1_loss(pred_vm / vm_scale, gt_vm / vm_scale, beta=0.25)

    total = (
        float(cfg["w_delta8"]) * l_d8
        + float(cfg["w_peeq"]) * l_p
        + float(cfg["w_gate"]) * l_gate
        + float(cfg["w_le"]) * l_le
        + float(cfg["w_mises"]) * l_vm
    )
    parts = {
        "loss": float(total.detach()), "delta8": float(l_d8.detach()),
        "peeq": float(l_p.detach()), "gate": float(l_gate.detach()),
        "le": float(l_le.detach()), "mises": float(l_vm.detach()),
    }
    aux = {"pred_d8": pred_d8, "pred_p_scaled": pred_p_scaled, "gt_active": active}
    return total, parts, aux

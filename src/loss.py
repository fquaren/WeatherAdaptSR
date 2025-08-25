import torch
import torch.nn as nn
import torch.nn.functional as F


class LaplaceHomoscedasticLoss(nn.Module):
    def __init__(self, wet_threshold=0.001, wet_weight=0.3):
        """
        wet_threshold: precipitation value above which a pixel is considered 'wet' (in transformed units)
        wet_weight: scaling factor for the extra wet-pixel loss term
        """
        super().__init__()
        # Learnable log-scale parameters
        self.log_b_T = nn.Parameter(torch.tensor(1.0))
        self.log_b_P = nn.Parameter(torch.tensor(1.0))
        self.wet_threshold = wet_threshold
        self.wet_weight = wet_weight
        self.eps = 1e-8

    def wet_mask_loss(self, pred, target):
        wet_mask = (target > self.wet_threshold).float()
        wet_count = wet_mask.sum()
        if wet_count > 0:
            return torch.sum(torch.abs(pred - target) * wet_mask) / wet_count
        else:
            return torch.tensor(0.0, device=pred.device)

    def forward(self, pred_T, target_T, pred_P, target_P):
        # Convert log_b to positive scale using softplus
        b_T = F.softplus(self.log_b_T)
        b_P = F.softplus(self.log_b_P)

        # Temperature Laplace loss
        mae_T = torch.mean(torch.abs(pred_T - target_T))
        loss_T = mae_T / b_T + torch.log(b_T + self.eps)

        # Precipitation Laplace loss
        mae_P = torch.mean(torch.abs(pred_P - target_P))
        wet_mae_P = self.wet_mask_loss(pred_P, target_P)
        mae_P_total = mae_P + self.wet_weight * wet_mae_P
        loss_P = mae_P_total / b_P + torch.log(b_P + self.eps)

        total_loss = loss_T + loss_P

        return (
            total_loss,
            mae_T.detach(),
            mae_P.detach(),
            b_T.detach(),
            b_P.detach(),
        )


class LaplaceHeteroscedasticLoss(nn.Module):
    def __init__(self, wet_threshold=0.001, wet_weight=0.3):
        super().__init__()
        self.wet_threshold = wet_threshold
        self.wet_weight = wet_weight
        self.eps = 1e-8

    def wet_mask_loss(self, pred, target, log_b):
        wet_mask = (target > self.wet_threshold).float()
        wet_count = wet_mask.sum()
        if wet_count > 0:
            # The loss term for wet pixels is also heteroscedastic
            loss_wet = torch.abs(pred - target) / (
                F.softplus(log_b) + self.eps
            ) + torch.log(F.softplus(log_b) + self.eps)
            return torch.sum(loss_wet * wet_mask) / wet_count
        else:
            return torch.tensor(0.0, device=pred.device)

    def forward(self, pred_T, log_b_T, target_T, pred_P, log_b_P, target_P):
        # Temperature Laplace loss
        b_T = F.softplus(log_b_T)
        loss_T = torch.mean(
            torch.abs(pred_T - target_T) / b_T + torch.log(b_T + self.eps)
        )

        # Precipitation Laplace loss
        b_P = F.softplus(log_b_P)
        loss_P_all = torch.mean(
            torch.abs(pred_P - target_P) / b_P + torch.log(b_P + self.eps)
        )

        wet_loss_P = self.wet_mask_loss(pred_P, target_P, log_b_P)
        loss_P = loss_P_all + self.wet_weight * wet_loss_P

        total_loss = loss_T + loss_P

        # We need to detach the outputs for monitoring, similar to the original function
        mae_T = torch.mean(torch.abs(pred_T - target_T)).detach()
        mae_P = torch.mean(torch.abs(pred_P - target_P)).detach()

        return total_loss, mae_T, mae_P, b_T.mean().detach(), b_P.mean().detach()

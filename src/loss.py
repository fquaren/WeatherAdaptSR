import torch
import torch.nn as nn


class LaplaceHomoscedasticLoss(nn.Module):
    """
    Laplace negative log-likelihood loss with learnable scale parameters (b = exp(eta)).
    Returns a differentiable total loss for backward() and detached monitoring values.
    """

    def __init__(self, init_logb_T: float, init_logb_P: float, clamp_range=(-5.0, 5.0)):
        super().__init__()
        # eta = log(b), optimized directly
        self.eta_T = nn.Parameter(torch.tensor(float(init_logb_T)))
        self.eta_P = nn.Parameter(torch.tensor(float(init_logb_P)))
        self.clamp_range = clamp_range

    def forward(self, pred_T, target_T, pred_P, target_P):
        # Mean absolute errors
        mae_T = torch.mean(torch.abs(pred_T - target_T))
        mae_P = torch.mean(torch.abs(pred_P - target_P))

        # Clamp eta to avoid exploding/vanishing exponentials
        eta_T_clamped = torch.clamp(self.eta_T, *self.clamp_range)
        eta_P_clamped = torch.clamp(self.eta_P, *self.clamp_range)

        # Laplace NLL: |e|/b + log(b)  with b = exp(eta)
        loss_T = mae_T * torch.exp(-eta_T_clamped) + eta_T_clamped
        loss_P = mae_P * torch.exp(-eta_P_clamped) + eta_P_clamped
        total_loss = loss_T + loss_P

        # Detach monitoring values (no gradients)
        mae_T_det = mae_T.detach()
        mae_P_det = mae_P.detach()
        b_T = torch.exp(eta_T_clamped).detach()
        b_P = torch.exp(eta_P_clamped).detach()

        # For training: only total_loss is used in backward()
        return total_loss, mae_T_det, mae_P_det, b_T, b_P

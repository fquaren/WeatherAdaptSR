import torch

# Custom loss function for engression model
def engression_loss(pred, target):
    mean, log_var = pred  # Extract mean and log variance
    precision = torch.exp(-log_var)  # Compute precision (1 / variance)
    
    # Negative log-likelihood loss
    loss = precision * (target - mean) ** 2 + log_var  # NLL loss
    return loss.mean()  # Average loss over batch


def engression_loss_with_reg(pred, target, input_log_var, reg_lambda=1e-4):
    mean, log_var = pred  # Extract mean and log variance
    precision = torch.exp(-log_var)  # Compute precision (1 / variance)
    
    # Negative log-likelihood loss
    loss = precision * (target - mean) ** 2 + log_var  # NLL loss

    # L2 regularization on the log variance (for noise regularization)
    l2_reg = reg_lambda * torch.sum(input_log_var**2)  # L2 penalty on input_log_var

    return loss.mean() + l2_reg  # Final loss including regularization
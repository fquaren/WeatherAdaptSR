import torch


def deep_coral_loss(source_features, target_features):
    """
    Aligns the covariance matrices of source and target spatial feature maps.
    """
    b, c, h, w = source_features.size()
    source_flat = source_features.view(b, c, -1).transpose(1, 2).reshape(-1, c)
    target_flat = target_features.view(b, c, -1).transpose(1, 2).reshape(-1, c)

    cov_source = torch.cov(source_flat.T)
    cov_target = torch.cov(target_flat.T)

    return torch.mean((cov_source - cov_target) ** 2)


def spectral_density_loss(pred_target, y_target_unlabeled):
    """
    Enforces the power-law scaling of the predicted precipitation to match the target climatology.
    """
    fft_pred = torch.fft.rfft2(pred_target)
    fft_target = torch.fft.rfft2(y_target_unlabeled)

    psd_pred = torch.abs(fft_pred) ** 2
    psd_target = torch.abs(fft_target) ** 2

    mean_psd_pred = torch.mean(psd_pred, dim=0)
    mean_psd_target = torch.mean(psd_target, dim=0)

    return torch.nn.functional.mse_loss(mean_psd_pred, mean_psd_target)


def mmd_loss(source_features, target_features, sigma=1.0):
    """
    Maximum mean discrepancy using a Gaussian kernel.
    """
    b = source_features.size(0)
    source_flat = source_features.view(b, -1)
    target_flat = target_features.view(b, -1)

    xx = torch.matmul(source_flat, source_flat.T)
    yy = torch.matmul(target_flat, target_flat.T)
    xy = torch.matmul(source_flat, target_flat.T)

    rx = torch.diag(xx).unsqueeze(0).expand_as(xx)
    ry = torch.diag(yy).unsqueeze(0).expand_as(yy)

    dxx = rx.T + rx - 2.0 * xx
    dyy = ry.T + ry - 2.0 * yy
    dxy = rx.T + ry - 2.0 * xy

    k_xx = torch.exp(-dxx / (2.0 * sigma**2))
    k_yy = torch.exp(-dyy / (2.0 * sigma**2))
    k_xy = torch.exp(-dxy / (2.0 * sigma**2))

    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()


def sinkhorn_divergence(source_features, target_features, epsilon=0.1, n_iters=50):
    """
    Approximates the 1-Wasserstein distance using entropic regularization.
    """
    b = source_features.size(0)
    source_flat = source_features.view(b, -1)
    target_flat = target_features.view(b, -1)

    C = torch.cdist(source_flat, target_flat, p=2)
    K = torch.exp(-C / epsilon)

    u = torch.ones(b, device=source_features.device) / b
    v = torch.ones(b, device=target_features.device) / b

    for _ in range(n_iters):
        u = (1.0 / b) / (torch.matmul(K, v) + 1e-8)
        v = (1.0 / b) / (torch.matmul(K.T, u) + 1e-8)

    gamma = torch.diag(u) @ K @ torch.diag(v)
    loss = torch.sum(gamma * C)
    return loss


def fourier_domain_adaptation(source_img, target_img, beta=0.01):
    """
    Swaps the low-frequency amplitude of the source input with the target input.
    """
    fft_src = torch.fft.fftn(source_img, dim=(-2, -1))
    fft_tgt = torch.fft.fftn(target_img, dim=(-2, -1))

    amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)
    amp_tgt = torch.abs(fft_tgt)

    fft_src_shifted = torch.fft.fftshift(amp_src, dim=(-2, -1))
    fft_tgt_shifted = torch.fft.fftshift(amp_tgt, dim=(-2, -1))

    h, w = source_img.shape[-2:]
    b_h, b_w = int(h * beta), int(w * beta)
    c_h, c_w = h // 2, w // 2

    mask = torch.zeros_like(amp_src)
    mask[..., c_h - b_h : c_h + b_h, c_w - b_w : c_w + b_w] = 1.0

    amp_src_shifted_new = fft_src_shifted * (1 - mask) + fft_tgt_shifted * mask
    amp_src_new = torch.fft.ifftshift(amp_src_shifted_new, dim=(-2, -1))

    fft_src_new = amp_src_new * torch.exp(1j * pha_src)
    src_in_tgt = torch.fft.ifftn(fft_src_new, dim=(-2, -1)).real
    return src_in_tgt


def quantile_mapping(source_pred, target_obs):
    """
    Enforces the predicted cumulative distribution function of the target domain to map to the historical empirical cumulative distribution function.
    """
    b = source_pred.size(0)
    source_flat = source_pred.view(b, -1)
    target_flat = target_obs.view(b, -1)

    source_sorted, indices = torch.sort(source_flat, dim=1)
    target_sorted, _ = torch.sort(target_flat, dim=1)

    mapped_source = torch.zeros_like(source_flat)
    mapped_source.scatter_(1, indices, target_sorted)

    return mapped_source.view_as(source_pred)

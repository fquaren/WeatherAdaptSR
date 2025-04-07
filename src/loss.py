
import torch

def compute_mmd(source_features, target_features, kernel='rbf', sigma=1.0):
    """
    Computes the Maximum Mean Discrepancy (MMD) between source and target features.
    
    Args:
        source_features: Tensor of shape (N, D) from the source domain.
        target_features: Tensor of shape (M, D) from the target domain.
        kernel: Kernel type ('rbf' for Gaussian, 'linear' for simple dot product).
        sigma: Bandwidth for the RBF kernel.
        
    Returns:
        MMD loss (scalar)
    """
    def rbf_kernel(X, Y, sigma):
        """
        Computes the Gaussian RBF kernel between X and Y.
        """
        XX = torch.matmul(X, X.t())  # Shape: (N, N)
        YY = torch.matmul(Y, Y.t())  # Shape: (M, M)
        XY = torch.matmul(X, Y.t())  # Shape: (N, M)
        
        X_sqnorms = XX.diag().unsqueeze(1)  # Shape: (N, 1)
        Y_sqnorms = YY.diag().unsqueeze(0)  # Shape: (1, M)
        
        dists = X_sqnorms - 2 * XY + Y_sqnorms  # Squared L2 distance
        K = torch.exp(-dists / (2 * sigma ** 2))  # Apply RBF kernel
        return K
    
    if kernel == 'rbf':
        K_ss = rbf_kernel(source_features, source_features, sigma)
        K_tt = rbf_kernel(target_features, target_features, sigma)
        K_st = rbf_kernel(source_features, target_features, sigma)
    elif kernel == 'linear':
        K_ss = torch.matmul(source_features, source_features.t())
        K_tt = torch.matmul(target_features, target_features.t())
        K_st = torch.matmul(source_features, target_features.t())
    else:
        raise ValueError("Invalid kernel type. Choose 'rbf' or 'linear'.")
    
    # Compute MMD loss
    mmd_loss = K_ss.mean() + K_tt.mean() - 2 * K_st.mean()
    return mmd_loss
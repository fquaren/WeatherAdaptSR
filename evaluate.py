import argparse
import os
import torch
import numpy as np
import json
from data.dataset import ClimateSRDataset
from deterministic_model import EDSRModel


def inverse_arcsinh(x):
    """Converts transformed variables back to physical units (mm)."""
    return np.sinh(x)


def evaluate(source_domain, target_domain, device):
    # 1. Load the model trained on the source domain
    model_path = f"models/model_{source_domain}.pth"
    model = EDSRModel(dynamic_in_channels=9, static_in_channels=2, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. Load the target domain test dataset
    target_cluster_path = f"/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift/{target_domain}"
    stats_path = f"/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift/stats_{target_domain}.json"

    with open(stats_path, "r") as f:
        target_stats = json.load(f)

    # Note: Use a dedicated 'test' split here, not the validation split used during training
    test_dataset = ClimateSRDataset(
        cluster_path=target_cluster_path,
        split="test",
        normalization_stats=target_stats,
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    total_mse = 0.0
    total_samples = 0

    with torch.no_grad():
        for x_dyn, x_stat, y_true_transformed in test_loader:
            x_dyn, x_stat = x_dyn.to(device), x_stat.to(device)

            # Forward pass
            y_pred_transformed = model(x_dyn, x_stat).cpu().numpy()
            y_true_transformed = y_true_transformed.numpy()

            # Transform back to physical space (mm of precipitation)
            y_pred_physical = inverse_arcsinh(y_pred_transformed)
            y_true_physical = inverse_arcsinh(y_true_transformed)

            # Compute physical Mean Squared Error
            batch_mse = np.sum((y_pred_physical - y_true_physical) ** 2)
            total_mse += batch_mse
            total_samples += y_true_physical.size

    rmse_physical = np.sqrt(total_mse / total_samples)

    # Save the result to a central matrix file
    result_file = "results/generalization_matrix.csv"
    os.makedirs("results", exist_ok=True)

    if not os.path.exists(result_file):
        with open(result_file, "w") as f:
            f.write("source,target,rmse_physical\n")

    with open(result_file, "a") as f:
        f.write(f"{source_domain},{target_domain},{rmse_physical}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate(args.source, args.target, device)

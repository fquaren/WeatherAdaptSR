import argparse
import os
from src.models import UNet8x
from src.train import train_model
from data.dataloader import get_dataloaders


def main():
    data_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data"
    elevation_dir = os.path.join(data_path, "dem_squares")

    parser = argparse.ArgumentParser(description="Run UNet8x baseline experiment.")
    parser.add_argument("cluster_id", type=int, help="Cluster ID for data selection")
    parser.add_argument("exp_id", type=str, help="Experiment ID for logging")
    args = parser.parse_args()
    
    cluster_id = args.cluster_id  # Retrieve the cluster ID from arguments
    exp_id = args.exp_id  # Retrieve the experiment ID from arguments

    input_dir = os.path.join(data_path, f"1h_2D_sel_cropped_gridded_clustered_threshold_blurred/cluster_{cluster_id}")
    target_dir = os.path.join(data_path, f"1h_2D_sel_cropped_gridded_clustered_threshold/cluster_{cluster_id}")

    variable = "T_2M"

    exp_dir = f"UNet8x-baseline-T2M/{exp_id}"
    output_dir = os.path.join("/scratch/fquareng/experiments/", exp_dir)
    os.makedirs(output_dir, exist_ok=True)

    batch_size = 16
    num_epochs = 50
    patience = 5
    device = "cuda"

    print(f"Running experiment {exp_id} for variable {variable} ...")
    # best_model_path = os.path.join(output_dir, f"best_model_{exp_id}.pth")
    train_loader, val_loader, _ = get_dataloaders(variable, input_dir, target_dir, elevation_dir, batch_size)
    model = UNet8x()
    train_model(model, train_loader, val_loader, save_path=output_dir, num_epochs=num_epochs, device=device, patience=patience)
    # model.load_state_dict(torch.load(best_model_path))
    # model.to(device)
    # evaluate_and_plot(model, test_loader, device)

    return 0

if __name__ == "__main__":
    main()
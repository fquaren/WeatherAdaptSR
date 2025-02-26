import os
import random
import string
import torch

# from configs import config_local as config
from configs import config_curnagl as config

from src.train import train_model
from src.evaluate import evaluate_and_plot
from data.dataloader import get_dataloaders


# Generate random experiment ID
def generate_experiment_id(length=4):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


# Launch experiment
def main():
    
    # Set device and seed
    device = config["experiment"]["device"]
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["experiment"]["seed"])
    else:
        device = "cpu"

    # Setup experiment
    model = config["experiment"]["model"]
    exp_path = config["experiment"]["save_dir"]
    exp_name = config["experiment"]["name"]
    exp_id = generate_experiment_id()
    output_dir = os.path.join(exp_path, exp_name, exp_id)
    os.makedirs(output_dir, exist_ok=True)

    # Get data paths
    data_path = config["data"]["data_path"]
    input_path = config["data"]["input_path"]
    target_path = config["data"]["target_path"]
    variable = config["data"]["variable"]
    dem_dir = config["data"]["dem_path"] 
    cluster_id = config["data"]["cluster_id"]
    input_dir = os.path.join(data_path, input_path, f"cluster_{cluster_id}")
    assert os.path.exists(input_dir), f"Inputs directory {input_dir} does not exist."
    target_dir = os.path.join(data_path, target_path, f"cluster_{cluster_id}")
    assert os.path.exists(target_dir), f"Targets directory {target_dir} does not exist."

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders(
        variable, input_dir, target_dir, dem_dir, config["training"]["batch_size"])    
    
    # Train model
    best_model_path = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config["training"],
        device=device,
    )

    # Evaluate model
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    evaluate_and_plot(model, test_loader, device)

    return 0

if __name__ == "__main__":
    main()
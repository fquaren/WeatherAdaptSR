import os
import random
import string
import torch
import yaml
import argparse
import pandas as pd

from data.dataloader import get_dataloaders, get_cluster_dataloaders
from src.models import unet
from src.train import train_model, train_model_step_1
from src.evaluate import evaluate_and_plot, evaluate_and_plot_step_1


# Generate random experiment ID
def generate_experiment_id(length=4):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


# Launch experiment
def main():

    # Get argument for local or curnagl config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="curnagl", help="Local or curnagl config")
    args = parser.parse_args()
    config = "config_local" if args.config == "local" else "config_curnagl"
    print("Using config: ", config)

    # Load local config
    config_path = os.path.join(os.path.dirname(__file__), "configs", f"{config}.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Set device and seed
    device = config["experiment"]["device"]
    if device == "gpu" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["experiment"]["seed"])
        print("CUDA is available!")
    if device == "gpu" and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS is available!")
    else:
        device = torch.device("cpu")
        print("Neither NVIDIA nor MPS not available, using CPU.")

    # Setup experiment
    model = config["experiment"]["model"]
    exp_path = config["experiment"]["save_dir"]
    exp_name = config["experiment"]["name"]
    exp_id = generate_experiment_id()
    output_dir = os.path.join(exp_path, exp_name, exp_id)
    print(f"Experiment ID: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, "config.yaml"), "w") as file:
        yaml.dump(config, file)

    # Log experiment in experiments.csv: (Time, Experiment, Experiment ID, Model)
    time = str(pd.Timestamp.now()) 
    with open("experiments.csv", "a") as file:
        file.write(f"{time},{exp_name},{exp_id},{model}\n")

    # Get data paths
    data_path = config["data"]["data_path"]
    input_path = config["data"]["input_path"]
    target_path = config["data"]["target_path"]
    variable = config["data"]["variable"]
    dem_path = config["data"]["dem_path"] 

    input_dir = os.path.join(data_path, input_path)
    assert os.path.exists(input_dir), f"Inputs directory {input_dir} does not exist."
    target_dir = os.path.join(data_path, target_path)
    assert os.path.exists(target_dir), f"Targets directory {target_dir} does not exist."
    dem_dir = os.path.join(data_path, dem_path)
    assert os.path.exists(dem_dir), f"DEM directory {dem_dir} does not exist."

    # Load data
    train_loaders, val_loaders, test_loaders = get_dataloaders(
        variable, input_dir, target_dir, dem_dir, config["training"]["batch_size"])
    
    # Load model
    model = getattr(unet, model)()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)  # Wrap model for multi-GPU
    
    # Train model
    best_model_path = train_model(
        model=model,
        train_loader=train_loaders,
        val_loader=val_loaders,
        config=config["training"],
        device=device,
        save_path=output_dir
    )

    # Train model
    # best_model_path = train_model_step_1(
    #     model=model,
    #     train_loaders=train_loaders,
    #     val_loaders=val_loaders,
    #     config=config["training"],
    #     device=device,
    #     save_path=output_dir
    # )

    # Evaluate model
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    
    test_loss = evaluate_and_plot(
        model=model,
        config=config["testing"],
        test_loader=test_loaders,
        save_path=output_dir,
        device=device)
    
    # test_loss = evaluate_and_plot_step_1(
    #     model=model,
    #     config=config["testing"],
    #     test_loader=test_loaders,
    #     save_path=output_dir,
    #     device=device)
    
    print(f"Test loss: {test_loss:.4f}")
    
    return

if __name__ == "__main__":
    main()
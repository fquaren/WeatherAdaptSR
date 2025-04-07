import os
import random
import string
import torch
import yaml
import argparse
import pandas as pd
import glob

from data.dataloader import get_dataloaders
from src.models import unet
from src.train import train_model


# Generate random experiment ID
def generate_experiment_id(length=4):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


# Launch experiment
def main():

    # Get argument for local or curnagl config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="curnagl", help="Local or curnagl config")
    parser.add_argument("--resume_exp", type=str, default=None, help="Local or curnagl config")
    args = parser.parse_args()
    config = "config_local" if args.config == "local" else "config_curnagl"
    resume_exp = args.resume_exp
    print("Using config: ", config)

    # Load local config
    config_path = os.path.join(os.path.dirname(__file__), "configs", f"{config}.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Set device
    device = config["experiment"]["device"]

    # Setup experiment
    model = config["experiment"]["model"]
    exp_path = config["experiment"]["save_dir"]
    exp_name = config["experiment"]["name"]
    
    # Experiment id
    # Get exp id from argparse
    if resume_exp is None:
        # Generate a new experiment ID
        exp_id = generate_experiment_id()
        print(f"Generated new experiment ID: {exp_id}")
    else:
        # Check if the experiment ID already exists
        exp_id = resume_exp
        existing_ids = [f.split("/")[-1] for f in glob.glob(os.path.join(exp_path, exp_name, "*"))]
        # Check if the experiment ID exists
        if exp_id in existing_ids:
            print(f"Found experiment at {os.path.join(exp_path, exp_name, exp_id)} already exists. Resuming training.")
        else:
            print(f"Experiment ID {exp_id} does not exist. Retry.")
            return
        print(f"Resuming experiment with ID: {exp_id}")
        
    # exp_id = generate_experiment_id()
    output_dir = os.path.join(exp_path, exp_name, exp_id)
    print(f"Experiment ID: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Save config experiment
    time = str(pd.Timestamp.now()) 
    with open(os.path.join(output_dir, "config.yaml"), "w") as file:
        yaml.dump(config, file)
        # Save experiment metadata
        file.write(f"EXPERIENT_ID: {exp_id}\n")
        file.write(f"EXPERIMENT_DATE: {time}\n")

    # Use saved config file for experiment
    config_path = os.path.join(output_dir, f"config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        
    # Log experiment in experiments.csv: (Time, Model, Path)
    local_dir = "../../work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR"
    if os.path.exists(local_dir):
        if not os.path.exists(os.path.join(local_dir, "experiments.csv")):
            with open(os.path.join(local_dir, "experiments.csv"), "w") as file:
                file.write("Time,Model,Path\n")
        with open(os.path.join(local_dir, "experiments.csv"), "a") as file:
            # Append a line to the csv file
            file.write(f"{time},{model},{output_dir}\n")

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

    # Load model
    model = getattr(unet, model)()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)  # Wrap model for multi-GPU
    model.to(device)

    # Load data 
    dataloaders = get_dataloaders(
        input_dir=input_dir,
        target_dir=target_dir,
        elev_dir=dem_dir,
        variable=variable,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"]
    )

    print(f"Number of domains: {len(dataloaders)}")
    # Print lenght of each dataloader
    for k, dataloader in dataloaders.items():
        print(len(dataloader["train"]))

    # Train in a leave-one-cluster-out cross-validation fashion
    for excluded_cluster, loaders in dataloaders.items():
        print(f"Training excluding cluster: {excluded_cluster}")
        train_loaders = loaders["train"]
        val_loaders = loaders["val"]
    
        _  = train_model(
            model=model,
            excluding_cluster=excluded_cluster,
            train_loader=train_loaders,
            val_loader=val_loaders,
            config=config["training"],
            device=device,
            save_path=output_dir,
        )
    
    return

if __name__ == "__main__":
    main()
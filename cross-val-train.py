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
from src.train_mdan import train_model_mdan
from src.train_mmd import train_model_mmd



# Generate random experiment ID
def generate_experiment_id(length=4):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


# Launch experiment
def main():

    # Get argument for local or curnagl config
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="mmd", help="Domain adaptaion method.")
    parser.add_argument("--resume", type=str, default=None, help="Path experiment to resume.")
    args = parser.parse_args()

    method = args.method
    if method == "mdan":
        config = "config_mdan"
    if method == "mmd":
        config = "config_mmd"

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "configs", f"{config}.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Set device and seed
    device = config["experiment"]["device"]

    # Setup experiment
    model = config["experiment"]["model"]
    exp_path = config["experiment"]["save_dir"]
    exp_name = config["experiment"]["name"]
    
    # Experiment id
    # Generate a new experiment ID
    resume = args.resume
    if resume:
        if os.path.isdir(resume):
            output_dir = resume
            exp_id = resume.split("/")[-1]
            print(f"Resuming experiment {exp_id} stored at {output_dir} ...")
        else:
            print(f"{output_dir} not found. Provide different path.")
    else:
        print("Starting new experiment ...")
        exp_id = generate_experiment_id()
        print(f"Generated new experiment ID: {exp_id}")
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
    local_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR"
    if os.path.exists(local_dir):
        if not os.path.exists(os.path.join(local_dir, "experiments.csv")):
            with open(os.path.join(local_dir, "experiments.csv"), "w") as file:
                file.write("Time,Model,Path\n")
        with open(os.path.join(local_dir, "experiments.csv"), "a") as file:
            print(f"Appending to {os.path.join(local_dir, 'experiments.csv')}")
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
        num_workers=config["training"]["num_workers"],
    )

    # Train in a leave-one-cluster-out cross-validation fashion
    if method == "mdan":
        train_model_mdan(
            model=model,
            dataloaders=dataloaders,
            config=config["training"],
            device=device,
            save_path=output_dir,
        )
    if method == "mmd":
        train_model_mmd(
            model=model,
            dataloaders=dataloaders,
            config=config["training"],
            device=device,
            save_path=output_dir,
        )
    if method == None:
        print("Select domain adaptation method. Options: mdan, mmd")
    
    return

if __name__ == "__main__":
    main()
import os
import random
import string
import torch
import yaml
import argparse
import pandas as pd
import glob
import numpy as np
import optuna
import gc
import time

from data.dataloader import get_clusters_dataloader
from src.models import unet
from src.train import train_model, objective


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

    # Set random seed for reproducibility
    # random.seed(42)
    # np.random.seed(42)
    # torch.manual_seed(42)
    # torch.cuda.manual_seed(42)

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
    
    # Experiment ID
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
    local_dir = config["paths"]["local_dir"]
    if os.path.exists(local_dir):
        if not os.path.exists(os.path.join(local_dir, "experiments.csv")):
            with open(os.path.join(local_dir, "experiments.csv"), "w") as file:
                file.write("Time,Model,Path\n")
        with open(os.path.join(local_dir, "experiments.csv"), "a") as file:
            # Append a line to the csv file
            file.write(f"{time},{model},{output_dir}\n")

    # Load model on device (multi-GPU if available)
    model = getattr(unet, model)()
    # Move model to device
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)  # Wrap model for multi-GPU
    model.to(device)

    # Load data
    data_path = config["paths"]["data_path"]
    # Choose clusters to train on for parallelization of training
    cluster_names = config["paths"]["clusters"]
    if cluster_names is None:
        print("No cluster names provided. Loading all clusters from data path.")
        cluster_names = sorted([c for c in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, c))])
    if config["training"]["load_data_on_gpu"]:
        device_data = "cuda"
    else:
        device_data = "cpu"

    # Optimize hyperparameters
    if config["optimization"]["num_epochs"] != 0:
        # Note: Only optimize excluding the first cluster (assumption: it is representative)
        print(f"Optimizing for training ...")
        cluster_dataloaders = get_clusters_dataloader(
            data_path=config["paths"]["data_path"],
            elev_dir=config["paths"]["elev_path"],
            excluded_cluster="cluster_0",  # Assuming the first cluster is representative
            batch_size=config["training"]["batch_size"],
            num_workers=config["training"]["num_workers"],
            use_theta_e=config["training"]["use_theta_e"],
            device=device_data,
            config=config,
        )
        train_loader = cluster_dataloaders["train"]
        val_loader = cluster_dataloaders["val"]
        num_epochs = config["optimization"]["num_epochs"]
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: objective(trial, model, num_epochs, train_loader, val_loader, config, device),
            n_trials=config["optimization"]["num_trials"]
        )
        
        for cluster in cluster_names:
            # Update the optimizer_params per cluster
            if cluster in config["domain_specific"]:
                config["domain_specific"][cluster]["optimizer_params"].update(study.best_params)
            # Save back to YAML
            with open(config_path, "w") as f:
                yaml.dump(config, f, sort_keys=False)

        # Load config file for experiment # TODO: check if this is needed
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

    # Train in a leave-one-cluster-out cross-validation fashion
    for excluded_cluster in cluster_names:
        print(f"Training excluding cluster: {excluded_cluster}")
        cluster_dataloaders = get_clusters_dataloader(
            data_path=config["paths"]["data_path"],
            elev_dir=config["paths"]["elev_path"],
            excluded_cluster=excluded_cluster,
            batch_size=config["training"]["batch_size"],
            num_workers=config["training"]["num_workers"],
            use_theta_e=config["training"]["use_theta_e"],
            device=device_data,
            config=config,
        )
        train_loader = cluster_dataloaders["train"]
        val_loader = cluster_dataloaders["val"]
        _ = train_model(
            model=model,
            excluding_cluster=excluded_cluster,
            num_epochs=config["training"]["num_epochs"],
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            save_path=output_dir,
        )
        # Empty gpu memory
        print(f"Finished training excluding cluster: {excluded_cluster}")
        print(f"Emptying GPU memory for cluster: {excluded_cluster}")
        # Check memory usage
        for dataset in cluster_dataloaders["train"].dataset.datasets:
            dataset.unload_from_gpu()
        for dataset in cluster_dataloaders["val"].dataset.datasets:
            dataset.unload_from_gpu()
        for dataset in cluster_dataloaders["test"].dataset.datasets:
            dataset.unload_from_gpu()
        del train_loader, val_loader, cluster_dataloaders, model
        torch.cuda.empty_cache()
        gc.collect()
        print(f"GPU memory emptied for cluster: {excluded_cluster}")

        # Reset model to initial state for next training
        print(f"Model reset for next training ...")
        model = getattr(unet, model)()
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = torch.nn.DataParallel(model)
        model.to(device)
        print("Done. Continuing to next cluster ...")

    return

if __name__ == "__main__":
    main()
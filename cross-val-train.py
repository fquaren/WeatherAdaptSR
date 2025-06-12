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

    print("Starting cross-validation training ...")
    start_time = pd.Timestamp.now()
    print("Time: ", start_time)

    # Get argument for local or curnagl config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="curnagl", help="Local or curnagl config")
    parser.add_argument("--resume_exp", type=str, default=None, help="Local or curnagl config")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    args = parser.parse_args()
    config = "config_local" if args.config == "local" else "config_curnagl"
    resume_exp = args.resume_exp
    model_name = args.model
    print("Using config: ", config)

    # Load local config
    config_path = os.path.join(os.path.dirname(__file__), "configs", f"{config}.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    device = config["experiment"]["device"]
    # model_name = config["experiment"]["model"]
    exp_path = config["experiment"]["save_dir"]
    exp_name = config["experiment"]["name"]
    
    # Experiment ID
    if resume_exp is None:
        exp_id = generate_experiment_id()
        print(f"Generated new experiment ID: {exp_id}")
    else:
        exp_id = resume_exp
        existing_ids = [f.split("/")[-1] for f in glob.glob(os.path.join(exp_path, exp_name, "*"))]
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
    with open(os.path.join(output_dir, "config.yaml"), "w") as file:
        yaml.dump(config, file)
        file.write(f"EXPERIMENT_ID: {exp_id}\n")
        file.write(f"EXPERIMENT_START_TIME: {start_time}\n")

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
            file.write(f"{start_time},{model_name},{output_dir}\n")

    # Load data
    data_path = config["paths"]["data_path"]
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
        for cluster in cluster_names:
            # Load model on device (multi-GPU if available)
            print(f"Loading model: {model_name} ...")
            model = getattr(unet, model_name)()
            if model is None:
                print(f"Model {model_name} not found in unet module. Please check the model name.")
                return
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs!")
                model = torch.nn.DataParallel(model)  # Wrap model for multi-GPU
            else:
                print("Using single GPU or CPU.")
            print(f"Moving model to device: {device} ...")
            model.to(device)

            print(f"Optimizing model excluding {cluster} for training ...")
            num_epochs = config["optimization"]["num_epochs"]
            study = optuna.create_study(direction="minimize")
            study.optimize(
                lambda trial: objective(trial, model, num_epochs, cluster, config, device, device_data),
                n_trials=config["optimization"]["num_trials"]
            )

            # Update params        
            if cluster in config["domain_specific"]:
                config["domain_specific"][cluster]["optimizer_params"].update(study.best_params)
            with open(config_path, "w") as f:
                yaml.dump(config, f, sort_keys=False)
    else:
        print("Skipping hyperparameter optimization as num_epochs is set to 0.")

    # Load config file for experiment
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # If method=vanilla

    # Train in a leave-one-cluster-out cross-validation fashion
    for excluded_cluster in cluster_names:
        
        # Load model on device (multi-GPU if available)
        print(f"Loading model: {model_name} ...")
        model = getattr(unet, model_name)()
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = torch.nn.DataParallel(model)  # Wrap model for multi-GPU
        else:
            print("Using single GPU or CPU.")
        print(f"Moving model to device: {device} ...")
        model.to(device)
        
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

    # If method=mmd

    # If method=mdan

    print("All clusters trained. Experiment completed.")
    print(f"Experiment saved at: {output_dir}")
    print("Elapsed time: ", pd.Timestamp.now() - start_time)

    # Save config experiment
    with open(os.path.join(output_dir, "config.yaml"), "w") as file:
        yaml.dump(config, file)
        file.write(f"EXPERIMENT_ELAPSED_TIME: {pd.Timestamp.now() - start_time}\n")

    return output_dir

if __name__ == "__main__":
    main()
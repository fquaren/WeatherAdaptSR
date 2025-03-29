import os
import random
import string
import torch
import yaml
import argparse
import pandas as pd
import numpy as np

from data.dataloader import get_dataloaders
from src.models import unet
from src.train import train_model

import glob

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

    # Train in a leave-one-cluster-out cross-validation fashion
    for excluded_cluster, loaders in dataloaders.items():
        print(f"Training excluding cluster: {excluded_cluster}")
        train_loaders = loaders["train"]
        val_loaders = loaders["val"]
    
        cluster_best_model_path, cluster_train_losses, cluster_val_losses  = train_model(
            model=model,
            excluding_cluster=excluded_cluster,
            train_loader=train_loaders,
            val_loader=val_loaders,
            config=config["training"],
            device=device,
            save_path=output_dir,
        )

        np.save(os.path.join(output_dir, f"train_losses_{excluded_cluster}.npy"), np.array(cluster_train_losses))
        np.save(os.path.join(output_dir, f"val_losses_{excluded_cluster}.npy"), np.array(cluster_val_losses))

        # Save best model path
        with open(os.path.join(output_dir, f"best_model_paths_{excluded_cluster}.yaml"), "w") as file:
            yaml.dump(cluster_best_model_path, file)


    # # Train model
    # _ = train_model(
    #     model=model,
    #     train_loader=train_loaders,
    #     val_loader=val_loaders,
    #     config=config["training"],
    #     device=device,
    #     save_path=output_dir,
    #     model_path="/scratch/fquareng/experiments/UNet-8x-baseline-T2M/953m/",
    #     model_name="checkpoint_epoch_35.pth",
    #     fine_tuning=True
    # )

    # Pretrain model on all clusters
    # best_pretrained_model_path = train_model_step_1(
    #     model=model,
    #     train_loaders=train_loaders,
    #     val_loaders=val_loaders,
    #     config=config["training"],
    #     device=device,
    #     save_path=output_dir,
    #     model_path="/scratch/fquareng/experiments/UNet-8x-baseline-T2M/x50d/",
    #     model_name="checkpoint_epoch_15.pth"
    # )

    # Finetune models on each of the clusters
    # best_pretrained_model_path = "/scratch/fquareng/experiments/UNet-8x-baseline-T2M/953m/best_model.pth"
    # train_model_step_2(
    #     model=model,
    #     train_loaders=train_loaders,
    #     val_loaders=val_loaders,
    #     config=config["training"],
    #     device=device,
    #     save_path=output_dir,
    #     model_path=best_pretrained_model_path,
    # )

    # Evaluate model
    # model.load_state_dict(torch.load(best_model_path))
    # model.to(device)
    
    # test_loss = evaluate_and_plot(
    #     model=model,
    #     config=config["testing"],
    #     test_loader=test_loaders,
    #     save_path=output_dir,
    #     device=device)
    
    # test_loss = evaluate_and_plot_step_1(
    #     model=model,
    #     config=config["testing"],
    #     test_loader=test_loaders,
    #     save_path=output_dir,
    #     device=device)
    
    # print(f"Test loss: {test_loss:.4f}")
    
    return

if __name__ == "__main__":
    main()
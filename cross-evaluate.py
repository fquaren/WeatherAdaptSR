import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import os
import argparse
import yaml
from tqdm import tqdm

from data.dataloader import get_dataloaders
from src.models import unet


def evaluate_model(model, criterion, test_loader, device="cuda"):
    """
    Evaluates the model on the test datasets from multiple clusters, computes test loss, and plots results.
    
    Args:
        model: The PyTorch model to evaluate.
        config (dict): The configuration containing criterion and other settings.
        test_loader (array): Array where values are DataLoader instances for testing.
        save_path (str): Directory to save the evaluation results.
        device (str): Device to run the evaluation on ('cpu', 'cuda', etc.).
        save (bool): Whether to save the plots and losses.
    
    Returns:
        float: The mean test loss across all clusters.
    """
      
    test_losses, predictions, targets, elevations, inputs = [], [], [], [], []
    with torch.no_grad():
        for temperature, elevation, target in tqdm(test_loader):
            # TODO: get file name from the dataloader
            temperature, elevation, target = temperature.to(device), elevation.to(device), target.to(device)
            output = model(temperature, elevation)
            loss = criterion(output, target).item()
            test_losses.append(loss)
            predictions.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
            elevations.append(elevation.cpu().numpy())
            inputs.append(temperature.cpu().numpy()) 
    
    test_losses = np.array(test_losses)
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0) 
    elevations = np.concatenate(elevations, axis=0)
    inputs = np.concatenate(inputs, axis=0)

    evaluation_results = {
        "test_losses": test_losses,
        "predictions": predictions,
        "targets": targets,
        "elevations": elevations,
        "inputs": inputs
    }

    return evaluation_results


def plot_results(evaluation_results, eval_on_cluster, cluster_name, save_path, save=True):
    """"
    Plots the worst and best 5 examples based on test loss for a given cluster."
    """
    # Unpack evaluation results
    test_losses = evaluation_results["test_losses"]
    predictions = evaluation_results["predictions"]
    targets = evaluation_results["targets"]
    elevations = evaluation_results["elevations"]
    inputs = evaluation_results["inputs"]
    
    # Create directory for saving results
    os.makedirs(save_path, exist_ok=True)
    
    # Get top 5 and bottom 5 examples based on loss
    top_5_idx = test_losses.argsort()[-5:][::-1]  # Highest loss
    bottom_5_idx = test_losses.argsort()[:5]      # Lowest loss
    
    # Plot results for the current cluster
    fig, axes = plt.subplots(5, 4, figsize=(10, 15))
    plt.suptitle(f"WORST 5 - Mean Test Loss for model eval on {eval_on_cluster}\nand tested on {cluster_name}: {test_losses.mean():.4f}")
    for i, idx in enumerate(top_5_idx):
        for j, (data, cmap, title) in enumerate(zip(
            [inputs[idx][0], predictions[idx][0], targets[idx][0], elevations[idx][0]], 
            ['coolwarm', 'coolwarm', 'coolwarm', 'viridis'], 
            [f"Input", f"Prediction (Loss: {test_losses[idx]:.4f})", 
            f"Target", f"Elevation"])):
            
            # Display image
            img = axes[i, j].imshow(data, cmap=cmap)
            axes[i, j].set_title(title)
            axes[i, j].axis("off")  # Hide axes
            
            # Add colorbar
            cbar = fig.colorbar(img, ax=axes[i, j], fraction=0.046, pad=0.04)  
            cbar.ax.tick_params(labelsize=8)  # Adjust tick size
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(save_path, f"evaluation_results_worst_{eval_on_cluster}_{cluster_name}.png"))
    
    fig, axes = plt.subplots(5, 4, figsize=(10, 15))
    plt.suptitle(f"BEST 5 - Mean Test Loss for model eval on {eval_on_cluster}\nand tested on {cluster_name}: {test_losses.mean():.4f}")
    for i, idx in enumerate(bottom_5_idx):
        for j, (data, cmap, title) in enumerate(zip(
            [inputs[idx][0], predictions[idx][0], targets[idx][0], elevations[idx][0]], 
            ['coolwarm', 'coolwarm', 'coolwarm', 'viridis'], 
            [f"Input", f"Prediction (Loss: {test_losses[idx]:.4f})", 
            f"Target", f"Elevation"])):
            
            # Display image
            img = axes[i, j].imshow(data, cmap=cmap)
            axes[i, j].set_title(title)
            axes[i, j].axis("off")  # Hide axes
            
            # Add colorbar
            cbar = fig.colorbar(img, ax=axes[i, j], fraction=0.046, pad=0.04)  
            cbar.ax.tick_params(labelsize=8)  # Adjust tick size

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(save_path, f"evaluation_results_best_{eval_on_cluster}_{cluster_name}.png"))
    plt.close(fig)  # Close the figure to free memory


def plot_training_metrics(save_path, evaluation_path, model_architecture, excluded_cluster):
    # Plot training metrics
    train_losses = np.load(os.path.join(save_path, "train_losses.npy"))
    val_losses = np.load(os.path.join(save_path, "val_losses.npy"))
    _ = plt.figure()
    plt.title(f"Training metrics {model_architecture} model trained on {excluded_cluster}")
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.yscale("log")
    plt.savefig(os.path.join(evaluation_path, f"training_metrics.png"))
    plt.close()


def plot_mean_test_loss_matrix(mean_test_loss_matrix, dataloaders, save_path):
    """"
    Plots the mean test loss matrix for all clusters.
    """
    # Plot mean test loss matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(mean_test_loss_matrix, cmap='coolwarm')
    plt.colorbar(cax)
    ax.set_xticks(np.arange(len(dataloaders)))
    ax.set_yticks(np.arange(len(dataloaders)))
    ax.set_xticklabels(list(dataloaders.keys()))
    ax.set_yticklabels(list(dataloaders.keys()))
    plt.xlabel("Excluded Cluster")
    plt.ylabel("Test Cluster")
    plt.title("Mean Test Loss Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "mean_test_loss_matrix.png"))
    plt.close(fig)  # Close the figure to free memory


def main():
    # Get model path from arg command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for evaluation (cpu or cuda)")
    parser.add_argument("--exp_path", type=str, default=None, help="Path of model to evaluate")
    parser.add_argument("--local", type=str, default=None, help="Evaluation on local machine")
    args = parser.parse_args()    
    
    exp_path = args.exp_path
    if exp_path is None:
        raise ValueError("Please provide the path to the model to evaluate using --exp_path")
    print("Using model path: ", exp_path)
    
    # Load config file
    config_path = os.path.join(exp_path, "config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Set device
    device = args.device
    print("Device: ", device)
    
    # Load model architecture from config
    model_architecture = config["experiment"]["model"]
    # Check if model_architecture is a valid attribute of the unet module
    if not hasattr(unet, model_architecture):
        raise ValueError(f"Model architecture '{model_architecture}' not found in unet.py module.")
    # Check if model_architecture is a callable class
    if not callable(getattr(unet, model_architecture)):
        raise ValueError(f"Model architecture '{model_architecture}' is not callable.")
    # Initialize model
    model = getattr(unet, model_architecture)()
    
    # Get dataloaders
    if args.local is not None:
        input_dir="/Users/fquareng/data/1d-PS-RELHUM_2M-T_2M_cropped_gridded_clustered_threshold_12_blurred"
        target_dir="/Users/fquareng/data/1d-PS-RELHUM_2M-T_2M_cropped_gridded_clustered_threshold_12"
        dem_dir="/Users/fquareng/data/dem_squares"
    else:
        input_dir = os.path.join(config["data"]["data_path"], config["data"]["input_path"])
        target_dir = os.path.join(config["data"]["data_path"], config["data"]["target_path"])
        dem_dir = os.path.join(config["data"]["data_path"], config["data"]["dem_path"])

    # Get dataloaders
    print("Getting dataloaders...")
    dataloaders = get_dataloaders(
        input_dir=input_dir,
        target_dir=target_dir,
        elev_dir=dem_dir,
        variable=config["data"]["variable"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        transform=config["training"]["transform"],
    )

    # Load criterion
    criterion = getattr(torch.nn, config["testing"]["criterion"])()
    if criterion is None:
        raise ValueError(f"Criterion '{config['experiment']['criterion']}' not found.")
    criterion.to(device)

    # Test in a leave-one-cluster-out cross-validation fashion
    # Compute mean test loss for each cluster
    mean_test_loss_matrix = np.zeros((len(dataloaders), len(dataloaders)))
    for i, (excluded_cluster, loaders) in enumerate(dataloaders.items()):
        
        print(f"Evaluating model trained on cluster: {excluded_cluster}")

        save_path = os.path.join(exp_path, excluded_cluster)
        evaluation_path = os.path.join(save_path, "evaluation")
        os.makedirs(evaluation_path, exist_ok=True)

        plot_training_metrics(save_path, evaluation_path, model_architecture, excluded_cluster)

        snapshot_path = os.path.join(save_path, "best_snapshot.pth")
        if os.path.exists(snapshot_path):
            snapshot = torch.load(os.path.join(snapshot_path), map_location=torch.device(device), weights_only=False)
            model.load_state_dict(snapshot["model_state_dict"])
            print(f"Loaded model from {snapshot_path}")
        else:
            raise FileNotFoundError(f"Snapshot file {snapshot_path} does not exist.")
        
        # Set model to evaluation mode
        model.to(device)
        model.eval()

        # Evaluate model on the test dataset of the excluded cluster
        for j, (cluster, loaders) in enumerate(dataloaders.items()):
            
            # Get cluster test loader
            test_loader = loaders["test"]
            evaluation_results = evaluate_model(
                model,
                criterion,
                test_loader,
                device=device,
            )

            # Save evaluation results
            # np.savez(os.path.join(evaluation_path, f"eval_{excluded_cluster}_on_{cluster}.npz"), **evaluation_results)
            # print(f"Evaluation results saved to {evaluation_path}/eval_{excluded_cluster}_on_{cluster}.npz")

            # Plot results
            plot_results(
                evaluation_results,
                excluded_cluster,
                cluster,
                evaluation_path,
                save=True
            )
            # Compute mean test loss
            mean_test_loss_matrix[i, j] = np.mean(evaluation_results["test_losses"])
    # Save mean test loss matrix
    np.savez(os.path.join(save_path, "mean_test_loss_matrix.npz"), mean_test_loss_matrix)
    print("Mean test loss matrix saved to ", os.path.join(exp_path, "mean_test_loss_matrix.npz"))
    
    # Plot mean test loss matrix
    plot_mean_test_loss_matrix(mean_test_loss_matrix, dataloaders, exp_path)
    print("Mean test loss matrix plotted and saved.")

if __name__ == "__main__":
    main()
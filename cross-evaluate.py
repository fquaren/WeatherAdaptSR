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
from src.train import train_model


def evaluate_model(model, criterion, test_loader, save_path, device="cuda", save=True):
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

    model.eval()
      
    test_losses, predictions, targets, elevations, inputs = [], [], [], [], []
    with torch.no_grad():
        for temperature, elevation, target in tqdm(test_loader):
            temperature, elevation, target = temperature.to(device), elevation.to(device), target.to(device)
            output = model(temperature, elevation)
            loss = criterion(output, target).item()
            test_losses.append(loss.cpu().numpy())
            predictions.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
            elevations.append(elevation.cpu().numpy())
            inputs.append(temperature.cpu().numpy())
    
    test_losses = np.concatenate(test_losses, axis=0)
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
    if save:
        # Save evaluation results
        np.savez(os.path.join(save_path, "evaluation_results.npz"), **evaluation_results)
        print(f"Evaluation results saved to {save_path}/evaluation_results.npz")

    return evaluation_results


def plot_results(evaluation_results, cluster_name, save_path, save=True):
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
    plt.suptitle(f"WORST 5 - Mean Test Loss for {cluster_name}: {test_losses.mean():.4f}")
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
        plt.savefig(os.path.join(save_path, f"evaluation_results_worst_{cluster_name}.png"))
    
    fig, axes = plt.subplots(5, 4, figsize=(10, 15))
    plt.suptitle(f"BEST 5 - Mean Test Loss for {cluster_name}: {test_losses.mean():.4f}")
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
        plt.savefig(os.path.join(save_path, f"evaluation_results_best_{cluster_name}.png"))
    plt.close(fig)  # Close the figure to free memory


def main():
    # Get model path from arg command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation (cpu or cuda)")
    parser.add_argument("--exp_path", type=str, default=None, help="Path of model to evaluate")
    args = parser.parse_args()
    
    # Set device
    device = args.device
    print("Using device: ", device)    
    
    exp_path = args.exp_path
    if exp_path is None:
        raise ValueError("Please provide the path to the model to evaluate using --exp_path")
    print("Using model path: ", exp_path)
    
    # Load config file
    config_path = os.path.join(exp_path, "config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
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
    dataloaders = get_dataloaders(
        input_dir=config["data"]["input_path"],
        target_dir=config["data"]["target_path"],
        elev_dir=config["data"]["dem_path"],
        variable=config["data"]["variable"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"]
    )

    # Load criterion
    criterion = getattr(torch.nn, config["testing"]["criterion"])()
    if criterion is None:
        raise ValueError(f"Criterion '{config['experiment']['criterion']}' not found.")
    criterion.to(device)

    # Test in a leave-one-cluster-out cross-validation fashion
    # Compute mean test loss for each cluster
    mean_test_loss_matrix = np.zeros((len(dataloaders), len(dataloaders)))
    for i, excluded_cluster, loaders in enumerate(dataloaders.items()):
        print(f"Evaluating excluding cluster: {excluded_cluster}")

        save_path = os.path.join(exp_path, excluded_cluster)
        snapshot_path = os.path.join(save_path, "best_snapshot.pth")
        if os.path.exists(snapshot_path):
            snapshot = torch.load(os.path.join(snapshot_path))
            model.load_state_dict(torch.load(snapshot["model_state_dict"]))
            print(f"Loaded model from {snapshot_path}")
        else:
            raise FileNotFoundError(f"Snapshot file {snapshot_path} does not exist.")
        
        # Set model to evaluation mode
        model.to(device)

        # Get cluster test loader
        evaluation_path = os.path.join(save_path, "evaluation")
        for j, cluster, loaders in enumerate(dataloaders.items()):
            test_loader = loaders["cluster"]["test"]
            evaluation_results = evaluate_model(
                model,
                criterion,
                test_loader,
                evaluation_path,
                device=device,
                save=True
            )
            # Plot results
            plot_results(
                evaluation_results.cpu(),
                cluster,
                evaluation_path,
                save=True
            )
            # Compute mean test loss
            mean_test_loss_matrix[i, j] = np.mean(evaluation_results["test_losses"])
        
    return

if __name__ == "__main__":
    main()
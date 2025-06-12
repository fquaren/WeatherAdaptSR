import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import os
import argparse
import yaml
from tqdm import tqdm
import gc

from data.dataloader import get_single_cluster_dataloader, get_clusters_dataloader
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


def plot_mean_test_loss_matrix(mean_test_loss_matrix, cluster_names, save_path):
    """"
    Plots the mean test loss matrix for all clusters.
    """
    assert mean_test_loss_matrix.shape[0] == mean_test_loss_matrix.shape[1], "Matrix must be square"
    N = mean_test_loss_matrix.shape[0]

    arr = mean_test_loss_matrix.copy()
    cols_to_move = arr[:, [2, 3]].copy()
    mean_test_loss_matrix[:, 2:9] = mean_test_loss_matrix[:, 4:11]
    mean_test_loss_matrix[:, 9:11] = cols_to_move

    rows_to_move = mean_test_loss_matrix[[2, 3], :].copy()
    mean_test_loss_matrix[2:9, :] = mean_test_loss_matrix[4:11, :]
    mean_test_loss_matrix[9:11, :] = rows_to_move
    
    # 1. Compute the mean of the diagonal (same cluster train-test)
    mean_diagonal = np.mean(np.diag(mean_test_loss_matrix))

    # 2. Compute the mean difference for each column
    sum_differences = []
    for i in range(mean_test_loss_matrix.shape[1]):
        column_values = mean_test_loss_matrix[:, i]
        diff = column_values - column_values[i]  # Difference from the diagonal value
        diff = np.maximum(diff, 0)               # Apply relu to difference to avoid negative values
        sum_diff = np.sum(np.delete(diff, i))    # Exclude the diagonal element
        sum_differences.append(sum_diff)

    # Convert to a NumPy array for easy handling
    consistency = 1/(N*(N-1))*np.sum(np.array(sum_differences))

    print(f"Mean diagonal MSE: {mean_diagonal}")
    print(f"Mean non-diagonal MSE: {np.mean(np.delete(mean_test_loss_matrix, np.diag_indices(N)))}")
    print(f"Mean overall MSE: {np.mean(mean_test_loss_matrix)}")
    print(f"Consistency metric: {consistency}")

    # Plot mean test loss matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(mean_test_loss_matrix, cmap='viridis', vmin=0, vmax=np.max(mean_test_loss_matrix))
    plt.colorbar(cax)
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(cluster_names, rotation=45)
    ax.set_yticklabels(cluster_names)
    plt.xlabel("Domain")
    plt.ylabel("Model")
    plt.title(f"Mean Diagonal MSE: {mean_diagonal:.4f}, Mean Non-Diagonal MSE: {np.mean(np.delete(mean_test_loss_matrix, np.diag_indices(N)))}, \nMean Test Loss Matrix. MSE: {np.mean(mean_test_loss_matrix):.4f}, Consistency: {consistency:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "mean_test_loss_matrix.png"))
    plt.close(fig)  # Close the figure to free memory


def main():
    # Get model path from arg command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for evaluation (cpu or cuda)")
    parser.add_argument("--model", type=str, default=None, help="Model architecture to use for evaluation")
    parser.add_argument("--exp_path", type=str, default=None, help="Path of model to evaluate")
    parser.add_argument("--local", type=str, default=None, help="Evaluation on local machine")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of workers (optional)")
    parser.add_argument("--save_eval", default=False, help="Save evaluation results to disk")
    args = parser.parse_args()

    save_eval = args.save_eval
    
    exp_path = args.exp_path
    if exp_path is None:
        raise ValueError("Please provide the path to the model to evaluate using --exp_path")
    print("Using model path: ", exp_path)
    
    # Load config file
    config_path = os.path.join(exp_path, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist. Please provide a valid path.")
    print("Loading config from: ", config_path)
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Num workers 
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = config["training"]["num_workers"]
    else:
        num_workers = args.num_workers
    print(f"Num workers: {num_workers}")

    # Set device
    device = args.device
    print("Device: ", device)
    
    # Load model architecture from config
    model_architecture = args.model
    # model_architecture = config["experiment"]["model"]
    # Check if model_architecture is a valid attribute of the unet module
    if not hasattr(unet, model_architecture):
        raise ValueError(f"Model architecture '{model_architecture}' not found in unet.py module.")
    # Check if model_architecture is a callable class
    if not callable(getattr(unet, model_architecture)):
        raise ValueError(f"Model architecture '{model_architecture}' is not callable.")
    # Initialize model
    model = getattr(unet, model_architecture)()

    # Load data path and cluster names
    data_path = config["paths"]["data_path"]
    cluster_names = sorted([c for c in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, c))])
    if config["training"]["load_data_on_gpu"]:
        device_data = "cuda"
    else:
        device_data = "cpu"

    # Load criterion
    criterion = getattr(torch.nn, config["testing"]["criterion"])()
    if criterion is None:
        raise ValueError(f"Criterion '{config['experiment']['criterion']}' not found.")
    criterion.to(device)

    # Test in a leave-one-cluster-out cross-validation fashion
    # Compute mean test loss for each cluster
    mean_test_loss_matrix = np.zeros((len(cluster_names), len(cluster_names)))
    for i, excluded_cluster in enumerate(cluster_names):
        
        # Load model trained on all but excluded cluster
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
        
        model.to(device)
        model.eval()

        # Evaluate on all clusters
        for j, cluster in enumerate(cluster_names):

            # Load dataloaders
            cluster_dataloader = get_single_cluster_dataloader(
                data_path=config["paths"]["data_path"],
                elev_dir=config["paths"]["elev_path"],
                cluster=cluster,
                batch_size=config["training"]["batch_size"],
                num_workers=config["training"]["num_workers"],
                use_theta_e=config["training"]["use_theta_e"],
                device=device_data,
            )
            test_loader = cluster_dataloader["test"]

            evaluation_results = evaluate_model(
                model,
                criterion,
                test_loader,
                device=device,
            )

            # Save evaluation results
            if save_eval:
                np.savez(os.path.join(evaluation_path, f"eval_{excluded_cluster}_on_{cluster}.npz"), **evaluation_results)
                print(f"Evaluation results saved to {evaluation_path}/eval_{excluded_cluster}_on_{cluster}.npz")

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

            # Free up memory
            cluster_dataloader["train"].dataset.unload_from_gpu()            
            cluster_dataloader["val"].dataset.unload_from_gpu()
            cluster_dataloader["test"].dataset.unload_from_gpu()
            del test_loader, cluster_dataloader
            torch.cuda.empty_cache()
            gc.collect()

    np.savez(os.path.join(exp_path, "mean_test_loss_matrix.npz"), mean_test_loss_matrix)
    print("Mean test loss matrix saved to ", os.path.join(exp_path, "mean_test_loss_matrix.npz"))
    plot_mean_test_loss_matrix(mean_test_loss_matrix, cluster_names, exp_path)
    print("Mean test loss matrix plotted and saved.")

if __name__ == "__main__":
    main()
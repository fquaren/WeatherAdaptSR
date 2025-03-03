import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os


def evaluate_and_plot(model, config, test_loader, save_path, device="cuda", save=True):
    """This function evaluates the model on the test dataset, computes test loss, and plots the results.
    """
    model.eval()

    criterion = getattr(torch.nn, config["criterion"])()
    test_losses = []
    predictions, targets = [], []
    
    with torch.no_grad():
        for temperature, elevation, target in test_loader:
            temperature, elevation, target = temperature.to(device), elevation.to(device), target.to(device)
            output = model(temperature, elevation)
            loss = criterion(output, target).item()
            test_losses.append(loss)
            predictions.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
    
    test_losses = np.array(test_losses)
    predictions = np.array(predictions)
    targets = np.array(targets)

    # Save all test losses
    np.save(os.path.join(save_path, "test_losses.npy"), test_losses)
    
    # Get top 5 and bottom 5 examples based on loss
    top_5_idx = test_losses.argsort()[-5:][::-1]  # Highest loss
    bottom_5_idx = test_losses.argsort()[:5]      # Lowest loss
    
    # Plot results
    _, axes = plt.subplots(5, 2, figsize=(10, 15))
    plt.suptitle("Mean Test Loss: {:.4f}".format(test_losses.mean()))
    for i, idx in enumerate(top_5_idx):
        axes[i, 0].imshow(predictions[idx][0], cmap='coolwarm')
        axes[i, 0].set_title(f"Top {i+1} - Prediction (Loss: {test_losses[idx]:.4f})")
        axes[i, 1].imshow(targets[idx][0], cmap='coolwarm')
        axes[i, 1].set_title(f"Top {i+1} - Target")
    
    for i, idx in enumerate(bottom_5_idx):
        axes[i, 0].imshow(predictions[idx][0], cmap='coolwarm')
        axes[i, 0].set_title(f"Bottom {i+1} - Prediction (Loss: {test_losses[idx]:.4f})")
        axes[i, 1].imshow(targets[idx][0], cmap='coolwarm')
        axes[i, 1].set_title(f"Bottom {i+1} - Target")
    
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(save_path, "evaluation_results.png"))

    return test_losses.mean()


def evaluate_and_plot_step_1(model, config, test_loaders, save_path, device="cuda", save=True):
    """
    Evaluates the model on the test datasets from multiple clusters, computes test loss, and plots results.
    
    Args:
        model: The PyTorch model to evaluate.
        config (dict): The configuration containing criterion and other settings.
        test_loaders (dict): A dictionary where keys are cluster names and values are DataLoader instances for testing.
        save_path (str): Directory to save the evaluation results.
        device (str): Device to run the evaluation on ('cpu', 'cuda', etc.).
        save (bool): Whether to save the plots and losses.
    
    Returns:
        float: The mean test loss across all clusters.
    """
    model.eval()
    
    criterion = getattr(torch.nn, config["criterion"])()
    all_test_losses = []
    all_predictions, all_targets = [], []
    
    # Iterate through each cluster
    for cluster_name, test_loader in test_loaders.items():
        test_losses = []
        predictions, targets = [], []
        
        with torch.no_grad():
            for temperature, elevation, target in test_loader:
                temperature, elevation, target = temperature.to(device), elevation.to(device), target.to(device)
                output = model(temperature, elevation)
                loss = criterion(output, target).item()
                test_losses.append(loss)
                predictions.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())
        
        test_losses = np.array(test_losses)
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Save test losses for the current cluster
        np.save(os.path.join(save_path, f"{cluster_name}_test_losses.npy"), test_losses)
        
        all_test_losses.extend(test_losses)  # Accumulate all cluster test losses
        all_predictions.extend(predictions)  # Accumulate all cluster predictions
        all_targets.extend(targets)  # Accumulate all cluster targets

        # Get top 5 and bottom 5 examples based on loss
        top_5_idx = test_losses.argsort()[-5:][::-1]  # Highest loss
        bottom_5_idx = test_losses.argsort()[:5]      # Lowest loss
        
        # Plot results for the current cluster
        _, axes = plt.subplots(5, 2, figsize=(10, 15))
        plt.suptitle(f"Mean Test Loss for {cluster_name}: {test_losses.mean():.4f}")
        
        for i, idx in enumerate(top_5_idx):
            axes[i, 0].imshow(predictions[idx][0], cmap='coolwarm')
            axes[i, 0].set_title(f"Top {i+1} - Prediction (Loss: {test_losses[idx]:.4f})")
            axes[i, 1].imshow(targets[idx][0], cmap='coolwarm')
            axes[i, 1].set_title(f"Top {i+1} - Target")
        
        for i, idx in enumerate(bottom_5_idx):
            axes[i, 0].imshow(predictions[idx][0], cmap='coolwarm')
            axes[i, 0].set_title(f"Bottom {i+1} - Prediction (Loss: {test_losses[idx]:.4f})")
            axes[i, 1].imshow(targets[idx][0], cmap='coolwarm')
            axes[i, 1].set_title(f"Bottom {i+1} - Target")
        
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(save_path, f"{cluster_name}_evaluation_results.png"))
            plt.close()

    # Combine test losses across all clusters and compute the mean
    all_test_losses = np.array(all_test_losses)
    mean_test_loss = all_test_losses.mean()

    # Save the combined test losses
    np.save(os.path.join(save_path, "all_clusters_test_losses.npy"), all_test_losses)
    
    # Final summary
    print(f"Mean Test Loss across all clusters: {mean_test_loss:.4f}")
    
    return mean_test_loss
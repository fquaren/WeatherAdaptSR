import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os


def evaluate_and_plot(model, config, test_loader, save_path, device="cuda", save=True):

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
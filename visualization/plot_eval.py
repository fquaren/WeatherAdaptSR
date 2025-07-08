from matplotlib import pyplot as plt
import numpy as np
import os


def plot_results(
    evaluation_results, eval_on_cluster, cluster_name, save_path, save=True
):
    """
    Plots the worst and best 5 examples based on test loss for a given cluster.
    """
    # Unpack evaluation results
    test_losses = evaluation_results["test_losses"]
    predictions = evaluation_results["predictions"]
    targets = evaluation_results["targets"]
    elevations = evaluation_results["elevations"]
    inputs = evaluation_results["inputs"]

    # Create directory for saving results
    os.makedirs(save_path, exist_ok=True)

    # Get top 5 and bottom 5 indices based on loss
    top_5_idx = test_losses.argsort()[-5:][::-1]
    bottom_5_idx = test_losses.argsort()[:5]

    def plot_subset(indices, title_prefix, filename_suffix):
        fig, axes = plt.subplots(5, 5, figsize=(10, 15))
        plt.suptitle(
            f"{title_prefix} 5 - Mean Test Loss for model trained excluding {eval_on_cluster}\n"
            f"and tested on {cluster_name}: {test_losses.mean():.4f}"
        )

        for i, idx in enumerate(indices):

            input_img = inputs[idx][0]
            pred_img = predictions[idx][0]
            target_img = targets[idx][0]
            elev_img = elevations[idx][0]
            mse = np.mean(np.subtract(pred_img, target_img) ** 2)  # Pixel wise mse

            images = [input_img, pred_img, target_img, mse, elev_img]
            cmaps = ["coolwarm", "coolwarm", "coolwarm", "plasma", "viridis"]
            titles = [
                "Input",
                "Prediction",
                "Target",
                f"Pixel-wise MSE (Mean: {test_losses[idx]:.4f})",
                "Elevation",
            ]

            for j, (data, cmap, title) in enumerate(zip(images, cmaps, titles)):
                img = axes[i, j].imshow(data, cmap=cmap)

                axes[i, j].set_title(title)
                axes[i, j].axis("off")
                cbar = fig.colorbar(img, ax=axes[i, j], fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)

        plt.tight_layout()
        if save:
            plt.savefig(
                os.path.join(
                    save_path,
                    f"evaluation_results_{filename_suffix}_{eval_on_cluster}_{cluster_name}.png",
                )
            )
        plt.close(fig)

    # Plot worst and best
    plot_subset(top_5_idx, "WORST", "worst")
    plot_subset(bottom_5_idx, "BEST", "best")


def plot_training_metrics(
    save_path, evaluation_path, model_architecture, trained_on_label, logger
):
    # Plot training metrics
    train_losses_path = os.path.join(save_path, "train_losses.npy")
    val_losses_path = os.path.join(save_path, "val_losses.npy")

    if not os.path.exists(train_losses_path) or not os.path.exists(val_losses_path):
        logger.warning(
            f"Training loss files not found in {save_path}. Skipping training metrics plot."
        )
        return

    train_losses = np.load(train_losses_path)
    val_losses = np.load(val_losses_path)
    _ = plt.figure()
    plt.title(
        f"Training metrics {model_architecture} model trained on {trained_on_label}"
    )
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.yscale("log")
    plt.savefig(os.path.join(evaluation_path, "training_metrics.png"))
    plt.close()


def plot_eval_matrix(
    mean_eval_matrix, cluster_names, model_architecture, metric, save_path, logger
):
    """
    Plots the mean test loss matrix for all clusters.
    """
    assert (
        mean_eval_matrix.shape[0] == mean_eval_matrix.shape[1]
    ), "Matrix must be square"
    N = mean_eval_matrix.shape[0]

    # 1. Compute the mean of the diagonal (same cluster train-test)
    mean_diagonal = np.mean(np.diag(mean_eval_matrix))
    mean_off_diagonal = np.mean(np.delete(mean_eval_matrix, np.diag_indices(N)))

    # 2. Compute the mean difference for each column
    sum_differences = []
    for i in range(mean_eval_matrix.shape[1]):
        column_values = mean_eval_matrix[:, i]
        diff = column_values[i] - column_values  # Difference from the diagonal value
        # diff = np.maximum(diff, 0)  # Apply relu to difference to avoid negative values
        sum_diff = np.sum(np.delete(diff, i))  # Exclude the diagonal element
        sum_differences.append(sum_diff)

    # Convert to a NumPy array for easy handling
    consistency = 1 / (N * (N - 1)) * np.sum(np.array(sum_differences))

    logger.info(f"EVALUATION: Mean diagonal {metric}: {mean_diagonal}")
    logger.info(f"EVALUATION: Mean off-diagonal {metric}: {mean_off_diagonal}")
    logger.info(f"EVALUATION: Mean overall {metric}: {np.mean(mean_eval_matrix)}")
    logger.info(f"EVALUATION: Consistency metric: {consistency}")
    logger.info(
        f"EVALUATION: Difference Diag-OffDiag: {
            mean_diagonal - mean_off_diagonal}"
    )

    # Plot mean test loss matrix
    cmap = "bwr_r" if metric == "SSIM" else "bwr"
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(
        mean_eval_matrix,
        cmap=cmap,
    )
    plt.colorbar(cax)
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(cluster_names, rotation=45)
    ax.set_yticklabels(cluster_names)
    plt.xlabel("Domain")
    plt.ylabel("Model")
    plt.title(
        f"Model: {model_architecture}\n"
        f"Mean Test Loss Matrix {metric}: {np.mean(mean_eval_matrix):.6f},\n"
        f"Mean Diagonal {metric}: {mean_diagonal:.6f},\n"
        f"Mean Non-Diagonal {metric}: {np.mean(np.delete(mean_eval_matrix, np.diag_indices(N))):.6f},\n"
        f"Difference Mean Diag & Mean Non-Diagonal {metric}: {
            np.mean(mean_eval_matrix) - np.mean(np.delete(mean_eval_matrix, np.diag_indices(N)))},\n"
        f"Consistency: {consistency:.6f}"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{metric}_matrix.png"))
    plt.close(fig)  # Close the figure to free memory

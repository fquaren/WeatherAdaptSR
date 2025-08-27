import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os


# def plot_results(
#     evaluation_results, eval_on_cluster, cluster_name, save_path, save=True
# ):
#     """
#     Plots the worst and best 5 examples based on test loss for a given cluster.
#     """
#     # Unpack evaluation results
#     test_losses = evaluation_results["test_losses"]
#     predictions = evaluation_results["predictions"]
#     targets = evaluation_results["targets"]
#     elevations = evaluation_results["elevations"]
#     inputs = evaluation_results["inputs"]

#     # Create directory for saving results
#     os.makedirs(save_path, exist_ok=True)

#     # Get top 5 and bottom 5 indices based on loss
#     top_5_idx = test_losses.argsort()[-5:][::-1]
#     bottom_5_idx = test_losses.argsort()[:5]

#     def plot_subset(indices, title_prefix, filename_suffix):
#         fig, axes = plt.subplots(5, 5, figsize=(10, 15))
#         plt.suptitle(
#             f"{title_prefix} 5 - Mean Test Loss for model trained excluding {eval_on_cluster}\n"
#             f"and tested on {cluster_name}: {test_losses.mean():.4f}"
#         )

#         for i, idx in enumerate(indices):

#             input_img = inputs[idx][0]
#             pred_img = predictions[idx][0]
#             target_img = targets[idx][0]
#             elev_img = elevations[idx][0]
#             mse_img = np.subtract(pred_img, target_img) ** 2  # Pixel wise mse

#             images = [input_img, pred_img, target_img, mse_img, elev_img]
#             vmin = min(img.min() for img in images[:3])
#             vmax = max(img.max() for img in images[:3])
#             cmaps = ["coolwarm", "coolwarm", "coolwarm", "plasma", "viridis"]
#             titles = [
#                 "Input",
#                 "Prediction",
#                 "Target",
#                 f"Pixel-wise MSE (Mean: {np.mean(mse_img):.4f})",
#                 "Elevation",
#             ]

#             for j, (data, cmap, title) in enumerate(zip(images, cmaps, titles)):
#                 if j < 3:
#                     img = axes[i, j].imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
#                 else:
#                     img = axes[i, j].imshow(data, cmap=cmap)

#                 axes[i, j].set_title(title)
#                 axes[i, j].axis("off")
#                 cbar = fig.colorbar(img, ax=axes[i, j], fraction=0.046, pad=0.04)
#                 cbar.ax.tick_params(labelsize=8)

#         plt.tight_layout()
#         if save:
#             plt.savefig(
#                 os.path.join(
#                     save_path,
#                     f"evaluation_results_{filename_suffix}_{eval_on_cluster}_{cluster_name}.png",
#                 )
#             )
#         plt.close(fig)


#     # Plot worst and best
#     plot_subset(top_5_idx, "WORST", "worst")
#     plot_subset(bottom_5_idx, "BEST", "best")


def plot_results(
    evaluation_results, eval_on_cluster, cluster_name, save_path, save=True
):
    """
    Plots the worst and best 5 examples for each variable (T_2M and TOT_PREC)
    on a single figure.
    """
    # Unpack evaluation results
    test_losses = evaluation_results["test_losses"]
    predictions = evaluation_results["predictions"]
    targets = evaluation_results["targets"]
    inputs = evaluation_results["inputs"]

    # Create directory for saving results
    os.makedirs(save_path, exist_ok=True)

    # Get top 5 and bottom 5 indices based on loss
    top_5_idx = test_losses.argsort()[-5:][::-1]
    bottom_5_idx = test_losses.argsort()[:5]

    def plot_subset(indices, title_prefix, filename_suffix):
        fig, axes = plt.subplots(10, 4, figsize=(10, 25))
        plt.suptitle(
            f"{title_prefix} 5 examples for T_2M and TOT_PREC\n"
            f"Mean Test Loss for model trained excluding {eval_on_cluster}\n"
            f"and tested on {cluster_name}: {test_losses.mean():.4f}"
        )

        variable_names = ["T_2M", "TOT_PREC"]
        titles = ["Input", "Prediction", "Target", "Elevation"]
        cmaps = ["coolwarm", "coolwarm", "coolwarm", "plasma"]

        for var_idx, var_name in enumerate(variable_names):
            for i, idx in enumerate(indices):
                row_offset = var_idx * 5 + i

                # Extract specific channel for the current variable
                input_img = inputs[idx][var_idx]
                pred_img = predictions[idx][var_idx]
                target_img = targets[idx][var_idx]
                elev_img = inputs[idx][-1]

                images = [input_img, pred_img, target_img, elev_img]
                vmin = np.amin(target_img)
                vmax = np.amax(target_img)

                for j, (data, title, cmap) in enumerate(zip(images, titles, cmaps)):
                    ax = axes[row_offset, j]
                    if j < 3:
                        img = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
                    else:
                        img = ax.imshow(data, cmap=cmap)

                    ax.set_title(f"{var_name} - {title}")
                    ax.axis("off")
                    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
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

    # Plot temp training metrics
    train_losses_path = os.path.join(save_path, "train_temp_losses.npy")
    val_losses_path = os.path.join(save_path, "val_temp_losses.npy")

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
    plt.savefig(os.path.join(evaluation_path, "training_temp_metrics.png"))
    plt.close()

    # Plot precip training metrics
    train_losses_path = os.path.join(save_path, "train_precip_losses.npy")
    val_losses_path = os.path.join(save_path, "val_precip_losses.npy")

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
    plt.savefig(os.path.join(evaluation_path, "training_precip_metrics.png"))
    plt.close()

    # Plot precip training metrics
    b_T_path = os.path.join(save_path, "b_T.npy")
    b_P_path = os.path.join(save_path, "b_P.npy")
    val_b_T_path = os.path.join(save_path, "val_b_T.npy")
    val_b_P_path = os.path.join(save_path, "val_b_P.npy")

    if not os.path.exists(b_T_path) or not os.path.exists(b_P_path):
        logger.warning(
            f"Training loss files not found in {save_path}. Skipping training metrics plot."
        )
        return

    b_T = np.load(b_T_path)
    val_b_T = np.load(val_b_T_path)
    b_P = np.load(b_P_path)
    val_b_P = np.load(val_b_P_path)
    _ = plt.figure()
    plt.title(
        f"Training metrics {model_architecture} model trained on {trained_on_label}"
    )
    plt.plot(np.log(b_T), linestyle=":", label="Train b_T")
    plt.plot(np.log(val_b_T), linestyle="-", label="Validation b_T")
    plt.plot(np.log(b_P), linestyle="-.", label="Train b_P")
    plt.plot(np.log(val_b_P), linestyle="--", label="Validation b_P")
    plt.xlabel("Epoch")
    plt.ylabel("Parameter value")
    plt.yscale("log")
    plt.legend()
    plt.savefig(os.path.join(evaluation_path, "log_params.png"))
    plt.close()


def plot_eval_matrix(mean_eval_matrix, cluster_names, metric, save_path):
    """
    Plots the mean test loss matrix for all clusters, highlighting the diagonal.
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
        diff = np.abs(column_values[i] - column_values)
        sum_diff = np.sum(np.delete(diff, i))
        sum_differences.append(sum_diff)

    consistency = 1 / (N * (N - 1)) * np.sum(np.array(sum_differences))

    print(f"EVALUATION: Mean diagonal {metric}: {mean_diagonal}")
    print(f"EVALUATION: Mean off-diagonal {metric}: {mean_off_diagonal}")
    print(f"EVALUATION: Mean overall {metric}: {np.mean(mean_eval_matrix)}")
    print(f"EVALUATION: Consistency metric: {consistency}")
    print(
        f"EVALUATION: Absolute Difference Diag-OffDiag: {np.abs(mean_diagonal - mean_off_diagonal)}"
    )

    # Plot mean test loss matrix
    cmap = "bwr_r" if metric == "SSIM" else "bwr"
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(
        mean_eval_matrix,
        cmap=cmap,
    )
    plt.colorbar(cax, label="MAE (K)")
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(cluster_names, rotation=45, ha="left")
    ax.set_yticklabels(cluster_names)
    plt.xlabel("Model evaluated on:")
    plt.ylabel("Model trained on: ")
    plt.title(
        f"Model: UNet\n"
        f"Mean Test Loss Matrix {metric}: {np.mean(mean_eval_matrix):.6f},\n"
        f"Mean Diagonal {metric}: {mean_diagonal:.6f},\n"
        f"Mean Non-Diagonal {metric}: {mean_off_diagonal:.6f},\n"
        f"Difference Mean Diag & Mean Non-Diagonal {metric}: {mean_diagonal - mean_off_diagonal:.6f},\n"
        f"Consistency: {consistency:.6f}"
    )

    # Highlight the diagonal cells
    for i in range(N):
        rect = patches.Rectangle(
            (i - 0.5, i - 0.5),
            1,
            1,
            linewidth=3,
            edgecolor="y",
            facecolor="none",
            alpha=1,
        )
        ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{metric}.png"))
    plt.close(fig)

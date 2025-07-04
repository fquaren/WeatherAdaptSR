import torch
import numpy as np
from matplotlib import pyplot as plt
import os
import argparse
import yaml
from tqdm import tqdm
import gc
import logging
from data.dataloader import get_single_cluster_dataloader
from src.models import unet
from src.logger import setup_logger


def spectral_nrmse_from_fields(observation, forecast):
    """
    Compute normalized RMSE between the radially averaged power spectra
    of observation and forecast fields.

    Returns:
        nrmse: float
    """

    def radial_power(field):
        # Remove mean
        field = field - np.mean(field)
        # 2D FFT and shift
        ps = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2
        # Radial average
        ny, nx = ps.shape
        y, x = np.indices((ny, nx))
        r = np.sqrt((x - nx // 2) ** 2 + (y - ny // 2) ** 2).astype(np.int32)
        tbin = np.bincount(r.ravel(), ps.ravel())
        nr = np.bincount(r.ravel())
        return tbin / (nr + 1e-8)

    radial_obs = radial_power(observation)
    radial_pred = radial_power(forecast)

    # NRMSE
    num = np.sqrt(np.sum((radial_obs - radial_pred) ** 2))
    denom = np.sqrt(np.sum(radial_obs**2)) + 1e-8
    nrmse = num / denom

    return nrmse


def compute_ssim(x, y, C1=0.01**2, C2=0.03**2):
    """
    Compute SSIM (Structural Similarity Index) between two images (2D tensors).

    Args:
        x, y: input images (2D tensors), assumed to be normalized to [0, 1]
        C1, C2: SSIM constants (default values from the original paper)

    Returns:
        SSIM index (scalar)
    """
    assert x.shape == y.shape, "SSIM: input shapes must match"

    x = x.float()
    y = y.float()

    mu_x = x.mean()
    mu_y = y.mean()

    sigma_x = ((x - mu_x) ** 2).mean()
    sigma_y = ((y - mu_y) ** 2).mean()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

    return numerator / denominator


def evaluate_model(model, criterion, test_loader, device="cuda"):
    """
    Evaluates the model on the test datasets from multiple clusters, computes test loss, and plots results.

    Args:
        model: The PyTorch model to evaluate.
        criterion: The loss function.
        test_loader (array): Array where values are DataLoader instances for testing.
        device (str): Device to run the evaluation on ('cpu', 'cuda', etc.).

    Returns:
        dict: Evaluation results containing test_losses, predictions, targets, elevations, inputs.
    """

    test_losses, predictions, targets, elevations, inputs = [], [], [], [], []
    with torch.no_grad():
        for temperature, elevation, target in tqdm(test_loader):
            # TODO: get file name from the dataloader
            temperature, elevation, target = (
                temperature.to(device),
                elevation.to(device),
                target.to(device),
            )
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
        "inputs": inputs,
    }

    return evaluation_results


def evaluate_model_mmd(model, criterion, test_loader, device="cuda"):
    """
    Evaluates the model on the test datasets from multiple clusters, computes test loss, and plots results.

    Args:
        model: The PyTorch model to evaluate.
        criterion: The loss function.
        test_loader (array): Array where values are DataLoader instances for testing.
        device (str): Device to run the evaluation on ('cpu', 'cuda', etc.).

    Returns:
        dict: Evaluation results containing test_losses, predictions, targets, elevations, inputs.
    """

    test_losses, predictions, targets, elevations, inputs = [], [], [], [], []
    with torch.no_grad():
        for temperature, elevation, target in tqdm(test_loader):
            # TODO: get file name from the dataloader
            temperature, elevation, target = (
                temperature.to(device),
                elevation.to(device),
                target.to(device),
            )
            output, _ = model(
                temperature,
                elevation,
                target_variable=torch.zeros_like(temperature),
                target_elevation=torch.zeros_like(elevation),
            )
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
        "inputs": inputs,
    }

    return evaluation_results


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
        fig, axes = plt.subplots(5, 4, figsize=(10, 15))
        plt.suptitle(
            f"{title_prefix} 5 - Mean Test Loss for model trained excluding {eval_on_cluster}\n"
            f"and tested on {cluster_name}: {test_losses.mean():.4f}"
        )

        for i, idx in enumerate(indices):
            input_img = inputs[idx][0]
            pred_img = predictions[idx][0]
            target_img = targets[idx][0]
            elev_img = elevations[idx][0]

            images = [input_img, pred_img, target_img, elev_img]
            cmaps = ["coolwarm", "coolwarm", "coolwarm", "viridis"]
            titles = [
                "Input",
                f"Prediction (Loss: {test_losses[idx]:.4f})",
                "Target",
                "Elevation",
            ]

            for j, (data, cmap, title) in enumerate(zip(images, cmaps, titles)):
                if j < 3:
                    img = axes[i, j].imshow(data, cmap=cmap)
                else:
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


def main():
    # Get model path from arg command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for evaluation (cpu or cuda)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model architecture to use for evaluation",
    )
    parser.add_argument(
        "--exp_path", type=str, default=None, help="Path of model to evaluate"
    )
    parser.add_argument(
        "--local", type=str, default=None, help="Evaluation on local machine"
    )
    parser.add_argument(
        "--num_workers", type=int, default=None, help="Number of workers (optional)"
    )
    parser.add_argument(
        "--save_eval",
        default=True,
        action="store_true",
        help="Save evaluation results to disk",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="cross-val",
        help="Method name (single, all, cross-val, mmd)",
    )
    args = parser.parse_args()

    save_eval = args.save_eval
    method = args.method

    exp_path = args.exp_path
    if exp_path is None:
        raise ValueError(
            "Please provide the path to the model to evaluate using --exp_path"
        )

    LOGGER = setup_logger(exp_path)
    LOGGER.info(f"EVALUATION: Using model path: {exp_path}")

    evaluation_path = os.path.join(exp_path, "evaluation_results")
    os.makedirs(evaluation_path, exist_ok=True)

    # Load config file
    config_path = os.path.join(exp_path, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file {config_path} does not exist. Please provide a valid path."
        )
    LOGGER.info(f"EVALUATION: Loading config from: {config_path}")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Num workers
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = config["training"]["num_workers"]
    else:
        num_workers = args.num_workers
    LOGGER.info(f"EVALUATION: Num workers: {num_workers}")

    # Set device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("INFO: CUDA is not available. Using CPU instead.")
        device = "cpu"
    elif device == "cuda":
        LOGGER.info(f"INFO: Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        LOGGER.info("INFO: Using CPU.")
    LOGGER.info(f"EVALUATION: Device: {device}")

    # Load model architecture
    model_architecture = args.model
    if not hasattr(unet, model_architecture):
        raise ValueError(
            f"Model architecture '{model_architecture}' not found in unet.py module."
        )
    if not callable(getattr(unet, model_architecture)):
        raise ValueError(f"Model architecture '{model_architecture}' is not callable.")

    # Data paths
    data_path = config["paths"]["data_path"]
    elev_dir = config["paths"]["elev_path"]
    cluster_names = config["paths"]["clusters"]
    if cluster_names is None:
        cluster_names = sorted(
            [
                c
                for c in os.listdir(data_path)
                if os.path.isdir(os.path.join(data_path, c))
            ]
        )

    # Loss function (assuming MSELoss for now, can be made configurable)
    criterion = torch.nn.MSELoss()

    if method == "single":
        LOGGER.info(
            "EVALUATION: Starting 'single' method evaluation (evaluating on all clusters)..."
        )
        single_cluster_name = config["paths"]["single_cluster"]
        if single_cluster_name is None:
            raise ValueError(
                "For 'single' method, 'single_cluster' must be specified in the config file under 'paths'."
            )

        model_save_path = os.path.join(exp_path, single_cluster_name)
        model_state_dict_path = os.path.join(model_save_path, "best_snapshot.pth")

        if not os.path.exists(model_state_dict_path):
            LOGGER.error(
                f"EVALUATION: Model not found for single cluster '{single_cluster_name}' \
                    at {model_state_dict_path}. Exiting."
            )
            return

        model = getattr(unet, model_architecture)()
        checkpoint = torch.load(model_state_dict_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        # Plot training metrics for this model
        evaluation_save_path = os.path.join(
            exp_path, "evaluation_results", f"model_trained_on_{single_cluster_name}"
        )
        os.makedirs(evaluation_save_path, exist_ok=True)
        plot_training_metrics(
            model_save_path,
            evaluation_save_path,
            model_architecture,
            single_cluster_name,
            LOGGER,
        )

        mean_eval_losses = []
        mean_eval_ssims = []
        mean_eval_snrmse = []

        for j, test_cluster in enumerate(cluster_names):
            LOGGER.info(
                f"EVALUATION: Evaluating model trained on {single_cluster_name} on test data from {test_cluster}..."
            )

            # Get test loader for the current cluster
            loaders = get_single_cluster_dataloader(
                data_path=data_path,
                elev_dir=elev_dir,
                cluster=test_cluster,
                batch_size=config["training"]["batch_size"],
                num_workers=num_workers,
                use_theta_e=config["training"]["use_theta_e"],
                device="cpu",  # Always load data to CPU first
                augment=False,  # No augmentation for evaluation
            )
            test_loader = loaders["test"]

            # Decide which evaluate function to use based on model type (e.g., if it was trained with MMD)
            # For simplicity here, assuming a standard UNet. If config has use_mmd_loss, you'd use evaluate_model_mmd.
            # Example:
            # if config["training"].get("use_mmd_loss", False): # Assuming a flag in config
            #     results = evaluate_model_mmd(model, criterion, test_loader, device)
            # else:
            #     results = evaluate_model(model, criterion, test_loader, device)
            results = evaluate_model(model, criterion, test_loader, device)

            mean_test_loss = np.mean(results["test_losses"])
            mean_eval_losses.append(mean_test_loss)

            # Compute SSIM
            ssim_scores = []
            for k in range(results["predictions"].shape[0]):
                ssim_scores.append(
                    compute_ssim(
                        torch.from_numpy(results["predictions"][k, 0]),
                        torch.from_numpy(results["targets"][k, 0]),
                    ).item()
                )

            # Compute SSIM
            snrmse_scores = []
            for k in range(results["predictions"].shape[0]):
                snrmse_scores.append(
                    spectral_nrmse_from_fields(
                        results["predictions"][k, 0, :, :],
                        results["targets"][k, 0, :, :],
                    )
                )

            mean_ssim = np.mean(ssim_scores)
            mean_eval_ssims.append(mean_ssim)
            mean_eval_snrmse.append(snrmse_scores)

            LOGGER.info(
                f"EVALUATION: Mean test loss for {test_cluster}: {mean_test_loss:.4f}"
            )
            LOGGER.info(f"EVALUATION: Mean SSIM for {test_cluster}: {mean_ssim:.4f}")

            if save_eval:
                plot_results(
                    results,
                    single_cluster_name,
                    test_cluster,
                    evaluation_save_path,
                    save=True,
                )

            # Clean up
            _ = loaders["test"].dataset.unload_from_gpu()
            del loaders
            torch.cuda.empty_cache()
            gc.collect()
            LOGGER.info(f"EVALUATION: GPU memory emptied for cluster: {test_cluster}")

        # Clean up for the model
        del model
        torch.cuda.empty_cache()
        gc.collect()

        LOGGER.info(
            "\n--- Overall Results for 'single' method model evaluated on all clusters ---"
        )
        LOGGER.info(f"Model trained on: {single_cluster_name}")
        for idx, cluster in enumerate(cluster_names):
            LOGGER.info(
                f"  Tested on {cluster} | Mean MSE Loss: {mean_eval_losses[idx]:.4f} \
                    | Mean SSIM: {mean_eval_ssims[idx]:.4f} | Mean rNRMSE: {mean_eval_snrmse[idx]:.4f} "
            )
        LOGGER.info(
            f"Overall Average MSE Loss across all test clusters: {np.mean(mean_eval_losses):.4f}"
        )
        LOGGER.info(
            f"Overall Average SSIM across all test clusters: {np.mean(mean_eval_ssims):.4f}"
        )
        LOGGER.info(
            f"Overall Average sNRMSE across all test clusters: {np.mean(mean_eval_snrmse):.4f}"
        )

    elif method == "cross-val":
        LOGGER.info("EVALUATION: Starting cross-validation evaluation...")
        mean_eval_matrix = np.zeros((len(cluster_names), len(cluster_names)))
        ssim_eval_matrix = np.zeros((len(cluster_names), len(cluster_names)))
        snrmse_eval_matrix = np.zeros((len(cluster_names), len(cluster_names)))

        for i, excluded_cluster in enumerate(cluster_names):
            model_save_path = os.path.join(exp_path, excluded_cluster)
            model_state_dict_path = os.path.join(model_save_path, "best_snapshot.pth")

            if not os.path.exists(model_state_dict_path):
                LOGGER.warning(
                    f"EVALUATION: Model not found for excluded cluster {excluded_cluster} \
                        at {model_state_dict_path}. Skipping."
                )
                continue

            model = getattr(unet, model_architecture)()
            checkpoint = torch.load(model_state_dict_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()

            # Plot training metrics for this model
            evaluation_save_path = os.path.join(
                exp_path, "evaluation_results", excluded_cluster
            )
            os.makedirs(evaluation_save_path, exist_ok=True)
            plot_training_metrics(
                model_save_path,
                evaluation_save_path,
                model_architecture,
                excluded_cluster,
                LOGGER,
            )

            for j, test_cluster in enumerate(cluster_names):
                LOGGER.info(
                    f"EVALUATION: Evaluating model trained excluding {excluded_cluster} \
                        on test data from {test_cluster}..."
                )

                # Get test loader for the current cluster
                loaders = get_single_cluster_dataloader(
                    data_path=data_path,
                    elev_dir=elev_dir,
                    cluster_name=test_cluster,
                    batch_size=config["training"]["batch_size"],
                    num_workers=num_workers,
                    use_theta_e=config["training"]["use_theta_e"],
                    device="cpu",
                    stats_path="crossval_normalization_stats.json",
                    augment=False,
                )
                test_loader = loaders["test"]

                results = evaluate_model(model, criterion, test_loader, device)
                mean_test_loss = np.mean(results["test_losses"])
                mean_eval_matrix[i, j] = mean_test_loss

                # Compute SSIM
                ssim_scores = []
                for k in range(results["predictions"].shape[0]):
                    ssim_scores.append(
                        compute_ssim(
                            torch.from_numpy(results["predictions"][k, 0]),
                            torch.from_numpy(results["targets"][k, 0]),
                        ).item()
                    )
                ssim_eval_matrix[i, j] = np.mean(ssim_scores)

                # Compute normalized spectral error
                snrmse_scores = []
                for k in range(results["predictions"].shape[0]):
                    snrmse_scores.append(
                        spectral_nrmse_from_fields(
                            results["predictions"][k, 0, :, :],
                            results["targets"][k, 0, :, :],
                        )
                    )
                snrmse_eval_matrix[i, j] = np.mean(snrmse_scores)

                LOGGER.info(
                    f"EVALUATION: Mean test loss for {test_cluster}: {mean_test_loss:.4f}"
                )
                LOGGER.info(
                    f"EVALUATION: Mean SSIM for {test_cluster}: {ssim_eval_matrix[i, j]:.4f}"
                )
                LOGGER.info(
                    f"EVALUATION: Mean sNRMSE for {test_cluster}: {snrmse_eval_matrix[i, j]:.4f}"
                )

                if save_eval:
                    # Save plots for each test cluster
                    plot_results(
                        results,
                        excluded_cluster,
                        test_cluster,
                        evaluation_save_path,
                        save=True,
                    )

                # Clean up
                _ = loaders["test"].dataset.unload_from_gpu()
                del loaders
                torch.cuda.empty_cache()
                gc.collect()
                LOGGER.info(
                    f"EVALUATION: GPU memory emptied for cluster: {test_cluster}"
                )

            # Clean up for the model
            del model
            torch.cuda.empty_cache()
            gc.collect()

        # Plot overall evaluation matrices
        plot_eval_matrix(
            mean_eval_matrix,
            cluster_names,
            model_architecture,
            "MSE",
            os.path.join(exp_path, "evaluation_results"),
            LOGGER,
        )

        def standardize(matrix):
            # Standardize each row between 0 and 1
            row_min = matrix.min(axis=1, keepdims=True)
            row_max = matrix.max(axis=1, keepdims=True)
            denominator = row_max - row_min
            # To avoid division by zero if all elements in the row are identical
            denominator[denominator == 0] = 1
            return (matrix - row_min) / denominator

        plot_eval_matrix(
            standardize(mean_eval_matrix),
            cluster_names,
            model_architecture,
            "standardized_MSE",
            os.path.join(exp_path, "evaluation_results"),
            LOGGER,
        )
        plot_eval_matrix(
            ssim_eval_matrix,
            cluster_names,
            model_architecture,
            "SSIM",
            os.path.join(exp_path, "evaluation_results"),
            LOGGER,
        )
        plot_eval_matrix(
            standardize(ssim_eval_matrix),
            cluster_names,
            model_architecture,
            "standardized_SSIM",
            os.path.join(exp_path, "evaluation_results"),
            LOGGER,
        )
        plot_eval_matrix(
            snrmse_eval_matrix,
            cluster_names,
            model_architecture,
            "sNRMSE",
            os.path.join(exp_path, "evaluation_results"),
            LOGGER,
        )
        plot_eval_matrix(
            standardize(snrmse_eval_matrix),
            cluster_names,
            model_architecture,
            "standardized_sNRMSE",
            os.path.join(exp_path, "evaluation_results"),
            LOGGER,
        )

    elif method == "all":
        LOGGER.info("EVALUATION: Starting 'all' method evaluation...")
        mean_eval_matrix = np.zeros(
            (1, len(cluster_names))
        )  # Single model trained on all, evaluated on each
        ssim_eval_matrix = np.zeros((1, len(cluster_names)))
        snrmse_eval_matrix = np.zeros((1, len(cluster_names)))

        model_save_path = os.path.join(
            exp_path, "all_clusters"
        )  # Assuming "all_clusters" is the save directory for the 'all' method
        model_state_dict_path = os.path.join(model_save_path, "best_snapshot.pth")

        if not os.path.exists(model_state_dict_path):
            LOGGER.error(
                f"EVALUATION: Model not found for 'all_clusters' method at {model_state_dict_path}. Exiting."
            )
            return

        model = getattr(unet, model_architecture)()
        checkpoint = torch.load(model_state_dict_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        # Plot training metrics for this model
        evaluation_save_path = os.path.join(
            exp_path, "evaluation_results", "all_clusters_model"
        )  # Distinct folder for this model's evaluation
        os.makedirs(evaluation_save_path, exist_ok=True)
        plot_training_metrics(
            model_save_path,
            evaluation_save_path,
            model_architecture,
            "all_clusters",
            LOGGER,
        )

        for j, test_cluster in enumerate(cluster_names):
            LOGGER.info(
                f"EVALUATION: Evaluating model trained on all clusters on test data from {test_cluster}..."
            )

            # Get test loader for the current cluster
            # Use get_single_cluster_dataloader for evaluation on individual test clusters
            loaders = get_single_cluster_dataloader(
                data_path=data_path,
                elev_dir=elev_dir,
                cluster=test_cluster,
                batch_size=config["training"]["batch_size"],
                num_workers=num_workers,
                use_theta_e=config["training"]["use_theta_e"],
                device="cpu",  # Always load data to CPU first
                stats_path="crossval_normalization_stats.json",
                augment=False,  # No augmentation for evaluation
            )
            test_loader = loaders["test"]

            results = evaluate_model(model, criterion, test_loader, device)
            mean_test_loss = np.mean(results["test_losses"])
            mean_eval_matrix[0, j] = (
                mean_test_loss  # Store in the first row as there's only one model
            )

            # Compute SSIM
            ssim_scores = []
            for k in range(results["predictions"].shape[0]):
                ssim_scores.append(
                    compute_ssim(
                        torch.from_numpy(results["predictions"][k, 0]),
                        torch.from_numpy(results["targets"][k, 0]),
                    ).item()
                )
            ssim_eval_matrix[0, j] = np.mean(ssim_scores)  # Store in the first row

            # Compute sNRMSE
            snrmse_scores = []
            for k in range(results["predictions"].shape[0]):
                snrmse_scores.append(
                    spectral_nrmse_from_fields(
                        results["predictions"][k, 0, :, :],
                        results["targets"][k, 0, :, :],
                    )
                )
            snrmse_eval_matrix[0, j] = np.mean(snrmse_scores)  # Store in the first row

            LOGGER.info(
                f"EVALUATION: Mean test loss for {test_cluster}: {mean_test_loss:.4f}"
            )
            LOGGER.info(
                f"EVALUATION: Mean SSIM for {test_cluster}: {ssim_eval_matrix[0, j]:.4f}"
            )
            LOGGER.info(
                f"EVALUATION: Mean sNRMSE for {test_cluster}: {snrmse_eval_matrix[0, j]:.4f}"
            )

            if save_eval:
                # Save plots for each test cluster
                plot_results(
                    results,
                    "all_clusters_model",
                    test_cluster,
                    evaluation_save_path,
                    save=True,
                )

            # Clean up
            _ = loaders["test"].dataset.unload_from_gpu()
            del loaders
            torch.cuda.empty_cache()
            gc.collect()
            LOGGER.info(f"EVALUATION: GPU memory emptied for cluster: {test_cluster}")

        # Clean up for the model
        del model
        torch.cuda.empty_cache()
        gc.collect()

        # For "all" method, the matrix will be 1xN, so adjust plotting or just log the results
        LOGGER.info("\n--- Overall Results for 'all' method ---")
        for idx, cluster in enumerate(cluster_names):
            LOGGER.info(
                f"Cluster: {cluster} | Mean MSE Loss: {mean_eval_matrix[0, idx]:.4f} \
                    | Mean SSIM: {ssim_eval_matrix[0, idx]:.4f}"
            )
        LOGGER.info(f"Overall Average MSE Loss: {np.mean(mean_eval_matrix):.4f}")
        LOGGER.info(f"Overall Average SSIM: {np.mean(ssim_eval_matrix):.4f}")

    elif method == "mmd":
        LOGGER.info("EVALUATION: Starting MMD evaluation...")
        mean_eval_matrix = np.zeros((len(cluster_names), len(cluster_names)))
        ssim_eval_matrix = np.zeros((len(cluster_names), len(cluster_names)))
        snrmse_eval_matrix = np.zeros((len(cluster_names), len(cluster_names)))

        for i, excluded_cluster in enumerate(cluster_names):
            model_save_path = os.path.join(exp_path, excluded_cluster)
            model_state_dict_path = os.path.join(model_save_path, "best_snapshot.pth")

            if not os.path.exists(model_state_dict_path):
                LOGGER.warning(
                    f"EVALUATION: Model not found for excluded cluster {excluded_cluster} \
                        at {model_state_dict_path}. Skipping."
                )
                continue

            model = getattr(unet, model_architecture)()
            checkpoint = torch.load(model_state_dict_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()

            # Plot training metrics for this model
            evaluation_save_path = os.path.join(
                exp_path, "evaluation_results", excluded_cluster
            )
            os.makedirs(evaluation_save_path, exist_ok=True)
            plot_training_metrics(
                model_save_path,
                evaluation_save_path,
                model_architecture,
                excluded_cluster,
                LOGGER,
            )

            for j, test_cluster in enumerate(cluster_names):
                LOGGER.info(
                    f"EVALUATION: Evaluating model trained excluding {excluded_cluster} \
                        with MMD on test data from {test_cluster}..."
                )

                # Get test loader for the current cluster
                loaders = get_single_cluster_dataloader(
                    data_path=data_path,
                    elev_dir=elev_dir,
                    cluster=test_cluster,
                    batch_size=config["training"]["batch_size"],
                    num_workers=num_workers,
                    use_theta_e=config["training"]["use_theta_e"],
                    device="cpu",  # Always load data to CPU first
                    stats_path="da_normalization_stats.json",
                    augment=False,  # No augmentation for evaluation
                )
                test_loader = loaders["test"]

                results = evaluate_model_mmd(model, criterion, test_loader, device)
                mean_test_loss = np.mean(results["test_losses"])
                mean_eval_matrix[i, j] = mean_test_loss

                # Compute SSIM
                ssim_scores = []
                for k in range(results["predictions"].shape[0]):
                    ssim_scores.append(
                        compute_ssim(
                            torch.from_numpy(results["predictions"][k, 0]),
                            torch.from_numpy(results["targets"][k, 0]),
                        ).item()
                    )
                ssim_eval_matrix[i, j] = np.mean(ssim_scores)

                # Compute sNRMSE
                snrmse_scores = []
                for k in range(results["predictions"].shape[0]):
                    snrmse_scores.append(
                        spectral_nrmse_from_fields(
                            results["predictions"][k, 0, :, :],
                            results["targets"][k, 0, :, :],
                        )
                    )
                snrmse_eval_matrix[0, j] = np.mean(snrmse_scores)

                LOGGER.info(
                    f"EVALUATION: Mean test loss for {test_cluster}: {mean_test_loss:.4f}"
                )
                LOGGER.info(
                    f"EVALUATION: Mean SSIM for {test_cluster}: {ssim_eval_matrix[i, j]:.4f}"
                )
                LOGGER.info(
                    f"EVALUATION: Mean sNRMSE for {test_cluster}: {snrmse_eval_matrix[i, j]:.4f}"
                )

                if save_eval:
                    # Save plots for each test cluster
                    plot_results(
                        results,
                        excluded_cluster,
                        test_cluster,
                        evaluation_save_path,
                        save=True,
                    )

                # Clean up
                _ = loaders["test"].dataset.unload_from_gpu()
                del loaders
                torch.cuda.empty_cache()
                gc.collect()
                LOGGER.info(
                    f"EVALUATION: GPU memory emptied for cluster: {test_cluster}"
                )

            # Clean up for the model
            del model
            torch.cuda.empty_cache()
            gc.collect()

        # Plot overall evaluation matrices
        plot_eval_matrix(
            mean_eval_matrix,
            cluster_names,
            model_architecture,
            "MSE",
            evaluation_save_path,
            LOGGER,
        )
        plot_eval_matrix(
            ssim_eval_matrix,
            cluster_names,
            model_architecture,
            "SSIM",
            evaluation_save_path,
            LOGGER,
        )
        plot_eval_matrix(
            ssim_eval_matrix,
            cluster_names,
            model_architecture,
            "sNRMSE",
            evaluation_save_path,
            LOGGER,
        )

    else:
        LOGGER.error(
            f"EVALUATION: Unknown method '{method}'. Please use 'single', 'all', 'cross-val', or 'mmd'."
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)  # Basic logger setup for main execution
    main()

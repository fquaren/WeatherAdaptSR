import matplotlib.pyplot as plt
import os
import yaml
import numpy as np
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from scipy.stats import pearsonr
import random


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

    x = x
    y = y

    mu_x = x.mean()
    mu_y = y.mean()

    sigma_x = ((x - mu_x) ** 2).mean()
    sigma_y = ((y - mu_y) ** 2).mean()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

    return numerator / denominator


def calculate_pearson_correlation(array1, array2):
    """
    Computes the Pearson correlation coefficient and p-value between two NumPy arrays.

    Args:
        array1 (np.ndarray): The first NumPy array.
        array2 (np.ndarray): The second NumPy array.

    Returns:
        tuple: A tuple containing:
            - pearson_coeff (float): The Pearson correlation coefficient.
            - p_value (float): The two-tailed p-value.
        Raises:
            ValueError: If the input arrays are not 1-dimensional or do not have the same length.
    """
    # Ensure arrays are 1-dimensional
    if array1.ndim != 1 or array2.ndim != 1:
        raise ValueError("Input arrays must be 1-dimensional.")

    # Ensure arrays have the same length
    if len(array1) != len(array2):
        raise ValueError("Input arrays must have the same length.")

    # Calculate Pearson correlation coefficient and p-value
    pearson_coeff, p_value = pearsonr(array1, array2)

    return pearson_coeff, p_value


# --- Wasserstein Distance (Earth Mover's Distance) ---
def calculate_wasserstein_distance(p_data, q_data):
    """
    Calculates the Wasserstein-1 distance (Earth Mover's Distance) between two 1D distributions.
    """
    return wasserstein_distance(p_data, q_data)


# --- Plotting Code ---
def plot_temperature_distributions(
    num_clusters, temp_min, temp_max, config, splits=["train", "val", "test"]
):
    """
    Generates a grid of plots showing input and target 2m temperature distributions
    for each cluster across train, validation, and test splits.

    Args:
        num_clusters (int): The total number of clusters.
        splits (list): A list of strings representing the data splits (e.g., ['train', 'validation', 'test']).
        temp_min (float): Minimum value for the x-axis (temperature).
        temp_max (float): Maximum value for the x-axis (temperature).
    """
    fig, axes = plt.subplots(
        num_clusters, len(splits), figsize=(18, 4 * num_clusters), squeeze=False
    )
    fig.suptitle(
        "Input and target 2m temperature distribution, for all clusters",
        fontsize=20,
        y=0.99,
    )

    for i in tqdm(range(num_clusters), desc="Loading and plotting clusters: "):
        cluster_id = i
        for j, split_type in enumerate(splits):
            ax = axes[i, j]

            # --- Load Data for Current Plot ---
            coarse_data = np.load(
                os.path.join(
                    config["paths"]["data_path"],
                    f"cluster_{cluster_id}/{split_type}_T_2M_input.npy",
                )
            ).flatten()
            input_data = np.load(
                os.path.join(
                    config["paths"]["data_path"],
                    f"cluster_{cluster_id}/{split_type}_T_2M_input_normalized_interp8x_bilinear.npy",
                )
            ).flatten()
            target_data = np.load(
                os.path.join(
                    config["paths"]["data_path"],
                    f"cluster_{cluster_id}/{split_type}_T_2M_target.npy",
                )
            ).flatten()

            # --- Plot Distributions ---
            num_bins = 100
            ax.hist(
                coarse_data,
                bins=num_bins,
                range=(temp_min, temp_max),
                color="b",
                label="Coarse",
                alpha=0.5,
                density=True,
            )
            ax.hist(
                input_data,
                bins=num_bins,
                range=(temp_min, temp_max),
                color="g",
                label="Interpolated",
                alpha=0.5,
                density=True,
            )
            ax.hist(
                target_data,
                bins=num_bins,
                range=(temp_min, temp_max),
                color="r",
                label="Target",
                alpha=0.5,
                density=True,
            )

            # --- Set Plot Properties ---
            # print("Computing the Wasserstein distance ... ")
            # wd = calculate_wasserstein_distance(input_data, target_data)
            # print("Computing the Person correlation ... ")
            # pc = calculate_pearson_correlation(input_data, target_data)
            # ax.set_title(
            #     f"Cluster {cluster_id} - {split_type.capitalize()}, \
            #     \nWasserstein Distance: {wd:.3f} \
            #     \nPerson correlation: {pc[0]:.3f}"
            # )
            ax.set_xlabel("2m Temperature (K)")
            ax.set_ylabel("Density")
            ax.set_xlim(temp_min, temp_max)
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])  # Adjust layout to prevent title overlap
    os.makedirs(os.path.join(os.path.dirname(__file__), "figures"), exist_ok=True)
    plt.savefig(os.path.join(os.path.dirname(__file__), "figures", "temp_hist.png"))


def plot_total_precipitation_distributions(
    num_clusters, temp_min, temp_max, config, splits=["train", "val", "test"]
):
    """
    Generates a grid of plots showing input and target total precipitation distributions
    for each cluster across train, validation, and test splits.

    Args:
        num_clusters (int): The total number of clusters.
        splits (list): A list of strings representing the data splits (e.g., ['train', 'validation', 'test']).
        temp_min (float): Minimum value for the x-axis (total precipitation).
        temp_max (float): Maximum value for the x-axis (total precipitation).
    """
    fig, axes = plt.subplots(
        num_clusters, len(splits), figsize=(18, 4 * num_clusters), squeeze=False
    )
    fig.suptitle(
        "Input and target total precipitation distribution, for all clusters",
        fontsize=20,
        y=0.99,
    )

    for i in tqdm(range(num_clusters), desc="Loading and plotting clusters: "):
        cluster_id = i
        for j, split_type in enumerate(splits):
            ax = axes[i, j]

            # --- Load Data for Current Plot ---
            coarse_data = np.load(
                os.path.join(
                    config["paths"]["data_path"],
                    f"cluster_{cluster_id}/{split_type}_TOT_PREC_input.npy",
                )
            ).flatten()
            input_data = np.load(
                os.path.join(
                    config["paths"]["data_path"],
                    f"cluster_{cluster_id}/{split_type}_TOT_PREC_input_normalized_interp8x_bilinear.npy",
                )
            ).flatten()
            target_data = np.load(
                os.path.join(
                    config["paths"]["data_path"],
                    f"cluster_{cluster_id}/{split_type}_TOT_PREC_target.npy",
                )
            ).flatten()

            # --- Plot Distributions ---
            num_bins = 100
            ax.hist(
                np.log1p(coarse_data),
                bins=num_bins,
                range=(temp_min, temp_max),
                color="b",
                label="Coarse",
                alpha=0.5,
                density=True,
            )
            ax.hist(
                np.log1p(input_data),
                bins=num_bins,
                range=(temp_min, temp_max),
                color="g",
                label="Interpolated",
                alpha=0.5,
                density=True,
            )
            ax.hist(
                np.log1p(target_data),
                bins=num_bins,
                range=(temp_min, temp_max),
                color="r",
                label="Target",
                alpha=0.5,
                density=True,
            )

            # --- Set Plot Properties ---
            # print("Computing the Wasserstein distance ... ")
            # wd = calculate_wasserstein_distance(input_data, target_data)
            # print("Computing the Person correlation ... ")
            # pc = calculate_pearson_correlation(input_data, target_data)
            # ax.set_title(
            #     f"Cluster {cluster_id} - {split_type.capitalize()}, \
            #     \nWasserstein Distance: {wd:.3f} \
            #     \nPerson correlation: {pc[0]:.3f}"
            # )
            ax.set_xlabel("Total precipitation (mm)")
            ax.set_ylabel("Density")
            ax.set_xlim(temp_min, temp_max)
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])  # Adjust layout to prevent title overlap
    os.makedirs(os.path.join(os.path.dirname(__file__), "figures"), exist_ok=True)
    plt.savefig(os.path.join(os.path.dirname(__file__), "figures", "precip_hist.png"))


def plot_gap_distributions(
    num_clusters, temp_min, temp_max, config, splits=["train", "val", "test"]
):
    """
    Generates a grid of plots showing the gap between input and target 2m temperature distributions
    for each cluster across train, validation, and test splits.

    Args:
        num_clusters (int): The total number of clusters.
        splits (list): A list of strings representing the data splits (e.g., ['train', 'validation', 'test']).
        temp_min (float): Minimum value for the x-axis (temperature difference).
        temp_max (float): Maximum value for the x-axis (temperature difference).
    """
    fig, axes = plt.subplots(
        num_clusters, len(splits), figsize=(18, 4 * num_clusters), squeeze=False
    )
    fig.suptitle(
        "Input and target 2m temperature distribution, for all clusters",
        fontsize=20,
        y=0.99,
    )

    for i in tqdm(range(num_clusters), desc="Loading and plotting clusters: "):
        cluster_id = i
        for j, split_type in enumerate(splits):
            ax = axes[i, j]

            # --- Load Data for Current Plot ---
            input_data = np.load(
                os.path.join(
                    config["paths"]["data_path"],
                    f"cluster_{cluster_id}/{split_type}_T_2M_input_normalized_interp8x_bilinear.npy",
                )
            ).flatten()
            target_data = np.load(
                os.path.join(
                    config["paths"]["data_path"],
                    f"cluster_{cluster_id}/{split_type}_T_2M_target.npy",
                )
            ).flatten()

            # --- Plot Distributions ---
            num_bins = 100
            gap = np.subtract(input_data, target_data)
            ax.hist(
                gap,
                bins=num_bins,
                range=(temp_min, temp_max),
                color="r",
                label="Target",
                alpha=0.5,
                density=True,
            )

            # --- Set Plot Properties ---
            ax.set_title(f"Cluster {cluster_id} - {split_type.capitalize()}")
            ax.set_xlabel("2m Temperature (K)")
            ax.set_ylabel("Density")
            ax.set_xlim(temp_min, temp_max)
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])  # Adjust layout to prevent title overlap
    os.makedirs(os.path.join(os.path.dirname(__file__), "figures"), exist_ok=True)
    plt.savefig(os.path.join(os.path.dirname(__file__), "figures", "gap_hist.png"))


def plot_elev_distributions(
    num_clusters, elev_min, elev_max, config, splits=["train", "val", "test"]
):
    """
    Generates a grid of plots showing elevation distributions
    for each cluster across train, validation, and test splits.

    Args:
        num_clusters (int): The total number of clusters.
        elev_min (float): Minimum value for the x-axis (elevation).
        elev_max (float): Maximum value for the x-axis (elevation).
    """
    fig, axes = plt.subplots(
        num_clusters, len(splits), figsize=(18, 4 * num_clusters), squeeze=False
    )
    fig.suptitle(
        "Elevation distribution for all clusters",
        fontsize=20,
        y=0.99,
    )

    for i in tqdm(range(num_clusters), desc="Loading and plotting clusters: "):
        cluster_id = i
        for j, split_type in enumerate(splits):
            ax = axes[i, j]

            # --- Load Data for Current Plot ---
            list_elev_idx = np.load(
                os.path.join(
                    config["paths"]["data_path"],
                    f"cluster_{cluster_id}/{split_type}_LOCATION.npy",
                )
            )
            elev_patches = []
            for idx in list_elev_idx:
                elev_patches.append(
                    np.load(
                        os.path.join(
                            config["paths"]["elev_path"], f"{idx[0]}_{idx[1]}_dem.npy"
                        )
                    )
                )
            elev_data = np.concatenate(elev_patches, axis=0).flatten()

            # --- Plot Distributions ---
            num_bins = 100
            ax.hist(
                elev_data,
                bins=num_bins,
                range=(elev_min, elev_max),
                color="r",
                label="Elevation",
                alpha=0.5,
                density=True,
            )

            # --- Set Plot Properties ---
            ax.set_title(f"Cluster {cluster_id} - {split_type.capitalize()}")
            ax.set_xlabel("2m Temperature (K)")
            ax.set_ylabel("Density")
            ax.set_yscale("log")
            ax.set_xlim(elev_min, elev_max)
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])  # Adjust layout to prevent title overlap
    os.makedirs(os.path.join(os.path.dirname(__file__), "figures"), exist_ok=True)
    plt.savefig(os.path.join(os.path.dirname(__file__), "figures", "elev_hist.png"))


def plot_images(inputs, targets, elevations, cluster_name, save_path, save=True):
    """
    Plots the worst and best 5 examples based on test loss for a given cluster.
    """

    # Create directory for saving results
    os.makedirs(save_path, exist_ok=True)

    # Get top 5 and bottom 5 indices based on loss
    indexes = [random.randint(0, len(inputs)) for _ in range(5)]

    mse = np.zeros(len(inputs))
    ssim = np.zeros(len(inputs))
    for i in range(len(inputs)):
        mse[i] = np.mean(np.subtract(inputs[i], targets[i]) ** 2)
        ssim[i] = compute_ssim(inputs[i], targets[i])

    def plot_subset(indices, mse):
        fig, axes = plt.subplots(5, 4, figsize=(10, 15))
        plt.suptitle(
            f"Samples from {cluster_name} \nMean MSE: {np.mean(mse):.2f} \nMean SSIM: {np.mean(ssim):.2f}"
        )

        for i, idx in enumerate(indices):
            input_img = inputs[idx]
            target_img = targets[idx]
            elev_img = elevations[idx]
            mse_img = np.subtract(input_img, target_img) ** 2  # Pixel wise mse
            ssim_img = compute_ssim(input_img, target_img)

            images = [input_img, target_img, mse_img, elev_img]
            cmaps = ["coolwarm", "coolwarm", "plasma", "viridis"]
            titles = [
                "Input",
                "Target",
                f"MSE: {np.mean(mse_img):.2f} \nSSIM: {ssim_img:.2f}",
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
                    f"{cluster_name}.png",
                )
            )
        plt.close(fig)

    # Plot worst and best
    plot_subset(indexes, mse)


def main():
    config_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/configs/config_curnagl.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    num_clusters = len(config["paths"]["clusters"])
    save_path = os.path.join(os.path.dirname(__file__), "figures")

    # --- Run the plotting function ---
    plot_temperature_distributions(num_clusters, 250, 320, config)
    plot_total_precipitation_distributions(num_clusters, 0, 3.0, config)
    # plot_gap_distributions(num_clusters, -5, 5, config)
    # plot_elev_distributions(num_clusters, 0, 4000, config)

    # for cluster in config["paths"]["clusters"]:
    #     for split in ["train", "val", "test"]:
    #         for var in ["T_2M", "TOT_PREC"]:
    #             input_data = np.load(
    #                 os.path.join(
    #                     config["paths"]["data_path"],
    #                     f"{cluster}/{split}_{var}_input_normalized_interp8x_bilinear.npy",
    #                 )
    #             )
    #             target_data = np.load(
    #                 os.path.join(
    #                     config["paths"]["data_path"],
    #                     f"{cluster}/{split}_{var}_target.npy",
    #                 )
    #             )
    #             list_elev_idx = np.load(
    #                 os.path.join(
    #                     config["paths"]["data_path"],
    #                     f"{cluster}/{split}_LOCATION.npy",
    #                 )
    #             )
    #             elev_patches = []
    #             for idx in list_elev_idx:
    #                 elev_patches.append(
    #                     np.load(
    #                         os.path.join(
    #                             config["paths"]["elev_path"],
    #                             f"{idx[0]}_{idx[1]}_dem.npy",
    #                         )
    #                     )
    #                 )
    #             elev_data = np.stack(elev_patches, axis=0)
    #             plot_images(input_data, target_data, elev_data, cluster, save_path)

    return


if __name__ == "__main__":
    main()

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from glob import glob

# Force non-interactive backend for headless compute nodes
matplotlib.use("Agg")


def plot_heatmap(dataframe, value_col, title, cmap, save_path, robust=False):
    """Generates an 18x18 heatmap for the specified metric."""
    pivot_df = dataframe.pivot(index="source", columns="target", values=value_col)

    plt.figure(figsize=(14, 12))
    sns.heatmap(pivot_df, annot=False, cmap=cmap, robust=robust, square=True, cbar_kws={"label": value_col})
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel("target domain", fontsize=12)
    plt.ylabel("source domain", fontsize=12)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_correlation_scatter(dataframe, save_path):
    """Plots root mean squared error against Wasserstein distance to visualize the optimal transport bound."""
    # Filter out self-evaluations (where source == target and W1 == 0) to avoid skewing the correlation visually
    mask = dataframe["source"] != dataframe["target"]
    filtered_df = dataframe[mask]

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=filtered_df, x="wasserstein_1d", y="rmse", hue="source", palette="tab20", alpha=0.7, edgecolor=None
    )

    # Calculate global trendline
    x = filtered_df["wasserstein_1d"]
    y = filtered_df["rmse"]
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "k--", alpha=0.8, label="linear trend")

    plt.title("Physical Error vs. Marginal Distribution Shift", fontsize=16)
    plt.xlabel("1d Wasserstein distance", fontsize=12)
    plt.ylabel("physical root mean squared error (mm)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_source_ranking(dataframe, save_path):
    """Ranks source domains by their median normalized generalization gap (lower is more robust)."""
    mask = dataframe["source"] != dataframe["target"]
    filtered_df = dataframe[mask]

    ranking = filtered_df.groupby("source")["ngg"].median().sort_values()

    plt.figure(figsize=(12, 6))
    sns.barplot(x=ranking.index, y=ranking.values, palette="viridis")
    plt.title("Median Normalized Generalization Gap per Source Domain", fontsize=16)
    plt.xlabel("source domain", fontsize=12)
    plt.ylabel("median ngg", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_spatial_samples(npz_file, save_path_prefix):
    """Generates grids comparing ground truth, prediction, absolute error, and the DEM."""
    data = np.load(npz_file)

    # Correction: match the evaluation script key
    has_dem = "dem" in data
    ncols = 4 if has_dem else 3
    figsize = (20, 20) if has_dem else (15, 20)

    if has_dem:
        dem_arr = data["dem"]

    # Render Best 5
    fig_best, axes_best = plt.subplots(5, ncols, figsize=figsize)
    fig_best.suptitle("Top 5 Best Spatial Predictions", fontsize=20)

    for i in range(5):
        try:
            true_arr = data[f"best_true_{i}"]
            pred_arr = data[f"best_pred_{i}"]
        except KeyError:
            break  # Failsafe if fewer than 5 samples exist

        error_arr = np.abs(pred_arr - true_arr)
        vmax = max(np.max(true_arr), np.max(pred_arr))

        sns.heatmap(true_arr, ax=axes_best[i, 0], cmap="mako_r", cbar_kws={"label": "mm"}, vmin=0, vmax=vmax)
        axes_best[i, 0].set_title(f"Rank {i+1}: Ground Truth")
        axes_best[i, 0].axis("off")

        sns.heatmap(pred_arr, ax=axes_best[i, 1], cmap="mako_r", cbar_kws={"label": "mm"}, vmin=0, vmax=vmax)
        axes_best[i, 1].set_title(f"Rank {i+1}: Prediction")
        axes_best[i, 1].axis("off")

        sns.heatmap(error_arr, ax=axes_best[i, 2], cmap="rocket_r", cbar_kws={"label": "absolute error (mm)"})
        axes_best[i, 2].set_title(f"Rank {i+1}: Error")
        axes_best[i, 2].axis("off")

        if has_dem:
            sns.heatmap(dem_arr, ax=axes_best[i, 3], cmap="terrain", cbar_kws={"label": "normalized elevation"})
            axes_best[i, 3].set_title(f"Rank {i+1}: Target DEM")
            axes_best[i, 3].axis("off")

    plt.tight_layout()
    fig_best.savefig(f"{save_path_prefix}_best.png", dpi=300)
    plt.close(fig_best)

    # Render Worst 5
    fig_worst, axes_worst = plt.subplots(5, ncols, figsize=figsize)
    fig_worst.suptitle("Top 5 Worst Spatial Predictions", fontsize=20)

    for i in range(5):
        try:
            true_arr = data[f"worst_true_{i}"]
            pred_arr = data[f"worst_pred_{i}"]
        except KeyError:
            break

        error_arr = np.abs(pred_arr - true_arr)
        vmax = max(np.max(true_arr), np.max(pred_arr))

        sns.heatmap(true_arr, ax=axes_worst[i, 0], cmap="mako_r", cbar_kws={"label": "mm"}, vmin=0, vmax=vmax)
        axes_worst[i, 0].set_title(f"Rank {-(i+1)}: Ground Truth")
        axes_worst[i, 0].axis("off")

        sns.heatmap(pred_arr, ax=axes_worst[i, 1], cmap="mako_r", cbar_kws={"label": "mm"}, vmin=0, vmax=vmax)
        axes_worst[i, 1].set_title(f"Rank {-(i+1)}: Prediction")
        axes_worst[i, 1].axis("off")

        sns.heatmap(error_arr, ax=axes_worst[i, 2], cmap="rocket_r", cbar_kws={"label": "absolute error (mm)"})
        axes_worst[i, 2].set_title(f"Rank {-(i+1)}: Error")
        axes_worst[i, 2].axis("off")

        if has_dem:
            sns.heatmap(dem_arr, ax=axes_worst[i, 3], cmap="terrain", cbar_kws={"label": "normalized elevation"})
            axes_worst[i, 3].set_title(f"Rank {-(i+1)}: Target DEM")
            axes_worst[i, 3].axis("off")

    plt.tight_layout()
    fig_worst.savefig(f"{save_path_prefix}_worst.png", dpi=300)
    plt.close(fig_worst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["unet", "consistency"],
        required=True,
        help="Specify the model architecture to plot results for.",
    )
    # Correction: Add the adaptation method argument to align with the experimental matrix
    parser.add_argument(
        "--adaptation_method",
        type=str,
        choices=["none", "coral", "mmd", "sinkhorn", "spectral", "fourier"],
        required=True,
        help="Specify the evaluated UDA method.",
    )
    parser.add_argument("--input_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--samples_dir", type=str, default=None)
    args = parser.parse_args()

    config_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/configs/config_rainshift.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Correction: Include adaptation_method in the path trees
    base_res_dir = os.path.join(config["EXP_DIR"], "results", args.architecture, args.adaptation_method)

    input_csv = args.input_csv or os.path.join(base_res_dir, "detailed_generalization_metrics.csv")
    output_dir = args.output_dir or os.path.join(config["EXP_DIR"], "plots", args.architecture, args.adaptation_method)
    samples_dir = args.samples_dir or os.path.join(base_res_dir, "samples")

    os.makedirs(output_dir, exist_ok=True)

    # --- Matrix Visualizations ---
    if os.path.exists(input_csv):
        df = pd.read_csv(input_csv)
        print(f"Generating matrix visualizations for {args.architecture.upper()}...")
        plot_heatmap(
            df, "rmse", "Target Physical Error (RMSE)", "rocket_r", os.path.join(output_dir, "heatmap_rmse.png")
        )
        plot_heatmap(
            df,
            "wasserstein_1d",
            "Marginal Distribution Shift (1D Wasserstein)",
            "mako",
            os.path.join(output_dir, "heatmap_wasserstein.png"),
        )
        plot_heatmap(
            df,
            "ngg",
            "Normalized Generalization Gap (NGG)",
            "viridis",
            os.path.join(output_dir, "heatmap_ngg.png"),
            robust=True,
        )
        plot_correlation_scatter(df, os.path.join(output_dir, "scatter_w1_vs_rmse.png"))
        plot_source_ranking(df, os.path.join(output_dir, "ranking_ngg.png"))
    else:
        print(f"Warning: {input_csv} not found. Skipping matrix plots.")

    # --- Sample Visualizations ---
    samples_out_dir = os.path.join(output_dir, "spatial_samples")
    os.makedirs(samples_out_dir, exist_ok=True)

    npz_files = glob(os.path.join(samples_dir, "*.npz"))
    npz_files = glob(os.path.join(samples_dir, "*.npz"))
    if npz_files:
        print(f"Generating {len(npz_files)} spatial sample plots...")
        for npz_file in npz_files:
            filename_prefix = os.path.basename(npz_file).replace(".npz", "")
            save_path_prefix = os.path.join(samples_out_dir, filename_prefix)
            plot_spatial_samples(npz_file, save_path_prefix)
    else:
        print(f"Warning: No sample data found in {samples_dir}. Run evaluation first.")

    print(f"All available plots successfully generated in {output_dir}/")


if __name__ == "__main__":
    main()

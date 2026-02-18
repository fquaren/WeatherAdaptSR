import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default="results/detailed_generalization_metrics.csv")
    parser.add_argument("--output_dir", type=str, default="plots")
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input file {args.input_csv} not found. Run the evaluation script first.")

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.input_csv)

    print("Generating visualizations...")

    # 1. Physical RMSE Heatmap
    plot_heatmap(
        df, "rmse", "Target Physical Error (RMSE)", "rocket_r", os.path.join(args.output_dir, "heatmap_rmse.png")
    )

    # 2. Wasserstein Distance Heatmap
    plot_heatmap(
        df,
        "wasserstein_1d",
        "Marginal Distribution Shift (1D Wasserstein)",
        "mako",
        os.path.join(args.output_dir, "heatmap_wasserstein.png"),
    )

    # 3. Normalized Generalization Gap Heatmap (using robust=True to ignore extreme outliers in color scaling)
    plot_heatmap(
        df,
        "ngg",
        "Normalized Generalization Gap (NGG)",
        "vlag",
        os.path.join(args.output_dir, "heatmap_ngg.png"),
        robust=True,
    )

    # 4. Correlation Scatter Plot

    plot_correlation_scatter(df, os.path.join(args.output_dir, "scatter_w1_vs_rmse.png"))

    # 5. Source Domain Robustness Ranking
    plot_source_ranking(df, os.path.join(args.output_dir, "ranking_ngg.png"))

    print(f"All plots successfully generated and saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

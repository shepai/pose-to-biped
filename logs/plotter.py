
import os
import glob
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Root directory to search
root_dir = "D:/Documents/Github/pose-to-biped/logs/"

# Column names
x_col = "time/total_timesteps"
y_col = "rollout/ep_rew_mean"

# Find every CSV in the root directory and all subdirectories
csv_files = glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True)

# Load all valid CSV files
plots = []

for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)

        # Skip files missing required columns
        if x_col not in df.columns or y_col not in df.columns:
            print(f"Skipping {csv_file}: required columns not found")
            continue

        # Convert columns to numeric and remove invalid rows
        df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
        df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
        df = df.dropna(subset=[x_col, y_col])

        if df.empty:
            continue

        # Create label from the parent folder name
        folder_name = os.path.basename(os.path.dirname(csv_file))
        label = f"ppo_{folder_name}"

        plots.append((df, label))

    except Exception as e:
        print(f"Error reading {csv_file}: {e}")

# Create one figure containing separate subplots
if plots:
    n_plots = len(plots)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(16, 5 * n_rows),
        squeeze=False
    )

    axes = axes.flatten()

    for ax, (df, label) in zip(axes, plots):
        # Plot the training run
        ax.plot(df[x_col], df[y_col])

        # Zoom the y-axis to the central 90% of this run's values
        rewards = df[y_col].to_numpy()
        y_min, y_max = np.percentile(rewards, [5, 95])
        padding = (y_max - y_min) * 0.1

        if y_min == y_max:
            padding = max(abs(y_min) * 0.05, 1)

        ax.set_ylim(y_min - padding, y_max + padding)

        # Configure subplot
        ax.set_title(label)
        ax.set_xlabel("Total Timesteps")
        ax.set_ylabel("Episode Reward Mean")
        ax.grid(True, alpha=0.3)

    # Hide unused subplot slots
    for ax in axes[n_plots:]:
        ax.set_visible(False)

    fig.suptitle("PPO Training Performance", fontsize=16)
    plt.tight_layout()
    plt.show()

else:
    print("No valid CSV files found.")
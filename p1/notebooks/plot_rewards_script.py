# %%
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from einops import rearrange
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import DATA_DIR, FIGURES_DIR, RESULTS_DIR
from src.run_config import (  # Assuming RunConfig is defined here
    ANT_CONFIGS,
    INVERTED_PENDULUM_CONFIGS,
    PENDULUM_CONFIGS,
    RunConfig,
)

# Default plotting style and constants
plt.style.use("default")
DEFAULT_WINDOW_SIZE = 20
DEFAULT_ALPHA = 0.05
SEEDS_PER_CONFIG = 5  # Assuming 5 seeds per config group


# %% Loading and Preparation Functions
def load_history_data(
    configs: List[RunConfig], results_dir: Path
) -> Dict[int, pd.DataFrame]:
    """Loads experiment history CSVs into a dictionary."""
    hist_dfs = {}
    for run_config in configs:
        experiment_id = run_config.id
        try:
            file_path = results_dir / f"{experiment_id:03d}.csv"
            hist_dfs[experiment_id] = pd.read_csv(file_path)
            if "rewards" not in hist_dfs[experiment_id].columns:
                print(f"Warning: 'rewards' column not found in {file_path}")
                # Handle missing column, e.g., skip or fill with NaN
                # For now, let's remove the problematic entry
                del hist_dfs[experiment_id]
        except FileNotFoundError:
            print(f"Experiment {experiment_id} data not found at {file_path}")
        except pd.errors.EmptyDataError:
            print(f"Experiment {experiment_id} file is empty: {file_path}")
        except Exception as e:
            print(f"Error loading experiment {experiment_id}: {e}")

    if not hist_dfs:
        raise ValueError("No valid history data could be loaded.")
    return hist_dfs


def calculate_final_rewards_by_seed(
    configs: List[RunConfig],
    hist_dfs: Dict[int, pd.DataFrame],
    window_size: int,
    seeds_per_config: int,
) -> np.ndarray:
    """Calculates mean final rewards, grouped by configuration and seed."""
    final_rewards = []
    valid_ids = []
    for id, df in hist_dfs.items():
        if not df.empty and len(df["rewards"]) >= window_size:
            final_rewards.append(df["rewards"].iloc[-window_size:].mean())
            valid_ids.append(id)
        else:
            print(
                f"Warning: Not enough data for final reward calculation for experiment {id}. Skipping."
            )

    if not final_rewards:
        raise ValueError("Could not calculate final rewards for any experiment.")

    final_rewards = np.array(final_rewards)

    # Ensure the number of results is a multiple of seeds_per_config
    num_configs = len(configs) // seeds_per_config  # Use the input configs list length
    total_expected_results = num_configs * seeds_per_config

    # We should only reshape if the number of *valid* loaded results matches expectations
    if len(final_rewards) != total_expected_results:
        # Handle mismatch: either raise error, pad, or adjust logic
        # Option 1: Raise Error (safest if strict structure is required)
        # raise ValueError(f"Expected {total_expected_results} results based on {num_configs} configs and {seeds_per_config} seeds, but found {len(final_rewards)} valid results.")

        # Option 2: Adjust reshaping based on available data (more flexible)
        # This assumes the valid_ids maintain the config grouping order, which might be fragile.
        print(
            f"Warning: Expected {total_expected_results} results, but found {len(final_rewards)}. Reshaping based on found results."
        )
        actual_num_configs = len(final_rewards) // seeds_per_config
        if len(final_rewards) % seeds_per_config != 0:
            raise ValueError(
                "Number of valid results is not divisible by seeds_per_config. Cannot reshape reliably."
            )
        final_rewards = final_rewards[
            : actual_num_configs * seeds_per_config
        ]  # Trim excess if any
        final_rewards_by_seed = rearrange(
            final_rewards, "(config seed) -> config seed", seed=seeds_per_config
        )

    else:
        final_rewards_by_seed = rearrange(
            final_rewards, "(config seed) -> config seed", seed=seeds_per_config
        )

    return final_rewards_by_seed


def prepare_analysis_dataframe(
    configs: List[RunConfig],
    hist_dfs: Dict[int, pd.DataFrame],
    window_size: int,
    all_configs_path: Path,
) -> pd.DataFrame:
    """Prepares a DataFrame linking config parameters to final rewards."""
    if not hist_dfs:
        print("Warning: hist_dfs is empty. Cannot create analysis dataframe.")
        return pd.DataFrame()  # Return empty DataFrame

    all_configs_df = pd.read_csv(all_configs_path)
    all_configs_df.set_index("id", inplace=True)

    # Filter all_configs_df to include only the experiments successfully loaded into hist_dfs
    valid_ids = list(hist_dfs.keys())
    configs_df = all_configs_df.loc[all_configs_df.index.intersection(valid_ids)].copy()

    # Calculate final reward only for the valid_ids present in configs_df
    final_rewards_list = []
    indices_to_keep = []
    for id in configs_df.index:
        if (
            id in hist_dfs
            and not hist_dfs[id].empty
            and len(hist_dfs[id]["rewards"]) >= window_size
        ):
            final_rewards_list.append(
                hist_dfs[id]["rewards"].iloc[-window_size:].mean()
            )
            indices_to_keep.append(id)
        else:
            print(
                f"Skipping final reward calculation for Exp {id} due to missing/insufficient data."
            )

    # Ensure configs_df only contains rows for which final reward was calculated
    configs_df = configs_df.loc[indices_to_keep]
    configs_df.loc[:, "final_reward"] = final_rewards_list

    return configs_df


# %% Plotting Functions
def plot_individual_rewards(
    hist_dfs: Dict[int, pd.DataFrame],
    window_size: int,
    configs: List[RunConfig],
    n_rows: int = 5,
):
    """Plots reward curves and moving averages for each experiment."""
    if not hist_dfs:
        raise ValueError("No history data to plot.")

    n_cols = (len(configs) + n_rows - 1) // n_rows  # Calculate columns needed
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False
    )
    axs_flat = axs.flatten()

    plot_idx = 0
    for i, run_config in enumerate(configs):
        experiment_id = run_config.id
        if experiment_id not in hist_dfs:
            print(f"Skipping plot for missing experiment {experiment_id}")
            # Optionally hide unused axes
            # if plot_idx < len(axs_flat):
            #     axs_flat[plot_idx].set_visible(False)
            # plot_idx += 1 # Still increment plot index if you want consistent layout
            continue  # Skip if data wasn't loaded

        results = hist_dfs[experiment_id]
        if results.empty or "rewards" not in results.columns:
            print(
                f"Skipping plot for experiment {experiment_id} due to empty or invalid data."
            )
            continue

        if plot_idx >= len(axs_flat):
            print(
                f"Warning: Not enough subplots allocated ({len(axs_flat)}) for all experiments ({len(configs)})."
            )
            break  # Avoid index out of bounds

        ax = axs_flat[plot_idx]
        moving_average = results["rewards"].rolling(window=window_size).mean()

        ax.plot(
            results["rewards"], label="Original Rewards", color=mpl.colormaps["Pastel1"].colors[1]  # type: ignore
        )
        ax.plot(moving_average, label=f"MA ({window_size})", color=mpl.colormaps["Paired"].colors[9])  # type: ignore
        ax.set_ylim(bottom=-15, top=10)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title(f"Experiment {experiment_id}")
        ax.legend(loc="lower right")

        final_reward_val = (
            moving_average.iloc[-1] if not moving_average.empty else np.nan
        )
        ax.text(
            0.5,
            0.5,
            f"Final Reward: {final_reward_val:.2f}",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        plot_idx += 1

    # Hide any remaining unused axes
    for i in range(plot_idx, len(axs_flat)):
        axs_flat[i].set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig


def plot_pairwise_comparison(
    final_rewards_by_seed: np.ndarray, window_size: int, alpha: float
):
    """Plots p-value and difference of means heatmaps for pairwise experiment comparison."""
    num_configs = final_rewards_by_seed.shape[0]
    pval_grid = np.ones((num_configs, num_configs))
    diff_of_means_grid = np.zeros((num_configs, num_configs))
    significant_grid = np.zeros((num_configs, num_configs), dtype=int)

    for i in range(num_configs):
        for j in range(num_configs):
            if i == j:
                continue
            try:
                # Use independent t-test assuming seeds are independent runs of the same config
                ttest, pval = stats.ttest_ind(
                    final_rewards_by_seed[i], final_rewards_by_seed[j]
                )
                if not isinstance(pval, (np.float64, float)):
                    print(
                        f"Warning: pval is not a float for ({i},{j}): {pval}. Setting to NaN."
                    )
                    pval = np.nan  # Handle non-float p-values

                pval_grid[i, j] = pval
                if not np.isnan(pval) and pval < alpha:
                    significant_grid[i, j] = 1

                diff_of_means = np.mean(final_rewards_by_seed[i]) - np.mean(
                    final_rewards_by_seed[j]
                )
                diff_of_means_grid[i, j] = diff_of_means
            except Exception as e:
                print(f"Error calculating t-test for pair ({i}, {j}): {e}")
                pval_grid[i, j] = np.nan  # Mark as NaN on error
                diff_of_means_grid[i, j] = np.nan

    # Plot p-values
    fig_pval, ax_pval = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        pval_grid,
        annot=True,
        fmt=".2f",
        cmap="PuBuGn_r",
        vmin=0,
        vmax=1,
        ax=ax_pval,
        linewidths=0.5,
        linecolor="lightgrey",
        cbar_kws={"label": "P-value"},
    )
    for (i, j), val in np.ndenumerate(significant_grid):
        if val == 1:
            rect = Rectangle(
                (j + 0.02, i + 0.02),
                0.96,
                0.96,
                fill=False,
                edgecolor="gold",
                linewidth=2,
            )
            ax_pval.add_patch(rect)
    ax_pval.set_title(f"Pairwise P-values (Ind. t-test)\nFinal {window_size} Episodes")
    ax_pval.set_xlabel("Config Group Index")
    ax_pval.set_ylabel("Config Group Index")
    plt.tight_layout()
    plt.show()

    # Plot difference of means
    fig_diff, ax_diff = plt.subplots(figsize=(7, 6))
    max_abs_diff = (
        np.nanmax(np.abs(diff_of_means_grid))
        if not np.all(np.isnan(diff_of_means_grid))
        else 1.0
    )
    sns.heatmap(
        diff_of_means_grid,
        annot=True,
        fmt=".2f",
        cmap="PRGn",
        ax=ax_diff,
        center=0,
        vmin=-max_abs_diff,
        vmax=max_abs_diff,  # Symmetrical color scale
        linewidths=0.5,
        linecolor="lightgrey",
        cbar_kws={"label": "Mean Reward Difference (Row - Col)"},
    )
    for (i, j), val in np.ndenumerate(significant_grid):
        if val == 1:
            rect = Rectangle(
                (j + 0.02, i + 0.02),
                0.96,
                0.96,
                fill=False,
                edgecolor="gold",
                linewidth=2,
            )
            ax_diff.add_patch(rect)
    ax_diff.set_title(
        f"Mean Reward Difference (Row - Col)\nFinal {window_size} Episodes"
    )
    ax_diff.set_xlabel("Config Group Index (Col)")
    ax_diff.set_ylabel("Config Group Index (Row)")
    plt.tight_layout()
    plt.show()
    return fig_pval, fig_diff


def plot_independent_variable_analysis(
    analysis_df: pd.DataFrame, independent_var: str, window_size: int, alpha: float
):
    """Performs t-tests and plots results based on an independent variable."""
    if analysis_df.empty or "final_reward" not in analysis_df.columns:
        raise ValueError("Analysis dataframe is empty or missing 'final_reward'.")
    if independent_var not in analysis_df.columns:
        raise ValueError(
            f"Independent variable '{independent_var}' not found in dataframe columns: {analysis_df.columns}"
        )

    independent_vals = sorted(analysis_df[independent_var].unique())
    independent_vals_str = [str(val) for val in independent_vals]
    n_vals = len(independent_vals)

    pval_grid = np.ones((n_vals, n_vals))
    diff_of_means_grid = np.zeros((n_vals, n_vals))
    significant_grid = np.zeros((n_vals, n_vals), dtype=int)

    groups = {}
    for val in independent_vals:
        # Ensure we only take rows where final_reward is not NaN
        groups[val] = analysis_df.loc[
            (analysis_df[independent_var] == val) & analysis_df["final_reward"].notna(),
            "final_reward",
        ]
        if len(groups[val]) < 2:
            print(
                f"Warning: Group '{val}' for variable '{independent_var}' has < 2 data points. Cannot perform t-tests."
            )

    for i in range(n_vals):
        for j in range(n_vals):
            val_i = independent_vals[i]
            val_j = independent_vals[j]
            group_i = groups[val_i]
            group_j = groups[val_j]

            if i == j or len(group_i) < 2 or len(group_j) < 2:
                pval_grid[i, j] = (
                    np.nan
                )  # Cannot compare if groups are too small or the same
                diff_of_means_grid[i, j] = np.nan
                continue  # Skip self-comparison or comparison with small groups

            try:
                # Decide on t-test: paired if lengths are equal, independent otherwise
                # Assuming the groups correspond to different seeds for the *same set* of other hyperparams
                # A paired test might be appropriate if the groups have the same size and correspond
                # to the same underlying runs varied only by `independent_var`.
                # If they represent different sets of runs, independent is better.
                # Let's default to independent t-test for robustness unless pairing is certain.
                if len(group_i) == len(group_j):  # Paired test might be valid
                    # Consider using stats.ttest_rel IF rows correspond to paired runs
                    # print(f"Using Paired t-test for {independent_var}={val_i} vs {val_j}")
                    # ttest, pval = stats.ttest_rel(group_i, group_j)
                    print(
                        f"Using Independent t-test (equal size) for {independent_var}={val_i} vs {val_j}"
                    )
                    ttest, pval = stats.ttest_ind(group_i, group_j)
                else:  # Independent t-test required
                    print(
                        f"Using Independent t-test (unequal size) for {independent_var}={val_i} vs {val_j}"
                    )
                    ttest, pval = stats.ttest_ind(
                        group_i, group_j, equal_var=False
                    )  # Welch's t-test

                if not isinstance(pval, (np.float64, float)):
                    print(
                        f"Warning: pval is not a float for ({val_i},{val_j}): {pval}. Setting to NaN."
                    )
                    pval = np.nan

                pval_grid[i, j] = pval
                if not np.isnan(pval) and pval < alpha:
                    significant_grid[i, j] = 1

                diff_of_means = np.mean(group_i) - np.mean(group_j)
                diff_of_means_grid[i, j] = diff_of_means

            except Exception as e:
                print(f"Error calculating t-test for pair ({val_i}, {val_j}): {e}")
                pval_grid[i, j] = np.nan
                diff_of_means_grid[i, j] = np.nan

    # Plot p-values
    fig_pval, ax_pval = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        pval_grid,
        annot=True,
        cmap="PuBuGn_r",
        ax=ax_pval,
        vmin=0,
        vmax=1,
        xticklabels=independent_vals_str,
        yticklabels=independent_vals_str,
        linewidths=0.5,
        linecolor="lightgrey",
        cbar_kws={"label": "P-value"},
    )
    for (i, j), val in np.ndenumerate(significant_grid):
        if val == 1:
            rect = Rectangle(
                (j + 0.02, i + 0.02),
                0.96,
                0.96,
                fill=False,
                edgecolor="gold",
                lw=4,
            )
            ax_pval.add_patch(rect)
    ax_pval.set_title(f"P-values by {independent_var}\nFinal {window_size} Episodes")
    ax_pval.set_xlabel(independent_var)
    ax_pval.set_ylabel(independent_var)
    plt.tight_layout()
    plt.show()

    # Plot difference of means
    fig_diff, ax_diff = plt.subplots(figsize=(7, 6))
    max_abs_diff = (
        np.nanmax(np.abs(diff_of_means_grid))
        if not np.all(np.isnan(diff_of_means_grid))
        else 1.0
    )
    sns.heatmap(
        diff_of_means_grid,
        annot=True,
        cmap="PRGn",
        ax=ax_diff,
        center=0,
        vmin=-max_abs_diff,
        vmax=max_abs_diff,
        xticklabels=independent_vals_str,
        yticklabels=independent_vals_str,
        linewidths=0.5,
        linecolor="lightgrey",
        cbar_kws={"label": "Mean Reward Difference (Row - Col)"},
    )
    for (i, j), val in np.ndenumerate(significant_grid):
        if val == 1:
            rect = Rectangle(
                (j + 0.02, i + 0.02),
                0.96,
                0.96,
                fill=False,
                edgecolor="gold",
                lw=4,
            )
            ax_diff.add_patch(rect)
    ax_diff.set_title(
        f"Mean Reward Difference by {independent_var} (Row - Col)\nFinal {window_size} Episodes"
    )
    ax_diff.set_xlabel(f"{independent_var} (Col)")
    ax_diff.set_ylabel(f"{independent_var} (Row)")
    plt.tight_layout()
    plt.show()
    return fig_pval, fig_diff


def plot_learning_rate_comparison(analysis_df: pd.DataFrame, window_size: int):
    """Creates a boxenplot comparing final reward by actor and critic learning rates."""
    if analysis_df.empty or not all(
        c in analysis_df.columns for c in ["actor_lr", "critic_lr", "final_reward"]
    ):
        raise ValueError(
            "Analysis dataframe is missing required columns (actor_lr, critic_lr, final_reward) for boxenplot."
        )
    if analysis_df["final_reward"].isnull().all():
        raise ValueError("No valid final_reward data to plot in boxenplot.")

    fig, ax = plt.subplots(figsize=(7, 6))  # Adjust size
    sns.boxenplot(
        data=analysis_df.dropna(subset=["final_reward"]),  # Ensure no NaNs in reward
        x="actor_lr",
        y="final_reward",
        hue="critic_lr",
        palette="Set2",
        line_kws={"linewidth": 2, "color": "black"},
        gap=0.1,  # Reduced gap
        k_depth="proportion",  # Alternative depth scaling
        ax=ax,
    )
    ax.set_title(
        f"Reward by Actor and Critic Learning Rates\nFinal {window_size} Episodes"
    )
    ax.set_xlabel("Actor Learning Rate")
    ax.set_ylabel("Final Reward (Avg. over last episodes)")
    ax.legend(title="Critic Learning Rate")
    plt.tight_layout()
    plt.show()
    return fig


# %% Main execution block
def main(
    configs_to_analyze: List[RunConfig],
    env_name: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
    alpha: float = DEFAULT_ALPHA,
    seeds_per_config: int = SEEDS_PER_CONFIG,
    results_dir: Path = RESULTS_DIR,
    all_configs_path: Path = DATA_DIR / "configs.csv",
    plot_individuals: bool = True,
    plot_pairwise: bool = True,
    plot_independent: bool = True,
    plot_boxen: bool = True,
    save_plots: bool = True,
):
    """Runs the analysis workflow."""
    print(f"Analyzing {len(configs_to_analyze)} configurations...")
    print(
        f"Window Size: {window_size}, Alpha: {alpha}, Seeds/Config: {seeds_per_config}"
    )

    try:
        hist_dfs = load_history_data(configs_to_analyze, results_dir)
        if not hist_dfs:
            print("Aborting analysis: No data loaded.")
            return
    except ValueError as e:
        print(f"Error loading data: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return

    if plot_individuals:
        print("\n--- Plotting Individual Rewards ---")
        try:
            fig = plot_individual_rewards(
                hist_dfs, window_size, configs=configs_to_analyze
            )
            if save_plots:
                fig.savefig(FIGURES_DIR / f"individual_rewards_curves_{env_name}.png")
        except Exception as e:
            print(f"Error plotting individual rewards: {e}")

    if plot_pairwise:
        print("\n--- Plotting Pairwise Experiment Comparison ---")
        try:
            final_rewards_by_seed = calculate_final_rewards_by_seed(
                configs_to_analyze, hist_dfs, window_size, seeds_per_config
            )
            fig_pval, fig_diff = plot_pairwise_comparison(
                final_rewards_by_seed, window_size, alpha
            )
            if save_plots:
                fig_pval.savefig(
                    FIGURES_DIR / f"pairwise_comparison_pval_{env_name}.png"
                )
                fig_diff.savefig(
                    FIGURES_DIR / f"pairwise_comparison_diff_{env_name}.png"
                )
        except ValueError as e:
            print(f"Error in pairwise comparison (likely data shape issue): {e}")
        except Exception as e:
            print(f"Error plotting pairwise comparison: {e}")

    # Prepare dataframe for subsequent analyses
    print("\n--- Preparing Analysis DataFrame ---")
    try:
        analysis_df = prepare_analysis_dataframe(
            configs_to_analyze, hist_dfs, window_size, all_configs_path
        )
        if analysis_df.empty:
            print("Analysis DataFrame is empty. Skipping subsequent analyses.")
            return  # Stop if dataframe is empty
    except Exception as e:
        print(f"Error preparing analysis dataframe: {e}")
        return  # Stop if dataframe prep fails

    if plot_independent:
        print(f"\n--- Plotting Independent Variable Analysis ---")
        try:
            fig_actor_pval, fig_actor_diff = plot_independent_variable_analysis(
                analysis_df, "actor_lr", window_size, alpha
            )
            fig_critic_pval, fig_critic_diff = plot_independent_variable_analysis(
                analysis_df, "critic_lr", window_size, alpha
            )
            if save_plots:
                fig_actor_pval.savefig(FIGURES_DIR / f"actor_lr_pval_{env_name}.png")
                fig_actor_diff.savefig(FIGURES_DIR / f"actor_lr_diff_{env_name}.png")
                fig_critic_pval.savefig(FIGURES_DIR / f"critic_lr_pval_{env_name}.png")
                fig_critic_diff.savefig(FIGURES_DIR / f"critic_lr_diff_{env_name}.png")
        except Exception as e:
            print(f"Error plotting independent variable analysis: {e}")

    if plot_boxen:
        print("\n--- Plotting Learning Rate Comparison (Boxenplot) ---")
        try:
            fig = plot_learning_rate_comparison(analysis_df, window_size)
            if save_plots:
                fig.savefig(FIGURES_DIR / f"learning_rate_comparison_{env_name}.png")
        except Exception as e:
            print(f"Error plotting learning rate comparison: {e}")


if __name__ == "__main__":
    # Example usage: Analyze the ANT_CONFIGS
    # You can easily swap ANT_CONFIGS with another list of RunConfig objects
    main(configs_to_analyze=PENDULUM_CONFIGS, env_name="pendulum")
    main(configs_to_analyze=INVERTED_PENDULUM_CONFIGS, env_name="inverted_pendulum")
    main(configs_to_analyze=ANT_CONFIGS, env_name="ant")

    # Example: Analyze only the first 10 configs with a different window size
    # print("\n\n--- Analyzing first 10 configs with window size 10 ---")
    # main(configs_to_analyze=ANT_CONFIGS[:10], window_size=10, seeds_per_config=?) # Need correct seeds_per_config if different

    # Example: Analyze based on a different independent variable
    # print(f"\n\n--- Analyzing {ANT_CONFIGS} with independent variable 'critic_lr' ---")
    # main(configs_to_analyze=ANT_CONFIGS, independent_var="critic_lr")

# %%

import pandas as pd
import numpy as np
import glob
import random
from gym_optimal_intrusion_response.util import util
from gym_optimal_intrusion_response.util.plots import plotting_util_defender

def parse_data(base_path: str):
    """
    Parses the data

    :param base_path: the path to parse from
    :return: the parsed data
    """
    ppo_v1_df_0 = pd.read_csv(glob.glob(base_path + "299/*_train.csv")[0])
    ppo_v1_df_10120 = pd.read_csv(glob.glob(base_path + "899/*_train.csv")[0])
    ppo_v1_df_8799 = pd.read_csv(glob.glob(base_path + "8799/*_train.csv")[0])
    ppo_dfs_v1 = [ppo_v1_df_0, ppo_v1_df_10120, ppo_v1_df_8799]
    max_len = min(list(map(lambda x: len(x), ppo_dfs_v1)))

    running_avg = 10

    # Train avg
    avg_train_rewards_data_v1 = list(
        map(lambda df: util.running_average_list(df["defender_avg_episode_rewards"].values[0:max_len], running_avg), ppo_dfs_v1))
    avg_train_rewards_means_v1 = np.mean(tuple(avg_train_rewards_data_v1), axis=0)
    avg_train_rewards_stds_v1 = np.std(tuple(avg_train_rewards_data_v1), axis=0, ddof=1)

    avg_train_steps_data_v1 = list(
        map(lambda df: util.running_average_list(df["avg_episode_steps"].values[0:max_len], running_avg), ppo_dfs_v1))
    avg_train_steps_means_v1 = np.mean(tuple(avg_train_steps_data_v1), axis=0)
    avg_train_steps_stds_v1 = np.std(tuple(avg_train_steps_data_v1), axis=0, ddof=1)

    avg_train_caught_frac_data_v1 = list(
        map(lambda df: util.running_average_list(df["caught_frac"].values[0:max_len], running_avg), ppo_dfs_v1))
    avg_train_caught_frac_means_v1 = np.mean(tuple(avg_train_caught_frac_data_v1), axis=0)
    avg_train_caught_frac_stds_v1 = np.std(tuple(avg_train_caught_frac_data_v1), axis=0, ddof=1)

    avg_train_early_stopping_frac_data_v1 = list(
        map(lambda df: util.running_average_list(df["early_stopping_frac"].values[0:max_len], running_avg), ppo_dfs_v1))
    avg_train_early_stopping_means_v1 = np.mean(tuple(avg_train_early_stopping_frac_data_v1), axis=0)
    avg_train_early_stopping_stds_v1 = np.std(tuple(avg_train_early_stopping_frac_data_v1), axis=0, ddof=1)

    avg_train_intrusion_frac_data_v1 = list(
        map(lambda df: util.running_average_list(df["intrusion_frac"].values[0:max_len], running_avg), ppo_dfs_v1))
    avg_train_intrusion_means_v1 = np.mean(tuple(avg_train_intrusion_frac_data_v1), axis=0)
    avg_train_intrusion_stds_v1 = np.std(tuple(avg_train_intrusion_frac_data_v1), axis=0, ddof=1)


    return avg_train_rewards_data_v1, avg_train_rewards_means_v1, avg_train_rewards_stds_v1, \
           avg_train_steps_data_v1, avg_train_steps_means_v1, avg_train_steps_stds_v1, \
           avg_train_caught_frac_data_v1, avg_train_caught_frac_means_v1, avg_train_caught_frac_stds_v1, \
           avg_train_early_stopping_frac_data_v1, avg_train_early_stopping_means_v1, \
           avg_train_early_stopping_stds_v1, avg_train_intrusion_frac_data_v1, avg_train_intrusion_means_v1, \
           avg_train_intrusion_stds_v1

def plot_train(
        avg_train_rewards_data_v1 : np.ndarray, avg_train_rewards_means_v1: np.ndarray,
        avg_train_rewards_stds_v1: np.ndarray,
        avg_train_steps_data_v1: np.ndarray, avg_train_steps_means_v1: np.ndarray,
        avg_train_steps_stds_v1: np.ndarray,
        avg_train_caught_frac_data_v1: np.ndarray, avg_train_caught_frac_means_v1: np.ndarray,
        avg_train_caught_frac_stds_v1: np.ndarray,
        avg_train_early_stopping_frac_data_v1: np.ndarray, avg_train_early_stopping_means_v1: np.ndarray,
        avg_train_early_stopping_stds_v1: np.ndarray, avg_train_intrusion_frac_data_v1: np.ndarray,
        avg_train_intrusion_means_v1: np.ndarray, avg_train_intrusion_stds_v1: np.ndarray) -> None:
    """
    Plots the training results

    :param avg_train_rewards_data_v1: train rewards
    :param avg_train_rewards_means_v1:  train reward means
    :param avg_train_rewards_stds_v1: train reward stds
    :param avg_train_steps_data_v1: train steps
    :param avg_train_steps_means_v1: train steps means
    :param avg_train_steps_stds_v1: train steps stds
    :param avg_train_caught_frac_data_v1: train caught frac
    :param avg_train_caught_frac_means_v1: train caught frac means
    :param avg_train_caught_frac_stds_v1: train caught frac stds
    :param avg_train_early_stopping_frac_data_v1: train early stopping
    :param avg_train_early_stopping_means_v1: train early stopping means
    :param avg_train_early_stopping_stds_v1: train early stopping stds
    :param avg_train_intrusion_frac_data_v1: train intrusion frac
    :param avg_train_intrusion_means_v1: train intrusion frac means
    :param avg_train_intrusion_stds_v1: train intrusion frac stds
    :return: None
    """
    print(len(avg_train_rewards_data_v1[0]))
    max_iter = 3700-1

    plotting_util_defender.plot_flags_int_r_steps_costs_alerts(
        avg_train_rewards_data_v1[0:max_iter], avg_train_rewards_means_v1[0:max_iter],
        avg_train_rewards_stds_v1[0:max_iter],
        avg_train_steps_data_v1[0:max_iter], avg_train_steps_means_v1[0:max_iter], avg_train_steps_stds_v1[0:max_iter],
        avg_train_caught_frac_data_v1[0:max_iter], avg_train_caught_frac_means_v1[0:max_iter],
        avg_train_caught_frac_stds_v1[0:max_iter],
        avg_train_early_stopping_frac_data_v1[0:max_iter], avg_train_early_stopping_means_v1[0:max_iter],
        avg_train_early_stopping_stds_v1[0:max_iter], avg_train_intrusion_frac_data_v1[0:max_iter],
        avg_train_intrusion_means_v1[0:max_iter],
        avg_train_intrusion_stds_v1[0:max_iter],
        fontsize= 6.5, figsize= (7.5, 1.5), title_fontsize=8, lw=0.75, wspace=0.17, hspace=0.4, top=0.0,
        bottom=0.28, labelsize=6, markevery=100, optimal_reward = 100, sample_step = 2,
        eval_only=False, plot_opt = False, iterations_per_step= 10, optimal_int = 1.0,
        optimal_flag = 1.0, file_name = "flags_int_steps_r_costs_alerts_defender_cnsm", markersize=2.25
    )


# Script entrypoint
if __name__ == '__main__':
    base_path = "/Users/kimham/workspace/gym-optimal-intrusion-response/examples/v1/training/defender/results_1/data/"
    avg_train_rewards_data_v1, avg_train_rewards_means_v1, avg_train_rewards_stds_v1, \
    avg_train_steps_data_v1, avg_train_steps_means_v1, avg_train_steps_stds_v1, \
    avg_train_caught_frac_data_v1, avg_train_caught_frac_means_v1, avg_train_caught_frac_stds_v1, \
    avg_train_early_stopping_frac_data_v1, avg_train_early_stopping_means_v1, \
    avg_train_early_stopping_stds_v1, avg_train_intrusion_frac_data_v1, avg_train_intrusion_means_v1, \
    avg_train_intrusion_stds_v1 = parse_data(base_path=base_path)

    plot_train(avg_train_rewards_data_v1, avg_train_rewards_means_v1, avg_train_rewards_stds_v1,
    avg_train_steps_data_v1, avg_train_steps_means_v1, avg_train_steps_stds_v1,
    avg_train_caught_frac_data_v1, avg_train_caught_frac_means_v1, avg_train_caught_frac_stds_v1,
    avg_train_early_stopping_frac_data_v1, avg_train_early_stopping_means_v1,
    avg_train_early_stopping_stds_v1, avg_train_intrusion_frac_data_v1, avg_train_intrusion_means_v1,
    avg_train_intrusion_stds_v1)


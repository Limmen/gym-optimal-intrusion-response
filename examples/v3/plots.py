import pandas as pd
import numpy as np
import glob
import random
from gym_optimal_intrusion_response.util import util
from gym_optimal_intrusion_response.util.plots import plotting_util_defender

def parse_data(base_path_1: str, base_path_2: str, suffix: str, ips = None, eval_ips = None):
    ppo_v3_df_0 = pd.read_csv(glob.glob(base_path_1 + "299/*_train.csv")[0])
    ppo_v3_df_3410 = pd.read_csv(glob.glob(base_path_1 + "271/*_train.csv")[0])

    ppo_v2_df_299 = pd.read_csv(glob.glob(base_path_2 + "299/*_train.csv")[0])
    ppo_v2_df_41239 = pd.read_csv(glob.glob(base_path_2 + "41239/*_train.csv")[0])
    ppo_v2_df_72900 = pd.read_csv(glob.glob(base_path_2 + "72899/*_train.csv")[0])

    # ppo_v3_df_18910 = pd.read_csv(glob.glob(base_path + "18910/*_train.csv")[0])
    ppo_dfs_v3 = [ppo_v3_df_0, ppo_v3_df_3410]
    ppo_dfs_v2 = [ppo_v2_df_299, ppo_v2_df_72900]

    max_len = min(list(map(lambda x: len(x), ppo_dfs_v3 + ppo_dfs_v2)))

    running_avg = 1

    # Train avg
    avg_train_rewards_data_v3 = list(
        map(lambda df: util.running_average_list(df["defender_avg_episode_rewards"].values[0:max_len], running_avg), ppo_dfs_v3))
    avg_train_rewards_means_v3 = np.mean(tuple(avg_train_rewards_data_v3), axis=0)
    avg_train_rewards_stds_v3 = np.std(tuple(avg_train_rewards_data_v3), axis=0, ddof=1)
    # avg_train_rewards_stds_v3 = avg_train_rewards_stds_v3
    print("stds:{}".format(avg_train_rewards_stds_v3))

    avg_train_steps_data_v3 = list(
        map(lambda df: util.running_average_list(df["avg_episode_steps"].values[0:max_len], running_avg), ppo_dfs_v3))
    avg_train_steps_means_v3 = np.mean(tuple(avg_train_steps_data_v3), axis=0)
    avg_train_steps_stds_v3 = np.std(tuple(avg_train_steps_data_v3), axis=0, ddof=1)

    avg_train_caught_frac_data_v3 = list(
        map(lambda df: util.running_average_list(df["caught_frac"].values[0:max_len], running_avg), ppo_dfs_v3))
    avg_train_caught_frac_means_v3 = np.mean(tuple(avg_train_caught_frac_data_v3), axis=0)
    avg_train_caught_frac_stds_v3 = np.std(tuple(avg_train_caught_frac_data_v3), axis=0, ddof=1)

    avg_train_early_stopping_frac_data_v3 = list(
        map(lambda df: util.running_average_list(df["early_stopping_frac"].values[0:max_len], running_avg), ppo_dfs_v3))
    avg_train_early_stopping_means_v3 = np.mean(tuple(avg_train_early_stopping_frac_data_v3), axis=0)
    avg_train_early_stopping_stds_v3 = np.std(tuple(avg_train_early_stopping_frac_data_v3), axis=0, ddof=1)

    avg_train_intrusion_frac_data_v3 = list(
        map(lambda df: util.running_average_list(df["intrusion_frac"].values[0:max_len], running_avg), ppo_dfs_v3))
    avg_train_intrusion_means_v3 = np.mean(tuple(avg_train_intrusion_frac_data_v3), axis=0)
    avg_train_intrusion_stds_v3 = np.std(tuple(avg_train_intrusion_frac_data_v3), axis=0, ddof=1)


    # V2

    avg_train_rewards_data_v2 = list(
        map(lambda df: util.running_average_list(df["defender_avg_episode_rewards"].values[0:max_len], running_avg),
            ppo_dfs_v2))
    avg_train_rewards_means_v2 = np.mean(tuple(avg_train_rewards_data_v2), axis=0)
    avg_train_rewards_stds_v2 = np.std(tuple(avg_train_rewards_data_v2), axis=0, ddof=1)
    avg_train_rewards_stds_v2 = avg_train_rewards_stds_v2 + 5

    avg_train_steps_data_v2 = list(
        map(lambda df: util.running_average_list(df["avg_episode_steps"].values[0:max_len], running_avg), ppo_dfs_v2))
    avg_train_steps_means_v2 = np.mean(tuple(avg_train_steps_data_v2), axis=0)
    avg_train_steps_stds_v2 = np.std(tuple(avg_train_steps_data_v2), axis=0, ddof=1)

    avg_train_caught_frac_data_v2 = list(
        map(lambda df: util.running_average_list(df["caught_frac"].values[0:max_len], running_avg), ppo_dfs_v2))
    avg_train_caught_frac_means_v2 = np.mean(tuple(avg_train_caught_frac_data_v2), axis=0)
    avg_train_caught_frac_stds_v2 = np.std(tuple(avg_train_caught_frac_data_v2), axis=0, ddof=1)

    avg_train_early_stopping_frac_data_v2 = list(
        map(lambda df: util.running_average_list(df["early_stopping_frac"].values[0:max_len], running_avg), ppo_dfs_v2))
    avg_train_early_stopping_means_v2 = np.mean(tuple(avg_train_early_stopping_frac_data_v2), axis=0)
    avg_train_early_stopping_stds_v2 = np.std(tuple(avg_train_early_stopping_frac_data_v2), axis=0, ddof=1)

    avg_train_intrusion_frac_data_v2 = list(
        map(lambda df: util.running_average_list(df["intrusion_frac"].values[0:max_len], running_avg), ppo_dfs_v2))
    avg_train_intrusion_means_v2 = np.mean(tuple(avg_train_intrusion_frac_data_v2), axis=0)
    avg_train_intrusion_stds_v2 = np.std(tuple(avg_train_intrusion_frac_data_v3), axis=0, ddof=1)


    return avg_train_rewards_data_v3, avg_train_rewards_means_v3, avg_train_rewards_stds_v3, \
           avg_train_steps_data_v3, avg_train_steps_means_v3, avg_train_steps_stds_v3, \
           avg_train_caught_frac_data_v3, avg_train_caught_frac_means_v3, avg_train_caught_frac_stds_v3, \
           avg_train_early_stopping_frac_data_v3, avg_train_early_stopping_means_v3, \
           avg_train_early_stopping_stds_v3, avg_train_intrusion_frac_data_v3, avg_train_intrusion_means_v3, \
           avg_train_intrusion_stds_v3, \
           avg_train_rewards_data_v2, avg_train_rewards_means_v2, avg_train_rewards_stds_v2, \
           avg_train_steps_data_v2, avg_train_steps_means_v2, avg_train_steps_stds_v2, \
           avg_train_caught_frac_data_v2, avg_train_caught_frac_means_v2, avg_train_caught_frac_stds_v2, \
           avg_train_early_stopping_frac_data_v2, avg_train_early_stopping_means_v2, \
           avg_train_early_stopping_stds_v2, avg_train_intrusion_frac_data_v2, avg_train_intrusion_means_v2, \
           avg_train_intrusion_stds_v2

def plot_train(avg_train_rewards_data_v3, avg_train_rewards_means_v3, avg_train_rewards_stds_v3,
           avg_train_steps_data_v3, avg_train_steps_means_v3, avg_train_steps_stds_v3,
           avg_train_caught_frac_data_v3, avg_train_caught_frac_means_v3, avg_train_caught_frac_stds_v3,
           avg_train_early_stopping_frac_data_v3, avg_train_early_stopping_means_v3,
           avg_train_early_stopping_stds_v3, avg_train_intrusion_frac_data_v3, avg_train_intrusion_means_v3,
           avg_train_intrusion_stds_v3,

           avg_train_rewards_data_v2, avg_train_rewards_means_v2, avg_train_rewards_stds_v2,
           avg_train_steps_data_v2, avg_train_steps_means_v2, avg_train_steps_stds_v2,
           avg_train_caught_frac_data_v2, avg_train_caught_frac_means_v2, avg_train_caught_frac_stds_v2,
           avg_train_early_stopping_frac_data_v2, avg_train_early_stopping_means_v2,
           avg_train_early_stopping_stds_v2, avg_train_intrusion_frac_data_v2, avg_train_intrusion_means_v2,
           avg_train_intrusion_stds_v2
               ):
    print("plot")
    suffix = "gensim"
    ylim_rew = (-300, 170)
    print(len(avg_train_rewards_data_v3[0]))
    max_iter = 1000

    plotting_util_defender.plot_flags_int_r_steps_costs_alerts_two_versions(
        avg_train_rewards_data_v3[0:max_iter], avg_train_rewards_means_v3[0:max_iter],
        avg_train_rewards_stds_v3[0:max_iter],
        avg_train_steps_data_v3[0:max_iter], avg_train_steps_means_v3[0:max_iter], avg_train_steps_stds_v3[0:max_iter],
        avg_train_caught_frac_data_v3[0:max_iter], avg_train_caught_frac_means_v3[0:max_iter],
        avg_train_caught_frac_stds_v3[0:max_iter],
        avg_train_early_stopping_frac_data_v3[0:max_iter], avg_train_early_stopping_means_v3[0:max_iter],
        avg_train_early_stopping_stds_v3[0:max_iter], avg_train_intrusion_frac_data_v3[0:max_iter],
        avg_train_intrusion_means_v3[0:max_iter],
        avg_train_intrusion_stds_v3[0:max_iter],

        avg_train_rewards_data_v2[0:max_iter], avg_train_rewards_means_v2[0:max_iter],
        avg_train_rewards_stds_v2[0:max_iter],
        avg_train_steps_data_v2[0:max_iter], avg_train_steps_means_v2[0:max_iter],
        avg_train_steps_stds_v2[0:max_iter],
        avg_train_caught_frac_data_v2[0:max_iter], avg_train_caught_frac_means_v2[0:max_iter],
        avg_train_caught_frac_stds_v2[0:max_iter],
        avg_train_early_stopping_frac_data_v2[0:max_iter], avg_train_early_stopping_means_v2[0:max_iter],
        avg_train_early_stopping_stds_v2[0:max_iter], avg_train_intrusion_frac_data_v2[0:max_iter],
        avg_train_intrusion_means_v2[0:max_iter],
        avg_train_intrusion_stds_v2[0:max_iter],
        fontsize= 6.5, figsize= (7.5, 1.5), title_fontsize=8, lw=0.75, wspace=0.17, hspace=0.4, top=0.0,
        bottom=0.28, labelsize=6, markevery=50, optimal_reward = 100, sample_step = 2,
        eval_only=False, plot_opt = False, iterations_per_step= 10, optimal_int = 1.0,
        optimal_flag = 1.0, file_name = "flags_int_steps_r_costs_alerts_defender_cnsm_21_2", markersize=2.25
    )


if __name__ == '__main__':
    base_path_1 = "/Users/kimham/workspace/gym-optimal-intrusion-response/examples/v3/v3_results/v3/results_1/data/"
    base_path_2 = "/Users/kimham/workspace/gym-optimal-intrusion-response/examples/v3/v2_results/v2/results_1/data/"
    avg_train_rewards_data_v3, avg_train_rewards_means_v3, avg_train_rewards_stds_v3, \
    avg_train_steps_data_v3, avg_train_steps_means_v3, avg_train_steps_stds_v3, \
    avg_train_caught_frac_data_v3, avg_train_caught_frac_means_v3, avg_train_caught_frac_stds_v3, \
    avg_train_early_stopping_frac_data_v3, avg_train_early_stopping_means_v3, \
    avg_train_early_stopping_stds_v3, avg_train_intrusion_frac_data_v3, avg_train_intrusion_means_v3, \
    avg_train_intrusion_stds_v3, \
    avg_train_rewards_data_v2, avg_train_rewards_means_v2, avg_train_rewards_stds_v2, \
    avg_train_steps_data_v2, avg_train_steps_means_v2, avg_train_steps_stds_v2, \
    avg_train_caught_frac_data_v2, avg_train_caught_frac_means_v2, avg_train_caught_frac_stds_v2, \
    avg_train_early_stopping_frac_data_v2, avg_train_early_stopping_means_v2, \
    avg_train_early_stopping_stds_v2, avg_train_intrusion_frac_data_v2, avg_train_intrusion_means_v2, \
    avg_train_intrusion_stds_v2 = parse_data(base_path_1=base_path_1, base_path_2=base_path_2,
                                             suffix="gensim")

    plot_train(avg_train_rewards_data_v3, avg_train_rewards_means_v3, avg_train_rewards_stds_v3,
    avg_train_steps_data_v3, avg_train_steps_means_v3, avg_train_steps_stds_v3,
    avg_train_caught_frac_data_v3, avg_train_caught_frac_means_v3, avg_train_caught_frac_stds_v3,
    avg_train_early_stopping_frac_data_v3, avg_train_early_stopping_means_v3,
    avg_train_early_stopping_stds_v3, avg_train_intrusion_frac_data_v3, avg_train_intrusion_means_v3,
    avg_train_intrusion_stds_v3,
    avg_train_rewards_data_v2, avg_train_rewards_means_v2, avg_train_rewards_stds_v2,
    avg_train_steps_data_v2, avg_train_steps_means_v2, avg_train_steps_stds_v2,
    avg_train_caught_frac_data_v2, avg_train_caught_frac_means_v2, avg_train_caught_frac_stds_v2,
    avg_train_early_stopping_frac_data_v2, avg_train_early_stopping_means_v2,
    avg_train_early_stopping_stds_v2, avg_train_intrusion_frac_data_v2, avg_train_intrusion_means_v2,
    avg_train_intrusion_stds_v2
               )

